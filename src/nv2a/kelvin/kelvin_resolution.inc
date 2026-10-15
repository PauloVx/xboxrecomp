/* Opt-in full-screen raster surfaces. Included only by kelvin_render.c.
 * Guest allocations, register geometry and source textures remain native.
 * Cache metadata is owned by the GPU thread; workers hold immutable pixel
 * pointers. Eviction, imports and native mirrors follow a queue drain.
 */
#ifdef KRES_PC_VIDEO
#include "../../video/pc_video.h"
#endif
#define KRES_SLOTS 16
#define KRES_CACHE_LIMIT ((size_t)384 * 1024 * 1024)
#define KRES_SNAPSHOT_LIMIT ((size_t)256 * 1024 * 1024)

typedef struct KResSurface {
    uint32_t va, format;
    unsigned int pitch, bpp;
    bool depth, dirty;
    uint8_t *pixels, *native;
    size_t bytes, native_bytes;
    uint64_t used;
} KResSurface;

static KResSurface s_res[KRES_SLOTS];
static size_t s_res_bytes;
static uint64_t s_res_clock;
static unsigned int s_res_w, s_res_h;
static int s_res_ready;
static KResSurface *s_res_pinned;

static bool kres_dimension(const char *e, unsigned int *v)
{
    unsigned long n;
    char *end;
    if (!e || !*e || *e < '0' || *e > '9') return false;
    n = strtoul(e, &end, 10);
    if (*end || n < 240 || n > 8192) return false;
    *v = (unsigned int)n;
    return true;
}

static bool kres_enabled(void)
{
    /* The GPU backend owns host-resolution surfaces itself (kelvin_gpu.c). */
    if (kgpu_enabled()) return false;
    if (!s_res_ready) {
        const char *w = getenv("RECOMP_RENDER_WIDTH");
        const char *h = getenv("RECOMP_RENDER_HEIGHT");
#ifdef KRES_PC_VIDEO
        /* The production build takes the raster size from the player's video settings (pc_video.c): the internal
         * resolution, or RECOMP_RENDER_WIDTH/HEIGHT when those are set. The CPU renderer applies it at start. */
        char wbuf[16], hbuf[16];
        {
            unsigned int rw, rh;
            pc_video_render_size(&rw, &rh);
            snprintf(wbuf, sizeof(wbuf), "%u", rw);
            snprintf(hbuf, sizeof(hbuf), "%u", rh);
            w = wbuf; h = hbuf;
        }
#endif
        s_res_ready = 1;
        if (!w && !h) return false;
        if (!kres_dimension(w, &s_res_w) || !kres_dimension(h, &s_res_h) ||
            (uint64_t)s_res_w * s_res_h > 16777216u) {
            s_res_w = s_res_h = 0;
            fprintf(stderr, "[KRES] rejected render dimensions: expected 240..8192"
                            " and at most 16777216 pixels; native raster retained\n");
        } else if (s_res_w == 640 && s_res_h == 480) {
            s_res_w = s_res_h = 0;
        } else {
            fprintf(stderr, "[KRES] full-screen raster %ux%u; guest 640x480;"
                            " cache limit 384 MiB, feedback snapshots 256 MiB\n",
                    s_res_w, s_res_h);
        }
    }
    return s_res_w && s_res_h;
}

static void kres_import(KResSurface *s, const uint8_t *source)
{
    unsigned int x, y;
    for (y = 0; y < s_res_h; y++) {
        unsigned int sy = (unsigned int)(((uint64_t)(2 * y + 1) * 480) / (2 * s_res_h));
        uint8_t *row = s->pixels + (size_t)y * s_res_w * s->bpp;
        const uint8_t *src = source + (size_t)sy * s->pitch;
        for (x = 0; x < s_res_w; x++) {
            unsigned int sx = (unsigned int)(((uint64_t)(2 * x + 1) * 640) / (2 * s_res_w));
            memcpy(row + (size_t)x * s->bpp, src + (size_t)sx * s->bpp, s->bpp);
        }
    }
    memcpy(s->native, source, s->native_bytes);
    s->dirty = false;
}

/* CPU edits do not discard high-resolution detail in untouched native
 * pixels. Replace only the blocks whose native texels actually changed. */
static void kres_import_updates(KResSurface *s, const uint8_t *source,
                                const uint8_t *changed)
{
    unsigned int x, y;
    for (y = 0; y < 480; y++) {
        for (x = 0; x < 640; x++) {
            size_t index = (size_t)y * 640 + x;
            size_t off = index * s->bpp;
            unsigned int x0, x1, y0, y1, hx, hy;
            if (!(changed[index >> 3] & (1u << (index & 7)))) continue;
            memcpy(kelvin_guest(s->va) + (size_t)y * s->pitch + (size_t)x * s->bpp,
                   source + off, s->bpp);
            x0 = (unsigned int)(((uint64_t)2 * x * s_res_w + 639) / 1280);
            x1 = (unsigned int)(((uint64_t)2 * (x + 1) * s_res_w + 639) / 1280);
            y0 = (unsigned int)(((uint64_t)2 * y * s_res_h + 479) / 960);
            y1 = (unsigned int)(((uint64_t)2 * (y + 1) * s_res_h + 479) / 960);
            for (hy = y0; hy < y1; hy++) {
                for (hx = x0; hx < x1; hx++)
                    memcpy(s->pixels + ((size_t)hy * s_res_w + hx) * s->bpp,
                           source + off, s->bpp);
            }
        }
    }
    memcpy(s->native, kelvin_guest(s->va), s->native_bytes);
    s->dirty = false;
}

/* GPU idle, notify, semaphore and flip callers use this after every queued
 * lane has completed. These readback mirrors never replace the high-res
 * image used by subsequent render-to-texture draws. */
static void kres_mirror(void)
{
    unsigned int i, x, y;
    for (i = 0; i < KRES_SLOTS; i++) {
        KResSurface *s = &s_res[i];
        uint8_t *guest;
        if (!s->pixels || !s->dirty) continue;
        guest = kelvin_guest(s->va);
        for (y = 0; y < 480; y++) {
            unsigned int sy = (unsigned int)(((uint64_t)(2 * y + 1) * s_res_h) / 960);
            const uint8_t *row = s->pixels + (size_t)sy * s_res_w * s->bpp;
            uint8_t *dst = guest + (size_t)y * s->pitch;
            for (x = 0; x < 640; x++) {
                unsigned int sx = (unsigned int)(((uint64_t)(2 * x + 1) * s_res_w) / 1280);
                memcpy(dst + (size_t)x * s->bpp, row + (size_t)sx * s->bpp, s->bpp);
            }
        }
        memcpy(s->native, guest, s->native_bytes);
        s->dirty = false;
    }
}

static void kres_drop(KResSurface *s)
{
    s_res_bytes -= s->bytes + s->native_bytes;
    free(s->pixels);
    free(s->native);
    memset(s, 0, sizeof(*s));
}

static bool kres_overlap(uint32_t va, size_t bytes, const KResSurface *s)
{
    return s->pixels && (uint64_t)va < (uint64_t)s->va + s->native_bytes &&
           (uint64_t)s->va < (uint64_t)va + bytes;
}

/* A different target layout or a GPU blit over an allocation ends that
 * cache identity. It must not leave old high-res pixels hiding new bytes. */
static void kres_invalidate(uint32_t va, size_t bytes)
{
    unsigned int i;
    bool any = false;
    for (i = 0; i < KRES_SLOTS; i++) any |= kres_overlap(va, bytes, &s_res[i]);
    if (!any) return;
    kelvin_render_sync();
    for (i = 0; i < KRES_SLOTS; i++) {
        if (kres_overlap(va, bytes, &s_res[i])) kres_drop(&s_res[i]);
    }
}

static KResSurface *kres_surface(uint32_t va, unsigned int pitch,
                                unsigned int bpp, uint32_t format, bool depth)
{
    unsigned int i;
    KResSurface *s = NULL;
    size_t native_bytes = (size_t)pitch * 480;
    size_t bytes = (size_t)s_res_w * s_res_h * bpp;
    if (!va || !bpp || pitch < 640 * bpp) return NULL;
    for (i = 0; i < KRES_SLOTS; i++) {
        KResSurface *q = &s_res[i];
        if (q->pixels && q->va == va && q->pitch == pitch && q->bpp == bpp &&
            q->format == format && q->depth == depth) {
            if (memcmp(q->native, kelvin_guest(va), native_bytes)) {
                /* Preserve a CPU update even if an outstanding raster batch
                 * still has this target: the import follows its completion. */
                size_t compact_bytes = (size_t)640 * 480 * bpp;
                uint8_t *copy = (uint8_t *)malloc(compact_bytes + 640 * 480 / 8);
                uint8_t *changed;
                unsigned int x, y;
                if (!copy) {
                    fprintf(stderr, "[KRES] CPU-update capture allocation failed; native target retained\n");
                    return NULL;
                }
                changed = copy + compact_bytes;
                memset(changed, 0, 640 * 480 / 8);
                for (y = 0; y < 480; y++) {
                    const uint8_t *row = kelvin_guest(va) + (size_t)y * pitch;
                    memcpy(copy + (size_t)y * 640 * bpp, row, (size_t)640 * bpp);
                    for (x = 0; x < 640; x++) {
                        size_t index = (size_t)y * 640 + x;
                        if (memcmp(row + (size_t)x * bpp,
                                   q->native + (size_t)y * pitch + (size_t)x * bpp, bpp))
                            changed[index >> 3] |= (uint8_t)(1u << (index & 7));
                    }
                }
                kelvin_render_sync();
                /* Compare against the pre-drain baseline above: a pending
                 * draw changing other texels must not look like CPU edits. */
                kres_import_updates(q, copy, changed);
                free(copy);
            }
            q->used = ++s_res_clock;
            return q;
        }
    }
    kres_invalidate(va, native_bytes);
    for (;;) {
        for (i = 0; i < KRES_SLOTS; i++) {
            if (!s_res[i].pixels) { s = &s_res[i]; break; }
        }
        if (s && bytes + native_bytes <= KRES_CACHE_LIMIT - s_res_bytes) break;
        /* Every pointer in deferred geometry and raster jobs must retire
         * before freeing an LRU surface, including sampled-only surfaces. */
        kelvin_render_sync();
        {
            KResSurface *old = NULL;
            for (i = 0; i < KRES_SLOTS; i++) {
                if (s_res[i].pixels && &s_res[i] != s_res_pinned &&
                    (!old || s_res[i].used < old->used)) old = &s_res[i];
            }
            if (!old) return NULL;
            kres_drop(old);
        }
        s = NULL;
    }
    s->pixels = (uint8_t *)malloc(bytes);
    s->native = (uint8_t *)malloc(native_bytes);
    if (!s->pixels || !s->native) {
        free(s->pixels); free(s->native); memset(s, 0, sizeof(*s));
        fprintf(stderr, "[KRES] host allocation failed (%zu bytes); native target retained\n",
                bytes + native_bytes);
        return NULL;
    }
    s->va = va; s->pitch = pitch; s->bpp = bpp; s->format = format;
    s->depth = depth; s->bytes = bytes; s->native_bytes = native_bytes;
    s->used = ++s_res_clock;
    s_res_bytes += bytes + native_bytes;
    kres_import(s, kelvin_guest(va));
    fprintf(stderr, "[KRES] %s 0x%08X: guest640x480 pitch%u -> raster%ux%u"
                    " pitch%u; cache%zu bytes\n", depth ? "depth" : "color", va,
            pitch, s_res_w, s_res_h, s_res_w * bpp, s_res_bytes);
    return s;
}

static void kres_bind(KelvinState *pg, KelvinTarget *t)
{
    KResSurface *color = NULL, *depth = NULL;
    bool eligible;
    if (!kres_enabled()) return;
    eligible = !t->swizzled && t->aa_x == 1 && t->aa_y == 1 &&
               t->width == 640 && t->height == 480;
    if (t->color && t->zeta &&
        (uint64_t)t->color_va < (uint64_t)t->zeta_va + t->guest_zeta_bytes &&
        (uint64_t)t->zeta_va < (uint64_t)t->color_va + t->guest_color_bytes) {
        static int said;
        if (!said++) fprintf(stderr, "[KRES] overlapping color/depth allocations; native raster retained\n");
        eligible = false;
    }
    if (!eligible) {
        if (t->color) kres_invalidate(t->color_va, t->guest_color_bytes);
        if (t->zeta) kres_invalidate(t->zeta_va, t->guest_zeta_bytes);
        return;
    }
    if (t->color) color = kres_surface(t->color_va, t->color_pitch,
        t->color_bpp, pg->surface_shape.color_format, false);
    if (t->zeta) depth = kres_surface(t->zeta_va, t->zeta_pitch,
        t->zeta_bpp, pg->surface_shape.zeta_format, true);
    if ((t->color && !color) || (t->zeta && !depth)) {
        /* Allocation failure never mixes high-res color with native depth. */
        kelvin_render_sync();
        fprintf(stderr, "[KRES] incomplete target allocation; native raster retained\n");
        return;
    }
    if (!color && !depth) return;
    t->raster_x = (double)s_res_w / 640.0;
    t->raster_y = (double)s_res_h / 480.0;
    t->width = s_res_w; t->height = s_res_h;
    t->clip_x0 = (int)floor(t->clip_x0 * t->raster_x);
    t->clip_y0 = (int)floor(t->clip_y0 * t->raster_y);
    t->clip_x1 = (int)ceil(t->clip_x1 * t->raster_x);
    t->clip_y1 = (int)ceil(t->clip_y1 * t->raster_y);
    if (color) {
        t->host_color = (unsigned int)(color - s_res) + 1;
        t->color = color->pixels; t->color_pitch = s_res_w * t->color_bpp;
    }
    if (depth) {
        t->host_zeta = (unsigned int)(depth - s_res) + 1;
        t->zeta = depth->pixels; t->zeta_pitch = s_res_w * t->zeta_bpp;
    }
}

static void kres_written(const KelvinTarget *t, bool color, bool depth)
{
    if (color && t->host_color) s_res[t->host_color - 1].dirty = true;
    if (depth && t->host_zeta) s_res[t->host_zeta - 1].dirty = true;
}

static bool kres_draw_needs_zeta(KelvinState *pg)
{
    return !kres_enabled() ||
        (pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0) & NV_PGRAPH_CONTROL_0_ZENABLE) ||
        (pgraph_reg_r(pg, NV_PGRAPH_CONTROL_1) & NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE);
}

static void kres_texture(KTexture *t)
{
    unsigned int i;
    if (!kres_enabled() || !t->valid) return;
    for (i = 0; i < KRES_SLOTS; i++) {
        KResSurface *s = &s_res[i];
        if (!s->pixels || t->base != s->va || !t->linear || t->cubemap ||
            t->width != 640 || t->height != 480 || t->bpp != s->bpp ||
            t->pitch != s->pitch || t->levels != 1 || t->depth_format != s->depth) continue;
        /* Read any CPU update through the same import path as a bound target. */
        s = kres_surface(s->va, s->pitch, s->bpp, s->format, s->depth);
        if (!s) break;
        t->mem = s->pixels;
        t->host_bytes = s->bytes;
        t->pitch = s_res_w * t->bpp;
        t->level_w[0] = s_res_w; t->level_h[0] = s_res_h;
        return;
    }
    /* Other texture views retain native addressing. Mirror if their guest
     * bytes overlap a high-res target before they can be sampled. */
    for (i = 0; i < KRES_SLOTS; i++) {
        if (kres_overlap(t->base, (size_t)(t->cubemap ? 6 : 1) * t->face_stride,
                         &s_res[i]) && s_res[i].dirty) {
            kelvin_render_sync();
            break;
        }
    }
}

/* A native whole-frame SRCCOPY keeps the larger pixels too. Partial blits,
 * blending and differently laid out targets retain the original guest path.
 * Called after native rows have been copied and the queue has drained. */
static bool kres_blit_prepare(uint32_t src, unsigned int pitch, unsigned int bpp)
{
    unsigned int i;
    if (!kres_enabled()) return true;
    for (i = 0; i < KRES_SLOTS; i++) {
        KResSurface *s = &s_res[i];
        if (s->pixels && !s->depth && s->va == src && s->pitch == pitch && s->bpp == bpp) {
            /* Detect CPU edits before SRCCOPY/forced-alpha mutates any rows,
             * including a copy onto the very same allocation. */
            return kres_surface(s->va, s->pitch, s->bpp, s->format, false) != NULL;
        }
    }
    return true;
}

static void kres_blit_copy(uint32_t src, unsigned int src_pitch,
                           uint32_t dst, unsigned int dst_pitch,
                           unsigned int bpp, uint32_t format, int alpha)
{
    unsigned int i;
    KResSurface *from = NULL, *to;
    if (!kres_enabled()) return;
    for (i = 0; i < KRES_SLOTS; i++) {
        if (s_res[i].pixels && !s_res[i].depth && s_res[i].va == src &&
            s_res[i].pitch == src_pitch && s_res[i].bpp == bpp) { from = &s_res[i]; break; }
    }
    if (!from) return;
    if (src == dst && src_pitch == dst_pitch) {
        from->format = format;
        if (bpp == 4 && alpha >= 0) {
            size_t p;
            for (p = 3; p < from->bytes; p += 4) from->pixels[p] = (uint8_t)alpha;
        }
        memcpy(from->native, kelvin_guest(src), from->native_bytes);
        from->dirty = false;
        return;
    }
    if (((uint64_t)src < (uint64_t)dst + (size_t)dst_pitch * 480 &&
                  (uint64_t)dst < (uint64_t)src + from->native_bytes)) return;
    from->used = ++s_res_clock;
    /* The source may be the oldest cache entry under pressure. It remains
     * pinned across destination allocation; the two disjoint surfaces always
     * fit within the cache bound at the accepted resolution limit. */
    s_res_pinned = from;
    to = kres_surface(dst, dst_pitch, bpp, format, false);
    s_res_pinned = NULL;
    if (!to) {
        fprintf(stderr, "[KRES] whole-frame blit host allocation failed; native copy retained\n");
        return;
    }
    memcpy(to->pixels, from->pixels, from->bytes);
    if (bpp == 4 && alpha >= 0) {
        size_t p;
        for (p = 3; p < to->bytes; p += 4) to->pixels[p] = (uint8_t)alpha;
    }
    /* Guest SRCCOPY above already produced exactly the same native mirror. */
    to->dirty = false;
}
