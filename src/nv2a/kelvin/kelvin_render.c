/*
 * Kelvin renderer: surfaces, clears, blits, fences and the draw path --
 * vertex fetch, primitive assembly, clipping and rasterization.
 *
 * Draw semantics follow xemu: vertex sources as gl/draw.c pgraph_gl_flush_draw
 * and gl/vertex.c bind them, primitive assembly as glsl/geom.c splits quads,
 * strips and fans, depth as the zvalue computation at the top of glsl/psh.c
 * (a GPL-2.0+ reference for behaviour only; implemented independently),
 * clears as gl/draw.c pgraph_gl_clear_surface and blits as gl/blit.c. Pixels
 * are produced by kelvin_pixel.c into guest memory, or into larger owned
 * full-screen surfaces when RECOMP_RENDER_WIDTH/HEIGHT are supplied.
 *
 * Copyright (c) 2012 espes; 2015 Jannik Vogel; 2018-2025 Matt Borgerson
 * (xemu, whose semantics these are)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <stdlib.h>
#include <limits.h>
#include "kelvin_render.h"
#include "kelvin_trace.h"
#include "kelvin_workers.h"
#include "kelvin_gpu.h"
#include <intrin.h>
#include <immintrin.h>

#include "../../video/fb_present.h"

/* ---- targets ------------------------------------------------------------- */

static unsigned int color_bpp(unsigned int fmt)
{
    switch (fmt) {
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_Z1R5G5B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_O1R5G5B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_G8B8:
        return 2;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1A7R8G8B8_Z1A7R8G8B8:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1A7R8G8B8_O1A7R8G8B8:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8:
        return 4;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_B8:
        return 1;
    default:
        return 0;
    }
}

static void swizzle_masks(unsigned int width, unsigned int height,
                          uint32_t *mx, uint32_t *my)
{
    uint32_t x = 0, y = 0, bit = 1, mask_bit = 1;
    bool done;
    do {
        done = true;
        if (bit < width) { x |= mask_bit; mask_bit <<= 1; done = false; }
        if (bit < height) { y |= mask_bit; mask_bit <<= 1; done = false; }
        bit <<= 1;
    } while (!done);
    *mx = x;
    *my = y;
}

#include "kelvin_resolution.inc"

bool kelvin_bind_target(KelvinState *pg, KelvinTarget *t, bool need_color,
                        bool need_zeta)
{
    const SurfaceShape *s = &pg->surface_shape;
    unsigned int log_w = s->log_width, log_h = s->log_height;

    memset(t, 0, sizeof(*t));
    switch (s->anti_aliasing) {
    case NV097_SET_SURFACE_FORMAT_ANTI_ALIASING_CENTER_CORNER_2:
        t->aa_x = 2; t->aa_y = 1; break;
    case NV097_SET_SURFACE_FORMAT_ANTI_ALIASING_SQUARE_OFFSET_4:
        t->aa_x = 2; t->aa_y = 2; break;
    default:
        t->aa_x = 1; t->aa_y = 1; break;
    }
    t->swizzled = pg->surface_type == NV097_SET_SURFACE_FORMAT_TYPE_SWIZZLE;
    if (t->swizzled) {
        t->width = 1u << log_w;
        t->height = 1u << log_h;
        swizzle_masks(t->width, t->height, &t->swz_mask_x, &t->swz_mask_y);
    } else {
        t->width = (s->clip_x + s->clip_width) * t->aa_x;
        t->height = (s->clip_y + s->clip_height) * t->aa_y;
    }
    t->clip_x0 = s->clip_x * t->aa_x;
    t->clip_y0 = s->clip_y * t->aa_y;
    t->clip_x1 = (s->clip_x + s->clip_width) * t->aa_x;
    t->clip_y1 = (s->clip_y + s->clip_height) * t->aa_y;
    if (t->clip_x1 > (int)t->width) t->clip_x1 = t->width;
    if (t->clip_y1 > (int)t->height) t->clip_y1 = t->height;

    if (need_color && pg->dma_color) {
        t->color_bpp = color_bpp(s->color_format);
        t->color_pitch = pg->surface_color.pitch;
        if (!t->color_bpp) {
            kelvin_unsupported(KU_SURFACE_FORMAT, "colour surface format",
                               s->color_format);
        } else {
            uint32_t bytes = t->swizzled ? t->width * t->height * t->color_bpp
                                         : t->color_pitch * t->height;
            t->color_va = kelvin_dma_va(pg->dma_color, pg->surface_color.offset,
                                        bytes);
            if (t->color_va) {
                t->color = kelvin_guest(t->color_va);
            } else {
                kelvin_unsupported(KU_DMA_UNRESOLVED, "colour surface",
                                   pg->surface_color.offset);
            }
        }
    }
    if (need_zeta && pg->dma_zeta && s->zeta_format) {
        t->zeta_bpp = s->zeta_format == NV097_SET_SURFACE_FORMAT_ZETA_Z16 ? 2 : 4;
        t->zeta_pitch = pg->surface_zeta.pitch;
        {
            uint32_t bytes = t->swizzled ? t->width * t->height * t->zeta_bpp
                                         : t->zeta_pitch * t->height;
            t->zeta_va = kelvin_dma_va(pg->dma_zeta, pg->surface_zeta.offset,
                                       bytes);
            if (t->zeta_va) {
                t->zeta = kelvin_guest(t->zeta_va);
            } else {
                kelvin_unsupported(KU_DMA_UNRESOLVED, "zeta surface",
                                   pg->surface_zeta.offset);
            }
        }
    }
    t->guest_width = t->width; t->guest_height = t->height;
    t->guest_color_pitch = t->color_pitch; t->guest_zeta_pitch = t->zeta_pitch;
    t->guest_color_bytes = t->color ? (t->swizzled ? (size_t)t->width * t->height * t->color_bpp
                                                  : (size_t)t->color_pitch * t->height) : 0;
    t->guest_zeta_bytes = t->zeta ? (t->swizzled ? (size_t)t->width * t->height * t->zeta_bpp
                                              : (size_t)t->zeta_pitch * t->height) : 0;
    kres_bind(pg, t);
    return t->color || t->zeta;
}

/* ---- clears (gl/draw.c pgraph_gl_clear_surface) --------------------------- */

void kelvin_render_clear_surface(KelvinState *pg, uint32_t parameter)
{
    bool write_color = parameter & NV097_CLEAR_SURFACE_COLOR;
    bool write_z = parameter & NV097_CLEAR_SURFACE_Z;
    bool write_s = parameter & NV097_CLEAR_SURFACE_STENCIL;
    KelvinTarget t;
    uint32_t rx = pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTX);
    uint32_t ry = pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTY);
    int xmin = GET_MASK(rx, NV_PGRAPH_CLEARRECTX_XMIN);
    int xmax = GET_MASK(rx, NV_PGRAPH_CLEARRECTX_XMAX);
    int ymin = GET_MASK(ry, NV_PGRAPH_CLEARRECTY_YMIN);
    int ymax = GET_MASK(ry, NV_PGRAPH_CLEARRECTY_YMAX);
    uint32_t cv = pgraph_reg_r(pg, NV_PGRAPH_COLORCLEARVALUE);
    uint32_t zv = pgraph_reg_r(pg, NV_PGRAPH_ZSTENCILCLEARVALUE);
    int x0, y0, x1, y1, x, y;

    kelvin_render_sync_for(KT_SYNC_CLEAR);
    if (!kelvin_bind_target(pg, &t, write_color, write_z || write_s)) {
        return;
    }
    pg->stats.clears++;
    if (t.swizzled) pg->stats.swizzled_clears++;
    x0 = (int)floor(xmin * kelvin_raster_x(&t));
    y0 = (int)floor(ymin * kelvin_raster_y(&t));
    x1 = (int)ceil((xmax + 1) * kelvin_raster_x(&t));
    y1 = (int)ceil((ymax + 1) * kelvin_raster_y(&t));
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int)t.width) x1 = t.width;
    if (y1 > (int)t.height) y1 = t.height;
    if (kgpu_active()) {
        unsigned int chan = ((parameter & NV097_CLEAR_SURFACE_B) ? 1u : 0u) |
                            ((parameter & NV097_CLEAR_SURFACE_G) ? 2u : 0u) |
                            ((parameter & NV097_CLEAR_SURFACE_R) ? 4u : 0u) |
                            ((parameter & NV097_CLEAR_SURFACE_A) ? 8u : 0u);
        kgpu_clear(&t, write_color && chan, write_z, write_s, cv, zv, chan, x0, y0, x1, y1,
                   pg->surface_shape.color_format);
        if (!kgpu_compare_enabled()) return;
    }
    kres_written(&t, write_color, write_z || write_s);
    if (write_color && t.color) {
        uint8_t mask[4] = {
            (parameter & NV097_CLEAR_SURFACE_B) ? 0xFF : 0,
            (parameter & NV097_CLEAR_SURFACE_G) ? 0xFF : 0,
            (parameter & NV097_CLEAR_SURFACE_R) ? 0xFF : 0,
            (parameter & NV097_CLEAR_SURFACE_A) ? 0xFF : 0,
        };
        unsigned int fmt = pg->surface_shape.color_format;
        bool words = !t.swizzled && t.color_bpp == 4;
        if (words) {
            /* The same bytes a pixel at a time: the masked value as a
             * little-endian word. */
            uint32_t v = cv, m = (uint32_t)mask[0] | ((uint32_t)mask[1] << 8) |
                                 ((uint32_t)mask[2] << 16) | ((uint32_t)mask[3] << 24);
            if (fmt == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8) {
                v &= 0x00FFFFFFu;
            } else if (fmt == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8) {
                v |= 0xFF000000u;
            }
            v &= m;
            for (y = y0; y < y1; y++) {
                uint8_t *row = t.color + (size_t)y * t.color_pitch + (size_t)x0 * 4;
                for (x = 0; x < x1 - x0; x++) {
                    uint32_t old;
                    memcpy(&old, row + 4 * x, 4);
                    old = (old & ~m) | v;
                    memcpy(row + 4 * x, &old, 4);
                }
            }
        }
        for (y = y0; y < y1 && !words; y++) {
            for (x = x0; x < x1; x++) {
                uint8_t *p = kelvin_color_ptr(&t, x, y);
                if (t.color_bpp == 4) {
                    uint8_t v[4];
                    int k;
                    v[0] = (uint8_t)cv; v[1] = (uint8_t)(cv >> 8);
                    v[2] = (uint8_t)(cv >> 16); v[3] = (uint8_t)(cv >> 24);
                    if (fmt == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8) {
                        v[3] = 0;
                    } else if (fmt == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8) {
                        v[3] = 0xFF;
                    }
                    for (k = 0; k < 4; k++) {
                        p[k] = (uint8_t)((p[k] & ~mask[k]) | (v[k] & mask[k]));
                    }
                } else if (t.color_bpp == 2) {
                    uint16_t old = (uint16_t)(p[0] | (p[1] << 8)), m = 0, nv;
                    if (fmt == NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5) {
                        m = (uint16_t)((mask[2] ? 0xF800 : 0) | (mask[1] ? 0x07E0 : 0) |
                                       (mask[0] ? 0x001F : 0));
                    } else {
                        m = (uint16_t)((mask[2] ? 0x7C00 : 0) | (mask[1] ? 0x03E0 : 0) |
                                       (mask[0] ? 0x001F : 0) | (mask[3] ? 0x8000 : 0));
                    }
                    nv = (uint16_t)((old & ~m) | (cv & m));
                    p[0] = (uint8_t)nv; p[1] = (uint8_t)(nv >> 8);
                } else if (mask[0]) {
                    *p = (uint8_t)cv;
                }
            }
        }
    }
    if ((write_z || write_s) && t.zeta && !t.swizzled && t.zeta_bpp == 4) {
        /* The same bytes a pixel at a time, as a little-endian word. */
        uint32_t m = (write_z ? 0xFFFFFF00u : 0) | (write_s ? 0xFFu : 0);
        for (y = y0; y < y1; y++) {
            uint8_t *row = t.zeta + (size_t)y * t.zeta_pitch + (size_t)x0 * 4;
            for (x = 0; x < x1 - x0; x++) {
                uint32_t old;
                memcpy(&old, row + 4 * x, 4);
                old = (old & ~m) | (zv & m);
                memcpy(row + 4 * x, &old, 4);
            }
        }
    } else if ((write_z || write_s) && t.zeta) {
        for (y = y0; y < y1; y++) {
            for (x = x0; x < x1; x++) {
                uint8_t *p = kelvin_zeta_ptr(&t, x, y);
                if (t.zeta_bpp == 4) {
                    uint32_t old = p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
                    uint32_t m = (write_z ? 0xFFFFFF00u : 0) | (write_s ? 0xFFu : 0);
                    uint32_t nv = (old & ~m) | (zv & m);
                    p[0] = (uint8_t)nv; p[1] = (uint8_t)(nv >> 8);
                    p[2] = (uint8_t)(nv >> 16); p[3] = (uint8_t)(nv >> 24);
                } else if (write_z) {
                    p[0] = (uint8_t)zv; p[1] = (uint8_t)(zv >> 8);
                }
            }
        }
    }
}

/* ---- fences ----------------------------------------------------------------- */

/* A semaphore release waiting behind the raster work queued before it. At
 * most KWORK_QUEUE jobs are unfinished, so a record is free again long
 * before the ring comes back to it. */
typedef struct SemaphoreRelease {
    volatile uint32_t *where;
    uint32_t value;
} SemaphoreRelease;

static SemaphoreRelease s_releases[2 * KWORK_QUEUE];
static unsigned int s_release_next;

static void semaphore_write(void *arg)
{
    const SemaphoreRelease *r = (const SemaphoreRelease *)arg;
    MemoryBarrier();
    *r->where = r->value;
}

/* RECOMP_KELVIN_SYNC_SEMAPHORES=1 drains the queue at each release instead,
 * as before. */
static bool sync_semaphores(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_KELVIN_SYNC_SEMAPHORES");
        on = e && *e && *e != '0';
    }
    return on != 0;
}

static void semaphore_write_thunk(uintptr_t where, uintptr_t value)
{
    *(volatile uint32_t *)where = (uint32_t)value;
}

void kelvin_render_semaphore_release(KelvinState *pg, uint32_t value)
{
    uint32_t offset = pgraph_reg_r(pg, NV_PGRAPH_SEMAPHOREOFFSET);
    uint32_t va = kelvin_dma_va(pg->dma_semaphore, offset, 4);

    if (!va) {
        kelvin_render_sync_for(KT_SYNC_SEMAPHORE);
        kelvin_unsupported(KU_DMA_UNRESOLVED, "semaphore", offset);
        return;
    }
    pg->stats.semaphores++;
    /* The title reads the semaphore as "the GPU has got this far": the
     * value lands once everything drawn before it is in memory. The value
     * is written from the queue, after the work ahead of it, while the
     * pusher goes on; every later drain (notify, idle, a readback) waits
     * for it too. */
    if (kgpu_active() || kres_enabled() || sync_semaphores() || !kwork_workers()) {
        if (kgpu_async_active()) {
            /* written by the GPU thread, after the draws queued before it */
            kgpu_defer_call(semaphore_write_thunk, (uintptr_t)kelvin_guest(va), (uintptr_t)value);
            return;
        }
        kelvin_render_sync_for(KT_SYNC_SEMAPHORE);
        *(volatile uint32_t *)kelvin_guest(va) = value;
        return;
    }
    /* Behind every draw before it: their triangles are queued first. */
    kelvin_render_geometry_sync();
    {
        SemaphoreRelease *r = &s_releases[s_release_next++ % (2 * KWORK_QUEUE)];
        r->where = (volatile uint32_t *)kelvin_guest(va);
        r->value = value;
        kwork_submit_lanes(semaphore_write, NULL, r, NULL, 0, KWORK_BARRIER);
    }
}

/* ---- image blit (gl/blit.c) ----------------------------------------------- */

void kelvin_render_image_blit(KelvinState *pg)
{
    ContextSurfaces2DState *cs = &pg->context_surfaces_2d;
    ImageBlitState *ib = &pg->image_blit;
    unsigned int bpp, y, x;
    uint32_t src_va, dst_va;
    size_t row_pixels;
    bool host_copy_ready = true;

    if (kgpu_active()) {
        /* BLACK issues no image blits; surfaces live on the GPU. */
        kelvin_unsupported(KU_IMAGE_BLIT_OP, "image blit with GPU surfaces", ib->operation);
        return;
    }
    if (!kres_enabled()) kelvin_render_sync();

    switch (cs->color_format) {
    case NV062_SET_COLOR_FORMAT_LE_Y8: bpp = 1; break;
    case NV062_SET_COLOR_FORMAT_LE_R5G6B5: bpp = 2; break;
    case NV062_SET_COLOR_FORMAT_LE_A8R8G8B8:
    case NV062_SET_COLOR_FORMAT_LE_X8R8G8B8:
    case NV062_SET_COLOR_FORMAT_LE_X8R8G8B8_Z8R8G8B8:
    case NV062_SET_COLOR_FORMAT_LE_Y32:
        bpp = 4; break;
    default:
        kelvin_unsupported(KU_IMAGE_BLIT_OP, "blit surface format", cs->color_format);
        return;
    }
    src_va = kelvin_dma_va(cs->dma_image_source,
                           cs->source_offset + ib->in_y * cs->source_pitch +
                           ib->in_x * bpp, 0);
    dst_va = kelvin_dma_va(cs->dma_image_dest,
                           cs->dest_offset + ib->out_y * cs->dest_pitch +
                           ib->out_x * bpp, 0);
    if (!src_va || !dst_va) {
        kelvin_unsupported(KU_DMA_UNRESOLVED, "blit surface", cs->dest_offset);
        return;
    }
    row_pixels = MIN(MIN(cs->source_pitch, cs->dest_pitch) / bpp, ib->width);
    pg->stats.image_blits++;
    if (ib->operation == NV09F_SET_OPERATION_SRCCOPY && row_pixels == 640 && ib->height == 480)
        host_copy_ready = kres_blit_prepare(src_va, cs->source_pitch, bpp);
    if (kres_enabled()) kelvin_render_sync();
    if (!host_copy_ready) kres_invalidate(src_va, (size_t)cs->source_pitch * 480);
    if (ib->height && !(ib->operation == NV09F_SET_OPERATION_SRCCOPY &&
        src_va == dst_va && cs->source_pitch == cs->dest_pitch &&
        ((row_pixels == 640 && ib->height == 480) ||
         cs->color_format == NV062_SET_COLOR_FORMAT_LE_A8R8G8B8 ||
         cs->color_format == NV062_SET_COLOR_FORMAT_LE_R5G6B5))) kres_invalidate(dst_va,
        (size_t)(ib->height - 1) * cs->dest_pitch + row_pixels * bpp);

    for (y = 0; y < ib->height; y++) {
        uint8_t *s = kelvin_guest(src_va + y * cs->source_pitch);
        uint8_t *d = kelvin_guest(dst_va + y * cs->dest_pitch);
        if (ib->operation == NV09F_SET_OPERATION_SRCCOPY) {
            memmove(d, s, row_pixels * bpp);
        } else if (ib->operation == NV09F_SET_OPERATION_BLEND_AND) {
            uint32_t max_mult = 0x7f80, mult = pg->beta.beta >> 16;
            uint32_t inv = max_mult - mult;
            for (x = 0; x < row_pixels; x++) {
                unsigned int ch;
                for (ch = 0; ch < 3; ch++) {
                    d[x * 4 + ch] = (uint8_t)((s[x * 4 + ch] * mult +
                                               d[x * 4 + ch] * inv) / max_mult);
                }
            }
        } else {
            kelvin_unsupported(KU_IMAGE_BLIT_OP, "blit operation", ib->operation);
            return;
        }
        if (cs->color_format == NV062_SET_COLOR_FORMAT_LE_X8R8G8B8 ||
            cs->color_format == NV062_SET_COLOR_FORMAT_LE_X8R8G8B8_Z8R8G8B8) {
            uint8_t a = cs->color_format == NV062_SET_COLOR_FORMAT_LE_X8R8G8B8
                            ? 0xFF : 0x00;
            for (x = 0; x < row_pixels; x++) {
                d[x * 4 + 3] = a;
            }
        }
    }
    if (host_copy_ready && ib->operation == NV09F_SET_OPERATION_SRCCOPY && row_pixels == 640 && ib->height == 480) {
        unsigned int fmt = 0;
        int alpha = -1;
        if (cs->color_format == NV062_SET_COLOR_FORMAT_LE_A8R8G8B8)
            fmt = NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8;
        else if (cs->color_format == NV062_SET_COLOR_FORMAT_LE_R5G6B5)
            fmt = NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5;
        else if (cs->color_format == NV062_SET_COLOR_FORMAT_LE_X8R8G8B8) {
            fmt = NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8; alpha = 255;
        } else if (cs->color_format == NV062_SET_COLOR_FORMAT_LE_X8R8G8B8_Z8R8G8B8) {
            fmt = NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8; alpha = 0;
        }
        if (fmt) kres_blit_copy(src_va, cs->source_pitch, dst_va, cs->dest_pitch, bpp, fmt, alpha);
    }
}

/* ---- frames --------------------------------------------------------------- */

static uint64_t s_frame_hash_prev;
static uint64_t s_frames, s_frames_distinct;

/* ---- draw trace ---------------------------------------------------------------
 *
 * RECOMP_KELVIN_TRACE=first[-last] prints, for every draw of those frames
 * (numbered as the frame log numbers them), the state the pipeline used:
 * primitive and vertex source, vertex-stage mode, render target, each
 * enabled texture, combiner and output-merger state, the first vertices
 * after the vertex stage (RECOMP_KELVIN_TRACE_VERTS of them, default 4), and
 * how many pixels the draw wrote. With RECOMP_KELVIN_TRACE_TEX set to a path
 * prefix, each texture those draws sample is also written out once per
 * frame, decoded by this renderer's own sampler: colour on the left, alpha on
 * the right. A range may carry a step, first-last/step, to sample a long run. */
static unsigned long long s_trace_first = 1, s_trace_last, s_trace_step = 1;
static uint64_t s_trace_draw;
static size_t s_trace_verts = 4;     /* RECOMP_KELVIN_TRACE_VERTS */
static const char *s_trace_tex;      /* RECOMP_KELVIN_TRACE_TEX */

static bool trace_frame(void)
{
    static int init;
    if (!init) {
        const char *e = getenv("RECOMP_KELVIN_TRACE");
        const char *v = getenv("RECOMP_KELVIN_TRACE_VERTS");
        init = 1;
        s_trace_tex = getenv("RECOMP_KELVIN_TRACE_TEX");
        if (v && *v) {
            s_trace_verts = (size_t)strtoul(v, NULL, 0);
        }
        if (e && *e) {
            char *end;
            s_trace_first = strtoull(e, &end, 0);
            s_trace_last = s_trace_first;
            if (*end == '-') {
                s_trace_last = strtoull(end + 1, &end, 0);
            }
            if (*end == '/') {
                s_trace_step = strtoull(end + 1, NULL, 0);
                if (!s_trace_step) {
                    s_trace_step = 1;
                }
            }
        }
    }
    return s_trace_last && s_frames + 1 >= s_trace_first &&
           s_frames + 1 <= s_trace_last &&
           (s_frames + 1 - s_trace_first) % s_trace_step == 0;
}


static uint32_t hash_surface(const KelvinTarget *t)
{
    uint32_t h = 2166136261u;
    unsigned int y, x;
    for (y = 0; y < t->height; y++) {
        const uint8_t *row = t->color + (size_t)y * t->color_pitch;
        for (x = 0; x < t->width * t->color_bpp; x++) {
            h = (h ^ row[x]) * 16777619u;
        }
    }
    return h;
}

/* A frame's identity for counting distinct frames, cheaper than
 * hash_surface's byte-serial FNV: four interleaved 64-bit lanes. */
static uint64_t quick_hash_surface(const KelvinTarget *t)
{
    uint64_t h[4] = { 1469598103934665603ull, 1099511628211ull, 0x9E3779B97F4A7C15ull, 7 };
    unsigned int y, row = t->width * t->color_bpp, x, k;
    for (y = 0; y < t->height; y++) {
        const uint8_t *p = t->color + (size_t)y * t->color_pitch;
        for (x = 0; x + 32 <= row; x += 32) {
            for (k = 0; k < 4; k++) {
                uint64_t v;
                memcpy(&v, p + x + 8 * k, 8);
                h[k] = (h[k] ^ v) * 0x100000001B3ull;
            }
        }
        for (; x < row; x++) {
            h[0] = (h[0] ^ p[x]) * 0x100000001B3ull;
        }
    }
    return h[0] ^ (h[1] * 3) ^ (h[2] * 5) ^ (h[3] * 7);
}

static void dump_bmp(const char *path, const KelvinTarget *t)
{
    FILE *f = fopen(path, "wb");
    uint32_t w = t->width, h = t->height, y, x, row_bytes, pad, filesz;
    uint8_t hdr[54];
    uint8_t *bgr_row;

    if (!f) {
        return;
    }
    row_bytes = w * 3;
    pad = (4 - (row_bytes & 3)) & 3;
    bgr_row = (uint8_t *)malloc(row_bytes + pad);
    if (!bgr_row) { fclose(f); return; }
    memset(bgr_row + row_bytes, 0, pad);
    filesz = 54 + (row_bytes + pad) * h;
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54; hdr[14] = 40;
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, sizeof(hdr), f);
    for (y = h; y-- > 0;) {
        const uint8_t *row = t->color + (size_t)y * t->color_pitch;
        for (x = 0; x < w; x++) {
            uint8_t *bgr = bgr_row + x * 3;
            if (t->color_bpp == 4) {
                bgr[0] = row[x * 4]; bgr[1] = row[x * 4 + 1]; bgr[2] = row[x * 4 + 2];
            } else {
                uint16_t v = (uint16_t)(row[x * 2] | (row[x * 2 + 1] << 8));
                bgr[0] = (uint8_t)((v & 0x1F) << 3);
                bgr[1] = (uint8_t)(((v >> 5) & 0x3F) << 2);
                bgr[2] = (uint8_t)(((v >> 11) & 0x1F) << 3);
            }
        }
        fwrite(bgr_row, 1, row_bytes + pad, f);
    }
    free(bgr_row);
    fclose(f);
}

/* FLIP_STALL: the frame in the current colour surface is complete. Record
 * it, show it, and optionally write it out. */
/* ---- job trace -------------------------------------------------------------
 *
 * RECOMP_KELVIN_JOBTRACE=n records frame n's timeline: for each batch, when
 * the GPU thread handed it over, when its first lane started and its last
 * ended, its target, triangles and kind; and each wait of the GPU thread for
 * the queue. Printed at that frame's flip, in milliseconds from the flip
 * before. */
typedef struct JobTrace {
    long long submit;
    volatile LONG64 start, end;
    volatile LONG64 cpu, written;      /* lane time summed, fragments written */
    uint32_t sig;                      /* the pixel state's signature */
    char desc[96];
    uint32_t target;
    int w, h, ntris, kind;         /* kind: 0 lanes, 1 conflict, 2 snapshot, 3 one thread */
} JobTrace;

#define JT_MAX 8192
static JobTrace s_jt[JT_MAX];
static int s_njt;
static struct { long long t0, t1; int cause; uint32_t method; } s_jw[1024];
static int s_njw;
static long long s_jt_origin;
static long long s_jt_frame = -2;          /* -2 unread, -1 off */
static bool s_jt_on;

static long long jt_ticks(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

static void jt_max(volatile LONG64 *v, long long x)
{
    LONG64 old = *v;
    while (x > old) {
        LONG64 seen = InterlockedCompareExchange64(v, x, old);
        if (seen == old) break;
        old = seen;
    }
}

static void jt_min_nonzero(volatile LONG64 *v, long long x)
{
    LONG64 old = *v;
    while (old == 0 || x < old) {
        LONG64 seen = InterlockedCompareExchange64(v, x, old);
        if (seen == old) break;
        old = seen;
    }
}

/* GPU-thread events for the job trace: draws, draw-job waits, and loops of
 * the GPU thread that read no word (kelvin_render_trace_event). */
static struct { long long t0, t1; int kind; uint32_t detail; } s_je[16384];
static int s_nje;

bool g_kelvin_tracing;

void kelvin_render_trace_event(int kind, long long t0, long long t1, uint32_t detail)
{
    if (s_jt_on && s_nje < (int)(sizeof(s_je) / sizeof(s_je[0]))) {
        s_je[s_nje].t0 = t0;
        s_je[s_nje].t1 = t1;
        s_je[s_nje].kind = kind;
        s_je[s_nje].detail = detail;
        s_nje++;
    }
}

static void jobtrace_flip(void)
{
    if (s_jt_frame == -2) {
        const char *e = getenv("RECOMP_KELVIN_JOBTRACE");
        s_jt_frame = e && *e ? (long long)strtoull(e, NULL, 0) : -1;
    }
    if (s_jt_frame < 0) {
        return;
    }
    if (s_jt_on) {
        LARGE_INTEGER f;
        double k;
        int i;
        kwork_drain();                   /* every traced batch has ended */
        QueryPerformanceFrequency(&f);
        k = 1000.0 / (double)f.QuadPart;
        fprintf(stderr, "[KJOBS] frame %lld: %d batches, %d waits, flip at %.2f ms\n",
                s_jt_frame, s_njt, s_njw, (jt_ticks() - s_jt_origin) * k);
        for (i = 0; i < s_njt; i++) {
            const JobTrace *j = &s_jt[i];
            static const char *const kinds[] = { "lanes", "conflict", "snapshot", "single" };
            fprintf(stderr, "[KJOBS] b%d submit %.2f start %.2f end %.2f | %s 0x%08X %dx%d"
                            " %d tris | cpu %.3f px %lld sig %08X %s\n", i,
                    (j->submit - s_jt_origin) * k,
                    j->start ? (j->start - s_jt_origin) * k : -1.0,
                    j->end ? (j->end - s_jt_origin) * k : -1.0, kinds[j->kind],
                    j->target, j->w, j->h, j->ntris, j->cpu * k,
                    (long long)j->written, j->sig, j->desc);
        }
        for (i = 0; i < s_njw; i++) {
            fprintf(stderr, "[KJOBS] wait cause %d method %X %.2f .. %.2f\n", s_jw[i].cause, s_jw[i].method,
                    (s_jw[i].t0 - s_jt_origin) * k, (s_jw[i].t1 - s_jt_origin) * k);
        }
        for (i = 0; i < s_nje; i++) {
            fprintf(stderr, "[KJOBS] event %d %X %.3f .. %.3f\n", s_je[i].kind,
                    s_je[i].detail, (s_je[i].t0 - s_jt_origin) * k,
                    (s_je[i].t1 - s_jt_origin) * k);
        }
        fflush(stderr);
        s_jt_on = g_kelvin_tracing = false;
        s_jt_frame = -1;
        return;
    }
    if ((long long)s_frames + 1 == s_jt_frame) {
        /* The flip before the traced frame: its start. */
        s_njt = s_njw = s_nje = 0;
        s_jt_origin = jt_ticks();
        s_jt_on = g_kelvin_tracing = true;
    }
}

static void capture_flip(void);

void kelvin_render_flip_stall(KelvinState *pg,
                              uint32_t aspect_w, uint32_t aspect_h)
{
    KelvinTarget t;
    uint32_t h;
    static const char *dump = (const char *)-1;
    static unsigned int every = 1;
    static long long frame_qpc_frequency;
    long long flip_t0;

    kelvin_render_sync_for(KT_SYNC_FLIP);
    jobtrace_flip();
    capture_flip();
    flip_t0 = jt_ticks();

    if (kgpu_active() && !kgpu_compare_enabled()) {
        if (kelvin_bind_target(pg, &t, true, false) && t.color_va) {
            s_frames++;
            kgpu_flip(&t, pg->surface_shape.color_format, aspect_w, aspect_h);
        }
        return;
    }

    if (dump == (const char *)-1) {
        const char *e = getenv("RECOMP_KELVIN_EVERY");
        dump = getenv("RECOMP_KELVIN_DUMP");
        if (e) {
            every = (unsigned int)strtoul(e, NULL, 0);
            if (!every) every = 1;
        }
    }
    if (!kelvin_bind_target(pg, &t, true, false) || !t.color || t.swizzled ||
        t.color_bpp < 2 || !t.height ||
        t.color_pitch > UINT32_MAX / t.height) {
        return;
    }
    if (kgpu_active()) kgpu_compare_frame(&t, pg->surface_shape.color_format);
    /* The FNV hash only for frames whose hash is printed or named; the
     * same-or-new count goes by the quick one. */
    {
        uint64_t n = s_frames + 1;
        bool traced = trace_frame();
        bool logged = n <= 8 || (n % 60) == 0;
        bool dumped = dump && *dump && (n % every) == 0;
        uint64_t quick = quick_hash_surface(&t);
        bool same = s_frames > 0 && quick == s_frame_hash_prev;
        h = traced || logged || dumped ? hash_surface(&t) : 0;
        if (traced) {
            fprintf(stderr, "[KTRACE] f%llu complete: hash %08X\n",
                    (unsigned long long)n, h);
            s_trace_draw = 0;
        }
        s_frames = n;
        if (!same) {
            s_frames_distinct++;
        }
        if (logged) {
            if (!frame_qpc_frequency) {
                LARGE_INTEGER frequency;
                QueryPerformanceFrequency(&frequency);
                frame_qpc_frequency = frequency.QuadPart;
            }
            /* flip_t0 already marks completed raster work before hashing,
             * display and optional disk dumps. Reuse it for dump-free rates. */
            fprintf(stderr, "[KELVIN] frame %llu: surface 0x%08X %ux%u %s hash"
                            " %08X (%llu distinct so far) [qpc %lld freq %lld]\n",
                    (unsigned long long)s_frames, t.color_va, t.width, t.height,
                    same ? "same" : "new", h,
                    (unsigned long long)s_frames_distinct,
                    flip_t0, frame_qpc_frequency);
            fflush(stderr);
        }
        s_frame_hash_prev = quick;
    }

    /* Raster/geometry work was drained above. Publish an owned linear image
     * before the guest can reuse this target; UI work never reads guest RAM. */
    /* High-resolution scene pixels are already laid out at the requested
     * output aspect. Preserve explicit per-frame movie aspect tags, while
     * avoiding a second anamorphic reinterpretation by presentation policy. */
    if (t.host_color && !(aspect_w && aspect_h)) {
        aspect_w = t.width; aspect_h = t.height;
    }
    xbox_FramebufferPublishAspect(t.color, (uint32_t)(t.color_pitch * t.height),
        t.color_va, t.width, t.height, t.color_pitch, pg->surface_shape.color_format,
        aspect_w, aspect_h);
    xbox_FramebufferWindowStart();

    if (dump && *dump && (s_frames % every) == 0) {
        char path[512];
        snprintf(path, sizeof(path), "%s%05llu_%08X.bmp", dump,
                 (unsigned long long)s_frames, h);
        dump_bmp(path, &t);
    }
    kelvin_render_trace_event(7, flip_t0, jt_ticks(), 0);
}

/* ---- vertex fetch (gl/vertex.c pgraph_gl_bind_vertex_attributes) --------- */

typedef enum { SRC_ARRAYS, SRC_INLINE_ARRAY, SRC_INLINE_BUFFER } SrcKind;

typedef struct DrawSource {
    SrcKind kind;
    uint32_t attr_va[NV2A_VERTEXSHADER_ATTRIBUTES];
    unsigned int vertex_size;          /* inline array */
    /* fetch_vertex's plan (plan_fetch): every attribute's value when it is
     * not read per vertex, and the attributes that are. */
    bool planned;
    int nactive;
    uint8_t active[NV2A_VERTEXSHADER_ATTRIBUTES];
    float tmpl[NV2A_VERTEXSHADER_ATTRIBUTES][4];
    /* What the plan reads per vertex: pg's attributes and inline array, or
     * a draw job's copies of them (geom_capture). */
    const VertexAttribute *attrs;
    const uint32_t *inline_array;
    unsigned int inline_array_length;  /* words */
} DrawSource;

static void decode_attr(const VertexAttribute *a, const uint8_t *p, float out[4])
{
    unsigned int i;
    out[0] = out[1] = out[2] = 0.0f;
    out[3] = 1.0f;
    switch (a->format) {
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_D3D:
        /* GL_BGRA: a D3DCOLOR's bytes are B, G, R, A. */
        out[0] = p[2] / 255.0f;
        out[1] = p[1] / 255.0f;
        out[2] = p[0] / 255.0f;
        out[3] = p[3] / 255.0f;
        return;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_OGL:
        for (i = 0; i < a->count && i < 4; i++) out[i] = p[i] / 255.0f;
        return;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S1:
        for (i = 0; i < a->count && i < 4; i++) {
            int16_t v;
            memcpy(&v, p + i * 2, 2);
            out[i] = MAX(-1.0f, (float)v / 32767.0f);
        }
        return;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_F:
        for (i = 0; i < a->count && i < 4; i++) memcpy(&out[i], p + i * 4, 4);
        return;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S32K:
        for (i = 0; i < a->count && i < 4; i++) {
            int16_t v;
            memcpy(&v, p + i * 2, 2);
            out[i] = (float)v;
        }
        return;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_CMP: {
        int32_t v, x, y, z;
        memcpy(&v, p, 4);
        x = v & 0x7FF; if (x & 0x400) x |= (int32_t)0xFFFFF800;
        y = (v >> 11) & 0x7FF; if (y & 0x400) y |= (int32_t)0xFFFFF800;
        z = (v >> 22) & 0x3FF; if (z & 0x200) z |= (int32_t)0xFFFFFC00;
        out[0] = (float)x / 1023.0f;
        out[1] = (float)y / 1023.0f;
        out[2] = (float)z / 511.0f;
        return;
    }
    default:
        return;
    }
}

/* Work out once per draw what fetch_vertex does for each attribute that does
 * not depend on the vertex: the attributes read from memory per vertex, and
 * the inline value every other one takes. */
static void plan_fetch(KelvinState *pg, DrawSource *src)
{
    int i;
    src->attrs = pg->vertex_attributes;
    src->inline_array = pg->inline_array;
    src->inline_array_length = pg->inline_array_length;
    src->nactive = 0;
    for (i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        const VertexAttribute *a = &pg->vertex_attributes[i];
        bool per_vertex;
        memcpy(src->tmpl[i], a->inline_value, 16);
        if (src->kind == SRC_INLINE_BUFFER) {
            per_vertex = a->inline_buffer_populated;
        } else if (src->kind == SRC_INLINE_ARRAY) {
            per_vertex = a->count != 0;
        } else {
            per_vertex = a->count != 0 && src->attr_va[i] != 0;
        }
        if (per_vertex) {
            src->active[src->nactive++] = (uint8_t)i;
        }
    }
    src->planned = true;
}

static void fetch_vertex(KelvinState *pg, const DrawSource *src, uint32_t index,
                         float in[NV2A_VERTEXSHADER_ATTRIBUTES][4])
{
    int i;
    if (src->planned) {
        /* As below, per attribute: the inline value, or what the vertex
         * reads -- or the inline value again if that is out of range. */
        int j;
        memcpy(in, src->tmpl, sizeof(src->tmpl));
        for (j = 0; j < src->nactive; j++) {
            const VertexAttribute *a;
            i = src->active[j];
            a = &src->attrs[i];
            if (src->kind == SRC_INLINE_BUFFER) {
                memcpy(in[i], &a->inline_buffer[index * 4], 16);
            } else if (src->kind == SRC_INLINE_ARRAY) {
                size_t at = (size_t)index * src->vertex_size + a->inline_array_offset;
                if (at + a->size * a->count <= (size_t)src->inline_array_length * 4) {
                    decode_attr(a, (const uint8_t *)src->inline_array + at, in[i]);
                }
            } else {
                uint32_t va = src->attr_va[i];
                va += a->stride ? index * a->stride : 0;
                if (!(va < KELVIN_PHYS_WINDOW ||
                      va + a->size * a->count > KELVIN_PHYS_WINDOW + KELVIN_PHYS_SIZE)) {
                    decode_attr(a, kelvin_guest(va), in[i]);
                }
            }
        }
        return;
    }
    for (i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *a = &pg->vertex_attributes[i];
        if (src->kind == SRC_INLINE_BUFFER) {
            if (a->inline_buffer_populated) {
                memcpy(in[i], &a->inline_buffer[index * 4], 16);
            } else {
                memcpy(in[i], a->inline_value, 16);
            }
            continue;
        }
        if (!a->count) {
            memcpy(in[i], a->inline_value, 16);
            continue;
        }
        if (src->kind == SRC_INLINE_ARRAY) {
            size_t at = (size_t)index * src->vertex_size + a->inline_array_offset;
            if (at + a->size * a->count > (size_t)pg->inline_array_length * 4) {
                memcpy(in[i], a->inline_value, 16);
                continue;
            }
            decode_attr(a, (const uint8_t *)pg->inline_array + at, in[i]);
        } else {
            uint32_t va = src->attr_va[i];
            if (!va) {
                memcpy(in[i], a->inline_value, 16);
                continue;
            }
            va += a->stride ? index * a->stride : 0;
            if (va < KELVIN_PHYS_WINDOW ||
                va + a->size * a->count > KELVIN_PHYS_WINDOW + KELVIN_PHYS_SIZE) {
                memcpy(in[i], a->inline_value, 16);
                continue;
            }
            decode_attr(a, kelvin_guest(va), in[i]);
        }
    }
}

/* ---- rasterization ---------------------------------------------------------- */

typedef struct RVert {
    double x, y;         /* stored pixels */
    double lx, ly;       /* logical pixels (depth slope) */
    double z, w;
    /* The varyings, in KV order: a vertex's own (make_rvert points at its
     * KVertex), or `own` for one clipping made (lerp_rvert). */
    const float *var;
    float own[KV_VARYINGS];
} RVert;

/* A KVertex's varyings from d0 on are the KV order. */
typedef char kvertex_varyings_in_order[
    offsetof(KVertex, d1) - offsetof(KVertex, d0) == 4 * sizeof(float) &&
    offsetof(KVertex, b0) - offsetof(KVertex, d0) == 8 * sizeof(float) &&
    offsetof(KVertex, b1) - offsetof(KVertex, d0) == 12 * sizeof(float) &&
    offsetof(KVertex, fog) - offsetof(KVertex, d0) == 16 * sizeof(float) &&
    offsetof(KVertex, t) - offsetof(KVertex, d0) == 17 * sizeof(float) ? 1 : -1];

/* A triangle ready to scan: its edge functions, the depth and perspective
 * planes, its bounds, and what its fragments take from it. Set up once on the
 * GPU thread, then read by whichever raster threads own its rows. */
typedef struct TriSetup {
    double A[3], B[3], Cc[3];
    bool tl[3];
    double zx, zy, z0, qx, qy, q0;
    double px_[KV_VARYINGS], py_[KV_VARYINGS], p0_[KV_VARYINGS];
    int xmin, xmax, ymin, ymax;
    bool flat_shaded, front;
    float flat[16];
    double tri_mz;
} TriSetup;

/* Triangles one chunk of a draw set up in parallel (assemble_parallel). */
typedef struct TriChunk {
    TriSetup *tris;
    size_t n, cap;
    uint64_t triangles, clamped;     /* its share of the draw's stats */
    bool failed;                     /* out of memory */
} TriChunk;

typedef struct DrawCtx {
    KelvinState *pg;
    KelvinTarget t;
    KPixelState ps;
    bool cull_enable, ccw_front, provoking_first;
    int cull_ctrl;
    bool poly_offset, line_offset;
    float depth_offset, depth_factor;
    bool tex_needs_lod[4];
    /* Varyings no fragment of the draw reads, left uninterpolated
     * (skip_unread_varyings); all false interpolates every one. */
    bool var_skip[KV_VARYINGS];
    TriChunk *chunk;             /* set up into this instead (assemble_parallel) */
    TriSetup *tris;              /* set up, not yet handed over (flush_tris) */
    size_t ntris;
} DrawCtx;


/* Initial front colours are only consumed through combiner inputs; texture
 * stages use the texture coordinate sets. An input that reads V1+R0 also
 * consumes V1. Count every stage's reads even when an earlier stage writes
 * the register, so this deliberately retains more than a full liveness
 * analysis would. */
static void combiner_color_reads(const KPixelState *ps, bool read[2])
{
    int i, k;
    read[0] = read[1] = false;
    for (i = 0; i < ps->num_stages; i++) {
        for (k = 0; k < 4; k++) {
            int rgb = ps->stage[i].rgb[k].reg, alpha = ps->stage[i].alpha[k].reg;
            read[0] |= rgb == PS_REGISTER_V0 || alpha == PS_REGISTER_V0;
            read[1] |= rgb == PS_REGISTER_V1 || alpha == PS_REGISTER_V1 ||
                       rgb == PS_REGISTER_V1R0_SUM || alpha == PS_REGISTER_V1R0_SUM;
        }
    }
    if (ps->final_enabled) {
        for (k = 0; k < 7; k++) {
            int reg = ps->fin[k].reg;
            read[0] |= reg == PS_REGISTER_V0;
            read[1] |= reg == PS_REGISTER_V1 || reg == PS_REGISTER_V1R0_SUM;
        }
    }
}

/* Leave out of interpolation what kpix_fragment never consumes: the back
 * colours (it takes d0 and d1 whichever face), and the texture coordinates
 * of a stage that neither samples nor passes them on nor needs a LOD from
 * them. The dot-product reflection modes also read their neighbouring
 * stages' coordinates, so any of them keeps all four sets. */
static void skip_unread_varyings(DrawCtx *c)
{
    const KPixelState *ps = &c->ps;
    bool neighbours = false;
    bool color_read[2];
    int i, k;

    for (i = 0; i < 4; i++) {
        if (ps->tex_mode[i] == PS_TEXTUREMODES_DOT_RFLCT_DIFF ||
            ps->tex_mode[i] == PS_TEXTUREMODES_DOT_RFLCT_SPEC) {
            neighbours = true;
        }
    }
    combiner_color_reads(ps, color_read);
    for (k = 0; k < 8; k++) {
        c->var_skip[k] = !color_read[k / 4];
    }
    for (k = 8; k < 16; k++) {
        c->var_skip[k] = true;                  /* b0, b1 */
    }
    for (i = 0; i < 4; i++) {
        bool read = neighbours || c->tex_needs_lod[i] ||
                    ps->tex_mode[i] != PS_TEXTUREMODES_NONE;
        for (k = 0; k < 4; k++) {
            c->var_skip[17 + 4 * i + k] = !read;
        }
    }
}

static void make_rvert(const DrawCtx *c, const KVertex *v, RVert *r)
{
    r->lx = v->pos[0];
    r->ly = v->pos[1];
    r->x = v->pos[0] * kelvin_raster_x(&c->t);
    r->y = v->pos[1] * kelvin_raster_y(&c->t);
    r->z = v->pos[2];
    r->w = v->pos[3];
    r->var = v->d0;
}

/* One varying's perspective planes, as the scan evaluates them. */
KELVIN_INLINE void plane1(TriSetup *s, const RVert *const v[3],
                          const double iw[3], double area, int k)
{
    double sx = 0, sy = 0, s0 = 0;
    int i;
    for (i = 0; i < 3; i++) {
        double f = v[i]->var[k] * iw[i];
        sx += f * s->A[i];
        sy += f * s->B[i];
        s0 += f * s->Cc[i];
    }
    s->px_[k] = sx / area;
    s->py_[k] = sy / area;
    s->p0_[k] = s0 / area;
}

/* plane1 for varyings k..k+3, in each lane the same operations. */
KELVIN_INLINE void plane4(TriSetup *s, const RVert *const v[3],
                          const double iw[3], double area, int k)
{
#if defined(__AVX__)
    __m256d sx = _mm256_setzero_pd(), sy = sx, s0 = sx, a = _mm256_set1_pd(area);
    int i;
    for (i = 0; i < 3; i++) {
        __m256d f = _mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(&v[i]->var[k])),
                                  _mm256_set1_pd(iw[i]));
        sx = _mm256_add_pd(sx, _mm256_mul_pd(f, _mm256_set1_pd(s->A[i])));
        sy = _mm256_add_pd(sy, _mm256_mul_pd(f, _mm256_set1_pd(s->B[i])));
        s0 = _mm256_add_pd(s0, _mm256_mul_pd(f, _mm256_set1_pd(s->Cc[i])));
    }
    _mm256_storeu_pd(&s->px_[k], _mm256_div_pd(sx, a));
    _mm256_storeu_pd(&s->py_[k], _mm256_div_pd(sy, a));
    _mm256_storeu_pd(&s->p0_[k], _mm256_div_pd(s0, a));
#else
    int j;
    for (j = 0; j < 4; j++) {
        plane1(s, v, iw, area, k + j);
    }
#endif
}

/* Set up one triangle whose vertices all have w > 0. `flat` holds the
 * provoking vertex's colours when shading is flat. False if its bounding
 * box inside the scissor is empty. */
static bool tri_setup(DrawCtx *c, const RVert *v0, const RVert *v1,
                      const RVert *v2, const float *flat, bool front,
                      double tri_mz, TriSetup *s)
{
    const RVert *v[3] = { v0, v1, v2 };
    double area = (v1->x - v0->x) * (v2->y - v0->y) -
                  (v2->x - v0->x) * (v1->y - v0->y);
    double iw[3];
    double min_x, max_x, min_y, max_y;
    int xmin, xmax, ymin, ymax, k, i;

    if (area == 0.0 || !isfinite(area)) {
        return false;
    }
    if (area < 0.0) {
        const RVert *tmp = v[1];
        v[1] = v[2];
        v[2] = tmp;
        area = -area;
    }

    /* Edge i is opposite vertex i: E_i(p) = A_i x + B_i y + C_i. */
    for (i = 0; i < 3; i++) {
        const RVert *a = v[(i + 1) % 3], *b = v[(i + 2) % 3];
        double dx = b->x - a->x, dy = b->y - a->y;
        s->A[i] = -dy;
        s->B[i] = dx;
        s->Cc[i] = dy * a->x - dx * a->y;
        s->tl[i] = (dy < 0.0) || (dy == 0.0 && dx > 0.0);
        iw[i] = 1.0 / v[i]->w;
    }

    /* Screen-linear depth plane, and the perspective planes Q = sum l_i/w_i
     * and P_k = sum var_k l_i / w_i. */
    s->zx = s->zy = s->z0 = s->qx = s->qy = s->q0 = 0.0;
    for (i = 0; i < 3; i++) {
        s->zx += v[i]->z * s->A[i];
        s->zy += v[i]->z * s->B[i];
        s->z0 += v[i]->z * s->Cc[i];
        s->qx += iw[i] * s->A[i];
        s->qy += iw[i] * s->B[i];
        s->q0 += iw[i] * s->Cc[i];
    }
    s->zx /= area; s->zy /= area; s->z0 /= area;
    s->qx /= area; s->qy /= area; s->q0 /= area;
    /* P_k = sum over the vertices of var_k / w_i times the edge terms,
     * summed from 0 in vertex order: whole groups of four (a colour, a
     * texture coordinate set) at a time, each lane that sum. */
    {
        static const int group_start[8] = { 0, 4, 8, 12, 17, 21, 25, 29 };
        int g;
        for (g = 0; g < 8; g++) {
            int k0 = group_start[g];
            if (!c->var_skip[k0] && !c->var_skip[k0 + 1] &&
                !c->var_skip[k0 + 2] && !c->var_skip[k0 + 3]) {
                plane4(s, v, iw, area, k0);
                continue;
            }
            for (k = k0; k < k0 + 4; k++) {
                if (!c->var_skip[k]) {
                    plane1(s, v, iw, area, k);
                }
            }
        }
        if (!c->var_skip[16]) {
            plane1(s, v, iw, area, 16);
        }
    }

    min_x = floor(MIN(MIN(v[0]->x, v[1]->x), v[2]->x));
    max_x = ceil(MAX(MAX(v[0]->x, v[1]->x), v[2]->x));
    min_y = floor(MIN(MIN(v[0]->y, v[1]->y), v[2]->y));
    max_y = ceil(MAX(MAX(v[0]->y, v[1]->y), v[2]->y));
    if (min_x < INT_MIN || max_x > INT_MAX ||
        min_y < INT_MIN || max_y > INT_MAX) {
        if (c->chunk) {
            c->chunk->clamped++;
        } else {
            c->pg->stats.triangles_bounds_clamped++;
        }
    }
    /* Eye-plane clipping can produce finite projected coordinates far beyond
     * int's range. Clamp in double first: converting a large positive maximum
     * to int can otherwise turn it negative and discard a visible triangle. */
    xmin = (int)MAX(c->t.clip_x0, MIN(c->t.clip_x1, min_x));
    xmax = (int)MAX(c->t.clip_x0, MIN(c->t.clip_x1, max_x));
    ymin = (int)MAX(c->t.clip_y0, MIN(c->t.clip_y1, min_y));
    ymax = (int)MAX(c->t.clip_y0, MIN(c->t.clip_y1, max_y));
    if (xmin >= xmax || ymin >= ymax) {
        return false;
    }
    s->xmin = xmin;
    s->xmax = xmax;
    s->ymin = ymin;
    s->ymax = ymax;
    s->flat_shaded = flat != NULL;
    if (flat) {
        memcpy(s->flat, flat, sizeof(s->flat));
    }
    s->front = front;
    s->tri_mz = tri_mz;
    if (c->chunk) {
        c->chunk->triangles++;
    } else {
        c->pg->stats.triangles++;
    }
    return true;
}

/* Whether column x of the row whose centres are at pyc is inside edge i:
 * tri_scan's test, rounded the same way. */
static bool edge_in(const TriSetup *s, int i, int x, double pyc)
{
    double pxc = x + 0.5;
    double e = s->A[i] * pxc + s->B[i] * pyc + s->Cc[i];
    return !(e < 0.0 || (e == 0.0 && !s->tl[i]));
}

/* The first column in [lo, hi) where edge_in is `want`, given that it is
 * not before that column and is from there on; hi if none. Searches out
 * from `guess`, then halves. */
static int first_column(const TriSetup *s, int i, double pyc, bool want,
                        int lo, int hi, double guess)
{
    int g, b, step;

    if (!(guess >= (double)lo)) {           /* also NaN */
        g = lo;
    } else if (!(guess < (double)(hi - 1))) {
        g = hi - 1;
    } else {
        g = (int)guess;
    }
    if (edge_in(s, i, g, pyc) == want) {
        hi = g;
        for (step = 1; (b = g - step) >= lo; step *= 2) {
            if (edge_in(s, i, b, pyc) != want) {
                lo = b + 1;
                break;
            }
            hi = b;
        }
    } else {
        lo = g + 1;
        for (step = 1; (b = g + step) < hi; step *= 2) {
            if (edge_in(s, i, b, pyc) == want) {
                hi = b;
                break;
            }
            lo = b + 1;
        }
    }
    while (lo < hi) {
        int m = lo + (hi - lo) / 2;
        if (edge_in(s, i, m, pyc) == want) {
            hi = m;
        } else {
            lo = m + 1;
        }
    }
    return lo;
}

/* Narrow [*lo, *hi) to the columns of row pyc inside all three edges. With
 * finite coefficients an edge's value, as rounded, never decreases along
 * the row when A > 0 and never increases when A < 0 (each rounding step is
 * monotonic), so the columns inside one edge are one run: out then in, or
 * in then out. This finds where each run ends instead of testing every
 * column of the bounds; the columns kept are exactly those the test
 * passes. */
static void row_span(const TriSetup *s, double pyc, int *lo, int *hi)
{
    int i;
    for (i = 0; i < 3 && *lo < *hi; i++) {
        double a = s->A[i];
        double guess;
        if (a == 0.0) {
            if (!edge_in(s, i, *lo, pyc)) {
                *hi = *lo;
            }
            continue;
        }
        guess = -(s->B[i] * pyc + s->Cc[i]) / a - 0.5;
        if (a > 0.0) {
            *lo = first_column(s, i, pyc, true, *lo, *hi, guess);
        } else {
            *hi = first_column(s, i, pyc, false, *lo, *hi, guess);
        }
    }
}

/* Rows at least this wide are searched for their span (row_span); narrower
 * ones are tested column by column. Tests may set it to compare the two. */
static int s_span_min_width = 8;

/* A fragment's varyings, in KV order, are its first KV_VARYINGS floats:
 * tri_scan interpolates straight into them. */
typedef char kfragment_varyings_in_order[
    offsetof(KFragment, d1) == 4 * sizeof(float) &&
    offsetof(KFragment, fog) == 16 * sizeof(float) &&
    offsetof(KFragment, t) == 17 * sizeof(float) &&
    offsetof(KFragment, dtdx) == KV_VARYINGS * sizeof(float) ? 1 : -1];

/* Four varyings from k: (p0 + px * pxc + py * pyc) * iq, each as the scalar
 * expression rounds it, then to float. */
KELVIN_INLINE void interp4(const TriSetup *s, int k, double pxc, double pyc,
                           double iq, float *out)
{
#if defined(__AVX__)
    __m256d v = _mm256_add_pd(
        _mm256_add_pd(_mm256_loadu_pd(&s->p0_[k]),
                      _mm256_mul_pd(_mm256_loadu_pd(&s->px_[k]), _mm256_set1_pd(pxc))),
        _mm256_mul_pd(_mm256_loadu_pd(&s->py_[k]), _mm256_set1_pd(pyc)));
    _mm_storeu_ps(out, _mm256_cvtpd_ps(_mm256_mul_pd(v, _mm256_set1_pd(iq))));
#else
    __m128d x = _mm_set1_pd(pxc), y = _mm_set1_pd(pyc), q = _mm_set1_pd(iq);
    __m128d lo = _mm_add_pd(_mm_add_pd(_mm_loadu_pd(&s->p0_[k]),
                                       _mm_mul_pd(_mm_loadu_pd(&s->px_[k]), x)),
                            _mm_mul_pd(_mm_loadu_pd(&s->py_[k]), y));
    __m128d hi = _mm_add_pd(_mm_add_pd(_mm_loadu_pd(&s->p0_[k + 2]),
                                       _mm_mul_pd(_mm_loadu_pd(&s->px_[k + 2]), x)),
                            _mm_mul_pd(_mm_loadu_pd(&s->py_[k + 2]), y));
    _mm_storeu_ps(out, _mm_movelh_ps(_mm_cvtpd_ps(_mm_mul_pd(lo, q)),
                                     _mm_cvtpd_ps(_mm_mul_pd(hi, q))));
#endif
}

/* A texture coordinate set's screen derivatives, dA/dx = (P_x - A Q_x) / Q
 * and likewise in y, from its four interpolated values a. */
KELVIN_INLINE void derivs4(const TriSetup *s, int k, const float *a, double iq,
                           float *dx, float *dy)
{
#if defined(__AVX__)
    __m256d av = _mm256_cvtps_pd(_mm_loadu_ps(a)), q = _mm256_set1_pd(iq);
    _mm_storeu_ps(dx, _mm256_cvtpd_ps(_mm256_mul_pd(
        _mm256_sub_pd(_mm256_loadu_pd(&s->px_[k]),
                      _mm256_mul_pd(av, _mm256_set1_pd(s->qx))), q)));
    _mm_storeu_ps(dy, _mm256_cvtpd_ps(_mm256_mul_pd(
        _mm256_sub_pd(_mm256_loadu_pd(&s->py_[k]),
                      _mm256_mul_pd(av, _mm256_set1_pd(s->qy))), q)));
#else
    int j;
    for (j = 0; j < 4; j++) {
        double v = a[j];
        dx[j] = (float)((s->px_[k + j] - v * s->qx) * iq);
        dy[j] = (float)((s->py_[k + j] - v * s->qy) * iq);
    }
#endif
}

/* Shade rows [y0, y1) of a set-up triangle. Reads only the draw's state and
 * writes only those rows' pixels, so raster threads owning different rows
 * can scan the same triangle at once. Returns the pixels written. */
/* Integer-only interpolation lists. DrawCtx/TriSetup stay capture-compatible.
 * A batch owns both variants until its queued callbacks have completed. */
typedef struct TriInterpPlan {
    int groups[8], single[KV_VARYINGS], lod[4];
    int ngroups, nsingle, nlod;
} TriInterpPlan;

static void prepare_interp_plan(const DrawCtx *c, bool flat_shaded,
                                TriInterpPlan *plan)
{
    static const int group_start[8] = { 0, 4, 8, 12, 17, 21, 25, 29 };
    int i, k;
    plan->ngroups = plan->nsingle = plan->nlod = 0;
    for (i = 0; i < 8; i++) {
        int k0 = group_start[i];
        bool all = true, none = true;
        if (flat_shaded && k0 < 16) {
            continue;
        }
        for (k = k0; k < k0 + 4; k++) {
            all &= !c->var_skip[k];
            none &= c->var_skip[k];
        }
        if (all) {
            plan->groups[plan->ngroups++] = k0;
        } else if (!none) {
            for (k = k0; k < k0 + 4; k++) {
                if (!c->var_skip[k]) plan->single[plan->nsingle++] = k;
            }
        }
    }
    if (!c->var_skip[16]) {
        plan->single[plan->nsingle++] = 16;
    }
    for (i = 0; i < 4; i++) {
        if (c->tex_needs_lod[i]) {
            plan->lod[plan->nlod++] = i;
        }
    }
}

static void prepare_interp_pair(const DrawCtx *c, TriInterpPlan plans[2])
{
    prepare_interp_plan(c, false, &plans[0]);
    prepare_interp_plan(c, true, &plans[1]);
}

static uint64_t tri_scan_prepared(const DrawCtx *c, const TriSetup *s, int y0, int y1,
                                  const TriInterpPlan *plan, KPixelProgram program)
{
    const KPixelState *ps = &c->ps;
    const float *flat = s->flat_shaded ? s->flat : NULL;
    uint64_t written = 0, scanned = 0, inside_count = 0;
    int x, y, k, i, j;
    const int ngroups = plan->ngroups, nsingle = plan->nsingle, nlod = plan->nlod;
    const int *groups = plan->groups, *single = plan->single, *lod = plan->lod;
    float *fv;
    KFragment f, blk[8], *fp;
    float zvs[8];
    int nb = 0, ninit = 0;
    bool spans = s->xmax - s->xmin >= s_span_min_width;

    for (i = 0; i < 3; i++) {
        if (!isfinite(s->A[i]) || !isfinite(s->B[i]) || !isfinite(s->Cc[i])) {
            spans = false;
        }
    }
    /* What every fragment of the triangle shares: varyings never
     * interpolated stay 0, as do the derivatives of stages without a LOD,
     * and flat shading's colours replace the interpolated ones. kpix_fragment
     * does not write the fragment. */
    memset(&f, 0, sizeof(f));
    fv = f.d0;
    if (flat) {
        memcpy(fv, flat, 16 * sizeof(float));
    }
    f.front = s->front;
    if (y0 < s->ymin) y0 = s->ymin;
    if (y1 > s->ymax) y1 = s->ymax;
    for (y = y0; y < y1; y++) {
        double pyc = y + 0.5;
        int x0 = s->xmin, x1 = s->xmax;
        if (spans) {
            row_span(s, pyc, &x0, &x1);
        }
        if (x1 > x0) {
            scanned += (uint64_t)(x1 - x0);
        }
        for (x = x0; x < x1; x++) {
            double pxc = x + 0.5;
            double q, z;
            float zv;
            bool inside = true;
            for (i = 0; i < 3 && !spans; i++) {
                double e = s->A[i] * pxc + s->B[i] * pyc + s->Cc[i];
                if (e < 0.0 || (e == 0.0 && !s->tl[i])) {
                    inside = false;
                    break;
                }
            }
            if (!inside) {
                continue;
            }
            q = s->q0 + s->qx * pxc + s->qy * pyc;
            if (q == 0.0) {
                continue;
            }
            {
                double iq = 1.0 / q;
                if (ps->z_perspective) {
                    z = iq;
                    if (!(z > 0.0) || !isfinite(z)) {
                        z = 3.402823466e38;
                    } else {
                        z += c->depth_offset + c->depth_factor * s->tri_mz * z * z;
                    }
                } else {
                    z = s->z0 + s->zx * pxc + s->zy * pyc;
                    z += c->depth_offset + c->depth_factor * s->tri_mz;
                }
                f.z = (float)z;
                inside_count++;
                /* A fragment the depth test (or clip) drops before shading
                 * needs no varyings. */
                zv = f.z;
                if (!kpix_keep(ps, &c->t, x, y, &zv)) {
                    continue;
                }
                /* Into the next of a block of eight, shaded together
                 * (kpix_shade8); a slot is set up from f when first used,
                 * and each fragment then writes the same fields. */
                if (nb == ninit) {
                    blk[ninit++] = f;
                }
                fp = &blk[nb];
                fv = fp->d0;
                for (j = 0; j < ngroups; j++) {
                    interp4(s, groups[j], pxc, pyc, iq, fv + groups[j]);
                }
                for (j = 0; j < nsingle; j++) {
                    k = single[j];
                    fv[k] = (float)((s->p0_[k] + s->px_[k] * pxc + s->py_[k] * pyc) * iq);
                }
                for (j = 0; j < nlod; j++) {
                    i = lod[j];
                    derivs4(s, 17 + i * 4, fp->t[i], iq, fp->dtdx[i], fp->dtdy[i]);
                }
                fp->z = f.z;
                fp->x = x;
                fp->y = y;
                zvs[nb++] = zv;
                if (nb == 8) {
                    written += (uint64_t)kpix_shade8_program(ps, &c->t, blk, zvs, nb, program);
                    nb = 0;
                }
            }
        }
    }
    if (nb) {
        written += (uint64_t)kpix_shade8_program(ps, &c->t, blk, zvs, nb, program);
    }
    if (g_kelvin_timing > 0) {
        _InterlockedExchangeAdd64(&g_kelvin_scan[0], (long long)scanned);
        _InterlockedExchangeAdd64(&g_kelvin_scan[1], (long long)inside_count);
        _InterlockedExchangeAdd64(&g_kelvin_scan[2], (long long)written);
    }
    return written;
}

/* Compatibility for direct private callers: prepare only this triangle's
 * integer lists. Batched scans use plans prepared once for their context. */
static uint64_t tri_scan(const DrawCtx *c, const TriSetup *s, int y0, int y1)
{
    TriInterpPlan plan;
    prepare_interp_plan(c, s->flat_shaded, &plan);
    return tri_scan_prepared(c, s, y0, y1, &plan, KPIX_PROGRAM_GENERIC);
}

/* ---- scanning a draw's triangles on the raster threads --------------------
 *
 * A batch's rows are cut into bands of KR_BAND, and each triangle is listed
 * in every band it touches, in triangle order. Band b is in worker lane
 * b % KWORK_LANES (kelvin_workers.h): one raster thread at a time scans a
 * lane, a batch's bands there after the same bands of every earlier batch,
 * each band's triangles in order. So each pixel sees the same writes in the
 * same order as on one thread, and the arithmetic per pixel is the same code
 * on the same inputs. Threads claim lanes rather than being dealt them,
 * which keeps every thread busy whatever the scene, and on a CPU whose cores
 * differ in speed. A batch whose rows share bytes (rows_shared) is scanned
 * by one raster thread, in triangle order, after every earlier batch and
 * before any later one.
 *
 * The GPU thread does not wait for the batches it hands over. It queues them
 * (kwork_submit_lanes) and goes on to the next draws' vertices. Lanes run
 * independently, so a later batch's rows may be drawn before an earlier
 * batch's other rows. That is safe where the two meet only in the same
 * pixels of a surface both lay out alike, which a lane keeps in order. A
 * batch that samples memory an unfinished batch draws to, draws to memory an
 * unfinished batch samples, or draws over one with another layout, waits for
 * every earlier batch instead (batch_conflicts); so does one that samples its
 * own target, for its snapshot. Anything else that touches surfaces, or
 * tells the title how far the GPU has got, calls kelvin_render_sync first,
 * which waits for the queue to empty: clears, blits, lines, flips,
 * semaphores, notifies, reports, WAIT_FOR_IDLE, and the pusher before it
 * reports GET at PUT. The GPU thread's own work on later draws reads vertex
 * data and registers, never surfaces.
 *
 * RECOMP_KELVIN_VERIFY_THREADS=1 checks the threads on the running title:
 * each batch they scan is scanned again on the GPU thread from the same
 * starting surfaces, and the two results are compared byte for byte. */

/* Only the producer reads the immutable option, before publishing a batch. */
static int s_band_rows;
static int raster_band_rows(void)
{
    if (!s_band_rows) {
        const char *e = getenv("RECOMP_KELVIN_BAND_ROWS");
        s_band_rows = e && strcmp(e, "2") == 0 ? 2 : 4;
        if (s_band_rows == 2) {
            fprintf(stderr, "[KELVIN] isolated two-row raster bands enabled\n");
        }
    }
    return s_band_rows;
}

#define KR_BAND 4
/* Triangles set up before they are handed over; bounds each batch's buffer
 * (about a kilobyte a triangle) for draws of any size. */
#define KR_BATCH 1024

/* A draw's triangles and everything scanning them reads: the draw's state
 * is copied in, because the draw is gone by the time the threads get to it. */
typedef struct RasterBatch {
    DrawCtx ctx;
    TriSetup *tris;
    size_t cap;
    /* Band b lists bin_items[bin_start[b]] up to bin_items[bin_start[b + 1]],
     * in triangle order. */
    uint32_t *bin_start, *bin_items;
    size_t bin_start_cap, bin_items_cap;
    int nbands;
    uint8_t lanes[KWORK_LANES];    /* the lanes its bands are in */
    int nlanes;
    long job;                      /* its number in the worker queue */
    bool single;                   /* rows share bytes: one thread, in order */
    bool counted;                  /* its pixels go to s_written */
    uint64_t written[KWORK_MAX];   /* by each worker */
    uint8_t *snap[4];              /* texture copies (snapshot_textures) */
    size_t snap_cap[4];
    int jt;                        /* its job trace record, or -1 */
    uint64_t lane_mask[2];         /* the lanes it lists, as bits */
    /* Its begin callback copies the textures overlapping its target
     * (snapshot_textures) into room already made for them: from then on it
     * reads those from the copies alone. */
    bool snapshotting;
    int band_rows; /* immutable producer snapshot; callback selected at submit */
    TriInterpPlan interpolation[2]; /* normal / flat; immutable after binning */
    KPixelProgram pixel_program; /* immutable selectors; retained through completion */
} RasterBatch;


/* A ring of batches: the one filling on the GPU thread, and up to
 * KWORK_QUEUE handed over. Batch n lives in slot n % KR_SLOTS; while it
 * fills, batch n - KR_SLOTS has finished, since at most KWORK_QUEUE wait. */
#define KR_SLOTS (KWORK_QUEUE + 1)
static RasterBatch s_batch[KR_SLOTS];
static unsigned long s_fill;
/* Pixels the raster threads wrote that kelvin_render_sync has not counted. */
static volatile __int64 s_written;
static KelvinState *s_pg;

static bool grow_u32(uint32_t **buf, size_t *cap, size_t need)
{
    size_t n = *cap ? *cap : 1024;
    uint32_t *grown;

    if (need <= *cap) {
        return true;
    }
    while (n < need) {
        n *= 2;
    }
    grown = (uint32_t *)realloc(*buf, n * sizeof(uint32_t));
    if (!grown) {
        return false;
    }
    *buf = grown;
    *cap = n;
    return true;
}

/* List each triangle in every band it touches, and the lanes of the bands
 * with any; false if out of memory. */
static bool bin_batch4(RasterBatch *rb)
{
    const DrawCtx *c = &rb->ctx;
    bool used[KWORK_LANES];
    int nbands = 0, b, l;
    size_t i, total;

    for (i = 0; i < c->ntris; i++) {
        int last = (c->tris[i].ymax - 1) / KR_BAND + 1;
        if (last > nbands) nbands = last;
    }
    if (!grow_u32(&rb->bin_start, &rb->bin_start_cap, (size_t)nbands + 2)) {
        return false;
    }
    memset(rb->bin_start, 0, ((size_t)nbands + 2) * sizeof(uint32_t));
    for (i = 0; i < c->ntris; i++) {
        const TriSetup *t = &c->tris[i];
        for (b = t->ymin / KR_BAND; b * KR_BAND < t->ymax; b++) {
            rb->bin_start[b + 2]++;
        }
    }
    /* Offsets shifted by one, so filling can use bin_start[b + 1] as band
     * b's cursor and leave it as band b + 1's start. */
    for (b = 0; b < nbands; b++) {
        rb->bin_start[b + 2] += rb->bin_start[b + 1];
    }
    total = rb->bin_start[nbands + 1];
    if (!grow_u32(&rb->bin_items, &rb->bin_items_cap, total ? total : 1)) {
        return false;
    }
    for (i = 0; i < c->ntris; i++) {
        const TriSetup *t = &c->tris[i];
        for (b = t->ymin / KR_BAND; b * KR_BAND < t->ymax; b++) {
            rb->bin_items[rb->bin_start[b + 1]++] = (uint32_t)i;
        }
    }
    rb->nbands = nbands;
    memset(used, 0, sizeof(used));
    for (b = 0; b < nbands; b++) {
        if (rb->bin_start[b + 1] > rb->bin_start[b]) {
            used[b % KWORK_LANES] = true;
        }
    }
    rb->nlanes = 0;
    rb->lane_mask[0] = rb->lane_mask[1] = 0;
    for (l = 0; l < KWORK_LANES; l++) {
        if (used[l]) {
            rb->lanes[rb->nlanes++] = (uint8_t)l;
            rb->lane_mask[l / 64] |= 1ull << (l % 64);
        }
    }
    memset(rb->written, 0, sizeof(rb->written));
    return true;
}

static bool bin_batch2(RasterBatch *rb)
{
    const DrawCtx *c = &rb->ctx;
    bool used[KWORK_LANES];
    int nbands = 0, b, l;
    size_t i, total;

    for (i = 0; i < c->ntris; i++) {
        int last = (c->tris[i].ymax - 1) / 2 + 1;
        if (last > nbands) nbands = last;
    }
    if (!grow_u32(&rb->bin_start, &rb->bin_start_cap, (size_t)nbands + 2)) {
        return false;
    }
    memset(rb->bin_start, 0, ((size_t)nbands + 2) * sizeof(uint32_t));
    for (i = 0; i < c->ntris; i++) {
        const TriSetup *t = &c->tris[i];
        for (b = t->ymin / 2; b * 2 < t->ymax; b++) {
            rb->bin_start[b + 2]++;
        }
    }
    /* Offsets shifted by one, so filling can use bin_start[b + 1] as band
     * b's cursor and leave it as band b + 1's start. */
    for (b = 0; b < nbands; b++) {
        rb->bin_start[b + 2] += rb->bin_start[b + 1];
    }
    total = rb->bin_start[nbands + 1];
    if (!grow_u32(&rb->bin_items, &rb->bin_items_cap, total ? total : 1)) {
        return false;
    }
    for (i = 0; i < c->ntris; i++) {
        const TriSetup *t = &c->tris[i];
        for (b = t->ymin / 2; b * 2 < t->ymax; b++) {
            rb->bin_items[rb->bin_start[b + 1]++] = (uint32_t)i;
        }
    }
    rb->nbands = nbands;
    memset(used, 0, sizeof(used));
    for (b = 0; b < nbands; b++) {
        if (rb->bin_start[b + 1] > rb->bin_start[b]) {
            used[b % KWORK_LANES] = true;
        }
    }
    rb->nlanes = 0;
    rb->lane_mask[0] = rb->lane_mask[1] = 0;
    for (l = 0; l < KWORK_LANES; l++) {
        if (used[l]) {
            rb->lanes[rb->nlanes++] = (uint8_t)l;
            rb->lane_mask[l / 64] |= 1ull << (l % 64);
        }
    }
    memset(rb->written, 0, sizeof(rb->written));
    return true;
}

/* Dispatch once per batch. The four-row body remains the frozen implementation,
 * and the two-row body differs only by its compile-time divisor/multiplier. */
static bool bin_batch(RasterBatch *rb)
{
    rb->pixel_program = kpix_program(&rb->ctx.ps);
    rb->band_rows = raster_band_rows();
    prepare_interp_pair(&rb->ctx, rb->interpolation);
    return rb->band_rows == 2 ? bin_batch2(rb) : bin_batch4(rb);
}

static uint64_t scan_serial(const RasterBatch *rb)
{
    const DrawCtx *c = &rb->ctx;
    uint64_t written = 0;
    size_t i;
    TriInterpPlan interpolation[2];

    /* Also covers shared-row and allocation-failure contexts that have no
     * published plan. Prepare once here, without requiring valid bins. */
    KPixelProgram program = kpix_program(&c->ps);
    prepare_interp_pair(c, interpolation);
    for (i = 0; i < c->ntris; i++) {
        written += tri_scan_prepared(c, &c->tris[i], c->tris[i].ymin, c->tris[i].ymax,
                                     &interpolation[c->tris[i].flat_shaded ? 1 : 0], program);
    }
    return written;
}

/* One lane of a batch: its bands in that lane, or the whole batch in
 * triangle order when its rows share bytes (it then has lane 0 alone). */
static void batch_lane4(void *arg, int lane, int worker)
{
    RasterBatch *rb = (RasterBatch *)arg;
    const DrawCtx *c = &rb->ctx;
    uint64_t written = 0;
    int b;

    long long lane_t0 = 0;

    if (rb->jt >= 0 && s_jt_on) {
        lane_t0 = jt_ticks();
        jt_min_nonzero(&s_jt[rb->jt].start, lane_t0);
    }
    if (rb->single) {
        written = scan_serial(rb);
    } else {
        for (b = lane; b < rb->nbands; b += KWORK_LANES) {
            uint32_t k;
            for (k = rb->bin_start[b]; k < rb->bin_start[b + 1]; k++) {
                const TriSetup *tri = &c->tris[rb->bin_items[k]];
                written += tri_scan_prepared(c, tri, b * KR_BAND, (b + 1) * KR_BAND,
                                             &rb->interpolation[tri->flat_shaded ? 1 : 0], rb->pixel_program);
            }
        }
    }
    /* Only this worker adds to its count, though other workers may be
     * scanning other lanes of the batch. */
    rb->written[worker] += written;
    if (rb->counted && written) {
        _InterlockedExchangeAdd64(&s_written, (__int64)written);
    }
    if (rb->jt >= 0 && s_jt_on) {
        long long now = jt_ticks();
        jt_max(&s_jt[rb->jt].end, now);
        _InterlockedExchangeAdd64(&s_jt[rb->jt].cpu, now - lane_t0);
        _InterlockedExchangeAdd64(&s_jt[rb->jt].written, (__int64)written);
    }
}

static void batch_lane2(void *arg, int lane, int worker)
{
    RasterBatch *rb = (RasterBatch *)arg;
    const DrawCtx *c = &rb->ctx;
    uint64_t written = 0;
    int b;

    long long lane_t0 = 0;

    if (rb->jt >= 0 && s_jt_on) {
        lane_t0 = jt_ticks();
        jt_min_nonzero(&s_jt[rb->jt].start, lane_t0);
    }
    if (rb->single) {
        written = scan_serial(rb);
    } else {
        for (b = lane; b < rb->nbands; b += KWORK_LANES) {
            uint32_t k;
            for (k = rb->bin_start[b]; k < rb->bin_start[b + 1]; k++) {
                const TriSetup *tri = &c->tris[rb->bin_items[k]];
                written += tri_scan_prepared(c, tri, b * 2, (b + 1) * 2,
                                             &rb->interpolation[tri->flat_shaded ? 1 : 0], rb->pixel_program);
            }
        }
    }
    /* Only this worker adds to its count, though other workers may be
     * scanning other lanes of the batch. */
    rb->written[worker] += written;
    if (rb->counted && written) {
        _InterlockedExchangeAdd64(&s_written, (__int64)written);
    }
    if (rb->jt >= 0 && s_jt_on) {
        long long now = jt_ticks();
        jt_max(&s_jt[rb->jt].end, now);
        _InterlockedExchangeAdd64(&s_jt[rb->jt].cpu, now - lane_t0);
        _InterlockedExchangeAdd64(&s_jt[rb->jt].written, (__int64)written);
    }
}

typedef void (*RasterLane)(void *, int, int);
static RasterLane batch_lane_for(const RasterBatch *rb)
{
    return rb->band_rows == 2 ? batch_lane2 : batch_lane4;
}

/* Private direct-call entry used by existing raster regression tests. Real
 * queue submissions select the constant callback once before publication. */
static void batch_lane(void *arg, int lane, int worker)
{
    batch_lane_for((const RasterBatch *)arg)(arg, lane, worker);
}

static uint64_t batch_written(const RasterBatch *rb)
{
    uint64_t written = 0;
    /* The submitter may scan lanes while draining, with worker number
     * kwork_workers(). Count its slot as well (slot 0 without workers). */
    int i, workers = kwork_workers() + 1;

    for (i = 0; i < workers; i++) {
        written += rb->written[i];
    }
    return written;
}

/* Scan on the raster threads and wait for them (the thread check). */
static uint64_t scan_parallel_now(RasterBatch *rb)
{
    if (!bin_batch(rb)) {
        return scan_serial(rb);
    }
    rb->single = false;
    rb->counted = false;
    kwork_submit_lanes(NULL, batch_lane_for(rb), rb, rb->lanes, rb->nlanes,
                       KWORK_BARRIER);
    kwork_drain();
    return batch_written(rb);
}

static size_t surface_bytes(const KelvinTarget *t, bool color)
{
    unsigned int bpp = color ? t->color_bpp : t->zeta_bpp;
    unsigned int pitch = color ? t->color_pitch : t->zeta_pitch;

    if (!(color ? t->color : t->zeta)) {
        return 0;
    }
    return t->swizzled ? (size_t)t->width * t->height * bpp
                       : (size_t)pitch * t->height;
}

/* Queue conflicts are expressed in guest allocations, never the much larger
 * host surface length; host addresses themselves stay immutable until drain. */
static size_t guest_surface_bytes(const KelvinTarget *t, bool color)
{
    size_t n = color ? t->guest_color_bytes : t->guest_zeta_bytes;
    return n ? n : surface_bytes(t, color);
}

static bool uses_color(const DrawCtx *c)
{
    return c->t.color && (c->ps.mask_r || c->ps.mask_g || c->ps.mask_b ||
                          c->ps.mask_a);
}

/* A zeta surface is bound whenever the title set one, but only a depth or
 * stencil test reaches it; BLACK leaves one at 0x80000000, where its own
 * data lives, while it renders small targets with both tests off. */
static bool uses_zeta(const DrawCtx *c)
{
    return c->t.zeta && (c->ps.depth_test || c->ps.stencil_test);
}

/* NULL when different threads may scan different rows and get what one
 * thread gets: no pixel reaches bytes that a pixel in another row reads or
 * writes; otherwise why not. Rows narrower than their pitch says, colour and
 * zeta surfaces that overlap, and a palette read from a surface the draw
 * writes all break that, and such a batch is scanned by one thread, in the
 * order the hardware would. A texture over such a surface is read from a
 * copy (snapshot_textures), so it breaks nothing. */
static const char *rows_shared(const DrawCtx *c)
{
    const KelvinTarget *t = &c->t;
    const KPixelState *ps = &c->ps;
    bool color = uses_color(c), zeta = uses_zeta(c);
    uint64_t cs = 0, ce = 0, zs = 0, ze = 0;
    int i;

    if (color) {
        if (!t->swizzled &&
            t->color_pitch < (unsigned int)t->clip_x1 * t->color_bpp) {
            return "colour rows overlap";
        }
        cs = t->color_va;
        ce = cs + guest_surface_bytes(t, true);
    }
    if (zeta) {
        if (!t->swizzled &&
            t->zeta_pitch < (unsigned int)t->clip_x1 * t->zeta_bpp) {
            return "zeta rows overlap";
        }
        zs = t->zeta_va;
        ze = zs + guest_surface_bytes(t, false);
    }
    if (color && zeta && cs < ze && zs < ce) {
        return "colour and zeta overlap";
    }
    for (i = 0; i < 4; i++) {
        const KTexture *x = &ps->tex[i];
        uint64_t s0, s1;
        if (!x->valid) {
            continue;
        }
        /* A texture over a surface the draw writes is read from a copy
         * (snapshot_textures), so it shares nothing; a palette is not. */
        if (x->palette) {
            s0 = x->palette;
            s1 = s0 + (uint64_t)x->palette_len * 4;
            if ((color && s0 < ce && cs < s1) ||
                (zeta && s0 < ze && zs < s1)) {
                return "a palette overlaps a surface it draws to";
            }
        }
    }
    return NULL;
}

/* Whether a texture's memory overlaps a surface the draw writes. */
static bool tex_overlaps_target(const DrawCtx *c, const KTexture *x)
{
    uint64_t s0 = x->base;
    uint64_t s1 = s0 + (uint64_t)(x->cubemap ? 6 : 1) * x->face_stride;

    if (uses_color(c)) {
        uint64_t cs = c->t.color_va, ce = cs + guest_surface_bytes(&c->t, true);
        if (s0 < ce && cs < s1) {
            return true;
        }
    }
    if (uses_zeta(c)) {
        uint64_t zs = c->t.zeta_va, ze = zs + guest_surface_bytes(&c->t, false);
        if (s0 < ze && zs < s1) {
            return true;
        }
    }
    return false;
}

static bool needs_snapshot(const DrawCtx *c)
{
    int i;
    for (i = 0; i < 4; i++) {
        if (c->ps.tex[i].valid && tex_overlaps_target(c, &c->ps.tex[i])) {
            return true;
        }
    }
    return false;
}

/* ---- batches that may overlap on the raster threads -------------------- */

typedef struct MemSpan {
    uint64_t start, end;
} MemSpan;

static bool spans_meet(const MemSpan *a, const MemSpan *b)
{
    return a->start < b->end && b->start < a->end;
}

/* The colour (or zeta) surface a batch's pixels read and write, if any. */
static bool drawn_span(const DrawCtx *c, bool color, MemSpan *s)
{
    if (!(color ? uses_color(c) : uses_zeta(c))) {
        return false;
    }
    s->start = color ? c->t.color_va : c->t.zeta_va;
    s->end = s->start + guest_surface_bytes(&c->t, color);
    return true;
}

/* Every texture and palette a batch may sample. A texture it copies counts
 * too, since the copy is made when the batch starts -- unless `copied`:
 * then the textures its begin callback copies (those overlapping its
 * target, snapshot_textures) are left out, as they are read from memory
 * only before the batch's lanes run. */
static int sampled_spans(const DrawCtx *c, MemSpan s[8], bool copied)
{
    int i, n = 0;

    for (i = 0; i < 4; i++) {
        const KTexture *x = &c->ps.tex[i];
        if (!x->valid) {
            continue;
        }
        if (!(copied && tex_overlaps_target(c, x))) {
            s[n].start = x->base;
            s[n].end = x->base + (uint64_t)(x->cubemap ? 6 : 1) * x->face_stride;
            n++;
        }
        if (x->palette) {
            s[n].start = x->palette;
            s[n].end = x->palette + (uint64_t)x->palette_len * 4;
            n++;
        }
    }
    return n;
}

/* Whether two batches put each pixel of their colour (or zeta) surfaces at
 * the same bytes. Each batch's pixels are at distinct bytes (rows_shared),
 * so a byte both touch is the same pixel, in the same row and lane. */
static bool same_layout(const KelvinTarget *a, const KelvinTarget *b,
                        bool color)
{
    if (a->swizzled != b->swizzled) {
        return false;
    }
    if (color ? a->color_va != b->color_va || a->color_bpp != b->color_bpp
              : a->zeta_va != b->zeta_va || a->zeta_bpp != b->zeta_bpp) {
        return false;
    }
    if (a->swizzled) {
        return a->width == b->width && a->height == b->height;
    }
    return color ? a->color_pitch == b->color_pitch
                 : a->zeta_pitch == b->zeta_pitch;
}

/* Whether a later batch must wait for an earlier, unfinished one to finish,
 * rather than follow it lane by lane: it samples what the earlier batch
 * draws, draws what the earlier batch samples, or draws over the earlier
 * batch's surfaces laid out differently or in their other role. */
static bool batch_conflicts(const DrawCtx *early, const DrawCtx *late,
                            bool early_copied)
{
    MemSpan ew[2], lw[2], er[8], lr[8];
    bool ehas[2], lhas[2];
    int ne, nl, i, j;

    for (i = 0; i < 2; i++) {
        ehas[i] = drawn_span(early, i == 0, &ew[i]);
        lhas[i] = drawn_span(late, i == 0, &lw[i]);
    }
    ne = sampled_spans(early, er, early_copied);
    nl = sampled_spans(late, lr, false);
    for (i = 0; i < 2; i++) {
        if (!lhas[i]) {
            continue;
        }
        for (j = 0; j < 2; j++) {
            if (ehas[j] && spans_meet(&lw[i], &ew[j]) &&
                (i != j || !same_layout(&early->t, &late->t, i == 0))) {
                return true;
            }
        }
        for (j = 0; j < ne; j++) {
            if (spans_meet(&lw[i], &er[j])) {
                return true;
            }
        }
    }
    for (i = 0; i < nl; i++) {
        for (j = 0; j < 2; j++) {
            if (ehas[j] && spans_meet(&lr[i], &ew[j])) {
                return true;
            }
        }
    }
    return false;
}

/* Whether a batch about to be handed over conflicts with any handed over
 * before it that has not finished. */
static bool conflicts_in_flight(const RasterBatch *rb)
{
    long finished = kwork_finished();
    unsigned long n;

    for (n = 1; n <= KWORK_QUEUE && n <= s_fill; n++) {
        const RasterBatch *e = &s_batch[(s_fill - n) % KR_SLOTS];
        if ((long)((unsigned long)e->job - (unsigned long)finished) < 0) {
            break;            /* it, and every batch before it, has finished */
        }
        if (e->single) {
            return true;
        }
        /* A batch that draws where an earlier one's snapshot comes from
         * need not wait for it to finish, only for the copy, if each of its
         * lanes is one of the earlier batch's: a lane runs there only after
         * the earlier batch's lane, which runs only after its begin. */
        if (batch_conflicts(&e->ctx, &rb->ctx, false) &&
            !(e->snapshotting &&
              !(rb->lane_mask[0] & ~e->lane_mask[0]) &&
              !(rb->lane_mask[1] & ~e->lane_mask[1]) &&
              !batch_conflicts(&e->ctx, &rb->ctx, true))) {
            return true;
        }
    }
    return false;
}

/* Large feedback copies must not accumulate once per ring slot forever.
 * Only the producer reclaims buffers, after their jobs have completed. */
static bool snapshot_room(RasterBatch *rb, int stage, size_t size)
{
    size_t total = 0;
    unsigned int n;
    int i;
    uint8_t *grown;
    if (size <= rb->snap_cap[stage]) return true;
    if (kres_enabled()) {
        for (n = 0; n < KR_SLOTS; n++) {
            for (i = 0; i < 4; i++) total += s_batch[n].snap_cap[i];
        }
        if (size - rb->snap_cap[stage] > KRES_SNAPSHOT_LIMIT - MIN(total, KRES_SNAPSHOT_LIMIT)) {
            /* The current batch has not been submitted. Draining raster
             * jobs is sufficient; deferred geometry owns no snapshots. */
            kwork_drain();
            for (n = 0; n < KR_SLOTS; n++) {
                if (&s_batch[n] == rb) continue;
                for (i = 0; i < 4; i++) {
                    free(s_batch[n].snap[i]); s_batch[n].snap[i] = NULL;
                    s_batch[n].snap_cap[i] = 0;
                }
            }
            total = 0;
            for (i = 0; i < 4; i++) total += rb->snap_cap[i];
            if (size - rb->snap_cap[stage] > KRES_SNAPSHOT_LIMIT - MIN(total, KRES_SNAPSHOT_LIMIT)) {
                fprintf(stderr, "[KRES] feedback snapshot exceeds 256 MiB budget\n");
                return false;
            }
        }
    }
    grown = (uint8_t *)realloc(rb->snap[stage], size);
    if (!grown) return false;
    rb->snap[stage] = grown; rb->snap_cap[stage] = size;
    return true;
}

/* Give each texture that overlaps a surface the batch writes a copy of its
 * memory as it is now -- after every earlier batch, before this one -- and
 * sample that, so the draw sees the surface as it was before the draw
 * whichever thread scans which row. Called once the batch before it is
 * done. BLACK's full-screen passes sample the frame they draw to. With
 * `all`, every texture gets a copy (the thread check, below). */
static void snapshot_textures(RasterBatch *rb, bool all)
{
    int i;

    for (i = 0; i < 4; i++) {
        KTexture *x = &rb->ctx.ps.tex[i];
        uint64_t size, end = (uint64_t)KELVIN_PHYS_WINDOW + KELVIN_PHYS_SIZE;

        x->host = NULL;
        if (!x->valid || (!all && !tex_overlaps_target(&rb->ctx, x))) {
            continue;
        }
        size = x->host_bytes ? x->host_bytes : (uint64_t)(x->cubemap ? 6 : 1) * x->face_stride;
        if (x->base < KELVIN_PHYS_WINDOW || x->base >= end) {
            continue;
        }
        if (!x->host_bytes && x->base + size > end) {
            size = end - x->base;
        }
        if (size > rb->snap_cap[i]) {
            /* Host feedback was reserved on the producer before submission;
             * callbacks never reclaim memory or drain their own queue. */
            if (x->host_bytes || !snapshot_room(rb, i, (size_t)size)) {
                fprintf(stderr, "[KRES] missing reserved feedback snapshot\n");
                continue;
            }
        }
        memcpy(rb->snap[i], x->host_bytes ? x->mem : kelvin_guest(x->base), (size_t)size);
        x->host = rb->snap[i];
    }
}

/* Make room for snapshot_textures' copies before the batch is handed
 * over, so they cannot fail for want of memory; false if they might. */
static bool snapshot_reserve(RasterBatch *rb, bool all)
{
    int i;

    for (i = 0; i < 4; i++) {
        const KTexture *x = &rb->ctx.ps.tex[i];
        uint64_t size, end = (uint64_t)KELVIN_PHYS_WINDOW + KELVIN_PHYS_SIZE;

        if (!x->valid || (!all && !tex_overlaps_target(&rb->ctx, x))) {
            continue;
        }
        size = x->host_bytes ? x->host_bytes : (uint64_t)(x->cubemap ? 6 : 1) * x->face_stride;
        if (x->base < KELVIN_PHYS_WINDOW || x->base >= end) {
            return false;              /* not copied: read where it is */
        }
        if (!x->host_bytes && x->base + size > end) {
            size = end - x->base;
        }
        if (size > rb->snap_cap[i]) {
            if (!snapshot_room(rb, i, (size_t)size)) {
                return false;
            }
        }
    }
    return true;
}

/* Run by one raster thread once every earlier batch has finished, before
 * any of this one is scanned. */
static void batch_begin(void *arg)
{
    snapshot_textures((RasterBatch *)arg, false);
}

static bool verify_threads(void)
{
    static int on = -1;
    if (on < 0) {
        on = getenv("RECOMP_KELVIN_VERIFY_THREADS") != NULL;
    }
    return on != 0;
}

/* Scan the batch on the raster threads, then again on this thread from the
 * same starting surfaces, and compare. The serial result is the one kept. */
static uint64_t scan_verified(RasterBatch *rb)
{
    static uint64_t batches, differed;
    const DrawCtx *c = &rb->ctx;
    size_t cb = uses_color(c) ? surface_bytes(&c->t, true) : 0;
    size_t zb = uses_zeta(c) ? surface_bytes(&c->t, false) : 0;
    uint8_t *before = (uint8_t *)malloc(cb + zb + 1);
    uint8_t *threaded = (uint8_t *)malloc(cb + zb + 1);
    uint64_t wp, ws;
    size_t i;

    if (!before || !threaded) {
        free(before);
        free(threaded);
        return scan_parallel_now(rb);
    }
    /* Both scans read copies of every texture: the title may be writing
     * one meanwhile -- BLACK's movie decoder fills the texture its title
     * screen shows -- and that is not what is being checked. */
    if (kres_enabled() && !snapshot_reserve(rb, true)) {
        fprintf(stderr, "[KRES] thread-check snapshot allocation failed\n");
        free(before); free(threaded);
        return 0;
    }
    snapshot_textures(rb, true);
    if (cb) memcpy(before, c->t.color, cb);
    if (zb) memcpy(before + cb, c->t.zeta, zb);
    wp = scan_parallel_now(rb);
    if (cb) memcpy(threaded, c->t.color, cb);
    if (zb) memcpy(threaded + cb, c->t.zeta, zb);
    if (cb) memcpy(c->t.color, before, cb);
    if (zb) memcpy(c->t.zeta, before + cb, zb);
    ws = scan_serial(rb);

    batches++;
    for (i = 0; i < cb + zb; i++) {
        uint8_t now = i < cb ? c->t.color[i] : c->t.zeta[i - cb];
        if (now != threaded[i]) {
            break;
        }
    }
    if (i < cb + zb || wp != ws) {
        differed++;
        if (differed <= 20) {
            int k;
            fprintf(stderr, "[KELVIN] thread check: batch %llu differs (%s byte"
                            " %zu; %llu vs %llu pixels written; %zu triangles)\n",
                    (unsigned long long)batches,
                    i < cb ? "colour" : (i < cb + zb ? "zeta" : "no"),
                    i < cb ? i : i - cb, (unsigned long long)wp,
                    (unsigned long long)ws, c->ntris);
            fprintf(stderr, "[KELVIN]   %ux%u%s clip %d,%d-%d,%d; colour 0x%08X"
                            " +%zu pitch %u bpp %u; zeta 0x%08X +%zu pitch %u"
                            " bpp %u; depth %d stencil %d\n",
                    c->t.width, c->t.height, c->t.swizzled ? " swizzled" : "",
                    c->t.clip_x0, c->t.clip_y0, c->t.clip_x1, c->t.clip_y1,
                    c->t.color_va, cb, c->t.color_pitch, c->t.color_bpp,
                    c->t.zeta_va, zb, c->t.zeta_pitch, c->t.zeta_bpp,
                    c->ps.depth_test, c->ps.stencil_test);
            for (k = 0; k < 4; k++) {
                if (c->ps.tex[k].valid) {
                    fprintf(stderr, "[KELVIN]   texture %d at 0x%08X +%u\n",
                            k, c->ps.tex[k].base,
                            (c->ps.tex[k].cubemap ? 6 : 1) *
                                c->ps.tex[k].face_stride);
                }
            }
            fflush(stderr);
        }
    }
    if ((batches & (batches - 1)) == 0 || batches % 10000 == 0) {
        fprintf(stderr, "[KELVIN] thread check: %llu batches compared, %llu"
                        " differed\n", (unsigned long long)batches,
                (unsigned long long)differed);
        fflush(stderr);
    }
    free(before);
    free(threaded);
    return ws;
}

/* Wait until every batch handed over has finished, and count its pixels.
 * Everything that touches surfaces or reports the GPU's progress to the
 * title calls this first. */
static void geom_flush(void);

volatile long g_kelvin_sync_count[16];

void kelvin_render_sync_for(int cause)
{
    long long t0 = g_kelvin_timing > 0 || s_jt_on ? kelvin_ticks() : 0;
    if (cause >= 0 && cause < 16) g_kelvin_sync_count[cause]++;
    /* With asynchronous GPU submission the work is queued in order and the other causes need no wait; a notify tells the
     * title the GPU got here, so what was queued is finished first. */
    if (cause == KT_SYNC_NOTIFY) kgpu_drain();
    geom_flush();
    kwork_drain();
    kres_mirror();
    if (s_jt_on && s_njw < 1024 && kelvin_ticks() - t0 > 100) {
        s_jw[s_njw].t0 = t0;
        s_jw[s_njw].t1 = kelvin_ticks();
        s_jw[s_njw].cause = cause;
        s_jw[s_njw].method = g_kelvin_method;
        s_njw++;
    }
    if (s_pg) {
        s_pg->stats.pixels += (uint64_t)_InterlockedExchange64(&s_written, 0);
    }
    if (g_kelvin_timing > 0) {
        g_kelvin_ticks[cause] += kelvin_ticks() - t0;
    }
}

void kelvin_render_sync(void)
{
    kelvin_render_sync_for(KT_SYNC_OTHER);
}

int g_kelvin_timing = -1;
long long g_kelvin_ticks[KT_COUNT];
volatile long long g_kelvin_scan[3];
uint32_t g_kelvin_method;
uint32_t g_kelvin_word_get;

long long kelvin_ticks(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

/* The pixel state in short, for the job trace: a signature over what
 * shapes the per-fragment work, and the textures and stages. */
static void jt_describe(const DrawCtx *c, JobTrace *j)
{
    const KPixelState *ps = &c->ps;
    uint32_t h = 2166136261u;
    int i, n = 0;
#define MIX(v) (h = (h ^ (uint32_t)(v)) * 16777619u)
    MIX(ps->num_stages); MIX(ps->final_enabled); MIX(ps->blend); MIX(ps->depth_test);
    MIX(ps->alpha_test); MIX(ps->stencil_test);
    for (i = 0; i < ps->num_stages; i++) {
        const KCombinerStage *s = &ps->stage[i];
        const int *w = (const int *)s;
        size_t k;
        for (k = 0; k < offsetof(KCombinerStage, c0) / sizeof(int); k++) MIX(w[k]);
    }
    n += snprintf(j->desc + n, sizeof(j->desc) - n, "st%d%s%s%s", ps->num_stages,
                  ps->final_enabled ? "f" : "", ps->blend ? " bl" : "",
                  ps->depth_test ? " z" : "");
    for (i = 0; i < 4 && n < (int)sizeof(j->desc) - 1; i++) {
        const KTexture *t = &ps->tex[i];
        MIX(ps->tex_mode[i]);
        if (ps->tex_mode[i] == PS_TEXTUREMODES_NONE) continue;
        MIX(t->color_format); MIX(t->min_filter); MIX(t->mag_filter); MIX(t->levels);
        n += snprintf(j->desc + n, sizeof(j->desc) - n, " t%d:m%d f%02X %ux%u L%u %d/%d%s", i,
                      ps->tex_mode[i], t->color_format, t->width, t->height, t->levels,
                      t->min_filter, t->mag_filter, c->tex_needs_lod[i] ? " lod" : "");
    }
#undef MIX
    j->sig = h;
}

/* ---- batch capture ---------------------------------------------------------
 *
 * RECOMP_KELVIN_CAPTURE=frame:path writes frame `frame`'s batches to path,
 * for tests/kelvin_bench to replay: each batch's DrawCtx and triangles, and
 * before it the guest memory it touches that no batch before it did, read
 * once every earlier batch has finished (the capture drains before each
 * batch). Records: 'R' va len bytes; 'B' sizeof(DrawCtx) DrawCtx
 * sizeof(TriSetup) ntris TriSetups; 'E'. Pointers in the DrawCtx are the
 * game's; the replay points them at its own memory. */
static FILE *s_cap;
static long long s_cap_frame = -2;             /* -2 unread, -1 off */
static char s_cap_path[512];
static struct { uint32_t start, end; } s_cap_ranges[4096];
static int s_ncap;

static void cap_u32(uint32_t v)
{
    fwrite(&v, 4, 1, s_cap);
}

static void cap_region(uint32_t va, uint64_t bytes)
{
    uint64_t end = (uint64_t)KELVIN_PHYS_WINDOW + KELVIN_PHYS_SIZE;
    int i;
    if (!bytes || va < KELVIN_PHYS_WINDOW || va >= end) {
        return;
    }
    if (va + bytes > end) {
        bytes = end - va;
    }
    for (i = 0; i < s_ncap; i++) {
        if (va >= s_cap_ranges[i].start && va + bytes <= s_cap_ranges[i].end) {
            return;
        }
    }
    if (s_ncap < (int)(sizeof(s_cap_ranges) / sizeof(s_cap_ranges[0]))) {
        s_cap_ranges[s_ncap].start = va;
        s_cap_ranges[s_ncap].end = (uint32_t)(va + bytes);
        s_ncap++;
    }
    cap_u32('R');
    cap_u32(va);
    cap_u32((uint32_t)bytes);
    fwrite(kelvin_guest(va), 1, (size_t)bytes, s_cap);
}

static size_t surface_bytes(const KelvinTarget *t, bool color);
static bool uses_color(const DrawCtx *c);
static bool uses_zeta(const DrawCtx *c);

static void capture_batch(const DrawCtx *c)
{
    int i;
    kelvin_render_sync();
    if (uses_color(c)) {
        cap_region(c->t.color_va, guest_surface_bytes(&c->t, true));
    }
    if (uses_zeta(c)) {
        cap_region(c->t.zeta_va, guest_surface_bytes(&c->t, false));
    }
    for (i = 0; i < 4; i++) {
        const KTexture *x = &c->ps.tex[i];
        if (!x->valid) {
            continue;
        }
        cap_region(x->base, (uint64_t)(x->cubemap ? 6 : 1) * x->face_stride);
        if (x->palette) {
            cap_region(x->palette, (uint64_t)x->palette_len * 4);
        }
    }
    cap_u32('B');
    cap_u32((uint32_t)sizeof(DrawCtx));
    fwrite(c, sizeof(DrawCtx), 1, s_cap);
    cap_u32((uint32_t)sizeof(TriSetup));
    cap_u32((uint32_t)c->ntris);
    fwrite(c->tris, sizeof(TriSetup), c->ntris, s_cap);
}

/* At each flip: start the capture at the one before its frame, end it at
 * its frame's. */
static void capture_flip(void)
{
    if (s_cap_frame == -2) {
        const char *e = getenv("RECOMP_KELVIN_CAPTURE");
        const char *colon = e ? strchr(e, ':') : NULL;
        s_cap_frame = -1;
        if (e && kres_enabled()) {
            fprintf(stderr, "[KRES] legacy guest-only KCAP disabled for high-resolution surfaces; use BMP/QPC evidence\n");
            return;
        }
        if (colon && colon[1]) {
            s_cap_frame = (long long)strtoull(e, NULL, 0);
            snprintf(s_cap_path, sizeof(s_cap_path), "%s", colon + 1);
        }
    }
    if (s_cap) {
        cap_u32('E');
        fclose(s_cap);
        s_cap = NULL;
        s_cap_frame = -1;
        fprintf(stderr, "[KELVIN] capture of frame %llu written to %s\n",
                (unsigned long long)s_frames, s_cap_path);
        fflush(stderr);
        return;
    }
    if (s_cap_frame >= 0 && (long long)s_frames + 1 == s_cap_frame) {
        s_cap = fopen(s_cap_path, "wb");
        s_ncap = 0;
    }
}

/* Hand every triangle set up so far to the raster threads, in order, and
 * start the next batch. The raster job the batch became, or -1 if it was
 * scanned here or there was none. */
static long flush_tris(DrawCtx *c)
{
    RasterBatch *rb = &s_batch[s_fill % KR_SLOTS];
    const char *shared;
    long queued = -1;

    if (!c->ntris) {
        return -1;
    }
    if (s_cap) {
        capture_batch(c);
    }
    s_pg = c->pg;
    rb->ctx = *c;
    rb->jt = -1;
    if (kres_enabled() && needs_snapshot(&rb->ctx) && !snapshot_reserve(rb, false)) {
        fprintf(stderr, "[KRES] feedback allocation failed; draw skipped (no native-resolution substitution)\n");
        c->ntris = 0;
        return -1;
    }
    rb->snapshotting = false;
    /* A reused batch must not keep an old snapshot when no begin callback
     * is needed. Aliasing textures are still copied by the workers, after
     * every earlier batch has finished. */
    for (int i = 0; i < 4; i++) {
        rb->ctx.ps.tex[i].host = NULL;
    }
    shared = rows_shared(c);
    if (shared && verify_threads()) {
        static unsigned said;
        if (said++ < 8) {
            int k;
            fprintf(stderr, "[KELVIN] thread check: one thread for a batch"
                            " (%s): %ux%u%s colour 0x%08X +%zu zeta 0x%08X"
                            " +%zu\n",
                    shared, c->t.width, c->t.height,
                    c->t.swizzled ? " swizzled" : "", c->t.color_va,
                    uses_color(c) ? surface_bytes(&c->t, true) : 0,
                    c->t.zeta_va,
                    uses_zeta(c) ? surface_bytes(&c->t, false) : 0);
            for (k = 0; k < 4; k++) {
                const KTexture *x = &c->ps.tex[k];
                if (x->valid) {
                    fprintf(stderr, "[KELVIN]   texture %d at 0x%08X +%u,"
                                    " %ux%u, %u level(s)\n", k, x->base,
                            (x->cubemap ? 6 : 1) * x->face_stride,
                            x->width, x->height, x->levels);
                }
            }
            fflush(stderr);
        }
    }
    if (!bin_batch(rb)) {
        /* No memory for the bands: scan it here, after the queue. */
        kelvin_render_sync();
        snapshot_textures(rb, false);
        kres_written(&c->t, uses_color(c), uses_zeta(c));
        c->pg->stats.pixels += scan_serial(rb);
    } else if (verify_threads() && !shared && kwork_workers()) {
        kelvin_render_sync();
        snapshot_textures(rb, false);
        kres_written(&c->t, uses_color(c), uses_zeta(c));
        c->pg->stats.pixels += scan_verified(rb);
    } else {
        static const uint8_t lane0 = 0;
        bool snapshot = needs_snapshot(&rb->ctx);
        unsigned int flags = 0;

        rb->single = shared != NULL;
        rb->counted = true;
        rb->snapshotting = snapshot && !rb->single && snapshot_reserve(rb, false);
        c->pg->stats.raster_batches++;
        if (rb->single) {
            flags = KWORK_EXCLUSIVE;
            c->pg->stats.raster_batches_single++;
        } else if (snapshot) {
            c->pg->stats.raster_batches_snapshot++;
        } else if (conflicts_in_flight(rb)) {
            flags = KWORK_BARRIER;
            c->pg->stats.raster_batches_conflict++;
        }
        rb->jt = -1;
        if (s_jt_on && s_njt < JT_MAX) {
            JobTrace *j = &s_jt[s_njt];
            rb->jt = s_njt++;
            j->submit = jt_ticks();
            j->start = j->end = 0;
            j->target = c->t.color_va ? c->t.color_va : c->t.zeta_va;
            j->w = (int)c->t.width;
            j->h = (int)c->t.height;
            j->ntris = (int)c->ntris;
            j->kind = rb->single ? 3 : snapshot ? 2 : flags ? 1 : 0;
            j->cpu = j->written = 0;
            jt_describe(c, j);
        }
        {
            long long t0 = g_kelvin_timing > 0 ? kelvin_ticks() : 0;
            kres_written(&c->t, uses_color(c), uses_zeta(c));
            rb->job = kwork_submit_lanes(snapshot ? batch_begin : NULL,
                                         batch_lane_for(rb), rb,
                                         rb->single ? &lane0 : rb->lanes,
                                         rb->single ? 1 : rb->nlanes, flags);
            if (g_kelvin_timing > 0) {
                g_kelvin_ticks[KT_SUBMIT] += kelvin_ticks() - t0;
            }
        }
        queued = rb->job;
        s_fill++;
    }
    c->tris = s_batch[s_fill % KR_SLOTS].tris;
    c->ntris = 0;
    return queued;
}

/* Set up one triangle whose vertices all have w > 0 and add it to the batch;
 * flush_tris hands it over. */
static void raster_triangle(DrawCtx *c, const RVert *v0, const RVert *v1,
                            const RVert *v2, const float *flat, bool front,
                            double tri_mz)
{
    RasterBatch *rb;

    if (c->chunk) {
        TriChunk *k = c->chunk;
        if (k->n == k->cap) {
            size_t cap = k->cap ? k->cap * 2 : 256;
            TriSetup *grown = (TriSetup *)realloc(k->tris, cap * sizeof(TriSetup));
            if (!grown) {
                k->failed = true;
                return;
            }
            k->tris = grown;
            k->cap = cap;
        }
        if (tri_setup(c, v0, v1, v2, flat, front, tri_mz, &k->tris[k->n])) {
            k->n++;
        }
        return;
    }
    if (c->ntris == KR_BATCH) {
        flush_tris(c);
    }
    rb = &s_batch[s_fill % KR_SLOTS];
    if (c->ntris == rb->cap) {
        size_t cap = rb->cap ? rb->cap * 2 : 256;
        TriSetup *grown = (TriSetup *)realloc(rb->tris, cap * sizeof(TriSetup));
        if (!grown) {
            /* Out of memory: scan this one triangle here, after the queue. */
            TriSetup one;
            flush_tris(c);
            kelvin_render_sync();
            rb = &s_batch[s_fill % KR_SLOTS];
            if (tri_setup(c, v0, v1, v2, flat, front, tri_mz, &one)) {
                rb->ctx = *c;
                rb->ctx.tris = &one;
                rb->ctx.ntris = 1;
                snapshot_textures(rb, false);
                c->pg->stats.pixels += scan_serial(rb);
            }
            return;
        }
        rb->tris = grown;
        rb->cap = cap;
    }
    c->tris = rb->tris;
    if (tri_setup(c, v0, v1, v2, flat, front, tri_mz, &c->tris[c->ntris])) {
        c->ntris++;
    }
}

/* Polygon-offset slope (glsl/geom.c calc_triz), in logical pixels. */
static double tri_slope(const DrawCtx *c, const RVert *a, const RVert *b,
                        const RVert *d)
{
    double m00 = b->lx - a->lx, m01 = b->ly - a->ly;
    double m10 = d->lx - a->lx, m11 = d->ly - a->ly;
    double bx, by, det, dzx, dzy;
    if (c->ps.z_perspective) {
        bx = (a->w - b->w) / (b->w * a->w);
        by = (a->w - d->w) / (d->w * a->w);
    } else {
        bx = b->z - a->z;
        by = d->z - a->z;
    }
    det = m00 * m11 - m10 * m01;
    if (det == 0.0) {
        return 0.0;
    }
    dzx = (bx * m11 - by * m01) / det;
    dzy = (by * m00 - bx * m10) / det;
    dzx = fabs(dzx);
    dzy = fabs(dzy);
    return isfinite(dzx) && isfinite(dzy) ? MAX(dzx, dzy) : 0.0;
}

static void lerp_rvert(const RVert *a, const RVert *b, double t, RVert *o)
{
    /* Linear in clip space: X = x w, Y = y w, Z = z w, W = w. */
    double w = a->w + (b->w - a->w) * t;
    double X = a->x * a->w + (b->x * b->w - a->x * a->w) * t;
    double Y = a->y * a->w + (b->y * b->w - a->y * a->w) * t;
    double LX = a->lx * a->w + (b->lx * b->w - a->lx * a->w) * t;
    double LY = a->ly * a->w + (b->ly * b->w - a->ly * a->w) * t;
    double Z = a->z * a->w + (b->z * b->w - a->z * a->w) * t;
    int k;
    o->w = w;
    o->x = X / w;
    o->y = Y / w;
    o->lx = LX / w;
    o->ly = LY / w;
    o->z = Z / w;
    for (k = 0; k < KV_VARYINGS; k++) {
        o->own[k] = (float)(a->var[k] + (b->var[k] - a->var[k]) * t);
    }
    o->var = o->own;
}

static void draw_triangle(DrawCtx *c, const KVertex *k0, const KVertex *k1,
                          const KVertex *k2, const KVertex *provoking)
{
    RVert r[3];
    double area;
    bool front;
    float flat[16];
    const float *flatp = NULL;
    double mz;

    make_rvert(c, k0, &r[0]);
    make_rvert(c, k1, &r[1]);
    make_rvert(c, k2, &r[2]);

    /* NV2A screen coordinates have their origin at the top left: negative
     * signed area is counter-clockwise, positive area is clockwise. */
    area = (r[1].x - r[0].x) * (r[2].y - r[0].y) -
           (r[2].x - r[0].x) * (r[1].y - r[0].y);
    if (r[0].w < 0 || r[1].w < 0 || r[2].w < 0) {
        /* With a vertex behind the eye the projected winding is not the
         * triangle's; GL decides in clip space, which here is the sign of
         * the determinant of (X, Y, W). */
        double m[3][3];
        int i;
        for (i = 0; i < 3; i++) {
            m[i][0] = r[i].x * r[i].w;
            m[i][1] = r[i].y * r[i].w;
            m[i][2] = r[i].w;
        }
        area = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    front = (area < 0.0) == c->ccw_front;
    if (c->cull_enable) {
        if (c->cull_ctrl == NV_PGRAPH_SETUPRASTER_CULLCTRL_FRONT_AND_BACK) return;
        if (c->cull_ctrl == NV_PGRAPH_SETUPRASTER_CULLCTRL_FRONT && front) return;
        if (c->cull_ctrl == NV_PGRAPH_SETUPRASTER_CULLCTRL_BACK && !front) return;
    }
    if (!c->ps.smooth) {
        memcpy(flat + 0, provoking->d0, 16);
        memcpy(flat + 4, provoking->d1, 16);
        memcpy(flat + 8, provoking->b0, 16);
        memcpy(flat + 12, provoking->b1, 16);
        flatp = flat;
    }
    mz = c->poly_offset ? tri_slope(c, &r[0], &r[1], &r[2]) : 0.0;

    if (r[0].w > 0 && r[1].w > 0 && r[2].w > 0) {
        raster_triangle(c, &r[0], &r[1], &r[2], flatp, front, mz);
        return;
    }
    /* Clip against w >= eps (Sutherland-Hodgman, homogeneous space). */
    {
        const double eps = 1e-6;
        RVert in[4], out[4];
        int n = 0, i;
        for (i = 0; i < 3; i++) {
            const RVert *a = &r[i], *b = &r[(i + 1) % 3];
            bool ain = a->w >= eps, bin = b->w >= eps;
            if (ain) {
                in[n++] = *a;
            }
            if (ain != bin) {
                double t = (eps - a->w) / (b->w - a->w);
                lerp_rvert(a, b, t, &in[n++]);
            }
        }
        (void)out;
        for (i = 1; i + 1 < n; i++) {
            raster_triangle(c, &in[0], &in[i], &in[i + 1], flatp, front, mz);
        }
    }
}

/* The minor-axis coordinate of step i of a line, before raster_line floors
 * it: the expression its loop evaluates, operation for operation. */
KELVIN_INLINE double line_minor(int i, double m0, double dm, double n0, double dn)
{
    double t = (i + 0.5 - m0) / dm;
    return n0 + t * dn;
}

/* The first step of [lo, hi) whose minor coordinate is at least `bound`
 * (rising) or below it (falling), or hi; the coordinate must only rise, or
 * only fall, over the steps. */
static int line_first_step(int lo, int hi, double m0, double dm, double n0,
                           double dn, bool rising, double bound)
{
    while (lo < hi) {
        int mid = lo + (int)(((unsigned int)hi - (unsigned int)lo) / 2);
        double u = line_minor(mid, m0, dm, n0, dn);
        if (rising ? u >= bound : u < bound) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

/* Narrow raster_line's steps [*i0, *i1) to those whose pixel can be inside
 * the scissor; the loop still tests each one. A step's major coordinate is
 * its number, so the major range is a clamp. Its minor coordinate is
 * (int)floor(u), u = line_minor(step): with c0 and c1 integers that is in
 * [c0, c1) exactly when c0 <= u < c1 -- also when floor(u) is outside int's
 * range or u is NaN, which convert to INT_MIN, below any c0 above INT_MIN.
 * Every operation giving u rounds monotonically, so u only rises or only
 * falls from step to step, the steps with c0 <= u < c1 are one run, and
 * bisection on u itself finds its ends. */
static void line_clip_steps(const KelvinTarget *t, bool xmajor, double m0, double dm,
                            double n0, double dn, int *i0, int *i1)
{
    int lo = *i0, hi = *i1;
    int cmaj0 = xmajor ? t->clip_x0 : t->clip_y0, cmaj1 = xmajor ? t->clip_x1 : t->clip_y1;
    int cmin0 = xmajor ? t->clip_y0 : t->clip_x0, cmin1 = xmajor ? t->clip_y1 : t->clip_x1;

    if (lo < cmaj0) {
        lo = cmaj0;
    }
    if (hi > cmaj1) {
        hi = cmaj1;
    }
    if (lo < hi && cmin0 > INT_MIN) {
        bool rising = line_minor(lo, m0, dm, n0, dn) <= line_minor(hi - 1, m0, dm, n0, dn);
        if (rising) {
            lo = line_first_step(lo, hi, m0, dm, n0, dn, true, (double)cmin0);
            hi = line_first_step(lo, hi, m0, dm, n0, dn, true, (double)cmin1);
        } else {
            lo = line_first_step(lo, hi, m0, dm, n0, dn, false, (double)cmin1);
            hi = line_first_step(lo, hi, m0, dm, n0, dn, false, (double)cmin0);
        }
    }
    *i0 = lo;
    *i1 = hi;
}

/* One width-1, non-antialiased line whose ends both have w > 0.
 *
 * GL's rule for these is the diamond-exit test; stepping the major axis over
 * pixel centres, half-open at the far end, picks the same pixels except where
 * a line starts or ends inside a diamond, which the GL specification leaves
 * to the implementation anyway. Varyings are perspective-correct, depth is
 * screen-linear (or 1/q for a w-buffer), and texture derivatives run along
 * the major axis. Only the steps that can land inside the scissor are taken
 * (line_clip_steps): an end near w = 0 projects millions of pixels away. */
static void raster_line(DrawCtx *c, const RVert *a, const RVert *b,
                        const float *flat, double depth_bias)
{
    const KPixelState *ps = &c->ps;
    double dx = b->x - a->x, dy = b->y - a->y;
    bool xmajor = fabs(dx) >= fabs(dy);
    double m0 = xmajor ? a->x : a->y, m1 = xmajor ? b->x : b->y;
    double n0 = xmajor ? a->y : a->x, dn = xmajor ? dy : dx;
    double dm = m1 - m0;
    double iwa = 1.0 / a->w, iwb = 1.0 / b->w;
    int i, i0, i1, k, t4;

    if (dm == 0.0 || !isfinite(dm) || !isfinite(dn)) {
        return;
    }
    i0 = (int)ceil(MIN(m0, m1) - 0.5);
    i1 = (int)ceil(MAX(m0, m1) - 0.5);
    line_clip_steps(&c->t, xmajor, m0, dm, n0, dn, &i0, &i1);
    for (i = i0; i < i1; i++) {
        double t = (i + 0.5 - m0) / dm;
        int j = (int)floor(n0 + t * dn);
        int x = xmajor ? i : j, y = xmajor ? j : i;
        double q = iwa * (1.0 - t) + iwb * t, iq, z;
        float var[KV_VARYINGS];
        KFragment f;

        if (x < c->t.clip_x0 || x >= c->t.clip_x1 ||
            y < c->t.clip_y0 || y >= c->t.clip_y1 || q == 0.0) {
            continue;
        }
        iq = 1.0 / q;
        for (k = 0; k < KV_VARYINGS; k++) {
            var[k] = (float)((a->var[k] * iwa * (1.0 - t) +
                              b->var[k] * iwb * t) * iq);
        }
        if (flat) {
            memcpy(f.d0, flat + 0, 16);
            memcpy(f.d1, flat + 4, 16);
            memcpy(f.b0, flat + 8, 16);
            memcpy(f.b1, flat + 12, 16);
        } else {
            memcpy(f.d0, var + 0, 16);
            memcpy(f.d1, var + 4, 16);
            memcpy(f.b0, var + 8, 16);
            memcpy(f.b1, var + 12, 16);
        }
        f.fog = var[16];
        memcpy(f.t, var + 17, 64);
        for (t4 = 0; t4 < 4; t4++) {
            memset(f.dtdx[t4], 0, 16);
            memset(f.dtdy[t4], 0, 16);
            if (!c->tex_needs_lod[t4]) {
                continue;
            }
            for (k = 0; k < 4; k++) {
                /* d(attr)/d(major) of the perspective-correct attribute. */
                int idx = 17 + t4 * 4 + k;
                double dq = (iwb - iwa) / dm;
                double dp = (b->var[idx] * iwb - a->var[idx] * iwa) / dm;
                float d = (float)((dp - var[idx] * dq) * iq);
                if (xmajor) {
                    f.dtdx[t4][k] = d;
                } else {
                    f.dtdy[t4][k] = d;
                }
            }
        }
        if (ps->z_perspective) {
            z = iq;
            z = (z > 0.0 && isfinite(z)) ? z + depth_bias : 3.402823466e38;
        } else {
            z = a->z + t * (b->z - a->z) + depth_bias;
        }
        f.z = (float)z;
        f.x = x;
        f.y = y;
        f.front = true;
        if (kpix_fragment(ps, &c->t, &f)) {
            c->pg->stats.pixels++;
        }
    }
}

static void draw_line(DrawCtx *c, const KVertex *k0, const KVertex *k1,
                      const KVertex *provoking)
{
    RVert r[2];
    float flat[16];
    const float *flatp = NULL;
    const double eps = 1e-6;

    kelvin_render_sync();     /* lines are drawn here, not batched */
    kres_written(&c->t, uses_color(c), uses_zeta(c));

    make_rvert(c, k0, &r[0]);
    make_rvert(c, k1, &r[1]);
    if (!c->ps.smooth) {
        memcpy(flat + 0, provoking->d0, 16);
        memcpy(flat + 4, provoking->d1, 16);
        memcpy(flat + 8, provoking->b0, 16);
        memcpy(flat + 12, provoking->b1, 16);
        flatp = flat;
    }
    /* Clip where an end is at or behind the eye, against w >= eps, as
     * triangles are; any positive w is drawn as it stands. */
    if (!(r[0].w > 0) && !(r[1].w > 0)) {
        return;
    }
    if (!(r[0].w > 0) || !(r[1].w > 0)) {
        int in = r[0].w > 0 ? 0 : 1;
        double t = (eps - r[in].w) / (r[1 - in].w - r[in].w);
        RVert cut;
        if (r[in].w <= eps) {
            return;      /* what is in front is all within eps of the eye */
        }
        lerp_rvert(&r[in], &r[1 - in], t, &cut);
        r[1 - in] = cut;
    }
    c->pg->stats.lines++;
    raster_line(c, &r[0], &r[1], flatp, c->line_offset ? c->depth_offset : 0.0);
}

/* ---- primitive assembly (glsl/geom.c) --------------------------------------- */

typedef struct VCache {
    uint32_t *keys;
    KVertex *verts;
    uint32_t *stamp;           /* a slot holds a vertex of this draw when == gen */
    uint32_t gen;
    size_t cap;
} VCache;

static const KVertex *get_vertex(KelvinState *pg, const DrawSource *src,
                                 VCache *vc, uint32_t index)
{
    size_t h = (index * 2654435761u) & (vc->cap - 1);
    while (vc->stamp[h] == vc->gen) {
        if (vc->keys[h] == index) {
            return &vc->verts[h];
        }
        h = (h + 1) & (vc->cap - 1);
    }
    {
        float in[NV2A_VERTEXSHADER_ATTRIBUTES][4];
        fetch_vertex(pg, src, index, in);
        kvsh_run(pg, in, &vc->verts[h]);
        pg->stats.vertices++;
        vc->keys[h] = index;
        vc->stamp[h] = vc->gen;
        return &vc->verts[h];
    }
}

static void assemble(DrawCtx *c, const DrawSource *src, VCache *vc,
                     const uint32_t *idx, size_t n)
{
    KelvinState *pg = c->pg;
    size_t i;
#define V(k) get_vertex(pg, src, vc, idx[k])

    switch (pg->primitive_mode) {
    case PRIM_TYPE_TRIANGLES:
        for (i = 0; i + 2 < n; i += 3) {
            draw_triangle(c, V(i), V(i + 1), V(i + 2),
                          c->provoking_first ? V(i) : V(i + 2));
        }
        break;
    case PRIM_TYPE_TRIANGLE_STRIP:
        for (i = 0; i + 2 < n; i++) {
            if (i & 1) {
                draw_triangle(c, V(i + 1), V(i), V(i + 2),
                              c->provoking_first ? V(i) : V(i + 2));
            } else {
                draw_triangle(c, V(i), V(i + 1), V(i + 2),
                              c->provoking_first ? V(i) : V(i + 2));
            }
        }
        break;
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        for (i = 1; i + 1 < n; i++) {
            draw_triangle(c, V(0), V(i), V(i + 1),
                          c->provoking_first ? V(0) : V(i + 1));
        }
        break;
    case PRIM_TYPE_QUADS:
        for (i = 0; i + 3 < n; i += 4) {
            draw_triangle(c, V(i + 1), V(i + 2), V(i), V(i + 3));
            draw_triangle(c, V(i + 2), V(i + 3), V(i), V(i + 3));
        }
        break;
    case PRIM_TYPE_QUAD_STRIP:
        for (i = 0; i + 3 < n; i += 2) {
            draw_triangle(c, V(i), V(i + 1), V(i + 2), V(i + 3));
            draw_triangle(c, V(i + 2), V(i + 1), V(i + 3), V(i + 3));
        }
        break;
    case PRIM_TYPE_POINTS:
        kelvin_unsupported(KU_POINT_PRIM, "points", (uint32_t)n);
        break;
    case PRIM_TYPE_LINES:
        for (i = 0; i + 1 < n; i += 2) {
            draw_line(c, V(i), V(i + 1), c->provoking_first ? V(i) : V(i + 1));
        }
        break;
    case PRIM_TYPE_LINE_STRIP:
    case PRIM_TYPE_LINE_LOOP:
        for (i = 0; i + 1 < n; i++) {
            draw_line(c, V(i), V(i + 1), c->provoking_first ? V(i) : V(i + 1));
        }
        if (pg->primitive_mode == PRIM_TYPE_LINE_LOOP && n > 2) {
            draw_line(c, V(n - 1), V(0), c->provoking_first ? V(n - 1) : V(0));
        }
        break;
    default:
        break;
    }
#undef V
}

static const KVertex *get_vertex(KelvinState *pg, const DrawSource *src,
                                 VCache *vc, uint32_t index);
static void trace_dump_texture(const KTexture *t);

/* RECOMP_KELVIN_TRACE on the GPU renderer: the same draw state, before the draw goes to the GPU (no rasterised
 * pixels or post-transform vertices to report there), with every combiner stage's inputs. */
static void trace_draw_gpu(DrawCtx *c, const DrawSource *src, size_t n)
{
    static const char *const kinds[] = { "arrays", "inline-array", "inline-buffer" };
    KelvinState *pg = c->pg;
    const KPixelState *ps = &c->ps;
    const uint32_t csv0_d = pgraph_reg_r(pg, NV_PGRAPH_CSV0_D);
    int k, s, j;
    fprintf(stderr, "[KTRACE] f%llu d%llu prim %u n %zu %s vsh %s | target 0x%08X %ux%u zeta 0x%08X | gpu\n",
            (unsigned long long)s_frames + 1, (unsigned long long)s_trace_draw++, pg->primitive_mode, n,
            kinds[src->kind], GET_MASK(csv0_d, NV_PGRAPH_CSV0_D_MODE) == 2 ? "program" : "fixed",
            c->t.color_va, c->t.width, c->t.height, c->t.zeta_va);
    fprintf(stderr, "[KTRACE]   combiners %d%s | blend %d sf 0x%X df 0x%X | alpha test %d | depth %d/%d | cull %d\n",
            ps->num_stages, ps->final_enabled ? " +final" : "", ps->blend, ps->blend_sfactor, ps->blend_dfactor,
            ps->alpha_test, ps->depth_test, ps->depth_write, c->cull_enable);
    for (s = 0; s < ps->num_stages && s < 8; s++) {
        const KCombinerStage *st = &ps->stage[s];
        fprintf(stderr, "[KTRACE]   stage %d rgb", s);
        for (j = 0; j < 4; j++) fprintf(stderr, " %d/%d/%d", st->rgb[j].reg, st->rgb[j].mod, st->rgb[j].chan);
        fprintf(stderr, " -> ab %d cd %d sum %d ops %d/%d/%d map %d | alpha", st->rgb_out.ab, st->rgb_out.cd, st->rgb_out.muxsum,
                st->rgb_out.ab_op, st->rgb_out.cd_op, st->rgb_out.muxsum_op, st->rgb_out.mapping);
        for (j = 0; j < 4; j++) fprintf(stderr, " %d/%d/%d", st->alpha[j].reg, st->alpha[j].mod, st->alpha[j].chan);
        fprintf(stderr, " | c0 %.2f %.2f %.2f %.2f c1 %.2f %.2f %.2f %.2f\n",
                (double)st->c0[0], (double)st->c0[1], (double)st->c0[2], (double)st->c0[3],
                (double)st->c1[0], (double)st->c1[1], (double)st->c1[2], (double)st->c1[3]);
    }
    if (ps->final_enabled) {
        fprintf(stderr, "[KTRACE]   final");
        for (j = 0; j < 7; j++) fprintf(stderr, " %d/%d/%d", ps->fin[j].reg, ps->fin[j].mod, ps->fin[j].chan);
        fprintf(stderr, "\n");
    }
    kvsh_trace_program();
    for (k = 0; k < 4; k++) {
        const KTexture *t = &ps->tex[k];
        if (!t->valid && !ps->tex_mode[k]) continue;
        fprintf(stderr, "[KTRACE]   tex%d mode %d %s fmt 0x%02X %ux%u levels %u base 0x%08X addr %u/%u%s\n",
                k, ps->tex_mode[k], t->valid ? "valid" : "INVALID", t->color_format, t->width, t->height, t->levels,
                t->base, t->addr[0], t->addr[1], t->linear ? " linear" : " swizzled");
        if (t->valid) trace_dump_texture(t);
    }
}

/* Level 0 of a 2D texture as a BMP: colour | alpha, side by side. */
static void trace_dump_texture(const KTexture *t)
{
    static uint32_t seen[64][2];
    static unsigned nseen;
    static uint64_t seen_frame;
    char path[512];
    unsigned int w = t->width, h = t->height, x, y, i;
    uint32_t row_bytes, pad, filesz;
    uint8_t hdr[54];
    FILE *f;

    if (!s_trace_tex || !*s_trace_tex || t->dimensionality != 2 ||
        t->cubemap || !w || !h || w > 4096 || h > 4096) {
        return;
    }
    /* Once per texture per frame: a video or render target changes. */
    if (seen_frame != s_frames) {
        seen_frame = s_frames;
        nseen = 0;
    }
    for (i = 0; i < nseen; i++) {
        if (seen[i][0] == t->base && seen[i][1] == t->color_format) {
            return;
        }
    }
    if (nseen < 64) {
        seen[nseen][0] = t->base;
        seen[nseen][1] = t->color_format;
        nseen++;
    }
    snprintf(path, sizeof(path), "%sf%05llu_%08X_fmt%02X_%ux%u.bmp",
             s_trace_tex, (unsigned long long)s_frames + 1, t->base,
             t->color_format, w, h);
    f = fopen(path, "wb");
    if (!f) {
        return;
    }
    row_bytes = w * 2 * 3;
    pad = (4 - row_bytes % 4) % 4;
    filesz = 54 + (row_bytes + pad) * h;
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    memcpy(hdr + 2, &filesz, 4);
    hdr[10] = 54; hdr[14] = 40;
    {
        uint32_t ww = w * 2;
        memcpy(hdr + 18, &ww, 4);
    }
    memcpy(hdr + 22, &h, 4);
    hdr[26] = 1; hdr[28] = 24;
    fwrite(hdr, 1, sizeof(hdr), f);
    for (y = h; y-- > 0;) {
        uint8_t *row = (uint8_t *)malloc(row_bytes + pad);
        memset(row, 0, row_bytes + pad);
        for (x = 0; x < w; x++) {
            float o[4];
            float sc = (x + 0.5f) / w, tc = (y + 0.5f) / h;
            ktex_sample_2d(t, 0, sc, tc, 0.0f, o);
            for (i = 0; i < 3; i++) {
                row[x * 3 + i] = (uint8_t)(fminf(fmaxf(o[2 - i], 0.0f), 1.0f)
                                           * 255.0f + 0.5f);
                row[(w + x) * 3 + i] = (uint8_t)(fminf(fmaxf(o[3], 0.0f),
                                                       1.0f) * 255.0f + 0.5f);
            }
        }
        fwrite(row, 1, row_bytes + pad, f);
        free(row);
    }
    fclose(f);
    fprintf(stderr, "[KTRACE]   texture written: %s\n", path);
}

static void trace_draw(DrawCtx *c, const DrawSource *src, VCache *vc,
                       const uint32_t *idx, size_t n, uint64_t pixels_before)
{
    static const char *const kinds[] = { "arrays", "inline-array",
                                         "inline-buffer" };
    KelvinState *pg = c->pg;
    const KPixelState *ps = &c->ps;
    uint32_t csv0_d = pgraph_reg_r(pg, NV_PGRAPH_CSV0_D);
    size_t i;
    int k;

    fprintf(stderr, "[KTRACE] f%llu d%llu prim %u n %zu %s%s vsh %s | target"
                    " 0x%08X fmt %u %ux%u zeta 0x%08X | px %llu\n",
            (unsigned long long)s_frames + 1,
            (unsigned long long)s_trace_draw++, pg->primitive_mode, n,
            kinds[src->kind], pg->inline_elements_length ? " indexed" : "",
            GET_MASK(csv0_d, NV_PGRAPH_CSV0_D_MODE) == 2 ? "program" : "fixed",
            c->t.color_va, ps->color_format, c->t.width, c->t.height,
            c->t.zeta_va,
            (unsigned long long)(pg->stats.pixels - pixels_before));
    {
        /* Where the draw's vertices landed, over all of them. */
        float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
        for (i = 0; i < n; i++) {
            const KVertex *v = get_vertex(pg, src, vc, idx[i]);
            x0 = fminf(x0, v->pos[0]); x1 = fmaxf(x1, v->pos[0]);
            y0 = fminf(y0, v->pos[1]); y1 = fmaxf(y1, v->pos[1]);
        }
        fprintf(stderr, "[KTRACE]   bbox x %.1f..%.1f y %.1f..%.1f\n", x0, x1,
                y0, y1);
    }
    fprintf(stderr, "[KTRACE]   combiners %d%s | blend %d sf 0x%X df 0x%X"
                    " eq 0x%X | alpha test %d func %d ref %d | depth %d/%d"
                    " func %d | stencil %d | mask %d%d%d%d | cull %d\n",
            ps->num_stages, ps->final_enabled ? " +final" : "", ps->blend,
            ps->blend_sfactor, ps->blend_dfactor, ps->blend_eqn,
            ps->alpha_test, ps->alpha_func, ps->alpha_ref, ps->depth_test,
            ps->depth_write, ps->depth_func, ps->stencil_test, ps->mask_r,
            ps->mask_g, ps->mask_b, ps->mask_a, c->cull_enable);
    fprintf(stderr, "[KTRACE]   regs: combinectl %08X color i/o %08X/%08X"
                    " alpha i/o %08X/%08X specfog %08X %08X factor0 %08X"
                    " shaderprog %08X\n",
            pgraph_reg_r(pg, NV_PGRAPH_COMBINECTL),
            pgraph_reg_r(pg, NV_PGRAPH_COMBINECOLORI0),
            pgraph_reg_r(pg, NV_PGRAPH_COMBINECOLORO0),
            pgraph_reg_r(pg, NV_PGRAPH_COMBINEALPHAI0),
            pgraph_reg_r(pg, NV_PGRAPH_COMBINEALPHAO0),
            pgraph_reg_r(pg, NV_PGRAPH_COMBINESPECFOG0),
            pgraph_reg_r(pg, NV_PGRAPH_COMBINESPECFOG1),
            pgraph_reg_r(pg, NV_PGRAPH_COMBINEFACTOR0),
            pgraph_reg_r(pg, NV_PGRAPH_SHADERPROG));
    kvsh_trace_program();
    for (k = 0; k < 4; k++) {
        const KTexture *t = &ps->tex[k];
        if (!t->valid && !ps->tex_mode[k]) {
            continue;
        }
        fprintf(stderr, "[KTRACE]   tex%d mode %d %s fmt 0x%02X %ux%u"
                        " levels %u base 0x%08X filter %u/%u addr %u/%u%s\n",
                k, ps->tex_mode[k], t->valid ? "valid" : "INVALID",
                t->color_format, t->width, t->height, t->levels, t->base,
                t->min_filter, t->mag_filter, t->addr[0], t->addr[1],
                t->linear ? " linear" : " swizzled");
        if (t->valid) {
            trace_dump_texture(t);
        }
    }
    for (i = 0; i < n && i < s_trace_verts; i++) {
        const KVertex *v = get_vertex(pg, src, vc, idx[i]);
        fprintf(stderr, "[KTRACE]   v%zu pos %.1f %.1f %.4g w %.4g | d0 %.3f"
                        " %.3f %.3f %.3f | t0 %.3f %.3f %.3f %.3f\n",
                i, v->pos[0], v->pos[1], v->pos[2], v->pos[3], v->d0[0],
                v->d0[1], v->d0[2], v->d0[3], v->t[0][0], v->t[0][1],
                v->t[0][2], v->t[0][3]);
        {
            /* What the vertex stage was given: every attribute the draw
             * supplies, as fetched. */
            float in[NV2A_VERTEXSHADER_ATTRIBUTES][4];
            fetch_vertex(pg, src, idx[i], in);
            kelvin_trace_record_begin(stderr);
            fprintf(stderr, "[KTRACE]      in");
            for (k = 0; k < NV2A_VERTEXSHADER_ATTRIBUTES; k++) {
                if (pg->vertex_attributes[k].count ||
                    (src->kind == SRC_INLINE_BUFFER &&
                     pg->vertex_attributes[k].inline_buffer_populated)) {
                    fprintf(stderr, " | v%d %.4g %.4g %.4g %.4g", k, in[k][0],
                            in[k][1], in[k][2], in[k][3]);
                }
            }
            fprintf(stderr, "\n");
            kelvin_trace_record_end(stderr);
        }
    }
    if (src->kind == SRC_INLINE_ARRAY) {
        kelvin_trace_record_begin(stderr);
        fprintf(stderr, "[KTRACE]   inline array: vertex %u bytes, %u words;",
                src->vertex_size, pg->inline_array_length);
        for (k = 0; k < NV2A_VERTEXSHADER_ATTRIBUTES; k++) {
            const VertexAttribute *a = &pg->vertex_attributes[k];
            if (a->count) {
                fprintf(stderr, " v%d@%u fmt %u x%u", k,
                        a->inline_array_offset, a->format, a->count);
            }
        }
        fprintf(stderr, "\n");
        kelvin_trace_record_end(stderr);
    }
    fflush(stderr);
}

/* One cache, reused draw after draw on the GPU thread: allocating one per
 * draw put large draws' caches in fresh pages every time (the heap hands
 * big blocks to VirtualAlloc), and the page faults cost more than the
 * shading. Slots are claimed by stamping them with the draw's generation,
 * so nothing is cleared between draws. */
static struct {
    uint32_t *keys;
    KVertex *verts;
    uint32_t *stamp;
    size_t cap;
    uint32_t gen;
} s_vcache;

static void vcache_init(VCache *vc, size_t n)
{
    size_t cap = 64;
    while (cap < n * 2) cap <<= 1;
    if (cap > s_vcache.cap) {
        free(s_vcache.keys);
        free(s_vcache.verts);
        free(s_vcache.stamp);
        s_vcache.keys = (uint32_t *)malloc(cap * sizeof(uint32_t));
        s_vcache.verts = (KVertex *)malloc(cap * sizeof(KVertex));
        s_vcache.stamp = (uint32_t *)calloc(cap, sizeof(uint32_t));
        s_vcache.cap = cap;
        s_vcache.gen = 0;
    }
    if (++s_vcache.gen == 0) {
        memset(s_vcache.stamp, 0, s_vcache.cap * sizeof(uint32_t));
        s_vcache.gen = 1;
    }
    vc->keys = s_vcache.keys;
    vc->verts = s_vcache.verts;
    vc->stamp = s_vcache.stamp;
    vc->gen = s_vcache.gen;
    vc->cap = cap;
}

static void vcache_free(VCache *vc)
{
    (void)vc;                  /* kept for the next draw */
}

/* ---- shading a draw's vertices on the raster threads -------------------
 *
 * A large draw's distinct vertices are entered in its vertex cache first
 * and shaded in parallel (kwork_parallel), so that assembly finds every one
 * there. Each is shaded by the same code from the same inputs as get_vertex
 * would, under the GPU thread's MXCSR, so the results are the same; only
 * draws whose vertices do not depend on one another qualify
 * (kvsh_parallel_ok). The loop runs ahead of queued raster work, which
 * reads its own copies of the triangles, never vertex data or the cache. */

/* Vertices per parallel item: enough to outweigh handing an item over. */
#define KV_CHUNK 32
/* Distinct vertices a draw needs before they are shaded in parallel. */
#define KV_PARALLEL_MIN 96

typedef struct VertexWork {
    KelvinState *pg;
    const DrawSource *src;
    VCache *vc;
    uint32_t *slots;                 /* cache entries to shade */
    size_t count;
} VertexWork;

static uint32_t *s_vslots;
static size_t s_vslots_cap;

static void shade_vertices(void *arg, int item, int worker)
{
    const VertexWork *w = (const VertexWork *)arg;
    size_t i = (size_t)item * KV_CHUNK, end = MIN(i + KV_CHUNK, w->count);

    (void)worker;
    /* KVSH_LANES at a time through the program (kvsh_run8 or kvsh_run4). */
    for (; i < end; i += KVSH_LANES) {
        float in[KVSH_LANES][NV2A_VERTEXSHADER_ATTRIBUTES][4];
        KVertex *out[KVSH_LANES];
        int n = (int)MIN(KVSH_LANES, end - i), j;
        for (j = 0; j < n; j++) {
            uint32_t h = w->slots[i + j];
            fetch_vertex(w->pg, w->src, w->vc->keys[h], in[j]);
            out[j] = &w->vc->verts[h];
        }
#if KVSH_LANES == 8
        kvsh_run8(w->pg, in, out, n);
#else
        kvsh_run4(w->pg, in, out, n);
#endif
    }
}

/* RECOMP_KELVIN_VERIFY_VERTICES=1 shades every vertex shaded in parallel
 * again here, one at a time, and compares the two byte for byte. */
static bool verify_vertices(void)
{
    static int on = -1;
    if (on < 0) {
        on = getenv("RECOMP_KELVIN_VERIFY_VERTICES") != NULL;
    }
    return on != 0;
}

/* The first float at which two shaded vertices differ, or -1: equal bit
 * for bit, except that any two NaNs match. Which NaN an SSE operation keeps
 * of two differs from a compiler's scalar code (kvsh_run8 against kvsh_run),
 * and a NaN's payload changes nothing downstream: a NaN position drops the
 * triangle, a colour clamps it to 1, a coordinate samples as NaN does. */
static int vertex_differs(const KVertex *a, const KVertex *b)
{
    const float *x = (const float *)a, *y = (const float *)b;
    int f;
    for (f = 0; f < (int)(sizeof(KVertex) / sizeof(float)); f++) {
        if (memcmp(&x[f], &y[f], 4) != 0 && !(isnan(x[f]) && isnan(y[f]))) {
            return f;
        }
    }
    return -1;
}

static void check_preshaded(const VertexWork *w)
{
    static uint64_t checked, differed;
    size_t i;

    for (i = 0; i < w->count; i++) {
        uint32_t h = w->slots[i];
        float in[NV2A_VERTEXSHADER_ATTRIBUTES][4];
        KVertex serial;
        fetch_vertex(w->pg, w->src, w->vc->keys[h], in);
        kvsh_run(w->pg, in, &serial);
        if (vertex_differs(&serial, &w->vc->verts[h]) >= 0 && ++differed <= 20) {
            fprintf(stderr, "[KELVIN] vertex check: index %u differs\n",
                    w->vc->keys[h]);
        }
    }
    checked += w->count;
    if ((checked >> 16) != ((checked - w->count) >> 16)) {
        fprintf(stderr, "[KELVIN] vertex check: %llu vertices compared, %llu"
                        " differed\n", (unsigned long long)checked,
                (unsigned long long)differed);
        fflush(stderr);
    }
}

/* Shade the vertices idx[0..n) name that the cache lacks, four at a time,
 * and in parallel when there are enough of them. Whether every vertex the
 * list names is then in the cache. */
static bool preshade(KelvinState *pg, const DrawSource *src, VCache *vc,
                     const uint32_t *idx, size_t n)
{
    VertexWork w;
    size_t i;

    if (n < 4 || !kvsh_parallel_ok()) {
        return false;
    }
    if (s_vslots_cap < n) {
        uint32_t *grown = (uint32_t *)realloc(s_vslots, n * sizeof(uint32_t));
        if (!grown) {
            return false;
        }
        s_vslots = grown;
        s_vslots_cap = n;
    }
    w.pg = pg;
    w.src = src;
    w.vc = vc;
    w.slots = s_vslots;
    w.count = 0;
    for (i = 0; i < n; i++) {
        uint32_t index = idx[i];
        size_t h = (index * 2654435761u) & (vc->cap - 1);   /* get_vertex */
        while (vc->stamp[h] == vc->gen && vc->keys[h] != index) {
            h = (h + 1) & (vc->cap - 1);
        }
        if (vc->stamp[h] != vc->gen) {
            vc->stamp[h] = vc->gen;
            vc->keys[h] = index;
            w.slots[w.count++] = (uint32_t)h;
        }
    }
    if (w.count < KV_PARALLEL_MIN || !kwork_workers()) {
        pg->stats.vertices += w.count;
        for (i = 0; i * KV_CHUNK < w.count; i++) {
            shade_vertices(&w, (int)i, 0);
        }
        return true;
    }
    pg->stats.vertices += w.count;
    pg->stats.vertices_parallel += w.count;
    {
        long long t0 = g_kelvin_timing > 0 ? kelvin_ticks() : 0;
        kwork_parallel(shade_vertices, &w,
                       (int)((w.count + KV_CHUNK - 1) / KV_CHUNK));
        if (g_kelvin_timing > 0) {
            g_kelvin_ticks[KT_PRESHADE] += kelvin_ticks() - t0;
        }
    }
    if (verify_vertices()) {
        check_preshaded(&w);
    }
    return true;
}

/* ---- setting a draw's triangles up on the raster threads ---------------
 *
 * With every vertex of a large draw already in the cache, its primitives
 * are split into chunks set up in parallel (assemble_chunk), each by the
 * same per-primitive code assemble runs, into its own buffer. The GPU thread
 * then appends the chunks in order into batches exactly as raster_triangle
 * would have filled them, so the batches, their boundaries and their
 * triangles are the same as serial assembly's. Setup reads only the draw's
 * state and cached vertices. */

/* Primitives per parallel item. */
#define KA_CHUNK 64
/* Triangle primitives a draw needs before it is set up in parallel. */
#define KA_PARALLEL_MIN 128

/* How many primitives assemble makes of n indices, for triangle modes;
 * zero for points and lines, which stay serial. */
static size_t tri_prims(unsigned int mode, size_t n)
{
    switch (mode) {
    case PRIM_TYPE_TRIANGLES:
        return n / 3;
    case PRIM_TYPE_TRIANGLE_STRIP:
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        return n >= 3 ? n - 2 : 0;
    case PRIM_TYPE_QUADS:
        return n / 4;
    case PRIM_TYPE_QUAD_STRIP:
        return n >= 4 ? (n - 4) / 2 + 1 : 0;
    default:
        return 0;
    }
}

/* Primitive p of the list, as assemble's loops draw it. */
static void emit_prim(DrawCtx *c, const DrawSource *src, VCache *vc,
                      const uint32_t *idx, unsigned int mode, size_t p)
{
    KelvinState *pg = c->pg;
    size_t i;
#define V(k) get_vertex(pg, src, vc, idx[k])

    switch (mode) {
    case PRIM_TYPE_TRIANGLES:
        i = 3 * p;
        draw_triangle(c, V(i), V(i + 1), V(i + 2),
                      c->provoking_first ? V(i) : V(i + 2));
        break;
    case PRIM_TYPE_TRIANGLE_STRIP:
        i = p;
        if (i & 1) {
            draw_triangle(c, V(i + 1), V(i), V(i + 2),
                          c->provoking_first ? V(i) : V(i + 2));
        } else {
            draw_triangle(c, V(i), V(i + 1), V(i + 2),
                          c->provoking_first ? V(i) : V(i + 2));
        }
        break;
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        i = p + 1;
        draw_triangle(c, V(0), V(i), V(i + 1),
                      c->provoking_first ? V(0) : V(i + 1));
        break;
    case PRIM_TYPE_QUADS:
        i = 4 * p;
        draw_triangle(c, V(i + 1), V(i + 2), V(i), V(i + 3));
        draw_triangle(c, V(i + 2), V(i + 3), V(i), V(i + 3));
        break;
    case PRIM_TYPE_QUAD_STRIP:
        i = 2 * p;
        draw_triangle(c, V(i), V(i + 1), V(i + 2), V(i + 3));
        draw_triangle(c, V(i + 2), V(i + 1), V(i + 3), V(i + 3));
        break;
    default:
        break;
    }
#undef V
}

typedef struct AssembleWork {
    const DrawCtx *c;
    const DrawSource *src;
    VCache *vc;
    const uint32_t *idx;
    unsigned int mode;
    size_t prims;
    TriChunk *chunks;
} AssembleWork;

static TriChunk *s_chunks;
static size_t s_chunks_cap;

static void assemble_chunk(void *arg, int item, int worker)
{
    const AssembleWork *w = (const AssembleWork *)arg;
    TriChunk *out = &w->chunks[item];
    size_t p = (size_t)item * KA_CHUNK, end = MIN(p + KA_CHUNK, w->prims);
    DrawCtx c = *w->c;

    (void)worker;
    out->n = 0;
    out->triangles = out->clamped = 0;
    out->failed = false;
    c.chunk = out;
    for (; p < end && !out->failed; p++) {
        emit_prim(&c, w->src, w->vc, w->idx, w->mode, p);
    }
}

/* raster_triangle's batching, for a triangle already set up. */
static void append_tri(DrawCtx *c, const TriSetup *t)
{
    RasterBatch *rb;

    if (c->ntris == KR_BATCH) {
        flush_tris(c);
    }
    rb = &s_batch[s_fill % KR_SLOTS];
    if (c->ntris == rb->cap) {
        size_t cap = rb->cap ? rb->cap * 2 : 256;
        TriSetup *grown = (TriSetup *)realloc(rb->tris, cap * sizeof(TriSetup));
        if (!grown) {
            /* Out of memory: scan this one triangle here, after the queue. */
            TriSetup one = *t;
            flush_tris(c);
            kelvin_render_sync();
            rb = &s_batch[s_fill % KR_SLOTS];
            rb->ctx = *c;
            rb->ctx.tris = &one;
            rb->ctx.ntris = 1;
            snapshot_textures(rb, false);
            c->pg->stats.pixels += scan_serial(rb);
            return;
        }
        rb->tris = grown;
        rb->cap = cap;
    }
    c->tris = rb->tris;
    c->tris[c->ntris++] = *t;
}

/* Set the draw's triangles up in parallel and batch them; false, having
 * batched nothing, when the draw does not qualify or memory ran out. Every
 * vertex idx names must be in the cache. */
static bool assemble_parallel(DrawCtx *c, const DrawSource *src, VCache *vc,
                              const uint32_t *idx, size_t n)
{
    unsigned int mode = c->pg->primitive_mode;
    size_t prims = tri_prims(mode, n), chunks, k, i;
    AssembleWork w;

    if (prims < KA_PARALLEL_MIN || !kwork_workers()) {
        return false;
    }
    chunks = (prims + KA_CHUNK - 1) / KA_CHUNK;
    if (chunks > s_chunks_cap) {
        TriChunk *grown = (TriChunk *)realloc(s_chunks, chunks * sizeof(TriChunk));
        if (!grown) {
            return false;
        }
        memset(grown + s_chunks_cap, 0, (chunks - s_chunks_cap) * sizeof(TriChunk));
        s_chunks = grown;
        s_chunks_cap = chunks;
    }
    w.c = c;
    w.src = src;
    w.vc = vc;
    w.idx = idx;
    w.mode = mode;
    w.prims = prims;
    w.chunks = s_chunks;
    kwork_parallel(assemble_chunk, &w, (int)chunks);
    for (k = 0; k < chunks; k++) {
        if (s_chunks[k].failed) {
            return false;
        }
    }
    for (k = 0; k < chunks; k++) {
        const TriChunk *out = &s_chunks[k];
        c->pg->stats.triangles += out->triangles;
        c->pg->stats.triangles_bounds_clamped += out->clamped;
        for (i = 0; i < out->n; i++) {
            append_tri(c, &out->tris[i]);
        }
    }
    return true;
}

/* ---- draw jobs: a draw's geometry on the raster threads -----------------
 *
 * A draw of triangles whose vertices go through a program that writes no
 * constants (kvsh_parallel_ok) is set up after the GPU thread has gone on
 * to the next methods. The GPU thread captures what the draw's geometry
 * reads -- its DrawCtx, vertex constants, attribute formats, inline data and
 * index list -- into a job (geom_capture), and a task on the raster threads
 * (ktask_submit) enters the draw's distinct vertices in the job's vertex
 * cache (phase 0), shades them KVSH_LANES at a time with the job's copy of
 * the vertex state (phase 1), and sets its primitives up in chunks (phase
 * 2), each into a buffer of its own, by the per-primitive code assemble
 * runs, on the same vertices: kvsh_run8 and kvsh_run4 give what kvsh_run
 * gives. Small draws do it all in phase 0.
 *
 * The GPU thread hands a finished job's chunks to the raster threads in
 * draw order, each chunk a batch of its own (geom_submit): a chunk holds at
 * most KR_BATCH triangles, so batches keep their bound. Where a draw's
 * batches split changes no pixel -- a lane scans every band's triangles in
 * order whatever the batch -- except for a draw that samples its own
 * target, whose batches each snapshot it; such draws are not deferred. The
 * batches read the chunks where they are, so a job's slot is reused only
 * once its last batch has finished.
 *
 * Work reaches the raster queue in the order the methods came: anything
 * else that queues work or waits for the queue first finishes every job
 * and hands its triangles over (geom_flush), and so does a draw that is not
 * deferred. GET is held at the oldest draw not yet handed over
 * (kelvin_render_get_bound): its vertex arrays are guest memory the title
 * may reuse once the GPU is past it.
 */

#define KG_JOBS 128             /* job slots */
#define KG_VCHUNK 64            /* vertices per shading item */
/* Fewer distinct vertices, and primitives, are done in phase 0. */
#define KG_EARLY_VERTICES 96
#define KG_EARLY_PRIMS 128
/* Words a draw job may keep GET behind the pusher. Include the retained
 * draw word and up to 255 unpublished words: 4 * (lag + 256) must remain
 * below D3D's 0x2400-byte fence margin. Control transfers finish geometry
 * and publish GET immediately (kelvin.c). The normal default is 2 KiB;
 * the opt-in lookahead experiment stays within the same bound. */
#define KG_GET_LAG 512
#define KG_GET_LOOKAHEAD 1536
#if 4 * (KG_GET_LAG + 256) >= 0x2400 || 4 * (KG_GET_LOOKAHEAD + 256) >= 0x2400
#error KG_GET_LAG exceeds the D3D fence rewrite safety margin
#endif

static int s_gj_get_lag = -1;

static unsigned geom_get_lag_from(const char *value)
{
    return value && strcmp(value, "1536") == 0 ? KG_GET_LOOKAHEAD : KG_GET_LAG;
}

static unsigned geom_get_lag(void)
{
    if (s_gj_get_lag < 0) {
        s_gj_get_lag = (int)geom_get_lag_from(getenv("RECOMP_KELVIN_LOOKAHEAD"));
        if (s_gj_get_lag == KG_GET_LOOKAHEAD) {
            fprintf(stderr, "[KELVIN] RECOMP_KELVIN_LOOKAHEAD: geometry GET lag %u words\n",
                    (unsigned)s_gj_get_lag);
        }
    }
    return (unsigned)s_gj_get_lag;
}

typedef struct GeomChunk {
    size_t run;                      /* its run of the index list */
    size_t first, count;             /* its primitives in that run */
    TriChunk out;
} GeomChunk;

typedef struct GeomJob {
    DrawCtx c;                       /* the draw's state; c.pg for stats only */
    DrawSource src;                  /* reading the copies below */
    KVshState vs;                    /* reading consts */
    unsigned int mode;
    uint32_t get;                    /* the draw's word in the pushbuffer */
    uint64_t words;                  /* words read by then */
    long task;
    long last_batch;                 /* raster job of its last batch, or -1 */
    VertexAttribute attrs[NV2A_VERTEXSHADER_ATTRIBUTES];
    uint32_t consts[NV2A_VERTEXSHADER_CONSTANTS][4];
    uint32_t *idx;                   /* the index list, runs one after another */
    size_t n, idx_cap;
    size_t (*runs)[2];               /* each run's start in idx and length */
    size_t nruns, runs_cap;
    uint32_t *inl;                   /* inline array words or buffer values */
    size_t inl_cap;
    VCache vc;
    size_t vc_cap;
    uint32_t *slots;                 /* the cache entries to shade */
    size_t nslots, slots_cap;
    GeomChunk *chunks;
    size_t nchunks, chunks_cap;
    bool early;                      /* shaded and set up in phase 0 */
    bool verify;
} GeomJob;

static GeomJob *s_gj[KG_JOBS];
/* Jobs captured, handed over, and whose slots are free again; job n is in
 * slot n % KG_JOBS. */
static unsigned long s_gj_head, s_gj_sub, s_gj_free;
/* Handing jobs over: a sync that comes of it only drains the queue. */
static bool s_gj_busy;

static GeomJob *gj_slot(unsigned long n)
{
    return s_gj[n % KG_JOBS];
}

static bool grow_array(void **buf, size_t *cap, size_t need, size_t size)
{
    size_t n = *cap ? *cap : 64;
    void *grown;
    if (need <= *cap) {
        return true;
    }
    while (n < need) {
        n *= 2;
    }
    grown = realloc(*buf, n * size);
    if (!grown) {
        return false;
    }
    *buf = grown;
    *cap = n;
    return true;
}

/* Shade cache entries slots[from..to) with the job's state, KVSH_LANES at
 * a time. */
/* RECOMP_KELVIN_VERIFY_VERTICES: shade vertex k of a group again, one at a
 * time from the same inputs, and compare; and fetch its inputs again, which
 * differ only if guest memory changed under the job. */
static void gj_verify(const GeomJob *j, uint32_t h,
                      const float in[NV2A_VERTEXSHADER_ATTRIBUTES][4])
{
    static volatile LONG64 checked, differed, refetched;
    float again[NV2A_VERTEXSHADER_ATTRIBUTES][4];
    KVertex serial;
    LONG64 c;
    int f;

    kvsh_run_state(&j->vs, in, &serial);
    f = vertex_differs(&serial, &j->vc.verts[h]);
    if (f >= 0 && InterlockedIncrement64(&differed) <= 20) {
        fprintf(stderr, "[KELVIN] vertex check: draw job index %u differs at"
                        " float %d: %.9g serial, %.9g in lanes\n",
                j->vc.keys[h], f, ((const float *)&serial)[f],
                ((const float *)&j->vc.verts[h])[f]);
    }
    fetch_vertex(NULL, &j->src, j->vc.keys[h], again);
    if (memcmp(again, in, sizeof(again)) != 0 &&
        InterlockedIncrement64(&refetched) <= 20) {
        fprintf(stderr, "[KELVIN] vertex check: draw job index %u (source %d,"
                        " attribute 0 at 0x%08X) read differently a second"
                        " time: guest memory changed\n", j->vc.keys[h],
                (int)j->src.kind, j->src.attr_va[0]);
    }
    c = InterlockedIncrement64(&checked);
    if ((c & 0xFFFF) == 0) {
        fprintf(stderr, "[KELVIN] vertex check: %lld draw job vertices"
                        " compared, %lld differed, %lld read differently\n",
                (long long)c, (long long)differed, (long long)refetched);
        fflush(stderr);
    }
}

static void gj_shade_range(GeomJob *j, size_t from, size_t to)
{
    size_t i;
    for (i = from; i < to; i += KVSH_LANES) {
        float in[KVSH_LANES][NV2A_VERTEXSHADER_ATTRIBUTES][4];
        KVertex *out[KVSH_LANES];
        int n = (int)MIN(KVSH_LANES, to - i), k;
        for (k = 0; k < n; k++) {
            uint32_t h = j->slots[i + k];
            fetch_vertex(NULL, &j->src, j->vc.keys[h], in[k]);
            out[k] = &j->vc.verts[h];
        }
#if KVSH_LANES == 8
        kvsh_run8_state(&j->vs, in, out, n);
#else
        kvsh_run4_state(&j->vs, in, out, n);
#endif
        if (j->verify) {
            for (k = 0; k < n; k++) {
                gj_verify(j, j->slots[i + k], in[k]);
            }
        }
    }
}

/* Chunk k's primitives, set up into its buffer. */
static void gj_assemble_chunk(GeomJob *j, size_t k)
{
    GeomChunk *ch = &j->chunks[k];
    DrawCtx c = j->c;
    size_t p, end = ch->first + ch->count;

    ch->out.n = 0;
    ch->out.triangles = ch->out.clamped = 0;
    ch->out.failed = false;
    c.chunk = &ch->out;
    for (p = ch->first; p < end && !ch->out.failed; p++) {
        emit_prim(&c, &j->src, &j->vc, j->idx + j->runs[ch->run][0], j->mode, p);
    }
}

/* Phase 0: the distinct vertices, in the cache as get_vertex looks. */
static void gj_index(void *arg, int item, int worker)
{
    GeomJob *j = (GeomJob *)arg;
    VCache *vc = &j->vc;
    size_t i, prims = 0;

    (void)item;
    (void)worker;
    j->nslots = 0;
    for (i = 0; i < j->n; i++) {
        uint32_t index = j->idx[i];
        size_t h = (index * 2654435761u) & (vc->cap - 1);   /* get_vertex */
        while (vc->stamp[h] == vc->gen && vc->keys[h] != index) {
            h = (h + 1) & (vc->cap - 1);
        }
        if (vc->stamp[h] != vc->gen) {
            vc->stamp[h] = vc->gen;
            vc->keys[h] = index;
            j->slots[j->nslots++] = (uint32_t)h;
        }
    }
    for (i = 0; i < j->nchunks; i++) {
        prims += j->chunks[i].count;
    }
    j->early = j->nslots < KG_EARLY_VERTICES && prims <= KG_EARLY_PRIMS;
    if (j->early) {
        gj_shade_range(j, 0, j->nslots);
        for (i = 0; i < j->nchunks; i++) {
            gj_assemble_chunk(j, i);
        }
    }
}

static void gj_shade(void *arg, int item, int worker)
{
    GeomJob *j = (GeomJob *)arg;
    size_t from = (size_t)item * KG_VCHUNK;
    (void)worker;
    gj_shade_range(j, from, MIN(from + KG_VCHUNK, j->nslots));
}

static void gj_assemble(void *arg, int item, int worker)
{
    (void)worker;
    gj_assemble_chunk((GeomJob *)arg, (size_t)item);
}

static int gj_step(void *arg, int phase, void (**run)(void *, int, int))
{
    GeomJob *j = (GeomJob *)arg;
    switch (phase) {
    case 0:
        *run = gj_index;
        return 1;
    case 1:
        *run = gj_shade;
        return j->early ? 0 : (int)((j->nslots + KG_VCHUNK - 1) / KG_VCHUNK);
    case 2:
        *run = gj_assemble;
        return (int)j->nchunks;
    default:
        return 0;
    }
}

/* Hand job j's triangles over, in its chunks' order, and count its work. */
static void geom_submit(GeomJob *j)
{
    KelvinState *pg = j->c.pg;
    bool failed = false;
    size_t k;

    pg->stats.vertices += j->nslots;
    if (!j->early) {
        pg->stats.vertices_parallel += j->nslots;
    }
    pg->stats.draws_deferred++;
    j->last_batch = -1;
    for (k = 0; k < j->nchunks; k++) {
        failed |= j->chunks[k].out.failed;
    }
    if (failed) {
        /* Out of memory for a chunk: set the draw up here instead, from its
         * shaded vertices, into batches of the ring's own. */
        j->c.chunk = NULL;
        j->c.tris = NULL;
        j->c.ntris = 0;
        for (k = 0; k < j->nchunks; k++) {
            const GeomChunk *ch = &j->chunks[k];
            size_t p;
            for (p = ch->first; p < ch->first + ch->count; p++) {
                emit_prim(&j->c, &j->src, &j->vc, j->idx + j->runs[ch->run][0],
                          j->mode, p);
            }
        }
        flush_tris(&j->c);
        return;
    }
    for (k = 0; k < j->nchunks; k++) {
        TriChunk *out = &j->chunks[k].out;
        pg->stats.triangles += out->triangles;
        pg->stats.triangles_bounds_clamped += out->clamped;
        if (out->n) {
            long b;
            j->c.tris = out->tris;
            j->c.ntris = out->n;
            b = flush_tris(&j->c);
            if (b >= 0) {
                j->last_batch = b;
            }
        }
    }
}

/* Wait for the oldest job not handed over, and hand it over. */
static void geom_finish_one(void)
{
    GeomJob *j = gj_slot(s_gj_sub);
    if (!ktask_done(j->task)) {
        long long t0 = g_kelvin_timing > 0 || s_jt_on ? kelvin_ticks() : 0;
        ktask_wait(j->task);
        if (g_kelvin_timing > 0) {
            g_kelvin_ticks[KT_GEOMETRY] += kelvin_ticks() - t0;
        }
        if (s_jt_on) {
            kelvin_render_trace_event(2, t0, kelvin_ticks(), 0);
        }
    }
    geom_submit(j);
    s_gj_sub++;
}

/* Hand over, in order, every job that has finished. */
static void geom_retire(void)
{
    if (s_gj_busy) {
        return;
    }
    s_gj_busy = true;
    while (s_gj_sub != s_gj_head && ktask_done(gj_slot(s_gj_sub)->task)) {
        geom_submit(gj_slot(s_gj_sub));
        s_gj_sub++;
    }
    s_gj_busy = false;
}

/* Finish every job and hand it over. */
static void geom_flush(void)
{
    if (s_gj_busy) {
        return;
    }
    s_gj_busy = true;
    while (s_gj_sub != s_gj_head) {
        geom_finish_one();
    }
    s_gj_busy = false;
}

void kelvin_render_geometry_sync(void)
{
    geom_flush();
    kgpu_drain();
}

uint32_t kelvin_render_get_bound(uint32_t get, uint64_t words)
{
    unsigned lag = geom_get_lag();
    if (kgpu_async_active()) return kgpu_get_bound(get);
    if (s_gj_sub == s_gj_head) {
        return get;
    }
    geom_retire();
    if (!s_gj_busy) {
        s_gj_busy = true;
        while (s_gj_sub != s_gj_head && gj_slot(s_gj_sub)->words + lag < words) {
            geom_finish_one();
        }
        s_gj_busy = false;
    }
    return s_gj_sub != s_gj_head ? gj_slot(s_gj_sub)->get : get;
}

/* The slot the next job captures into, free: its last job handed over, and
 * that job's batches finished. NULL if out of memory. */
static GeomJob *geom_next_slot(void)
{
    GeomJob *j;
    while (s_gj_head - s_gj_free == KG_JOBS) {
        GeomJob *old = gj_slot(s_gj_free);
        if (s_gj_free == s_gj_sub) {
            s_gj_busy = true;
            geom_finish_one();
            s_gj_busy = false;
        }
        if (old->last_batch >= 0) {
            kwork_wait(old->last_batch);
        }
        s_gj_free++;
    }
    /* Free the slots whose batches are done anyway, oldest first. */
    while (s_gj_free != s_gj_sub &&
           (gj_slot(s_gj_free)->last_batch < 0 ||
            (long)(kwork_finished() - gj_slot(s_gj_free)->last_batch) > 0)) {
        s_gj_free++;
    }
    j = s_gj[s_gj_head % KG_JOBS];
    if (!j) {
        j = (GeomJob *)calloc(1, sizeof(GeomJob));
        s_gj[s_gj_head % KG_JOBS] = j;
    }
    return j;
}

/* Whether the draw's geometry may be set up later, on the raster threads. */
static bool geom_deferrable(KelvinState *pg, const DrawCtx *c)
{
    switch (pg->primitive_mode) {
    case PRIM_TYPE_TRIANGLES:
    case PRIM_TYPE_TRIANGLE_STRIP:
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
    case PRIM_TYPE_QUADS:
    case PRIM_TYPE_QUAD_STRIP:
        break;
    default:
        return false;
    }
    return kwork_workers() && kvsh_parallel_ok() && !needs_snapshot(c);
}

/* Capture the draw in slot j and start its task; false, having started
 * nothing, if memory ran out. c is j->c, src the draw's source over pg. */
static bool geom_capture(KelvinState *pg, GeomJob *j, const DrawSource *src)
{
    unsigned int mode = pg->primitive_mode;
    size_t n = 0, i, k, cap, chunk_prims;

    j->src = *src;
    j->mode = mode;
    j->nruns = 0;
    /* The index list, in runs assembled one by one. */
    if (pg->draw_arrays_length) {
        for (i = 0; i < pg->draw_arrays_length; i++) {
            n += (size_t)pg->draw_arrays_count[i];
        }
        if (!grow_array((void **)&j->idx, &j->idx_cap, n ? n : 1, sizeof(uint32_t)) ||
            !grow_array((void **)&j->runs, &j->runs_cap, pg->draw_arrays_length,
                        sizeof(j->runs[0]))) {
            return false;
        }
        n = 0;
        for (i = 0; i < pg->draw_arrays_length; i++) {
            size_t cnt = (size_t)pg->draw_arrays_count[i], m;
            j->runs[i][0] = n;
            j->runs[i][1] = cnt;
            for (m = 0; m < cnt; m++) {
                j->idx[n + m] = (uint32_t)(pg->draw_arrays_start[i] + (int32_t)m);
            }
            n += cnt;
        }
        j->nruns = pg->draw_arrays_length;
    } else {
        if (pg->inline_elements_length) {
            n = pg->inline_elements_length;
        } else {
            n = src->kind == SRC_INLINE_BUFFER
                    ? pg->inline_buffer_length
                    : (src->vertex_size ? pg->inline_array_length * 4 / src->vertex_size : 0);
        }
        if (!grow_array((void **)&j->idx, &j->idx_cap, n ? n : 1, sizeof(uint32_t)) ||
            !grow_array((void **)&j->runs, &j->runs_cap, 1, sizeof(j->runs[0]))) {
            return false;
        }
        if (pg->inline_elements_length) {
            memcpy(j->idx, pg->inline_elements, n * sizeof(uint32_t));
        } else {
            for (i = 0; i < n; i++) {
                j->idx[i] = (uint32_t)i;
            }
        }
        j->runs[0][0] = 0;
        j->runs[0][1] = n;
        j->nruns = 1;
    }
    j->n = n;

    /* What fetching reads: attribute formats, and the inline data. */
    memcpy(j->attrs, pg->vertex_attributes, sizeof(j->attrs));
    j->src.attrs = j->attrs;
    if (src->kind == SRC_INLINE_ARRAY) {
        if (!grow_array((void **)&j->inl, &j->inl_cap,
                        pg->inline_array_length ? pg->inline_array_length : 1,
                        sizeof(uint32_t))) {
            return false;
        }
        memcpy(j->inl, pg->inline_array, pg->inline_array_length * sizeof(uint32_t));
        j->src.inline_array = j->inl;
    } else if (src->kind == SRC_INLINE_BUFFER) {
        size_t per = (size_t)pg->inline_buffer_length * 4;
        if (!grow_array((void **)&j->inl, &j->inl_cap,
                        per * (src->nactive ? src->nactive : 1), sizeof(uint32_t))) {
            return false;
        }
        for (k = 0; k < (size_t)src->nactive; k++) {
            int a = src->active[k];
            float *copy = (float *)j->inl + k * per;
            memcpy(copy, pg->vertex_attributes[a].inline_buffer, per * sizeof(float));
            j->attrs[a].inline_buffer = copy;
        }
    }
    memcpy(j->consts, pg->vsh_constants, sizeof(j->consts));
    kvsh_state(&j->vs, (const float *)j->consts);

    /* Its vertex cache, and the slot list phase 0 fills. */
    cap = 64;
    while (cap < n * 2) {
        cap <<= 1;
    }
    if (cap > j->vc_cap) {
        uint32_t *keys = (uint32_t *)malloc(cap * sizeof(uint32_t));
        KVertex *verts = (KVertex *)malloc(cap * sizeof(KVertex));
        uint32_t *stamp = (uint32_t *)calloc(cap, sizeof(uint32_t));
        if (!keys || !verts || !stamp) {
            free(keys);
            free(verts);
            free(stamp);
            return false;
        }
        free(j->vc.keys);
        free(j->vc.verts);
        free(j->vc.stamp);
        j->vc.keys = keys;
        j->vc.verts = verts;
        j->vc.stamp = stamp;
        j->vc.gen = 0;
        j->vc_cap = cap;
    }
    if (++j->vc.gen == 0) {
        memset(j->vc.stamp, 0, j->vc_cap * sizeof(uint32_t));
        j->vc.gen = 1;
    }
    j->vc.cap = cap;
    if (!grow_array((void **)&j->slots, &j->slots_cap, n ? n : 1, sizeof(uint32_t))) {
        return false;
    }

    /* Chunks of primitives whose triangles fit a batch: four at most from
     * a quad (two triangles, each clipped into two), two from the rest. */
    chunk_prims = KR_BATCH / (mode == PRIM_TYPE_QUADS || mode == PRIM_TYPE_QUAD_STRIP ? 4 : 2);
    j->nchunks = 0;
    for (i = 0; i < j->nruns; i++) {
        size_t prims = tri_prims(mode, j->runs[i][1]), p;
        for (p = 0; p < prims; p += chunk_prims) {
            GeomChunk *ch;
            if (j->nchunks == j->chunks_cap) {
                size_t old = j->chunks_cap;
                if (!grow_array((void **)&j->chunks, &j->chunks_cap, old + 1,
                                sizeof(GeomChunk))) {
                    return false;
                }
                memset(j->chunks + old, 0, (j->chunks_cap - old) * sizeof(GeomChunk));
            }
            ch = &j->chunks[j->nchunks++];
            ch->run = i;
            ch->first = p;
            ch->count = MIN(chunk_prims, prims - p);
        }
    }
    j->nslots = 0;
    j->early = false;
    j->last_batch = -1;
    j->get = g_kelvin_word_get;
    j->words = pg->stats.words;
    j->verify = verify_vertices();
    j->task = ktask_submit(gj_step, j);
    s_gj_head++;
    return true;
}

void kelvin_render_draw_begin(KelvinState *pg)
{
    (void)pg;
}

void kelvin_render_draw_end(KelvinState *pg)
{
    uint32_t control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    bool color_write = control_0 & (NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE |
                                    NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE |
                                    NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE |
                                    NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE);
    bool depth_test = control_0 & NV_PGRAPH_CONTROL_0_ZENABLE;
    bool stencil_test = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_1) &
                        NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE;

    pg->stats.draws++;
    if (!(color_write || depth_test || stencil_test)) {
        pg->stats.draws_nop++;
        return;
    }
    kelvin_render_flush_draw(pg);
}

static void flush_draw(KelvinState *pg);

/* An attribute's current value becomes what it held for the last vertex
 * (gl/draw.c and gl/vertex.c update inline_value from the provoking
 * element). */
static void update_inline_values(KelvinState *pg, const DrawSource *src)
{
    int a;
    if (src->kind == SRC_INLINE_BUFFER && pg->inline_buffer_length) {
        for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
            VertexAttribute *attr = &pg->vertex_attributes[a];
            if (attr->inline_buffer_populated) {
                memcpy(attr->inline_value,
                       attr->inline_buffer + (pg->inline_buffer_length - 1) * 4,
                       sizeof(attr->inline_value));
                attr->inline_buffer_populated = false;
            }
        }
    }
}

/* ---- the GPU backend's draw glue (kelvin_gpu.c) ------------------------------------
 *
 * The CPU vertex stage runs exactly as for the CPU renderer; its shaded vertices
 * become records for the GPU, and primitive assembly becomes an index list in the
 * order draw_triangle would be called, so winding and provoking vertices agree. */

typedef struct GpuBuild {
    uint32_t *remap, *stamp;        /* vertex cache slot -> record */
    size_t cap;
    uint32_t gen;
    float *recs;                    /* nrec * stride float4s */
    size_t recs_cap;                /* in floats */
    uint32_t *idx;
    size_t idx_cap;
    uint32_t nrec, nidx, stride;
    int slot[KG_SLOTS];
    bool flat;
} GpuBuild;
static GpuBuild s_gb;

static bool gb_reserve(size_t recs_floats, size_t idx)
{
    if (recs_floats > s_gb.recs_cap) {
        size_t cap = s_gb.recs_cap ? s_gb.recs_cap : 65536;
        float *g;
        while (cap < recs_floats) cap *= 2;
        g = (float *)realloc(s_gb.recs, cap * sizeof(float));
        if (!g) return false;
        s_gb.recs = g; s_gb.recs_cap = cap;
    }
    if (idx > s_gb.idx_cap) {
        size_t cap = s_gb.idx_cap ? s_gb.idx_cap : 65536;
        uint32_t *g;
        while (cap < idx) cap *= 2;
        g = (uint32_t *)realloc(s_gb.idx, cap * sizeof(uint32_t));
        if (!g) return false;
        s_gb.idx = g; s_gb.idx_cap = cap;
    }
    return true;
}

static void gb_pack(float *r, const KVertex *v, const KVertex *prov)
{
    int s = s_gb.slot[KG_POS];
    memcpy(r + 4 * s, v->pos, 16);
    if ((s = s_gb.slot[KG_D0]) >= 0) memcpy(r + 4 * s, prov ? prov->d0 : v->d0, 16);
    if ((s = s_gb.slot[KG_D1]) >= 0) memcpy(r + 4 * s, prov ? prov->d1 : v->d1, 16);
    if ((s = s_gb.slot[KG_FOG]) >= 0) { r[4 * s] = v->fog; r[4 * s + 1] = r[4 * s + 2] = r[4 * s + 3] = 0.0f; }
    if ((s = s_gb.slot[KG_T0]) >= 0) memcpy(r + 4 * s, v->t[0], 16);
    if ((s = s_gb.slot[KG_T1]) >= 0) memcpy(r + 4 * s, v->t[1], 16);
    if ((s = s_gb.slot[KG_T2]) >= 0) memcpy(r + 4 * s, v->t[2], 16);
    if ((s = s_gb.slot[KG_T3]) >= 0) memcpy(r + 4 * s, v->t[3], 16);
}

/* The record for a shaded vertex of this draw; flat shading gives every triangle
 * its own copies carrying the provoking vertex's colours. */
static uint32_t gb_vertex(const VCache *vc, const KVertex *v, const KVertex *prov)
{
    uint32_t rec;
    if (s_gb.flat) {
        rec = s_gb.nrec++;
        gb_pack(s_gb.recs + (size_t)rec * s_gb.stride * 4, v, prov);
        return rec;
    }
    {
        size_t h = (size_t)(v - vc->verts);
        if (s_gb.stamp[h] == s_gb.gen) return s_gb.remap[h];
        rec = s_gb.nrec++;
        gb_pack(s_gb.recs + (size_t)rec * s_gb.stride * 4, v, NULL);
        s_gb.stamp[h] = s_gb.gen;
        s_gb.remap[h] = rec;
        return rec;
    }
}

static void gb_tri(const VCache *vc, const KVertex *a, const KVertex *b, const KVertex *c,
                   const KVertex *prov)
{
    s_gb.idx[s_gb.nidx++] = gb_vertex(vc, a, prov);
    s_gb.idx[s_gb.nidx++] = gb_vertex(vc, b, prov);
    s_gb.idx[s_gb.nidx++] = gb_vertex(vc, c, prov);
}

/* A line as a quad one guest pixel wide (the CPU rasterizer's width-1 lines);
 * culling is off for line draws, so its winding does not matter. */
static void gb_line(const KVertex *a, const KVertex *b)
{
    float dx = b->pos[0] - a->pos[0], dy = b->pos[1] - a->pos[1];
    float len = sqrtf(dx * dx + dy * dy), nx, ny;
    KVertex c[4];
    uint32_t r[4];
    int k;
    if (!(len > 0.0f) || !isfinite(len)) return;
    nx = -dy / len * 0.5f;
    ny = dx / len * 0.5f;
    c[0] = *a; c[1] = *a; c[2] = *b; c[3] = *b;
    c[0].pos[0] += nx; c[0].pos[1] += ny;
    c[1].pos[0] -= nx; c[1].pos[1] -= ny;
    c[2].pos[0] += nx; c[2].pos[1] += ny;
    c[3].pos[0] -= nx; c[3].pos[1] -= ny;
    for (k = 0; k < 4; k++) {
        r[k] = s_gb.nrec++;
        gb_pack(s_gb.recs + (size_t)r[k] * s_gb.stride * 4, &c[k], s_gb.flat ? a : NULL);
    }
    s_gb.idx[s_gb.nidx++] = r[0]; s_gb.idx[s_gb.nidx++] = r[1]; s_gb.idx[s_gb.nidx++] = r[2];
    s_gb.idx[s_gb.nidx++] = r[2]; s_gb.idx[s_gb.nidx++] = r[1]; s_gb.idx[s_gb.nidx++] = r[3];
}

/* assemble()'s primitives, as records and indices. */
static void gpu_assemble(DrawCtx *c, const DrawSource *src, VCache *vc,
                         const uint32_t *idx, size_t n)
{
    KelvinState *pg = c->pg;
    size_t i, tris = tri_prims(pg->primitive_mode, n);
    unsigned int mode = pg->primitive_mode;
    size_t quad = (mode == PRIM_TYPE_QUADS || mode == PRIM_TYPE_QUAD_STRIP) ? 2 : 1;
    if (mode >= PRIM_TYPE_LINES && mode <= PRIM_TYPE_LINE_STRIP) {
        size_t segs = mode == PRIM_TYPE_LINES ? n / 2 : n < 2 ? 0 : (mode == PRIM_TYPE_LINE_LOOP ? n : n - 1);
        if (!gb_reserve(((size_t)s_gb.nrec + 4 * segs) * s_gb.stride * 4, (size_t)s_gb.nidx + 6 * segs)) return;
        for (i = 0; i < segs; i++) {
            size_t a = mode == PRIM_TYPE_LINES ? 2 * i : i;
            size_t b = mode == PRIM_TYPE_LINES ? 2 * i + 1 : (i + 1) % n;
            gb_line(get_vertex(pg, src, vc, idx[a]), get_vertex(pg, src, vc, idx[b]));
        }
        return;
    }
    if (!gb_reserve(((size_t)s_gb.nrec + (s_gb.flat ? 3 * tris * quad : n)) * s_gb.stride * 4,
                    (size_t)s_gb.nidx + 3 * tris * quad)) {
        return;
    }
#define V(k) get_vertex(pg, src, vc, idx[k])
#define PROV(a, b) (s_gb.flat ? (c->provoking_first ? (a) : (b)) : NULL)
    switch (mode) {
    case PRIM_TYPE_TRIANGLES:
        for (i = 0; i + 2 < n; i += 3) {
            const KVertex *p0 = V(i), *p1 = V(i + 1), *p2 = V(i + 2);
            gb_tri(vc, p0, p1, p2, PROV(p0, p2));
        }
        break;
    case PRIM_TYPE_TRIANGLE_STRIP:
        for (i = 0; i + 2 < n; i++) {
            const KVertex *p0 = V(i), *p1 = V(i + 1), *p2 = V(i + 2);
            if (i & 1) gb_tri(vc, p1, p0, p2, PROV(p0, p2));
            else gb_tri(vc, p0, p1, p2, PROV(p0, p2));
        }
        break;
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        for (i = 1; i + 1 < n; i++) {
            const KVertex *p0 = V(0), *p1 = V(i), *p2 = V(i + 1);
            gb_tri(vc, p0, p1, p2, PROV(p0, p2));
        }
        break;
    case PRIM_TYPE_QUADS:
        for (i = 0; i + 3 < n; i += 4) {
            const KVertex *q0 = V(i), *q1 = V(i + 1), *q2 = V(i + 2), *q3 = V(i + 3);
            gb_tri(vc, q1, q2, q0, s_gb.flat ? q3 : NULL);
            gb_tri(vc, q2, q3, q0, s_gb.flat ? q3 : NULL);
        }
        break;
    case PRIM_TYPE_QUAD_STRIP:
        for (i = 0; i + 3 < n; i += 2) {
            const KVertex *q0 = V(i), *q1 = V(i + 1), *q2 = V(i + 2), *q3 = V(i + 3);
            gb_tri(vc, q0, q1, q2, s_gb.flat ? q3 : NULL);
            gb_tri(vc, q2, q1, q3, s_gb.flat ? q3 : NULL);
        }
        break;
    default:
        break;
    }
#undef V
#undef PROV
}

/* The triangle list of a primitive run as element numbers (the GPU shades the
 * vertices itself): the same order assemble() and emit_prim() use. */
static void gb_idx_tri(uint32_t a, uint32_t b, uint32_t c)
{
    s_gb.idx[s_gb.nidx++] = a;
    s_gb.idx[s_gb.nidx++] = b;
    s_gb.idx[s_gb.nidx++] = c;
}

/* Triangle (a, b, c) with the provoking vertex first, the orientation kept; pv is its position in (a, b, c). */
static void gb_idx_tri_pv(uint32_t a, uint32_t b, uint32_t c, int pv)
{
    if (pv == 0) gb_idx_tri(a, b, c);
    else if (pv == 1) gb_idx_tri(b, c, a);
    else gb_idx_tri(c, a, b);
}

/* gpu_assemble_idx for flat shading: D3D11 takes flat colours from a triangle's first vertex, so each triangle starts with
 * the vertex the NV2A takes them from (the first or last of the primitive's triangle; the last vertex of a quad). */
static bool gpu_assemble_idx_flat(unsigned int mode, const uint32_t *idx, size_t n, bool first)
{
    size_t i, tris = tri_prims(mode, n);
    size_t quad = (mode == PRIM_TYPE_QUADS || mode == PRIM_TYPE_QUAD_STRIP) ? 2 : 1;
    if (!gb_reserve(0, (size_t)s_gb.nidx + 3 * tris * quad)) return false;
    switch (mode) {
    case PRIM_TYPE_TRIANGLES:
        for (i = 0; i + 2 < n; i += 3) gb_idx_tri_pv(idx[i], idx[i + 1], idx[i + 2], first ? 0 : 2);
        break;
    case PRIM_TYPE_TRIANGLE_STRIP:
        for (i = 0; i + 2 < n; i++) {
            if (i & 1) gb_idx_tri_pv(idx[i + 1], idx[i], idx[i + 2], first ? 1 : 2);
            else gb_idx_tri_pv(idx[i], idx[i + 1], idx[i + 2], first ? 0 : 2);
        }
        break;
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        for (i = 1; i + 1 < n; i++) gb_idx_tri_pv(idx[0], idx[i], idx[i + 1], first ? 0 : 2);
        break;
    case PRIM_TYPE_QUADS:
        for (i = 0; i + 3 < n; i += 4) {
            gb_idx_tri(idx[i + 3], idx[i], idx[i + 1]);
            gb_idx_tri(idx[i + 3], idx[i + 1], idx[i + 2]);
        }
        break;
    case PRIM_TYPE_QUAD_STRIP:
        for (i = 0; i + 3 < n; i += 2) {
            gb_idx_tri(idx[i + 3], idx[i], idx[i + 1]);
            gb_idx_tri(idx[i + 3], idx[i + 2], idx[i]);
        }
        break;
    default:
        break;
    }
    return true;
}

static bool gpu_assemble_idx(unsigned int mode, const uint32_t *idx, size_t n)
{
    size_t i, tris = tri_prims(mode, n);
    size_t quad = (mode == PRIM_TYPE_QUADS || mode == PRIM_TYPE_QUAD_STRIP) ? 2 : 1;
    if (!gb_reserve(0, (size_t)s_gb.nidx + 3 * tris * quad)) return false;
    switch (mode) {
    case PRIM_TYPE_TRIANGLES:
        for (i = 0; i + 2 < n; i += 3) gb_idx_tri(idx[i], idx[i + 1], idx[i + 2]);
        break;
    case PRIM_TYPE_TRIANGLE_STRIP:
        for (i = 0; i + 2 < n; i++) {
            if (i & 1) gb_idx_tri(idx[i + 1], idx[i], idx[i + 2]);
            else gb_idx_tri(idx[i], idx[i + 1], idx[i + 2]);
        }
        break;
    case PRIM_TYPE_TRIANGLE_FAN:
    case PRIM_TYPE_POLYGON:
        for (i = 1; i + 1 < n; i++) gb_idx_tri(idx[0], idx[i], idx[i + 1]);
        break;
    case PRIM_TYPE_QUADS:
        for (i = 0; i + 3 < n; i += 4) {
            gb_idx_tri(idx[i + 1], idx[i + 2], idx[i]);
            gb_idx_tri(idx[i + 2], idx[i + 3], idx[i]);
        }
        break;
    case PRIM_TYPE_QUAD_STRIP:
        for (i = 0; i + 3 < n; i += 2) {
            gb_idx_tri(idx[i], idx[i + 1], idx[i + 2]);
            gb_idx_tri(idx[i + 2], idx[i + 1], idx[i + 3]);
        }
        break;
    default:
        break;
    }
    return true;
}

/* Where each attribute's data lives, for the GPU vertex path. */
static unsigned int gpu_attr_format(unsigned int f)
{
    switch (f) {
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_D3D: return 1;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_OGL: return 2;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S1: return 3;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_F: return 4;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S32K: return 5;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_CMP: return 6;
    default: return 0;
    }
}

static void gpu_attr_sources(KelvinState *pg, const DrawSource *src, uint32_t minidx,
                             uint32_t maxidx, KGpuVsDraw *g)
{
    int a;
    for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES && a < 16; a++) {
        const VertexAttribute *at = &pg->vertex_attributes[a];
        KGpuAttrSrc *o = &g->attr[a];
        bool per;
        memset(o, 0, sizeof(*o));
        memcpy(o->inline_value, at->inline_value, 16);
        if (src->kind == SRC_ARRAYS) per = at->count && src->attr_va[a];
        else if (src->kind == SRC_INLINE_ARRAY) per = at->count != 0;
        else per = at->inline_buffer_populated;
        if (!per) continue;
        if (src->kind == SRC_INLINE_BUFFER) {
            o->fmt = 4; o->count = 4; o->stride = 16;
            o->data = (const uint8_t *)at->inline_buffer;
            o->lo = 0;
            o->hi = (int64_t)pg->inline_buffer_length * 16;
            continue;
        }
        o->fmt = gpu_attr_format(at->format);
        if (!o->fmt) {
            o->inline_value[0] = o->inline_value[1] = o->inline_value[2] = 0.0f;
            o->inline_value[3] = 1.0f;
            continue;
        }
        o->count = at->count;
        if (src->kind == SRC_INLINE_ARRAY) {
            unsigned int nverts = src->vertex_size ? src->inline_array_length * 4 / src->vertex_size : 0;
            o->stride = src->vertex_size;
            o->data = (const uint8_t *)src->inline_array + at->inline_array_offset;
            o->lo = 0;
            o->hi = nverts ? (int64_t)(nverts - 1) * src->vertex_size + at->size * at->count : 0;
        } else {
            uint32_t va = src->attr_va[a];
            uint64_t first = (uint64_t)va + (uint64_t)minidx * at->stride;
            uint64_t last = (uint64_t)va + (uint64_t)maxidx * at->stride + (uint64_t)at->size * at->count;
            if (first < KELVIN_PHYS_WINDOW || last > (uint64_t)KELVIN_PHYS_WINDOW + KELVIN_PHYS_SIZE) {
                o->fmt = 0;             /* the CPU stage reads the inline value out of range */
                continue;
            }
            o->stride = at->stride;
            o->data = kelvin_guest(va);
            o->lo = at->stride ? (int64_t)minidx * at->stride : 0;
            o->hi = at->stride ? (int64_t)maxidx * at->stride + at->size * at->count : at->size * at->count;
        }
    }
}

static bool combiner_reads_fog(const KPixelState *ps)
{
    int i, k;
    for (i = 0; i < ps->num_stages; i++) {
        for (k = 0; k < 4; k++) {
            if (ps->stage[i].rgb[k].reg == PS_REGISTER_FOG || ps->stage[i].alpha[k].reg == PS_REGISTER_FOG) return true;
        }
    }
    if (ps->final_enabled) {
        for (k = 0; k < 7; k++) if (ps->fin[k].reg == PS_REGISTER_FOG) return true;
    }
    return false;
}

/* A draw whose vertices the GPU shades: the index list over the draw's element numbers. Triangle lists and strips go to
 * the GPU as they are (a strip draws as a strip; the runs of a multi-range draw are joined with restart indices);
 * the other primitives are assembled into triangle lists here. */
static bool gpu_draw_gvs(KelvinState *pg, DrawCtx *c, const DrawSource *src,
                         const KVshGpuInfo *gi)
{
    static KGpuVsDraw gv;
    static uint32_t *s_tmp;
    static size_t s_tmp_cap;
    KGpuDraw gd;
    unsigned int mode = pg->primitive_mode;
    uint32_t minidx = UINT32_MAX, maxidx = 0;
    size_t i, n = 0, j, tris = 0;
    long long t0 = g_kelvin_timing > 0 ? kelvin_ticks() : 0;
    bool ok = true;
    unsigned int topo = KG_TOPO_LIST;
    bool direct = c->ps.smooth && (mode == PRIM_TYPE_TRIANGLES || mode == PRIM_TYPE_TRIANGLE_STRIP);
    bool strip = mode == PRIM_TYPE_TRIANGLE_STRIP;

    s_gb.nidx = 0;
    if (direct) {
        topo = strip ? KG_TOPO_STRIP : KG_TOPO_LIST;
        if (pg->draw_arrays_length) {
            size_t total = 0;
            for (i = 0; i < pg->draw_arrays_length; i++) total += (size_t)pg->draw_arrays_count[i] + 1;
            if (!gb_reserve(0, total)) return false;
            for (i = 0; i < pg->draw_arrays_length; i++) {
                size_t cnt = (size_t)pg->draw_arrays_count[i];
                uint32_t start = (uint32_t)pg->draw_arrays_start[i];
                if (!strip) cnt -= cnt % 3;
                if (cnt < 3) continue;
                if (strip && s_gb.nidx) s_gb.idx[s_gb.nidx++] = 0xFFFFFFFFu;
                for (j = 0; j < cnt; j++) s_gb.idx[s_gb.nidx + j] = start + (uint32_t)j;
                s_gb.nidx += (uint32_t)cnt;
                if (start < minidx) minidx = start;
                if (start + (uint32_t)cnt - 1 > maxidx) maxidx = start + (uint32_t)cnt - 1;
                tris += strip ? cnt - 2 : cnt / 3;
            }
        } else {
            if (pg->inline_elements_length) {
                n = pg->inline_elements_length;
                if (!gb_reserve(0, n)) return false;
                memcpy(s_gb.idx, pg->inline_elements, n * sizeof(uint32_t));
            } else {
                n = src->kind == SRC_INLINE_BUFFER
                        ? pg->inline_buffer_length
                        : (src->vertex_size ? pg->inline_array_length * 4 / src->vertex_size : 0);
                if (!gb_reserve(0, n ? n : 1)) return false;
                for (i = 0; i < n; i++) s_gb.idx[i] = (uint32_t)i;
            }
            if (!strip) n -= n % 3;
            if (n >= 3) {
                uint32_t lo = UINT32_MAX, hi = 0;
                for (i = 0; i < n; i++) {
                    uint32_t v = s_gb.idx[i];
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
                s_gb.nidx = (uint32_t)n;
                minidx = lo; maxidx = hi;
                tris = strip ? n - 2 : n / 3;
            }
        }
    } else if (pg->draw_arrays_length) {
        for (i = 0; i < pg->draw_arrays_length && ok; i++) {
            size_t cnt = (size_t)pg->draw_arrays_count[i];
            uint32_t start = (uint32_t)pg->draw_arrays_start[i];
            if (cnt > s_tmp_cap) {
                uint32_t *g = (uint32_t *)realloc(s_tmp, cnt * sizeof(uint32_t));
                if (!g) return false;
                s_tmp = g; s_tmp_cap = cnt;
            }
            for (j = 0; j < cnt; j++) s_tmp[j] = start + (uint32_t)j;
            if (cnt) {
                if (start < minidx) minidx = start;
                if (start + (uint32_t)cnt - 1 > maxidx) maxidx = start + (uint32_t)cnt - 1;
            }
            ok = c->ps.smooth ? gpu_assemble_idx(mode, s_tmp, cnt) : gpu_assemble_idx_flat(mode, s_tmp, cnt, c->provoking_first);
        }
        tris = s_gb.nidx / 3;
    } else {
        if (pg->inline_elements_length) {
            n = pg->inline_elements_length;
            if (n > s_tmp_cap) {
                uint32_t *g = (uint32_t *)realloc(s_tmp, n * sizeof(uint32_t));
                if (!g) return false;
                s_tmp = g; s_tmp_cap = n;
            }
            memcpy(s_tmp, pg->inline_elements, n * sizeof(uint32_t));
        } else {
            n = src->kind == SRC_INLINE_BUFFER
                    ? pg->inline_buffer_length
                    : (src->vertex_size ? pg->inline_array_length * 4 / src->vertex_size : 0);
            if (n > s_tmp_cap) {
                uint32_t *g = (uint32_t *)realloc(s_tmp, (n ? n : 1) * sizeof(uint32_t));
                if (!g) return false;
                s_tmp = g; s_tmp_cap = n ? n : 1;
            }
            for (i = 0; i < n; i++) s_tmp[i] = (uint32_t)i;
        }
        for (i = 0; i < n; i++) {
            if (s_tmp[i] < minidx) minidx = s_tmp[i];
            if (s_tmp[i] > maxidx) maxidx = s_tmp[i];
        }
        ok = c->ps.smooth ? gpu_assemble_idx(mode, s_tmp, n) : gpu_assemble_idx_flat(mode, s_tmp, n, c->provoking_first);
        tris = s_gb.nidx / 3;
    }
    if (!ok) return false;
    if (!s_gb.nidx) return true;
    gpu_attr_sources(pg, src, minidx, maxidx, &gv);
    gv.program = gi->program;
    gv.hash = gi->hash;
    gv.start = gi->start;
    gv.constants = gi->constants;
    gv.fog_enable = gi->fog_enable;
    gv.fog_mode = gi->fog_mode;
    gv.fog_param[0] = gi->fog_param[0];
    gv.fog_param[1] = gi->fog_param[1];
    gv.specular_enable = gi->specular_enable;
    gv.ignore_specular_alpha = gi->ignore_specular_alpha;
    memset(&gd, 0, sizeof(gd));
    gd.t = &c->t;
    gd.ps = &c->ps;
    gd.cull_enable = c->cull_enable;
    gd.cull_ctrl = c->cull_ctrl;
    gd.ccw_front = c->ccw_front;
    gd.zbias = c->poly_offset ? c->depth_offset : 0.0f;
    gd.zslope = c->poly_offset ? c->depth_factor : 0.0f;
    gd.gvs = &gv;
    gd.indices = s_gb.idx;
    gd.icount = s_gb.nidx;
    gd.topology = topo;
    gd.index_limit = maxidx < 0xFFFFFFFEu ? maxidx + 1 : 0;
    pg->stats.triangles += tris;
    if (g_kelvin_timing > 0) g_kgpu_ticks[KGP_VERTEX] += kelvin_ticks() - t0;
    kgpu_draw(&gd);
    return true;
}

volatile long g_kgpu_cpuv[8];
volatile long g_kgpu_flat_mode[16];
volatile long long g_kgpu_flat_elems;
static int s_nodraw;
static void gpu_flush_draw(KelvinState *pg, bool commit_inline)
{
    DrawCtx c;
    DrawSource src;
    VCache vc;
    KGpuDraw gd;
    uint32_t setupraster = pgraph_reg_r(pg, NV_PGRAPH_SETUPRASTER);
    uint32_t *idx = NULL;
    size_t n = 0, i, total = 0;
    int a;
    unsigned int mode = pg->primitive_mode;
    long long vert_t0;

    {
        /* RECOMP_KGPU_NODRAW=1: drop every draw before any work (the engine's own limit: guest main thread
         * and presentation only). =2: do the CPU work of the draw, skip only the Direct3D submission. */
        static int nodraw = -1;
        if (nodraw < 0) { const char *e = getenv("RECOMP_KGPU_NODRAW"); nodraw = e && *e ? atoi(e) : 0; }
        if (nodraw == 1) return;
        s_nodraw = nodraw;
    }
    memset(&c, 0, sizeof(c));
    memset(&src, 0, sizeof(src));
    c.pg = pg;
    if (!kelvin_bind_target(pg, &c.t, true, kres_draw_needs_zeta(pg))) {
        return;
    }
    if (c.t.guest_width != 640 || c.t.guest_height != 480) {
        static uint32_t seen[16];
        static unsigned nseen;
        unsigned k;
        for (k = 0; k < nseen && seen[k] != c.t.color_va; k++) {}
        if (k == nseen && nseen < 16) {
            seen[nseen++] = c.t.color_va;
            fprintf(stderr, "[KGPU] small target 0x%08X %ux%u %s fmt %u pitch %u zeta 0x%08X\n", c.t.color_va, c.t.guest_width,
                    c.t.guest_height, c.t.swizzled ? "swizzled" : "linear", pg->surface_shape.color_format,
                    c.t.guest_color_pitch, c.t.zeta_va);
        }
    }
    if (mode < PRIM_TYPE_LINES || mode > PRIM_TYPE_POLYGON) {
        kelvin_unsupported(KU_POINT_PRIM, "GPU backend: point primitive", mode);
        return;
    }
    kvsh_begin_draw(pg);
    kpix_setup(pg, &c.ps, &c.t);
    for (i = 0; i < 4; i++) {
        const KTexture *tx = &c.ps.tex[i];
        c.tex_needs_lod[i] = tx->valid && !c.ps.lod_free[i] &&
            (tx->levels > 1 || tx->min_filter != tx->mag_filter);
    }
    skip_unread_varyings(&c);
    c.cull_enable = setupraster & NV_PGRAPH_SETUPRASTER_CULLENABLE;
    c.cull_ctrl = GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_CULLCTRL);
    c.ccw_front = GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACE);
    {
        /* RECOMP_KGPU_DBG=cull,alpha,depth,blend (lab diagnostic): switch those stages off for every draw. */
        static const char *dbg = (const char *)-1;
        if (dbg == (const char *)-1) dbg = getenv("RECOMP_KGPU_DBG");
        if (dbg) {
            if (strstr(dbg, "cull")) c.cull_enable = false;
            if (strstr(dbg, "alpha")) c.ps.alpha_test = false;
            if (strstr(dbg, "depth")) c.ps.depth_test = false;
            if (strstr(dbg, "blend")) c.ps.blend = false;
        }
    }
    c.provoking_first = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CONTROL_3),
                                 NV_PGRAPH_CONTROL_3_PROVOKING_VERTEX)
                        == NV_PGRAPH_CONTROL_3_PROVOKING_VERTEX_FIRST;
    if (GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACEMODE) !=
        NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_FILL) {
        kelvin_unsupported(KU_POLY_MODE, "polygon mode",
                           GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACEMODE));
    }
    /* Polygon offset for filled triangles (the game's bullet-hole and shadow decals): a depth bias in zeta units and a
     * slope factor, which the rasterizer state applies (kgpu get_raster). */
    c.poly_offset = false;
    c.depth_offset = c.depth_factor = 0.0f;
    {
        static int poff_on = -1;
        if (poff_on < 0) { const char *e = getenv("RECOMP_KGPU_POLY_OFFSET"); poff_on = !(e && *e == '0'); }
        if (!poff_on) setupraster &= ~NV_PGRAPH_SETUPRASTER_POFFSETFILLENABLE;
    }
    if ((setupraster & NV_PGRAPH_SETUPRASTER_POFFSETFILLENABLE) && mode >= PRIM_TYPE_TRIANGLES &&
        GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACEMODE) == NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_FILL) {
        c.poly_offset = true;
        c.depth_offset = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_ZOFFSETBIAS));
        c.depth_factor = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_ZOFFSETFACTOR));
    }

    if (pg->draw_arrays_length || pg->inline_elements_length) {
        src.kind = SRC_ARRAYS;
        for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
            VertexAttribute *attr = &pg->vertex_attributes[a];
            if (attr->count) {
                src.attr_va[a] = kelvin_dma_va(attr->dma_select ? pg->dma_vertex_b : pg->dma_vertex_a,
                                               attr->offset, 0);
            }
        }
    } else if (pg->inline_buffer_length) {
        src.kind = SRC_INLINE_BUFFER;
    } else if (pg->inline_array_length) {
        unsigned int offset = 0;
        src.kind = SRC_INLINE_ARRAY;
        for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
            VertexAttribute *attr = &pg->vertex_attributes[a];
            if (!attr->count) continue;
            offset = ROUND_UP(offset, attr->size);
            attr->inline_array_offset = offset;
            offset += attr->size * attr->count;
            offset = ROUND_UP(offset, attr->size);
        }
        src.vertex_size = offset;
    } else {
        return;
    }
    plan_fetch(pg, &src);

    /* record layout: position, then what the pixel stage reads */
    {
        int k = 1;
        s_gb.slot[KG_POS] = 0;
        s_gb.slot[KG_D0] = !c.var_skip[0] ? k++ : -1;
        s_gb.slot[KG_D1] = !c.var_skip[4] ? k++ : -1;
        s_gb.slot[KG_FOG] = combiner_reads_fog(&c.ps) ? k++ : -1;
        for (i = 0; i < 4; i++) s_gb.slot[KG_T0 + i] = !c.var_skip[17 + 4 * i] ? k++ : -1;
        s_gb.stride = (uint32_t)k;
        s_gb.flat = !c.ps.smooth;
        s_gb.nrec = s_gb.nidx = 0;
    }

    if (trace_frame()) trace_draw_gpu(&c, &src, (size_t)(pg->inline_elements_length + pg->inline_array_length));
    /* counts of why a draw goes through the CPU vertex stage (reported by kgpu_report) */
    if (kgpu_vs_enabled() && mode >= PRIM_TYPE_TRIANGLES) {
        KVshGpuInfo gi;
        if (kvsh_gpu_info(&gi)) {
            if (kgpu_vs_ready(gi.program, gi.hash, gi.start, gi.words, gi.nwords)) {
                if (gpu_draw_gvs(pg, &c, &src, &gi)) {
                    if (commit_inline) update_inline_values(pg, &src);
                    return;
                }
                g_kgpu_cpuv[5]++;
            } else {
                g_kgpu_cpuv[4]++;
            }
        } else {
            g_kgpu_cpuv[3]++;
        }
    } else if (!kgpu_vs_enabled()) {
        g_kgpu_cpuv[0]++;
    } else if (!c.ps.smooth) {
        size_t el = 0, q;
        g_kgpu_cpuv[1]++;
        g_kgpu_flat_mode[mode & 15]++;
        if (pg->draw_arrays_length) for (q = 0; q < pg->draw_arrays_length; q++) el += (size_t)pg->draw_arrays_count[q];
        else if (pg->inline_elements_length) el = pg->inline_elements_length;
        else if (src.vertex_size) el = pg->inline_array_length * 4 / src.vertex_size;
        else el = pg->inline_buffer_length;
        g_kgpu_flat_elems += el;
    } else {
        g_kgpu_cpuv[2]++;
    }

    if (pg->draw_arrays_length) {
        for (i = 0; i < pg->draw_arrays_length; i++) total += (size_t)pg->draw_arrays_count[i];
        vcache_init(&vc, total);
    } else {
        if (pg->inline_elements_length) {
            n = pg->inline_elements_length;
            idx = (uint32_t *)malloc(n * sizeof(uint32_t));
            memcpy(idx, pg->inline_elements, n * sizeof(uint32_t));
        } else {
            n = src.kind == SRC_INLINE_BUFFER
                    ? pg->inline_buffer_length
                    : (src.vertex_size ? pg->inline_array_length * 4 / src.vertex_size : 0);
            idx = (uint32_t *)malloc((n ? n : 1) * sizeof(uint32_t));
            for (i = 0; i < n; i++) idx[i] = (uint32_t)i;
        }
        vcache_init(&vc, n);
    }
    /* the remap table follows the vertex cache's size */
    if (vc.cap > s_gb.cap) {
        free(s_gb.remap); free(s_gb.stamp);
        s_gb.remap = (uint32_t *)malloc(vc.cap * sizeof(uint32_t));
        s_gb.stamp = (uint32_t *)calloc(vc.cap, sizeof(uint32_t));
        s_gb.cap = vc.cap;
        s_gb.gen = 0;
    }
    if (++s_gb.gen == 0) { memset(s_gb.stamp, 0, s_gb.cap * sizeof(uint32_t)); s_gb.gen = 1; }

    vert_t0 = kelvin_ticks();
    if (pg->draw_arrays_length) {
        for (i = 0; i < pg->draw_arrays_length; i++) {
            size_t cnt = (size_t)pg->draw_arrays_count[i], j;
            uint32_t *run = (uint32_t *)malloc((cnt ? cnt : 1) * sizeof(uint32_t));
            for (j = 0; j < cnt; j++) run[j] = (uint32_t)(pg->draw_arrays_start[i] + (int32_t)j);
            preshade(pg, &src, &vc, run, cnt);
            gpu_assemble(&c, &src, &vc, run, cnt);
            free(run);
        }
    } else {
        preshade(pg, &src, &vc, idx, n);
        gpu_assemble(&c, &src, &vc, idx, n);
    }
    free(idx);
    g_kgpu_ticks[KGP_VERTEX] += kelvin_ticks() - vert_t0;

    if (s_gb.nidx) {
        memset(&gd, 0, sizeof(gd));
        gd.t = &c.t;
        gd.ps = &c.ps;
        gd.cull_enable = c.cull_enable && mode >= PRIM_TYPE_TRIANGLES;
        gd.cull_ctrl = c.cull_ctrl;
        gd.ccw_front = c.ccw_front;
        gd.zbias = c.poly_offset ? c.depth_offset : 0.0f;
        gd.zslope = c.poly_offset ? c.depth_factor : 0.0f;
        gd.recs = s_gb.recs;
        gd.vcount = s_gb.nrec;
        gd.stride = s_gb.stride;
        for (i = 0; i < KG_SLOTS; i++) gd.slot[i] = s_gb.slot[i];
        gd.indices = s_gb.idx;
        gd.icount = s_gb.nidx;
        pg->stats.triangles += s_gb.nidx / 3;
        if (s_nodraw != 2) kgpu_draw(&gd);
    }
    if (commit_inline) update_inline_values(pg, &src);
}

void kelvin_render_flush_draw(KelvinState *pg)
{
    long long t0;
    if (g_kelvin_timing < 0) {
        g_kelvin_timing = getenv("RECOMP_KELVIN_TIMING") != NULL;
    }
    t0 = g_kelvin_timing > 0 ? kelvin_ticks() : 0;
    if (kgpu_active()) {
        bool both = kgpu_compare_enabled();
        gpu_flush_draw(pg, !both);
        if (!both) {
            if (g_kelvin_timing > 0) g_kelvin_ticks[KT_DRAW] += kelvin_ticks() - t0;
            return;
        }
    }
    flush_draw(pg);
    if (kgpu_active() && kgpu_compare_enabled()) kgpu_pass_check();
    if (g_kelvin_timing > 0) {
        g_kelvin_ticks[KT_DRAW] += kelvin_ticks() - t0;
    }
}

static void flush_draw(KelvinState *pg)
{
    GeomJob *job;
    DrawCtx *c;
    DrawSource src;
    VCache vc;
    uint32_t setupraster = pgraph_reg_r(pg, NV_PGRAPH_SETUPRASTER);
    uint32_t *idx = NULL;
    size_t n = 0, i;
    int a;
    bool trace = trace_frame();
    uint64_t pixels_before = pg->stats.pixels;

    /* The draw is built in the next job slot, and captured there if it can
     * be set up later; otherwise it is drawn here and the slot stays free. */
    if (s_jt_on) {
        long long now = jt_ticks();
        kelvin_render_trace_event(1, now, now, pg->primitive_mode);
    }
    geom_retire();
    job = geom_next_slot();
    if (!job) {
        return;
    }
    c = &job->c;
    memset(c, 0, sizeof(*c));
    c->pg = pg;
    /* BLACK may leave zeta pointing at title data during full-screen passes
     * with both tests disabled. Only real depth/stencil consumers may cache
     * it; the default guest renderer retains its existing binding contract. */
    bool need_zeta = kres_draw_needs_zeta(pg);
    if (!kelvin_bind_target(pg, &c->t, true, need_zeta)) {
        return;
    }
    kvsh_begin_draw(pg);
    kpix_setup(pg, &c->ps, &c->t);
    for (i = 0; i < 4; i++) kres_texture(&c->ps.tex[i]);
    for (i = 0; i < 4; i++) {
        const KTexture *tx = &c->ps.tex[i];
        c->tex_needs_lod[i] = tx->valid && !c->ps.lod_free[i] &&
            (tx->levels > 1 || tx->min_filter != tx->mag_filter);
    }
    skip_unread_varyings(c);
    c->cull_enable = setupraster & NV_PGRAPH_SETUPRASTER_CULLENABLE;
    c->cull_ctrl = GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_CULLCTRL);
    c->ccw_front = GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACE);
    c->provoking_first = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CONTROL_3),
                                  NV_PGRAPH_CONTROL_3_PROVOKING_VERTEX)
                         == NV_PGRAPH_CONTROL_3_PROVOKING_VERTEX_FIRST;
    if (GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACEMODE) !=
        NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_FILL &&
        pg->primitive_mode >= PRIM_TYPE_TRIANGLES) {
        kelvin_unsupported(KU_POLY_MODE, "polygon mode",
                           GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACEMODE));
    }
    {
        uint32_t mode = GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_FRONTFACEMODE);
        c->poly_offset = pg->primitive_mode >= PRIM_TYPE_TRIANGLES &&
            ((mode == NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_FILL &&
              (setupraster & NV_PGRAPH_SETUPRASTER_POFFSETFILLENABLE)) ||
             (mode == NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_LINE &&
              (setupraster & NV_PGRAPH_SETUPRASTER_POFFSETLINEENABLE)) ||
             (mode == NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_POINT &&
              (setupraster & NV_PGRAPH_SETUPRASTER_POFFSETPOINTENABLE)));
        /* Lines use the line enable whatever the polygon mode. */
        c->line_offset = setupraster & NV_PGRAPH_SETUPRASTER_POFFSETLINEENABLE;
        if (c->poly_offset || c->line_offset) {
            c->depth_offset = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_ZOFFSETBIAS));
            c->depth_factor = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_ZOFFSETFACTOR));
        }
    }

    memset(&src, 0, sizeof(src));
    if (pg->draw_arrays_length || pg->inline_elements_length) {
        src.kind = SRC_ARRAYS;
        for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
            VertexAttribute *attr = &pg->vertex_attributes[a];
            if (attr->count) {
                src.attr_va[a] = kelvin_dma_va(attr->dma_select ? pg->dma_vertex_b
                                                                : pg->dma_vertex_a,
                                               attr->offset, 0);
            }
        }
    } else if (pg->inline_buffer_length) {
        src.kind = SRC_INLINE_BUFFER;
    } else if (pg->inline_array_length) {
        unsigned int offset = 0;
        src.kind = SRC_INLINE_ARRAY;
        for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
            VertexAttribute *attr = &pg->vertex_attributes[a];
            if (!attr->count) {
                continue;
            }
            offset = ROUND_UP(offset, attr->size);
            attr->inline_array_offset = offset;
            offset += attr->size * attr->count;
            offset = ROUND_UP(offset, attr->size);
        }
        src.vertex_size = offset;
    } else {
        return;
    }
    plan_fetch(pg, &src);

    if (!trace && geom_deferrable(pg, c) && geom_capture(pg, job, &src)) {
        update_inline_values(pg, &src);
        return;
    }
    geom_flush();

    if (pg->draw_arrays_length) {
        size_t total = 0;
        for (i = 0; i < pg->draw_arrays_length; i++) {
            total += (size_t)pg->draw_arrays_count[i];
        }
        vcache_init(&vc, total);
        for (i = 0; i < pg->draw_arrays_length; i++) {
            size_t cnt = (size_t)pg->draw_arrays_count[i], j;
            uint32_t *run = (uint32_t *)malloc(cnt * sizeof(uint32_t));
            for (j = 0; j < cnt; j++) {
                run[j] = (uint32_t)(pg->draw_arrays_start[i] + (int32_t)j);
            }
            if (!preshade(pg, &src, &vc, run, cnt) ||
                !assemble_parallel(c, &src, &vc, run, cnt)) {
                assemble(c, &src, &vc, run, cnt);
            }
            if (trace && i + 1 == pg->draw_arrays_length) {
                flush_tris(c);
                kelvin_render_sync();
                trace_draw(c, &src, &vc, run, cnt, pixels_before);
            }
            free(run);
        }
        flush_tris(c);
        vcache_free(&vc);
    } else {
        if (pg->inline_elements_length) {
            n = pg->inline_elements_length;
            idx = (uint32_t *)malloc(n * sizeof(uint32_t));
            memcpy(idx, pg->inline_elements, n * sizeof(uint32_t));
        } else {
            n = src.kind == SRC_INLINE_BUFFER
                    ? pg->inline_buffer_length
                    : (src.vertex_size ? pg->inline_array_length * 4 / src.vertex_size : 0);
            idx = (uint32_t *)malloc((n ? n : 1) * sizeof(uint32_t));
            for (i = 0; i < n; i++) {
                idx[i] = (uint32_t)i;
            }
        }
        vcache_init(&vc, n);
        if (!preshade(pg, &src, &vc, idx, n) ||
            !assemble_parallel(c, &src, &vc, idx, n)) {
            assemble(c, &src, &vc, idx, n);
        }
        flush_tris(c);
        if (trace) {
            kelvin_render_sync();
            trace_draw(c, &src, &vc, idx, n, pixels_before);
        }
        vcache_free(&vc);
        free(idx);
    }

    update_inline_values(pg, &src);
}
