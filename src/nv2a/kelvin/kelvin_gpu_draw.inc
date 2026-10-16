/* The draw, included by kelvin_gpu.c after feedback_copy.
 *
 * kgpu_draw_now draws one batch of triangles. Most of what a draw needs is the same as the draw before it: the
 * surfaces, the combiner constants and the shader they select, the blend, depth and raster states, the textures. The
 * caller says what it knows has not changed (KGpuDraw.clean), and what is derived from it is kept in s_dc; and the
 * Direct3D state last set on the context is kept in s_bd, so a call that would set what is already set is not made.
 * Everything else that touches the context's bindings (clears, the conversion passes) calls kg_bound_reset first, and
 * so does anything that releases an object the cache may name (a texture, a surface, a state object). */

typedef struct KgBound {
    bool valid;
    bool mrt;                    /* the screen-space reflection targets are bound beside the colour */
    ID3D11RenderTargetView *rtv;
    ID3D11DepthStencilView *dsv;
    ID3D11BlendState *bs;
    FLOAT bf[4];
    ID3D11DepthStencilState *dss;
    UINT sref;
    ID3D11RasterizerState *rs;
    D3D11_VIEWPORT vp;
    D3D11_RECT sc;
    ID3D11PixelShader *ps;
    ID3D11VertexShader *vs;
    ID3D11ShaderResourceView *srv[4];
    ID3D11SamplerState *smp[4];
    ID3D11ShaderResourceView *vsrv;
    int mode;                    /* 1: GPU vertex shading (constant buffers b1..b3, arena at t9), 2: CPU vertices */
    D3D11_PRIMITIVE_TOPOLOGY topo;
    DXGI_FORMAT ibfmt;
    bool fixed;                  /* the constant buffer, input layout and vertex-stage buffers that never change are bound */
} KgBound;

typedef struct KgDc {
    bool valid;
    int reject;                  /* 0 draw; 1 skip counted; 2 skip silently */
    KGpuSurf *cs, *zs;
    double sx, sy;
    bool has_alpha;
    unsigned int alpha_mode;
    PSConst pc;
    ID3D11PixelShader *psx;
    bool psx_pending;
    bool pc_changed;
    ID3D11BlendState *bs;
    ID3D11DepthStencilState *dss;
    ID3D11RasterizerState *rs;
    D3D11_VIEWPORT vp;
    D3D11_RECT sc;
    FLOAT bf[4];
    UINT sref;
    float zmax;
    int cull_enable, cull_ctrl, ccw_front;
    float zbias, zslope;
    ID3D11ShaderResourceView *srv[4];
    ID3D11SamplerState *smp[4];
    KGpuTex *te[4];              /* the cache entries behind srv (ray tracing shades with them) */
    bool tex_ok;                 /* the slots resolved to plain textures: nothing per draw */
    uint64_t tex_epoch, tex_surfgen;
} KgDc;

/* Ray tracing capture (kelvin_gpu_rt.inc, included after this file). */
static bool kg_rtc_begin(const KGpuDraw *d, const KgDc *dc, KGpuVs *ve, float zmax);
static void kg_rtc_end(void);
static void kg_rtc_note_draw(const KPixelState *ps, const KGpuSurf *cs, KGpuSurf *zs);

static KgBound s_bd;
static KgDc s_dc;
static void kg_bound_reset(void)
{
    s_bd.valid = false;
}

static void kg_cache_invalidate(void)
{
    s_bd.valid = false;
    s_dc.valid = false;
}

/* RECOMP_KGPU_PROBE=frame,x,y[,frames] (lab diagnostic, synchronous submission only): after every draw and clear of
 * the flips in [frame, frame+frames) print the colour the target holds at (x, y) when it changed, with the draw that
 * changed it. Finds which draw puts a pixel there. */
static struct { int mode, f0, n, x, y; unsigned int ord; uint64_t frame; uint32_t last; bool have; ID3D11Texture2D *stage; } s_probe = { -1 };

static void kg_probe(const KGpuSurf *cs, const KGpuDraw *d, const char *what)
{
    D3D11_BOX box;
    D3D11_MAPPED_SUBRESOURCE ms;
    uint32_t px;
    uint64_t fr;
    if (s_probe.mode < 0) {
        const char *e = getenv("RECOMP_KGPU_PROBE");
        int f = 0, x = 0, y = 0, n = 1;
        s_probe.mode = e && sscanf(e, "%d,%d,%d,%d", &f, &x, &y, &n) >= 3 ? 1 : 0;
        s_probe.f0 = f; s_probe.x = x; s_probe.y = y; s_probe.n = n > 0 ? n : 1;
    }
    if (s_probe.mode != 1 || !cs || !cs->tex || cs->bpp != 4) return;
    fr = kgpu_compare_enabled() ? s_cmp_frames + 1 : G.frames;
    if (fr < (uint64_t)s_probe.f0 || fr >= (uint64_t)s_probe.f0 + (uint64_t)s_probe.n) return;
    if (s_probe.frame != fr) { s_probe.frame = fr; s_probe.ord = 0; s_probe.have = false; }
    s_probe.ord++;
    if ((unsigned int)s_probe.x >= cs->hw || (unsigned int)s_probe.y >= cs->hh) return;
    if (!s_probe.stage) {
        D3D11_TEXTURE2D_DESC td;
        memset(&td, 0, sizeof(td));
        td.Width = 1; td.Height = 1; td.MipLevels = td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_probe.stage))) return;
    }
    box.left = (UINT)s_probe.x; box.right = box.left + 1; box.top = (UINT)s_probe.y; box.bottom = box.top + 1;
    box.front = 0; box.back = 1;
    ID3D11DeviceContext_CopySubresourceRegion(G.ctx, (ID3D11Resource *)s_probe.stage, 0, 0, 0, 0, (ID3D11Resource *)cs->tex, 0, &box);
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)s_probe.stage, 0, D3D11_MAP_READ, 0, &ms))) return;
    px = *(const uint32_t *)ms.pData;
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)s_probe.stage, 0);
    if (!s_probe.have || px != s_probe.last) {
        if (d) {
            const KPixelState *ps = d->ps;
            fprintf(stderr, "[KPROBE] f%llu #%u %s: %08X -> %08X  icount %u vcount %u topo %u %s vs %08X blend %d(%d,%d) z %d/%d/%d tex0 %ux%u fmt %u\n",
                    (unsigned long long)fr, s_probe.ord, what, s_probe.have ? s_probe.last : 0, px, d->icount, d->vcount, d->topology,
                    d->gvs ? "gvs" : "cpu", d->gvs ? d->gvs->hash : 0, ps->blend, ps->blend_sfactor, ps->blend_dfactor,
                    ps->depth_test, ps->depth_func, ps->depth_write, ps->tex[0].width, ps->tex[0].height, ps->tex[0].color_format);
        } else {
            fprintf(stderr, "[KPROBE] f%llu #%u %s: %08X -> %08X\n", (unsigned long long)fr, s_probe.ord, what,
                    s_probe.have ? s_probe.last : 0, px);
        }
        s_probe.last = px; s_probe.have = true;
    }
}

/* RECOMP_KGPU_PASSCMP=frame[,frames] (lab diagnostic, compare mode only): after each draw of those flips, finish the
 * CPU renderer's work and compare the GPU's copy of the draw's colour target with the CPU's. Prints a line whenever a
 * draw makes the two differ more than before, with the draw's state, and dumps the first few such targets. */
static struct {
    int mode, f0, n;
    bool have;
    KelvinTarget t;
    KGpuSurf *cs, *zs;
    unsigned int ord;
    uint64_t frame;
    char what[200];
    uint64_t zbad_prev;
    int dumped;
    struct { uint32_t va; double mean; uint64_t gt8; } seen[16];
} s_pc = { -1 };

static void kg_pass_note(const KGpuDraw *d, KGpuSurf *cs, KGpuSurf *zs)
{
    const KPixelState *ps = d->ps;
    uint64_t fr;
    if (s_pc.mode < 0) {
        const char *e = getenv("RECOMP_KGPU_PASSCMP");
        int f = 0, n = 1;
        s_pc.mode = e && sscanf(e, "%d,%d", &f, &n) >= 1 ? 1 : 0;
        s_pc.f0 = f; s_pc.n = n > 0 ? n : 1;
    }
    s_pc.have = false;
    if (s_pc.mode != 1 || !kgpu_compare_enabled() || !cs || cs->bpp != 4 || !d->t->color) return;
    fr = s_cmp_frames + 1;
    if (fr < (uint64_t)s_pc.f0 || fr >= (uint64_t)s_pc.f0 + (uint64_t)s_pc.n) return;
    if (s_pc.frame != fr) { s_pc.frame = fr; s_pc.ord = 0; memset(s_pc.seen, 0, sizeof(s_pc.seen)); }
    s_pc.ord++;
    s_pc.t = *d->t;
    s_pc.cs = cs;
    s_pc.zs = zs;
    s_pc.have = true;
    snprintf(s_pc.what, sizeof(s_pc.what), "icount %u topo %u %s vs %08X blend %d(%d,%d) z %d/%d/%d tex0 %ux%u fmt %u tex1 %ux%u target %ux%u va %08X zva %08X real %d fromc %d",
             d->icount, d->topology, d->gvs ? "gvs" : "cpu", d->gvs ? d->gvs->hash : 0, ps->blend, ps->blend_sfactor, ps->blend_dfactor,
             ps->depth_test, ps->depth_func, ps->depth_write, ps->tex[0].width, ps->tex[0].height, ps->tex[0].color_format,
             ps->tex[1].width, ps->tex[1].height, d->t->width, d->t->height, d->t->color_va,
             zs ? zs->va : 0, zs ? (int)zs->real : -1, zs ? (int)zs->from_color : -1);
}

static uint32_t kg_pdep(uint32_t v, uint32_t mask)
{
    uint32_t r = 0, bit;
    for (bit = 1; mask; bit <<= 1) {
        uint32_t low = mask & (0u - mask);
        if (v & bit) r |= low;
        mask &= mask - 1;
    }
    return r;
}

/* Called by the CPU path once it has finished its own copy of the draw that kg_pass_note saw. */
void kgpu_pass_check(void)
{
    ID3D11Texture2D *st;
    D3D11_MAPPED_SUBRESOURCE ms;
    unsigned int x, y, slot;
    uint64_t sum = 0, gt8 = 0, gt32 = 0, pixels;
    unsigned int maxd = 0;
    double mean;
    const KelvinTarget *t = &s_pc.t;
    if (!s_pc.have) return;
    s_pc.have = false;
    kelvin_render_sync();
    if (s_pc.cs->hw != t->width || s_pc.cs->hh != t->height) return;
    if (!readback(s_pc.cs, &st, &ms)) return;
    pixels = (uint64_t)t->width * t->height;
    for (y = 0; y < t->height; y++) {
        const uint8_t *g = (const uint8_t *)ms.pData + (size_t)y * ms.RowPitch;
        for (x = 0; x < t->width; x++) {
            const uint8_t *c;
            unsigned int d0, d1, d2, d;
            if (t->swizzled) c = t->color + (size_t)(kg_pdep(x, t->swz_mask_x) | kg_pdep(y, t->swz_mask_y)) * 4;
            else c = t->color + (size_t)y * t->color_pitch + (size_t)x * 4;
            d0 = (unsigned int)abs((int)g[x * 4] - (int)c[0]);
            d1 = (unsigned int)abs((int)g[x * 4 + 1] - (int)c[1]);
            d2 = (unsigned int)abs((int)g[x * 4 + 2] - (int)c[2]);
            d = d0 > d1 ? d0 : d1;
            if (d2 > d) d = d2;
            sum += d0 + d1 + d2;
            if (d > maxd) maxd = d;
            if (d > 8) gt8++;
            if (d > 32) gt32++;
        }
    }
    mean = (double)sum / (3.0 * (double)pixels);
    for (slot = 0; slot < 16 && s_pc.seen[slot].va && s_pc.seen[slot].va != t->color_va; slot++) {}
    if (slot < 16) {
        if (!s_pc.seen[slot].va) { s_pc.seen[slot].va = t->color_va; }
        if (mean > s_pc.seen[slot].mean + 0.01 || gt8 > s_pc.seen[slot].gt8 + 40) {
            fprintf(stderr, "[KPASS] f%llu #%u mean %.3f (was %.3f) max %u >8: %llu (was %llu) >32: %llu  %s\n",
                    (unsigned long long)s_pc.frame, s_pc.ord, mean, s_pc.seen[slot].mean, maxd, (unsigned long long)gt8,
                    (unsigned long long)s_pc.seen[slot].gt8, (unsigned long long)gt32, s_pc.what);
            if (G.dump && *G.dump && s_pc.dumped < 24 && t->width * t->height >= 160 * 120) {
                char path[512];
                uint8_t *buf = (uint8_t *)malloc((size_t)t->width * t->height * 4);
                if (buf) {
                    for (y = 0; y < t->height; y++)
                        for (x = 0; x < t->width; x++) {
                            const uint8_t *c;
                            if (t->swizzled) c = t->color + (size_t)(kg_pdep(x, t->swz_mask_x) | kg_pdep(y, t->swz_mask_y)) * 4;
                            else c = t->color + (size_t)y * t->color_pitch + (size_t)x * 4;
                            memcpy(buf + ((size_t)y * t->width + x) * 4, c, 4);
                        }
                    snprintf(path, sizeof(path), "%spass%03d_%u_cpu.bmp", G.dump, s_pc.dumped, s_pc.ord);
                    write_bmp(path, buf, t->width, t->height, t->width * 4);
                    snprintf(path, sizeof(path), "%spass%03d_%u_gpu.bmp", G.dump, s_pc.dumped, s_pc.ord);
                    write_bmp(path, (const uint8_t *)ms.pData, t->width, t->height, ms.RowPitch);
                    free(buf);
                    s_pc.dumped++;
                }
            }
        }
        s_pc.seen[slot].mean = mean; s_pc.seen[slot].gt8 = gt8;
    }
    if (s_pc.zs && s_pc.zs->tex && t->zeta && t->zeta_bpp == 4 && !t->swizzled && s_pc.zs->hw == t->width && s_pc.zs->hh == t->height) {
        static ID3D11Texture2D *zst;
        static unsigned int zw, zh;
        D3D11_TEXTURE2D_DESC td;
        D3D11_MAPPED_SUBRESOURCE zm;
        ID3D11Texture2D_GetDesc(s_pc.zs->tex, &td);
        if (zst && (zw != td.Width || zh != td.Height)) { IUnknown_Release((IUnknown *)zst); zst = NULL; }
        if (!zst) {
            td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
            if (SUCCEEDED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &zst))) { zw = td.Width; zh = td.Height; }
        }
        if (zst) {
            ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)zst, (ID3D11Resource *)s_pc.zs->tex);
            if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)zst, 0, D3D11_MAP_READ, 0, &zm))) {
                uint64_t bad = 0, far_cpu_near_gpu = 0;
                int fx = -1, fy = -1;
                uint32_t fc = 0, fg = 0;
                for (y = 0; y < t->height; y++) {
                    const float *gz = (const float *)((const uint8_t *)zm.pData + (size_t)y * zm.RowPitch);
                    const uint32_t *cz = (const uint32_t *)(t->zeta + (size_t)y * t->zeta_pitch);
                    for (x = 0; x < t->width; x++) {
                        double gf = gz[x * 2];
                        long gi = (long)floor((gf < 0 ? 0 : gf > 1 ? 1 : gf) * 16777215.0 + 0.5);
                        long ci = (long)(cz[x] >> 8);
                        if (labs(gi - ci) > 4096) {
                            if (!bad) { fx = (int)x; fy = (int)y; fc = (uint32_t)ci; fg = (uint32_t)gi; }
                            bad++;
                            if (ci > 0xF00000 && gi < 0xE00000) far_cpu_near_gpu++;
                        }
                    }
                }
                if (bad > s_pc.zbad_prev + 20) {
                    fprintf(stderr, "[KZ] f%llu #%u depth differs (>4096 of 24 bits): %llu pixels (was %llu), cpu far/gpu not: %llu, first at (%d,%d) cpu %06X gpu %06X  %s\n",
                            (unsigned long long)s_pc.frame, s_pc.ord, (unsigned long long)bad, (unsigned long long)s_pc.zbad_prev,
                            (unsigned long long)far_cpu_near_gpu, fx, fy, fc, fg, s_pc.what);
                }
                s_pc.zbad_prev = bad;
                ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)zst, 0);
            }
        }
    }
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)st, 0);
    IUnknown_Release((IUnknown *)st);
}

/* RECOMP_KGPU_DRAWLOG=frame[,frames] (lab diagnostic): a line for every draw and clear of those presented frames. */
static struct { int mode, f0, n; unsigned int ord; uint64_t frame; float last[16]; bool have; } s_dl = { -1 };

static bool kg_dl_on(void)
{
    uint64_t fr;
    if (s_dl.mode < 0) {
        const char *e = getenv("RECOMP_KGPU_DRAWLOG");
        int f = 0, n = 1;
        s_dl.mode = e && sscanf(e, "%d,%d", &f, &n) >= 1 ? 1 : 0;
        s_dl.f0 = f; s_dl.n = n > 0 ? n : 1;
    }
    if (s_dl.mode != 1) return false;
    fr = kgpu_compare_enabled() ? s_cmp_frames + 1 : G.frames;
    if (fr < (uint64_t)s_dl.f0 || fr >= (uint64_t)s_dl.f0 + (uint64_t)s_dl.n) return false;
    if (s_dl.frame != fr) { s_dl.frame = fr; s_dl.ord = 0; s_dl.have = false; }
    s_dl.ord++;
    return true;
}

static void kg_dl_draw(const KGpuDraw *d)
{
    const KPixelState *ps = d->ps;
    if (!kg_dl_on()) return;
    fprintf(stderr, "[KDL] f%llu #%u draw vs %08X icount %u topo %u target %ux%u z %d/%d/%d blend %d(%d,%d) at %d cull %d tex0 %ux%u zofs %g zmax %g\n",
            (unsigned long long)s_dl.frame, s_dl.ord, d->gvs ? d->gvs->hash : 0, d->icount, d->topology, d->t->width, d->t->height,
            ps->depth_test, ps->depth_func, ps->depth_write, ps->blend, ps->blend_sfactor, ps->blend_dfactor, ps->alpha_test,
            d->cull_enable, ps->tex[0].width, ps->tex[0].height, (double)d->zbias, (double)ps->zmax);
    fprintf(stderr, "[KDL]   targetva %08X depthva %08X depthclip %d zclip %g,%g cullctrl %d frontccw %d texturebase %08X,%08X,%08X,%08X texturevalid %d,%d,%d,%d\n",
            d->t->color_va, d->t->zeta_va, (int)ps->depth_clip_cull, (double)ps->zclip_min, (double)ps->zclip_max,
            d->cull_ctrl, (int)d->ccw_front,
            ps->tex[0].base, ps->tex[1].base, ps->tex[2].base, ps->tex[3].base,
            (int)ps->tex[0].valid, (int)ps->tex[1].valid, (int)ps->tex[2].valid, (int)ps->tex[3].valid);
    if (d->gvs && d->gvs->constants) {
        static float vp_last[8];
        const float *v = d->gvs->constants + 4 * 58;       /* c[-38] viewport scale, c[-37] offset */
        if (memcmp(vp_last, v, sizeof(vp_last)) != 0) {
            memcpy(vp_last, v, sizeof(vp_last));
            fprintf(stderr, "[KDL]   viewport scale %g %g %g %g offset %g %g %g %g\n", (double)v[0], (double)v[1], (double)v[2], (double)v[3],
                    (double)v[4], (double)v[5], (double)v[6], (double)v[7]);
        }
    }
    if (d->gvs && d->gvs->constants && (!s_dl.have || memcmp(s_dl.last, d->gvs->constants, sizeof(s_dl.last)) != 0)) {
        const float *c = d->gvs->constants;
        memcpy(s_dl.last, c, sizeof(s_dl.last)); s_dl.have = true;
        fprintf(stderr, "[KDL]   c0 %g %g %g %g | c1 %g %g %g %g | c2 %g %g %g %g | c3 %g %g %g %g\n",
                (double)c[0], (double)c[1], (double)c[2], (double)c[3], (double)c[4], (double)c[5], (double)c[6], (double)c[7],
                (double)c[8], (double)c[9], (double)c[10], (double)c[11], (double)c[12], (double)c[13], (double)c[14], (double)c[15]);
    }
}

static void kgpu_draw_now(const KGpuDraw *d)
{
    const KelvinTarget *t = d->t;
    const KPixelState *ps = d->ps;
    KgDc *dc = &s_dc;
    VSConst vc;
    ID3D11ShaderResourceView *vbsrv;
    D3D11_MAPPED_SUBRESOURCE ms;
    size_t at = 0, ib_at;
    unsigned int i, k;
    void *p;
    long long prof_t;
    bool new_t, new_ps, cull_changed;
    KGpuVs *rt_ve = NULL;
    bool rt_cap;
    bool short_idx = d->index_limit && d->index_limit <= 0xFFFFu;
    D3D11_PRIMITIVE_TOPOLOGY topo = d->topology == KG_TOPO_STRIP ? D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP
                                                                   : D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    DXGI_FORMAT ibfmt = short_idx ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
    size_t ib_bytes;

    if (!G.ok || !d->icount || (!d->gvs && !d->vcount)) return;
    {
        /* RECOMP_KGPU_SKIPVS=hash,hash (lab diagnostic): do not draw with those vertex programs. */
        static const char *skip = (const char *)-1;
        if (skip == (const char *)-1) skip = getenv("RECOMP_KGPU_SKIPVS");
        if (skip && d->gvs) {
            char key[16];
            snprintf(key, sizeof(key), "%08X", d->gvs->hash);
            if (strstr(skip, key)) return;
        }
    }
    kt_begin();
    if (!s_bd.valid) {
        memset(&s_bd, 0, sizeof(s_bd));
        s_bd.valid = true;
        s_bd.vp.Width = -1.0f;                 /* not a real viewport or scissor: the first draw sets them */
        s_bd.sc.left = -1;
    }

    new_t = !dc->valid || !(d->clean & KG_CLEAN_T);
    new_ps = !dc->valid || !(d->clean & KG_CLEAN_PS);
    cull_changed = dc->cull_enable != (int)d->cull_enable || dc->cull_ctrl != d->cull_ctrl || dc->ccw_front != (int)d->ccw_front ||
                   dc->zbias != d->zbias || dc->zslope != d->zslope;
    if (new_t || new_ps || cull_changed) {
        KGpuSurf *cs = NULL, *zs = NULL;
        const KGpuSurf *ref;
        D3D11_VIEWPORT *vp = &dc->vp;
        D3D11_RECT *sc = &dc->sc;
        dc->valid = true;
        dc->reject = 0;
        dc->cull_enable = d->cull_enable ? 1 : 0; dc->cull_ctrl = d->cull_ctrl; dc->ccw_front = d->ccw_front ? 1 : 0;
        dc->zbias = d->zbias; dc->zslope = d->zslope;
        if (t->color) cs = surf_get(t, false, ps->color_format);
        if (t->zeta && (ps->depth_test || ps->stencil_test)) zs = surf_get(t, true, ps->zeta_format);
        dc->cs = cs; dc->zs = zs;
        if (!cs && !zs) dc->reject = 1;
        else if (d->cull_enable && d->cull_ctrl == NV_PGRAPH_SETUPRASTER_CULLCTRL_FRONT_AND_BACK) dc->reject = 2;
        else if (cs && zs && (cs->hw != zs->hw || cs->hh != zs->hh)) dc->reject = 1;
        if (!dc->reject) {
            dc->sx = cs ? (double)cs->hw / (double)cs->gw : (double)zs->hw / (double)zs->gw;
            dc->sy = cs ? (double)cs->hh / (double)cs->gh : (double)zs->hh / (double)zs->gh;
            dc->has_alpha = ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8 ||
                            ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X1A7R8G8B8_Z1A7R8G8B8 ||
                            ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X1A7R8G8B8_O1A7R8G8B8;
            dc->alpha_mode = 0;
            if (ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8) dc->alpha_mode = 1;
            if (ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8) dc->alpha_mode = 2;
            fill_ps_const(&dc->pc, ps, dc->alpha_mode);
            dc->pc_changed = true;
            dc->psx = ps_lookup(&dc->pc);
            dc->psx_pending = dc->psx == NULL;
            dc->zmax = ps->zmax > 0.0f ? ps->zmax : 16777215.0f;
            dc->bs = get_blend(ps, dc->alpha_mode, dc->has_alpha, false);
            dc->dss = get_depth(ps, zs != NULL);
            dc->rs = get_raster(d);
            dc->bf[0] = ps->blend_color[0]; dc->bf[1] = ps->blend_color[1];
            dc->bf[2] = ps->blend_color[2]; dc->bf[3] = ps->blend_color[3];
            dc->sref = (UINT)ps->stencil_ref;
            ref = cs ? cs : zs;
            vp->TopLeftX = vp->TopLeftY = 0.0f; vp->Width = (FLOAT)ref->hw; vp->Height = (FLOAT)ref->hh;
            vp->MinDepth = 0.0f; vp->MaxDepth = 1.0f;
            sc->left = (LONG)floor(t->clip_x0 * dc->sx); sc->top = (LONG)floor(t->clip_y0 * dc->sy);
            sc->right = (LONG)ceil(t->clip_x1 * dc->sx); sc->bottom = (LONG)ceil(t->clip_y1 * dc->sy);
            if (sc->right > (LONG)ref->hw) sc->right = (LONG)ref->hw;
            if (sc->bottom > (LONG)ref->hh) sc->bottom = (LONG)ref->hh;
            if (sc->left < 0) sc->left = 0;
            if (sc->top < 0) sc->top = 0;
            dc->tex_ok = false;
        }
    }
    if (dc->reject) {
        if (dc->reject == 1) G.skipped++;
        return;
    }
    kg_rtc_note_draw(ps, dc->cs, dc->zs);
    kg_post_note_draw(ps, dc->cs, dc->zs);       /* the picture effects run when the HUD begins (kelvin_gpu_post.inc) */
    if (dc->zs) {
        if (ps->depth_write || ps->stencil_write) dc->zs->real = true;
        depth_surface_refresh(dc->zs);
    }

    /* textures */
    prof_t = kg_tick();
    if (!dc->tex_ok || dc->tex_epoch != G.epoch || dc->tex_surfgen != s_surf_gen) {
        KGpuSurf *cs = dc->cs;
        bool plain = true;
        for (i = 0; i < 4; i++) {
            const KTexture *tx = &ps->tex[i];
            KGpuSurf *fs;
            dc->srv[i] = G.dummy_srv;
            dc->smp[i] = NULL;
            dc->te[i] = NULL;
            if (ps->tex_mode[i] != PS_TEXTUREMODES_PROJECT2D || !tx->valid) continue;
            fs = surf_for_texture(tx);
            if (fs) {
                if (cs && fs == cs) {
                    /* sampling the surface being drawn: read a copy of its pre-draw state, made for every draw */
                    ID3D11Texture2D *cp = feedback_copy(fs);
                    plain = false;
                    if (cp) {
                        static ID3D11ShaderResourceView *cp_srv;
                        static ID3D11Texture2D *cp_tex;
                        ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)cp, (ID3D11Resource *)fs->tex);
                        if (cp_tex != cp) {
                            SAFE_RELEASE(cp_srv);
                            ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)cp, NULL, &cp_srv);
                            cp_tex = cp;
                        }
                        dc->srv[i] = cp_srv;
                    }
                } else {
                    dc->srv[i] = fs->srv;
                }
            } else {
                KGpuSurf *ds = surf_depth_for_texture(tx);
                ID3D11ShaderResourceView *v;
                if (ds) plain = false;
                v = ds ? depth_as_color(ds) : tex_get(tx);
                if (v) dc->srv[i] = v;
                if (!ds && v && G.last_tex && G.last_tex->srv == v) dc->te[i] = G.last_tex;
            }
            dc->smp[i] = get_sampler(tx);
        }
        dc->tex_ok = plain;
        dc->tex_epoch = G.epoch;
        dc->tex_surfgen = s_surf_gen;
    }
    g_kgpu_ticks[KGP_TEX] += kg_tick() - prof_t;
    prof_t = kg_tick();

    if (!s_bd.valid) {
        memset(&s_bd, 0, sizeof(s_bd));
        s_bd.valid = true;
        s_bd.vp.Width = -1.0f;
        s_bd.sc.left = -1;
    }

    /* the combiner constants and the pixel shader they select */
    if (dc->psx_pending) {
        dc->psx = ps_lookup(&dc->pc);
        if (dc->psx) dc->psx_pending = false;
    }
    if (dc->pc_changed) {
        static PSConst last;
        static bool have;
        dc->pc_changed = false;
        if (!have || memcmp(&last, &dc->pc, sizeof(last)) != 0) {
            if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.cb_ps, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
                memcpy(ms.pData, &dc->pc, sizeof(dc->pc));
                ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.cb_ps, 0);
                memcpy(&last, &dc->pc, sizeof(last));
                have = true;
            }
        }
    }

    /* geometry */
    if (d->gvs) {
        /* raw guest vertex data: the ranges the attributes read, merged, in the arena */
        typedef struct Rng { const uint8_t *lo, *hi; size_t off; } Rng;
        const KGpuVsDraw *gv = d->gvs;
        Rng rg[16], mg[16];
        int nr = 0, nm = 0, a, j;
        size_t total = 0;
        uint32_t adesc[16][4];
        float inl[16][4];
        VSG gc;
        const KGpuSurf *ref = dc->cs ? dc->cs : dc->zs;
        for (a = 0; a < 16; a++) {
            const KGpuAttrSrc *as = &gv->attr[a];
            memset(adesc[a], 0, sizeof(adesc[a]));
            memcpy(inl[a], as->inline_value, 16);
            if (!as->fmt) continue;
            rg[nr].lo = as->data + as->lo;
            rg[nr].hi = as->data + as->hi;
            rg[nr].off = (size_t)a;
            for (j = nr; j > 0 && rg[j - 1].lo > rg[j].lo; j--) { Rng t2 = rg[j]; rg[j] = rg[j - 1]; rg[j - 1] = t2; }
            nr++;
        }
        for (a = 0; a < nr; a++) {
            if (nm && rg[a].lo <= mg[nm - 1].hi + 64) {
                if (rg[a].hi > mg[nm - 1].hi) mg[nm - 1].hi = rg[a].hi;
            } else {
                mg[nm] = rg[a];
                nm++;
            }
        }
        for (a = 0; a < nm; a++) {
            mg[a].off = total;
            total += ((size_t)(mg[a].hi - mg[a].lo) + 15) & ~(size_t)15;
        }
        if (total == 0) total = 16;
        p = ring_map(G.arena, &G.arena_pos, G.arena_cap, total / 16, 16, &at, &ms);
        if (!p) return;
        for (a = 0; a < nm; a++) memcpy((uint8_t *)p + mg[a].off, mg[a].lo, (size_t)(mg[a].hi - mg[a].lo));
        ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.arena, 0);
        G.vb_bytes += total;
        for (a = 0; a < 16; a++) {
            const KGpuAttrSrc *as = &gv->attr[a];
            const uint8_t *lo;
            int64_t base;
            if (!as->fmt) continue;
            lo = as->data + as->lo;
            for (j = 0; j < nm; j++) if (lo >= mg[j].lo && lo < mg[j].hi + 1) break;
            if (j == nm) continue;
            base = (int64_t)(at * 16 + mg[j].off) + (int64_t)(as->data - mg[j].lo);
            adesc[a][0] = (uint32_t)base;
            adesc[a][1] = as->stride;
            adesc[a][2] = as->fmt;
            adesc[a][3] = as->count;
        }
        memset(&gc, 0, sizeof(gc));
        gc.scale[0] = 2.0f / (float)ref->gw;
        gc.scale[1] = 2.0f / (float)ref->gh;
        gc.scale[2] = 1.0f / dc->zmax;
        gc.scale[3] = 18446744073709551616.0f;
        gc.fog[0] = gv->fog_param[0]; gc.fog[1] = gv->fog_param[1];
        gc.flags[0] = gv->fog_enable ? 1u : 0u;
        gc.flags[1] = (uint32_t)gv->fog_mode;
        gc.flags[2] = gv->specular_enable ? 1u : 0u;
        gc.flags[3] = gv->ignore_specular_alpha ? 1u : 0u;
        gc.vm[0] = kg_viewmodel_scale(gv->hash);
        {
            static VSG last_gc;
            static bool have_gc;
            if (!have_gc || memcmp(&last_gc, &gc, sizeof(gc)) != 0) {
                if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.cb_gvs, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
                    memcpy(ms.pData, &gc, sizeof(gc));
                    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.cb_gvs, 0);
                    memcpy(&last_gc, &gc, sizeof(gc));
                    have_gc = true;
                }
            }
        }
        if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.cb_a, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
            memcpy(ms.pData, adesc, sizeof(adesc));
            memcpy((uint8_t *)ms.pData + sizeof(adesc), inl, sizeof(inl));
            ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.cb_a, 0);
        }
        {
            static float last_c[192 * 4];
            static bool have_last;
            if (!have_last || memcmp(last_c, gv->constants, sizeof(last_c)) != 0) {
                if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.cb_c, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
                    memcpy(ms.pData, gv->constants, sizeof(last_c));
                    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.cb_c, 0);
                    memcpy(last_c, gv->constants, sizeof(last_c));
                    have_last = true;
                }
            }
        }
    } else {
        const KGpuSurf *ref = dc->cs ? dc->cs : dc->zs;
        p = ring_map(G.vb, &G.vb_pos, G.vb_cap, (size_t)d->vcount * d->stride, 16, &at, &ms);
        if (!p) return;
        memcpy(p, d->recs, (size_t)d->vcount * d->stride * 16);
        ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.vb, 0);
        G.vb_bytes += (size_t)d->vcount * d->stride * 16;
        memset(&vc, 0, sizeof(vc));
        vc.scale[0] = 2.0f / (float)ref->gw;
        vc.scale[1] = 2.0f / (float)ref->gh;
        vc.scale[2] = 1.0f / dc->zmax;
        vc.scale[3] = 18446744073709551616.0f;
        for (i = 0; i < KG_SLOTS; i++) {
            uint32_t s = d->slot[i] < 0 ? 0xFFFFFFFFu : (uint32_t)d->slot[i];
            switch (i) {
            case KG_POS: vc.rec0[2] = s; break;
            case KG_D0: vc.rec0[3] = s; break;
            case KG_D1: vc.rec1[0] = s; break;
            case KG_FOG: vc.rec1[1] = s; break;
            case KG_T0: vc.rec1[2] = s; break;
            case KG_T1: vc.rec1[3] = s; break;
            case KG_T2: vc.rec2[0] = s; break;
            case KG_T3: vc.rec2[1] = s; break;
            }
        }
        vc.rec0[1] = d->stride;
        vc.rec0[0] = (uint32_t)at;
        if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.cb_vs, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
            memcpy(ms.pData, &vc, sizeof(vc));
            ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.cb_vs, 0);
        }
    }
    ib_bytes = (size_t)d->icount * (short_idx ? 2 : 4);
    p = ring_map(G.ib, &G.ib_pos, G.ib_cap, (ib_bytes + 3) / 4, 4, &ib_at, &ms);
    if (!p) return;
    if (short_idx) {
        uint16_t *w = (uint16_t *)p;
        const uint32_t *src = d->indices;
        for (i = 0; i < d->icount; i++) w[i] = (uint16_t)src[i];
    } else {
        memcpy(p, d->indices, ib_bytes);
    }
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.ib, 0);

    g_kgpu_ticks[KGP_UPLOAD] += kg_tick() - prof_t;
    prof_t = kg_tick();
    if (!dc->bs || !dc->dss || !dc->rs) return;

    {
        KgBound *b = &s_bd;
        ID3D11RenderTargetView *rtv = dc->cs ? dc->cs->rtv : NULL;
        ID3D11DepthStencilView *dsv = dc->zs ? dc->zs->dsv : NULL;
        ID3D11PixelShader *psx = dc->psx ? dc->psx : (ps->smooth || !G.ps_flat) ? G.ps : G.ps_flat;
        ID3D11BlendState *bs = dc->bs, *ssr_bs = NULL;
        ID3D11PixelShader *ssr_ps = NULL;
        /* an environment-mapped draw also writes the reflection targets (kelvin_gpu_ssr.inc) */
        const bool mrt = dc->psx && d->gvs &&
                         kg_ssr_draw(dc->cs, dc->zs, &dc->pc, dc->alpha_mode, dc->has_alpha, ps,
                                     vs_lookup(d->gvs->program, d->gvs->hash, d->gvs->start, NULL, 0), &ssr_ps, &ssr_bs);
        if (mrt) { psx = ssr_ps; bs = ssr_bs; }
        if (!b->fixed) {
            ID3D11DeviceContext_IASetInputLayout(G.ctx, NULL);
            ID3D11DeviceContext_PSSetConstantBuffers(G.ctx, 0, 1, &G.cb_ps);
            b->fixed = true;
        }
        if (b->rtv != rtv || b->dsv != dsv || b->mrt != mrt) {
            if (mrt) {
                ID3D11RenderTargetView *rtvs[4];
                rtvs[0] = rtv; rtvs[1] = s_ssr.rtv[0]; rtvs[2] = s_ssr.rtv[1]; rtvs[3] = s_ssr.rtv[2];
                ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 4, rtvs, dsv);
            } else {
                ID3D11DeviceContext_OMSetRenderTargets(G.ctx, rtv ? 1 : 0, rtv ? &rtv : NULL, dsv);
            }
            b->rtv = rtv; b->dsv = dsv; b->mrt = mrt;
            /* binding a target takes its resource out of every input slot */
            for (k = 0; k < 4; k++) b->srv[k] = NULL;
        }
        if (b->bs != bs || memcmp(b->bf, dc->bf, sizeof(b->bf)) != 0) {
            ID3D11DeviceContext_OMSetBlendState(G.ctx, bs, dc->bf, 0xFFFFFFFFu);
            b->bs = bs; memcpy(b->bf, dc->bf, sizeof(b->bf));
        }
        if (b->dss != dc->dss || b->sref != dc->sref) {
            ID3D11DeviceContext_OMSetDepthStencilState(G.ctx, dc->dss, dc->sref);
            b->dss = dc->dss; b->sref = dc->sref;
        }
        if (b->rs != dc->rs) {
            ID3D11DeviceContext_RSSetState(G.ctx, dc->rs);
            b->rs = dc->rs;
        }
        if (memcmp(&b->vp, &dc->vp, sizeof(b->vp)) != 0) {
            ID3D11DeviceContext_RSSetViewports(G.ctx, 1, &dc->vp);
            b->vp = dc->vp;
        }
        if (memcmp(&b->sc, &dc->sc, sizeof(b->sc)) != 0) {
            ID3D11DeviceContext_RSSetScissorRects(G.ctx, 1, &dc->sc);
            b->sc = dc->sc;
        }
        if (b->topo != topo) {
            ID3D11DeviceContext_IASetPrimitiveTopology(G.ctx, topo);
            b->topo = topo;
        }
        if (b->ibfmt != ibfmt) {
            ID3D11DeviceContext_IASetIndexBuffer(G.ctx, G.ib, ibfmt, 0);
            b->ibfmt = ibfmt;
        }
        if (b->ps != psx) {
            ID3D11DeviceContext_PSSetShader(G.ctx, psx, NULL, 0);
            b->ps = psx;
        }
        if (d->gvs) {
            KGpuVs *ve = vs_lookup(d->gvs->program, d->gvs->hash, d->gvs->start, NULL, 0);
            if (!ve || !ve->vs) return;
            rt_ve = ve;
            if (b->vs != ve->vs) {
                ID3D11DeviceContext_VSSetShader(G.ctx, ve->vs, NULL, 0);
                b->vs = ve->vs;
            }
            if (b->mode != 1) {
                ID3D11Buffer *cbs[3];
                cbs[0] = G.cb_gvs; cbs[1] = G.cb_c; cbs[2] = G.cb_a;
                ID3D11DeviceContext_VSSetConstantBuffers(G.ctx, 1, 3, cbs);
                vbsrv = G.arena_srv;
                ID3D11DeviceContext_VSSetShaderResources(G.ctx, 9, 1, &vbsrv);
                b->mode = 1;
            }
        } else {
            if (b->vs != G.vs) {
                ID3D11DeviceContext_VSSetShader(G.ctx, G.vs, NULL, 0);
                b->vs = G.vs;
            }
            if (b->mode != 2) {
                ID3D11DeviceContext_VSSetConstantBuffers(G.ctx, 1, 1, &G.cb_vs);
                vbsrv = G.vb_srv;
                ID3D11DeviceContext_VSSetShaderResources(G.ctx, 8, 1, &vbsrv);
                b->mode = 2;
            }
        }
        for (k = 0; k < 4; k++) {
            ID3D11SamplerState *s = dc->smp[k];
            if (!s) s = G.default_smp;
            if (b->srv[k] != dc->srv[k]) {
                ID3D11DeviceContext_PSSetShaderResources(G.ctx, k, 1, &dc->srv[k]);
                b->srv[k] = dc->srv[k];
            }
            if (s && b->smp[k] != s) {
                ID3D11DeviceContext_PSSetSamplers(G.ctx, k, 1, &s);
                b->smp[k] = s;
            }
        }
    }
    g_kgpu_ticks[KGP_STATE] += kg_tick() - prof_t;
    prof_t = kg_tick();
    if (s_kp_mode != 0) {
        KpInfo ki;
        const KGpuSurf *rf = dc->cs ? dc->cs : dc->zs;
        uint64_t h = 1469598103934665603ULL;
        unsigned int q, nt = 0;
        const PSConst *pc = &dc->pc;
        memset(&ki, 0, sizeof(ki));
        for (q = 0; q < 8; q++) {
            unsigned int w;
            for (w = 0; w < 4; w++) h = (h ^ pc->stage[q].rgb_in[w]) * 1099511628211ULL;
            for (w = 0; w < 4; w++) h = (h ^ pc->stage[q].alpha_in[w]) * 1099511628211ULL;
            for (w = 0; w < 2; w++) h = (h ^ pc->stage[q].outs[w]) * 1099511628211ULL;
        }
        for (q = 0; q < 4; q++) { h = (h ^ pc->texmode[q]) * 1099511628211ULL; if (d->slot[q] >= 0) nt++; }
        ki.kind = 0; ki.psh = (uint32_t)(h ^ (h >> 32)); ki.icount = d->icount;
        ki.tw = rf ? rf->hw : 0; ki.th = rf ? rf->hh : 0;
        ki.blend = ps->blend ? 1 : 0; ki.ztest = (dc->zs && ps->depth_test) ? 1 : 0; ki.zwrite = (dc->zs && ps->depth_write) ? 1 : 0;
        ki.ntex = (uint8_t)nt; ki.tex0w = ps->tex[0].width; ki.tex0h = ps->tex[0].height;
        kp_mark(&ki);
    }
    rt_cap = kg_rtc_begin(d, dc, rt_ve, dc->zmax);
    ID3D11DeviceContext_DrawIndexed(G.ctx, d->icount, (UINT)(short_idx ? ib_at * 2 : ib_at), 0);
    if (rt_cap) kg_rtc_end();
    if (s_kp_mode != 0) kp_done();
    g_kgpu_ticks[KGP_DRAW] += kg_tick() - prof_t;
    if (s_dl.mode != 0) kg_dl_draw(d);
    if (s_probe.mode != 0) kg_probe(dc->cs, d, "draw");
    if (s_pc.mode != 0) kg_pass_note(d, dc->cs, dc->zs);
    if (dc->cs) { dc->cs->ver++; dc->cs->wseq = ++s_wseq; }
    if (dc->zs && (ps->depth_write || ps->stencil_write)) { dc->zs->ver++; dc->zs->wseq = ++s_wseq; }
    G.draws++;
    if (d->gvs) G.gvs_draws++;
    G.tris += d->topology == KG_TOPO_STRIP ? (d->icount > 2 ? d->icount - 2 : 0) : d->icount / 3;
}
