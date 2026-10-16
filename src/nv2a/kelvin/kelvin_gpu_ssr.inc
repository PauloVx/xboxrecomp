/* Screen-space reflections, included by kelvin_gpu.c after kelvin_gpu_post.inc (the draw path calls kg_ssr_draw).
 *
 * BLACK's reflective materials (wet mud and cobbles, windows, water, weapon metal) add a small environment map --
 * a 128x128 sphere map -- through the register combiners, at texture coordinates the vertex program derives from
 * the eye direction reflected about the normal (vsh_env_stages). This replaces that map with what the screen shows:
 *
 *   - Each env-mapped draw on the main target uses a variant of its specialised pixel shader (PS_TEX_SSR) that also
 *     writes three targets: how much a texel of the env map adds to the pixel (the combiner run with that texel at 0
 *     and at 1), what the sampled texel added, and the pixel's depth with a normal from its screen derivatives.
 *     The extra targets are blended like the colour for transparent surfaces; depth and normal are written as is.
 *   - Before the HUD, a full-screen pass marches each such pixel's reflected ray through the depth buffer. Where it
 *     hits, the pixel's env contribution is taken out and the scene at the hit put in, scaled the same way; where it
 *     leaves the screen, points back at the camera or finds nothing, the original env map stays.
 *
 * The extra targets are cleared after each frame's pass. Draws whose specialised shader is still compiling, CPU
 * vertex draws and draws on other targets keep the original reflection. */

static struct {
    bool tried, ok, used;
    unsigned int w, h;
    ID3D11Texture2D *tex[3];
    ID3D11RenderTargetView *rtv[3];
    ID3D11ShaderResourceView *srv[3];
    ID3D11Buffer *cb;                   /* SSRV, b4 of the env draws: camera half-frustum, 1 / target size */
    float cb_vals[4];
    ID3D11PixelShader *ps;
    ID3D11Texture2D *blur;              /* the scene with mips: hits blur with distance, misses read it as an environment */
    ID3D11ShaderResourceView *blur_srv;
    unsigned int blur_w, blur_h, blur_levels;
    unsigned long long draws, frames;   /* env draws that wrote the targets, passes run (RECOMP_KGPU_POST_TRACE) */
    /* RECOMP_KGPU_SSR_DUMP=PREFIX[,N] (diagnostic): the scene before and after the pass, and the reflective pixels'
     * weight, as PREFIX_before.bmp / _after.bmp / _weight.bmp, at the Nth pass and every 300th after it. */
    int dump_init;
    char dump[MAX_PATH];
    unsigned long long dump_at;
    /* ray-traced reflections this frame (kelvin_gpu_rt.inc): rgb a colour, a = 0 not traced, 1 no colour, 2 a hit */
    ID3D11ShaderResourceView *rt_srv;
    bool rt_valid;
} s_ssr;

static void ssr_free_targets(void)
{
    unsigned int i;
    for (i = 0; i < 3; i++) { SAFE_RELEASE(s_ssr.srv[i]); SAFE_RELEASE(s_ssr.rtv[i]); SAFE_RELEASE(s_ssr.tex[i]); }
    SAFE_RELEASE(s_ssr.blur_srv); SAFE_RELEASE(s_ssr.blur);
    s_ssr.w = s_ssr.h = s_ssr.blur_w = s_ssr.blur_h = 0;
}

/* The mip chain of the scene copy (GenerateMips needs a render-target texture). */
static bool ssr_blur(unsigned int w, unsigned int h)
{
    D3D11_TEXTURE2D_DESC td;
    if (s_ssr.blur && s_ssr.blur_w == w && s_ssr.blur_h == h) return true;
    SAFE_RELEASE(s_ssr.blur_srv); SAFE_RELEASE(s_ssr.blur);
    memset(&td, 0, sizeof(td));
    td.Width = w; td.Height = h; td.MipLevels = 0; td.ArraySize = 1; td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET; td.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_ssr.blur)) ||
        FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)s_ssr.blur, NULL, &s_ssr.blur_srv))) {
        SAFE_RELEASE(s_ssr.blur);
        return false;
    }
    ID3D11Texture2D_GetDesc(s_ssr.blur, &td);
    s_ssr.blur_w = w; s_ssr.blur_h = h; s_ssr.blur_levels = td.MipLevels;
    return true;
}

static bool ssr_init(void)
{
    if (s_ssr.tried) return s_ssr.ok;
    s_ssr.tried = true;
    s_ssr.ps = post_init() ? post_shader("ssr_ps") : NULL;
    s_ssr.ok = s_ssr.ps && make_buffer(16, D3D11_BIND_CONSTANT_BUFFER, &s_ssr.cb);
    if (!s_ssr.ok) fprintf(stderr, "[KGPU] screen-space reflections unavailable: their shader did not compile\n");
    return s_ssr.ok;
}

static void ssr_clear(void)
{
    static const FLOAT zero[4] = { 0, 0, 0, 0 };
    unsigned int i;
    for (i = 0; i < 3; i++) if (s_ssr.rtv[i]) ID3D11DeviceContext_ClearRenderTargetView(G.ctx, s_ssr.rtv[i], zero);
}

static bool ssr_targets(unsigned int w, unsigned int h)
{
    static const DXGI_FORMAT fmt[3] = { DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT };
    unsigned int i;
    if (s_ssr.tex[0] && s_ssr.w == w && s_ssr.h == h) return true;
    ssr_free_targets();
    for (i = 0; i < 3; i++)
        if (!post_texture(w, h, fmt[i], &s_ssr.tex[i], &s_ssr.srv[i], &s_ssr.rtv[i])) { ssr_free_targets(); return false; }
    s_ssr.w = w; s_ssr.h = h;
    ssr_clear();
    return true;
}

/* The draw path asks for every draw on the main target: whether this draw writes the reflection targets, and with
 * which pixel shader and blend state. */
static bool kg_ssr_draw(KGpuSurf *cs, KGpuSurf *zs, const PSConst *dpc, unsigned int alpha_mode, bool has_alpha,
                        const KPixelState *ps, const KGpuVs *ve, ID3D11PixelShader **psx, ID3D11BlendState **bs)
{
    PSConst pc;
    unsigned int i, env = 0;
    float hx, hy;
    if (s_ssr_setting <= 0 || !ve || !ve->env || !cs || !zs || !post_main_target(cs)) return false;
    pc = *dpc;
    for (i = 0; i < 4; i++)
        if ((ve->env & (1u << i)) && pc.texmode[i] == 1u && (pc.texflags[i] & 1u)) { pc.texflags[i] |= PS_TEX_SSR; env++; }
    if (!env || !kg_ao_camera(s_ao_draw_camera, &hx, &hy) || !ssr_init() || !ssr_targets(cs->hw, cs->hh)) return false;
    *psx = ps_lookup(&pc);
    if (!*psx) return false;                        /* still compiling: the original reflection this time */
    *bs = get_blend(ps, alpha_mode, has_alpha, true);
    if (!*bs) return false;
    {
        const float v[4] = { hx, hy, 1.0f / (float)cs->hw, 1.0f / (float)cs->hh };
        D3D11_MAPPED_SUBRESOURCE ms;
        if (memcmp(v, s_ssr.cb_vals, sizeof(v)) &&
            SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)s_ssr.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
            memcpy(ms.pData, v, sizeof(v));
            ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)s_ssr.cb, 0);
            memcpy(s_ssr.cb_vals, v, sizeof(v));
        }
        ID3D11DeviceContext_PSSetConstantBuffers(G.ctx, 4, 1, &s_ssr.cb);
    }
    s_ssr.used = true;
    s_ssr.draws++;
    return true;
}

/* The weight target as an 8-bit picture (diagnostic): where the env maps were and how strongly they add. */
static void ssr_dump_weight(unsigned int w, unsigned int h)
{
    D3D11_TEXTURE2D_DESC td;
    ID3D11Texture2D *stage = NULL;
    D3D11_MAPPED_SUBRESOURCE ms;
    char path[MAX_PATH + 32];
    uint8_t *px;
    unsigned int x, y;
    memset(&td, 0, sizeof(td));
    td.Width = w; td.Height = h; td.MipLevels = td.ArraySize = 1; td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &stage))) return;
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)stage, (ID3D11Resource *)s_ssr.tex[0]);
    px = (uint8_t *)malloc((size_t)w * h * 4);
    if (px && SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)stage, 0, D3D11_MAP_READ, 0, &ms))) {
        for (y = 0; y < h; y++) {
            const uint16_t *row = (const uint16_t *)((const uint8_t *)ms.pData + (size_t)y * ms.RowPitch);
            for (x = 0; x < w; x++) {
                unsigned int k;
                for (k = 0; k < 3; k++) {
                    /* half to float, enough for 0..1 */
                    const uint16_t hv = row[x * 4 + k];
                    const int ex = (hv >> 10) & 31;
                    const float f = ex ? (float)ldexp(1.0 + (hv & 1023) / 1024.0, ex - 15) : 0.0f;
                    const float c = f * 255.0f;
                    px[((size_t)y * w + x) * 4 + (2 - k)] = (uint8_t)(c > 255.0f ? 255 : c < 0.0f || (hv & 0x8000) ? 0 : c);
                }
                px[((size_t)y * w + x) * 4 + 3] = 255;
            }
        }
        ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)stage, 0);
        snprintf(path, sizeof(path), "%s_%llu_weight.bmp", s_ssr.dump, s_ssr.frames + 1);
        write_bmp(path, px, w, h, w * 4);
    }
    free(px);
    SAFE_RELEASE(stage);
}

/* Before the HUD (kg_post_scene): the reflections, for the pixels the env draws marked. */
static void kg_ssr_pass(KGpuSurf *cs, KGpuSurf *zs)
{
    static const float steps[4] = { 24, 40, 64, 96 };
    ID3D11ShaderResourceView *in[9], *none[9] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    ID3D11SamplerState *smp[2];
    D3D11_MAPPED_SUBRESOURCE ms;
    KgPostCb c;
    int q = s_ssr_setting;
    bool dumping;
    const bool rt = s_ssr.rt_valid && s_ssr.rt_srv;
    s_ssr.rt_valid = false;
    if (!s_ssr.used) return;
    s_ssr.used = false;
    if (q < 1 || !s_ssr.ok || !zs || !zs->depth || !zs->real || !zs->srv || zs->hw != cs->hw || zs->hh != cs->hh ||
        s_ssr.w != cs->hw || s_ssr.h != cs->hh || !ssr_blur(cs->hw, cs->hh)) { ssr_clear(); return; }
    memset(&c, 0, sizeof(c));
    if (!kg_ao_camera(s_post.ao_camera, &c.p3[0], &c.p3[1])) { ssr_clear(); return; }
    c.p3[2] = rt ? 1.0f : 0.0f;
    {
        static int dbg = -1;
        if (dbg < 0) { const char *e = getenv("RECOMP_KGPU_RT_REFL_DEBUG"); dbg = e && *e && *e != '0'; }
        c.p3[3] = rt && dbg ? 1.0f : 0.0f;
    }
    if (q > 4) q = 4;
    c.p0[0] = 1.0f / cs->hw; c.p0[1] = 1.0f / cs->hh; c.p0[2] = (float)cs->hw; c.p0[3] = (float)cs->hh;
    c.p1[0] = steps[q - 1]; c.p1[1] = 1.6f; c.p1[2] = 0.06f; c.p1[3] = s_ssr_strength;
    /* misses: how much the screen environment replaces the original map, and how blurred (a 128-texel map's detail);
     * hits: how much they blur at the end of the reach */
    c.p2[0] = s_ssr_env_mix; c.p2[1] = (float)(s_ssr.blur_levels > 7 ? s_ssr.blur_levels - 4 : 3); c.p2[2] = 2.5f;
    c.p2[3] = (float)(s_ssr.blur_levels ? s_ssr.blur_levels - 1 : 0);
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)s_post.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) { ssr_clear(); return; }
    memcpy(ms.pData, &c, sizeof(c));
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)s_post.cb, 0);
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_post.copy, (ID3D11Resource *)cs->tex);
    ID3D11DeviceContext_CopySubresourceRegion(G.ctx, (ID3D11Resource *)s_ssr.blur, 0, 0, 0, 0, (ID3D11Resource *)cs->tex, 0, NULL);
    ID3D11DeviceContext_GenerateMips(G.ctx, s_ssr.blur_srv);
    if (!s_ssr.dump_init) {
        const char *e = getenv("RECOMP_KGPU_SSR_DUMP"), *comma;
        s_ssr.dump_init = 1;
        if (e && *e) {
            comma = strchr(e, ',');
            snprintf(s_ssr.dump, sizeof(s_ssr.dump), "%.*s", comma ? (int)(comma - e) : (int)strlen(e), e);
            s_ssr.dump_at = comma ? strtoull(comma + 1, NULL, 0) : 600;
        }
    }
    dumping = s_ssr.dump[0] && s_ssr.frames + 1 >= s_ssr.dump_at && (s_ssr.frames + 1 - s_ssr.dump_at) % 300 == 0;
    if (dumping) {
        char path[MAX_PATH + 32];
        snprintf(path, sizeof(path), "%s_%llu_before.bmp", s_ssr.dump, s_ssr.frames + 1);
        free(post_ao_colour_capture(cs->tex, cs->hw, cs->hh, path));
        ssr_dump_weight(cs->hw, cs->hh);
    }
    in[0] = s_post.copy_srv; in[1] = NULL; in[2] = zs->srv; in[3] = NULL;
    in[4] = s_ssr.srv[0]; in[5] = s_ssr.srv[1]; in[6] = s_ssr.srv[2]; in[7] = s_ssr.blur_srv;
    in[8] = rt ? s_ssr.rt_srv : NULL;
    smp[0] = s_post.smp_lin; smp[1] = s_post.smp_pt;
    ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 1, &cs->rtv, NULL);
    fullscreen_pass_state(cs->hw, cs->hh);
    ID3D11DeviceContext_OMSetDepthStencilState(G.ctx, G.ds_none, 0);
    ID3D11DeviceContext_PSSetShader(G.ctx, s_ssr.ps, NULL, 0);
    ID3D11DeviceContext_PSSetConstantBuffers(G.ctx, 0, 1, &s_post.cb);
    ID3D11DeviceContext_PSSetSamplers(G.ctx, 0, 2, smp);
    ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 9, in);
    ID3D11DeviceContext_Draw(G.ctx, 3, 0);
    ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 9, none);
    if (dumping) {
        char path[MAX_PATH + 32];
        snprintf(path, sizeof(path), "%s_%llu_after.bmp", s_ssr.dump, s_ssr.frames + 1);
        free(post_ao_colour_capture(cs->tex, cs->hw, cs->hh, path));
        fprintf(stderr, "[KSSR] pass %llu dumped to %s_%llu_*.bmp\n", s_ssr.frames + 1, s_ssr.dump, s_ssr.frames + 1);
    }
    ssr_clear();
    s_ssr.frames++;
    if (s_post.post_trace && (s_ssr.frames == 1 || s_ssr.frames % 600 == 0))
        fprintf(stderr, "[KSSR] %llu frames with reflections, %llu env draws\n", s_ssr.frames, s_ssr.draws);
    kg_cache_invalidate();
}

/* A frame whose pass did not run (no HUD point reached on the main target) leaves nothing for the next one. */
static void kg_ssr_after_flip(void)
{
    s_ssr.rt_valid = false;
    if (!s_ssr.used) return;
    s_ssr.used = false;
    ssr_clear();
}
