/* DLAA (NVIDIA DLSS at the internal resolution), included by kelvin_gpu.c after kelvin_gpu_rt.inc.
 *
 * With anti-aliasing set to DLAA, every draw of the world (the main target, before the HUD) is shifted by a
 * different sub-pixel offset each frame (g_jit in the vertex programs' epilogue, a Halton 2,3 sequence over 8 frames).
 * The camera is the matrix most of the world's draws share (C[10..13], rows x y z w of clip space, with the viewport
 * transform C[58] C[59]); each pixel's world position comes back from the depth buffer through its inverse, and last
 * frame's camera says where it was then: the motion vectors. Very near pixels -- the first-person weapon, drawn with
 * a camera of its own -- move with the view and get none. At the HUD (kelvin_gpu_post.inc) DLSS takes the frame, its
 * depth and the motion, and its output replaces the frame before the HUD is drawn over it. Without DLSS (no SDK at
 * build time, no RTX GPU) the frame gets FXAA instead. */

#include "kelvin_dlss.h"

#define KG_DLSS_CAMERAS 8

static struct {
    int tried, ok;
    unsigned phase;
    float shift[2];                     /* this frame's offset of the world, render pixels (x right, y down) */
    struct { float m[16], c58[4], c59[4], zmax; uint32_t n; } cam[KG_DLSS_CAMERAS];
    unsigned ncam;
    float prev_m[16], prev_c58[4], prev_c59[4];
    bool have_prev;
    unsigned w, h, out_w, out_h;
    ID3D11Texture2D *mv, *out;
    /* upscaling: the full-size texture the main target is swapped to at the HUD, and what it had before */
    ID3D11Texture2D *big; ID3D11RenderTargetView *big_rtv; ID3D11ShaderResourceView *big_srv;
    KGpuSurf *swapped; ID3D11Texture2D *lowres; ID3D11RenderTargetView *lowres_rtv; ID3D11ShaderResourceView *lowres_srv;
    unsigned lowres_w, lowres_h, phases;
    bool refresh;
    ID3D11RenderTargetView *mv_rtv;
    ID3D11ShaderResourceView *out_srv;
    ID3D11PixelShader *ps_mv, *ps_copy;
    ID3D11Buffer *cb;
    long long last_qpc;
} s_dlss;

typedef struct KgDlssCb {
    float inv[4][4], prev[4][4], c58[4], c59[4], pc58[4], pc59[4], size[4], misc[4];
} KgDlssCb;

static const char kg_dlss_hlsl[] =
"cbuffer DlssCB : register(b1) {\n"
"    float4 inv0, inv1, inv2, inv3;\n"      /* rows of the inverse camera: clip/w -> world (homogeneous) */
"    float4 prev0, prev1, prev2, prev3;\n"  /* rows of last frame's camera */
"    float4 c58, c59, pc58, pc59;\n"        /* viewport transforms, this frame's and last frame's */
"    float4 size;\n"                        /* x,y the target in pixels; z,w the guest window */
"    float4 misc;\n"                        /* x zmax; y the nearest view distance that moves with the world */
"};\n"
"Texture2D<float4> Src : register(t0);\n"
"Texture2D<float> DepthTex : register(t2);\n"
"float2 mv_ps(float4 p : SV_Position) : SV_Target {\n"
"    float d = DepthTex.Load(int3(p.xy, 0));\n"
"    float2 g = p.xy / size.xy * size.zw;\n"
"    float3 n = (float3(g, d * misc.x) - c59.xyz) / c58.xyz;\n"
"    float4 v = float4(n, 1.0);\n"
"    float4 h = float4(dot(inv0, v), dot(inv1, v), dot(inv2, v), dot(inv3, v));\n"
"    if (abs(h.w) < 1.0e-20) return float2(0.0, 0.0);\n"
"    float w = 1.0 / h.w;\n"
"    if (w > 0.0 && w < misc.y) return float2(0.0, 0.0);\n"
"    float4 world = float4(h.xyz / h.w, 1.0);\n"
"    float4 c = float4(dot(prev0, world), dot(prev1, world), dot(prev2, world), dot(prev3, world));\n"
"    if (c.w <= 1.0e-6) return float2(0.0, 0.0);\n"
"    float2 q = c.xy / c.w * pc58.xy + pc59.xy;\n"
"    return q / size.zw * size.xy - p.xy;\n"
"}\n"
"float4 copy_ps(float4 p : SV_Position) : SV_Target { return float4(saturate(Src.Load(int3(p.xy, 0)).rgb), 1.0); }\n";

static ID3D11PixelShader *kg_dlss_shader(const char *entry)
{
    ID3DBlob *b = compile_hlsl("", kg_dlss_hlsl, entry, "ps_5_0");
    ID3D11PixelShader *ps = NULL;
    if (!b) return NULL;
    if (FAILED(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(b), ID3D10Blob_GetBufferSize(b), NULL, &ps))) ps = NULL;
    ID3D10Blob_Release(b);
    return ps;
}

static bool kg_dlss_ready(void)
{
    if (s_dlss.tried) return s_dlss.ok;
    s_dlss.tried = 1;
    if (!kdlss_init(G.dev)) return false;
    s_dlss.ps_mv = kg_dlss_shader("mv_ps");
    s_dlss.ps_copy = kg_dlss_shader("copy_ps");
    if (!s_dlss.ps_mv || !s_dlss.ps_copy || !make_buffer(sizeof(KgDlssCb), D3D11_BIND_CONSTANT_BUFFER, &s_dlss.cb)) {
        fprintf(stderr, "[DLSS] the motion vector shaders did not compile: DLSS off\n");
        return false;
    }
    s_dlss.ok = 1;
    return true;
}

/* Whether this frame's world is drawn for DLAA (the setting, and DLSS able to run). */
static bool kg_dlss_on(void)
{
    return s_post.aa >= PCV_AA_DLAA && s_post.aa < PCV_AA_COUNT && (!s_dlss.tried || s_dlss.ok);
}

static float kg_halton(unsigned i, unsigned b)
{
    float f = 1.0f, r = 0.0f;
    while (i) { f /= (float)b; r += f * (float)(i % b); i /= b; }
    return r;
}

/* Vertex programs: the jitter for this draw (clip-space offsets per unit w), and the camera vote. */
static void kg_dlss_vertex(const KGpuDraw *d, const KgDc *dc, float jit[4])
{
    const KPixelState *ps = d->ps;
    jit[0] = jit[1] = jit[2] = jit[3] = 0.0f;
    if (!kg_dlss_on() || s_post.done || !post_main_target(dc->cs)) return;
    jit[0] = 2.0f * s_dlss.shift[0] / (float)dc->cs->hw;
    jit[1] = -2.0f * s_dlss.shift[1] / (float)dc->cs->hh;

    if (d->gvs && d->gvs->constants && ps->depth_write && dc->zs && kg_viewmodel_scale(d->gvs->hash) == 1.0f) {
        const float *c = d->gvs->constants;
        unsigned i;
        for (i = 0; i < s_dlss.ncam; i++)
            if (!memcmp(s_dlss.cam[i].m, c + 40, sizeof(s_dlss.cam[i].m)) && !memcmp(s_dlss.cam[i].c58, c + 4 * 58, 16)) break;
        if (i == s_dlss.ncam && s_dlss.ncam < KG_DLSS_CAMERAS) {
            memcpy(s_dlss.cam[i].m, c + 40, sizeof(s_dlss.cam[i].m));
            memcpy(s_dlss.cam[i].c58, c + 4 * 58, 16);
            memcpy(s_dlss.cam[i].c59, c + 4 * 59, 16);
            s_dlss.cam[i].zmax = dc->zmax;
            s_dlss.cam[i].n = 0;
            s_dlss.ncam++;
        }
        if (i < s_dlss.ncam) s_dlss.cam[i].n++;
    }
}

/* At the HUD: DLAA over the world. False when it did not run (the caller anti-aliases another way). */
static bool kg_dlss_run(KGpuSurf *cs, KGpuSurf *zs)
{
    KgDlssCb cb;
    double inv[16];
    unsigned i, best = 0, out_w, out_h;
    int quality;
    double ratio;
    bool reset;
    long long now = qpc_now();
    D3D11_MAPPED_SUBRESOURCE ms;
    if (!kg_dlss_ready() || !zs || !zs->depth || !zs->srv || !zs->tex || zs->hw != cs->hw || zs->hh != cs->hh || !s_dlss.ncam) return false;
    for (i = 1; i < s_dlss.ncam; i++) if (s_dlss.cam[i].n > s_dlss.cam[best].n) best = i;
    if (!kg_rtc_inv4(s_dlss.cam[best].m, inv)) return false;
    {
        const bool up = s_post.aa >= PCV_AA_DLSS_QUALITY && G.out_w > cs->hw && G.out_h > cs->hh && cs->gw == 640 && cs->gh == 480;
        out_w = up ? G.out_w : cs->hw; out_h = up ? G.out_h : cs->hh;
        ratio = (double)cs->hh / (double)out_h;
        quality = ratio > 0.99 ? KDLSS_DLAA : ratio > 0.64 ? KDLSS_QUALITY : ratio > 0.55 ? KDLSS_BALANCED : KDLSS_PERFORMANCE;
    }
    if (s_dlss.w != cs->hw || s_dlss.h != cs->hh || s_dlss.out_w != out_w || s_dlss.out_h != out_h) {
        D3D11_TEXTURE2D_DESC td;
        SAFE_RELEASE(s_dlss.mv_rtv); SAFE_RELEASE(s_dlss.mv); SAFE_RELEASE(s_dlss.out_srv); SAFE_RELEASE(s_dlss.out);
        SAFE_RELEASE(s_dlss.big_rtv); SAFE_RELEASE(s_dlss.big_srv); SAFE_RELEASE(s_dlss.big);
        s_dlss.w = s_dlss.h = s_dlss.out_w = s_dlss.out_h = 0;
        memset(&td, 0, sizeof(td));
        td.Width = cs->hw; td.Height = cs->hh; td.MipLevels = 1; td.ArraySize = 1; td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.Format = DXGI_FORMAT_R16G16_FLOAT; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_dlss.mv)) ||
            FAILED(ID3D11Device_CreateRenderTargetView(G.dev, (ID3D11Resource *)s_dlss.mv, NULL, &s_dlss.mv_rtv))) return false;
        td.Width = out_w; td.Height = out_h;
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; td.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_dlss.out)) ||
            FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)s_dlss.out, NULL, &s_dlss.out_srv))) return false;
        if (out_w != cs->hw || out_h != cs->hh) {
            td.Format = cs->dxfmt; td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_dlss.big)) ||
                FAILED(ID3D11Device_CreateRenderTargetView(G.dev, (ID3D11Resource *)s_dlss.big, NULL, &s_dlss.big_rtv)) ||
                FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)s_dlss.big, NULL, &s_dlss.big_srv))) return false;
        }
        s_dlss.w = cs->hw; s_dlss.h = cs->hh; s_dlss.out_w = out_w; s_dlss.out_h = out_h;
        s_dlss.have_prev = false;
    }
    reset = !s_dlss.have_prev || !s_dlss.last_qpc || (double)(now - s_dlss.last_qpc) / (double)G.qpf > 0.25;
    memset(&cb, 0, sizeof(cb));
    for (i = 0; i < 16; i++) cb.inv[i / 4][i % 4] = (float)inv[i];
    memcpy(cb.prev, s_dlss.have_prev ? s_dlss.prev_m : s_dlss.cam[best].m, sizeof(cb.prev));
    memcpy(cb.c58, s_dlss.cam[best].c58, 16); memcpy(cb.c59, s_dlss.cam[best].c59, 16);
    memcpy(cb.pc58, s_dlss.have_prev ? s_dlss.prev_c58 : s_dlss.cam[best].c58, 16);
    memcpy(cb.pc59, s_dlss.have_prev ? s_dlss.prev_c59 : s_dlss.cam[best].c59, 16);
    cb.size[0] = (float)cs->hw; cb.size[1] = (float)cs->hh; cb.size[2] = (float)cs->gw; cb.size[3] = (float)cs->gh;
    cb.misc[0] = s_dlss.cam[best].zmax; cb.misc[1] = 1.0f;
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)s_dlss.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) return false;
    memcpy(ms.pData, &cb, sizeof(cb));
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)s_dlss.cb, 0);
    ID3D11DeviceContext_PSSetConstantBuffers(G.ctx, 1, 1, &s_dlss.cb);
    post_pass(s_dlss.ps_mv, s_dlss.mv_rtv, cs->hw, cs->hh, NULL, NULL, zs->srv, NULL);
    ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 0, NULL, NULL);
    if (!kdlss_evaluate(G.ctx, (ID3D11Resource *)cs->tex, (ID3D11Resource *)zs->tex, (ID3D11Resource *)s_dlss.mv,
                        (ID3D11Resource *)s_dlss.out, cs->hw, cs->hh, out_w, out_h, quality,
                        s_dlss.shift[0], s_dlss.shift[1], reset)) {   /* measured: this sign is the steady, sharp one */
        kg_cache_invalidate();
        if (!kdlss_available()) s_dlss.ok = 0;
        return false;
    }
    kg_cache_invalidate();
    if (s_dlss.big) {
        /* the HUD and the flip now see the main target at the internal resolution, until the flip is done */
        post_pass(s_dlss.ps_copy, s_dlss.big_rtv, out_w, out_h, s_dlss.out_srv, NULL, NULL, NULL);
        s_dlss.swapped = cs;
        s_dlss.lowres = cs->tex; s_dlss.lowres_rtv = cs->rtv; s_dlss.lowres_srv = cs->srv;
        s_dlss.lowres_w = cs->hw; s_dlss.lowres_h = cs->hh;
        cs->tex = s_dlss.big; cs->rtv = s_dlss.big_rtv; cs->srv = s_dlss.big_srv; cs->hw = out_w; cs->hh = out_h;
        s_dlss.refresh = true;
    } else
        post_pass(s_dlss.ps_copy, cs->rtv, cs->hw, cs->hh, s_dlss.out_srv, NULL, NULL, NULL);
    memcpy(s_dlss.prev_m, s_dlss.cam[best].m, sizeof(s_dlss.prev_m));
    memcpy(s_dlss.prev_c58, s_dlss.cam[best].c58, 16);
    memcpy(s_dlss.prev_c59, s_dlss.cam[best].c59, 16);
    s_dlss.have_prev = true;
    s_dlss.last_qpc = now;
    kg_cache_invalidate();
    return true;
}

static const KGpuSurf *kg_dlss_swapped_surface(void) { return s_dlss.swapped; }

/* Once after the swap: the draw that set it off recomputes its viewport and targets. */
static bool kg_dlss_take_refresh(void)
{
    const bool r = s_dlss.refresh;
    s_dlss.refresh = false;
    return r;
}

/* After a flip: the next frame's jitter, and a fresh camera vote. */
static void kg_dlss_after_flip(void)
{
    if (s_dlss.swapped) {
        KGpuSurf *cs = s_dlss.swapped;
        if (cs->used && cs->tex == s_dlss.big) {
            cs->tex = s_dlss.lowres; cs->rtv = s_dlss.lowres_rtv; cs->srv = s_dlss.lowres_srv;
            cs->hw = s_dlss.lowres_w; cs->hh = s_dlss.lowres_h;
        }
        s_dlss.swapped = NULL;
        kg_cache_invalidate();
    }
    s_dlss.phases = s_dlss.out_h && s_dlss.h ? (unsigned)(8.0 * (double)s_dlss.out_h * s_dlss.out_h / ((double)s_dlss.h * s_dlss.h) + 0.5) : 8u;
    if (s_dlss.phases < 8u) s_dlss.phases = 8u;
    if (s_dlss.phases > 64u) s_dlss.phases = 64u;
    s_dlss.phase = (s_dlss.phase + 1u) % s_dlss.phases;
    s_dlss.shift[0] = kg_halton(s_dlss.phase + 1u, 2) - 0.5f;
    s_dlss.shift[1] = kg_halton(s_dlss.phase + 1u, 3) - 0.5f;
    s_dlss.ncam = 0;
}
