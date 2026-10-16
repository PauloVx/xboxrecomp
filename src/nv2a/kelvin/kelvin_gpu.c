/*
 * Kelvin D3D11 backend. See kelvin_gpu.h.
 *
 * Pixel semantics follow kelvin_pixel.c (the CPU reference); the shader that
 * evaluates them is kelvin_gpu_hlsl.inc.
 */
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include "kelvin_rt.h"
#include <intrin.h>
#include <nmmintrin.h>
#include <tlhelp32.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include "kelvin_gpu.h"
#include "kelvin_int.h"
#include "../../video/fb_present.h"
#include "../../video/pc_video.h"

#include "kelvin_gpu_hlsl.inc"

/* ---- constant buffer layouts (mirror the HLSL) ------------------------------- */

typedef struct PSStage {
    uint32_t rgb_in[4], alpha_in[4], outs[4];
    float c0[4], c1[4];
} PSStage;

typedef struct PSConst {
    PSStage stage[8];
    uint32_t fin_abcd[4];
    uint32_t fin_efg[4];
    float final_c0[4], final_c1[4];
    float fogcolor[4];
    uint32_t gen[4];
    uint32_t texmode[4];
    uint32_t texflags[4];
    float texrcp[4][4];
    uint32_t ckey[4];
    uint32_t ckeymask[4];
    uint32_t ofs[4];
} PSConst;

typedef struct VSG {
    float scale[4];
    float fog[4];
    uint32_t flags[4];
    float vm[4];                 /* x: viewmodel scale about the viewport centre (1 for none) */
} VSG;

static volatile uint32_t s_vm_hash[8];
static volatile int s_vm_count;
static volatile float s_vm_scale = 1.0f;

/* The camera hook runs on the title thread while the GPU consumes the pair.
 * Publish both frustum dimensions together, including when aim/FOV changes. */
static volatile LONG64 s_ao_camera;
static uint64_t s_ao_draw_camera; /* GPU-owned camera snapshot of the draw currently being consumed */

void kgpu_set_ao_camera(float half_x, float half_y)
{
    union { float half[2]; LONG64 bits; } camera;
    if (!(half_x > 0.001f && half_x < 100.0f && half_y > 0.001f && half_y < 100.0f)) {
        InterlockedExchange64(&s_ao_camera, 0);
        return;
    }
    camera.half[0] = half_x; camera.half[1] = half_y;
    InterlockedExchange64(&s_ao_camera, camera.bits);
}

static uint64_t kg_ao_camera_bits(void)
{
    return (uint64_t)InterlockedCompareExchange64(&s_ao_camera, 0, 0);
}

static bool kg_ao_camera(uint64_t bits, float *half_x, float *half_y)
{
    union { float half[2]; LONG64 bits; } camera;
    camera.bits = (LONG64)bits;
    if (!camera.bits) return false;
    *half_x = camera.half[0]; *half_y = camera.half[1];
    return true;
}

void kgpu_set_viewmodel(const uint32_t *vs_hashes, int count, float scale)
{
    int i;
    if (count > 8) count = 8;
    s_vm_count = 0;
    for (i = 0; i < count; i++) s_vm_hash[i] = vs_hashes[i];
    s_vm_scale = scale;
    s_vm_count = count < 0 ? 0 : count;
}

static float kg_viewmodel_scale(uint32_t vs_hash)
{
    int i, n = s_vm_count;
    for (i = 0; i < n; i++) if (s_vm_hash[i] == vs_hash) return s_vm_scale;
    return 1.0f;
}

typedef struct VSConst {
    float scale[4];
    uint32_t rec0[4], rec1[4], rec2[4];
    float misc[4];
} VSConst;

/* ---- state ----------------------------------------------------------------------- */

#define KG_MAX_SURF 32
#define KG_TEX_SLOTS 4096
#define KG_STAGING 3

typedef struct KGpuSurf {
    bool used, depth, swz;
    uint32_t va, pitch, format;
    unsigned int gw, gh, hw, hh, bpp;
    DXGI_FORMAT dxfmt;
    ID3D11Texture2D *tex;
    ID3D11RenderTargetView *rtv;
    ID3D11DepthStencilView *dsv;
    ID3D11ShaderResourceView *srv;
    uint64_t clock;
    /* Games reuse surface memory as another kind of surface (BLACK's glare pass reads the
     * depth buffer as an ARGB texture and uses a colour target as a depth buffer). `ver`
     * counts the writes; conv_* is the depth buffer read back as packed Z24S8 words, and a
     * depth surface that was made from a colour surface remembers which write it holds. */
    uint64_t ver;
    ID3D11Texture2D *conv_tex;
    ID3D11RenderTargetView *conv_rtv;
    ID3D11ShaderResourceView *conv_srv;
    uint64_t conv_ver;
    bool from_color, real;        /* real: cleared or drawn to as a depth buffer */
    uint64_t src_ver;
    /* Which of two surfaces over the same memory was written last (s_wseq at the write): the one written later
     * holds what the memory holds. */
    uint64_t wseq;
} KGpuSurf;

typedef struct KGpuTex {
    bool used;
    uint32_t base, fmt, w, h, pitch, levels, palette, palette_len;
    uint32_t span;
    uint64_t sig[2];
    uint64_t epoch, clock;
    size_t bytes;
    ID3D11Texture2D *tex;
    ID3D11ShaderResourceView *srv;
    HANDLE rt_h;                       /* ray tracing: the texture shared with the D3D12 side, */
    int rt_index;                      /* and its index there once a captured draw used it (-1 before) */
} KGpuTex;

typedef struct KeyedPtr { uint64_t key; void *ptr; } KeyedPtr;

static struct {
    int tried, ok;
    ID3D11Device *dev;
    ID3D11DeviceContext *ctx;
    ID3D11DeviceContext1 *ctx1;
    KGpuTex *last_tex;                 /* the cache entry tex_get last looked at */
    HMODULE compiler;
    HRESULT (WINAPI *compile)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO *,
                              ID3DInclude *, LPCSTR, LPCSTR, UINT, UINT,
                              ID3DBlob **, ID3DBlob **);
    ID3D11VertexShader *vs;
    ID3D11PixelShader *ps, *ps_flat;
    ID3D11VertexShader *clear_vs;
    ID3D11PixelShader *clear_ps, *clear_depth_ps, *d2c_ps, *c2d_ps;
    ID3D11BlendState *bs_none;
    ID3D11DepthStencilState *ds_none, *ds_always;
    ID3D11RasterizerState *rs_none;
    ID3D11Buffer *cb_ps, *cb_vs, *cb_clear, *vb, *ib;
    ID3D11Buffer *arena, *cb_gvs, *cb_c, *cb_a;
    ID3D11ShaderResourceView *arena_srv;
    size_t arena_cap, arena_pos;           /* in float4s */
    ID3D11ShaderResourceView *vb_srv;
    size_t vb_cap, vb_pos, ib_cap, ib_pos;     /* in float4s / indices */
    ID3D11ShaderResourceView *dummy_srv;
    ID3D11Texture2D *dummy_tex;
    ID3D11SamplerState *default_smp;
    KGpuSurf surf[KG_MAX_SURF];
    KGpuTex *tex;
    size_t tex_bytes;
    KeyedPtr blend[256], depth[256], raster[128], sampler[256];
    int nblend, ndepth, nraster, nsampler;
    ID3D11Texture2D *fb_copy;
    unsigned int fb_w, fb_h;
    DXGI_FORMAT fb_fmt;
    ID3D11Texture2D *stage[KG_STAGING];
    uint64_t stage_frame[KG_STAGING];
    uint32_t stage_va[KG_STAGING];
    unsigned int stage_aspect_w[KG_STAGING], stage_aspect_h[KG_STAGING];
    unsigned int stage_w, stage_h;
    uint64_t epoch, clock, frames;
    const char *dump;
    unsigned int dump_every;
    long long qpf;
    unsigned int host_w, host_h;               /* the raster size of the full-screen targets */
    unsigned int out_w, out_h;                 /* the size presented: the raster, or half of it under super-sampling */
    /* counters */
    uint64_t gvs_draws, draws, tris, tex_uploads, tex_hits, tex_validations, vb_bytes, skipped;
} G;

long long g_kgpu_ticks[KGP_COUNT];
static long long s_flip_prev_qpc;
static unsigned int s_hist[6];            /* flip intervals: <=18, <=20, <=25, <=34, <=70, longer (ms) */
static unsigned int s_fine[40];           /* the same in bins of half a millisecond, up to 20 ms */
static double s_hist_max_ms;
static volatile LONG64 s_flip_count;      /* flips so far and the qpc of each, for the slow-flip sampler */
static volatile long long s_flip_q[1024];
void kgpu_profile_maybe_start(void);
static int s_enabled = -1;

static long long qpc_now(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}
/* The per-stage CPU times of the periodic report cost eight clock reads per draw; they are on only with
 * RECOMP_KGPU_TIMES=1. */
static int s_times = -1;
static long long kg_tick(void)
{
    if (s_times < 0) { const char *e = getenv("RECOMP_KGPU_TIMES"); s_times = e && *e && *e != '0'; }
    return s_times ? qpc_now() : 0;
}
#define PROF(which, stmt) do { long long p0_ = kg_tick(); stmt; g_kgpu_ticks[which] += kg_tick() - p0_; } while (0)


bool kgpu_enabled(void)
{
    if (s_enabled < 0) {
        const char *e = getenv("RECOMP_KELVIN_GPU");
        s_enabled = e && *e && *e != '0';
    }
    return s_enabled != 0;
}

bool kgpu_active(void)
{
    return G.ok != 0;
}

/* Texture contents are revalidated once per kick. With asynchronous submission the kick counter travels with each
 * draw (kelvin_gpu_async.inc) and the submitting thread sets G.epoch from it. */
static uint64_t s_kick_epoch;
static uint64_t s_surf_gen;              /* bumped when a surface is created or freed */
static void kg_bound_reset(void);        /* the draw path's record of the context's bindings is stale */
static void kg_cache_invalidate(void);   /* ... and so is what it derived from surfaces, states and textures */
static void kg_post_poll(void);          /* the picture settings (kelvin_gpu_post.inc) */
static void kgq_start(void);
static bool kgq_enabled(void);
static void kgq_report(void);
/* Hand a finished frame to the presenter: at once, or at the time the flip was queued with (kelvin_gpu_async.inc). */
static void kgq_publish_frame(unsigned slot, uint64_t value, uint32_t va, uint32_t aspect_w, uint32_t aspect_h);

void kgpu_note_kick(void)
{
    s_kick_epoch++;
    if (!kgq_enabled()) G.epoch++;
}

void kgpu_host_size(unsigned int *w, unsigned int *h)
{
    *w = G.host_w;
    *h = G.host_h;
}

#include "kelvin_gpu_time.inc"
#include "kelvin_gpu_prof.inc"

#define SAFE_RELEASE(p) do { if (p) { IUnknown_Release((IUnknown *)(p)); (p) = NULL; } } while (0)
#define HR_CHECK(expr, what) do { HRESULT hr_ = (expr); if (FAILED(hr_)) { \
    fprintf(stderr, "[KGPU] %s failed: 0x%08lX\n", what, (unsigned long)hr_); return false; } } while (0)

/* ---- hashing -------------------------------------------------------------------- */

static void sig_bytes(const uint8_t *p, size_t n, uint64_t out[2])
{
    uint64_t a = 0x9E3779B97F4A7C15ull, b = 0xC2B2AE3D27D4EB4Full, c = 0x165667B19E3779F9ull;
    size_t i = 0;
    for (; i + 24 <= n; i += 24) {
        uint64_t x, y, z;
        memcpy(&x, p + i, 8); memcpy(&y, p + i + 8, 8); memcpy(&z, p + i + 16, 8);
        a = _mm_crc32_u64(a, x);
        b = _mm_crc32_u64(b, y);
        c = _mm_crc32_u64(c, z);
    }
    for (; i < n; i++) {
        a = _mm_crc32_u8((unsigned int)a, p[i]);
    }
    out[0] = (a & 0xFFFFFFFFull) | (b << 32);
    out[1] = (c & 0xFFFFFFFFull) | ((uint64_t)n << 32);
}

/* ---- device ---------------------------------------------------------------------- */

static ID3DBlob *compile_hlsl(const char *a, const char *b, const char *entry,
                              const char *target)
{
    size_t la = strlen(a), lb = strlen(b);
    char *src = (char *)malloc(la + lb + 1);
    ID3DBlob *code = NULL, *errors = NULL;
    HRESULT hr;
    if (!src) return NULL;
    memcpy(src, a, la);
    memcpy(src + la, b, lb + 1);
    hr = G.compile(src, la + lb, "kelvin_gpu", NULL, NULL, entry, target,
                   D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(hr)) {
        fprintf(stderr, "[KGPU] HLSL %s failed 0x%08lX:\n%s\n", entry, (unsigned long)hr,
                errors ? (const char *)ID3D10Blob_GetBufferPointer(errors) : "(no log)");
        code = NULL;
    } else if (errors && getenv("RECOMP_KGPU_SHADER_LOG")) {
        fprintf(stderr, "[KGPU] HLSL %s warnings:\n%s\n", entry,
                (const char *)ID3D10Blob_GetBufferPointer(errors));
    }
    if (errors) ID3D10Blob_Release(errors);
    free(src);
    return code;
}

/* The ubershader with flat shading (colours from a triangle's first vertex), for draws whose specialised shader is not ready. */
static ID3DBlob *compile_flat_ps(void)
{
    static const char def[] = "#define KG_FLAT 1\n";
    size_t ld = sizeof(def) - 1, lc = strlen(kgpu_hlsl_common), lp = strlen(kgpu_hlsl_ps);
    char *src = (char *)malloc(ld + lc + lp + 1);
    ID3DBlob *code = NULL, *errors = NULL;
    HRESULT hr;
    if (!src) return NULL;
    memcpy(src, def, ld); memcpy(src + ld, kgpu_hlsl_common, lc); memcpy(src + ld + lc, kgpu_hlsl_ps, lp + 1);
    hr = G.compile(src, ld + lc + lp, "kgpu_flat", NULL, NULL, "ps_main", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(hr)) {
        fprintf(stderr, "[KGPU] flat-shading pixel shader failed 0x%08lX: %s\n", (unsigned long)hr,
                errors ? (const char *)ID3D10Blob_GetBufferPointer(errors) : "(no log)");
        code = NULL;
    }
    if (errors) ID3D10Blob_Release(errors);
    free(src);
    return code;
}

static bool make_buffer(UINT bytes, UINT bind, ID3D11Buffer **out)
{
    D3D11_BUFFER_DESC d;
    memset(&d, 0, sizeof(d));
    d.ByteWidth = bytes;
    d.Usage = D3D11_USAGE_DYNAMIC;
    d.BindFlags = bind;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    HR_CHECK(ID3D11Device_CreateBuffer(G.dev, &d, NULL, out), "CreateBuffer");
    return true;
}

/* d2c_ps: above 2^23 a float holds whole numbers only, so adding 0.5 rounds to even: it moved odd depths up by one and
 * turned the far plane (1.0 -> 16777215.5) into 2^24, which packs as zero. */
static const char kgpu_hlsl_clear[] =
"cbuffer ClearCB : register(b0) { float4 color; float depth; float3 pad; };\n"
"float4 clear_vs(uint i : SV_VertexID) : SV_Position {\n"
"    float2 uv = float2((i << 1) & 2, i & 2);\n"
"    return float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1);\n"
"}\n"
"float4 clear_ps(float4 p : SV_Position) : SV_Target { return color; }\n"
"float clear_depth_ps(float4 p : SV_Position) : SV_Depth { return depth; }\n"
"Texture2D<float> DepthTex : register(t0);\n"
"Texture2D<float4> ColorTex : register(t0);\n"
"float4 d2c_ps(float4 p : SV_Position) : SV_Target {\n"
"    float s = saturate(DepthTex.Load(int3((int2)p.xy, 0))) * 16777215.0;\n"
"    uint z = s >= 8388608.0 ? (uint)s : (uint)(s + 0.5);\n"
"    return float4((float)((z >> 8) & 255u) / 255.0, (float)(z & 255u) / 255.0, 0.0, (float)((z >> 16) & 255u) / 255.0);\n"
"}\n"
"float c2d_ps(float4 p : SV_Position) : SV_Depth {\n"
"    float4 c = ColorTex.Load(int3((int2)p.xy, 0));\n"
"    uint z = ((uint)(c.a * 255.0 + 0.5) << 16) | ((uint)(c.r * 255.0 + 0.5) << 8) | (uint)(c.g * 255.0 + 0.5);\n"
"    return (float)z / 16777215.0;\n"
"}\n";

/* The Video Settings page asks for another raster size: it is applied between two frames, on the thread that
 * owns the surfaces (kg_apply_pending_size, at the end of a flip), so no draw ever straddles it. */
static volatile LONG64 s_pending_size;      /* raster width, raster height, output width, output height: 16 bits each */
static void kgpu_request_host_size(unsigned int w, unsigned int h, unsigned int ow, unsigned int oh)
{
    if (w < 640 || h < 480 || w > 8192 || h > 8192 || (uint64_t)w * h > 16777216u || ow < 640 || oh < 480 || ow > w || oh > h) return;
    InterlockedExchange64(&s_pending_size, ((LONG64)w << 48) | ((LONG64)h << 32) | ((LONG64)ow << 16) | (LONG64)oh);
}

static void kgpu_vs_precompile(void);

bool kgpu_init(void)
{
    IDXGIFactory1 *factory = NULL;
    IDXGIAdapter1 *adapter = NULL, *best = NULL;
    SIZE_T best_mem = 0;
    LUID best_luid = { 0, 0 };
    UINT i, flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL levels[2] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL got;
    ID3DBlob *vsb = NULL, *psb = NULL, *psfb = NULL, *cvs = NULL, *cps = NULL, *cdps = NULL;
    D3D11_SHADER_RESOURCE_VIEW_DESC srvd;
    LARGE_INTEGER f;
    HRESULT hr;

    if (G.tried) return G.ok != 0;
    G.tried = 1;
    if (!kgpu_enabled()) return false;
    QueryPerformanceFrequency(&f);
    G.qpf = f.QuadPart;
    {
        /* The raster size is the player's internal resolution (pc_video.c): RECOMP_RENDER_WIDTH/HEIGHT still
         * override it, and the Video Settings page changes it while the game runs (kgpu_request_host_size). */
        unsigned int rw, rh, ow, oh;
        pc_video_raster_size(&rw, &rh);
        pc_video_render_size(&ow, &oh);
        G.host_w = rw; G.host_h = rh;
        G.out_w = ow < rw ? ow : rw; G.out_h = oh < rh ? oh : rh;
    }
    if (G.host_w < 640 || G.host_h < 480 || G.host_w > 8192 || G.host_h > 8192) {
        G.host_w = 640; G.host_h = 480;
    }
    if (G.out_w < 640 || G.out_h < 480) { G.out_w = G.host_w; G.out_h = G.host_h; }
    G.dump = getenv("RECOMP_KELVIN_DUMP");
    {
        const char *e = getenv("RECOMP_KELVIN_EVERY");
        G.dump_every = e ? (unsigned int)strtoul(e, NULL, 0) : 1;
        if (!G.dump_every) G.dump_every = 1;
    }

    G.compiler = LoadLibraryA("d3dcompiler_47.dll");
    if (G.compiler) *(FARPROC *)&G.compile = GetProcAddress(G.compiler, "D3DCompile");
    if (!G.compile) { fprintf(stderr, "[KGPU] d3dcompiler_47.dll unavailable\n"); return false; }

    hr = CreateDXGIFactory1(&IID_IDXGIFactory1, (void **)&factory);
    if (FAILED(hr)) { fprintf(stderr, "[KGPU] DXGI factory failed 0x%08lX\n", (unsigned long)hr); return false; }
    for (i = 0; IDXGIFactory1_EnumAdapters1(factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 d;
        IDXGIAdapter1_GetDesc1(adapter, &d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.DedicatedVideoMemory > best_mem) {
            if (best) IDXGIAdapter1_Release(best);
            best = adapter; best_mem = d.DedicatedVideoMemory; best_luid = d.AdapterLuid;
            fprintf(stderr, "[KGPU] adapter %u: %ls, %zu MiB\n", i, d.Description,
                    (size_t)(d.DedicatedVideoMemory >> 20));
        } else {
            IDXGIAdapter1_Release(adapter);
        }
    }
    IDXGIFactory1_Release(factory);
    if (!best) { fprintf(stderr, "[KGPU] no hardware adapter\n"); return false; }
    xbox_FramebufferGpuAdapter((uint32_t)best_luid.LowPart, (int32_t)best_luid.HighPart);
    if (getenv("RECOMP_KGPU_DEBUG")) flags |= D3D11_CREATE_DEVICE_DEBUG;
    hr = D3D11CreateDevice((IDXGIAdapter *)best, D3D_DRIVER_TYPE_UNKNOWN, NULL, flags,
                           levels, 2, D3D11_SDK_VERSION, &G.dev, &got, &G.ctx);
    if (FAILED(hr) && (flags & D3D11_CREATE_DEVICE_DEBUG)) {
        flags &= ~D3D11_CREATE_DEVICE_DEBUG;
        hr = D3D11CreateDevice((IDXGIAdapter *)best, D3D_DRIVER_TYPE_UNKNOWN, NULL, flags,
                               levels, 2, D3D11_SDK_VERSION, &G.dev, &got, &G.ctx);
    }
    IDXGIAdapter1_Release(best);
    if (FAILED(hr)) { fprintf(stderr, "[KGPU] D3D11CreateDevice failed 0x%08lX\n", (unsigned long)hr); return false; }
    ID3D11DeviceContext_QueryInterface(G.ctx, &IID_ID3D11DeviceContext1, (void **)&G.ctx1);
    if (getenv("RECOMP_KGPU_RT")) krt_init(G.dev, G.ctx, best_luid);

    vsb = compile_hlsl(kgpu_hlsl_common, kgpu_hlsl_vs, "vs_main", "vs_5_0");
    psb = compile_hlsl(kgpu_hlsl_common, kgpu_hlsl_ps, "ps_main", "ps_5_0");
    psfb = compile_flat_ps();
    cvs = compile_hlsl("", kgpu_hlsl_clear, "clear_vs", "vs_5_0");
    cps = compile_hlsl("", kgpu_hlsl_clear, "clear_ps", "ps_5_0");
    cdps = compile_hlsl("", kgpu_hlsl_clear, "clear_depth_ps", "ps_5_0");
    if (!vsb || !psb || !cvs || !cps || !cdps) return false;
    HR_CHECK(ID3D11Device_CreateVertexShader(G.dev, ID3D10Blob_GetBufferPointer(vsb),
             ID3D10Blob_GetBufferSize(vsb), NULL, &G.vs), "CreateVertexShader");
    HR_CHECK(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(psb),
             ID3D10Blob_GetBufferSize(psb), NULL, &G.ps), "CreatePixelShader");
    if (psfb) ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(psfb), ID3D10Blob_GetBufferSize(psfb), NULL, &G.ps_flat);
    HR_CHECK(ID3D11Device_CreateVertexShader(G.dev, ID3D10Blob_GetBufferPointer(cvs),
             ID3D10Blob_GetBufferSize(cvs), NULL, &G.clear_vs), "CreateVertexShader(clear)");
    HR_CHECK(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(cps),
             ID3D10Blob_GetBufferSize(cps), NULL, &G.clear_ps), "CreatePixelShader(clear)");
    HR_CHECK(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(cdps),
             ID3D10Blob_GetBufferSize(cdps), NULL, &G.clear_depth_ps), "CreatePixelShader(clear depth)");
    ID3D10Blob_Release(vsb); ID3D10Blob_Release(psb);
    ID3D10Blob_Release(cvs); ID3D10Blob_Release(cps); ID3D10Blob_Release(cdps);
    {
        ID3DBlob *d2c = compile_hlsl("", kgpu_hlsl_clear, "d2c_ps", "ps_5_0");
        ID3DBlob *c2d = compile_hlsl("", kgpu_hlsl_clear, "c2d_ps", "ps_5_0");
        D3D11_BLEND_DESC bd;
        D3D11_DEPTH_STENCIL_DESC dd;
        D3D11_RASTERIZER_DESC rd;
        if (!d2c || !c2d) return false;
        HR_CHECK(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(d2c),
                 ID3D10Blob_GetBufferSize(d2c), NULL, &G.d2c_ps), "CreatePixelShader(d2c)");
        HR_CHECK(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(c2d),
                 ID3D10Blob_GetBufferSize(c2d), NULL, &G.c2d_ps), "CreatePixelShader(c2d)");
        ID3D10Blob_Release(d2c); ID3D10Blob_Release(c2d);
        memset(&bd, 0, sizeof(bd));
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        memset(&dd, 0, sizeof(dd));
        memset(&rd, 0, sizeof(rd));
        rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.DepthClipEnable = FALSE;
        HR_CHECK(ID3D11Device_CreateBlendState(G.dev, &bd, &G.bs_none), "CreateBlendState(none)");
        HR_CHECK(ID3D11Device_CreateDepthStencilState(G.dev, &dd, &G.ds_none), "CreateDepthStencilState(none)");
        dd.DepthEnable = TRUE; dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
        HR_CHECK(ID3D11Device_CreateDepthStencilState(G.dev, &dd, &G.ds_always), "CreateDepthStencilState(always)");
        HR_CHECK(ID3D11Device_CreateRasterizerState(G.dev, &rd, &G.rs_none), "CreateRasterizerState(none)");
    }

    if (!make_buffer(sizeof(PSConst), D3D11_BIND_CONSTANT_BUFFER, &G.cb_ps) ||
        !make_buffer(sizeof(VSConst), D3D11_BIND_CONSTANT_BUFFER, &G.cb_vs) ||
        !make_buffer(32, D3D11_BIND_CONSTANT_BUFFER, &G.cb_clear)) return false;
    G.vb_cap = (size_t)4 * 1024 * 1024;           /* float4s: 64 MiB */
    G.ib_cap = (size_t)4 * 1024 * 1024;           /* indices: 16 MiB */
    if (!make_buffer((UINT)(G.vb_cap * 16), D3D11_BIND_SHADER_RESOURCE, &G.vb) ||
        !make_buffer((UINT)(G.ib_cap * 4), D3D11_BIND_INDEX_BUFFER, &G.ib)) return false;
    G.arena_cap = (size_t)4 * 1024 * 1024;        /* float4s: 64 MiB of raw vertex data */
    if (!make_buffer((UINT)(G.arena_cap * 16), D3D11_BIND_SHADER_RESOURCE, &G.arena) ||
        !make_buffer(64, D3D11_BIND_CONSTANT_BUFFER, &G.cb_gvs) ||
        !make_buffer(192 * 16, D3D11_BIND_CONSTANT_BUFFER, &G.cb_c) ||
        !make_buffer(32 * 16, D3D11_BIND_CONSTANT_BUFFER, &G.cb_a)) return false;
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC ad;
        memset(&ad, 0, sizeof(ad));
        ad.Format = DXGI_FORMAT_R32_UINT;
        ad.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        ad.Buffer.FirstElement = 0;
        ad.Buffer.NumElements = (UINT)(G.arena_cap * 4);
        HR_CHECK(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)G.arena, &ad, &G.arena_srv),
                 "CreateShaderResourceView(arena)");
    }
    memset(&srvd, 0, sizeof(srvd));
    srvd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvd.Buffer.FirstElement = 0;
    srvd.Buffer.NumElements = (UINT)G.vb_cap;
    HR_CHECK(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)G.vb, &srvd, &G.vb_srv),
             "CreateShaderResourceView(vb)");
    {
        D3D11_TEXTURE2D_DESC td;
        uint32_t px = 0xFFFFFFFFu;
        D3D11_SUBRESOURCE_DATA init;
        memset(&td, 0, sizeof(td));
        td.Width = td.Height = 1; td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        init.pSysMem = &px; init.SysMemPitch = 4; init.SysMemSlicePitch = 0;
        HR_CHECK(ID3D11Device_CreateTexture2D(G.dev, &td, &init, &G.dummy_tex), "CreateTexture2D(dummy)");
        HR_CHECK(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)G.dummy_tex, NULL,
                 &G.dummy_srv), "CreateShaderResourceView(dummy)");
    }
    G.tex = (KGpuTex *)calloc(KG_TEX_SLOTS, sizeof(KGpuTex));
    if (!G.tex) return false;
    {
        D3D11_SAMPLER_DESC sd;
        memset(&sd, 0, sizeof(sd));
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxAnisotropy = 1; sd.ComparisonFunc = D3D11_COMPARISON_NEVER; sd.MaxLOD = D3D11_FLOAT32_MAX;
        HR_CHECK(ID3D11Device_CreateSamplerState(G.dev, &sd, &G.default_smp), "CreateSamplerState(default)");
    }
    G.ok = 1;
    kgpu_vs_precompile();
    kg_post_poll();
    pc_video_set_resize_hook(kgpu_request_host_size);
    kgq_start();
    kgpu_profile_maybe_start();
    fprintf(stderr, "[KGPU] D3D11 feature level 0x%X; host raster %ux%u, presented at %ux%u\n", (unsigned)got,
            G.host_w, G.host_h, G.out_w, G.out_h);
    return true;
}

/* ---- state objects ----------------------------------------------------------------- */

static void *keyed_get(KeyedPtr *tab, int *n, int cap, uint64_t key)
{
    int i;
    for (i = 0; i < *n && i < cap; i++) if (tab[i].key == key) return tab[i].ptr;
    return NULL;
}
static void keyed_put(KeyedPtr *tab, int *n, int cap, uint64_t key, void *ptr)
{
    if (*n < cap) { tab[*n].key = key; tab[*n].ptr = ptr; (*n)++; }
    else {   /* full: replace round-robin (state objects are cheap to recreate) */
        static unsigned rr;
        unsigned i = rr++ % (unsigned)cap;
        kg_cache_invalidate();
        IUnknown_Release((IUnknown *)tab[i].ptr);
        tab[i].key = key; tab[i].ptr = ptr;
    }
}

static D3D11_BLEND map_blend(int f, bool alpha)
{
    switch (f) {
    case 0: return D3D11_BLEND_ZERO;
    case 1: return D3D11_BLEND_ONE;
    case 2: return alpha ? D3D11_BLEND_SRC_ALPHA : D3D11_BLEND_SRC_COLOR;
    case 3: return alpha ? D3D11_BLEND_INV_SRC_ALPHA : D3D11_BLEND_INV_SRC_COLOR;
    case 4: return D3D11_BLEND_SRC_ALPHA;
    case 5: return D3D11_BLEND_INV_SRC_ALPHA;
    case 6: return D3D11_BLEND_DEST_ALPHA;
    case 7: return D3D11_BLEND_INV_DEST_ALPHA;
    case 8: return alpha ? D3D11_BLEND_DEST_ALPHA : D3D11_BLEND_DEST_COLOR;
    case 9: return alpha ? D3D11_BLEND_INV_DEST_ALPHA : D3D11_BLEND_INV_DEST_COLOR;
    case 10: return D3D11_BLEND_SRC_ALPHA_SAT;
    case 12: case 14: return D3D11_BLEND_BLEND_FACTOR;
    case 13: case 15: return D3D11_BLEND_INV_BLEND_FACTOR;
    default: return D3D11_BLEND_ZERO;
    }
}

/* With `mrt` the screen-space reflection targets 1..3 are bound too: 1 and 2 blend like the colour (a transparent
 * reflective surface adds its share), 3 (depth and normal) is written as is. */
static ID3D11BlendState *get_blend(const KPixelState *ps, unsigned int alpha_mode,
                                   bool has_alpha, bool mrt)
{
    uint64_t key;
    ID3D11BlendState *bs;
    D3D11_BLEND_DESC d;
    unsigned int mask = (ps->mask_r ? 1 : 0) | (ps->mask_g ? 2 : 0) | (ps->mask_b ? 4 : 0) |
                        (ps->mask_a && has_alpha ? 8 : 0);
    if (alpha_mode) mask |= 8;
    key = (uint64_t)(ps->blend ? 1 : 0) | ((uint64_t)(ps->blend_sfactor & 15) << 1) |
          ((uint64_t)(ps->blend_dfactor & 15) << 5) | ((uint64_t)(ps->blend_eqn & 7) << 9) |
          ((uint64_t)mask << 12) | ((uint64_t)alpha_mode << 16) | ((uint64_t)(mrt ? 1 : 0) << 20);
    bs = (ID3D11BlendState *)keyed_get(G.blend, &G.nblend, 256, key);
    if (bs) return bs;
    memset(&d, 0, sizeof(d));
    d.RenderTarget[0].RenderTargetWriteMask = (UINT8)mask;
    if (ps->blend) {
        D3D11_BLEND_OP op = D3D11_BLEND_OP_ADD;
        switch (ps->blend_eqn) {
        case 0: op = D3D11_BLEND_OP_SUBTRACT; break;
        case 1: case 5: op = D3D11_BLEND_OP_REV_SUBTRACT; break;
        case 3: op = D3D11_BLEND_OP_MIN; break;
        case 4: op = D3D11_BLEND_OP_MAX; break;
        default: break;
        }
        d.RenderTarget[0].BlendEnable = TRUE;
        d.RenderTarget[0].SrcBlend = map_blend(ps->blend_sfactor, false);
        d.RenderTarget[0].DestBlend = map_blend(ps->blend_dfactor, false);
        d.RenderTarget[0].BlendOp = op;
        d.RenderTarget[0].SrcBlendAlpha = map_blend(ps->blend_sfactor, true);
        d.RenderTarget[0].DestBlendAlpha = map_blend(ps->blend_dfactor, true);
        d.RenderTarget[0].BlendOpAlpha = op;
        if (op == D3D11_BLEND_OP_MIN || op == D3D11_BLEND_OP_MAX) {
            /* factors are ignored by MIN/MAX but must be legal */
            d.RenderTarget[0].SrcBlend = d.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
            d.RenderTarget[0].SrcBlendAlpha = d.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
        }
    }
    if (alpha_mode) {
        /* X8R8G8B8 targets: the alpha byte is forced; the shader outputs the
         * forced value and the alpha channel passes it through. */
        if (!ps->blend) {
            d.RenderTarget[0].BlendEnable = TRUE;
            d.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
            d.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
            d.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        }
        d.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        d.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
        d.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    }
    if (mrt) {
        d.IndependentBlendEnable = TRUE;
        d.RenderTarget[1] = d.RenderTarget[0];
        d.RenderTarget[2] = d.RenderTarget[0];
        d.RenderTarget[1].RenderTargetWriteMask = d.RenderTarget[2].RenderTargetWriteMask = (UINT8)(mask ? 15 : 0);
        d.RenderTarget[3].BlendEnable = FALSE;
        d.RenderTarget[3].RenderTargetWriteMask = (UINT8)(mask & 7 ? 15 : 0);
    }
    if (FAILED(ID3D11Device_CreateBlendState(G.dev, &d, &bs))) return NULL;
    keyed_put(G.blend, &G.nblend, 256, key, bs);
    return bs;
}

static ID3D11DepthStencilState *get_depth(const KPixelState *ps, bool have_zeta)
{
    uint64_t key;
    ID3D11DepthStencilState *ds;
    D3D11_DEPTH_STENCIL_DESC d;
    bool test = have_zeta && ps->depth_test;
    bool stencil = have_zeta && ps->stencil_test &&
                   ps->zeta_format == NV097_SET_SURFACE_FORMAT_ZETA_Z24S8;
    key = (test ? 1u : 0u) | ((test && ps->depth_write) ? 2u : 0u) |
          ((uint64_t)(ps->depth_func & 7) << 2) | ((stencil ? 1ull : 0ull) << 5);
    if (stencil) {
        key |= ((uint64_t)(ps->stencil_func & 7) << 6) | ((uint64_t)(ps->stencil_op_fail & 15) << 9) |
               ((uint64_t)(ps->stencil_op_zfail & 15) << 13) | ((uint64_t)(ps->stencil_op_zpass & 15) << 17) |
               ((uint64_t)(ps->stencil_mask_read & 255) << 21) |
               ((uint64_t)(ps->stencil_write ? ps->stencil_mask_write & 255 : 0) << 29);
    }
    ds = (ID3D11DepthStencilState *)keyed_get(G.depth, &G.ndepth, 256, key);
    if (ds) return ds;
    memset(&d, 0, sizeof(d));
    d.DepthEnable = test;
    d.DepthWriteMask = (test && ps->depth_write) ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
    d.DepthFunc = (D3D11_COMPARISON_FUNC)((ps->depth_func & 7) + 1);
    d.StencilEnable = stencil;
    d.StencilReadMask = (UINT8)ps->stencil_mask_read;
    d.StencilWriteMask = (UINT8)(ps->stencil_write ? ps->stencil_mask_write : 0);
    d.FrontFace.StencilFailOp = (D3D11_STENCIL_OP)(ps->stencil_op_fail ? ps->stencil_op_fail : 1);
    d.FrontFace.StencilDepthFailOp = (D3D11_STENCIL_OP)(ps->stencil_op_zfail ? ps->stencil_op_zfail : 1);
    d.FrontFace.StencilPassOp = (D3D11_STENCIL_OP)(ps->stencil_op_zpass ? ps->stencil_op_zpass : 1);
    d.FrontFace.StencilFunc = (D3D11_COMPARISON_FUNC)((ps->stencil_func & 7) + 1);
    d.BackFace = d.FrontFace;
    if (FAILED(ID3D11Device_CreateDepthStencilState(G.dev, &d, &ds))) return NULL;
    keyed_put(G.depth, &G.ndepth, 256, key, ds);
    return ds;
}

static ID3D11RasterizerState *get_raster(const KGpuDraw *dr)
{
    const KPixelState *ps = dr->ps;
    uint64_t key;
    ID3D11RasterizerState *rs;
    D3D11_RASTERIZER_DESC d;
    unsigned int cull = 0;               /* 0 none, 1 front, 2 back */
    int zb;
    float zs;
    uint32_t zsbits;
    if (dr->cull_enable) {
        cull = dr->cull_ctrl == NV_PGRAPH_SETUPRASTER_CULLCTRL_FRONT ? 1 : 2;
        if (dr->cull_ctrl == NV_PGRAPH_SETUPRASTER_CULLCTRL_BACK) cull = 2;
    }
    /* The bias is a whole number of the depth buffer's smallest steps (float depth: 2^-24 in [0.5, 1), what a zeta unit is). */
    {
        double zm = ps->zmax > 0.0f ? (double)ps->zmax : 16777215.0;
        double u = (double)dr->zbias * (16777216.0 / zm);       /* zeta units -> steps of 2^-24 */
        zb = (int)(u < 0.0 ? u - 0.5 : u + 0.5);
    }
    zs = dr->zslope;
    memcpy(&zsbits, &zs, sizeof(zsbits));
    key = cull | ((uint64_t)(dr->ccw_front ? 1 : 0) << 2) | ((uint64_t)(ps->depth_clip_cull ? 1 : 0) << 3) |
          ((uint64_t)(uint32_t)zb << 8) | ((uint64_t)(zsbits >> 8) << 40);
    rs = (ID3D11RasterizerState *)keyed_get(G.raster, &G.nraster, 128, key);
    if (rs) return rs;
    memset(&d, 0, sizeof(d));
    d.FillMode = D3D11_FILL_SOLID;
    d.CullMode = cull == 0 ? D3D11_CULL_NONE : cull == 1 ? D3D11_CULL_FRONT : D3D11_CULL_BACK;
    d.FrontCounterClockwise = dr->ccw_front ? TRUE : FALSE;
    d.DepthClipEnable = ps->depth_clip_cull ? TRUE : FALSE;
    d.ScissorEnable = TRUE;
    d.DepthBias = zb;
    d.SlopeScaledDepthBias = zs;
    if (FAILED(ID3D11Device_CreateRasterizerState(G.dev, &d, &rs))) return NULL;
    keyed_put(G.raster, &G.nraster, 128, key, rs);
    return rs;
}

static int s_af = 1;                       /* anisotropic filtering level (kg_post_poll) */

static ID3D11SamplerState *get_sampler(const KTexture *t)
{
    uint64_t key;
    ID3D11SamplerState *ss;
    D3D11_SAMPLER_DESC d;
    bool min_lin, mip_lin, mag_lin, lod0;
    float lod_bias = t->lod_bias, minl = t->min_lod, maxl = t->max_lod;
    uint32_t bc = t->border_color;
    /* key on the quantised parameters */
    key = (uint64_t)(t->min_filter & 15) | ((uint64_t)(t->mag_filter & 15) << 4) |
          ((uint64_t)(t->addr[0] & 7) << 8) | ((uint64_t)(t->addr[1] & 7) << 11) |
          ((uint64_t)(t->levels & 31) << 14) | ((uint64_t)(int)(lod_bias * 256) << 20) |
          ((uint64_t)(unsigned)(minl * 16) << 36) | ((uint64_t)(unsigned)(maxl * 16) << 44);
    key ^= (uint64_t)bc * 0x9E3779B97F4A7C15ull;
    key ^= (uint64_t)s_af * 0xD6E8FEB86659FD93ull;
    ss = (ID3D11SamplerState *)keyed_get(G.sampler, &G.nsampler, 256, key);
    if (ss) return ss;
    switch (t->min_filter) {
    case NV_PGRAPH_TEXFILTER0_MIN_BOX_LOD0:        min_lin = false; mip_lin = false; lod0 = true; break;
    case NV_PGRAPH_TEXFILTER0_MIN_TENT_LOD0:       min_lin = true;  mip_lin = false; lod0 = true; break;
    case NV_PGRAPH_TEXFILTER0_MIN_BOX_NEARESTLOD:  min_lin = false; mip_lin = false; lod0 = false; break;
    case NV_PGRAPH_TEXFILTER0_MIN_TENT_NEARESTLOD: min_lin = true;  mip_lin = false; lod0 = false; break;
    case NV_PGRAPH_TEXFILTER0_MIN_BOX_TENT_LOD:    min_lin = false; mip_lin = true;  lod0 = false; break;
    case NV_PGRAPH_TEXFILTER0_MIN_TENT_TENT_LOD:   min_lin = true;  mip_lin = true;  lod0 = false; break;
    default:                                       min_lin = true;  mip_lin = false; lod0 = true; break;
    }
    mag_lin = t->mag_filter != 1;
    memset(&d, 0, sizeof(d));
    d.Filter = (D3D11_FILTER)((min_lin ? 0x10 : 0) | (mag_lin ? 0x04 : 0) | (mip_lin ? 0x01 : 0));
    d.MaxAnisotropy = 1;
    if (s_af > 1 && min_lin && mag_lin && !lod0 && t->levels > 1) {
        /* anisotropic filtering needs linear everything; it only matters for a texture with mip levels */
        d.Filter = D3D11_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = (UINT)s_af;
    }
    d.AddressU = (D3D11_TEXTURE_ADDRESS_MODE)(t->addr[0] >= 1 && t->addr[0] <= 4 ? t->addr[0] : 3);
    d.AddressV = (D3D11_TEXTURE_ADDRESS_MODE)(t->addr[1] >= 1 && t->addr[1] <= 4 ? t->addr[1] : 3);
    d.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    d.MipLODBias = lod_bias;
    d.ComparisonFunc = D3D11_COMPARISON_NEVER;
    d.BorderColor[0] = ((bc >> 16) & 0xFF) / 255.0f;
    d.BorderColor[1] = ((bc >> 8) & 0xFF) / 255.0f;
    d.BorderColor[2] = (bc & 0xFF) / 255.0f;
    d.BorderColor[3] = ((bc >> 24) & 0xFF) / 255.0f;
    d.MinLOD = minl;
    d.MaxLOD = lod0 ? minl : maxl;
    if (d.MaxLOD < d.MinLOD) d.MaxLOD = d.MinLOD;
    if (FAILED(ID3D11Device_CreateSamplerState(G.dev, &d, &ss))) return NULL;
    keyed_put(G.sampler, &G.nsampler, 256, key, ss);
    return ss;
}

/* ---- surfaces ------------------------------------------------------------------------ */

static void surf_free(KGpuSurf *s)
{
    kg_cache_invalidate();
    s_surf_gen++;
    SAFE_RELEASE(s->conv_srv); SAFE_RELEASE(s->conv_rtv); SAFE_RELEASE(s->conv_tex);
    SAFE_RELEASE(s->srv); SAFE_RELEASE(s->dsv); SAFE_RELEASE(s->rtv); SAFE_RELEASE(s->tex);
    memset(s, 0, sizeof(*s));
}

static DXGI_FORMAT color_dxgi(unsigned int fmt, unsigned int *bpp)
{
    switch (fmt) {
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_Z1R5G5B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_O1R5G5B5:
        *bpp = 2; return DXGI_FORMAT_B5G5R5A1_UNORM;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5:
        *bpp = 2; return DXGI_FORMAT_B5G6R5_UNORM;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_B8:
        *bpp = 1; return DXGI_FORMAT_R8_UNORM;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_G8B8:
        *bpp = 2; return DXGI_FORMAT_R8G8_UNORM;
    default:
        *bpp = 4; return DXGI_FORMAT_B8G8R8A8_UNORM;
    }
}

/* The GPU surface for the target's colour or zeta allocation, created on first
 * use. Full-screen guest targets (640x480, linear, no AA) render at host size. */
static KGpuSurf *surf_get(const KelvinTarget *t, bool depth, unsigned int format)
{
    uint32_t va = depth ? t->zeta_va : t->color_va;
    uint32_t pitch = t->swizzled ? 0 : depth ? t->guest_zeta_pitch : t->guest_color_pitch;
    unsigned int gw = t->guest_width, gh = t->guest_height, hw, hh, bpp, i;
    DXGI_FORMAT fmt;
    KGpuSurf *s = NULL, *victim = NULL;
    D3D11_TEXTURE2D_DESC td;
    if (!va) return NULL;
    if (t->aa_x != 1 || t->aa_y != 1) {
        kelvin_unsupported(KU_SURFACE_FORMAT, "GPU surface with anti-aliasing", va);
        return NULL;
    }
    if (gw == 640 && gh == 480) { hw = G.host_w; hh = G.host_h; } else { hw = gw; hh = gh; }
    fmt = depth ? DXGI_FORMAT_R32G8X24_TYPELESS : color_dxgi(format, &bpp);
    if (depth) bpp = 4;
    for (i = 0; i < KG_MAX_SURF; i++) {
        KGpuSurf *q = &G.surf[i];
        if (!q->used) { if (!s) s = q; continue; }
        if (q->va == va && q->depth == depth) {
            if (q->pitch == pitch && q->gw == gw && q->gh == gh && q->format == format && q->swz == t->swizzled) {
                q->clock = ++G.clock;
                return q;
            }
            surf_free(q);
            s = q;
            break;
        }
        if (!victim || q->clock < victim->clock) victim = q;
    }
    if (!s) { surf_free(victim); s = victim; }
    memset(&td, 0, sizeof(td));
    td.Width = hw; td.Height = hh; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = fmt; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = depth ? (D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE)
                         : (D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s->tex))) {
        fprintf(stderr, "[KGPU] surface %ux%u create failed\n", hw, hh);
        memset(s, 0, sizeof(*s));
        return NULL;
    }
    if (depth) {
        D3D11_DEPTH_STENCIL_VIEW_DESC dd;
        memset(&dd, 0, sizeof(dd));
        dd.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        dd.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
        ID3D11Device_CreateDepthStencilView(G.dev, (ID3D11Resource *)s->tex, &dd, &s->dsv);
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC sd;
            memset(&sd, 0, sizeof(sd));
            sd.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            sd.Texture2D.MipLevels = 1;
            ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)s->tex, &sd, &s->srv);
        }
    } else {
        ID3D11Device_CreateRenderTargetView(G.dev, (ID3D11Resource *)s->tex, NULL, &s->rtv);
        ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)s->tex, NULL, &s->srv);
    }
    s->used = true; s->depth = depth; s->swz = t->swizzled; s->va = va; s->pitch = pitch; s->format = format;
    s->gw = gw; s->gh = gh; s->hw = hw; s->hh = hh; s->bpp = bpp; s->dxfmt = fmt;
    s->clock = ++G.clock;
    s_surf_gen++;
    fprintf(stderr, "[KGPU] %s surface 0x%08X: guest %ux%u pitch %u -> host %ux%u\n",
            depth ? "depth" : "colour", va, gw, gh, pitch, hw, hh);
    if (!depth) {
        /* A fresh colour surface starts black. */
        const FLOAT zero[4] = { 0, 0, 0, 0 };
        ID3D11DeviceContext_ClearRenderTargetView(G.ctx, s->rtv, zero);
    } else {
        ID3D11DeviceContext_ClearDepthStencilView(G.ctx, s->dsv,
            D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
        s->ver++;
    }
    return s;
}

static void depth_surface_refresh(KGpuSurf *z);
static uint64_t s_wseq;

/* A colour surface (not depth) whose guest allocation is exactly this
 * texture's, to sample it without a round trip through guest memory. */
static KGpuSurf *surf_for_texture(const KTexture *t)
{
    unsigned int i;
    for (i = 0; i < KG_MAX_SURF; i++) {
        KGpuSurf *s = &G.surf[i];
        if (!s->used || s->depth || s->va != t->base) continue;
        if (t->cubemap || t->levels != 1 || t->width != s->gw || t->height != s->gh) continue;
        if (s->swz) {
            /* a swizzled target sampled as a swizzled texture of the same shape: the layout is
             * the GPU's own either way */
            if (t->linear || t->kind != TK_8888 || s->bpp != 4) continue;
        } else if (!t->linear || t->pitch != s->pitch) {
            continue;
        }
        return s;
    }
    return NULL;
}

/* ---- reinterpreting surfaces ------------------------------------------------------------------------ */

static KGpuSurf *surf_find(uint32_t va, bool depth)
{
    unsigned int i;
    for (i = 0; i < KG_MAX_SURF; i++) {
        KGpuSurf *s = &G.surf[i];
        if (s->used && s->va == va && s->depth == depth) return s;
    }
    return NULL;
}

/* A depth surface whose guest allocation is exactly this (linear, whole-surface) texture's. */
static KGpuSurf *surf_depth_for_texture(const KTexture *t)
{
    KGpuSurf *s = surf_find(t->base, true);
    if (!s || s->swz || !t->linear || t->cubemap || t->levels != 1) return NULL;
    if (t->width != s->gw || t->height != s->gh || t->pitch != s->pitch) return NULL;
    return s;
}

static void fullscreen_pass_state(unsigned int w, unsigned int h)
{
    D3D11_VIEWPORT vp;
    kg_bound_reset();
    vp.TopLeftX = vp.TopLeftY = 0.0f; vp.Width = (FLOAT)w; vp.Height = (FLOAT)h; vp.MinDepth = 0.0f; vp.MaxDepth = 1.0f;
    ID3D11DeviceContext_OMSetBlendState(G.ctx, G.bs_none, NULL, 0xFFFFFFFFu);
    ID3D11DeviceContext_RSSetState(G.ctx, G.rs_none);
    ID3D11DeviceContext_RSSetViewports(G.ctx, 1, &vp);
    ID3D11DeviceContext_IASetInputLayout(G.ctx, NULL);
    ID3D11DeviceContext_IASetPrimitiveTopology(G.ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_VSSetShader(G.ctx, G.clear_vs, NULL, 0);
}

/* The depth buffer as an ARGB texture of packed Z24S8 words (what a title that samples its
 * depth buffer as colour reads), regenerated when the depth buffer was written since. */
static ID3D11ShaderResourceView *depth_as_color(KGpuSurf *d)
{
    ID3D11ShaderResourceView *none = NULL;
    if (!d->conv_tex) {
        D3D11_TEXTURE2D_DESC td;
        memset(&td, 0, sizeof(td));
        td.Width = d->hw; td.Height = d->hh; td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &d->conv_tex)) ||
            FAILED(ID3D11Device_CreateRenderTargetView(G.dev, (ID3D11Resource *)d->conv_tex, NULL, &d->conv_rtv)) ||
            FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)d->conv_tex, NULL, &d->conv_srv))) {
            SAFE_RELEASE(d->conv_srv); SAFE_RELEASE(d->conv_rtv); SAFE_RELEASE(d->conv_tex);
            return NULL;
        }
        d->conv_ver = ~(uint64_t)0;
    }
    if (d->conv_ver != d->ver) {
        ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 1, &d->conv_rtv, NULL);
        fullscreen_pass_state(d->hw, d->hh);
        ID3D11DeviceContext_OMSetDepthStencilState(G.ctx, G.ds_none, 0);
        ID3D11DeviceContext_PSSetShader(G.ctx, G.d2c_ps, NULL, 0);
        ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 1, &d->srv);
        if (s_kp_mode != 0) { KpInfo ki; memset(&ki, 0, sizeof(ki)); ki.kind = 2; ki.psh = 0xD2C; ki.tw = d->hw; ki.th = d->hh; kp_mark(&ki); }
        ID3D11DeviceContext_Draw(G.ctx, 3, 0);
        if (s_kp_mode != 0) kp_done();
        ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 1, &none);
        d->conv_ver = d->ver;
    }
    return d->conv_srv;
}

/* A colour surface's packed Z24S8 words as the contents of a depth surface. */
static void depth_from_color(KGpuSurf *z, KGpuSurf *c)
{
    ID3D11ShaderResourceView *none = NULL;
    ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 0, NULL, z->dsv);
    fullscreen_pass_state(z->hw, z->hh);
    ID3D11DeviceContext_OMSetDepthStencilState(G.ctx, G.ds_always, 0);
    ID3D11DeviceContext_PSSetShader(G.ctx, G.c2d_ps, NULL, 0);
    ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 1, &c->srv);
    if (s_kp_mode != 0) { KpInfo ki; memset(&ki, 0, sizeof(ki)); ki.kind = 2; ki.psh = 0xC2D; ki.tw = z->hw; ki.th = z->hh; kp_mark(&ki); }
    ID3D11DeviceContext_Draw(G.ctx, 3, 0);
    if (s_kp_mode != 0) kp_done();
    ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 1, &none);
    z->src_ver = c->ver;
    z->ver++;
    z->wseq = ++s_wseq;
}

/* Before a draw uses depth surface `z`: if a colour surface occupies the same memory and was written after the depth
 * surface, the memory holds that colour surface's packed words (BLACK's glare pass downsamples the depth buffer into a
 * colour target and then depth-tests its quads against that target used as a depth buffer; the target had been a
 * depth buffer before, so "never written as depth" does not tell). */
static void depth_surface_refresh(KGpuSurf *z)
{
    KGpuSurf *c = surf_find(z->va, false);
    if (!c || c->gw != z->gw || c->gh != z->gh) { z->from_color = false; return; }
    if (c->wseq > z->wseq) {
        z->from_color = true;
        depth_from_color(z, c);
    }
}

/* ---- textures -------------------------------------------------------------------------- */

static uint32_t spread_bits16(uint32_t v)
{
    v &= 0xFFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

static uint32_t swz_off(unsigned int w, unsigned int h, unsigned int x, unsigned int y)
{
    unsigned long m;
    uint32_t low;
    _BitScanForward(&m, w < h ? w : h);
    low = (1u << m) - 1;
    return spread_bits16(x & low) | (spread_bits16(y & low) << 1) |
           (((x >> m) | (y >> m)) << (2 * m));
}

static uint32_t pack_rgba(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
{
    return r | (g << 8) | (b << 16) | (a << 24);
}

static uint32_t expand5(uint32_t v) { return (v * 255 + 15) / 31; }
static uint32_t expand6(uint32_t v) { return (v * 255 + 31) / 63; }
static uint32_t expand4(uint32_t v) { return v * 17; }

static uint8_t clip_byte(int x) { return (uint8_t)(x < 0 ? 0 : x > 255 ? 255 : x); }

static uint32_t texel_rgba(const KTexture *t, const uint8_t *p)
{
    uint32_t v, l;
    switch (t->kind) {
    case TK_Y8:   return pack_rgba(p[0], p[0], p[0], 255);
    case TK_AY8:  return pack_rgba(p[0], p[0], p[0], p[0]);
    case TK_A8:   return pack_rgba(255, 255, 255, p[0]);
    case TK_A8Y8: return pack_rgba(p[0], p[0], p[0], p[1]);
    case TK_G8B8: return pack_rgba(p[0], p[1], p[0], p[1]);
    case TK_R8B8: return pack_rgba(p[1], p[0], p[0], p[1]);
    case TK_1555: case TK_X555:
        v = p[0] | (p[1] << 8);
        return pack_rgba(expand5((v >> 10) & 31), expand5((v >> 5) & 31), expand5(v & 31),
                         t->kind == TK_X555 ? 255 : ((v >> 15) & 1) * 255);
    case TK_4444:
        v = p[0] | (p[1] << 8);
        return pack_rgba(expand4((v >> 8) & 15), expand4((v >> 4) & 15), expand4(v & 15),
                         expand4((v >> 12) & 15));
    case TK_565:
        v = p[0] | (p[1] << 8);
        return pack_rgba(expand5((v >> 11) & 31), expand6((v >> 5) & 63), expand5(v & 31), 255);
    case TK_8888: return pack_rgba(p[2], p[1], p[0], p[3]);
    case TK_X888: return pack_rgba(p[2], p[1], p[0], 255);
    case TK_ABGR: return pack_rgba(p[0], p[1], p[2], p[3]);
    case TK_BGRA: return pack_rgba(p[1], p[2], p[3], p[0]);
    case TK_RGBA: return pack_rgba(p[3], p[2], p[1], p[0]);
    case TK_I8: {
        const uint8_t *q;
        if (!t->palette_mem || p[0] >= t->palette_len) return pack_rgba(0, 0, 0, 255);
        q = t->palette_mem + p[0] * 4;
        return pack_rgba(q[2], q[1], q[0], q[3]);
    }
    case TK_Y16:
        v = p[0] | (p[1] << 8);
        l = (v * 255 + 32767) / 65535;
        return pack_rgba(l, l, l, 255);
    default:
        return pack_rgba(0, 0, 0, 255);
    }
}

static uint32_t yuv_rgba(const KTexture *t, const uint8_t *line, unsigned int x)
{
    int c, d, e;
    if (t->kind == TK_YUY2) {
        c = (int)line[x * 2] - 16;
        if (x % 2) { d = (int)line[x * 2 - 1] - 128; e = (int)line[x * 2 + 1] - 128; }
        else       { d = (int)line[x * 2 + 1] - 128; e = (int)line[x * 2 + 3] - 128; }
    } else {
        c = (int)line[x * 2 + 1] - 16;
        if (x % 2) { d = (int)line[x * 2 - 2] - 128; e = (int)line[x * 2 + 0] - 128; }
        else       { d = (int)line[x * 2 + 0] - 128; e = (int)line[x * 2 + 2] - 128; }
    }
    return pack_rgba(clip_byte((298 * c + 409 * e + 128) >> 8),
                     clip_byte((298 * c - 100 * d - 208 * e + 128) >> 8),
                     clip_byte((298 * c + 516 * d + 128) >> 8), 255);
}

static void decode_level(const KTexture *t, unsigned int level, uint32_t *out)
{
    unsigned int w = t->level_w[level], h = t->level_h[level], x, y;
    const uint8_t *base = t->mem + t->level_offset[level];
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            uint32_t px;
            if (t->kind == TK_YUY2 || t->kind == TK_UYVY) {
                px = yuv_rgba(t, base + (size_t)y * t->pitch, x);
            } else if (t->linear) {
                px = texel_rgba(t, base + (size_t)y * t->pitch + (size_t)x * t->bpp);
            } else {
                px = texel_rgba(t, base + (size_t)swz_off(w, h, x, y) * t->bpp);
            }
            out[(size_t)y * w + x] = px;
        }
    }
}

static bool kind_is_dxt(unsigned int kind)
{
    return kind == TK_DXT1 || kind == TK_DXT3 || kind == TK_DXT5;
}

/* Bytes of guest memory the texture occupies. */
static uint32_t texture_span(const KTexture *t)
{
    uint32_t total = 0;
    unsigned int level;
    for (level = 0; level < t->levels && level < KTEX_MAX_LEVELS; level++) {
        unsigned int w = t->level_w[level], h = t->level_h[level];
        if (kind_is_dxt(t->kind)) {
            total = t->level_offset[level] + ((w + 3) / 4) * ((h + 3) / 4) * (t->kind == TK_DXT1 ? 8 : 16);
        } else if (t->linear) {
            total = t->level_offset[level] + t->pitch * h;
        } else {
            total = t->level_offset[level] + w * h * t->bpp;
        }
    }
    return total;
}

/* Reference BC1/2/3 decoder for levels D3D11 refuses (not a multiple of four). */
static void dxt_decode_block(unsigned int kind, const uint8_t *b, uint32_t px[16])
{
    const uint8_t *cb = kind == TK_DXT1 ? b : b + 8;
    uint32_t c0 = cb[0] | (cb[1] << 8), c1 = cb[2] | (cb[3] << 8), idx, pal[4], i;
    uint8_t alpha[16];
    #define EXP(c, r, g, bb) do { r = expand5((c >> 11) & 31); g = expand6((c >> 5) & 63); bb = expand5(c & 31); } while (0)
    uint32_t r0, g0, b0, r1, g1, b1;
    EXP(c0, r0, g0, b0); EXP(c1, r1, g1, b1);
    pal[0] = pack_rgba(r0, g0, b0, 255);
    pal[1] = pack_rgba(r1, g1, b1, 255);
    if (kind == TK_DXT1 && c0 <= c1) {
        pal[2] = pack_rgba((r0 + r1) / 2, (g0 + g1) / 2, (b0 + b1) / 2, 255);
        pal[3] = pack_rgba(0, 0, 0, 0);
    } else {
        pal[2] = pack_rgba((2 * r0 + r1) / 3, (2 * g0 + g1) / 3, (2 * b0 + b1) / 3, 255);
        pal[3] = pack_rgba((r0 + 2 * r1) / 3, (g0 + 2 * g1) / 3, (b0 + 2 * b1) / 3, 255);
    }
    memcpy(&idx, cb + 4, 4);
    if (kind == TK_DXT3) {
        for (i = 0; i < 16; i++) alpha[i] = (uint8_t)expand4((b[i / 2] >> ((i & 1) * 4)) & 15);
    } else if (kind == TK_DXT5) {
        uint64_t bits = 0;
        uint8_t a0 = b[0], a1 = b[1], ap[8];
        memcpy(&bits, b + 2, 6);
        ap[0] = a0; ap[1] = a1;
        if (a0 > a1) { for (i = 1; i < 7; i++) ap[i + 1] = (uint8_t)(((7 - i) * a0 + i * a1) / 7); }
        else { for (i = 1; i < 5; i++) ap[i + 1] = (uint8_t)(((5 - i) * a0 + i * a1) / 5); ap[6] = 0; ap[7] = 255; }
        for (i = 0; i < 16; i++) alpha[i] = ap[(bits >> (3 * i)) & 7];
    } else {
        memset(alpha, 255, sizeof(alpha));
    }
    for (i = 0; i < 16; i++) {
        uint32_t c = pal[(idx >> (2 * i)) & 3];
        px[i] = kind == TK_DXT1 ? c : (c & 0x00FFFFFFu) | ((uint32_t)alpha[i] << 24);
    }
    #undef EXP
}

static void decode_dxt_level(const KTexture *t, unsigned int level, uint32_t *out)
{
    unsigned int w = t->level_w[level], h = t->level_h[level], bw = (w + 3) / 4, bh = (h + 3) / 4;
    unsigned int block = t->kind == TK_DXT1 ? 8 : 16, bx, by, i;
    const uint8_t *base = t->mem + t->level_offset[level];
    for (by = 0; by < bh; by++) {
        for (bx = 0; bx < bw; bx++) {
            uint32_t px[16];
            dxt_decode_block(t->kind, base + ((size_t)by * bw + bx) * block, px);
            for (i = 0; i < 16; i++) {
                unsigned int x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x < w && y < h) out[(size_t)y * w + x] = px[i];
            }
        }
    }
}

static void tex_free(KGpuTex *e)
{
    kg_cache_invalidate();
    if (e->rt_index >= 0 && e->used) krt_tex_remove(e->rt_index);
    if (e->rt_h) CloseHandle(e->rt_h);
    SAFE_RELEASE(e->srv);
    SAFE_RELEASE(e->tex);
    G.tex_bytes -= e->bytes;
    memset(e, 0, sizeof(*e));
}

static void tex_signature(const KTexture *t, uint32_t span, uint64_t sig[2])
{
    sig_bytes(t->mem, span, sig);
    if (t->kind == TK_I8 && t->palette_mem) {
        uint64_t p[2];
        sig_bytes(t->palette_mem, t->palette_len * 4, p);
        sig[0] ^= p[0] * 0x9E3779B97F4A7C15ull;
        sig[1] ^= p[1];
    }
}

/* Create or refresh the cached GPU copy of a texture; the SRV to bind. */
static ID3D11ShaderResourceView *tex_get(const KTexture *t)
{
    uint32_t span, h;
    unsigned int slot, probes, level;
    KGpuTex *e = NULL;
    uint64_t sig[2];
    D3D11_TEXTURE2D_DESC td;
    D3D11_SUBRESOURCE_DATA init[KTEX_MAX_LEVELS];
    uint32_t *buffers[KTEX_MAX_LEVELS];
    bool dxt = kind_is_dxt(t->kind), native_dxt = false;
    unsigned int levels = t->levels ? t->levels : 1;

    if (!t->valid || !t->mem || t->cubemap || t->width == 0 || t->height == 0) return NULL;
    span = texture_span(t);
    h = (t->base * 2654435761u) ^ (t->color_format * 40503u) ^ (t->width * 977u) ^ (t->height * 131u);
    for (probes = 0, slot = h % KG_TEX_SLOTS; probes < KG_TEX_SLOTS; probes++, slot = (slot + 1) % KG_TEX_SLOTS) {
        KGpuTex *q = &G.tex[slot];
        if (!q->used) { if (!e) e = q; break; }
        if (q->base == t->base && q->fmt == t->color_format && q->w == t->width &&
            q->h == t->height && q->pitch == t->pitch && q->levels == levels &&
            q->palette == t->palette) {
            e = q;
            break;
        }
    }
    if (!e) return NULL;
    G.last_tex = e;
    if (e->used) {
        if (e->epoch == G.epoch) { e->clock = G.clock; G.tex_hits++; return e->srv; }
        G.tex_validations++;
        tex_signature(t, span, sig);
        e->epoch = G.epoch;
        e->clock = G.clock;
        if (sig[0] == e->sig[0] && sig[1] == e->sig[1] && e->span == span) {
            G.tex_hits++;
            return e->srv;
        }
        /* content changed: rebuild the GPU copy */
        tex_free(e);
    } else {
        tex_signature(t, span, sig);
    }

    memset(&td, 0, sizeof(td));
    td.Width = t->width; td.Height = t->height; td.MipLevels = levels; td.ArraySize = 1;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (krt_available()) {
        /* rays shade what they hit with the game's textures: shared with the D3D12 side */
        td.Usage = D3D11_USAGE_DEFAULT;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    }
    memset(buffers, 0, sizeof(buffers));
    if (dxt && (t->width % 4) == 0 && (t->height % 4) == 0) {
        native_dxt = true;
        td.Format = t->kind == TK_DXT1 ? DXGI_FORMAT_BC1_UNORM : t->kind == TK_DXT3 ? DXGI_FORMAT_BC2_UNORM
                                                                              : DXGI_FORMAT_BC3_UNORM;
        for (level = 0; level < levels; level++) {
            unsigned int bw = (t->level_w[level] + 3) / 4;
            init[level].pSysMem = t->mem + t->level_offset[level];
            init[level].SysMemPitch = bw * (t->kind == TK_DXT1 ? 8 : 16);
            init[level].SysMemSlicePitch = 0;
        }
    } else {
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        for (level = 0; level < levels; level++) {
            size_t n = (size_t)t->level_w[level] * t->level_h[level];
            buffers[level] = (uint32_t *)malloc(n * 4);
            if (!buffers[level]) goto fail;
            if (dxt) decode_dxt_level(t, level, buffers[level]);
            else decode_level(t, level, buffers[level]);
            {
                static int texlog = -1, said;
                if (texlog < 0) texlog = getenv("RECOMP_KGPU_TEXLOG") != NULL;
                if (texlog && G.frames > 14000 && said++ < 400)
                    fprintf(stderr, "[KTEX] upload base 0x%08X kind %u fmt 0x%02X %ux%u level %u/%u %s pitch %u (%s)\n", t->base, t->kind, t->color_format,
                            t->level_w[level], t->level_h[level], level, levels, t->linear ? "linear" : "swizzled", t->pitch, dxt ? "dxt" : "plain");
            }
            init[level].pSysMem = buffers[level];
            init[level].SysMemPitch = t->level_w[level] * 4;
            init[level].SysMemSlicePitch = 0;
        }
    }
    {
        ID3D11Texture2D *tex = NULL;
        ID3D11ShaderResourceView *srv = NULL;
        HRESULT hr = ID3D11Device_CreateTexture2D(G.dev, &td, init, &tex);
        if (FAILED(hr) && native_dxt) {
            /* fall back to a CPU decode */
            native_dxt = false;
            td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            for (level = 0; level < levels; level++) {
                size_t n = (size_t)t->level_w[level] * t->level_h[level];
                buffers[level] = (uint32_t *)malloc(n * 4);
                if (!buffers[level]) goto fail;
                decode_dxt_level(t, level, buffers[level]);
                init[level].pSysMem = buffers[level];
                init[level].SysMemPitch = t->level_w[level] * 4;
            }
            hr = ID3D11Device_CreateTexture2D(G.dev, &td, init, &tex);
        }
        if (FAILED(hr) || FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)tex, NULL, &srv))) {
            static int said;
            if (said++ < 8) fprintf(stderr, "[KGPU] texture 0x%08X fmt 0x%02X %ux%u create failed 0x%08lX\n",
                                    t->base, t->color_format, t->width, t->height, (unsigned long)hr);
            SAFE_RELEASE(tex); SAFE_RELEASE(srv);
            goto fail;
        }
        e->used = true;
        e->base = t->base; e->fmt = t->color_format; e->w = t->width; e->h = t->height;
        e->pitch = t->pitch; e->levels = levels; e->palette = t->palette; e->palette_len = t->palette_len;
        e->span = span; e->sig[0] = sig[0]; e->sig[1] = sig[1];
        e->epoch = G.epoch; e->clock = G.clock;
        e->tex = tex; e->srv = srv;
        e->rt_h = NULL; e->rt_index = -1;
        if (td.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE) {
            IDXGIResource1 *r1 = NULL;
            if (SUCCEEDED(ID3D11Texture2D_QueryInterface(tex, &IID_IDXGIResource1, (void **)&r1))) {
                if (FAILED(IDXGIResource1_CreateSharedHandle(r1, NULL, DXGI_SHARED_RESOURCE_READ, NULL, &e->rt_h))) e->rt_h = NULL;
                IDXGIResource1_Release(r1);
            }
        }
        {
            size_t bytes = 0;
            for (level = 0; level < levels; level++) {
                size_t texels = (size_t)t->level_w[level] * t->level_h[level];
                bytes += native_dxt ? (t->kind == TK_DXT1 ? texels / 2 : texels) : texels * 4;
            }
            e->bytes = bytes + 64;
        }
        G.tex_bytes += e->bytes;
        G.tex_uploads++;
    }
    for (level = 0; level < levels; level++) free(buffers[level]);
    return e->srv;
fail:
    for (level = 0; level < levels; level++) free(buffers[level]);
    return NULL;
}

/* ---- ring buffers ---------------------------------------------------------------------- */

static void *ring_map(ID3D11Buffer *buf, size_t *pos, size_t cap, size_t count,
                      size_t elem, size_t *at, D3D11_MAPPED_SUBRESOURCE *ms)
{
    D3D11_MAP mode = D3D11_MAP_WRITE_NO_OVERWRITE;
    if (*pos + count > cap) { *pos = 0; mode = D3D11_MAP_WRITE_DISCARD; }
    else if (*pos == 0) mode = D3D11_MAP_WRITE_DISCARD;
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)buf, 0, mode, 0, ms))) return NULL;
    *at = *pos;
    *pos += count;
    return (uint8_t *)ms->pData + *at * elem;
}

#include "kelvin_gpu_vsh.inc"
#include "kelvin_gpu_pss.inc"

/* ---- draw ---------------------------------------------------------------------------------- */

static uint32_t raw_in(const KCombinerInput *in) { return (uint32_t)(in->reg | in->chan | in->mod); }
static uint32_t raw_out(const KCombinerOutput *o)
{
    return (uint32_t)(o->cd | (o->ab << 4) | (o->muxsum << 8) | (o->flags << 12));
}

static void fill_ps_const(PSConst *c, const KPixelState *ps, unsigned int alpha_mode)
{
    int i, k;
    memset(c, 0, sizeof(*c));
    for (i = 0; i < 8; i++) {
        const KCombinerStage *s = &ps->stage[i];
        for (k = 0; k < 4; k++) {
            c->stage[i].rgb_in[k] = raw_in(&s->rgb[k]);
            c->stage[i].alpha_in[k] = raw_in(&s->alpha[k]);
        }
        c->stage[i].outs[0] = raw_out(&s->rgb_out);
        c->stage[i].outs[1] = raw_out(&s->alpha_out);
        memcpy(c->stage[i].c0, s->c0, 16);
        memcpy(c->stage[i].c1, s->c1, 16);
    }
    for (k = 0; k < 4; k++) c->fin_abcd[k] = raw_in(&ps->fin[k]);
    c->fin_efg[0] = raw_in(&ps->fin[4]);
    c->fin_efg[1] = raw_in(&ps->fin[5]);
    c->fin_efg[2] = raw_in(&ps->fin[6]);
    c->fin_efg[3] = (ps->final_enabled ? 1u : 0u) | (ps->clamp_sum ? 2u : 0u) | (ps->inv_v1 ? 4u : 0u) |
                    (ps->inv_r0 ? 8u : 0u) | ((ps->flags & PS_COMBINERCOUNT_MUX_MSB) ? 0x10u : 0u) |
                    ((ps->flags & PS_COMBINERCOUNT_UNIQUE_C0) ? 0x20u : 0u) |
                    ((ps->flags & PS_COMBINERCOUNT_UNIQUE_C1) ? 0x40u : 0u);
    memcpy(c->final_c0, ps->final_c0, 16);
    memcpy(c->final_c1, ps->final_c1, 16);
    c->fogcolor[0] = ps->fog_color[0]; c->fogcolor[1] = ps->fog_color[1]; c->fogcolor[2] = ps->fog_color[2];
    c->gen[0] = (uint32_t)ps->num_stages;
    c->gen[1] = ps->alpha_test ? (uint32_t)ps->alpha_func : 7u;
    c->gen[2] = (uint32_t)ps->alpha_ref;
    c->gen[3] = ps->smooth ? 0u : 1u;          /* flat shading */
    for (i = 0; i < 4; i++) {
        const KTexture *t = &ps->tex[i];
        c->texmode[i] = (uint32_t)ps->tex_mode[i];
        c->texflags[i] = (t->valid ? 1u : 0u) | (t->linear ? 2u : 0u) | (t->alphakill ? 4u : 0u) |
                         ((uint32_t)t->colorkey_mode << 4);
        c->texrcp[i][0] = t->width ? 1.0f / (float)t->width : 1.0f;
        c->texrcp[i][1] = t->height ? 1.0f / (float)t->height : 1.0f;
        c->ckey[i] = ps->color_key[i];
        c->ckeymask[i] = ps->color_key_mask[i];
    }
    c->ofs[0] = alpha_mode;
}

static ID3D11Texture2D *feedback_copy(const KGpuSurf *s)
{
    D3D11_TEXTURE2D_DESC td;
    if (G.fb_copy && (G.fb_w != s->hw || G.fb_h != s->hh || G.fb_fmt != s->dxfmt)) SAFE_RELEASE(G.fb_copy);
    if (!G.fb_copy) {
        memset(&td, 0, sizeof(td));
        td.Width = s->hw; td.Height = s->hh; td.MipLevels = td.ArraySize = 1;
        td.Format = s->dxfmt; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &G.fb_copy))) return NULL;
        G.fb_w = s->hw; G.fb_h = s->hh; G.fb_fmt = s->dxfmt;
    }
    return G.fb_copy;
}

static uint64_t s_cmp_frames;       /* flips seen by compare mode (G.frames only counts the GPU's own presented frames) */
static bool readback(const KGpuSurf *cs, ID3D11Texture2D **stage_out, D3D11_MAPPED_SUBRESOURCE *ms);
static void write_bmp(const char *path, const uint8_t *bgra, unsigned int w, unsigned int h, unsigned int pitch);
#include "kelvin_gpu_post.inc"
#include "kelvin_gpu_ssr.inc"
#include "kelvin_gpu_draw.inc"
#include "kelvin_gpu_rt.inc"

/* ---- clears ------------------------------------------------------------------------------------- */

static void kgpu_clear_now(const KelvinTarget *t, bool color, bool depth, bool stencil,
                           uint32_t color_value, uint32_t zs_value, unsigned int channel_mask,
                           int x0, int y0, int x1, int y1, unsigned int color_format)
{
    KGpuSurf *cs = NULL, *zs = NULL;
    double sx = 1.0, sy = 1.0;
    unsigned int zfmt = 0;
    if (!G.ok) return;
    kt_begin();
    kg_bound_reset();
    if (color && t->color) cs = surf_get(t, false, color_format);
    if ((depth || stencil) && t->zeta) {
        zfmt = t->zeta_bpp == 2 ? NV097_SET_SURFACE_FORMAT_ZETA_Z16 : NV097_SET_SURFACE_FORMAT_ZETA_Z24S8;
        zs = surf_get(t, true, zfmt);
    }
    if (cs) { sx = (double)cs->hw / cs->gw; sy = (double)cs->hh / cs->gh; }
    else if (zs) { sx = (double)zs->hw / zs->gw; sy = (double)zs->hh / zs->gh; }
    if (s_dl.mode != 0 && kg_dl_on())
        fprintf(stderr, "[KDL] f%llu #%u clear colour %d depth %d stencil %d rect %d,%d-%d,%d target %ux%u\n", (unsigned long long)s_dl.frame, s_dl.ord,
                (int)color, (int)depth, (int)stencil, x0, y0, x1, y1, t->width, t->height);
    if (cs) { cs->ver++; cs->wseq = ++s_wseq; }
    if (zs) { zs->real = true; zs->from_color = false; zs->ver++; zs->wseq = ++s_wseq; }
    /* the rectangle arrives in guest-stored pixels */
    x0 = (int)floor(x0 * sx); y0 = (int)floor(y0 * sy);
    x1 = (int)ceil(x1 * sx); y1 = (int)ceil(y1 * sy);
    if (cs) {
        bool full = x0 <= 0 && y0 <= 0 && x1 >= (int)cs->hw && y1 >= (int)cs->hh;
        float c[4];
        c[0] = ((color_value >> 16) & 0xFF) / 255.0f;
        c[1] = ((color_value >> 8) & 0xFF) / 255.0f;
        c[2] = (color_value & 0xFF) / 255.0f;
        c[3] = ((color_value >> 24) & 0xFF) / 255.0f;
        if (color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8) c[3] = 0.0f;
        if (color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8) c[3] = 1.0f;
        if (full && channel_mask == 0xF) {
            if (s_kp_mode != 0) { KpInfo ki; memset(&ki, 0, sizeof(ki)); ki.kind = 1; ki.psh = 0xC1; ki.tw = cs->hw; ki.th = cs->hh; kp_mark(&ki); }
            ID3D11DeviceContext_ClearRenderTargetView(G.ctx, cs->rtv, c);
            if (s_kp_mode != 0) kp_done();
        } else if (x1 > x0 && y1 > y0) {
            /* masked / partial: a rectangle drawn with a colour write mask */
            D3D11_BLEND_DESC bd;
            ID3D11BlendState *bs = NULL;
            D3D11_RASTERIZER_DESC rd;
            ID3D11RasterizerState *rs = NULL;
            D3D11_DEPTH_STENCIL_DESC dd;
            ID3D11DepthStencilState *ds = NULL;
            D3D11_MAPPED_SUBRESOURCE ms;
            D3D11_VIEWPORT vp;
            D3D11_RECT sc;
            memset(&bd, 0, sizeof(bd));
            /* channel_mask bits: B=1, G=2, R=4, A=8 */
            bd.RenderTarget[0].RenderTargetWriteMask = (UINT8)(((channel_mask & 4) ? 1 : 0) |
                ((channel_mask & 2) ? 2 : 0) | ((channel_mask & 1) ? 4 : 0) | ((channel_mask & 8) ? 8 : 0));
            memset(&rd, 0, sizeof(rd));
            rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.ScissorEnable = TRUE; rd.DepthClipEnable = TRUE;
            memset(&dd, 0, sizeof(dd));
            ID3D11Device_CreateBlendState(G.dev, &bd, &bs);
            ID3D11Device_CreateRasterizerState(G.dev, &rd, &rs);
            ID3D11Device_CreateDepthStencilState(G.dev, &dd, &ds);
            if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.cb_clear, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
                memcpy(ms.pData, c, 16);
                ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.cb_clear, 0);
            }
            ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 1, &cs->rtv, NULL);
            ID3D11DeviceContext_OMSetBlendState(G.ctx, bs, NULL, 0xFFFFFFFFu);
            ID3D11DeviceContext_OMSetDepthStencilState(G.ctx, ds, 0);
            ID3D11DeviceContext_RSSetState(G.ctx, rs);
            vp.TopLeftX = vp.TopLeftY = 0; vp.Width = (FLOAT)cs->hw; vp.Height = (FLOAT)cs->hh;
            vp.MinDepth = 0; vp.MaxDepth = 1;
            ID3D11DeviceContext_RSSetViewports(G.ctx, 1, &vp);
            sc.left = x0 < 0 ? 0 : x0; sc.top = y0 < 0 ? 0 : y0;
            sc.right = x1 > (int)cs->hw ? (LONG)cs->hw : x1; sc.bottom = y1 > (int)cs->hh ? (LONG)cs->hh : y1;
            ID3D11DeviceContext_RSSetScissorRects(G.ctx, 1, &sc);
            ID3D11DeviceContext_IASetInputLayout(G.ctx, NULL);
            ID3D11DeviceContext_IASetPrimitiveTopology(G.ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D11DeviceContext_VSSetShader(G.ctx, G.clear_vs, NULL, 0);
            ID3D11DeviceContext_PSSetShader(G.ctx, G.clear_ps, NULL, 0);
            ID3D11DeviceContext_PSSetConstantBuffers(G.ctx, 0, 1, &G.cb_clear);
            ID3D11DeviceContext_Draw(G.ctx, 3, 0);
            SAFE_RELEASE(bs); SAFE_RELEASE(rs); SAFE_RELEASE(ds);
        }
    }
    if (zs) {
        bool full = x0 <= 0 && y0 <= 0 && x1 >= (int)zs->hw && y1 >= (int)zs->hh;
        float zv = zfmt == NV097_SET_SURFACE_FORMAT_ZETA_Z16 ? (float)(zs_value & 0xFFFF) / 65535.0f
                                                           : (float)(zs_value >> 8) / 16777215.0f;
        if (full) {
            UINT fl = 0;
            if (depth) fl |= D3D11_CLEAR_DEPTH;
            if (stencil) fl |= D3D11_CLEAR_STENCIL;
            if (s_kp_mode != 0) { KpInfo ki; memset(&ki, 0, sizeof(ki)); ki.kind = 1; ki.psh = 0xC2; ki.tw = zs->hw; ki.th = zs->hh; kp_mark(&ki); }
            ID3D11DeviceContext_ClearDepthStencilView(G.ctx, zs->dsv, fl, zv, (UINT8)(zs_value & 0xFF));
            if (s_kp_mode != 0) kp_done();
        } else if (depth && x1 > x0 && y1 > y0) {
            D3D11_BLEND_DESC bd;
            ID3D11BlendState *bs = NULL;
            D3D11_RASTERIZER_DESC rd;
            ID3D11RasterizerState *rs = NULL;
            D3D11_DEPTH_STENCIL_DESC dd;
            ID3D11DepthStencilState *ds = NULL;
            D3D11_MAPPED_SUBRESOURCE ms;
            D3D11_VIEWPORT vp;
            D3D11_RECT sc;
            float c[8];
            memset(&bd, 0, sizeof(bd));
            memset(&rd, 0, sizeof(rd));
            rd.FillMode = D3D11_FILL_SOLID; rd.CullMode = D3D11_CULL_NONE; rd.ScissorEnable = TRUE; rd.DepthClipEnable = FALSE;
            memset(&dd, 0, sizeof(dd));
            dd.DepthEnable = TRUE; dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
            ID3D11Device_CreateBlendState(G.dev, &bd, &bs);
            ID3D11Device_CreateRasterizerState(G.dev, &rd, &rs);
            ID3D11Device_CreateDepthStencilState(G.dev, &dd, &ds);
            memset(c, 0, sizeof(c)); c[4] = zv;
            if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.cb_clear, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
                memcpy(ms.pData, c, 32);
                ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.cb_clear, 0);
            }
            ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 0, NULL, zs->dsv);
            ID3D11DeviceContext_OMSetBlendState(G.ctx, bs, NULL, 0xFFFFFFFFu);
            ID3D11DeviceContext_OMSetDepthStencilState(G.ctx, ds, 0);
            ID3D11DeviceContext_RSSetState(G.ctx, rs);
            vp.TopLeftX = vp.TopLeftY = 0; vp.Width = (FLOAT)zs->hw; vp.Height = (FLOAT)zs->hh;
            vp.MinDepth = 0; vp.MaxDepth = 1;
            ID3D11DeviceContext_RSSetViewports(G.ctx, 1, &vp);
            sc.left = x0 < 0 ? 0 : x0; sc.top = y0 < 0 ? 0 : y0;
            sc.right = x1 > (int)zs->hw ? (LONG)zs->hw : x1; sc.bottom = y1 > (int)zs->hh ? (LONG)zs->hh : y1;
            ID3D11DeviceContext_RSSetScissorRects(G.ctx, 1, &sc);
            ID3D11DeviceContext_IASetInputLayout(G.ctx, NULL);
            ID3D11DeviceContext_IASetPrimitiveTopology(G.ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D11DeviceContext_VSSetShader(G.ctx, G.clear_vs, NULL, 0);
            ID3D11DeviceContext_PSSetShader(G.ctx, G.clear_depth_ps, NULL, 0);
            ID3D11DeviceContext_PSSetConstantBuffers(G.ctx, 0, 1, &G.cb_clear);
            ID3D11DeviceContext_Draw(G.ctx, 3, 0);
            SAFE_RELEASE(bs); SAFE_RELEASE(rs); SAFE_RELEASE(ds);
        }
    }
}

/* ---- presentation (readback path) ------------------------------------------------------------------ */

static void write_bmp(const char *path, const uint8_t *bgra, unsigned int w, unsigned int h,
                      unsigned int pitch)
{
    FILE *f = fopen(path, "wb");
    uint32_t row_bytes = w * 3, pad = (4 - (row_bytes & 3)) & 3, filesz, y, x;
    uint8_t hdr[54], *row;
    if (!f) return;
    row = (uint8_t *)malloc(row_bytes + pad);
    if (!row) { fclose(f); return; }
    memset(row + row_bytes, 0, pad);
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
        const uint8_t *src = bgra + (size_t)y * pitch;
        for (x = 0; x < w; x++) { row[x * 3] = src[x * 4]; row[x * 3 + 1] = src[x * 4 + 1]; row[x * 3 + 2] = src[x * 4 + 2]; }
        fwrite(row, 1, row_bytes + pad, f);
    }
    free(row);
    fclose(f);
}

static uint32_t crc_frame(const uint8_t *p, unsigned int w, unsigned int h, unsigned int pitch)
{
    uint64_t s[2];
    uint32_t acc = 0;
    unsigned int y;
    for (y = 0; y < h; y++) {
        sig_bytes(p + (size_t)y * pitch, (size_t)w * 4, s);
        acc = (acc * 16777619u) ^ (uint32_t)s[0] ^ (uint32_t)(s[0] >> 32);
    }
    return acc;
}

static void publish_staged(unsigned int slot, unsigned int format)
{
    D3D11_MAPPED_SUBRESOURCE ms;
    uint64_t n = G.stage_frame[slot];
    bool logged = n <= 8 || (n % 60) == 0;
    bool dumped = G.dump && *G.dump && (n % G.dump_every) == 0;
    if (!G.stage[slot] || !n) return;
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)G.stage[slot], 0, D3D11_MAP_READ, 0, &ms))) return;
    {
        uint32_t h = logged || dumped ? crc_frame((const uint8_t *)ms.pData, G.stage_w, G.stage_h, ms.RowPitch) : 0;
        if (logged) {
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            fprintf(stderr, "[KELVIN] frame %llu: surface 0x%08X %ux%u gpu hash %08X (%llu distinct so far) [qpc %lld freq %lld]\n",
                    (unsigned long long)n, G.stage_va[slot], G.stage_w, G.stage_h, h,
                    (unsigned long long)n, (long long)now.QuadPart, (long long)G.qpf);
            fflush(stderr);
        }
        if (dumped) {
            char path[512];
            snprintf(path, sizeof(path), "%s%05llu_%08X.bmp", G.dump, (unsigned long long)n, h);
            write_bmp(path, (const uint8_t *)ms.pData, G.stage_w, G.stage_h, ms.RowPitch);
        }
        xbox_FramebufferPublishAspect(ms.pData, (size_t)ms.RowPitch * G.stage_h, G.stage_va[slot],
                                      G.stage_w, G.stage_h, ms.RowPitch, format,
                                      G.stage_aspect_w[slot], G.stage_aspect_h[slot]);
    }
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)G.stage[slot], 0);
    G.stage_frame[slot] = 0;
}

static int s_compare = -1;
bool kgpu_compare_enabled(void)
{
    if (s_compare < 0) {
        const char *e = getenv("RECOMP_KGPU_COMPARE");
        s_compare = e && *e ? (int)strtol(e, NULL, 10) : 0;
        if (s_compare < 0) s_compare = 0;
    }
    return kgpu_enabled() && s_compare > 0;
}

/* Readback of one colour surface, synchronously. */
static bool readback(const KGpuSurf *cs, ID3D11Texture2D **stage_out, D3D11_MAPPED_SUBRESOURCE *ms)
{
    D3D11_TEXTURE2D_DESC td;
    ID3D11Texture2D *st = NULL;
    memset(&td, 0, sizeof(td));
    td.Width = cs->hw; td.Height = cs->hh; td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &st))) return false;
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)st, (ID3D11Resource *)cs->tex);
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)st, 0, D3D11_MAP_READ, 0, ms))) {
        IUnknown_Release((IUnknown *)st);
        return false;
    }
    *stage_out = st;
    return true;
}

void kgpu_compare_frame(const KelvinTarget *t, unsigned int color_format)
{
    static uint64_t n, last;
    KGpuSurf *cs;
    ID3D11Texture2D *st;
    D3D11_MAPPED_SUBRESOURCE ms;
    unsigned int x, y;
    uint64_t sum = 0, gt8 = 0, gt32 = 0, pixels;
    unsigned int maxd = 0;
    if (!G.ok || !t->color || t->color_bpp != 4) return;
    n++;
    s_cmp_frames = n;
    if (s_compare <= 0 || n % (uint64_t)s_compare) return;
    cs = surf_get(t, false, color_format);
    if (!cs || cs->hw != t->width || cs->hh != t->height) return;
    if (!readback(cs, &st, &ms)) return;
    pixels = (uint64_t)t->width * t->height;
    for (y = 0; y < t->height; y++) {
        const uint8_t *g = (const uint8_t *)ms.pData + (size_t)y * ms.RowPitch;
        const uint8_t *c = t->color + (size_t)y * t->color_pitch;
        for (x = 0; x < t->width; x++) {
            unsigned int d0 = (unsigned int)abs((int)g[x * 4] - (int)c[x * 4]);
            unsigned int d1 = (unsigned int)abs((int)g[x * 4 + 1] - (int)c[x * 4 + 1]);
            unsigned int d2 = (unsigned int)abs((int)g[x * 4 + 2] - (int)c[x * 4 + 2]);
            unsigned int d = d0 > d1 ? d0 : d1;
            if (d2 > d) d = d2;
            sum += d0 + d1 + d2;
            if (d > maxd) maxd = d;
            if (d > 8) gt8++;
            if (d > 32) gt32++;
        }
    }
    fprintf(stderr, "[KCMP] flip %llu: mean abs diff %.3f, max %u, >8: %.3f%%, >32: %.3f%%\n",
            (unsigned long long)n, (double)sum / (3.0 * (double)pixels), maxd,
            100.0 * (double)gt8 / (double)pixels, 100.0 * (double)gt32 / (double)pixels);
    if (G.dump && *G.dump && ((double)sum / (3.0 * (double)pixels) > 0.5 || (n / (uint64_t)s_compare) % 20 == 0)) {
        char path[512];
        uint8_t *buf = (uint8_t *)malloc((size_t)t->width * t->height * 4);
        if (buf) {
            for (y = 0; y < t->height; y++)
                memcpy(buf + (size_t)y * t->width * 4, t->color + (size_t)y * t->color_pitch, (size_t)t->width * 4);
            snprintf(path, sizeof(path), "%scmp%05llu_cpu.bmp", G.dump, (unsigned long long)n);
            write_bmp(path, buf, t->width, t->height, t->width * 4);
            snprintf(path, sizeof(path), "%scmp%05llu_gpu.bmp", G.dump, (unsigned long long)n);
            write_bmp(path, (const uint8_t *)ms.pData, t->width, t->height, ms.RowPitch);
            for (y = 0; y < t->height; y++) {
                const uint8_t *g = (const uint8_t *)ms.pData + (size_t)y * ms.RowPitch;
                for (x = 0; x < t->width * 4; x++) {
                    int d = abs((int)g[x] - (int)buf[(size_t)y * t->width * 4 + x]) * 4;
                    buf[(size_t)y * t->width * 4 + x] = (uint8_t)(d > 255 ? 255 : d);
                }
            }
            snprintf(path, sizeof(path), "%scmp%05llu_diff.bmp", G.dump, (unsigned long long)n);
            write_bmp(path, buf, t->width, t->height, t->width * 4);
            free(buf);
        }
    }
    last = n;
    (void)last;
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)st, 0);
    IUnknown_Release((IUnknown *)st);
}

/* ---- direct presentation through shared textures ----------------------------------------------------- */

#define KG_RING 4
static ID3D11Device5 *s_dev5;
static ID3D11DeviceContext4 *s_ctx4;
static ID3D11Fence *s_frame_fence, *s_done_fence;
static HANDLE s_frame_fence_h, s_done_fence_h;
static ID3D11Texture2D *s_ring[KG_RING];
static HANDLE s_ring_h[KG_RING];
static uint64_t s_ring_value[KG_RING], s_fence_next;
static unsigned int s_ring_w, s_ring_h_dim, s_ring_next;
static int s_shared_mode = -1;      /* -1 untried, 0 readback, 1 shared textures */

static void ring_free(void)
{
    unsigned int i;
    for (i = 0; i < KG_RING; i++) {
        SAFE_RELEASE(s_ring[i]);
        if (s_ring_h[i]) { CloseHandle(s_ring_h[i]); s_ring_h[i] = NULL; }
        s_ring_value[i] = 0;
    }
    s_ring_w = s_ring_h_dim = 0;
}

static bool ring_setup(unsigned int w, unsigned int h)
{
    unsigned int i;
    D3D11_TEXTURE2D_DESC td;
    if (!s_dev5) {
        if (FAILED(ID3D11Device_QueryInterface(G.dev, &IID_ID3D11Device5, (void **)&s_dev5)) ||
            FAILED(ID3D11DeviceContext_QueryInterface(G.ctx, &IID_ID3D11DeviceContext4, (void **)&s_ctx4))) {
            SAFE_RELEASE(s_dev5); SAFE_RELEASE(s_ctx4);
            return false;
        }
    }
    if (!s_frame_fence) {
        if (FAILED(ID3D11Device5_CreateFence(s_dev5, 0, D3D11_FENCE_FLAG_SHARED, &IID_ID3D11Fence, (void **)&s_frame_fence)) ||
            FAILED(ID3D11Device5_CreateFence(s_dev5, 0, D3D11_FENCE_FLAG_SHARED, &IID_ID3D11Fence, (void **)&s_done_fence)) ||
            FAILED(ID3D11Fence_CreateSharedHandle(s_frame_fence, NULL, GENERIC_ALL, NULL, &s_frame_fence_h)) ||
            FAILED(ID3D11Fence_CreateSharedHandle(s_done_fence, NULL, GENERIC_ALL, NULL, &s_done_fence_h))) {
            return false;
        }
        s_fence_next = 0;
    }
    ring_free();
    memset(&td, 0, sizeof(td));
    td.Width = w; td.Height = h; td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    for (i = 0; i < KG_RING; i++) {
        IDXGIResource1 *r1 = NULL;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_ring[i]))) { ring_free(); return false; }
        if (FAILED(ID3D11Texture2D_QueryInterface(s_ring[i], &IID_IDXGIResource1, (void **)&r1)) ||
            FAILED(IDXGIResource1_CreateSharedHandle(r1, NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                     NULL, &s_ring_h[i]))) {
            if (r1) IDXGIResource1_Release(r1);
            ring_free();
            return false;
        }
        IDXGIResource1_Release(r1);
    }
    s_ring_w = w; s_ring_h_dim = h; s_ring_next = 0;
    if (!xbox_FramebufferGpuRegister((void *const *)s_ring_h, KG_RING, s_frame_fence_h, s_done_fence_h, w, h)) {
        ring_free();
        return false;
    }
    fprintf(stderr, "[KGPU] direct presentation: %u shared %ux%u textures\n", KG_RING, w, h);
    return true;
}

static void flip_readback(KGpuSurf *cs, uint32_t aspect_w, uint32_t aspect_h);

/* A finished frame to the presenter without touching the CPU. */
static bool flip_shared(KGpuSurf *cs, uint32_t aspect_w, uint32_t aspect_h)
{
    unsigned int k, slot = 0;
    uint64_t done;
    bool found = false;
    ID3D11Texture2D *src;
    unsigned int sw, sh;
    src = kg_post_flip_source(cs, &sw, &sh);
    if (s_ring_w != sw || s_ring_h_dim != sh) {
        if (!ring_setup(sw, sh)) return false;
    }
    done = ID3D11Fence_GetCompletedValue(s_done_fence);
    for (k = 0; k < KG_RING; k++) {
        slot = (s_ring_next + k) % KG_RING;
        if (s_ring_value[slot] <= done) { found = true; break; }
    }
    if (!found) return true;             /* the presenter is behind: this frame is dropped */
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_ring[slot], (ID3D11Resource *)src);
    ID3D11DeviceContext4_Signal(s_ctx4, s_frame_fence, ++s_fence_next);
    s_ring_value[slot] = s_fence_next;
    ID3D11DeviceContext_Flush(G.ctx);
    kgq_publish_frame(slot, s_fence_next, cs->va, aspect_w, aspect_h);
    s_ring_next = (slot + 1) % KG_RING;
    return true;
}

static void kgpu_flip_inner(const KelvinTarget *t, unsigned int color_format, uint32_t aspect_w, uint32_t aspect_h)
{
    KGpuSurf *cs;
    long long prof_t = kg_tick();
    uint64_t n;
    if (!G.ok) return;
    cs = surf_get(t, false, color_format);
    if (!cs || cs->bpp != 4) return;
    if (s_shared_mode < 0) {
        const char *rb = getenv("RECOMP_KGPU_READBACK");
        s_shared_mode = !(rb && *rb && *rb != '0') && xbox_FramebufferGpuWanted() ? 1 : 0;
        fprintf(stderr, "[KGPU] presentation: %s\n", s_shared_mode ? "shared textures" : "CPU readback");
    }
    n = ++G.frames;
    kt_end(n);
    kt_poll();
    if ((n % 120) == 0 && G.dev) {
        /* A removed or reset device (driver update, TDR) leaves every later call failing quietly; say so once. */
        static bool lost_said;
        HRESULT removed = ID3D11Device_GetDeviceRemovedReason(G.dev);
        if (removed != S_OK && !lost_said) {
            lost_said = true;
            fprintf(stderr, "[KGPU] the GPU device was removed (0x%08lX): the picture will stop updating; "
                            "restart the game, or turn off the GPU renderer in the launcher\n", (unsigned long)removed);
            fflush(stderr);
        }
    }
    {
        long long now = qpc_now();
        if (s_flip_prev_qpc) {
            double ms = (double)(now - s_flip_prev_qpc) * 1000.0 / (double)G.qpf;
            s_hist[ms <= 18.0 ? 0 : ms <= 20.0 ? 1 : ms <= 25.0 ? 2 : ms <= 34.0 ? 3 : ms <= 70.0 ? 4 : 5]++;
            if (ms > s_hist_max_ms) s_hist_max_ms = ms;
            s_fine[ms >= 19.99 ? 39 : (unsigned int)(ms * 2.0)]++;
        }
        s_flip_prev_qpc = now;
        s_flip_q[n & 1023] = now;
        s_flip_count = (LONG64)n;
    }
    xbox_FramebufferWindowStart();
    if (s_shared_mode && xbox_FramebufferGpuFailed()) {
        fprintf(stderr, "[KGPU] the presenter could not open the shared textures; using CPU readback\n");
        s_shared_mode = 0;
    }
    if (s_shared_mode) {
        bool logged = n <= 8 || (n % 60) == 0;
        bool dumped = G.dump && *G.dump && (n % G.dump_every) == 0;
        if (logged || dumped) {
            D3D11_MAPPED_SUBRESOURCE ms;
            ID3D11Texture2D *st = NULL;
            uint32_t h = 0;
            if (dumped && readback(cs, &st, &ms)) {
                char path[512];
                h = crc_frame((const uint8_t *)ms.pData, cs->hw, cs->hh, ms.RowPitch);
                snprintf(path, sizeof(path), "%s%05llu_%08X.bmp", G.dump, (unsigned long long)n, h);
                write_bmp(path, (const uint8_t *)ms.pData, cs->hw, cs->hh, ms.RowPitch);
                ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)st, 0);
                IUnknown_Release((IUnknown *)st);
            }
            if (logged) {
                LARGE_INTEGER now;
                QueryPerformanceCounter(&now);
                fprintf(stderr, "[KELVIN] frame %llu: surface 0x%08X %ux%u gpu hash %08X (%llu distinct so far) [qpc %lld freq %lld]\n",
                        (unsigned long long)n, cs->va, cs->hw, cs->hh, h, (unsigned long long)n,
                        (long long)now.QuadPart, (long long)G.qpf);
                fflush(stderr);
            }
        }
        if (flip_shared(cs, aspect_w, aspect_h)) {
            g_kgpu_ticks[KGP_FLIP] += kg_tick() - prof_t;
            return;
        }
        fprintf(stderr, "[KGPU] shared textures unavailable; using CPU readback\n");
        s_shared_mode = 0;
    }
    flip_readback(cs, aspect_w, aspect_h);
    g_kgpu_ticks[KGP_FLIP] += kg_tick() - prof_t;
}

static void flip_readback(KGpuSurf *cs, uint32_t aspect_w, uint32_t aspect_h)
{
    unsigned int slot, prev, i, sw, sh;
    ID3D11Texture2D *src = kg_post_flip_source(cs, &sw, &sh);
    if (G.stage_w != sw || G.stage_h != sh) {
        for (i = 0; i < KG_STAGING; i++) { SAFE_RELEASE(G.stage[i]); G.stage_frame[i] = 0; }
        G.stage_w = sw; G.stage_h = sh;
    }
    slot = (unsigned int)(G.frames % KG_STAGING);
    if (!G.stage[slot]) {
        D3D11_TEXTURE2D_DESC td;
        memset(&td, 0, sizeof(td));
        td.Width = sw; td.Height = sh; td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM; td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &G.stage[slot]))) return;
    }
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)G.stage[slot], (ID3D11Resource *)src);
    G.stage_frame[slot] = G.frames;
    G.stage_va[slot] = cs->va;
    G.stage_aspect_w[slot] = aspect_w ? aspect_w : sw;
    G.stage_aspect_h[slot] = aspect_h ? aspect_h : sh;
    ID3D11DeviceContext_Flush(G.ctx);
    /* publish the frame two flips back, so the copy has had time to finish */
    prev = (unsigned int)((G.frames + KG_STAGING - 2) % KG_STAGING);
    if (G.frames > 1) publish_staged(prev, 8);
}

/* The raster size changes between frames: every surface rendered at the host size is dropped and made again at the
 * new one the next time a draw or clear names it. Their contents are one frame old at most. */
static void kg_apply_pending_size(void)
{
    const ULONG64 want = (ULONG64)s_pending_size;
    unsigned int w = (unsigned int)(want >> 48), h = (unsigned int)((want >> 32) & 0xFFFFu);
    unsigned int ow = (unsigned int)((want >> 16) & 0xFFFFu), oh = (unsigned int)(want & 0xFFFFu), i;
    if (!want || (w == G.host_w && h == G.host_h && ow == G.out_w && oh == G.out_h)) return;
    if (w != G.host_w || h != G.host_h) {
        for (i = 0; i < KG_MAX_SURF; i++)
            if (G.surf[i].used && G.surf[i].gw == 640 && G.surf[i].gh == 480) surf_free(&G.surf[i]);
        SAFE_RELEASE(G.fb_copy);
        kg_post_free();
    }
    kg_cache_invalidate();
    fprintf(stderr, "[KGPU] host raster %ux%u -> %ux%u, presented at %ux%u\n", G.host_w, G.host_h, w, h, ow, oh);
    G.host_w = w; G.host_h = h; G.out_w = ow; G.out_h = oh;
}

static void kgpu_flip_now(const KelvinTarget *t, unsigned int color_format, uint32_t aspect_w, uint32_t aspect_h)
{
    kg_rtc_before_flip();
    kg_post_before_flip(t, color_format);
    kgpu_flip_inner(t, color_format, aspect_w, aspect_h);
    kg_post_after_flip();
    kg_apply_pending_size();
}

#include <tlhelp32.h>
extern volatile DWORD g_black_main_tid;
extern volatile DWORD g_kelvin_pusher_tid;
extern volatile DWORD g_kgq_tid;
/* RECOMP_KGPU_THREADS=1: which threads of this process burned CPU since the last report. */
static void thread_cpu_report(void)
{
    static ULONGLONG prev_ticks[4096][2];
    static DWORD prev_ids[4096];
    static unsigned int nprev;
    static long long last_qpc;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te;
    long long now = qpc_now();
    double wall = last_qpc ? (double)(now - last_qpc) / (double)G.qpf : 0.0;
    DWORD pid = GetCurrentProcessId();
    unsigned int seen = 0;
    if (snap == INVALID_HANDLE_VALUE) return;
    te.dwSize = sizeof(te);
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID == pid && seen < 4096) {
                HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
                FILETIME c, e, k, u;
                if (h) {
                    if (GetThreadTimes(h, &c, &e, &k, &u)) {
                        ULONGLONG kk = ((ULONGLONG)k.dwHighDateTime << 32) | k.dwLowDateTime;
                        ULONGLONG uu = ((ULONGLONG)u.dwHighDateTime << 32) | u.dwLowDateTime;
                        unsigned int i, found = nprev;
                        for (i = 0; i < nprev; i++) if (prev_ids[i] == te.th32ThreadID) { found = i; break; }
                        if (found < nprev && wall > 0.0) {
                            double dk = (double)(kk - prev_ticks[found][0]) * 1e-7;
                            double du = (double)(uu - prev_ticks[found][1]) * 1e-7;
                            if ((dk + du) / wall > 0.02) {
                                PWSTR name = NULL;
                                typedef HRESULT (WINAPI *GetDesc)(HANDLE, PWSTR *);
                                static GetDesc get_desc;
                                if (!get_desc) get_desc = (GetDesc)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription");
                                if (get_desc) get_desc(h, &name);
                                fprintf(stderr, "[KTHR] tid %lu %-14ls cpu %5.1f%% (user %5.1f%%, kernel %5.1f%%)\n",
                                        (unsigned long)te.th32ThreadID,
                                        te.th32ThreadID == g_black_main_tid ? L"MAIN" : te.th32ThreadID == g_kelvin_pusher_tid ? L"PUSHER" :
                                        te.th32ThreadID == g_kgq_tid ? L"GPUQ" :
                                        name && *name ? name : L"-",
                                        100.0 * (dk + du) / wall, 100.0 * du / wall, 100.0 * dk / wall);
                                if (name) LocalFree(name);
                            }
                        }
                        if (found == nprev && nprev < 4096) { prev_ids[nprev++] = te.th32ThreadID; }
                        if (found < 4096) { prev_ticks[found][0] = kk; prev_ticks[found][1] = uu; }
                        seen++;
                    }
                    CloseHandle(h);
                }
            }
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    last_qpc = now;
}

void kgpu_report(void)
{
    static long long prev[KGP_COUNT];
    static uint64_t prev_frames;
    static const char *const names[KGP_COUNT] = { "vertex", "tex", "upload", "state", "draw", "flip", "clear" };
    double k;
    int i;
    if (!G.ok) return;
    if (getenv("RECOMP_KGPU_THREADS")) thread_cpu_report();
    kt_report();
    kgq_report();
    {
        extern volatile long g_kgpu_cpuv[8];
        static long prev[8];
        fprintf(stderr, "[KGPU] draws on the CPU vertex stage since the last report: no GPU VS %ld, flat shading %ld, lines/points %ld, no vertex program %ld, "
                        "program not ready %ld, GPU path refused %ld\n", g_kgpu_cpuv[0] - prev[0], g_kgpu_cpuv[1] - prev[1], g_kgpu_cpuv[2] - prev[2],
                g_kgpu_cpuv[3] - prev[3], g_kgpu_cpuv[4] - prev[4], g_kgpu_cpuv[5] - prev[5]);
        memcpy(prev, (const void *)g_kgpu_cpuv, sizeof(prev));
        {
            extern volatile long g_kgpu_flat_mode[16];
            extern volatile long long g_kgpu_flat_elems;
            static long pm[16];
            static long long pe;
            int q;
            fprintf(stderr, "[KGPU] flat-shaded draws by primitive mode:");
            for (q = 0; q < 16; q++) if (g_kgpu_flat_mode[q] - pm[q]) fprintf(stderr, " mode%d x%ld", q, g_kgpu_flat_mode[q] - pm[q]);
            fprintf(stderr, "; elements %lld\n", g_kgpu_flat_elems - pe);
            memcpy(pm, (const void *)g_kgpu_flat_mode, sizeof(pm));
            pe = g_kgpu_flat_elems;
        }
    }
    k = G.frames > prev_frames ? 1000.0 / (double)G.qpf / (double)(G.frames - prev_frames) : 0.0;
    fprintf(stderr, "[KGPU] ms per flip over %llu flips:", (unsigned long long)(G.frames - prev_frames));
    for (i = 0; i < KGP_COUNT; i++) {
        fprintf(stderr, " %s %.2f", names[i], (g_kgpu_ticks[i] - prev[i]) * k);
        prev[i] = g_kgpu_ticks[i];
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "[KGPU] flip intervals (ms): <=18 %u, <=20 %u, <=25 %u, <=34 %u, <=70 %u, longer %u; longest %.1f\n",
            s_hist[0], s_hist[1], s_hist[2], s_hist[3], s_hist[4], s_hist[5], s_hist_max_ms);
    {
        char line[400];
        int len = 0, b;
        for (b = 0; b < 40; b++) {
            if (s_fine[b] && len < 360) len += snprintf(line + len, sizeof(line) - len, " %.1f:%u", b * 0.5, s_fine[b]);
        }
        line[len] = 0;
        fprintf(stderr, "[KGPU] flip intervals by 0.5 ms bin:%s\n", line);
    }
    memset(s_fine, 0, sizeof(s_fine));
    memset(s_hist, 0, sizeof(s_hist));
    s_hist_max_ms = 0.0;
    prev_frames = G.frames;
    fprintf(stderr, "[KGPU] draws %llu (%llu with GPU vertices), triangles %llu, skipped %llu; textures: %llu uploads, %llu hits, "
                    "%llu validations, %.1f MiB; vertex bytes %.1f MiB\n",
            (unsigned long long)G.draws, (unsigned long long)G.gvs_draws, (unsigned long long)G.tris, (unsigned long long)G.skipped,
            (unsigned long long)G.tex_uploads, (unsigned long long)G.tex_hits,
            (unsigned long long)G.tex_validations, G.tex_bytes / 1048576.0, G.vb_bytes / 1048576.0);
}

/* ---- sampling profiler (diagnostic) -------------------------------------------------------------------
 * RECOMP_KGPU_PROFILE=start,seconds: after `start` seconds, sample the instruction pointer of every
 * thread of the process once a millisecond for `seconds`, then print each busy thread's hottest symbols
 * (the generated guest functions are named sub_XXXXXXXX; needs the PDB beside the executable). */
#include <dbghelp.h>
typedef struct ProfSym { uint64_t rip; char name[96]; } ProfSym;
static DWORD WINAPI profile_thread(LPVOID arg)
{
    unsigned int start = ((unsigned int)(uintptr_t)arg) & 0xFFFF, secs = ((unsigned int)(uintptr_t)arg) >> 16;
    typedef struct TS { DWORD tid; HANDLE h; uint64_t total; uint32_t nsym; struct { uint64_t rip; uint32_t n; } sym[4096]; } TS;
    TS *ts = (TS *)calloc(256, sizeof(TS));
    unsigned int nts = 0, i, k;
    DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();
    long long t0, deadline;
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    Sleep(start * 1000);
    {
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        THREADENTRY32 te;
        te.dwSize = sizeof(te);
        if (snap != INVALID_HANDLE_VALUE && Thread32First(snap, &te)) {
            do {
                if (te.th32OwnerProcessID == pid && te.th32ThreadID != self && nts < 256) {
                    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
                    if (h) { ts[nts].tid = te.th32ThreadID; ts[nts].h = h; nts++; }
                }
            } while (Thread32Next(snap, &te));
        }
        if (snap != INVALID_HANDLE_VALUE) CloseHandle(snap);
    }
    t0 = qpc_now();
    deadline = t0 + (long long)secs * f.QuadPart;
    while (qpc_now() < deadline) {
        for (i = 0; i < nts; i++) {
            CONTEXT ctx;
            ctx.ContextFlags = CONTEXT_CONTROL;
            if (SuspendThread(ts[i].h) == (DWORD)-1) continue;
            if (GetThreadContext(ts[i].h, &ctx)) {
                TS *t = &ts[i];
                t->total++;
                for (k = 0; k < t->nsym; k++) if (t->sym[k].rip == ctx.Rip) break;
                if (k == t->nsym && t->nsym < 4096) { t->sym[k].rip = ctx.Rip; t->sym[k].n = 0; t->nsym++; }
                if (k < t->nsym) t->sym[k].n++;
            }
            ResumeThread(ts[i].h);
        }
        Sleep(1);
    }
    for (i = 0; i < nts; i++) {
        TS *t = &ts[i];
        typedef struct { char name[96]; uint32_t n; } Agg;
        Agg agg[256];
        unsigned int na = 0;
        if (t->total < 50) continue;
        for (k = 0; k < t->nsym; k++) {
            char buf[sizeof(SYMBOL_INFO) + 128];
            SYMBOL_INFO *si = (SYMBOL_INFO *)buf;
            DWORD64 disp = 0;
            unsigned int j;
            char nm[96];
            memset(buf, 0, sizeof(buf));
            si->SizeOfStruct = sizeof(SYMBOL_INFO);
            si->MaxNameLen = 120;
            if (SymFromAddr(GetCurrentProcess(), t->sym[k].rip, &disp, si)) snprintf(nm, sizeof(nm), "%s", si->Name);
            else snprintf(nm, sizeof(nm), "?%llX", (unsigned long long)(t->sym[k].rip >> 12));
            for (j = 0; j < na; j++) if (!strcmp(agg[j].name, nm)) break;
            if (j == na && na < 256) { snprintf(agg[na].name, sizeof(agg[na].name), "%s", nm); agg[na].n = 0; na++; }
            if (j < na) agg[j].n += t->sym[k].n;
        }
        for (k = 0; k < na; k++) {
            unsigned int j;
            for (j = k + 1; j < na; j++) if (agg[j].n > agg[k].n) { Agg tmp = agg[k]; agg[k] = agg[j]; agg[j] = tmp; }
        }
        fprintf(stderr, "[KPROF] tid %lu: %llu samples:", (unsigned long)t->tid, (unsigned long long)t->total);
        {
            const char *topn = getenv("RECOMP_KGPU_PROFILE_TOP");
            unsigned int top = topn && *topn ? (unsigned int)atoi(topn) : 8;
            for (k = 0; k < na && k < top; k++) fprintf(stderr, " %s %.1f%%", agg[k].name, 100.0 * agg[k].n / (double)t->total);
        }
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    return 0;
}

#include "kelvin_gpu_slow.inc"
#include "kelvin_gpu_incl.inc"
volatile DWORD g_black_main_tid;

void kgpu_profile_maybe_start(void)
{
    const char *e = getenv("RECOMP_KGPU_PROFILE");
    unsigned int a = 0, b = 0;
    if (e && sscanf(e, "%u,%u", &a, &b) == 2 && b)
        CloseHandle(CreateThread(NULL, 0, profile_thread, (LPVOID)(uintptr_t)((b << 16) | (a & 0xFFFF)), 0, NULL));
    incl_maybe_start();
    e = getenv("RECOMP_KGPU_SLOW");
    a = 0; b = 0;
    if (e && sscanf(e, "%u,%u", &a, &b) >= 1 && a)
        CloseHandle(CreateThread(NULL, 0, slow_thread, (LPVOID)(uintptr_t)(((b & 0xFFFF) << 16) | (a & 0xFFFF)), 0, NULL));
}

#include "kelvin_gpu_async.inc"
