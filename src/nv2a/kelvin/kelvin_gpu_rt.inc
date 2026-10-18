/* Ray tracing geometry capture, included by kelvin_gpu.c after kelvin_gpu_ssr.inc (the draw path calls kg_rtc_begin
 * and kg_rtc_end around a draw, kg_rtc_note_draw for every draw and kg_rtc_before_flip at the flip).
 *
 * The emulator never sees BLACK's scene, only triangles after its vertex programs. Every draw of the world on the main
 * target (it writes depth there; the first-person weapon, with its own projection, is left out) streams its
 * triangles out as it is rasterised (vs_make_so): window position, diffuse colour and two texture coordinate sets.
 *
 * When the HUD begins (or at the flip), a compute pass turns them into world space. The window position is undone to
 * the program's clip space with the viewport (c[58], c[59]) and the renderer's own scale, then taken back through the
 * camera's view-projection. That matrix is the one most of the frame's world draws transform with (c[10..13]): the
 * level is stored in world space, so its many draws all use the camera's matrix alone, and a model with a transform
 * of its own comes out in world space all the same. The result goes to a texture the D3D12 side shares, two texels a
 * vertex: (world xyz, colour RGBA8) and (uv0, uv1), three vertices a triangle in draw order.
 *
 * RECOMP_KGPU_RT_CHECK=1 (diagnostic): every 300 frames, the converted first vertex of the level's draws is compared
 * with the game's own world-space position of that vertex. */

#define KRT_SO_VERTS (1u << 20)
#define KRT_GEO_W 2048u
#define KRT_DRAWS_MAX 4096u

typedef struct KrtCapDraw {
    uint32_t first, count;              /* vertices in the stream */
    uint32_t tex[2], tex_w[2], tex_h[2], tex_fmt[2];
    uint32_t vs, alpha_test;
    uint32_t rt_tex[2], rt_flags;       /* for the D3D12 side: texture indices and KRT_* flags */
    bool sh;
    float sh_l1[3];                     /* lit by ambient coefficients: their linear term, in the model's space */
    bool skinned;                       /* its program blends bones: a character */
    uint64_t key;                       /* the same model from one frame to the next: its vertices, program, count */
    float m[16];                        /* the program's c[10..13] */
    bool raw_ok;
    float raw[3];                       /* the first vertex's input position, for the check */
} KrtCapDraw;

static struct {
    int state;                          /* 0 untried, 1 on, -1 off */
    ID3D11Buffer *so;
    ID3D11ShaderResourceView *so_srv;
    ID3D11Texture2D *geo;
    ID3D11UnorderedAccessView *geo_uav;
    HANDLE geo_h;
    ID3D11ComputeShader *cs;
    ID3D11Buffer *cb;
    uint32_t used;
    KrtCapDraw draws[KRT_DRAWS_MAX];
    unsigned ndraws;
    bool overflow, done;
    float c58[4], c59[4], scale[4];
    unsigned frames, check;
    ID3D11Texture2D *out;               /* what the D3D12 passes wrote, at the raster size */
    ID3D11Texture2D *in[3];             /* their inputs: the picture, the reflection weight and surface */
    HANDLE in_h[3];
    ID3D11ShaderResourceView *pic_srv;  /* the picture copy, read back by the lighting composite */
    ID3D11Texture2D *depth, *lit;       /* the depth buffer as R32F (written here), the lighting factor (by D3D12) */
    ID3D11RenderTargetView *depth_rtv;
    ID3D11ShaderResourceView *lit_srv;
    HANDLE depth_h, lit_h;
    ID3D11PixelShader *ps_depth, *ps_lit, *ps_view_half;
    ID3D11ShaderResourceView *out_srv;
    HANDLE out_h;
    unsigned out_w, out_hh;
    ID3D11PixelShader *ps_view;
    int view;                           /* RECOMP_KGPU_RT_VIEW: 1 normals, 2 shaded, 3 calibration, 4 lighting (diagnostic) */
    unsigned calib;
    float light[3];                     /* towards the main light, smoothed over frames */
    bool have_light;
    float shadow, ao, ao_radius, gi;    /* strengths (RECOMP_KGPU_RT_SHADOW/_AO/_GI, 0 off) */
    uint64_t frame;
    /* what has a transform of its own, by key: its last world transform and when it last moved */
    struct { uint64_t key, moved, seen; float world[12]; } models[8192];
    bool reflect;                       /* ray-traced reflections (RECOMP_KGPU_RT_REFL=0 turns them off) */
    uint32_t first[KRT_DRAWS_MAX], count[KRT_DRAWS_MAX], tex0[KRT_DRAWS_MAX], tex1[KRT_DRAWS_MAX], flags[KRT_DRAWS_MAX];
} s_rtc;

/* The depth buffer copied as plain floats for D3D12; the picture multiplied by the lighting factor. */
static const char kg_rtc_depth_hlsl[] =
    "Texture2D<float4> z : register(t0);\n"
    "float main(float4 p : SV_Position) : SV_Target { return z.Load(int3(p.xy, 0)).r; }\n";
/* The lighting factor is made at half the size: scaled up with a linear filter. */
static const char kg_rtc_lit_hlsl[] =
    "Texture2D<float4> pic : register(t0);\n"
    "Texture2D<float4> lit : register(t1);\n"
    "SamplerState lin : register(s0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "    uint w, h; pic.GetDimensions(w, h);\n"
    "    float4 c = pic.Load(int3(p.xy, 0));\n"
    "    return float4(c.rgb * lit.SampleLevel(lin, p.xy / float2(w, h), 0).rgb, c.a);\n"
    "}\n";
static const char kg_rtc_view_half_hlsl[] =
    "Texture2D<float4> rt : register(t0);\n"
    "SamplerState lin : register(s0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "    uint w, h; rt.GetDimensions(w, h);\n"
    "    if (p.x > w) discard;   // the left half: the picture stays on the right to compare\n"
    "    return float4(rt.SampleLevel(lin, p.xy / float2(w * 2, h * 2), 0).rgb, 1.0);\n"
    "}\n";

static const char kg_rtc_view_hlsl[] =
    "Texture2D<float4> rt : register(t0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "    uint w, h; rt.GetDimensions(w, h);\n"
    "    if (p.x > w * 0.5) discard;   // the left half: the picture stays on the right to compare\n"
    "    return float4(rt.Load(int3(p.xy, 0)).rgb, 1.0);\n"
    "}\n";

static const char kg_rtc_hlsl[] =
    "ByteAddressBuffer so : register(t0);\n"
    "RWTexture2D<float4> geo : register(u0);\n"
    "cbuffer RTC : register(b0) { float4 inv0; float4 inv1; float4 inv2; float4 c58; float4 c59; float4 scale; uint4 n; };\n"
    "uint pack(float4 c) { uint4 u = (uint4)(saturate(c) * 255.0 + 0.5); return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24); }\n"
    "[numthreads(64, 1, 1)] void cs_main(uint3 id : SV_DispatchThreadID) {\n"
    "    uint i = id.x, b = i * 64;\n"
    "    if (i >= n.x) return;\n"
    "    float4 pos = asfloat(so.Load4(b)), col = asfloat(so.Load4(b + 16));\n"
    "    float4 t0 = asfloat(so.Load4(b + 32)), t1 = asfloat(so.Load4(b + 48));\n"
    "    float w = pos.w;\n"
    "    float2 pp = float2((pos.x / w + 1.0) / scale.x, (1.0 - pos.y / w) / scale.y);\n"
    "    float3 r = float3((pp - c59.xy) / c58.xy * w, w) - float3(inv0.w, inv1.w, inv2.w);\n"
    "    float3 wp = float3(dot(inv0.xyz, r), dot(inv1.xyz, r), dot(inv2.xyz, r));\n"
    "    uint2 at = uint2((i * 2) % n.y, (i * 2) / n.y);\n"
    "    geo[at] = float4(wp, asfloat(pack(col)));\n"
    "    geo[at + uint2(1, 0)] = float4(t0.xy, t1.xy);\n"
    "}\n";

/* inv: the inverse of the camera matrix's x, y and w rows (in xyz), each with that row's offset (in w). Depth is left
 * out: it is stored with far less precision than w, the distance along the view direction. */
typedef struct KrtCb { float inv[12], c58[4], c59[4], scale[4]; uint32_t n[4]; } KrtCb;

static bool kg_rtc_setup(void)
{
    D3D11_BUFFER_DESC bd;
    D3D11_SHADER_RESOURCE_VIEW_DESC sd;
    D3D11_TEXTURE2D_DESC td;
    IDXGIResource1 *r1 = NULL;
    ID3DBlob *code;
    const char *e;
    if (!krt_available()) return false;
    memset(&bd, 0, sizeof(bd));
    bd.ByteWidth = KRT_SO_VERTS * KG_SO_STRIDE;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_STREAM_OUTPUT | D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    if (FAILED(ID3D11Device_CreateBuffer(G.dev, &bd, NULL, &s_rtc.so))) { fprintf(stderr, "[KRT] stream-output buffer failed\n"); return false; }
    memset(&sd, 0, sizeof(sd));
    sd.Format = DXGI_FORMAT_R32_TYPELESS;
    sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    sd.BufferEx.NumElements = bd.ByteWidth / 4;
    sd.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    if (FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)s_rtc.so, &sd, &s_rtc.so_srv))) return false;

    memset(&td, 0, sizeof(td));
    td.Width = KRT_GEO_W; td.Height = KRT_SO_VERTS * 2 / KRT_GEO_W; td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_rtc.geo)) ||
        FAILED(ID3D11Device_CreateUnorderedAccessView(G.dev, (ID3D11Resource *)s_rtc.geo, NULL, &s_rtc.geo_uav)) ||
        FAILED(ID3D11Texture2D_QueryInterface(s_rtc.geo, &IID_IDXGIResource1, (void **)&r1)) ||
        FAILED(IDXGIResource1_CreateSharedHandle(r1, NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, NULL, &s_rtc.geo_h))) {
        if (r1) IDXGIResource1_Release(r1);
        fprintf(stderr, "[KRT] shared geometry texture failed\n");
        return false;
    }
    IDXGIResource1_Release(r1);
    if (!krt_set_geometry(s_rtc.geo_h, td.Width, td.Height)) return false;

    code = compile_hlsl("", kg_rtc_hlsl, "cs_main", "cs_5_0");
    if (!code || FAILED(ID3D11Device_CreateComputeShader(G.dev, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code),
                                                         NULL, &s_rtc.cs))) {
        if (code) ID3D10Blob_Release(code);
        return false;
    }
    ID3D10Blob_Release(code);
    if (!make_buffer(sizeof(KrtCb), D3D11_BIND_CONSTANT_BUFFER, &s_rtc.cb)) return false;
    e = getenv("RECOMP_KGPU_RT_CHECK");
    s_rtc.check = e && *e && *e != '0';
    e = getenv("RECOMP_KGPU_RT_VIEW");
    s_rtc.view = e && *e ? atoi(e) : 0;
    e = getenv("RECOMP_KGPU_RT_SHADOW");
    s_rtc.shadow = e && *e ? (float)atof(e) : 0.55f;
    e = getenv("RECOMP_KGPU_RT_AO");
    s_rtc.ao = e && *e ? (float)atof(e) : 0.6f;
    s_rtc.ao_radius = 1.5f;
    e = getenv("RECOMP_KGPU_RT_GI");
    s_rtc.gi = e && *e ? (float)atof(e) : 2.0f;
    e = getenv("RECOMP_KGPU_RT_REFL");
    s_rtc.reflect = !(e && *e == '0');
    code = compile_hlsl("", kg_rtc_view_hlsl, "main", "ps_5_0");
    if (code) {
        ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code), NULL, &s_rtc.ps_view);
        ID3D10Blob_Release(code);
    }
    code = compile_hlsl("", kg_rtc_depth_hlsl, "main", "ps_5_0");
    if (code) {
        ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code), NULL, &s_rtc.ps_depth);
        ID3D10Blob_Release(code);
    }
    code = compile_hlsl("", kg_rtc_view_half_hlsl, "main", "ps_5_0");
    if (code) {
        ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code), NULL, &s_rtc.ps_view_half);
        ID3D10Blob_Release(code);
    }
    code = compile_hlsl("", kg_rtc_lit_hlsl, "main", "ps_5_0");
    if (code) {
        ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code), NULL, &s_rtc.ps_lit);
        ID3D10Blob_Release(code);
    }
    fprintf(stderr, "[KRT] geometry capture on: up to %u vertices a frame\n", KRT_SO_VERTS);
    return true;
}

/* The Video Settings' switch (pc_video), read again only when the settings change. */
static bool kg_rtc_wanted(void)
{
    static unsigned generation = ~0u;
    static bool on;
    const unsigned g = pc_video_generation();
    if (g != generation) {
        PcVideoSettings v;
        pc_video_get(&v);
        on = v.ray_tracing != 0;
        generation = g;
    }
    return on;
}

static bool kg_rtc_on(void)
{
    if (s_rtc.state == 0) s_rtc.state = kg_rtc_setup() ? 1 : -1;
    return s_rtc.state > 0;
}

/* Triangles the draw makes, as stream output writes them: a list's, or each strip run's (restarts cut a strip). */
static uint32_t kg_rtc_prims(const KGpuDraw *d)
{
    uint32_t i, n = 0, run = 0;
    if (d->topology == KG_TOPO_LIST) return d->icount / 3;
    for (i = 0; i < d->icount; i++) {
        if (d->indices[i] == 0xFFFFFFFFu) { if (run >= 3) n += run - 2; run = 0; }
        else run++;
    }
    if (run >= 3) n += run - 2;
    return n;
}

/* Before a draw is issued: streams its triangles out if it draws the world on the main target. */
static bool kg_rtc_begin(const KGpuDraw *d, const KgDc *dc, KGpuVs *ve, float zmax)
{
    const KPixelState *ps = d->ps;
    const float *c;
    KrtCapDraw *r;
    uint32_t prims, i;
    UINT offset;
    if (!ve || !ve->so_gs || !d->gvs || !d->gvs->constants || !post_main_target(dc->cs) || !dc->zs || !ps->depth_write ||
        s_rtc.done || kg_viewmodel_scale(d->gvs->hash) != 1.0f || !kg_rtc_wanted() || !kg_rtc_on())
        return false;
    prims = kg_rtc_prims(d);
    if (!prims) return false;
    if (s_rtc.ndraws >= KRT_DRAWS_MAX || s_rtc.used + prims * 3 > KRT_SO_VERTS) { s_rtc.overflow = true; return false; }
    c = d->gvs->constants;
    r = &s_rtc.draws[s_rtc.ndraws++];
    memset(r, 0, sizeof(*r));
    r->first = s_rtc.used; r->count = prims * 3;
    for (i = 0; i < 2; i++) {
        if (!ps->tex[i].valid) continue;
        r->tex[i] = ps->tex[i].base; r->tex_w[i] = ps->tex[i].width; r->tex_h[i] = ps->tex[i].height;
        r->tex_fmt[i] = ps->tex[i].color_format;
    }
    r->vs = d->gvs->hash;
    r->alpha_test = ps->alpha_test ? 1u : 0u;
    for (i = 0; i < 2; i++) {
        KGpuTex *te = dc->te[i];
        if (!te || !te->rt_h) continue;
        if (te->rt_index == -1) { te->rt_index = krt_tex_add(te->rt_h); if (te->rt_index < 0) te->rt_index = -2; }
        if (te->rt_index < 0) continue;
        r->rt_tex[i] = (uint32_t)te->rt_index;
        r->rt_flags |= i ? KRT_TEX1 : KRT_TEX0;
        if (ps->tex[i].linear) r->rt_flags |= i ? KRT_TEX1_TEXELS : KRT_TEX0_TEXELS;
    }
    if (ps->alpha_test)
        r->rt_flags |= KRT_ALPHA_TEST | ((uint32_t)(ps->alpha_func & 7) << 8) | ((uint32_t)(ps->alpha_ref & 255) << 16);
    if (!ve->rt_sh) {
        /* a program lit by ambient coefficients reads the normal (v2) and c[2..5]; c[4] scales the normal's
         * components, the coefficients' linear term (BLACK's props and characters) */
        ve->rt_sh = ve->source && strstr(ve->source, "FetchAttr(2, vid)") && strstr(ve->source, "(C[4].xyz") &&
                 strstr(ve->source, "(C[5].xyz") ? 2 : 1;
    }
    if (ve->rt_sh == 2) { r->sh = true; memcpy(r->sh_l1, c + 4 * 4, sizeof(r->sh_l1)); }
    if (!ve->rt_skin) ve->rt_skin = ve->source && strstr(ve->source, "+ (int)a0,") ? 2 : 1;   /* bones by relative address */
    r->skinned = ve->rt_skin == 2;
    {
        const KGpuAttrSrc *a = &d->gvs->attr[0];
        uint64_t k = (uint64_t)(uintptr_t)a->data * 0x9E3779B97F4A7C15ull;
        k ^= ((uint64_t)d->gvs->hash << 32) ^ (uint64_t)d->icount * 0xC2B2AE3D27D4EB4Full ^ d->indices[0];
        r->key = k ? k : 1;
    }
    memcpy(r->m, c + 4 * 10, sizeof(r->m));
    {
        const KGpuAttrSrc *a = &d->gvs->attr[0];
        uint32_t first = d->indices[0];
        if (a->fmt == 4 && a->count >= 3 && a->data && first != 0xFFFFFFFFu) {
            const int64_t at = (int64_t)first * a->stride;
            if (at >= a->lo && at + 12 <= a->hi + 1) { memcpy(r->raw, a->data + at, 12); r->raw_ok = true; }
        }
    }
    memcpy(s_rtc.c58, c + 4 * 58, sizeof(s_rtc.c58));
    memcpy(s_rtc.c59, c + 4 * 59, sizeof(s_rtc.c59));
    s_rtc.scale[0] = 2.0f / (float)dc->cs->gw; s_rtc.scale[1] = 2.0f / (float)dc->cs->gh; s_rtc.scale[2] = 1.0f / zmax;
    offset = s_rtc.used * KG_SO_STRIDE;
    s_rtc.used += prims * 3;
    ID3D11DeviceContext_GSSetShader(G.ctx, ve->so_gs, NULL, 0);
    ID3D11DeviceContext_SOSetTargets(G.ctx, 1, &s_rtc.so, &offset);
    return true;
}

static void kg_rtc_end(void)
{
    ID3D11Buffer *none = NULL;
    UINT zero = 0;
    ID3D11DeviceContext_GSSetShader(G.ctx, NULL, NULL, 0);
    ID3D11DeviceContext_SOSetTargets(G.ctx, 1, &none, &zero);
}

/* The camera matrix m (rows c[10..13]) gives clip x, y and w as a linear function of the world position: the inverse
 * of those three rows, laid out as KrtCb.inv. */
static bool kg_rtc_invert(const float *m, float *out)
{
    const double a[9] = { m[0], m[1], m[2], m[4], m[5], m[6], m[12], m[13], m[14] };
    double inv[9], det;
    int i;
    inv[0] = a[4] * a[8] - a[5] * a[7];
    inv[1] = a[2] * a[7] - a[1] * a[8];
    inv[2] = a[1] * a[5] - a[2] * a[4];
    inv[3] = a[5] * a[6] - a[3] * a[8];
    inv[4] = a[0] * a[8] - a[2] * a[6];
    inv[5] = a[2] * a[3] - a[0] * a[5];
    inv[6] = a[3] * a[7] - a[4] * a[6];
    inv[7] = a[1] * a[6] - a[0] * a[7];
    inv[8] = a[0] * a[4] - a[1] * a[3];
    det = a[0] * inv[0] + a[1] * inv[3] + a[2] * inv[6];
    if (det == 0.0 || det != det) return false;
    for (i = 0; i < 3; i++) {
        out[i * 4 + 0] = (float)(inv[i * 3 + 0] / det);
        out[i * 4 + 1] = (float)(inv[i * 3 + 1] / det);
        out[i * 4 + 2] = (float)(inv[i * 3 + 2] / det);
    }
    out[3] = m[3]; out[7] = m[7]; out[11] = m[15];
    return true;
}

/* The camera's view-projection: the matrix most of the frame's world triangles were drawn with. */
static const float *kg_rtc_camera(void)
{
    unsigned i, j, best = 0;
    uint32_t best_n = 0;
    for (i = 0; i < s_rtc.ndraws; i++) {
        uint32_t n = 0;
        for (j = 0; j < s_rtc.ndraws; j++)
            if (memcmp(s_rtc.draws[i].m, s_rtc.draws[j].m, sizeof(s_rtc.draws[i].m)) == 0) n += s_rtc.draws[j].count;
        if (n > best_n) { best_n = n; best = i; }
        if (best_n * 2 > s_rtc.used) break;
    }
    return best_n ? s_rtc.draws[best].m : NULL;
}

static void kg_rtc_check(const float *vp)
{
    D3D11_TEXTURE2D_DESC td;
    D3D11_MAPPED_SUBRESOURCE ms;
    ID3D11Texture2D *st = NULL;
    D3D11_BOX box;
    unsigned i, checked = 0, rows;
    double worst = 0.0;
    rows = (s_rtc.used * 2 + KRT_GEO_W - 1) / KRT_GEO_W;
    if (!rows) return;
    ID3D11Texture2D_GetDesc(s_rtc.geo, &td);
    td.Height = rows; td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &st))) return;
    box.left = 0; box.top = 0; box.front = 0; box.right = KRT_GEO_W; box.bottom = rows; box.back = 1;
    ID3D11DeviceContext_CopySubresourceRegion(G.ctx, (ID3D11Resource *)st, 0, 0, 0, 0, (ID3D11Resource *)s_rtc.geo, 0, &box);
    if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)st, 0, D3D11_MAP_READ, 0, &ms))) {
        for (i = 0; i < s_rtc.ndraws; i++) {
            const KrtCapDraw *r = &s_rtc.draws[i];
            const float *p;
            double dx, dy, dz, err;
            uint32_t t = r->first * 2;
            if (!r->raw_ok || memcmp(r->m, vp, sizeof(r->m)) != 0) continue;
            p = (const float *)((const uint8_t *)ms.pData + (size_t)(t / KRT_GEO_W) * ms.RowPitch) + (t % KRT_GEO_W) * 4;
            dx = p[0] - r->raw[0]; dy = p[1] - r->raw[1]; dz = p[2] - r->raw[2];
            err = sqrt(dx * dx + dy * dy + dz * dz);
            if (err > worst) worst = err;
            checked++;
        }
        ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)st, 0);
    }
    ID3D11Texture2D_Release(st);
    fprintf(stderr, "[KRT] check: %u draws, %u triangles%s; %u level draws, worst world-position error %.4f\n",
            s_rtc.ndraws, s_rtc.used / 3, s_rtc.overflow ? " (stream full)" : "", checked, worst);
}

/* A texture D3D12 shares, by NT handle. */
static bool kg_rtc_shared(unsigned w, unsigned h, DXGI_FORMAT fmt, UINT bind, ID3D11Texture2D **tex,
                          ID3D11ShaderResourceView **srv, HANDLE *handle)
{
    D3D11_TEXTURE2D_DESC td;
    IDXGIResource1 *r1 = NULL;
    memset(&td, 0, sizeof(td));
    td.Width = w; td.Height = h; td.MipLevels = td.ArraySize = 1;
    td.Format = fmt; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = bind | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, tex)) ||
        (srv && FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)*tex, NULL, srv))) ||
        FAILED(ID3D11Texture2D_QueryInterface(*tex, &IID_IDXGIResource1, (void **)&r1)) ||
        FAILED(IDXGIResource1_CreateSharedHandle(r1, NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, NULL, handle))) {
        if (r1) IDXGIResource1_Release(r1);
        return false;
    }
    IDXGIResource1_Release(r1);
    return true;
}

static void kg_rtc_free_textures(void)
{
    unsigned i;
    SAFE_RELEASE(s_rtc.out_srv); SAFE_RELEASE(s_rtc.out);
    for (i = 0; i < 3; i++) SAFE_RELEASE(s_rtc.in[i]);
    SAFE_RELEASE(s_rtc.pic_srv); SAFE_RELEASE(s_rtc.depth_rtv); SAFE_RELEASE(s_rtc.depth); SAFE_RELEASE(s_rtc.lit_srv); SAFE_RELEASE(s_rtc.lit);
    if (s_rtc.depth_h) { CloseHandle(s_rtc.depth_h); s_rtc.depth_h = NULL; }
    if (s_rtc.lit_h) { CloseHandle(s_rtc.lit_h); s_rtc.lit_h = NULL; }
    if (s_rtc.out_h) { CloseHandle(s_rtc.out_h); s_rtc.out_h = NULL; }
    for (i = 0; i < 3; i++) if (s_rtc.in_h[i]) { CloseHandle(s_rtc.in_h[i]); s_rtc.in_h[i] = NULL; }
    s_rtc.out_w = s_rtc.out_hh = 0;
}

/* What the D3D12 passes read and write, at the main target's raster size; made again when that size changes: the
 * output, and the reflections' inputs (the picture, and the screen-space reflection targets' weight and surface,
 * in their formats -- see ssr_targets). */
static bool kg_rtc_output(unsigned w, unsigned h)
{
    static const DXGI_FORMAT in_fmt[3] = { DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT };
    unsigned i;
    if (s_rtc.out && s_rtc.out_w == w && s_rtc.out_hh == h) return true;
    kg_rtc_free_textures();
    if (!kg_rtc_shared(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_UNORDERED_ACCESS, &s_rtc.out, &s_rtc.out_srv, &s_rtc.out_h))
        goto fail;
    for (i = 0; i < 3; i++)
        if (!kg_rtc_shared(w, h, in_fmt[i], 0, &s_rtc.in[i], i ? NULL : &s_rtc.pic_srv, &s_rtc.in_h[i])) goto fail;
    if (!kg_rtc_shared(w, h, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_RENDER_TARGET, &s_rtc.depth, NULL, &s_rtc.depth_h) ||
        FAILED(ID3D11Device_CreateRenderTargetView(G.dev, (ID3D11Resource *)s_rtc.depth, NULL, &s_rtc.depth_rtv)) ||
        !kg_rtc_shared((w + 1) / 2, (h + 1) / 2, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D11_BIND_UNORDERED_ACCESS, &s_rtc.lit, &s_rtc.lit_srv,
                       &s_rtc.lit_h))
        goto fail;
    if (!krt_set_output(s_rtc.out_h, w, h) ||
        !krt_set_inputs(s_rtc.in_h[0], s_rtc.in_h[1], s_rtc.in_h[2], s_rtc.depth_h, s_rtc.lit_h)) goto fail;
    s_rtc.out_w = w; s_rtc.out_hh = h;
    return true;
fail:
    fprintf(stderr, "[KRT] the %ux%u textures shared with D3D12 failed\n", w, h);
    kg_rtc_free_textures();
    return false;
}

static float kg_rtc_half(uint16_t h)
{
    const int e = (h >> 10) & 31, m = h & 1023;
    const float f = e ? (float)ldexp(1.0 + m / 1024.0, e - 15) : (float)ldexp(m / 1024.0, -14);
    return (h & 0x8000) ? -f : f;
}

static int kg_rtc_cmp(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return x < y ? -1 : x > y;
}

/* RECOMP_KGPU_RT_VIEW=3 (diagnostic): how much brighter the picture is than the hits shaded from the textures, the
 * median over the pixels lit by a lightmap, by vertex colour and by neither (krt_view.hlsl). */
static void kg_rtc_calibration(void)
{
    D3D11_TEXTURE2D_DESC td;
    D3D11_MAPPED_SUBRESOURCE ms;
    ID3D11Texture2D *st = NULL;
    static float *ratios[3];
    unsigned n[3] = { 0, 0, 0 }, x, y, k, cap = (s_rtc.out_w / 4 + 1) * (s_rtc.out_hh / 4 + 1);
    for (k = 0; k < 3; k++) if (!ratios[k] && !(ratios[k] = (float *)malloc(cap * sizeof(float)))) return;
    ID3D11Texture2D_GetDesc(s_rtc.out, &td);
    td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &st))) return;
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)st, (ID3D11Resource *)s_rtc.out);
    if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)st, 0, D3D11_MAP_READ, 0, &ms))) {
        for (y = 0; y < td.Height; y += 4) {
            const uint16_t *row = (const uint16_t *)((const uint8_t *)ms.pData + (size_t)y * ms.RowPitch);
            for (x = 0; x < td.Width; x += 4) {
                const float r = kg_rtc_half(row[x * 4]), lp = kg_rtc_half(row[x * 4 + 1]), kind = kg_rtc_half(row[x * 4 + 3]);
                const int c = (int)(kind + 0.5f) - 1;
                if (c < 0 || c > 2 || lp < 0.02f || n[c] >= cap) continue;   /* misses, and black pixels say little */
                ratios[c][n[c]++] = r;
            }
        }
        ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)st, 0);
    }
    ID3D11Texture2D_Release(st);
    for (k = 0; k < 3; k++) if (n[k]) qsort(ratios[k], n[k], sizeof(float), kg_rtc_cmp);
    fprintf(stderr, "[KRT] calibration, picture over shaded (median, quartiles): lightmap %.3f %.3f-%.3f (%u), vertex %.3f %.3f-%.3f (%u), "
                    "neither %.3f (%u)\n",
            n[0] ? ratios[0][n[0] / 2] : 0.0f, n[0] ? ratios[0][n[0] / 4] : 0.0f, n[0] ? ratios[0][n[0] * 3 / 4] : 0.0f, n[0],
            n[1] ? ratios[1][n[1] / 2] : 0.0f, n[1] ? ratios[1][n[1] / 4] : 0.0f, n[1] ? ratios[1][n[1] * 3 / 4] : 0.0f, n[1],
            n[2] ? ratios[2][n[2] / 2] : 0.0f, n[2]);
}

/* The inverse of a 4x4 matrix (rows), false if it has none. */
static bool kg_rtc_inv4(const float *m, double *inv)
{
    double a[16], det;
    int i;
    for (i = 0; i < 16; i++) a[i] = m[i];
    inv[0] = a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4] = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8] = a[4]*a[9]*a[15] - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14] + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1] = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5] = a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9] = -a[0]*a[9]*a[15] + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] = a[0]*a[9]*a[14] - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2] = a[1]*a[6]*a[15] - a[1]*a[7]*a[14] - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7] - a[13]*a[3]*a[6];
    inv[6] = -a[0]*a[6]*a[15] + a[0]*a[7]*a[14] + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7] + a[12]*a[3]*a[6];
    inv[10] = a[0]*a[5]*a[15] - a[0]*a[7]*a[13] - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7] - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14] + a[0]*a[6]*a[13] + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6] + a[12]*a[2]*a[5];
    inv[3] = -a[1]*a[6]*a[11] + a[1]*a[7]*a[10] + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7] + a[9]*a[3]*a[6];
    inv[7] = a[0]*a[6]*a[11] - a[0]*a[7]*a[10] - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7] - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11] + a[0]*a[7]*a[9] + a[4]*a[1]*a[11] - a[4]*a[3]*a[9] - a[8]*a[1]*a[7] + a[8]*a[3]*a[5];
    inv[15] = a[0]*a[5]*a[10] - a[0]*a[6]*a[9] - a[4]*a[1]*a[10] + a[4]*a[2]*a[9] + a[8]*a[1]*a[6] - a[8]*a[2]*a[5];
    det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    if (det == 0.0 || det != det) return false;
    for (i = 0; i < 16; i++) inv[i] /= det;
    return true;
}

/* Whether a draw with a transform of its own is dynamic: a character (its program blends bones), or a model that
 * moved in the last two seconds. A model that stays put -- a tree, a crate, rubble -- is in the lightmaps already,
 * its shadow too, so it is left to them. */
static bool kg_rtc_dynamic(const KrtCapDraw *r, const double *ivp)
{
    float world[12];
    unsigned j, k, slot, probe;
    if (r->skinned) return true;
    for (j = 0; j < 3; j++)
        for (k = 0; k < 4; k++)
            world[j * 4 + k] = (float)(ivp[j * 4 + 0] * r->m[0 * 4 + k] + ivp[j * 4 + 1] * r->m[1 * 4 + k] +
                                       ivp[j * 4 + 2] * r->m[2 * 4 + k] + ivp[j * 4 + 3] * r->m[3 * 4 + k]);
    slot = (unsigned)(r->key % 8192u);
    for (probe = 0; probe < 16; probe++, slot = (slot + 1) % 8192u) {
        if (s_rtc.models[slot].key == r->key) break;
        if (!s_rtc.models[slot].key || s_rtc.frame - s_rtc.models[slot].seen > 600) {
            s_rtc.models[slot].key = r->key;
            s_rtc.models[slot].moved = 0;
            memcpy(s_rtc.models[slot].world, world, sizeof(world));
            s_rtc.models[slot].seen = s_rtc.frame;
            return false;
        }
    }
    if (probe == 16) return false;
    for (j = 0; j < 12; j++) {
        const float tol = (j % 4) == 3 ? 0.01f : 0.002f;   /* translation in world units, rotation in cosines */
        if (fabsf(world[j] - s_rtc.models[slot].world[j]) > tol) { s_rtc.models[slot].moved = s_rtc.frame; break; }
    }
    memcpy(s_rtc.models[slot].world, world, sizeof(world));
    s_rtc.models[slot].seen = s_rtc.frame;
    return s_rtc.models[slot].moved && s_rtc.frame - s_rtc.models[slot].moved < 240;
}

/* The main light: BLACK bakes its lighting, but the ambient coefficients that light its props and characters have a
 * linear term pointing at where most light comes from. Each such draw's, turned from its model's space into the
 * world (the camera's inverse times its own matrix), summed and smoothed over frames. */
static void kg_rtc_light(const float *vp)
{
    double ivp[16], sum[3] = { 0, 0, 0 }, len;
    unsigned i, j, k;
    if (!kg_rtc_inv4(vp, ivp)) return;
    for (i = 0; i < s_rtc.ndraws; i++) {
        const KrtCapDraw *r = &s_rtc.draws[i];
        double w[9], d[3];
        if (!r->sh) continue;
        /* the model-to-world rotation: (camera^-1 * draw matrix), upper 3x3 */
        for (j = 0; j < 3; j++)
            for (k = 0; k < 3; k++)
                w[j * 3 + k] = ivp[j * 4 + 0] * r->m[0 * 4 + k] + ivp[j * 4 + 1] * r->m[1 * 4 + k] +
                               ivp[j * 4 + 2] * r->m[2 * 4 + k] + ivp[j * 4 + 3] * r->m[3 * 4 + k];
        for (j = 0; j < 3; j++) d[j] = w[j * 3] * r->sh_l1[0] + w[j * 3 + 1] * r->sh_l1[1] + w[j * 3 + 2] * r->sh_l1[2];
        for (j = 0; j < 3; j++) sum[j] += d[j];
    }
    len = sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    if (len < 1e-6) return;
    for (j = 0; j < 3; j++) sum[j] /= len;
    if (!s_rtc.have_light) {
        for (j = 0; j < 3; j++) s_rtc.light[j] = (float)sum[j];
        s_rtc.have_light = true;
        fprintf(stderr, "[KRT] main light towards %.2f %.2f %.2f\n", sum[0], sum[1], sum[2]);
    } else {
        for (j = 0; j < 3; j++) s_rtc.light[j] = s_rtc.light[j] * 0.97f + (float)sum[j] * 0.03f;
        len = sqrt((double)s_rtc.light[0] * s_rtc.light[0] + (double)s_rtc.light[1] * s_rtc.light[1] + (double)s_rtc.light[2] * s_rtc.light[2]);
        if (len > 1e-6) for (j = 0; j < 3; j++) s_rtc.light[j] = (float)(s_rtc.light[j] / len);
    }
}

/* The frame's world is drawn (cs is the main target, or NULL at a flip): turn what was captured into world space,
 * hand it to the D3D12 side and run the passes there. */
static void kg_rtc_finish(KGpuSurf *cs, KGpuSurf *zs)
{
    const float *vp;
    double ivp[16];
    bool have_ivp;
    KrtCb cb;
    KrtFrame f;
    D3D11_MAPPED_SUBRESOURCE ms;
    ID3D11ShaderResourceView *none_srv = NULL;
    ID3D11UnorderedAccessView *none_uav = NULL;
    unsigned i;
    bool sized;
    if (s_rtc.state <= 0 || s_rtc.done) return;
    s_rtc.done = true;
    if (!s_rtc.used || !(vp = kg_rtc_camera()) || !kg_rtc_invert(vp, cb.inv)) return;
    memcpy(cb.c58, s_rtc.c58, sizeof(cb.c58));
    memcpy(cb.c59, s_rtc.c59, sizeof(cb.c59));
    memcpy(cb.scale, s_rtc.scale, sizeof(cb.scale));
    cb.n[0] = s_rtc.used; cb.n[1] = KRT_GEO_W; cb.n[2] = cb.n[3] = 0;
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)s_rtc.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) return;
    memcpy(ms.pData, &cb, sizeof(cb));
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)s_rtc.cb, 0);
    ID3D11DeviceContext_CSSetShader(G.ctx, s_rtc.cs, NULL, 0);
    ID3D11DeviceContext_CSSetConstantBuffers(G.ctx, 0, 1, &s_rtc.cb);
    ID3D11DeviceContext_CSSetShaderResources(G.ctx, 0, 1, &s_rtc.so_srv);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(G.ctx, 0, 1, &s_rtc.geo_uav, NULL);
    ID3D11DeviceContext_Dispatch(G.ctx, (s_rtc.used + 63) / 64, 1, 1);
    ID3D11DeviceContext_CSSetShaderResources(G.ctx, 0, 1, &none_srv);
    ID3D11DeviceContext_CSSetUnorderedAccessViews(G.ctx, 0, 1, &none_uav, NULL);
    ID3D11DeviceContext_CSSetShader(G.ctx, NULL, NULL, 0);
    if (s_rtc.check && ++s_rtc.frames % 300 == 0) kg_rtc_check(vp);

    memset(&f, 0, sizeof(f));
    f.history = cs ? cs->view : 0;
    s_rtc.frame++;
    have_ivp = kg_rtc_inv4(vp, ivp);
    for (i = 0; i < s_rtc.ndraws; i++) {
        s_rtc.first[i] = s_rtc.draws[i].first;
        s_rtc.count[i] = s_rtc.draws[i].count;
        s_rtc.tex0[i] = s_rtc.draws[i].rt_tex[0];
        s_rtc.tex1[i] = s_rtc.draws[i].rt_tex[1];
        s_rtc.flags[i] = s_rtc.draws[i].rt_flags;
        if (memcmp(s_rtc.draws[i].m, vp, sizeof(s_rtc.draws[i].m)) != 0 && have_ivp && kg_rtc_dynamic(&s_rtc.draws[i], ivp))
            s_rtc.flags[i] |= KRT_DYNAMIC;
    }
    f.nverts = s_rtc.used; f.ndraws = s_rtc.ndraws;
    f.first = s_rtc.first; f.count = s_rtc.count; f.tex0 = s_rtc.tex0; f.tex1 = s_rtc.tex1; f.flags = s_rtc.flags;
    memcpy(f.inv, cb.inv, sizeof(f.inv));
    memcpy(f.c58, cb.c58, sizeof(f.c58));
    memcpy(f.c59, cb.c59, sizeof(f.c59));
    memcpy(f.vp, vp, sizeof(f.vp));
    kg_rtc_light(vp);
    f.light[0] = s_rtc.light[0]; f.light[1] = s_rtc.light[1]; f.light[2] = s_rtc.light[2];
    f.light[3] = s_rtc.have_light ? s_rtc.shadow : 0.0f;
    f.ao = s_rtc.ao; f.ao_radius = s_rtc.ao_radius; f.gi = s_rtc.gi;
    f.zmax = 1.0f / s_rtc.scale[2];
    f.guest_w = 2.0f / s_rtc.scale[0]; f.guest_h = 2.0f / s_rtc.scale[1];
    sized = cs && kg_rtc_output(cs->hw, cs->hh);
    f.view = sized && s_rtc.view < 4 ? s_rtc.view : 0;    /* 4 is the lighting factor, shown here */
    /* reflections: when the env draws marked pixels this frame, at the picture's size and format */
    f.reflect = s_rtc.reflect && sized && s_ssr.used && s_ssr.w == cs->hw && s_ssr.h == cs->hh &&
                cs->dxfmt == DXGI_FORMAT_B8G8R8A8_UNORM;
    /* the picture so far: the reflections read it, and shading learns its brightness from it */
    f.calibrate = sized && cs->dxfmt == DXGI_FORMAT_B8G8R8A8_UNORM;
    if (f.calibrate) ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_rtc.in[0], (ID3D11Resource *)cs->tex);
    /* shadows, occlusion and bounce colour need the depth buffer, at the picture's size */
    f.lighting = f.calibrate && zs && zs->depth && zs->srv && zs->hw == cs->hw && zs->hh == cs->hh && s_rtc.ps_depth &&
                 s_rtc.ps_lit && (f.light[3] > 0.0f || f.ao > 0.0f || f.gi > 0.0f) && !f.view && post_init();
    if (f.lighting) post_pass(s_rtc.ps_depth, s_rtc.depth_rtv, cs->hw, cs->hh, zs->srv, NULL, NULL, NULL);
    if (f.reflect) {
        ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_rtc.in[1], (ID3D11Resource *)s_ssr.tex[0]);
        ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_rtc.in[2], (ID3D11Resource *)s_ssr.tex[2]);
    }
    if (!krt_frame(&f)) return;
    if (f.view >= 3) {
        if (++s_rtc.calib % 120 == 0) kg_rtc_calibration();
    } else if (f.view && s_rtc.ps_view && post_init()) {
        post_pass(s_rtc.ps_view, cs->rtv, cs->hw, cs->hh, s_rtc.out_srv, NULL, NULL, NULL);
        kg_cache_invalidate();
    } else {
        if (f.lighting) {
            if (s_rtc.view == 4) post_pass(s_rtc.ps_view_half, cs->rtv, cs->hw, cs->hh, s_rtc.lit_srv, NULL, NULL, NULL);
            else post_pass(s_rtc.ps_lit, cs->rtv, cs->hw, cs->hh, s_rtc.pic_srv, s_rtc.lit_srv, NULL, NULL);
            kg_cache_invalidate();
        }
        if (f.reflect) {
            s_ssr.rt_srv = s_rtc.out_srv;
            s_ssr.rt_valid = true;
        }
    }
}

/* Called for every draw once its surfaces are known: the HUD's first draw ends the world. */
static void kg_rtc_note_draw(const KPixelState *ps, const KGpuSurf *cs, KGpuSurf *zs)
{
    if (s_rtc.state > 0 && !s_rtc.done && s_rtc.ndraws >= 4 && post_main_target(cs) &&
        ps->depth_test && ps->depth_func == 7 && !ps->depth_write)
        kg_rtc_finish((KGpuSurf *)cs, zs);
}

/* The end of a world (a frame, or a split-screen view): its effects now if the HUD did not start them, on `cs`. */
static void kg_rtc_end_frame(KGpuSurf *cs, KGpuSurf *zs)
{
    if (!(s_rtc.state > 0 && !s_rtc.done && s_rtc.ndraws >= 4) || !post_main_target(cs)) cs = zs = NULL;
    kg_rtc_finish(cs, zs);
    s_rtc.used = 0;
    s_rtc.ndraws = 0;
    s_rtc.overflow = false;
    s_rtc.done = false;
}

/* At the flip: a frame with no HUD (photo mode, some cut scenes) has its effects now, on the target it shows. */
static void kg_rtc_before_flip(const KelvinTarget *t, unsigned int color_format)
{
    KGpuSurf *cs = NULL, *zs = NULL;
    if (s_rtc.state > 0 && !s_rtc.done && s_rtc.ndraws >= 4) {
        cs = surf_get(t, false, color_format);
        zs = t->zeta_va ? surf_find(t->zeta_va, true) : NULL;
        if (!post_main_target(cs)) cs = zs = NULL;
    }
    kg_rtc_finish(cs, zs);
    s_rtc.used = 0;
    s_rtc.ndraws = 0;
    s_rtc.overflow = false;
    s_rtc.done = false;
}
