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
    ID3D11ShaderResourceView *out_srv;
    HANDLE out_h;
    unsigned out_w, out_hh;
    ID3D11PixelShader *ps_view;
    bool view;                          /* RECOMP_KGPU_RT_VIEW=1 */
    uint32_t first[KRT_DRAWS_MAX], count[KRT_DRAWS_MAX];
    uint8_t opaque[KRT_DRAWS_MAX];
} s_rtc;

static const char kg_rtc_view_hlsl[] =
    "Texture2D<float4> rt : register(t0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target { return float4(rt.Load(int3(p.xy, 0)).rgb, 1.0); }\n";

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
    s_rtc.view = e && *e && *e != '0';
    code = compile_hlsl("", kg_rtc_view_hlsl, "main", "ps_5_0");
    if (code) {
        ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code), NULL, &s_rtc.ps_view);
        ID3D10Blob_Release(code);
    }
    fprintf(stderr, "[KRT] geometry capture on: up to %u vertices a frame\n", KRT_SO_VERTS);
    return true;
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
        s_rtc.done || kg_viewmodel_scale(d->gvs->hash) != 1.0f || !kg_rtc_on())
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

/* The texture the D3D12 passes write, at the main target's raster size; made again when that size changes. */
static bool kg_rtc_output(unsigned w, unsigned h)
{
    D3D11_TEXTURE2D_DESC td;
    IDXGIResource1 *r1 = NULL;
    if (s_rtc.out && s_rtc.out_w == w && s_rtc.out_hh == h) return true;
    SAFE_RELEASE(s_rtc.out_srv); SAFE_RELEASE(s_rtc.out);
    if (s_rtc.out_h) { CloseHandle(s_rtc.out_h); s_rtc.out_h = NULL; }
    s_rtc.out_w = s_rtc.out_hh = 0;
    memset(&td, 0, sizeof(td));
    td.Width = w; td.Height = h; td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &s_rtc.out)) ||
        FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)s_rtc.out, NULL, &s_rtc.out_srv)) ||
        FAILED(ID3D11Texture2D_QueryInterface(s_rtc.out, &IID_IDXGIResource1, (void **)&r1)) ||
        FAILED(IDXGIResource1_CreateSharedHandle(r1, NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, NULL, &s_rtc.out_h)) ||
        !krt_set_output(s_rtc.out_h, w, h)) {
        if (r1) IDXGIResource1_Release(r1);
        SAFE_RELEASE(s_rtc.out_srv); SAFE_RELEASE(s_rtc.out);
        return false;
    }
    IDXGIResource1_Release(r1);
    s_rtc.out_w = w; s_rtc.out_hh = h;
    return true;
}

/* The frame's world is drawn (cs is the main target, or NULL at a flip): turn what was captured into world space,
 * hand it to the D3D12 side and run the passes there. */
static void kg_rtc_finish(KGpuSurf *cs)
{
    const float *vp;
    KrtCb cb;
    KrtFrame f;
    D3D11_MAPPED_SUBRESOURCE ms;
    ID3D11ShaderResourceView *none_srv = NULL;
    ID3D11UnorderedAccessView *none_uav = NULL;
    unsigned i;
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
    for (i = 0; i < s_rtc.ndraws; i++) {
        s_rtc.first[i] = s_rtc.draws[i].first;
        s_rtc.count[i] = s_rtc.draws[i].count;
        s_rtc.opaque[i] = s_rtc.draws[i].alpha_test ? 0 : 1;
    }
    f.nverts = s_rtc.used; f.ndraws = s_rtc.ndraws;
    f.first = s_rtc.first; f.count = s_rtc.count; f.opaque = s_rtc.opaque;
    memcpy(f.inv, cb.inv, sizeof(f.inv));
    memcpy(f.c58, cb.c58, sizeof(f.c58));
    memcpy(f.c59, cb.c59, sizeof(f.c59));
    f.guest_w = 2.0f / s_rtc.scale[0]; f.guest_h = 2.0f / s_rtc.scale[1];
    f.view = s_rtc.view && cs && kg_rtc_output(cs->hw, cs->hh);
    if (krt_frame(&f) && f.view && s_rtc.ps_view && post_init()) {
        post_pass(s_rtc.ps_view, cs->rtv, cs->hw, cs->hh, s_rtc.out_srv, NULL, NULL, NULL);
        kg_cache_invalidate();
    }
}

/* Called for every draw once its surfaces are known: the HUD's first draw ends the world. */
static void kg_rtc_note_draw(const KPixelState *ps, const KGpuSurf *cs)
{
    if (s_rtc.state > 0 && !s_rtc.done && s_rtc.ndraws >= 4 && post_main_target(cs) &&
        ps->depth_test && ps->depth_func == 7 && !ps->depth_write)
        kg_rtc_finish((KGpuSurf *)cs);
}

static void kg_rtc_before_flip(void)
{
    kg_rtc_finish(NULL);
    s_rtc.used = 0;
    s_rtc.ndraws = 0;
    s_rtc.overflow = false;
    s_rtc.done = false;
}
