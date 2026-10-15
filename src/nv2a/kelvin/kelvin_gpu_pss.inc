/* Specialised pixel shaders, included by kelvin_gpu.c after kelvin_gpu_vsh.inc.
 *
 * The ubershader in kelvin_gpu_hlsl.inc interprets the register combiners at run time: dynamic register
 * indexing, a loop over stages, nested mapping chains, four texture samples whether used or not. Most of the GPU
 * time of a frame went into that interpretation. Here the same program is written out for one combiner
 * configuration: registers are plain locals, the stage loop is unrolled, every mapping and output selection is
 * folded, and only the textures a draw uses are sampled. Everything that varies per draw but is not part of the
 * configuration (stage constants, fog colour, texture size reciprocals, colour keys, alpha reference) stays in
 * the PSConst constant buffer, so one shader serves every draw with the same configuration.
 *
 * Shaders compile on the worker threads shared with the vertex programs; until one is ready the draw uses the
 * ubershader, so there are no stalls, only a slower first few frames. Compiled code is cached on disk next to the
 * vertex programs (ps_<key>_<source hash>.bin). RECOMP_KGPU_PS_SPEC=0 turns this off.
 *
 * The generator mirrors the arithmetic and ordering of the ubershader statement by statement (every input of a
 * stage is read before any output of it is written; the outputs are written rgb-ab, rgb-cd, rgb-mux, then the
 * alpha ones); RECOMP_KGPU_COMPARE shows the difference to the CPU renderer for either path. */

typedef struct KGpuPs {
    bool used;
    uint64_t key;
    volatile LONG state;               /* 0 compiling, 1 ready, 2 unusable */
    ID3D11PixelShader *ps;
    char *source;
} KGpuPs;

#define KG_PS_MAX 2048
/* PSConst.texflags: this stage is an environment map the draw writes screen-space reflection targets for */
#define PS_TEX_SSR 0x100u
static KGpuPs s_ps[KG_PS_MAX];
static int s_ps_spec = -1;

/* The ubershader's map3/map1 for a known modifier. `x` is the operand expression. */
static void ps_map(char *out, size_t cap, const char *x, uint32_t mod)
{
    switch (mod & 0xE0u) {
    case 0x00: snprintf(out, cap, "max(%s, 0.0)", x); break;
    case 0x20: snprintf(out, cap, "(1.0 - clamp(%s, 0.0, 1.0))", x); break;
    case 0x40: snprintf(out, cap, "(2.0 * max(%s, 0.0) - 1.0)", x); break;
    case 0x60: snprintf(out, cap, "(-2.0 * max(%s, 0.0) + 1.0)", x); break;
    case 0x80: snprintf(out, cap, "(max(%s, 0.0) - 0.5)", x); break;
    case 0xA0: snprintf(out, cap, "(-max(%s, 0.0) + 0.5)", x); break;
    case 0xE0: snprintf(out, cap, "(-%s)", x); break;
    default:   snprintf(out, cap, "%s", x); break;
    }
}

/* mapout: x scaled / biased by the output mapping bits (rfl & 0x38). */
static void ps_mapout(char *out, size_t cap, const char *x, uint32_t m)
{
    switch (m & 0x38u) {
    case 0x08: snprintf(out, cap, "(%s - 0.5)", x); break;
    case 0x10: snprintf(out, cap, "(%s * 2.0)", x); break;
    case 0x18: snprintf(out, cap, "((%s - 0.5) * 2.0)", x); break;
    case 0x20: snprintf(out, cap, "(%s * 4.0)", x); break;
    case 0x30: snprintf(out, cap, "(%s * 0.5)", x); break;
    default:   snprintf(out, cap, "%s", x); break;
    }
}

static bool ps_writable(uint32_t r) { return r == 4u || r == 5u || (r >= 8u && r <= 13u); }

/* Input operand of a combiner: register `in & 15`, alpha replicated when bit 4 is set, mapped. */
static void ps_input3(char *out, size_t cap, uint32_t in)
{
    char x[40];
    snprintf(x, sizeof(x), "r%u.%s", in & 15u, (in & 0x10u) ? "aaa" : "rgb");
    ps_map(out, cap, x, in & 0xE0u);
}

static void ps_input1(char *out, size_t cap, uint32_t in)
{
    char x[40];
    snprintf(x, sizeof(x), "r%u.%s", in & 15u, (in & 0x10u) ? "a" : "b");
    ps_map(out, cap, x, in & 0xE0u);
}

static uint64_t ps_key_of(const PSConst *c)
{
    uint64_t h = 1469598103934665603ULL;
    uint32_t nst = c->gen[0] > 8 ? 8 : c->gen[0];
    unsigned int i, k;
#define PS_MIX(v) h = (h ^ (uint64_t)(uint32_t)(v)) * 1099511628211ULL
    PS_MIX(nst); PS_MIX(c->gen[1]); PS_MIX(c->ofs[0]); PS_MIX(c->gen[3]);
    for (k = 0; k < 4; k++) { PS_MIX(c->texmode[k]); PS_MIX(c->texflags[k]); PS_MIX(c->fin_abcd[k]); PS_MIX(c->fin_efg[k]); }
    for (i = 0; i < nst; i++) {
        for (k = 0; k < 4; k++) { PS_MIX(c->stage[i].rgb_in[k]); PS_MIX(c->stage[i].alpha_in[k]); }
        PS_MIX(c->stage[i].outs[0]); PS_MIX(c->stage[i].outs[1]);
    }
#undef PS_MIX
    return h ? h : 1;
}

/* The shader for one configuration. Returns a malloc'd string (the caller prepends the common declarations). */
static char *ps_generate(const PSConst *c)
{
    StrBuf b = { NULL, 0, 0 }, mn = { NULL, 0, 0 }, cmb = { NULL, 0, 0 }, tl = { NULL, 0, 0 };
    uint32_t env = 0;
    uint32_t nst = c->gen[0] > 8 ? 8 : c->gen[0], fflags = c->fin_efg[3], func = c->gen[1];
    unsigned int i, k;
    char ex[96], ey[96], buf[256];
    if (c->gen[3]) sb_add(&b, "//KG_FLAT\n");
    sb_add(&b,
        "Texture2D<float4> Tex0 : register(t0);\nTexture2D<float4> Tex1 : register(t1);\n"
        "Texture2D<float4> Tex2 : register(t2);\nTexture2D<float4> Tex3 : register(t3);\n"
        "SamplerState Smp0 : register(s0);\nSamplerState Smp1 : register(s1);\n"
        "SamplerState Smp2 : register(s2);\nSamplerState Smp3 : register(s3);\n");
    for (i = 0; i < 4; i++) if (c->texflags[i] & PS_TEX_SSR) env |= 1u << i;
    /* textures: only the ones in use are sampled; the sample sits outside any branch so its derivatives are defined */
    for (i = 0; i < 4; i++) {
        uint32_t mode = c->texmode[i], fl = c->texflags[i];
        const char *comp = i == 0 ? "x" : i == 1 ? "y" : i == 2 ? "z" : "w";
        if (mode == 1u && (fl & 1u)) {
            sb_add(&mn,
                "    float2 uv%u = float2(0.0, 0.0);\n"
                "    if (v.t%u.w != 0.0) uv%u = v.t%u.xy / v.t%u.w;\n", i, i, i, i, i);
            if (fl & 2u) sb_add(&mn, "    uv%u *= texrcp[%u].xy;\n", i, i);
            sb_add(&mn,
                "    float4 s%u = Tex%u.Sample(Smp%u, uv%u);\n"
                "    float4 t%u = float4(0.0, 0.0, 0.0, 1.0);\n"
                "    if (v.t%u.w != 0.0) {\n"
                "        t%u = s%u;\n", i, i, i, i, i, i, i, i);
            if (fl & 4u) sb_add(&mn, "        if (t%u.a == 0.0) killed = true;\n", i);
            if ((fl >> 4) & 3u) {
                uint32_t ckm = (fl >> 4) & 3u;
                sb_add(&mn,
                    "        {\n"
                    "            uint4 q = (uint4)(saturate(t%u) * 255.0 + 0.5);\n"
                    "            uint cc = (q.a << 24) | (q.r << 16) | (q.g << 8) | q.b;\n"
                    "            if ((cc & ckeymask.%s) == (ckey.%s & ckeymask.%s)) {\n", i, comp, comp, comp);
                if (ckm == 3u) sb_add(&mn, "                killed = true;\n");
                else if (ckm == 1u) sb_add(&mn, "                t%u.a = 0.0;\n", i);
                else sb_add(&mn, "                t%u = float4(0.0, 0.0, 0.0, 0.0);\n", i);
                sb_add(&mn, "            }\n        }\n");
            }
            sb_add(&mn, "    }\n");
        } else {
            sb_add(&mn, "    float4 t%u = float4(0.0, 0.0, 0.0, 1.0);\n", i);
        }
    }
    sb_add(&cmb,
        "    float4 r0 = float4(0, 0, 0, 0);\n"
        "    float4 r1 = stage[0].c0;\n"
        "    float4 r2 = stage[0].c1;\n"
        "    float4 r3 = float4(fogcolor.rgb, saturate(v.misc.x));\n"
        "    float4 r4 = v.d0;\n"
        "    float4 r5 = v.d1;\n"
        "    float4 r6 = float4(0, 0, 0, 0);\n"
        "    float4 r7 = float4(0, 0, 0, 0);\n"
        "    float4 r8 = t0; float4 r9 = t1; float4 r10 = t2; float4 r11 = t3;\n"
        "    float4 r12 = float4(0.0, 0.0, 0.0, %s);\n"
        "    float4 r13 = float4(0, 0, 0, 0);\n"
        "    float4 r14 = float4(0, 0, 0, 0);\n"
        "    float4 r15 = float4(0, 0, 0, 0);\n", c->texmode[0] != 0u ? "t0.a" : "1.0");

    for (i = 0; i < nst; i++) {
        uint32_t ro = c->stage[i].outs[0], ao = c->stage[i].outs[1];
        uint32_t rcd = ro & 15u, rab = (ro >> 4) & 15u, rmx = (ro >> 8) & 15u, rfl = ro >> 12, rmap = rfl & 0x38u;
        uint32_t acd = ao & 15u, aab = (ao >> 4) & 15u, amx = (ao >> 8) & 15u, afl = ao >> 12, amap = afl & 0x38u;
        sb_add(&cmb, "    {\n");
        sb_add(&cmb, "        r1 = stage[%u].c0;\n", (fflags & 0x20u) ? i : 0u);
        sb_add(&cmb, "        r2 = stage[%u].c1;\n", (fflags & 0x40u) ? i : 0u);
        sb_add(&cmb, "        {\n            float3 a1 = %s;\n            float3 b1 = %s;\n            float3 sm = a1 + b1;\n",
               (fflags & 4u) ? "1.0 - r5.rgb" : "r5.rgb", (fflags & 8u) ? "1.0 - r12.rgb" : "r12.rgb");
        if (fflags & 2u) sb_add(&cmb, "            sm = clamp(sm, 0.0, 1.0);\n");
        sb_add(&cmb, "            r14 = float4(sm, 0.0);\n        }\n");
        for (k = 0; k < 4; k++) {
            ps_input3(ex, sizeof(ex), c->stage[i].rgb_in[k]);
            ps_input1(ey, sizeof(ey), c->stage[i].alpha_in[k]);
            sb_add(&cmb, "        float3 ra%u = %s;\n        float aa%u = %s;\n", k, ex, k, ey);
        }
        sb_add(&cmb, "        float3 ab3 = %s;\n", (rfl & 2u) ? "dot(ra0, ra1).xxx" : "ra0 * ra1");
        sb_add(&cmb, "        float3 cd3 = %s;\n", (rfl & 1u) ? "dot(ra2, ra3).xxx" : "ra2 * ra3");
        if ((rfl & 4u) || (afl & 4u)) {
            if (fflags & 0x10u) sb_add(&cmb, "        bool selr = r12.a >= 0.5;\n");
            else sb_add(&cmb, "        bool selr = ((((uint)(r12.a * 255.0)) & 1u) == 1u);\n");
        }
        sb_add(&cmb, "        float3 mx3 = %s;\n", (rfl & 4u) ? "(selr ? cd3 : ab3)" : "(ab3 + cd3)");
        ps_mapout(ex, sizeof(ex), "ab3", rmap); sb_add(&cmb, "        float3 rab3 = clamp(%s, -1.0, 1.0);\n", ex);
        ps_mapout(ex, sizeof(ex), "cd3", rmap); sb_add(&cmb, "        float3 rcd3 = clamp(%s, -1.0, 1.0);\n", ex);
        ps_mapout(ex, sizeof(ex), "mx3", rmap); sb_add(&cmb, "        float3 rmx3 = clamp(%s, -1.0, 1.0);\n", ex);
        sb_add(&cmb, "        float ab1 = aa0 * aa1;\n        float cd1 = aa2 * aa3;\n");
        sb_add(&cmb, "        float mx1 = %s;\n", (afl & 4u) ? "(selr ? cd1 : ab1)" : "(ab1 + cd1)");
        ps_mapout(ex, sizeof(ex), "ab1", amap); sb_add(&cmb, "        float rab1 = clamp(%s, -1.0, 1.0);\n", ex);
        ps_mapout(ex, sizeof(ex), "cd1", amap); sb_add(&cmb, "        float rcd1 = clamp(%s, -1.0, 1.0);\n", ex);
        ps_mapout(ex, sizeof(ex), "mx1", amap); sb_add(&cmb, "        float rmx1 = clamp(%s, -1.0, 1.0);\n", ex);
        if (ps_writable(rab)) {
            sb_add(&cmb, "        r%u.rgb = rab3;\n", rab);
            if (rfl & 0x80u) sb_add(&cmb, "        r%u.a = rab3.b;\n", rab);
        }
        if (ps_writable(rcd)) {
            sb_add(&cmb, "        r%u.rgb = rcd3;\n", rcd);
            if (rfl & 0x40u) sb_add(&cmb, "        r%u.a = rcd3.b;\n", rcd);
        }
        if (ps_writable(rmx)) sb_add(&cmb, "        r%u.rgb = rmx3;\n", rmx);
        if (ps_writable(aab)) sb_add(&cmb, "        r%u.a = rab1;\n", aab);
        if (ps_writable(acd)) sb_add(&cmb, "        r%u.a = rcd1;\n", acd);
        if (ps_writable(amx)) sb_add(&cmb, "        r%u.a = rmx1;\n", amx);
        sb_add(&cmb, "    }\n");
        (void)buf;
    }

    sb_add(&cmb, "    float4 frag;\n");
    if (fflags & 1u) {
        uint32_t ine = c->fin_efg[0], inf = c->fin_efg[1], ing = c->fin_efg[2];
        sb_add(&cmb, "    {\n        r1 = final_c0;\n        r2 = final_c1;\n");
        sb_add(&cmb, "        {\n            float3 a1 = %s;\n            float3 b1 = %s;\n            float3 sm = a1 + b1;\n",
               (fflags & 4u) ? "1.0 - r5.rgb" : "r5.rgb", (fflags & 8u) ? "1.0 - r12.rgb" : "r12.rgb");
        if (fflags & 2u) sb_add(&cmb, "            sm = clamp(sm, 0.0, 1.0);\n");
        sb_add(&cmb, "            r14 = float4(sm, 0.0);\n        }\n");
        ps_input3(ex, sizeof(ex), ine); ps_input3(ey, sizeof(ey), inf);
        sb_add(&cmb, "        float3 e3 = %s;\n        float3 f3 = %s;\n        r15 = float4(e3 * f3, 0.0);\n", ex, ey);
        ps_input3(ex, sizeof(ex), c->fin_abcd[0]); sb_add(&cmb, "        float3 a3 = %s;\n", ex);
        ps_input3(ex, sizeof(ex), c->fin_abcd[1]); sb_add(&cmb, "        float3 b3 = %s;\n", ex);
        ps_input3(ex, sizeof(ex), c->fin_abcd[2]); sb_add(&cmb, "        float3 c3 = %s;\n", ex);
        ps_input3(ex, sizeof(ex), c->fin_abcd[3]); sb_add(&cmb, "        float3 d3 = %s;\n", ex);
        ps_input1(ex, sizeof(ex), ing); sb_add(&cmb, "        float g1 = %s;\n", ex);
        sb_add(&cmb, "        frag = float4(d3 + c3 * (1.0 - a3) + b3 * a3, g1);\n    }\n");
    } else {
        sb_add(&cmb, "    frag = r12;\n");
    }
    if (func != 7u) {
        static const char *const cmp[] = { NULL, "fa < ref", "fa == ref", "fa <= ref", "fa > ref", "fa != ref", "fa >= ref" };
        if (func == 0u) {
            sb_add(&tl, "    discard;\n");
        } else if (func < 7u) {
            sb_add(&tl, "    {\n        int fa = (int)floor(frag.a * 255.0 + 0.5);\n        int ref = (int)gen.z;\n"
                       "        if (!(%s)) discard;\n    }\n", cmp[func]);
        }
    }
    sb_add(&tl, "    float4 o = saturate(frag);\n");
    if (c->ofs[0] == 1u) sb_add(&tl, "    o.a = 1.0;\n");
    else if (c->ofs[0] == 2u) sb_add(&tl, "    o.a = 0.0;\n");
    /* the combiner as a function, so the SSR variant can run it with the env texel at 0 and at 1 as well */
    sb_add(&b, "float4 kg_combine(VSOut v, float4 t0, float4 t1, float4 t2, float4 t3) {\n%s    return frag;\n}\n",
           cmb.p ? cmb.p : "");
    if (env) {
        /* Screen-space reflections (kelvin_gpu_post.inc): besides the colour, how much each env map texel adds (the
         * combiner's change from texel 0 to texel 1), what the sampled one added, and where the pixel is. */
        sb_add(&b,
            "cbuffer SSRV : register(b4) { float4 ssr_view; };   // x,y: camera half-frustum; z,w: 1 / target size\n"
            "struct KgEnvOut { float4 c : SV_Target0; float4 w : SV_Target1; float4 e : SV_Target2; float4 g : SV_Target3; };\n"
            "KgEnvOut ps_main(VSOut v) {\n");
    } else {
        sb_add(&b, "float4 ps_main(VSOut v) : SV_Target {\n");
    }
    sb_add(&b, "    bool killed = false;\n%s    if (killed) discard;\n    float4 frag = kg_combine(v, t0, t1, t2, t3);\n%s",
           mn.p ? mn.p : "", tl.p ? tl.p : "");
    if (!env) {
        sb_add(&b, "    return o;\n}\n");
    } else {
        sb_add(&b, "    KgEnvOut r;\n    r.c = o;\n    float3 w = float3(0, 0, 0), e = float3(0, 0, 0);\n");
        for (i = 0; i < 4; i++) {
            static const char *const tn[4] = { "t0", "t1", "t2", "t3" };
            char a0[4][16], a1[4][16];
            if (!(env & (1u << i))) continue;
            for (k = 0; k < 4; k++) {
                snprintf(a0[k], sizeof(a0[k]), "%s", k == i ? "tz" : tn[k]);
                snprintf(a1[k], sizeof(a1[k]), "%s", k == i ? "tn" : tn[k]);
            }
            sb_add(&b,
                "    {\n"
                "        float4 tz = float4(0.0, 0.0, 0.0, %s.a), tn = float4(1.0, 1.0, 1.0, %s.a);\n"
                "        float3 f0 = saturate(kg_combine(v, %s, %s, %s, %s)).rgb;\n"
                "        float3 f1 = saturate(kg_combine(v, %s, %s, %s, %s)).rgb;\n"
                "        w += f1 - f0;\n"
                "        e += o.rgb - f0;\n"
                "    }\n", tn[i], tn[i], a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]);
        }
        sb_add(&b,
            "    r.w = float4(max(w, 0.0), o.a);\n"
            "    r.e = float4(e, o.a);\n"
            "    {\n"
            "        // view-space position as the post passes reconstruct it (perspective depth 1 - near / z)\n"
            "        float vz = min(4096.0, 1.0 / max(0.0001, 1.0 - v.pos.z));\n"
            "        float2 ndc = float2(v.pos.x * ssr_view.z * 2.0 - 1.0, 1.0 - v.pos.y * ssr_view.w * 2.0);\n"
            "        float3 p = float3(ndc * ssr_view.xy * vz, vz);\n"
            "        float3 n = cross(ddx(p), ddy(p));\n"
            "        float n2 = dot(n, n);\n"
            "        n = n2 > 1e-20 ? n * rsqrt(n2) : float3(0, 0, -1);\n"
            "        if (dot(n, p) > 0.0) n = -n;\n"
            "        r.g = float4(v.pos.z, n);\n"
            "    }\n"
            "    return r;\n}\n");
    }
    free(mn.p); free(cmb.p); free(tl.p);
    return b.p;
}

static void ps_cache_path(char *out, size_t cap, const KGpuPs *e)
{
    const char *dir = getenv("RECOMP_KGPU_CACHE");
    uint32_t sh = e->source ? fnv_bytes(e->source, strlen(e->source)) : 0;
    snprintf(out, cap, "%s/ps_%08X%08X_%08X.bin", dir && *dir ? dir : "kgpu-cache",
             (uint32_t)(e->key >> 32), (uint32_t)e->key, sh);
}

static bool ps_try_cache(KGpuPs *e)
{
    char path[512];
    FILE *f;
    void *buf;
    long n;
    ps_cache_path(path, sizeof(path), e);
    f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = n > 0 ? malloc((size_t)n) : NULL;
    if (buf && fread(buf, 1, (size_t)n, f) == (size_t)n &&
        SUCCEEDED(ID3D11Device_CreatePixelShader(G.dev, buf, (SIZE_T)n, NULL, &e->ps))) {
        free(buf); fclose(f);
        return true;
    }
    free(buf); fclose(f);
    return false;
}

static void ps_compile(KGpuPs *e)
{
    ID3DBlob *code = NULL, *errors = NULL;
    HRESULT hr;
    size_t lc = strlen(kgpu_hlsl_common), ls = strlen(e->source);
    char *full = (char *)malloc(lc + ls + 1);
    long long t0 = qpc_now();
    if (!full) { InterlockedExchange(&e->state, 2); return; }
    if (!strncmp(e->source, "//KG_FLAT", 9)) {
        /* the flat-shading variant: defined before the common declarations see it */
        static const char flat_def[] = "#define KG_FLAT 1\n";
        size_t lf = sizeof(flat_def) - 1;
        char *g = (char *)realloc(full, lf + lc + ls + 1);
        if (!g) { free(full); InterlockedExchange(&e->state, 2); return; }
        full = g;
        memcpy(full, flat_def, lf);
        memcpy(full + lf, kgpu_hlsl_common, lc);
        memcpy(full + lf + lc, e->source, ls + 1);
        lc += lf;
    } else {
        memcpy(full, kgpu_hlsl_common, lc);
        memcpy(full + lc, e->source, ls + 1);
    }
    hr = G.compile(full, lc + ls, "kgpu_ps", NULL, NULL, "ps_main", "ps_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (SUCCEEDED(hr) &&
        SUCCEEDED(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code), NULL, &e->ps))) {
        char path[512];
        FILE *f;
        CreateDirectoryA(getenv("RECOMP_KGPU_CACHE") && *getenv("RECOMP_KGPU_CACHE") ? getenv("RECOMP_KGPU_CACHE") : "kgpu-cache", NULL);
        ps_cache_path(path, sizeof(path), e);
        f = fopen(path, "wb");
        if (f) { fwrite(ID3D10Blob_GetBufferPointer(code), 1, ID3D10Blob_GetBufferSize(code), f); fclose(f); }
        InterlockedExchange(&e->state, 1);
        fprintf(stderr, "[KGPU] pixel program %016llX compiled in %.0f ms\n", (unsigned long long)e->key,
                (double)(qpc_now() - t0) * 1000.0 / (double)G.qpf);
    } else {
        fprintf(stderr, "[KGPU] pixel program %016llX did not compile: %s\n", (unsigned long long)e->key,
                errors ? (const char *)ID3D10Blob_GetBufferPointer(errors) : "(no log)");
        if (getenv("RECOMP_KGPU_SHADER_LOG")) fprintf(stderr, "%s\n", e->source);
        InterlockedExchange(&e->state, 2);
    }
    if (errors) ID3D10Blob_Release(errors);
    if (code) ID3D10Blob_Release(code);
    free(full);
}

/* The specialised shader for this draw's configuration once it is ready, else NULL (use the ubershader). */
static ID3D11PixelShader *ps_lookup(const PSConst *c)
{
    uint64_t key;
    unsigned int i, slot;
    KGpuPs *e = NULL;
    if (s_ps_spec < 0) {
        const char *env = getenv("RECOMP_KGPU_PS_SPEC");
        s_ps_spec = !(env && *env && *env == '0');
    }
    if (!s_ps_spec) return NULL;
    key = ps_key_of(c);
    slot = (unsigned int)(key % KG_PS_MAX);
    for (i = 0; i < KG_PS_MAX; i++, slot = (slot + 1) % KG_PS_MAX) {
        KGpuPs *q = &s_ps[slot];
        if (!q->used) { e = q; break; }
        if (q->key == key) return q->state == 1 ? q->ps : NULL;
    }
    if (!e) return NULL;
    e->used = true; e->key = key; e->state = 0;
    e->source = ps_generate(c);
    if (!e->source) { e->state = 2; return NULL; }
    if (ps_try_cache(e)) {
        e->state = 1;
        return e->ps;
    }
    kg_job_post(1, e);
    return NULL;
}

static void kg_job_ps(void *p) { ps_compile((KGpuPs *)p); }
