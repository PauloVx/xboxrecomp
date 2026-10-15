/* The GPU vertex path of the Kelvin D3D11 backend (included by kelvin_gpu.c).
 *
 * A transform program (nv2a_vsh_cpu's parsed steps, straight-line code) becomes
 * an HLSL vertex shader that also fetches and decodes the draw's attributes from
 * a raw copy of the guest vertex data. Semantics follow nv2a_vsh_cpu.c and the
 * epilogue of kelvin_vsh.c (1/16 pixel position snapping, w clamping, fog factor,
 * colour clamping), so the shaded vertices equal the CPU stage's.
 */
#include "nv2a_vsh_cpu/nv2a_vsh_disassembler.h"

static const char kgpu_vsh_head[] =
"struct VSOut {\n"
"    float4 pos : SV_Position;\n"
"    float4 d0  : TEXCOORD0;\n"
"    float4 d1  : TEXCOORD1;\n"
"    float4 misc: TEXCOORD2;\n"
"    float4 t0  : TEXCOORD3;\n"
"    float4 t1  : TEXCOORD4;\n"
"    float4 t2  : TEXCOORD5;\n"
"    float4 t3  : TEXCOORD6;\n"
"};\n"
"cbuffer VSG : register(b1) {\n"
"    float4 g_scale;\n"       /* x 2/guestW, y 2/guestH, z 1/zmax, w 2^64 */
"    float4 g_fog;\n"         /* x fog param 0, y fog param 1 */
"    uint4  g_flags;\n"       /* x fog enable, y fog mode, z specular enable, w ignore specular alpha */
"    float4 g_vm;\n"          /* x viewmodel scale about the viewport centre c[-37] */
"};\n"
"cbuffer VCB : register(b2) { float4 C[192]; };\n"
"cbuffer ACB : register(b3) { uint4 adesc[16]; float4 inl[16]; };\n"
"Buffer<uint> VAB : register(t9);\n"
"\n"
"uint LoadU32(uint addr) {\n"
"    uint a0 = addr & ~3u;\n"
"    uint sh = (addr & 3u) * 8u;\n"
"    uint lo = VAB[a0 >> 2];\n"
"    if (sh == 0u) return lo;\n"
"    uint hi = VAB[(a0 >> 2) + 1u];\n"
"    return (lo >> sh) | (hi << (32u - sh));\n"
"}\n"
"uint LoadU8(uint addr) { return (VAB[addr >> 2] >> ((addr & 3u) * 8u)) & 0xFFu; }\n"
"int LoadS16(uint addr) { return (int)(LoadU32(addr) << 16) >> 16; }\n"
"float4 FetchAttr(uint a, uint e) {\n"
"    uint4 d = adesc[a];\n"
"    float4 r = float4(0.0, 0.0, 0.0, 1.0);\n"
"    if (d.z == 0u) return inl[a];\n"
"    uint addr = d.x + e * d.y;\n"
"    if (d.z == 1u) {\n"
"        uint w = LoadU32(addr);\n"
"        r = float4((float)((w >> 16) & 255u), (float)((w >> 8) & 255u), (float)(w & 255u), (float)(w >> 24)) / 255.0;\n"
"    } else if (d.z == 2u) {\n"
"        for (uint i = 0; i < 4u; i++) if (i < d.w) r[i] = (float)LoadU8(addr + i) / 255.0;\n"
"    } else if (d.z == 3u) {\n"
"        for (uint i = 0; i < 4u; i++) if (i < d.w) r[i] = max(-1.0, (float)LoadS16(addr + i * 2u) / 32767.0);\n"
"    } else if (d.z == 4u) {\n"
"        for (uint i = 0; i < 4u; i++) if (i < d.w) r[i] = asfloat(LoadU32(addr + i * 4u));\n"
"    } else if (d.z == 5u) {\n"
"        for (uint i = 0; i < 4u; i++) if (i < d.w) r[i] = (float)LoadS16(addr + i * 2u);\n"
"    } else if (d.z == 6u) {\n"
"        uint v = LoadU32(addr);\n"
"        int x = (int)(v << 21) >> 21;\n"
"        int y = (int)((v >> 11) << 21) >> 21;\n"
"        int z = (int)v >> 22;\n"
"        r = float4((float)x / 1023.0, (float)y / 1023.0, (float)z / 511.0, 1.0);\n"
"    }\n"
"    return r;\n"
"}\n"
"\n"
"float vm(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }\n"
"float4 vmul(float4 a, float4 b) { return float4(vm(a.x, b.x), vm(a.y, b.y), vm(a.z, b.z), vm(a.w, b.w)); }\n"
"float4 vdp3(float4 a, float4 b) { return (vm(a.x, b.x) + vm(a.y, b.y) + vm(a.z, b.z)).xxxx; }\n"
"float4 vdph(float4 a, float4 b) { return (vm(a.x, b.x) + vm(a.y, b.y) + vm(a.z, b.z) + b.w).xxxx; }\n"
"float4 vdp4(float4 a, float4 b) { return (vm(a.x, b.x) + vm(a.y, b.y) + vm(a.z, b.z) + vm(a.w, b.w)).xxxx; }\n"
"float4 vdst(float4 a, float4 b) { return float4(1.0, vm(a.y, b.y), a.z, b.w); }\n"
"float4 vmin(float4 a, float4 b) { return float4(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z, a.w < b.w ? a.w : b.w); }\n"
"float4 vmax(float4 a, float4 b) { return float4(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z, a.w > b.w ? a.w : b.w); }\n"
"float lt1(float a, float b) { return (a < b || (asuint(a) == 0x80000000u && asuint(b) == 0u)) ? 1.0 : 0.0; }\n"
"float4 vslt(float4 a, float4 b) { return float4(lt1(a.x, b.x), lt1(a.y, b.y), lt1(a.z, b.z), lt1(a.w, b.w)); }\n"
"float4 vsge(float4 a, float4 b) { return 1.0 - vslt(a, b); }\n"
"float4 vrcp(float4 a) {\n"
"    float x = a.x, r;\n"
"    if (x == 1.0) r = 1.0;\n"
"    else if (x == 0.0) r = (asuint(x) & 0x80000000u) ? -asfloat(0x7F800000u) : asfloat(0x7F800000u);\n"
"    else r = 1.0 / x;\n"
"    return r.xxxx;\n"
"}\n"
"float4 vrcc(float4 a) {\n"
"    float x = a.x, r;\n"
"    if (x == 1.0) r = 1.0;\n"
"    else {\n"
"        r = 1.0 / x;\n"
"        if (r > 0.0) {\n"
"            if (r < 5.42101e-020) r = 5.42101e-020;\n"
"            else if (r > 1.884467e+019) r = asfloat(0x5F800000u);\n"
"        } else {\n"
"            if (r < -1.884467e+019) r = asfloat(0xDF800000u);\n"
"            else if (r > -5.42101e-020) r = -5.42101e-020;\n"
"        }\n"
"    }\n"
"    return r.xxxx;\n"
"}\n"
"float4 vrsq(float4 a) {\n"
"    float x = abs(a.x), r;\n"
"    if (x == 1.0) r = 1.0;\n"
"    else if (x == 0.0) r = asfloat(0x7F800000u);\n"
"    else r = 1.0 / sqrt(x);\n"
"    return r.xxxx;\n"
"}\n"
"float4 vexp(float4 a) {\n"
"    float t = floor(a.x);\n"
"    return float4(exp2(t), a.x - t, exp2(a.x), 1.0);\n"
"}\n"
"float4 vlog(float4 a) {\n"
"    float t = abs(a.x);\n"
"    if (t == 0.0) return float4(-asfloat(0x7F800000u), 1.0, -asfloat(0x7F800000u), 1.0);\n"
"    if (isinf(t)) return float4(asfloat(0x7F800000u), 1.0, asfloat(0x7F800000u), 1.0);\n"
"    uint bits = asuint(t);\n"
"    float e = (float)((int)((bits >> 23) & 0xFFu) - 127);\n"
"    float m = asfloat((bits & 0x7FFFFFu) | 0x3F800000u);\n"
"    return float4(e, m, log2(t), 1.0);\n"
"}\n"
"float4 vlit(float4 a) {\n"
"    float4 r = float4(1.0, 0.0, 0.0, 1.0);\n"
"    float power = clamp(a.w, -127.9961, 127.9961);\n"
"    if (a.x > 0.0) {\n"
"        r.y = a.x;\n"
"        if (a.y > 0.0) r.z = pow(a.y, power);\n"
"    }\n"
"    return r;\n"
"}\n"
"float clamp_away(float t) {\n"
"    float lo = asfloat(0x1F800000u), hi = asfloat(0x5F800000u);\n"
"    if (t > 0.0 || asuint(t) == 0u) return t < lo ? lo : (t > hi ? hi : t);\n"
"    return t < -hi ? -hi : (t > -lo ? -lo : t);\n"
"}\n"
"float4 clamp_color(float4 c) {\n"
"    float4 r = c;\n"
"    r.x = isnan(c.x) ? 1.0 : clamp(c.x, 0.0, 1.0);\n"
"    r.y = isnan(c.y) ? 1.0 : clamp(c.y, 0.0, 1.0);\n"
"    r.z = isnan(c.z) ? 1.0 : clamp(c.z, 0.0, 1.0);\n"
"    r.w = isnan(c.w) ? 1.0 : clamp(c.w, 0.0, 1.0);\n"
"    return r;\n"
"}\n"
"float fog_factor(float d) {\n"
"    uint m = g_flags.y;\n"
"    float f = 1.0, infr = 0.0, nanr = 0.0;\n"
"    if (m == 0u || m == 4u) { infr = 1.0; nanr = 1.0; f = g_fog.x + d * g_fog.y - 1.0; }\n"
"    else if (m == 1u) { infr = 1.0; nanr = 1.0; f = g_fog.x + exp2(d * g_fog.y * 16.0) - 1.5; }\n"
"    else if (m == 5u) { f = g_fog.x + exp2(d * g_fog.y * 16.0) - 1.5; }\n"
"    else if (m == 3u || m == 7u) { f = g_fog.x + exp2(-d * d * g_fog.y * g_fog.y * 32.0) - 1.5; }\n"
"    if (m == 4u || m == 5u || m == 7u) f = abs(f);\n"
"    if (isinf(d)) return infr;\n"
"    if (isnan(f)) f = nanr;\n"
"    return clamp(f, -3.402823466e38, 3.402823466e38);\n"
"}\n";

/* ---- program -> HLSL ----------------------------------------------------------------- */

typedef struct StrBuf { char *p; size_t n, cap; } StrBuf;

static void sb_add(StrBuf *b, const char *fmt, ...)
{
    va_list ap;
    int need;
    va_start(ap, fmt);
    need = _vscprintf(fmt, ap);
    va_end(ap);
    if (need < 0) return;
    if (b->n + (size_t)need + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        char *g;
        while (cap < b->n + (size_t)need + 1) cap *= 2;
        g = (char *)realloc(b->p, cap);
        if (!g) return;
        b->p = g; b->cap = cap;
    }
    va_start(ap, fmt);
    vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    b->n += (size_t)need;
}

/* An operand as an HLSL expression. */
static bool vsh_operand(const Nv2aVshInput *in, char *out, size_t cap, uint32_t *attr_mask)
{
    char base[48];
    static const char sw[4] = { 'x', 'y', 'z', 'w' };
    switch (in->type) {
    case NV2ART_TEMPORARY:
        if (in->index == 12) snprintf(base, sizeof(base), "o0");
        else if (in->index < 12) snprintf(base, sizeof(base), "r%u", in->index);
        else return false;
        break;
    case NV2ART_INPUT:
        if (in->index > 15) return false;
        snprintf(base, sizeof(base), "v%u", in->index);
        *attr_mask |= 1u << in->index;
        break;
    case NV2ART_CONTEXT:
        if (in->index > 191) return false;
        if (in->is_relative) snprintf(base, sizeof(base), "C[clamp((int)%u + (int)a0, 0, 191)]", in->index);
        else snprintf(base, sizeof(base), "C[%u]", in->index);
        break;
    default:
        return false;
    }
    snprintf(out, cap, "%s(%s.%c%c%c%c)", in->is_negated ? "-" : "", base,
             sw[in->swizzle[0] & 3], sw[in->swizzle[1] & 3], sw[in->swizzle[2] & 3], sw[in->swizzle[3] & 3]);
    return true;
}

static bool vsh_op(StrBuf *b, const Nv2aVshOperation *op, const char *tag, int step,
                   uint32_t *attr_mask, bool *needs_a0)
{
    char in[3][96];
    const char *fn = NULL;
    int nin = 0, k;
    char dest[2][16];
    (void)needs_a0;
    if (op->opcode == NV2AOP_NOP) return true;
    switch (op->opcode) {
    case NV2AOP_MOV: case NV2AOP_ARL: case NV2AOP_RCP: case NV2AOP_RCC: case NV2AOP_RSQ:
    case NV2AOP_EXP: case NV2AOP_LOG: case NV2AOP_LIT:
        nin = 1; break;
    case NV2AOP_MUL: case NV2AOP_ADD: case NV2AOP_DP3: case NV2AOP_DPH: case NV2AOP_DP4:
    case NV2AOP_DST: case NV2AOP_MIN: case NV2AOP_MAX: case NV2AOP_SLT: case NV2AOP_SGE:
        nin = 2; break;
    case NV2AOP_MAD:
        nin = 3; break;
    default:
        return false;
    }
    for (k = 0; k < nin; k++) {
        if (!vsh_operand(&op->inputs[k], in[k], sizeof(in[k]), attr_mask)) return false;
    }
    for (k = 0; k < nin; k++) sb_add(b, "        float4 %s_i%d = %s;\n", tag, k, in[k]);
    switch (op->opcode) {
    case NV2AOP_MOV: sb_add(b, "        float4 %s_res = %s_i0;\n", tag, tag); break;
    case NV2AOP_ARL: sb_add(b, "        float4 %s_res = floor(%s_i0.x + 0.001).xxxx;\n", tag, tag); break;
    case NV2AOP_MUL: fn = "vmul"; break;
    case NV2AOP_ADD: sb_add(b, "        float4 %s_res = %s_i0 + %s_i1;\n", tag, tag, tag); break;
    case NV2AOP_MAD: sb_add(b, "        float4 %s_res = vmul(%s_i0, %s_i1) + %s_i2;\n", tag, tag, tag, tag); break;
    case NV2AOP_DP3: fn = "vdp3"; break;
    case NV2AOP_DPH: fn = "vdph"; break;
    case NV2AOP_DP4: fn = "vdp4"; break;
    case NV2AOP_DST: fn = "vdst"; break;
    case NV2AOP_MIN: fn = "vmin"; break;
    case NV2AOP_MAX: fn = "vmax"; break;
    case NV2AOP_SLT: fn = "vslt"; break;
    case NV2AOP_SGE: fn = "vsge"; break;
    case NV2AOP_RCP: sb_add(b, "        float4 %s_res = vrcp(%s_i0);\n", tag, tag); break;
    case NV2AOP_RCC: sb_add(b, "        float4 %s_res = vrcc(%s_i0);\n", tag, tag); break;
    case NV2AOP_RSQ: sb_add(b, "        float4 %s_res = vrsq(%s_i0);\n", tag, tag); break;
    case NV2AOP_EXP: sb_add(b, "        float4 %s_res = vexp(%s_i0);\n", tag, tag); break;
    case NV2AOP_LOG: sb_add(b, "        float4 %s_res = vlog(%s_i0);\n", tag, tag); break;
    case NV2AOP_LIT: sb_add(b, "        float4 %s_res = vlit(%s_i0);\n", tag, tag); break;
    default: return false;
    }
    if (fn) sb_add(b, "        float4 %s_res = %s(%s_i0, %s_i1);\n", tag, fn, tag, tag);
    (void)step;
    (void)dest;
    return true;
}

static bool vsh_writes(StrBuf *b, const Nv2aVshOperation *op, const char *tag)
{
    int k;
    if (op->opcode == NV2AOP_NOP) return true;
    for (k = 0; k < 2; k++) {
        const Nv2aVshOutput *o = &op->outputs[k];
        char name[16];
        switch (o->type) {
        case NV2ART_NONE: continue;
        case NV2ART_ADDRESS:
            sb_add(b, "        a0 = %s_res.x;\n", tag);
            continue;
        case NV2ART_OUTPUT:
            if (o->index > 12) return false;
            snprintf(name, sizeof(name), "o%u", o->index);
            break;
        case NV2ART_TEMPORARY:
            if (o->index > 11) return false;
            snprintf(name, sizeof(name), "r%u", o->index);
            break;
        default:
            return false;       /* constant writes: the CPU stage runs this program */
        }
        if (o->writemask & NV2AWM_X) sb_add(b, "        %s.x = %s_res.x;\n", name, tag);
        if (o->writemask & NV2AWM_Y) sb_add(b, "        %s.y = %s_res.y;\n", name, tag);
        if (o->writemask & NV2AWM_Z) sb_add(b, "        %s.z = %s_res.z;\n", name, tag);
        if (o->writemask & NV2AWM_W) sb_add(b, "        %s.w = %s_res.w;\n", name, tag);
    }
    return true;
}

/* The complete vertex shader source for a program, or NULL if the GPU cannot run it. */
static char *vsh_generate(const Nv2aVshProgram *prog)
{
    StrBuf b = { NULL, 0, 0 };
    uint32_t attr_mask = 0;
    bool needs_a0 = false;
    const Nv2aVshStep *st;
    int step = 0, i;
    sb_add(&b, "%s", kgpu_vsh_head);
    /* the body is generated first so the attribute list is known */
    {
        StrBuf body = { NULL, 0, 0 };
        for (st = prog->steps;; st++, step++) {
            sb_add(&body, "    { // step %d\n", step);
            if (!vsh_op(&body, &st->mac, "m", step, &attr_mask, &needs_a0) ||
                !vsh_op(&body, &st->ilu, "i", step, &attr_mask, &needs_a0) ||
                !vsh_writes(&body, &st->mac, "m") ||
                !vsh_writes(&body, &st->ilu, "i")) {
                free(body.p); free(b.p);
                return NULL;
            }
            sb_add(&body, "    }\n");
            if (st->is_final || step > 400) break;
        }
        sb_add(&b, "VSOut vs_main(uint vid : SV_VertexID) {\n");
        for (i = 0; i < 16; i++) {
            if (attr_mask & (1u << i)) sb_add(&b, "    float4 v%d = FetchAttr(%d, vid);\n", i, i);
        }
        for (i = 0; i < 12; i++) sb_add(&b, "    float4 r%d = float4(0.0, 0.0, 0.0, 0.0);\n", i);
        for (i = 0; i < 13; i++) sb_add(&b, "    float4 o%d = float4(0.0, 0.0, 0.0, 1.0);\n", i);
        sb_add(&b, "    float a0 = 0.0;\n");
        sb_add(&b, "%s", body.p ? body.p : "");
        free(body.p);
    }
    sb_add(&b,
        "    float4 pp = o0;\n"
        "    if (g_vm.x != 1.0) pp.xy = C[59].xy + (pp.xy - C[59].xy) * g_vm.x;\n"
        "    pp.x = trunc(pp.x * 16.0) / 16.0;\n"
        "    pp.y = trunc(pp.y * 16.0) / 16.0;\n"
        "    pp.w = clamp_away(pp.w);\n"
        "    float fogv = g_flags.x != 0u ? fog_factor(o5.x) : 1.0;\n"
        "    float4 c0 = clamp_color(o3);\n"
        "    float4 c1 = float4(0.0, 0.0, 0.0, 1.0);\n"
        "    if (g_flags.z != 0u) { c1 = clamp_color(o4); if (g_flags.w != 0u) c1.w = 1.0; }\n"
        "    VSOut o;\n"
        "    float w = pp.w;\n"
        "    if (abs(w) < 1.0e-12) w *= g_scale.w;\n"
        "    float2 ndc = float2(pp.x * g_scale.x - 1.0, 1.0 - pp.y * g_scale.y);\n"
        "    o.pos = float4(ndc * w, pp.z * g_scale.z * w, w);\n"
        "    o.d0 = c0;\n"
        "    o.d1 = c1;\n"
        "    o.misc = float4(fogv, 0.0, 0.0, 0.0);\n"
        "    o.t0 = o9;\n"
        "    o.t1 = o10;\n"
        "    o.t2 = o11;\n"
        "    o.t3 = o12;\n"
        "    return o;\n"
        "}\n");
    return b.p;
}

/* ---- environment maps ---------------------------------------------------------------------------- */

/* The texture stages a program feeds with an environment map, bit k = stage k (output o9 + k). BLACK reflects the
 * eye direction about the vertex normal (input v2), turns it into the env map's frame and writes map coordinates
 * as r * s + s with the same constant for scale and bias (0.5: sphere map), directly or projectively. An output is
 * an env coordinate when its value depends on the normal through such a multiply-add. Steps read before they
 * write, the MAC and ILU halves alike, as the hardware does. */
#define VENV_NORMAL 1u
#define VENV_HALF 2u
static unsigned vsh_op_inputs(const Nv2aVshOperation *op)
{
    switch (op->opcode) {
    case NV2AOP_NOP: return 0;
    case NV2AOP_MOV: case NV2AOP_ARL: case NV2AOP_RCP: case NV2AOP_RCC: case NV2AOP_RSQ:
    case NV2AOP_EXP: case NV2AOP_LOG: case NV2AOP_LIT: return 1;
    case NV2AOP_MAD: return 3;
    default: return 2;
    }
}

static unsigned vsh_op_flags(const Nv2aVshOperation *op, const unsigned temp[13], unsigned a0)
{
    const unsigned n = vsh_op_inputs(op);
    unsigned flags = 0, k;
    for (k = 0; k < n; k++) {
        const Nv2aVshInput *in = &op->inputs[k];
        if (in->type == NV2ART_INPUT && in->index == 2) flags |= VENV_NORMAL;
        else if (in->type == NV2ART_TEMPORARY && in->index <= 12) flags |= temp[in->index];
        if (in->is_relative) flags |= a0;
    }
    if (op->opcode == NV2AOP_MAD && op->inputs[1].type == NV2ART_CONTEXT && op->inputs[2].type == NV2ART_CONTEXT &&
        op->inputs[1].index == op->inputs[2].index && !op->inputs[1].is_relative && !op->inputs[2].is_relative &&
        !memcmp(op->inputs[1].swizzle, op->inputs[2].swizzle, 4) && (flags & VENV_NORMAL))
        flags |= VENV_HALF;
    return flags;
}

static void vsh_op_store(const Nv2aVshOperation *op, unsigned flags, unsigned temp[13], unsigned out[16], unsigned *a0)
{
    int k;
    if (op->opcode == NV2AOP_NOP) return;
    for (k = 0; k < 2; k++) {
        const Nv2aVshOutput *o = &op->outputs[k];
        if (o->type == NV2ART_TEMPORARY && o->index <= 12) {
            if (o->writemask == NV2AWM_XYZW) temp[o->index] = flags; else temp[o->index] |= flags;
        } else if (o->type == NV2ART_OUTPUT && o->index < 16) {
            out[o->index] |= flags;
        } else if (o->type == NV2ART_ADDRESS) {
            *a0 = flags;
        }
    }
}

static unsigned vsh_env_stages(const Nv2aVshProgram *prog)
{
    unsigned temp[13], out[16], a0 = 0, mask = 0, k;
    const Nv2aVshStep *st;
    int step = 0;
    memset(temp, 0, sizeof(temp));
    memset(out, 0, sizeof(out));
    for (st = prog->steps;; st++, step++) {
        const unsigned fm = vsh_op_flags(&st->mac, temp, a0), fi = vsh_op_flags(&st->ilu, temp, a0);
        vsh_op_store(&st->mac, fm, temp, out, &a0);
        vsh_op_store(&st->ilu, fi, temp, out, &a0);
        if (st->is_final || step > 400) break;
    }
    for (k = 0; k < 4; k++)
        if ((out[9 + k] & (VENV_NORMAL | VENV_HALF)) == (VENV_NORMAL | VENV_HALF)) mask |= 1u << k;
    return mask;
}

/* ---- shader cache with an asynchronous compiler -------------------------------------------- */
/*
 * Compiled programs are cached on disk by their HLSL (vs_<source hash>_<length>.bin), so a program has one file
 * whatever program memory holds behind its final instruction. Each program's microcode is also recorded in
 * vs_manifest.bin there. At startup kgpu_vs_precompile compiles every recorded program the cache lacks on all
 * cores, before the game runs: a rebuild or a cleared cache costs a pause at launch instead of hitches in play
 * (RECOMP_KGPU_PRECOMPILE=0 skips it). A program first met in play gets a quick unoptimised build, then an
 * optimised one replaces it and is cached (RECOMP_KGPU_VS_QUICK=0 builds only the optimised one).
 */

typedef struct KGpuVs {
    bool used;
    uint32_t hash, start;
    volatile LONG state;               /* 0 compiling, 1 ready, 2 unusable */
    ID3D11VertexShader *volatile vs;   /* the optimised build replaces the quick one */
    ID3D11VertexShader *vs_quick;      /* kept: a queued draw may still bind it */
    char *source;
    uint32_t sh, len;                  /* source hash and length: the cache key */
    uint32_t *words;                   /* microcode up to the final instruction, for the manifest */
    uint32_t nwords;
    unsigned env;                      /* the stages its texture coordinates give environment maps (vsh_env_stages) */
    ID3D11GeometryShader *volatile so_gs; /* ray tracing: the same program streaming its triangles out (kelvin_gpu_rt.inc) */
} KGpuVs;

/* A vertex shader's outputs streamed as they are rasterised: position, diffuse colour and the first two texture
 * coordinate sets, 64 bytes a vertex. Made from the vertex shader's own bytecode, so it is a pass-through stage. */
#define KG_SO_STRIDE 64u
static void vs_make_so(KGpuVs *e, const void *code, size_t n)
{
    static const D3D11_SO_DECLARATION_ENTRY decl[4] = {
        { 0, "SV_POSITION", 0, 0, 4, 0 }, { 0, "TEXCOORD", 0, 0, 4, 0 },
        { 0, "TEXCOORD", 3, 0, 4, 0 }, { 0, "TEXCOORD", 4, 0, 4, 0 },
    };
    const UINT stride = KG_SO_STRIDE;
    ID3D11GeometryShader *gs = NULL;
    if (e->so_gs || !krt_available()) return;
    if (FAILED(ID3D11Device_CreateGeometryShaderWithStreamOutput(G.dev, code, (SIZE_T)n, decl, 4, &stride, 1, 0, NULL, &gs)))
        return;
    if (InterlockedCompareExchangePointer((PVOID volatile *)&e->so_gs, gs, NULL) != NULL) ID3D11GeometryShader_Release(gs);
}

#define KG_VS_MAX 512
static KGpuVs s_vs[KG_VS_MAX];
/* Compile jobs for the worker threads: vertex programs (kind 0), specialised pixel programs (kind 1) and the
 * optimised build of a vertex program that runs its quick build (kind 2). */
typedef struct KgJob { int kind; void *p; } KgJob;
#define KG_JOB_MAX 4096
static KgJob s_jobs[KG_JOB_MAX];
static volatile LONG s_job_head, s_job_claim;    /* jobs queued / jobs taken by a compiler thread */
static HANDLE s_job_sem;                         /* released once per queued job */
static SRWLOCK s_job_lock = SRWLOCK_INIT;        /* the pusher, the submission thread and the workers queue jobs */
#define KG_VS_WORKERS_MAX 8
static void kg_job_ps(void *p);
static void kg_job_post(int kind, void *p);
static int s_vs_enabled = -1;
static int s_vs_quick = -1;

static uint32_t fnv_bytes(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint32_t h = 2166136261u;
    size_t i;
    for (i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
    return h;
}

static const char *kg_cache_dir(void)
{
    const char *dir = getenv("RECOMP_KGPU_CACHE");
    return dir && *dir ? dir : "kgpu-cache";
}

static void vs_cache_path(char *out, size_t cap, uint32_t sh, uint32_t len)
{
    snprintf(out, cap, "%s/vs_%08X_%05X.bin", kg_cache_dir(), sh, len);
}

static void *kg_read_file(const char *path, long *size)
{
    FILE *f = fopen(path, "rb");
    void *buf;
    long n;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = n > 0 ? malloc((size_t)n) : NULL;
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); buf = NULL; }
    fclose(f);
    *size = n;
    return buf;
}

/* Written beside and renamed, so an interrupted write never leaves a file that looks complete. */
static void kg_write_file(const char *path, const void *p, size_t n)
{
    char tmp[600];
    FILE *f;
    CreateDirectoryA(kg_cache_dir(), NULL);
    snprintf(tmp, sizeof(tmp), "%s.%lu.tmp", path, (unsigned long)GetCurrentThreadId());
    f = fopen(tmp, "wb");
    if (!f) return;
    if (fwrite(p, 1, n, f) != n) { fclose(f); DeleteFileA(tmp); return; }
    fclose(f);
    if (!MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING)) DeleteFileA(tmp);
}

static bool vs_try_cache(KGpuVs *e)
{
    char path[512];
    ID3D11VertexShader *vs = NULL;
    void *buf;
    long n = 0;
    vs_cache_path(path, sizeof(path), e->sh, e->len);
    buf = kg_read_file(path, &n);
    if (!buf) {
        /* a cache written before the names were keyed by the source alone */
        char old[512];
        snprintf(old, sizeof(old), "%s/vs_%08X_%08X_%08X.bin", kg_cache_dir(), e->hash, e->start, e->sh);
        buf = kg_read_file(old, &n);
        if (buf) kg_write_file(path, buf, (size_t)n);
    }
    if (!buf) return false;
    if (FAILED(ID3D11Device_CreateVertexShader(G.dev, buf, (SIZE_T)n, NULL, &vs))) vs = NULL;
    if (vs) vs_make_so(e, buf, (size_t)n);
    free(buf);
    e->vs = vs;
    return vs != NULL;
}

/* ---- the manifest of the programs met so far ---- */

#define KG_MAN_MAGIC 0x3153564Bu       /* "KVS1"; a record is magic, source hash, source length, word count, words */
#define KG_MAN_MAX 4096
static uint64_t s_man_keys[KG_MAN_MAX];
static int s_man_n;
static bool s_man_loaded;
static SRWLOCK s_man_lock = SRWLOCK_INIT;

static void manifest_path(char *out, size_t cap)
{
    snprintf(out, cap, "%s/vs_manifest.bin", kg_cache_dir());
}

static bool manifest_has(uint64_t key)
{
    int i;
    for (i = 0; i < s_man_n; i++) if (s_man_keys[i] == key) return true;
    return false;
}

static void manifest_note(uint64_t key)
{
    if (s_man_n < KG_MAN_MAX && !manifest_has(key)) s_man_keys[s_man_n++] = key;
}

/* Calls fn for each whole record; a torn record at the end (an interrupted write) ends the list. */
static void manifest_read(void (*fn)(void *ctx, const uint32_t *hdr, const uint32_t *words), void *ctx)
{
    char path[512];
    void *buf;
    long n = 0, at = 0;
    manifest_path(path, sizeof(path));
    buf = kg_read_file(path, &n);
    if (!buf) return;
    while (at + 16 <= n) {
        const uint32_t *hdr = (const uint32_t *)((const uint8_t *)buf + at);
        if (hdr[0] != KG_MAN_MAGIC || !hdr[3] || hdr[3] % VSH_TOKEN_SIZE ||
            hdr[3] > NV2A_MAX_TRANSFORM_PROGRAM_LENGTH * VSH_TOKEN_SIZE || at + 16 + (long)hdr[3] * 4 > n) break;
        fn(ctx, hdr, hdr + 4);
        at += 16 + (long)hdr[3] * 4;
    }
    free(buf);
}

static void manifest_note_record(void *ctx, const uint32_t *hdr, const uint32_t *words)
{
    (void)ctx; (void)words;
    manifest_note(((uint64_t)hdr[1] << 32) | hdr[2]);
}

static void manifest_add(const KGpuVs *e)
{
    uint64_t key = ((uint64_t)e->sh << 32) | e->len;
    if (!e->words || !e->nwords) return;
    AcquireSRWLockExclusive(&s_man_lock);
    if (!s_man_loaded) { manifest_read(manifest_note_record, NULL); s_man_loaded = true; }
    if (!manifest_has(key) && s_man_n < KG_MAN_MAX) {
        char path[512];
        FILE *f;
        uint32_t hdr[4];
        hdr[0] = KG_MAN_MAGIC; hdr[1] = e->sh; hdr[2] = e->len; hdr[3] = e->nwords;
        CreateDirectoryA(kg_cache_dir(), NULL);
        manifest_path(path, sizeof(path));
        f = fopen(path, "ab");
        if (f) {
            fwrite(hdr, 4, 4, f);
            fwrite(e->words, 4, e->nwords, f);
            fclose(f);
            s_man_keys[s_man_n++] = key;
        }
    }
    ReleaseSRWLockExclusive(&s_man_lock);
}

/* ---- compiling ---- */

static ID3DBlob *vs_build(const char *source, bool quick, ID3DBlob **errors)
{
    ID3DBlob *code = NULL;
    HRESULT hr = G.compile(source, strlen(source), "kgpu_vsh", NULL, NULL, "vs_main", "vs_5_0",
                           quick ? D3DCOMPILE_SKIP_OPTIMIZATION : D3DCOMPILE_OPTIMIZATION_LEVEL1, 0, &code, errors);
    if (FAILED(hr) && code) { ID3D10Blob_Release(code); code = NULL; }
    return code;
}

static void vs_compile(KGpuVs *e, bool quick)
{
    ID3DBlob *code, *errors = NULL;
    ID3D11VertexShader *vs = NULL;
    long long t0 = qpc_now();
    code = vs_build(e->source, quick, &errors);
    if (code && SUCCEEDED(ID3D11Device_CreateVertexShader(G.dev, ID3D10Blob_GetBufferPointer(code),
                                                          ID3D10Blob_GetBufferSize(code), NULL, &vs))) {
        double ms = (double)(qpc_now() - t0) * 1000.0 / (double)G.qpf;
        if (quick) {
            e->vs_quick = vs;
            e->vs = vs;
            InterlockedExchange(&e->state, 1);
            fprintf(stderr, "[KGPU] vertex program %08X/%u quick build in %.0f ms\n", e->hash, e->start, ms);
            kg_job_post(2, e);
        } else {
            char path[512];
            vs_cache_path(path, sizeof(path), e->sh, e->len);
            kg_write_file(path, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code));
            vs_make_so(e, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code));
            InterlockedExchangePointer((PVOID volatile *)&e->vs, vs);
            InterlockedExchange(&e->state, 1);
            manifest_add(e);
            fprintf(stderr, "[KGPU] vertex program %08X/%u compiled in %.0f ms\n", e->hash, e->start, ms);
        }
    } else if (quick) {
        /* the optimised build may still succeed */
        kg_job_post(2, e);
    } else {
        fprintf(stderr, "[KGPU] vertex program %08X/%u did not compile: %s\n", e->hash, e->start,
                errors ? (const char *)ID3D10Blob_GetBufferPointer(errors) : "(no log)");
        if (e->state != 1) InterlockedExchange(&e->state, 2);    /* a working quick build stays in use */
    }
    if (errors) ID3D10Blob_Release(errors);
    if (code) ID3D10Blob_Release(code);
}

static DWORD WINAPI vs_worker(LPVOID arg)
{
    (void)arg;
    for (;;) {
        LONG idx;
        KgJob j;
        WaitForSingleObject(s_job_sem, INFINITE);
        idx = InterlockedIncrement(&s_job_claim) - 1;
        j = s_jobs[idx % KG_JOB_MAX];
        if (j.kind == 0) vs_compile((KGpuVs *)j.p, s_vs_quick != 0);
        else if (j.kind == 2) vs_compile((KGpuVs *)j.p, false);
        else kg_job_ps(j.p);
    }
    return 0;
}

static void kg_job_post(int kind, void *p)
{
    AcquireSRWLockExclusive(&s_job_lock);
    if (!s_job_sem) {
        /* Programs compile in parallel: a level's first frames name dozens of them at once, and until one is
         * ready its draws run the CPU vertex stage (vertex programs) or the ubershader (pixel programs).
         * RECOMP_KGPU_VS_THREADS overrides the count. */
        const char *e2 = getenv("RECOMP_KGPU_VS_THREADS");
        int n = e2 && *e2 ? atoi(e2) : 4, i;
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        if (n > (int)si.dwNumberOfProcessors / 2) n = (int)si.dwNumberOfProcessors / 2;
        if (n < 1) n = 1;
        if (n > KG_VS_WORKERS_MAX) n = KG_VS_WORKERS_MAX;
        s_job_sem = CreateSemaphoreA(NULL, 0, KG_JOB_MAX, NULL);
        for (i = 0; i < n; i++) {
            HANDLE th = CreateThread(NULL, 0, vs_worker, NULL, 0, NULL);
            if (th) { SetThreadPriority(th, THREAD_PRIORITY_BELOW_NORMAL); CloseHandle(th); }
        }
    }
    s_jobs[s_job_head % KG_JOB_MAX].kind = kind;
    s_jobs[s_job_head % KG_JOB_MAX].p = p;
    InterlockedIncrement(&s_job_head);
    ReleaseSRWLockExclusive(&s_job_lock);
    ReleaseSemaphore(s_job_sem, 1, NULL);
}

/* ---- compiling the recorded programs before the game starts ---- */

typedef struct KgPre { char *source; uint32_t sh, len; } KgPre;
typedef struct KgPreList { KgPre *items; int n, cap, cached, unusable; } KgPreList;
static volatile LONG s_pre_next, s_pre_failed;

static void precompile_note(void *ctx, const uint32_t *hdr, const uint32_t *words)
{
    KgPreList *l = (KgPreList *)ctx;
    Nv2aVshProgram prog;
    char *source, path[512];
    uint32_t sh, len;
    int i;
    memset(&prog, 0, sizeof(prog));
    if (nv2a_vsh_parse_program(&prog, words, hdr[3] / VSH_TOKEN_SIZE) != NV2AVPR_SUCCESS) { l->unusable++; return; }
    source = vsh_generate(&prog);
    nv2a_vsh_program_destroy(&prog);
    if (!source) { l->unusable++; return; }
    /* this build's translation, which may differ from the one that recorded the program */
    len = (uint32_t)strlen(source);
    sh = fnv_bytes(source, len);
    manifest_note(((uint64_t)hdr[1] << 32) | hdr[2]);
    manifest_note(((uint64_t)sh << 32) | len);
    vs_cache_path(path, sizeof(path), sh, len);
    if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) { l->cached++; free(source); return; }
    for (i = 0; i < l->n; i++) {
        if (l->items[i].sh == sh && l->items[i].len == len) { free(source); return; }
    }
    if (l->n == l->cap) {
        int cap = l->cap ? l->cap * 2 : 64;
        KgPre *g = (KgPre *)realloc(l->items, (size_t)cap * sizeof(*g));
        if (!g) { free(source); return; }
        l->items = g; l->cap = cap;
    }
    l->items[l->n].source = source; l->items[l->n].sh = sh; l->items[l->n].len = len;
    l->n++;
}

static DWORD WINAPI precompile_worker(LPVOID arg)
{
    KgPreList *l = (KgPreList *)arg;
    for (;;) {
        LONG i = InterlockedIncrement(&s_pre_next) - 1;
        ID3DBlob *code, *errors = NULL;
        if (i >= l->n) break;
        code = vs_build(l->items[i].source, false, &errors);
        if (code) {
            char path[512];
            vs_cache_path(path, sizeof(path), l->items[i].sh, l->items[i].len);
            kg_write_file(path, ID3D10Blob_GetBufferPointer(code), ID3D10Blob_GetBufferSize(code));
            ID3D10Blob_Release(code);
        } else {
            InterlockedIncrement(&s_pre_failed);
        }
        if (errors) ID3D10Blob_Release(errors);
    }
    return 0;
}

static void kgpu_vs_precompile(void)
{
    const char *env = getenv("RECOMP_KGPU_PRECOMPILE");
    KgPreList l;
    HANDLE th[32];
    SYSTEM_INFO si;
    int nth = 0, want, i;
    long long t0 = qpc_now();
    if ((env && *env == '0') || !kgpu_vs_enabled() || !G.compile) return;
    memset(&l, 0, sizeof(l));
    AcquireSRWLockExclusive(&s_man_lock);
    manifest_read(precompile_note, &l);
    s_man_loaded = true;
    ReleaseSRWLockExclusive(&s_man_lock);
    if (l.n) {
        GetSystemInfo(&si);
        want = (int)si.dwNumberOfProcessors > 1 ? (int)si.dwNumberOfProcessors - 1 : 1;
        if (want > l.n) want = l.n;
        if (want > 32) want = 32;
        fprintf(stderr, "[KGPU] precompiling %d vertex programs on %d threads\n", l.n, want);
        s_pre_next = 0; s_pre_failed = 0;
        for (i = 0; i < want; i++) {
            th[nth] = CreateThread(NULL, 0, precompile_worker, &l, 0, NULL);
            if (th[nth]) nth++;
        }
        if (nth) WaitForMultipleObjects((DWORD)nth, th, TRUE, INFINITE);
        else precompile_worker(&l);
        for (i = 0; i < nth; i++) CloseHandle(th[i]);
    }
    fprintf(stderr, "[KGPU] shader manifest: %d vertex programs compiled before start in %.0f ms (%ld failed), "
                    "%d already cached, %d unusable\n", l.n,
            (double)(qpc_now() - t0) * 1000.0 / (double)G.qpf, (long)s_pre_failed, l.cached, l.unusable);
    for (i = 0; i < l.n; i++) free(l.items[i].source);
    free(l.items);
}

bool kgpu_vs_enabled(void)
{
    if (s_vs_enabled < 0) {
        const char *e = getenv("RECOMP_KGPU_CPU_VERTICES");
        s_vs_enabled = !(e && *e && *e != '0');
    }
    return s_vs_enabled != 0 && G.ok;
}

/* The compiled shader for a program, once ready. Starts its translation and
 * compilation the first time a program is seen; until then (or if the GPU cannot
 * run it) false, and the draw uses the CPU vertex stage. */
static KGpuVs *vs_lookup(const void *program, uint32_t hash, uint32_t start, const uint32_t *words, uint32_t nwords)
{
    KGpuVs *e = NULL;
    unsigned int i, slot = (hash ^ (start * 40503u)) % KG_VS_MAX;
    for (i = 0; i < KG_VS_MAX; i++, slot = (slot + 1) % KG_VS_MAX) {
        KGpuVs *q = &s_vs[slot];
        if (!q->used) { e = q; break; }
        if (q->hash == hash && q->start == start) return q->state == 1 ? q : NULL;
    }
    if (!e) return NULL;
    if (s_vs_quick < 0) {
        const char *env = getenv("RECOMP_KGPU_VS_QUICK");
        s_vs_quick = !(env && *env == '0');
    }
    e->used = true; e->hash = hash; e->start = start; e->state = 0;
    e->env = vsh_env_stages((const Nv2aVshProgram *)program);
    if (e->env) fprintf(stderr, "[KGPU] vertex program %08X/%u gives stages %X environment maps\n", hash, start, e->env);
    e->source = vsh_generate((const Nv2aVshProgram *)program);
    if (!e->source) {
        e->state = 2;
        fprintf(stderr, "[KGPU] vertex program %08X/%u runs on the CPU (unsupported operation)\n", hash, start);
        return NULL;
    }
    e->len = (uint32_t)strlen(e->source);
    e->sh = fnv_bytes(e->source, e->len);
    if (words && nwords) {
        e->words = (uint32_t *)malloc((size_t)nwords * 4);
        if (e->words) { memcpy(e->words, words, (size_t)nwords * 4); e->nwords = nwords; }
    }
    if (vs_try_cache(e)) {
        e->state = 1;
        manifest_add(e);
        fprintf(stderr, "[KGPU] vertex program %08X/%u loaded from the shader cache\n", hash, start);
        return e;
    }
    kg_job_post(0, e);
    return NULL;
}

bool kgpu_vs_ready(const void *program, uint32_t hash, uint32_t start, const uint32_t *words, uint32_t nwords)
{
    return vs_lookup(program, hash, start, words, nwords) != NULL;
}
