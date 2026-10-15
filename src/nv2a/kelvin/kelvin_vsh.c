/*
 * Kelvin vertex stage.
 *
 * Programmable mode runs the title's transform program through nv2a_vsh_cpu
 * (xemu's own CPU vertex-program interpreter, vendored in nv2a_vsh_cpu/).
 * Fixed-function mode evaluates what xemu's glsl/vsh-ff.c generates. Both end
 * in the epilogue of glsl/vsh.c: 1/16-pixel truncation of the screen position,
 * w kept away from zero and infinity, fog factor from the fog mode, colours
 * clamped with NaN read as 1. How a program's screen-space output is treated
 * follows the behaviour of xemu's glsl/vsh-prog.c (GPL-2.0+), implemented
 * independently here, not copied (see NOTICE).
 *
 * Copyright (c) 2012 espes; 2015 Jannik Vogel; 2018-2025 Matt Borgerson
 * (xemu glsl/vsh.c and glsl/vsh-ff.c, whose vertex semantics these are)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <float.h>
#include "kelvin_render.h"
#include "kelvin_trace.h"
#include "nv2a_vsh_cpu/nv2a_vsh_emulator.h"
#include "nv2a_vsh_cpu/nv2a_vsh_emulator4.h"
#include "nv2a_vsh_cpu/nv2a_vsh_emulator8.h"

/* A parsed transform program. Entries are kept, never changed, until the
 * cache is full, so a draw shaded later still has its program when the
 * title has moved on to another (kvsh_state). */
typedef struct ProgEntry {
    uint32_t start, hash;
    bool writes_constants;        /* c[] outputs: vertices depend on order */
    uint32_t nslots;              /* instructions up to the final one */
    Nv2aVshProgram prog;
    uint32_t words[NV2A_MAX_TRANSFORM_PROGRAM_LENGTH][VSH_TOKEN_SIZE];
} ProgEntry;

#define KVSH_PROGRAMS 256
static ProgEntry *s_progs[KVSH_PROGRAMS];
static int s_nprogs;

typedef struct VshDrawState {
    bool program;
    const ProgEntry *entry;       /* the draw's parsed program, or NULL */
    bool prog_valid;
    bool prog_writes_constants;
    uint32_t prog_start, prog_hash;
    uint32_t prog_writes;         /* pg->program_writes when last hashed */

    bool fog_enable;
    int fog_mode;
    float fog_param[2];
    bool specular_enable, separate_specular, ignore_specular_alpha;
    float point_size;
    bool point_params_enable;

    /* fixed function */
    int skinning;
    bool normalization, local_eye, lighting;
    int emission_src, ambient_src, diffuse_src, specular_src;
    bool texture_matrix_enable[4];
    int texgen[4][4];
    int light[NV2A_MAX_LIGHTS];
    int foggen;
    float specular_power;
    float material_alpha;
} VshDrawState;

static VshDrawState s_vs;
/* What program-mode shading reads, for the current draw (kvsh_begin_draw). */
static KVshState s_cur;

static const float *C(KelvinState *pg, int row)
{
    return (const float *)pg->vsh_constants[row];
}

static float clamp_away_zero_inf(float t)
{
    const float lo = kelvin_u2f(0x1F800000), hi = kelvin_u2f(0x5F800000);
    if (t > 0.0f || kelvin_f2u(t) == 0) {
        return t < lo ? lo : (t > hi ? hi : t);
    }
    return t < -hi ? -hi : (t > -lo ? -lo : t);
}

static float round_screen(float v)
{
    return truncf(v * 16.0f) / 16.0f;
}

static uint32_t hash_words(const uint32_t *w, size_t n)
{
    uint32_t h = 2166136261u;
    size_t i;
    for (i = 0; i < n; i++) {
        h = (h ^ w[i]) * 16777619u;
    }
    return h;
}

/* The parsed program at `start` whose words hash to h, parsed now if the
 * cache lacks it; NULL if it does not parse. A full cache is emptied, once
 * no draw waiting to be shaded can still use it. */
static const ProgEntry *find_program(KelvinState *pg, uint32_t start, uint32_t h)
{
    ProgEntry *e;
    const Nv2aVshStep *step;
    size_t bytes = (NV2A_MAX_TRANSFORM_PROGRAM_LENGTH - start) * sizeof(e->words[0]);
    int i;

    for (i = 0; i < s_nprogs; i++) {
        e = s_progs[i];
        if (e->start == start && e->hash == h &&
            memcmp(e->words, pg->program_data[start], bytes) == 0) {
            return e;
        }
    }
    if (s_nprogs == KVSH_PROGRAMS) {
        kelvin_render_geometry_sync();
        for (i = 0; i < s_nprogs; i++) {
            nv2a_vsh_program_destroy(&s_progs[i]->prog);
            free(s_progs[i]);
        }
        s_nprogs = 0;
    }
    e = (ProgEntry *)calloc(1, sizeof(*e));
    if (!e) {
        return NULL;
    }
    if (nv2a_vsh_parse_program(&e->prog, pg->program_data[start],
                               NV2A_MAX_TRANSFORM_PROGRAM_LENGTH - start)
        != NV2AVPR_SUCCESS) {
        free(e);
        return NULL;
    }
    e->start = start;
    e->hash = h;
    memcpy(e->words, pg->program_data[start], bytes);
    for (step = e->prog.steps;; step++) {
        int k;
        e->nslots++;
        for (k = 0; k < 2; k++) {
            if (step->mac.outputs[k].type == NV2ART_CONTEXT ||
                step->ilu.outputs[k].type == NV2ART_CONTEXT) {
                e->writes_constants = true;
            }
        }
        if (step->is_final) {
            break;
        }
    }
    s_progs[s_nprogs++] = e;
    return e;
}

void kvsh_begin_draw(KelvinState *pg)
{
    uint32_t csv0_c = pgraph_reg_r(pg, NV_PGRAPH_CSV0_C);
    uint32_t csv0_d = pgraph_reg_r(pg, NV_PGRAPH_CSV0_D);
    uint32_t control_3 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_3);
    unsigned int mode = GET_MASK(csv0_d, NV_PGRAPH_CSV0_D_MODE);
    int i, j;

    s_vs.program = mode == 2;
    if (mode != 0 && mode != 2) {
        kelvin_unsupported(KU_KELVIN_METHOD, "transform execution mode", mode);
    }

    s_vs.fog_enable = control_3 & NV_PGRAPH_CONTROL_3_FOGENABLE;
    s_vs.fog_mode = GET_MASK(control_3, NV_PGRAPH_CONTROL_3_FOG_MODE);
    s_vs.fog_param[0] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_FOGPARAM0));
    s_vs.fog_param[1] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_FOGPARAM1));
    s_vs.specular_enable = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_SPECULAR_ENABLE);
    s_vs.separate_specular = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_SEPARATE_SPECULAR);
    s_vs.ignore_specular_alpha =
        !GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_ALPHA_FROM_MATERIAL_SPECULAR);
    s_vs.point_size = pgraph_reg_r(pg, NV_PGRAPH_POINTSIZE) / 8.0f;
    s_vs.point_params_enable = GET_MASK(csv0_d, NV_PGRAPH_CSV0_D_POINTPARAMSENABLE);
    s_vs.specular_power = pg->specular_power;
    s_vs.material_alpha = pg->material_alpha;
    s_cur.constants = (const float *)pg->vsh_constants;
    s_cur.fog_enable = s_vs.fog_enable;
    s_cur.fog_mode = s_vs.fog_mode;
    s_cur.fog_param[0] = s_vs.fog_param[0];
    s_cur.fog_param[1] = s_vs.fog_param[1];
    s_cur.specular_enable = s_vs.specular_enable;
    s_cur.ignore_specular_alpha = s_vs.ignore_specular_alpha;
    s_cur.point_size = s_vs.point_size;

    if (s_vs.program) {
        unsigned int start = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_CHEOPS_PROGRAM_START);
        uint32_t h;
        if (start >= NV2A_MAX_TRANSFORM_PROGRAM_LENGTH) {
            start = 0;
        }
        /* Nothing written to the program memory since: the same program. */
        if (s_vs.prog_valid && s_vs.prog_start == start &&
            s_vs.prog_writes == pg->program_writes) {
            return;
        }
        s_vs.prog_writes = pg->program_writes;
        h = hash_words(&pg->program_data[start][0],
                       (NV2A_MAX_TRANSFORM_PROGRAM_LENGTH - start) * 4);
        if (!s_vs.prog_valid || s_vs.prog_start != start || s_vs.prog_hash != h) {
            const ProgEntry *e = find_program(pg, start, h);
            s_vs.entry = e;
            s_vs.prog_valid = e != NULL;
            if (e) {
                s_vs.prog_start = start;
                s_vs.prog_hash = h;
                s_vs.prog_writes_constants = e->writes_constants;
            } else {
                kelvin_unsupported(KU_KELVIN_METHOD,
                                   "transform program parse", start);
            }
            s_cur.program = e ? &e->prog : NULL;
        }
        return;
    }

    /* fixed function (xemu glsl/vsh.c set_fixed_function_vsh_state) */
    s_vs.skinning = GET_MASK(csv0_d, NV_PGRAPH_CSV0_D_SKIN);
    s_vs.normalization = csv0_c & NV_PGRAPH_CSV0_C_NORMALIZATION_ENABLE;
    s_vs.local_eye = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_LOCALEYE);
    s_vs.emission_src = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_EMISSION);
    s_vs.ambient_src = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_AMBIENT);
    s_vs.diffuse_src = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_DIFFUSE);
    s_vs.specular_src = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_SPECULAR);
    for (i = 0; i < 4; i++) {
        unsigned int reg = (i < 2) ? NV_PGRAPH_CSV1_A : NV_PGRAPH_CSV1_B;
        unsigned int masks[4] = {
            (i % 2) ? NV_PGRAPH_CSV1_A_T1_S : NV_PGRAPH_CSV1_A_T0_S,
            (i % 2) ? NV_PGRAPH_CSV1_A_T1_T : NV_PGRAPH_CSV1_A_T0_T,
            (i % 2) ? NV_PGRAPH_CSV1_A_T1_R : NV_PGRAPH_CSV1_A_T0_R,
            (i % 2) ? NV_PGRAPH_CSV1_A_T1_Q : NV_PGRAPH_CSV1_A_T0_Q,
        };
        s_vs.texture_matrix_enable[i] = pg->texture_matrix_enable[i];
        for (j = 0; j < 4; j++) {
            s_vs.texgen[i][j] = GET_MASK(pgraph_reg_r(pg, reg), masks[j]);
        }
    }
    s_vs.lighting = GET_MASK(csv0_c, NV_PGRAPH_CSV0_C_LIGHTING);
    for (i = 0; i < NV2A_MAX_LIGHTS; i++) {
        s_vs.light[i] = s_vs.lighting
            ? (int)GET_MASK(csv0_d, NV_PGRAPH_CSV0_D_LIGHT0 << (i * 2)) : 0;
    }
    s_vs.foggen = s_vs.fog_enable
        ? (int)GET_MASK(csv0_d, NV_PGRAPH_CSV0_D_FOGGENMODE) : 0;
}

/* GLSL `v * mat4(c[b], c[b+1], c[b+2], c[b+3])`: out[j] = dot(v, c[b+j]). */
static void vec_mat(KelvinState *pg, const float v[4], int base, float out[4])
{
    int j;
    for (j = 0; j < 4; j++) {
        const float *r = C(pg, base + j);
        out[j] = v[0] * r[0] + v[1] * r[1] + v[2] * r[2] + v[3] * r[3];
    }
}

static float dot3(const float a[3], const float b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void normalize3(float v[3])
{
    float l = sqrtf(dot3(v, v));
    if (l > 0.0f) {
        v[0] /= l; v[1] /= l; v[2] /= l;
    }
}

static float nan_to(float v, float r)
{
    return isnan(v) ? r : v;
}

static void clamp_color(float c[4])
{
    int k;
    for (k = 0; k < 4; k++) {
        float v = nan_to(c[k], 1.0f);
        c[k] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }
}

/* xemu vsh.c fog factor from a fog distance. */
static float fog_factor(const KVshState *vs, float d)
{
    float f, inf_result = 0.0f, nan_result = 0.0f;
    switch (vs->fog_mode) {
    case FOG_MODE_LINEAR:
    case FOG_MODE_LINEAR_ABS:
        inf_result = 1.0f; nan_result = 1.0f;
        f = vs->fog_param[0] + d * vs->fog_param[1] - 1.0f;
        break;
    case FOG_MODE_EXP:
        inf_result = 1.0f; nan_result = 1.0f;
        /* fallthrough */
    case FOG_MODE_EXP_ABS:
        f = vs->fog_param[0] + exp2f(d * vs->fog_param[1] * 16.0f) - 1.5f;
        break;
    case FOG_MODE_EXP2:
    case FOG_MODE_EXP2_ABS:
        f = vs->fog_param[0] +
            exp2f(-d * d * vs->fog_param[1] * vs->fog_param[1] * 32.0f) - 1.5f;
        break;
    default:
        f = 1.0f;
        break;
    }
    if (vs->fog_mode == FOG_MODE_LINEAR_ABS || vs->fog_mode == FOG_MODE_EXP_ABS ||
        vs->fog_mode == FOG_MODE_EXP2_ABS) {
        f = fabsf(f);
    }
    if (isinf(d)) {
        return inf_result;
    }
    f = nan_to(f, nan_result);
    if (f > FLT_MAX) f = FLT_MAX;
    if (f < -FLT_MAX) f = -FLT_MAX;
    return f;
}

static void run_program(const KVshState *vs,
                        const float in[NV2A_VERTEXSHADER_ATTRIBUTES][4],
                        float o[13][4])
{
    float temps[12 * 4];
    float addr[4] = { 0, 0, 0, 0 };
    Nv2aVshExecutionState st;
    int i;

    memset(temps, 0, sizeof(temps));
    for (i = 0; i < 13; i++) {
        o[i][0] = o[i][1] = o[i][2] = 0.0f;
        o[i][3] = 1.0f;
    }
    st.input_regs = (float *)in;
    st.output_regs = &o[0][0];
    st.temp_regs = temps;
    /* Written only by a program that writes c[], which is shaded on the
     * GPU thread, in pg's constants. */
    st.context_regs = (float *)vs->constants;
    st.address_reg = addr;
    st.context_dirty = NULL;
    if (vs->program) {
        nv2a_vsh_emu_execute(&st, (const Nv2aVshProgram *)vs->program);
    }
}

static void skin(KelvinState *pg, const float v[4], int mat0, float w4, bool xyz_only,
                 const float weight[4], float out[4])
{
    unsigned int count;
    bool mix;
    int i;

    (void)w4;
    switch (s_vs.skinning) {
    case SKINNING_1WEIGHTS: mix = true; count = 2; break;
    case SKINNING_2WEIGHTS2MATRICES: mix = false; count = 2; break;
    case SKINNING_2WEIGHTS: mix = true; count = 3; break;
    case SKINNING_3WEIGHTS3MATRICES: mix = false; count = 3; break;
    case SKINNING_3WEIGHTS: mix = true; count = 4; break;
    case SKINNING_4WEIGHTS4MATRICES: mix = false; count = 4; break;
    default: mix = false; count = 0; break;
    }
    if (count == 0) {
        vec_mat(pg, v, mat0, out);
    } else {
        float weight_n = 1.0f, t[4];
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        for (i = 0; i < (int)count; i++) {
            float wi;
            if (mix) {
                if (i < (int)count - 1) {
                    wi = weight[i];
                    weight_n -= wi;
                } else {
                    wi = weight_n;
                }
            } else {
                wi = weight[i];
            }
            vec_mat(pg, v, mat0 + i * 8, t);
            out[0] += t[0] * wi; out[1] += t[1] * wi;
            out[2] += t[2] * wi; out[3] += t[3] * wi;
        }
    }
    if (xyz_only) {
        out[3] = 0.0f;
    }
}

static void run_fixed_function(KelvinState *pg,
                               const float in[NV2A_VERTEXSHADER_ATTRIBUTES][4],
                               float o[13][4], float *fog_distance)
{
    const float *position = in[0], *weight = in[1], *diffuse = in[3],
                *specular = in[4], *back_diffuse = in[7],
                *back_specular = in[8];
    float normal4[4] = { in[2][0], in[2][1], in[2][2], 0.0f };
    float tpos[4], tnorm4[4], tnorm[3];
    int i, j;

    for (i = 0; i < 13; i++) {
        o[i][0] = o[i][1] = o[i][2] = 0.0f;
        o[i][3] = 1.0f;
    }

    skin(pg, position, NV_IGRAPH_XF_XFCTX_MMAT0, 1.0f, false, weight, tpos);
    skin(pg, normal4, NV_IGRAPH_XF_XFCTX_IMMAT0, 0.0f, true, weight, tnorm4);
    tnorm[0] = tnorm4[0]; tnorm[1] = tnorm4[1]; tnorm[2] = tnorm4[2];
    if (s_vs.normalization) {
        normalize3(tnorm);
    }

    /* texgen */
    for (i = 0; i < 4; i++) {
        const float *texture = in[9 + i];
        float *ot = o[NV2AOR_TEX0 + i];
        for (j = 0; j < 4; j++) {
            switch (s_vs.texgen[i][j]) {
            case TEXGEN_DISABLE:
                ot[j] = texture[j];
                break;
            case TEXGEN_EYE_LINEAR: {
                const float *p = C(pg, NV_IGRAPH_XF_XFCTX_TG0MAT + i * 8 + j);
                ot[j] = p[0] * tpos[0] + p[1] * tpos[1] + p[2] * tpos[2] +
                        p[3] * tpos[3];
                break;
            }
            case TEXGEN_OBJECT_LINEAR: {
                const float *p = C(pg, NV_IGRAPH_XF_XFCTX_TG0MAT + i * 8 + j);
                ot[j] = p[0] * position[0] + p[1] * position[1] +
                        p[2] * position[2] + p[3] * position[3];
                break;
            }
            case TEXGEN_SPHERE_MAP:
            case TEXGEN_REFLECTION_MAP: {
                float u[3] = { tpos[0], tpos[1], tpos[2] }, r[3], d;
                normalize3(u);
                d = dot3(tnorm, u);
                r[0] = u[0] - 2.0f * d * tnorm[0];
                r[1] = u[1] - 2.0f * d * tnorm[1];
                r[2] = u[2] - 2.0f * d * tnorm[2];
                if (s_vs.texgen[i][j] == TEXGEN_SPHERE_MAP) {
                    float ro[3] = { r[0], r[1], r[2] + 1.0f };
                    float inv_m = 1.0f / (2.0f * sqrtf(dot3(ro, ro)));
                    ot[j] = j < 3 ? r[j] * inv_m + 0.5f : 0.0f;
                } else {
                    ot[j] = j < 3 ? r[j] : 0.0f;
                }
                break;
            }
            case TEXGEN_NORMAL_MAP:
                ot[j] = j < 3 ? tnorm[j] : 0.0f;
                break;
            default:
                kelvin_unsupported(KU_FF_TEXGEN, "texgen mode", s_vs.texgen[i][j]);
                break;
            }
        }
    }
    for (i = 0; i < 4; i++) {
        if (s_vs.texture_matrix_enable[i]) {
            float t[4];
            vec_mat(pg, o[NV2AOR_TEX0 + i], NV_IGRAPH_XF_XFCTX_T0MAT + i * 8, t);
            memcpy(o[NV2AOR_TEX0 + i], t, sizeof(t));
        }
    }

    if (!s_vs.lighting) {
        memcpy(o[NV2AOR_DIFFUSE], diffuse, 16);
        memcpy(o[NV2AOR_SPECULAR], specular, 16);
        memcpy(o[NV2AOR_BACK_DIFFUSE], back_diffuse, 16);
        memcpy(o[NV2AOR_BACK_SPECULAR], back_specular, 16);
    } else {
        const float *scene_ambient =
            (const float *)pg->ltctxa[NV_IGRAPH_XF_LTCTXA_FR_AMB];
        const float *emission = (const float *)pg->ltctxa[NV_IGRAPH_XF_LTCTXA_CM_COL];
        float *d0 = o[NV2AOR_DIFFUSE], *d1 = o[NV2AOR_SPECULAR];
        float alpha = s_vs.diffuse_src == MATERIAL_COLOR_SRC_MATERIAL
                          ? s_vs.material_alpha
                          : (s_vs.diffuse_src == MATERIAL_COLOR_SRC_SPECULAR
                                 ? specular[3] : diffuse[3]);
        const float *amb = s_vs.ambient_src == MATERIAL_COLOR_SRC_MATERIAL
                               ? scene_ambient
                               : (s_vs.ambient_src == MATERIAL_COLOR_SRC_DIFFUSE
                                      ? diffuse : specular);
        const float *emi = s_vs.emission_src == MATERIAL_COLOR_SRC_MATERIAL
                               ? scene_ambient
                               : (s_vs.emission_src == MATERIAL_COLOR_SRC_DIFFUSE
                                      ? diffuse : specular);
        float vpeye[3] = { 0, 0, 0 };

        for (j = 0; j < 3; j++) {
            d0[j] = amb[j] * emission[j] + emi[j];
        }
        d0[3] = alpha;
        d1[0] = d1[1] = d1[2] = 0.0f;
        d1[3] = specular[3];

        if (s_vs.local_eye) {
            const float *eye = C(pg, NV_IGRAPH_XF_XFCTX_EYEP);
            for (j = 0; j < 3; j++) {
                vpeye[j] = eye[j] / eye[3] - tpos[j] / tpos[3];
            }
            normalize3(vpeye);
        }

        for (i = 0; i < NV2A_MAX_LIGHTS; i++) {
            float attenuation = 1.0f, ndotvp = 0.0f, ndothv = 0.0f, pf;
            float vp[3] = { 0, 0, 0 };
            const float *la, *ld, *ls;
            bool lit = true;

            if (s_vs.light[i] == LIGHT_OFF) {
                continue;
            }
            if (s_vs.light[i] == LIGHT_LOCAL || s_vs.light[i] == LIGHT_SPOT) {
                float tp[3] = { tpos[0] / tpos[3], tpos[1] / tpos[3],
                                tpos[2] / tpos[3] };
                float d, hv[3];
                const float *att = pg->light_local_attenuation[i];
                for (j = 0; j < 3; j++) {
                    vp[j] = pg->light_local_position[i][j] - tp[j];
                }
                d = sqrtf(dot3(vp, vp));
                if (d > kelvin_u2f(pg->ltc1[NV_IGRAPH_XF_LTC1_r0 + i][0])) {
                    lit = false;
                } else {
                    normalize3(vp);
                    attenuation = 1.0f / (att[0] + att[1] * d + att[2] * d * d);
                    for (j = 0; j < 3; j++) {
                        hv[j] = vp[j] + vpeye[j];
                    }
                    normalize3(hv);
                    ndotvp = MAX(0.0f, dot3(tnorm, vp));
                    ndothv = MAX(0.0f, dot3(tnorm, hv));
                }
                if (lit && s_vs.light[i] == LIGHT_SPOT) {
                    const float *spot = (const float *)
                        pg->ltctxa[NV_IGRAPH_XF_LTCTXA_L0_SPT + i * 2];
                    float inv_scale = 1.0f / sqrtf(dot3(spot, spot));
                    float cos_half_phi = -inv_scale * spot[3];
                    float cos_half_theta = inv_scale + cos_half_phi;
                    float sd = dot3(spot, vp);
                    float rho = inv_scale * sd;
                    if (rho > cos_half_theta) {
                    } else if (rho <= cos_half_phi) {
                        attenuation = 0.0f;
                    } else {
                        attenuation *= sd + spot[3];
                    }
                }
            } else {
                float dir[3] = { pg->light_infinite_direction[i][0],
                                 pg->light_infinite_direction[i][1],
                                 pg->light_infinite_direction[i][2] };
                normalize3(dir);
                ndotvp = MAX(0.0f, dot3(tnorm, dir));
                if (s_vs.local_eye) {
                    float hv[3] = { dir[0] + vpeye[0], dir[1] + vpeye[1],
                                    dir[2] + vpeye[2] };
                    normalize3(hv);
                    ndothv = MAX(0.0f, dot3(tnorm, hv));
                } else {
                    ndothv = MAX(0.0f, dot3(tnorm,
                                            pg->light_infinite_half_vector[i]));
                }
            }
            if (!lit) {
                continue;
            }
            pf = (ndotvp == 0.0f || ndothv == 0.0f)
                     ? 0.0f : powf(ndothv, s_vs.specular_power);
            la = (const float *)pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_AMB + i * 6];
            ld = (const float *)pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_DIF + i * 6];
            ls = (const float *)pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_SPC + i * 6];
            for (j = 0; j < 3; j++) {
                float lamb = la[j] * attenuation;
                float ldif = ld[j] * attenuation * ndotvp;
                float lspc = ls[j] * attenuation * pf;
                d0[j] += lamb;
                d0[j] += s_vs.diffuse_src == MATERIAL_COLOR_SRC_MATERIAL
                             ? ldif
                             : (s_vs.diffuse_src == MATERIAL_COLOR_SRC_DIFFUSE
                                    ? diffuse[j] * ldif : specular[j] * ldif);
                d1[j] += s_vs.specular_src == MATERIAL_COLOR_SRC_MATERIAL
                             ? lspc
                             : (s_vs.specular_src == MATERIAL_COLOR_SRC_DIFFUSE
                                    ? diffuse[j] * lspc : specular[j] * lspc);
            }
        }
        memcpy(o[NV2AOR_BACK_DIFFUSE], back_diffuse, 16);
        memcpy(o[NV2AOR_BACK_SPECULAR], back_specular, 16);
    }

    if (!s_vs.specular_enable) {
        static const float black[4] = { 0, 0, 0, 1 };
        memcpy(o[NV2AOR_SPECULAR], black, 16);
        memcpy(o[NV2AOR_BACK_SPECULAR], black, 16);
    } else {
        if (!s_vs.separate_specular) {
            if (s_vs.lighting) {
                for (j = 0; j < 3; j++) {
                    o[NV2AOR_DIFFUSE][j] += o[NV2AOR_SPECULAR][j];
                    o[NV2AOR_BACK_DIFFUSE][j] += o[NV2AOR_BACK_SPECULAR][j];
                }
            }
            memcpy(o[NV2AOR_SPECULAR], specular, 16);
            memcpy(o[NV2AOR_BACK_SPECULAR], back_specular, 16);
        }
        if (s_vs.ignore_specular_alpha) {
            o[NV2AOR_SPECULAR][3] = 1.0f;
            o[NV2AOR_BACK_SPECULAR][3] = 1.0f;
        }
    }

    if (s_vs.fog_enable) {
        switch (s_vs.foggen) {
        case FOGGEN_SPEC_ALPHA: {
            float a = specular[3];
            *fog_distance = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
            break;
        }
        case FOGGEN_RADIAL:
            *fog_distance = sqrtf(tpos[0] * tpos[0] + tpos[1] * tpos[1] +
                                  tpos[2] * tpos[2]);
            break;
        case FOGGEN_PLANAR:
        case FOGGEN_ABS_PLANAR: {
            const float *fp = C(pg, NV_IGRAPH_XF_XFCTX_FOG);
            float d = fp[0] * tpos[0] + fp[1] * tpos[1] + fp[2] * tpos[2] + fp[3];
            *fog_distance = s_vs.foggen == FOGGEN_ABS_PLANAR ? fabsf(d) : d;
            break;
        }
        case FOGGEN_FOG_X:
            *fog_distance = in[5][0];
            break;
        default:
            kelvin_unsupported(KU_FF_FOGGEN, "fog gen mode", s_vs.foggen);
            *fog_distance = 0.0f;
            break;
        }
    }

    /* If skinning is off the composite matrix already includes the MV matrix */
    if (s_vs.skinning == SKINNING_OFF) {
        memcpy(tpos, position, sizeof(tpos));
    }
    {
        float p[4];
        const float *vpoff = C(pg, NV_IGRAPH_XF_XFCTX_VPOFF);
        vec_mat(pg, tpos, NV_IGRAPH_XF_XFCTX_CMAT0, p);
        p[3] = clamp_away_zero_inf(p[3]);
        p[0] = p[0] / p[3] + vpoff[0];
        p[1] = p[1] / p[3] + vpoff[1];
        p[2] = p[2] / p[3];
        memcpy(o[NV2AOR_POS], p, sizeof(p));
    }
}

/* The current transform program, one line per instruction slot, for the
 * draw trace. Printed once per distinct program. */
void kvsh_trace_program(void)
{
    static const char *const ops[] = {
        "nop", "mov", "mul", "add", "mad", "dp3", "dph", "dp4", "dst", "min",
        "max", "slt", "sge", "arl", "rcp", "rcc", "rsq", "exp", "log", "lit",
    };
    static const char *const regs = "?rvocA";
    static uint32_t last_hash, last_start = ~0u;
    const Nv2aVshStep *st;
    int n, k, j, m;

    if (!s_vs.program || !s_vs.prog_valid ||
        (s_vs.prog_hash == last_hash && s_vs.prog_start == last_start)) {
        return;
    }
    last_hash = s_vs.prog_hash;
    last_start = s_vs.prog_start;
    fprintf(stderr, "[KTRACE]   program at slot %u (hash %08X):\n",
            s_vs.prog_start, s_vs.prog_hash);
    for (n = 0, st = s_vs.entry->prog.steps; n < NV2A_MAX_TRANSFORM_PROGRAM_LENGTH;
         n++, st++) {
        const Nv2aVshOperation *pair[2] = { &st->mac, &st->ilu };
        kelvin_trace_record_begin(stderr);
        fprintf(stderr, "[KTRACE]     %3d:", n);
        for (k = 0; k < 2; k++) {
            const Nv2aVshOperation *op = pair[k];
            if (op->opcode == NV2AOP_NOP) {
                continue;
            }
            fprintf(stderr, " %s%s", k ? " || " : "",
                    op->opcode < sizeof(ops) / sizeof(ops[0])
                        ? ops[op->opcode] : "?");
            for (j = 0; j < 2; j++) {
                const Nv2aVshOutput *o = &op->outputs[j];
                if (o->type == NV2ART_NONE) {
                    continue;
                }
                fprintf(stderr, " %c%u.", regs[o->type], o->index);
                for (m = 0; m < 4; m++) {
                    if (o->writemask & (8 >> m)) {
                        fputc("xyzw"[m], stderr);
                    }
                }
            }
            for (j = 0; j < 3; j++) {
                const Nv2aVshInput *in = &op->inputs[j];
                if (in->type == NV2ART_NONE) {
                    continue;
                }
                fprintf(stderr, ", %s%c%s%u.%c%c%c%c", in->is_negated ? "-" : "",
                        regs[in->type], in->is_relative ? "[a0]+" : "",
                        in->index, "xyzw"[in->swizzle[0] & 3],
                        "xyzw"[in->swizzle[1] & 3], "xyzw"[in->swizzle[2] & 3],
                        "xyzw"[in->swizzle[3] & 3]);
            }
        }
        fputc('\n', stderr);
        kelvin_trace_record_end(stderr);
        if (st->is_final) {
            break;
        }
    }
}

/* A vertex program that writes no constant registers makes each vertex
 * depend only on its own inputs and the draw's state, so kvsh_run may shade
 * several of the draw's vertices at once, on several threads. The fixed
 * function path reports unsupported modes per vertex; it stays on one. */
bool kvsh_parallel_ok(void)
{
    return s_vs.program && s_vs.prog_valid && !s_vs.prog_writes_constants;
}

/* The current draw's program for the GPU vertex path (kelvin_gpu.c), if its
 * vertices can be shaded there: a valid program that writes no constants. */
bool kvsh_gpu_info(KVshGpuInfo *out)
{
    if (!s_vs.program || !s_vs.prog_valid || s_vs.prog_writes_constants || !s_vs.entry) {
        return false;
    }
    out->program = &s_vs.entry->prog;
    out->hash = s_vs.prog_hash;
    out->start = s_vs.prog_start;
    out->words = &s_vs.entry->words[0][0];
    out->nwords = s_vs.entry->nslots * VSH_TOKEN_SIZE;
    out->constants = s_cur.constants;
    out->fog_enable = s_cur.fog_enable;
    out->fog_mode = s_cur.fog_mode;
    out->fog_param[0] = s_cur.fog_param[0];
    out->fog_param[1] = s_cur.fog_param[1];
    out->specular_enable = s_cur.specular_enable;
    out->ignore_specular_alpha = s_cur.ignore_specular_alpha;
    out->point_size = s_cur.point_size;
    return true;
}

void kvsh_state(KVshState *out, const float *constants)
{
    *out = s_cur;
    out->constants = constants;
}

/* What the rasterizer takes from a vertex's outputs. */
static void finish_vertex(const KVshState *vs, float o[13][4], float fog_distance,
                          KVertex *out)
{
    memcpy(out->pos, o[NV2AOR_POS], sizeof(out->pos));
    out->fog = vs->fog_enable ? fog_factor(vs, fog_distance) : 1.0f;

    memcpy(out->d0, o[NV2AOR_DIFFUSE], 16);
    memcpy(out->b0, o[NV2AOR_BACK_DIFFUSE], 16);
    clamp_color(out->d0);
    clamp_color(out->b0);
    if (vs->specular_enable) {
        memcpy(out->d1, o[NV2AOR_SPECULAR], 16);
        memcpy(out->b1, o[NV2AOR_BACK_SPECULAR], 16);
        clamp_color(out->d1);
        clamp_color(out->b1);
        if (vs->ignore_specular_alpha) {
            out->d1[3] = 1.0f;
            out->b1[3] = 1.0f;
        }
    } else {
        out->d1[0] = out->d1[1] = out->d1[2] = 0.0f; out->d1[3] = 1.0f;
        out->b1[0] = out->b1[1] = out->b1[2] = 0.0f; out->b1[3] = 1.0f;
    }
    memcpy(out->t, &o[NV2AOR_TEX0], sizeof(out->t));
    out->pts = vs->point_size <= 0.0f ? 1.0f : vs->point_size;
}

/* A program's outputs, with the vsh-prog.c epilogue. */
static void finish_program(const KVshState *vs, float o[13][4], KVertex *out)
{
    float fog_distance = o[NV2AOR_FOG_COORD][0];
    o[0][0] = round_screen(o[0][0]);
    o[0][1] = round_screen(o[0][1]);
    o[0][3] = clamp_away_zero_inf(o[0][3]);
    finish_vertex(vs, o, fog_distance, out);
}

/* The current draw's state with pg's constants. */
static KVshState current(KelvinState *pg)
{
    KVshState vs = s_cur;
    vs.constants = (const float *)pg->vsh_constants;
    return vs;
}

void kvsh_run_state(const KVshState *vs,
                    const float in[NV2A_VERTEXSHADER_ATTRIBUTES][4], KVertex *out)
{
    float o[13][4];
    run_program(vs, in, o);
    finish_program(vs, o, out);
}

void kvsh_run(KelvinState *pg, const float in[NV2A_VERTEXSHADER_ATTRIBUTES][4],
              KVertex *out)
{
    float o[13][4];
    float fog_distance = 0.0f;
    KVshState vs = current(pg);

    if (s_vs.program) {
        kvsh_run_state(&vs, in, out);
        return;
    }
    run_fixed_function(pg, in, o, &fog_distance);
    o[0][0] = round_screen(o[0][0]);
    o[0][1] = round_screen(o[0][1]);
    finish_vertex(&vs, o, fog_distance, out);
}

#if defined(__AVX2__)
void kvsh_run8_state(const KVshState *vs,
                     const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
                     KVertex *const out[], int n)
{
    Nv2aVshState8 st;
    float o[8][13][4];
    int lane, a, r, k;

    nv2a_vsh_emu8_reset(&st);
    st.context = vs->constants;
    /* Vertices in, components out: two 4x4 transposes per attribute, lanes
     * 0-3 low and 4-7 high; lanes past n repeat the first vertex, and
     * nothing reads them. */
    for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
        __m128 v[8];
        for (lane = 0; lane < 8; lane++) {
            v[lane] = _mm_loadu_ps(in[lane < n ? lane : 0][a]);
        }
        _MM_TRANSPOSE4_PS(v[0], v[1], v[2], v[3]);
        _MM_TRANSPOSE4_PS(v[4], v[5], v[6], v[7]);
        for (k = 0; k < 4; k++) {
            st.input[a].c[k] = _mm256_insertf128_ps(_mm256_castps128_ps256(v[k]),
                                                    v[4 + k], 1);
        }
    }
    nv2a_vsh_emu8_execute(&st, (const Nv2aVshProgram *)vs->program);
    for (r = 0; r < 13; r++) {
        __m128 lo[4], hi[4];
        for (k = 0; k < 4; k++) {
            lo[k] = _mm256_castps256_ps128(st.output[r].c[k]);
            hi[k] = _mm256_extractf128_ps(st.output[r].c[k], 1);
        }
        _MM_TRANSPOSE4_PS(lo[0], lo[1], lo[2], lo[3]);
        _MM_TRANSPOSE4_PS(hi[0], hi[1], hi[2], hi[3]);
        for (k = 0; k < 4; k++) {
            _mm_storeu_ps(o[k][r], lo[k]);
            _mm_storeu_ps(o[4 + k][r], hi[k]);
        }
    }
    for (lane = 0; lane < n; lane++) {
        finish_program(vs, o[lane], out[lane]);
    }
}

void kvsh_run8(KelvinState *pg,
               const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
               KVertex *const out[], int n)
{
    KVshState vs = current(pg);
    kvsh_run8_state(&vs, in, out, n);
}
#endif

void kvsh_run4_state(const KVshState *vs,
                     const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
                     KVertex *const out[], int n)
{
    Nv2aVshState4 st;
    float o[4][13][4];
    int lane, a, r;

    nv2a_vsh_emu4_reset(&st);
    st.context = vs->constants;
    /* Vertices in, components out, by transposing; lanes past n repeat the
     * first vertex, and nothing reads them. */
    for (a = 0; a < NV2A_VERTEXSHADER_ATTRIBUTES; a++) {
        __m128 c0 = _mm_loadu_ps(in[0][a]);
        __m128 c1 = _mm_loadu_ps(in[n > 1][a]);
        __m128 c2 = _mm_loadu_ps(in[n > 2 ? 2 : 0][a]);
        __m128 c3 = _mm_loadu_ps(in[n > 3 ? 3 : 0][a]);
        _MM_TRANSPOSE4_PS(c0, c1, c2, c3);
        st.input[a].c[0] = c0;
        st.input[a].c[1] = c1;
        st.input[a].c[2] = c2;
        st.input[a].c[3] = c3;
    }
    nv2a_vsh_emu4_execute(&st, (const Nv2aVshProgram *)vs->program);
    for (r = 0; r < 13; r++) {
        __m128 c0 = st.output[r].c[0], c1 = st.output[r].c[1];
        __m128 c2 = st.output[r].c[2], c3 = st.output[r].c[3];
        _MM_TRANSPOSE4_PS(c0, c1, c2, c3);
        _mm_storeu_ps(o[0][r], c0);
        _mm_storeu_ps(o[1][r], c1);
        _mm_storeu_ps(o[2][r], c2);
        _mm_storeu_ps(o[3][r], c3);
    }
    for (lane = 0; lane < n; lane++) {
        finish_program(vs, o[lane], out[lane]);
    }
}

void kvsh_run4(KelvinState *pg,
               const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
               KVertex *const out[], int n)
{
    KVshState vs = current(pg);
    kvsh_run4_state(&vs, in, out, n);
}
