/*
 * Kelvin pixel stage: texture shader, register combiners, alpha test,
 * stencil/depth and the output merger.
 *
 * The texture modes, combiner arithmetic and final combiner compute what the
 * hardware computes for the same registers, including its order: every input
 * of a combiner stage is read before any of its outputs is written. The
 * reference for that behaviour is xemu's glsl/psh.c, which is GPL-2.0+ and
 * generates GLSL; nothing of it is copied here -- this is an independent CPU
 * evaluator of the same register semantics (see NOTICE). Blending, stencil
 * and depth tests follow the GL state xemu's LGPL gl/draw.c sets from the
 * PGRAPH registers.
 *
 * Copyright (c) 2012 espes; 2015 Jannik Vogel; 2018-2025 Matt Borgerson
 * (xemu gl/draw.c, whose state mapping the output merger follows)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include "kelvin_render.h"

#include <emmintrin.h>

/* Integer selectors only: uniforms retain their per-batch float values.
 * This lives outside KPixelState so captures retain their original ABI. */
typedef struct CombinerShape {
    int num_stages, flags;
    bool final_enabled, clamp_sum, inv_v1, inv_r0;
    KCombinerInput fin[7];
    KCombinerStage stage[6];
} CombinerShape;

static const CombinerShape world_shape = {
    .num_stages = 2, .flags = 273,
    .final_enabled = true,
    .clamp_sum = true,
    .inv_v1 = false, .inv_r0 = false,
    .fin = { { 3, 0, 16 }, { 12, 0, 0 }, { 3, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 32, 0 } },
    .stage = {
        { .rgb = { { 8, 192, 0 }, { 0, 32, 0 }, { 9, 192, 0 }, { 0, 64, 0 } },
          .alpha = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .rgb_out = { 0, 0, 12, 0, 0, 0, 0, 0 },
          .alpha_out = { 0, 0, 0, 0, 0, 0, 0, 0 } },
        { .rgb = { { 12, 192, 0 }, { 9, 192, 16 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .alpha = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .rgb_out = { 12, 0, 0, 0, 0, 0, 0, 0 },
          .alpha_out = { 0, 0, 0, 0, 0, 0, 0, 0 } }
    }
};

static const CombinerShape background_shape = {
    .num_stages = 6, .flags = 273,
    .final_enabled = true,
    .clamp_sum = true,
    .inv_v1 = false, .inv_r0 = false,
    .fin = { { 0, 0, 0 }, { 0, 0, 0 }, { 12, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
    .stage = {
        { .rgb = { { 9, 192, 0 }, { 4, 192, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .alpha = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .rgb_out = { 9, 0, 0, 0, 0, 0, 0, 0 },
          .alpha_out = { 0, 0, 0, 0, 0, 0, 0, 0 } },
        { .rgb = { { 1, 192, 0 }, { 9, 192, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .alpha = { { 1, 192, 16 }, { 9, 192, 16 }, { 0, 0, 16 }, { 0, 0, 16 } },
          .rgb_out = { 13, 0, 0, 0, 0, 0, 0, 0 },
          .alpha_out = { 13, 0, 0, 0, 0, 0, 0, 0 } },
        { .rgb = { { 8, 192, 0 }, { 4, 192, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .alpha = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .rgb_out = { 8, 0, 0, 0, 0, 0, 0, 0 },
          .alpha_out = { 0, 0, 0, 0, 0, 0, 0, 0 } },
        { .rgb = { { 8, 192, 0 }, { 4, 192, 16 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .alpha = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .rgb_out = { 8, 0, 0, 0, 0, 0, 0, 0 },
          .alpha_out = { 0, 0, 0, 0, 0, 0, 0, 0 } },
        { .rgb = { { 1, 192, 0 }, { 8, 192, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .alpha = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
          .rgb_out = { 8, 0, 0, 0, 0, 0, 0, 0 },
          .alpha_out = { 0, 0, 0, 0, 0, 0, 0, 0 } },
        { .rgb = { { 13, 192, 0 }, { 13, 192, 16 }, { 8, 192, 0 }, { 13, 32, 16 } },
          .alpha = { { 13, 192, 16 }, { 13, 192, 16 }, { 8, 192, 16 }, { 13, 32, 16 } },
          .rgb_out = { 0, 0, 12, 0, 0, 0, 0, 0 },
          .alpha_out = { 0, 0, 12, 0, 0, 0, 0, 0 } }
    }
};

static bool input_is(const KCombinerInput *a, const KCombinerInput *b)
{
    return a->reg == b->reg && a->mod == b->mod && a->chan == b->chan;
}

static bool output_is(const KCombinerOutput *a, const KCombinerOutput *b)
{
    return a->ab == b->ab && a->cd == b->cd && a->muxsum == b->muxsum &&
           a->flags == b->flags && a->ab_op == b->ab_op &&
           a->cd_op == b->cd_op && a->muxsum_op == b->muxsum_op &&
           a->mapping == b->mapping;
}

static bool combiner_is(const KPixelState *ps, const CombinerShape *shape)
{
    if (ps->num_stages != shape->num_stages || ps->flags != shape->flags ||
        ps->final_enabled != shape->final_enabled ||
        ps->clamp_sum != shape->clamp_sum || ps->inv_v1 != shape->inv_v1 ||
        ps->inv_r0 != shape->inv_r0) return false;
    for (int i = 0; i < 7; i++) {
        if (!input_is(&ps->fin[i], &shape->fin[i])) return false;
    }
    for (int i = 0; i < shape->num_stages; i++) {
        const KCombinerStage *a = &ps->stage[i], *b = &shape->stage[i];
        for (int k = 0; k < 4; k++) {
            if (!input_is(&a->rgb[k], &b->rgb[k]) ||
                !input_is(&a->alpha[k], &b->alpha[k])) return false;
        }
        if (!output_is(&a->rgb_out, &b->rgb_out) ||
            !output_is(&a->alpha_out, &b->alpha_out)) return false;
    }
    return true;
}

/* Private high bits carry integer classifications of immutable draw state.
 * The low combiner ID and the public state/capture layouts stay unchanged.
 * Float-dependent merger guards stay in the fragment path: moving those
 * comparisons to the producer could change each worker's MXCSR status. */
#define KPIX_SELECT_COMBINER 3u
#define KPIX_SELECT_STAGE(i) (16u << (i))
#define KPIX_SELECT_PREPARED 256u
#define KPIX_SELECT_INDEPENDENT 512u
#define KPIX_SELECT_TAIL_STATE 1024u

static bool stage8_eligible(const KPixelState *ps, int i)
{
    const KTexture *tx = &ps->tex[i];
    return ps->tex_mode[i] == PS_TEXTUREMODES_PROJECT2D && tx->valid &&
           !tx->cubemap && !tx->colorkey_mode;
}

static bool tail8_state_eligible(const KPixelState *ps)
{
    return !(ps->alpha_test && ps->alpha_func != 7) && !ps->stencil_test &&
           (ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8 ||
            ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8 ||
            ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8);
}

KPixelProgram kpix_program(const KPixelState *ps)
{
    unsigned int program = KPIX_SELECT_PREPARED;
    bool independent = true;
    if (combiner_is(ps, &world_shape)) program |= KPIX_PROGRAM_WORLD_FOG;
    else if (combiner_is(ps, &background_shape)) program |= KPIX_PROGRAM_BACKGROUND;
    for (int i = 0; i < 4; i++) {
        if (stage8_eligible(ps, i)) program |= KPIX_SELECT_STAGE(i);
        else if (ps->tex_mode[i] != PS_TEXTUREMODES_NONE) independent = false;
    }
    if (independent) program |= KPIX_SELECT_INDEPENDENT;
    if (tail8_state_eligible(ps)) program |= KPIX_SELECT_TAIL_STATE;
    return (KPixelProgram)program;
}

static void argb_to_rgba(uint32_t argb, float *rgba)
{
    rgba[0] = ((argb >> 16) & 0xFF) / 255.0f;
    rgba[1] = ((argb >> 8) & 0xFF) / 255.0f;
    rgba[2] = (argb & 0xFF) / 255.0f;
    rgba[3] = ((argb >> 24) & 0xFF) / 255.0f;
}

static void parse_input(KCombinerInput *in, int value)
{
    in->reg = value & 0xF;
    in->chan = value & 0x10;
    in->mod = value & 0xE0;
}

static void parse_inputs(uint32_t value, KCombinerInput in[4])
{
    parse_input(&in[3], value & 0xFF);          /* d */
    parse_input(&in[2], (value >> 8) & 0xFF);   /* c */
    parse_input(&in[1], (value >> 16) & 0xFF);  /* b */
    parse_input(&in[0], (value >> 24) & 0xFF);  /* a */
}

static void parse_output(uint32_t value, KCombinerOutput *out)
{
    int flags = value >> 12;
    out->cd = value & 0xF;
    out->ab = (value >> 4) & 0xF;
    out->muxsum = (value >> 8) & 0xF;
    out->flags = flags;
    out->cd_op = flags & 1;
    out->ab_op = flags & 2;
    out->muxsum_op = flags & 4;
    out->mapping = flags & 0x38;
}

static bool stage_active(uint32_t shaderprog, int stage)
{
    uint32_t mode = (shaderprog >> (stage * 5)) & 0x1F;
    return mode != 0 && mode != 4;
}

void kpix_setup(KelvinState *pg, KPixelState *ps, const KelvinTarget *t)
{
    uint32_t combinectl = pgraph_reg_r(pg, NV_PGRAPH_COMBINECTL);
    uint32_t shaderprog = pgraph_reg_r(pg, NV_PGRAPH_SHADERPROG);
    uint32_t shaderctl = pgraph_reg_r(pg, NV_PGRAPH_SHADERCTL);
    uint32_t fin0 = pgraph_reg_r(pg, NV_PGRAPH_COMBINESPECFOG0);
    uint32_t fin1 = pgraph_reg_r(pg, NV_PGRAPH_COMBINESPECFOG1);
    uint32_t control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    uint32_t control_1 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_1);
    uint32_t control_2 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_2);
    uint32_t control_3 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_3);
    uint32_t blend = pgraph_reg_r(pg, NV_PGRAPH_BLEND);
    uint32_t setupraster = pgraph_reg_r(pg, NV_PGRAPH_SETUPRASTER);
    uint32_t clipmode = pgraph_reg_r(pg, NV_PGRAPH_SHADERCLIPMODE);
    uint32_t fog = pgraph_reg_r(pg, NV_PGRAPH_FOGCOLOR);
    int i, j;

    memset(ps, 0, sizeof(*ps));

    ps->num_stages = combinectl & 0xFF;
    if (ps->num_stages > 8) {
        ps->num_stages = 8;
    }
    ps->flags = combinectl >> 8;
    for (i = 0; i < ps->num_stages; i++) {
        KCombinerStage *s = &ps->stage[i];
        parse_inputs(pgraph_reg_r(pg, NV_PGRAPH_COMBINECOLORI0 + i * 4), s->rgb);
        parse_inputs(pgraph_reg_r(pg, NV_PGRAPH_COMBINEALPHAI0 + i * 4), s->alpha);
        parse_output(pgraph_reg_r(pg, NV_PGRAPH_COMBINECOLORO0 + i * 4), &s->rgb_out);
        parse_output(pgraph_reg_r(pg, NV_PGRAPH_COMBINEALPHAO0 + i * 4), &s->alpha_out);
    }
    for (i = 0; i < 8; i++) {
        argb_to_rgba(pgraph_reg_r(pg, NV_PGRAPH_COMBINEFACTOR0 + i * 4),
                     ps->stage[i].c0);
        argb_to_rgba(pgraph_reg_r(pg, NV_PGRAPH_COMBINEFACTOR1 + i * 4),
                     ps->stage[i].c1);
    }
    argb_to_rgba(pgraph_reg_r(pg, NV_PGRAPH_SPECFOGFACTOR0), ps->final_c0);
    argb_to_rgba(pgraph_reg_r(pg, NV_PGRAPH_SPECFOGFACTOR1), ps->final_c1);

    ps->final_enabled = fin0 || fin1;
    if (ps->final_enabled) {
        KCombinerInput blank[4];
        parse_inputs(fin0, &ps->fin[0]);          /* a b c d */
        parse_inputs(fin1, blank);                /* e f g - */
        ps->fin[4] = blank[0];
        ps->fin[5] = blank[1];
        ps->fin[6] = blank[2];
        ps->clamp_sum = fin1 & PS_FINALCOMBINERSETTING_CLAMP_SUM;
        ps->inv_v1 = fin1 & PS_FINALCOMBINERSETTING_COMPLEMENT_V1;
        ps->inv_r0 = fin1 & PS_FINALCOMBINERSETTING_COMPLEMENT_R0;
    }

    for (i = 0; i < 4; i++) {
        ps->tex_mode[i] = (shaderprog >> (i * 5)) & 0x1F;
    }
    ps->dot_map[0] = 0;
    ps->dot_map[1] = shaderctl & 0xF;
    ps->dot_map[2] = (shaderctl >> 4) & 0xF;
    ps->dot_map[3] = (shaderctl >> 8) & 0xF;
    ps->input_tex[0] = -1;
    ps->input_tex[1] = 0;
    ps->input_tex[2] = (shaderctl >> 16) & 0xF;
    ps->input_tex[3] = (shaderctl >> 20) & 0xF;

    for (i = 0; i < 4; i++) {
        uint32_t ctl_0 = pgraph_reg_r(pg, NV_PGRAPH_TEXCTL0_0 + i * 4);
        for (j = 0; j < 4; j++) {
            ps->compare_mode[i][j] = (clipmode >> (4 * i + j)) & 1;
        }
        if (stage_active(shaderprog, i) && (ctl_0 & NV_PGRAPH_TEXCTL0_0_ENABLE)) {
            ktex_setup(pg, i, &ps->tex[i]);
            ps->rect_tex[i] = ps->tex[i].linear;
            ps->lod_free[i] = ktex_lod_free(&ps->tex[i]);
            if (ps->tex[i].min_filter == NV_PGRAPH_TEXFILTER0_MIN_CONVOLUTION_2D_LOD0) {
                kelvin_unsupported(KU_CONVOLUTION, "convolution filter", i);
            }
        }
        ps->color_key[i] = pgraph_reg_r(pg, NV_PGRAPH_COLORKEYCOLOR0 + i * 4);
        {
            unsigned int cf = ps->tex[i].color_format;
            ps->color_key_mask[i] =
                (cf == NV097_SET_TEXTURE_FORMAT_COLOR_SZ_X1R5G5B5 ||
                 cf == NV097_SET_TEXTURE_FORMAT_COLOR_SZ_X8R8G8B8 ||
                 cf == NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_X1R5G5B5 ||
                 cf == NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_X8R8G8B8)
                    ? 0x00FFFFFF : 0xFFFFFFFF;
        }
        if (i > 0) {
            ps->bump_mat[i][0] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_BUMPMAT00 + 4 * (i - 1)));
            ps->bump_mat[i][1] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_BUMPMAT01 + 4 * (i - 1)));
            ps->bump_mat[i][2] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_BUMPMAT10 + 4 * (i - 1)));
            ps->bump_mat[i][3] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_BUMPMAT11 + 4 * (i - 1)));
            ps->bump_scale[i] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_BUMPSCALE1 + (i - 1) * 4));
            ps->bump_offset[i] = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_BUMPOFFSET1 + (i - 1) * 4));
        }
    }

    ps->fog_color[0] = GET_MASK(fog, NV_PGRAPH_FOGCOLOR_RED) / 255.0f;
    ps->fog_color[1] = GET_MASK(fog, NV_PGRAPH_FOGCOLOR_GREEN) / 255.0f;
    ps->fog_color[2] = GET_MASK(fog, NV_PGRAPH_FOGCOLOR_BLUE) / 255.0f;

    ps->alpha_test = control_0 & NV_PGRAPH_CONTROL_0_ALPHATESTENABLE;
    ps->alpha_func = GET_MASK(control_0, NV_PGRAPH_CONTROL_0_ALPHAFUNC);
    ps->alpha_ref = GET_MASK(control_0, NV_PGRAPH_CONTROL_0_ALPHAREF);

    ps->depth_test = control_0 & NV_PGRAPH_CONTROL_0_ZENABLE;
    ps->depth_write = (control_0 & NV_PGRAPH_CONTROL_0_ZWRITEENABLE) && ps->depth_test;
    ps->depth_func = GET_MASK(control_0, NV_PGRAPH_CONTROL_0_ZFUNC);
    ps->stencil_test = control_1 & NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE;
    ps->stencil_write = control_0 & NV_PGRAPH_CONTROL_0_STENCIL_WRITE_ENABLE;
    ps->stencil_func = GET_MASK(control_1, NV_PGRAPH_CONTROL_1_STENCIL_FUNC);
    ps->stencil_ref = GET_MASK(control_1, NV_PGRAPH_CONTROL_1_STENCIL_REF);
    ps->stencil_mask_read = GET_MASK(control_1, NV_PGRAPH_CONTROL_1_STENCIL_MASK_READ);
    ps->stencil_mask_write = GET_MASK(control_1, NV_PGRAPH_CONTROL_1_STENCIL_MASK_WRITE);
    ps->stencil_op_fail = GET_MASK(control_2, NV_PGRAPH_CONTROL_2_STENCIL_OP_FAIL);
    ps->stencil_op_zfail = GET_MASK(control_2, NV_PGRAPH_CONTROL_2_STENCIL_OP_ZFAIL);
    ps->stencil_op_zpass = GET_MASK(control_2, NV_PGRAPH_CONTROL_2_STENCIL_OP_ZPASS);
    ps->zeta_format = pg->surface_shape.zeta_format;
    ps->z_float = GET_MASK(setupraster, NV_PGRAPH_SETUPRASTER_Z_FORMAT);
    ps->z_perspective = control_0 & NV_PGRAPH_CONTROL_0_Z_PERSPECTIVE_ENABLE;
    ps->zmax = ps->zeta_format == NV097_SET_SURFACE_FORMAT_ZETA_Z16
                   ? (ps->z_float ? 511.9375f : 65535.0f)
                   : (ps->z_float ? 1.0e30f : 16777215.0f);
    ps->zclip_min = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_ZCLIPMIN));
    ps->zclip_max = kelvin_u2f(pgraph_reg_r(pg, NV_PGRAPH_ZCLIPMAX));
    ps->depth_clip_cull =
        GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_ZCOMPRESSOCCLUDE),
                 NV_PGRAPH_ZCOMPRESSOCCLUDE_ZCLAMP_EN)
        == NV_PGRAPH_ZCOMPRESSOCCLUDE_ZCLAMP_EN_CULL;
    if (ps->z_float && (ps->depth_test || ps->stencil_test)) {
        kelvin_unsupported(KU_ZETA_FLOAT, "float depth buffer", ps->zeta_format);
    }

    ps->blend = blend & NV_PGRAPH_BLEND_EN;
    ps->blend_sfactor = GET_MASK(blend, NV_PGRAPH_BLEND_SFACTOR);
    ps->blend_dfactor = GET_MASK(blend, NV_PGRAPH_BLEND_DFACTOR);
    ps->blend_eqn = GET_MASK(blend, NV_PGRAPH_BLEND_EQN);
    argb_to_rgba(pgraph_reg_r(pg, NV_PGRAPH_BLENDCOLOR), ps->blend_color);
    if (blend & NV_PGRAPH_BLEND_LOGICOP_ENABLE) {
        kelvin_unsupported(KU_LOGIC_OP, "logic op", blend);
    }
    ps->mask_r = control_0 & NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE;
    ps->mask_g = control_0 & NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE;
    ps->mask_b = control_0 & NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE;
    ps->mask_a = control_0 & NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE;
    ps->color_format = pg->surface_shape.color_format;

    ps->window_clip_exclusive = setupraster & NV_PGRAPH_SETUPRASTER_WINDOWCLIPTYPE;
    for (i = 0; i < 8; i++) {
        uint32_t x = pgraph_reg_r(pg, NV_PGRAPH_WINDOWCLIPX0 + i * 4);
        uint32_t y = pgraph_reg_r(pg, NV_PGRAPH_WINDOWCLIPY0 + i * 4);
        ps->clip_region[i][0] = (int)floor(GET_MASK(x, NV_PGRAPH_WINDOWCLIPX0_XMIN) * kelvin_raster_x(t));
        ps->clip_region[i][1] = (int)floor(GET_MASK(y, NV_PGRAPH_WINDOWCLIPY0_YMIN) * kelvin_raster_y(t));
        ps->clip_region[i][2] = (int)ceil((GET_MASK(x, NV_PGRAPH_WINDOWCLIPX0_XMAX) + 1) * kelvin_raster_x(t));
        ps->clip_region[i][3] = (int)ceil((GET_MASK(y, NV_PGRAPH_WINDOWCLIPY0_YMAX) + 1) * kelvin_raster_y(t));
    }
    ps->smooth = GET_MASK(control_3, NV_PGRAPH_CONTROL_3_SHADEMODE)
                 == NV_PGRAPH_CONTROL_3_SHADEMODE_SMOOTH;
}

/* ---- combiners ------------------------------------------------------------ */

typedef struct Regs {
    float c0[4], c1[4], fog[4], v0[4], v1[4], t[4][4], r0[4], r1[4];
    float ef[4];                     /* lane 3 unused */
} Regs;

static const float s_zero4[4] = { 0, 0, 0, 0 };

KELVIN_INLINE const float *reg_ptr(const KPixelState *ps, const Regs *r, int reg,
                            int stage, float scratch[4])
{
    switch (reg) {
    case PS_REGISTER_C0:
        if ((ps->flags & PS_COMBINERCOUNT_UNIQUE_C0) || stage == 8) {
            return stage == 8 ? ps->final_c0 : ps->stage[stage].c0;
        }
        return ps->stage[0].c0;
    case PS_REGISTER_C1:
        if ((ps->flags & PS_COMBINERCOUNT_UNIQUE_C1) || stage == 8) {
            return stage == 8 ? ps->final_c1 : ps->stage[stage].c1;
        }
        return ps->stage[0].c1;
    case PS_REGISTER_FOG: return r->fog;
    case PS_REGISTER_V0: return r->v0;
    case PS_REGISTER_V1: return r->v1;
    case PS_REGISTER_T0: return r->t[0];
    case PS_REGISTER_T1: return r->t[1];
    case PS_REGISTER_T2: return r->t[2];
    case PS_REGISTER_T3: return r->t[3];
    case PS_REGISTER_R0: return r->r0;
    case PS_REGISTER_R1: return r->r1;
    case PS_REGISTER_V1R0_SUM: {
        int k;
        for (k = 0; k < 3; k++) {
            float a = ps->inv_v1 ? 1.0f - r->v1[k] : r->v1[k];
            float b = ps->inv_r0 ? 1.0f - r->r0[k] : r->r0[k];
            float s = a + b;
            if (ps->clamp_sum) {
                s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
            }
            scratch[k] = s;
        }
        scratch[3] = 0.0f;
        return scratch;
    }
    case PS_REGISTER_EF_PROD:
        scratch[0] = r->ef[0];
        scratch[1] = r->ef[1];
        scratch[2] = r->ef[2];
        scratch[3] = 0.0f;
        return scratch;
    default:
        return s_zero4;
    }
}

KELVIN_INLINE float map_input(float x, int mod)
{
    switch (mod) {
    case PS_INPUTMAPPING_UNSIGNED_IDENTITY: return x > 0.0f ? x : 0.0f;
    case PS_INPUTMAPPING_UNSIGNED_INVERT:
        return 1.0f - (x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x));
    case PS_INPUTMAPPING_EXPAND_NORMAL: return 2.0f * (x > 0.0f ? x : 0.0f) - 1.0f;
    case PS_INPUTMAPPING_EXPAND_NEGATE: return -2.0f * (x > 0.0f ? x : 0.0f) + 1.0f;
    case PS_INPUTMAPPING_HALFBIAS_NORMAL: return (x > 0.0f ? x : 0.0f) - 0.5f;
    case PS_INPUTMAPPING_HALFBIAS_NEGATE: return -(x > 0.0f ? x : 0.0f) + 0.5f;
    case PS_INPUTMAPPING_SIGNED_IDENTITY: return x;
    case PS_INPUTMAPPING_SIGNED_NEGATE: return -x;
    default: return x;
    }
}

/* map_input on three lanes, choosing the mapping once. */
KELVIN_INLINE void map_input3(const float v[3], int mod, float out[3])
{
    int k;
    switch (mod) {
    case PS_INPUTMAPPING_UNSIGNED_IDENTITY:
        for (k = 0; k < 3; k++) out[k] = v[k] > 0.0f ? v[k] : 0.0f;
        return;
    case PS_INPUTMAPPING_UNSIGNED_INVERT:
        for (k = 0; k < 3; k++) {
            out[k] = 1.0f - (v[k] < 0.0f ? 0.0f : (v[k] > 1.0f ? 1.0f : v[k]));
        }
        return;
    case PS_INPUTMAPPING_EXPAND_NORMAL:
        for (k = 0; k < 3; k++) out[k] = 2.0f * (v[k] > 0.0f ? v[k] : 0.0f) - 1.0f;
        return;
    case PS_INPUTMAPPING_EXPAND_NEGATE:
        for (k = 0; k < 3; k++) out[k] = -2.0f * (v[k] > 0.0f ? v[k] : 0.0f) + 1.0f;
        return;
    case PS_INPUTMAPPING_HALFBIAS_NORMAL:
        for (k = 0; k < 3; k++) out[k] = (v[k] > 0.0f ? v[k] : 0.0f) - 0.5f;
        return;
    case PS_INPUTMAPPING_HALFBIAS_NEGATE:
        for (k = 0; k < 3; k++) out[k] = -(v[k] > 0.0f ? v[k] : 0.0f) + 0.5f;
        return;
    case PS_INPUTMAPPING_SIGNED_NEGATE:
        for (k = 0; k < 3; k++) out[k] = -v[k];
        return;
    default:                            /* SIGNED_IDENTITY and anything else */
        for (k = 0; k < 3; k++) out[k] = v[k];
        return;
    }
}

/* An input variable as xemu's get_input_var builds it: 3 lanes for RGB,
 * 1 lane for alpha. */
KELVIN_INLINE void read_input(const KPixelState *ps, const Regs *r,
                       const KCombinerInput *in, int stage, bool is_alpha,
                       float out[3])
{
    float scratch[4];
    const float *v = reg_ptr(ps, r, in->reg, stage, scratch);
    if (!is_alpha) {
        if (in->chan == PS_CHANNEL_ALPHA) {
            out[0] = out[1] = out[2] = map_input(v[3], in->mod);
        } else {
            map_input3(v, in->mod, out);
        }
    } else {
        out[0] = map_input(in->chan == PS_CHANNEL_ALPHA ? v[3] : v[2], in->mod);
    }
}

KELVIN_INLINE float map_output(float x, int mapping)
{
    switch (mapping) {
    case PS_COMBINEROUTPUT_BIAS: return x - 0.5f;
    case PS_COMBINEROUTPUT_SHIFTLEFT_1: return x * 2.0f;
    case PS_COMBINEROUTPUT_SHIFTLEFT_1_BIAS: return (x - 0.5f) * 2.0f;
    case PS_COMBINEROUTPUT_SHIFTLEFT_2: return x * 4.0f;
    case PS_COMBINEROUTPUT_SHIFTRIGHT_1: return x / 2.0f;
    default: return x;
    }
}

KELVIN_INLINE float clamp1(float x)
{
    return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

KELVIN_INLINE float *dest_ptr(Regs *r, int reg)
{
    switch (reg) {
    case PS_REGISTER_V0: return r->v0;
    case PS_REGISTER_V1: return r->v1;
    case PS_REGISTER_T0: return r->t[0];
    case PS_REGISTER_T1: return r->t[1];
    case PS_REGISTER_T2: return r->t[2];
    case PS_REGISTER_T3: return r->t[3];
    case PS_REGISTER_R0: return r->r0;
    case PS_REGISTER_R1: return r->r1;
    default: return NULL;           /* discard (and anything unwritable) */
    }
}

typedef struct PortionResult {
    float ab[4], cd[4], mux[4];      /* RGB in 0-2, alpha in 0 */
    bool w_ab, w_cd, w_mux;
} PortionResult;

#ifdef _MSC_VER
#define KPIX_NOINLINE __declspec(noinline)
#else
#define KPIX_NOINLINE __attribute__((noinline))
#endif

/* Whether a mux picks CD over AB: r0.a's top bit, or the low bit of r0.a
 * as an 8-bit value -- a conversion whose result out of range is the
 * compiler's, so every path (one fragment or eight) takes it from this one
 * function. */
static KPIX_NOINLINE bool mux_select(const KPixelState *ps, float r0a)
{
    if (ps->flags & PS_COMBINERCOUNT_MUX_MSB) {
        return r0a >= 0.5f;
    }
    return (((unsigned int)(r0a * 255.0f)) & 1u) == 1u;
}

/* ---- the RGB portion on SSE lanes 0-2 ------------------------------------
 *
 * Each lane does what the scalar code does to that component, and each
 * operation below gives exactly the scalar expression's result, NaNs and
 * signed zeros included: MAXPS(a, b) is a > b ? a : b and MINPS(a, b) is
 * a < b ? a : b, so max(x, 0) is x > 0 ? x : 0, max(0, min(1, x)) is the
 * [0, 1] clamp and max(-1, min(1, x)) is clamp1; negation flips the sign
 * bit as unary minus does. Lane 3 carries whatever and is never stored. */

KELVIN_INLINE __m128 neg4(__m128 x)
{
    return _mm_xor_ps(x, _mm_set1_ps(-0.0f));
}

KELVIN_INLINE __m128 map_input4(__m128 x, int mod)
{
    __m128 zero = _mm_setzero_ps(), one = _mm_set1_ps(1.0f);
    switch (mod) {
    case PS_INPUTMAPPING_UNSIGNED_IDENTITY:
        return _mm_max_ps(x, zero);
    case PS_INPUTMAPPING_UNSIGNED_INVERT:
        return _mm_sub_ps(one, _mm_max_ps(zero, _mm_min_ps(one, x)));
    case PS_INPUTMAPPING_EXPAND_NORMAL:
        return _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(2.0f), _mm_max_ps(x, zero)), one);
    case PS_INPUTMAPPING_EXPAND_NEGATE:
        return _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-2.0f), _mm_max_ps(x, zero)), one);
    case PS_INPUTMAPPING_HALFBIAS_NORMAL:
        return _mm_sub_ps(_mm_max_ps(x, zero), _mm_set1_ps(0.5f));
    case PS_INPUTMAPPING_HALFBIAS_NEGATE:
        return _mm_add_ps(neg4(_mm_max_ps(x, zero)), _mm_set1_ps(0.5f));
    case PS_INPUTMAPPING_SIGNED_NEGATE:
        return neg4(x);
    default:                            /* SIGNED_IDENTITY and anything else */
        return x;
    }
}

KELVIN_INLINE __m128 map_output4(__m128 x, int mapping)
{
    switch (mapping) {
    case PS_COMBINEROUTPUT_BIAS: return _mm_sub_ps(x, _mm_set1_ps(0.5f));
    case PS_COMBINEROUTPUT_SHIFTLEFT_1: return _mm_mul_ps(x, _mm_set1_ps(2.0f));
    case PS_COMBINEROUTPUT_SHIFTLEFT_1_BIAS:
        return _mm_mul_ps(_mm_sub_ps(x, _mm_set1_ps(0.5f)), _mm_set1_ps(2.0f));
    case PS_COMBINEROUTPUT_SHIFTLEFT_2: return _mm_mul_ps(x, _mm_set1_ps(4.0f));
    case PS_COMBINEROUTPUT_SHIFTRIGHT_1: return _mm_div_ps(x, _mm_set1_ps(2.0f));
    default: return x;
    }
}

KELVIN_INLINE __m128 clamp1_4(__m128 x)
{
    return _mm_max_ps(_mm_set1_ps(-1.0f), _mm_min_ps(_mm_set1_ps(1.0f), x));
}

/* read_input for RGB. */
KELVIN_INLINE __m128 read_rgb(const KPixelState *ps, const Regs *r,
                       const KCombinerInput *in, int stage)
{
    float scratch[4];
    __m128 v = _mm_loadu_ps(reg_ptr(ps, r, in->reg, stage, scratch));
    if (in->chan == PS_CHANNEL_ALPHA) {
        v = _mm_shuffle_ps(v, v, 0xFF);
    }
    return map_input4(v, in->mod);
}

/* a[0] * b[0] + a[1] * b[1] + a[2] * b[2], in that order, in every lane. */
KELVIN_INLINE __m128 dot3_4(__m128 a, __m128 b)
{
    float p[4], dp;
    _mm_storeu_ps(p, _mm_mul_ps(a, b));
    dp = p[0] + p[1] + p[2];
    return _mm_set1_ps(dp);
}

static void eval_rgb(const KPixelState *ps, const Regs *r, int stage,
                     const KCombinerInput in[4], const KCombinerOutput *o,
                     PortionResult *res)
{
    __m128 ab = _mm_setzero_ps(), cd = _mm_setzero_ps();

    if (res->w_ab || res->w_mux) {
        __m128 a = read_rgb(ps, r, &in[0], stage), b = read_rgb(ps, r, &in[1], stage);
        ab = o->ab_op == PS_COMBINEROUTPUT_AB_DOT_PRODUCT ? dot3_4(a, b)
                                                           : _mm_mul_ps(a, b);
    }
    if (res->w_cd || res->w_mux) {
        __m128 c = read_rgb(ps, r, &in[2], stage), d = read_rgb(ps, r, &in[3], stage);
        cd = o->cd_op == PS_COMBINEROUTPUT_CD_DOT_PRODUCT ? dot3_4(c, d)
                                                           : _mm_mul_ps(c, d);
    }
    if (res->w_ab) {
        _mm_storeu_ps(res->ab, clamp1_4(map_output4(ab, o->mapping)));
    }
    if (res->w_cd) {
        _mm_storeu_ps(res->cd, clamp1_4(map_output4(cd, o->mapping)));
    }
    if (res->w_mux) {
        __m128 m;
        if (o->muxsum_op == PS_COMBINEROUTPUT_AB_CD_SUM) {
            m = _mm_add_ps(ab, cd);
        } else {
            m = mux_select(ps, r->r0[3]) ? cd : ab;
        }
        _mm_storeu_ps(res->mux, clamp1_4(map_output4(m, o->mapping)));
    }
}

/* Only what a written output uses is computed: AB for the AB output, CD for
 * the CD output, both for the sum or mux. An output going nowhere is never
 * read (write_portion), and reading inputs has no side effects. */
static void eval_portion(const KPixelState *ps, const Regs *r, int stage,
                         const KCombinerInput in[4], const KCombinerOutput *o,
                         bool is_alpha, PortionResult *res)
{
    float a[3], b[3], c[3], d[3], ab[3], cd[3];
    int n = is_alpha ? 1 : 3, k;

    res->w_ab = dest_ptr((Regs *)r, o->ab) != NULL;
    res->w_cd = dest_ptr((Regs *)r, o->cd) != NULL;
    res->w_mux = dest_ptr((Regs *)r, o->muxsum) != NULL;
    if (!is_alpha) {
        eval_rgb(ps, r, stage, in, o, res);
        return;
    }

    if (res->w_ab || res->w_mux) {
        read_input(ps, r, &in[0], stage, is_alpha, a);
        read_input(ps, r, &in[1], stage, is_alpha, b);
        if (!is_alpha && o->ab_op == PS_COMBINEROUTPUT_AB_DOT_PRODUCT) {
            float dp = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            ab[0] = ab[1] = ab[2] = dp;
        } else {
            for (k = 0; k < n; k++) ab[k] = a[k] * b[k];
        }
    }
    if (res->w_cd || res->w_mux) {
        read_input(ps, r, &in[2], stage, is_alpha, c);
        read_input(ps, r, &in[3], stage, is_alpha, d);
        if (!is_alpha && o->cd_op == PS_COMBINEROUTPUT_CD_DOT_PRODUCT) {
            float dp = c[0] * d[0] + c[1] * d[1] + c[2] * d[2];
            cd[0] = cd[1] = cd[2] = dp;
        } else {
            for (k = 0; k < n; k++) cd[k] = c[k] * d[k];
        }
    }

    if (res->w_ab) {
        for (k = 0; k < n; k++) res->ab[k] = clamp1(map_output(ab[k], o->mapping));
    }
    if (res->w_cd) {
        for (k = 0; k < n; k++) res->cd[k] = clamp1(map_output(cd[k], o->mapping));
    }
    if (res->w_mux) {
        if (o->muxsum_op == PS_COMBINEROUTPUT_AB_CD_SUM) {
            for (k = 0; k < n; k++) {
                res->mux[k] = clamp1(map_output(ab[k] + cd[k], o->mapping));
            }
        } else {
            /* The same select for every component: r0.a's. */
            bool sel = mux_select(ps, r->r0[3]);
            for (k = 0; k < n; k++) {
                res->mux[k] = clamp1(map_output(sel ? cd[k] : ab[k], o->mapping));
            }
        }
    }
}

static void write_portion(Regs *r, const KCombinerOutput *o, bool is_alpha,
                          const PortionResult *res)
{
    float *dst;
    if (res->w_ab) {
        dst = dest_ptr(r, o->ab);
        if (is_alpha) {
            dst[3] = res->ab[0];
        } else {
            dst[0] = res->ab[0]; dst[1] = res->ab[1]; dst[2] = res->ab[2];
            if (o->flags & PS_COMBINEROUTPUT_AB_BLUE_TO_ALPHA) {
                dst[3] = res->ab[2];
            }
        }
    }
    if (res->w_cd) {
        dst = dest_ptr(r, o->cd);
        if (is_alpha) {
            dst[3] = res->cd[0];
        } else {
            dst[0] = res->cd[0]; dst[1] = res->cd[1]; dst[2] = res->cd[2];
            if (o->flags & PS_COMBINEROUTPUT_CD_BLUE_TO_ALPHA) {
                dst[3] = res->cd[2];
            }
        }
    }
    if (res->w_mux) {
        dst = dest_ptr(r, o->muxsum);
        if (is_alpha) {
            dst[3] = res->mux[0];
        } else {
            dst[0] = res->mux[0]; dst[1] = res->mux[1]; dst[2] = res->mux[2];
        }
    }
}

/* ---- texture shader --------------------------------------------------------- */

static float sign1(float x) { x *= 255.0f; return (x - 128.0f) / 127.0f; }
static float sign2(float x)
{
    x *= 255.0f;
    return x >= 128.0f ? (x - 255.5f) / 127.5f : (x + 0.5f) / 127.5f;
}
static float sign3(float x)
{
    x *= 255.0f;
    return x >= 128.0f ? (x - 256.0f) / 127.0f : x / 127.0f;
}

static void dotmap(int mode, const float col[4], float out[3])
{
    int k;
    switch (mode) {
    case PS_DOTMAPPING_MINUS1_TO_1_D3D:
        for (k = 0; k < 3; k++) out[k] = sign1(col[k]);
        return;
    case PS_DOTMAPPING_MINUS1_TO_1_GL:
        for (k = 0; k < 3; k++) out[k] = sign2(col[k]);
        return;
    case PS_DOTMAPPING_MINUS1_TO_1:
        for (k = 0; k < 3; k++) out[k] = sign3(col[k]);
        return;
    case PS_DOTMAPPING_HILO_1: {
        unsigned int hi = ((unsigned int)(col[3] * 255.0f) << 8) |
                          (unsigned int)(col[0] * 255.0f);
        unsigned int lo = ((unsigned int)(col[1] * 255.0f) << 8) |
                          (unsigned int)(col[2] * 255.0f);
        out[0] = (float)hi / 65535.0f;
        out[1] = (float)lo / 65535.0f;
        out[2] = 1.0f;
        return;
    }
    case PS_DOTMAPPING_ZERO_TO_ONE:
        out[0] = col[0]; out[1] = col[1]; out[2] = col[2];
        return;
    default:
        kelvin_unsupported(KU_DOTMAP_HILO, "dot mapping", mode);
        out[0] = col[0]; out[1] = col[1]; out[2] = col[2];
        return;
    }
}

/* Normalize texel-space coordinates for a linear ("rect") texture. */
KELVIN_INLINE void rect_norm(const KPixelState *ps, int i, float *s, float *t)
{
    if (ps->rect_tex[i]) {
        const KTexture *tx = &ps->tex[i];
        *s /= tx->width ? (float)tx->width : 1.0f;
        *t /= tx->height ? (float)tx->height : 1.0f;
    }
}

/* LOD from the derivatives of a projected coordinate s = x / w. */
KELVIN_INLINE float proj_lod(const KPixelState *ps, int i, const KFragment *f)
{
    const float *p = f->t[i], *dx = f->dtdx[i], *dy = f->dtdy[i];
    float w = p[3], iw, s, tt, dsdx, dtdx, dsdy, dtdy;
    if (w == 0.0f) {
        return 0.0f;
    }
    iw = 1.0f / w;
    s = p[0] * iw;
    tt = p[1] * iw;
    dsdx = (dx[0] - s * dx[3]) * iw;
    dtdx = (dx[1] - tt * dx[3]) * iw;
    dsdy = (dy[0] - s * dy[3]) * iw;
    dtdy = (dy[1] - tt * dy[3]) * iw;
    if (ps->rect_tex[i]) {
        const KTexture *tx = &ps->tex[i];
        float iwd = tx->width ? 1.0f / tx->width : 1.0f;
        float ihd = tx->height ? 1.0f / tx->height : 1.0f;
        dsdx *= iwd; dsdy *= iwd; dtdx *= ihd; dtdy *= ihd;
    }
    return ktex_lod(&ps->tex[i], dsdx, dtdx, dsdy, dtdy);
}

KELVIN_INLINE float plain_lod(const KPixelState *ps, int i, const KFragment *f)
{
    return ktex_lod(&ps->tex[i], f->dtdx[i][0], f->dtdx[i][1],
                    f->dtdy[i][0], f->dtdy[i][1]);
}

KELVIN_INLINE void sample2d(const KPixelState *ps, int i, float s, float t, float lod,
                     float out[4])
{
    rect_norm(ps, i, &s, &t);
    ktex_sample_2d(&ps->tex[i], 0, s, t, lod, out);
}

/* Returns false if the fragment is discarded. */
static bool texture_stage(const KPixelState *ps, const KFragment *f, Regs *r,
                          int i, float dots[4])
{
    int mode = ps->tex_mode[i];
    const float *p = f->t[i];
    float *out = r->t[i];
    bool fetched = false;
    const KTexture *tx = &ps->tex[i];

    switch (mode) {
    case PS_TEXTUREMODES_NONE:
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return true;
    case PS_TEXTUREMODES_PROJECT2D:
    case PS_TEXTUREMODES_PROJECT3D:
        if (tx->cubemap) {
            kelvin_unsupported(KU_TEX_CUBE, "projected cube texture", i);
        }
        if (p[3] == 0.0f) {
            out[0] = out[1] = out[2] = 0.0f;
            out[3] = 1.0f;
        } else {
            sample2d(ps, i, p[0] / p[3], p[1] / p[3],
                     ps->lod_free[i] ? 0.0f : proj_lod(ps, i, f), out);
        }
        fetched = true;
        break;
    case PS_TEXTUREMODES_CUBEMAP:
        if (tx->cubemap) {
            ktex_sample_cube(tx, p[0], p[1], p[2],
                             ps->lod_free[i] ? 0.0f : plain_lod(ps, i, f), out);
        } else {
            kelvin_unsupported(KU_TEX_CUBE, "cube mode on 2D texture", i);
            out[0] = out[1] = out[2] = 0.0f;
            out[3] = 1.0f;
        }
        fetched = true;
        break;
    case PS_TEXTUREMODES_PASSTHRU:
        memcpy(out, p, 16);
        return true;
    case PS_TEXTUREMODES_CLIPPLANE: {
        int j;
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        for (j = 0; j < 4; j++) {
            bool discard = ps->compare_mode[i][j] ? p[j] >= 0.0f : p[j] < 0.0f;
            if (discard) {
                return false;
            }
        }
        return true;
    }
    case PS_TEXTUREMODES_BUMPENVMAP:
    case PS_TEXTUREMODES_BUMPENVMAP_LUM: {
        const float *in = r->t[ps->input_tex[i]];
        const float *m = ps->bump_mat[i];
        float ds = sign3(in[2]), dt = sign3(in[1]);
        float bs = m[0] * ds + m[2] * dt;
        float bt = m[1] * ds + m[3] * dt;
        sample2d(ps, i, p[0] + bs, p[1] + bt,
                 ps->lod_free[i] ? 0.0f : plain_lod(ps, i, f), out);
        if (mode == PS_TEXTUREMODES_BUMPENVMAP_LUM) {
            float l = ps->bump_scale[i] * in[0] + ps->bump_offset[i];
            out[0] *= l; out[1] *= l; out[2] *= l; out[3] *= l;
        }
        fetched = true;
        break;
    }
    case PS_TEXTUREMODES_DOTPRODUCT:
    case PS_TEXTUREMODES_DOT_ZW: {
        float n[3];
        dotmap(ps->dot_map[i], r->t[ps->input_tex[i]], n);
        dots[i] = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        return true;
    }
    case PS_TEXTUREMODES_DOT_ST: {
        float n[3];
        dotmap(ps->dot_map[i], r->t[ps->input_tex[i]], n);
        dots[i] = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
        sample2d(ps, i, dots[i - 1], dots[i], 0.0f, out);
        fetched = true;
        break;
    }
    case PS_TEXTUREMODES_DOT_STR_3D:
    case PS_TEXTUREMODES_DOT_STR_CUBE:
    case PS_TEXTUREMODES_DOT_RFLCT_DIFF:
    case PS_TEXTUREMODES_DOT_RFLCT_SPEC: {
        float n[3], v[3];
        dotmap(ps->dot_map[i], r->t[ps->input_tex[i]], n);
        dots[i] = p[0] * n[0] + p[1] * n[1] + p[2] * n[2];
        if (mode == PS_TEXTUREMODES_DOT_RFLCT_DIFF) {
            float n2[3];
            const float *p1 = f->t[i + 1];
            dotmap(ps->dot_map[i + 1], r->t[ps->input_tex[i + 1]], n2);
            v[0] = dots[i - 1]; v[1] = dots[i];
            v[2] = p1[0] * n2[0] + p1[1] * n2[1] + p1[2] * n2[2];
        } else if (mode == PS_TEXTUREMODES_DOT_RFLCT_SPEC) {
            float nn[3] = { dots[i - 2], dots[i - 1], dots[i] };
            float e[3] = { f->t[i - 2][3], f->t[i - 1][3], f->t[i][3] };
            float nde = nn[0] * e[0] + nn[1] * e[1] + nn[2] * e[2];
            float ndn = nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2];
            int k;
            for (k = 0; k < 3; k++) {
                v[k] = ndn != 0.0f ? 2.0f * nn[k] * nde / ndn - e[k] : -e[k];
            }
        } else {
            v[0] = dots[i - 2]; v[1] = dots[i - 1]; v[2] = dots[i];
        }
        if (mode == PS_TEXTUREMODES_DOT_STR_3D) {
            if (tx->dimensionality == 2) {
                sample2d(ps, i, v[0], v[1], 0.0f, out);
            } else {
                kelvin_unsupported(KU_TEX_3D, "DOT_STR_3D volume", i);
                out[0] = out[1] = out[2] = 0.0f;
                out[3] = 1.0f;
            }
        } else if (tx->cubemap) {
            ktex_sample_cube(tx, v[0], v[1], v[2], 0.0f, out);
        } else {
            kelvin_unsupported(KU_TEX_CUBE, "dot cube on 2D texture", i);
            out[0] = out[1] = out[2] = 0.0f;
            out[3] = 1.0f;
        }
        fetched = true;
        break;
    }
    case PS_TEXTUREMODES_DPNDNT_AR: {
        const float *in = r->t[ps->input_tex[i]];
        sample2d(ps, i, in[3], in[0], 0.0f, out);
        fetched = true;
        break;
    }
    case PS_TEXTUREMODES_DPNDNT_GB: {
        const float *in = r->t[ps->input_tex[i]];
        sample2d(ps, i, in[1], in[2], 0.0f, out);
        fetched = true;
        break;
    }
    default:
        kelvin_unsupported(KU_TEX_MODE, "texture shader mode", mode);
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return true;
    }

    if (fetched) {
        if (tx->alphakill && out[3] == 0.0f) {
            return false;
        }
        if (tx->colorkey_mode) {
            uint32_t c =
                ((uint32_t)(out[3] * 255.0f + 0.5f) << 24) |
                ((uint32_t)(out[0] * 255.0f + 0.5f) << 16) |
                ((uint32_t)(out[1] * 255.0f + 0.5f) << 8) |
                (uint32_t)(out[2] * 255.0f + 0.5f);
            uint32_t mask = ps->color_key_mask[i];
            if ((c & mask) == (ps->color_key[i] & mask)) {
                switch (tx->colorkey_mode) {
                case COLOR_KEY_DISCARD:
                    return false;
                case COLOR_KEY_KILL_ALPHA:
                    out[3] = 0.0f;
                    break;
                case COLOR_KEY_KILL_COLOR_AND_ALPHA:
                    out[0] = out[1] = out[2] = out[3] = 0.0f;
                    break;
                }
            }
        }
    }
    return true;
}

/* ---- surfaces ------------------------------------------------------------- */

KELVIN_INLINE void read_color(const KPixelState *ps, const uint8_t *p, float c[4])
{
    uint32_t v;
    switch (ps->color_format) {
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_Z1R5G5B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_O1R5G5B5:
        v = p[0] | (p[1] << 8);
        c[0] = ((v >> 10) & 0x1F) / 31.0f;
        c[1] = ((v >> 5) & 0x1F) / 31.0f;
        c[2] = (v & 0x1F) / 31.0f;
        c[3] = 1.0f;
        return;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5:
        v = p[0] | (p[1] << 8);
        c[0] = ((v >> 11) & 0x1F) / 31.0f;
        c[1] = ((v >> 5) & 0x3F) / 63.0f;
        c[2] = (v & 0x1F) / 31.0f;
        c[3] = 1.0f;
        return;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8:
        c[0] = p[2] / 255.0f; c[1] = p[1] / 255.0f; c[2] = p[0] / 255.0f;
        c[3] = p[3] / 255.0f;
        return;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_B8:
        c[0] = c[1] = c[2] = p[0] / 255.0f; c[3] = 1.0f;
        return;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_G8B8:
        c[0] = p[0] / 255.0f; c[1] = p[1] / 255.0f; c[2] = 0.0f; c[3] = 1.0f;
        return;
    default:                         /* the X8R8G8B8 variants */
        c[0] = p[2] / 255.0f; c[1] = p[1] / 255.0f; c[2] = p[0] / 255.0f;
        c[3] = 1.0f;
        return;
    }
}

KELVIN_INLINE uint32_t unorm(float v, uint32_t max)
{
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return (uint32_t)(v * (float)max + 0.5f);
}

KELVIN_INLINE void write_color(const KPixelState *ps, uint8_t *p, const float c[4])
{
    switch (ps->color_format) {
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_Z1R5G5B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_O1R5G5B5: {
        uint32_t v = p[0] | (p[1] << 8);
        if (ps->mask_r) v = (v & ~0x7C00u) | (unorm(c[0], 31) << 10);
        if (ps->mask_g) v = (v & ~0x03E0u) | (unorm(c[1], 31) << 5);
        if (ps->mask_b) v = (v & ~0x001Fu) | unorm(c[2], 31);
        v = (v & 0x7FFF) |
            (ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_O1R5G5B5
                 ? 0x8000 : 0);
        p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
        return;
    }
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5: {
        uint32_t v = p[0] | (p[1] << 8);
        if (ps->mask_r) v = (v & ~0xF800u) | (unorm(c[0], 31) << 11);
        if (ps->mask_g) v = (v & ~0x07E0u) | (unorm(c[1], 63) << 5);
        if (ps->mask_b) v = (v & ~0x001Fu) | unorm(c[2], 31);
        p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
        return;
    }
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_B8:
        if (ps->mask_b) p[0] = (uint8_t)unorm(c[2], 255);
        return;
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_G8B8:
        if (ps->mask_r) p[0] = (uint8_t)unorm(c[0], 255);
        if (ps->mask_g) p[1] = (uint8_t)unorm(c[1], 255);
        return;
    default:
        if (ps->mask_b) p[0] = (uint8_t)unorm(c[2], 255);
        if (ps->mask_g) p[1] = (uint8_t)unorm(c[1], 255);
        if (ps->mask_r) p[2] = (uint8_t)unorm(c[0], 255);
        if (ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8) {
            if (ps->mask_a) p[3] = (uint8_t)unorm(c[3], 255);
        } else if (ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8) {
            p[3] = 0xFF;
        } else {
            p[3] = 0x00;
        }
        return;
    }
}

KELVIN_INLINE float blend_factor(int f, const float src[4], const float dst[4],
                          const float cc[4], int k)
{
    switch (f) {
    case 0: return 0.0f;
    case 1: return 1.0f;
    case 2: return src[k];
    case 3: return 1.0f - src[k];
    case 4: return src[3];
    case 5: return 1.0f - src[3];
    case 6: return dst[3];
    case 7: return 1.0f - dst[3];
    case 8: return dst[k];
    case 9: return 1.0f - dst[k];
    case 10: return k == 3 ? 1.0f : MIN(src[3], 1.0f - dst[3]);
    case 12: return cc[k];
    case 13: return 1.0f - cc[k];
    case 14: return cc[3];
    case 15: return 1.0f - cc[3];
    default: return 0.0f;
    }
}

/* x with lane 3 replaced by a's lane 3. */
KELVIN_INLINE __m128 with_alpha4(__m128 x, __m128 a)
{
    __m128 hi = _mm_unpackhi_ps(x, a);             /* x2 a2 x3 a3 */
    return _mm_shuffle_ps(x, hi, _MM_SHUFFLE(3, 0, 1, 0));
}

/* read_color as a vector: the same divisions per lane. */
KELVIN_INLINE __m128 read_color4(const KPixelState *ps, const uint8_t *p)
{
    switch (ps->color_format) {
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_Z1R5G5B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_O1R5G5B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_B8:
    case NV097_SET_SURFACE_FORMAT_COLOR_LE_G8B8: {
        float c[4];
        read_color(ps, p, c);
        return _mm_loadu_ps(c);
    }
    default: {                       /* A8R8G8B8 and the X8R8G8B8 variants */
        uint32_t v;
        __m128i b;
        __m128 c;
        memcpy(&v, p, 4);
        b = _mm_cvtsi32_si128((int)v);
        b = _mm_unpacklo_epi16(_mm_unpacklo_epi8(b, _mm_setzero_si128()),
                               _mm_setzero_si128());
        c = _mm_div_ps(_mm_cvtepi32_ps(b), _mm_set1_ps(255.0f));   /* B G R A */
        c = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 1, 2));
        if (ps->color_format != NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8) {
            c = with_alpha4(c, _mm_set1_ps(1.0f));
        }
        return c;
    }
    }
}

/* blend_factor for all four channels. */
KELVIN_INLINE __m128 blend_factor4(int f, __m128 src, __m128 dst, const float cc[4])
{
    __m128 one = _mm_set1_ps(1.0f);
    switch (f) {
    case 0: return _mm_setzero_ps();
    case 1: return one;
    case 2: return src;
    case 3: return _mm_sub_ps(one, src);
    case 4: return _mm_shuffle_ps(src, src, 0xFF);
    case 5: return _mm_sub_ps(one, _mm_shuffle_ps(src, src, 0xFF));
    case 6: return _mm_shuffle_ps(dst, dst, 0xFF);
    case 7: return _mm_sub_ps(one, _mm_shuffle_ps(dst, dst, 0xFF));
    case 8: return dst;
    case 9: return _mm_sub_ps(one, dst);
    case 10:                         /* MIN(src.a, 1 - dst.a); alpha 1 */
        return with_alpha4(_mm_min_ps(_mm_shuffle_ps(src, src, 0xFF),
                                      _mm_sub_ps(one, _mm_shuffle_ps(dst, dst, 0xFF))),
                           one);
    case 12: return _mm_loadu_ps(cc);
    case 13: return _mm_sub_ps(one, _mm_loadu_ps(cc));
    case 14: return _mm_set1_ps(cc[3]);
    case 15: return _mm_sub_ps(one, _mm_set1_ps(cc[3]));
    default: return _mm_setzero_ps();
    }
}

KELVIN_INLINE bool compare(int func, uint32_t a, uint32_t b)
{
    switch (func) {
    case 0: return false;
    case 1: return a < b;
    case 2: return a == b;
    case 3: return a <= b;
    case 4: return a > b;
    case 5: return a != b;
    case 6: return a >= b;
    default: return true;
    }
}

KELVIN_INLINE uint32_t stencil_op(int op, uint32_t s, uint32_t ref)
{
    switch (op) {
    case NV_PGRAPH_CONTROL_2_STENCIL_OP_V_ZERO: return 0;
    case NV_PGRAPH_CONTROL_2_STENCIL_OP_V_REPLACE: return ref & 0xFF;
    case NV_PGRAPH_CONTROL_2_STENCIL_OP_V_INCRSAT: return s < 255 ? s + 1 : 255;
    case NV_PGRAPH_CONTROL_2_STENCIL_OP_V_DECRSAT: return s > 0 ? s - 1 : 0;
    case NV_PGRAPH_CONTROL_2_STENCIL_OP_V_INVERT: return (~s) & 0xFF;
    case NV_PGRAPH_CONTROL_2_STENCIL_OP_V_INCR: return (s + 1) & 0xFF;
    case NV_PGRAPH_CONTROL_2_STENCIL_OP_V_DECR: return (s - 1) & 0xFF;
    default: return s;               /* KEEP */
    }
}

/* What kpix_fragment tests before shading, from the position and depth
 * alone: the window clip, the depth clip range and the early depth test.
 * False if they drop the fragment; they write nothing. *zvalue becomes the
 * depth the rest of the pipeline uses. */
KELVIN_INLINE bool fragment_kept(const KPixelState *ps, const KelvinTarget *t,
                                 int x, int y, float *zvalue)
{
    int i;

    /* window clip (psh.c) */
    {
        bool contained = false;
        for (i = 0; i < 8; i++) {
            const int *cr = ps->clip_region[i];
            bool outside = x < cr[0] || y < cr[1] || x >= cr[2] || y >= cr[3];
            if (!outside) {
                contained = true;
                break;
            }
        }
        if (ps->window_clip_exclusive ? contained : !contained) {
            return false;
        }
    }

    /* depth clip range */
    if (ps->depth_clip_cull) {
        if (*zvalue < ps->zclip_min || ps->zclip_max < *zvalue) {
            return false;
        }
    } else {
        *zvalue = *zvalue < ps->zclip_min ? ps->zclip_min
                                          : (*zvalue > ps->zclip_max ? ps->zclip_max : *zvalue);
    }

    /* Texture stages and combiners cannot change depth. Reject hidden
     * fragments before shading, but leave all writes after alpha/discard.
     * Stencil failures may write, so retain their original late ordering.
     * Nonfinite values use the existing path without an earlier conversion. */
    if (t->zeta && ps->depth_test && !ps->stencil_test && isfinite(*zvalue)) {
        const uint8_t *zp = kelvin_zeta_ptr(t, x, y);
        bool z24 = ps->zeta_format == NV097_SET_SURFACE_FORMAT_ZETA_Z24S8;
        uint32_t sz = z24 ? ((uint32_t)zp[1] | ((uint32_t)zp[2] << 8) |
                              ((uint32_t)zp[3] << 16))
                         : (uint32_t)(zp[0] | (zp[1] << 8));
        float zf = floorf(*zvalue);
        if (zf < 0.0f) zf = 0.0f;
        if (zf > ps->zmax) zf = ps->zmax;
        if (!compare(ps->depth_func, (uint32_t)zf, sz)) {
            return false;
        }
    }
    return true;
}

bool kpix_keep(const KPixelState *ps, const KelvinTarget *t, int x, int y,
               float *zvalue)
{
    return fragment_kept(ps, t, x, y, zvalue);
}

bool kpix_fragment(const KPixelState *ps, const KelvinTarget *t, KFragment *f)
{
    float zvalue = f->z;

    if (!fragment_kept(ps, t, f->x, f->y, &zvalue)) {
        return false;
    }
    return kpix_shade(ps, t, f, zvalue);
}

static bool shade_tail(const KPixelState *ps, const KelvinTarget *t, int x, int y,
                       float zvalue, const float frag[4]);
static void combine(const KPixelState *ps, Regs *rp, float frag[4]);

bool kpix_shade(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                float zvalue)
{
    Regs r;
    float dots[4] = { 0, 0, 0, 0 };
    float frag[4];
    int i;

    memcpy(r.v0, f->d0, 16);
    memcpy(r.v1, f->d1, 16);
    r.fog[0] = ps->fog_color[0];
    r.fog[1] = ps->fog_color[1];
    r.fog[2] = ps->fog_color[2];
    r.fog[3] = f->fog < 0.0f ? 0.0f : (f->fog > 1.0f ? 1.0f : f->fog);
    memset(r.r0, 0, sizeof(r.r0));
    memset(r.r1, 0, sizeof(r.r1));
    memset(r.t, 0, sizeof(r.t));

    for (i = 0; i < 4; i++) {
        if (ps->tex_mode[i] == PS_TEXTUREMODES_NONE) {
            r.t[i][0] = r.t[i][1] = r.t[i][2] = 0.0f;   /* texture_stage's */
            r.t[i][3] = 1.0f;
            continue;
        }
        if (!texture_stage(ps, f, &r, i, dots)) {
            return false;
        }
    }
    r.r0[3] = ps->tex_mode[0] != PS_TEXTUREMODES_NONE ? r.t[0][3] : 1.0f;
    combine(ps, &r, frag);
    return shade_tail(ps, t, f->x, f->y, zvalue, frag);
}

/* The combiner stages and the final combiner on a fragment's registers:
 * its colour in frag. */
static void combine(const KPixelState *ps, Regs *rp, float frag[4])
{
    Regs r = *rp;
    int i;

    for (i = 0; i < ps->num_stages; i++) {
        const KCombinerStage *s = &ps->stage[i];
        PortionResult rgb, alpha;
        eval_portion(ps, &r, i, s->rgb, &s->rgb_out, false, &rgb);
        eval_portion(ps, &r, i, s->alpha, &s->alpha_out, true, &alpha);
        write_portion(&r, &s->rgb_out, false, &rgb);
        write_portion(&r, &s->alpha_out, true, &alpha);
    }

    if (ps->final_enabled) {
        /* On SSE lanes 0-2, as the stages' RGB portions (eval_rgb): EF from
         * e and f first, then d + c (1 - a) + b a; G is the alpha input. */
        float g[3];
        __m128 a, b, c, d;
        _mm_storeu_ps(r.ef, _mm_mul_ps(read_rgb(ps, &r, &ps->fin[4], 8),
                                       read_rgb(ps, &r, &ps->fin[5], 8)));
        a = read_rgb(ps, &r, &ps->fin[0], 8);
        b = read_rgb(ps, &r, &ps->fin[1], 8);
        c = read_rgb(ps, &r, &ps->fin[2], 8);
        d = read_rgb(ps, &r, &ps->fin[3], 8);
        read_input(ps, &r, &ps->fin[6], 8, true, g);
        _mm_storeu_ps(frag, _mm_add_ps(
            _mm_add_ps(d, _mm_mul_ps(c, _mm_sub_ps(_mm_set1_ps(1.0f), a))),
            _mm_mul_ps(b, a)));
        frag[3] = g[0];
    } else {
        /* xemu leaves fragColor unwritten here; r0 is what the combiners
         * produced last, which is the hardware's output register. */
        memcpy(frag, r.r0, 4 * sizeof(float));
    }
    *rp = r;
}

/* What follows the combiners for the fragment at (x, y) whose colour is
 * frag: the alpha test, stencil and depth, blending and the write. Whether
 * it wrote anything. */
static bool shade_tail(const KPixelState *ps, const KelvinTarget *t, int x, int y,
                       float zvalue, const float frag[4])
{
    if (ps->alpha_test && ps->alpha_func != 7) {
        int fa = (int)floorf(frag[3] * 255.0f + 0.5f);
        bool pass;
        switch (ps->alpha_func) {
        case 0: pass = false; break;
        case 1: pass = fa < ps->alpha_ref; break;
        case 2: pass = fa == ps->alpha_ref; break;
        case 3: pass = fa <= ps->alpha_ref; break;
        case 4: pass = fa > ps->alpha_ref; break;
        case 5: pass = fa != ps->alpha_ref; break;
        case 6: pass = fa >= ps->alpha_ref; break;
        default: pass = true; break;
        }
        if (!pass) {
            return false;
        }
    }

    /* stencil and depth */
    if (t->zeta && (ps->depth_test || ps->stencil_test)) {
        uint8_t *zp = kelvin_zeta_ptr(t, x, y);
        bool z24 = ps->zeta_format == NV097_SET_SURFACE_FORMAT_ZETA_Z24S8;
        uint32_t stored = z24 ? (zp[0] | (zp[1] << 8) | (zp[2] << 16) |
                                 ((uint32_t)zp[3] << 24))
                              : (uint32_t)(zp[0] | (zp[1] << 8));
        uint32_t sz = z24 ? stored >> 8 : stored;
        uint32_t st = z24 ? stored & 0xFF : 0;
        uint32_t nz;
        float zf = floorf(zvalue);
        bool write_back = false;

        if (zf < 0.0f) zf = 0.0f;
        if (zf > ps->zmax) zf = ps->zmax;
        nz = (uint32_t)zf;

        if (ps->stencil_test && z24) {
            uint32_t m = ps->stencil_mask_read;
            if (!compare(ps->stencil_func, ps->stencil_ref & m, st & m)) {
                uint32_t ns = stencil_op(ps->stencil_op_fail, st, ps->stencil_ref);
                if (ps->stencil_write) {
                    st = (st & ~ps->stencil_mask_write) | (ns & ps->stencil_mask_write);
                    stored = (stored & 0xFFFFFF00u) | st;
                    zp[0] = (uint8_t)stored;
                }
                return false;
            }
        }
        if (ps->depth_test && !compare(ps->depth_func, nz, sz)) {
            if (ps->stencil_test && z24) {
                uint32_t ns = stencil_op(ps->stencil_op_zfail, st, ps->stencil_ref);
                if (ps->stencil_write) {
                    st = (st & ~ps->stencil_mask_write) | (ns & ps->stencil_mask_write);
                    zp[0] = (uint8_t)st;
                }
            }
            return false;
        }
        if (ps->stencil_test && z24) {
            uint32_t ns = stencil_op(ps->stencil_op_zpass, st, ps->stencil_ref);
            if (ps->stencil_write) {
                st = (st & ~ps->stencil_mask_write) | (ns & ps->stencil_mask_write);
                write_back = true;
            }
        }
        if (ps->depth_write) {
            sz = nz;
            write_back = true;
        }
        if (write_back) {
            if (z24) {
                stored = (sz << 8) | (st & 0xFF);
                zp[0] = (uint8_t)stored; zp[1] = (uint8_t)(stored >> 8);
                zp[2] = (uint8_t)(stored >> 16); zp[3] = (uint8_t)(stored >> 24);
            } else {
                zp[0] = (uint8_t)sz; zp[1] = (uint8_t)(sz >> 8);
            }
        }
    }

    if (!t->color || !(ps->mask_r || ps->mask_g || ps->mask_b || ps->mask_a)) {
        return true;
    }
    {
        uint8_t *cp = kelvin_color_ptr(t, x, y);
        __m128 v = _mm_loadu_ps(frag), src, o;
        float out[4];
        /* NaN or below 0: +0; above 1: 1; otherwise v, -0 included. MINPS
         * (1, v) is 1 < v ? 1 : v; MINPS and MAXPS are MIN and MAX. */
        __m128 drop = _mm_or_ps(_mm_cmpunord_ps(v, v), _mm_cmplt_ps(v, _mm_setzero_ps()));
        src = _mm_andnot_ps(drop, _mm_min_ps(_mm_set1_ps(1.0f), v));
        if (ps->blend) {
            __m128 dst = read_color4(ps, cp);
            __m128 s = _mm_mul_ps(src, blend_factor4(ps->blend_sfactor, src, dst, ps->blend_color));
            __m128 d = _mm_mul_ps(dst, blend_factor4(ps->blend_dfactor, src, dst, ps->blend_color));
            switch (ps->blend_eqn) {
            case 0: o = _mm_sub_ps(s, d); break;
            case 1: case 5: o = _mm_sub_ps(d, s); break;
            case 3: o = _mm_min_ps(src, dst); break;
            case 4: o = _mm_max_ps(src, dst); break;
            default: o = _mm_add_ps(s, d); break;
            }
        } else {
            o = src;
        }
        _mm_storeu_ps(out, o);
        write_color(ps, cp, out);
    }
    return true;
}

/* ---- eight fragments at a time ---------------------------------------------
 *
 * kpix_shade8 shades up to eight fragments of one triangle, each as
 * kpix_shade would: the texture stages, alpha test, stencil, depth and
 * output run per fragment through the same code, and the combiners and the
 * final combiner run on all eight at once, one channel of eight fragments to
 * a vector. Each vector operation is the one the single-fragment code does
 * to that channel (eval_rgb's SSE lanes, or the scalar alpha code, whose
 * comparisons MAXPS and MINPS make as the RGB comments above show), in the
 * same order, so every fragment gets the same bits. The mux choice, whose
 * conversion is the compiler's, comes from mux_select for each fragment. The
 * fragments are distinct pixels, so running their stages side by side
 * changes nothing either writes. */

#if defined(__AVX2__)

/* The common 32-bit output merger, one channel of eight fragments per vector.
 * Stencil, alpha tests, unusual surface layouts and aliased colour/depth
 * retain shade_tail's ordered scalar path. The rasterizer's block contains
 * distinct pixels; disjoint, non-overlapping rows make reading its destination
 * colours and depths before writing any of them equivalent to lane order. */
static bool tail8_eligible(const KPixelState *ps, const KelvinTarget *t,
                           bool state_eligible)
{
    uint64_t color_size, zeta_size;
    uintptr_t color, zeta;

    if (!state_eligible || !t->color || t->swizzled || t->color_bpp != 4 ||
        !t->width || !t->height ||
        (uint64_t)t->color_pitch < (uint64_t)t->width * 4) {
        return false;
    }
    if (ps->blend) {
        /* Hardware constants are UNORM8; retain scalar behaviour for callers
         * supplying other constants directly, including nonfinite values. */
        int k;
        for (k = 0; k < 4; k++) {
            if (!(ps->blend_color[k] >= 0.0f && ps->blend_color[k] <= 1.0f)) return false;
        }
    }
    if (!t->zeta || !ps->depth_test) return true;
    if (!(ps->zmax >= 0.0f && ps->zmax <= 16777215.0f) ||
        t->zeta_bpp != (ps->zeta_format == NV097_SET_SURFACE_FORMAT_ZETA_Z24S8 ? 4u : 2u) ||
        (uint64_t)t->zeta_pitch < (uint64_t)t->width * t->zeta_bpp) {
        return false;
    }
    color_size = (uint64_t)(t->height - 1) * t->color_pitch + (uint64_t)t->width * 4;
    zeta_size = (uint64_t)(t->height - 1) * t->zeta_pitch + (uint64_t)t->width * t->zeta_bpp;
    color = (uintptr_t)t->color;
    zeta = (uintptr_t)t->zeta;
    /* Subtraction avoids wrapping an end pointer. */
    return color < zeta ? zeta - color >= color_size : color - zeta >= zeta_size;
}

/* blend_factor4, with one fragment per lane. Choose the factor once for
 * the block, including SRC_ALPHA_SATURATE's different alpha factor. */
static void blend_factor8(int f, const __m256 src[4], const __m256 dst[4],
                          const float cc[4], __m256 out[4])
{
    __m256 one = _mm256_set1_ps(1.0f), v;
    int k;
    switch (f) {
    case 1: v = one; break;
    case 4: v = src[3]; break;
    case 5: v = _mm256_sub_ps(one, src[3]); break;
    case 6: v = dst[3]; break;
    case 7: v = _mm256_sub_ps(one, dst[3]); break;
    case 14: v = _mm256_set1_ps(cc[3]); break;
    case 15: v = _mm256_sub_ps(one, _mm256_set1_ps(cc[3])); break;
    case 2: case 3: case 8: case 9: case 12: case 13:
        for (k = 0; k < 4; k++) {
            v = f == 2 || f == 3 ? src[k] :
                (f == 8 || f == 9 ? dst[k] : _mm256_set1_ps(cc[k]));
            out[k] = f == 3 || f == 9 || f == 13 ? _mm256_sub_ps(one, v) : v;
        }
        return;
    case 10:
        out[0] = out[1] = out[2] = _mm256_min_ps(src[3], _mm256_sub_ps(one, dst[3]));
        out[3] = one;
        return;
    default: v = _mm256_setzero_ps(); break;
    }
    for (k = 0; k < 4; k++) out[k] = v;
}

/* -1 means no operation was performed and shade_tail must be used. Colour
 * clamps, blend arithmetic and UNORM rounding have exactly its operand order.
 * Finite, clamped depths fit signed 32-bit conversion; nonfinite depths keep
 * the scalar compiler's original conversion through the fallback. */
static int shade_tail8_prepared(const KPixelState *ps, const KelvinTarget *t,
                       const KFragment *f, const float *zvalue, int n,
                       const bool keep[8], const float frag[4][8],
                       bool state_eligible)
{
    uint8_t *cp[8], *zp[8];
    uint32_t old[8] = { 0 }, stored[8], nz[8], packed[8];
    uint32_t write_mask = (ps->mask_b ? 0x000000FFu : 0) |
                          (ps->mask_g ? 0x0000FF00u : 0) |
                          (ps->mask_r ? 0x00FF0000u : 0) |
                          (ps->mask_a ? 0xFF000000u : 0);
    unsigned int active = 0;
    bool a8 = ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8;
    bool depth = t->zeta && ps->depth_test;
    bool z24 = ps->zeta_format == NV097_SET_SURFACE_FORMAT_ZETA_Z24S8;
    bool any_mask = ps->mask_r || ps->mask_g || ps->mask_b || ps->mask_a;
    __m256 src[4], dst[4], out[4], zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    __m256i color;
    uintptr_t previous_color = 0;
    int l, k, written = 0;

    if (!tail8_eligible(ps, t, state_eligible)) return -1;
    if (depth) {
        float z[8] = { 0 };
        for (l = 0; l < n; l++) {
            if (keep[l] && !isfinite(zvalue[l])) return -1;
            z[l] = keep[l] ? zvalue[l] : 0.0f;
        }
        _mm256_storeu_si256((__m256i *)nz, _mm256_cvttps_epi32(_mm256_min_ps(
            _mm256_max_ps(_mm256_floor_ps(_mm256_loadu_ps(z)), zero),
            _mm256_set1_ps(ps->zmax))));
    }
    for (l = 0; l < n; l++) {
        if (!keep[l]) continue;
        if (f[l].x < 0 || f[l].y < 0 || (unsigned)f[l].x >= t->width ||
            (unsigned)f[l].y >= t->height) return -1;
        cp[l] = kelvin_color_ptr(t, f[l].x, f[l].y);
        /* Scan blocks increase in address. Other callers may repeat pixels;
         * their depth comparisons or blends depend on previous lane writes. */
        if ((uintptr_t)cp[l] <= previous_color) return -1;
        previous_color = (uintptr_t)cp[l];
        if (depth) {
            uint32_t sz;
            zp[l] = kelvin_zeta_ptr(t, f[l].x, f[l].y);
            if (z24) {
                memcpy(&stored[l], zp[l], 4);
                sz = stored[l] >> 8;
            } else {
                sz = zp[l][0] | (zp[l][1] << 8);
            }
            if (!compare(ps->depth_func, nz[l], sz)) continue;
        }
        active |= 1u << l;
        if (any_mask && (ps->blend || write_mask != UINT32_MAX)) {
            memcpy(&old[l], cp[l], 4);
        }
    }
    if (!active) return 0;
    if (any_mask) {
        for (k = 0; k < 4; k++) {
            __m256 v = _mm256_loadu_ps(frag[k]);
            __m256 drop = _mm256_or_ps(_mm256_cmp_ps(v, v, _CMP_UNORD_Q),
                                       _mm256_cmp_ps(v, zero, _CMP_LT_OQ));
            src[k] = _mm256_andnot_ps(drop, _mm256_min_ps(one, v));
        }
        if (ps->blend) {
            __m256 sf[4], df[4];
            __m256i bytes = _mm256_loadu_si256((const __m256i *)old);
            const int shift[4] = { 16, 8, 0, 24 };
            for (k = 0; k < 4; k++) {
                dst[k] = k == 3 && !a8 ? one : _mm256_div_ps(_mm256_cvtepi32_ps(
                    _mm256_and_si256(_mm256_srlv_epi32(bytes, _mm256_set1_epi32(shift[k])),
                                       _mm256_set1_epi32(255))), _mm256_set1_ps(255.0f));
            }
            blend_factor8(ps->blend_sfactor, src, dst, ps->blend_color, sf);
            blend_factor8(ps->blend_dfactor, src, dst, ps->blend_color, df);
            for (k = 0; k < 4; k++) {
                __m256 s = _mm256_mul_ps(src[k], sf[k]), d = _mm256_mul_ps(dst[k], df[k]);
                switch (ps->blend_eqn) {
                case 0: out[k] = _mm256_sub_ps(s, d); break;
                case 1: case 5: out[k] = _mm256_sub_ps(d, s); break;
                case 3: out[k] = _mm256_min_ps(src[k], dst[k]); break;
                case 4: out[k] = _mm256_max_ps(src[k], dst[k]); break;
                default: out[k] = _mm256_add_ps(s, d); break;
                }
            }
        } else {
            for (k = 0; k < 4; k++) out[k] = src[k];
        }
        color = _mm256_setzero_si256();
        for (k = 0; k < 4; k++) {
            static const int shift[4] = { 16, 8, 0, 24 };
            __m256 clamped = _mm256_max_ps(zero, _mm256_min_ps(one, out[k]));
            __m256i b = _mm256_cvttps_epi32(_mm256_add_ps(
                _mm256_mul_ps(clamped, _mm256_set1_ps(255.0f)), _mm256_set1_ps(0.5f)));
            color = _mm256_or_si256(color, _mm256_sllv_epi32(b, _mm256_set1_epi32(shift[k])));
        }
        if (!a8) {
            write_mask |= 0xFF000000u;
            color = _mm256_and_si256(color, _mm256_set1_epi32(0x00FFFFFF));
            if (ps->color_format == NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_O8R8G8B8) {
                color = _mm256_or_si256(color, _mm256_set1_epi32((int)0xFF000000u));
            }
        }
        color = _mm256_or_si256(_mm256_and_si256(color, _mm256_set1_epi32((int)write_mask)),
            _mm256_andnot_si256(_mm256_set1_epi32((int)write_mask),
                                _mm256_loadu_si256((const __m256i *)old)));
        _mm256_storeu_si256((__m256i *)packed, color);
    }
    for (l = 0; l < n; l++) {
        if (!(active & (1u << l))) continue;
        if (depth && ps->depth_write) {
            if (z24) {
                uint32_t z = (nz[l] << 8) | (stored[l] & 0xFF);
                memcpy(zp[l], &z, 4);
            } else {
                zp[l][0] = (uint8_t)nz[l];
                zp[l][1] = (uint8_t)(nz[l] >> 8);
            }
        }
        if (any_mask) memcpy(cp[l], &packed[l], 4);
        written++;
    }
    return written;
}

/* Private compatibility entry for callers that have no immutable plan. */
static int shade_tail8(const KPixelState *ps, const KelvinTarget *t,
                       const KFragment *f, const float *zvalue, int n,
                       const bool keep[8], const float frag[4][8])
{
    return shade_tail8_prepared(ps, t, f, zvalue, n, keep, frag,
                                tail8_state_eligible(ps));
}

typedef struct Regs8 {
    __m256 fog[4], v0[4], v1[4], t[4][4], r0[4], r1[4], ef[3];
} Regs8;

typedef struct Portion8 {
    __m256 ab[3], cd[3], mux[3];     /* RGB in 0-2, alpha in 0 */
    bool w_ab, w_cd, w_mux;
} Portion8;

KELVIN_INLINE void copy4_8(__m256 out[4], const __m256 in[4])
{
    out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; out[3] = in[3];
}

KELVIN_INLINE void const4_8(__m256 out[4], const float k[4])
{
    out[0] = _mm256_set1_ps(k[0]); out[1] = _mm256_set1_ps(k[1]);
    out[2] = _mm256_set1_ps(k[2]); out[3] = _mm256_set1_ps(k[3]);
}

/* reg_ptr for eight fragments: the register's four channels. */
KELVIN_INLINE void reg8(const KPixelState *ps, const Regs8 *r, int reg, int stage,
                        __m256 out[4])
{
    int k;
    switch (reg) {
    case PS_REGISTER_C0:
        const4_8(out, (ps->flags & PS_COMBINERCOUNT_UNIQUE_C0) || stage == 8
                          ? (stage == 8 ? ps->final_c0 : ps->stage[stage].c0)
                          : ps->stage[0].c0);
        return;
    case PS_REGISTER_C1:
        const4_8(out, (ps->flags & PS_COMBINERCOUNT_UNIQUE_C1) || stage == 8
                          ? (stage == 8 ? ps->final_c1 : ps->stage[stage].c1)
                          : ps->stage[0].c1);
        return;
    case PS_REGISTER_FOG: copy4_8(out, r->fog); return;
    case PS_REGISTER_V0: copy4_8(out, r->v0); return;
    case PS_REGISTER_V1: copy4_8(out, r->v1); return;
    case PS_REGISTER_T0: copy4_8(out, r->t[0]); return;
    case PS_REGISTER_T1: copy4_8(out, r->t[1]); return;
    case PS_REGISTER_T2: copy4_8(out, r->t[2]); return;
    case PS_REGISTER_T3: copy4_8(out, r->t[3]); return;
    case PS_REGISTER_R0: copy4_8(out, r->r0); return;
    case PS_REGISTER_R1: copy4_8(out, r->r1); return;
    case PS_REGISTER_V1R0_SUM: {
        /* s < 0 ? 0 : (s > 1 ? 1 : s) is MAXPS(0, MINPS(1, s)). */
        __m256 one = _mm256_set1_ps(1.0f);
        for (k = 0; k < 3; k++) {
            __m256 a = ps->inv_v1 ? _mm256_sub_ps(one, r->v1[k]) : r->v1[k];
            __m256 b = ps->inv_r0 ? _mm256_sub_ps(one, r->r0[k]) : r->r0[k];
            __m256 s = _mm256_add_ps(a, b);
            if (ps->clamp_sum) {
                s = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(one, s));
            }
            out[k] = s;
        }
        out[3] = _mm256_setzero_ps();
        return;
    }
    case PS_REGISTER_EF_PROD:
        out[0] = r->ef[0]; out[1] = r->ef[1]; out[2] = r->ef[2];
        out[3] = _mm256_setzero_ps();
        return;
    default:
        out[0] = out[1] = out[2] = out[3] = _mm256_setzero_ps();
        return;
    }
}

KELVIN_INLINE __m256 neg8(__m256 x)
{
    return _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
}

/* map_input4, eight wide. */
KELVIN_INLINE __m256 map_input8(__m256 x, int mod)
{
    __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    switch (mod) {
    case PS_INPUTMAPPING_UNSIGNED_IDENTITY:
        return _mm256_max_ps(x, zero);
    case PS_INPUTMAPPING_UNSIGNED_INVERT:
        return _mm256_sub_ps(one, _mm256_max_ps(zero, _mm256_min_ps(one, x)));
    case PS_INPUTMAPPING_EXPAND_NORMAL:
        return _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), _mm256_max_ps(x, zero)), one);
    case PS_INPUTMAPPING_EXPAND_NEGATE:
        return _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), _mm256_max_ps(x, zero)), one);
    case PS_INPUTMAPPING_HALFBIAS_NORMAL:
        return _mm256_sub_ps(_mm256_max_ps(x, zero), _mm256_set1_ps(0.5f));
    case PS_INPUTMAPPING_HALFBIAS_NEGATE:
        return _mm256_add_ps(neg8(_mm256_max_ps(x, zero)), _mm256_set1_ps(0.5f));
    case PS_INPUTMAPPING_SIGNED_NEGATE:
        return neg8(x);
    default:                            /* SIGNED_IDENTITY and anything else */
        return x;
    }
}

/* map_output4, eight wide. */
KELVIN_INLINE __m256 map_output8(__m256 x, int mapping)
{
    switch (mapping) {
    case PS_COMBINEROUTPUT_BIAS: return _mm256_sub_ps(x, _mm256_set1_ps(0.5f));
    case PS_COMBINEROUTPUT_SHIFTLEFT_1: return _mm256_mul_ps(x, _mm256_set1_ps(2.0f));
    case PS_COMBINEROUTPUT_SHIFTLEFT_1_BIAS:
        return _mm256_mul_ps(_mm256_sub_ps(x, _mm256_set1_ps(0.5f)), _mm256_set1_ps(2.0f));
    case PS_COMBINEROUTPUT_SHIFTLEFT_2: return _mm256_mul_ps(x, _mm256_set1_ps(4.0f));
    case PS_COMBINEROUTPUT_SHIFTRIGHT_1: return _mm256_div_ps(x, _mm256_set1_ps(2.0f));
    default: return x;
    }
}

KELVIN_INLINE __m256 clamp1_8(__m256 x)
{
    return _mm256_max_ps(_mm256_set1_ps(-1.0f), _mm256_min_ps(_mm256_set1_ps(1.0f), x));
}

/* read_input for RGB (read_rgb), eight wide. */
KELVIN_INLINE void read_rgb8(const KPixelState *ps, const Regs8 *r,
                             const KCombinerInput *in, int stage, __m256 out[3])
{
    __m256 v[4];
    int k;
    reg8(ps, r, in->reg, stage, v);
    for (k = 0; k < 3; k++) {
        out[k] = map_input8(in->chan == PS_CHANNEL_ALPHA ? v[3] : v[k], in->mod);
    }
}

/* read_input for alpha, eight wide. */
KELVIN_INLINE __m256 read_alpha8(const KPixelState *ps, const Regs8 *r,
                                 const KCombinerInput *in, int stage)
{
    __m256 v[4];
    reg8(ps, r, in->reg, stage, v);
    return map_input8(in->chan == PS_CHANNEL_ALPHA ? v[3] : v[2], in->mod);
}

/* a[0] * b[0] + a[1] * b[1] + a[2] * b[2], in that order (dot3_4). */
KELVIN_INLINE __m256 dot3_8(const __m256 a[3], const __m256 b[3])
{
    return _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(a[0], b[0]),
                                       _mm256_mul_ps(a[1], b[1])),
                         _mm256_mul_ps(a[2], b[2]));
}

/* dest_ptr, eight wide. */
KELVIN_INLINE __m256 *dest8(Regs8 *r, int reg)
{
    switch (reg) {
    case PS_REGISTER_V0: return r->v0;
    case PS_REGISTER_V1: return r->v1;
    case PS_REGISTER_T0: return r->t[0];
    case PS_REGISTER_T1: return r->t[1];
    case PS_REGISTER_T2: return r->t[2];
    case PS_REGISTER_T3: return r->t[3];
    case PS_REGISTER_R0: return r->r0;
    case PS_REGISTER_R1: return r->r1;
    default: return NULL;
    }
}

/* eval_portion, eight wide; sel is the mux's choice per fragment. */
static void eval_portion8(const KPixelState *ps, const Regs8 *r, int stage,
                          const KCombinerInput in[4], const KCombinerOutput *o,
                          bool is_alpha, __m256 sel, Portion8 *res)
{
    int n = is_alpha ? 1 : 3, k;
    __m256 ab[3], cd[3];

    res->w_ab = dest8((Regs8 *)r, o->ab) != NULL;
    res->w_cd = dest8((Regs8 *)r, o->cd) != NULL;
    res->w_mux = dest8((Regs8 *)r, o->muxsum) != NULL;
    for (k = 0; k < 3; k++) {
        ab[k] = cd[k] = _mm256_setzero_ps();
    }
    if (res->w_ab || res->w_mux) {
        __m256 a[3], b[3];
        if (is_alpha) {
            a[0] = read_alpha8(ps, r, &in[0], stage);
            b[0] = read_alpha8(ps, r, &in[1], stage);
            ab[0] = _mm256_mul_ps(a[0], b[0]);
        } else {
            read_rgb8(ps, r, &in[0], stage, a);
            read_rgb8(ps, r, &in[1], stage, b);
            if (o->ab_op == PS_COMBINEROUTPUT_AB_DOT_PRODUCT) {
                ab[0] = ab[1] = ab[2] = dot3_8(a, b);
            } else {
                for (k = 0; k < 3; k++) ab[k] = _mm256_mul_ps(a[k], b[k]);
            }
        }
    }
    if (res->w_cd || res->w_mux) {
        __m256 c[3], d[3];
        if (is_alpha) {
            c[0] = read_alpha8(ps, r, &in[2], stage);
            d[0] = read_alpha8(ps, r, &in[3], stage);
            cd[0] = _mm256_mul_ps(c[0], d[0]);
        } else {
            read_rgb8(ps, r, &in[2], stage, c);
            read_rgb8(ps, r, &in[3], stage, d);
            if (o->cd_op == PS_COMBINEROUTPUT_CD_DOT_PRODUCT) {
                cd[0] = cd[1] = cd[2] = dot3_8(c, d);
            } else {
                for (k = 0; k < 3; k++) cd[k] = _mm256_mul_ps(c[k], d[k]);
            }
        }
    }
    for (k = 0; k < n; k++) {
        if (res->w_ab) {
            res->ab[k] = clamp1_8(map_output8(ab[k], o->mapping));
        }
        if (res->w_cd) {
            res->cd[k] = clamp1_8(map_output8(cd[k], o->mapping));
        }
        if (res->w_mux) {
            __m256 m = o->muxsum_op == PS_COMBINEROUTPUT_AB_CD_SUM
                           ? _mm256_add_ps(ab[k], cd[k])
                           : _mm256_blendv_ps(ab[k], cd[k], sel);
            res->mux[k] = clamp1_8(map_output8(m, o->mapping));
        }
    }
}

/* write_portion, eight wide. */
static void write_portion8(Regs8 *r, const KCombinerOutput *o, bool is_alpha,
                           const Portion8 *res)
{
    __m256 *dst;
    if (res->w_ab) {
        dst = dest8(r, o->ab);
        if (is_alpha) {
            dst[3] = res->ab[0];
        } else {
            dst[0] = res->ab[0]; dst[1] = res->ab[1]; dst[2] = res->ab[2];
            if (o->flags & PS_COMBINEROUTPUT_AB_BLUE_TO_ALPHA) {
                dst[3] = res->ab[2];
            }
        }
    }
    if (res->w_cd) {
        dst = dest8(r, o->cd);
        if (is_alpha) {
            dst[3] = res->cd[0];
        } else {
            dst[0] = res->cd[0]; dst[1] = res->cd[1]; dst[2] = res->cd[2];
            if (o->flags & PS_COMBINEROUTPUT_CD_BLUE_TO_ALPHA) {
                dst[3] = res->cd[2];
            }
        }
    }
    if (res->w_mux) {
        dst = dest8(r, o->muxsum);
        if (is_alpha) {
            dst[3] = res->mux[0];
        } else {
            dst[0] = res->mux[0]; dst[1] = res->mux[1]; dst[2] = res->mux[2];
        }
    }
}

/* Eight floats, one per fragment, as a vector. */
#define GATHER8(expr) _mm256_setr_ps(expr(0), expr(1), expr(2), expr(3), \
                                     expr(4), expr(5), expr(6), expr(7))

/* The mux's choice for each fragment, as a blend mask: from mux_select,
 * whatever r0.a holds (a stage's portions all read r0 before it is
 * written). */
static __m256 mux_mask8(const KPixelState *ps, __m256 r0a)
{
    float a[8];
    int32_t m[8];
    int l;
    _mm256_storeu_ps(a, r0a);
    for (l = 0; l < 8; l++) {
        m[l] = mux_select(ps, a[l]) ? -1 : 0;
    }
    return _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *)m));
}

static bool stage_needs_mux(const KCombinerStage *s)
{
    return (s->rgb_out.muxsum_op != PS_COMBINEROUTPUT_AB_CD_SUM &&
            s->rgb_out.muxsum != 0) ||
           (s->alpha_out.muxsum_op != PS_COMBINEROUTPUT_AB_CD_SUM &&
            s->alpha_out.muxsum != 0);
}

/* combine, eight wide: each fragment's colour in out[channel][fragment]. */
static void combine8(const KPixelState *ps, Regs8 *r, float out[4][8])
{
    int i, k;

    for (i = 0; i < ps->num_stages; i++) {
        const KCombinerStage *s = &ps->stage[i];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8(ps, r, i, s->rgb, &s->rgb_out, false, sel, &rgb);
        eval_portion8(ps, r, i, s->alpha, &s->alpha_out, true, sel, &alpha);
        write_portion8(r, &s->rgb_out, false, &rgb);
        write_portion8(r, &s->alpha_out, true, &alpha);
    }

    if (ps->final_enabled) {
        /* As combine: EF first, then d + c (1 - a) + b a; G is alpha. */
        __m256 e[3], ff[3], a[3], b[3], c[3], d[3], one = _mm256_set1_ps(1.0f);
        read_rgb8(ps, r, &ps->fin[4], 8, e);
        read_rgb8(ps, r, &ps->fin[5], 8, ff);
        for (k = 0; k < 3; k++) {
            r->ef[k] = _mm256_mul_ps(e[k], ff[k]);
        }
        read_rgb8(ps, r, &ps->fin[0], 8, a);
        read_rgb8(ps, r, &ps->fin[1], 8, b);
        read_rgb8(ps, r, &ps->fin[2], 8, c);
        read_rgb8(ps, r, &ps->fin[3], 8, d);
        for (k = 0; k < 3; k++) {
            _mm256_storeu_ps(out[k], _mm256_add_ps(
                _mm256_add_ps(d[k], _mm256_mul_ps(c[k], _mm256_sub_ps(one, a[k]))),
                _mm256_mul_ps(b[k], a[k])));
        }
        _mm256_storeu_ps(out[3], read_alpha8(ps, r, &ps->fin[6], 8));
    } else {
        for (k = 0; k < 4; k++) {
            _mm256_storeu_ps(out[k], r->r0[k]);
        }
    }
}

KELVIN_INLINE void reg8_fixed(const KPixelState *ps, const Regs8 *r, int reg, int stage,
                        const CombinerShape *shape, __m256 out[4])
{
    int k;
    switch (reg) {
    case PS_REGISTER_C0:
        const4_8(out, (shape->flags & PS_COMBINERCOUNT_UNIQUE_C0) || stage == 8
                          ? (stage == 8 ? ps->final_c0 : ps->stage[stage].c0)
                          : ps->stage[0].c0);
        return;
    case PS_REGISTER_C1:
        const4_8(out, (shape->flags & PS_COMBINERCOUNT_UNIQUE_C1) || stage == 8
                          ? (stage == 8 ? ps->final_c1 : ps->stage[stage].c1)
                          : ps->stage[0].c1);
        return;
    case PS_REGISTER_FOG: copy4_8(out, r->fog); return;
    case PS_REGISTER_V0: copy4_8(out, r->v0); return;
    case PS_REGISTER_V1: copy4_8(out, r->v1); return;
    case PS_REGISTER_T0: copy4_8(out, r->t[0]); return;
    case PS_REGISTER_T1: copy4_8(out, r->t[1]); return;
    case PS_REGISTER_T2: copy4_8(out, r->t[2]); return;
    case PS_REGISTER_T3: copy4_8(out, r->t[3]); return;
    case PS_REGISTER_R0: copy4_8(out, r->r0); return;
    case PS_REGISTER_R1: copy4_8(out, r->r1); return;
    case PS_REGISTER_V1R0_SUM: {
        /* s < 0 ? 0 : (s > 1 ? 1 : s) is MAXPS(0, MINPS(1, s)). */
        __m256 one = _mm256_set1_ps(1.0f);
        for (k = 0; k < 3; k++) {
            __m256 a = shape->inv_v1 ? _mm256_sub_ps(one, r->v1[k]) : r->v1[k];
            __m256 b = shape->inv_r0 ? _mm256_sub_ps(one, r->r0[k]) : r->r0[k];
            __m256 s = _mm256_add_ps(a, b);
            if (shape->clamp_sum) {
                s = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(one, s));
            }
            out[k] = s;
        }
        out[3] = _mm256_setzero_ps();
        return;
    }
    case PS_REGISTER_EF_PROD:
        out[0] = r->ef[0]; out[1] = r->ef[1]; out[2] = r->ef[2];
        out[3] = _mm256_setzero_ps();
        return;
    default:
        out[0] = out[1] = out[2] = out[3] = _mm256_setzero_ps();
        return;
    }
}

KELVIN_INLINE void read_rgb8_fixed(const KPixelState *ps, const Regs8 *r,
                             const KCombinerInput *in, int stage, const CombinerShape *shape, __m256 out[3])
{
    __m256 v[4];
    int k;
    reg8_fixed(ps, r, in->reg, stage, shape, v);
    for (k = 0; k < 3; k++) {
        out[k] = map_input8(in->chan == PS_CHANNEL_ALPHA ? v[3] : v[k], in->mod);
    }
}

KELVIN_INLINE __m256 read_alpha8_fixed(const KPixelState *ps, const Regs8 *r,
                                 const KCombinerInput *in, int stage,
                                 const CombinerShape *shape)
{
    __m256 v[4];
    reg8_fixed(ps, r, in->reg, stage, shape, v);
    return map_input8(in->chan == PS_CHANNEL_ALPHA ? v[3] : v[2], in->mod);
}

KELVIN_INLINE void eval_portion8_fixed(const KPixelState *ps, const Regs8 *r, int stage,
                          const KCombinerInput in[4], const KCombinerOutput *o,
                          bool is_alpha, __m256 sel, Portion8 *res,
                          const CombinerShape *shape)
{
    int n = is_alpha ? 1 : 3, k;
    __m256 ab[3], cd[3];

    res->w_ab = dest8((Regs8 *)r, o->ab) != NULL;
    res->w_cd = dest8((Regs8 *)r, o->cd) != NULL;
    res->w_mux = dest8((Regs8 *)r, o->muxsum) != NULL;
    for (k = 0; k < 3; k++) {
        ab[k] = cd[k] = _mm256_setzero_ps();
    }
    if (res->w_ab || res->w_mux) {
        __m256 a[3], b[3];
        if (is_alpha) {
            a[0] = read_alpha8_fixed(ps, r, &in[0], stage, shape);
            b[0] = read_alpha8_fixed(ps, r, &in[1], stage, shape);
            ab[0] = _mm256_mul_ps(a[0], b[0]);
        } else {
            read_rgb8_fixed(ps, r, &in[0], stage, shape, a);
            read_rgb8_fixed(ps, r, &in[1], stage, shape, b);
            if (o->ab_op == PS_COMBINEROUTPUT_AB_DOT_PRODUCT) {
                ab[0] = ab[1] = ab[2] = dot3_8(a, b);
            } else {
                for (k = 0; k < 3; k++) ab[k] = _mm256_mul_ps(a[k], b[k]);
            }
        }
    }
    if (res->w_cd || res->w_mux) {
        __m256 c[3], d[3];
        if (is_alpha) {
            c[0] = read_alpha8_fixed(ps, r, &in[2], stage, shape);
            d[0] = read_alpha8_fixed(ps, r, &in[3], stage, shape);
            cd[0] = _mm256_mul_ps(c[0], d[0]);
        } else {
            read_rgb8_fixed(ps, r, &in[2], stage, shape, c);
            read_rgb8_fixed(ps, r, &in[3], stage, shape, d);
            if (o->cd_op == PS_COMBINEROUTPUT_CD_DOT_PRODUCT) {
                cd[0] = cd[1] = cd[2] = dot3_8(c, d);
            } else {
                for (k = 0; k < 3; k++) cd[k] = _mm256_mul_ps(c[k], d[k]);
            }
        }
    }
    for (k = 0; k < n; k++) {
        if (res->w_ab) {
            res->ab[k] = clamp1_8(map_output8(ab[k], o->mapping));
        }
        if (res->w_cd) {
            res->cd[k] = clamp1_8(map_output8(cd[k], o->mapping));
        }
        if (res->w_mux) {
            __m256 m = o->muxsum_op == PS_COMBINEROUTPUT_AB_CD_SUM
                           ? _mm256_add_ps(ab[k], cd[k])
                           : _mm256_blendv_ps(ab[k], cd[k], sel);
            res->mux[k] = clamp1_8(map_output8(m, o->mapping));
        }
    }
}

KELVIN_INLINE void write_portion8_fixed(Regs8 *r, const KCombinerOutput *o, bool is_alpha,
                           const Portion8 *res)
{
    __m256 *dst;
    if (res->w_ab) {
        dst = dest8(r, o->ab);
        if (is_alpha) {
            dst[3] = res->ab[0];
        } else {
            dst[0] = res->ab[0]; dst[1] = res->ab[1]; dst[2] = res->ab[2];
            if (o->flags & PS_COMBINEROUTPUT_AB_BLUE_TO_ALPHA) {
                dst[3] = res->ab[2];
            }
        }
    }
    if (res->w_cd) {
        dst = dest8(r, o->cd);
        if (is_alpha) {
            dst[3] = res->cd[0];
        } else {
            dst[0] = res->cd[0]; dst[1] = res->cd[1]; dst[2] = res->cd[2];
            if (o->flags & PS_COMBINEROUTPUT_CD_BLUE_TO_ALPHA) {
                dst[3] = res->cd[2];
            }
        }
    }
    if (res->w_mux) {
        dst = dest8(r, o->muxsum);
        if (is_alpha) {
            dst[3] = res->mux[0];
        } else {
            dst[0] = res->mux[0]; dst[1] = res->mux[1]; dst[2] = res->mux[2];
        }
    }
}

KELVIN_INLINE bool stage_needs_mux_fixed(const KCombinerStage *s)
{
    return (s->rgb_out.muxsum_op != PS_COMBINEROUTPUT_AB_CD_SUM &&
            s->rgb_out.muxsum != 0) ||
           (s->alpha_out.muxsum_op != PS_COMBINEROUTPUT_AB_CD_SUM &&
            s->alpha_out.muxsum != 0);
}

static void combine8_world_fog(const KPixelState *ps, Regs8 *r, float out[4][8])
{
    const CombinerShape *shape = &world_shape;
    int k;

    {
        const KCombinerStage *s = &shape->stage[0];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 0, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 0, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    {
        const KCombinerStage *s = &shape->stage[1];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 1, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 1, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    if (shape->final_enabled) {
        /* As combine: EF first, then d + c (1 - a) + b a; G is alpha. */
        __m256 e[3], ff[3], a[3], b[3], c[3], d[3], one = _mm256_set1_ps(1.0f);
        read_rgb8_fixed(ps, r, &shape->fin[4], 8, shape, e);
        read_rgb8_fixed(ps, r, &shape->fin[5], 8, shape, ff);
        for (k = 0; k < 3; k++) {
            r->ef[k] = _mm256_mul_ps(e[k], ff[k]);
        }
        read_rgb8_fixed(ps, r, &shape->fin[0], 8, shape, a);
        read_rgb8_fixed(ps, r, &shape->fin[1], 8, shape, b);
        read_rgb8_fixed(ps, r, &shape->fin[2], 8, shape, c);
        read_rgb8_fixed(ps, r, &shape->fin[3], 8, shape, d);
        for (k = 0; k < 3; k++) {
            _mm256_storeu_ps(out[k], _mm256_add_ps(
                _mm256_add_ps(d[k], _mm256_mul_ps(c[k], _mm256_sub_ps(one, a[k]))),
                _mm256_mul_ps(b[k], a[k])));
        }
        _mm256_storeu_ps(out[3], read_alpha8_fixed(ps, r, &shape->fin[6], 8, shape));
    } else {
        for (k = 0; k < 4; k++) {
            _mm256_storeu_ps(out[k], r->r0[k]);
        }
    }
}

static void combine8_background(const KPixelState *ps, Regs8 *r, float out[4][8])
{
    const CombinerShape *shape = &background_shape;
    int k;

    {
        const KCombinerStage *s = &shape->stage[0];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 0, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 0, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    {
        const KCombinerStage *s = &shape->stage[1];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 1, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 1, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    {
        const KCombinerStage *s = &shape->stage[2];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 2, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 2, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    {
        const KCombinerStage *s = &shape->stage[3];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 3, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 3, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    {
        const KCombinerStage *s = &shape->stage[4];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 4, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 4, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    {
        const KCombinerStage *s = &shape->stage[5];
        Portion8 rgb, alpha;
        __m256 sel = stage_needs_mux_fixed(s) ? mux_mask8(ps, r->r0[3]) : _mm256_setzero_ps();
        eval_portion8_fixed(ps, r, 5, s->rgb, &s->rgb_out, false, sel, &rgb, shape);
        eval_portion8_fixed(ps, r, 5, s->alpha, &s->alpha_out, true, sel, &alpha, shape);
        write_portion8_fixed(r, &s->rgb_out, false, &rgb);
        write_portion8_fixed(r, &s->alpha_out, true, &alpha);
    }

    if (shape->final_enabled) {
        /* As combine: EF first, then d + c (1 - a) + b a; G is alpha. */
        __m256 e[3], ff[3], a[3], b[3], c[3], d[3], one = _mm256_set1_ps(1.0f);
        read_rgb8_fixed(ps, r, &shape->fin[4], 8, shape, e);
        read_rgb8_fixed(ps, r, &shape->fin[5], 8, shape, ff);
        for (k = 0; k < 3; k++) {
            r->ef[k] = _mm256_mul_ps(e[k], ff[k]);
        }
        read_rgb8_fixed(ps, r, &shape->fin[0], 8, shape, a);
        read_rgb8_fixed(ps, r, &shape->fin[1], 8, shape, b);
        read_rgb8_fixed(ps, r, &shape->fin[2], 8, shape, c);
        read_rgb8_fixed(ps, r, &shape->fin[3], 8, shape, d);
        for (k = 0; k < 3; k++) {
            _mm256_storeu_ps(out[k], _mm256_add_ps(
                _mm256_add_ps(d[k], _mm256_mul_ps(c[k], _mm256_sub_ps(one, a[k]))),
                _mm256_mul_ps(b[k], a[k])));
        }
        _mm256_storeu_ps(out[3], read_alpha8_fixed(ps, r, &shape->fin[6], 8, shape));
    } else {
        for (k = 0; k < 4; k++) {
            _mm256_storeu_ps(out[k], r->r0[k]);
        }
    }
}

/* Constant propagation can change NaN payloads and subnormal FP status.
 * Classify consumed raw bits without FP comparisons; exceptional operands
 * keep the generic instruction sequence. Signed zero remains eligible. */
KELVIN_INLINE bool normal_or_zero_vector8(__m256 value)
{
    __m256i bits = _mm256_castps_si256(value);
    __m256i exponent = _mm256_and_si256(bits,
                                       _mm256_set1_epi32(0x7F800000));
    __m256i magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFFFF));
    __m256i zero = _mm256_setzero_si256();
    __m256i subnormal = _mm256_andnot_si256(_mm256_cmpeq_epi32(magnitude, zero),
                                           _mm256_cmpeq_epi32(exponent, zero));
    __m256i bad = _mm256_or_si256(subnormal,
        _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7F800000)));
    return _mm256_movemask_epi8(bad) == 0;
}

KELVIN_INLINE bool normal_or_zero_uniform(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t exponent = bits & 0x7F800000u;
    return exponent != 0x7F800000u && (exponent != 0 || (bits & 0x7FFFFFFFu) == 0);
}

static bool world_regular8(const Regs8 *r)
{
    for (int k = 0; k < 3; k++) if (!normal_or_zero_vector8(r->t[0][k])) return false;
    for (int k = 0; k < 4; k++) {
        if (!normal_or_zero_vector8(r->t[1][k]) || !normal_or_zero_vector8(r->fog[k])) return false;
    }
    return true;
}

static bool background_regular8(const KPixelState *ps, const Regs8 *r)
{
    for (int k = 0; k < 4; k++) {
        if (!normal_or_zero_vector8(r->t[0][k]) || !normal_or_zero_vector8(r->t[1][k]) ||
            !normal_or_zero_vector8(r->v0[k]) || !normal_or_zero_uniform(ps->stage[1].c0[k])) return false;
    }
    for (int k = 0; k < 3; k++) if (!normal_or_zero_uniform(ps->stage[4].c0[k])) return false;
    return true;
}

/* The producer validates and snapshots this ID before queue publication.
 * It is not derived from mutable uniforms or texture pointers. */
static void combine_program8(const KPixelState *ps, Regs8 *r, float out[4][8],
                              KPixelProgram program)
{
    if ((unsigned int)program & KPIX_SELECT_PREPARED)
        program = (KPixelProgram)((unsigned int)program & KPIX_SELECT_COMBINER);
    switch (program) {
    case KPIX_PROGRAM_WORLD_FOG:
        if (world_regular8(r)) combine8_world_fog(ps, r, out);
        else combine8(ps, r, out);
        break;
    case KPIX_PROGRAM_BACKGROUND:
        if (background_regular8(ps, r)) combine8_background(ps, r, out);
        else combine8(ps, r, out);
        break;
    default: combine8(ps, r, out); break;
    }
}

/* texture_stage for stage i of eight fragments at once, when its mode and
 * texture have an eight-wide form: PROJECT2D on a 2D texture that samples
 * the same at every LOD (lod_free, so the LOD is 0 as texture_stage passes
 * it) with no colour key. The same as texture_stage per fragment: q == 0
 * gives (0, 0, 0, 1), otherwise p / q (and rect_norm's division) sampled by
 * ktex_sample_2d8, then the alpha kill. Results into each lane's registers;
 * false, having done nothing, if the stage has no such form. */
static bool stage8_sample(const KPixelState *ps, const KFragment *f, int n, int i,
                          bool keep[8], float out[4][8], bool eligible)
{
    const KTexture *tx = &ps->tex[i];
    float s[8], tc[8], q[8];
    unsigned int mask = 0;
    int l, k;

    if (!eligible) {
        return false;
    }
    for (l = 0; l < 8; l++) {
        const float *p = f[l < n ? l : 0].t[i];
        q[l] = p[3];
        s[l] = p[0];
        tc[l] = p[1];
    }
    if (!ps->lod_free[i]) {
        /* proj_lod, then ktex_lod, lane by lane in vectors; each lane's
         * last step (0.5 log2 rho^2) as ktex_lod takes it. */
        float lod[8], rho2[8];
        __m256 p0 = _mm256_loadu_ps(s), p1 = _mm256_loadu_ps(tc), w = _mm256_loadu_ps(q);
        __m256 dx[4], dy[4], iw, ps0, tt, dsdx, dtdx, dsdy, dtdy, ax, ay, bx, by;
        for (k = 0; k < 4; k++) {
#define DX(l) f[(l) < n ? (l) : 0].dtdx[i][k]
#define DY(l) f[(l) < n ? (l) : 0].dtdy[i][k]
            dx[k] = _mm256_setr_ps(DX(0), DX(1), DX(2), DX(3), DX(4), DX(5), DX(6), DX(7));
            dy[k] = _mm256_setr_ps(DY(0), DY(1), DY(2), DY(3), DY(4), DY(5), DY(6), DY(7));
#undef DX
#undef DY
        }
        iw = _mm256_div_ps(_mm256_set1_ps(1.0f), w);
        ps0 = _mm256_mul_ps(p0, iw);
        tt = _mm256_mul_ps(p1, iw);
        dsdx = _mm256_mul_ps(_mm256_sub_ps(dx[0], _mm256_mul_ps(ps0, dx[3])), iw);
        dtdx = _mm256_mul_ps(_mm256_sub_ps(dx[1], _mm256_mul_ps(tt, dx[3])), iw);
        dsdy = _mm256_mul_ps(_mm256_sub_ps(dy[0], _mm256_mul_ps(ps0, dy[3])), iw);
        dtdy = _mm256_mul_ps(_mm256_sub_ps(dy[1], _mm256_mul_ps(tt, dy[3])), iw);
        if (ps->rect_tex[i]) {
            __m256 iwd = _mm256_set1_ps(tx->width ? 1.0f / tx->width : 1.0f);
            __m256 ihd = _mm256_set1_ps(tx->height ? 1.0f / tx->height : 1.0f);
            dsdx = _mm256_mul_ps(dsdx, iwd);
            dsdy = _mm256_mul_ps(dsdy, iwd);
            dtdx = _mm256_mul_ps(dtdx, ihd);
            dtdy = _mm256_mul_ps(dtdy, ihd);
        }
        ax = _mm256_mul_ps(dsdx, _mm256_set1_ps((float)tx->width));
        ay = _mm256_mul_ps(dtdx, _mm256_set1_ps((float)tx->height));
        bx = _mm256_mul_ps(dsdy, _mm256_set1_ps((float)tx->width));
        by = _mm256_mul_ps(dtdy, _mm256_set1_ps((float)tx->height));
        /* MAX(a, b) is a > b ? a : b, which MAXPS(a, b) is */
        _mm256_storeu_ps(rho2, _mm256_max_ps(
            _mm256_add_ps(_mm256_mul_ps(ax, ax), _mm256_mul_ps(ay, ay)),
            _mm256_add_ps(_mm256_mul_ps(bx, bx), _mm256_mul_ps(by, by))));
        for (l = 0; l < 8; l++) {
            lod[l] = q[l] == 0.0f ? 0.0f : ktex_lod_rho2(rho2[l]);
        }
        {
            __m256 qv = _mm256_loadu_ps(q);
            __m256 sv = _mm256_div_ps(p0, qv);
            __m256 tv = _mm256_div_ps(p1, qv);
            if (ps->rect_tex[i]) {
                sv = _mm256_div_ps(sv, _mm256_set1_ps(tx->width ? (float)tx->width : 1.0f));
                tv = _mm256_div_ps(tv, _mm256_set1_ps(tx->height ? (float)tx->height : 1.0f));
            }
            _mm256_storeu_ps(s, sv);
            _mm256_storeu_ps(tc, tv);
        }
        for (l = 0; l < 8; l++) {
            if (keep[l] && q[l] != 0.0f) {
                mask |= 1u << l;
            }
        }
        ktex_sample_2d8_lod(tx, s, tc, lod, mask, out);
        goto store;
    }
    {
        __m256 qv = _mm256_loadu_ps(q);
        __m256 sv = _mm256_div_ps(_mm256_loadu_ps(s), qv);
        __m256 tv = _mm256_div_ps(_mm256_loadu_ps(tc), qv);
        if (ps->rect_tex[i]) {
            sv = _mm256_div_ps(sv, _mm256_set1_ps(tx->width ? (float)tx->width : 1.0f));
            tv = _mm256_div_ps(tv, _mm256_set1_ps(tx->height ? (float)tx->height : 1.0f));
        }
        _mm256_storeu_ps(s, sv);
        _mm256_storeu_ps(tc, tv);
    }
    for (l = 0; l < 8; l++) {
        if (keep[l] && q[l] != 0.0f) {
            mask |= 1u << l;
        }
    }
    ktex_sample_2d8(tx, s, tc, mask, out);
store:
    for (l = 0; l < 8; l++) {
        if (q[l] == 0.0f) {
            out[0][l] = out[1][l] = out[2][l] = 0.0f;
            out[3][l] = 1.0f;
        }
        if (tx->alphakill && out[3][l] == 0.0f) {
            keep[l] = false;
        }
    }
    return true;
}

static bool stage8(const KPixelState *ps, const KFragment *f, int n, int i,
                   Regs rl[8], bool keep[8], KPixelProgram program)
{
    float out[4][8];
    bool eligible = (unsigned int)program & KPIX_SELECT_PREPARED
        ? ((unsigned int)program & KPIX_SELECT_STAGE(i)) != 0
        : stage8_eligible(ps, i);
    if (!stage8_sample(ps, f, n, i, keep, out, eligible)) return false;
    for (int l = 0; l < 8; l++) {
        for (int k = 0; k < 4; k++) rl[l].t[i][k] = out[k][l];
    }
    return true;
}

/* PROJECT2D stages read only their own fragment coordinates. If every
 * stage is PROJECT2D or NONE, the combiner can consume the sampler's lane
 * vectors directly; dependent modes keep the scalar register file. */
static bool independent_stages8(const KPixelState *ps)
{
    for (int i = 0; i < 4; i++) {
        const KTexture *tx = &ps->tex[i];
        if (ps->tex_mode[i] == PS_TEXTUREMODES_NONE) continue;
        if (ps->tex_mode[i] != PS_TEXTUREMODES_PROJECT2D || !tx->valid ||
            tx->cubemap || tx->colorkey_mode) return false;
    }
    return true;
}

int kpix_shade8_program(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                const float *zvalue, int n, KPixelProgram program)
{
    Regs8 r;
    float tex[4][4][8];                /* texture stage results by lane */
    float fogc[8], out[4][8];
    bool keep[8];
    int written = 0, i, k, l;

    if (n <= 0) {
        return 0;
    }
    /* The texture stages, a stage at a time: eight fragments at once where
     * texture_stage's sampling has an eight-wide form (stage8), one
     * fragment at a time otherwise, a stage reading the earlier stages'
     * results per fragment either way. A fragment a stage discards runs no
     * later stage alone; eight wide, its lane runs on and is dropped. Lanes
     * past n repeat the first fragment and are never written. */
    if ((unsigned int)program & KPIX_SELECT_PREPARED
        ? ((unsigned int)program & KPIX_SELECT_INDEPENDENT) != 0
        : independent_stages8(ps)) {
        for (l = 0; l < 8; l++) keep[l] = l < n;
        for (i = 0; i < 4; i++) {
            if (ps->tex_mode[i] == PS_TEXTUREMODES_NONE) {
                for (k = 0; k < 4; k++) {
                    r.t[i][k] = _mm256_set1_ps(k == 3 ? 1.0f : 0.0f);
                }
            } else {
                stage8_sample(ps, f, n, i, keep, tex[i], true);
                for (k = 0; k < 4; k++) r.t[i][k] = _mm256_loadu_ps(tex[i][k]);
            }
        }
    } else {
        Regs rl[8];
        float dots[8][4];
        for (l = 0; l < 8; l++) {
            keep[l] = l < n;
            memset(rl[l].t, 0, sizeof(rl[l].t));
            memset(dots[l], 0, sizeof(dots[l]));
        }
        for (i = 0; i < 4; i++) {
            if (ps->tex_mode[i] == PS_TEXTUREMODES_NONE) {
                for (l = 0; l < 8; l++) {
                    rl[l].t[i][0] = rl[l].t[i][1] = rl[l].t[i][2] = 0.0f;
                    rl[l].t[i][3] = 1.0f;
                }
                continue;
            }
            if (stage8(ps, f, n, i, rl, keep, program)) {
                continue;
            }
            for (l = 0; l < 8; l++) {
                if (keep[l] && !texture_stage(ps, &f[l], &rl[l], i, dots[l])) {
                    keep[l] = false;
                }
            }
        }
        for (l = 0; l < 8; l++) {
            for (i = 0; i < 4; i++) {
                for (k = 0; k < 4; k++) {
                    tex[i][k][l] = rl[l].t[i][k];
                }
            }
        }
        for (k = 0; k < 4; k++) {
            for (i = 0; i < 4; i++) r.t[i][k] = _mm256_loadu_ps(tex[i][k]);
        }
    }
    for (l = 0; l < 8; l++) {
        const KFragment *fl = &f[l < n ? l : 0];
        fogc[l] = fl->fog < 0.0f ? 0.0f : (fl->fog > 1.0f ? 1.0f : fl->fog);
    }
#define LANE_F(l) (&f[(l) < n ? (l) : 0])
    for (k = 0; k < 4; k++) {
#define D0(l) LANE_F(l)->d0[k]
#define D1(l) LANE_F(l)->d1[k]
        r.v0[k] = GATHER8(D0);
        r.v1[k] = GATHER8(D1);
#undef D0
#undef D1
        r.r0[k] = r.r1[k] = _mm256_setzero_ps();
    }
#undef LANE_F
    r.fog[0] = _mm256_set1_ps(ps->fog_color[0]);
    r.fog[1] = _mm256_set1_ps(ps->fog_color[1]);
    r.fog[2] = _mm256_set1_ps(ps->fog_color[2]);
    r.fog[3] = _mm256_loadu_ps(fogc);
    r.r0[3] = ps->tex_mode[0] != PS_TEXTUREMODES_NONE ? r.t[0][3] : _mm256_set1_ps(1.0f);
    combine_program8(ps, &r, out, program);

    written = shade_tail8_prepared(ps, t, f, zvalue, n, keep, out,
        (unsigned int)program & KPIX_SELECT_PREPARED
            ? ((unsigned int)program & KPIX_SELECT_TAIL_STATE) != 0
            : tail8_state_eligible(ps));
    if (written >= 0) return written;
    written = 0;
    for (l = 0; l < n; l++) {
        float frag[4];
        if (!keep[l]) {
            continue;
        }
        frag[0] = out[0][l]; frag[1] = out[1][l];
        frag[2] = out[2][l]; frag[3] = out[3][l];
        written += shade_tail(ps, t, f[l].x, f[l].y, zvalue[l], frag);
    }
    return written;
}

int kpix_shade8(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                const float *zvalue, int n)
{
    return kpix_shade8_program(ps, t, f, zvalue, n, KPIX_PROGRAM_GENERIC);
}


#undef GATHER8

#else

int kpix_shade8(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                const float *zvalue, int n)
{
    int written = 0, l;
    for (l = 0; l < n; l++) {
        written += kpix_shade(ps, t, &f[l], zvalue[l]);
    }
    return written;
}

#endif


#if !defined(__AVX2__)
int kpix_shade8_program(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                        const float *zvalue, int n, KPixelProgram program)
{
    (void)program;
    return kpix_shade8(ps, t, f, zvalue, n);
}
#endif
