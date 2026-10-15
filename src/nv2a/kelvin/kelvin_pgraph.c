/*
 * Kelvin (NV2A 3D engine) method handlers.
 *
 * Ported from xemu hw/xbox/nv2a/pgraph/pgraph.c and pgraph/vertex.c, where
 * each method is translated into PGRAPH register fields exactly as xemu does.
 * Differences from upstream: renderer operations call the CPU renderer
 * (kelvin_render_*), GCC case ranges are if-chains, asserts log instead of
 * aborting, and FLIP_INCREMENT_WRITE does not divide by a zero modulo.
 *
 * Copyright (c) 2012 espes
 * Copyright (c) 2015 Jannik Vogel
 * Copyright (c) 2018-2025 Matt Borgerson
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <stdlib.h>
#include "kelvin_int.h"
#include "nv2a_vsh_cpu/nv2a_vsh_emulator.h"

#define KELVIN_DPRINTF(...) do { } while (0)
#define KELVIN_ASSERT(cond)                                              \
    do {                                                                 \
        if (!(cond)) {                                                   \
            static int said_;                                            \
            if (!said_++)                                                \
                fprintf(stderr, "[KELVIN] check failed: %s (%s:%d)\n",   \
                        #cond, __FILE__, __LINE__);                      \
        }                                                                \
    } while (0)

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static void pgraph_expand_draw_arrays(KelvinState *pg);
static void pgraph_check_within_begin_end_block(KelvinState *pg);

static void kelvin_SET_OBJECT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->kelvin_object_instance = parameter;
}

volatile long long g_kelvin_notify_raise_qpc, g_kelvin_notify_ticks, g_kelvin_notify_count;

static void kelvin_NO_OPERATION(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    /* The bios uses nop as a software method call -
     * it seems to expect a notify interrupt if the parameter isn't 0.
     * According to a nouveau guy it should still be a nop regardless
     * of the parameter. It's possible a debug register enables this,
     * but nothing obvious sticks out. Weird.
     */
    if (parameter == 0) {
        return;
    }
    /* The title waits on this notify as "rendering reached here". */
    kelvin_render_sync_for(KT_SYNC_NOTIFY);

    unsigned channel_id =
        PG_GET_MASK(NV_PGRAPH_CTX_USER, NV_PGRAPH_CTX_USER_CHID);

    KELVIN_ASSERT(!(pg->pending_interrupts & NV_PGRAPH_INTR_ERROR));

    PG_SET_MASK(NV_PGRAPH_TRAPPED_ADDR, NV_PGRAPH_TRAPPED_ADDR_CHID,
             channel_id);
    PG_SET_MASK(NV_PGRAPH_TRAPPED_ADDR, NV_PGRAPH_TRAPPED_ADDR_SUBCH,
             subchannel);
    PG_SET_MASK(NV_PGRAPH_TRAPPED_ADDR, NV_PGRAPH_TRAPPED_ADDR_MTHD,
             method);
    pgraph_reg_w(pg, NV_PGRAPH_TRAPPED_DATA_LOW, parameter);
    pgraph_reg_w(pg, NV_PGRAPH_NSOURCE,
             NV_PGRAPH_NSOURCE_NOTIFICATION); /* TODO: check this */
    pg->pending_interrupts |= NV_PGRAPH_INTR_ERROR;
    pg->waiting_for_nop = true;
    pg->stats.notifies++;
    {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        g_kelvin_notify_raise_qpc = now.QuadPart;
    }

    kelvin_irq_update();
}

static void kelvin_WAIT_FOR_IDLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    kelvin_render_sync();
}

static void kelvin_SET_FLIP_READ(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SURFACE, NV_PGRAPH_SURFACE_READ_3D,
             parameter);
}

static void kelvin_SET_FLIP_WRITE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SURFACE, NV_PGRAPH_SURFACE_WRITE_3D,
             parameter);
}

static void kelvin_SET_FLIP_MODULO(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SURFACE, NV_PGRAPH_SURFACE_MODULO_3D,
             parameter);
}

static void kelvin_FLIP_INCREMENT_WRITE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    uint32_t modulo = PG_GET_MASK(NV_PGRAPH_SURFACE,
                                  NV_PGRAPH_SURFACE_MODULO_3D);
    uint32_t next = PG_GET_MASK(NV_PGRAPH_SURFACE,
                                NV_PGRAPH_SURFACE_WRITE_3D) + 1;

    /* xemu divides unconditionally; a title that increments before setting
     * the modulo would fault the host here, so wrap only when it is set. */
    if (modulo) {
        next %= modulo;
    }
    PG_SET_MASK(NV_PGRAPH_SURFACE, NV_PGRAPH_SURFACE_WRITE_3D, next);
    pg->stats.flips++;
}

static void kelvin_FLIP_STALL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    kelvin_render_flip_stall(pg, g_kelvin_frame_aspect_w, g_kelvin_frame_aspect_h);
    pg->waiting_for_flip = true;
}

// TODO: these should be loading the dma objects from ramin here?

static void kelvin_SET_CONTEXT_DMA_NOTIFIES(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_notifies = parameter;
}

static void kelvin_SET_CONTEXT_DMA_A(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_a = parameter;
}

static void kelvin_SET_CONTEXT_DMA_B(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_b = parameter;
}

static void kelvin_SET_CONTEXT_DMA_STATE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_state = parameter;
}

static void kelvin_SET_CONTEXT_DMA_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    /* try to get any straggling draws in before the surface's changed :/ */

    pg->dma_color = parameter;
}

static void kelvin_SET_CONTEXT_DMA_ZETA(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_zeta = parameter;
}

static void kelvin_SET_CONTEXT_DMA_VERTEX_A(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_vertex_a = parameter;
}

static void kelvin_SET_CONTEXT_DMA_VERTEX_B(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_vertex_b = parameter;
}

static void kelvin_SET_CONTEXT_DMA_SEMAPHORE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->dma_semaphore = parameter;
}

static void kelvin_SET_CONTEXT_DMA_REPORT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{

    pg->dma_report = parameter;
}

static void kelvin_SET_SURFACE_CLIP_HORIZONTAL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{

    pg->surface_shape.clip_x =
        GET_MASK(parameter, NV097_SET_SURFACE_CLIP_HORIZONTAL_X);
    pg->surface_shape.clip_width =
        GET_MASK(parameter, NV097_SET_SURFACE_CLIP_HORIZONTAL_WIDTH);
}

static void kelvin_SET_SURFACE_CLIP_VERTICAL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{

    pg->surface_shape.clip_y =
        GET_MASK(parameter, NV097_SET_SURFACE_CLIP_VERTICAL_Y);
    pg->surface_shape.clip_height =
        GET_MASK(parameter, NV097_SET_SURFACE_CLIP_VERTICAL_HEIGHT);
}

static void kelvin_SET_SURFACE_FORMAT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{

    pg->surface_shape.color_format =
        GET_MASK(parameter, NV097_SET_SURFACE_FORMAT_COLOR);
    pg->surface_shape.zeta_format =
        GET_MASK(parameter, NV097_SET_SURFACE_FORMAT_ZETA);
    pg->surface_shape.anti_aliasing =
        GET_MASK(parameter, NV097_SET_SURFACE_FORMAT_ANTI_ALIASING);
    pg->surface_shape.log_width =
        GET_MASK(parameter, NV097_SET_SURFACE_FORMAT_WIDTH);
    pg->surface_shape.log_height =
        GET_MASK(parameter, NV097_SET_SURFACE_FORMAT_HEIGHT);

    int surface_type = GET_MASK(parameter, NV097_SET_SURFACE_FORMAT_TYPE);
    if (surface_type != pg->surface_type) {
        pg->surface_type = surface_type;
    }
}

static void kelvin_SET_SURFACE_PITCH(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
        unsigned int color_pitch = GET_MASK(parameter, NV097_SET_SURFACE_PITCH_COLOR);
    unsigned int zeta_pitch  = GET_MASK(parameter, NV097_SET_SURFACE_PITCH_ZETA);
    pg->surface_color.pitch = color_pitch;
    pg->surface_zeta.pitch = zeta_pitch;
}

static void kelvin_SET_SURFACE_COLOR_OFFSET(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->surface_color.offset = parameter;
}

static void kelvin_SET_SURFACE_ZETA_OFFSET(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->surface_zeta.offset = parameter;
}

static void kelvin_SET_COMBINER_ALPHA_ICW(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COMBINER_ALPHA_ICW) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_COMBINEALPHAI0 + slot * 4, parameter);
}

static void kelvin_SET_COMBINER_SPECULAR_FOG_CW0(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_COMBINESPECFOG0, parameter);
}

static void kelvin_SET_COMBINER_SPECULAR_FOG_CW1(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_COMBINESPECFOG1, parameter);
}

static void kelvin_SET_TEXTURE_ADDRESS(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_ADDRESS) / 64;
    pgraph_reg_w(pg, NV_PGRAPH_TEXADDRESS0 + slot * 4, parameter);
}

static void kelvin_SET_CONTROL0(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{

    bool stencil_write_enable =
        parameter & NV097_SET_CONTROL0_STENCIL_WRITE_ENABLE;
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_STENCIL_WRITE_ENABLE,
             stencil_write_enable);

    uint32_t z_format = GET_MASK(parameter, NV097_SET_CONTROL0_Z_FORMAT);
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_Z_FORMAT, z_format);

    bool z_perspective =
        parameter & NV097_SET_CONTROL0_Z_PERSPECTIVE_ENABLE;
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_Z_PERSPECTIVE_ENABLE,
             z_perspective);
}

static void kelvin_SET_LIGHT_CONTROL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_SEPARATE_SPECULAR,
             (parameter & NV097_SET_LIGHT_CONTROL_SEPARATE_SPECULAR) != 0);

    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_LOCALEYE,
             (parameter & NV097_SET_LIGHT_CONTROL_LOCALEYE) != 0);

    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_ALPHA_FROM_MATERIAL_SPECULAR,
             (parameter & NV097_SET_LIGHT_CONTROL_ALPHA_FROM_MATERIAL_SPECULAR) != 0);
}

static void kelvin_SET_COLOR_MATERIAL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_EMISSION,
             (parameter >> 0) & 3);
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_AMBIENT,
             (parameter >> 2) & 3);
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_DIFFUSE,
             (parameter >> 4) & 3);
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_SPECULAR,
             (parameter >> 6) & 3);
}

static void kelvin_SET_FOG_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    /* FIXME: There is also NV_PGRAPH_CSV0_D_FOG_MODE */
    unsigned int mode;
    switch (parameter) {
    case NV097_SET_FOG_MODE_V_LINEAR:
        mode = NV_PGRAPH_CONTROL_3_FOG_MODE_LINEAR; break;
    case NV097_SET_FOG_MODE_V_EXP:
        mode = NV_PGRAPH_CONTROL_3_FOG_MODE_EXP; break;
    case NV097_SET_FOG_MODE_V_EXP2:
        mode = NV_PGRAPH_CONTROL_3_FOG_MODE_EXP2; break;
    case NV097_SET_FOG_MODE_V_EXP_ABS:
        mode = NV_PGRAPH_CONTROL_3_FOG_MODE_EXP_ABS; break;
    case NV097_SET_FOG_MODE_V_EXP2_ABS:
        mode = NV_PGRAPH_CONTROL_3_FOG_MODE_EXP2_ABS; break;
    case NV097_SET_FOG_MODE_V_LINEAR_ABS:
        mode = NV_PGRAPH_CONTROL_3_FOG_MODE_LINEAR_ABS; break;
    default:
        KELVIN_ASSERT(!"Invalid or unimplemented fog mode");
        break;
    }
    PG_SET_MASK(NV_PGRAPH_CONTROL_3, NV_PGRAPH_CONTROL_3_FOG_MODE,
             mode);
}

static void kelvin_SET_FOG_GEN_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    unsigned int mode;
    switch (parameter) {
    case NV097_SET_FOG_GEN_MODE_V_SPEC_ALPHA:
        mode = NV_PGRAPH_CSV0_D_FOGGENMODE_SPEC_ALPHA; break;
    case NV097_SET_FOG_GEN_MODE_V_RADIAL:
        mode = NV_PGRAPH_CSV0_D_FOGGENMODE_RADIAL; break;
    case NV097_SET_FOG_GEN_MODE_V_PLANAR:
        mode = NV_PGRAPH_CSV0_D_FOGGENMODE_PLANAR; break;
    case NV097_SET_FOG_GEN_MODE_V_ABS_PLANAR:
        mode = NV_PGRAPH_CSV0_D_FOGGENMODE_ABS_PLANAR; break;
    case NV097_SET_FOG_GEN_MODE_V_FOG_X:
        mode = NV_PGRAPH_CSV0_D_FOGGENMODE_FOG_X; break;
    default:
        KELVIN_ASSERT(!"Invalid fog gen mode");
        break;
    }
    PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_FOGGENMODE, mode);
}

static void kelvin_SET_FOG_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    /*
      FIXME: There is also:
        PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_FOGENABLE,
             parameter);
    */
    PG_SET_MASK(NV_PGRAPH_CONTROL_3, NV_PGRAPH_CONTROL_3_FOGENABLE,
         parameter);
}

static void kelvin_SET_FOG_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    /* PGRAPH channels are ARGB, parameter channels are ABGR */
    uint8_t red = GET_MASK(parameter, NV097_SET_FOG_COLOR_RED);
    uint8_t green = GET_MASK(parameter, NV097_SET_FOG_COLOR_GREEN);
    uint8_t blue = GET_MASK(parameter, NV097_SET_FOG_COLOR_BLUE);
    uint8_t alpha = GET_MASK(parameter, NV097_SET_FOG_COLOR_ALPHA);
    PG_SET_MASK(NV_PGRAPH_FOGCOLOR, NV_PGRAPH_FOGCOLOR_RED, red);
    PG_SET_MASK(NV_PGRAPH_FOGCOLOR, NV_PGRAPH_FOGCOLOR_GREEN, green);
    PG_SET_MASK(NV_PGRAPH_FOGCOLOR, NV_PGRAPH_FOGCOLOR_BLUE, blue);
    PG_SET_MASK(NV_PGRAPH_FOGCOLOR, NV_PGRAPH_FOGCOLOR_ALPHA, alpha);
}

static void kelvin_SET_WINDOW_CLIP_TYPE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_WINDOWCLIPTYPE, parameter);
}

static void kelvin_SET_WINDOW_CLIP_HORIZONTAL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_WINDOW_CLIP_HORIZONTAL) / 4;
    for (; slot < 8; ++slot) {
        pgraph_reg_w(pg, NV_PGRAPH_WINDOWCLIPX0 + slot * 4, parameter);
    }
}

static void kelvin_SET_WINDOW_CLIP_VERTICAL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_WINDOW_CLIP_VERTICAL) / 4;
    for (; slot < 8; ++slot) {
        pgraph_reg_w(pg, NV_PGRAPH_WINDOWCLIPY0 + slot * 4, parameter);
    }
}

static void kelvin_SET_ALPHA_TEST_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_ALPHATESTENABLE, parameter);
}

static void kelvin_SET_BLEND_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_BLEND, NV_PGRAPH_BLEND_EN, parameter);
}

static void kelvin_SET_CULL_FACE_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_CULLENABLE,
             parameter);
}

static void kelvin_SET_DEPTH_TEST_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_0, NV_PGRAPH_CONTROL_0_ZENABLE,
             parameter);
}

static void kelvin_SET_DITHER_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_DITHERENABLE, parameter);
}

static void kelvin_SET_LIGHTING_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_LIGHTING,
             parameter);
}

static void kelvin_SET_POINT_PARAMS_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_POINTPARAMSENABLE,
             parameter);
    PG_SET_MASK(NV_PGRAPH_CONTROL_3,
             NV_PGRAPH_CONTROL_3_POINTPARAMSENABLE, parameter);
}

static void kelvin_SET_POINT_SMOOTH_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_POINTSMOOTHENABLE, parameter);
}

static void kelvin_SET_LINE_SMOOTH_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_LINESMOOTHENABLE, parameter);
}

static void kelvin_SET_POLY_SMOOTH_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_POLYSMOOTHENABLE, parameter);
}

static void kelvin_SET_SKIN_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_SKIN,
             parameter);
}

static void kelvin_SET_STENCIL_TEST_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_1,
             NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE, parameter);
}

static void kelvin_SET_POLY_OFFSET_POINT_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_POFFSETPOINTENABLE, parameter);
}

static void kelvin_SET_POLY_OFFSET_LINE_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_POFFSETLINEENABLE, parameter);
}

static void kelvin_SET_POLY_OFFSET_FILL_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_POFFSETFILLENABLE, parameter);
}

static void kelvin_SET_ALPHA_FUNC(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_ALPHAFUNC, parameter & 0xF);
}

static void kelvin_SET_ALPHA_REF(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_ALPHAREF, parameter);
}

static void kelvin_SET_BLEND_FUNC_SFACTOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    unsigned int factor;
    switch (parameter) {
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ZERO:
        factor = NV_PGRAPH_BLEND_SFACTOR_ZERO; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ONE:
        factor = NV_PGRAPH_BLEND_SFACTOR_ONE; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_SRC_COLOR:
        factor = NV_PGRAPH_BLEND_SFACTOR_SRC_COLOR; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ONE_MINUS_SRC_COLOR:
        factor = NV_PGRAPH_BLEND_SFACTOR_ONE_MINUS_SRC_COLOR; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_SRC_ALPHA:
        factor = NV_PGRAPH_BLEND_SFACTOR_SRC_ALPHA; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ONE_MINUS_SRC_ALPHA:
        factor = NV_PGRAPH_BLEND_SFACTOR_ONE_MINUS_SRC_ALPHA; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_DST_ALPHA:
        factor = NV_PGRAPH_BLEND_SFACTOR_DST_ALPHA; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ONE_MINUS_DST_ALPHA:
        factor = NV_PGRAPH_BLEND_SFACTOR_ONE_MINUS_DST_ALPHA; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_DST_COLOR:
        factor = NV_PGRAPH_BLEND_SFACTOR_DST_COLOR; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ONE_MINUS_DST_COLOR:
        factor = NV_PGRAPH_BLEND_SFACTOR_ONE_MINUS_DST_COLOR; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_SRC_ALPHA_SATURATE:
        factor = NV_PGRAPH_BLEND_SFACTOR_SRC_ALPHA_SATURATE; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_CONSTANT_COLOR:
        factor = NV_PGRAPH_BLEND_SFACTOR_CONSTANT_COLOR; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ONE_MINUS_CONSTANT_COLOR:
        factor = NV_PGRAPH_BLEND_SFACTOR_ONE_MINUS_CONSTANT_COLOR; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_CONSTANT_ALPHA:
        factor = NV_PGRAPH_BLEND_SFACTOR_CONSTANT_ALPHA; break;
    case NV097_SET_BLEND_FUNC_SFACTOR_V_ONE_MINUS_CONSTANT_ALPHA:
        factor = NV_PGRAPH_BLEND_SFACTOR_ONE_MINUS_CONSTANT_ALPHA; break;
    default:
        KELVIN_DPRINTF("Unknown blend source factor: 0x%08x\n", parameter);
        return; /* discard */
    }
    PG_SET_MASK(NV_PGRAPH_BLEND, NV_PGRAPH_BLEND_SFACTOR, factor);
}

static void kelvin_SET_BLEND_FUNC_DFACTOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    unsigned int factor;
    switch (parameter) {
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ZERO:
        factor = NV_PGRAPH_BLEND_DFACTOR_ZERO; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ONE:
        factor = NV_PGRAPH_BLEND_DFACTOR_ONE; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_SRC_COLOR:
        factor = NV_PGRAPH_BLEND_DFACTOR_SRC_COLOR; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ONE_MINUS_SRC_COLOR:
        factor = NV_PGRAPH_BLEND_DFACTOR_ONE_MINUS_SRC_COLOR; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_SRC_ALPHA:
        factor = NV_PGRAPH_BLEND_DFACTOR_SRC_ALPHA; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ONE_MINUS_SRC_ALPHA:
        factor = NV_PGRAPH_BLEND_DFACTOR_ONE_MINUS_SRC_ALPHA; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_DST_ALPHA:
        factor = NV_PGRAPH_BLEND_DFACTOR_DST_ALPHA; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ONE_MINUS_DST_ALPHA:
        factor = NV_PGRAPH_BLEND_DFACTOR_ONE_MINUS_DST_ALPHA; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_DST_COLOR:
        factor = NV_PGRAPH_BLEND_DFACTOR_DST_COLOR; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ONE_MINUS_DST_COLOR:
        factor = NV_PGRAPH_BLEND_DFACTOR_ONE_MINUS_DST_COLOR; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_SRC_ALPHA_SATURATE:
        factor = NV_PGRAPH_BLEND_DFACTOR_SRC_ALPHA_SATURATE; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_CONSTANT_COLOR:
        factor = NV_PGRAPH_BLEND_DFACTOR_CONSTANT_COLOR; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ONE_MINUS_CONSTANT_COLOR:
        factor = NV_PGRAPH_BLEND_DFACTOR_ONE_MINUS_CONSTANT_COLOR; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_CONSTANT_ALPHA:
        factor = NV_PGRAPH_BLEND_DFACTOR_CONSTANT_ALPHA; break;
    case NV097_SET_BLEND_FUNC_DFACTOR_V_ONE_MINUS_CONSTANT_ALPHA:
        factor = NV_PGRAPH_BLEND_DFACTOR_ONE_MINUS_CONSTANT_ALPHA; break;
    default:
        KELVIN_DPRINTF("Unknown blend destination factor: 0x%08x\n", parameter);
        return; /* discard */
    }
    PG_SET_MASK(NV_PGRAPH_BLEND, NV_PGRAPH_BLEND_DFACTOR, factor);
}

static void kelvin_SET_BLEND_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_BLENDCOLOR, parameter);
}

static void kelvin_SET_BLEND_EQUATION(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    unsigned int equation;
    switch (parameter) {
    case NV097_SET_BLEND_EQUATION_V_FUNC_SUBTRACT:
        equation = 0; break;
    case NV097_SET_BLEND_EQUATION_V_FUNC_REVERSE_SUBTRACT:
        equation = 1; break;
    case NV097_SET_BLEND_EQUATION_V_FUNC_ADD:
        equation = 2; break;
    case NV097_SET_BLEND_EQUATION_V_MIN:
        equation = 3; break;
    case NV097_SET_BLEND_EQUATION_V_MAX:
        equation = 4; break;
    case NV097_SET_BLEND_EQUATION_V_FUNC_REVERSE_SUBTRACT_SIGNED:
        equation = 5; break;
    case NV097_SET_BLEND_EQUATION_V_FUNC_ADD_SIGNED:
        equation = 6; break;
    default:
        KELVIN_DPRINTF("Unknown blend equation: 0x%08x\n", parameter);
        return; /* discard */
    }
    PG_SET_MASK(NV_PGRAPH_BLEND, NV_PGRAPH_BLEND_EQN, equation);
}

static void kelvin_SET_DEPTH_FUNC(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    if (parameter >= 0x200 && parameter <= 0x207) {
        PG_SET_MASK(NV_PGRAPH_CONTROL_0, NV_PGRAPH_CONTROL_0_ZFUNC,
                    parameter & 0xF);
    }
}

static void kelvin_SET_COLOR_MASK(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{

    bool alpha = parameter & NV097_SET_COLOR_MASK_ALPHA_WRITE_ENABLE;
    bool red = parameter & NV097_SET_COLOR_MASK_RED_WRITE_ENABLE;
    bool green = parameter & NV097_SET_COLOR_MASK_GREEN_WRITE_ENABLE;
    bool blue = parameter & NV097_SET_COLOR_MASK_BLUE_WRITE_ENABLE;
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE, alpha);
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE, red);
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE, green);
    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE, blue);
}

static void kelvin_SET_DEPTH_MASK(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{

    PG_SET_MASK(NV_PGRAPH_CONTROL_0,
             NV_PGRAPH_CONTROL_0_ZWRITEENABLE, parameter);
}

static void kelvin_SET_STENCIL_MASK(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_1,
             NV_PGRAPH_CONTROL_1_STENCIL_MASK_WRITE, parameter);
}

static void kelvin_SET_STENCIL_FUNC(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_1,
             NV_PGRAPH_CONTROL_1_STENCIL_FUNC, parameter & 0xF);
}

static void kelvin_SET_STENCIL_FUNC_REF(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_1,
             NV_PGRAPH_CONTROL_1_STENCIL_REF, parameter);
}

static void kelvin_SET_STENCIL_FUNC_MASK(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_1,
             NV_PGRAPH_CONTROL_1_STENCIL_MASK_READ, parameter);
}

static unsigned int kelvin_map_stencil_op(uint32_t parameter)
{
    unsigned int op;
    switch (parameter) {
    case NV097_SET_STENCIL_OP_V_KEEP:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_KEEP; break;
    case NV097_SET_STENCIL_OP_V_ZERO:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_ZERO; break;
    case NV097_SET_STENCIL_OP_V_REPLACE:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_REPLACE; break;
    case NV097_SET_STENCIL_OP_V_INCRSAT:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_INCRSAT; break;
    case NV097_SET_STENCIL_OP_V_DECRSAT:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_DECRSAT; break;
    case NV097_SET_STENCIL_OP_V_INVERT:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_INVERT; break;
    case NV097_SET_STENCIL_OP_V_INCR:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_INCR; break;
    case NV097_SET_STENCIL_OP_V_DECR:
        op = NV_PGRAPH_CONTROL_2_STENCIL_OP_V_DECR; break;
    default:
        KELVIN_ASSERT(!"Invalid kelvin map stencil operation");
        break;
    }
    return op;
}

static void kelvin_SET_STENCIL_OP_FAIL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_2,
             NV_PGRAPH_CONTROL_2_STENCIL_OP_FAIL,
             kelvin_map_stencil_op(parameter));
}

static void kelvin_SET_STENCIL_OP_ZFAIL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_2,
             NV_PGRAPH_CONTROL_2_STENCIL_OP_ZFAIL,
             kelvin_map_stencil_op(parameter));
}

static void kelvin_SET_STENCIL_OP_ZPASS(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CONTROL_2,
             NV_PGRAPH_CONTROL_2_STENCIL_OP_ZPASS,
             kelvin_map_stencil_op(parameter));
}

static void kelvin_SET_SHADE_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    switch (parameter) {
    case NV097_SET_SHADE_MODE_V_FLAT:
        PG_SET_MASK(NV_PGRAPH_CONTROL_3, NV_PGRAPH_CONTROL_3_SHADEMODE,
                 NV_PGRAPH_CONTROL_3_SHADEMODE_FLAT);
        break;
    case NV097_SET_SHADE_MODE_V_SMOOTH:
        PG_SET_MASK(NV_PGRAPH_CONTROL_3, NV_PGRAPH_CONTROL_3_SHADEMODE,
                 NV_PGRAPH_CONTROL_3_SHADEMODE_SMOOTH);
        break;
    default:
        /* Discard */
        break;
    }
}

static void kelvin_SET_PROVOKING_VERTEX(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    KELVIN_ASSERT((parameter & ~1) == 0);
    PG_SET_MASK(NV_PGRAPH_CONTROL_3, NV_PGRAPH_CONTROL_3_PROVOKING_VERTEX,
             parameter);
}

static void kelvin_SET_POLYGON_OFFSET_SCALE_FACTOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_ZOFFSETFACTOR, parameter);
}

static void kelvin_SET_POLYGON_OFFSET_BIAS(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_ZOFFSETBIAS, parameter);
}

static unsigned int kelvin_map_polygon_mode(uint32_t parameter)
{
    unsigned int mode;
    switch (parameter) {
    case NV097_SET_FRONT_POLYGON_MODE_V_POINT:
        mode = NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_POINT; break;
    case NV097_SET_FRONT_POLYGON_MODE_V_LINE:
        mode = NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_LINE; break;
    case NV097_SET_FRONT_POLYGON_MODE_V_FILL:
        mode = NV_PGRAPH_SETUPRASTER_FRONTFACEMODE_FILL; break;
    default:
        KELVIN_ASSERT(!"Invalid kelvin map polygon mode");
        break;
    }
    return mode;
}

static void kelvin_SET_FRONT_POLYGON_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_FRONTFACEMODE,
             kelvin_map_polygon_mode(parameter));
}

static void kelvin_SET_BACK_POLYGON_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER,
             NV_PGRAPH_SETUPRASTER_BACKFACEMODE,
             kelvin_map_polygon_mode(parameter));
}

static void kelvin_SET_CLIP_MIN(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_ZCLIPMIN, parameter);
}

static void kelvin_SET_CLIP_MAX(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_ZCLIPMAX, parameter);
}

static void kelvin_SET_CULL_FACE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    unsigned int face;
    switch (parameter) {
    case NV097_SET_CULL_FACE_V_FRONT:
        face = NV_PGRAPH_SETUPRASTER_CULLCTRL_FRONT; break;
    case NV097_SET_CULL_FACE_V_BACK:
        face = NV_PGRAPH_SETUPRASTER_CULLCTRL_BACK; break;
    case NV097_SET_CULL_FACE_V_FRONT_AND_BACK:
        face = NV_PGRAPH_SETUPRASTER_CULLCTRL_FRONT_AND_BACK; break;
    default:
        KELVIN_ASSERT(!"Invalid cull face parameter");
        break;
    }
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER, NV_PGRAPH_SETUPRASTER_CULLCTRL, face);
}

static void kelvin_SET_FRONT_FACE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    bool ccw;
    switch (parameter) {
    case NV097_SET_FRONT_FACE_V_CW:
        ccw = false; break;
    case NV097_SET_FRONT_FACE_V_CCW:
        ccw = true; break;
    default:
        KELVIN_DPRINTF("Unknown front face: 0x%08x\n", parameter);
        return; /* discard */
    }
    PG_SET_MASK(NV_PGRAPH_SETUPRASTER, NV_PGRAPH_SETUPRASTER_FRONTFACE,
                ccw ? 1 : 0);
}

static void kelvin_SET_NORMALIZATION_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_NORMALIZATION_ENABLE,
                parameter);
}

static void kelvin_SET_MATERIAL_EMISSION(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_MATERIAL_EMISSION) / 4;
    // FIXME: Verify NV_IGRAPH_XF_LTCTXA_CM_COL is correct
    pg->ltctxa[NV_IGRAPH_XF_LTCTXA_CM_COL][slot] = parameter;
}

static void kelvin_SET_MATERIAL_ALPHA(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->material_alpha = kelvin_u2f(parameter);
}

static void kelvin_SET_SPECULAR_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_C, NV_PGRAPH_CSV0_C_SPECULAR_ENABLE, parameter);
}

static void kelvin_SET_LIGHT_ENABLE_MASK(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_LIGHTS, parameter);
}

static unsigned int kelvin_map_texgen(uint32_t parameter, unsigned int channel)
{
    KELVIN_ASSERT(channel < 4);
    unsigned int texgen;
    switch (parameter) {
    case NV097_SET_TEXGEN_S_DISABLE:
        texgen = NV_PGRAPH_CSV1_A_T0_S_DISABLE; break;
    case NV097_SET_TEXGEN_S_EYE_LINEAR:
        texgen = NV_PGRAPH_CSV1_A_T0_S_EYE_LINEAR; break;
    case NV097_SET_TEXGEN_S_OBJECT_LINEAR:
        texgen = NV_PGRAPH_CSV1_A_T0_S_OBJECT_LINEAR; break;
    case NV097_SET_TEXGEN_S_SPHERE_MAP:
        KELVIN_ASSERT(channel < 2);
        texgen = NV_PGRAPH_CSV1_A_T0_S_SPHERE_MAP; break;
    case NV097_SET_TEXGEN_S_REFLECTION_MAP:
        KELVIN_ASSERT(channel < 3);
        texgen = NV_PGRAPH_CSV1_A_T0_S_REFLECTION_MAP; break;
    case NV097_SET_TEXGEN_S_NORMAL_MAP:
        KELVIN_ASSERT(channel < 3);
        texgen = NV_PGRAPH_CSV1_A_T0_S_NORMAL_MAP; break;
    default:
        KELVIN_ASSERT(!"Invalid kelvin map texgen parameter");
        break;
    }
    return texgen;
}

static void kelvin_SET_TEXGEN_S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXGEN_S) / 16;
    unsigned int reg = (slot < 2) ? NV_PGRAPH_CSV1_A
                                  : NV_PGRAPH_CSV1_B;
    unsigned int mask = (slot % 2) ? NV_PGRAPH_CSV1_A_T1_S
                                   : NV_PGRAPH_CSV1_A_T0_S;
    PG_SET_MASK(reg, mask, kelvin_map_texgen(parameter, 0));
}

static void kelvin_SET_TEXGEN_T(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXGEN_T) / 16;
    unsigned int reg = (slot < 2) ? NV_PGRAPH_CSV1_A
                                  : NV_PGRAPH_CSV1_B;
    unsigned int mask = (slot % 2) ? NV_PGRAPH_CSV1_A_T1_T
                                   : NV_PGRAPH_CSV1_A_T0_T;
    PG_SET_MASK(reg, mask, kelvin_map_texgen(parameter, 1));
}

static void kelvin_SET_TEXGEN_R(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXGEN_R) / 16;
    unsigned int reg = (slot < 2) ? NV_PGRAPH_CSV1_A
                                  : NV_PGRAPH_CSV1_B;
    unsigned int mask = (slot % 2) ? NV_PGRAPH_CSV1_A_T1_R
                                   : NV_PGRAPH_CSV1_A_T0_R;
    PG_SET_MASK(reg, mask, kelvin_map_texgen(parameter, 2));
}

static void kelvin_SET_TEXGEN_Q(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXGEN_Q) / 16;
    unsigned int reg = (slot < 2) ? NV_PGRAPH_CSV1_A
                                  : NV_PGRAPH_CSV1_B;
    unsigned int mask = (slot % 2) ? NV_PGRAPH_CSV1_A_T1_Q
                                   : NV_PGRAPH_CSV1_A_T0_Q;
    PG_SET_MASK(reg, mask, kelvin_map_texgen(parameter, 3));
}

static void kelvin_SET_TEXTURE_MATRIX_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_MATRIX_ENABLE) / 4;
    pg->texture_matrix_enable[slot] = parameter;
}

static void kelvin_SET_POINT_SIZE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    if (parameter > NV097_SET_POINT_SIZE_V_MAX) {
        return;
    }

    pgraph_reg_w(pg, NV_PGRAPH_POINTSIZE, parameter);
}

static void kelvin_SET_PROJECTION_MATRIX(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_PROJECTION_MATRIX) / 4;
    // pg->projection_matrix[slot] = kelvin_u2f(parameter);
    unsigned int row = NV_IGRAPH_XF_XFCTX_PMAT0 + slot/4;
    pg->vsh_constants[row][slot%4] = parameter;
}

static void kelvin_SET_MODEL_VIEW_MATRIX(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_MODEL_VIEW_MATRIX) / 4;
    unsigned int matnum = slot / 16;
    unsigned int entry = slot % 16;
    unsigned int row = NV_IGRAPH_XF_XFCTX_MMAT0 + matnum*8 + entry/4;
    pg->vsh_constants[row][entry % 4] = parameter;
}

static void kelvin_SET_INVERSE_MODEL_VIEW_MATRIX(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_INVERSE_MODEL_VIEW_MATRIX) / 4;
    unsigned int matnum = slot / 16;
    unsigned int entry = slot % 16;
    unsigned int row = NV_IGRAPH_XF_XFCTX_IMMAT0 + matnum*8 + entry/4;
    pg->vsh_constants[row][entry % 4] = parameter;
}

static void kelvin_SET_COMPOSITE_MATRIX(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COMPOSITE_MATRIX) / 4;
    unsigned int row = NV_IGRAPH_XF_XFCTX_CMAT0 + slot/4;
    pg->vsh_constants[row][slot%4] = parameter;
}

static void kelvin_SET_TEXTURE_MATRIX(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_MATRIX) / 4;
    unsigned int tex = slot / 16;
    unsigned int entry = slot % 16;
    unsigned int row = NV_IGRAPH_XF_XFCTX_T0MAT + tex*8 + entry/4;
    pg->vsh_constants[row][entry%4] = parameter;
}

static void kelvin_SET_FOG_PARAMS(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_FOG_PARAMS) / 4;
    if (slot < 2) {
        pgraph_reg_w(pg, NV_PGRAPH_FOGPARAM0 + slot*4, parameter);
    } else {
        /* FIXME: No idea where slot = 2 is */
    }

    pg->ltctxa[NV_IGRAPH_XF_LTCTXA_FOG_K][slot] = parameter;
}

/* Handles NV097_SET_TEXGEN_PLANE_S,T,R,Q */
static void kelvin_SET_TEXGEN_PLANE_S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXGEN_PLANE_S) / 4;
    unsigned int tex = slot / 16;
    unsigned int entry = slot % 16;
    unsigned int row = NV_IGRAPH_XF_XFCTX_TG0MAT + tex*8 + entry/4;
    pg->vsh_constants[row][entry%4] = parameter;
}

static void kelvin_SET_TEXGEN_VIEW_MODEL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_TEXGEN_REF,
             parameter);
}

static void kelvin_SET_FOG_PLANE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_FOG_PLANE) / 4;
    pg->vsh_constants[NV_IGRAPH_XF_XFCTX_FOG][slot] = parameter;
}

struct CurveCoefficients {
  float a;
  float b;
  float c;
};

static const struct CurveCoefficients curve_coefficients[] = {
  {1.000108475163, -9.838607076280, 54.829089549713},
  {1.199164441703, -3.292603784852, 7.799987995214},
  {8.653441252033, 29.189473787191, 43.586027561823},
  {-531.307758450301, 117.398468683934, 113.155490738338},
  {-4.662713151292, 1.221108944572, 1.217360986939},
  {-124.435242105211, 35.401219563514, 35.408114377045},
  {10672560.259502287954, 21565843.555823743343, 10894794.336297152564},
  {-51973801.463933646679, -104199997.554352939129, -52225454.356278456748},
  {972270.324080004124, 2025882.096547174733, 1054898.052467488218},
};

static const float kCoefficient0StepPoints[] = {
  -0.022553957999, // power = 1.25
  -0.421539008617, // power = 4.00
  -0.678715527058, // power = 9.00
  -0.838916420937, // power = 20.00
  -0.961754500866, // power = 90.00
  -0.990773200989, // power = 375.00
  -0.994858562946, // power = 650.00
  -0.996561050415, // power = 1000.00
  -0.999547004700, // power = 1250.00
};

static float reconstruct_quadratic(float c0, const struct CurveCoefficients *coefficients) {
  return coefficients->a + coefficients->b * c0 + coefficients->c * c0 * c0;
}

static float reconstruct_saturation_growth_rate(float c0, const struct CurveCoefficients *coefficients) {
  return (coefficients->a * c0) / (coefficients->b + coefficients->c * c0);
}

static float (* const reconstruct_func_map[])(float, const struct CurveCoefficients *) = {
  reconstruct_quadratic, // 1.0..1.25 max error 0.01 %
  reconstruct_quadratic, // 1.25..4.0 max error 2.2 %
  reconstruct_quadratic, // 4.0..9.0 max error 2.3 %
  reconstruct_saturation_growth_rate, // 9.0..20.0 max error 1.4 %
  reconstruct_saturation_growth_rate, // 20.0..90.0 max error 2.1 %
  reconstruct_saturation_growth_rate, // 90.0..375.0 max error 2.8%
  reconstruct_quadratic, // 375..650 max error 1.0 %
  reconstruct_quadratic, // 650..1000 max error 1.7%
  reconstruct_quadratic, // 1000..1250 max error 1.0%
};

static float reconstruct_specular_power(const float *params) {
  // See https://github.com/dracc/xgu/blob/db3172d8c983629f0dc971092981846da22438ae/xgux.h#L279

  // Values < 1.0 will result in a positive c1 and (c2 - c0 * 2) will be very
  // close to the original value.
  if (params[1] > 0.0f && params[2] < 1.0f) {
    return params[2] - (params[0] * 2.0f);
  }

  float c0 = params[0];
  float c3 = params[3];
  // FIXME: This handling is not correct, but is distinct without crashing.
  // It does not appear possible for a DirectX-generated value to be positive,
  // so while this differs from hardware behavior, it may be irrelevant in
  // practice.
  if (c0 > 0.0f || c3 > 0.0f) {
    return 0.0001f;
  }

  float reconstructed_power = 0.f;
  for (uint32_t i = 0; i < sizeof(kCoefficient0StepPoints) / sizeof(kCoefficient0StepPoints[0]); ++i) {
    if (c0 > kCoefficient0StepPoints[i]) {
      reconstructed_power = reconstruct_func_map[i](c0, &curve_coefficients[i]);
      break;
    }
  }

  float reconstructed_half_power = 0.f;
  for (uint32_t i = 0; i < sizeof(kCoefficient0StepPoints) / sizeof(kCoefficient0StepPoints[0]); ++i) {
    if (c3 > kCoefficient0StepPoints[i]) {
      reconstructed_half_power = reconstruct_func_map[i](c3, &curve_coefficients[i]);
      break;
    }
  }

  // The range can be extended beyond 1250 by using the half power params. This
  // will only work for DirectX generated values, arbitrary params could
  // erroneously trigger this.
  //
  // There are some very low power (~1) values that have inverted powers, but
  // they are easily identified by comparatively high c0 parameters.
  if (reconstructed_power == 0.f || (reconstructed_half_power > reconstructed_power && c0 < -0.1f)) {
    return reconstructed_half_power * 2.f;
  }

  return reconstructed_power;
}

static void kelvin_SET_SPECULAR_PARAMS(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_SPECULAR_PARAMS) / 4;
    pg->specular_params[slot] = kelvin_u2f(parameter);
    if (slot == 5) {
        pg->specular_power = reconstruct_specular_power(pg->specular_params);
    }
}

static void kelvin_SET_SCENE_AMBIENT_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_SCENE_AMBIENT_COLOR) / 4;
    // ??
    pg->ltctxa[NV_IGRAPH_XF_LTCTXA_FR_AMB][slot] = parameter;
}

static void kelvin_SET_VIEWPORT_OFFSET(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VIEWPORT_OFFSET) / 4;
    pg->vsh_constants[NV_IGRAPH_XF_XFCTX_VPOFF][slot] = parameter;
}

static void kelvin_SET_POINT_PARAMS(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_POINT_PARAMS) / 4;
    pg->point_params[slot] = kelvin_u2f(parameter); /* FIXME: Where? */
}

static void kelvin_SET_EYE_POSITION(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_EYE_POSITION) / 4;
    pg->vsh_constants[NV_IGRAPH_XF_XFCTX_EYEP][slot] = parameter;
}

static void kelvin_SET_COMBINER_FACTOR0(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COMBINER_FACTOR0) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_COMBINEFACTOR0 + slot*4, parameter);
}

static void kelvin_SET_COMBINER_FACTOR1(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COMBINER_FACTOR1) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_COMBINEFACTOR1 + slot*4, parameter);
}

static void kelvin_SET_COMBINER_ALPHA_OCW(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COMBINER_ALPHA_OCW) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_COMBINEALPHAO0 + slot*4, parameter);
}

static void kelvin_SET_COMBINER_COLOR_ICW(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COMBINER_COLOR_ICW) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_COMBINECOLORI0 + slot*4, parameter);
}

static void kelvin_SET_COLOR_KEY_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COLOR_KEY_COLOR) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_COLORKEYCOLOR0 + slot * 4, parameter);
}

static void kelvin_SET_VIEWPORT_SCALE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VIEWPORT_SCALE) / 4;
    pg->vsh_constants[NV_IGRAPH_XF_XFCTX_VPSCL][slot] = parameter;
}

static void kelvin_SET_TRANSFORM_PROGRAM(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TRANSFORM_PROGRAM) / 4;

    int program_load = PG_GET_MASK(NV_PGRAPH_CHEOPS_OFFSET,
                                NV_PGRAPH_CHEOPS_OFFSET_PROG_LD_PTR);

    KELVIN_ASSERT(program_load < NV2A_MAX_TRANSFORM_PROGRAM_LENGTH);
    pg->program_data[program_load][slot%4] = parameter;
    pg->program_writes++;

    if (slot % 4 == 3) {
        PG_SET_MASK(NV_PGRAPH_CHEOPS_OFFSET,
                 NV_PGRAPH_CHEOPS_OFFSET_PROG_LD_PTR, program_load+1);
    }
}

static void kelvin_SET_TRANSFORM_CONSTANT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TRANSFORM_CONSTANT) / 4;
    int const_load = PG_GET_MASK(NV_PGRAPH_CHEOPS_OFFSET,
                              NV_PGRAPH_CHEOPS_OFFSET_CONST_LD_PTR);

    KELVIN_ASSERT(const_load < NV2A_VERTEXSHADER_CONSTANTS);
    // VertexShaderConstant *constant = &pg->constants[const_load];
    pg->vsh_constants_dirty[const_load] |=
        (parameter != pg->vsh_constants[const_load][slot%4]);
    pg->vsh_constants[const_load][slot%4] = parameter;

    if (slot % 4 == 3) {
        PG_SET_MASK(NV_PGRAPH_CHEOPS_OFFSET,
                 NV_PGRAPH_CHEOPS_OFFSET_CONST_LD_PTR, const_load+1);
    }
}

static void kelvin_SET_VERTEX3F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX3F) / 4;
    VertexAttribute *attribute =
        &pg->vertex_attributes[NV2A_VERTEX_ATTR_POSITION];
    pgraph_allocate_inline_buffer_vertices(pg, NV2A_VERTEX_ATTR_POSITION);
    attribute->inline_value[slot] = kelvin_u2f(parameter);
    attribute->inline_value[3] = 1.0f;
    if (slot == 2) {
        pgraph_finish_inline_buffer_vertex(pg);
    }
}

/* Handles NV097_SET_BACK_LIGHT_* */
static void kelvin_SET_BACK_LIGHT_AMBIENT_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_BACK_LIGHT_AMBIENT_COLOR) / 4;
    unsigned int part = NV097_SET_BACK_LIGHT_AMBIENT_COLOR / 4 + slot % 16;
    unsigned int m;
    slot /= 16; /* [Light index] */
    if (slot >= 8) {
        return;
    }
    m = part * 4;
    if (m >= NV097_SET_BACK_LIGHT_AMBIENT_COLOR &&
        m <= NV097_SET_BACK_LIGHT_AMBIENT_COLOR + 8) {
        part -= NV097_SET_BACK_LIGHT_AMBIENT_COLOR / 4;
        pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_BAMB + slot*6][part] = parameter;
    } else if (m >= NV097_SET_BACK_LIGHT_DIFFUSE_COLOR &&
               m <= NV097_SET_BACK_LIGHT_DIFFUSE_COLOR + 8) {
        part -= NV097_SET_BACK_LIGHT_DIFFUSE_COLOR / 4;
        pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_BDIF + slot*6][part] = parameter;
    } else if (m >= NV097_SET_BACK_LIGHT_SPECULAR_COLOR &&
               m <= NV097_SET_BACK_LIGHT_SPECULAR_COLOR + 8) {
        part -= NV097_SET_BACK_LIGHT_SPECULAR_COLOR / 4;
        pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_BSPC + slot*6][part] = parameter;
    }
}

/* Handles all the light source props except for NV097_SET_BACK_LIGHT_* */
static void kelvin_SET_LIGHT_AMBIENT_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_LIGHT_AMBIENT_COLOR) / 4;
    unsigned int part = NV097_SET_LIGHT_AMBIENT_COLOR / 4 + slot % 32;
    unsigned int m;
    slot /= 32; /* [Light index] */
    if (slot >= 8) {
        return;
    }
    m = part * 4;
#define IN_RANGE(base, span) (m >= (base) && m <= (base) + (span))
    if (IN_RANGE(NV097_SET_LIGHT_AMBIENT_COLOR, 8)) {
        part -= NV097_SET_LIGHT_AMBIENT_COLOR / 4;
        pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_AMB + slot*6][part] = parameter;
    } else if (IN_RANGE(NV097_SET_LIGHT_DIFFUSE_COLOR, 8)) {
        part -= NV097_SET_LIGHT_DIFFUSE_COLOR / 4;
        pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_DIF + slot*6][part] = parameter;
    } else if (IN_RANGE(NV097_SET_LIGHT_SPECULAR_COLOR, 8)) {
        part -= NV097_SET_LIGHT_SPECULAR_COLOR / 4;
        pg->ltctxb[NV_IGRAPH_XF_LTCTXB_L0_SPC + slot*6][part] = parameter;
    } else if (m == NV097_SET_LIGHT_LOCAL_RANGE) {
        pg->ltc1[NV_IGRAPH_XF_LTC1_r0 + slot][0] = parameter;
    } else if (IN_RANGE(NV097_SET_LIGHT_INFINITE_HALF_VECTOR, 8)) {
        part -= NV097_SET_LIGHT_INFINITE_HALF_VECTOR / 4;
        pg->light_infinite_half_vector[slot][part] = kelvin_u2f(parameter);
    } else if (IN_RANGE(NV097_SET_LIGHT_INFINITE_DIRECTION, 8)) {
        part -= NV097_SET_LIGHT_INFINITE_DIRECTION / 4;
        pg->light_infinite_direction[slot][part] = kelvin_u2f(parameter);
    } else if (IN_RANGE(NV097_SET_LIGHT_SPOT_FALLOFF, 8)) {
        part -= NV097_SET_LIGHT_SPOT_FALLOFF / 4;
        pg->ltctxa[NV_IGRAPH_XF_LTCTXA_L0_K + slot*2][part] = parameter;
    } else if (IN_RANGE(NV097_SET_LIGHT_SPOT_DIRECTION, 12)) {
        part -= NV097_SET_LIGHT_SPOT_DIRECTION / 4;
        pg->ltctxa[NV_IGRAPH_XF_LTCTXA_L0_SPT + slot*2][part] = parameter;
    } else if (IN_RANGE(NV097_SET_LIGHT_LOCAL_POSITION, 8)) {
        part -= NV097_SET_LIGHT_LOCAL_POSITION / 4;
        pg->light_local_position[slot][part] = kelvin_u2f(parameter);
    } else if (IN_RANGE(NV097_SET_LIGHT_LOCAL_ATTENUATION, 8)) {
        part -= NV097_SET_LIGHT_LOCAL_ATTENUATION / 4;
        pg->light_local_attenuation[slot][part] = kelvin_u2f(parameter);
    }
#undef IN_RANGE
}

static void kelvin_SET_VERTEX4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX4F) / 4;
    VertexAttribute *attribute =
        &pg->vertex_attributes[NV2A_VERTEX_ATTR_POSITION];
    pgraph_allocate_inline_buffer_vertices(pg, NV2A_VERTEX_ATTR_POSITION);
    attribute->inline_value[slot] = kelvin_u2f(parameter);
    if (slot == 3) {
        pgraph_finish_inline_buffer_vertex(pg);
    }
}

static void kelvin_SET_FOG_COORD(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    VertexAttribute *attribute = &pg->vertex_attributes[NV2A_VERTEX_ATTR_FOG];
    pgraph_allocate_inline_buffer_vertices(pg, NV2A_VERTEX_ATTR_FOG);
    attribute->inline_value[0] = kelvin_u2f(parameter);
    attribute->inline_value[1] = attribute->inline_value[0];
    attribute->inline_value[2] = attribute->inline_value[0];
    attribute->inline_value[3] = attribute->inline_value[0];
}

static void kelvin_SET_WEIGHT1F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    VertexAttribute *attribute = &pg->vertex_attributes[NV2A_VERTEX_ATTR_WEIGHT];
    pgraph_allocate_inline_buffer_vertices(pg, NV2A_VERTEX_ATTR_WEIGHT);
    attribute->inline_value[0] = kelvin_u2f(parameter);
    attribute->inline_value[1] = 0.f;
    attribute->inline_value[2] = 0.f;
    attribute->inline_value[3] = 1.f;
}

static void kelvin_SET_NORMAL3S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_NORMAL3S) / 4;
    unsigned int part = slot % 2;
    VertexAttribute *attribute =
        &pg->vertex_attributes[NV2A_VERTEX_ATTR_NORMAL];
    pgraph_allocate_inline_buffer_vertices(pg, NV2A_VERTEX_ATTR_NORMAL);
    int16_t val = parameter & 0xFFFF;
    attribute->inline_value[part * 2 + 0] = MAX(-1.0f, (float)val / 32767.0f);
    val = parameter >> 16;
    attribute->inline_value[part * 2 + 1] = MAX(-1.0f, (float)val / 32767.0f);
}

#define SET_VERTEX_ATTRIBUTE_4S(command, attr_index)                     \
    do {                                                                   \
        int slot = (method - (command)) / 4;                               \
        unsigned int part = slot % 2;                                      \
        VertexAttribute *attribute = &pg->vertex_attributes[(attr_index)]; \
        pgraph_allocate_inline_buffer_vertices(pg, (attr_index));          \
        attribute->inline_value[part * 2 + 0] =                            \
            (float)(int16_t)(parameter & 0xFFFF);                          \
        attribute->inline_value[part * 2 + 1] =                            \
            (float)(int16_t)(parameter >> 16);                             \
    } while (0)

static void kelvin_SET_TEXCOORD0_4S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_4S(NV097_SET_TEXCOORD0_4S, NV2A_VERTEX_ATTR_TEXTURE0);
}

static void kelvin_SET_TEXCOORD1_4S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_4S(NV097_SET_TEXCOORD1_4S, NV2A_VERTEX_ATTR_TEXTURE1);
}

static void kelvin_SET_TEXCOORD2_4S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_4S(NV097_SET_TEXCOORD2_4S, NV2A_VERTEX_ATTR_TEXTURE2);
}

static void kelvin_SET_TEXCOORD3_4S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_4S(NV097_SET_TEXCOORD3_4S, NV2A_VERTEX_ATTR_TEXTURE3);
}

#undef SET_VERTEX_ATTRIBUTE_4S

#define SET_VERTEX_ATRIBUTE_TEX_2S(attr_index)                             \
    do {                                                                   \
        VertexAttribute *attribute = &pg->vertex_attributes[(attr_index)]; \
        pgraph_allocate_inline_buffer_vertices(pg, (attr_index));          \
        attribute->inline_value[0] = (float)(int16_t)(parameter & 0xFFFF); \
        attribute->inline_value[1] = (float)(int16_t)(parameter >> 16);    \
        attribute->inline_value[2] = 0.0f;                                 \
        attribute->inline_value[3] = 1.0f;                                 \
    } while (0)

static void kelvin_SET_TEXCOORD0_2S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2S(NV2A_VERTEX_ATTR_TEXTURE0);
}

static void kelvin_SET_TEXCOORD1_2S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2S(NV2A_VERTEX_ATTR_TEXTURE1);
}

static void kelvin_SET_TEXCOORD2_2S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2S(NV2A_VERTEX_ATTR_TEXTURE2);
}

static void kelvin_SET_TEXCOORD3_2S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2S(NV2A_VERTEX_ATTR_TEXTURE3);
}

#undef SET_VERTEX_ATRIBUTE_TEX_2S

#define SET_VERTEX_COLOR_3F(command, attr_index)                           \
    do {                                                                   \
        int slot = (method - (command)) / 4;                               \
        VertexAttribute *attribute = &pg->vertex_attributes[(attr_index)]; \
        pgraph_allocate_inline_buffer_vertices(pg, (attr_index));          \
        attribute->inline_value[slot] = kelvin_u2f(parameter);               \
        attribute->inline_value[3] = 1.0f;                                 \
    } while (0)

static void kelvin_SET_DIFFUSE_COLOR3F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_COLOR_3F(NV097_SET_DIFFUSE_COLOR3F, NV2A_VERTEX_ATTR_DIFFUSE);
}

static void kelvin_SET_SPECULAR_COLOR3F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_COLOR_3F(NV097_SET_SPECULAR_COLOR3F, NV2A_VERTEX_ATTR_SPECULAR);
}

#undef SET_VERTEX_COLOR_3F

#define SET_VERTEX_ATTRIBUTE_F(command, attr_index)                        \
    do {                                                                   \
        int slot = (method - (command)) / 4;                               \
        VertexAttribute *attribute = &pg->vertex_attributes[(attr_index)]; \
        pgraph_allocate_inline_buffer_vertices(pg, (attr_index));          \
        attribute->inline_value[slot] = kelvin_u2f(parameter);               \
    } while (0)

static void kelvin_SET_NORMAL3F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_NORMAL3F, NV2A_VERTEX_ATTR_NORMAL);
}

static void kelvin_SET_DIFFUSE_COLOR4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_DIFFUSE_COLOR4F, NV2A_VERTEX_ATTR_DIFFUSE);
}

static void kelvin_SET_SPECULAR_COLOR4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_SPECULAR_COLOR4F,
                           NV2A_VERTEX_ATTR_SPECULAR);
}

static void kelvin_SET_TEXCOORD0_4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_TEXCOORD0_4F, NV2A_VERTEX_ATTR_TEXTURE0);
}

static void kelvin_SET_TEXCOORD1_4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_TEXCOORD1_4F, NV2A_VERTEX_ATTR_TEXTURE1);
}

static void kelvin_SET_TEXCOORD2_4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_TEXCOORD2_4F, NV2A_VERTEX_ATTR_TEXTURE2);
}

static void kelvin_SET_TEXCOORD3_4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_TEXCOORD3_4F, NV2A_VERTEX_ATTR_TEXTURE3);
}

static void kelvin_SET_WEIGHT4F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_F(NV097_SET_WEIGHT4F, NV2A_VERTEX_ATTR_WEIGHT);
}

#undef SET_VERTEX_ATTRIBUTE_F

static void kelvin_SET_WEIGHT2F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_WEIGHT2F) / 4;
    VertexAttribute *attribute =
        &pg->vertex_attributes[NV2A_VERTEX_ATTR_WEIGHT];
    pgraph_allocate_inline_buffer_vertices(pg, NV2A_VERTEX_ATTR_WEIGHT);
    attribute->inline_value[slot] = kelvin_u2f(parameter);
    attribute->inline_value[2] = 0.0f;
    attribute->inline_value[3] = 1.0f;
}

static void kelvin_SET_WEIGHT3F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_WEIGHT3F) / 4;
    VertexAttribute *attribute =
        &pg->vertex_attributes[NV2A_VERTEX_ATTR_WEIGHT];
    pgraph_allocate_inline_buffer_vertices(pg, NV2A_VERTEX_ATTR_WEIGHT);
    attribute->inline_value[slot] = kelvin_u2f(parameter);
    attribute->inline_value[3] = 1.0f;
}

#define SET_VERTEX_ATRIBUTE_TEX_2F(command, attr_index)                    \
    do {                                                                   \
        int slot = (method - (command)) / 4;                               \
        VertexAttribute *attribute = &pg->vertex_attributes[(attr_index)]; \
        pgraph_allocate_inline_buffer_vertices(pg, (attr_index));          \
        attribute->inline_value[slot] = kelvin_u2f(parameter);               \
        attribute->inline_value[2] = 0.0f;                                 \
        attribute->inline_value[3] = 1.0f;                                 \
    } while (0)

static void kelvin_SET_TEXCOORD0_2F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2F(NV097_SET_TEXCOORD0_2F,
                               NV2A_VERTEX_ATTR_TEXTURE0);
}

static void kelvin_SET_TEXCOORD1_2F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2F(NV097_SET_TEXCOORD1_2F,
                               NV2A_VERTEX_ATTR_TEXTURE1);
}

static void kelvin_SET_TEXCOORD2_2F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2F(NV097_SET_TEXCOORD2_2F,
                               NV2A_VERTEX_ATTR_TEXTURE2);
}

static void kelvin_SET_TEXCOORD3_2F(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATRIBUTE_TEX_2F(NV097_SET_TEXCOORD3_2F,
                               NV2A_VERTEX_ATTR_TEXTURE3);
}

#undef SET_VERTEX_ATRIBUTE_TEX_2F

#define SET_VERTEX_ATTRIBUTE_4UB(command, attr_index)                       \
    do {                                                                   \
        VertexAttribute *attribute = &pg->vertex_attributes[(attr_index)]; \
        pgraph_allocate_inline_buffer_vertices(pg, (attr_index));          \
        attribute->inline_value[0] = (parameter & 0xFF) / 255.0f;          \
        attribute->inline_value[1] = ((parameter >> 8) & 0xFF) / 255.0f;   \
        attribute->inline_value[2] = ((parameter >> 16) & 0xFF) / 255.0f;  \
        attribute->inline_value[3] = ((parameter >> 24) & 0xFF) / 255.0f;  \
    } while (0)

static void kelvin_SET_DIFFUSE_COLOR4UB(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_4UB(NV097_SET_DIFFUSE_COLOR4UB,
                             NV2A_VERTEX_ATTR_DIFFUSE);
}

static void kelvin_SET_SPECULAR_COLOR4UB(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    SET_VERTEX_ATTRIBUTE_4UB(NV097_SET_SPECULAR_COLOR4UB,
                             NV2A_VERTEX_ATTR_SPECULAR);
}

#undef SET_VERTEX_ATTRIBUTE_4UB

static void kelvin_SET_VERTEX_DATA_ARRAY_FORMAT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX_DATA_ARRAY_FORMAT) / 4;
    VertexAttribute *attr = &pg->vertex_attributes[slot];
    attr->format = GET_MASK(parameter, NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE);
    attr->count = GET_MASK(parameter, NV097_SET_VERTEX_DATA_ARRAY_FORMAT_SIZE);
    attr->stride = GET_MASK(parameter,
                            NV097_SET_VERTEX_DATA_ARRAY_FORMAT_STRIDE);

    KELVIN_DPRINTF("vertex data array format=%d, count=%d, stride=%d\n",
                 attr->format, attr->count, attr->stride);

    switch (attr->format) {
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_D3D:
        attr->size = 1;
        KELVIN_ASSERT(attr->count == 4);
        break;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_OGL:
        attr->size = 1;
        break;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S1:
        attr->size = 2;
        break;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_F:
        attr->size = 4;
        break;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S32K:
        attr->size = 2;
        break;
    case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_CMP:
        /* 3 signed, normalized components packed in 32-bits. (11,11,10) */
        attr->size = 4;
        KELVIN_ASSERT(attr->count == 1);
        break;
    default:
        fprintf(stderr, "Unknown vertex type: 0x%x\n", attr->format);
        KELVIN_ASSERT(!"Unknown vertex type");
        break;
    }

    if (attr->format == NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_CMP) {
        pg->compressed_attrs |= (1 << slot);
    } else {
        pg->compressed_attrs &= ~(1 << slot);
    }
}

static void kelvin_SET_VERTEX_DATA_ARRAY_OFFSET(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX_DATA_ARRAY_OFFSET) / 4;

    pg->vertex_attributes[slot].dma_select = parameter & 0x80000000;
    pg->vertex_attributes[slot].offset = parameter & 0x7fffffff;
}

static void kelvin_SET_LOGIC_OP_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_BLEND, NV_PGRAPH_BLEND_LOGICOP_ENABLE,
             parameter);
}

static void kelvin_SET_LOGIC_OP(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_BLEND, NV_PGRAPH_BLEND_LOGICOP,
             parameter & 0xF);
}

static void kelvin_CLEAR_REPORT_VALUE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    kelvin_render_sync();
    kelvin_unsupported(KU_REPORT, "CLEAR_REPORT_VALUE", parameter);
}

static void kelvin_SET_ZPASS_PIXEL_COUNT_ENABLE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->zpass_pixel_count_enable = parameter;
}

static void kelvin_GET_REPORT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    uint8_t type = GET_MASK(parameter, NV097_GET_REPORT_TYPE);
    KELVIN_ASSERT(type == NV097_GET_REPORT_TYPE_ZPASS_PIXEL_CNT);

    kelvin_render_sync();
    kelvin_unsupported(KU_REPORT, "GET_REPORT", parameter);
}

static void kelvin_SET_EYE_DIRECTION(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_EYE_DIRECTION) / 4;
    pg->ltctxa[NV_IGRAPH_XF_LTCTXA_EYED][slot] = parameter;
}

static void kelvin_SET_BEGIN_END(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    if (parameter == NV097_SET_BEGIN_END_OP_END) {
        if (pg->primitive_mode == PRIM_TYPE_INVALID) {
            KELVIN_DPRINTF("End without Begin!\n");
            pgraph_reset_inline_buffers(pg);
            return;
        }
        kelvin_render_draw_end(pg);
        pgraph_reset_inline_buffers(pg);
        pg->primitive_mode = PRIM_TYPE_INVALID;
    } else {
        if (pg->primitive_mode != PRIM_TYPE_INVALID) {
            KELVIN_DPRINTF("Begin without End!\n");
            return;
        }
        KELVIN_ASSERT(parameter <= NV097_SET_BEGIN_END_OP_POLYGON);
        pg->primitive_mode = parameter;
        pgraph_reset_inline_buffers(pg);
        kelvin_render_draw_begin(pg);
    }
}

static void kelvin_SET_TEXTURE_OFFSET(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_OFFSET) / 64;
    pgraph_reg_w(pg, NV_PGRAPH_TEXOFFSET0 + slot * 4, parameter);
}

static void kelvin_SET_TEXTURE_FORMAT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_FORMAT) / 64;

    bool dma_select =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_CONTEXT_DMA) == 2;
    bool cubemap =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_CUBEMAP_ENABLE);
    unsigned int border_source =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_BORDER_SOURCE);
    unsigned int dimensionality =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_DIMENSIONALITY);
    unsigned int color_format =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_COLOR);
    unsigned int levels =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_MIPMAP_LEVELS);
    unsigned int log_width =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_BASE_SIZE_U);
    unsigned int log_height =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_BASE_SIZE_V);
    unsigned int log_depth =
        GET_MASK(parameter, NV097_SET_TEXTURE_FORMAT_BASE_SIZE_P);

    unsigned int reg = NV_PGRAPH_TEXFMT0 + slot * 4;
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_CONTEXT_DMA, dma_select);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_CUBEMAPENABLE, cubemap);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_BORDER_SOURCE, border_source);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_DIMENSIONALITY, dimensionality);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_COLOR, color_format);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_MIPMAP_LEVELS, levels);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_BASE_SIZE_U, log_width);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_BASE_SIZE_V, log_height);
    PG_SET_MASK(reg, NV_PGRAPH_TEXFMT0_BASE_SIZE_P, log_depth);
}

static void kelvin_SET_TEXTURE_CONTROL0(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_CONTROL0) / 64;
    pgraph_reg_w(pg, NV_PGRAPH_TEXCTL0_0 + slot*4, parameter);
}

static void kelvin_SET_TEXTURE_CONTROL1(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_CONTROL1) / 64;
    pgraph_reg_w(pg, NV_PGRAPH_TEXCTL1_0 + slot*4, parameter);
}

static void kelvin_SET_TEXTURE_FILTER(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_FILTER) / 64;
    pgraph_reg_w(pg, NV_PGRAPH_TEXFILTER0 + slot * 4, parameter);
}

static void kelvin_SET_TEXTURE_IMAGE_RECT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_IMAGE_RECT) / 64;
    pgraph_reg_w(pg, NV_PGRAPH_TEXIMAGERECT0 + slot * 4, parameter);
}

static void kelvin_SET_TEXTURE_PALETTE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_PALETTE) / 64;

    bool dma_select =
        GET_MASK(parameter, NV097_SET_TEXTURE_PALETTE_CONTEXT_DMA) == 1;
    unsigned int length =
        GET_MASK(parameter, NV097_SET_TEXTURE_PALETTE_LENGTH);
    unsigned int offset =
        GET_MASK(parameter, NV097_SET_TEXTURE_PALETTE_OFFSET);

    unsigned int reg = NV_PGRAPH_TEXPALETTE0 + slot * 4;
    PG_SET_MASK(reg, NV_PGRAPH_TEXPALETTE0_CONTEXT_DMA, dma_select);
    PG_SET_MASK(reg, NV_PGRAPH_TEXPALETTE0_LENGTH, length);
    PG_SET_MASK(reg, NV_PGRAPH_TEXPALETTE0_OFFSET, offset);
}

static void kelvin_SET_TEXTURE_BORDER_COLOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_BORDER_COLOR) / 64;
    pgraph_reg_w(pg, NV_PGRAPH_BORDERCOLOR0 + slot * 4, parameter);
}

static void kelvin_SET_TEXTURE_SET_BUMP_ENV_MAT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_SET_BUMP_ENV_MAT) / 4;
    if (slot < 16) {
        /* discard */
        return;
    }

    slot -= 16;
    const int swizzle[4] = { NV_PGRAPH_BUMPMAT00, NV_PGRAPH_BUMPMAT01,
                             NV_PGRAPH_BUMPMAT11, NV_PGRAPH_BUMPMAT10 };
    pgraph_reg_w(pg, swizzle[slot % 4] + slot / 4, parameter);
}

static void kelvin_SET_TEXTURE_SET_BUMP_ENV_SCALE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_SET_BUMP_ENV_SCALE) / 64;
    if (slot == 0) {
        /* discard */
        return;
    }

    slot--;
    pgraph_reg_w(pg, NV_PGRAPH_BUMPSCALE1 + slot * 4, parameter);
}

static void kelvin_SET_TEXTURE_SET_BUMP_ENV_OFFSET(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TEXTURE_SET_BUMP_ENV_OFFSET) / 64;
    if (slot == 0) {
        /* discard */
        return;
    }

    slot--;
    pgraph_reg_w(pg, NV_PGRAPH_BUMPOFFSET1 + slot * 4, parameter);
}

static void pgraph_expand_draw_arrays(KelvinState *pg)
{
    uint32_t start = pg->draw_arrays_start[pg->draw_arrays_length - 1];
    uint32_t count = pg->draw_arrays_count[pg->draw_arrays_length - 1];

    /* Render any previously squashed DRAW_ARRAYS calls. This case would be
     * triggered if a set of BEGIN+DA+END triplets is followed by the
     * BEGIN+DA+ARRAY_ELEMENT+... chain that caused this expansion. */
    if (pg->draw_arrays_length > 1) {
        kelvin_render_flush_draw(pg);
        pgraph_reset_inline_buffers(pg);
    }
    KELVIN_ASSERT((pg->inline_elements_length + count) < NV2A_MAX_BATCH_LENGTH);
    for (unsigned int i = 0; i < count; i++) {
        pg->inline_elements[pg->inline_elements_length++] = start + i;
    }

    pgraph_reset_draw_arrays(pg);
}

static void pgraph_check_within_begin_end_block(KelvinState *pg)
{
    if (pg->primitive_mode == PRIM_TYPE_INVALID) {
        KELVIN_DPRINTF("Vertex data being sent outside of begin/end block!\n");
    }
}

static void kelvin_ARRAY_ELEMENT16(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_check_within_begin_end_block(pg);

    if (pg->draw_arrays_length) {
        pgraph_expand_draw_arrays(pg);
    }

    KELVIN_ASSERT(pg->inline_elements_length < NV2A_MAX_BATCH_LENGTH);
    pg->inline_elements[pg->inline_elements_length++] = parameter & 0xFFFF;
    pg->inline_elements[pg->inline_elements_length++] = parameter >> 16;
}

static void kelvin_ARRAY_ELEMENT32(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_check_within_begin_end_block(pg);

    if (pg->draw_arrays_length) {
        pgraph_expand_draw_arrays(pg);
    }

    KELVIN_ASSERT(pg->inline_elements_length < NV2A_MAX_BATCH_LENGTH);
    pg->inline_elements[pg->inline_elements_length++] = parameter;
}

static void kelvin_DRAW_ARRAYS(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_check_within_begin_end_block(pg);

    int32_t start = GET_MASK(parameter, NV097_DRAW_ARRAYS_START_INDEX);
    int32_t count = GET_MASK(parameter, NV097_DRAW_ARRAYS_COUNT) + 1;

    if (pg->inline_elements_length) {
        /* FIXME: HW throws an exception if the start index is > 0xFFFF. This
         * would prevent this assert from firing for any reasonable choice of
         * NV2A_MAX_BATCH_LENGTH (which must be larger to accommodate
         * NV097_INLINE_ARRAY anyway)
         */
        KELVIN_ASSERT((pg->inline_elements_length + count) < NV2A_MAX_BATCH_LENGTH);
        KELVIN_ASSERT(!pg->draw_arrays_prevent_connect);

        for (unsigned int i = 0; i < count; i++) {
            pg->inline_elements[pg->inline_elements_length++] = start + i;
        }
        return;
    }

    pg->draw_arrays_min_start = MIN(pg->draw_arrays_min_start, start);
    pg->draw_arrays_max_count = MAX(pg->draw_arrays_max_count, start + count);

    KELVIN_ASSERT(pg->draw_arrays_length < ARRAY_SIZE(pg->draw_arrays_start));

    /* Attempt to connect contiguous primitives */
    if (!pg->draw_arrays_prevent_connect && pg->draw_arrays_length > 0) {
        unsigned int last_start =
            pg->draw_arrays_start[pg->draw_arrays_length - 1];
        int32_t *last_count =
            &pg->draw_arrays_count[pg->draw_arrays_length - 1];
        if (start == (last_start + *last_count)) {
            *last_count += count;
            return;
        }
    }

    pg->draw_arrays_start[pg->draw_arrays_length] = start;
    pg->draw_arrays_count[pg->draw_arrays_length] = count;
    pg->draw_arrays_length++;
    pg->draw_arrays_prevent_connect = false;
}

static void kelvin_INLINE_ARRAY(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_check_within_begin_end_block(pg);
    KELVIN_ASSERT(pg->inline_array_length < NV2A_MAX_BATCH_LENGTH);
    pg->inline_array[pg->inline_array_length++] = parameter;
}

static void kelvin_SET_EYE_VECTOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_EYE_VECTOR) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_EYEVEC0 + slot * 4, parameter);
}

static void kelvin_SET_VERTEX_DATA2F_M(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX_DATA2F_M) / 4;
    unsigned int part = slot % 2;
    slot /= 2;
    VertexAttribute *attribute = &pg->vertex_attributes[slot];
    pgraph_allocate_inline_buffer_vertices(pg, slot);
    attribute->inline_value[part] = kelvin_u2f(parameter);
    /* FIXME: Should these really be set to 0.0 and 1.0 ? Conditions? */
    attribute->inline_value[2] = 0.0;
    attribute->inline_value[3] = 1.0;
    if ((slot == 0) && (part == 1)) {
        pgraph_finish_inline_buffer_vertex(pg);
    }
}

static void kelvin_SET_VERTEX_DATA4F_M(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX_DATA4F_M) / 4;
    unsigned int part = slot % 4;
    slot /= 4;
    VertexAttribute *attribute = &pg->vertex_attributes[slot];
    pgraph_allocate_inline_buffer_vertices(pg, slot);
    attribute->inline_value[part] = kelvin_u2f(parameter);
    if ((slot == 0) && (part == 3)) {
        pgraph_finish_inline_buffer_vertex(pg);
    }
}

static void kelvin_SET_VERTEX_DATA2S(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX_DATA2S) / 4;
    VertexAttribute *attribute = &pg->vertex_attributes[slot];
    pgraph_allocate_inline_buffer_vertices(pg, slot);
    attribute->inline_value[0] = (float)(int16_t)(parameter & 0xFFFF);
    attribute->inline_value[1] = (float)(int16_t)(parameter >> 16);
    attribute->inline_value[2] = 0.0;
    attribute->inline_value[3] = 1.0;
    if (slot == 0) {
        pgraph_finish_inline_buffer_vertex(pg);
    }
}

static void kelvin_SET_VERTEX_DATA4UB(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX_DATA4UB) / 4;
    VertexAttribute *attribute = &pg->vertex_attributes[slot];
    pgraph_allocate_inline_buffer_vertices(pg, slot);
    attribute->inline_value[0] = (parameter & 0xFF) / 255.0;
    attribute->inline_value[1] = ((parameter >> 8) & 0xFF) / 255.0;
    attribute->inline_value[2] = ((parameter >> 16) & 0xFF) / 255.0;
    attribute->inline_value[3] = ((parameter >> 24) & 0xFF) / 255.0;
    if (slot == 0) {
        pgraph_finish_inline_buffer_vertex(pg);
    }
}

static void kelvin_SET_VERTEX_DATA4S_M(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_VERTEX_DATA4S_M) / 4;
    unsigned int part = slot % 2;
    slot /= 2;
    VertexAttribute *attribute = &pg->vertex_attributes[slot];
    pgraph_allocate_inline_buffer_vertices(pg, slot);

    attribute->inline_value[part * 2 + 0] = (float)(int16_t)(parameter & 0xFFFF);
    attribute->inline_value[part * 2 + 1] = (float)(int16_t)(parameter >> 16);
    if ((slot == 0) && (part == 1)) {
        pgraph_finish_inline_buffer_vertex(pg);
    }
}

static void kelvin_SET_SEMAPHORE_OFFSET(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_SEMAPHOREOFFSET, parameter);
}

static void kelvin_BACK_END_WRITE_SEMAPHORE_RELEASE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    kelvin_render_semaphore_release(pg, parameter);
}

static void kelvin_SET_ZMIN_MAX_CONTROL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    switch (GET_MASK(parameter, NV097_SET_ZMIN_MAX_CONTROL_ZCLAMP_EN)) {
    case NV097_SET_ZMIN_MAX_CONTROL_ZCLAMP_EN_CULL:
        PG_SET_MASK(NV_PGRAPH_ZCOMPRESSOCCLUDE,
                 NV_PGRAPH_ZCOMPRESSOCCLUDE_ZCLAMP_EN,
                 NV_PGRAPH_ZCOMPRESSOCCLUDE_ZCLAMP_EN_CULL);
        break;
    case NV097_SET_ZMIN_MAX_CONTROL_ZCLAMP_EN_CLAMP:
        PG_SET_MASK(NV_PGRAPH_ZCOMPRESSOCCLUDE,
                 NV_PGRAPH_ZCOMPRESSOCCLUDE_ZCLAMP_EN,
                 NV_PGRAPH_ZCOMPRESSOCCLUDE_ZCLAMP_EN_CLAMP);
        break;
    default:
        /* FIXME: Should raise NV_PGRAPH_NSOURCE_DATA_ERROR_PENDING */
        KELVIN_ASSERT(!"Invalid zclamp value");
        break;
    }
}

static void kelvin_SET_ANTI_ALIASING_CONTROL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_ANTIALIASING, NV_PGRAPH_ANTIALIASING_ENABLE,
             GET_MASK(parameter, NV097_SET_ANTI_ALIASING_CONTROL_ENABLE));
    // FIXME: Handle the remaining bits (observed values 0xFFFF0000, 0xFFFF0001)
}

static void kelvin_SET_ZSTENCIL_CLEAR_VALUE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_ZSTENCILCLEARVALUE, parameter);
}

static void kelvin_SET_COLOR_CLEAR_VALUE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_COLORCLEARVALUE, parameter);
}

static void kelvin_CLEAR_SURFACE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    kelvin_render_clear_surface(pg, parameter);
}

static void kelvin_SET_CLEAR_RECT_HORIZONTAL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_CLEARRECTX, parameter);
}

static void kelvin_SET_CLEAR_RECT_VERTICAL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_CLEARRECTY, parameter);
}

static void kelvin_SET_SPECULAR_FOG_FACTOR(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_SPECULAR_FOG_FACTOR) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_SPECFOGFACTOR0 + slot*4, parameter);
}

static void kelvin_SET_SPECULAR_PARAMS_BACK(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_SPECULAR_PARAMS_BACK) / 4;
    pg->specular_params_back[slot] = kelvin_u2f(parameter);
    if (slot == 5) {
        pg->specular_power_back = reconstruct_specular_power(pg->specular_params_back);
    }
}

static void kelvin_SET_SHADER_CLIP_PLANE_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_SHADERCLIPMODE, parameter);
}

static void kelvin_SET_COMBINER_COLOR_OCW(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_COMBINER_COLOR_OCW) / 4;
    pgraph_reg_w(pg, NV_PGRAPH_COMBINECOLORO0 + slot*4, parameter);
}

static void kelvin_SET_COMBINER_CONTROL(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_COMBINECTL, parameter);
}

static void kelvin_SET_SHADOW_ZSLOPE_THRESHOLD(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_SHADOWZSLOPETHRESHOLD, parameter);
    KELVIN_ASSERT(parameter == 0x7F800000); /* FIXME: Unimplemented */
}

static void kelvin_SET_SHADOW_DEPTH_FUNC(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SHADOWCTL, NV_PGRAPH_SHADOWCTL_SHADOW_ZFUNC,
             parameter);
}

static void kelvin_SET_SHADER_STAGE_PROGRAM(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pgraph_reg_w(pg, NV_PGRAPH_SHADERPROG, parameter);
}

static void kelvin_SET_DOT_RGBMAPPING(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SHADERCTL, 0xFFF,
             GET_MASK(parameter, 0xFFF));
}

static void kelvin_SET_SHADER_OTHER_STAGE_INPUT(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_SHADERCTL, 0xFFFF000,
             GET_MASK(parameter, 0xFFFF000));
}

static void kelvin_SET_TRANSFORM_DATA(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    int slot = (method - NV097_SET_TRANSFORM_DATA) / 4;
    pg->vertex_state_shader_v0[slot] = parameter;
}

static void kelvin_LAUNCH_TRANSFORM_PROGRAM(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    unsigned int program_start = parameter;
    Nv2aVshProgram program;
    Nv2aVshCPUXVSSExecutionState state_linkage;
    Nv2aVshExecutionState state;

    if (program_start >= NV2A_MAX_TRANSFORM_PROGRAM_LENGTH) {
        return;
    }
    if (nv2a_vsh_parse_program(
            &program, pg->program_data[program_start],
            NV2A_MAX_TRANSFORM_PROGRAM_LENGTH - program_start)
        != NV2AVPR_SUCCESS) {
        kelvin_unsupported(KU_KELVIN_METHOD,
                           "LAUNCH_TRANSFORM_PROGRAM parse", program_start);
        return;
    }

    state = nv2a_vsh_emu_initialize_xss_execution_state(
            &state_linkage, (float*)pg->vsh_constants);
    memcpy(state_linkage.input_regs, pg->vertex_state_shader_v0,
           sizeof(pg->vertex_state_shader_v0));
    nv2a_vsh_emu_execute_track_context_writes(&state, &program,
                                              pg->vsh_constants_dirty);
    nv2a_vsh_program_destroy(&program);
}

static void kelvin_SET_TRANSFORM_EXECUTION_MODE(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_MODE,
             GET_MASK(parameter,
                      NV097_SET_TRANSFORM_EXECUTION_MODE_MODE));
    PG_SET_MASK(NV_PGRAPH_CSV0_D, NV_PGRAPH_CSV0_D_RANGE_MODE,
             GET_MASK(parameter,
                      NV097_SET_TRANSFORM_EXECUTION_MODE_RANGE_MODE));
}

static void kelvin_SET_TRANSFORM_PROGRAM_CXT_WRITE_EN(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    pg->enable_vertex_program_write = parameter;
}

static void kelvin_SET_TRANSFORM_PROGRAM_LOAD(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    KELVIN_ASSERT(parameter < NV2A_MAX_TRANSFORM_PROGRAM_LENGTH);
    PG_SET_MASK(NV_PGRAPH_CHEOPS_OFFSET,
             NV_PGRAPH_CHEOPS_OFFSET_PROG_LD_PTR, parameter);
}

static void kelvin_SET_TRANSFORM_PROGRAM_START(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    KELVIN_ASSERT(parameter < NV2A_MAX_TRANSFORM_PROGRAM_LENGTH);
    PG_SET_MASK(NV_PGRAPH_CSV0_C,
             NV_PGRAPH_CSV0_C_CHEOPS_PROGRAM_START, parameter);
}

static void kelvin_SET_TRANSFORM_CONSTANT_LOAD(KelvinState *pg, unsigned int subchannel, unsigned int method, uint32_t parameter)
{
    KELVIN_ASSERT(parameter < NV2A_VERTEXSHADER_CONSTANTS);
    PG_SET_MASK(NV_PGRAPH_CHEOPS_OFFSET,
             NV_PGRAPH_CHEOPS_OFFSET_CONST_LD_PTR, parameter);
}


/* ---- dispatch ------------------------------------------------------------ */

typedef void (*KelvinMethodFunc)(KelvinState *pg, unsigned int subchannel,
                                 unsigned int method, uint32_t parameter);
static KelvinMethodFunc s_kelvin_methods[0x2000 / 4];

static void reg_method(uint32_t base, unsigned int count, unsigned int stride,
                       KelvinMethodFunc fn)
{
    unsigned int i;
    for (i = 0; i < count; i++) {
        uint32_t m = base + i * stride;
        if (m < 0x2000) {
            s_kelvin_methods[m / 4] = fn;
        }
    }
}

static void register_methods(void)
{
    reg_method(NV097_SET_OBJECT, 1, 4, kelvin_SET_OBJECT);
    reg_method(NV097_NO_OPERATION, 1, 4, kelvin_NO_OPERATION);
    reg_method(NV097_WAIT_FOR_IDLE, 1, 4, kelvin_WAIT_FOR_IDLE);
    reg_method(NV097_SET_FLIP_READ, 1, 4, kelvin_SET_FLIP_READ);
    reg_method(NV097_SET_FLIP_WRITE, 1, 4, kelvin_SET_FLIP_WRITE);
    reg_method(NV097_SET_FLIP_MODULO, 1, 4, kelvin_SET_FLIP_MODULO);
    reg_method(NV097_FLIP_INCREMENT_WRITE, 1, 4, kelvin_FLIP_INCREMENT_WRITE);
    reg_method(NV097_FLIP_STALL, 1, 4, kelvin_FLIP_STALL);
    reg_method(NV097_SET_CONTEXT_DMA_NOTIFIES, 1, 4, kelvin_SET_CONTEXT_DMA_NOTIFIES);
    reg_method(NV097_SET_CONTEXT_DMA_A, 1, 4, kelvin_SET_CONTEXT_DMA_A);
    reg_method(NV097_SET_CONTEXT_DMA_B, 1, 4, kelvin_SET_CONTEXT_DMA_B);
    reg_method(NV097_SET_CONTEXT_DMA_STATE, 1, 4, kelvin_SET_CONTEXT_DMA_STATE);
    reg_method(NV097_SET_CONTEXT_DMA_COLOR, 1, 4, kelvin_SET_CONTEXT_DMA_COLOR);
    reg_method(NV097_SET_CONTEXT_DMA_ZETA, 1, 4, kelvin_SET_CONTEXT_DMA_ZETA);
    reg_method(NV097_SET_CONTEXT_DMA_VERTEX_A, 1, 4, kelvin_SET_CONTEXT_DMA_VERTEX_A);
    reg_method(NV097_SET_CONTEXT_DMA_VERTEX_B, 1, 4, kelvin_SET_CONTEXT_DMA_VERTEX_B);
    reg_method(NV097_SET_CONTEXT_DMA_SEMAPHORE, 1, 4, kelvin_SET_CONTEXT_DMA_SEMAPHORE);
    reg_method(NV097_SET_CONTEXT_DMA_REPORT, 1, 4, kelvin_SET_CONTEXT_DMA_REPORT);
    reg_method(NV097_SET_SURFACE_CLIP_HORIZONTAL, 1, 4, kelvin_SET_SURFACE_CLIP_HORIZONTAL);
    reg_method(NV097_SET_SURFACE_CLIP_VERTICAL, 1, 4, kelvin_SET_SURFACE_CLIP_VERTICAL);
    reg_method(NV097_SET_SURFACE_FORMAT, 1, 4, kelvin_SET_SURFACE_FORMAT);
    reg_method(NV097_SET_SURFACE_PITCH, 1, 4, kelvin_SET_SURFACE_PITCH);
    reg_method(NV097_SET_SURFACE_COLOR_OFFSET, 1, 4, kelvin_SET_SURFACE_COLOR_OFFSET);
    reg_method(NV097_SET_SURFACE_ZETA_OFFSET, 1, 4, kelvin_SET_SURFACE_ZETA_OFFSET);
    reg_method(NV097_SET_COMBINER_ALPHA_ICW, 8, 4, kelvin_SET_COMBINER_ALPHA_ICW);
    reg_method(NV097_SET_COMBINER_SPECULAR_FOG_CW0, 1, 4, kelvin_SET_COMBINER_SPECULAR_FOG_CW0);
    reg_method(NV097_SET_COMBINER_SPECULAR_FOG_CW1, 1, 4, kelvin_SET_COMBINER_SPECULAR_FOG_CW1);
    reg_method(NV097_SET_TEXTURE_ADDRESS, 4, 64, kelvin_SET_TEXTURE_ADDRESS);
    reg_method(NV097_SET_CONTROL0, 1, 4, kelvin_SET_CONTROL0);
    reg_method(NV097_SET_LIGHT_CONTROL, 1, 4, kelvin_SET_LIGHT_CONTROL);
    reg_method(NV097_SET_COLOR_MATERIAL, 1, 4, kelvin_SET_COLOR_MATERIAL);
    reg_method(NV097_SET_FOG_MODE, 1, 4, kelvin_SET_FOG_MODE);
    reg_method(NV097_SET_FOG_GEN_MODE, 1, 4, kelvin_SET_FOG_GEN_MODE);
    reg_method(NV097_SET_FOG_ENABLE, 1, 4, kelvin_SET_FOG_ENABLE);
    reg_method(NV097_SET_FOG_COLOR, 1, 4, kelvin_SET_FOG_COLOR);
    reg_method(NV097_SET_WINDOW_CLIP_TYPE, 1, 4, kelvin_SET_WINDOW_CLIP_TYPE);
    reg_method(NV097_SET_WINDOW_CLIP_HORIZONTAL, 8, 4, kelvin_SET_WINDOW_CLIP_HORIZONTAL);
    reg_method(NV097_SET_WINDOW_CLIP_VERTICAL, 8, 4, kelvin_SET_WINDOW_CLIP_VERTICAL);
    reg_method(NV097_SET_ALPHA_TEST_ENABLE, 1, 4, kelvin_SET_ALPHA_TEST_ENABLE);
    reg_method(NV097_SET_BLEND_ENABLE, 1, 4, kelvin_SET_BLEND_ENABLE);
    reg_method(NV097_SET_CULL_FACE_ENABLE, 1, 4, kelvin_SET_CULL_FACE_ENABLE);
    reg_method(NV097_SET_DEPTH_TEST_ENABLE, 1, 4, kelvin_SET_DEPTH_TEST_ENABLE);
    reg_method(NV097_SET_DITHER_ENABLE, 1, 4, kelvin_SET_DITHER_ENABLE);
    reg_method(NV097_SET_LIGHTING_ENABLE, 1, 4, kelvin_SET_LIGHTING_ENABLE);
    reg_method(NV097_SET_POINT_PARAMS_ENABLE, 1, 4, kelvin_SET_POINT_PARAMS_ENABLE);
    reg_method(NV097_SET_POINT_SMOOTH_ENABLE, 1, 4, kelvin_SET_POINT_SMOOTH_ENABLE);
    reg_method(NV097_SET_LINE_SMOOTH_ENABLE, 1, 4, kelvin_SET_LINE_SMOOTH_ENABLE);
    reg_method(NV097_SET_POLY_SMOOTH_ENABLE, 1, 4, kelvin_SET_POLY_SMOOTH_ENABLE);
    reg_method(NV097_SET_SKIN_MODE, 1, 4, kelvin_SET_SKIN_MODE);
    reg_method(NV097_SET_STENCIL_TEST_ENABLE, 1, 4, kelvin_SET_STENCIL_TEST_ENABLE);
    reg_method(NV097_SET_POLY_OFFSET_POINT_ENABLE, 1, 4, kelvin_SET_POLY_OFFSET_POINT_ENABLE);
    reg_method(NV097_SET_POLY_OFFSET_LINE_ENABLE, 1, 4, kelvin_SET_POLY_OFFSET_LINE_ENABLE);
    reg_method(NV097_SET_POLY_OFFSET_FILL_ENABLE, 1, 4, kelvin_SET_POLY_OFFSET_FILL_ENABLE);
    reg_method(NV097_SET_ALPHA_FUNC, 1, 4, kelvin_SET_ALPHA_FUNC);
    reg_method(NV097_SET_ALPHA_REF, 1, 4, kelvin_SET_ALPHA_REF);
    reg_method(NV097_SET_BLEND_FUNC_SFACTOR, 1, 4, kelvin_SET_BLEND_FUNC_SFACTOR);
    reg_method(NV097_SET_BLEND_FUNC_DFACTOR, 1, 4, kelvin_SET_BLEND_FUNC_DFACTOR);
    reg_method(NV097_SET_BLEND_COLOR, 1, 4, kelvin_SET_BLEND_COLOR);
    reg_method(NV097_SET_BLEND_EQUATION, 1, 4, kelvin_SET_BLEND_EQUATION);
    reg_method(NV097_SET_DEPTH_FUNC, 1, 4, kelvin_SET_DEPTH_FUNC);
    reg_method(NV097_SET_COLOR_MASK, 1, 4, kelvin_SET_COLOR_MASK);
    reg_method(NV097_SET_DEPTH_MASK, 1, 4, kelvin_SET_DEPTH_MASK);
    reg_method(NV097_SET_STENCIL_MASK, 1, 4, kelvin_SET_STENCIL_MASK);
    reg_method(NV097_SET_STENCIL_FUNC, 1, 4, kelvin_SET_STENCIL_FUNC);
    reg_method(NV097_SET_STENCIL_FUNC_REF, 1, 4, kelvin_SET_STENCIL_FUNC_REF);
    reg_method(NV097_SET_STENCIL_FUNC_MASK, 1, 4, kelvin_SET_STENCIL_FUNC_MASK);
    reg_method(NV097_SET_STENCIL_OP_FAIL, 1, 4, kelvin_SET_STENCIL_OP_FAIL);
    reg_method(NV097_SET_STENCIL_OP_ZFAIL, 1, 4, kelvin_SET_STENCIL_OP_ZFAIL);
    reg_method(NV097_SET_STENCIL_OP_ZPASS, 1, 4, kelvin_SET_STENCIL_OP_ZPASS);
    reg_method(NV097_SET_SHADE_MODE, 1, 4, kelvin_SET_SHADE_MODE);
    reg_method(NV097_SET_PROVOKING_VERTEX, 1, 4, kelvin_SET_PROVOKING_VERTEX);
    reg_method(NV097_SET_POLYGON_OFFSET_SCALE_FACTOR, 1, 4, kelvin_SET_POLYGON_OFFSET_SCALE_FACTOR);
    reg_method(NV097_SET_POLYGON_OFFSET_BIAS, 1, 4, kelvin_SET_POLYGON_OFFSET_BIAS);
    reg_method(NV097_SET_FRONT_POLYGON_MODE, 1, 4, kelvin_SET_FRONT_POLYGON_MODE);
    reg_method(NV097_SET_BACK_POLYGON_MODE, 1, 4, kelvin_SET_BACK_POLYGON_MODE);
    reg_method(NV097_SET_CLIP_MIN, 1, 4, kelvin_SET_CLIP_MIN);
    reg_method(NV097_SET_CLIP_MAX, 1, 4, kelvin_SET_CLIP_MAX);
    reg_method(NV097_SET_CULL_FACE, 1, 4, kelvin_SET_CULL_FACE);
    reg_method(NV097_SET_FRONT_FACE, 1, 4, kelvin_SET_FRONT_FACE);
    reg_method(NV097_SET_NORMALIZATION_ENABLE, 1, 4, kelvin_SET_NORMALIZATION_ENABLE);
    reg_method(NV097_SET_MATERIAL_EMISSION, 3, 4, kelvin_SET_MATERIAL_EMISSION);
    reg_method(NV097_SET_MATERIAL_ALPHA, 1, 4, kelvin_SET_MATERIAL_ALPHA);
    reg_method(NV097_SET_SPECULAR_ENABLE, 1, 4, kelvin_SET_SPECULAR_ENABLE);
    reg_method(NV097_SET_LIGHT_ENABLE_MASK, 1, 4, kelvin_SET_LIGHT_ENABLE_MASK);
    reg_method(NV097_SET_TEXGEN_S, 4, 16, kelvin_SET_TEXGEN_S);
    reg_method(NV097_SET_TEXGEN_T, 4, 16, kelvin_SET_TEXGEN_T);
    reg_method(NV097_SET_TEXGEN_R, 4, 16, kelvin_SET_TEXGEN_R);
    reg_method(NV097_SET_TEXGEN_Q, 4, 16, kelvin_SET_TEXGEN_Q);
    reg_method(NV097_SET_TEXTURE_MATRIX_ENABLE, 4, 4, kelvin_SET_TEXTURE_MATRIX_ENABLE);
    reg_method(NV097_SET_POINT_SIZE, 1, 4, kelvin_SET_POINT_SIZE);
    reg_method(NV097_SET_PROJECTION_MATRIX, 16, 4, kelvin_SET_PROJECTION_MATRIX);
    reg_method(NV097_SET_MODEL_VIEW_MATRIX, 16*4, 4, kelvin_SET_MODEL_VIEW_MATRIX);
    reg_method(NV097_SET_INVERSE_MODEL_VIEW_MATRIX, 16*4, 4, kelvin_SET_INVERSE_MODEL_VIEW_MATRIX);
    reg_method(NV097_SET_COMPOSITE_MATRIX, 16, 4, kelvin_SET_COMPOSITE_MATRIX);
    reg_method(NV097_SET_TEXTURE_MATRIX, 16*4, 4, kelvin_SET_TEXTURE_MATRIX);
    reg_method(NV097_SET_FOG_PARAMS, 3, 4, kelvin_SET_FOG_PARAMS);
    reg_method(NV097_SET_TEXGEN_PLANE_S, 4*4*4, 4, kelvin_SET_TEXGEN_PLANE_S);
    reg_method(NV097_SET_TEXGEN_VIEW_MODEL, 1, 4, kelvin_SET_TEXGEN_VIEW_MODEL);
    reg_method(NV097_SET_FOG_PLANE, 4, 4, kelvin_SET_FOG_PLANE);
    reg_method(NV097_SET_SPECULAR_PARAMS, 6, 4, kelvin_SET_SPECULAR_PARAMS);
    reg_method(NV097_SET_SCENE_AMBIENT_COLOR, 3, 4, kelvin_SET_SCENE_AMBIENT_COLOR);
    reg_method(NV097_SET_VIEWPORT_OFFSET, 4, 4, kelvin_SET_VIEWPORT_OFFSET);
    reg_method(NV097_SET_POINT_PARAMS, 8, 4, kelvin_SET_POINT_PARAMS);
    reg_method(NV097_SET_EYE_POSITION, 4, 4, kelvin_SET_EYE_POSITION);
    reg_method(NV097_SET_COMBINER_FACTOR0, 8, 4, kelvin_SET_COMBINER_FACTOR0);
    reg_method(NV097_SET_COMBINER_FACTOR1, 8, 4, kelvin_SET_COMBINER_FACTOR1);
    reg_method(NV097_SET_COMBINER_ALPHA_OCW, 8, 4, kelvin_SET_COMBINER_ALPHA_OCW);
    reg_method(NV097_SET_COMBINER_COLOR_ICW, 8, 4, kelvin_SET_COMBINER_COLOR_ICW);
    reg_method(NV097_SET_COLOR_KEY_COLOR, 4, 4, kelvin_SET_COLOR_KEY_COLOR);
    reg_method(NV097_SET_VIEWPORT_SCALE, 4, 4, kelvin_SET_VIEWPORT_SCALE);
    reg_method(NV097_SET_TRANSFORM_PROGRAM, 32, 4, kelvin_SET_TRANSFORM_PROGRAM);
    reg_method(NV097_SET_TRANSFORM_CONSTANT, 32, 4, kelvin_SET_TRANSFORM_CONSTANT);
    reg_method(NV097_SET_VERTEX3F, 3, 4, kelvin_SET_VERTEX3F);
    reg_method(NV097_SET_BACK_LIGHT_AMBIENT_COLOR, 8*16, 4, kelvin_SET_BACK_LIGHT_AMBIENT_COLOR);
    reg_method(NV097_SET_LIGHT_AMBIENT_COLOR, 8*32, 4, kelvin_SET_LIGHT_AMBIENT_COLOR);
    reg_method(NV097_SET_VERTEX4F, 4, 4, kelvin_SET_VERTEX4F);
    reg_method(NV097_SET_NORMAL3F, 3, 4, kelvin_SET_NORMAL3F);
    reg_method(NV097_SET_NORMAL3S, 3, 4, kelvin_SET_NORMAL3S);
    reg_method(NV097_SET_DIFFUSE_COLOR4F, 4, 4, kelvin_SET_DIFFUSE_COLOR4F);
    reg_method(NV097_SET_DIFFUSE_COLOR3F, 3, 4, kelvin_SET_DIFFUSE_COLOR3F);
    reg_method(NV097_SET_DIFFUSE_COLOR4UB, 1, 4, kelvin_SET_DIFFUSE_COLOR4UB);
    reg_method(NV097_SET_SPECULAR_COLOR4F, 4, 4, kelvin_SET_SPECULAR_COLOR4F);
    reg_method(NV097_SET_SPECULAR_COLOR3F, 3, 4, kelvin_SET_SPECULAR_COLOR3F);
    reg_method(NV097_SET_SPECULAR_COLOR4UB, 1, 4, kelvin_SET_SPECULAR_COLOR4UB);
    reg_method(NV097_SET_TEXCOORD0_2F, 2, 4, kelvin_SET_TEXCOORD0_2F);
    reg_method(NV097_SET_TEXCOORD0_4F, 4, 4, kelvin_SET_TEXCOORD0_4F);
    reg_method(NV097_SET_TEXCOORD0_2S, 1, 4, kelvin_SET_TEXCOORD0_2S);
    reg_method(NV097_SET_TEXCOORD0_4S, 2, 4, kelvin_SET_TEXCOORD0_4S);
    reg_method(NV097_SET_TEXCOORD1_2F, 2, 4, kelvin_SET_TEXCOORD1_2F);
    reg_method(NV097_SET_TEXCOORD1_4F, 4, 4, kelvin_SET_TEXCOORD1_4F);
    reg_method(NV097_SET_TEXCOORD1_2S, 1, 4, kelvin_SET_TEXCOORD1_2S);
    reg_method(NV097_SET_TEXCOORD1_4S, 2, 4, kelvin_SET_TEXCOORD1_4S);
    reg_method(NV097_SET_TEXCOORD2_2F, 2, 4, kelvin_SET_TEXCOORD2_2F);
    reg_method(NV097_SET_TEXCOORD2_4F, 4, 4, kelvin_SET_TEXCOORD2_4F);
    reg_method(NV097_SET_TEXCOORD2_2S, 1, 4, kelvin_SET_TEXCOORD2_2S);
    reg_method(NV097_SET_TEXCOORD2_4S, 2, 4, kelvin_SET_TEXCOORD2_4S);
    reg_method(NV097_SET_TEXCOORD3_2F, 2, 4, kelvin_SET_TEXCOORD3_2F);
    reg_method(NV097_SET_TEXCOORD3_4F, 4, 4, kelvin_SET_TEXCOORD3_4F);
    reg_method(NV097_SET_TEXCOORD3_2S, 1, 4, kelvin_SET_TEXCOORD3_2S);
    reg_method(NV097_SET_TEXCOORD3_4S, 2, 4, kelvin_SET_TEXCOORD3_4S);
    reg_method(NV097_SET_FOG_COORD, 1, 4, kelvin_SET_FOG_COORD);
    reg_method(NV097_SET_WEIGHT1F, 1, 4, kelvin_SET_WEIGHT1F);
    reg_method(NV097_SET_WEIGHT2F, 2, 4, kelvin_SET_WEIGHT2F);
    reg_method(NV097_SET_WEIGHT3F, 3, 4, kelvin_SET_WEIGHT3F);
    reg_method(NV097_SET_WEIGHT4F, 4, 4, kelvin_SET_WEIGHT4F);
    reg_method(NV097_SET_VERTEX_DATA_ARRAY_FORMAT, 16, 4, kelvin_SET_VERTEX_DATA_ARRAY_FORMAT);
    reg_method(NV097_SET_VERTEX_DATA_ARRAY_OFFSET, 16, 4, kelvin_SET_VERTEX_DATA_ARRAY_OFFSET);
    reg_method(NV097_SET_LOGIC_OP_ENABLE, 1, 4, kelvin_SET_LOGIC_OP_ENABLE);
    reg_method(NV097_SET_LOGIC_OP, 1, 4, kelvin_SET_LOGIC_OP);
    reg_method(NV097_CLEAR_REPORT_VALUE, 1, 4, kelvin_CLEAR_REPORT_VALUE);
    reg_method(NV097_SET_ZPASS_PIXEL_COUNT_ENABLE, 1, 4, kelvin_SET_ZPASS_PIXEL_COUNT_ENABLE);
    reg_method(NV097_GET_REPORT, 1, 4, kelvin_GET_REPORT);
    reg_method(NV097_SET_EYE_DIRECTION, 3, 4, kelvin_SET_EYE_DIRECTION);
    reg_method(NV097_SET_BEGIN_END, 1, 4, kelvin_SET_BEGIN_END);
    reg_method(NV097_SET_TEXTURE_OFFSET, 4, 64, kelvin_SET_TEXTURE_OFFSET);
    reg_method(NV097_SET_TEXTURE_FORMAT, 4, 64, kelvin_SET_TEXTURE_FORMAT);
    reg_method(NV097_SET_TEXTURE_CONTROL0, 4, 64, kelvin_SET_TEXTURE_CONTROL0);
    reg_method(NV097_SET_TEXTURE_CONTROL1, 4, 64, kelvin_SET_TEXTURE_CONTROL1);
    reg_method(NV097_SET_TEXTURE_FILTER, 4, 64, kelvin_SET_TEXTURE_FILTER);
    reg_method(NV097_SET_TEXTURE_IMAGE_RECT, 4, 64, kelvin_SET_TEXTURE_IMAGE_RECT);
    reg_method(NV097_SET_TEXTURE_PALETTE, 4, 64, kelvin_SET_TEXTURE_PALETTE);
    reg_method(NV097_SET_TEXTURE_BORDER_COLOR, 4, 64, kelvin_SET_TEXTURE_BORDER_COLOR);
    reg_method(NV097_SET_TEXTURE_SET_BUMP_ENV_MAT, 4, 64, kelvin_SET_TEXTURE_SET_BUMP_ENV_MAT);
    reg_method(NV097_SET_TEXTURE_SET_BUMP_ENV_MAT + 4, 4, 64, kelvin_SET_TEXTURE_SET_BUMP_ENV_MAT);
    reg_method(NV097_SET_TEXTURE_SET_BUMP_ENV_MAT + 8, 4, 64, kelvin_SET_TEXTURE_SET_BUMP_ENV_MAT);
    reg_method(NV097_SET_TEXTURE_SET_BUMP_ENV_MAT + 12, 4, 64, kelvin_SET_TEXTURE_SET_BUMP_ENV_MAT);
    reg_method(NV097_SET_TEXTURE_SET_BUMP_ENV_SCALE, 4, 64, kelvin_SET_TEXTURE_SET_BUMP_ENV_SCALE);
    reg_method(NV097_SET_TEXTURE_SET_BUMP_ENV_OFFSET, 4, 64, kelvin_SET_TEXTURE_SET_BUMP_ENV_OFFSET);
    reg_method(NV097_ARRAY_ELEMENT16, 1, 4, kelvin_ARRAY_ELEMENT16);
    reg_method(NV097_ARRAY_ELEMENT32, 1, 4, kelvin_ARRAY_ELEMENT32);
    reg_method(NV097_DRAW_ARRAYS, 1, 4, kelvin_DRAW_ARRAYS);
    reg_method(NV097_INLINE_ARRAY, 1, 4, kelvin_INLINE_ARRAY);
    reg_method(NV097_SET_EYE_VECTOR, 3, 4, kelvin_SET_EYE_VECTOR);
    reg_method(NV097_SET_VERTEX_DATA2F_M, 32, 4, kelvin_SET_VERTEX_DATA2F_M);
    reg_method(NV097_SET_VERTEX_DATA4F_M, 64, 4, kelvin_SET_VERTEX_DATA4F_M);
    reg_method(NV097_SET_VERTEX_DATA2S, 16, 4, kelvin_SET_VERTEX_DATA2S);
    reg_method(NV097_SET_VERTEX_DATA4UB, 16, 4, kelvin_SET_VERTEX_DATA4UB);
    reg_method(NV097_SET_VERTEX_DATA4S_M, 32, 4, kelvin_SET_VERTEX_DATA4S_M);
    reg_method(NV097_SET_SEMAPHORE_OFFSET, 1, 4, kelvin_SET_SEMAPHORE_OFFSET);
    reg_method(NV097_BACK_END_WRITE_SEMAPHORE_RELEASE, 1, 4, kelvin_BACK_END_WRITE_SEMAPHORE_RELEASE);
    reg_method(NV097_SET_ZMIN_MAX_CONTROL, 1, 4, kelvin_SET_ZMIN_MAX_CONTROL);
    reg_method(NV097_SET_ANTI_ALIASING_CONTROL, 1, 4, kelvin_SET_ANTI_ALIASING_CONTROL);
    reg_method(NV097_SET_ZSTENCIL_CLEAR_VALUE, 1, 4, kelvin_SET_ZSTENCIL_CLEAR_VALUE);
    reg_method(NV097_SET_COLOR_CLEAR_VALUE, 1, 4, kelvin_SET_COLOR_CLEAR_VALUE);
    reg_method(NV097_CLEAR_SURFACE, 1, 4, kelvin_CLEAR_SURFACE);
    reg_method(NV097_SET_CLEAR_RECT_HORIZONTAL, 1, 4, kelvin_SET_CLEAR_RECT_HORIZONTAL);
    reg_method(NV097_SET_CLEAR_RECT_VERTICAL, 1, 4, kelvin_SET_CLEAR_RECT_VERTICAL);
    reg_method(NV097_SET_SPECULAR_FOG_FACTOR, 2, 4, kelvin_SET_SPECULAR_FOG_FACTOR);
    reg_method(NV097_SET_SPECULAR_PARAMS_BACK, 6, 4, kelvin_SET_SPECULAR_PARAMS_BACK);
    reg_method(NV097_SET_SHADER_CLIP_PLANE_MODE, 1, 4, kelvin_SET_SHADER_CLIP_PLANE_MODE);
    reg_method(NV097_SET_COMBINER_COLOR_OCW, 8, 4, kelvin_SET_COMBINER_COLOR_OCW);
    reg_method(NV097_SET_COMBINER_CONTROL, 1, 4, kelvin_SET_COMBINER_CONTROL);
    reg_method(NV097_SET_SHADOW_ZSLOPE_THRESHOLD, 1, 4, kelvin_SET_SHADOW_ZSLOPE_THRESHOLD);
    reg_method(NV097_SET_SHADOW_DEPTH_FUNC, 1, 4, kelvin_SET_SHADOW_DEPTH_FUNC);
    reg_method(NV097_SET_SHADER_STAGE_PROGRAM, 1, 4, kelvin_SET_SHADER_STAGE_PROGRAM);
    reg_method(NV097_SET_DOT_RGBMAPPING, 1, 4, kelvin_SET_DOT_RGBMAPPING);
    reg_method(NV097_SET_SHADER_OTHER_STAGE_INPUT, 1, 4, kelvin_SET_SHADER_OTHER_STAGE_INPUT);
    reg_method(NV097_SET_TRANSFORM_DATA, 4, 4, kelvin_SET_TRANSFORM_DATA);
    reg_method(NV097_LAUNCH_TRANSFORM_PROGRAM, 1, 4, kelvin_LAUNCH_TRANSFORM_PROGRAM);
    reg_method(NV097_SET_TRANSFORM_EXECUTION_MODE, 1, 4, kelvin_SET_TRANSFORM_EXECUTION_MODE);
    reg_method(NV097_SET_TRANSFORM_PROGRAM_CXT_WRITE_EN, 1, 4, kelvin_SET_TRANSFORM_PROGRAM_CXT_WRITE_EN);
    reg_method(NV097_SET_TRANSFORM_PROGRAM_LOAD, 1, 4, kelvin_SET_TRANSFORM_PROGRAM_LOAD);
    reg_method(NV097_SET_TRANSFORM_PROGRAM_START, 1, 4, kelvin_SET_TRANSFORM_PROGRAM_START);
    reg_method(NV097_SET_TRANSFORM_CONSTANT_LOAD, 1, 4, kelvin_SET_TRANSFORM_CONSTANT_LOAD);
}

static const char *class_name(unsigned int cls)
{
    switch (cls) {
    case NV_BETA: return "BETA";
    case NV_CONTEXT_PATTERN: return "CONTEXT_PATTERN";
    case NV_CONTEXT_SURFACES_2D: return "CONTEXT_SURFACES_2D";
    case NV_IMAGE_BLIT: return "IMAGE_BLIT";
    case NV_KELVIN_PRIMITIVE: return "KELVIN";
    default: return "?";
    }
}

void kelvin_pgraph_init(KelvinState *pg, uint32_t *regs)
{
    int i;

    memset(pg, 0, sizeof(*pg));
    pg->regs = regs;
    pg->primitive_mode = PRIM_TYPE_INVALID;
    for (i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attr = &pg->vertex_attributes[i];
        attr->inline_buffer = (float *)calloc(NV2A_MAX_BATCH_LENGTH,
                                              sizeof(float) * 4);
        attr->inline_value[3] = 1.0f;
    }
    pg->material_alpha = 0.0f;
    pgraph_reset_inline_buffers(pg);
    register_methods();
}

/* A run of data words for one non-incrementing method, taken in one call. Vertex indices arrive as runs of
 * ARRAY_ELEMENT16 / ARRAY_ELEMENT32 words and inline vertex data as INLINE_ARRAY words -- about 100,000 words
 * a frame in BLACK -- and the same per-word dispatch cost as any other method made them most of the pusher's
 * time. Returns false when the run is not one of these (the caller then dispatches the words one at a time);
 * the effect is exactly that of the per-word handlers. */
bool kelvin_pgraph_method_run(KelvinState *pg, unsigned int subchannel, unsigned int method,
                              const uint32_t *words, uint32_t n)
{
    uint32_t i;
    if (pg->subch_class[subchannel] != NV_KELVIN_PRIMITIVE) return false;
    if (method == NV097_ARRAY_ELEMENT16) {
        if (pg->inline_elements_length + 2u * n >= NV2A_MAX_BATCH_LENGTH) return false;
        pgraph_check_within_begin_end_block(pg);
        if (pg->draw_arrays_length) pgraph_expand_draw_arrays(pg);
        {
            uint32_t *out = pg->inline_elements + pg->inline_elements_length;
            for (i = 0; i < n; i++) {
                uint32_t w = words[i];
                out[2 * i] = w & 0xFFFF;
                out[2 * i + 1] = w >> 16;
            }
            pg->inline_elements_length += 2u * n;
        }
    } else if (method == NV097_ARRAY_ELEMENT32) {
        if (pg->inline_elements_length + n >= NV2A_MAX_BATCH_LENGTH) return false;
        pgraph_check_within_begin_end_block(pg);
        if (pg->draw_arrays_length) pgraph_expand_draw_arrays(pg);
        memcpy(pg->inline_elements + pg->inline_elements_length, words, (size_t)n * 4);
        pg->inline_elements_length += n;
    } else if (method == NV097_INLINE_ARRAY) {
        if (pg->inline_array_length + n >= NV2A_MAX_BATCH_LENGTH) return false;
        pgraph_check_within_begin_end_block(pg);
        memcpy(pg->inline_array + pg->inline_array_length, words, (size_t)n * 4);
        pg->inline_array_length += n;
    } else {
        return false;
    }
    pg->stats.methods += n;
    g_kelvin_method = (subchannel << 16) | method;
    return true;
}

/* One method word. `method` is the byte offset, `parameter` already has any
 * object handle (0x180..0x1FC) translated to an instance address, as PFIFO
 * does on hardware. `inc` is informational: PFIFO has already advanced
 * `method` for increasing runs. */
void kelvin_pgraph_method(KelvinState *pg, unsigned int subchannel,
                          unsigned int method, uint32_t parameter, bool inc)
{
    unsigned int cls;
    (void)inc;

    pg->stats.methods++;
    g_kelvin_method = (subchannel << 16) | method;
    if (method == 0) {
        /* SET_OBJECT: parameter is the object's instance address. Its first
         * word carries the graphics class, which decides how every later
         * method on this subchannel is interpreted. */
        pg->subch_instance[subchannel] = parameter;
        pg->subch_class[subchannel] =
            kelvin_pramin_r32(parameter) & NV_PGRAPH_CTX_SWITCH1_GRCLASS;
    }
    cls = pg->subch_class[subchannel];

    switch (cls) {
    case NV_BETA:
        switch (method) {
        case NV012_SET_OBJECT:
            pg->beta.object_instance = parameter;
            return;
        case NV012_SET_BETA:
            if (parameter & 0x80000000) {
                pg->beta.beta = 0;
            } else {
                pg->beta.beta = parameter & 0x7f800000;
            }
            return;
        }
        break;
    case NV_CONTEXT_PATTERN:
        switch (method) {
        case NV044_SET_MONOCHROME_COLOR0:
            pgraph_reg_w(pg, NV_PGRAPH_PATT_COLOR0, parameter);
            return;
        }
        break;
    case NV_CONTEXT_SURFACES_2D: {
        ContextSurfaces2DState *c = &pg->context_surfaces_2d;
        switch (method) {
        case NV062_SET_OBJECT: c->object_instance = parameter; return;
        case NV062_SET_CONTEXT_DMA_IMAGE_SOURCE: c->dma_image_source = parameter; return;
        case NV062_SET_CONTEXT_DMA_IMAGE_DESTIN: c->dma_image_dest = parameter; return;
        case NV062_SET_COLOR_FORMAT: c->color_format = parameter; return;
        case NV062_SET_PITCH:
            c->source_pitch = parameter & 0xFFFF;
            c->dest_pitch = parameter >> 16;
            return;
        case NV062_SET_OFFSET_SOURCE: c->source_offset = parameter & 0x07FFFFFF; return;
        case NV062_SET_OFFSET_DESTIN: c->dest_offset = parameter & 0x07FFFFFF; return;
        }
        break;
    }
    case NV_IMAGE_BLIT: {
        ImageBlitState *b = &pg->image_blit;
        switch (method) {
        case NV09F_SET_OBJECT: b->object_instance = parameter; return;
        case NV09F_SET_CONTEXT_SURFACES: b->context_surfaces = parameter; return;
        case NV09F_SET_OPERATION: b->operation = parameter; return;
        case NV09F_CONTROL_POINT_IN:
            b->in_x = parameter & 0xFFFF;
            b->in_y = parameter >> 16;
            return;
        case NV09F_CONTROL_POINT_OUT:
            b->out_x = parameter & 0xFFFF;
            b->out_y = parameter >> 16;
            return;
        case NV09F_SIZE:
            b->width = parameter & 0xFFFF;
            b->height = parameter >> 16;
            if (b->width && b->height) {
                kelvin_render_image_blit(pg);
            }
            return;
        }
        break;
    }
    case NV_KELVIN_PRIMITIVE: {
        KelvinMethodFunc handler = method < 0x2000
                                       ? s_kelvin_methods[method / 4] : NULL;
        if (handler) {
            handler(pg, subchannel, method, parameter);
            return;
        }
        pg->stats.unhandled_methods++;
        kelvin_unsupported(KU_KELVIN_METHOD, "Kelvin method", method);
        return;
    }
    default:
        break;
    }

    if (method == 0) {
        return;
    }
    pg->stats.unhandled_methods++;
    {
        static unsigned said;
        if (said++ < 16) {
            fprintf(stderr, "[KELVIN] unhandled method 0x%04X on subchannel %u"
                            " (class 0x%02X %s) param 0x%08X\n",
                    method, subchannel, cls, class_name(cls), parameter);
        }
    }
    kelvin_unsupported(KU_CLASS, "method on unmodelled class", cls);
}

/* ---- vertex bookkeeping (xemu pgraph/vertex.c) ---------------------------- */

void pgraph_update_inline_value(VertexAttribute *attr, const uint8_t *data)
{
    KELVIN_ASSERT(attr->count <= 4);
    attr->inline_value[0] = 0.0f;
    attr->inline_value[1] = 0.0f;
    attr->inline_value[2] = 0.0f;
    attr->inline_value[3] = 1.0f;

    switch (attr->format) {
        case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_D3D:
        case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_UB_OGL:
            for (uint32_t i = 0; i < attr->count; ++i) {
                attr->inline_value[i] = (float)data[i] / 255.0f;
            }
            break;
        case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S1: {
            const int16_t *val = (const int16_t *) data;
            for (uint32_t i = 0; i < attr->count; ++i, ++val) {
                attr->inline_value[i] = MAX(-1.0f, (float) *val / 32767.0f);
            }
            break;
        }
        case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_F:
            memcpy(attr->inline_value, data, attr->size * attr->count);
            break;
        case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_S32K: {
            const int16_t *val = (const int16_t *) data;
            for (uint32_t i = 0; i < attr->count; ++i, ++val) {
                attr->inline_value[i] = (float)*val;
            }
            break;
        }
        case NV097_SET_VERTEX_DATA_ARRAY_FORMAT_TYPE_CMP: {
            /* 3 signed, normalized components packed in 32-bits. (11,11,10) */
            const int32_t val = *(const int32_t *)data;
            int32_t x = val & 0x7FF;
            if (x & 0x400) {
                x |= 0xFFFFF800;
            }
            int32_t y = (val >> 11) & 0x7FF;
            if (y & 0x400) {
                y |= 0xFFFFF800;
            }
            int32_t z = (val >> 22) & 0x7FF;
            if (z & 0x200) {
                z |= 0xFFFFFC00;
            }

            attr->inline_value[0] = MAX(-1.0f, (float)x / 1023.0f);
            attr->inline_value[1] = MAX(-1.0f, (float)y / 1023.0f);
            attr->inline_value[2] = MAX(-1.0f, (float)z / 511.0f);
            break;
        }
    default:
        fprintf(stderr, "Unknown vertex attribute type: for format 0x%x\n",
                attr->format);
        KELVIN_ASSERT(!"Unsupported attribute type");
        break;
    }
}

void pgraph_allocate_inline_buffer_vertices(KelvinState *pg, unsigned int attr)
{
    VertexAttribute *attribute = &pg->vertex_attributes[attr];

    if (attribute->inline_buffer_populated || pg->inline_buffer_length == 0) {
        return;
    }

    /* Now upload the previous attribute value */
    attribute->inline_buffer_populated = true;
    for (int i = 0; i < pg->inline_buffer_length; i++) {
        memcpy(&attribute->inline_buffer[i * 4], attribute->inline_value,
               sizeof(float) * 4);
    }
}

void pgraph_finish_inline_buffer_vertex(KelvinState *pg)
{
    pgraph_check_within_begin_end_block(pg);
    KELVIN_ASSERT(pg->inline_buffer_length < NV2A_MAX_BATCH_LENGTH);

    for (int i = 0; i < NV2A_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attribute = &pg->vertex_attributes[i];
        if (attribute->inline_buffer_populated) {
            memcpy(&attribute->inline_buffer[pg->inline_buffer_length * 4],
                   attribute->inline_value, sizeof(float) * 4);
        }
    }

    pg->inline_buffer_length++;
}

void pgraph_reset_inline_buffers(KelvinState *pg)
{
    pg->inline_elements_length = 0;
    pg->inline_array_length = 0;
    pg->inline_buffer_length = 0;
    pgraph_reset_draw_arrays(pg);
}

void pgraph_reset_draw_arrays(KelvinState *pg)
{
    pg->draw_arrays_length = 0;
    pg->draw_arrays_min_start = -1;
    pg->draw_arrays_max_count = 0;
    pg->draw_arrays_prevent_connect = false;
}
