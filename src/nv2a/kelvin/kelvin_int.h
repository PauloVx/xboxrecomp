/*
 * Kelvin: a CPU implementation of the NV2A 3D engine, for titles that drive
 * the GPU themselves through a statically linked D3D8 (D3D8LTCG and friends).
 *
 * Structure and semantics follow xemu (hw/xbox/nv2a): the method handlers and
 * PGRAPH register model are ported from pgraph/pgraph.c, and the vertex and
 * pixel pipelines implement what xemu's GLSL generators (glsl/vsh*.c,
 * glsl/psh.c, glsl/geom.c) emit, evaluated per vertex and per pixel instead of
 * compiled. By default rendering writes into guest surfaces. The opt-in
 * resolution path owns larger full-screen surfaces and mirrors them at GPU
 * fences while retaining their detail for render-target texture feedback.
 *
 * Copyright (c) 2012 espes
 * Copyright (c) 2015 Jannik Vogel
 * Copyright (c) 2018-2025 Matt Borgerson
 * (xemu PGRAPH, from which the state model and method handlers are ported)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */
#ifndef XBOX_NV2A_KELVIN_INT_H
#define XBOX_NV2A_KELVIN_INT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "../qemu_shim.h"
#include "../nv2a_regs.h"
#include "kelvin_enums.h"
#include "psh_regs.h"

#define KELVIN_SUBCHANNELS 8

/* Guest-visible NV2A aperture layout. */
#define KELVIN_APERTURE_VA   0xFD000000u
#define KELVIN_PFIFO_BASE    0x00002000u
#define KELVIN_PGRAPH_BASE   0x00400000u
#define KELVIN_PRAMIN_BASE   0x00700000u
#define KELVIN_PRAMIN_SIZE   0x00100000u
#define KELVIN_USER_BASE     0x00800000u
#define KELVIN_USER_DMA_PUT  0x00800040u
#define KELVIN_USER_DMA_GET  0x00800044u

/* Every surface, texture and vertex address is physical; the contiguous
 * window is the physical view of guest RAM. */
#define KELVIN_PHYS_WINDOW   0x80000000u
/* 128 MB: the contiguous window's 64 MB, then low RAM's own physical view
 * above it (xbox_PhysicalAddressOf), both reached as 0x80000000 + P. */
#define KELVIN_PHYS_SIZE     0x08000000u

typedef struct VertexAttribute {
    bool dma_select;
    uint32_t offset;
    unsigned int inline_array_offset;
    float inline_value[4];
    unsigned int format;
    unsigned int size;   /* size of the data type */
    unsigned int count;  /* number of components */
    uint32_t stride;
    float *inline_buffer;
    bool inline_buffer_populated;
} VertexAttribute;

typedef struct SurfaceShape {
    unsigned int z_format;
    unsigned int color_format;
    unsigned int zeta_format;
    unsigned int log_width, log_height;
    unsigned int clip_x, clip_y;
    unsigned int clip_width, clip_height;
    unsigned int anti_aliasing;
} SurfaceShape;

typedef struct Surface {
    unsigned int pitch;
    uint32_t offset;
} Surface;

typedef struct ContextSurfaces2DState {
    uint32_t object_instance;
    uint32_t dma_image_source;
    uint32_t dma_image_dest;
    unsigned int color_format;
    unsigned int source_pitch, dest_pitch;
    uint32_t source_offset, dest_offset;
} ContextSurfaces2DState;

typedef struct ImageBlitState {
    uint32_t object_instance;
    uint32_t context_surfaces;
    unsigned int operation;
    unsigned int in_x, in_y;
    unsigned int out_x, out_y;
    unsigned int width, height;
} ImageBlitState;

typedef struct BetaState {
    uint32_t object_instance;
    uint32_t beta;
} BetaState;

/* What the engine did, for the report and for deciding what to build next.
 * A feature the pipeline does not implement is counted here and never
 * silently approximated. */
typedef struct KelvinStats {
    uint64_t methods, words, jumps, calls;
    uint64_t draws, draws_nop, prims, triangles, lines, pixels;
    uint64_t triangles_bounds_clamped;
    uint64_t clears, flips, semaphores, image_blits, notifies;
    uint64_t swizzled_clears;
    /* Batches handed to the raster threads, and those that waited for
     * every earlier batch: scanned by one thread, sampling their own
     * target, or conflicting with an unfinished batch. */
    uint64_t raster_batches, raster_batches_single, raster_batches_snapshot,
        raster_batches_conflict;
    /* Vertices shaded, and those shaded on several threads at once. */
    uint64_t vertices, vertices_parallel;
    /* Draws whose vertices were shaded and triangles set up on the raster
     * threads after the GPU thread went on (kelvin_render.c draw jobs). */
    uint64_t draws_deferred;
    uint64_t unhandled_methods;
    uint64_t unsupported[64];
} KelvinStats;

/* Unimplemented features, indexed into KelvinStats.unsupported. */
enum {
    KU_CLASS = 0,           /* method on an object class we do not model */
    KU_KELVIN_METHOD,       /* Kelvin method with no handler */
    KU_FF_LIGHTING,
    KU_FF_TEXGEN,
    KU_FF_SKINNING,
    KU_FF_FOGGEN,
    KU_POINT_PRIM,
    KU_POLY_MODE,
    KU_TEX_MODE,
    KU_TEX_FORMAT,
    KU_TEX_3D,
    KU_TEX_CUBE,
    KU_TEX_BORDER,
    KU_SHADOW_MAP,
    KU_SURFACE_FORMAT,
    KU_SWIZZLED_SURFACE,
    KU_ZETA_FLOAT,
    KU_LOGIC_OP,
    KU_WINDOW_CLIP,
    KU_REPORT,
    KU_DMA_UNRESOLVED,
    KU_IMAGE_BLIT_OP,
    KU_CONVOLUTION,
    KU_DOTMAP_HILO,
    KU_SPECULAR_POWER,
    KU_COUNT
};

typedef struct KelvinState {
    /* PGRAPH registers, NV_PGRAPH_* offsets. The title reaches the same
     * array through the trapped PGRAPH page (kelvin_mmio.c). */
    uint32_t *regs;

    /* NV_PGRAPH_INTR and NV_PGRAPH_INTR_EN, which are not storage: reads of
     * INTR return what is pending and writes clear what they name. */
    uint32_t pending_interrupts;
    uint32_t enabled_interrupts;

    unsigned int subch_class[KELVIN_SUBCHANNELS];
    uint32_t subch_instance[KELVIN_SUBCHANNELS];

    ContextSurfaces2DState context_surfaces_2d;
    ImageBlitState image_blit;
    BetaState beta;
    uint32_t kelvin_object_instance;

    uint32_t dma_color, dma_zeta;
    Surface surface_color, surface_zeta;
    unsigned int surface_type;
    SurfaceShape surface_shape;

    uint32_t dma_a, dma_b;
    bool texture_matrix_enable[NV2A_MAX_TEXTURES];

    uint32_t dma_state, dma_notifies, dma_semaphore, dma_report;
    bool zpass_pixel_count_enable;
    uint32_t dma_vertex_a, dma_vertex_b;

    uint32_t primitive_mode;
    bool enable_vertex_program_write;

    uint32_t vertex_state_shader_v0[4];
    uint32_t program_data[NV2A_MAX_TRANSFORM_PROGRAM_LENGTH][VSH_TOKEN_SIZE];
    uint32_t program_writes;         /* words written to program_data */
    uint32_t vsh_constants[NV2A_VERTEXSHADER_CONSTANTS][4];
    bool vsh_constants_dirty[NV2A_VERTEXSHADER_CONSTANTS];

    uint32_t ltctxa[NV2A_LTCTXA_COUNT][4];
    uint32_t ltctxb[NV2A_LTCTXB_COUNT][4];
    uint32_t ltc1[NV2A_LTC1_COUNT][4];

    float material_alpha;
    float light_infinite_half_vector[NV2A_MAX_LIGHTS][3];
    float light_infinite_direction[NV2A_MAX_LIGHTS][3];
    float light_local_position[NV2A_MAX_LIGHTS][3];
    float light_local_attenuation[NV2A_MAX_LIGHTS][3];
    float specular_params[6];
    float specular_power;
    float specular_params_back[6];
    float specular_power_back;
    float point_params[8];

    VertexAttribute vertex_attributes[NV2A_VERTEXSHADER_ATTRIBUTES];
    uint16_t compressed_attrs;

    unsigned int inline_array_length;
    uint32_t inline_array[NV2A_MAX_BATCH_LENGTH];
    unsigned int inline_elements_length;
    uint32_t inline_elements[NV2A_MAX_BATCH_LENGTH];
    unsigned int inline_buffer_length;

    unsigned int draw_arrays_length;
    unsigned int draw_arrays_min_start;
    unsigned int draw_arrays_max_count;
    int32_t draw_arrays_start[1250];
    int32_t draw_arrays_count[1250];
    bool draw_arrays_prevent_connect;

    bool waiting_for_flip;
    bool waiting_for_nop;
    bool waiting_for_context_switch;

    KelvinStats stats;
} KelvinState;

extern KelvinState g_kelvin;

/* For the small helpers of the per-pixel path, which the compiler would
 * otherwise call. */
#ifdef _MSC_VER
#define KELVIN_INLINE static __forceinline
#else
#define KELVIN_INLINE static inline __attribute__((always_inline))
#endif

static inline uint32_t pgraph_reg_r(KelvinState *pg, unsigned int r)
{
    return pg->regs[r / 4];
}

static inline void pgraph_reg_w(KelvinState *pg, unsigned int r, uint32_t v)
{
    pg->regs[r / 4] = v;
}

#define PG_GET_MASK(reg, mask) GET_MASK(pgraph_reg_r(pg, reg), mask)
#define PG_SET_MASK(reg, mask, value)        \
    do {                                     \
        uint32_t rv = pgraph_reg_r(pg, reg); \
        SET_MASK(rv, mask, value);           \
        pgraph_reg_w(pg, reg, rv);           \
    } while (0)

static inline float kelvin_u2f(uint32_t v)
{
    float f;
    memcpy(&f, &v, 4);
    return f;
}

static inline uint32_t kelvin_f2u(float f)
{
    uint32_t v;
    memcpy(&v, &f, 4);
    return v;
}

/* Count an unimplemented feature, and say so the first time. */
void kelvin_unsupported(int what, const char *detail, uint32_t value);

/* Guest memory. */
uint8_t *kelvin_guest(uint32_t va);
uint8_t *kelvin_aperture(uint32_t offset);
uint32_t kelvin_aperture_r32(uint32_t offset);

/* Objects: instance address of a RAMHT handle, and the DMA objects there. */
bool kelvin_ramht_lookup(uint32_t handle, uint32_t *instance,
                         unsigned int *engine);
uint32_t kelvin_pramin_r32(uint32_t offset);
/* Guest VA of `offset` bytes into the DMA object at `instance` (an instance
 * address, as PGRAPH receives it), or 0 if out of range. */
uint32_t kelvin_dma_va(uint32_t instance, uint32_t offset, uint32_t bytes);
uint32_t kelvin_dma_limit(uint32_t instance);

/* Registers with behaviour (kelvin_mmio.c). The lock serialises the engine
 * against the title's register accesses, as xemu's pgraph lock does; every
 * method runs under it, and so does every trapped access. */
void kelvin_lock(void);
void kelvin_unlock(void);
void kelvin_lock_yield(void);
/* Whether a register access or vblank is waiting for the lock. */
bool kelvin_lock_wanted(void);
bool kelvin_mmio_start(KelvinState *pg);
void kelvin_mmio_report(void);
/* Recompute the interrupt line after a source changed. Lock held. */
void kelvin_irq_update(void);
/* Before a method on channel `chid`: PGRAPH holds another channel's context,
 * so raise the context-switch interrupt and wait for the title to load it
 * (xemu pgraph_context_switch). Lock held. */
void kelvin_context_switch(KelvinState *pg, unsigned int chid);

/* PGRAPH, ported from xemu. */
void kelvin_pgraph_init(KelvinState *pg, uint32_t *regs);
bool kelvin_pgraph_method_run(KelvinState *pg, unsigned int subchannel, unsigned int method,
                              const uint32_t *words, uint32_t n);
void kelvin_pgraph_method(KelvinState *pg, unsigned int subchannel,
                          unsigned int method, uint32_t parameter, bool inc);

/* Vertex bookkeeping (xemu pgraph/vertex.c). */
void pgraph_update_inline_value(VertexAttribute *attr, const uint8_t *data);
void pgraph_allocate_inline_buffer_vertices(KelvinState *pg, unsigned int attr);
void pgraph_finish_inline_buffer_vertex(KelvinState *pg);
void pgraph_reset_inline_buffers(KelvinState *pg);
void pgraph_reset_draw_arrays(KelvinState *pg);

/* The renderer: what xemu's renderer ops do, on the CPU. */
void kelvin_render_clear_surface(KelvinState *pg, uint32_t parameter);
void kelvin_render_draw_begin(KelvinState *pg);
void kelvin_render_draw_end(KelvinState *pg);
void kelvin_render_flush_draw(KelvinState *pg);
void kelvin_render_image_blit(KelvinState *pg);
void kelvin_render_flip_stall(KelvinState *pg,
                              uint32_t aspect_w, uint32_t aspect_h);
/* GPU-pusher current-method metadata; never sampled from title state. */
extern uint32_t g_kelvin_frame_aspect_w, g_kelvin_frame_aspect_h;
void kelvin_render_semaphore_release(KelvinState *pg, uint32_t value);
/* Wait until every pixel handed to the raster threads is written. Anything
 * that touches surfaces or tells the title how far the GPU has got calls it
 * first. */
void kelvin_render_sync(void);

/* RECOMP_KELVIN_TIMING=1: where the GPU thread's time goes, in
 * QueryPerformanceCounter ticks, reported per flip by kelvin_report. */
enum {
    KT_LOOP,            /* the GPU thread's whole loop */
    KT_PUSHER,          /* inside pusher_run */
    KT_DRAW,            /* kelvin_render_flush_draw */
    KT_PRESHADE,        /* its parallel vertex shading */
    KT_SUBMIT,          /* its waits for queue room */
    KT_GEOMETRY,        /* waits for draw jobs to finish, wherever */
    KT_SYNC_CLEAR,      /* waits for the raster queue to empty, by cause */
    KT_SYNC_SEMAPHORE,
    KT_SYNC_FLIP,
    KT_SYNC_NOTIFY,
    KT_SYNC_IDLE,       /* the pusher reaching PUT */
    KT_SYNC_OTHER,
    KT_WAIT_PUT,        /* loops that read no word: nothing past GET, */
    KT_WAIT_FLIP,       /* a flip stall waiting for the vblank ISR, */
    KT_WAIT_ACK,        /* or a notify, context switch or FIFO access off */
    KT_COUNT
};
extern int g_kelvin_timing;
extern long long g_kelvin_ticks[KT_COUNT];
/* With timing: pixel positions scanned in triangles' bounds, fragments
 * inside them, and fragments written. */
extern volatile long long g_kelvin_scan[3];
/* The method the engine is executing, subchannel << 16 | method, for the
 * job trace. */
extern uint32_t g_kelvin_method;
long long kelvin_ticks(void);
/* kelvin_render_sync, charging the wait to one of the KT_SYNC_ causes. */
void kelvin_render_sync_for(int cause);
/* RECOMP_KELVIN_JOBTRACE: a GPU-thread event from t0 to t1 (kelvin_ticks):
 * 1 a draw, 2 a wait for a draw job, 3 a loop that read no word (detail:
 * the KT_WAIT_ cause), 4 pusher_run. */
void kelvin_render_trace_event(int kind, long long t0, long long t1, uint32_t detail);
/* Whether the job trace is recording (the traced frame). */
extern bool g_kelvin_tracing;
/* The pushbuffer address of the word the engine is executing, set by the
 * pusher before each method. */
extern uint32_t g_kelvin_word_get;
/* The GET to publish with the pusher at `get` after `words` words: never
 * past a draw whose vertices a draw job may still be reading from guest
 * memory, which the title may reuse once GET is past it. A draw job more
 * than a bounded number of words behind is finished first, so GET stays
 * well inside D3D's fence margin. */
uint32_t kelvin_render_get_bound(uint32_t get, uint64_t words);

/* Surface description, as the pipeline needs it. */
typedef struct KelvinTarget {
    uint8_t *color;          /* host pointer to the colour surface, or NULL */
    uint8_t *zeta;           /* host pointer to the zeta surface, or NULL */
    uint32_t color_va, zeta_va;
    unsigned int color_bpp, zeta_bpp;
    unsigned int color_pitch, zeta_pitch;
    bool swizzled;
    uint32_t swz_mask_x, swz_mask_y;     /* when swizzled */
    unsigned int width, height;          /* in stored pixels (AA-scaled) */
    unsigned int aa_x, aa_y;             /* anti-aliasing scale factors */
    int clip_x0, clip_y0, clip_x1, clip_y1;   /* scissor, stored pixels */
    /* Host raster size is independent of the unchanged guest allocation.
     * Zero scales retain the legacy AA factors (also for standalone tests). */
    double raster_x, raster_y;
    unsigned int guest_width, guest_height, guest_color_pitch, guest_zeta_pitch;
    size_t guest_color_bytes, guest_zeta_bytes;
    unsigned int host_color, host_zeta;   /* cache slot + 1, zero for guest */
} KelvinTarget;

bool kelvin_bind_target(KelvinState *pg, KelvinTarget *t, bool need_color,
                        bool need_zeta);

#endif
