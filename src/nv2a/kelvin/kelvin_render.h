/*
 * Kelvin CPU renderer: types shared by the vertex, raster, texture and pixel
 * stages. See kelvin_int.h for the overall design.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */
#ifndef XBOX_NV2A_KELVIN_RENDER_H
#define XBOX_NV2A_KELVIN_RENDER_H

#include "kelvin_int.h"

static inline double kelvin_raster_x(const KelvinTarget *t)
{ return t->raster_x > 0.0 ? t->raster_x : (double)t->aa_x; }
static inline double kelvin_raster_y(const KelvinTarget *t)
{ return t->raster_y > 0.0 ? t->raster_y : (double)t->aa_y; }

/* Clears and fragments address the same surface layout. Swizzled targets
 * interleave coordinate bits until each rectangular dimension is exhausted. */
static inline uint32_t kelvin_surface_deposit(uint32_t v, uint32_t mask)
{
    uint32_t out = 0, bit;
    for (bit = 1; mask; bit <<= 1) {
        uint32_t low = mask & (~mask + 1);
        if (v & bit) out |= low;
        mask &= mask - 1;
    }
    return out;
}

static inline uint8_t *kelvin_color_ptr(const KelvinTarget *t, int x, int y)
{
    if (t->swizzled) {
        return t->color + (kelvin_surface_deposit((uint32_t)x, t->swz_mask_x) |
                           kelvin_surface_deposit((uint32_t)y, t->swz_mask_y)) * t->color_bpp;
    }
    return t->color + (size_t)y * t->color_pitch + (size_t)x * t->color_bpp;
}

static inline uint8_t *kelvin_zeta_ptr(const KelvinTarget *t, int x, int y)
{
    if (t->swizzled) {
        return t->zeta + (kelvin_surface_deposit((uint32_t)x, t->swz_mask_x) |
                          kelvin_surface_deposit((uint32_t)y, t->swz_mask_y)) * t->zeta_bpp;
    }
    return t->zeta + (size_t)y * t->zeta_pitch + (size_t)x * t->zeta_bpp;
}

#define KTEX_MAX_LEVELS 16

/* How a texture format's bytes become RGBA. */
enum TexelKind {
    TK_NONE = 0,
    TK_Y8,          /* (Y, Y, Y, 1) */
    TK_AY8,         /* (Y, Y, Y, Y) */
    TK_A8,          /* (1, 1, 1, A) */
    TK_A8Y8,        /* byte0 = Y, byte1 = A */
    TK_G8B8,        /* GL RG8, swizzle R G R G */
    TK_R8B8,        /* GL RG8, swizzle G R R G */
    TK_1555,        /* A1R5G5B5 */
    TK_X555,        /* X1R5G5B5 */
    TK_4444,
    TK_565,
    TK_8888,        /* A8R8G8B8 in the dword */
    TK_X888,
    TK_ABGR,        /* A8B8G8R8: bytes R G B A */
    TK_BGRA,        /* B8G8R8A8: bytes A R G B */
    TK_RGBA,        /* R8G8B8A8: bytes A B G R */
    TK_I8,          /* palette index */
    TK_DXT1, TK_DXT3, TK_DXT5,
    TK_YUY2, TK_UYVY,
    TK_Y16,
    TK_UNSUPPORTED,
};

typedef struct KTexture {
    bool valid;
    unsigned int color_format, bpp, kind;
    bool linear, cubemap, border, depth_format;
    unsigned int dimensionality;
    unsigned int width, height, depth, pitch, levels;
    float min_lod, max_lod, lod_bias;
    unsigned int min_filter, mag_filter;
    unsigned int addr[3];
    uint32_t border_color;
    bool alphakill;
    unsigned int colorkey_mode;
    uint32_t base, face_stride;
    uint32_t level_offset[KTEX_MAX_LEVELS];
    unsigned int level_w[KTEX_MAX_LEVELS], level_h[KTEX_MAX_LEVELS];
    uint32_t mask_x[KTEX_MAX_LEVELS], mask_y[KTEX_MAX_LEVELS];
    uint32_t palette;
    unsigned int palette_len;
    const uint8_t *palette_mem;      /* guest memory at palette, or NULL */
    /* When set, the texels (not the palette) are read from this copy of the
     * texture's memory instead of from guest memory: a draw that samples a
     * surface it is drawing to sees the surface as it was before the draw,
     * as xemu's copy of a surface into the texture it is bound as gives. */
    const uint8_t *host;
    /* Otherwise guest memory at `base` (ktex_setup), so a texel's address
     * needs no call; NULL reads through kelvin_guest. */
    const uint8_t *mem;
    /* A full-screen render-target view keeps its guest width/height and
     * guest span above, but samples larger level-0 texels at host pitch. */
    size_t host_bytes;
} KTexture;

void ktex_init(void);
void ktex_setup(KelvinState *pg, int stage, KTexture *t);
bool ktex_lod_free(const KTexture *t);
float ktex_lod(const KTexture *t, float dsdx, float dtdx, float dsdy,
               float dtdy);
/* ktex_lod's last step: the LOD of a footprint of squared size rho2. */
float ktex_lod_rho2(float rho2);
void ktex_sample_2d(const KTexture *t, unsigned int face, float s, float tc,
                    float lod, float out[4]);
void ktex_sample_cube(const KTexture *t, float x, float y, float z,
                      float lod, float out[4]);
#if defined(__AVX2__)
/* ktex_sample_2d(t, 0, s[l], tc[l], lod, out) for eight lanes, for a valid
 * texture that samples the same at every LOD (ktex_lod_free, so any lod):
 * RGBA in out[channel][lane]. Lanes outside mask may hold anything. */
void ktex_sample_2d8(const KTexture *t, const float *s, const float *tc,
                     unsigned int mask, float out[4][8]);
/* ktex_sample_2d(t, 0, s[l], tc[l], lod[l], out) for eight lanes of a valid
 * texture, at any LOD. */
void ktex_sample_2d8_lod(const KTexture *t, const float *s, const float *tc,
                         const float *lod, unsigned int mask, float out[4][8]);
#endif

/* A vertex after the vertex stage: what xemu's vertex shader hands the
 * rasterizer (vtxPos and the vtx* varyings). */
typedef struct KVertex {
    float pos[4];        /* screen x, y (logical pixels), z (depth units), w */
    float d0[4], d1[4], b0[4], b1[4];
    float fog;
    float t[4][4];
    float pts;
} KVertex;

/* Number of floats interpolated per vertex (colours, fog, texcoords). */
#define KV_VARYINGS (4 * 4 + 1 + 16)

/* Decoded per-draw state for the pixel stage. */
typedef struct KCombinerInput { int reg, mod, chan; } KCombinerInput;
typedef struct KCombinerOutput {
    int ab, cd, muxsum, flags, ab_op, cd_op, muxsum_op, mapping;
} KCombinerOutput;
typedef struct KCombinerStage {
    KCombinerInput rgb[4], alpha[4];
    KCombinerOutput rgb_out, alpha_out;
    float c0[4], c1[4];
} KCombinerStage;

typedef struct KPixelState {
    int num_stages;
    int flags;
    KCombinerStage stage[8];
    bool final_enabled;
    KCombinerInput fin[7];
    bool clamp_sum, inv_v1, inv_r0;
    float final_c0[4], final_c1[4];

    int tex_mode[4];
    int dot_map[4];
    int input_tex[4];
    KTexture tex[4];
    bool rect_tex[4];
    /* The stage's texture samples the same at every LOD (ktex_lod_free):
     * its LOD is not worked out. */
    bool lod_free[4];
    bool compare_mode[4][4];
    uint32_t color_key[4], color_key_mask[4];
    float bump_mat[4][4], bump_scale[4], bump_offset[4];

    float fog_color[3];

    bool alpha_test;
    int alpha_func, alpha_ref;

    /* depth / stencil */
    bool depth_test, depth_write, stencil_test, stencil_write;
    int depth_func;
    int zeta_format;         /* NV097_SET_SURFACE_FORMAT_ZETA_* */
    bool z_float;
    bool z_perspective;
    float zmax;
    float zclip_min, zclip_max;
    bool depth_clip_cull;
    float depth_offset, depth_factor;
    int stencil_func, stencil_ref, stencil_mask_read, stencil_mask_write;
    int stencil_op_fail, stencil_op_zfail, stencil_op_zpass;

    /* output merger */
    bool blend;
    int blend_sfactor, blend_dfactor, blend_eqn;
    float blend_color[4];
    bool mask_r, mask_g, mask_b, mask_a;
    int color_format;        /* NV097_SET_SURFACE_FORMAT_COLOR_* */

    /* window clip */
    bool window_clip_exclusive;
    int clip_region[8][4];

    bool smooth;
} KPixelState;

void kpix_setup(KelvinState *pg, KPixelState *ps, const KelvinTarget *t);

/* A fragment's inputs: interpolated varyings and derivatives for LOD. */
typedef struct KFragment {
    float d0[4], d1[4], b0[4], b1[4];
    float fog;
    float t[4][4];
    float dtdx[4][4], dtdy[4][4];
    float z;                 /* depth in zeta units, before offset */
    int x, y;                /* stored pixel */
    bool front;
} KFragment;

/* Shade and merge one fragment. Returns true if it wrote anything. */
/* kpix_fragment in two: kpix_keep makes the tests that need only the
 * position and depth (window clip, depth clip, early depth test) and writes
 * nothing, turning *zvalue (the fragment's z) into the depth the rest uses;
 * if it keeps the fragment, kpix_shade(..., *zvalue) does the rest. So a
 * dropped fragment's other inputs need not be worked out. */
bool kpix_keep(const KPixelState *ps, const KelvinTarget *t, int x, int y,
               float *zvalue);
bool kpix_shade(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                float zvalue);
bool kpix_fragment(const KPixelState *ps, const KelvinTarget *t,
                   KFragment *f);
/* kpix_shade for f[0..n-1], n at most 8, distinct pixels of one triangle,
 * each with the zvalue kpix_keep left it: the same writes as kpix_shade on
 * each in turn. Returns how many fragments wrote anything. */
int kpix_shade8(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                const float *zvalue, int n);

/* Private prepared raster entry. KPixelState and DrawCtx remain unchanged.
 * A non-generic ID is valid only for the immutable ps that kpix_program
 * examined. Other callers use the original kpix_shade8 entry. */
typedef enum KPixelProgram {
    KPIX_PROGRAM_GENERIC,
    KPIX_PROGRAM_WORLD_FOG,
    KPIX_PROGRAM_BACKGROUND
} KPixelProgram;
KPixelProgram kpix_program(const KPixelState *ps);
int kpix_shade8_program(const KPixelState *ps, const KelvinTarget *t, KFragment *f,
                        const float *zvalue, int n, KPixelProgram program);

/* Vertex stage: transform `count` vertices into out. */
/* What the GPU vertex path needs of the current draw's transform program. */
typedef struct KVshGpuInfo {
    const void *program;          /* Nv2aVshProgram */
    uint32_t hash, start;
    const uint32_t *words;        /* the microcode up to the final instruction */
    uint32_t nwords;
    const float *constants;       /* NV2A_VERTEXSHADER_CONSTANTS x 4 */
    bool fog_enable;
    int fog_mode;
    float fog_param[2];
    bool specular_enable, ignore_specular_alpha;
    float point_size;
} KVshGpuInfo;
bool kvsh_gpu_info(KVshGpuInfo *out);

void kvsh_begin_draw(KelvinState *pg);
void kvsh_trace_program(void);
/* Whether this draw's vertices may be shaded on several threads at once. */
bool kvsh_parallel_ok(void);
/* Up to four vertices (n) through the draw's vertex program at once, with
 * kvsh_run's results for each; only when kvsh_parallel_ok. */
void kvsh_run4(KelvinState *pg,
               const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
               KVertex *const out[], int n);
#if defined(__AVX2__)
/* The same with up to eight vertices, on AVX2. */
void kvsh_run8(KelvinState *pg,
               const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
               KVertex *const out[], int n);
#define KVSH_LANES 8
#else
#define KVSH_LANES 4
#endif
void kvsh_run(KelvinState *pg, const float in[NV2A_VERTEXSHADER_ATTRIBUTES][4],
              KVertex *out);

/* What shading a draw's vertices through its program reads, so they can be
 * shaded after the draw, on another thread: the parsed program (kept while
 * the draw may still need it), the fog, specular and point-size state, and
 * the constants, which the caller copies. */
typedef struct KVshState {
    const void *program;          /* Nv2aVshProgram, or NULL: no output */
    const float *constants;       /* NV2A_VERTEXSHADER_CONSTANTS x 4 */
    bool fog_enable;
    int fog_mode;
    float fog_param[2];
    bool specular_enable, ignore_specular_alpha;
    float point_size;
} KVshState;

/* The current draw's state, only when kvsh_parallel_ok, with `constants`
 * for the constants; the kvsh_run*_state calls then give what kvsh_run,
 * kvsh_run4 and kvsh_run8 give for the draw, given the same constants. */
void kvsh_state(KVshState *out, const float *constants);
void kvsh_run_state(const KVshState *vs,
                    const float in[NV2A_VERTEXSHADER_ATTRIBUTES][4], KVertex *out);
void kvsh_run4_state(const KVshState *vs,
                     const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
                     KVertex *const out[], int n);
#if defined(__AVX2__)
void kvsh_run8_state(const KVshState *vs,
                     const float in[][NV2A_VERTEXSHADER_ATTRIBUTES][4],
                     KVertex *const out[], int n);
#endif

/* Finish every draw whose vertices are still being shaded on the raster
 * threads, and hand its triangles over, before a program it may use is
 * freed. */
void kelvin_render_geometry_sync(void);

#endif
