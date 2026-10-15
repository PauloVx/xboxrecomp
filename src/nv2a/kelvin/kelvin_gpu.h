/*
 * Kelvin D3D11 backend: rasterization, pixel shading and output merging of
 * the NV2A on the host GPU, for BLACK's draw subset.
 *
 * Enabled with RECOMP_KELVIN_GPU=1 together with RECOMP_NV2A_KELVIN=1. The
 * CPU vertex stage (kelvin_vsh.c) still transforms the vertices; this module
 * takes the transformed vertices of a draw, the decoded pixel/combiner state
 * (KPixelState) and the bound target, and draws them. Surfaces live on the GPU
 * at the requested host resolution; guest memory is not mirrored.
 *
 * Rendering entry points belong to the Kelvin GPU thread. The camera and
 * viewmodel publication functions below may be called by the title thread.
 */
#ifndef XBOX_NV2A_KELVIN_GPU_H
#define XBOX_NV2A_KELVIN_GPU_H

#include "kelvin_render.h"

/* RECOMP_KELVIN_GPU asks for the backend. */
bool kgpu_enabled(void);
/* The backend initialised and owns rendering. */
bool kgpu_active(void);
/* Create the device and shaders (idempotent). False: the CPU renderer stays. */
bool kgpu_init(void);
void kgpu_report(void);
/* Profiling: QPC ticks the GPU thread spent in each part of a draw. */
enum { KGP_VERTEX, KGP_TEX, KGP_UPLOAD, KGP_STATE, KGP_DRAW, KGP_FLIP, KGP_CLEAR, KGP_COUNT };
extern long long g_kgpu_ticks[KGP_COUNT];

/* The pusher read a new PUT: texture contents are revalidated once per kick. */
void kgpu_note_kick(void);

/* Host raster size for a full-screen guest target. */
void kgpu_host_size(unsigned int *w, unsigned int *h);

/* Slots of a vertex record, in float4 units; -1 where the draw has none. */
enum { KG_POS, KG_D0, KG_D1, KG_FOG, KG_T0, KG_T1, KG_T2, KG_T3, KG_SLOTS };

/* One vertex attribute of a draw whose vertices the GPU shades. */
typedef struct KGpuAttrSrc {
    uint32_t fmt;              /* 0: the inline value; 1 UB_D3D, 2 UB_OGL, 3 S1, 4 F, 5 S32K, 6 CMP */
    uint32_t count;            /* components */
    uint32_t stride;           /* bytes between elements */
    const uint8_t *data;       /* element 0 */
    int64_t lo, hi;            /* bytes of it the draw reads, relative to data */
    float inline_value[4];
} KGpuAttrSrc;

typedef struct KGpuVsDraw {
    const void *program;       /* Nv2aVshProgram */
    uint32_t hash, start;
    const float *constants;    /* 192 x 4 */
    bool fog_enable;
    int fog_mode;
    float fog_param[2];
    bool specular_enable, ignore_specular_alpha;
    KGpuAttrSrc attr[16];
} KGpuVsDraw;

/* The GPU can shade this program's vertices now (its shader is compiled). `words` is the program's microcode up
 * to its final instruction (nwords words), recorded so later runs can compile it before the game starts. */
bool kgpu_vs_enabled(void);
bool kgpu_vs_ready(const void *program, uint32_t hash, uint32_t start, const uint32_t *words, uint32_t nwords);

/* KGpuDraw.clean: what the caller knows is unchanged since its previous draw (so what derives from it is kept). A
 * caller that cannot tell leaves it zero. */
enum { KG_CLEAN_T = 1, KG_CLEAN_PS = 2 };
/* KGpuDraw.topology */
enum { KG_TOPO_LIST = 0, KG_TOPO_STRIP = 1 };

typedef struct KGpuDraw {
    const KelvinTarget *t;
    const KPixelState *ps;
    bool cull_enable, ccw_front;
    int cull_ctrl;
    /* Polygon offset: bias in zeta units (zero for none) and the slope factor. */
    float zbias, zslope;
    /* Vertex records: vcount records of `stride` float4s, laid out by slot. */
    const float *recs;
    uint32_t vcount, stride;
    int slot[KG_SLOTS];
    /* With gvs set the GPU shades the vertices: recs/vcount/stride/slot are unused
     * and indices are element numbers of the draw's vertex arrays. */
    const KGpuVsDraw *gvs;
    /* Triangle list over the records, or a triangle strip (restarts are 0xFFFFFFFF). */
    const uint32_t *indices;
    uint32_t icount;
    unsigned int topology;
    uint32_t index_limit;      /* every real index is below this; zero if not known (indices stay 32-bit) */
    uint32_t clean;
} KGpuDraw;

void kgpu_draw(const KGpuDraw *d);

/* NV097_CLEAR_SURFACE for the rectangle (host-stored pixels, scaled by the
 * caller's kelvin_raster scale), channel mask in bit order B G R A. */
void kgpu_clear(const KelvinTarget *t, bool color, bool depth, bool stencil,
                uint32_t color_value, uint32_t zs_value, unsigned int channel_mask,
                int x0, int y0, int x1, int y1, unsigned int color_format);

/* Viewmodel scale: draws made with one of these vertex programs (hash up to 8, as kgpu's vertex program hash) have their
 * window position scaled about the viewport centre by `scale` (the caller's ratio between the world camera's window and
 * the one the first-person model was made for). Pass count 0 to clear. */
void kgpu_set_viewmodel(const uint32_t *vs_hashes, int count, float scale);

/* AO position reconstruction: publish the real camera's horizontal/vertical
 * half-frustum after aspect, aim zoom and optional PC FOV. Invalid input disables
 * AO until a valid camera arrives. Safe on the title thread. */
void kgpu_set_ao_camera(float half_x, float half_y);

/* RECOMP_KGPU_COMPARE=N: draw every frame with both renderers and compare the
 * finished frame every N flips (640x480 host size only); the CPU frame is shown. */
bool kgpu_compare_enabled(void);
void kgpu_compare_frame(const KelvinTarget *t, unsigned int color_format);
/* RECOMP_KGPU_PASSCMP (lab diagnostic): called by the CPU path after it has drawn what the GPU just drew. */
void kgpu_pass_check(void);

/* Asynchronous submission (kelvin_gpu_async.inc): kgpu_draw, kgpu_clear and kgpu_flip queue their work for a thread
 * of their own. Everything the title can see of the GPU's progress goes through the queue as well. */
bool kgpu_async_active(void);
/* Wait until everything queued so far has been executed. */
void kgpu_drain(void);
/* Run fn(a, b) on the GPU thread behind everything queued before it (at once, if submission is synchronous). */
void kgpu_defer_call(void (*fn)(uintptr_t, uintptr_t), uintptr_t a, uintptr_t b);
/* The GET to publish given the pusher's `get`: no further than the oldest draw not yet executed. */
uint32_t kgpu_get_bound(uint32_t get);

/* FLIP_STALL: the target `t` holds the finished frame. */
void kgpu_flip(const KelvinTarget *t, unsigned int color_format,
               uint32_t aspect_w, uint32_t aspect_h);

#endif
