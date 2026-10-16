#ifndef KELVIN_RT_H
#define KELVIN_RT_H

/* Ray tracing for the Kelvin D3D11 backend (kelvin_rt.c).
 *
 * DXR needs Direct3D 12, so a D3D12 device on the renderer's adapter does the tracing next to the D3D11 device that
 * draws. The two share textures through NT handles and order their work with one shared fence: the D3D11 side
 * signals it when what the rays read is ready, the D3D12 side waits on it, traces, and signals it again when what it
 * wrote may be read; the D3D11 side waits for that before it reads. Neither CPU waits for the other's GPU work. */

#include <stdbool.h>
#include <stdint.h>

#ifdef _WIN32
#define COBJMACROS
#include <d3d11_4.h>

/* Brings up the D3D12 device on the adapter with this LUID and checks it can trace rays (DXR tier 1.1, for inline
 * ray queries). `dev`/`ctx` are the renderer's; it keeps them. False, with the reason logged, when ray tracing is
 * unavailable: no d3d12.dll, no DXR, no compiled ray tracing shaders, or the two devices cannot share (as under some
 * Wine set-ups). */
bool krt_init(ID3D11Device *dev, ID3D11DeviceContext *ctx, LUID adapter);
/* Whether the adapter with this LUID could trace rays (DXR tier 1.1, and the shaders built in), without bringing
 * anything up: a D3D12 device is made, asked and released. */
bool krt_supported(LUID adapter);
bool krt_available(void);
void krt_shutdown(void);

/* The texture (created by D3D11, shared by NT handle) the captured triangles reach the D3D12 side in: two RGBA32F
 * texels a vertex, see kelvin_gpu_rt.inc. It holds width * height / 2 vertices. */
bool krt_set_geometry(HANDLE texture, unsigned width, unsigned height);

/* The texture (RGBA16F, created by D3D11 with unordered access, shared by NT handle) the passes write their results
 * to. */
bool krt_set_output(HANDLE texture, unsigned width, unsigned height);

/* A game texture (created by D3D11, shared by NT handle) the passes may shade hits with; its index for KrtFrame.
 * -1 when it cannot be opened. Removing it releases it once the GPU no longer uses it. */
int krt_tex_add(HANDLE texture);
void krt_tex_remove(int index);

/* KrtFrame.flags, a draw's */
enum {
    KRT_TEX0 = 1u, KRT_TEX1 = 2u,      /* textured on stage 0 (coordinates uv0) / stage 1 (uv1) */
    KRT_TEX0_TEXELS = 4u, KRT_TEX1_TEXELS = 8u, /* those coordinates count texels (a linear texture) */
    KRT_ALPHA_TEST = 16u,              /* alpha test: function in bits 8..10, reference in bits 16..23 */
    KRT_DYNAMIC = 32u                  /* drawn with a transform of its own (a character, a prop), not the level */
};

/* One frame's captured world: `nverts` vertices, three a triangle, in `ndraws` draws of `count[i]` vertices from
 * `first[i]`, textured with tex0[i]/tex1[i] (krt_tex_add's indices) as flags[i] says. The camera: `inv` the inverse
 * of its matrix's x, y and w rows with their offsets (KrtCb in kelvin_gpu_rt.inc), the viewport c58/c59 and the
 * guest window size. */
typedef struct KrtFrame {
    uint32_t nverts;
    unsigned ndraws;
    const uint32_t *first, *count, *tex0, *tex1, *flags;
    float inv[12], c58[4], c59[4];
    float vp[16];                      /* the camera matrix itself, rows c[10..13] */
    float zmax;                        /* the depth buffer's largest value */
    float guest_w, guest_h;
    int view;                          /* RECOMP_KGPU_RT_VIEW: the camera's view traced into the output (1 normals, 2 shaded) */
    bool reflect;                      /* reflections for the pixels the screen-space targets marked */
    bool calibrate;                    /* the picture input holds this frame: learn shading brightness from it */
    float light[4];                    /* towards the main light (xyz, world space) and the shadows' strength (w) */
    float ao, ao_radius, gi;           /* occlusion strength and reach, bounce-colour strength (0 off) */
    bool lighting;                     /* the depth input holds this frame: shadows, occlusion and bounce colour */
} KrtFrame;

/* The passes' per-frame textures, made by D3D11 at the output's size and shared by NT handle: the picture so far
 * (BGRA8), the screen-space reflection targets' weight (RGBA16F) and surface (RGBA32F), the depth buffer (R32F), and
 * the lighting factor the lighting pass writes (RGBA16F, unordered access). */
bool krt_set_inputs(HANDLE picture, HANDLE weight, HANDLE surface, HANDLE depth, HANDLE light);

/* Called on the renderer's thread, after the D3D11 work that wrote the geometry texture has been issued. Builds the
 * scene's acceleration structure and runs the passes; D3D11 work issued afterwards waits for them. False if nothing
 * was written to the output. */
bool krt_frame(const KrtFrame *f);
#endif

#endif
