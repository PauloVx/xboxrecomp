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
bool krt_available(void);
void krt_shutdown(void);

/* The texture (created by D3D11, shared by NT handle) the captured triangles reach the D3D12 side in: two RGBA32F
 * texels a vertex, see kelvin_gpu_rt.inc. It holds width * height / 2 vertices. */
bool krt_set_geometry(HANDLE texture, unsigned width, unsigned height);

/* The texture (RGBA8, created by D3D11 with unordered access, shared by NT handle) the passes write their results to. */
bool krt_set_output(HANDLE texture, unsigned width, unsigned height);

/* One frame's captured world: `nverts` vertices, three a triangle, in `ndraws` draws of `count[i]` vertices from
 * `first[i]`; `opaque[i]` is zero for a draw whose texture's alpha cuts it out. The camera: `inv` the inverse of its
 * matrix's x, y and w rows with their offsets (KrtCb in kelvin_gpu_rt.inc), the viewport c58/c59 and the guest
 * window size. */
typedef struct KrtFrame {
    uint32_t nverts;
    unsigned ndraws;
    const uint32_t *first, *count;
    const uint8_t *opaque;
    float inv[12], c58[4], c59[4];
    float guest_w, guest_h;
    bool view;                         /* RECOMP_KGPU_RT_VIEW: the camera's view, traced, into the output */
} KrtFrame;

/* Called on the renderer's thread, after the D3D11 work that wrote the geometry texture has been issued. Builds the
 * scene's acceleration structure and runs the passes; D3D11 work issued afterwards waits for them. False if nothing
 * was written to the output. */
bool krt_frame(const KrtFrame *f);
#endif

#endif
