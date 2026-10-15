#ifndef KELVIN_RT_H
#define KELVIN_RT_H

/* Ray tracing for the Kelvin D3D11 backend (kelvin_rt.c).
 *
 * DXR needs Direct3D 12, so a D3D12 device on the renderer's adapter does the tracing next to the D3D11 device that
 * draws. The two share textures and buffers through NT handles and order their work with one shared fence: the
 * D3D11 side signals it when what the rays read is ready, the D3D12 side waits on it, traces, and signals it again
 * when what it wrote may be read. */

#include <stdbool.h>
#include <stdint.h>

#ifdef _WIN32
#define COBJMACROS
#include <d3d11_4.h>

/* Brings up the D3D12 device on the adapter with this LUID and checks it can trace rays (DXR tier 1.1, for inline
 * ray queries). `dev`/`ctx` are the renderer's; it keeps them. False, with the reason logged, when ray tracing is
 * unavailable: no d3d12.dll, no DXR, or the two devices cannot share (as under some Wine set-ups). */
bool krt_init(ID3D11Device *dev, ID3D11DeviceContext *ctx, LUID adapter);
bool krt_available(void);
void krt_shutdown(void);

/* The texture (created by D3D11, shared by NT handle) the captured triangles reach the D3D12 side in: two RGBA32F
 * texels a vertex, see kelvin_gpu_rt.inc. */
bool krt_set_geometry(HANDLE texture, unsigned width, unsigned height);
#endif

#endif
