#ifndef KELVIN_DLSS_H
#define KELVIN_DLSS_H

/* NVIDIA DLSS (NGX) for the Kelvin D3D11 backend (kelvin_dlss.c), at the internal resolution (DLAA) or upscaling to it.
 *
 * Built only when the DLSS SDK is next to the sources (third_party/dlss, see the nv2a CMakeLists); without it, and on
 * a GPU or driver without DLSS, kdlss_available is false and the renderer keeps its own anti-aliasing. The runtime,
 * nvngx_dlss.dll, is loaded from the game's folder. */

#include <stdbool.h>

#ifdef _WIN32
#define COBJMACROS
#include <d3d11_4.h>

/* Quality: what the input size is against the output (DLAA: the same). */
enum { KDLSS_DLAA, KDLSS_QUALITY, KDLSS_BALANCED, KDLSS_PERFORMANCE };

/* Brings NGX up on the renderer's device, once; false (with the reason logged) when DLSS cannot run here. */
bool kdlss_init(ID3D11Device *dev);
bool kdlss_available(void);
/* The input size DLSS wants for an output of out_w x out_h at this quality (out_w x out_h for DLAA). */
void kdlss_render_size(int quality, unsigned out_w, unsigned out_h, unsigned *in_w, unsigned *in_h);
/* One frame: colour, depth and motion vectors (RG16F, render pixels, from the current to the previous position) at
 * in_w x in_h; output (with unordered access) at out_w x out_h. jitter: the sample's offset from the pixel centre this
 * frame, in render pixels. reset: the history is not this picture's (a cut, a new size). False when nothing ran. */
bool kdlss_evaluate(ID3D11DeviceContext *ctx, ID3D11Resource *color, ID3D11Resource *depth, ID3D11Resource *motion,
                    ID3D11Resource *output, unsigned in_w, unsigned in_h, unsigned out_w, unsigned out_h, int quality,
                    float jitter_x, float jitter_y, bool reset);
void kdlss_shutdown(void);
/* Which history the evaluations that follow use: split-screen views each keep their own (0 for the whole screen). */
#define KDLSS_SLOTS 3
void kdlss_select(int slot);
#endif

#endif
