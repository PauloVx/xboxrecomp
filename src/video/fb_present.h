#ifndef XBOXRECOMP_FB_PRESENT_H
#define XBOXRECOMP_FB_PRESENT_H

#include <stdint.h>
#include <stddef.h>

/* Publication accepts a completed, linear surface. The caller must finish all
 * raster writes first and keep source readable until this synchronous copy
 * returns. No guest pointer survives the call; the UI owns its pixel copy.
 * Numeric formats are the NV097 surface colour values, not D3DFMT values. */
void xbox_FramebufferPublish(const void *source, size_t source_bytes,
                             uint32_t fb_va, uint32_t width, uint32_t height,
                             uint32_t pitch, uint32_t nv_color_format);
/* A completed native frame may carry its own display aspect. The metadata is
 * copied with these pixels, so later gameplay cannot change a queued movie's
 * shape. Both aspect values zero select the ordinary presentation policy. */
void xbox_FramebufferPublishAspect(const void *source, size_t source_bytes,
    uint32_t fb_va, uint32_t width, uint32_t height, uint32_t pitch,
    uint32_t nv_color_format, uint32_t aspect_width, uint32_t aspect_height);
void xbox_FramebufferWindowSet(uint32_t fb_va, uint32_t pitch);
void xbox_FramebufferWindowStart(void);
int xbox_FramebufferDumpBmp(const char *path);

/* GPU-resident frames (RECOMP_KELVIN_GPU). The renderer owns a D3D11 device and a ring of
 * shared textures it copies each finished frame into; the presenter opens them on its own
 * device and draws them without a CPU copy. Two shared fences order the two devices:
 * `frame_fence` is signalled by the renderer once a slot's copy has executed, `done_fence`
 * by the presenter once it has finished sampling a slot, so the renderer can reuse it.
 * Handles are NT handles that stay valid for the process; the renderer keeps ownership. */
void xbox_FramebufferGpuAdapter(uint32_t luid_low, int32_t luid_high);
int xbox_FramebufferGpuRegister(void *const *texture_handles, unsigned count, void *frame_fence,
                                void *done_fence, uint32_t width, uint32_t height);
/* With on, the frame publishers do not call pc_input_frame_complete for rendered (non-movie) frames: the caller does, when
 * the game's flip reaches it. A renderer that finishes frames later than the game issues them needs this, because the
 * input context counts game frames, not presented ones. */
void xbox_FramebufferInputByCaller(int on);
void xbox_FramebufferPublishGpu(unsigned slot, uint64_t fence_value, uint32_t fb_va,
                                uint32_t aspect_width, uint32_t aspect_height);
/* Nonzero once the presenter could not open the shared resources: use the CPU path. */
int xbox_FramebufferGpuFailed(void);
/* Nonzero when a window is being presented (RECOMP_FB_WINDOW). */
int xbox_FramebufferGpuWanted(void);

/* One active movie owns presentation priority. Tokens reject publications from
 * a stopped/replaced decoder. BGRA bytes and encoded display aspect are owned
 * by the same synchronous copy; the caller calls WindowStart normally. */
uint64_t xbox_FramebufferMovieBegin(void);
void xbox_FramebufferMoviePublish(uint64_t token, const void *bgra,
    size_t source_bytes, uint32_t width, uint32_t height, uint32_t pitch,
    uint32_t aspect_width, uint32_t aspect_height);
void xbox_FramebufferMovieEnd(uint64_t token);
/* Nonzero while a movie owns the presenter (Escape then belongs to the movie's own skip). */
int xbox_FramebufferMovieActive(void);

/* One entry of the display's colour lookup (the gamma ramp the game writes to the DAC palette). Every presented frame
 * goes through it; until the game writes one it is the identity. */
void xbox_FramebufferGammaEntry(unsigned index, uint8_t red, uint8_t green, uint8_t blue);
/* The whole lookup at once (256 entries per channel), for a title that hands it over without the DAC's 768 writes. */
void xbox_FramebufferGammaRamp(const uint8_t *red, const uint8_t *green, const uint8_t *blue);

/* Asks the presenter's window to close, as its close button does (the game exits with 0). 0 = no window. */
int xbox_FramebufferRequestClose(void);

#endif
