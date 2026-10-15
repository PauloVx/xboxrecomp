#ifndef XBOXRECOMP_KELVIN_POST_ORDER_H
#define XBOXRECOMP_KELVIN_POST_ORDER_H

#include <stdbool.h>
#include <math.h>

static inline bool kg_post_order_should_apply_ao_before_blend(unsigned int world_draws,
                                                               bool depth_write,
                                                               bool blend,
                                                               const bool *ao_done)
{
    if (!ao_done || *ao_done || world_draws < 4 || depth_write || !blend) return false;
    return true;
}

/* Frame-rate-independent focus response. A loading/pause gap invalidates the
 * old scene's focus instead of slowly blending it into an unrelated view. */
static inline float kg_post_focus_blend(double elapsed_seconds)
{
    if (!(elapsed_seconds > 0.0) || elapsed_seconds >= 0.5) return 1.0f;
    return (float)(1.0 - exp(-elapsed_seconds / 0.25));
}

#endif
