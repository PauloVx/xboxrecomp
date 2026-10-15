/*
 * Kelvin: NV2A 3D engine on the CPU -- public interface for the runtime.
 *
 * Enabled with RECOMP_NV2A_KELVIN=1. See kelvin.c.
 */
#ifndef XBOX_NV2A_KELVIN_H
#define XBOX_NV2A_KELVIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Whether RECOMP_NV2A_KELVIN asks for the engine. */
int kelvin_enabled(void);

/* Whether the engine is running. While it is, it owns DMA_GET: nothing else
 * may acknowledge pushbuffer progress on its behalf. */
int kelvin_active(void);

/* Attach presentation aspect to one not-yet-published FLIP_STALL parameter.
 * The caller registers the exact physical guest VA before publishing PUT.
 * The GPU consumes it once at that word; no later title state is sampled.
 * Returns zero on invalid input, a duplicate outstanding VA or a full queue.
 * Zero aspect belongs to ordinary unmarked frames, so is not registered. */
int kelvin_mark_frame_aspect(uint32_t flip_data_va,
                            uint32_t aspect_w, uint32_t aspect_h);

/* Start the GPU thread. Call once the NV2A aperture is mapped. */
void kelvin_start(void);

/* Counters and unimplemented features seen so far, to stderr. */
void kelvin_report(void);

/* The GPU interrupt, for the kernel's interrupt thread.
 *
 * While the engine runs, the PMC, PGRAPH and PCRTC register pages have real
 * interrupt semantics (see kelvin_mmio.c), so the line is level-triggered as
 * on hardware: kelvin_irq_asserted() says whether the title's ISR should run
 * now, and kelvin_irq_event() is signalled whenever the line may have gone
 * up, so the interrupt thread need not wait out its tick. */
int kelvin_irq_asserted(void);
void *kelvin_irq_event(void);

/* The display's vertical blank: sets PCRTC_INTR_0's VBLANK bit through the
 * register model. */
void kelvin_vblank(void);

/* Whether the register at this aperture offset belongs to the register model
 * while the engine runs -- host code must not treat it as plain memory. */
int kelvin_owns_register(unsigned int offset);

#ifdef __cplusplus
}
#endif

#endif
