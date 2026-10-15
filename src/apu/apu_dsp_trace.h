#pragma once

#include <stdint.h>

#define MCPX_DSP_TRACE_SEN              (1u << 0)
#define MCPX_DSP_TRACE_STACK_OVERFLOW   (1u << 1)
#define MCPX_DSP_TRACE_STACK_UNDERFLOW  (1u << 2)
#define MCPX_DSP_TRACE_STACK_IRQ        (1u << 3)
#define MCPX_DSP_TRACE_STACK_MASK       (MCPX_DSP_TRACE_STACK_OVERFLOW | \
                                         MCPX_DSP_TRACE_STACK_UNDERFLOW | \
                                         MCPX_DSP_TRACE_STACK_IRQ)
#define MCPX_DSP_TRACE_SAMPLE_PERIOD    60u
#define MCPX_DSP_TRACE_AUDIO_PERIOD     188u

static inline int mcpx_dsp_trace_sample_due(uint64_t frame)
{
    return frame == 1 || (frame % MCPX_DSP_TRACE_SAMPLE_PERIOD) == 0;
}

static inline int mcpx_dsp_trace_audio_sample_due(uint64_t block)
{
    return block == 1 || (block % MCPX_DSP_TRACE_AUDIO_PERIOD) == 0;
}

static inline uint32_t mcpx_dsp_trace_events(uint32_t omr, uint32_t sp,
                                              uint64_t pending_bits)
{
    uint32_t events = 0;
    if (omr & (1u << 20)) events |= MCPX_DSP_TRACE_SEN;
    if (sp & (1u << 4)) events |= MCPX_DSP_TRACE_STACK_OVERFLOW;
    if (sp & (1u << 5)) events |= MCPX_DSP_TRACE_STACK_UNDERFLOW;
    if (pending_bits & (1ull << 1)) events |= MCPX_DSP_TRACE_STACK_IRQ;
    return events;
}
