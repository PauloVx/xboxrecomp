#ifndef MCPX_APU_TIMING_H
#define MCPX_APU_TIMING_H

#include <stdint.h>

/* Scale QPC without overflowing after long Windows uptime. */
static inline int64_t mcpx_apu_clock_microseconds(int64_t count, int64_t frequency)
{
    return (count / frequency) * INT64_C(1000000) +
           (count % frequency) * INT64_C(1000000) / frequency;
}

/* Keep ordinary scheduling jitter as sample debt. Reset only after a long
 * interruption: catching up unbounded old audio would delay current gameplay.
 * 64 ms is the capacity of the three 1024-sample XAudio2 packets. */
static inline int mcpx_apu_rebase_deadline(int64_t *deadline, unsigned *fraction,
                                          int64_t now)
{
    if (!*deadline || (now > *deadline && now - *deadline > 64000)) {
        *deadline = now;
        *fraction = 0;
        return 1;
    }
    return 0;
}

/* 256 samples at 48 kHz: 5333 + 1/3 microseconds. Truncating every period
 * made the producer outrun the device by 337.5 ms over a 90-minute session. */
static inline void mcpx_apu_advance_deadline(int64_t *deadline, unsigned *fraction)
{
    *deadline += 5333;
    if (++*fraction == 3) {
        ++*deadline;
        *fraction = 0;
    }
}

#endif
