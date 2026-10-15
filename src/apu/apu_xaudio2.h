/**
 * XAudio2 Audio Output Backend
 */
#ifndef APU_XAUDIO2_H
#define APU_XAUDIO2_H

#include <stdint.h>

/* Initialize and shut down on the APU submission thread. Returns 1 on success,
 * 0 on failure. That thread owns the COM apartment and all graph resources. */
int xa2_init(void);

/* Shut down XAudio2 and release all resources. */
void xa2_shutdown(void);

/* Returns 1 while selected, including a confirmed device-error recovery. */
int xa2_is_active(void);

enum {
    XA2_SUBMIT_ERROR = 0,
    XA2_SUBMIT_ACCEPTED = 1,
    XA2_SUBMIT_BACKPRESSURE = -1
};
/* Accepted means the backend owns this PCM, even if the subsequent Start
 * fails. Retry only BACKPRESSURE; ERROR preserves device recovery semantics. */
int xa2_submit_samples(const int16_t *samples, int num_samples);

/* Get the preferred buffer size in samples. */
int xa2_get_buffer_size(void);

typedef struct Xa2OutputStats {
    uint64_t submitted_samples, played_samples, queue_full, submit_failed;
    uint64_t start_failed, stop_failed, empty_queue, reprimes;
    uint64_t recovery_attempts, recoveries, recovery_failed, discarded_samples;
    unsigned critical_errors;
    int32_t critical_error_status;
    unsigned queued_buffers;
    unsigned mmcss_registered;
} Xa2OutputStats;
/* Read only on the APU submission thread (or after that thread has joined). */
void xa2_get_output_stats(Xa2OutputStats *stats);

#endif /* APU_XAUDIO2_H */
