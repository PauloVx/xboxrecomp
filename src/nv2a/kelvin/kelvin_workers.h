/*
 * Kelvin's raster workers: host threads that run jobs the GPU thread hands
 * them while it goes on with its own work.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */
#ifndef XBOX_NV2A_KELVIN_WORKERS_H
#define XBOX_NV2A_KELVIN_WORKERS_H

#include <stdint.h>

#define KWORK_MAX 64

/* Jobs submitted and not yet finished, at most; a power of two. Draws
 * touching few rows need many in flight to keep every worker busy. */
#define KWORK_QUEUE 64

/* A job is split into lanes, 0..KWORK_LANES-1: lane l of a job runs after
 * lane l of every job submitted before it, and different lanes run at the
 * same time. */
#define KWORK_LANES 128

/* The job starts once every job submitted before it has finished. */
#define KWORK_BARRIER 1u
/* A barrier that the next job waits for in turn. */
#define KWORK_EXCLUSIVE 2u

/* How many worker threads run jobs. RECOMP_KELVIN_THREADS counts the GPU
 * thread too, so 1 means no workers; the default is half the logical
 * processors, at most KWORK_MAX. The first call starts the workers, so make
 * it from the GPU thread. */
int kwork_workers(void);

/* Queue a job and return its number, counting from 0; wait only while
 * KWORK_QUEUE jobs are unfinished. run(arg, lane, worker) is called once for
 * each of the nlanes distinct lanes listed, by any worker, 0..workers-1,
 * or by the submitting thread (worker number `workers`) during a wait:
 * lane l of this job runs after lane l of every earlier job has returned,
 * and lanes of this job run alongside lanes of earlier and later jobs. With
 * KWORK_BARRIER or KWORK_EXCLUSIVE, or when the job before it was
 * KWORK_EXCLUSIVE, no lane starts until every earlier job has finished. A
 * begin callback makes the job a barrier; one worker calls begin(arg) before
 * any of its lanes run. The workers run with the submitting thread's
 * floating-point control word (MXCSR), so they compute exactly as it would.
 * arg must stay valid until the job has finished. Without workers the job
 * runs here and now, as worker 0. Only one non-reentrant thread may submit
 * or wait. Queued callbacks can run inline during its waits and must not
 * recursively submit, wait or drain, or wait for work depending on their
 * own unfinished callback. RECOMP_KELVIN_ASSIST_QUEUE=1 also helps old lanes
 * while the producer waits for queue space; the default only waits. */
long kwork_submit_lanes(void (*begin)(void *arg),
                        void (*run)(void *arg, int lane, int worker),
                        void *arg, const uint8_t *lanes, int nlanes,
                        unsigned int flags);

/* A job every worker takes part in: an exclusive job with one lane per
 * worker, calling run(arg, share, shares) for shares 0..shares-1, shares
 * being the number of workers (1, run here, without workers). */
void kwork_submit(void (*begin)(void *arg),
                  void (*run)(void *arg, int share, int shares), void *arg);

/* Call fn(arg, item, worker) once for each item, 0..items-1, on the workers
 * and this thread (worker number `workers`), and return when every call has
 * returned. It runs outside the queue, ahead of queued lanes, which is
 * right only for work that touches nothing queued jobs touch. Calls run with
 * this thread's MXCSR. Only the submitting thread may call it. */
void kwork_parallel(void (*fn)(void *arg, int item, int worker), void *arg,
                    int items);

/* Tasks submitted and not yet finished, at most; a power of two. */
#define KTASK_QUEUE 64

/* A task's phases: step(arg, phase, &run) returns how many items phase
 * `phase` (0, 1, ...) has and sets run; zero ends the task. It is called
 * for phase 0 by ktask_submit, and for each later phase by whichever thread
 * returns the last item of the one before, so a phase can size itself from
 * what the earlier ones made. run(arg, item, worker) is called once for
 * each item 0..items-1, on any worker or on the submitting thread (worker
 * number `workers`) while it waits in ktask_wait: the items of a phase run
 * at the same time, and each phase after every item of the one before has
 * returned. Workers take task items oldest task first, ahead of parallel
 * loops and lanes, so tasks must touch nothing queued lanes touch. Every
 * call runs with the submitting thread's MXCSR. Only one thread may submit;
 * it waits (running items) while KTASK_QUEUE tasks are unfinished. Without
 * workers the whole task runs here and now, as worker 0. Returns the task's
 * number, counting from 0. arg must stay valid until the task has
 * finished. */
typedef int (*ktask_step_fn)(void *arg, int phase,
                             void (**run)(void *arg, int item, int worker));
long ktask_submit(ktask_step_fn step, void *arg);

/* Whether the task has finished: every phase, every item returned. */
int ktask_done(long task);

/* Run items of unfinished tasks until the task has finished. Only the
 * submitting thread may call it. */
void ktask_wait(long task);

/* With RECOMP_KELVIN_TIMING set: QueryPerformanceCounter ticks the workers
 * (and the submitting thread helping) have spent running lanes, and parallel
 * loop and task items, summed over threads; zero otherwise. */
void kwork_busy_ticks(long long *lanes, long long *loops);

/* How many jobs have finished in submission order: every job numbered
 * below it has finished. */
long kwork_finished(void);

/* Help run queued lanes and return once every submitted job has finished.
 * Only the submitting thread may call it; its MXCSR is preserved. */
void kwork_drain(void);

/* Help run queued lanes and return once the job and every job submitted
 * before it have finished. Later queued lanes may also run while helping.
 * Only the submitting thread may call it; its MXCSR is preserved. */
void kwork_wait(long job);

#endif
