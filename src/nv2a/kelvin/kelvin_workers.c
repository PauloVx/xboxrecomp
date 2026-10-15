/*
 * Kelvin's raster workers: a fixed pool of host threads that run the jobs
 * the GPU thread submits, while the GPU thread goes on to the next draw.
 *
 * A job is a set of lanes. Each lane has its own queue of the jobs that
 * list it, in submission order, and one worker at a time owns a lane: it
 * claims the lane, runs every job waiting there that may start, in order,
 * and lets the lane go. So lane l of a job sees everything lane l of every
 * earlier job wrote, while other lanes -- of the same job, or of earlier and
 * later ones -- run on other workers meanwhile. The renderer's lanes are
 * rows of the render target: draws to one surface follow each other row by
 * row, and no worker waits for every row of one draw before it starts the
 * next.
 *
 * A barrier job waits instead until every job before it has finished; then
 * one worker runs its begin callback, and its lanes run. A job has finished
 * once all its lanes have. Jobs can finish out of order: `tail` counts those
 * finished in submission order. When the GPU thread reuses a job's slot, or
 * the memory a finished job used, no worker is still looking at it. The GPU
 * thread waits only while KWORK_QUEUE jobs are unfinished, or when it drains
 * the queue.
 * Only one non-reentrant producer may submit or wait. Its waits may run
 * callbacks inline; those callbacks must not submit, wait or drain recursively,
 * since that could wait for the callback's own unfinished lane.
 *
 * A parallel loop (kwork_parallel) bypasses the queue: workers take its
 * items between lanes, ahead of any lane still waiting, and the GPU thread
 * takes items too until the loop is done.
 *
 * Tasks (ktask_submit) bypass it too, ahead of parallel loops and lanes:
 * each is a sequence of phases, each phase a number of items any thread may
 * run, and the phases one after another. The GPU thread submits them and
 * does not wait, unless it needs one finished (ktask_wait), when it runs
 * items itself meanwhile.
 *
 * Waiting threads spin for a short while -- jobs follow each other closely,
 * and waking a sleeping thread costs more than many of them take -- then
 * sleep on a condition variable.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include <xmmintrin.h>

#include "kelvin_workers.h"

/* Normal spin budget in microseconds. The optional smaller budgets change
 * only when a waiter sleeps, retaining the same predicate/wakeup protocol.
 * Timed, not counted, because how long a pause takes differs several times
 * over between a P-core and an E-core. */
#define KWORK_SPIN_US 1000

static int spin_us_env(const char *env)
{
    if (env && strcmp(env, "50") == 0) return 50;
    if (env && strcmp(env, "250") == 0) return 250;
    return KWORK_SPIN_US;
}

static int assist_queue_env(const char *env)
{
    return env && strcmp(env, "1") == 0;
}

#ifdef _MSC_VER
#define KWORK_LINE __declspec(align(64))
#else
#define KWORK_LINE __attribute__((aligned(64)))
#endif

typedef struct KWorkSlot {
    void (*begin)(void *arg);
    void (*run_lane)(void *arg, int lane, int worker);
    void (*run_share)(void *arg, int share, int shares);
    void *arg;
    unsigned int mxcsr;
    LONG job;
    int barrier;
    volatile LONG begun;                /* 0, 1 while beginning, 2 begun */
    volatile LONG pending;              /* lanes not yet run */
    volatile LONG done;                 /* job + 1 once every lane has run */
} KWorkSlot;

/* The jobs listing a lane, in order: the submitting thread adds them at wr,
 * the worker owning the lane takes them at rd. It holds unfinished jobs
 * only, so never more than KWORK_QUEUE. */
typedef struct KWorkLane {
    /* Aligned, so no two lanes share a cache line. */
    KWORK_LINE volatile LONG owner;     /* 1 while a worker runs the lane */
    volatile LONG rd;
    volatile LONG wr;
    LONG job[KWORK_QUEUE];
} KWorkLane;

static struct {
    int started;
    int workers;
    int after_exclusive;                /* the last job was KWORK_EXCLUSIVE */
    KWorkSlot ring[KWORK_QUEUE];
    SRWLOCK lock;
    CONDITION_VARIABLE cv;              /* head, tail or gen moved */
    LONGLONG spin_ticks;
    int assist_queue;                   /* opt-in producer help at capacity */
    KWORK_LINE volatile LONG head;      /* jobs submitted */
    KWORK_LINE volatile LONG tail;      /* jobs finished, in order */
    /* Moves whenever a job is queued, a begin callback returns or tail
     * moves: whatever can let a lane's next job start. */
    KWORK_LINE volatile LONG gen;
    KWORK_LINE volatile LONG sleepers;
    KWORK_LINE KWorkLane lane[KWORK_LANES];
} s_kw;

/* The parallel loop (kwork_parallel), outside the queue: workers take its
 * items ahead of any lane. */
static struct {
    void (*fn)(void *arg, int item, int worker);
    void *arg;
    LONG items;
    unsigned int mxcsr;
    KWORK_LINE volatile LONG active;
    KWORK_LINE volatile LONG next;      /* items taken */
    KWORK_LINE volatile LONG remaining; /* items not yet returned */
    KWORK_LINE volatile LONG busy;      /* workers looking at the loop */
} s_par;

/* Tasks, in a ring of their own. A slot's claim word is its ticket -- a
 * count of the phases ever published in the slot, whichever task they
 * belonged to -- and the items of that phase taken so far. A phase's item
 * count and function are stored under its ticket (mod KTASK_TICKETS) before
 * the claim word shows the ticket, and stay until four phases later, after
 * the word has moved on: so a thread that reads the word, then the count,
 * and moves the word on by compare-exchange has claimed an item of the phase
 * it read, and no item is ever run twice. */
#define KTASK_TICKETS 4

typedef struct KTaskSlot {
    KWORK_LINE volatile LONG64 claim;   /* ticket << 32 | items taken */
    KWORK_LINE volatile LONG remaining; /* items of the phase not returned */
    volatile LONG done;                 /* task + 1 once finished */
    LONG task;                          /* the task in the slot */
    ktask_step_fn step;
    void *arg;
    unsigned int mxcsr;
    int phase;
    LONG items[KTASK_TICKETS];
    void (*run[KTASK_TICKETS])(void *arg, int item, int worker);
} KTaskSlot;

static struct {
    KTaskSlot ring[KTASK_QUEUE];
    KWORK_LINE volatile LONG head;      /* tasks submitted */
    KWORK_LINE volatile LONG tail;      /* tasks finished, in order */
} s_tk;

static int s_timing;
static volatile LONG64 s_lane_ticks, s_loop_ticks;

static long long ticks(void)
{
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

void kwork_busy_ticks(long long *lanes, long long *loops)
{
    *lanes = s_lane_ticks;
    *loops = s_loop_ticks;
}

static LONG load(volatile LONG *v)
{
    /* Poll without taking exclusive ownership of the shared cache line.
     * Interlocked publication of head, tail, gen and begun supplies the
     * release side; the acquire read keeps slot data behind that
     * publication. */
#ifdef _MSC_VER
    return ReadAcquire(v);
#else
    return __atomic_load_n(v, __ATOMIC_ACQUIRE);
#endif
}

#ifdef KWORK_TEST
/* Test probes are absent from production builds and do not mutate state. */
int kwork_test_parse_spin_us(const char *env)
{
    return spin_us_env(env);
}

long long kwork_test_spin_ticks(void)
{
    return s_kw.spin_ticks;
}

long kwork_test_sleepers(void)
{
    return load(&s_kw.sleepers);
}

int kwork_test_parse_assist_queue(const char *env)
{
    return assist_queue_env(env);
}

int kwork_test_assist_queue_enabled(void)
{
    return s_kw.assist_queue;
}
#endif

static void store_release(volatile LONG *v, LONG value)
{
#ifdef _MSC_VER
    WriteRelease(v, value);
#else
    __atomic_store_n(v, value, __ATOMIC_RELEASE);
#endif
}

/* Job numbers wrap; they are compared and indexed as unsigned, so the
 * queue's length must divide 2^32. */
typedef char kwork_queue_is_a_power_of_two[
    (KWORK_QUEUE & (KWORK_QUEUE - 1)) == 0 ? 1 : -1];
static LONG next_job(LONG job)
{
    return (LONG)((ULONG)job + 1);
}

static KWorkSlot *slot_of(LONG job)
{
    return &s_kw.ring[(ULONG)job % KWORK_QUEUE];
}

static int queue_has_room(LONG unused)
{
    (void)unused;
    return (ULONG)load(&s_kw.head) - (ULONG)load(&s_kw.tail) < KWORK_QUEUE;
}

static int queue_empty(LONG unused)
{
    (void)unused;
    return load(&s_kw.head) == load(&s_kw.tail);
}

static int gen_moved(LONG seen)
{
    return load(&s_kw.gen) != seen;
}

static int parallel_done(LONG unused)
{
    (void)unused;
    return load(&s_par.remaining) == 0;
}

static LONG64 load64(volatile LONG64 *v)
{
#ifdef _MSC_VER
    return ReadAcquire64(v);
#else
    return __atomic_load_n(v, __ATOMIC_ACQUIRE);
#endif
}

/* Spin until ready(arg), for a while, then sleep until it is. A waiter
 * counts itself among the sleepers before its last look, and whoever moves
 * head, tail or gen looks at the count after moving it. The sleeper
 * increment, those publications and the sleeper-count read retain full
 * interlocked barriers, so one side always sees the other: no wakeup is
 * lost. */
static void wait_for(int (*ready)(LONG), LONG arg)
{
    LARGE_INTEGER start, now;
    int k = 0;

    if (ready(arg)) {
        return;
    }
    QueryPerformanceCounter(&start);
    for (;;) {
        _mm_pause();
        if (ready(arg)) {
            return;
        }
        if (++k & 63) {
            continue;
        }
        QueryPerformanceCounter(&now);
        if (now.QuadPart - start.QuadPart > s_kw.spin_ticks) {
            break;
        }
    }
    AcquireSRWLockExclusive(&s_kw.lock);
    InterlockedIncrement(&s_kw.sleepers);
    while (!ready(arg)) {
        SleepConditionVariableSRW(&s_kw.cv, &s_kw.lock, INFINITE, 0);
    }
    InterlockedDecrement(&s_kw.sleepers);
    ReleaseSRWLockExclusive(&s_kw.lock);
}

/* After moving head, tail, gen or begun: a lane's next job may now start. */
static void notify(void)
{
    InterlockedIncrement(&s_kw.gen);
    /* Keep a full barrier for the sleep/wake handshake, independently of
     * the acquire-only polling above. */
    if (InterlockedCompareExchange(&s_kw.sleepers, 0, 0)) {
        AcquireSRWLockExclusive(&s_kw.lock);
        WakeAllConditionVariable(&s_kw.cv);
        ReleaseSRWLockExclusive(&s_kw.lock);
    }
}

/* Move tail over every job that has finished. A job marks itself done
 * before it looks at tail, and each move is a compare-exchange, so of two
 * jobs finishing together one always sees the other: tail never stops short
 * of a finished job. */
static void advance_tail(void)
{
    int moved = 0;

    for (;;) {
        LONG t = load(&s_kw.tail);
        if (load(&slot_of(t)->done) != next_job(t)) {
            break;
        }
        if (InterlockedCompareExchange(&s_kw.tail, next_job(t), t) == t) {
            moved = 1;
        }
    }
    if (moved) {
        notify();
    }
}

/* ---- tasks ---------------------------------------------------------------- */

typedef char ktask_queue_is_a_power_of_two[
    (KTASK_QUEUE & (KTASK_QUEUE - 1)) == 0 ? 1 : -1];

static KTaskSlot *task_slot(LONG task)
{
    return &s_tk.ring[(ULONG)task % KTASK_QUEUE];
}

/* Move the task tail over every task that has finished; as advance_tail. */
static void advance_task_tail(void)
{
    for (;;) {
        LONG t = load(&s_tk.tail);
        if (t == load(&s_tk.head) || load(&task_slot(t)->done) != next_job(t)) {
            break;
        }
        InterlockedCompareExchange(&s_tk.tail, next_job(t), t);
    }
}

/* Set the slot's next phase going: its items under the next ticket, then
 * the claim word. With no items, the task has finished. The caller alone
 * looks at the slot's phase state: it submitted the task, or returned the
 * previous phase's last item. */
static void start_phase(KTaskSlot *k)
{
    void (*run)(void *arg, int item, int worker) = NULL;
    int items;
    ULONG ticket = (ULONG)(load64(&k->claim) >> 32) + 1;

    _mm_setcsr(k->mxcsr);
    items = k->step(k->arg, k->phase, &run);
    if (items <= 0) {
        InterlockedExchange(&k->done, next_job(k->task));
        advance_task_tail();
        notify();
        return;
    }
    k->items[ticket % KTASK_TICKETS] = items;
    k->run[ticket % KTASK_TICKETS] = run;
    k->remaining = items;
    /* A full barrier: the count and function are visible first. */
    InterlockedExchange64(&k->claim, (LONG64)ticket << 32);
    notify();
}

/* Run one item of any unfinished task, oldest first; whether one ran. A
 * slot the search reaches late may hold a later task by then; an item of
 * that is as good. A finished task's items are all taken. */
static int run_task(int self)
{
    LONG t = load(&s_tk.tail), h = load(&s_tk.head);

    for (; t != h; t = next_job(t)) {
        KTaskSlot *k = task_slot(t);
        for (;;) {
            LONG64 w = load64(&k->claim);
            ULONG ticket = (ULONG)(w >> 32);
            LONG item = (LONG)(w & 0xFFFFFFFF);
            void (*run)(void *arg, int item, int worker);
            long long t0;

            if (item >= k->items[ticket % KTASK_TICKETS]) {
                break;
            }
            if (InterlockedCompareExchange64(&k->claim, w + 1, w) != w) {
                continue;
            }
            run = k->run[ticket % KTASK_TICKETS];
            t0 = s_timing ? ticks() : 0;
            _mm_setcsr(k->mxcsr);
            run(k->arg, (int)item, self);
            if (s_timing) {
                InterlockedExchangeAdd64(&s_loop_ticks, ticks() - t0);
            }
            if (InterlockedDecrement(&k->remaining) == 0) {
                /* The phase's last item: this thread starts the next. The
                 * task cannot finish, nor its slot be reused, meanwhile. */
                k->phase++;
                start_phase(k);
            }
            return 1;
        }
    }
    return 0;
}

int ktask_done(long task)
{
    LONG t = (LONG)task;
    /* A slot is reused only once the tail has passed its task. */
    return (LONG)(load(&s_tk.tail) - t) > 0 || load(&task_slot(t)->done) == next_job(t);
}

void ktask_wait(long task)
{
    int self = s_kw.workers;
    unsigned int mxcsr = _mm_getcsr();

    while (!ktask_done(task)) {
        LONG seen = load(&s_kw.gen);
        if (run_task(self)) {
            /* Items run with their task's MXCSR; this thread keeps its own. */
            _mm_setcsr(mxcsr);
            continue;
        }
        /* Every item left is running elsewhere: wait for a phase to start
         * or the task to finish, either of which moves gen. */
        if (!ktask_done(task)) {
            wait_for(gen_moved, seen);
        }
    }
}

long ktask_submit(ktask_step_fn step, void *arg)
{
    unsigned int mxcsr = _mm_getcsr();
    KTaskSlot *k;
    LONG task;

    if (!kwork_workers()) {
        /* Every phase here and now, as worker 0, each call with this
         * thread's MXCSR as it was. */
        int phase, items, i;
        for (phase = 0;; phase++) {
            void (*run)(void *arg, int item, int worker) = NULL;
            _mm_setcsr(mxcsr);
            items = step(arg, phase, &run);
            if (items <= 0) {
                break;
            }
            for (i = 0; i < items; i++) {
                _mm_setcsr(mxcsr);
                run(arg, i, 0);
            }
        }
        _mm_setcsr(mxcsr);
        task = s_tk.head;
        s_tk.head = next_job(task);
        s_tk.tail = s_tk.head;
        return task;
    }
    while ((ULONG)s_tk.head - (ULONG)load(&s_tk.tail) >= KTASK_QUEUE) {
        ktask_wait(load(&s_tk.tail));
    }
    task = s_tk.head;
    k = task_slot(task);
    k->task = task;
    k->step = step;
    k->arg = arg;
    k->mxcsr = mxcsr;
    k->phase = 0;
    k->done = 0;
    /* Counted before its first phase can finish it. */
    InterlockedIncrement(&s_tk.head);
    start_phase(k);
    return task;
}

/* Take items of the parallel loop until none are left. Whether any ran. */
static int run_parallel(int self)
{
    int ran = 0;
    LONG item;

    /* Counted first, so the next loop is not set up under this worker. */
    InterlockedIncrement(&s_par.busy);
    if (load(&s_par.active)) {
        _mm_setcsr(s_par.mxcsr);
        while ((item = InterlockedIncrement(&s_par.next) - 1) < s_par.items) {
            long long t0 = s_timing ? ticks() : 0;
            s_par.fn(s_par.arg, (int)item, self);
            if (s_timing) {
                InterlockedExchangeAdd64(&s_loop_ticks, ticks() - t0);
            }
            ran = 1;
            if (InterlockedDecrement(&s_par.remaining) == 0) {
                notify();
            }
        }
    }
    InterlockedDecrement(&s_par.busy);
    return ran;
}

/* Run the jobs waiting in a lane this worker owns, in order, until it is
 * empty or its next job may not start yet. Whether any ran. A parallel loop
 * waits for the end of the lane: leaving a lane early costs its rows' cache
 * locality, which costs more than the loop gains. */
static int run_lane(KWorkLane *ln, int lane, int self)
{
    LONG rd = ln->rd;
    int ran = 0;

    while (rd != load(&ln->wr)) {
        LONG job = ln->job[(ULONG)rd % KWORK_QUEUE];
        KWorkSlot *slot = slot_of(job);

        if (slot->barrier) {
            /* tail cannot pass a job with a lane still to run. */
            if (load(&s_kw.tail) != job) {
                break;
            }
            if (slot->begin && load(&slot->begun) != 2) {
                if (InterlockedCompareExchange(&slot->begun, 1, 0) != 0) {
                    break;               /* another worker is beginning it */
                }
                _mm_setcsr(slot->mxcsr);
                slot->begin(slot->arg);
                InterlockedExchange(&slot->begun, 2);
                notify();
            }
        }
        _mm_setcsr(slot->mxcsr);
        if (slot->run_share) {
            slot->run_share(slot->arg, lane, s_kw.workers);
        } else {
            slot->run_lane(slot->arg, lane, self);
        }
        rd = next_job(rd);
        /* Taken before the job can finish, so the lane has room for the
         * next job the GPU thread may then submit. */
        store_release(&ln->rd, rd);
        ran = 1;
        if (InterlockedDecrement(&slot->pending) == 0) {
            InterlockedExchange(&slot->done, next_job(job));
            advance_tail();
        }
    }
    return ran;
}

/* Claim a lane with jobs waiting, starting the search at *next, and run
 * them. Whether anything ran. */
static int run_some(int self, int *next)
{
    int i;

    for (i = 0; i < KWORK_LANES; i++) {
        int l = (*next + i) & (KWORK_LANES - 1);
        KWorkLane *ln = &s_kw.lane[l];
        int ran;

        if (load(&ln->wr) == load(&ln->rd) || load(&ln->owner)) {
            continue;
        }
        if (InterlockedCompareExchange(&ln->owner, 1, 0) != 0) {
            continue;
        }
        {
            long long t0 = s_timing ? ticks() : 0;
            ran = run_lane(ln, l, self);
            if (s_timing) {
                InterlockedExchangeAdd64(&s_lane_ticks, ticks() - t0);
            }
        }
        InterlockedExchange(&ln->owner, 0);
        if (ran) {
            *next = l;
            return 1;
        }
    }
    return 0;
}

extern void xbox_PinThread(const char *role);

static DWORD WINAPI kwork_thread(LPVOID param)
{
    int self = (int)(intptr_t)param;
    xbox_PinThread("WORKER");
    /* Start the workers' searches apart, and keep each on the rows it last
     * ran: they are still in its caches. */
    int next = (self * 8) & (KWORK_LANES - 1);

    for (;;) {
        /* Read before searching: anything that happens during a fruitless
         * search moves gen, and the search runs again. */
        LONG seen = load(&s_kw.gen);
        if (load(&s_tk.head) != load(&s_tk.tail) && run_task(self)) {
            continue;
        }
        if (load(&s_par.active) && run_parallel(self)) {
            continue;
        }
        if (!run_some(self, &next)) {
            wait_for(gen_moved, seen);
        }
    }
    return 0;
}

int kwork_workers(void)
{
    const char *env;
    SYSTEM_INFO si;
    LARGE_INTEGER f;
    int want, i, spin_us;

    if (s_kw.started) {
        return s_kw.workers;
    }
    s_kw.started = 1;
    GetSystemInfo(&si);
    env = getenv("RECOMP_KELVIN_THREADS");
    want = env && *env ? atoi(env) : (int)si.dwNumberOfProcessors / 2;
    if (want < 1) want = 1;
    if (want > KWORK_MAX) want = KWORK_MAX;
    {
        /* With the Direct3D backend the raster threads only shade the vertices of the few draws it cannot take, and
         * idle ones cost a measurable part of a frame (they keep polling for work). RECOMP_KELVIN_THREADS_FORCE=1
         * keeps the requested count. */
        extern _Bool kgpu_enabled(void);
        const char *force = getenv("RECOMP_KELVIN_THREADS_FORCE");
        if (kgpu_enabled() && want > 4 && !(force && *force && *force != '0')) {
            fprintf(stderr, "[KELVIN] GPU renderer: %d raster threads requested, using 4\n", want);
            want = 4;
        }
    }

    QueryPerformanceFrequency(&f);
    spin_us = spin_us_env(getenv("RECOMP_KELVIN_SPIN_US"));
    s_kw.spin_ticks = f.QuadPart * spin_us / 1000000;
    s_kw.assist_queue = assist_queue_env(getenv("RECOMP_KELVIN_ASSIST_QUEUE"));
    s_timing = getenv("RECOMP_KELVIN_TIMING") != NULL;
    InitializeSRWLock(&s_kw.lock);
    InitializeConditionVariable(&s_kw.cv);
    /* Worker numbers must be 0..workers-1 with no gaps, so stop at the
     * first thread that cannot be created and run with the ones that
     * were. */
    for (i = 0; i < want - 1; i++) {
        HANDLE h = CreateThread(NULL, 0, kwork_thread, (LPVOID)(intptr_t)i,
                                0, NULL);
        if (!h) {
            break;
        }
        CloseHandle(h);
    }
    s_kw.workers = i;
    fprintf(stderr, "[KELVIN] GPU thread and %d raster worker(s)"
                    " (RECOMP_KELVIN_THREADS)\n", s_kw.workers);
    if (spin_us != KWORK_SPIN_US) {
        fprintf(stderr, "[KELVIN] Worker spin budget %d us"
                        " (RECOMP_KELVIN_SPIN_US experiment)\n", spin_us);
    }
    if (s_kw.assist_queue) {
        fprintf(stderr, "[KELVIN] Queue capacity assistance enabled"
                        " (RECOMP_KELVIN_ASSIST_QUEUE experiment)\n");
    }
    fflush(stderr);
    return s_kw.workers;
}

static void wait_with_lanes(int (*ready)(LONG), LONG arg);

static void no_lane(void *arg, int lane, int worker)
{
    (void)arg;
    (void)lane;
    (void)worker;
}

static LONG submit(void (*begin)(void *arg),
                   void (*run_lane)(void *arg, int lane, int worker),
                   void (*run_share)(void *arg, int share, int shares),
                   void *arg, const uint8_t *lanes, int nlanes,
                   unsigned int flags)
{
    KWorkSlot *slot;
    LONG job;
    int i;

    /* Help only previously published jobs, before claiming the new slot.
     * The shared helper preserves this producer's MXCSR even if a begin
     * callback changed it during a search that ran no lane. */
    if (s_kw.assist_queue) {
        wait_with_lanes(queue_has_room, 0);
    } else {
        wait_for(queue_has_room, 0);
    }
    job = s_kw.head;
    slot = slot_of(job);
    slot->begin = begin;
    slot->run_lane = run_lane;
    slot->run_share = run_share;
    slot->arg = arg;
    slot->mxcsr = _mm_getcsr();
    slot->job = job;
    slot->barrier = begin || (flags & (KWORK_BARRIER | KWORK_EXCLUSIVE)) ||
                    s_kw.after_exclusive;
    s_kw.after_exclusive = (flags & KWORK_EXCLUSIVE) != 0;
    slot->begun = 0;
    slot->pending = nlanes;
    /* A full barrier: the slot is visible before any lane lists the job,
     * and head counts the job before it can finish. */
    InterlockedIncrement(&s_kw.head);
    for (i = 0; i < nlanes; i++) {
        KWorkLane *ln = &s_kw.lane[lanes[i] & (KWORK_LANES - 1)];
        LONG wr = ln->wr;
        ln->job[(ULONG)wr % KWORK_QUEUE] = job;
        store_release(&ln->wr, next_job(wr));
    }
    notify();
    return job;
}

/* Without workers every job runs as it is submitted; head and tail still
 * number them. */
static LONG ran_here(void)
{
    LONG job = s_kw.head;
    s_kw.head = next_job(job);
    s_kw.tail = s_kw.head;
    return job;
}

long kwork_submit_lanes(void (*begin)(void *arg),
                        void (*run)(void *arg, int lane, int worker),
                        void *arg, const uint8_t *lanes, int nlanes,
                        unsigned int flags)
{
    static const uint8_t lane0 = 0;
    int i;

    if (!kwork_workers()) {
        if (begin) {
            begin(arg);
        }
        for (i = 0; i < nlanes; i++) {
            run(arg, lanes[i], 0);
        }
        return ran_here();
    }
    if (nlanes <= 0) {
        /* Nothing to run, but it still takes its place in the order. */
        return submit(begin, no_lane, NULL, arg, &lane0, 1, flags);
    }
    return submit(begin, run, NULL, arg, lanes, nlanes, flags);
}

void kwork_submit(void (*begin)(void *arg),
                  void (*run)(void *arg, int share, int shares), void *arg)
{
    uint8_t lanes[KWORK_MAX];
    int i, n = kwork_workers();

    if (!n) {
        if (begin) {
            begin(arg);
        }
        run(arg, 0, 1);
        ran_here();
        return;
    }
    for (i = 0; i < n; i++) {
        lanes[i] = (uint8_t)i;
    }
    submit(begin, NULL, run, arg, lanes, n, KWORK_EXCLUSIVE);
}

void kwork_parallel(void (*fn)(void *arg, int item, int worker), void *arg,
                    int items)
{
    int i, workers = kwork_workers();

    if (items <= 0) {
        return;
    }
    if (!workers || items == 1) {
        for (i = 0; i < items; i++) {
            fn(arg, i, workers);
        }
        return;
    }
    /* No worker may still be looking at the previous loop. One that is
     * has run out of items and is about to leave; yield if it was
     * preempted there. */
    for (i = 0; load(&s_par.busy); i++) {
        if (i & 1023) {
            _mm_pause();
        } else if (i) {
            SwitchToThread();
        }
    }
    s_par.fn = fn;
    s_par.arg = arg;
    s_par.items = items;
    s_par.mxcsr = _mm_getcsr();
    s_par.next = 0;
    s_par.remaining = items;
    /* A full barrier: the loop is set up before any worker sees it. */
    InterlockedExchange(&s_par.active, 1);
    notify();
    run_parallel(workers);
    /* Items are short: this spins, and sleeps only if one is held up. */
    wait_for(parallel_done, 0);
    InterlockedExchange(&s_par.active, 0);
}

long kwork_finished(void)
{
    return load(&s_kw.tail);
}

static int job_finished(LONG job)
{
    return (LONG)(load(&s_kw.tail) - job) > 0;
}

/* The submitting thread has nothing else to do while it waits for raster
 * work. Take lanes through the same ownership and ordering checks as the
 * workers, preserving its control word even if a begin callback changed it
 * during a search that could not run a lane. Only the submitter calls this;
 * keeping its search position also keeps recently scanned rows in cache. */
static void wait_with_lanes(int (*ready)(LONG), LONG arg)
{
    static int next;
    unsigned int mxcsr = _mm_getcsr();
    int self = s_kw.workers;

    while (!ready(arg)) {
        LONG seen = load(&s_kw.gen);
        int ran = run_some(self, &next);
        _mm_setcsr(mxcsr);
        if (!ran && !ready(arg)) {
            /* Every runnable lane is owned elsewhere, or waits for earlier
             * work. Finishing it or publishing a begin callback moves gen. */
            wait_for(gen_moved, seen);
        }
    }
}

void kwork_wait(long job)
{
    if (kwork_workers()) {
        wait_with_lanes(job_finished, (LONG)job);
    }
}

void kwork_drain(void)
{
    if (kwork_workers()) {
        wait_with_lanes(queue_empty, 0);
    }
}
