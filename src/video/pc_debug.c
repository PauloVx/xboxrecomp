#include "pc_debug.h"
#include "pc_video.h"
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>

#define REFRESH 0.25            /* seconds between new rows */

static SRWLOCK s_lock = SRWLOCK_INIT;
static double (*s_clock)(void);
static volatile LONG s_visible;
/* since the last rows */
static unsigned s_presents, s_publishes, s_latencies, s_gpu_frames;
static double s_interval_sum, s_interval_max, s_latency_sum, s_gpu_sum, s_last_present = -1.0, s_window_at = -1.0;
static unsigned s_draws, s_raster_w, s_raster_h, s_out_w, s_out_h;
static uint64_t s_vram;
static int s_ping = -1;
static PcDebugRow s_game[PC_DEBUG_GAME_ROWS];
static PcDebugRow s_rows[PC_DEBUG_ROWS];
static int s_nrows;
static unsigned s_rows_gen, s_rows_seen = (unsigned)-1;
static double s_game_frame_ms;
static ULONGLONG s_cpu_prev, s_cpu_wall_prev;

static double now_seconds(void)
{
    static LARGE_INTEGER frequency;
    LARGE_INTEGER t;
    if (s_clock) return s_clock();
    if (!frequency.QuadPart) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)frequency.QuadPart;
}

/* RECOMP_DEBUG_OVERLAY=1 shows it from the start */
static void read_environment(void)
{
    static volatile LONG done;
    if (!done && InterlockedCompareExchange(&done, 1, 0) == 0) {
        const char *e = getenv("RECOMP_DEBUG_OVERLAY");
        if (e && *e && *e != '0') InterlockedExchange(&s_visible, 1);
    }
}

void pc_debug_toggle(void) { read_environment(); InterlockedXor(&s_visible, 1); }
int pc_debug_visible(void) { read_environment(); return InterlockedCompareExchange(&s_visible, 0, 0) != 0; }

int64_t pc_debug_note_publish(void)
{
    const double now = now_seconds();
    if (!pc_debug_visible()) return (int64_t)(now * 1e6);
    AcquireSRWLockExclusive(&s_lock);
    s_publishes++;
    ReleaseSRWLockExclusive(&s_lock);
    return (int64_t)(now * 1e6);
}

void pc_debug_note_present(int64_t published, int first)
{
    const double now = now_seconds();
    if (!first || !pc_debug_visible()) { if (first) s_last_present = now; return; }
    AcquireSRWLockExclusive(&s_lock);
    if (s_last_present >= 0.0 && now - s_last_present < 1.0) {
        const double interval = now - s_last_present;
        s_interval_sum += interval;
        if (interval > s_interval_max) s_interval_max = interval;
        s_presents++;
    }
    s_last_present = now;
    if (published > 0 && now - (double)published * 1e-6 < 1.0) {
        s_latency_sum += now - (double)published * 1e-6;
        s_latencies++;
    }
    ReleaseSRWLockExclusive(&s_lock);
}

void pc_debug_note_vram(uint64_t bytes) { s_vram = bytes; }

void pc_debug_note_render(double gpu_ms, unsigned draws, unsigned raster_w, unsigned raster_h, unsigned out_w, unsigned out_h)
{
    AcquireSRWLockExclusive(&s_lock);
    if (gpu_ms >= 0.0) { s_gpu_sum += gpu_ms; s_gpu_frames++; }
    s_draws = draws;
    s_raster_w = raster_w; s_raster_h = raster_h; s_out_w = out_w; s_out_h = out_h;
    ReleaseSRWLockExclusive(&s_lock);
}

void pc_debug_note_ping(int ms) { InterlockedExchange((volatile LONG *)&s_ping, ms < 0 ? -1 : ms); }

void pc_debug_game_row(unsigned slot, const char *label, const char *fmt, ...)
{
    PcDebugRow row;
    va_list args;
    if (slot >= PC_DEBUG_GAME_ROWS || !pc_debug_visible()) return;
    memset(&row, 0, sizeof(row));
    if (label && *label) {
        snprintf(row.label, sizeof(row.label), "%s", label);
        va_start(args, fmt);
        vsnprintf(row.value, sizeof(row.value), fmt, args);
        va_end(args);
    }
    AcquireSRWLockExclusive(&s_lock);
    s_game[slot] = row;
    ReleaseSRWLockExclusive(&s_lock);
}

static void row(PcDebugRow *rows, int *n, int max, const char *label, const char *fmt, ...)
{
    va_list args;
    if (*n >= max) return;
    snprintf(rows[*n].label, sizeof(rows[*n].label), "%s", label);
    va_start(args, fmt);
    vsnprintf(rows[*n].value, sizeof(rows[*n].value), fmt, args);
    va_end(args);
    ++*n;
}

static const char *aa_name(unsigned aa)
{
    static const char *const names[PCV_AA_COUNT] = { "OFF", "FXAA", "SSAA", "DLAA", "DLSS QUALITY", "DLSS BALANCED",
                                                     "DLSS PERFORMANCE" };
    return aa < PCV_AA_COUNT ? names[aa] : "?";
}

/* Process CPU time over the wall time since the last call, as a share of every core. -1 the first time. */
static int cpu_percent(void)
{
    FILETIME created, exited, kernel, user, wall;
    ULONGLONG busy, clock;
    SYSTEM_INFO si;
    int percent = -1;
    if (s_clock || !GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return -1;
    GetSystemTimeAsFileTime(&wall);
    busy = ((ULONGLONG)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime) + ((ULONGLONG)user.dwHighDateTime << 32 | user.dwLowDateTime);
    clock = (ULONGLONG)wall.dwHighDateTime << 32 | wall.dwLowDateTime;
    GetSystemInfo(&si);
    if (s_cpu_wall_prev && clock > s_cpu_wall_prev && si.dwNumberOfProcessors)
        percent = (int)((busy - s_cpu_prev) * 100ull / ((clock - s_cpu_wall_prev) * si.dwNumberOfProcessors));
    s_cpu_prev = busy; s_cpu_wall_prev = clock;
    return percent;
}

static void build(double now)
{
    PcDebugRow rows[PC_DEBUG_ROWS];
    PcVideoSettings v;
    PROCESS_MEMORY_COUNTERS_EX memory;
    double span, interval, game_fps, latency, gpu;
    unsigned presents, draws, rw, rh, ow, oh, i;
    int n = 0, ping, cpu;
    uint64_t vram;
    PcDebugRow game[PC_DEBUG_GAME_ROWS];
    AcquireSRWLockExclusive(&s_lock);
    span = s_window_at >= 0.0 ? now - s_window_at : 0.0;
    presents = s_presents;
    interval = presents ? s_interval_sum / presents : 0.0;
    game_fps = span > 0.0 ? s_publishes / span : 0.0;
    if (game_fps > 1.0) s_game_frame_ms = 1000.0 / game_fps;
    /* from the game starting the frame (about one of its frames before publishing it) to the screen */
    latency = s_latencies ? s_latency_sum / s_latencies * 1000.0 + s_game_frame_ms : -1.0;
    gpu = s_gpu_frames ? s_gpu_sum / s_gpu_frames : -1.0;
    draws = s_draws; rw = s_raster_w; rh = s_raster_h; ow = s_out_w; oh = s_out_h;
    memcpy(game, s_game, sizeof(game));
    {
        const double max_ms = s_interval_max * 1000.0;
        s_presents = s_publishes = s_latencies = s_gpu_frames = 0;
        s_interval_sum = s_interval_max = s_latency_sum = s_gpu_sum = 0.0;
        s_window_at = now;
        ReleaseSRWLockExclusive(&s_lock);
        row(rows, &n, PC_DEBUG_ROWS, "FPS", presents && interval > 0.0 ? "%.0f" : "-", interval > 0.0 ? 1.0 / interval : 0.0);
        row(rows, &n, PC_DEBUG_ROWS, "FRAME TIME", presents ? "%.1f MS   MAX %.1f" : "-", interval * 1000.0, max_ms);
    }
    row(rows, &n, PC_DEBUG_ROWS, "GAME", game_fps > 0.0 ? "%.0f FPS" : "-", game_fps);
    if (gpu >= 0.0) row(rows, &n, PC_DEBUG_ROWS, "GPU", "%.1f MS   %u DRAWS", gpu, draws);
    else row(rows, &n, PC_DEBUG_ROWS, "GPU", draws ? "%u DRAWS" : "-", draws);
    row(rows, &n, PC_DEBUG_ROWS, "LATENCY", latency >= 0.0 ? "%.0f MS" : "-", latency);
    ping = (int)InterlockedCompareExchange((volatile LONG *)&s_ping, 0, 0);
    if (ping >= 0) row(rows, &n, PC_DEBUG_ROWS, "PING", "%d MS", ping);      /* only once a network session reports one */
    if (rw && ow && (rw != ow || rh != oh)) row(rows, &n, PC_DEBUG_ROWS, "RESOLUTION", "%u X %u > %u X %u", rw, rh, ow, oh);
    else if (rw) row(rows, &n, PC_DEBUG_ROWS, "RESOLUTION", "%u X %u", rw, rh);
    pc_video_get(&v);
    row(rows, &n, PC_DEBUG_ROWS, "ANTI-ALIASING", "%s", aa_name(v.aa));
    row(rows, &n, PC_DEBUG_ROWS, "RAY TRACING", v.ray_tracing ? "ON" : "OFF");
    cpu = cpu_percent();
    row(rows, &n, PC_DEBUG_ROWS, "CPU", cpu >= 0 ? "%d %%" : "-", cpu);
    memset(&memory, 0, sizeof(memory));
    memory.cb = sizeof(memory);
    vram = s_vram;
    if (!s_clock && K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&memory, sizeof(memory))) {
        if (vram) row(rows, &n, PC_DEBUG_ROWS, "MEMORY", "%.2f GB   VRAM %.2f GB", memory.PrivateUsage / 1073741824.0, vram / 1073741824.0);
        else row(rows, &n, PC_DEBUG_ROWS, "MEMORY", "%.2f GB", memory.PrivateUsage / 1073741824.0);
    }
    for (i = 0; i < PC_DEBUG_GAME_ROWS; i++)
        if (game[i].label[0]) row(rows, &n, PC_DEBUG_ROWS, game[i].label, "%s", game[i].value);
    if (n != s_nrows || memcmp(rows, s_rows, sizeof(PcDebugRow) * (size_t)n)) {
        memcpy(s_rows, rows, sizeof(PcDebugRow) * (size_t)n);
        s_nrows = n;
        s_rows_gen++;
    }
}

int pc_debug_rows(PcDebugRow *rows, int max, int *changed)
{
    static double built_at = -1e9;
    const double now = now_seconds();
    int n;
    if (!pc_debug_visible()) { *changed = 0; s_window_at = -1.0; built_at = -1e9; s_rows_seen = (unsigned)-1; return 0; }
    if (s_window_at < 0.0) {
        AcquireSRWLockExclusive(&s_lock);
        s_presents = s_publishes = s_latencies = s_gpu_frames = 0;
        s_interval_sum = s_interval_max = s_latency_sum = s_gpu_sum = 0.0;
        s_window_at = now;
        ReleaseSRWLockExclusive(&s_lock);
    }
    if (now - built_at >= REFRESH) { build(now); built_at = now; }
    n = s_nrows < max ? s_nrows : max;
    memcpy(rows, s_rows, sizeof(PcDebugRow) * (size_t)n);
    *changed = s_rows_seen != s_rows_gen;
    s_rows_seen = s_rows_gen;
    return n;
}

void pc_debug_test_clock(double (*now)(void)) { s_clock = now; }

void pc_debug_test_reset(void)
{
    AcquireSRWLockExclusive(&s_lock);
    s_presents = s_publishes = s_latencies = s_gpu_frames = 0;
    s_interval_sum = s_interval_max = s_latency_sum = s_gpu_sum = 0.0;
    s_last_present = s_window_at = -1.0;
    s_draws = s_raster_w = s_raster_h = s_out_w = s_out_h = 0;
    s_vram = 0; s_ping = -1; s_nrows = 0; s_rows_gen = 0; s_rows_seen = (unsigned)-1; s_game_frame_ms = 0.0;
    s_cpu_prev = s_cpu_wall_prev = 0;
    memset(s_game, 0, sizeof(s_game));
    ReleaseSRWLockExclusive(&s_lock);
    InterlockedExchange(&s_visible, 0);
}

#else
void pc_debug_toggle(void) {}
int pc_debug_visible(void) { return 0; }
int64_t pc_debug_note_publish(void) { return 0; }
void pc_debug_note_present(int64_t published, int first) { (void)published; (void)first; }
void pc_debug_note_vram(uint64_t bytes) { (void)bytes; }
void pc_debug_note_render(double gpu_ms, unsigned draws, unsigned raster_w, unsigned raster_h, unsigned out_w, unsigned out_h)
{ (void)gpu_ms; (void)draws; (void)raster_w; (void)raster_h; (void)out_w; (void)out_h; }
void pc_debug_note_ping(int ms) { (void)ms; }
void pc_debug_game_row(unsigned slot, const char *label, const char *fmt, ...) { (void)slot; (void)label; (void)fmt; }
int pc_debug_rows(PcDebugRow *rows, int max, int *changed) { (void)rows; (void)max; *changed = 0; return 0; }
void pc_debug_test_clock(double (*now)(void)) { (void)now; }
void pc_debug_test_reset(void) {}
#endif
