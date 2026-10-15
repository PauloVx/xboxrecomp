/**
 * XAudio2 Audio Output Backend
 *
 * Provides low-latency audio output via XAudio2 (Win7+).
 * Called from the APU monitor frame to submit mixed samples.
 * Falls back gracefully if XAudio2 is unavailable.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "apu_xaudio2.h"

/* The XAudio2 backend is Windows-only. On Linux all xa2_* functions are
 * stubbed to report inactive; real audio output via SDL2 comes later. */
#if defined(_WIN32)

#define COBJMACROS
#include <windows.h>
#include <xaudio2.h>

#pragma comment(lib, "xaudio2.lib")
#pragma comment(lib, "ole32.lib")

#define XA2_SAMPLE_RATE   48000
#define XA2_CHANNELS      2
#define XA2_BUF_SAMPLES   1024   /* ~21ms per submission */
#define XA2_NUM_BUFS      3
#define XA2_RECOVERY_RETRY_MS 1000

static IXAudio2               *g_xa2 = NULL;
static IXAudio2MasteringVoice *g_xa2_master = NULL;
static IXAudio2SourceVoice    *g_xa2_source = NULL;
static int16_t                 g_xa2_bufs[XA2_NUM_BUFS][XA2_BUF_SAMPLES][2];
static int                     g_xa2_next_buf = 0;
static int                     g_xa2_initialized = 0;
static int                     g_xa2_started = 0;
static int                     g_xa2_frames_written = 0;
static Xa2OutputStats           g_xa2_stats;
static volatile LONG           g_xa2_critical_errors;
static volatile LONG           g_xa2_critical_status;
static volatile LONG           g_xa2_critical_pending;
static int                     g_xa2_callback_registered;
static DWORD                   g_xa2_owner_thread;
static int                     g_xa2_com_initialized;
static ULONGLONG               g_xa2_retry_at;
static uint64_t                g_xa2_played_base;
static uint64_t                g_xa2_generation_submitted;
static uint64_t                g_xa2_generation_played;

/* The APU producer, rather than only XAudio2's engine, needs scheduling
 * headroom under a busy renderer. Optional MMCSS ownership follows the
 * submission thread and does not change graph recovery or COM ownership. */
typedef HANDLE (WINAPI *Xa2AvSetCharacteristics)(LPCWSTR, LPDWORD);
typedef BOOL (WINAPI *Xa2AvRevertCharacteristics)(HANDLE);
static HMODULE g_xa2_avrt;
static HANDLE g_xa2_mmcss;
static Xa2AvRevertCharacteristics g_xa2_av_revert;

static void xa2_register_producer(void)
{
    DWORD task_index = 0;
    g_xa2_avrt = LoadLibraryExW(L"avrt.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (g_xa2_avrt) {
        Xa2AvSetCharacteristics av_set = (Xa2AvSetCharacteristics)
            GetProcAddress(g_xa2_avrt, "AvSetMmThreadCharacteristicsW");
        g_xa2_av_revert = (Xa2AvRevertCharacteristics)
            GetProcAddress(g_xa2_avrt, "AvRevertMmThreadCharacteristics");
        if (av_set && g_xa2_av_revert)
            g_xa2_mmcss = av_set(L"Pro Audio", &task_index);
    }
    fprintf(stderr, "[XA2] APU producer MMCSS Pro Audio %s\n",
            g_xa2_mmcss ? "registered" : "unavailable");
}

static void xa2_unregister_producer(void)
{
    if (g_xa2_mmcss) g_xa2_av_revert(g_xa2_mmcss);
    g_xa2_mmcss = NULL;
    g_xa2_av_revert = NULL;
    if (g_xa2_avrt) FreeLibrary(g_xa2_avrt);
    g_xa2_avrt = NULL;
}

/* Audio callbacks must not log, wait or take the APU lock. The submission
 * thread reports these atomic observations with the other output counters. */
static void STDMETHODCALLTYPE xa2_pass_start(IXAudio2EngineCallback *callback)
{ (void)callback; }
static void STDMETHODCALLTYPE xa2_pass_end(IXAudio2EngineCallback *callback)
{ (void)callback; }
static void STDMETHODCALLTYPE xa2_critical_error(IXAudio2EngineCallback *callback, HRESULT error)
{
    (void)callback;
    InterlockedExchange(&g_xa2_critical_status, (LONG)error);
    InterlockedIncrement(&g_xa2_critical_errors);
    InterlockedExchange(&g_xa2_critical_pending, 1);
}
static IXAudio2EngineCallbackVtbl g_xa2_callback_vtable = {
    xa2_pass_start, xa2_pass_end, xa2_critical_error
};
static IXAudio2EngineCallback g_xa2_callback = { &g_xa2_callback_vtable };

/* DestroyVoice waits until the voice can no longer read its PCM buffers.
 * Release the whole old graph before any ring slot can be reused. */
static void xa2_release_graph(void)
{
    if (g_xa2 && g_xa2_callback_registered) {
        IXAudio2_UnregisterForCallbacks(g_xa2, &g_xa2_callback);
        g_xa2_callback_registered = 0;
    }
    if (g_xa2_source) {
        XAUDIO2_VOICE_STATE state;
        IXAudio2SourceVoice_GetState(g_xa2_source, &state, 0);
        if (state.SamplesPlayed > g_xa2_generation_played)
            g_xa2_generation_played = state.SamplesPlayed;
        g_xa2_source->lpVtbl->DestroyVoice(g_xa2_source);
        g_xa2_source = NULL;
    }
    if (g_xa2_master) {
        g_xa2_master->lpVtbl->DestroyVoice(g_xa2_master);
        g_xa2_master = NULL;
    }
    if (g_xa2) {
        IXAudio2_Release(g_xa2);
        g_xa2 = NULL;
    }
    g_xa2_played_base += g_xa2_generation_played;
    if (g_xa2_generation_submitted > g_xa2_generation_played)
        g_xa2_stats.discarded_samples +=
            g_xa2_generation_submitted - g_xa2_generation_played;
    g_xa2_stats.played_samples = g_xa2_played_base;
    g_xa2_stats.queued_buffers = 0;
    g_xa2_generation_submitted = g_xa2_generation_played = 0;
    g_xa2_started = 0;
}

/* Graph creation reuses the owner's existing COM lease. Failed stages release
 * every resource they acquired, without ending that apartment. */
static int xa2_create_graph(void)
{
    HRESULT hr;
    WAVEFORMATEX wfx = { 0 };

    hr = XAudio2Create(&g_xa2, 0, XAUDIO2_DEFAULT_PROCESSOR);
    if (FAILED(hr) || !g_xa2) {
        fprintf(stderr, "[XA2] XAudio2Create failed: 0x%08lX\n", hr);
        goto fail;
    }

    hr = IXAudio2_RegisterForCallbacks(g_xa2, &g_xa2_callback);
    if (FAILED(hr)) {
        fprintf(stderr, "[XA2] RegisterForCallbacks failed: 0x%08lX\n", hr);
        goto fail;
    }
    g_xa2_callback_registered = 1;

    hr = IXAudio2_CreateMasteringVoice(g_xa2, &g_xa2_master,
        XA2_CHANNELS, XA2_SAMPLE_RATE, 0, NULL, NULL, 0);
    if (FAILED(hr)) {
        fprintf(stderr, "[XA2] CreateMasteringVoice failed: 0x%08lX\n", hr);
        goto fail;
    }

    wfx.wFormatTag      = WAVE_FORMAT_PCM;
    wfx.nChannels       = XA2_CHANNELS;
    wfx.nSamplesPerSec  = XA2_SAMPLE_RATE;
    wfx.wBitsPerSample  = 16;
    wfx.nBlockAlign     = XA2_CHANNELS * 2;
    wfx.nAvgBytesPerSec = XA2_SAMPLE_RATE * wfx.nBlockAlign;

    hr = IXAudio2_CreateSourceVoice(g_xa2, &g_xa2_source,
        &wfx, 0, XAUDIO2_DEFAULT_FREQ_RATIO, NULL, NULL, NULL);
    if (FAILED(hr)) {
        fprintf(stderr, "[XA2] CreateSourceVoice failed: 0x%08lX\n", hr);
        goto fail;
    }

    g_xa2_next_buf = 0;
    g_xa2_started = 0;
    return 1;

fail:
    xa2_release_graph();
    return 0;
}

int xa2_init(void)
{
    HRESULT hr;
    if (g_xa2_initialized)
        return g_xa2_owner_thread == GetCurrentThreadId();
    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
        fprintf(stderr, "[XA2] CoInitializeEx failed: 0x%08lX\n", hr);
        return 0;
    }
    g_xa2_owner_thread = GetCurrentThreadId();
    g_xa2_com_initialized = SUCCEEDED(hr);
    g_xa2_frames_written = 0;
    g_xa2_played_base = 0;
    g_xa2_generation_submitted = g_xa2_generation_played = 0;
    g_xa2_retry_at = 0;
    memset(&g_xa2_stats, 0, sizeof(g_xa2_stats));
    InterlockedExchange(&g_xa2_critical_errors, 0);
    InterlockedExchange(&g_xa2_critical_status, 0);
    InterlockedExchange(&g_xa2_critical_pending, 0);
    if (!xa2_create_graph()) {
        if (g_xa2_com_initialized) CoUninitialize();
        g_xa2_com_initialized = 0;
        g_xa2_owner_thread = 0;
        return 0;
    }
    g_xa2_initialized = 1;
    xa2_register_producer();
    fprintf(stderr, "[XA2] XAudio2 initialized (%d Hz stereo 16-bit, %d x %d-sample buffers)\n",
            XA2_SAMPLE_RATE, XA2_NUM_BUFS, XA2_BUF_SAMPLES);
    return 1;
}

void xa2_shutdown(void)
{
    if (!g_xa2_initialized || g_xa2_owner_thread != GetCurrentThreadId()) return;
    xa2_release_graph();
    xa2_unregister_producer();
    InterlockedExchange(&g_xa2_critical_pending, 0);
    if (g_xa2_initialized)
        fprintf(stderr, "[XA2] Shut down (%d frames written)\n", g_xa2_frames_written);
    g_xa2_initialized = 0;
    if (g_xa2_com_initialized) CoUninitialize();
    g_xa2_com_initialized = 0;
    g_xa2_owner_thread = 0;
}

int xa2_is_active(void)
{
    return g_xa2_initialized;
}

static int xa2_recover_if_needed(void)
{
    if (InterlockedCompareExchange(&g_xa2_critical_pending, 0, 0)) {
        fprintf(stderr, "[XA2] Critical error: 0x%08lX; recreating output graph\n",
                (unsigned long)InterlockedCompareExchange(&g_xa2_critical_status, 0, 0));
        xa2_release_graph();
        /* Old engine callbacks have ended before clearing their signal. A
         * critical error from the newly created engine remains pending. */
        InterlockedExchange(&g_xa2_critical_pending, 0);
    }
    if (g_xa2_source) return 1;
    ULONGLONG now = GetTickCount64();
    if (now < g_xa2_retry_at) return 0;
    ++g_xa2_stats.recovery_attempts;
    if (!xa2_create_graph() ||
        InterlockedCompareExchange(&g_xa2_critical_pending, 0, 0)) {
        /* A replacement can fail critically during creation too. Its signal
         * belongs to the graph just released, so preserve the retry delay. */
        xa2_release_graph();
        InterlockedExchange(&g_xa2_critical_pending, 0);
        ++g_xa2_stats.recovery_failed;
        g_xa2_retry_at = GetTickCount64() + XA2_RECOVERY_RETRY_MS;
        return 0;
    }
    ++g_xa2_stats.recoveries;
    /* Also bound repeated asynchronous failures of newly created voices.
     * The initial graph's first critical error has no retry deadline. */
    g_xa2_retry_at = GetTickCount64() + XA2_RECOVERY_RETRY_MS;
    fprintf(stderr, "[XA2] Output graph recovered; priming fresh PCM\n");
    return 1;
}

static int xa2_start_if_primed(unsigned queued)
{
    if (g_xa2_started || queued < 2) return 1;
    HRESULT hr = IXAudio2SourceVoice_Start(g_xa2_source, 0, XAUDIO2_COMMIT_NOW);
    if (FAILED(hr)) {
        if (!g_xa2_stats.start_failed)
            fprintf(stderr, "[XA2] Start failed: 0x%08lX\n", hr);
        ++g_xa2_stats.start_failed;
        return 0;
    }
    g_xa2_started = 1;
    return 1;
}

/* Submit a buffer of mixed samples to XAudio2.
 * Called from APU frame thread. Returns 1 if buffer was submitted. */
int xa2_submit_samples(const int16_t *samples, int num_samples)
{
    XAUDIO2_VOICE_STATE state;
    XAUDIO2_BUFFER xbuf;
    int idx;
    int copy_samples;
    HRESULT hr;

    if (!g_xa2_initialized || g_xa2_owner_thread != GetCurrentThreadId() ||
        !samples || num_samples <= 0) return 0;
    if (!xa2_recover_if_needed()) return 0;

    IXAudio2SourceVoice_GetState(g_xa2_source, &state, XAUDIO2_VOICE_NOSAMPLESPLAYED);
    if (!state.BuffersQueued && g_xa2_generation_submitted) {
        ++g_xa2_stats.empty_queue;
        if (g_xa2_started) {
            /* No packet is owned by the voice. Restore startup headroom after
             * an underrun instead of chasing a single packet indefinitely. */
            hr = IXAudio2SourceVoice_Stop(g_xa2_source, 0, XAUDIO2_COMMIT_NOW);
            if (SUCCEEDED(hr)) {
                g_xa2_started = 0;
                ++g_xa2_stats.reprimes;
            } else {
                if (!g_xa2_stats.stop_failed)
                    fprintf(stderr, "[XA2] Stop failed: 0x%08lX\n", hr);
                ++g_xa2_stats.stop_failed;
            }
        }
    }
    /* Retry a failed start even if all retained packets fill the ring. */
    if (!xa2_start_if_primed(state.BuffersQueued)) return 0;
    if ((int)state.BuffersQueued >= XA2_NUM_BUFS) {
        ++g_xa2_stats.queue_full;
        return XA2_SUBMIT_BACKPRESSURE;
    }
    idx = g_xa2_next_buf;
    copy_samples = (num_samples > XA2_BUF_SAMPLES) ? XA2_BUF_SAMPLES : num_samples;
    memcpy(g_xa2_bufs[idx], samples, copy_samples * XA2_CHANNELS * sizeof(int16_t));

    memset(&xbuf, 0, sizeof(xbuf));
    xbuf.AudioBytes = copy_samples * XA2_CHANNELS * sizeof(int16_t);
    xbuf.pAudioData = (const BYTE *)g_xa2_bufs[idx];

    hr = IXAudio2SourceVoice_SubmitSourceBuffer(g_xa2_source, &xbuf, NULL);
    if (FAILED(hr)) {
        ++g_xa2_stats.submit_failed;
        return 0;
    }

    g_xa2_next_buf = (idx + 1) % XA2_NUM_BUFS;
    g_xa2_frames_written++;
    g_xa2_stats.submitted_samples += copy_samples;
    g_xa2_generation_submitted += copy_samples;
    /* Starting an empty voice made the producer chase the very last samples
     * of a single packet. Ordinary scheduling jitter could exhaust it even
     * when no packets were rejected. Prime two packets (42.7 ms capacity)
     * before playback, preserving one packet of scheduling headroom. */
    xa2_start_if_primed(state.BuffersQueued + 1);
    return XA2_SUBMIT_ACCEPTED;
}

int xa2_get_buffer_size(void)
{
    return XA2_BUF_SAMPLES;
}

void xa2_get_output_stats(Xa2OutputStats *stats)
{
    *stats = g_xa2_stats;
    stats->mmcss_registered = g_xa2_mmcss != NULL;
    stats->critical_errors = (unsigned)InterlockedCompareExchange(&g_xa2_critical_errors, 0, 0);
    stats->critical_error_status = (int32_t)InterlockedCompareExchange(&g_xa2_critical_status, 0, 0);
    if (g_xa2_initialized && g_xa2_source) {
        XAUDIO2_VOICE_STATE state;
        IXAudio2SourceVoice_GetState(g_xa2_source, &state, 0);
        if (state.SamplesPlayed > g_xa2_generation_played)
            g_xa2_generation_played = state.SamplesPlayed;
        stats->played_samples = g_xa2_played_base + g_xa2_generation_played;
        stats->queued_buffers = state.BuffersQueued;
    }
}

#else /* !_WIN32 -- POSIX stubs (no audio output yet) */

int  xa2_init(void)                                   { return 0; }
void xa2_shutdown(void)                               {}
int  xa2_is_active(void)                              { return 0; }
int  xa2_submit_samples(const int16_t *s, int n)      { (void)s; (void)n; return 0; }
int  xa2_get_buffer_size(void)                        { return 0; }
void xa2_get_output_stats(Xa2OutputStats *s)           { memset(s, 0, sizeof(*s)); }

#endif /* _WIN32 */
