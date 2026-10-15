/**
 * Host-only movie pump. Media Foundation stays on this decoder thread; each
 * decoded BGRA frame is copied into the same completed-frame presenter used by
 * gameplay. No guest RAM, private movie window or second GPU device is used.
 * The title still selects the file through the opt-in RECOMP_FMV_HOST route.
 */
#if defined(_WIN32)
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "video_player.h"

static volatile LONG s_playing;
static char s_path[MAX_PATH];
static int s_escape_was_down;

static int vp_skip_pressed(void)
{
    DWORD pid = 0;
    int down;
    GetWindowThreadProcessId(GetForegroundWindow(), &pid);
    down = pid == GetCurrentProcessId() && (GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0;
    if (down == s_escape_was_down) return 0;
    s_escape_was_down = down;
    return down; /* one press skips one movie; held Escape cannot skip the next */
}

static DWORD WINAPI vp_thread(LPVOID unused)
{
    uint64_t movie = 0, sequence = 0;
    DWORD last;
    unsigned frames = 0;
    (void)unused;
    if (video_init() != 0) goto done;
    if (video_open_pixels(s_path) != 0) {
        fprintf(stderr, "  [VIDEO] could not open %s\n", s_path);
        goto done;
    }
    movie = xbox_FramebufferMovieBegin();
    if (!movie) {
        fprintf(stderr, "  [VIDEO] host playback requires the framebuffer presenter\n");
        goto done;
    }
    xbox_FramebufferWindowStart();
    fprintf(stderr, "  [VIDEO] playing %s in the gameplay window\n", s_path);
    last = GetTickCount();
    while (InterlockedCompareExchange(&s_playing, 1, 1)) {
        VideoFramePixels frame;
        DWORD now = GetTickCount();
        float dt = (float)(now - last) / 1000.0f;
        last = now;
        if (vp_skip_pressed()) break;
        if (video_update(dt) < 0 || video_is_finished()) break;
        if (video_get_frame_pixels(&frame) && frame.sequence != sequence) {
            xbox_FramebufferMoviePublish(movie, frame.pixels, frame.bytes,
                frame.width, frame.height, frame.pitch,
                frame.aspect_width, frame.aspect_height);
            sequence = frame.sequence;
            ++frames;
            /* Evidence remains the original decoded pixels, before scaling. */
            if ((frames == 60 || frames == 80) && getenv("RECOMP_FMV_DUMP")) {
                char out[MAX_PATH];
                _snprintf_s(out, sizeof(out), _TRUNCATE, "%s.%u.bmp",
                    getenv("RECOMP_FMV_DUMP"), frames);
                video_dump_frame_bmp(out);
            }
        }
        Sleep(1);
    }
    fprintf(stderr, "  [VIDEO] finished %s after %u published frames\n", s_path, frames);
done:
    if (movie) xbox_FramebufferMovieEnd(movie);
    video_shutdown();
    /* No new decoder can start until all old borrowed pixels are retired. */
    InterlockedExchange(&s_playing, 0);
    return 0;
}

int xbox_VideoPlayFile(const char *host_path)
{
    HANDLE th;
    if (!host_path || !*host_path) return -1;
    if (InterlockedCompareExchange(&s_playing, 1, 0) != 0) return -1;
    strncpy(s_path, host_path, sizeof(s_path) - 1);
    s_path[sizeof(s_path) - 1] = '\0';
    th = CreateThread(NULL, 0, vp_thread, NULL, 0, NULL);
    if (!th) { InterlockedExchange(&s_playing, 0); return -1; }
    CloseHandle(th);
    return 0;
}
int xbox_VideoIsPlaying(void) { return InterlockedCompareExchange(&s_playing, 0, 0) != 0; }
#else
int xbox_VideoPlayFile(const char *p) { (void)p; return -1; }
int xbox_VideoIsPlaying(void) { return 0; }
#endif
