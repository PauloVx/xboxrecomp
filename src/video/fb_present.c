/* Completed frames are copied by the rendering producer. Only the presentation
 * thread touches D3D11; drawing never holds the mailbox lock or reads guest RAM.
 * Display scaling and pacing do not change simulation or internal resolution. */
#include "fb_present.h"
#include "fb_present_policy.h"
#include "../input/pc_input.h"
#include "pc_video.h"
#include "pc_video_menu.h"
#include "pc_menu.h"
#include "pc_movie_skip.h"
#include "pc_photo.h"
#if defined(_WIN32)
#define COBJMACROS
#include <windows.h>
#include <d3d11.h>
#include <d3d11_4.h>
#include <dxgi1_5.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

enum { FB_NOT_STARTED, FB_RUNNING, FB_CLOSED };
typedef struct FbFrame {
    uint32_t width, height, va, aspect_w, aspect_h;
    uint64_t movie_token;
    uint64_t gpu_value;      /* a GPU-resident frame: no pixels, a slot of the shared ring */
    unsigned gpu_slot;
    int gpu;
    uint32_t pixels[];
} FbFrame;
typedef struct FbGpu {
    ID3D11Device *device; ID3D11DeviceContext *context;
    IDXGISwapChain1 *swap; ID3D11RenderTargetView *target;
    ID3D11Texture2D *texture; ID3D11ShaderResourceView *view;
    ID3D11VertexShader *vs, *vs_cinematic; ID3D11PixelShader *ps;
    ID3D11SamplerState *linear, *point; ID3D11RasterizerState *raster;
    unsigned width, height, tex_width, tex_height, tearing;
    ID3D11DeviceContext4 *ctx4; ID3D11Fence *frame_fence, *done_fence;
    ID3D11Texture2D *shared_tex[8]; ID3D11ShaderResourceView *shared_srv[8];
    LONG shared_gen;
    uint64_t done_signaled;
    /* the Video Settings page and its hint, drawn over the frame */
    ID3D11PixelShader *ovl_ps; ID3D11BlendState *ovl_blend;
    ID3D11Texture2D *ovl_tex; ID3D11ShaderResourceView *ovl_view;
    unsigned ovl_w, ovl_h; int overlay_on;
} FbGpu;
/* Test-only observers compile away in the runtime. The device fixture reads
 * the actual backbuffer before flip; it does not replace rendering behavior. */
#ifndef FB_PRESENT_BEFORE_PRESENT
#define FB_PRESENT_BEFORE_PRESENT(g, frame) ((void)0)
#endif
#ifndef FB_PRESENT_MESSAGE_HOOK
#define FB_PRESENT_MESSAGE_HOOK(h, m, w, l) 0
#endif
#define FB_GPU_SLOTS 8
static SRWLOCK s_gpu_lock = SRWLOCK_INIT;
static void *s_gpu_tex[FB_GPU_SLOTS], *s_gpu_frame_fence, *s_gpu_done_fence;
static unsigned s_gpu_count; static uint32_t s_gpu_w, s_gpu_h;
static volatile LONG s_gpu_gen, s_gpu_failed;
static LUID s_gpu_luid; static int s_gpu_luid_set;
static INIT_ONCE s_once = INIT_ONCE_STATIC_INIT;
static SRWLOCK s_frames_lock = SRWLOCK_INIT;
static volatile LONG s_state, s_close_requested, s_redraw, s_toggle;
static HANDLE s_frame_ready;
static HWND volatile s_window;      /* the presenter's window, once it exists */
static int s_enabled;
static FbFrame *s_pending, *s_displayed;
static uint64_t s_movie_token, s_movie_next;
static FbPresentPolicy s_policy;
static uint32_t s_pin_va;
static const char *s_dump;
static RECT s_window_rect;

static unsigned s_applied_w, s_applied_h;   /* the window size last set from the settings */
static const char *s_menu_shot;             /* RECOMP_VIDEO_MENU_SHOT: dump the back buffer whenever the page changes */
static unsigned s_menu_shots;

/* Everything in the presentation policy the Video Settings page can change. The window size and mode are
 * applied separately, because they act on the window. */
static void fb_policy_from_video(const PcVideoSettings *v)
{
    unsigned num, den;
    s_policy.width = v->window_width; s_policy.height = v->window_height;
    s_policy.scale = v->scale == PCV_SCALE_STRETCH ? FB_SCALE_STRETCH : v->scale == PCV_SCALE_INTEGER ? FB_SCALE_INTEGER : FB_SCALE_FIT;
    s_policy.nearest = v->nearest; s_policy.vsync = v->vsync; s_policy.fps_limit = v->fps_limit;
    pc_video_aspect_ratio(v->aspect, &num, &den);
    s_policy.aspect_w = num; s_policy.aspect_h = den;
    s_policy.borderless = v->borderless;
}

static BOOL CALLBACK fb_initialize(PINIT_ONCE once, PVOID arg, PVOID *context)
{
    const char *pin;
    (void)once; (void)arg; (void)context;
    s_enabled = getenv("RECOMP_FB_WINDOW") != NULL;
    {
        PcVideoSettings v;
        pc_video_get(&v);
        fb_policy_from_video(&v);
        s_applied_w = v.window_width; s_applied_h = v.window_height;
    }
    pin = getenv("RECOMP_FB_VA");
    s_pin_va = pin ? (uint32_t)strtoul(pin, NULL, 0) : 0;
    s_dump = getenv("RECOMP_FB_DUMP");
    s_menu_shot = getenv("RECOMP_VIDEO_MENU_SHOT");
    if (s_menu_shot && !*s_menu_shot) s_menu_shot = NULL;
    if (s_enabled) s_frame_ready = CreateEventA(NULL, FALSE, FALSE, NULL);
    return TRUE;
}
/* Legacy display-mode calls precede any completed pixels; do not sample RAM. */
void xbox_FramebufferWindowSet(uint32_t va, uint32_t pitch) { (void)va; (void)pitch; }

/* The renderer calls pc_input_frame_complete itself (see fb_publish). */
static volatile LONG s_input_by_caller;
void xbox_FramebufferInputByCaller(int on)
{
    InterlockedExchange(&s_input_by_caller, on ? 1 : 0);
}

static void fb_publish(const void *source, size_t source_bytes, uint32_t va,
    uint32_t width, uint32_t height, uint32_t pitch, uint32_t format,
    uint64_t movie_token, uint32_t aspect_w, uint32_t aspect_h)
{
    FbFrame *frame, *old;
    unsigned bpp, x, y;
    uint64_t row_bytes, required, count;
    int allowed;
    const uint8_t *bytes = (const uint8_t *)source;
    InitOnceExecuteOnce(&s_once, fb_initialize, NULL, NULL);
    if (!source || !width || !height ||
        width > 8192 || height > 8192 ||
        InterlockedCompareExchange(&s_state, 0, 0) == FB_CLOSED) return;
    if (movie_token && (!aspect_w || !aspect_h)) return;
    bpp = format >= 1 && format <= 3 ? 2 : format >= 4 && format <= 8 ? 4 : 0;
    if (!bpp) return; /* B8/G8B8 are not silently treated as display RGB. */
    row_bytes = (uint64_t)width * bpp;
    required = (uint64_t)(height - 1) * pitch + row_bytes;
    count = (uint64_t)width * height;
    if ((!aspect_w != !aspect_h) || pitch < row_bytes || required > source_bytes ||
        count > (SIZE_MAX - sizeof(*frame)) / sizeof(uint32_t)) return;
    AcquireSRWLockShared(&s_frames_lock);
    allowed = movie_token ? movie_token == s_movie_token : s_movie_token == 0;
    ReleaseSRWLockShared(&s_frames_lock);
    if (!allowed) return;
    /* This is a validated completed native/movie frame even in a headless
     * run. Input context follows frame completion, never display pacing. A renderer that finishes frames on a thread
     * of its own calls pc_input_frame_complete itself, at the flip, in the order the game polled (xbox_FramebufferInputByCaller). */
    if (movie_token || !s_input_by_caller) pc_input_frame_complete();
    if (!s_enabled || !s_frame_ready || (!movie_token && s_pin_va && s_pin_va != va)) return;
    frame = (FbFrame *)malloc(sizeof(*frame) + (size_t)count * sizeof(uint32_t));
    if (!frame) return;
    frame->width = width; frame->height = height; frame->va = va;
    frame->movie_token = movie_token; frame->aspect_w = aspect_w; frame->aspect_h = aspect_h;
    for (y = 0; y < height; ++y) {
        const uint8_t *row = bytes + (size_t)y * pitch;
        uint32_t *dst = frame->pixels + (size_t)y * width;
        if (bpp == 4) memcpy(dst, row, (size_t)width * 4);
        else for (x = 0; x < width; ++x) {
            uint16_t v; unsigned r, g, b;
            memcpy(&v, row + x * 2, sizeof(v));
            if (format == 3) { r = ((v >> 11) & 31) * 255u / 31u; g = ((v >> 5) & 63) * 255u / 63u; }
            else { r = ((v >> 10) & 31) * 255u / 31u; g = ((v >> 5) & 31) * 255u / 31u; }
            b = (v & 31) * 255u / 31u;
            dst[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
        }
    }
    AcquireSRWLockExclusive(&s_frames_lock);
    /* Closing can race the synchronous copy, but cannot leave a late frame
     * after the UI has freed its mailbox. */
    if (InterlockedCompareExchange(&s_state, 0, 0) == FB_CLOSED ||
        (movie_token ? movie_token != s_movie_token : s_movie_token != 0)) {
        ReleaseSRWLockExclusive(&s_frames_lock); free(frame); return;
    }
    old = s_pending; s_pending = frame;
    ReleaseSRWLockExclusive(&s_frames_lock);
    free(old); SetEvent(s_frame_ready);
}
void xbox_FramebufferGpuAdapter(uint32_t luid_low, int32_t luid_high)
{
    s_gpu_luid.LowPart = luid_low; s_gpu_luid.HighPart = luid_high; s_gpu_luid_set = 1;
}
int xbox_FramebufferGpuWanted(void)
{
    InitOnceExecuteOnce(&s_once, fb_initialize, NULL, NULL);
    return s_enabled && s_frame_ready;
}
int xbox_FramebufferGpuFailed(void) { return InterlockedCompareExchange(&s_gpu_failed, 0, 0) != 0; }
int xbox_FramebufferGpuRegister(void *const *textures, unsigned count, void *frame_fence,
                                void *done_fence, uint32_t width, uint32_t height)
{
    unsigned i;
    if (!count || count > FB_GPU_SLOTS || !frame_fence || !done_fence || !width || !height) return 0;
    AcquireSRWLockExclusive(&s_gpu_lock);
    for (i = 0; i < count; ++i) s_gpu_tex[i] = textures[i];
    s_gpu_count = count; s_gpu_frame_fence = frame_fence; s_gpu_done_fence = done_fence;
    s_gpu_w = width; s_gpu_h = height;
    InterlockedIncrement(&s_gpu_gen);
    ReleaseSRWLockExclusive(&s_gpu_lock);
    InterlockedExchange(&s_gpu_failed, 0);
    return 1;
}
void xbox_FramebufferPublishGpu(unsigned slot, uint64_t value, uint32_t va,
                                uint32_t aspect_w, uint32_t aspect_h)
{
    FbFrame *frame, *old;
    unsigned csr = _mm_getcsr();
    InitOnceExecuteOnce(&s_once, fb_initialize, NULL, NULL);
    if (!s_input_by_caller) pc_input_frame_complete();
    if (!s_enabled || !s_frame_ready || slot >= FB_GPU_SLOTS || (!aspect_w != !aspect_h) ||
        InterlockedCompareExchange(&s_state, 0, 0) == FB_CLOSED || (s_pin_va && s_pin_va != va)) {
        _mm_setcsr(csr); return;
    }
    frame = (FbFrame *)calloc(1, sizeof(*frame));
    if (!frame) { _mm_setcsr(csr); return; }
    AcquireSRWLockShared(&s_frames_lock);
    if (s_movie_token != 0) { ReleaseSRWLockShared(&s_frames_lock); free(frame); _mm_setcsr(csr); return; }
    ReleaseSRWLockShared(&s_frames_lock);
    frame->width = s_gpu_w; frame->height = s_gpu_h; frame->va = va;
    frame->aspect_w = aspect_w; frame->aspect_h = aspect_h;
    frame->gpu = 1; frame->gpu_slot = slot; frame->gpu_value = value;
    AcquireSRWLockExclusive(&s_frames_lock);
    if (InterlockedCompareExchange(&s_state, 0, 0) == FB_CLOSED || s_movie_token != 0) {
        ReleaseSRWLockExclusive(&s_frames_lock); free(frame); _mm_setcsr(csr); return;
    }
    old = s_pending; s_pending = frame;
    ReleaseSRWLockExclusive(&s_frames_lock);
    free(old); SetEvent(s_frame_ready);
    _mm_setcsr(csr);
}
void xbox_FramebufferPublish(const void *source, size_t source_bytes, uint32_t va,
    uint32_t width, uint32_t height, uint32_t pitch, uint32_t format)
{
    unsigned csr = _mm_getcsr();
    fb_publish(source, source_bytes, va, width, height, pitch, format, 0, 0, 0);
    _mm_setcsr(csr);
}
void xbox_FramebufferPublishAspect(const void *source, size_t source_bytes,
    uint32_t va, uint32_t width, uint32_t height, uint32_t pitch,
    uint32_t format, uint32_t aspect_w, uint32_t aspect_h)
{
    unsigned csr = _mm_getcsr();
    fb_publish(source, source_bytes, va, width, height, pitch, format,
        0, aspect_w, aspect_h);
    _mm_setcsr(csr);
}
uint64_t xbox_FramebufferMovieBegin(void)
{
    unsigned csr = _mm_getcsr();
    uint64_t token = 0;
    InitOnceExecuteOnce(&s_once, fb_initialize, NULL, NULL);
    AcquireSRWLockExclusive(&s_frames_lock);
    if (s_enabled && InterlockedCompareExchange(&s_state, 0, 0) != FB_CLOSED) {
        if (!++s_movie_next) ++s_movie_next;
        token = s_movie_token = s_movie_next;
        free(s_pending); s_pending = NULL;
    }
    ReleaseSRWLockExclusive(&s_frames_lock);
    _mm_setcsr(csr); return token;
}
void xbox_FramebufferMoviePublish(uint64_t token, const void *source, size_t bytes,
    uint32_t width, uint32_t height, uint32_t pitch, uint32_t aspect_w, uint32_t aspect_h)
{
    unsigned csr = _mm_getcsr();
    if (token) fb_publish(source, bytes, 0, width, height, pitch, 8, token, aspect_w, aspect_h);
    _mm_setcsr(csr);
}
int xbox_FramebufferMovieActive(void)
{
    int active;
    AcquireSRWLockShared(&s_frames_lock);
    active = s_movie_token != 0;
    ReleaseSRWLockShared(&s_frames_lock);
    return active;
}
void xbox_FramebufferMovieEnd(uint64_t token)
{
    unsigned csr = _mm_getcsr();
    AcquireSRWLockExclusive(&s_frames_lock);
    if (token && s_movie_token == token) {
        s_movie_token = 0;
        if (s_pending && s_pending->movie_token == token) { free(s_pending); s_pending = NULL; }
    }
    ReleaseSRWLockExclusive(&s_frames_lock);
    _mm_setcsr(csr);
}
static FbFrame *fb_take_latest(void)
{
    FbFrame *old = NULL, *frame;
    AcquireSRWLockExclusive(&s_frames_lock);
    if (s_pending) { old = s_displayed; s_displayed = s_pending; s_pending = NULL; }
    frame = s_displayed;
    ReleaseSRWLockExclusive(&s_frames_lock);
    free(old); return frame; /* only the UI thread retires displayed frames */
}
static void fb_discard_frames(void)
{
    AcquireSRWLockExclusive(&s_frames_lock);
    free(s_pending); free(s_displayed); s_pending = s_displayed = NULL;
    ReleaseSRWLockExclusive(&s_frames_lock);
}
int xbox_FramebufferDumpBmp(const char *path)
{
    FbFrame *copy = NULL; FILE *file; uint8_t header[54] = {0}, *line;
    uint32_t row, image, total, x, y;
    AcquireSRWLockShared(&s_frames_lock);
    if (s_displayed && !s_displayed->gpu) {
        size_t size = sizeof(*copy) + (size_t)s_displayed->width * s_displayed->height * 4;
        copy = (FbFrame *)malloc(size); if (copy) memcpy(copy, s_displayed, size);
    }
    ReleaseSRWLockShared(&s_frames_lock);
    if (!copy || !path) { free(copy); return -1; }
    row = (copy->width * 3u + 3u) & ~3u; image = row * copy->height; total = 54u + image;
    line = (uint8_t *)calloc(1, row); file = line ? fopen(path, "wb") : NULL;
    if (!file) { free(line); free(copy); return -1; }
    header[0] = 'B'; header[1] = 'M'; header[10] = 54; header[14] = 40;
    memcpy(header + 2, &total, 4); memcpy(header + 18, &copy->width, 4);
    memcpy(header + 22, &copy->height, 4); header[26] = 1; header[28] = 24; memcpy(header + 34, &image, 4);
    if (fwrite(header, 1, sizeof(header), file) != sizeof(header)) goto failed;
    for (y = 0; y < copy->height; ++y) {
        const uint32_t *src = copy->pixels + (size_t)(copy->height - 1 - y) * copy->width;
        for (x = 0; x < copy->width; ++x) {
            line[x * 3] = (uint8_t)src[x]; line[x * 3 + 1] = (uint8_t)(src[x] >> 8);
            line[x * 3 + 2] = (uint8_t)(src[x] >> 16);
        }
        if (fwrite(line, 1, row, file) != row) goto failed;
    }
    free(line); free(copy); return fclose(file) == 0 ? 0 : -1;
failed:
    fclose(file); free(line); free(copy); return -1;
}
/* ---- keyboard, mouse, focus and the pointer (UI thread only) ----------------------------------
 * pc_input.c owns what the keys and buttons mean; this owns the window side: which messages
 * feed it, and when the pointer is confined and hidden. The pointer is taken only while the
 * mission is being played (the camera update is running), the window is the foreground window
 * and it is not being moved, and is given back the moment any of that stops -- Escape (pause),
 * Alt-Tab, minimise, close, a menu, a movie. */
static int s_ui_captured, s_ui_cursor_hidden, s_ui_sizemove;
static unsigned s_mouse_down_mask;
static RECT s_ui_clip;

static void fb_capture_release(void)
{
    if (!s_ui_captured) return;
    ClipCursor(NULL);
    if (s_ui_cursor_hidden) { ShowCursor(TRUE); s_ui_cursor_hidden = 0; }
    s_ui_captured = 0;
    pc_input_capture_changed(0);
}
static int fb_movie_input_active(void)
{
    return xbox_FramebufferMovieActive() || pc_movie_skip_active();
}
static void fb_capture_update(HWND hwnd)
{
    int focused = !IsIconic(hwnd) && GetForegroundWindow() == hwnd;
    /* Foreground changes and minimize/restore need not send WM_SETFOCUS.
     * The idempotent setter also clears input and requests pause on loss. */
    pc_input_focus_event(focused);
    int want = !s_ui_sizemove && focused && pc_input_capture_wanted();
    if (!want) { fb_capture_release(); return; }
    {
        RECT client, clip; POINT top_left, bottom_right;
        GetClientRect(hwnd, &client);
        top_left.x = client.left; top_left.y = client.top;
        bottom_right.x = client.right; bottom_right.y = client.bottom;
        ClientToScreen(hwnd, &top_left); ClientToScreen(hwnd, &bottom_right);
        clip.left = top_left.x; clip.top = top_left.y; clip.right = bottom_right.x; clip.bottom = bottom_right.y;
        if (clip.right - clip.left > 2) { clip.left++; clip.right--; }
        if (clip.bottom - clip.top > 2) { clip.top++; clip.bottom--; }
        if (clip.right <= clip.left || clip.bottom <= clip.top) { fb_capture_release(); return; }
        if (!s_ui_captured || !EqualRect(&clip, &s_ui_clip)) { ClipCursor(&clip); s_ui_clip = clip; }
        if (!s_ui_captured) {
            s_ui_captured = 1;
            ShowCursor(FALSE); s_ui_cursor_hidden = 1;
            SetCursor(NULL);
            SetCursorPos((clip.left + clip.right) / 2, (clip.top + clip.bottom) / 2);
            pc_input_capture_changed(1);
        }
    }
}
static void fb_mouse_buttons_released(void)
{
    int b;
    for (b = 0; b < 5; ++b) if (s_mouse_down_mask & (1u << b)) pc_input_mouse_button_event(b, 0);
    s_mouse_down_mask = 0;
}
static int fb_input_message(HWND hwnd, UINT m, WPARAM w, LPARAM l, LRESULT *result)
{
    int button = -1, down = 0;
    switch (m) {
    case WM_KEYDOWN: case WM_SYSKEYDOWN: case WM_KEYUP: case WM_SYSKEYUP:
        if (m == WM_SYSKEYDOWN && w == VK_F4) return 0;      /* Alt+F4 still closes */
        down = m == WM_KEYDOWN || m == WM_SYSKEYDOWN;
        pc_input_key_event((unsigned)((l >> 16) & 0xFF), (int)((l >> 24) & 1), down, down && (l & (1 << 30)) != 0);
        *result = 0; return 1;
    case WM_SYSCHAR:
        *result = 0; return 1;                               /* no beeps for Alt+key */
    case WM_SYSCOMMAND:
        if ((w & 0xFFF0) == SC_KEYMENU && l == 0) { *result = 0; return 1; }   /* a bare Alt or F10 is a key */
        return 0;
    case WM_LBUTTONDOWN: button = 0; down = 1; break;
    case WM_LBUTTONUP: button = 0; break;
    case WM_RBUTTONDOWN: button = 1; down = 1; break;
    case WM_RBUTTONUP: button = 1; break;
    case WM_MBUTTONDOWN: button = 2; down = 1; break;
    case WM_MBUTTONUP: button = 2; break;
    case WM_XBUTTONDOWN: button = GET_XBUTTON_WPARAM(w) == XBUTTON1 ? 3 : 4; down = 1; break;
    case WM_XBUTTONUP: button = GET_XBUTTON_WPARAM(w) == XBUTTON1 ? 3 : 4; break;
    case WM_MOUSEWHEEL:
        pc_input_wheel_event(GET_WHEEL_DELTA_WPARAM(w));
        *result = 0; return 1;
    case WM_INPUT: {
        RAWINPUT raw = {0}; UINT size = sizeof(raw);
        UINT received = GetRawInputData((HRAWINPUT)l, RID_INPUT, &raw, &size, sizeof(RAWINPUTHEADER));
        if (received != (UINT)-1 && received >= sizeof(RAWINPUTHEADER) + sizeof(RAWMOUSE) &&
            received <= sizeof(raw) && raw.header.dwSize >= sizeof(RAWINPUTHEADER) + sizeof(RAWMOUSE) &&
            raw.header.dwType == RIM_TYPEMOUSE && !(raw.data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE))
            pc_input_mouse_motion(raw.data.mouse.lLastX, raw.data.mouse.lLastY);
        *result = DefWindowProcA(hwnd, m, w, l);             /* the system cleans up its raw packet */
        return 1;
    }
    case WM_ACTIVATE: {
        int focused = LOWORD(w) != WA_INACTIVE && HIWORD(w) == 0;
        if (!focused) { fb_capture_release(); fb_mouse_buttons_released(); }
        pc_input_focus_event(focused);
        return 0;  /* DefWindowProc sets keyboard focus during activation. */
    }
    case WM_SETFOCUS:
        pc_input_focus_event(1);
        *result = 0; return 1;
    case WM_KILLFOCUS:
        fb_capture_release();
        fb_mouse_buttons_released();
        pc_input_focus_event(0);
        *result = 0; return 1;
    case WM_CANCELMODE: case WM_CAPTURECHANGED:
        fb_mouse_buttons_released();
        return 0;
    case WM_ENTERSIZEMOVE:
        s_ui_sizemove = 1; fb_capture_release();
        return 0;
    case WM_EXITSIZEMOVE:
        s_ui_sizemove = 0;
        return 0;
    case WM_SETCURSOR:
        if (LOWORD(l) == HTCLIENT && pc_input_cursor_hidden_wanted()) { SetCursor(NULL); *result = TRUE; return 1; }
        return 0;
    default:
        return 0;
    }
    /* a mouse button */
    if (down) {
        if (!s_mouse_down_mask) SetCapture(hwnd);
        s_mouse_down_mask |= 1u << button;
    } else {
        s_mouse_down_mask &= ~(1u << button);
        if (!s_mouse_down_mask && GetCapture() == hwnd) ReleaseCapture();
    }
    pc_input_mouse_button_event(button, down);
    *result = (m == WM_XBUTTONDOWN || m == WM_XBUTTONUP) ? TRUE : 0;
    return 1;
}
static LRESULT CALLBACK fb_wndproc(HWND hwnd, UINT message, WPARAM w, LPARAM l)
{
    LRESULT input_result;
    if (FB_PRESENT_MESSAGE_HOOK(hwnd, message, w, l)) return 0;
    if (message == WM_CLOSE) {
        fb_capture_release();
        fb_mouse_buttons_released();
        pc_input_focus_event(0);
        InterlockedExchange(&s_close_requested, 1); InterlockedExchange(&s_state, FB_CLOSED); return 0;
    }
    if (message == WM_DESTROY) {
        fb_capture_release();
        fb_mouse_buttons_released();
        pc_input_focus_event(0);
        InterlockedExchange(&s_state, FB_CLOSED); return 0;
    }
    if (message == WM_SIZE || message == WM_PAINT) {
        if (message == WM_SIZE && w == SIZE_MINIMIZED) {
            fb_capture_release();
            fb_mouse_buttons_released();
            pc_input_focus_event(0);
        } else if (message == WM_SIZE && GetForegroundWindow() == hwnd) {
            /* Windows can restore a minimized foreground HWND without a
             * WM_SETFOCUS transition. Re-enable its cleared input state. */
            pc_input_focus_event(1);
        }
        InterlockedExchange(&s_redraw, 1);
        if (message == WM_PAINT) { PAINTSTRUCT p; BeginPaint(hwnd, &p); EndPaint(hwnd, &p); }
        return 0;
    }
    if (message == WM_KEYDOWN && w == VK_F11) {
        if (!(l & (1u << 30))) InterlockedExchange(&s_toggle, 1); return 0;
    }
    if (message == WM_DPICHANGED) {
        RECT *r = (RECT *)l;
        if (!s_policy.borderless) SetWindowPos(hwnd, NULL, r->left, r->top,
            r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    if (fb_input_message(hwnd, message, w, l, &input_result)) return input_result;
    return DefWindowProcA(hwnd, message, w, l);
}
static void fb_window_mode(HWND hwnd, unsigned borderless)
{
    if (borderless) {
        MONITORINFO monitor = { sizeof(monitor) }; GetWindowRect(hwnd, &s_window_rect);
        if (!GetMonitorInfoA(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor)) return;
        SetWindowLongPtrA(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
        SetWindowPos(hwnd, HWND_TOP, monitor.rcMonitor.left, monitor.rcMonitor.top,
            monitor.rcMonitor.right - monitor.rcMonitor.left, monitor.rcMonitor.bottom - monitor.rcMonitor.top,
            SWP_FRAMECHANGED);
    } else {
        SetWindowLongPtrA(hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
        SetWindowPos(hwnd, NULL, s_window_rect.left, s_window_rect.top,
            s_window_rect.right - s_window_rect.left, s_window_rect.bottom - s_window_rect.top,
            SWP_FRAMECHANGED | SWP_NOZORDER);
    }
    s_policy.borderless = borderless;
}
/* Sets the client size of the windowed window, keeping its top-left corner unless it would leave the work area. */
static void fb_resize_client(HWND hwnd, unsigned width, unsigned height)
{
    RECT want = {0, 0, (LONG)width, (LONG)height}, now, work;
    MONITORINFO monitor = { sizeof(monitor) };
    LONG x, y, w, h;
    if (IsZoomed(hwnd)) ShowWindow(hwnd, SW_RESTORE);
    AdjustWindowRect(&want, WS_OVERLAPPEDWINDOW, FALSE);
    w = want.right - want.left; h = want.bottom - want.top;
    GetWindowRect(hwnd, &now);
    x = now.left; y = now.top;
    if (GetMonitorInfoA(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &monitor)) {
        work = monitor.rcWork;
        if (x + w > work.right) x = work.right - w;
        if (y + h > work.bottom) y = work.bottom - h;
        if (x < work.left) x = work.left;
        if (y < work.top) y = work.top;
    }
    SetWindowPos(hwnd, NULL, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE);
}
/* The Video Settings page (or the launcher) changed something: bring the window and the policy along. */
static void fb_apply_video(HWND hwnd)
{
    PcVideoSettings v;
    const unsigned borderless_now = s_policy.borderless;
    pc_video_get(&v);
    fb_policy_from_video(&v);
    s_policy.borderless = borderless_now;
    if (v.borderless != borderless_now) {
        fb_window_mode(hwnd, v.borderless);
        fprintf(stderr, "[FBWIN] %s\n", v.borderless ? "borderless fullscreen" : "windowed");
    }
    if (!s_policy.borderless && (v.window_width != s_applied_w || v.window_height != s_applied_h)) {
        fb_resize_client(hwnd, v.window_width, v.window_height);
        s_applied_w = v.window_width; s_applied_h = v.window_height;
        fprintf(stderr, "[FBWIN] window client %ux%u\n", v.window_width, v.window_height);
    }
    fprintf(stderr, "[FBWIN] scale %u filter %s vsync %u cap %u camera aspect %u:%u\n", s_policy.scale,
            s_policy.nearest ? "nearest" : "linear", s_policy.vsync, s_policy.fps_limit, s_policy.aspect_w, s_policy.aspect_h);
}
#define FB_RELEASE(p) do { if (p) { IUnknown_Release((IUnknown *)(p)); (p) = NULL; } } while (0)
#define FB_TRY(expr) do { hr = (expr); if (FAILED(hr)) goto done; } while (0)
static void fb_gpu_close(FbGpu *g)
{
    if (g->context) { ID3D11DeviceContext_ClearState(g->context); ID3D11DeviceContext_Flush(g->context); }
    { unsigned i; for (i = 0; i < FB_GPU_SLOTS; ++i) { FB_RELEASE(g->shared_srv[i]); FB_RELEASE(g->shared_tex[i]); } }
    FB_RELEASE(g->frame_fence); FB_RELEASE(g->done_fence); FB_RELEASE(g->ctx4);
    FB_RELEASE(g->ovl_view); FB_RELEASE(g->ovl_tex); FB_RELEASE(g->ovl_blend); FB_RELEASE(g->ovl_ps);
    FB_RELEASE(g->raster); FB_RELEASE(g->point); FB_RELEASE(g->linear); FB_RELEASE(g->ps); FB_RELEASE(g->vs); FB_RELEASE(g->vs_cinematic);
    FB_RELEASE(g->view); FB_RELEASE(g->texture); FB_RELEASE(g->target); FB_RELEASE(g->swap);
    FB_RELEASE(g->context); FB_RELEASE(g->device);
}
static HRESULT fb_target(FbGpu *g, unsigned width, unsigned height)
{
    ID3D11Texture2D *back = NULL; HRESULT hr;
    if (!width || !height) return S_FALSE;
    ID3D11DeviceContext_OMSetRenderTargets(g->context, 0, NULL, NULL); FB_RELEASE(g->target);
    if (g->width) FB_TRY(IDXGISwapChain1_ResizeBuffers(g->swap, 2, width, height,
        DXGI_FORMAT_B8G8R8A8_UNORM, g->tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0));
    FB_TRY(IDXGISwapChain1_GetBuffer(g->swap, 0, &IID_ID3D11Texture2D, (void **)&back));
    FB_TRY(ID3D11Device_CreateRenderTargetView(g->device, (ID3D11Resource *)back, NULL, &g->target));
    g->width = width; g->height = height;
done:
    FB_RELEASE(back); return hr;
}
static HRESULT fb_gpu_open(FbGpu *g, HWND hwnd)
{
    static const char shader[] =
        "Texture2D image:register(t0); SamplerState filter:register(s0);"
        "struct V{float4 p:SV_Position;float2 uv:TEXCOORD0;};"
        "V vs(uint i:SV_VertexID){V v;float2 uv=float2((i<<1)&2,i&2);"
        "v.p=float4(uv.x*2-1,1-uv.y*2,0,1);v.uv=uv;return v;}"
        "V vs_cinematic(uint i:SV_VertexID){V v=vs(i);v.uv.y=(v.uv.y*352+64)/480;return v;}"
        "float4 ps(V v):SV_Target{return float4(image.Sample(filter,v.uv).rgb,1);}"
        "float4 ps_ovl(V v):SV_Target{return image.Sample(filter,v.uv);}";
    IDXGIDevice *dxgi_device = NULL; IDXGIAdapter *adapter = NULL;
    IDXGIFactory2 *factory = NULL; IDXGIFactory5 *factory5 = NULL;
    ID3DBlob *vs = NULL, *vs_cinematic = NULL, *ps = NULL, *ps_ovl = NULL, *errors = NULL; D3D_FEATURE_LEVEL level;
    D3D11_BLEND_DESC blend = {0};
    DXGI_SWAP_CHAIN_DESC1 swap = {0}; D3D11_SAMPLER_DESC sampler = {0};
    D3D11_RASTERIZER_DESC raster = {0}; RECT client; HRESULT hr; BOOL tearing = FALSE;
    const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
    if (s_gpu_luid_set) {
        /* the same adapter as the renderer, so its textures can be shared */
        IDXGIFactory4 *f4 = NULL; IDXGIAdapter1 *a1 = NULL;
        if (SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&f4)) &&
            SUCCEEDED(IDXGIFactory4_EnumAdapterByLuid(f4, s_gpu_luid, &IID_IDXGIAdapter1, (void **)&a1))) {
            hr = D3D11CreateDevice((IDXGIAdapter *)a1, D3D_DRIVER_TYPE_UNKNOWN, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                levels, sizeof(levels) / sizeof(levels[0]), D3D11_SDK_VERSION, &g->device, &level, &g->context);
        } else hr = E_FAIL;
        if (a1) IDXGIAdapter1_Release(a1);
        if (f4) IDXGIFactory4_Release(f4);
        if (FAILED(hr)) { g->device = NULL; g->context = NULL; }
    }
    if (!g->device)
    FB_TRY(D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        levels, sizeof(levels) / sizeof(levels[0]), D3D11_SDK_VERSION, &g->device, &level, &g->context));
    FB_TRY(ID3D11Device_QueryInterface(g->device, &IID_IDXGIDevice, (void **)&dxgi_device));
    FB_TRY(IDXGIDevice_GetAdapter(dxgi_device, &adapter));
    FB_TRY(IDXGIAdapter_GetParent(adapter, &IID_IDXGIFactory2, (void **)&factory));
    if (SUCCEEDED(IDXGIFactory2_QueryInterface(factory, &IID_IDXGIFactory5, (void **)&factory5)))
        IDXGIFactory5_CheckFeatureSupport(factory5, DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing));
    g->tearing = tearing ? 1 : 0; GetClientRect(hwnd, &client);
    swap.Width = client.right; swap.Height = client.bottom; swap.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swap.SampleDesc.Count = 1; swap.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; swap.BufferCount = 2;
    swap.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swap.Flags = g->tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    FB_TRY(IDXGIFactory2_CreateSwapChainForHwnd(factory, (IUnknown *)g->device, hwnd, &swap, NULL, NULL, &g->swap));
    IDXGIFactory2_MakeWindowAssociation(factory, hwnd, DXGI_MWA_NO_ALT_ENTER);
    FB_TRY(fb_target(g, swap.Width, swap.Height));
    FB_TRY(D3DCompile(shader, sizeof(shader) - 1, "framebuffer-present", NULL, NULL,
        "vs", "vs_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &vs, &errors)); FB_RELEASE(errors);
    FB_TRY(D3DCompile(shader, sizeof(shader) - 1, "framebuffer-present", NULL, NULL,
        "vs_cinematic", "vs_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &vs_cinematic, &errors)); FB_RELEASE(errors);
    FB_TRY(D3DCompile(shader, sizeof(shader) - 1, "framebuffer-present", NULL, NULL,
        "ps", "ps_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &ps, &errors));
    FB_TRY(D3DCompile(shader, sizeof(shader) - 1, "framebuffer-present", NULL, NULL,
        "ps_ovl", "ps_4_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &ps_ovl, &errors)); FB_RELEASE(errors);
    FB_TRY(ID3D11Device_CreatePixelShader(g->device, ps_ovl->lpVtbl->GetBufferPointer(ps_ovl),
        ps_ovl->lpVtbl->GetBufferSize(ps_ovl), NULL, &g->ovl_ps));
    blend.RenderTarget[0].BlendEnable = TRUE;
    blend.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE; blend.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blend.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blend.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE; blend.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    blend.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD; blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    FB_TRY(ID3D11Device_CreateBlendState(g->device, &blend, &g->ovl_blend));
    FB_TRY(ID3D11Device_CreateVertexShader(g->device, vs->lpVtbl->GetBufferPointer(vs),
        vs->lpVtbl->GetBufferSize(vs), NULL, &g->vs));
    FB_TRY(ID3D11Device_CreateVertexShader(g->device, vs_cinematic->lpVtbl->GetBufferPointer(vs_cinematic),
        vs_cinematic->lpVtbl->GetBufferSize(vs_cinematic), NULL, &g->vs_cinematic));
    FB_TRY(ID3D11Device_CreatePixelShader(g->device, ps->lpVtbl->GetBufferPointer(ps),
        ps->lpVtbl->GetBufferSize(ps), NULL, &g->ps));
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler.MaxLOD = D3D11_FLOAT32_MAX; sampler.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    FB_TRY(ID3D11Device_CreateSamplerState(g->device, &sampler, &g->linear));
    sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    FB_TRY(ID3D11Device_CreateSamplerState(g->device, &sampler, &g->point));
    raster.FillMode = D3D11_FILL_SOLID; raster.CullMode = D3D11_CULL_NONE; raster.DepthClipEnable = TRUE;
    FB_TRY(ID3D11Device_CreateRasterizerState(g->device, &raster, &g->raster));
done:
    if (FAILED(hr) && errors) fprintf(stderr, "[FBWIN] shader: %s\n", (char *)errors->lpVtbl->GetBufferPointer(errors));
    FB_RELEASE(errors); FB_RELEASE(ps_ovl); FB_RELEASE(ps); FB_RELEASE(vs); FB_RELEASE(vs_cinematic); FB_RELEASE(factory5); FB_RELEASE(factory);
    FB_RELEASE(adapter); FB_RELEASE(dxgi_device); return hr;
}
/* RECOMP_FB_DUMP: write what the presenter is about to show (a GPU frame has no CPU copy). */
static char s_backbuffer_dump[512];
static void fb_dump_backbuffer(FbGpu *g, const char *path)
{
    ID3D11Resource *res = NULL; ID3D11Texture2D *back = NULL, *stage = NULL; D3D11_TEXTURE2D_DESC desc;
    D3D11_MAPPED_SUBRESOURCE ms; FILE *file; uint8_t header[54] = {0}, *line; uint32_t row, image, total, x, y;
    ID3D11RenderTargetView_GetResource(g->target, &res);
    if (!res) return;
    if (FAILED(ID3D11Resource_QueryInterface(res, &IID_ID3D11Texture2D, (void **)&back))) { FB_RELEASE(res); return; }
    ID3D11Texture2D_GetDesc(back, &desc);
    desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = 0; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.MiscFlags = 0;
    if (SUCCEEDED(ID3D11Device_CreateTexture2D(g->device, &desc, NULL, &stage))) {
        ID3D11DeviceContext_CopyResource(g->context, (ID3D11Resource *)stage, (ID3D11Resource *)back);
        if (SUCCEEDED(ID3D11DeviceContext_Map(g->context, (ID3D11Resource *)stage, 0, D3D11_MAP_READ, 0, &ms))) {
            row = (desc.Width * 3u + 3u) & ~3u; image = row * desc.Height; total = 54u + image;
            line = (uint8_t *)calloc(1, row); file = line ? fopen(path, "wb") : NULL;
            if (file) {
                header[0] = 'B'; header[1] = 'M'; header[10] = 54; header[14] = 40;
                memcpy(header + 2, &total, 4); memcpy(header + 18, &desc.Width, 4);
                memcpy(header + 22, &desc.Height, 4); header[26] = 1; header[28] = 24; memcpy(header + 34, &image, 4);
                fwrite(header, 1, sizeof(header), file);
                for (y = 0; y < desc.Height; ++y) {
                    const uint8_t *src = (const uint8_t *)ms.pData + (size_t)(desc.Height - 1 - y) * ms.RowPitch;
                    for (x = 0; x < desc.Width; ++x) { line[x * 3] = src[x * 4]; line[x * 3 + 1] = src[x * 4 + 1]; line[x * 3 + 2] = src[x * 4 + 2]; }
                    fwrite(line, 1, row, file);
                }
                fclose(file);
            }
            free(line);
            ID3D11DeviceContext_Unmap(g->context, (ID3D11Resource *)stage, 0);
        }
        FB_RELEASE(stage);
    }
    FB_RELEASE(back); FB_RELEASE(res);
}
static HRESULT fb_gpu_shared_open(FbGpu *g)
{
    ID3D11Device1 *d1 = NULL; ID3D11Device5 *d5 = NULL; HRESULT hr; unsigned i, count;
    void *tex[FB_GPU_SLOTS], *ff, *df; LONG gen;
    AcquireSRWLockShared(&s_gpu_lock);
    gen = s_gpu_gen; count = s_gpu_count; ff = s_gpu_frame_fence; df = s_gpu_done_fence;
    for (i = 0; i < count; ++i) tex[i] = s_gpu_tex[i];
    ReleaseSRWLockShared(&s_gpu_lock);
    for (i = 0; i < FB_GPU_SLOTS; ++i) { FB_RELEASE(g->shared_srv[i]); FB_RELEASE(g->shared_tex[i]); }
    FB_RELEASE(g->frame_fence); FB_RELEASE(g->done_fence); FB_RELEASE(g->ctx4);
    FB_TRY(ID3D11Device_QueryInterface(g->device, &IID_ID3D11Device1, (void **)&d1));
    FB_TRY(ID3D11Device_QueryInterface(g->device, &IID_ID3D11Device5, (void **)&d5));
    FB_TRY(ID3D11DeviceContext_QueryInterface(g->context, &IID_ID3D11DeviceContext4, (void **)&g->ctx4));
    for (i = 0; i < count; ++i) {
        FB_TRY(ID3D11Device1_OpenSharedResource1(d1, (HANDLE)tex[i], &IID_ID3D11Texture2D, (void **)&g->shared_tex[i]));
        FB_TRY(ID3D11Device_CreateShaderResourceView(g->device, (ID3D11Resource *)g->shared_tex[i], NULL, &g->shared_srv[i]));
    }
    FB_TRY(ID3D11Device5_OpenSharedFence(d5, (HANDLE)ff, &IID_ID3D11Fence, (void **)&g->frame_fence));
    FB_TRY(ID3D11Device5_OpenSharedFence(d5, (HANDLE)df, &IID_ID3D11Fence, (void **)&g->done_fence));
    g->shared_gen = gen;
done:
    FB_RELEASE(d1); FB_RELEASE(d5);
    if (FAILED(hr)) InterlockedExchange(&s_gpu_failed, 1);
    return hr;
}
/* The overlay's pixels, premultiplied BGRA at the size of the client area. */
static void fb_overlay_upload(FbGpu *g, const uint32_t *pixels, unsigned width, unsigned height)
{
    unsigned box[4];
    /* The native menus change a few rows at a time: send only those once the texture exists. */
    if (g->ovl_tex && g->ovl_w == width && g->ovl_h == height && pc_menu_overlay_dirty(box) &&
        box[0] + box[2] <= width && box[1] + box[3] <= height && box[2] && box[3]) {
        D3D11_BOX b;
        b.left = box[0]; b.top = box[1]; b.front = 0; b.right = box[0] + box[2]; b.bottom = box[1] + box[3]; b.back = 1;
        ID3D11DeviceContext_UpdateSubresource(g->context, (ID3D11Resource *)g->ovl_tex, 0, &b,
            pixels + (size_t)box[1] * width + box[0], width * 4, 0);
        return;
    }
    if (!g->ovl_tex || g->ovl_w != width || g->ovl_h != height) {
        D3D11_TEXTURE2D_DESC desc = {0};
        ID3D11ShaderResourceView *null_view = NULL;
        ID3D11DeviceContext_PSSetShaderResources(g->context, 0, 1, &null_view);
        FB_RELEASE(g->ovl_view); FB_RELEASE(g->ovl_tex);
        desc.Width = width; desc.Height = height; desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(ID3D11Device_CreateTexture2D(g->device, &desc, NULL, &g->ovl_tex))) return;
        if (FAILED(ID3D11Device_CreateShaderResourceView(g->device, (ID3D11Resource *)g->ovl_tex, NULL, &g->ovl_view))) {
            FB_RELEASE(g->ovl_tex); return;
        }
        g->ovl_w = width; g->ovl_h = height;
    }
    ID3D11DeviceContext_UpdateSubresource(g->context, (ID3D11Resource *)g->ovl_tex, 0, NULL, pixels, width * 4, 0);
}
static HRESULT fb_gpu_draw(FbGpu *g, const FbFrame *frame)
{
    D3D11_VIEWPORT viewport;
    FbPresentRect r = fb_present_rect(g->width, g->height, frame->width, frame->height,
        s_policy.scale, s_policy.nearest,
        frame->aspect_w ? frame->aspect_w : s_policy.aspect_w,
        frame->aspect_h ? frame->aspect_h : s_policy.aspect_h);
    const FLOAT black[4] = {0, 0, 0, 1}; ID3D11SamplerState *sampler = r.nearest ? g->point : g->linear;
    HRESULT hr; ID3D11ShaderResourceView *view = g->view;
    if (frame->gpu) {
        if (g->shared_gen != InterlockedCompareExchange(&s_gpu_gen, 0, 0) || !g->frame_fence) {
            hr = fb_gpu_shared_open(g); if (FAILED(hr)) return S_OK;   /* drop; the renderer falls back */
        }
        if (frame->gpu_slot >= FB_GPU_SLOTS || !g->shared_srv[frame->gpu_slot]) return S_OK;
        ID3D11DeviceContext4_Wait(g->ctx4, g->frame_fence, frame->gpu_value);
        view = g->shared_srv[frame->gpu_slot];
    } else
    if (g->tex_width != frame->width || g->tex_height != frame->height) {
        D3D11_TEXTURE2D_DESC desc = {0}; ID3D11ShaderResourceView *null_view = NULL;
        ID3D11DeviceContext_PSSetShaderResources(g->context, 0, 1, &null_view);
        FB_RELEASE(g->view); FB_RELEASE(g->texture);
        desc.Width = frame->width; desc.Height = frame->height; desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM; desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        hr = ID3D11Device_CreateTexture2D(g->device, &desc, NULL, &g->texture); if (FAILED(hr)) return hr;
        hr = ID3D11Device_CreateShaderResourceView(g->device, (ID3D11Resource *)g->texture, NULL, &g->view);
        if (FAILED(hr)) return hr; g->tex_width = frame->width; g->tex_height = frame->height;
    }
    if (!frame->gpu) {
        ID3D11DeviceContext_UpdateSubresource(g->context, (ID3D11Resource *)g->texture, 0, NULL,
            frame->pixels, frame->width * 4, 0);
        view = g->view;
    }
    ID3D11DeviceContext_OMSetRenderTargets(g->context, 1, &g->target, NULL);
    ID3D11DeviceContext_ClearRenderTargetView(g->context, g->target, black);
    viewport.TopLeftX = (FLOAT)r.x; viewport.TopLeftY = (FLOAT)r.y;
    viewport.Width = (FLOAT)r.width; viewport.Height = (FLOAT)r.height; viewport.MinDepth = 0; viewport.MaxDepth = 1;
    ID3D11DeviceContext_RSSetViewports(g->context, 1, &viewport); ID3D11DeviceContext_RSSetState(g->context, g->raster);
    ID3D11DeviceContext_IASetInputLayout(g->context, NULL);
    ID3D11DeviceContext_IASetPrimitiveTopology(g->context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_VSSetShader(g->context,
        fb_cinematic_crop(frame->aspect_w, frame->aspect_h) ? g->vs_cinematic : g->vs, NULL, 0);
    ID3D11DeviceContext_PSSetShader(g->context, g->ps, NULL, 0);
    ID3D11DeviceContext_PSSetSamplers(g->context, 0, 1, &sampler);
    ID3D11DeviceContext_PSSetShaderResources(g->context, 0, 1, &view); ID3D11DeviceContext_Draw(g->context, 3, 0);
    if (g->overlay_on && g->ovl_view && g->ovl_ps && g->ovl_blend) {
        const FLOAT factor[4] = {0, 0, 0, 0};
        ID3D11SamplerState *ovl_sampler = g->ovl_w == g->width && g->ovl_h == g->height ? g->point : g->linear;
        viewport.TopLeftX = 0; viewport.TopLeftY = 0; viewport.Width = (FLOAT)g->width; viewport.Height = (FLOAT)g->height;
        ID3D11DeviceContext_RSSetViewports(g->context, 1, &viewport);
        ID3D11DeviceContext_VSSetShader(g->context, g->vs, NULL, 0);
        ID3D11DeviceContext_OMSetBlendState(g->context, g->ovl_blend, factor, 0xFFFFFFFFu);
        ID3D11DeviceContext_PSSetShader(g->context, g->ovl_ps, NULL, 0);
        ID3D11DeviceContext_PSSetSamplers(g->context, 0, 1, &ovl_sampler);
        ID3D11DeviceContext_PSSetShaderResources(g->context, 0, 1, &g->ovl_view);
        ID3D11DeviceContext_Draw(g->context, 3, 0);
        ID3D11DeviceContext_OMSetBlendState(g->context, NULL, factor, 0xFFFFFFFFu);
    }
    if (frame->gpu && frame->gpu_value > g->done_signaled) {
        ID3D11DeviceContext4_Signal(g->ctx4, g->done_fence, frame->gpu_value);
        g->done_signaled = frame->gpu_value;
    }
    if (s_backbuffer_dump[0]) { fb_dump_backbuffer(g, s_backbuffer_dump); s_backbuffer_dump[0] = 0; }
    FB_PRESENT_BEFORE_PRESENT(g, frame);
    return IDXGISwapChain1_Present(g->swap, s_policy.vsync ? 1 : 0,
        !s_policy.vsync && g->tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
}
static DWORD WINAPI fb_thread(LPVOID unused)
{
    WNDCLASSA wc = {0}; FbGpu gpu = {0}; HWND hwnd; RECT rect = {0, 0, 0, 0};
    FbFrame *frame = NULL; unsigned need_draw = 0, displayed = 0, applied_generation = pc_video_generation();
    LARGE_INTEGER frequency, next = {0}; HRESULT hr; (void)unused;
    /* Host graphics/driver code has its own standard FP environment; the
     * producer may have entered Start under any guest rounding/mask state. */
    _mm_setcsr(0x1F80);
    {
        typedef HANDLE (WINAPI *SetDpiContext)(HANDLE);
        SetDpiContext dpi = (SetDpiContext)GetProcAddress(GetModuleHandleA("user32.dll"), "SetThreadDpiAwarenessContext");
        if (dpi) dpi((HANDLE)(intptr_t)-4); /* PER_MONITOR_AWARE_V2, this thread only */
    }
    wc.lpfnWndProc = fb_wndproc; wc.hInstance = GetModuleHandleA(NULL);
    wc.hCursor = LoadCursorA(NULL, IDC_ARROW); wc.lpszClassName = "XboxRecompFramebuffer"; RegisterClassA(&wc);
    rect.right = (LONG)s_policy.width; rect.bottom = (LONG)s_policy.height;
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    hwnd = CreateWindowExA(0, wc.lpszClassName, "BLACK", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top, NULL, NULL, wc.hInstance, NULL);
    if (!hwnd) {
        fprintf(stderr, "[FBWIN] window creation failed: %lu\n", GetLastError());
        InterlockedExchange(&s_state, FB_CLOSED); fb_discard_frames(); ExitProcess(1);
    }
    if (s_policy.borderless) fb_window_mode(hwnd, 1);
    s_window = hwnd;
    pc_input_init();
    pc_video_menu_install();
    pc_menu_install();
    pc_movie_skip_install();
    pc_photo_install();
    pc_input_set_movie_active_query(fb_movie_input_active);
    {
        RAWINPUTDEVICE mouse;
        mouse.usUsagePage = 0x01; mouse.usUsage = 0x02; mouse.dwFlags = 0; mouse.hwndTarget = hwnd;
        if (!RegisterRawInputDevices(&mouse, 1, sizeof(mouse)))
            fprintf(stderr, "[FBWIN] raw mouse registration failed: %lu (mouse look unavailable)\n", GetLastError());
    }
    if (GetForegroundWindow() != hwnd) pc_input_focus_event(0);
    hr = fb_gpu_open(&gpu, hwnd);
    if (FAILED(hr)) {
        fprintf(stderr, "[FBWIN] D3D11 initialization failed: %08lX\n", (unsigned long)hr);
        InterlockedExchange(&s_state, FB_CLOSED);
    } else fprintf(stderr, "[FBWIN] D3D11 output %ux%u %s; scale %u filter %s vsync %u cap %u; F11 toggles fullscreen\n",
        gpu.width, gpu.height, s_policy.borderless ? "borderless" : "windowed", s_policy.scale,
        s_policy.nearest ? "nearest" : "linear", s_policy.vsync, s_policy.fps_limit);
    QueryPerformanceFrequency(&frequency);
    while (InterlockedCompareExchange(&s_state, 0, 0) == FB_RUNNING) {
        MSG message; LARGE_INTEGER now; DWORD wait_ms = 1000; FbFrame *latest;
        while (PeekMessageA(&message, NULL, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) { InterlockedExchange(&s_state, FB_CLOSED); break; }
            TranslateMessage(&message); DispatchMessageA(&message);
        }
        if (InterlockedCompareExchange(&s_state, 0, 0) != FB_RUNNING) break;
        if (InterlockedExchange(&s_toggle, 0)) {
            fb_window_mode(hwnd, !s_policy.borderless);
            pc_video_note_borderless(s_policy.borderless);
            need_draw = 1;
        }
        {
            const unsigned generation = pc_video_generation();
            if (generation != applied_generation) { applied_generation = generation; fb_apply_video(hwnd); need_draw = 1; }
        }
        {
            const uint32_t *overlay = NULL; int changed = 0;
            /* the native menus sit on the game picture, so they need where the picture is drawn */
            FbPresentRect pr = { 0, 0, gpu.width, gpu.height, 0 };
            if (frame) pr = fb_present_rect(gpu.width, gpu.height, frame->width, frame->height, s_policy.scale, s_policy.nearest,
                frame->aspect_w ? frame->aspect_w : s_policy.aspect_w, frame->aspect_h ? frame->aspect_h : s_policy.aspect_h);
            gpu.overlay_on = pc_menu_overlay(gpu.width, gpu.height, (int)pr.x, (int)pr.y, pr.width, pr.height, &overlay, &changed);
            if (changed) {
                need_draw = 1;
                if (gpu.overlay_on && overlay) fb_overlay_upload(&gpu, overlay, gpu.width, gpu.height);
                if (gpu.overlay_on && s_menu_shot && !s_backbuffer_dump[0])
                    snprintf(s_backbuffer_dump, sizeof(s_backbuffer_dump), "%s%03u.bmp", s_menu_shot, ++s_menu_shots);
            }
        }
        fb_capture_update(hwnd);
        latest = fb_take_latest(); if (latest != frame) { frame = latest; need_draw = 1; }
        if (InterlockedExchange(&s_redraw, 0)) need_draw = 1;
        QueryPerformanceCounter(&now);
        if (need_draw && frame && now.QuadPart >= next.QuadPart && !IsIconic(hwnd)) {
            RECT client; GetClientRect(hwnd, &client);
            if (client.right && client.bottom) {
                if ((unsigned)client.right != gpu.width || (unsigned)client.bottom != gpu.height)
                    hr = fb_target(&gpu, client.right, client.bottom);
                if (SUCCEEDED(hr)) hr = fb_gpu_draw(&gpu, frame);
                if (FAILED(hr)) { fprintf(stderr, "[FBWIN] D3D11 present failed: %08lX\n", (unsigned long)hr); break; }
                /* Cap presentation start-to-start. Adding a full interval
                 * after blocking Present would count the vsync wait twice. */
                next.QuadPart = s_policy.fps_limit ? now.QuadPart + frequency.QuadPart / s_policy.fps_limit : 0;
                need_draw = 0; QueryPerformanceCounter(&now);
                if (s_dump && ++displayed == 600) {
                    if (frame->gpu) { snprintf(s_backbuffer_dump, sizeof(s_backbuffer_dump), "%s", s_dump); }
                    else xbox_FramebufferDumpBmp(s_dump);
                }
                if (hr == DXGI_STATUS_OCCLUDED) wait_ms = 50;
            }
        }
        if (need_draw && next.QuadPart > now.QuadPart) {
            uint64_t ms = ((uint64_t)(next.QuadPart - now.QuadPart) * 1000 + frequency.QuadPart - 1) / frequency.QuadPart;
            if (ms < wait_ms) wait_ms = (DWORD)ms;
        }
        if (wait_ms > 40) wait_ms = 40;   /* the pointer follows the game's state within a few frames */
        MsgWaitForMultipleObjectsEx(1, &s_frame_ready, wait_ms, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
    }
    fb_capture_release();
    InterlockedExchange(&s_state, FB_CLOSED); fb_gpu_close(&gpu);
    s_window = NULL;
    if (IsWindow(hwnd)) DestroyWindow(hwnd);
    fb_discard_frames();
    if (InterlockedCompareExchange(&s_close_requested, 0, 0)) ExitProcess(0);
    if (FAILED(hr)) ExitProcess(1); /* do not leave the game silently windowless */
    return 0;
}
/* The same close as the window's own close button: the presenter ends and the process exits with 0. */
int xbox_FramebufferRequestClose(void)
{
    HWND hwnd = s_window;
    return hwnd && PostMessageA(hwnd, WM_CLOSE, 0, 0);
}
static void fb_start(void)
{
    HANDLE thread;
    InitOnceExecuteOnce(&s_once, fb_initialize, NULL, NULL);
    if (!s_enabled) return;
    if (!s_frame_ready) {
        fprintf(stderr, "[FBWIN] frame event creation failed\n"); ExitProcess(1);
    }
    if (InterlockedCompareExchange(&s_state, FB_RUNNING, FB_NOT_STARTED) != FB_NOT_STARTED) return;
    thread = CreateThread(NULL, 0, fb_thread, NULL, 0, NULL);
    if (thread) CloseHandle(thread);
    else {
        fprintf(stderr, "[FBWIN] presentation thread creation failed: %lu\n", GetLastError());
        InterlockedExchange(&s_state, FB_CLOSED); fb_discard_frames(); ExitProcess(1);
    }
}
void xbox_FramebufferWindowStart(void)
{
    unsigned csr = _mm_getcsr();
    fb_start();
    _mm_setcsr(csr);
}
#else
void xbox_FramebufferPublish(const void *s, size_t n, uint32_t va, uint32_t w,
    uint32_t h, uint32_t p, uint32_t f) { (void)s; (void)n; (void)va; (void)w; (void)h; (void)p; (void)f; }
void xbox_FramebufferPublishAspect(const void *s, size_t n, uint32_t va,
    uint32_t w, uint32_t h, uint32_t p, uint32_t f, uint32_t a, uint32_t b)
{ (void)s; (void)n; (void)va; (void)w; (void)h; (void)p; (void)f; (void)a; (void)b; }
void xbox_FramebufferWindowSet(uint32_t va, uint32_t p) { (void)va; (void)p; }
void xbox_FramebufferWindowStart(void) {}
int xbox_FramebufferDumpBmp(const char *path) { (void)path; return -1; }
uint64_t xbox_FramebufferMovieBegin(void) { return 0; }
int xbox_FramebufferMovieActive(void) { return 0; }
int xbox_FramebufferRequestClose(void) { return 0; }
void xbox_FramebufferMoviePublish(uint64_t t, const void *s, size_t n, uint32_t w,
    uint32_t h, uint32_t p, uint32_t a, uint32_t b) { (void)t; (void)s; (void)n; (void)w; (void)h; (void)p; (void)a; (void)b; }
void xbox_FramebufferMovieEnd(uint64_t t) { (void)t; }
void xbox_FramebufferInputByCaller(int on) { (void)on; }
#endif
