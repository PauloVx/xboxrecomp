/*
 * pc_split.c -- split-screen: the layout of the views, and player two's input. See pc_split.h.
 */
#include "pc_split.h"
#include "pc_video.h"
#include "../input/pc_input.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int pc_split_enabled(void) { return pc_video_split() == PCS_ON; }

void pc_split_rect_for(int split, int player, int r[4])
{
    r[0] = 0; r[1] = 0; r[2] = 640; r[3] = 480;
    if (split) { r[1] = player ? 240 : 0; r[3] = player ? 480 : 240; }
}

void pc_split_rect(int player, int r[4]) { pc_split_rect_for(pc_split_enabled(), player, r); }

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <xinput.h>

#define PCS_PI 3.14159265358979323846
#define PCS_DEG (PCS_PI / 180.0)
#define PCS_STICK_DEG 120.0              /* degrees a second, stick fully over */
#define PCS_SPEED 4.0                    /* metres a second */
#define PCS_DEAD 7800                    /* about a quarter of the throw */

/* Player two's keys, as the pad they stand for. Report layout: byte 2 digital (dpad 1 2 4 8, start 0x10, back 0x20,
 * thumbs 0x40 0x80), 4..11 analog A B X Y Black White LT RT, 12..19 the axes LX LY RX RY. */
enum { PK_DIGITAL, PK_ANALOG, PK_STICK };
static const struct { unsigned id; int kind, bit; } k_keys[] = {
    { 0x148, PK_STICK, 2 }, { 0x150, PK_STICK, 3 }, { 0x14B, PK_STICK, 1 }, { 0x14D, PK_STICK, 0 },   /* arrows: left stick */
    { 0x48, PK_STICK, 6 }, { 0x50, PK_STICK, 7 }, { 0x4B, PK_STICK, 5 }, { 0x4D, PK_STICK, 4 },       /* numpad 8 2 4 6: right stick */
    { 0x52, PK_ANALOG, 7 }, { 0x53, PK_ANALOG, 6 },                                                    /* numpad 0 RT, . LT */
    { 0x11C, PK_ANALOG, 0 }, { 0x4E, PK_ANALOG, 1 }, { 0x4A, PK_ANALOG, 2 }, { 0x37, PK_ANALOG, 3 },   /* enter A, + B, - X, * Y */
    { 0x49, PK_ANALOG, 4 }, { 0x47, PK_ANALOG, 5 },                                                    /* numpad 9 Black, 7 White */
    { 0x4F, PK_DIGITAL, 0x20 }, { 0x51, PK_DIGITAL, 0x10 },                                            /* numpad 1 Back, 3 Start */
    { 0x4C, PK_DIGITAL, 0x80 }, { 0x135, PK_DIGITAL, 0x40 },                                           /* numpad 5 R thumb, / L thumb */
};
#define PK_COUNT ((int)(sizeof(k_keys) / sizeof(k_keys[0])))

static volatile LONG s_frame, s_keys;
static int s_started;
static double s_pos[3], s_yaw, s_pitch, s_hand = 1.0, s_up_sign = 1.0;
static LONGLONG s_last_qpc;

void pc_split_note_frame(int split) { InterlockedExchange(&s_frame, split != 0); }
int pc_split_frame(void) { return (int)InterlockedCompareExchange(&s_frame, 0, 0); }

/* ---- player two's keys ---- */

static int key_index(unsigned id)
{
    int i;
    for (i = 0; i < PK_COUNT; i++) if (k_keys[i].id == id) return i;
    return -1;
}

static int f_key(unsigned id, int down, int repeat)
{
    const int i = key_index(id);
    const LONG bit = i >= 0 ? (LONG)1 << i : 0;
    (void)repeat;
    if (!bit) return 0;
    if (!pc_split_frame()) {
        /* a key held when the views went away is let go */
        if (!down && (InterlockedCompareExchange(&s_keys, 0, 0) & bit)) { InterlockedAnd(&s_keys, ~bit); return 1; }
        return 0;
    }
    if (down) InterlockedOr(&s_keys, bit); else InterlockedAnd(&s_keys, ~bit);
    return 1;
}

static void f_focus(int focused)
{
    if (!focused) InterlockedExchange(&s_keys, 0);
}

static const PcInputFilter k_filter = { f_key, NULL, NULL, NULL, f_focus, NULL };

void pc_split_install(void) { pc_input_set_filter_slot(4, &k_filter); }

static void put_axis(uint8_t *out, int axis, int v)
{
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    out[12 + axis * 2] = (uint8_t)(v & 0xFF);
    out[13 + axis * 2] = (uint8_t)((v >> 8) & 0xFF);
}

static int get_axis(const uint8_t *out, int axis) { return (int16_t)(out[12 + axis * 2] | (out[13 + axis * 2] << 8)); }

void pc_split_pad_report(uint8_t out[20])
{
    XINPUT_STATE x;
    const LONG keys = InterlockedCompareExchange(&s_keys, 0, 0);
    int axis[4] = { 0, 0, 0, 0 }, i;
    if (XInputGetState(1, &x) == ERROR_SUCCESS) {
        static const WORD analog[6] = { XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y,
                                        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_LEFT_SHOULDER };
        out[2] = (uint8_t)(x.Gamepad.wButtons & 0xFF);
        for (i = 0; i < 6; i++) out[4 + i] = (x.Gamepad.wButtons & analog[i]) ? 0xFF : 0;
        out[10] = x.Gamepad.bLeftTrigger;
        out[11] = x.Gamepad.bRightTrigger;
        axis[0] = x.Gamepad.sThumbLX; axis[1] = x.Gamepad.sThumbLY;
        axis[2] = x.Gamepad.sThumbRX; axis[3] = x.Gamepad.sThumbRY;
    }
    for (i = 0; i < PK_COUNT; i++) {
        if (!(keys & ((LONG)1 << i))) continue;
        if (k_keys[i].kind == PK_DIGITAL) out[2] |= (uint8_t)k_keys[i].bit;
        else if (k_keys[i].kind == PK_ANALOG) out[4 + k_keys[i].bit] = 0xFF;
        else axis[k_keys[i].bit >> 1] = (k_keys[i].bit & 1) ? -32768 : 32767;
    }
    for (i = 0; i < 4; i++) put_axis(out, i, axis[i]);
}

/* ---- the free camera (before player two has a character) ---- */

static void cross3(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

static int norm3(double v[3])
{
    const double l = sqrt(dot3(v, v));
    if (!(l > 1e-9)) return 0;
    v[0] /= l; v[1] /= l; v[2] /= l;
    return 1;
}

static double clampd(double v, double lo, double hi) { return v < lo ? lo : v > hi ? hi : v; }

/* As photo mode (pc_photo.c): forward, and right / up in the game's handedness, from yaw and pitch (world Y up). */
static void basis(double f[3], double r[3], double u[3])
{
    static const double world_up[3] = { 0, 1, 0 };
    f[0] = cos(s_pitch) * sin(s_yaw); f[1] = sin(s_pitch); f[2] = cos(s_pitch) * cos(s_yaw);
    cross3(world_up, f, r);
    if (!norm3(r)) { r[0] = cos(s_yaw); r[1] = 0; r[2] = -sin(s_yaw); }
    r[0] *= s_hand; r[1] *= s_hand; r[2] *= s_hand;
    cross3(f, r, u);
    norm3(u);
    u[0] *= s_up_sign; u[1] *= s_up_sign; u[2] *= s_up_sign;
}

static void start(const float m[4][3])
{
    double f[3] = { m[2][0], m[2][1], m[2][2] }, r[3], u[3], gr[3] = { m[0][0], m[0][1], m[0][2] }, gu[3] = { m[1][0], m[1][1], m[1][2] };
    int k;
    if (!norm3(f)) { f[0] = 0; f[1] = 0; f[2] = 1; }
    s_yaw = atan2(f[0], f[2]);
    s_pitch = asin(clampd(f[1], -1.0, 1.0));
    s_hand = 1.0; s_up_sign = 1.0;
    basis(f, r, u);
    if (dot3(r, gr) < 0) s_hand = -1.0;
    basis(f, r, u);
    if (dot3(u, gu) < 0) s_up_sign = -1.0;
    for (k = 0; k < 3; k++) s_pos[k] = m[3][k];
    s_started = 1;
}

static double stick(int v)
{
    const int a = v < 0 ? -v : v;
    if (a < PCS_DEAD) return 0.0;
    return (v < 0 ? -1.0 : 1.0) * (a - PCS_DEAD) / (32767.0 - PCS_DEAD);
}

void pc_split_camera(float m[4][3], const float p1[4][3])
{
    LARGE_INTEGER q, freq;
    uint8_t pad[20];
    double dt, f[3], r[3], u[3], flat[3], fwd, side, vert;
    int k;
    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&freq);
    memset(pad, 0, sizeof(pad));
    pc_split_pad_report(pad);
    if (!s_started || (pad[2] & 0x20)) {        /* Back: at player one's camera */
        start(p1);
        s_last_qpc = q.QuadPart;
    }
    dt = clampd((double)(q.QuadPart - s_last_qpc) / (double)freq.QuadPart, 0.0, 0.1);
    s_last_qpc = q.QuadPart;
    s_yaw -= s_hand * stick(get_axis(pad, 2)) * PCS_STICK_DEG * PCS_DEG * dt;
    s_pitch = clampd(s_pitch + stick(get_axis(pad, 3)) * PCS_STICK_DEG * PCS_DEG * dt, -89.0 * PCS_DEG, 89.0 * PCS_DEG);
    s_yaw = fmod(s_yaw, 2.0 * PCS_PI);
    fwd = stick(get_axis(pad, 1));
    side = stick(get_axis(pad, 0));
    vert = (pad[11] - pad[10]) / 255.0;
    basis(f, r, u);
    flat[0] = r[0]; flat[1] = 0; flat[2] = r[2];
    if (!norm3(flat)) { flat[0] = r[0]; flat[2] = r[2]; }
    /* the screen's right is the camera matrix's left */
    for (k = 0; k < 3; k++) s_pos[k] += (f[k] * fwd - flat[k] * side + (k == 1 ? vert : 0.0)) * PCS_SPEED * dt;
    for (k = 0; k < 3; k++) {
        m[0][k] = (float)r[k]; m[1][k] = (float)u[k]; m[2][k] = (float)f[k]; m[3][k] = (float)s_pos[k];
    }
}

#else

void pc_split_install(void) {}
void pc_split_pad_report(uint8_t out[20]) { (void)out; }
void pc_split_camera(float m[4][3], const float p1[4][3])
{
    int i, k;
    for (i = 0; i < 4; i++) for (k = 0; k < 3; k++) m[i][k] = p1[i][k];
}
void pc_split_note_frame(int split) { (void)split; }
int pc_split_frame(void) { return 0; }

#endif
