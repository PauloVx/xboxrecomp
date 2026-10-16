/*
 * pc_split.c -- split-screen: the layout of the views, and player two's free camera. See pc_split.h.
 */
#include "pc_split.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <xinput.h>

#define PCS_PI 3.14159265358979323846
#define PCS_DEG (PCS_PI / 180.0)
#define PCS_STICK_DEG 120.0              /* degrees a second, stick fully over */
#define PCS_SPEED 4.0                    /* metres a second; three times that with A held */
#define PCS_DEAD 7800                    /* about a quarter of the throw */

static volatile LONG s_frame;
static int s_started;
static double s_pos[3], s_yaw, s_pitch, s_hand = 1.0, s_up_sign = 1.0;
static LONGLONG s_last_qpc;

int pc_split_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("RECOMP_SPLITSCREEN");
        on = e && *e && *e != '0';
        if (on) fprintf(stderr, "[SPLIT] split-screen on: player two's camera on controller 2\n");
    }
    return on;
}

void pc_split_rect(int view, int r[4])
{
    r[0] = 0; r[2] = 640;
    r[1] = view == PCS_VIEW_P2 ? 240 : 0;
    r[3] = view == PCS_VIEW_P2 ? 480 : 240;
}

void pc_split_note_frame(int split) { InterlockedExchange(&s_frame, split != 0); }
int pc_split_frame(void) { return InterlockedCompareExchange(&s_frame, 0, 0) != 0; }

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

static double stick(SHORT v)
{
    const int a = v < 0 ? -v : v;
    if (a < PCS_DEAD) return 0.0;
    return (v < 0 ? -1.0 : 1.0) * (a - PCS_DEAD) / (32767.0 - PCS_DEAD);
}

void pc_split_camera(float m[4][3], const float p1[4][3])
{
    LARGE_INTEGER q, freq;
    XINPUT_STATE x;
    double dt, f[3], r[3], u[3], flat[3], fwd = 0, side = 0, vert = 0, speed = PCS_SPEED;
    int k, pad;
    QueryPerformanceCounter(&q);
    QueryPerformanceFrequency(&freq);
    pad = XInputGetState(1, &x) == ERROR_SUCCESS;
    if (!s_started || (pad && (x.Gamepad.wButtons & XINPUT_GAMEPAD_BACK))) { start(p1); s_last_qpc = q.QuadPart; }
    dt = clampd((double)(q.QuadPart - s_last_qpc) / (double)freq.QuadPart, 0.0, 0.1);
    s_last_qpc = q.QuadPart;
    if (pad) {
        s_yaw -= s_hand * stick(x.Gamepad.sThumbRX) * PCS_STICK_DEG * PCS_DEG * dt;
        s_pitch = clampd(s_pitch + stick(x.Gamepad.sThumbRY) * PCS_STICK_DEG * PCS_DEG * dt, -89.0 * PCS_DEG, 89.0 * PCS_DEG);
        s_yaw = fmod(s_yaw, 2.0 * PCS_PI);
        fwd = stick(x.Gamepad.sThumbLY);
        side = stick(x.Gamepad.sThumbLX);
        vert = (x.Gamepad.bRightTrigger - x.Gamepad.bLeftTrigger) / 255.0;
        if (x.Gamepad.wButtons & XINPUT_GAMEPAD_A) speed *= 3.0;
    }
    basis(f, r, u);
    flat[0] = r[0]; flat[1] = 0; flat[2] = r[2];
    if (!norm3(flat)) { flat[0] = r[0]; flat[2] = r[2]; }
    /* the screen's right is the camera matrix's left */
    for (k = 0; k < 3; k++) s_pos[k] += (f[k] * fwd - flat[k] * side + (k == 1 ? vert : 0.0)) * speed * dt;
    for (k = 0; k < 3; k++) {
        m[0][k] = (float)r[k]; m[1][k] = (float)u[k]; m[2][k] = (float)f[k]; m[3][k] = (float)s_pos[k];
    }
}

#else

int pc_split_enabled(void) { return 0; }
void pc_split_rect(int view, int r[4]) { (void)view; r[0] = r[1] = 0; r[2] = 640; r[3] = 480; }
void pc_split_camera(float m[4][3], const float p1[4][3])
{
    int i, k;
    for (i = 0; i < 4; i++) for (k = 0; k < 3; k++) m[i][k] = p1[i][k];
}
void pc_split_note_frame(int split) { (void)split; }
int pc_split_frame(void) { return 0; }

#endif
