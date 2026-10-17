/* The PC crosshair and hit markers. See pc_crosshair.h. */
#include "pc_crosshair.h"
#include "pc_video.h"
#include <math.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

/* The look, in pixels of a 1080-line picture, and how it moves. */
#define GAP_REST 7.0f
#define GAP_MAX 46.0f
#define MOVE_GAP 11.0f          /* at a run */
#define RUN_SPEED 5.0f          /* world units a second */
#define SHOT_KICK 4.5f
#define KICK_MAX 26.0f
#define KICK_RECOVERY 0.16      /* seconds: the kick falls to a third */
#define FOLLOW 22.0             /* how fast the drawn gap follows its target, per second */
#define CONTROL_TIMEOUT 0.25    /* no tick for this long: the player is not in control */
#define HIT_TIME 0.30
#define KILL_TIME 0.55

/* One per player: split-screen's player two has its own. */
typedef struct Aim {
    double last_tick, kick_at, shot_at, drawn_at;
    float speed, kick, zoom, drawn_gap;
    unsigned shots, have_shots;
    int control;
} Aim;

static SRWLOCK s_lock = SRWLOCK_INIT;
static double (*s_clock)(void);
static double s_mark_at[2] = { -1e9, -1e9 };
static int s_mark_kind[2];
static Aim s_aim[2] = { { -1e9, -1e9, -1e9, 0, 0, 0, 1.0f, GAP_REST, 0, 0, 0 }, { -1e9, -1e9, -1e9, 0, 0, 0, 1.0f, GAP_REST, 0, 0, 0 } };

static double now_seconds(void)
{
    static LARGE_INTEGER frequency;
    LARGE_INTEGER t;
    if (s_clock) return s_clock();
    if (!frequency.QuadPart) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)frequency.QuadPart;
}

void pc_crosshair_test_clock(double (*now)(void))
{
    AcquireSRWLockExclusive(&s_lock);
    s_clock = now;
    s_mark_at[0] = s_mark_at[1] = -1e9;
    for (int i = 0; i < 2; i++) {
        Aim *a = &s_aim[i];
        a->last_tick = a->kick_at = a->shot_at = -1e9; a->drawn_at = 0;
        a->speed = a->kick = 0; a->zoom = 1.0f; a->drawn_gap = GAP_REST; a->shots = a->have_shots = 0; a->control = 0;
    }
    ReleaseSRWLockExclusive(&s_lock);
}

static float kick_now(const Aim *a, double now)
{
    return a->kick * (float)exp(-(now - a->kick_at) / KICK_RECOVERY);
}

void pc_crosshair_tick_for(int player, int in_control, float speed, unsigned shots, float zoom)
{
    const double now = now_seconds();
    Aim *const a = &s_aim[player ? 1 : 0];
    AcquireSRWLockExclusive(&s_lock);
    a->last_tick = now;
    a->control = in_control;
    if (speed >= 0.0f && speed < 100.0f) a->speed += (speed - a->speed) * 0.35f;
    a->zoom = zoom > 0.0f && zoom <= 1.5f ? zoom : 1.0f;
    if (a->have_shots && shots > a->shots && shots - a->shots < 64u) {
        const float kick = kick_now(a, now) + SHOT_KICK * (float)(shots - a->shots);
        a->kick = kick < KICK_MAX ? kick : KICK_MAX;
        a->kick_at = a->shot_at = now;
    }
    a->shots = shots; a->have_shots = 1;
    ReleaseSRWLockExclusive(&s_lock);
}

void pc_crosshair_tick(int in_control, float speed, unsigned shots, float zoom)
{
    pc_crosshair_tick_for(0, in_control, speed, shots, zoom);
}

void pc_crosshair_hit_for(int player, int kind)
{
    const double now = now_seconds();
    const int p = player ? 1 : 0;
    AcquireSRWLockExclusive(&s_lock);
    /* a kill outranks a hit still showing */
    if (kind == PCX_MARK_KILL || now - s_mark_at[p] > HIT_TIME || s_mark_kind[p] != PCX_MARK_KILL) {
        s_mark_at[p] = now;
        s_mark_kind[p] = kind;
    }
    ReleaseSRWLockExclusive(&s_lock);
}

void pc_crosshair_hit(int kind) { pc_crosshair_hit_for(0, kind); }

int pc_crosshair_replaces_dot_for(int player)
{
    PcVideoSettings v;
    int replaces;
    pc_video_get(&v);
    if (v.crosshair == PCX_ORIGINAL || player < 0 || player > 1) return 0;
    AcquireSRWLockShared(&s_lock);
    replaces = s_aim[player].control && now_seconds() - s_aim[player].last_tick < CONTROL_TIMEOUT &&
               s_aim[player].zoom >= 0.9f;
    ReleaseSRWLockShared(&s_lock);
    return replaces;
}

int pc_crosshair_replaces_dot(void) { return pc_crosshair_replaces_dot_for(0); }

double pc_crosshair_since_shot(void)
{
    double since;
    AcquireSRWLockShared(&s_lock);
    since = now_seconds() - s_aim[0].shot_at;
    ReleaseSRWLockShared(&s_lock);
    return since;
}

int pc_crosshair_frame_for(int player, PcCrosshairDraw *out)
{
    PcVideoSettings v;
    double now, dt, since;
    float target, move, zoom_fade;
    int shown;
    Aim *const a = &s_aim[player ? 1 : 0];
    pc_video_get(&v);
    memset(out, 0, sizeof(*out));
    if (v.crosshair == PCX_ORIGINAL && !v.hitmarkers) return 0;
    now = now_seconds();
    AcquireSRWLockExclusive(&s_lock);
    shown = a->control && now - a->last_tick < CONTROL_TIMEOUT;
    move = a->speed / RUN_SPEED;
    if (move > 1.6f) move = 1.6f;
    target = GAP_REST + MOVE_GAP * move + kick_now(a, now);
    if (target > GAP_MAX) target = GAP_MAX;
    dt = now - a->drawn_at;
    if (dt < 0 || dt > 0.25) dt = 0.25;
    a->drawn_gap += (target - a->drawn_gap) * (float)(1.0 - exp(-FOLLOW * dt));
    if (target > a->drawn_gap) a->drawn_gap = target;       /* a shot opens it at once; it closes smoothly */
    a->drawn_at = now;
    /* aiming down the sights: the weapon's own sights take over */
    zoom_fade = a->zoom < 0.9f ? 0.0f : a->zoom < 0.98f ? (a->zoom - 0.9f) / 0.08f : 1.0f;
    out->style = v.crosshair;
    out->gap = a->drawn_gap;
    out->length = 10.0f;
    out->thickness = 2.2f;
    out->alpha = v.crosshair == PCX_ORIGINAL ? 0.0f : 0.92f * zoom_fade;
    since = now - s_mark_at[player ? 1 : 0];
    if (v.hitmarkers && since >= 0) {
        const double life = s_mark_kind[player ? 1 : 0] == PCX_MARK_KILL ? KILL_TIME : HIT_TIME;
        if (since < life) {
            const float t = (float)(since / life);
            out->marker_alpha = 1.0f - t * t;
            out->marker_kill = s_mark_kind[player ? 1 : 0] == PCX_MARK_KILL;
            out->marker_inner = (out->marker_kill ? 9.0f : 7.0f) + 3.0f * t;
            out->marker_outer = out->marker_inner + (out->marker_kill ? 10.0f : 7.0f);
        }
    }
    ReleaseSRWLockExclusive(&s_lock);
    if (out->style == PCX_ORIGINAL) out->style = PCX_CROSS;   /* markers only */
    return shown && (out->alpha > 0.0f || out->marker_alpha > 0.0f);
}

int pc_crosshair_frame(PcCrosshairDraw *out) { return pc_crosshair_frame_for(0, out); }

#else
void pc_crosshair_tick(int in_control, float speed, unsigned shots, float zoom) { (void)in_control; (void)speed; (void)shots; (void)zoom; }
void pc_crosshair_tick_for(int player, int in_control, float speed, unsigned shots, float zoom) { (void)player; (void)in_control; (void)speed; (void)shots; (void)zoom; }
int pc_crosshair_frame_for(int player, PcCrosshairDraw *out) { (void)player; memset(out, 0, sizeof(*out)); return 0; }
void pc_crosshair_hit(int kind) { (void)kind; }
void pc_crosshair_hit_for(int player, int kind) { (void)player; (void)kind; }
double pc_crosshair_since_shot(void) { return 1e9; }
int pc_crosshair_replaces_dot(void) { return 0; }
int pc_crosshair_replaces_dot_for(int player) { (void)player; return 0; }
int pc_crosshair_frame(PcCrosshairDraw *out) { memset(out, 0, sizeof(*out)); return 0; }
void pc_crosshair_test_clock(double (*now)(void)) { (void)now; }
#endif
