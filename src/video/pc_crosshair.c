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

static SRWLOCK s_lock = SRWLOCK_INIT;
static double (*s_clock)(void);
static double s_last_tick = -1e9, s_kick_at = -1e9, s_mark_at = -1e9, s_shot_at = -1e9, s_drawn_at;
static float s_speed, s_kick, s_zoom = 1.0f, s_drawn_gap = GAP_REST;
static unsigned s_shots, s_have_shots;
static int s_control, s_mark_kind;

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
    s_last_tick = s_kick_at = s_mark_at = s_shot_at = -1e9; s_drawn_at = 0;
    s_speed = s_kick = 0; s_zoom = 1.0f; s_drawn_gap = GAP_REST; s_shots = s_have_shots = 0; s_control = 0;
    ReleaseSRWLockExclusive(&s_lock);
}

static float kick_now(double now)
{
    return s_kick * (float)exp(-(now - s_kick_at) / KICK_RECOVERY);
}

void pc_crosshair_tick(int in_control, float speed, unsigned shots, float zoom)
{
    const double now = now_seconds();
    AcquireSRWLockExclusive(&s_lock);
    s_last_tick = now;
    s_control = in_control;
    if (speed >= 0.0f && speed < 100.0f) s_speed += (speed - s_speed) * 0.35f;
    s_zoom = zoom > 0.0f && zoom <= 1.5f ? zoom : 1.0f;
    if (s_have_shots && shots > s_shots && shots - s_shots < 64u) {
        const float kick = kick_now(now) + SHOT_KICK * (float)(shots - s_shots);
        s_kick = kick < KICK_MAX ? kick : KICK_MAX;
        s_kick_at = s_shot_at = now;
    }
    s_shots = shots; s_have_shots = 1;
    ReleaseSRWLockExclusive(&s_lock);
}

void pc_crosshair_hit(int kind)
{
    const double now = now_seconds();
    AcquireSRWLockExclusive(&s_lock);
    /* a kill outranks a hit still showing */
    if (kind == PCX_MARK_KILL || now - s_mark_at > HIT_TIME || s_mark_kind != PCX_MARK_KILL) {
        s_mark_at = now;
        s_mark_kind = kind;
    }
    ReleaseSRWLockExclusive(&s_lock);
}

double pc_crosshair_since_shot(void)
{
    double since;
    AcquireSRWLockShared(&s_lock);
    since = now_seconds() - s_shot_at;
    ReleaseSRWLockShared(&s_lock);
    return since;
}

int pc_crosshair_frame(PcCrosshairDraw *out)
{
    PcVideoSettings v;
    double now, dt, since;
    float target, move, zoom_fade;
    int shown;
    pc_video_get(&v);
    memset(out, 0, sizeof(*out));
    if (v.crosshair == PCX_ORIGINAL && !v.hitmarkers) return 0;
    now = now_seconds();
    AcquireSRWLockExclusive(&s_lock);
    shown = s_control && now - s_last_tick < CONTROL_TIMEOUT;
    move = s_speed / RUN_SPEED;
    if (move > 1.6f) move = 1.6f;
    target = GAP_REST + MOVE_GAP * move + kick_now(now);
    if (target > GAP_MAX) target = GAP_MAX;
    dt = now - s_drawn_at;
    if (dt < 0 || dt > 0.25) dt = 0.25;
    s_drawn_gap += (target - s_drawn_gap) * (float)(1.0 - exp(-FOLLOW * dt));
    if (target > s_drawn_gap) s_drawn_gap = target;         /* a shot opens it at once; it closes smoothly */
    s_drawn_at = now;
    /* aiming down the sights: the weapon's own sights take over */
    zoom_fade = s_zoom < 0.9f ? 0.0f : s_zoom < 0.98f ? (s_zoom - 0.9f) / 0.08f : 1.0f;
    out->style = v.crosshair;
    out->gap = s_drawn_gap;
    out->length = 10.0f;
    out->thickness = 2.2f;
    out->alpha = v.crosshair == PCX_ORIGINAL ? 0.0f : 0.92f * zoom_fade;
    since = now - s_mark_at;
    if (v.hitmarkers && since >= 0) {
        const double life = s_mark_kind == PCX_MARK_KILL ? KILL_TIME : HIT_TIME;
        if (since < life) {
            const float t = (float)(since / life);
            out->marker_alpha = 1.0f - t * t;
            out->marker_kill = s_mark_kind == PCX_MARK_KILL;
            out->marker_inner = (out->marker_kill ? 9.0f : 7.0f) + 3.0f * t;
            out->marker_outer = out->marker_inner + (out->marker_kill ? 10.0f : 7.0f);
        }
    }
    ReleaseSRWLockExclusive(&s_lock);
    if (out->style == PCX_ORIGINAL) out->style = PCX_CROSS;   /* markers only */
    return shown && (out->alpha > 0.0f || out->marker_alpha > 0.0f);
}

#else
void pc_crosshair_tick(int in_control, float speed, unsigned shots, float zoom) { (void)in_control; (void)speed; (void)shots; (void)zoom; }
void pc_crosshair_hit(int kind) { (void)kind; }
double pc_crosshair_since_shot(void) { return 1e9; }
int pc_crosshair_frame(PcCrosshairDraw *out) { memset(out, 0, sizeof(*out)); return 0; }
void pc_crosshair_test_clock(double (*now)(void)) { (void)now; }
#endif
