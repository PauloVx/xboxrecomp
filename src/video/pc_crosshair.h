/*
 * pc_crosshair.h -- the PC crosshair and hit markers, drawn by the presenter over the game picture (never in photos).
 *
 * The game thread reports what it sees once per update (pc_crosshair_tick): whether the player is in control, how
 * fast the player moves, the mission's bullets-fired count and the camera zoom. Each bullet kicks the crosshair
 * open; it closes again within a fraction of a second, and moving opens it as far as the speed asks. A kill (any
 * thread, pc_crosshair_hit) flashes a marker. The presenter asks pc_crosshair_frame for the look of the moment.
 *
 * Units are pixels of a 1080-line picture; the presenter scales them to the picture it draws.
 */
#ifndef XBOXRECOMP_PC_CROSSHAIR_H
#define XBOXRECOMP_PC_CROSSHAIR_H

#ifdef __cplusplus
extern "C" {
#endif

/* PcVideoSettings.crosshair: ORIGINAL leaves the game's own dot alone. */
enum { PCX_ORIGINAL, PCX_CROSS, PCX_T, PCX_CIRCLE, PCX_STYLE_COUNT };
enum { PCX_MARK_HIT, PCX_MARK_KILL };

typedef struct PcCrosshairDraw {
    unsigned style;              /* PCX_* (never ORIGINAL when something is drawn) */
    float gap, length, thickness, alpha;
    float marker_alpha, marker_inner, marker_outer;   /* 0 alpha: no marker */
    int marker_kill;
} PcCrosshairDraw;

/* Game thread, once per update. shots: the mission's bullets fired so far (it may restart at 0). zoom: the camera's
 * zoom compensation, 1 unzoomed and smaller when aiming. */
void pc_crosshair_tick(int in_control, float speed, unsigned shots, float zoom);
/* Split-screen: player 1 is player two, with a crosshair and hit markers of its own. */
void pc_crosshair_tick_for(int player, int in_control, float speed, unsigned shots, float zoom);
/* Any thread: something the player shot was hit (PCX_MARK_HIT) or killed (PCX_MARK_KILL). */
void pc_crosshair_hit(int kind);
void pc_crosshair_hit_for(int player, int kind);
/* Whether the game's own aiming dot should go: another style is chosen, the player is in control and not aiming
 * down the sights (where the dot comes back, as the crosshair goes). */
int pc_crosshair_replaces_dot(void);
/* Seconds since the player last fired, a large number before the first shot. */
double pc_crosshair_since_shot(void);
/* Presenter: the crosshair and marker to draw now. 0 when there is nothing (another style, no control, paused...). */
int pc_crosshair_frame(PcCrosshairDraw *out);
int pc_crosshair_frame_for(int player, PcCrosshairDraw *out);

/* Tests: a clock in seconds instead of the performance counter (NULL restores it). */
void pc_crosshair_test_clock(double (*now)(void));

#ifdef __cplusplus
}
#endif

#endif /* XBOXRECOMP_PC_CROSSHAIR_H */
