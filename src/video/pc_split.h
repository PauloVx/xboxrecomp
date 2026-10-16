/*
 * pc_split.h -- split-screen: the layout of the views, and player two's free camera.
 *
 * The Split Screen setting (pc_video.c; RECOMP_SPLITSCREEN=1 for a run) draws the world twice a frame: player one in
 * the top half, player two in the bottom half. Player two's camera is a free camera, read here on the host:
 *
 *   second controller (XInput user 1)  left stick move, right stick look, triggers down / up, A held faster,
 *                                      Back puts the camera at player one's
 *   keyboard                           arrows move, numpad 8 4 2 6 look, numpad 9 / 3 up / down, numpad 0 held faster,
 *                                      numpad . puts the camera at player one's
 *
 * The game project owns guest memory: it asks pc_split_camera for player two's camera every frame and marks where the
 * views begin in the pushbuffer (kelvin_mark_view). Rectangles are in pixels of the game's 640x480 picture.
 *
 * Threads: keys on the input threads, pc_split_camera on the game thread, pc_split_note_frame on the GPU thread,
 * pc_split_frame on the presenter.
 */
#ifndef XBOXRECOMP_PC_SPLIT_H
#define XBOXRECOMP_PC_SPLIT_H

#ifdef __cplusplus
extern "C" {
#endif

enum { PCS_OFF, PCS_ON, PCS_SETTING_COUNT };

/* The setting in force, or RECOMP_SPLITSCREEN (1 on, 0 off) when set. */
int pc_split_enabled(void);
/* Player `player`'s (0 or 1) view and HUD: x0, y0, x1, y1; the whole picture when `split` is 0. */
void pc_split_rect_for(int split, int player, int r[4]);
void pc_split_rect(int player, int r[4]);

/* Registers player two's keys with pc_input. Once, by the presenter. */
void pc_split_install(void);

/* Player two's camera: rows right, up, at (forward), position. `p1` is player one's camera this frame, where player
 * two's starts (and goes back to). */
void pc_split_camera(float m[4][3], const float p1[4][3]);

/* Whether the frame just finished was drawn split (GPU thread), and the last one (presenter). */
void pc_split_note_frame(int split);
int pc_split_frame(void);

#ifdef __cplusplus
}
#endif

#endif /* XBOXRECOMP_PC_SPLIT_H */
