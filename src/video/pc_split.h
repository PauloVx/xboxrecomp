/*
 * pc_split.h -- split-screen: the layout of the views, and player two's free camera.
 *
 * RECOMP_SPLITSCREEN=1 draws the world twice a frame: player one in the top half, player two in the bottom half.
 * Player two's camera is a free camera on the second controller (XInput user 1), read here on the host:
 *
 *   left stick move, right stick look, triggers down / up, A held faster, Back puts the camera at player one's
 *
 * The game project owns guest memory: it asks pc_split_camera for player two's camera every frame and marks where the
 * views begin in the pushbuffer (kelvin_mark_view). Rectangles are in pixels of the game's 640x480 picture.
 *
 * Threads: pc_split_camera on the game thread, pc_split_note_frame on the GPU thread, pc_split_frame on the presenter.
 */
#ifndef XBOXRECOMP_PC_SPLIT_H
#define XBOXRECOMP_PC_SPLIT_H

#ifdef __cplusplus
extern "C" {
#endif

enum { PCS_VIEW_P1, PCS_VIEW_P2, PCS_VIEW_HUD, PCS_VIEW_COUNT };

int pc_split_enabled(void);
/* The view's rectangle: x0, y0, x1, y1. */
void pc_split_rect(int view, int r[4]);

/* Player two's camera: rows right, up, at (forward), position. `p1` is player one's camera this frame, where player
 * two's starts (and goes back to on Back). */
void pc_split_camera(float m[4][3], const float p1[4][3]);

/* The frame just finished was drawn split (GPU thread), and the last one was (presenter). */
void pc_split_note_frame(int split);
int pc_split_frame(void);

#ifdef __cplusplus
}
#endif

#endif /* XBOXRECOMP_PC_SPLIT_H */
