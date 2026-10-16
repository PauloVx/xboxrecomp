/*
 * pc_debug.h -- the debug overlay (Insert): frame rate, frame times, latency, the network ping and what the renderer
 * and the game report, drawn by the presenter in the game's menu style (pc_menu_panel) over the picture's
 * top-left corner. Never in photos. RECOMP_DEBUG_OVERLAY=1 shows it from the start.
 *
 * Every producer reports from its own thread; the presenter asks for the rows a few times a second.
 */
#ifndef XBOXRECOMP_PC_DEBUG_H
#define XBOXRECOMP_PC_DEBUG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PC_DEBUG_ROWS 16
#define PC_DEBUG_LABEL 20
#define PC_DEBUG_VALUE 48
/* Rows the game project fills (position, players ...), after the presenter's and the renderer's. */
#define PC_DEBUG_GAME_ROWS 4

typedef struct PcDebugRow { char label[PC_DEBUG_LABEL], value[PC_DEBUG_VALUE]; } PcDebugRow;

/* Any thread. */
void pc_debug_toggle(void);
int pc_debug_visible(void);

/* The game published a frame (any thread); a performance-counter timestamp, stored with the frame. */
int64_t pc_debug_note_publish(void);
/* Presenter: a frame went to the screen. published: its pc_debug_note_publish stamp; first: its first present
 * (redraws of the same frame for an overlay are not frames). */
void pc_debug_note_present(int64_t published, int first);
/* Presenter: memory on the graphics card the process uses, in bytes (0: unknown). */
void pc_debug_note_vram(uint64_t bytes);
/* Renderer, once per frame: its GPU time (negative: not measured), draws, and the sizes rasterised and presented. */
void pc_debug_note_render(double gpu_ms, unsigned draws, unsigned raster_w, unsigned raster_h, unsigned out_w, unsigned out_h);
/* Network: the round trip to the other machine in milliseconds, or -1 when no session is connected (no PING row). */
void pc_debug_note_ping(int ms);
/* Game project: one of its rows (an empty label hides it). printf-style value. */
void pc_debug_game_row(unsigned slot, const char *label, const char *fmt, ...);

/* Presenter: the rows to show now. Returns how many; *changed says they differ from the last call's. */
int pc_debug_rows(PcDebugRow *rows, int max, int *changed);

/* Tests: a clock in seconds instead of the performance counter (NULL restores it), and a reset. */
void pc_debug_test_clock(double (*now)(void));
void pc_debug_test_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* XBOXRECOMP_PC_DEBUG_H */
