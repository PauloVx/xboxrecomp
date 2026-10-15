/*
 * pc_controls.h -- the player's controls, as the native Controls pages edit them and local\controls.ini keeps them.
 *
 * Keyboard and mouse are pc_input's bindings: RECOMP_BLACK_BIND_* (what the launcher exports) first, then the
 * file's [keyboard] section, which both the game and the launcher write.
 *
 * The controller is BLACK's own action map: which of its 28 controls each of the 37 action slots reads. The game
 * fills it from one of four presets ([0x2F3BD4], sub_001F1520). The player picks a layout -- one of those presets --
 * and may move single actions to other buttons; an action moved onto a button another listed action uses swaps
 * the two. Because the game reads its prompts and its keyboard routing from the same map, the button art in
 * "press ..." prompts and every keyboard key follow the change. The game project owns guest memory: it supplies
 * the preset tables and writes the map (PcControlsGame).
 *
 * Threads: the game thread (the menu tick, the game's own map fill). pc_input reads the keyboard bindings on the
 * input threads (see pc_input_bind).
 */
#ifndef XBOXRECOMP_PC_CONTROLS_H
#define XBOXRECOMP_PC_CONTROLS_H

#include <stddef.h>
#include <stdint.h>
#include "pc_input.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { PCC_PRESETS = 4, PCC_SLOTS = 37, PCC_CONTROLS = 28 };

typedef struct PcControlsGame {
    /* The control a preset gives a slot (0..27), -1 when the presets cannot be read yet. */
    int (*preset_control)(int preset, int slot);
    /* The layout changed: write it into the live map now (when there is one). */
    void (*apply)(void);
} PcControlsGame;

void pc_controls_set_game(const PcControlsGame *game);

/* Reads the file once (RECOMP_CONTROLS_INI, else local\controls.ini), after pc_input_init. */
void pc_controls_load(void);
/* Writes the file; 1 on success. Every change below saves by itself. */
int pc_controls_save(void);

/* ---- keyboard and mouse ---- */

/* Binds a captured source at a column (0 primary, 1 secondary); 0 clears. Saves. 0 when it cannot be bound. */
int pc_controls_key_bind(PcAction action, int column, unsigned source);
void pc_controls_key_reset(void);

/* ---- controller ---- */

/* Whether the Controls pages may move this action on the controller (a button action, not movement or pause). */
int pc_controls_pad_action(PcAction action);
/* Whether a control can be given to an action (buttons and the d-pad; START is kept for the pause menu). */
int pc_controls_pad_bindable(int control);
/* The layout in use, 0..3 (the game's own choice until the player picks one). */
int pc_controls_pad_layout(void);
/* Picks a layout, dir -1 / +1 through the four; per-action changes are dropped. Saves and applies. */
int pc_controls_pad_layout_step(int dir);
/* The control an action reads now, -1 when unknown. */
int pc_controls_pad_control(PcAction action);
/* Moves an action to a control (swapping with a listed action that had it); -1 puts it back as the layout has it. */
int pc_controls_pad_bind(PcAction action, int control);
void pc_controls_pad_reset(void);

/* The control of every slot for the game's map: the layout's preset, then the player's changes. The game calls it
 * after it filled the map itself (`preset` = the preset it used, its profile's choice). Returns 0 while the presets
 * cannot be read; `map` is then untouched. */
int pc_controls_pad_map(int preset, uint32_t map[PCC_SLOTS]);

/* The controls held in a 20-byte pad report, as a bit per control (bit i = control i; buttons and d-pad only). */
uint32_t pc_controls_pad_held(const uint8_t report[20]);

/* How a control is drawn: a glyph of the game's small menu font, and a short mark drawn on it ("LT", "START") or
 * NULL; a name for help texts ("RIGHT TRIGGER"). */
uint16_t pc_controls_pad_glyph(int control, const char **mark);
const char *pc_controls_pad_name(int control);

/* The slot an action is in the game's map, -1 for none. */
int pc_controls_action_slot(PcAction action);

#ifdef __cplusplus
}
#endif

#endif /* XBOXRECOMP_PC_CONTROLS_H */
