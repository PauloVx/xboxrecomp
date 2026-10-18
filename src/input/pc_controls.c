/*
 * pc_controls.c -- the player's controls: keyboard bindings and controller layout, kept in local\controls.ini.
 * See pc_controls.h.
 */
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "pc_controls.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <windows.h>

/* BLACK's controls (names table 0x2AC290). The menu font draws the face buttons, White, Black and the d-pad
 * directions; triggers, START, BACK and the stick clicks are its blank button and stick with a mark on them. */
static const struct PadControl {
    const char *token, *name;
    uint16_t glyph;
    const char *mark;
} k_pad[PCC_CONTROLS] = {
    { "DpadUp", "D-PAD UP", 0x25B2, NULL }, { "DpadDown", "D-PAD DOWN", 0x25BC, NULL },
    { "DpadLeft", "D-PAD LEFT", 0x25C4, NULL }, { "DpadRight", "D-PAD RIGHT", 0x25BA, NULL },
    { "LeftStickUp", "LEFT STICK UP", 0x2518, "L" }, { "LeftStickDown", "LEFT STICK DOWN", 0x2518, "L" },
    { "LeftStickLeft", "LEFT STICK LEFT", 0x2518, "L" }, { "LeftStickRight", "LEFT STICK RIGHT", 0x2518, "L" },
    { "RightStickUp", "RIGHT STICK UP", 0x2518, "R" }, { "RightStickDown", "RIGHT STICK DOWN", 0x2518, "R" },
    { "RightStickLeft", "RIGHT STICK LEFT", 0x2518, "R" }, { "RightStickRight", "RIGHT STICK RIGHT", 0x2518, "R" },
    { "Start", "START", 0x2514, "START" }, { "Back", "BACK", 0x2514, "BACK" },
    { "LeftThumb", "LEFT STICK CLICK", 0x2518, "L" }, { "RightThumb", "RIGHT STICK CLICK", 0x2518, "R" },
    { "A", "A", 0x00B2, NULL }, { "B", "B", 0x00B9, NULL }, { "X", "X", 0x00B3, NULL }, { "Y", "Y", 0x00B0, NULL },
    { "Black", "BLACK", 0x250C, NULL }, { "White", "WHITE", 0x2510, NULL },
    { "LT", "LEFT TRIGGER", 0x2514, "LT" }, { "RT", "RIGHT TRIGGER", 0x2514, "RT" },
    { "LeftStickX", "LEFT STICK", 0x2518, "L" }, { "LeftStickY", "LEFT STICK", 0x2518, "L" },
    { "RightStickX", "RIGHT STICK", 0x2518, "R" }, { "RightStickY", "RIGHT STICK", 0x2518, "R" },
};
enum { C_START = 12 };

static const PcControlsGame *s_game;
static int s_loaded;
static int s_layout = -1;                 /* the player's layout 0..3, -1 = the game's own (its profile) */
static int s_game_preset;                 /* the preset the game last filled the map from */
static int s_override[PCC_SLOTS];         /* a control the player chose for a slot, -1 = as the layout has it */

void pc_controls_set_game(const PcControlsGame *game) { s_game = game; }

int pc_controls_action_slot(PcAction action)
{
    return (unsigned)action < PC_ACTION_COUNT ? pc_input_actions[action].slot : -1;
}

int pc_controls_pad_action(PcAction action)
{
    switch (action) {
    case PC_ACTION_FIRE: case PC_ACTION_AIM: case PC_ACTION_RELOAD: case PC_ACTION_CROUCH: case PC_ACTION_USE:
    case PC_ACTION_NEXT_WEAPON: case PC_ACTION_PREV_WEAPON: case PC_ACTION_GRENADE: case PC_ACTION_FIRE_MODE:
    case PC_ACTION_SUPPRESSOR: case PC_ACTION_MELEE: case PC_ACTION_HEALTH_PACK: case PC_ACTION_CYCLE_ITEM:
    case PC_ACTION_BACK:
        return 1;
    default:
        return 0;
    }
}

int pc_controls_pad_bindable(int control)
{
    return (control >= 0 && control <= 3) || (control >= 13 && control <= 23);
}

static int layout_preset(void) { return s_layout >= 0 ? s_layout : s_game_preset; }

static int preset_control(int preset, int slot)
{
    return s_game && s_game->preset_control ? s_game->preset_control(preset, slot) : -1;
}

static int slot_control(int slot)
{
    if (slot < 0 || slot >= PCC_SLOTS) return -1;
    return s_override[slot] >= 0 ? s_override[slot] : preset_control(layout_preset(), slot);
}

int pc_controls_pad_layout(void) { pc_controls_load(); return layout_preset(); }

int pc_controls_pad_control(PcAction action)
{
    pc_controls_load();
    return pc_controls_pad_action(action) ? slot_control(pc_controls_action_slot(action)) : -1;
}

int pc_controls_pad_map(int preset, uint32_t map[PCC_SLOTS])
{
    int slot, c;
    pc_controls_load();
    if (preset >= 0 && preset < PCC_PRESETS) s_game_preset = preset;
    for (slot = 0; slot < PCC_SLOTS; slot++) {
        c = slot_control(slot);
        if (c < 0) return 0;
        map[slot] = (uint32_t)c;
    }
    return 1;
}

static void changed(void)
{
    pc_controls_save();
    if (s_game && s_game->apply) s_game->apply();
}

int pc_controls_pad_layout_step(int dir)
{
    int i;
    pc_controls_load();
    if (!dir || preset_control(0, 0) < 0) return 0;
    s_layout = ((layout_preset() + (dir < 0 ? -1 : 1)) % PCC_PRESETS + PCC_PRESETS) % PCC_PRESETS;
    for (i = 0; i < PCC_SLOTS; i++) s_override[i] = -1;
    changed();
    return 1;
}

int pc_controls_pad_bind(PcAction action, int control)
{
    const int slot = pc_controls_action_slot(action);
    const int restoring = control < 0;
    int old, a;
    pc_controls_load();
    if (!pc_controls_pad_action(action) || slot < 0) return 0;
    /* Restoring a default can displace the action that took this button in an earlier swap. */
    if (restoring) control = preset_control(layout_preset(), slot);
    if (!pc_controls_pad_bindable(control) || (old = slot_control(slot)) < 0) return 0;
    if (old == control) {
        /* A loaded explicit default must stop pinning this button when the profile changes presets. */
        if (restoring) { s_override[slot] = -1; changed(); }
        return 1;
    }
    /* a listed action that had this control takes this one's old control */
    for (a = 0; a < PC_ACTION_COUNT; a++) {
        const int other = pc_controls_action_slot((PcAction)a);
        if (a == (int)action || !pc_controls_pad_action((PcAction)a) || other < 0 || slot_control(other) != control) continue;
        s_override[other] = old;
        if (preset_control(layout_preset(), other) == old) s_override[other] = -1;
    }
    s_override[slot] = preset_control(layout_preset(), slot) == control ? -1 : control;
    changed();
    return 1;
}

void pc_controls_pad_reset(void)
{
    int i;
    pc_controls_load();
    s_layout = -1;
    for (i = 0; i < PCC_SLOTS; i++) s_override[i] = -1;
    changed();
}

uint32_t pc_controls_pad_held(const uint8_t r[20])
{
    static const int analog[8] = { 16, 17, 18, 19, 20, 21, 22, 23 };   /* A B X Y Black White LT RT */
    uint32_t bits = 0;
    int i;
    for (i = 0; i < 8; i++) if (r[2] & (1u << i)) bits |= 1u << (i < 4 ? i : 8 + i);   /* d-pad 0..3, START..RTHUMB 12..15 */
    for (i = 0; i < 8; i++) if (r[4 + i] > 100) bits |= 1u << analog[i];
    return bits;
}

uint16_t pc_controls_pad_glyph(int control, const char **mark)
{
    if (mark) *mark = NULL;
    if (control < 0 || control >= PCC_CONTROLS) return 0;
    if (mark) *mark = k_pad[control].mark;
    return k_pad[control].glyph;
}

const char *pc_controls_pad_name(int control)
{
    return control >= 0 && control < PCC_CONTROLS ? k_pad[control].name : "";
}

/* ---- keyboard ---- */

int pc_controls_key_bind(PcAction action, int column, unsigned source)
{
    pc_controls_load();
    if (!pc_input_bind(action, column, source)) return 0;
    pc_controls_save();
    return 1;
}

void pc_controls_key_reset(void)
{
    pc_controls_load();
    pc_input_bindings_reset();
    pc_controls_save();
}

/* ---- the file ---- */

static const char *path(void)
{
    const char *p = getenv("RECOMP_CONTROLS_INI");
    return p && *p ? p : "local\\controls.ini";
}

static int action_named(const char *name)
{
    int a;
    for (a = 0; a < PC_ACTION_COUNT; a++) if (!_stricmp(pc_input_actions[a].name, name)) return a;
    return -1;
}

static int control_named(const char *name)
{
    int c;
    for (c = 0; c < PCC_CONTROLS; c++) if (!_stricmp(k_pad[c].token, name)) return c;
    return -1;
}

static char *trim(char *s)
{
    char *e;
    while (*s == ' ' || *s == '\t') s++;
    e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) *--e = 0;
    return s;
}

void pc_controls_load(void)
{
    FILE *f;
    char line[256];
    int section = 0, i, n = 0;             /* 1 keyboard, 2 controller */
    if (s_loaded) return;
    s_loaded = 1;
    for (i = 0; i < PCC_SLOTS; i++) s_override[i] = -1;
    pc_input_init();
    f = fopen(path(), "r");
    if (!f) return;
    while (fgets(line, sizeof(line), f)) {
        char *s = trim(line), *eq, *key, *val;
        if (!*s || *s == ';' || *s == '#') continue;
        if (*s == '[') {
            section = !_strnicmp(s, "[keyboard]", 10) ? 1 : !_strnicmp(s, "[controller]", 12) ? 2 : 0;
            continue;
        }
        if (!(eq = strchr(s, '='))) continue;
        *eq = 0;
        key = trim(s); val = trim(eq + 1);
        if (section == 1) {
            const int a = action_named(key);
            if (a >= 0 && pc_input_set_binding_names((PcAction)a, val)) n++;
        } else if (section == 2) {
            if (!_stricmp(key, "layout")) {
                const int v = atoi(val);
                s_layout = v >= 1 && v <= PCC_PRESETS ? v - 1 : -1;
            } else {
                const int a = action_named(key), c = control_named(val);
                if (a >= 0 && pc_controls_pad_action((PcAction)a) && pc_controls_pad_bindable(c)) {
                    s_override[pc_controls_action_slot((PcAction)a)] = c;
                    n++;
                }
            }
        }
    }
    fclose(f);
    fprintf(stderr, "[INPUT] %s: %d control setting(s)%s\n", path(), n, s_layout >= 0 ? ", controller layout chosen" : "");
}

int pc_controls_save(void)
{
    char tmp[600], names[128], dir[512], *slash;
    FILE *f;
    int a, ok;
    snprintf(dir, sizeof(dir), "%s", path());
    slash = strrchr(dir, '\\');
    if (!slash) slash = strrchr(dir, '/');
    if (slash) { *slash = 0; CreateDirectoryA(dir, NULL); }
    snprintf(tmp, sizeof(tmp), "%s.tmp", path());
    f = fopen(tmp, "w");
    if (!f) return 0;
    fprintf(f, "; BLACK controls. Edited by the in-game Controls pages and by the launcher.\n");
    fprintf(f, "; [keyboard] ACTION=keys as RECOMP_BLACK_BIND_<ACTION> (comma separated, None = unbound).\n");
    fprintf(f, "; [controller] layout=1..4 is one of the game's presets (absent: the profile's own); ACTION=CONTROL moves\n");
    fprintf(f, "; one button action (A B X Y Black White LT RT Back LeftThumb RightThumb DpadUp DpadDown DpadLeft DpadRight).\n");
    fprintf(f, "[keyboard]\n");
    for (a = 0; a < PC_ACTION_COUNT; a++) {
        pc_input_binding_names((PcAction)a, names, sizeof(names));
        fprintf(f, "%s=%s\n", pc_input_actions[a].name, names);
    }
    fprintf(f, "[controller]\n");
    if (s_layout >= 0) fprintf(f, "layout=%d\n", s_layout + 1);
    for (a = 0; a < PC_ACTION_COUNT; a++) {
        const int slot = pc_controls_action_slot((PcAction)a);
        if (pc_controls_pad_action((PcAction)a) && slot >= 0 && s_override[slot] >= 0)
            fprintf(f, "%s=%s\n", pc_input_actions[a].name, k_pad[s_override[slot]].token);
    }
    ok = fclose(f) == 0 && MoveFileExA(tmp, path(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    if (!ok) DeleteFileA(tmp);
    return ok;
}
