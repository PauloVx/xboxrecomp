/*
 * pc_menu.h -- native, editable menus, drawn by the presenter over the game picture.
 *
 * A definition file (local\menus.ini; RECOMP_MENUS_INI moves it) describes pages by the names the
 * game itself gives them: a title, a few texts and a column of items whose actions are the game's
 * own menu calls. The game project supplies the services that touch guest state (PcMenuHost) and
 * calls pc_menu_tick once per front-end update on the game thread. A page the file does not list,
 * or lists with an error, is left to the game. The game project also supplies the built-in
 * definition, which is written to the file when the file does not exist, so there is always a
 * file to edit; RECOMP_NATIVE_MENUS=0 turns the native pages off for a run.
 *
 * Pages are drawn at window resolution into the presenter's overlay, in the original 640 x 480
 * menu space centred on the game picture, with fonts and images the game project registers from
 * the user's own game files (pc_menu_set_font / pc_menu_set_image).
 *
 * Threads: pc_menu_tick runs on the game thread; the input filter on the input threads (window
 * thread, OHCI thread); the overlay on the presenter thread. They share an action queue and a
 * display list under one lock; drawing works on a private copy of the display list.
 */
#ifndef XBOXRECOMP_PC_MENU_H
#define XBOXRECOMP_PC_MENU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- assets ------------------------------------------------------------------------------------------- */

enum { PCM_FONT_SMALL, PCM_FONT_BIG, PCM_FONT_COUNT };
enum { PCM_IMAGE_LOGO, PCM_IMAGE_BULLET, PCM_IMAGE_COUNT };

/* One glyph, in texels of the font's atlas: the cell (u, v, w, h), where the cell's top-left goes
 * relative to the pen (ox, oy) and how far the pen moves (advance). */
typedef struct PcMenuGlyph {
    uint16_t code;
    float u, v, w, h, ox, oy, advance;
} PcMenuGlyph;

/* A bitmap font: a straight-alpha BGRA atlas and its glyphs. `nominal` is the text size (in menu
 * units) at which one texel is one menu unit. Copied by pc_menu_set_font. */
typedef struct PcMenuFont {
    const uint32_t *pixels;
    unsigned width, height;
    float nominal;
    const PcMenuGlyph *glyphs;
    unsigned count;
} PcMenuFont;

int pc_menu_set_font(int slot, const PcMenuFont *font);
/* Any thread: a font has been successfully registered. Assets are registered during initialization;
 * use this to share that initialization between menus and movie hints without replacing a live atlas. */
int pc_menu_has_font(int slot);
/* A straight-alpha BGRA image, copied. */
int pc_menu_set_image(int slot, const uint32_t *pixels, unsigned width, unsigned height);

/* Decoders for the game's own files, in memory (pc_menu_assets.c). A "4.1v" bitmap font into a font slot;
 * images by id from an APT bundle's texture page (movie = the bundle's movie name, e.g. "core"). Return
 * 1 / the number of images registered, 0 when the data is not what they expect. */
int pc_menu_load_font_file(int slot, const uint8_t *data, size_t size);
int pc_menu_load_bundle_images(const uint8_t *data, size_t size, const char *movie, const uint32_t *ids, const int *slots, int n);
int pc_menu_decode_dxt(const uint8_t *src, size_t size, unsigned w, unsigned h, unsigned format, uint32_t *out);

/* ---- the game's services ------------------------------------------------------------------------------- */

/* Called on the game thread, from inside pc_menu_tick only. */
typedef struct PcMenuHost {
    void *ctx;
    /* The game's menu function NAME with up to three text arguments (NULL = "0"). Returns 1 when it was
     * requested, 0 when another request is still waiting and `force` is 0 (the original refuses then). */
    int (*gamecode)(void *ctx, const char *name, const char *const args[3], int force);
    /* Go to another page (the game unloads this one and loads that one). */
    void (*page)(void *ctx, const char *name);
    /* A number the game publishes to its menus (FEInvertLookFlag ...). 1 = found. */
    int (*value)(void *ctx, const char *name, int *out);
    /* Text for "$KEY" (the game's localized string) or "&KEY" (live text the game writes). Writes
     * UTF-16 and returns its length, or -1 when the game has no such text. */
    int (*text)(void *ctx, const char *spec, uint16_t *out, unsigned cap);
    /* Play one of the page's sound clips (SNDChange, SNDSelect, SNDBack, SNDError, SNDValueChange). */
    void (*sound)(void *ctx, const char *clip);
    /* Open the Video Settings page. */
    void (*video)(void *ctx);
    /* The game's own menu variables (what its pages keep in ActionScript, _global.NAME in BLACK's). `set` writes
     * them as well as the page's own, and @NAME reads them when no native page set it, so a page of the game's
     * own that follows a native one -- or the other way round -- finds what the first one set. May be NULL. */
    void (*set_var)(void *ctx, const char *name, const char *value);
    int (*get_var)(void *ctx, const char *name, char *out, unsigned cap);
    /* Close the game, the way closing its window does ("exit"). May be NULL. */
    void (*quit)(void *ctx);
} PcMenuHost;

/* ---- the game thread -------------------------------------------------------------------------------------- */

/* The definition the game ships with (UTF-8, the file syntax). Call before the first tick. */
void pc_menu_set_default_definition(const char *text);
/* Earlier built-in definitions, by the FNV-1a 64 hash of the file's bytes: a file that is still exactly one of them
 * (written by an older build, never edited) is replaced with the current one when it is read. */
void pc_menu_set_replaced_defaults(const uint64_t *fnv1a, int count);

/* Once per front-end update. `page` is the game's current page ("" or NULL for none), `ready` says it is
 * prepared and takes input. Returns 1 while a native page is shown: the game hides its own page then. */
int pc_menu_tick(const char *page, int ready, const PcMenuHost *host);

/* ---- presenter / input ----------------------------------------------------------------------------------- */

/* Registers the input filter and routes the Video Settings shortcut into the native action queue. */
void pc_menu_install(void);
/* Any thread: request native Video Settings. Revalidated and opened on the game thread. */
void pc_menu_request_video(void);

/* A native page is on screen (any thread). */
int pc_menu_active(void);
/* Window thread: the pointer moved to (x, y) in client pixels, the same space as pc_menu_overlay's picture
 * rectangle. On a native page the item under it takes the focus, and a left click acts on that item
 * (confirm; ON/OFF and sliders by where the click lands); a click away from the items does nothing. */
void pc_menu_pointer(int x, int y);

/* 1 when the player used a native page (a key, button or stick) since the last call; clears it (any thread). A
 * native page takes all input, so the game's own idle checks (an attract mode) would see none: the game project
 * counts this as input there. */
int pc_menu_take_input(void);

/* Any thread: show the cutscene's "Press again to skip" hint in the client area's bottom-right
 * corner, including when no native page is open. The game clears it on timeout, skip or movie end.
 * Uses the registered small font; this does not open a page or take input. */
void pc_menu_set_movie_skip_prompt(int visible);

/* Presenter thread: the overlay for a client area of w x h, with the game picture at (px, py, pw, ph).
 * Video Settings uses this same page renderer. The legacy overlay remains available for fallback runs. Same contract as
 * pc_video_menu_overlay: 1 = draw *pixels (w*h premultiplied BGRA, valid until the next call). */
int pc_menu_overlay(unsigned w, unsigned h, int px, int py, unsigned pw, unsigned ph,
                    const uint32_t **pixels, int *changed);

/* After pc_menu_overlay reported a change: the part of the pixels that changed (x, y, w, h), so only that part
 * needs uploading. 0 when it cannot say (upload everything). */
int pc_menu_overlay_dirty(unsigned box[4]);

/* Presenter thread: a box of label / value rows under a title, in the pages' look and fonts (the debug overlay), at
 * `scale` window pixels per menu unit. *pixels (premultiplied BGRA, *w x *h) stays valid until the next call. 0 when
 * no font is registered yet. */
int pc_menu_panel(const char *title, const char *const *labels, const char *const *values, int n, float scale,
                  const uint32_t **pixels, unsigned *w, unsigned *h);

/* ---- the definition -------------------------------------------------------------------------------------- */

/* Parses definition text. Returns the number of valid pages; errors are written one per line to `log`
 * ("menus.ini:LINE: message"). Used by the loader and the tests. */
int pc_menu_check_definition(const char *text, char *log, size_t log_cap);
const char *pc_menu_definition_path(void);

/* ---- tests ------------------------------------------------------------------------------------------------- */

/* START is the pad's start button: back on a page with "start = back", confirm elsewhere (as the original). */
enum { PCM_ACT_UP = 1, PCM_ACT_DOWN, PCM_ACT_LEFT, PCM_ACT_RIGHT, PCM_ACT_CONFIRM, PCM_ACT_BACK, PCM_ACT_START, PCM_ACT_VIDEO };
/* Tests: the menu-space to pixel mapping the pointer uses, as pc_menu_test_render sets it for w x h. */
void pc_menu_test_pointer_space(unsigned w, unsigned h);
void pc_menu_test_button(int button, int down);  /* a mouse button, through the native page's input filter */
int pc_menu_test_pointer_item(void);              /* the item under the pointer, -1 none */
void pc_menu_test_reset(const char *definition);   /* forget state, use this definition (no file) */
void pc_menu_test_action(int action);              /* as if the player pressed it */
int pc_menu_test_focus(void);                      /* index of the focused item, -1 none */
int pc_menu_test_load_file(const char *path);      /* read the definition from this file, as the game does; pages */
/* Renders the current display list into w x h premultiplied BGRA (picture = the whole buffer). */
int pc_menu_test_render(uint32_t *pixels, unsigned w, unsigned h);
/* The display list as text, one primitive per line (for tests and logs). */
size_t pc_menu_test_dump(char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* XBOXRECOMP_PC_MENU_H */
