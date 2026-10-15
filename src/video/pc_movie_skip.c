/* BLACK's native XMV input. Decoder ownership and EOF stay with the guest. */
#include "pc_movie_skip.h"
#include "pc_menu.h"
#include "../input/pc_input.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#define MOVIE_LEASE_MS 500u
#define CONFIRM_MS 3000u
#define KEY_ESC 0x001u
#define KEY_ENTER 0x01Cu
#define KEY_SPACE 0x039u
#define KEY_NUMENTER 0x11Cu
#define KEY_F11 0x057u

static SRWLOCK m_lock = SRWLOCK_INIT;
static uint64_t m_session, m_seen, m_confirm;
static uint64_t m_started;
static int m_kind, m_request, m_release;
static int m_subtitle_target, m_subtitle_loaded;
static char m_subtitle_stem[16];
static int m_focused = 1;
#define PC_MOVIE_CAPTIONS 64
#define PC_MOVIE_CAPTION_CHARS PC_MOVIE_SUBTITLE_CHARS
typedef struct MovieCaption { uint64_t start, end; uint16_t text[PC_MOVIE_CAPTION_CHARS]; } MovieCaption;
static MovieCaption m_captions[PC_MOVIE_CAPTIONS];
static unsigned m_caption_count;
/* Physical state survives movie transitions, so a held button cannot affect the
 * next movie. Consumed releases also stay consumed after the movie ends. */
static unsigned m_keys, m_pad, m_mouse;
static unsigned char m_swallowed[0x200];
static unsigned m_mouse_swallowed;
static int m_pad_swallowed;
#ifdef PC_MOVIE_SKIP_TEST
static uint64_t m_test_now;
void pc_movie_skip_test_time(uint64_t now) { m_test_now = now; }
static uint64_t now_ms(void) { return m_test_now; }
#else
static uint64_t now_ms(void) { return GetTickCount64(); }
#endif

static void retire_locked(void)
{
    m_session = m_seen = m_confirm = 0;
    m_started = 0;
    m_kind = PC_MOVIE_SKIP_NONE;
    m_request = 0;
    m_subtitle_target = m_subtitle_loaded = 0;
    m_subtitle_stem[0] = 0;
    m_caption_count = 0;
    m_release = (m_keys | m_pad | m_mouse) != 0;
    pc_menu_set_movie_skip_prompt(0);
}

static int active_locked(uint64_t now)
{
    if (m_kind && (now < m_seen || now - m_seen > MOVIE_LEASE_MS)) retire_locked();
    if (m_confirm && now >= m_confirm) {
        m_confirm = 0;
        pc_menu_set_movie_skip_prompt(0);
    }
    return m_kind != PC_MOVIE_SKIP_NONE;
}

static void press_locked(uint64_t now)
{
    if (!m_focused || !active_locked(now) || m_release || m_request) return;
    m_release = 1;
    if (m_kind == PC_MOVIE_SKIP_INTRO || m_confirm) {
        m_request = 1;
        m_confirm = 0;
        pc_menu_set_movie_skip_prompt(0);
        fprintf(stderr, "[MOVIE] skip requested (%s)\n",
                m_kind == PC_MOVIE_SKIP_INTRO ? "intro" : "confirmed");
    } else {
        m_confirm = now + CONFIRM_MS;
        pc_menu_set_movie_skip_prompt(1);
        fprintf(stderr, "[MOVIE] confirmation: Press again to skip\n");
    }
}

static void release_locked(void)
{
    if (!(m_keys | m_pad | m_mouse)) m_release = 0;
}

void pc_movie_skip_update(uint64_t session, int kind, int playing)
{
    const uint64_t now = now_ms();
    AcquireSRWLockExclusive(&m_lock);
    active_locked(now);
    if (!playing || !session || kind < PC_MOVIE_SKIP_INTRO || kind > PC_MOVIE_SKIP_CUTSCENE) {
        retire_locked();
    } else {
        if (m_session != session || m_kind != kind) {
            retire_locked();
            m_session = session;
            m_kind = kind;
            m_started = now;
        }
        m_seen = now;
    }
    ReleaseSRWLockExclusive(&m_lock);
}

static int srt_time_ms(const char *s, uint64_t *out)
{
    int h, m, sec, ms;
    if (sscanf(s, "%d:%d:%d,%d", &h, &m, &sec, &ms) != 4 ||
        h < 0 || m < 0 || m > 59 || sec < 0 || sec > 59 || ms < 0 || ms > 999) return 0;
    *out = (((uint64_t)h * 60u + (unsigned)m) * 60u + (unsigned)sec) * 1000u + (unsigned)ms;
    return 1;
}

static int subtitle_supported_stem(const char *stem)
{
    static const char *const supported[] = {
        "03_N", "05_N", "07_N", "09_N", "11_N", "13_N", "15_N", "17_N", "18_N"
    };
    unsigned i;
    if (!stem || !*stem) return 0;
    for (i = 0; i < sizeof(supported) / sizeof(supported[0]); ++i)
        if (!_stricmp(stem, supported[i])) return 1;
    return 0;
}

static void subtitle_load_locked(void)
{
    FILE *f;
    char line[1024];
    char subtitle_path[MAX_PATH];
    const char *language = getenv("RECOMP_LANGUAGE");
    const char *directory = language && !_stricmp(language, "pt-BR") ? "subtitles\\pt-BR\\" :
        language && !_stricmp(language, "es") ? "subtitles\\es\\" :
        language && !_stricmp(language, "fr") ? "subtitles\\fr\\" :
        language && !_stricmp(language, "de") ? "subtitles\\de\\" :
        language && !_stricmp(language, "it") ? "subtitles\\it\\" : "subtitles\\";
    m_subtitle_loaded = 1;
    m_caption_count = 0;
    if (!m_subtitle_target || !m_subtitle_stem[0] ||
        _snprintf_s(subtitle_path, sizeof(subtitle_path), _TRUNCATE,
                    "%s%s.srt", directory, m_subtitle_stem) < 0) return;
    f = fopen(subtitle_path, "rb");
    if (!f) {
        fprintf(stderr, "[SUBTITLE] %s not found\n", subtitle_path);
        return;
    }
    while (m_caption_count < PC_MOVIE_CAPTIONS && fgets(line, sizeof(line), f)) {
        uint64_t start, end;
        char *arrow, joined[PC_MOVIE_CAPTION_CHARS * 4];
        size_t used = 0;
        int wide_count;
        while (fgets(line, sizeof(line), f)) {
            char *p = line;
            while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
            if (*p) break;
        }
        if (!line[0]) break;
        arrow = strstr(line, " --> ");
        if (!arrow) continue;
        *arrow = 0;
        if (!srt_time_ms(line, &start) || !srt_time_ms(arrow + 5, &end) || end <= start) continue;
        joined[0] = 0;
        while (fgets(line, sizeof(line), f)) {
            size_t n = strlen(line);
            while (n && (line[n - 1] == '\r' || line[n - 1] == '\n')) line[--n] = 0;
            if (!n) break;
            if (used && used + 1 < sizeof(joined)) joined[used++] = '\n';
            if (n > sizeof(joined) - used - 1) n = sizeof(joined) - used - 1;
            memcpy(joined + used, line, n); used += n; joined[used] = 0;
        }
        if (!used) continue;
        wide_count = MultiByteToWideChar(CP_UTF8, 0, joined, -1,
            (wchar_t *)m_captions[m_caption_count].text, PC_MOVIE_CAPTION_CHARS);
        if (!wide_count) continue;
        m_captions[m_caption_count].start = start;
        m_captions[m_caption_count].end = end;
        m_caption_count++;
    }
    fclose(f);
    fprintf(stderr, "[SUBTITLE] loaded %u cues for %s\n", m_caption_count, m_subtitle_stem);
}

void pc_movie_skip_subtitle_target(uint64_t session, const char *stem)
{
    AcquireSRWLockExclusive(&m_lock);
    if (active_locked(now_ms()) && session == m_session) {
        const int target = subtitle_supported_stem(stem);
        if (target != m_subtitle_target ||
            (target && _stricmp(stem, m_subtitle_stem) != 0)) {
            m_subtitle_target = target;
            if (target) strncpy_s(m_subtitle_stem, sizeof(m_subtitle_stem), stem, _TRUNCATE);
            else m_subtitle_stem[0] = 0;
            m_subtitle_loaded = 0;
            m_caption_count = 0;
        }
    }
    ReleaseSRWLockExclusive(&m_lock);
}

int pc_movie_skip_subtitle(uint16_t *out, unsigned capacity)
{
    uint64_t elapsed;
    unsigned i;
    int found = 0;
    if (!out || !capacity) return 0;
    out[0] = 0;
    AcquireSRWLockExclusive(&m_lock);
    if (active_locked(now_ms()) && m_subtitle_target && m_started) {
        if (!m_subtitle_loaded) subtitle_load_locked();
        elapsed = now_ms() - m_started;
        for (i = 0; i < m_caption_count; ++i) {
            if (elapsed >= m_captions[i].start && elapsed < m_captions[i].end) {
                unsigned n = 0;
                while (m_captions[i].text[n] && n + 1 < capacity) { out[n] = m_captions[i].text[n]; n++; }
                out[n] = 0;
                found = n != 0;
                break;
            }
        }
    }
    ReleaseSRWLockExclusive(&m_lock);
    return found;
}

int pc_movie_skip_take_request(uint64_t session)
{
    int take;
    AcquireSRWLockExclusive(&m_lock);
    take = active_locked(now_ms()) && session == m_session && m_request;
    if (take) retire_locked();
    ReleaseSRWLockExclusive(&m_lock);
    return take;
}

void pc_movie_skip_clear(void)
{
    AcquireSRWLockExclusive(&m_lock);
    retire_locked();
    ReleaseSRWLockExclusive(&m_lock);
}

int pc_movie_skip_active(void)
{
    int active;
    AcquireSRWLockExclusive(&m_lock);
    active = active_locked(now_ms());
    ReleaseSRWLockExclusive(&m_lock);
    return active;
}

static unsigned key_bit(unsigned id)
{
    switch (id) {
    case KEY_ESC: return 1u;
    case KEY_ENTER: return 2u;
    case KEY_SPACE: return 4u;
    case KEY_NUMENTER: return 8u;
    default: return 0u;
    }
}

static int f_key(unsigned id, int down, int repeat)
{
    const uint64_t now = now_ms();
    const unsigned bit = key_bit(id);
    int active, take;
    if (id >= sizeof(m_swallowed) || id == KEY_F11) return 0;
    AcquireSRWLockExclusive(&m_lock);
    active = active_locked(now);
    take = active || m_swallowed[id];
    if (down) {
        const int edge = bit && !(m_keys & bit) && !repeat;
        m_keys |= bit;
        if (active) m_swallowed[id] = 1;
        if (edge) press_locked(now);
    } else {
        m_keys &= ~bit;
        m_swallowed[id] = 0;
        release_locked();
    }
    ReleaseSRWLockExclusive(&m_lock);
    return take;
}

static int f_button(int button, int down)
{
    const uint64_t now = now_ms();
    unsigned bit;
    int active, take;
    if (button < 0 || button > 4) return 0;
    bit = 1u << button;
    AcquireSRWLockExclusive(&m_lock);
    active = active_locked(now);
    take = active || (m_mouse_swallowed & bit) != 0;
    if (down) {
        const int edge = button == 0 && !m_mouse;
        if (button == 0) m_mouse = 1;
        if (active) m_mouse_swallowed |= bit;
        if (edge) press_locked(now);
    } else {
        if (button == 0) m_mouse = 0;
        m_mouse_swallowed &= ~bit;
        release_locked();
    }
    ReleaseSRWLockExclusive(&m_lock);
    return take;
}

static int f_wheel(int delta) { (void)delta; return pc_movie_skip_active(); }

static int f_report(uint8_t out[20], int pad_present)
{
    const uint64_t now = now_ms();
    /* Xbox report: digital START, analog A/B. A real or scripted pad
     * enters the same filter before the guest can act on its own skip input. */
    const unsigned down = pad_present ?
        ((out[2] & 0x10u) ? 1u : 0u) |
        ((out[4] > 30u) ? 2u : 0u) | ((out[5] > 30u) ? 4u : 0u) : 0u;
    int active, take;
    AcquireSRWLockExclusive(&m_lock);
    active = active_locked(now);
    take = active || (m_pad_swallowed && down);
    {
        const unsigned edge = down & ~m_pad;
        m_pad = down;
        if (active && down) m_pad_swallowed = 1;
        if (edge) press_locked(now);
        if (!down) m_pad_swallowed = 0;
        release_locked();
    }
    ReleaseSRWLockExclusive(&m_lock);
    return take;
}

static void f_focus(int focused)
{
    AcquireSRWLockExclusive(&m_lock);
    m_focused = focused != 0;
    m_release = 1; /* A held pad on focus return needs a fresh release. */
    if (!focused) {
        m_confirm = m_request = 0;
        pc_menu_set_movie_skip_prompt(0);
        /* Key-up may go to another window; require a fresh release on return. */
        m_keys = m_mouse = 0;
        m_release = 1;
    }
    ReleaseSRWLockExclusive(&m_lock);
}

static const PcInputFilter k_filter = { f_key, f_button, f_wheel, f_report, f_focus };
void pc_movie_skip_install(void) { pc_input_set_filter_slot(2, &k_filter); }

#ifdef PC_MOVIE_SKIP_TEST
void pc_movie_skip_test_reset(void)
{
    AcquireSRWLockExclusive(&m_lock);
    m_keys = m_pad = m_mouse = m_mouse_swallowed = 0;
    m_pad_swallowed = 0;
    m_focused = 1;
    memset(m_swallowed, 0, sizeof(m_swallowed));
    retire_locked();
    ReleaseSRWLockExclusive(&m_lock);
}
#endif
