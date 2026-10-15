#ifndef XBOXRECOMP_PC_MOVIE_SKIP_H
#define XBOXRECOMP_PC_MOVIE_SKIP_H
#include <stdint.h>

enum { PC_MOVIE_SKIP_NONE, PC_MOVIE_SKIP_INTRO, PC_MOVIE_SKIP_CUTSCENE };
#define PC_MOVIE_SUBTITLE_CHARS 256

/* Game thread: refresh a playing movie's identity each pump. Zero playing or NONE
 * retires it. A new identity always starts with an unarmed confirmation. */
void pc_movie_skip_update(uint64_t session, int kind, int playing);
/* Selects the one movie stem that owns the current subtitle track. Call on the game thread. */
void pc_movie_skip_subtitle_target(uint64_t session, const char *stem);
/* Presenter thread: copies the active UTF-16 caption and returns 1 while a cue is visible. */
int pc_movie_skip_subtitle(uint16_t *out, unsigned capacity);
int pc_movie_skip_take_request(uint64_t session);
void pc_movie_skip_clear(void);
/* Input/presenter threads: a fresh native movie currently owns input. */
int pc_movie_skip_active(void);
/* Install ahead of menu input filters. Called once by the presenter. */
void pc_movie_skip_install(void);

#ifdef PC_MOVIE_SKIP_TEST
void pc_movie_skip_test_time(uint64_t now);
void pc_movie_skip_test_reset(void);
#endif
#endif
