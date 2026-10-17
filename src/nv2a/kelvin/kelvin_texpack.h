#ifndef KELVIN_TEXPACK_H
#define KELVIN_TEXPACK_H

/* Replacement textures (kelvin_texpack.c): a texture pack made for another version of the game (a PCSX2 pack: DDS and
 * PNG files named by the PS2's own hashes, which say nothing about this version's textures) in the folder named by
 * RECOMP_TEXTURE_PACK. A game texture is replaced by the pack's file that looks like it: each pack file is reduced to
 * a small fingerprint (its picture at 32x32), and a game texture's fingerprint has to be close to one file's and
 * clearly closer to it than to any other. */

#include <stdbool.h>
#include <stdint.h>

#ifdef _WIN32
#define COBJMACROS
#include <d3d11_4.h>

bool ktp_enabled(void);
/* The pack file that replaces a game texture whose top level is `rgba` (w x h, R in the low byte), or -1. */
int ktp_match(const uint32_t *rgba, unsigned w, unsigned h);
/* That file's texture, made on first use with `usage` and `misc`; a reference for the caller to release, or NULL. */
bool ktp_texture(ID3D11Device *dev, int index, D3D11_USAGE usage, UINT misc, ID3D11Texture2D **tex,
                 ID3D11ShaderResourceView **srv, size_t *bytes);
#endif

#endif
