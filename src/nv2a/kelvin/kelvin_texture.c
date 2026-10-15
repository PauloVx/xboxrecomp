/*
 * Kelvin texture unit: fetch, address and filter a texture stage.
 *
 * Formats and channel mapping follow xemu's kelvin_color_format_gl_map
 * (hw/xbox/nv2a/pgraph/gl/constants.h) and the conversions in
 * pgraph/texture.c; the texture shape (levels, clamps, dimensions) follows
 * pgraph_get_texture_shape. Sampling reproduces the GL sampler state xemu
 * programs (gl/texture.c apply_texture_parameters): wrap/mirror/clamp/border
 * addressing, nearest/linear filtering with texel centres at +0.5, and mipmap
 * selection from the screen-space footprint.
 *
 * One deliberate difference: TEXCTL0 MIN/MAX_LOD_CLAMP are 12-bit fields,
 * read here as 4.8 fixed point. xemu uses the raw values as level numbers,
 * which agrees whenever the minimum is 0 (D3D's default).
 *
 * Every texel is read from guest memory when it is sampled, so a texture the
 * CPU rewrites between draws (video, render targets) is always current. The
 * one cache, of decoded DXT blocks, is keyed by the block's bytes.
 *
 * Copyright (c) 2012 espes; 2015 Jannik Vogel; 2018-2025 Matt Borgerson
 * (xemu, whose format tables and shape rules these are)
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include "kelvin_render.h"

#include <emmintrin.h>

/* enum TexelKind (how a format's bytes become RGBA) is in kelvin_render.h, for
 * the GPU backend's texture decoder. */

typedef struct TexFormat {
    unsigned char bpp;      /* bytes per texel; 4 for DXT per xemu's table */
    unsigned char linear;
    unsigned char kind;
    unsigned char depth;
} TexFormat;

static TexFormat s_formats[66];

static void init_formats(void)
{
#define F(fmt, b, lin, k) s_formats[NV097_SET_TEXTURE_FORMAT_COLOR_##fmt] = \
        (TexFormat){ b, lin, k, 0 }
#define D(fmt, b, lin) s_formats[NV097_SET_TEXTURE_FORMAT_COLOR_##fmt] = \
        (TexFormat){ b, lin, TK_UNSUPPORTED, 1 }
    F(SZ_Y8, 1, 0, TK_Y8);
    F(SZ_AY8, 1, 0, TK_AY8);
    F(SZ_A1R5G5B5, 2, 0, TK_1555);
    F(SZ_X1R5G5B5, 2, 0, TK_X555);
    F(SZ_A4R4G4B4, 2, 0, TK_4444);
    F(SZ_R5G6B5, 2, 0, TK_565);
    F(SZ_A8R8G8B8, 4, 0, TK_8888);
    F(SZ_X8R8G8B8, 4, 0, TK_X888);
    F(SZ_I8_A8R8G8B8, 1, 0, TK_I8);
    F(L_DXT1_A1R5G5B5, 4, 0, TK_DXT1);
    F(L_DXT23_A8R8G8B8, 4, 0, TK_DXT3);
    F(L_DXT45_A8R8G8B8, 4, 0, TK_DXT5);
    F(LU_IMAGE_A1R5G5B5, 2, 1, TK_1555);
    F(LU_IMAGE_R5G6B5, 2, 1, TK_565);
    F(LU_IMAGE_A8R8G8B8, 4, 1, TK_8888);
    F(LU_IMAGE_Y8, 1, 1, TK_Y8);
    F(LU_IMAGE_G8B8, 2, 1, TK_G8B8);
    F(SZ_A8, 1, 0, TK_A8);
    F(SZ_A8Y8, 2, 0, TK_A8Y8);
    F(LU_IMAGE_AY8, 1, 1, TK_AY8);
    F(LU_IMAGE_X1R5G5B5, 2, 1, TK_X555);
    F(LU_IMAGE_A4R4G4B4, 2, 1, TK_4444);
    F(LU_IMAGE_X8R8G8B8, 4, 1, TK_X888);
    F(LU_IMAGE_A8, 1, 1, TK_A8);
    F(LU_IMAGE_A8Y8, 2, 1, TK_A8Y8);
    F(SZ_R6G5B5, 2, 0, TK_UNSUPPORTED);   /* signed; xemu: FIXME */
    F(SZ_G8B8, 2, 0, TK_G8B8);
    F(SZ_R8B8, 2, 0, TK_R8B8);
    F(LC_IMAGE_CR8YB8CB8YA8, 2, 1, TK_YUY2);
    F(LC_IMAGE_YB8CR8YA8CB8, 2, 1, TK_UYVY);
    D(SZ_DEPTH_Y16_FIXED, 2, 0);
    D(LU_IMAGE_DEPTH_X8_Y24_FIXED, 4, 1);
    D(LU_IMAGE_DEPTH_X8_Y24_FLOAT, 4, 1);
    D(LU_IMAGE_DEPTH_Y16_FIXED, 2, 1);
    D(LU_IMAGE_DEPTH_Y16_FLOAT, 2, 1);
    F(LU_IMAGE_Y16, 2, 1, TK_Y16);
    F(SZ_A8B8G8R8, 4, 0, TK_ABGR);
    F(SZ_B8G8R8A8, 4, 0, TK_BGRA);
    F(SZ_R8G8B8A8, 4, 0, TK_RGBA);
    F(LU_IMAGE_A8B8G8R8, 4, 1, TK_ABGR);
    F(LU_IMAGE_B8G8R8A8, 4, 1, TK_BGRA);
    F(LU_IMAGE_R8G8B8A8, 4, 1, TK_RGBA);
#undef F
#undef D
}

/* pdep for the swizzle masks: scatter the bits of v into mask. */
static uint32_t deposit(uint32_t v, uint32_t mask)
{
    uint32_t out = 0, bit;
    for (bit = 1; mask; bit <<= 1) {
        uint32_t low = mask & (~mask + 1);
        if (v & bit) {
            out |= low;
        }
        mask &= mask - 1;
    }
    return out;
}

KELVIN_INLINE bool is_pow2(unsigned int v)
{
    return v && !(v & (v - 1));
}

/* log2 of a power of two. */
KELVIN_INLINE unsigned int log2_pow2(unsigned int v)
{
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward(&i, v);
    return (unsigned int)i;
#else
    return (unsigned int)__builtin_ctz(v);
#endif
}

/* The low 16 bits of v, moved to the even bit positions. */
KELVIN_INLINE uint32_t spread_bits(uint32_t v)
{
    v &= 0xFFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

/* deposit(x, mask_x) | deposit(y, mask_y) for a level whose masks are
 * swizzle_masks(w, h): while both coordinates have bits left they
 * alternate, x first, and the longer one's remaining bits follow. That is
 * exactly those masks when w and h are powers of two, as swizzled levels
 * are; anything else takes the general deposit. */
KELVIN_INLINE uint32_t swizzle_offset(const KTexture *t, unsigned int level,
                               unsigned int x, unsigned int y)
{
    unsigned int w = t->level_w[level], h = t->level_h[level], m;
    uint32_t low;

    if (!is_pow2(w) || !is_pow2(h) || w > 0x10000 || h > 0x10000) {
        return deposit(x, t->mask_x[level]) | deposit(y, t->mask_y[level]);
    }
    m = log2_pow2(MIN(w, h));
    low = (1u << m) - 1;
    return spread_bits(x & low) | (spread_bits(y & low) << 1) |
           (((x >> m) | (y >> m)) << (2 * m));
}

/* swizzle_offset(t, level, x, y) as the OR of an x part and a y part:
 * swz_part(t, level, x, 0) | swz_part(t, level, y, 1). Both forms above
 * split so: deposit per coordinate, and the shift of an OR is the OR of the
 * shifts. */
KELVIN_INLINE uint32_t swz_part(const KTexture *t, unsigned int level,
                                unsigned int c, int is_y)
{
    unsigned int w = t->level_w[level], h = t->level_h[level], m;
    uint32_t low;

    if (!is_pow2(w) || !is_pow2(h) || w > 0x10000 || h > 0x10000) {
        return deposit(c, is_y ? t->mask_y[level] : t->mask_x[level]);
    }
    m = log2_pow2(MIN(w, h));
    low = (1u << m) - 1;
    return (spread_bits(c & low) << is_y) | ((c >> m) << (2 * m));
}

/* xemu swizzle.c generate_swizzle_masks, 2D. */
static void swizzle_masks(unsigned int width, unsigned int height,
                          uint32_t *mx, uint32_t *my)
{
    uint32_t x = 0, y = 0, bit = 1, mask_bit = 1;
    bool done;
    do {
        done = true;
        if (bit < width) { x |= mask_bit; mask_bit <<= 1; done = false; }
        if (bit < height) { y |= mask_bit; mask_bit <<= 1; done = false; }
        bit <<= 1;
    } while (!done);
    *mx = x;
    *my = y;
}

KELVIN_INLINE float lin8(uint32_t v)
{
    return (float)v * (1.0f / 255.0f);
}

void ktex_init(void)
{
    static bool done;
    if (!done) {
        init_formats();
        done = true;
    }
}

/* xemu pgraph_get_texture_shape, plus the addressing facts sampling needs. */
void ktex_setup(KelvinState *pg, int i, KTexture *t)
{
    uint32_t ctl_0 = pgraph_reg_r(pg, NV_PGRAPH_TEXCTL0_0 + i * 4);
    uint32_t ctl_1 = pgraph_reg_r(pg, NV_PGRAPH_TEXCTL1_0 + i * 4);
    uint32_t fmt = pgraph_reg_r(pg, NV_PGRAPH_TEXFMT0 + i * 4);
    uint32_t filter = pgraph_reg_r(pg, NV_PGRAPH_TEXFILTER0 + i * 4);
    uint32_t address = pgraph_reg_r(pg, NV_PGRAPH_TEXADDRESS0 + i * 4);
    uint32_t rect = pgraph_reg_r(pg, NV_PGRAPH_TEXIMAGERECT0 + i * 4);
    uint32_t offset = pgraph_reg_r(pg, NV_PGRAPH_TEXOFFSET0 + i * 4);
    unsigned int log_w, log_h, log_d, level;
    uint32_t dma, lod_bias, total;

    ktex_init();
    memset(t, 0, sizeof(*t));
    t->color_format = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_COLOR);
    if (t->color_format >= ARRAY_SIZE(s_formats) ||
        !s_formats[t->color_format].bpp) {
        kelvin_unsupported(KU_TEX_FORMAT, "unknown texture format",
                           t->color_format);
        return;
    }
    t->bpp = s_formats[t->color_format].bpp;
    t->linear = s_formats[t->color_format].linear;
    t->kind = s_formats[t->color_format].kind;
    t->depth_format = s_formats[t->color_format].depth;
    t->cubemap = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_CUBEMAPENABLE);
    t->dimensionality = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_DIMENSIONALITY);
    t->border = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_BORDER_SOURCE)
                != NV_PGRAPH_TEXFMT0_BORDER_SOURCE_COLOR;
    log_w = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_BASE_SIZE_U);
    log_h = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_BASE_SIZE_V);
    log_d = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_BASE_SIZE_P);
    t->levels = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_MIPMAP_LEVELS);
    t->min_lod = GET_MASK(ctl_0, NV_PGRAPH_TEXCTL0_0_MIN_LOD_CLAMP) / 256.0f;
    t->max_lod = GET_MASK(ctl_0, NV_PGRAPH_TEXCTL0_0_MAX_LOD_CLAMP) / 256.0f;
    t->alphakill = ctl_0 & NV_PGRAPH_TEXCTL0_0_ALPHAKILLEN;
    t->colorkey_mode = GET_MASK(ctl_0, NV_PGRAPH_TEXCTL0_0_COLORKEYMODE);

    t->min_filter = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MIN);
    t->mag_filter = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MAG);
    lod_bias = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MIPMAP_LOD_BIAS);
    {
        int b = (int)lod_bias;
        if (lod_bias & (1 << 12)) {
            b |= ~NV_PGRAPH_TEXFILTER0_MIPMAP_LOD_BIAS;
        }
        t->lod_bias = (float)b / 256.0f;
    }
    if (filter & (NV_PGRAPH_TEXFILTER0_ASIGNED | NV_PGRAPH_TEXFILTER0_RSIGNED |
                  NV_PGRAPH_TEXFILTER0_GSIGNED | NV_PGRAPH_TEXFILTER0_BSIGNED)) {
        kelvin_unsupported(KU_TEX_FORMAT, "signed texture channels", filter);
    }
    t->addr[0] = GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRU);
    t->addr[1] = GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRV);
    t->addr[2] = GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRP);
    t->border_color = pgraph_reg_r(pg, NV_PGRAPH_BORDERCOLOR0 + i * 4);

    if (t->linear) {
        t->width = GET_MASK(rect, NV_PGRAPH_TEXIMAGERECT0_WIDTH);
        t->height = GET_MASK(rect, NV_PGRAPH_TEXIMAGERECT0_HEIGHT);
        t->depth = 1;
        t->pitch = GET_MASK(ctl_1, NV_PGRAPH_TEXCTL1_0_IMAGE_PITCH);
        t->levels = 1;
        /* xemu: mipmap min filters on a linear texture fall back to LOD0 */
        switch (t->min_filter) {
        case NV_PGRAPH_TEXFILTER0_MIN_BOX_NEARESTLOD:
        case NV_PGRAPH_TEXFILTER0_MIN_BOX_TENT_LOD:
            t->min_filter = NV_PGRAPH_TEXFILTER0_MIN_BOX_LOD0;
            break;
        case NV_PGRAPH_TEXFILTER0_MIN_TENT_NEARESTLOD:
        case NV_PGRAPH_TEXFILTER0_MIN_TENT_TENT_LOD:
            t->min_filter = NV_PGRAPH_TEXFILTER0_MIN_TENT_LOD0;
            break;
        }
    } else {
        t->width = 1u << log_w;
        t->height = 1u << log_h;
        t->depth = 1u << log_d;
        t->levels = MIN(t->levels, (unsigned)(MAX(log_w, log_h) + 1));
        if (t->levels == 0) {
            t->levels = 1;
        }
    }
    if (t->max_lod > (float)(t->levels - 1)) {
        t->max_lod = (float)(t->levels - 1);
    }
    if (t->min_lod > t->max_lod) {
        t->min_lod = t->max_lod;
    }

    dma = GET_MASK(fmt, NV_PGRAPH_TEXFMT0_CONTEXT_DMA) ? pg->dma_b : pg->dma_a;
    t->base = kelvin_dma_va(dma, offset, 1);
    if (!t->base) {
        kelvin_unsupported(KU_DMA_UNRESOLVED, "texture address", offset);
        return;
    }
    t->mem = kelvin_guest(t->base);

    /* Mip chain layout (xemu pgraph_get_texture_length). */
    total = 0;
    for (level = 0; level < t->levels && level < KTEX_MAX_LEVELS; level++) {
        unsigned int w = MAX(t->width >> level, 1u);
        unsigned int h = MAX(t->height >> level, 1u);
        t->level_offset[level] = total;
        t->level_w[level] = w;
        t->level_h[level] = h;
        if (t->kind == TK_DXT1 || t->kind == TK_DXT3 || t->kind == TK_DXT5) {
            unsigned int block = t->kind == TK_DXT1 ? 8 : 16;
            total += ((w + 3) / 4) * ((h + 3) / 4) * block;
        } else if (t->linear) {
            total += t->pitch * h;
        } else {
            total += w * h * t->bpp;
            swizzle_masks(w, h, &t->mask_x[level], &t->mask_y[level]);
        }
    }
    t->face_stride = (total + NV2A_CUBEMAP_FACE_ALIGNMENT - 1)
                     & ~(NV2A_CUBEMAP_FACE_ALIGNMENT - 1);

    if (t->kind == TK_I8) {
        uint32_t pal = pgraph_reg_r(pg, NV_PGRAPH_TEXPALETTE0 + i * 4);
        uint32_t pdma = GET_MASK(pal, NV_PGRAPH_TEXPALETTE0_CONTEXT_DMA)
                            ? pg->dma_b : pg->dma_a;
        static const unsigned int lengths[4] = { 256, 128, 64, 32 };
        t->palette_len = lengths[GET_MASK(pal, NV_PGRAPH_TEXPALETTE0_LENGTH)];
        t->palette = kelvin_dma_va(pdma, pal & NV_PGRAPH_TEXPALETTE0_OFFSET,
                                   t->palette_len * 4);
        t->palette_mem = t->palette ? kelvin_guest(t->palette) : NULL;
    }

    if (t->kind == TK_UNSUPPORTED) {
        kelvin_unsupported(t->depth_format ? KU_SHADOW_MAP : KU_TEX_FORMAT,
                           "texture colour format", t->color_format);
        return;
    }
    if (t->dimensionality == 3) {
        kelvin_unsupported(KU_TEX_3D, "3D texture", t->color_format);
        return;
    }
    if (t->border) {
        kelvin_unsupported(KU_TEX_BORDER, "texture with border", fmt);
    }
    t->valid = true;
}

/* Texels travel as SSE vectors, R G B A in lanes 0-3, with exactly the
 * values the scalar conversions give: a byte becomes (float)v * (1 / 255)
 * in its lane (lin8), and lanes are only moved, never recomputed. */

/* lin8 of bytes 0-3 of v in lanes 0-3. */
KELVIN_INLINE __m128 lin8x4(uint32_t v)
{
    __m128i b = _mm_cvtsi32_si128((int)v);
    b = _mm_unpacklo_epi8(b, _mm_setzero_si128());
    b = _mm_unpacklo_epi16(b, _mm_setzero_si128());
    return _mm_mul_ps(_mm_cvtepi32_ps(b), _mm_set1_ps(1.0f / 255.0f));
}

/* x with lane 3 replaced by a's lane 3. */
KELVIN_INLINE __m128 with_alpha(__m128 x, __m128 a)
{
    __m128 hi = _mm_unpackhi_ps(x, a);             /* x2 a2 x3 a3 */
    return _mm_shuffle_ps(x, hi, _MM_SHUFFLE(3, 0, 1, 0));
}

/* A little-endian B G R A dword (D3DCOLOR) as R G B A. */
KELVIN_INLINE __m128 bgra4(uint32_t v)
{
    __m128 x = lin8x4(v);
    return _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 0, 1, 2));
}

KELVIN_INLINE uint32_t load32(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

#ifdef _MSC_VER
#define KTEX_THREAD __declspec(thread)
#else
#define KTEX_THREAD __thread
#endif

/* One DXT block's colours by index and DXT5 alphas by index, decoded once
 * (the same colour and alpha rules as xemu's s3tc.c decode_bc1_colors / dxt3
 * / dxt5). `key` is the block's bytes, so an entry is used only for a block
 * whose bytes are the same, wherever and whenever it is read. */
typedef struct DxtBlock {
    uint64_t key[2];             /* key[1] is 0 for DXT1 */
    int kind;                    /* TK_NONE while empty */
    uint32_t indices;            /* 2 bits per texel */
    uint64_t abits;              /* DXT3: 4 bits per texel; DXT5: 3 */
    float rgba[4][4];            /* alpha as DXT1 has it */
    float alpha[8];              /* DXT5 */
} DxtBlock;

#define DXT_CACHE 64
static KTEX_THREAD DxtBlock s_dxt[DXT_CACHE];

/* Decoded bytes/255 floats belong to this thread's active rounding mode.
 * Read RC once per public DXT sampling call, never once per cached tap.
 * All nonzero normalized bytes are normal floats; FTZ/DAZ do not affect
 * this cache. Other sampler arithmetic still uses the full caller MXCSR. */
static KTEX_THREAD unsigned int s_dxt_rc = ~0u;

static void dxt_mode_begin(const KTexture *t)
{
    unsigned int rc;
    int i;
    if (t->kind != TK_DXT1 && t->kind != TK_DXT3 && t->kind != TK_DXT5) {
        return;
    }
    rc = _mm_getcsr() & 0x6000u;
    if (rc != s_dxt_rc) {
        for (i = 0; i < DXT_CACHE; i++) {
            s_dxt[i].kind = TK_NONE;
        }
        s_dxt_rc = rc;
    }
}

static void dxt_decode(DxtBlock *d)
{
    uint64_t cw = d->kind == TK_DXT1 ? d->key[0] : d->key[1];
    unsigned int c0 = (unsigned int)(cw & 0xFFFF), c1 = (unsigned int)((cw >> 16) & 0xFFFF);
    int r0 = (c0 >> 11) & 0x1F, g0 = (c0 >> 5) & 0x3F, b0 = c0 & 0x1F;
    int r1 = (c1 >> 11) & 0x1F, g1 = (c1 >> 5) & 0x3F, b1 = c1 & 0x1F;
    int rgb[4][3], i;

    r0 = (r0 << 3) | (r0 >> 2); g0 = (g0 << 2) | (g0 >> 4); b0 = (b0 << 3) | (b0 >> 2);
    r1 = (r1 << 3) | (r1 >> 2); g1 = (g1 << 2) | (g1 >> 4); b1 = (b1 << 3) | (b1 >> 2);
    rgb[0][0] = r0; rgb[0][1] = g0; rgb[0][2] = b0;
    rgb[1][0] = r1; rgb[1][1] = g1; rgb[1][2] = b1;
    if (d->kind == TK_DXT1 && c0 <= c1) {
        rgb[2][0] = (r0 + r1) / 2; rgb[2][1] = (g0 + g1) / 2; rgb[2][2] = (b0 + b1) / 2;
        rgb[3][0] = rgb[3][1] = rgb[3][2] = 0;
    } else {
        rgb[2][0] = (2 * r0 + r1) / 3; rgb[2][1] = (2 * g0 + g1) / 3; rgb[2][2] = (2 * b0 + b1) / 3;
        rgb[3][0] = (r0 + 2 * r1) / 3; rgb[3][1] = (g0 + 2 * g1) / 3; rgb[3][2] = (b0 + 2 * b1) / 3;
    }
    for (i = 0; i < 4; i++) {
        d->rgba[i][0] = (float)rgb[i][0] / 255.0f;
        d->rgba[i][1] = (float)rgb[i][1] / 255.0f;
        d->rgba[i][2] = (float)rgb[i][2] / 255.0f;
        d->rgba[i][3] = 1.0f;
    }
    if (d->kind == TK_DXT1 && c0 <= c1) {
        d->rgba[3][3] = 0.0f;
    }
    d->indices = (uint32_t)(cw >> 32);
    if (d->kind == TK_DXT3) {
        d->abits = d->key[0];
    } else if (d->kind == TK_DXT5) {
        unsigned int a0 = (unsigned int)(d->key[0] & 0xFF);
        unsigned int a1 = (unsigned int)((d->key[0] >> 8) & 0xFF);
        unsigned int ai, av;
        d->abits = d->key[0] >> 16;
        for (ai = 0; ai < 8; ai++) {
            if (ai == 0) {
                av = a0;
            } else if (ai == 1) {
                av = a1;
            } else if (a0 > a1) {
                av = ((8 - ai) * a0 + (ai - 1) * a1) / 7;
            } else if (ai == 6) {
                av = 0;
            } else if (ai == 7) {
                av = 255;
            } else {
                av = ((6 - ai) * a0 + (ai - 1) * a1) / 5;
            }
            d->alpha[ai] = (float)av / 255.0f;
        }
    }
}

/* The decoded block at blk, through this thread's cache. */
KELVIN_INLINE const DxtBlock *dxt_block(const uint8_t *blk, int kind)
{
    uint64_t k0, k1 = 0;
    uintptr_t a = (uintptr_t)blk >> (kind == TK_DXT1 ? 3 : 4);
    DxtBlock *d = &s_dxt[(a ^ (a >> 4) ^ (a >> 8)) & (DXT_CACHE - 1)];

    memcpy(&k0, blk, 8);
    if (kind != TK_DXT1) {
        memcpy(&k1, blk + 8, 8);
    }
    if (d->kind != kind || d->key[0] != k0 || d->key[1] != k1) {
        d->key[0] = k0;
        d->key[1] = k1;
        d->kind = kind;
        dxt_decode(d);
    }
    return d;
}

/* Texel (bx, by) of a decoded block. */
KELVIN_INLINE __m128 dxt_texel_of(const DxtBlock *d, int kind, unsigned int bx,
                                  unsigned int by)
{
    unsigned int n = by * 4 + bx;
    const float *c = d->rgba[(d->indices >> (2 * n)) & 3];
    if (kind == TK_DXT1) {
        return _mm_loadu_ps(c);
    } else if (kind == TK_DXT3) {
        unsigned int a4 = (unsigned int)((d->abits >> (4 * n)) & 0xF);
        return with_alpha(_mm_loadu_ps(c), _mm_set1_ps((float)(a4 * 17) / 255.0f));
    }
    return with_alpha(_mm_loadu_ps(c), _mm_set1_ps(d->alpha[(d->abits >> (3 * n)) & 7]));
}

/* One texel of a DXT block, through this thread's cache of decoded blocks. */
KELVIN_INLINE __m128 dxt_cached(const uint8_t *blk, int kind, unsigned int bx,
                                unsigned int by)
{
    return dxt_texel_of(dxt_block(blk, kind), kind, bx, by);
}

static uint8_t clip_byte(int x)
{
    return (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x));
}

/* One texel at integer coordinates, already wrapped into the level. */
KELVIN_INLINE const uint8_t *tex_ptr(const KTexture *t, uint32_t va)
{
    if (t->host) {
        return t->host + (va - t->base);
    }
    return t->mem ? t->mem + (va - t->base) : kelvin_guest(va);
}

/* A YUY2 or UYVY texel (xemu util.h convert_yuy2_to_rgb /
 * convert_uyvy_to_rgb on the 4-byte pair that holds it). */
static __m128 yuv_texel(const KTexture *t, uint32_t base, unsigned int x,
                        unsigned int y)
{
    const uint8_t *line = tex_ptr(t, base + y * t->pitch);
    int c, d, e;
    if (t->kind == TK_YUY2) {
        c = (int)line[x * 2] - 16;
        if (x % 2) {
            d = (int)line[x * 2 - 1] - 128;
            e = (int)line[x * 2 + 1] - 128;
        } else {
            d = (int)line[x * 2 + 1] - 128;
            e = (int)line[x * 2 + 3] - 128;
        }
    } else {
        c = (int)line[x * 2 + 1] - 16;
        if (x % 2) {
            d = (int)line[x * 2 - 2] - 128;
            e = (int)line[x * 2 + 0] - 128;
        } else {
            d = (int)line[x * 2 + 0] - 128;
            e = (int)line[x * 2 + 2] - 128;
        }
    }
    return _mm_setr_ps(lin8(clip_byte((298 * c + 409 * e + 128) >> 8)),
                       lin8(clip_byte((298 * c - 100 * d - 208 * e + 128) >> 8)),
                       lin8(clip_byte((298 * c + 516 * d + 128) >> 8)), 1.0f);
}

/* The texel whose bytes start at p, for every kind but DXT and YUV. */
KELVIN_INLINE __m128 texel_bytes(const KTexture *t, const uint8_t *p)
{
    uint32_t v;
    float l;

    switch (t->kind) {
    case TK_Y8:
        l = lin8(p[0]);
        return _mm_setr_ps(l, l, l, 1.0f);
    case TK_AY8:
        return _mm_set1_ps(lin8(p[0]));
    case TK_A8:
        return _mm_setr_ps(1.0f, 1.0f, 1.0f, lin8(p[0]));
    case TK_A8Y8:
        l = lin8(p[0]);
        return _mm_setr_ps(l, l, l, lin8(p[1]));
    case TK_G8B8: {
        float a = lin8(p[0]), b = lin8(p[1]);
        return _mm_setr_ps(a, b, a, b);
    }
    case TK_R8B8: {
        float a = lin8(p[1]), b = lin8(p[0]);
        return _mm_setr_ps(a, b, b, a);
    }
    case TK_1555:
    case TK_X555:
        v = p[0] | (p[1] << 8);
        return _mm_setr_ps((float)((v >> 10) & 0x1F) / 31.0f,
                           (float)((v >> 5) & 0x1F) / 31.0f,
                           (float)(v & 0x1F) / 31.0f,
                           t->kind == TK_X555 ? 1.0f : (float)((v >> 15) & 1));
    case TK_4444:
        v = p[0] | (p[1] << 8);
        return _mm_setr_ps((float)((v >> 8) & 0xF) / 15.0f,
                           (float)((v >> 4) & 0xF) / 15.0f,
                           (float)(v & 0xF) / 15.0f,
                           (float)((v >> 12) & 0xF) / 15.0f);
    case TK_565:
        v = p[0] | (p[1] << 8);
        return _mm_setr_ps((float)((v >> 11) & 0x1F) / 31.0f,
                           (float)((v >> 5) & 0x3F) / 63.0f,
                           (float)(v & 0x1F) / 31.0f, 1.0f);
    case TK_8888:
        return bgra4(load32(p));
    case TK_X888:
        return with_alpha(bgra4(load32(p)), _mm_set1_ps(1.0f));
    case TK_ABGR:                    /* bytes R G B A */
        return lin8x4(load32(p));
    case TK_BGRA: {                  /* bytes A R G B */
        __m128 x = lin8x4(load32(p));
        return _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 3, 2, 1));
    }
    case TK_RGBA: {                  /* bytes A B G R */
        __m128 x = lin8x4(load32(p));
        return _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 1, 2, 3));
    }
    case TK_I8:
        if (!t->palette || p[0] >= t->palette_len) {
            return _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
        }
        return bgra4(load32(t->palette_mem ? t->palette_mem + p[0] * 4
                                           : kelvin_guest(t->palette + p[0] * 4)));
    case TK_Y16:
        v = p[0] | (p[1] << 8);
        l = (float)v / 65535.0f;
        return _mm_setr_ps(l, l, l, 1.0f);
    default:
        return _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    }
}

KELVIN_INLINE bool is_dxt(unsigned int kind)
{
    return kind == TK_DXT1 || kind == TK_DXT3 || kind == TK_DXT5;
}

KELVIN_INLINE __m128 texel(const KTexture *t, unsigned int level, unsigned int face,
                           unsigned int x, unsigned int y)
{
    uint32_t base = t->base + face * t->face_stride + t->level_offset[level];
    const uint8_t *p;

    if (is_dxt(t->kind)) {
        unsigned int bw = (t->level_w[level] + 3) / 4;
        unsigned int block = t->kind == TK_DXT1 ? 8 : 16;
        p = tex_ptr(t, base + ((y / 4) * bw + (x / 4)) * block);
        return dxt_cached(p, t->kind, x & 3, y & 3);
    }
    if (t->kind == TK_YUY2 || t->kind == TK_UYVY) {
        return yuv_texel(t, base, x, y);
    }
    if (t->linear) {
        p = tex_ptr(t, base + y * t->pitch + x * t->bpp);
    } else {
        p = tex_ptr(t, base + swizzle_offset(t, level, x, y) * t->bpp);
    }
    return texel_bytes(t, p);
}

/* texel at (x0, y0), (x1, y0), (x0, y1) and (x1, y1) into c[0..3], sharing
 * what the four have in common: the level's address, rows, the swizzle's
 * column and row parts, and a DXT block when all four lie in one. The same
 * texels as four texel calls. False for YUV, which it leaves to those. */
KELVIN_INLINE bool texel_quad(const KTexture *t, unsigned int level,
                              unsigned int face, unsigned int x0,
                              unsigned int x1, unsigned int y0,
                              unsigned int y1, __m128 c[4])
{
    uint32_t base = t->base + face * t->face_stride + t->level_offset[level];
    const uint8_t *pb;

    if (t->kind == TK_YUY2 || t->kind == TK_UYVY) {
        return false;
    }
    /* tex_ptr is base + offset in every case, so offsets add to it. */
    pb = tex_ptr(t, base);
    if (is_dxt(t->kind)) {
        unsigned int bw = (t->level_w[level] + 3) / 4;
        unsigned int block = t->kind == TK_DXT1 ? 8 : 16;
        const uint8_t *r0 = pb + (y0 / 4) * bw * block, *r1 = pb + (y1 / 4) * bw * block;
        if (x0 / 4 == x1 / 4 && y0 / 4 == y1 / 4) {
            const DxtBlock *d = dxt_block(r0 + (x0 / 4) * block, t->kind);
            c[0] = dxt_texel_of(d, t->kind, x0 & 3, y0 & 3);
            c[1] = dxt_texel_of(d, t->kind, x1 & 3, y0 & 3);
            c[2] = dxt_texel_of(d, t->kind, x0 & 3, y1 & 3);
            c[3] = dxt_texel_of(d, t->kind, x1 & 3, y1 & 3);
        } else {
            c[0] = dxt_cached(r0 + (x0 / 4) * block, t->kind, x0 & 3, y0 & 3);
            c[1] = dxt_cached(r0 + (x1 / 4) * block, t->kind, x1 & 3, y0 & 3);
            c[2] = dxt_cached(r1 + (x0 / 4) * block, t->kind, x0 & 3, y1 & 3);
            c[3] = dxt_cached(r1 + (x1 / 4) * block, t->kind, x1 & 3, y1 & 3);
        }
        return true;
    }
    if (t->linear) {
        const uint8_t *r0 = pb + y0 * t->pitch, *r1 = pb + y1 * t->pitch;
        c[0] = texel_bytes(t, r0 + x0 * t->bpp);
        c[1] = texel_bytes(t, r0 + x1 * t->bpp);
        c[2] = texel_bytes(t, r1 + x0 * t->bpp);
        c[3] = texel_bytes(t, r1 + x1 * t->bpp);
    } else {
        uint32_t sx0 = swz_part(t, level, x0, 0), sx1 = swz_part(t, level, x1, 0);
        uint32_t sy0 = swz_part(t, level, y0, 1), sy1 = swz_part(t, level, y1, 1);
        c[0] = texel_bytes(t, pb + (sx0 | sy0) * t->bpp);
        c[1] = texel_bytes(t, pb + (sx1 | sy0) * t->bpp);
        c[2] = texel_bytes(t, pb + (sx0 | sy1) * t->bpp);
        c[3] = texel_bytes(t, pb + (sx1 | sy1) * t->bpp);
    }
    return true;
}

/* GL wrap of one integer coordinate. Returns false for a border texel. */
KELVIN_INLINE bool wrap(int c, unsigned int size, unsigned int mode, unsigned int *out)
{
    int n = (int)size;
    switch (mode) {
    case NV_PGRAPH_TEXADDRESS0_ADDRU_WRAP:
        /* For a power of two, the remainder made positive is the low bits
         * of the two's complement value. */
        if (is_pow2(size)) {
            c &= n - 1;
        } else {
            c %= n;
            if (c < 0) c += n;
        }
        break;
    case NV_PGRAPH_TEXADDRESS0_ADDRU_MIRROR: {
        int period = 2 * n;
        if (is_pow2(size)) {
            c &= period - 1;
        } else {
            c %= period;
            if (c < 0) c += period;
        }
        if (c >= n) c = period - 1 - c;
        break;
    }
    case NV_PGRAPH_TEXADDRESS0_ADDRU_BORDER:
        if (c < 0 || c >= n) return false;
        break;
    default:                     /* CLAMP_TO_EDGE, CLAMP_OGL (xemu: edge) */
        if (c < 0) c = 0;
        if (c >= n) c = n - 1;
        break;
    }
    *out = (unsigned int)c;
    return true;
}

KELVIN_INLINE __m128 border_texel(const KTexture *t)
{
    return bgra4(t->border_color);
}

KELVIN_INLINE __m128 fetch(const KTexture *t, unsigned int level, unsigned int face,
                           int x, int y)
{
    unsigned int wx, wy;
    if (!wrap(x, t->level_w[level], t->addr[0], &wx) ||
        !wrap(y, t->level_h[level], t->addr[1], &wy)) {
        return border_texel(t);
    }
    return texel(t, level, face, wx, wy);
}

/* a + (b - a) * f in every lane. */
KELVIN_INLINE __m128 lerp4(__m128 a, __m128 b, float f)
{
    return _mm_add_ps(a, _mm_mul_ps(_mm_sub_ps(b, a), _mm_set1_ps(f)));
}

/* Filter one level at normalized (s,t). */
static __m128 sample_level(const KTexture *t, unsigned int level,
                           unsigned int face, float s, float tc, bool linear)
{
    float u = s * (float)t->level_w[level];
    float v = tc * (float)t->level_h[level];

    if (!linear) {
        return fetch(t, level, face, (int)floorf(u), (int)floorf(v));
    }
    {
        float fu = u - 0.5f, fv = v - 0.5f;
        int x0 = (int)floorf(fu), y0 = (int)floorf(fv);
        float ax = fu - (float)x0, ay = fv - (float)y0;
        __m128 c00, c10, c01, c11;
        unsigned int wx0, wx1, wy0, wy1;
        /* The four texels share two columns and two rows: wrap each once.
         * A texel is the border colour when either coordinate is off the
         * edge, as fetch has it. */
        bool vx0 = wrap(x0, t->level_w[level], t->addr[0], &wx0);
        bool vx1 = wrap(x0 + 1, t->level_w[level], t->addr[0], &wx1);
        bool vy0 = wrap(y0, t->level_h[level], t->addr[1], &wy0);
        bool vy1 = wrap(y0 + 1, t->level_h[level], t->addr[1], &wy1);
        __m128 q[4];
        if (vx0 && vx1 && vy0 && vy1 && texel_quad(t, level, face, wx0, wx1, wy0, wy1, q)) {
            c00 = q[0];
            c10 = q[1];
            c01 = q[2];
            c11 = q[3];
        } else {
            c00 = vx0 && vy0 ? texel(t, level, face, wx0, wy0) : border_texel(t);
            c10 = vx1 && vy0 ? texel(t, level, face, wx1, wy0) : border_texel(t);
            c01 = vx0 && vy1 ? texel(t, level, face, wx0, wy1) : border_texel(t);
            c11 = vx1 && vy1 ? texel(t, level, face, wx1, wy1) : border_texel(t);
        }
        /* top = c00 + (c10 - c00) ax, bot likewise, top + (bot - top) ay */
        return lerp4(lerp4(c00, c10, ax), lerp4(c01, c11, ax), ay);
    }
}

/* Whether ktex_sample_2d and ktex_sample_cube give t's texels the same at
 * every LOD: a texture with one level, clamped to it, whose magnification
 * and minification filters are the same filter there (the mipmapped
 * minification modes then have only level 0 to use). */
bool ktex_lod_free(const KTexture *t)
{
    bool mag_linear, min_linear;

    if (!t->valid) {
        return true;                 /* sampled as a constant */
    }
    if (t->levels != 1 || t->min_lod != 0.0f || t->max_lod != 0.0f) {
        return false;
    }
    mag_linear = t->mag_filter != 1;
    switch (t->min_filter) {
    case NV_PGRAPH_TEXFILTER0_MIN_BOX_LOD0:
        min_linear = false;
        break;
    case NV_PGRAPH_TEXFILTER0_MIN_TENT_LOD0:
    case NV_PGRAPH_TEXFILTER0_MIN_CONVOLUTION_2D_LOD0:
        min_linear = true;
        break;
    default:
        min_linear = t->min_filter == NV_PGRAPH_TEXFILTER0_MIN_TENT_NEARESTLOD ||
                     t->min_filter == NV_PGRAPH_TEXFILTER0_MIN_TENT_TENT_LOD;
        break;
    }
    return mag_linear == min_linear;
}

float ktex_lod(const KTexture *t, float dsdx, float dtdx, float dsdy,
               float dtdy)
{
    float w = (float)t->width, h = (float)t->height;
    float ax = dsdx * w, ay = dtdx * h, bx = dsdy * w, by = dtdy * h;
    float rho2 = MAX(ax * ax + ay * ay, bx * bx + by * by);
    return ktex_lod_rho2(rho2);
}

float ktex_lod_rho2(float rho2)
{
    if (rho2 <= 0.0f) {
        return -1000.0f;
    }
    return 0.5f * log2f(rho2);
}

/* Sample at normalized (s, t) with a precomputed LOD (xemu's GL sampler:
 * the magnification filter at LOD <= 0, the minification filter otherwise,
 * mipmap levels per the min filter). */
/* What sample_2d does at an LOD: the level to sample and whether linearly,
 * and whether to blend a second level (its number, and the weight f). */
typedef struct LevelChoice {
    unsigned int a, b;
    float f;
    bool linear, blend;
} LevelChoice;

KELVIN_INLINE LevelChoice choose_levels(const KTexture *t, float lod)
{
    LevelChoice c;
    bool mag;
    unsigned int filt;

    c.b = 0;
    c.f = 0.0f;
    c.blend = false;
    lod += t->lod_bias;
    mag = lod <= 0.0f;
    filt = mag ? t->mag_filter : t->min_filter;
    if (mag) {
        /* MAG: 1 nearest, 2 linear, 4 convolution (xemu: linear). */
        c.a = (unsigned int)t->min_lod;
        c.linear = filt != 1;
        return c;
    }
    switch (filt) {
    case NV_PGRAPH_TEXFILTER0_MIN_BOX_LOD0:
    case NV_PGRAPH_TEXFILTER0_MIN_TENT_LOD0:
    case NV_PGRAPH_TEXFILTER0_MIN_CONVOLUTION_2D_LOD0:
        c.a = (unsigned int)t->min_lod;
        c.linear = filt != NV_PGRAPH_TEXFILTER0_MIN_BOX_LOD0;
        return c;
    default: {
        bool linear_mip = filt == NV_PGRAPH_TEXFILTER0_MIN_BOX_TENT_LOD ||
                          filt == NV_PGRAPH_TEXFILTER0_MIN_TENT_TENT_LOD;
        float l = lod;
        c.linear = filt == NV_PGRAPH_TEXFILTER0_MIN_TENT_NEARESTLOD ||
                   filt == NV_PGRAPH_TEXFILTER0_MIN_TENT_TENT_LOD;
        if (l < t->min_lod) l = t->min_lod;
        if (l > t->max_lod) l = t->max_lod;
        if (!linear_mip) {
            c.a = (unsigned int)floorf(l + 0.5f);
        } else {
            c.a = (unsigned int)floorf(l);
            c.b = MIN(c.a + 1, (unsigned int)t->max_lod);
            c.f = l - (float)c.a;
            c.blend = !(c.b == c.a || c.f == 0.0f);
        }
        return c;
    }
    }
}

static __m128 sample_2d(const KTexture *t, unsigned int face, float s, float tc,
                        float lod)
{
    LevelChoice c;
    __m128 a;

    if (!t->valid) {
        return _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f);
    }
    c = choose_levels(t, lod);
    a = sample_level(t, c.a, face, s, tc, c.linear);
    if (!c.blend) {
        return a;
    }
    return lerp4(a, sample_level(t, c.b, face, s, tc, c.linear), c.f);
}

void ktex_sample_2d(const KTexture *t, unsigned int face, float s, float tc,
                    float lod, float out[4])
{
    dxt_mode_begin(t);
    _mm_storeu_ps(out, sample_2d(t, face, s, tc, lod));
}

/* ---- eight samples at a time -----------------------------------------------
 *
 * ktex_sample_2d8 samples face 0 of a valid 2D texture whose texels are the
 * same at every LOD (ktex_lod_free) at eight coordinates. There sample_2d
 * always comes to sample_level at level 0 (min_lod is 0) with the
 * magnification filter's linearity, which the minification filter shares:
 * so this does sample_level's arithmetic, lane by lane in vectors -- the
 * same float operations in the same order, the same (int) conversions
 * (CVTTPS2DQ, as the scalar CVTTSS2SI), floorf as ROUNDPS -- and wrap's in
 * vector integers for power-of-two wraps and mirrors, clamps and borders,
 * calling wrap itself for the rest. Texels come from texel(), or for
 * A8R8G8B8 and X8R8G8B8 eight at a time with lin8x4's conversion. */

#if defined(__AVX2__)

/* wrap for eight coordinates: the wrapped ones, and a mask of those inside
 * (false for a border texel). */
static void wrap8(__m256i c, unsigned int size, unsigned int mode, __m256i *out,
                  __m256i *valid)
{
    __m256i n = _mm256_set1_epi32((int)size), ones = _mm256_set1_epi32(-1);
    switch (mode) {
    case NV_PGRAPH_TEXADDRESS0_ADDRU_WRAP:
        if (is_pow2(size)) {
            *out = _mm256_and_si256(c, _mm256_set1_epi32((int)size - 1));
            *valid = ones;
            return;
        }
        break;
    case NV_PGRAPH_TEXADDRESS0_ADDRU_MIRROR:
        if (is_pow2(size)) {
            __m256i period = _mm256_set1_epi32(2 * (int)size);
            __m256i m = _mm256_and_si256(c, _mm256_sub_epi32(period, _mm256_set1_epi32(1)));
            __m256i back = _mm256_sub_epi32(_mm256_sub_epi32(period, _mm256_set1_epi32(1)), m);
            /* c >= n: period - 1 - c */
            __m256i high = _mm256_cmpgt_epi32(m, _mm256_sub_epi32(n, _mm256_set1_epi32(1)));
            *out = _mm256_blendv_epi8(m, back, high);
            *valid = ones;
            return;
        }
        break;
    case NV_PGRAPH_TEXADDRESS0_ADDRU_BORDER:
        /* 0 <= c < n; c itself when inside */
        *valid = _mm256_andnot_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(), c),
                                     _mm256_cmpgt_epi32(n, c));
        *out = c;
        return;
    default:
        if (mode != NV_PGRAPH_TEXADDRESS0_ADDRU_WRAP &&
            mode != NV_PGRAPH_TEXADDRESS0_ADDRU_MIRROR) {
            /* CLAMP_TO_EDGE, CLAMP_OGL: c < 0 ? 0 : (c >= n ? n - 1 : c) */
            *out = _mm256_min_epi32(_mm256_max_epi32(c, _mm256_setzero_si256()),
                                    _mm256_sub_epi32(n, _mm256_set1_epi32(1)));
            *valid = ones;
            return;
        }
        break;
    }
    {
        /* NPOT framebuffer textures (including 3440x1440 feedback) usually
         * sample inside the image. WRAP and MIRROR are both the identity
         * there; avoid eight scalar remainder operations for that group.
         * A boundary-crossing group keeps the general wrap unchanged. */
        __m256i outside = _mm256_or_si256(
            _mm256_cmpgt_epi32(_mm256_setzero_si256(), c),
            _mm256_cmpgt_epi32(c, _mm256_sub_epi32(n, _mm256_set1_epi32(1))));
        if (!_mm256_movemask_ps(_mm256_castsi256_ps(outside))) {
            *out = c;
            *valid = ones;
            return;
        }
        /* The general cases, one lane at a time. */
        int32_t in[8], o[8], v[8];
        int l;
        _mm256_storeu_si256((__m256i *)in, c);
        for (l = 0; l < 8; l++) {
            unsigned int w;
            v[l] = wrap(in[l], size, mode, &w) ? -1 : 0;
            o[l] = (int32_t)w;
            if (!v[l]) {
                o[l] = in[l];
            }
        }
        *out = _mm256_loadu_si256((const __m256i *)o);
        *valid = _mm256_loadu_si256((const __m256i *)v);
    }
}

/* The texels at (x, y) where valid (the border colour elsewhere) of lanes in
 * mask, at level 0 of face 0: RGBA as channel vectors. */
static void fetch8(const KTexture *t, __m256i x, __m256i y, __m256i valid,
                   unsigned int mask, __m256 out[4])
{
    const uint8_t *base = t->host ? t->host : t->mem;
    /* Every lane's coordinates are inside the level, or it is a border
     * lane read at texel 0: its address is inside the texture, whatever the
     * mask. A8R8G8B8 and X8R8G8B8 texels are gathered; I8 indices are read
     * a lane at a time, then their palette entries gathered. */
    bool dword = (t->kind == TK_8888 || t->kind == TK_X888) && base;
    bool i8 = t->kind == TK_I8 && base && (t->palette_mem || !t->palette);
    bool gather = dword || i8;
    unsigned int w = t->level_w[0], h = t->level_h[0];

    if (gather && !t->linear &&
        (!is_pow2(w) || !is_pow2(h) || w > 0x10000 || h > 0x10000)) {
        gather = false;
    }
    if (gather) {
        __m256i off, v, border = _mm256_set1_epi32((int)t->border_color);
        __m256 k = _mm256_set1_ps(1.0f / 255.0f);
        __m256i byte = _mm256_set1_epi32(0xFF);
        __m256i x0 = _mm256_and_si256(x, valid), y0 = _mm256_and_si256(y, valid);
        __m256i unpaletted = _mm256_setzero_si256();
        int shift = dword ? 2 : 0;           /* log2 bpp */
        if (t->linear) {
            /* base + y * pitch + x * bpp (texel's address, level 0) */
            off = _mm256_add_epi32(_mm256_mullo_epi32(y0, _mm256_set1_epi32((int)t->pitch)),
                                   _mm256_slli_epi32(x0, shift));
        } else {
            /* swz_part(x, 0) | swz_part(y, 1), times 4 */
            unsigned int m = log2_pow2(MIN(w, h));
            __m256i low = _mm256_set1_epi32((int)((1u << m) - 1));
            __m256i sx = _mm256_and_si256(x0, low), sy = _mm256_and_si256(y0, low);
            __m256i hi;
#define SPREAD(v) \
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 8)), _mm256_set1_epi32(0x00FF00FF)); \
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 4)), _mm256_set1_epi32(0x0F0F0F0F)); \
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 2)), _mm256_set1_epi32(0x33333333)); \
    v = _mm256_and_si256(_mm256_or_si256(v, _mm256_slli_epi32(v, 1)), _mm256_set1_epi32(0x55555555))
            SPREAD(sx);
            SPREAD(sy);
#undef SPREAD
            hi = _mm256_sll_epi32(_mm256_or_si256(_mm256_srl_epi32(x0, _mm_cvtsi32_si128((int)m)),
                                                  _mm256_srl_epi32(y0, _mm_cvtsi32_si128((int)m))),
                                  _mm_cvtsi32_si128((int)(2 * m)));
            off = _mm256_slli_epi32(_mm256_or_si256(_mm256_or_si256(sx, _mm256_slli_epi32(sy, 1)),
                                                    hi), shift);
        }
        if (dword) {
            v = _mm256_i32gather_epi32((const int *)base, off, 1);
        } else {
            /* texel_bytes' I8: (0, 0, 0, 1) without a palette or past its
             * end, else the palette entry as a D3DCOLOR. */
            int32_t o[8], ix[8];
            __m256i idx;
            int l;
            _mm256_storeu_si256((__m256i *)o, off);
            for (l = 0; l < 8; l++) {
                ix[l] = base[o[l]];
            }
            idx = _mm256_loadu_si256((const __m256i *)ix);
            if (t->palette) {
                __m256i in = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)t->palette_len), idx);
                v = _mm256_i32gather_epi32((const int *)t->palette_mem,
                                           _mm256_slli_epi32(_mm256_and_si256(idx, in), 2), 1);
                unpaletted = _mm256_andnot_si256(in, valid);
            } else {
                v = _mm256_setzero_si256();
                unpaletted = valid;
            }
        }
        /* The border colour where outside: a D3DCOLOR too. */
        v = _mm256_blendv_epi8(border, v, valid);
        /* bgra4: bytes B G R A, lin8 each */
        out[2] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, byte)), k);
        out[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v, 8), byte)), k);
        out[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(v, 16), byte)), k);
        out[3] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(v, 24)), k);
        if (t->kind == TK_X888) {
            /* with_alpha(..., 1.0) -- but only for fetched texels; the
             * border colour keeps its alpha. */
            out[3] = _mm256_blendv_ps(out[3], _mm256_set1_ps(1.0f),
                                      _mm256_castsi256_ps(valid));
        }
        if (i8) {
            __m256 u = _mm256_castsi256_ps(unpaletted);
            out[0] = _mm256_blendv_ps(out[0], _mm256_setzero_ps(), u);
            out[1] = _mm256_blendv_ps(out[1], _mm256_setzero_ps(), u);
            out[2] = _mm256_blendv_ps(out[2], _mm256_setzero_ps(), u);
            out[3] = _mm256_blendv_ps(out[3], _mm256_set1_ps(1.0f), u);
        }
        return;
    }
    {
        int32_t xs[8], ys[8], vs[8];
        float c[4][8];
        int l, k;
        _mm256_storeu_si256((__m256i *)xs, x);
        _mm256_storeu_si256((__m256i *)ys, y);
        _mm256_storeu_si256((__m256i *)vs, valid);
        for (l = 0; l < 8; l++) {
            float v4[4] = { 0, 0, 0, 0 };
            if (mask & (1u << l)) {
                _mm_storeu_ps(v4, vs[l] ? texel(t, 0, 0, (unsigned int)xs[l], (unsigned int)ys[l])
                                        : border_texel(t));
            }
            for (k = 0; k < 4; k++) {
                c[k][l] = v4[k];
            }
        }
        for (k = 0; k < 4; k++) {
            out[k] = _mm256_loadu_ps(c[k]);
        }
    }
}

/* A bilinear DXT1 group at one mip level. Most neighbouring fragments pick
 * the same level: wrap their coordinates together, then transpose the cached
 * RGBA texels four lanes at a time and filter in registers. The general path
 * below handles groups whose levels or filters differ. */
static void sample_dxt1_level8(const KTexture *t, unsigned int level,
                              const float *s, const float *tc,
                              unsigned int mask, __m256 out[4])
{
    unsigned int w = t->level_w[level], h = t->level_h[level];
    unsigned int bw = (w + 3) / 4;
    const uint8_t *pb = tex_ptr(t, t->base + t->level_offset[level]);
    __m256 fu = _mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(s),
                                 _mm256_set1_ps((float)w)), _mm256_set1_ps(0.5f));
    __m256 fv = _mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(tc),
                                 _mm256_set1_ps((float)h)), _mm256_set1_ps(0.5f));
    __m256i x0 = _mm256_cvttps_epi32(_mm256_floor_ps(fu));
    __m256i y0 = _mm256_cvttps_epi32(_mm256_floor_ps(fv));
    __m256 ax = _mm256_sub_ps(fu, _mm256_cvtepi32_ps(x0));
    __m256 ay = _mm256_sub_ps(fv, _mm256_cvtepi32_ps(y0));
    __m256i wx0, wx1, wy0, wy1, vx0, vx1, vy0, vy1;
    int32_t xs[2][8], ys[2][8], valid[4][8];
    __m128 border = border_texel(t);
    int group, l, k;

    wrap8(x0, w, t->addr[0], &wx0, &vx0);
    wrap8(_mm256_add_epi32(x0, _mm256_set1_epi32(1)), w, t->addr[0], &wx1, &vx1);
    wrap8(y0, h, t->addr[1], &wy0, &vy0);
    wrap8(_mm256_add_epi32(y0, _mm256_set1_epi32(1)), h, t->addr[1], &wy1, &vy1);
    _mm256_storeu_si256((__m256i *)xs[0], wx0);
    _mm256_storeu_si256((__m256i *)xs[1], wx1);
    _mm256_storeu_si256((__m256i *)ys[0], wy0);
    _mm256_storeu_si256((__m256i *)ys[1], wy1);
    _mm256_storeu_si256((__m256i *)valid[0], _mm256_and_si256(vx0, vy0));
    _mm256_storeu_si256((__m256i *)valid[1], _mm256_and_si256(vx1, vy0));
    _mm256_storeu_si256((__m256i *)valid[2], _mm256_and_si256(vx0, vy1));
    _mm256_storeu_si256((__m256i *)valid[3], _mm256_and_si256(vx1, vy1));
    for (group = 0; group < 2; group++) {
        __m128 q[4][4];
        __m128 fx = group ? _mm256_extractf128_ps(ax, 1) : _mm256_castps256_ps128(ax);
        __m128 fy = group ? _mm256_extractf128_ps(ay, 1) : _mm256_castps256_ps128(ay);
        for (l = 0; l < 4; l++) {
            unsigned int lane = (unsigned int)(group * 4 + l);
            unsigned int x = (unsigned int)xs[0][lane], nx = (unsigned int)xs[1][lane];
            unsigned int y = (unsigned int)ys[0][lane], ny = (unsigned int)ys[1][lane];
            if (!(mask & (1u << lane))) {
                q[0][l] = q[1][l] = q[2][l] = q[3][l] = _mm_setzero_ps();
            } else if (valid[0][lane] && valid[1][lane] && valid[2][lane] &&
                       valid[3][lane] && x / 4 == nx / 4 && y / 4 == ny / 4) {
                const DxtBlock *d = dxt_block(pb + ((y / 4) * bw + x / 4) * 8, TK_DXT1);
                q[0][l] = dxt_texel_of(d, TK_DXT1, x & 3, y & 3);
                q[1][l] = dxt_texel_of(d, TK_DXT1, nx & 3, y & 3);
                q[2][l] = dxt_texel_of(d, TK_DXT1, x & 3, ny & 3);
                q[3][l] = dxt_texel_of(d, TK_DXT1, nx & 3, ny & 3);
            } else {
                q[0][l] = valid[0][lane] ? dxt_cached(pb + ((y / 4) * bw + x / 4) * 8,
                                                     TK_DXT1, x & 3, y & 3) : border;
                q[1][l] = valid[1][lane] ? dxt_cached(pb + ((y / 4) * bw + nx / 4) * 8,
                                                     TK_DXT1, nx & 3, y & 3) : border;
                q[2][l] = valid[2][lane] ? dxt_cached(pb + ((ny / 4) * bw + x / 4) * 8,
                                                     TK_DXT1, x & 3, ny & 3) : border;
                q[3][l] = valid[3][lane] ? dxt_cached(pb + ((ny / 4) * bw + nx / 4) * 8,
                                                     TK_DXT1, nx & 3, ny & 3) : border;
            }
        }
        for (k = 0; k < 4; k++) {
            _MM_TRANSPOSE4_PS(q[k][0], q[k][1], q[k][2], q[k][3]);
        }
        for (k = 0; k < 4; k++) {
            __m128 top = _mm_add_ps(q[0][k], _mm_mul_ps(_mm_sub_ps(q[1][k], q[0][k]), fx));
            __m128 bot = _mm_add_ps(q[2][k], _mm_mul_ps(_mm_sub_ps(q[3][k], q[2][k]), fx));
            __m128 value = _mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bot, top), fy));
            if (group) out[k] = _mm256_insertf128_ps(out[k], value, 1);
            else out[k] = _mm256_castps128_ps256(value);
        }
    }
}

/* sample_level for eight lanes (those in mask), each at its own level and
 * filter: the coordinates and filtering in vectors, as sample_level does
 * them per lane, and the wraps and texels per lane as it does. */
static void sample_levels8(const KTexture *t, const unsigned int level[8],
                           const bool linear[8], const float *s, const float *tc,
                           unsigned int mask, __m256 out[4])
{
    float wf[8], hf[8], near_c[4][8], quad[4][4][8];
    bool any_linear = false, any_near = false;
    __m256 u, v, ax, ay;
    int32_t xi[8], yi[8];
    int l, k, j;

    if (t->kind == TK_DXT1 && (mask & 0xFF)) {
        unsigned int first = 0;
        bool same = true;
        while (!(mask & (1u << first))) first++;
        for (l = 0; l < 8; l++) {
            if ((mask & (1u << l)) && (!linear[l] || level[l] != level[first])) {
                same = false;
                break;
            }
        }
        if (same) {
            sample_dxt1_level8(t, level[first], s, tc, mask, out);
            return;
        }
    }

    for (l = 0; l < 8; l++) {
        unsigned int lv = (mask & (1u << l)) ? level[l] : 0;
        wf[l] = (float)t->level_w[lv];
        hf[l] = (float)t->level_h[lv];
        if (mask & (1u << l)) {
            if (linear[l]) any_linear = true; else any_near = true;
        }
    }
    u = _mm256_mul_ps(_mm256_loadu_ps(s), _mm256_loadu_ps(wf));
    v = _mm256_mul_ps(_mm256_loadu_ps(tc), _mm256_loadu_ps(hf));
    memset(near_c, 0, sizeof(near_c));
    memset(quad, 0, sizeof(quad));
    if (any_near) {
        /* fetch(t, level, 0, (int)floorf(u), (int)floorf(v)) */
        _mm256_storeu_si256((__m256i *)xi, _mm256_cvttps_epi32(_mm256_floor_ps(u)));
        _mm256_storeu_si256((__m256i *)yi, _mm256_cvttps_epi32(_mm256_floor_ps(v)));
        for (l = 0; l < 8; l++) {
            float c4[4];
            if (!(mask & (1u << l)) || linear[l]) continue;
            _mm_storeu_ps(c4, fetch(t, level[l], 0, xi[l], yi[l]));
            for (k = 0; k < 4; k++) near_c[k][l] = c4[k];
        }
    }
    {
        __m256 fu = _mm256_sub_ps(u, _mm256_set1_ps(0.5f));
        __m256 fv = _mm256_sub_ps(v, _mm256_set1_ps(0.5f));
        __m256i x0 = _mm256_cvttps_epi32(_mm256_floor_ps(fu));
        __m256i y0 = _mm256_cvttps_epi32(_mm256_floor_ps(fv));
        ax = _mm256_sub_ps(fu, _mm256_cvtepi32_ps(x0));
        ay = _mm256_sub_ps(fv, _mm256_cvtepi32_ps(y0));
        _mm256_storeu_si256((__m256i *)xi, x0);
        _mm256_storeu_si256((__m256i *)yi, y0);
    }
    if (any_linear) {
        for (l = 0; l < 8; l++) {
            unsigned int lv = level[l], wx0, wx1, wy0, wy1;
            __m128 q[4];
            bool vx0, vx1, vy0, vy1;
            if (!(mask & (1u << l)) || !linear[l]) continue;
            /* sample_level's four texels, as it finds them */
            vx0 = wrap(xi[l], t->level_w[lv], t->addr[0], &wx0);
            vx1 = wrap(xi[l] + 1, t->level_w[lv], t->addr[0], &wx1);
            vy0 = wrap(yi[l], t->level_h[lv], t->addr[1], &wy0);
            vy1 = wrap(yi[l] + 1, t->level_h[lv], t->addr[1], &wy1);
            if (!(vx0 && vx1 && vy0 && vy1 && texel_quad(t, lv, 0, wx0, wx1, wy0, wy1, q))) {
                q[0] = vx0 && vy0 ? texel(t, lv, 0, wx0, wy0) : border_texel(t);
                q[1] = vx1 && vy0 ? texel(t, lv, 0, wx1, wy0) : border_texel(t);
                q[2] = vx0 && vy1 ? texel(t, lv, 0, wx0, wy1) : border_texel(t);
                q[3] = vx1 && vy1 ? texel(t, lv, 0, wx1, wy1) : border_texel(t);
            }
            for (j = 0; j < 4; j++) {
                float c4[4];
                _mm_storeu_ps(c4, q[j]);
                for (k = 0; k < 4; k++) quad[j][k][l] = c4[k];
            }
        }
    }
    {
        float lin_mask[8];
        for (l = 0; l < 8; l++) {
            uint32_t m = linear[l] ? 0xFFFFFFFFu : 0;
            memcpy(&lin_mask[l], &m, 4);
        }
        for (k = 0; k < 4; k++) {
            /* lerp4(lerp4(c00, c10, ax), lerp4(c01, c11, ax), ay) */
            __m256 c00 = _mm256_loadu_ps(quad[0][k]), c10 = _mm256_loadu_ps(quad[1][k]);
            __m256 c01 = _mm256_loadu_ps(quad[2][k]), c11 = _mm256_loadu_ps(quad[3][k]);
            __m256 top = _mm256_add_ps(c00, _mm256_mul_ps(_mm256_sub_ps(c10, c00), ax));
            __m256 bot = _mm256_add_ps(c01, _mm256_mul_ps(_mm256_sub_ps(c11, c01), ax));
            __m256 lin = _mm256_add_ps(top, _mm256_mul_ps(_mm256_sub_ps(bot, top), ay));
            out[k] = _mm256_blendv_ps(_mm256_loadu_ps(near_c[k]), lin,
                                      _mm256_loadu_ps(lin_mask));
        }
    }
}

/* Vector level choice for the common trilinear/min and linear/mag filters.
 * Ordered finite-or-infinite LODs clamp exactly as choose_levels does; NaN
 * keeps the scalar path. The valid texture's clamps bound conversion. */
static bool choose_linear_mip8(const KTexture *t, const float *lod,
                               unsigned int mask, unsigned int la[8],
                               unsigned int lb[8], bool linear[8], float f[8],
                               unsigned int *blend)
{
    __m256 l, mag, lo, hi, frac;
    __m256i ai, bi;
    int lane;
    if (t->min_filter != NV_PGRAPH_TEXFILTER0_MIN_TENT_TENT_LOD ||
        t->mag_filter == 1) return false;
    l = _mm256_add_ps(_mm256_loadu_ps(lod), _mm256_set1_ps(t->lod_bias));
    if (_mm256_movemask_ps(_mm256_cmp_ps(l, l, _CMP_UNORD_Q))) return false;
    lo = _mm256_set1_ps(t->min_lod);
    hi = _mm256_set1_ps(t->max_lod);
    mag = _mm256_cmp_ps(l, _mm256_setzero_ps(), _CMP_LE_OQ);
    l = _mm256_min_ps(_mm256_max_ps(l, lo), hi);
    ai = _mm256_cvttps_epi32(_mm256_floor_ps(l));
    bi = _mm256_min_epi32(_mm256_add_epi32(ai, _mm256_set1_epi32(1)),
                           _mm256_set1_epi32((int)t->max_lod));
    /* Magnification selects min_lod directly, not its clamped floor's next
     * mip, and always has zero fraction and no level blend. */
    ai = _mm256_blendv_epi8(ai, _mm256_set1_epi32((int)t->min_lod),
                             _mm256_castps_si256(mag));
    frac = _mm256_sub_ps(l, _mm256_cvtepi32_ps(ai));
    frac = _mm256_andnot_ps(mag, frac);
    bi = _mm256_andnot_si256(_mm256_castps_si256(mag), bi);
    *blend = (unsigned int)_mm256_movemask_ps(_mm256_andnot_ps(mag,
        _mm256_and_ps(_mm256_cmp_ps(frac, _mm256_setzero_ps(), _CMP_NEQ_OQ),
            _mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpeq_epi32(ai, bi),
                                                _mm256_set1_epi32(-1)))))) & mask;
    _mm256_storeu_si256((__m256i *)la, ai);
    _mm256_storeu_si256((__m256i *)lb, bi);
    _mm256_storeu_ps(f, frac);
    for (lane = 0; lane < 8; lane++) linear[lane] = true;
    return true;
}

void ktex_sample_2d8_lod(const KTexture *t, const float *s, const float *tc,
                         const float *lod, unsigned int mask, float out[4][8])
{
    dxt_mode_begin(t);
    unsigned int la[8], lb[8], blend = 0;
    bool linear[8];
    float f[8];
    __m256 a[4];
    int l, k;

    if (!choose_linear_mip8(t, lod, mask, la, lb, linear, f, &blend)) {
        for (l = 0; l < 8; l++) {
            LevelChoice c = choose_levels(t, lod[l]);
            la[l] = c.a;
            lb[l] = c.b;
            f[l] = c.f;
            linear[l] = c.linear;
            if (c.blend && (mask & (1u << l))) {
                blend |= 1u << l;
            }
        }
    }
    sample_levels8(t, la, linear, s, tc, mask, a);
    if (blend) {
        /* lerp4(a, b, f) where the choice blends a second level */
        __m256 b[4], fv = _mm256_loadu_ps(f);
        float bm[8];
        sample_levels8(t, lb, linear, s, tc, blend, b);
        for (l = 0; l < 8; l++) {
            uint32_t m = (blend & (1u << l)) ? 0xFFFFFFFFu : 0;
            memcpy(&bm[l], &m, 4);
        }
        for (k = 0; k < 4; k++) {
            __m256 mixed = _mm256_add_ps(a[k], _mm256_mul_ps(_mm256_sub_ps(b[k], a[k]), fv));
            a[k] = _mm256_blendv_ps(a[k], mixed, _mm256_loadu_ps(bm));
        }
    }
    for (k = 0; k < 4; k++) {
        _mm256_storeu_ps(out[k], a[k]);
    }
}

void ktex_sample_2d8(const KTexture *t, const float *s, const float *tc,
                     unsigned int mask, float out[4][8])
{
    dxt_mode_begin(t);
    __m256 u = _mm256_mul_ps(_mm256_loadu_ps(s), _mm256_set1_ps((float)t->level_w[0]));
    __m256 v = _mm256_mul_ps(_mm256_loadu_ps(tc), _mm256_set1_ps((float)t->level_h[0]));
    __m256 r[4];
    int k;

    if (t->mag_filter == 1) {
        /* fetch(t, 0, 0, (int)floorf(u), (int)floorf(v)) */
        __m256i x = _mm256_cvttps_epi32(_mm256_floor_ps(u));
        __m256i y = _mm256_cvttps_epi32(_mm256_floor_ps(v));
        __m256i wx, wy, vx, vy;
        wrap8(x, t->level_w[0], t->addr[0], &wx, &vx);
        wrap8(y, t->level_h[0], t->addr[1], &wy, &vy);
        fetch8(t, wx, wy, _mm256_and_si256(vx, vy), mask, r);
    } else {
        __m256 fu = _mm256_sub_ps(u, _mm256_set1_ps(0.5f));
        __m256 fv = _mm256_sub_ps(v, _mm256_set1_ps(0.5f));
        __m256i x0 = _mm256_cvttps_epi32(_mm256_floor_ps(fu));
        __m256i y0 = _mm256_cvttps_epi32(_mm256_floor_ps(fv));
        __m256 ax = _mm256_sub_ps(fu, _mm256_cvtepi32_ps(x0));
        __m256 ay = _mm256_sub_ps(fv, _mm256_cvtepi32_ps(y0));
        __m256i one = _mm256_set1_epi32(1);
        __m256i wx0, wx1, wy0, wy1, vx0, vx1, vy0, vy1;
        __m256 c00[4], c10[4], c01[4], c11[4];
        wrap8(x0, t->level_w[0], t->addr[0], &wx0, &vx0);
        wrap8(_mm256_add_epi32(x0, one), t->level_w[0], t->addr[0], &wx1, &vx1);
        wrap8(y0, t->level_h[0], t->addr[1], &wy0, &vy0);
        wrap8(_mm256_add_epi32(y0, one), t->level_h[0], t->addr[1], &wy1, &vy1);
        fetch8(t, wx0, wy0, _mm256_and_si256(vx0, vy0), mask, c00);
        fetch8(t, wx1, wy0, _mm256_and_si256(vx1, vy0), mask, c10);
        fetch8(t, wx0, wy1, _mm256_and_si256(vx0, vy1), mask, c01);
        fetch8(t, wx1, wy1, _mm256_and_si256(vx1, vy1), mask, c11);
        /* lerp4(lerp4(c00, c10, ax), lerp4(c01, c11, ax), ay) */
        for (k = 0; k < 4; k++) {
            __m256 top = _mm256_add_ps(c00[k], _mm256_mul_ps(_mm256_sub_ps(c10[k], c00[k]), ax));
            __m256 bot = _mm256_add_ps(c01[k], _mm256_mul_ps(_mm256_sub_ps(c11[k], c01[k]), ax));
            r[k] = _mm256_add_ps(top, _mm256_mul_ps(_mm256_sub_ps(bot, top), ay));
        }
    }
    for (k = 0; k < 4; k++) {
        _mm256_storeu_ps(out[k], r[k]);
    }
}

#endif

/* Cube sampling: pick the face by the major axis (GL rules). */
void ktex_sample_cube(const KTexture *t, float x, float y, float z, float lod,
                      float out[4])
{
    dxt_mode_begin(t);
    float ax = fabsf(x), ay = fabsf(y), az = fabsf(z), sc, tc, ma;
    unsigned int face;

    if (ax >= ay && ax >= az) {
        face = x > 0 ? 0 : 1;
        sc = x > 0 ? -z : z; tc = -y; ma = ax;
    } else if (ay >= ax && ay >= az) {
        face = y > 0 ? 2 : 3;
        sc = x; tc = y > 0 ? z : -z; ma = ay;
    } else {
        face = z > 0 ? 4 : 5;
        sc = z > 0 ? x : -x; tc = -y; ma = az;
    }
    if (ma == 0.0f) {
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    }
    _mm_storeu_ps(out, sample_2d(t, face, 0.5f * (sc / ma + 1.0f),
                                 0.5f * (tc / ma + 1.0f), lod));
}
