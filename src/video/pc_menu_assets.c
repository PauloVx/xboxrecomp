/* Assets for the native menus, decoded from the game's own files in memory (see pc_menu.h):
 *   - "4.1v" bitmap fonts (RenderWare-era EA fonts, BLACK's language/fonts/*.bin): texture offset at +4,
 *     atlas texels per text-size unit at +8/+12, 128 glyph chains from +0x20, 0x28-byte glyphs from
 *     +0x220 {u, v, w, h, x, y, advance (fractions of the atlas), code at +0x1C}, then the texture;
 *   - APT bundles (export/FrontEnd/*.bin): a table of {name, block}; a block holds the texture pages at
 *     +8 (a 0x40-byte header, then 0x80-byte texture headers each followed by the pixels) and the image
 *     map at +0x14 {magic, count, entries -> {id, 0, rect offset, 0}, rect = {u, v, w, h (fractions)}}.
 * Textures are Xbox DXT2/3 (0x0E) or DXT4/5 (0x0F), not swizzled.
 */
#include "pc_menu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t *d, size_t size, size_t at)
{
    uint32_t v;
    if (at > size || size - at < 4) return 0;
    memcpy(&v, d + at, 4);
    return v;
}

static float rdf(const uint8_t *d, size_t size, size_t at)
{
    float v;
    if (at > size || size - at < 4) return 0.0f;
    memcpy(&v, d + at, 4);
    return v;
}

static void rgb565(unsigned c, unsigned rgb[3])
{
    const unsigned r = (c >> 11) & 31u, g = (c >> 5) & 63u, b = c & 31u;
    rgb[0] = (r << 3) | (r >> 2); rgb[1] = (g << 2) | (g >> 4); rgb[2] = (b << 3) | (b >> 2);
}

int pc_menu_decode_dxt(const uint8_t *src, size_t size, unsigned w, unsigned h, unsigned format, uint32_t *out)
{
    unsigned bx, by, i;
    const uint8_t *b = src;
    if ((format != 0x0Eu && format != 0x0Fu) || !w || !h || (w & 3u) || (h & 3u) || (size_t)w * h > size) return 0;
    for (by = 0; by < h; by += 4) for (bx = 0; bx < w; bx += 4, b += 16) {
        unsigned alpha[16], pal[4][3];
        const unsigned c0 = b[8] | (b[9] << 8), c1 = b[10] | (b[11] << 8);
        const uint32_t idx = (uint32_t)b[12] | ((uint32_t)b[13] << 8) | ((uint32_t)b[14] << 16) | ((uint32_t)b[15] << 24);
        if (format == 0x0Eu) {
            for (i = 0; i < 16; i++) alpha[i] = ((b[i / 2] >> ((i & 1) * 4)) & 15u) * 17u;
        } else {
            unsigned ap[8];
            const unsigned a0 = b[0], a1 = b[1];
            uint64_t bits = 0;
            for (i = 0; i < 6; i++) bits |= (uint64_t)b[2 + i] << (8 * i);
            ap[0] = a0; ap[1] = a1;
            if (a0 > a1) for (i = 0; i < 6; i++) ap[2 + i] = ((6 - i) * a0 + (1 + i) * a1 + 3) / 7;
            else { for (i = 0; i < 4; i++) ap[2 + i] = ((4 - i) * a0 + (1 + i) * a1 + 2) / 5; ap[6] = 0; ap[7] = 255; }
            for (i = 0; i < 16; i++) alpha[i] = ap[(bits >> (3 * i)) & 7u];
        }
        rgb565(c0, pal[0]); rgb565(c1, pal[1]);
        for (i = 0; i < 3; i++) { pal[2][i] = (2 * pal[0][i] + pal[1][i] + 1) / 3; pal[3][i] = (pal[0][i] + 2 * pal[1][i] + 1) / 3; }
        for (i = 0; i < 16; i++) {
            const unsigned k = (idx >> (2 * i)) & 3u, x = bx + (i & 3u), y = by + (i >> 2);
            out[(size_t)y * w + x] = ((uint32_t)alpha[i] << 24) | (pal[k][0] << 16) | (pal[k][1] << 8) | pal[k][2];
        }
    }
    return 1;
}

int pc_menu_load_font_file(int slot, const uint8_t *d, size_t size)
{
    uint32_t tex, format, w, h, n = 0;
    size_t at;
    uint32_t *pixels = NULL;
    PcMenuGlyph *glyphs = NULL;
    PcMenuFont font;
    int ok = 0;
    if (!d || size < 0x300 || memcmp(d, "4.1v", 4)) return 0;
    tex = rd32(d, size, 4);
    if (tex < 0x220u || tex > size || size - tex < 0x80u) return 0;
    format = rd32(d, size, tex + 0x34u); w = rd32(d, size, tex + 0x38u); h = rd32(d, size, tex + 0x3Cu);
    if (!w || !h || w > 2048u || h > 2048u) return 0;
    pixels = (uint32_t *)malloc((size_t)w * h * 4u);
    glyphs = (PcMenuGlyph *)calloc((tex - 0x220u) / 0x28u + 1u, sizeof(PcMenuGlyph));
    if (!pixels || !glyphs || !pc_menu_decode_dxt(d + tex + 0x80u, size - tex - 0x80u, w, h, format, pixels)) goto done;
    for (at = 0x220u; at + 0x28u <= tex; at += 0x28u) {
        PcMenuGlyph *g = &glyphs[n];
        uint16_t code;
        memcpy(&code, d + at + 0x1Cu, 2);
        if (!code) continue;
        g->code = code;
        g->u = rdf(d, size, at) * (float)w; g->v = rdf(d, size, at + 4) * (float)h;
        g->w = rdf(d, size, at + 8) * (float)w; g->h = rdf(d, size, at + 12) * (float)h;
        g->ox = rdf(d, size, at + 16) * (float)w; g->oy = rdf(d, size, at + 20) * (float)h;
        g->advance = rdf(d, size, at + 24) * (float)w;
        if (g->u < 0 || g->v < 0) g->w = g->h = 0;            /* the space has no cell */
        n++;
    }
    font.pixels = pixels; font.width = w; font.height = h; font.glyphs = glyphs; font.count = n;
    /* +12 is the atlas height over the size at which one texel is one unit: 256 / 15 for Small, 256 / 30 for Big */
    font.nominal = rdf(d, size, 12) > 0 ? (float)h / rdf(d, size, 12) : 0.0f;
    ok = font.nominal > 1.0f && font.nominal < 200.0f && pc_menu_set_font(slot, &font);
done:
    free(pixels); free(glyphs);
    return ok;
}

int pc_menu_load_bundle_images(const uint8_t *d, size_t size, const char *movie, const uint32_t *ids, const int *slots, int n)
{
    uint32_t count, table, i, block = 0, texblock, map, format, w, h;
    uint32_t *page = NULL;
    int ok = 0;
    if (!d || size < 16) return 0;
    count = rd32(d, size, 8); table = rd32(d, size, 12);
    for (i = 0; i < count && i < 64u; i++) {
        const uint32_t e = rd32(d, size, table + i * 4u), name = rd32(d, size, e);
        const size_t len = strlen(movie) + 1;
        if (name < size && size - name >= len && !memcmp(d + name, movie, len)) { block = rd32(d, size, e + 4u); break; }
    }
    if (!block || block >= size) return 0;
    texblock = block + rd32(d, size, block + 8u);
    map = block + rd32(d, size, block + 0x14u);
    if (texblock == block || map == block || texblock > size || size - texblock < 0xC0u) return 0;
    format = rd32(d, size, texblock + 0x40u + 0x34u); w = rd32(d, size, texblock + 0x40u + 0x38u); h = rd32(d, size, texblock + 0x40u + 0x3Cu);
    if (!w || !h || w > 2048u || h > 2048u) return 0;
    page = (uint32_t *)malloc((size_t)w * h * 4u);
    if (!page || !pc_menu_decode_dxt(d + texblock + 0xC0u, size - texblock - 0xC0u, w, h, format, page)) { free(page); return 0; }
    {
        const uint32_t m = rd32(d, size, map + 8u), entries = map + rd32(d, size, map + 12u);
        for (i = 0; i < m && i < 256u; i++) {
            const uint32_t id = rd32(d, size, entries + i * 16u), r = map + rd32(d, size, entries + i * 16u + 8u);
            int k, slot = -1;
            unsigned x0, y0, iw, ih, y;
            uint32_t *img;
            for (k = 0; k < n; k++) if (ids[k] == id) slot = slots[k];
            if (slot < 0) continue;
            x0 = (unsigned)(rdf(d, size, r) * (float)w + 0.5f); y0 = (unsigned)(rdf(d, size, r + 4) * (float)h + 0.5f);
            iw = (unsigned)(rdf(d, size, r + 8) * (float)w + 0.5f); ih = (unsigned)(rdf(d, size, r + 12) * (float)h + 0.5f);
            if (!iw || !ih || x0 + iw > w || y0 + ih > h) continue;
            img = (uint32_t *)malloc((size_t)iw * ih * 4u);
            if (!img) continue;
            for (y = 0; y < ih; y++) memcpy(img + (size_t)y * iw, page + (size_t)(y0 + y) * w + x0, (size_t)iw * 4u);
            if (pc_menu_set_image(slot, img, iw, ih)) ok++;
            free(img);
        }
    }
    free(page);
    return ok;
}
