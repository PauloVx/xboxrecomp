/* Replacement textures from a texture pack made for another version of the game (kelvin_texpack.h). */

#define COBJMACROS
#include "kelvin_texpack.h"

#include <windows.h>
#include <wincodec.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KTP_SIDE 32
#define KTP_CELLS (KTP_SIDE * KTP_SIDE)
#define KTP_INDEX_VERSION 1u

/* A picture's fingerprint: its 32x32 reduction (colour premultiplied by alpha) as zero-mean, unit-length grey, its
 * colour and alpha on a 4x4 grid, whether any of it is see-through, and its shape. */
typedef struct KtpPrint {
    float grey[KTP_CELLS];
    float colour[48];
    float alpha[16];
    float aspect;
    uint8_t clear, ps2_alpha;
} KtpPrint;

typedef struct KtpFile {
    wchar_t *path;
    uint64_t size, written;
    KtpPrint print;
    bool ok;
    ID3D11Texture2D *tex;
    ID3D11ShaderResourceView *srv;
    size_t bytes;
} KtpFile;

static struct {
    int state;                         /* 0 not looked at, 1 ready, -1 off */
    wchar_t root[MAX_PATH];
    KtpFile *files;
    unsigned count, cap;
    IWICImagingFactory *wic;
} K;

/* ---- pictures ----------------------------------------------------------------------------------------------- */

static uint32_t ktp_565(uint16_t c)
{
    const uint32_t r = (c >> 11) & 31u, g = (c >> 5) & 63u, b = c & 31u;
    return ((r << 3) | (r >> 2)) | (((g << 2) | (g >> 4)) << 8) | (((b << 3) | (b >> 2)) << 16);
}

static uint32_t ktp_mix(uint32_t a, uint32_t b, unsigned wa, unsigned wb, unsigned d)
{
    uint32_t out = 0;
    unsigned k;
    for (k = 0; k < 24; k += 8) out |= ((((a >> k) & 255u) * wa + ((b >> k) & 255u) * wb) / d) << k;
    return out;
}

/* One 4x4 block of BC1 colour into `px` (row pitch `pitch` texels); `bc1` allows the transparent fourth colour. */
static void ktp_bc1(const uint8_t *s, uint32_t *px, unsigned pitch, bool bc1)
{
    const uint16_t c0 = (uint16_t)(s[0] | s[1] << 8), c1 = (uint16_t)(s[2] | s[3] << 8);
    uint32_t pal[4], bits = (uint32_t)s[4] | (uint32_t)s[5] << 8 | (uint32_t)s[6] << 16 | (uint32_t)s[7] << 24;
    unsigned i;
    pal[0] = ktp_565(c0) | 0xFF000000u;
    pal[1] = ktp_565(c1) | 0xFF000000u;
    if (c0 > c1 || !bc1) {
        pal[2] = ktp_mix(pal[0], pal[1], 2, 1, 3) | 0xFF000000u;
        pal[3] = ktp_mix(pal[0], pal[1], 1, 2, 3) | 0xFF000000u;
    } else {
        pal[2] = ktp_mix(pal[0], pal[1], 1, 1, 2) | 0xFF000000u;
        pal[3] = 0;
    }
    for (i = 0; i < 16; i++, bits >>= 2) px[(i / 4) * pitch + i % 4] = pal[bits & 3u];
}

static void ktp_bc3_alpha(const uint8_t *s, uint32_t *px, unsigned pitch)
{
    const unsigned a0 = s[0], a1 = s[1];
    unsigned pal[8], i;
    uint64_t bits = 0;
    for (i = 0; i < 6; i++) bits |= (uint64_t)s[2 + i] << (8 * i);
    pal[0] = a0; pal[1] = a1;
    if (a0 > a1) for (i = 1; i < 7; i++) pal[1 + i] = ((7 - i) * a0 + i * a1) / 7;
    else {
        for (i = 1; i < 5; i++) pal[1 + i] = ((5 - i) * a0 + i * a1) / 5;
        pal[6] = 0; pal[7] = 255;
    }
    for (i = 0; i < 16; i++, bits >>= 3) {
        uint32_t *p = &px[(i / 4) * pitch + i % 4];
        *p = (*p & 0x00FFFFFFu) | (uint32_t)pal[bits & 7u] << 24;
    }
}

static void ktp_bc2_alpha(const uint8_t *s, uint32_t *px, unsigned pitch)
{
    unsigned i;
    for (i = 0; i < 16; i++) {
        const unsigned a = (s[i / 2] >> (4 * (i & 1))) & 15u;
        uint32_t *p = &px[(i / 4) * pitch + i % 4];
        *p = (*p & 0x00FFFFFFu) | (uint32_t)(a * 17u) << 24;
    }
}

/* fourcc 1, 2 or 3 (BC1, BC2, BC3): one level into w x h texels */
static void ktp_decode_bc(int bc, const uint8_t *s, unsigned w, unsigned h, uint32_t *out)
{
    const unsigned bw = (w + 3) / 4, bh = (h + 3) / 4, step = bc == 1 ? 8u : 16u;
    uint32_t block[16];
    unsigned bx, by, x, y;
    for (by = 0; by < bh; by++)
        for (bx = 0; bx < bw; bx++, s += step) {
            ktp_bc1(bc == 1 ? s : s + 8, block, 4, bc == 1);
            if (bc == 3) ktp_bc3_alpha(s, block, 4);
            else if (bc == 2) ktp_bc2_alpha(s, block, 4);
            for (y = 0; y < 4 && by * 4 + y < h; y++)
                for (x = 0; x < 4 && bx * 4 + x < w; x++) out[(by * 4 + y) * w + bx * 4 + x] = block[y * 4 + x];
        }
}

typedef struct KtpDds {
    int bc;                            /* 1, 2, 3 */
    unsigned w, h, levels;
    size_t offset[16], bytes[16];
    unsigned lw[16], lh[16];
} KtpDds;

static bool ktp_dds_header(const uint8_t *hd, size_t size, KtpDds *d)
{
    uint32_t fourcc;
    size_t at = 128;
    unsigned i;
    if (size < 128 || memcmp(hd, "DDS ", 4)) return false;
    d->h = (unsigned)(hd[12] | hd[13] << 8 | hd[14] << 16 | (uint32_t)hd[15] << 24);
    d->w = (unsigned)(hd[16] | hd[17] << 8 | hd[18] << 16 | (uint32_t)hd[19] << 24);
    d->levels = (unsigned)(hd[28] | hd[29] << 8);
    memcpy(&fourcc, hd + 84, 4);
    if (fourcc == 0x31545844u) d->bc = 1;
    else if (fourcc == 0x33545844u) d->bc = 2;
    else if (fourcc == 0x35545844u) d->bc = 3;
    else return false;
    if (!d->w || !d->h || d->w > 16384 || d->h > 16384) return false;
    if (!d->levels) d->levels = 1;
    if (d->levels > 16) d->levels = 16;
    for (i = 0; i < d->levels; i++) {
        const unsigned w = d->w >> i ? d->w >> i : 1, h = d->h >> i ? d->h >> i : 1;
        d->lw[i] = w; d->lh[i] = h;
        d->offset[i] = at;
        d->bytes[i] = (size_t)((w + 3) / 4) * ((h + 3) / 4) * (d->bc == 1 ? 8u : 16u);
        at += d->bytes[i];
    }
    while (d->levels > 1 && d->offset[d->levels - 1] + d->bytes[d->levels - 1] > size) d->levels--;
    return d->offset[0] + d->bytes[0] <= size;
}

static uint8_t *ktp_read(const wchar_t *path, size_t at, size_t bytes, size_t *got)
{
    HANDLE f = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    LARGE_INTEGER size, pos;
    uint8_t *buf = NULL;
    DWORD n = 0;
    *got = 0;
    if (f == INVALID_HANDLE_VALUE) return NULL;
    if (GetFileSizeEx(f, &size) && (uint64_t)size.QuadPart > at) {
        if (!bytes || at + bytes > (uint64_t)size.QuadPart) bytes = (size_t)size.QuadPart - at;
        pos.QuadPart = (LONGLONG)at;
        if (bytes < 0x40000000u && (buf = (uint8_t *)malloc(bytes)) != NULL && SetFilePointerEx(f, pos, NULL, FILE_BEGIN) &&
            ReadFile(f, buf, (DWORD)bytes, &n, NULL) && n == bytes)
            *got = bytes;
        else { free(buf); buf = NULL; }
    }
    CloseHandle(f);
    return buf;
}

/* A PNG (or anything WIC reads) as RGBA, R in the low byte. */
static uint32_t *ktp_wic(const wchar_t *path, unsigned *w, unsigned *h)
{
    IWICBitmapDecoder *dec = NULL;
    IWICBitmapFrameDecode *frame = NULL;
    IWICFormatConverter *conv = NULL;
    uint32_t *px = NULL;
    UINT fw = 0, fh = 0;
    HRESULT hr;
    if (!K.wic) {
        CoInitializeEx(NULL, COINIT_MULTITHREADED);
        if (FAILED(CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, &IID_IWICImagingFactory,
                                    (void **)&K.wic)))
            return NULL;
    }
    hr = IWICImagingFactory_CreateDecoderFromFilename(K.wic, path, NULL, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec);
    if (SUCCEEDED(hr)) hr = IWICBitmapDecoder_GetFrame(dec, 0, &frame);
    if (SUCCEEDED(hr)) hr = IWICImagingFactory_CreateFormatConverter(K.wic, &conv);
    if (SUCCEEDED(hr)) hr = IWICFormatConverter_Initialize(conv, (IWICBitmapSource *)frame, &GUID_WICPixelFormat32bppRGBA,
                                                           WICBitmapDitherTypeNone, NULL, 0.0, WICBitmapPaletteTypeCustom);
    if (SUCCEEDED(hr)) hr = IWICBitmapSource_GetSize((IWICBitmapSource *)conv, &fw, &fh);
    if (SUCCEEDED(hr) && fw && fh && fw <= 16384 && fh <= 16384 && (px = (uint32_t *)malloc((size_t)fw * fh * 4)) != NULL) {
        hr = IWICBitmapSource_CopyPixels((IWICBitmapSource *)conv, NULL, fw * 4, fw * fh * 4, (BYTE *)px);
        if (FAILED(hr)) { free(px); px = NULL; }
    }
    if (conv) IWICFormatConverter_Release(conv);
    if (frame) IWICBitmapFrameDecode_Release(frame);
    if (dec) IWICBitmapDecoder_Release(dec);
    *w = fw; *h = fh;
    return px;
}

/* ---- fingerprints -------------------------------------------------------------------------------------------- */

/* `rgba` (w x h) reduced to 32x32 cells of RGBA as floats. */
static void ktp_reduce(const uint32_t *rgba, unsigned w, unsigned h, float cells[KTP_CELLS][4])
{
    unsigned cx, cy, x, y, k;
    for (cy = 0; cy < KTP_SIDE; cy++)
        for (cx = 0; cx < KTP_SIDE; cx++) {
            unsigned x0 = cx * w / KTP_SIDE, x1 = (cx + 1) * w / KTP_SIDE, y0 = cy * h / KTP_SIDE, y1 = (cy + 1) * h / KTP_SIDE;
            double sum[4] = { 0, 0, 0, 0 };
            unsigned n = 0;
            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;
            for (y = y0; y < y1 && y < h; y++)
                for (x = x0; x < x1 && x < w; x++, n++) {
                    const uint32_t c = rgba[(size_t)y * w + x];
                    for (k = 0; k < 4; k++) sum[k] += (c >> (8 * k)) & 255u;
                }
            for (k = 0; k < 4; k++) cells[cy * KTP_SIDE + cx][k] = n ? (float)(sum[k] / n) : 0.0f;
        }
}

/* `ps2` doubles alpha first (a picture whose alpha never passes 128: the PS2's, 128 opaque). */
static void ktp_print(float cells[KTP_CELLS][4], bool may_be_ps2, KtpPrint *p)
{
    float mean = 0.0f, norm = 0.0f, top = 0.0f;
    unsigned i, k, clear = 0;
    memset(p, 0, sizeof(*p));
    for (i = 0; i < KTP_CELLS; i++) if (cells[i][3] > top) top = cells[i][3];
    p->ps2_alpha = may_be_ps2 && top <= 130.0f;
    for (i = 0; i < KTP_CELLS; i++) {
        float a = p->ps2_alpha ? cells[i][3] * 2.0f : cells[i][3];
        if (a > 255.0f) a = 255.0f;
        cells[i][3] = a;
        if (a < 250.0f) clear++;
        for (k = 0; k < 3; k++) cells[i][k] *= a / 255.0f;
    }
    p->clear = clear > KTP_CELLS / 50;
    for (i = 0; i < KTP_CELLS; i++) {
        const unsigned gx = (i % KTP_SIDE) / 8, gy = (i / KTP_SIDE) / 8;
        p->grey[i] = (cells[i][0] + cells[i][1] + cells[i][2]) / 3.0f;
        mean += p->grey[i];
        for (k = 0; k < 3; k++) p->colour[(gy * 4 + gx) * 3 + k] += cells[i][k] / 64.0f;
        p->alpha[gy * 4 + gx] += cells[i][3] / 64.0f;
    }
    mean /= KTP_CELLS;
    for (i = 0; i < KTP_CELLS; i++) { p->grey[i] -= mean; norm += p->grey[i] * p->grey[i]; }
    norm = sqrtf(norm) + 1e-6f;
    for (i = 0; i < KTP_CELLS; i++) p->grey[i] /= norm;
}

static bool ktp_file_print(KtpFile *f)
{
    float (*cells)[4] = (float (*)[4])malloc(sizeof(float) * 4 * KTP_CELLS);
    size_t len = wcslen(f->path), got;
    bool ok = false;
    if (!cells) return false;
    if (len > 4 && !_wcsicmp(f->path + len - 4, L".dds")) {
        uint8_t *hd = ktp_read(f->path, 0, 128, &got);
        KtpDds d;
        if (hd && ktp_dds_header(hd, (size_t)f->size, &d)) {
            unsigned l = 0;
            uint8_t *data;
            uint32_t *px;
            while (l + 1 < d.levels && d.lw[l + 1] >= 64 && d.lh[l + 1] >= 64) l++;
            data = ktp_read(f->path, d.offset[l], d.bytes[l], &got);
            px = (uint32_t *)malloc((size_t)d.lw[l] * d.lh[l] * 4);
            if (data && px && got == d.bytes[l]) {
                ktp_decode_bc(d.bc, data, d.lw[l], d.lh[l], px);
                ktp_reduce(px, d.lw[l], d.lh[l], cells);
                f->print.aspect = (float)d.w / d.h;
                ok = true;
            }
            free(data); free(px);
        }
        free(hd);
    } else {
        unsigned w, h;
        uint32_t *px = ktp_wic(f->path, &w, &h);
        if (px) {
            ktp_reduce(px, w, h, cells);
            f->print.aspect = (float)w / h;
            ok = true;
            free(px);
        }
    }
    if (ok) {
        const float aspect = f->print.aspect;
        ktp_print(cells, true, &f->print);
        f->print.aspect = aspect;
    }
    free(cells);
    return ok;
}

/* ---- the pack ------------------------------------------------------------------------------------------------ */

static void ktp_scan(const wchar_t *dir, int depth)
{
    wchar_t pattern[MAX_PATH];
    WIN32_FIND_DATAW fd;
    HANDLE h;
    if (depth > 6 || _snwprintf(pattern, MAX_PATH, L"%s\\*", dir) < 0) return;
    pattern[MAX_PATH - 1] = 0;
    if ((h = FindFirstFileW(pattern, &fd)) == INVALID_HANDLE_VALUE) return;
    do {
        wchar_t path[MAX_PATH];
        size_t n;
        if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
        if (_snwprintf(path, MAX_PATH, L"%s\\%s", dir, fd.cFileName) < 0) continue;
        path[MAX_PATH - 1] = 0;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) { ktp_scan(path, depth + 1); continue; }
        n = wcslen(fd.cFileName);
        if (n < 5 || (_wcsicmp(fd.cFileName + n - 4, L".dds") && _wcsicmp(fd.cFileName + n - 4, L".png"))) continue;
        if (K.count == K.cap) {
            const unsigned cap = K.cap ? K.cap * 2 : 1024;
            KtpFile *grown = (KtpFile *)realloc(K.files, cap * sizeof(KtpFile));
            if (!grown) break;
            K.files = grown; K.cap = cap;
        }
        memset(&K.files[K.count], 0, sizeof(KtpFile));
        K.files[K.count].path = _wcsdup(path);
        K.files[K.count].size = ((uint64_t)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
        K.files[K.count].written = ((uint64_t)fd.ftLastWriteTime.dwHighDateTime << 32) | fd.ftLastWriteTime.dwLowDateTime;
        K.count++;
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}

static int ktp_order(const void *a, const void *b)
{
    return _wcsicmp(((const KtpFile *)a)->path, ((const KtpFile *)b)->path);
}

/* The fingerprints are kept in the pack's folder (black_recomp_index.bin): the files' names, sizes and times as they
 * were, and each one's fingerprint; a file that changed is looked at again. */
static void ktp_index(void)
{
    wchar_t cache[MAX_PATH];
    FILE *f;
    unsigned i, fresh = 0;
    _snwprintf(cache, MAX_PATH, L"%s\\black_recomp_index.bin", K.root);
    cache[MAX_PATH - 1] = 0;
    if ((f = _wfopen(cache, L"rb")) != NULL) {
        uint32_t version = 0, count = 0;
        if (fread(&version, 4, 1, f) == 1 && version == KTP_INDEX_VERSION && fread(&count, 4, 1, f) == 1) {
            for (; count; count--) {
                uint32_t len = 0;
                wchar_t name[MAX_PATH];
                uint64_t size, written;
                KtpPrint p;
                KtpFile key, *hit;
                if (fread(&len, 4, 1, f) != 1 || len >= MAX_PATH || fread(name, 2, len, f) != len ||
                    fread(&size, 8, 1, f) != 1 || fread(&written, 8, 1, f) != 1 || fread(&p, sizeof(p), 1, f) != 1)
                    break;
                name[len] = 0;
                key.path = name;
                hit = (KtpFile *)bsearch(&key, K.files, K.count, sizeof(KtpFile), ktp_order);
                if (hit && hit->size == size && hit->written == written) { hit->print = p; hit->ok = true; }
            }
        }
        fclose(f);
    }
    for (i = 0; i < K.count; i++)
        if (!K.files[i].ok) { K.files[i].ok = ktp_file_print(&K.files[i]); fresh++; }
    if (fresh && (f = _wfopen(cache, L"wb")) != NULL) {
        uint32_t version = KTP_INDEX_VERSION, count = 0;
        for (i = 0; i < K.count; i++) count += K.files[i].ok;
        fwrite(&version, 4, 1, f);
        fwrite(&count, 4, 1, f);
        for (i = 0; i < K.count; i++) {
            const KtpFile *e = &K.files[i];
            uint32_t len = (uint32_t)wcslen(e->path);
            if (!e->ok) continue;
            fwrite(&len, 4, 1, f);
            fwrite(e->path, 2, len, f);
            fwrite(&e->size, 8, 1, f);
            fwrite(&e->written, 8, 1, f);
            fwrite(&e->print, sizeof(e->print), 1, f);
        }
        fclose(f);
    }
    fprintf(stderr, "[TEXPACK] %u files, %u looked at now\n", K.count, fresh);
}

bool ktp_enabled(void)
{
    if (K.state == 0) {
        const char *dir = getenv("RECOMP_TEXTURE_PACK");
        K.state = -1;
        if (dir && *dir && MultiByteToWideChar(CP_UTF8, 0, dir, -1, K.root, MAX_PATH) > 0) {
            size_t n = wcslen(K.root);
            while (n && (K.root[n - 1] == L'\\' || K.root[n - 1] == L'/')) K.root[--n] = 0;
            ktp_scan(K.root, 0);
            if (K.count) {
                qsort(K.files, K.count, sizeof(KtpFile), ktp_order);
                ktp_index();
                K.state = 1;
            } else fprintf(stderr, "[TEXPACK] no DDS or PNG files under %s\n", dir);
        }
    }
    return K.state == 1;
}

/* ---- matching ------------------------------------------------------------------------------------------------ */

/* A file's likeness to a game texture: the grey pictures' correlation less the colour grids' distance / 100; none
 * when their shapes differ, one is see-through and the other not, or their alpha grids differ by 40 on average. */
static float ktp_score(const KtpPrint *a, const KtpPrint *b)
{
    float corr = 0.0f, cd = 0.0f, ad = 0.0f;
    unsigned i;
    if (fabsf(logf(a->aspect / b->aspect)) >= 0.1f || a->clear != b->clear) return -9.0f;
    for (i = 0; i < 16; i++) ad += fabsf(a->alpha[i] - b->alpha[i]);
    if (ad / 16.0f >= 40.0f) return -9.0f;
    for (i = 0; i < 48; i++) cd += (a->colour[i] - b->colour[i]) * (a->colour[i] - b->colour[i]);
    for (i = 0; i < KTP_CELLS; i++) corr += a->grey[i] * b->grey[i];
    return corr - sqrtf(cd / 48.0f) / 100.0f;
}

int ktp_match(const uint32_t *rgba, unsigned w, unsigned h)
{
    float (*cells)[4];
    KtpPrint p;
    float best = -9.0f, second = -9.0f;
    int at = -1;
    unsigned i;
    if (!ktp_enabled() || !rgba || w < 16 || h < 16) return -1;
    if (!(cells = (float (*)[4])malloc(sizeof(float) * 4 * KTP_CELLS))) return -1;
    ktp_reduce(rgba, w, h, cells);
    ktp_print(cells, false, &p);
    p.aspect = (float)w / h;
    free(cells);
    for (i = 0; i < K.count; i++) {
        float s;
        if (!K.files[i].ok) continue;
        s = ktp_score(&p, &K.files[i].print);
        if (s > best) { second = best; best = s; at = (int)i; }
        else if (s > second) second = s;
    }
    /* a see-through texture (effects, the HUD's) must look much more alike */
    if (at < 0 || best < (p.clear ? 0.85f : 0.55f) || best - second < 0.12f) return -1;
    return at;
}

/* ---- the replacement ----------------------------------------------------------------------------------------- */

/* BC3 alpha on the PS2's scale (128 opaque) to the full range: both end points doubled. */
static void ktp_bc3_double_alpha(uint8_t *s, size_t bytes)
{
    size_t i;
    for (i = 0; i + 16 <= bytes; i += 16) {
        const unsigned a0 = s[i], a1 = s[i + 1];
        const unsigned b0 = a0 * 2 > 255 ? 255 : a0 * 2, b1 = a1 * 2 > 255 ? 255 : a1 * 2;
        s[i] = (uint8_t)b0; s[i + 1] = (uint8_t)b1;
        if (a0 > a1 && b0 == b1) memset(s + i + 2, 0, 6);  /* both opaque now: every texel the first */
    }
}

static void ktp_downsample(const uint32_t *src, unsigned w, unsigned h, uint32_t *dst)
{
    const unsigned dw = w > 1 ? w / 2 : 1, dh = h > 1 ? h / 2 : 1;
    unsigned x, y, k;
    for (y = 0; y < dh; y++)
        for (x = 0; x < dw; x++) {
            const unsigned x0 = x * 2 < w ? x * 2 : w - 1, y0 = y * 2 < h ? y * 2 : h - 1;
            const unsigned x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
            uint32_t out = 0;
            for (k = 0; k < 32; k += 8) {
                const uint32_t s = ((src[y0 * w + x0] >> k) & 255u) + ((src[y0 * w + x1] >> k) & 255u) +
                                   ((src[y1 * w + x0] >> k) & 255u) + ((src[y1 * w + x1] >> k) & 255u);
                out |= ((s + 2) / 4) << k;
            }
            dst[y * dw + x] = out;
        }
}

bool ktp_texture(ID3D11Device *dev, int index, D3D11_USAGE usage, UINT misc, ID3D11Texture2D **tex,
                 ID3D11ShaderResourceView **srv, size_t *bytes)
{
    KtpFile *f;
    D3D11_TEXTURE2D_DESC td;
    D3D11_SUBRESOURCE_DATA init[16];
    uint8_t *data = NULL;
    uint32_t *levels[16];
    unsigned i, n = 0;
    size_t len, got, total = 0;
    *tex = NULL; *srv = NULL; *bytes = 0;
    if (!ktp_enabled() || index < 0 || (unsigned)index >= K.count) return false;
    f = &K.files[index];
    if (!f->tex) {
        memset(&td, 0, sizeof(td));
        memset(levels, 0, sizeof(levels));
        td.ArraySize = 1; td.SampleDesc.Count = 1; td.Usage = usage; td.BindFlags = D3D11_BIND_SHADER_RESOURCE; td.MiscFlags = misc;
        len = wcslen(f->path);
        if (len > 4 && !_wcsicmp(f->path + len - 4, L".dds")) {
            KtpDds d;
            data = ktp_read(f->path, 0, 0, &got);
            if (!data || !ktp_dds_header(data, got, &d)) goto fail;
            if (d.bc == 3 && f->print.ps2_alpha) ktp_bc3_double_alpha(data + 128, got - 128);
            td.Width = d.w; td.Height = d.h;
            td.Format = d.bc == 1 ? DXGI_FORMAT_BC1_UNORM : d.bc == 2 ? DXGI_FORMAT_BC2_UNORM : DXGI_FORMAT_BC3_UNORM;
            /* the block formats want every level's size a multiple of four but the 1x1 and 2x2 ends */
            for (n = 0; n < d.levels; n++) {
                init[n].pSysMem = data + d.offset[n];
                init[n].SysMemPitch = ((d.lw[n] + 3) / 4) * (d.bc == 1 ? 8u : 16u);
                init[n].SysMemSlicePitch = 0;
                total += d.bytes[n];
            }
            if ((d.w % 4) || (d.h % 4)) goto fail;
        } else {
            unsigned w, h;
            uint32_t *px = ktp_wic(f->path, &w, &h);
            if (!px) goto fail;
            if (f->print.ps2_alpha)
                for (i = 0; i < w * h; i++) {
                    const uint32_t a = px[i] >> 24;
                    px[i] = (px[i] & 0x00FFFFFFu) | (a * 2 > 255 ? 255u : a * 2) << 24;
                }
            td.Width = w; td.Height = h; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            levels[0] = px;
            for (n = 0; n < 16; n++) {
                const unsigned lw = w >> n ? w >> n : 1, lh = h >> n ? h >> n : 1;
                if (n && !levels[n]) {
                    const unsigned pw = w >> (n - 1) ? w >> (n - 1) : 1, ph = h >> (n - 1) ? h >> (n - 1) : 1;
                    if (!(levels[n] = (uint32_t *)malloc((size_t)lw * lh * 4))) goto fail;
                    ktp_downsample(levels[n - 1], pw, ph, levels[n]);
                }
                init[n].pSysMem = levels[n];
                init[n].SysMemPitch = lw * 4;
                init[n].SysMemSlicePitch = 0;
                total += (size_t)lw * lh * 4;
                if (lw == 1 && lh == 1) { n++; break; }
            }
        }
        td.MipLevels = n;
        if (FAILED(ID3D11Device_CreateTexture2D(dev, &td, init, &f->tex)) ||
            FAILED(ID3D11Device_CreateShaderResourceView(dev, (ID3D11Resource *)f->tex, NULL, &f->srv))) {
            if (f->tex) { ID3D11Texture2D_Release(f->tex); f->tex = NULL; }
            goto fail;
        }
        f->bytes = total;
        free(data);
        for (i = 0; i < 16; i++) free(levels[i]);
    }
    ID3D11Texture2D_AddRef(f->tex);
    ID3D11ShaderResourceView_AddRef(f->srv);
    *tex = f->tex; *srv = f->srv; *bytes = f->bytes;
    return true;
fail:
    {
        static int said;
        if (said++ < 8) fprintf(stderr, "[TEXPACK] could not load %ls\n", f->path);
    }
    free(data);
    for (i = 0; i < 16; i++) free(levels[i]);
    f->ok = false;
    return false;
}
