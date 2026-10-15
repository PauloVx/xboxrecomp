#ifndef XBOXRECOMP_FB_PRESENT_POLICY_H
#define XBOXRECOMP_FB_PRESENT_POLICY_H

#include <stdint.h>
#include <string.h>

enum { FB_SCALE_FIT, FB_SCALE_STRETCH, FB_SCALE_INTEGER };
typedef struct FbPresentPolicy {
    unsigned width, height, borderless, scale, nearest, vsync, fps_limit;
    unsigned aspect_w, aspect_h;
} FbPresentPolicy;
typedef struct FbPresentRect { unsigned x, y, width, height, nearest; } FbPresentRect;

/* Strict decimal parsing: malformed/overflowing/out-of-range values fall back. */
static unsigned fb_decimal(const char *text, unsigned lo, unsigned hi,
                           unsigned fallback)
{
    unsigned value = 0;
    const unsigned char *p = (const unsigned char *)text;
    if (!p || !*p) return fallback;
    do {
        unsigned digit = *p++ - '0';
        if (digit > 9 || value > (hi - digit) / 10) return fallback;
        value = value * 10 + digit;
    } while (*p);
    return value >= lo && value <= hi ? value : fallback;
}

static FbPresentPolicy fb_policy(const char *width, const char *height,
    const char *mode, const char *scale, const char *filter,
    const char *vsync, const char *limit)
{
    FbPresentPolicy p;
    p.aspect_w = p.aspect_h = 0;
    p.width = fb_decimal(width, 320, 7680, 1280);
    p.height = fb_decimal(height, 240, 4320, 960);
    p.borderless = mode && !strcmp(mode, "borderless");
    p.scale = scale && !strcmp(scale, "stretch") ? FB_SCALE_STRETCH :
              scale && !strcmp(scale, "integer") ? FB_SCALE_INTEGER : FB_SCALE_FIT;
    p.nearest = filter && !strcmp(filter, "nearest");
    p.vsync = !(vsync && !strcmp(vsync, "0"));
    p.fps_limit = fb_decimal(limit, 0, 240, 0);
    if (p.fps_limit != 0 && p.fps_limit != 30 && p.fps_limit != 60 &&
        p.fps_limit != 120 && p.fps_limit != 144 && p.fps_limit != 165 &&
        p.fps_limit != 240) p.fps_limit = 0;
    return p;
}

static void fb_policy_aspect(FbPresentPolicy *p, const char *aspect)
{
    p->aspect_w = p->aspect_h = 0;
    if (!aspect) return;
    if (!strcmp(aspect, "4:3")) { p->aspect_w = 4; p->aspect_h = 3; }
    if (!strcmp(aspect, "16:9")) { p->aspect_w = 16; p->aspect_h = 9; }
    if (!strcmp(aspect, "21:9")) { p->aspect_w = 21; p->aspect_h = 9; }
    if (!strcmp(aspect, "32:9")) { p->aspect_w = 32; p->aspect_h = 9; }
}

/* Exact 20:11 metadata opts into the known BLACK cinematic aperture:
 * source rows 64..415 of 480, including when the host raster is larger.
 * Reduced values cannot collide with ordinary source raster dimensions.
 * It travels with the completed frame, never with current decoder state. */
static unsigned fb_cinematic_crop(unsigned aspect_w, unsigned aspect_h)
{
    return aspect_w == 20u && aspect_h == 11u;
}

static FbPresentRect fb_present_rect(unsigned client_w, unsigned client_h,
    unsigned source_w, unsigned source_h, unsigned scale, unsigned nearest,
    unsigned aspect_w, unsigned aspect_h)
{
    FbPresentRect r = { 0, 0, 0, 0, nearest };
    uint64_t w, h;
    if (!client_w || !client_h || !source_w || !source_h) return r;
    if (fb_cinematic_crop(aspect_w, aspect_h)) scale = FB_SCALE_FIT;
    w = client_w; h = client_h;
    if (scale == FB_SCALE_INTEGER) {
        unsigned kx = client_w / source_w, ky = client_h / source_h;
        unsigned k = kx < ky ? kx : ky;
        if (k) { w = (uint64_t)source_w * k; h = (uint64_t)source_h * k; }
        else scale = FB_SCALE_FIT; /* fit small windows without cropping */
        /* An anamorphic source cannot both preserve its display aspect and
         * use one integral scale for square source pixels. Fit using the selected filter. */
        if (aspect_w && aspect_h &&
            (uint64_t)aspect_w * source_h != (uint64_t)aspect_h * source_w)
            scale = FB_SCALE_FIT;
    }
    if (scale == FB_SCALE_FIT) {
        w = client_w; h = client_h; /* discard any earlier integer-size choice */
        if (aspect_w && aspect_h) { source_w = aspect_w; source_h = aspect_h; }
        if ((uint64_t)client_w * source_h > (uint64_t)client_h * source_w)
            w = (uint64_t)client_h * source_w / source_h;
        else h = (uint64_t)client_w * source_h / source_w;
    }
    if (!w) w = 1;
    if (!h) h = 1;
    r.width = (unsigned)w; r.height = (unsigned)h;
    r.x = (client_w - r.width) / 2; r.y = (client_h - r.height) / 2;
    return r;
}

#endif
