// Ray-traced reflections for the pixels the screen-space reflection targets marked (kelvin_gpu_ssr.inc).
//
// The pixel's surface is found again by a ray from the camera: its exact position and its triangle's normal. If that
// ray hits farther than the depth the pixel was drawn at, the reflective surface is one the scene does not hold --
// the first-person weapon, BLACK's most reflective surface, a hand's length from the eye. Its reflections start at
// the eye, along the normal the screen-space targets hold, turned from view to world space. A ray follows the
// reflected direction. A hit that is on screen and seen there from the camera takes the picture's colour at that
// point: exactly what the game drew. Any other hit is shaded from the game's own textures (shade); nothing hit, the
// picture itself as a soft environment, looked up by direction as the screen-space pass does. The result is raw: one
// ray a pixel, so it is filtered next (krt_reflect_filter.hlsl). Raw: rgb the colour, a = 0 not reflective, 2 the
// scene's surface, 3 the weapon's.
#include "krt_common.hlsli"

Texture2D<float4> weight : register(t5);    // SsrW: how much the pixel's environment map adds
Texture2D<float4> picture : register(t6);   // the frame so far
Texture2D<float4> surface : register(t7);   // SsrG: the pixel's depth (x) and view-space normal (yzw)
RWTexture2D<float4> raw : register(u6);

// What a ray that leaves the scene sees: the picture, blurred, at where its direction points on the screen.
float3 environment(float3 dir)
{
    float3 v = float3(dot(dir, normalize(vp0.xyz)), dot(dir, normalize(vp1.xyz)), dot(dir, normalize(vp3.xyz)));
    float2 uv = float2(0.5 + v.x * 0.45, 0.5 - v.y * 0.45);
    float3 sum = 0.0;
    [unroll] for (int y = -1; y <= 1; y++)
        [unroll] for (int x = -1; x <= 1; x++) {
            int2 px = clamp(int2(uv * size.xy + float2(x, y) * size.y * 0.04), int2(0, 0), int2(size.xy) - 1);
            sum += picture.Load(int3(px, 0)).rgb;
        }
    return sum / 9.0;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= (uint2)size.xy)) return;
    float3 w = weight.Load(int3(id.xy, 0)).rgb;
    if (max(w.r, max(w.g, w.b)) < 0.002) { raw[id.xy] = 0.0; return; }

    RayDesc cr = camera_ray(id.xy);
    Hit h = first_hit(cr);
    float3 P = cr.Origin + cr.Direction * (h.found ? h.t : 1.0e6);
    // the depth the pixel was drawn at, as a distance along the view direction
    float4 drawn_at = surface.Load(int3(id.xy, 0));
    float drawn = view_w_from_depth(drawn_at.x);
    float here = view_w(P);
    float kind = 2.0;
    RayDesc rr;
    rr.TMin = 0.0;
    rr.TMax = 1.0e5;
    if (abs(here - drawn) > 0.02 * drawn + 0.05) {
        // not the scene's: the weapon
        float3 N = view_to_world(drawn_at.yzw);
        rr.Direction = reflect(cr.Direction, N);
        rr.Origin = cr.Origin + cr.Direction * 0.3;
        kind = 3.0;
    } else {
        float3 a, b, c;
        triangle_at(h.geometry, h.primitive, a, b, c);
        float3 N = normalize(cross(b - a, c - a));
        if (dot(N, cr.Direction) > 0.0) N = -N;
        rr.Direction = reflect(cr.Direction, N);
        rr.Origin = P + N * (0.002 * here + 0.005);
    }
    Hit rh = first_hit(rr);
    if (!rh.found) { raw[id.xy] = float4(environment(rr.Direction), kind); return; }
    float3 H = rr.Origin + rr.Direction * rh.t;

    // on screen, and the camera sees that very point?
    float2 px;
    if (to_pixel(H, px)) {
        RayDesc vr = camera_ray((uint2)px);
        Hit vh = first_hit(vr);
        if (vh.found) {
            float3 V = vr.Origin + vr.Direction * vh.t;
            if ((vh.geometry == rh.geometry && vh.primitive == rh.primitive) || distance(V, H) < 0.01 * view_w(H) + 0.02) {
                raw[id.xy] = float4(picture.Load(int3(px, 0)).rgb, kind);
                return;
            }
        }
    }
    raw[id.xy] = float4(shade(rh.geometry, rh.primitive, rh.bary, rh.t), kind);
}