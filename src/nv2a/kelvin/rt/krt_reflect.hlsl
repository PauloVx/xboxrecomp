// Ray-traced reflections for the pixels the screen-space reflection targets marked (kelvin_gpu_ssr.inc).
//
// The pixel's surface is found again by a ray from the camera: its exact position and its triangle's normal. If that
// ray hits farther than the depth the pixel was drawn at, the reflective surface is one the scene does not hold --
// the first-person weapon, BLACK's most reflective surface, a hand's length from the eye. Its reflections start at
// the eye, along the normal the screen-space targets hold, turned from view to world space. A ray follows the
// reflected direction. A hit that is on screen and seen there from the camera takes the picture's colour at that
// point: exactly what the game drew. Output: rgb that colour, a = 1 traced but no colour (the original environment),
// 2 a hit with its colour.
#include "krt_common.hlsli"

Texture2D<float4> weight : register(t5);    // SsrW: how much the pixel's environment map adds
Texture2D<float4> picture : register(t6);   // the frame so far
Texture2D<float4> surface : register(t7);   // SsrG: the pixel's depth (x)
RWTexture2D<float4> output : register(u0);

bool first_hit(RayDesc r, out float t, out uint geometry, out uint primitive)
{
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, r);
    q.Proceed();
    t = 0.0; geometry = 0; primitive = 0;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return false;
    t = q.CommittedRayT();
    geometry = q.CommittedGeometryIndex();
    primitive = q.CommittedPrimitiveIndex();
    return true;
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= (uint2)size.xy)) return;
    float3 w = weight.Load(int3(id.xy, 0)).rgb;
    if (max(w.r, max(w.g, w.b)) < 0.002) { output[id.xy] = 0.0; return; }

    RayDesc cr = camera_ray(id.xy);
    float t, t2;
    uint g, p, g2, p2;
    if (!first_hit(cr, t, g, p)) t = 1.0e6;
    float3 P = cr.Origin + cr.Direction * t;
    // the depth the pixel was drawn at, as a distance along the view direction
    float4 drawn_at = surface.Load(int3(id.xy, 0));
    float drawn = view_w_from_depth(drawn_at.x);
    float here = view_w(P);
    RayDesc rr;
    if (abs(here - drawn) > 0.02 * drawn + 0.05) {
        // not the scene's: the weapon
        float3 N = view_to_world(drawn_at.yzw);
        rr.Direction = reflect(cr.Direction, N);
        rr.Origin = cr.Origin + cr.Direction * 0.3;
    } else {
        float3 a, b, c;
        triangle_at(g, p, a, b, c);
        float3 N = normalize(cross(b - a, c - a));
        if (dot(N, cr.Direction) > 0.0) N = -N;
        rr.Direction = reflect(cr.Direction, N);
        rr.Origin = P + N * (0.002 * here + 0.005);
    }
    rr.TMin = 0.0;
    rr.TMax = 1.0e5;
    if (!first_hit(rr, t2, g2, p2)) { output[id.xy] = float4(0.0, 0.0, 0.0, 1.0); return; }
    float3 H = rr.Origin + rr.Direction * t2;

    // on screen, and the camera sees that very point?
    float2 px;
    if (to_pixel(H, px)) {
        RayDesc vr = camera_ray((uint2)px);
        float tv;
        uint gv, pv;
        if (first_hit(vr, tv, gv, pv)) {
            float3 V = vr.Origin + vr.Direction * tv;
            if ((gv == g2 && pv == p2) || distance(V, H) < 0.01 * view_w(H) + 0.02) {
                output[id.xy] = float4(picture.Load(int3(px, 0)).rgb, 2.0);
                return;
            }
        }
    }
    output[id.xy] = float4(0.0, 0.0, 0.0, 1.0);
}
