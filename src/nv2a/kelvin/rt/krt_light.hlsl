// Ray-traced shadows, occlusion and bounce colour, as one factor the picture is multiplied by (kelvin_gpu_rt.inc).
//
// BLACK bakes its lighting: the level is lit by lightmaps that already hold its shadows and bounce light, and its
// props and characters by a few ambient coefficients that hold none. So the rays look for what the bake could not
// know about. A surface of the level is shadowed and occluded only by what has a transform of its own (instance mask
// 2): characters, props, debris. Those are shadowed and occluded by everything, and take on the colour of what is
// around them. The light is the main light the ambient coefficients point at (kg_rtc_light).
//
// At half the output's width and height, a different one of each 2x2 block's pixels each frame: one shadow ray and two
// occlusion rays, averaged over frames. The history is last frame's result where this pixel's surface was then (last
// frame's camera), kept if it was the same surface. krt_light_filter.hlsl smooths it; the composite scales it up.
#include "krt_common.hlsli"

Texture2D<float> depth : register(t8);      // the frame's depth buffer
RWTexture2D<float4> light_out : register(u3);
RWTexture2D<float4> history_a : register(u4); // rgb the factor, a the surface's distance (0: none)
RWTexture2D<float4> history_b : register(u5); // even frames write a and read b, odd frames the other way round

float luminance(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

float hash(uint2 p, uint s)
{
    uint h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u ^ s * 0xcb1ab31fu;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (h & 0xffffffu) / 16777216.0;
}

// A direction about n, more of them near n (cosine-weighted).
float3 cosine_dir(float3 n, float2 u)
{
    float3 t = abs(n.y) < 0.99 ? normalize(cross(n, float3(0, 1, 0))) : normalize(cross(n, float3(1, 0, 0)));
    float3 b = cross(n, t);
    float r = sqrt(u.x), phi = 6.2831853 * u.y;
    return normalize(t * r * cos(phi) + b * r * sin(phi) + n * sqrt(max(0.0, 1.0 - u.x)));
}

bool prev_pixel(float3 p, out float2 px, out float w)
{
    float4 h = float4(p, 1.0);
    w = dot(prev_vp3, h);
    px = 0.0;
    if (w <= 0.01) return false;
    float2 pp = float2(dot(prev_vp0, h), dot(prev_vp1, h)) / w * c58.xy + c59.xy;
    px = pp / size.zw * size.xy;
    return all(px >= 0.0) && all(px < size.xy);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 half_size = ((uint2)size.xy + 1u) / 2u;
    if (any(id.xy >= half_size)) return;
    uint f = (uint)misc.x;
    bool odd = (f & 1u) != 0;
    uint2 px = min(id.xy * 2u + uint2(f & 1u, (f >> 1) & 1u), (uint2)size.xy - 1u);
    float4 now = float4(1.0, 1.0, 1.0, 0.0);
    float3 P = 0.0;
    float d = depth.Load(int3(px, 0));
    if (d < 0.99999) {
        RayDesc cr = camera_ray(px);
        Hit h = first_hit(cr);
        float drawn = view_w_from_depth(d);
        P = cr.Origin + cr.Direction * (h.found ? h.t : 0.0);
        // the surface drawn here is the scene's (not the weapon, a particle or the sky)
        if (h.found && abs(view_w(P) - drawn) <= 0.02 * drawn + 0.05) {
            uint4 info = draw_info[h.geometry];
            bool dynamic = (info.w & KRT_DYNAMIC) != 0;
            uint mask = dynamic ? 0xFF : KRT_DYNAMIC_MASK;
            float3 a, b, c;
            triangle_at(h.geometry, h.primitive, a, b, c);
            float3 N = normalize(cross(b - a, c - a));
            if (dot(N, cr.Direction) > 0.0) N = -N;
            float3 O = P + N * (0.002 * drawn + 0.01);
            float2 u = float2(hash(px, f * 3u), hash(px, f * 3u + 1u));

            float shadow = 1.0;
            float ndl = dot(N, light.xyz);
            if (light.w > 0.0 && ndl > 0.0) {
                // a soft edge: the light a little wider than a point, a different direction in it each frame
                float3 L = normalize(light.xyz + (cosine_dir(light.xyz, u) - light.xyz) * 0.02);
                RayDesc sr;
                sr.Origin = O; sr.Direction = L; sr.TMin = 0.0; sr.TMax = 300.0;
                if (occluded(sr, mask)) shadow = 1.0 - light.w * saturate(ndl * 2.0);
            }

            float occlusion = 0.0, seen = 0.0;
            float3 bounce = 0.0;
            if (fx.x > 0.0 || (dynamic && fx.z > 0.0)) {
                [unroll] for (uint k = 0; k < 2; k++) {
                    float2 uk = frac(u + float2(0.5, 0.25) * k);
                    RayDesc ar;
                    ar.Origin = O; ar.Direction = cosine_dir(N, uk); ar.TMin = 0.0; ar.TMax = dynamic ? fx.y * 4.0 : fx.y;
                    Hit ah = first_hit_in(ar, mask);
                    if (ah.found) {
                        occlusion += saturate(1.0 - ah.t / fx.y);
                        if (dynamic) { bounce += shade(ah.geometry, ah.primitive, ah.bary, ah.t); seen += 1.0; }
                    }
                }
            }
            float ao = 1.0 - fx.x * occlusion * 0.5;
            float3 tint = 1.0;
            if (dynamic && seen > 0.0 && fx.z > 0.0) {
                float3 e = bounce / seen;
                tint = lerp(1.0, e / max(luminance(e), 0.02), saturate(fx.z * seen * 0.5));
            }
            now = float4(shadow * ao * tint, view_w(P));
        }
    }

    // last frame's result for this surface, if it was there
    float4 result = now;
    if (now.a > 0.0 && misc.y > 0.5) {
        float2 pq;
        float w;
        if (prev_pixel(P, pq, w)) {
            uint2 hq = min((uint2)(pq * 0.5), half_size - 1u);
            float4 old = odd ? history_b[hq] : history_a[hq];
            if (old.a > 0.0 && abs(old.a - w) < 0.05 * w + 0.1) result.rgb = lerp(now.rgb, old.rgb, 0.9);
        }
    }
    if (odd) history_a[id.xy] = result; else history_b[id.xy] = result;   /* krt_light_filter.hlsl makes light_out */
}
