// Ray-traced shadows, occlusion and bounce colour, as one factor the picture is multiplied by (kelvin_gpu_rt.inc).
//
// BLACK bakes its lighting: the level is lit by lightmaps that already hold its shadows and bounce light, and its
// props and characters by a few ambient coefficients that hold none. So the rays look for what the bake could not
// know about. A surface of the level is shadowed only by what has a transform of its own (instance mask 2):
// characters, props, debris; it is occluded by those, and by the level itself only in contact, finer than the bake's
// texels. Characters and props are shadowed and occluded by everything. Both take on the colour of what is around
// them, followed over 4 bounces; the level only its colour, at the brightness the bake gave it. The light is the main
// light the ambient coefficients point at (kg_rtc_light).
//
// At half the output's width and height, a different one of each 2x2 block's pixels each frame: one shadow ray and two
// gather rays, averaged over frames. The history is last frame's result where this pixel's surface was then (last
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

static const uint KRT_GI_BOUNCES = 4u;

// A hit's base colour, which the light it passes on is multiplied by.
float3 albedo_at(uint geometry, uint primitive, float2 bary, float t)
{
    uint4 info = draw_info[geometry];
    if (!(info.w & KRT_TEX0)) return 0.5;
    float4 uvs, colour;
    attributes_at(geometry, primitive, bary, uvs, colour);
    return sample_stage(info, 0, uvs.xy, clamp(log2(max(t, 1.0) / 6.0), 0.0, 6.0)).rgb;
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
            if (fx.x > 0.0 || fx.z > 0.0) {
                [unroll] for (uint k = 0; k < 2; k++) {
                    float2 uk = frac(u + float2(0.5, 0.25) * k);
                    RayDesc ar;
                    ar.Origin = O; ar.Direction = cosine_dir(N, uk); ar.TMin = 0.0; ar.TMax = fx.y * 4.0;
                    Hit ah = first_hit_in(ar, 0xFF);
                    if (ah.found) {
                        // the level by the level only in contact: the bake already holds its larger occlusion
                        bool hit_dynamic = (draw_info[ah.geometry].w & KRT_DYNAMIC) != 0;
                        occlusion += saturate(1.0 - ah.t / (dynamic || hit_dynamic ? fx.y : fx.y * 0.15));
                        if (fx.z > 0.0) {
                            float3 sum = shade(ah.geometry, ah.primitive, ah.bary, ah.t), thr = 1.0;
                            Hit cur = ah;
                            RayDesc cr2 = ar;
                            [loop] for (uint b = 1; b < KRT_GI_BOUNCES; b++) {
                                float3 a2, b2, c2;
                                triangle_at(cur.geometry, cur.primitive, a2, b2, c2);
                                float3 n2 = normalize(cross(b2 - a2, c2 - a2));
                                if (dot(n2, cr2.Direction) > 0.0) n2 = -n2;
                                thr *= albedo_at(cur.geometry, cur.primitive, cur.bary, cur.t);
                                cr2.Origin = cr2.Origin + cr2.Direction * cur.t + n2 * 0.02;
                                cr2.Direction = cosine_dir(n2, float2(hash(px, f * 7u + b * 2u + k * 13u),
                                                                      hash(px, f * 7u + b * 2u + 1u + k * 13u)));
                                cr2.TMin = 0.0; cr2.TMax = fx.y * 4.0;
                                cur = first_hit_in(cr2, 0xFF);
                                if (!cur.found) break;
                                sum += thr * shade(cur.geometry, cur.primitive, cur.bary, cur.t);
                            }
                            bounce += sum; seen += 1.0;
                        }
                    }
                }
            }
            float ao = 1.0 - fx.x * occlusion * 0.5;
            float3 tint = 1.0;
            if (seen > 0.0 && fx.z > 0.0) {
                float3 e = bounce / seen;
                float le = luminance(e);
                if (dynamic) tint = lerp(1.0, e / max(le, 0.02), saturate(fx.z * seen * 0.5));
                else if (le > 0.005) {
                    // the bake holds the level's brightness: only the colour of what is around, at the same luminance
                    float3 c = lerp(1.0, clamp(e / le, 0.5, 1.8), saturate(fx.z * seen * 0.25));
                    tint = c / max(luminance(c), 0.05);
                }
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
