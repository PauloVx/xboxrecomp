// The colour the game draws what the rays hit in, learnt from the picture (see shade in krt_common.hlsli).
//
// shade_raw knows a surface's textures but not the register combiner program that lights them, so its colour and
// brightness can be off by the program's own scales, sums and tints (a lightmap's warm cast the program takes out). Every frame, one pixel in 64 (a different one each frame) is
// traced from the camera and the picture's brightness there set against shade_raw's: the sums go to the hit's base
// texture and to its kind of lighting (krt_calib_update.hlsl makes factors of them). Only near, mid-bright pixels
// count: far ones are fogged, black and white ones say nothing about a scale (dark ones do, noisily: BLACK is dark).
#include "krt_common.hlsli"

Texture2D<float4> picture : register(t6);   // the frame so far
RWByteAddressBuffer sums : register(u1);    // eight uints a key: picture rgb, 0, shaded rgb, 0, in 1/1024 units

float luminance(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint f = (uint)misc.x;
    uint2 px = id.xy * 8 + uint2(f % 8, (f / 8) % 8);
    if (any(px >= (uint2)size.xy)) return;
    Hit h = first_hit(camera_ray(px));
    if (!h.found || h.t > 60.0) return;
    float3 p = saturate(picture.Load(int3(px, 0)).rgb);
    float lp = luminance(p);
    if (lp < 0.01 || lp > 0.92 || max(p.r, max(p.g, p.b)) > 0.98) return;   // a clipped channel says nothing either
    uint kind;
    float3 s = shade_raw(h.geometry, h.primitive, h.bary, h.t, kind);
    if (luminance(s) < 0.01) return;
    uint4 info = draw_info[h.geometry];
    uint3 a = (uint3)(p * 1024.0), b = (uint3)(min(s, 4.0) * 1024.0);
    uint key[2] = { (info.w & KRT_TEX0) ? info.y : 0xFFFFFFFFu, KRT_FACTOR_KIND + kind };
    [unroll] for (uint k = 0; k < 2; k++) {
        if (key[k] == 0xFFFFFFFFu) continue;
        uint at = key[k] * 32;
        sums.InterlockedAdd(at, a.r); sums.InterlockedAdd(at + 4, a.g); sums.InterlockedAdd(at + 8, a.b);
        sums.InterlockedAdd(at + 16, b.r); sums.InterlockedAdd(at + 20, b.g); sums.InterlockedAdd(at + 24, b.b);
    }
}
