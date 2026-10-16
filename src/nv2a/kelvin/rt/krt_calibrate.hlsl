// How bright the game draws what the rays hit, learnt from the picture (see shade in krt_common.hlsli).
//
// shade_raw knows a surface's textures but not the register combiner program that lights them, so its brightness
// can be off by the program's own scales and sums. Every frame, one pixel in 64 (a different one each frame) is
// traced from the camera and the picture's brightness there set against shade_raw's: the sums go to the hit's base
// texture and to its kind of lighting (krt_calib_update.hlsl makes factors of them). Only near, mid-bright pixels
// count: far ones are fogged, black and white ones say nothing about a scale.
#include "krt_common.hlsli"

Texture2D<float4> picture : register(t6);   // the frame so far
RWByteAddressBuffer sums : register(u1);    // two uints a key: picture and shaded brightness, 1/1024 units

float luminance(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint f = (uint)misc.x;
    uint2 px = id.xy * 8 + uint2(f % 8, (f / 8) % 8);
    if (any(px >= (uint2)size.xy)) return;
    Hit h = first_hit(camera_ray(px));
    if (!h.found || h.t > 60.0) return;
    float lp = luminance(picture.Load(int3(px, 0)).rgb);
    if (lp < 0.03 || lp > 0.92) return;
    uint kind;
    float ls = luminance(shade_raw(h.geometry, h.primitive, h.bary, h.t, kind));
    if (ls < 0.01) return;
    uint4 info = draw_info[h.geometry];
    uint a = (uint)(lp * 1024.0), b = (uint)(ls * 1024.0);
    if (info.w & KRT_TEX0) {
        sums.InterlockedAdd(info.y * 8, a);
        sums.InterlockedAdd(info.y * 8 + 4, b);
    }
    sums.InterlockedAdd((KRT_FACTOR_KIND + kind) * 8, a);
    sums.InterlockedAdd((KRT_FACTOR_KIND + kind) * 8 + 4, b);
}
