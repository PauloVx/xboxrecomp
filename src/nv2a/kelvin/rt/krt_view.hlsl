// RECOMP_KGPU_RT_VIEW (diagnostic): one ray a pixel from the camera. 1: the normal of what it hits as a colour -- if
// that matches the rasterised picture, the captured scene and its acceleration structure are right. 2: what it hits
// shaded as the reflections shade hits off screen, to compare with the picture.
// 3: calibration -- for each pixel, the picture's brightness over the shaded hit's, the two brightnesses, and what
// lit the hit (1 a lightmap, 2 vertex colour, 3 neither), for kelvin_gpu_rt.inc to sum up.
#include "krt_common.hlsli"

Texture2D<float4> picture : register(t6);   // the frame so far
RWTexture2D<float4> output : register(u0);

float luminance(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= (uint2)size.xy)) return;
    RayDesc r = camera_ray(id.xy);
    Hit h = first_hit(r);
    float4 c = float4(0.0, 0.0, 0.0, 1.0);
    if (zp.w > 2.5) {
        c = 0.0;
        if (h.found) {
            uint4 info = draw_info[h.geometry];
            float4 uvs, colour;
            attributes_at(h.geometry, h.primitive, h.bary, uvs, colour);
            float ls = luminance(shade(h.geometry, h.primitive, h.bary, h.t)), lp = luminance(picture.Load(int3(id.xy, 0)).rgb);
            float kind = (info.w & KRT_TEX1) ? 1.0 : dot(colour.rgb, 1.0) > 0.003 ? 2.0 : 3.0;
            c = float4(lp / max(ls, 0.002), lp, ls, kind);
        }
    } else if (h.found && zp.w > 1.5) {
        c = float4(shade(h.geometry, h.primitive, h.bary, h.t), 1.0);
    } else if (h.found) {
        float3 a, b, cc;
        triangle_at(h.geometry, h.primitive, a, b, cc);
        float3 n = normalize(cross(b - a, cc - a));
        if (dot(n, r.Direction) > 0.0) n = -n;
        c = float4(n * 0.5 + 0.5, 1.0);
    }
    output[id.xy] = c;
}
