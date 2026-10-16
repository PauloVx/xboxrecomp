// The reflections' one ray a pixel (krt_reflect.hlsl), made smooth: each pixel averages the reflective pixels of its
// kind around it whose colour is close (a slightly rough surface, edges kept). The weapon is rough, flat-faceted metal
// that bobs against the camera: it averages over a much wider neighbourhood, and a little over frames.
// Output for the screen-space pass: rgb the colour, a = 2 traced on the scene, 3 on the weapon, 0 not.
#include "krt_common.hlsli"

RWTexture2D<float4> output : register(u0);
RWTexture2D<float4> raw : register(u6);
RWTexture2D<float4> history : register(u7);  // the weapon's: rgb its last colour, a 1 if it was the weapon's

float luminance(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= (uint2)size.xy)) return;
    float4 me = raw[id.xy];
    if (me.a < 1.5) { output[id.xy] = 0.0; history[id.xy] = 0.0; return; }
    bool weapon = me.a > 2.5;
    int stride = weapon ? 2 : 1, reach = weapon ? 4 : 2;
    float spread = weapon ? 0.06 : 0.25, edge = weapon ? 1.5 : 4.0;
    float lm = luminance(me.rgb);
    float3 sum = 0.0;
    float total = 0.0;
    for (int y = -reach; y <= reach; y++)
        for (int x = -reach; x <= reach; x++) {
            int2 q = int2(id.xy) + int2(x, y) * stride;
            if (any(q < 0) || any(q >= int2(size.xy))) continue;
            float4 o = raw[q];
            if (abs(o.a - me.a) > 0.5) continue;               // the same kind of surface only
            float wt = exp(-spread * (x * x + y * y)) * exp(-abs(luminance(o.rgb) - lm) * edge);
            sum += o.rgb * wt;
            total += wt;
        }
    float3 c = sum / max(total, 1e-4);
    if (weapon) {
        float4 old = history[id.xy];
        if (old.a > 0.5) c = lerp(c, old.rgb, 0.6);
        history[id.xy] = float4(c, 1.0);
    } else {
        history[id.xy] = 0.0;
    }
    output[id.xy] = float4(c, me.a);
}
