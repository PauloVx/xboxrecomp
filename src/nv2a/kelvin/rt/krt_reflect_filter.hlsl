// The reflections' one ray a pixel (krt_reflect.hlsl), made smooth: each pixel averages the reflective pixels around
// it whose colour is close (a slightly rough surface, edges kept), and the weapon's pixels, which move with the camera,
// average over frames too. Output for the screen-space pass: rgb the colour, a = 2 traced, 0 not.
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
    float lm = luminance(me.rgb);
    float3 sum = 0.0;
    float total = 0.0;
    [unroll] for (int y = -2; y <= 2; y++)
        [unroll] for (int x = -2; x <= 2; x++) {
            int2 q = int2(id.xy) + int2(x, y);
            if (any(q < 0) || any(q >= int2(size.xy))) continue;
            float4 o = raw[q];
            if (abs(o.a - me.a) > 0.5) continue;               // the same kind of surface only
            float wt = exp(-0.25 * (x * x + y * y)) * exp(-abs(luminance(o.rgb) - lm) * 4.0);
            sum += o.rgb * wt;
            total += wt;
        }
    float3 c = sum / max(total, 1e-4);
    if (me.a > 2.5) {
        float4 old = history[id.xy];
        if (old.a > 0.5) c = lerp(c, old.rgb, 0.9);
        history[id.xy] = float4(c, 1.0);
    } else {
        history[id.xy] = 0.0;
    }
    output[id.xy] = float4(c, 2.0);
}