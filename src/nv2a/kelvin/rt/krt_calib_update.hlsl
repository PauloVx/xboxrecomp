// The colour factors from krt_calibrate.hlsl's sums: each key with enough samples moves its factor a step towards
// what this frame measured, and the sums start again. A channel the shading has little of takes the brightness ratio.
// The casts are strong: a draw's second texture is not always a lightmap (a detail or bump map multiplies in its own
// colour), so the channels are free.
#include "krt_common.hlsli"

RWByteAddressBuffer sums : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= KRT_FACTOR_KEYS) return;
    float3 p = (float3)sums.Load3(i * 32), s = (float3)sums.Load3(i * 32 + 16);
    const float3 Y = float3(0.299, 0.587, 0.114);
    float ls = dot(s, Y);
    if (ls >= 2048.0) {                      // shaded brightness summing to at least 2: a few dozen pixels
        float rl = clamp(dot(p, Y) / ls, 0.125, 8.0);
        float3 measured = select(s > 0.1 * ls, p / max(s, 1.0), rl);
        measured = clamp(measured, 0.125, 8.0);
        float3 old = asfloat(factors.Load3(i * 16));
        factors.Store3(i * 16, asuint(old.g > 0.0 ? lerp(old, measured, 0.1) : measured));
    }
    sums.Store4(i * 32, uint4(0, 0, 0, 0));
    sums.Store4(i * 32 + 16, uint4(0, 0, 0, 0));
}
