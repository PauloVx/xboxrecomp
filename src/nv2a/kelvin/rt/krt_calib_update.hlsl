// The brightness factors from krt_calibrate.hlsl's sums: each key with enough samples moves its factor a step towards
// what this frame measured, and the sums start again.
#include "krt_common.hlsli"

RWByteAddressBuffer sums : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= KRT_FACTOR_KEYS) return;
    uint2 s = sums.Load2(i * 8);
    if (s.y >= 2048) {                       // shaded brightness summing to at least 2: a few dozen pixels
        float measured = clamp((float)s.x / (float)s.y, 0.125, 8.0);
        float old = asfloat(factors.Load(i * 4));
        factors.Store(i * 4, asuint(old > 0.0 ? lerp(old, measured, 0.1) : measured));
    }
    sums.Store2(i * 8, uint2(0, 0));
}
