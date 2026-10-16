// The lighting pass's result for this frame (krt_light.hlsl, its history), smoothed: each pixel averages the 5x5 around
// it that are about as far away (no bleeding across a silhouette). The composite scales it up to the picture.
#include "krt_common.hlsli"

RWTexture2D<float4> light_out : register(u3);
RWTexture2D<float4> history_a : register(u4);
RWTexture2D<float4> history_b : register(u5);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 half_size = ((uint2)size.xy + 1u) / 2u;
    if (any(id.xy >= half_size)) return;
    bool odd = (((uint)misc.x) & 1u) != 0;
    float4 me = odd ? history_a[id.xy] : history_b[id.xy];
    if (me.a <= 0.0) { light_out[id.xy] = float4(me.rgb, 1.0); return; }
    float3 sum = 0.0;
    float total = 0.0;
    [unroll] for (int y = -2; y <= 2; y++)
        [unroll] for (int x = -2; x <= 2; x++) {
            int2 q = int2(id.xy) + int2(x, y);
            if (any(q < 0) || any(q >= int2(half_size))) continue;
            float4 o = odd ? history_a[q] : history_b[q];
            if (o.a <= 0.0 || abs(o.a - me.a) > 0.03 * me.a + 0.05) continue;
            float wt = exp(-0.3 * (x * x + y * y));
            sum += o.rgb * wt;
            total += wt;
        }
    light_out[id.xy] = float4(sum / max(total, 1e-4), 1.0);
}