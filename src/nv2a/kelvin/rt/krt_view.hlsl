// RECOMP_KGPU_RT_VIEW=1 (diagnostic): one ray a pixel from the camera, the normal of what it hits as a colour.
// If this matches the rasterised picture, the captured scene and its acceleration structure are right.
#include "krt_common.hlsli"

RWTexture2D<float4> output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= (uint2)size.xy)) return;
    RayDesc r = camera_ray(id.xy);
    RayQuery<RAY_FLAG_CULL_NON_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, r);
    q.Proceed();
    float4 c = float4(0.0, 0.0, 0.0, 1.0);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float3 a, b, cc;
        triangle_at(q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex(), a, b, cc);
        float3 n = normalize(cross(b - a, cc - a));
        if (dot(n, r.Direction) > 0.0) n = -n;
        c = float4(n * 0.5 + 0.5, 1.0);
    }
    output[id.xy] = c;
}
