// Shared by the ray tracing passes: the frame's camera and the scene the rays see.
RaytracingAccelerationStructure scene : register(t0);
ByteAddressBuffer pos : register(t1);
ByteAddressBuffer uv : register(t2);
ByteAddressBuffer col : register(t3);
StructuredBuffer<uint> draw_first : register(t4);    // a captured draw's first vertex, by geometry index

cbuffer Frame : register(b0)
{
    float4 inv0, inv1, inv2;    // world position from clip (x, y, w): the camera matrix's inverse, offsets in .w
    float4 c58, c59;            // the viewport: window = clip / w * c58 + c59
    float4 size;                // output width, height; guest window width, height
};

// The world position of window position pp (guest pixels) at distance w along the view direction; w = 0 is the eye.
float3 world_at(float2 pp, float w)
{
    float3 r = float3((pp - c59.xy) / c58.xy * w, w) - float3(inv0.w, inv1.w, inv2.w);
    return float3(dot(inv0.xyz, r), dot(inv1.xyz, r), dot(inv2.xyz, r));
}

RayDesc camera_ray(uint2 px)
{
    float2 pp = (px + 0.5) / size.xy * size.zw;
    float3 o = world_at(pp, 0.0), d = world_at(pp, 1.0) - o;
    RayDesc r;
    r.Origin = o;
    r.Direction = normalize(d);
    r.TMin = 0.0;
    r.TMax = 1.0e6;
    return r;
}

// The hit triangle's corners.
void triangle_at(uint geometry, uint primitive, out float3 a, out float3 b, out float3 c)
{
    uint v = (draw_first[geometry] + primitive * 3) * 12;
    a = asfloat(pos.Load3(v));
    b = asfloat(pos.Load3(v + 12));
    c = asfloat(pos.Load3(v + 24));
}
