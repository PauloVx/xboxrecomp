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
    float4 vp0, vp1, vp3;       // the camera matrix's x, y and w rows
    float4 zp;                  // its z row as zp.x * (w row) + zp.y; the depth buffer's largest value (zp.z)
};

// The distance along the view direction (clip w) of a world position.
float view_w(float3 p) { return dot(vp3, float4(p, 1.0)); }

// The output pixel a world position is drawn at, if it is in front of the camera and on screen.
bool to_pixel(float3 p, out float2 px)
{
    float4 h = float4(p, 1.0);
    float w = dot(vp3, h);
    px = 0.0;
    if (w <= 0.01) return false;
    float2 pp = float2(dot(vp0, h), dot(vp1, h)) / w * c58.xy + c59.xy;
    px = pp / size.zw * size.xy;
    return all(px >= 0.0) && all(px < size.xy);
}

// A view-space direction (x right, y up, z forward, as the screen-space passes use) in world space: the camera's axes
// are its matrix's x, y and w rows, scaled by the projection.
float3 view_to_world(float3 v)
{
    return normalize(v.x * normalize(vp0.xyz) + v.y * normalize(vp1.xyz) + v.z * normalize(vp3.xyz));
}

// A depth buffer value (0..1, as the rasteriser wrote it) as a distance along the view direction.
float view_w_from_depth(float d)
{
    float zw = (d * zp.z - c59.z) / c58.z;      // clip z / w
    return zp.y / (zw - zp.x);
}

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
