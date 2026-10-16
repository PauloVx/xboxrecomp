// Shared by the ray tracing passes: the frame's camera and the scene the rays see.
RaytracingAccelerationStructure scene : register(t0);
ByteAddressBuffer pos : register(t1);
ByteAddressBuffer uv : register(t2);                 // uv0.xy, uv1.xy a vertex
ByteAddressBuffer col : register(t3);                // RGBA8 a vertex
StructuredBuffer<uint4> draw_info : register(t4);    // by geometry index: first vertex, tex0, tex1, flags (KRT_*)
Texture2D<float4> textures[4096] : register(t0, space1);
SamplerState wrap : register(s0);

cbuffer Frame : register(b0)
{
    float4 inv0, inv1, inv2;    // world position from clip (x, y, w): the camera matrix's inverse, offsets in .w
    float4 c58, c59;            // the viewport: window = clip / w * c58 + c59
    float4 size;                // output width, height; guest window width, height
    float4 vp0, vp1, vp3;       // the camera matrix's x, y and w rows
    float4 zp;                  // its z row as zp.x * (w row) + zp.y; the depth buffer's largest value (zp.z); the
                                // diagnostic view (zp.w: 1 normals, 2 shaded, 3 calibration)
    float4 misc;                // the frame's number
};

// Brightness factors for shade (krt_calibrate.hlsl): one a texture index, then one a kind of lighting.
RWByteAddressBuffer factors : register(u2);
static const uint KRT_FACTOR_KIND = 4096u, KRT_FACTOR_KEYS = 4096u + 3u;

static const uint KRT_TEX0 = 1u, KRT_TEX1 = 2u, KRT_TEX0_TEXELS = 4u, KRT_TEX1_TEXELS = 8u, KRT_ALPHA_TEST = 16u;

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

float4 unpack_rgba8(uint c) { return float4(c & 255u, (c >> 8) & 255u, (c >> 16) & 255u, c >> 24); }

// The hit triangle's corners.
void triangle_at(uint geometry, uint primitive, out float3 a, out float3 b, out float3 c)
{
    uint v = (draw_info[geometry].x + primitive * 3) * 12;
    a = asfloat(pos.Load3(v));
    b = asfloat(pos.Load3(v + 12));
    c = asfloat(pos.Load3(v + 24));
}

// A point on the hit triangle: its texture coordinates (uv0, uv1) and colour.
void attributes_at(uint geometry, uint primitive, float2 bary, out float4 uvs, out float4 colour)
{
    uint v = draw_info[geometry].x + primitive * 3;
    float3 w = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    uvs = w.x * asfloat(uv.Load4(v * 16)) + w.y * asfloat(uv.Load4(v * 16 + 16)) + w.z * asfloat(uv.Load4(v * 16 + 32));
    colour = w.x * unpack_rgba8(col.Load(v * 4)) + w.y * unpack_rgba8(col.Load(v * 4 + 4)) +
             w.z * unpack_rgba8(col.Load(v * 4 + 8));
    colour /= 255.0;
}

float4 sample_stage(uint4 info, uint stage, float2 coords, float lod)
{
    Texture2D<float4> t = textures[NonUniformResourceIndex(stage == 0 ? info.y : info.z)];
    if (info.w & (stage == 0 ? KRT_TEX0_TEXELS : KRT_TEX1_TEXELS)) {
        uint w, h;
        t.GetDimensions(w, h);
        coords /= float2(w, h);
    }
    return t.SampleLevel(wrap, coords, lod);
}

// The colour of a hit from its textures: the base texture, lit by the lightmap -- or, for a draw without one, by its
// vertex colour (the lightmapped programs leave that colour unset). Distant hits read coarser mips. kind: what lit
// it (0 a lightmap, 1 vertex colour, 2 neither).
float3 shade_raw(uint geometry, uint primitive, float2 bary, float t, out uint kind)
{
    uint4 info = draw_info[geometry];
    float4 uvs, colour;
    attributes_at(geometry, primitive, bary, uvs, colour);
    float lod = clamp(log2(max(t, 1.0) / 6.0), 0.0, 6.0);
    float3 albedo = (info.w & KRT_TEX0) ? sample_stage(info, 0, uvs.xy, lod).rgb : 1.0;
    float3 light = 1.0;
    kind = 2u;
    if (info.w & KRT_TEX1) { light = sample_stage(info, 1, uvs.zw, lod).rgb; kind = 0u; }
    else if (dot(colour.rgb, 1.0) > 0.003) { light = colour.rgb; kind = 1u; }
    return albedo * light;
}

// The colour of a hit as the game draws it: shade_raw, at the brightness the picture shows that texture -- or that
// kind of lighting, until the texture has been seen on screen.
float3 shade(uint geometry, uint primitive, float2 bary, float t)
{
    uint kind;
    float3 c = shade_raw(geometry, primitive, bary, t, kind);
    uint4 info = draw_info[geometry];
    float f = (info.w & KRT_TEX0) ? asfloat(factors.Load(info.y * 4)) : 0.0;
    if (!(f > 0.0)) f = asfloat(factors.Load((KRT_FACTOR_KIND + kind) * 4));
    return c * (f > 0.0 ? f : 2.0);
}
// Whether a cut-out (alpha-tested) triangle is there at that point, by the draw's own test.
bool alpha_passes(uint geometry, uint primitive, float2 bary)
{
    uint4 info = draw_info[geometry];
    if (!(info.w & KRT_TEX0)) return true;
    float4 uvs, colour;
    attributes_at(geometry, primitive, bary, uvs, colour);
    float a = sample_stage(info, 0, uvs.xy, 0.0).a * 255.0, ref = (float)((info.w >> 16) & 255u);
    switch ((info.w >> 8) & 7u) {
    case 0: return false;
    case 1: return a < ref;
    case 2: return abs(a - ref) < 0.5;
    case 3: return a <= ref;
    case 4: return a > ref;
    case 5: return abs(a - ref) >= 0.5;
    case 6: return a >= ref;
    default: return true;
    }
}

struct Hit { bool found; float t; uint geometry, primitive; float2 bary; };

// The nearest surface along a ray, cut-out surfaces tested as the game tests them.
Hit first_hit(RayDesc r)
{
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alpha_passes(q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    Hit h;
    h.found = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    h.t = h.found ? q.CommittedRayT() : 0.0;
    h.geometry = h.found ? q.CommittedGeometryIndex() : 0;
    h.primitive = h.found ? q.CommittedPrimitiveIndex() : 0;
    h.bary = h.found ? q.CommittedTriangleBarycentrics() : 0.0;
    return h;
}

// Whether anything is between a point and a direction's end (shadow rays): the first hit ends the search.
bool occluded(RayDesc r)
{
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, r);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alpha_passes(q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
}
