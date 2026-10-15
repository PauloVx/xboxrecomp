// The captured triangles (two texels a vertex, see kelvin_gpu_rt.inc) into the buffers the acceleration structure
// and the shading read: positions (3 floats), texture coordinates (uv0, uv1) and colours (RGBA8).
Texture2D<float4> geo : register(t0);
RWByteAddressBuffer pos : register(u0);
RWByteAddressBuffer uv : register(u1);
RWByteAddressBuffer col : register(u2);
cbuffer Unpack : register(b0) { uint n; uint width; uint2 pad; };

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= n) return;
    uint t = i * 2;
    uint2 at = uint2(t % width, t / width);
    float4 p = geo[at], u = geo[at + uint2(1, 0)];
    pos.Store3(i * 12, asuint(p.xyz));
    uv.Store4(i * 16, asuint(u));
    col.Store(i * 4, asuint(p.w));
}
