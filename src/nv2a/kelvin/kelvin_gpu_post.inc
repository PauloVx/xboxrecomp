/* Picture effects of the GPU renderer, included by kelvin_gpu.c after the surface and pass helpers.
 *
 * What the player chooses in the Video Settings page (pc_video.c) that the renderer does itself:
 *
 *   - AO runs on opaque scene colour just before the first blended draw, so transparent smoke is not darkened with its
 *     background. Depth of field, FXAA and sharpening still run just before the HUD (or at flip if there is no HUD).
 *     Each effect copies the colour target and draws back into it.
 *   - super-sampling (SSAA) rasterises at twice the width and height (the raster size, G.host_*); at the flip a four-tap
 *     tent resolve brings the frame down to the internal resolution (G.out_*), which is what is presented.
 *   - anisotropic filtering is a sampler setting (get_sampler reads s_af).
 *
 * Everything is on the thread that owns the surfaces and the context, between draws; the passes leave the context's
 * bindings to be set again (kg_cache_invalidate). */

#include "kelvin_post_order.h"
#include <wincodec.h>

static const char kgpu_hlsl_post[] =
"cbuffer PostCB : register(b0) {\n"
"    float4 p0;   // x,y: 1 / scene size; z,w: scene size in pixels\n"
"    float4 p1;   // DOF: radius/taps/smoothing/depth scale; AO: radius/directions/steps/strength\n"
"    float4 p2;   // DOF: focus zone/near blur/sharpening; AO: bias/range/detail blend\n"
"    float4 p3;   // Resolve: target size/reciprocal; AO: real camera half-frustum in x,y\n"
"};\n"
"Texture2D<float4> SceneTex : register(t0);\n"
"Texture2D<float>  CocTex   : register(t1);\n"
"Texture2D<float>  DepthTex : register(t2);\n"
"Texture2D<float>  FocusTex : register(t3);\n"
"SamplerState LinSmp : register(s0);\n"
"SamplerState PtSmp  : register(s1);\n"
"\n"
"// Focus on a surface represented by the central aperture. A mean between foreground foliage\n"
"// and the aimed background describes neither surface and pumps the whole picture's blur.\n"
"// The median rejects minority occluders; a foreground subject filling the aperture still wins.\n"
"// Cleared/invalid depth is not a subject, and a sky-only aperture preserves the last focus.\n"
"float focus_ps(float4 p : SV_Position) : SV_Target {\n"
"    float prev = FocusTex.Load(int3(0, 0, 0));\n"
"    float samples[25];\n"
"    [unroll] for (int k = 0; k < 25; ++k) samples[k] = 1.0;\n"
"    int n = 0;\n"
"    [unroll] for (int j = -2; j <= 2; j++) {\n"
"        [unroll] for (int i = -2; i <= 2; i++) {\n"
"            float2 uv = float2(0.5 + i * 0.010 * p0.w / p0.z, 0.5 + j * 0.010);\n"
"            float d = DepthTex.SampleLevel(PtSmp, uv, 0);\n"
"            if (d >= 0.0 && d < 0.99999) {\n"
"                int k = n;\n"
"                [loop] while (k > 0) {\n"
"                    if (samples[k - 1] <= d) break;\n"
"                    samples[k] = samples[k - 1];\n"
"                    --k;\n"
"                }\n"
"                samples[k] = d;\n"
"                ++n;\n"
"            }\n"
"        }\n"
"    }\n"
"    float target = n > 0 ? samples[n / 2] : prev;\n"
"    if (!(prev >= 0.0 && prev <= 1.0)) return n > 0 ? target : 1.0;\n"
"    return lerp(prev, target, p1.z);\n"
"}\n"
"\n"
"// How much each pixel is out of focus, 0..1 of the largest blur: only what is behind the focus blurs, so the weapon and\n"
"// hands in front of it stay sharp.\n"
"float coc_ps(float4 p : SV_Position) : SV_Target {\n"
"    float d = DepthTex.Load(int3((int2)p.xy, 0));\n"
"    float f = FocusTex.Load(int3(0, 0, 0));\n"
"    float dz = d - f;\n"
"    float far = saturate((dz - p2.x) * p1.w);\n"
"    float nr = saturate((-dz - p2.x) * p1.w) * p2.y;\n"
"    return max(far, nr);\n"
"}\n"
"\n"
"float4 dof_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    float4 center = SceneTex.SampleLevel(LinSmp, uv, 0);\n"
"    float R = CocTex.SampleLevel(PtSmp, uv, 0) * p1.x;\n"
"    if (R < 0.6) return center;\n"
"    float rot = 6.2831853 * frac(52.9829189 * frac(dot(p.xy, float2(0.06711056, 0.00583715))));\n"
"    float3 acc = center.rgb;\n"
"    float wsum = 1.0;\n"
"    int taps = (int)p1.y;\n"
"    [loop] for (int i = 0; i < taps; i++) {\n"
"        float r = sqrt((i + 0.5) / taps) * R;\n"
"        float a = i * 2.39996323 + rot;\n"
"        float2 o = float2(cos(a), sin(a)) * r;\n"
"        float2 suv = uv + o * p0.xy;\n"
"        float sc = CocTex.SampleLevel(PtSmp, suv, 0) * p1.x;\n"
"        float w = saturate(sc - r + 1.0);\n"
"        acc += SceneTex.SampleLevel(LinSmp, suv, 0).rgb * w;\n"
"        wsum += w;\n"
"    }\n"
"    return float4(acc / wsum, center.a);\n"
"}\n"
"\n"
"// FXAA: find the edges by their luma contrast, then blend along them. Pixels with little local contrast are left\n"
"// alone, so fine texture detail is not softened.\n"
"float luma(float3 c) { return dot(c, float3(0.299, 0.587, 0.114)); }\n"
"float4 fxaa_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    float2 t = p0.xy;\n"
"    float4 m4 = SceneTex.SampleLevel(LinSmp, uv, 0);\n"
"    float3 rgbNW = SceneTex.SampleLevel(LinSmp, uv + float2(-1, -1) * t, 0).rgb;\n"
"    float3 rgbNE = SceneTex.SampleLevel(LinSmp, uv + float2( 1, -1) * t, 0).rgb;\n"
"    float3 rgbSW = SceneTex.SampleLevel(LinSmp, uv + float2(-1,  1) * t, 0).rgb;\n"
"    float3 rgbSE = SceneTex.SampleLevel(LinSmp, uv + float2( 1,  1) * t, 0).rgb;\n"
"    float lumaNW = luma(rgbNW), lumaNE = luma(rgbNE), lumaSW = luma(rgbSW), lumaSE = luma(rgbSE), lumaM = luma(m4.rgb);\n"
"    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));\n"
"    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));\n"
"    if (lumaMax - lumaMin < max(0.0312, lumaMax * 0.125)) return m4;\n"
"    float2 dir;\n"
"    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));\n"
"    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));\n"
"    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * 0.125), 1.0 / 128.0);\n"
"    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);\n"
"    dir = clamp(dir * rcpDirMin, -8.0, 8.0) * t;\n"
"    float3 rgbA = 0.5 * (SceneTex.SampleLevel(LinSmp, uv + dir * (1.0 / 3.0 - 0.5), 0).rgb + SceneTex.SampleLevel(LinSmp, uv + dir * (2.0 / 3.0 - 0.5), 0).rgb);\n"
"    float3 rgbB = rgbA * 0.5 + 0.25 * (SceneTex.SampleLevel(LinSmp, uv + dir * -0.5, 0).rgb + SceneTex.SampleLevel(LinSmp, uv + dir * 0.5, 0).rgb);\n"
"    float lumaB = luma(rgbB);\n"
"    return float4((lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB, m4.a);\n"
"}\n"
"\n"
"// Contrast-adaptive sharpening (the AMD CAS idea): sharpen less where the neighbourhood is already contrasty.\n"
"float4 sharpen_ps(float4 p : SV_Position) : SV_Target {\n"
"    int2 q = (int2)p.xy;\n"
"    float4 c = SceneTex.Load(int3(q, 0));\n"
"    float3 a = SceneTex.Load(int3(q + int2(0, -1), 0)).rgb;\n"
"    float3 b = SceneTex.Load(int3(q + int2(-1, 0), 0)).rgb;\n"
"    float3 d = SceneTex.Load(int3(q + int2(1, 0), 0)).rgb;\n"
"    float3 e = SceneTex.Load(int3(q + int2(0, 1), 0)).rgb;\n"
"    float3 mn = min(c.rgb, min(min(a, b), min(d, e)));\n"
"    float3 mx = max(c.rgb, max(max(a, b), max(d, e)));\n"
"    float3 amp = sqrt(saturate(min(mn, 1.0 - mx) / max(mx, 0.0001)));\n"
"    float3 w = amp * lerp(-0.125, -0.2, p2.z);\n"
"    return float4(saturate((a * w + b * w + d * w + e * w + c.rgb) / (1.0 + 4.0 * w)), c.a);\n"
"}\n"
"\n"
"// The super-sampling resolve: four bilinear taps at +-0.75 source texels are a 4x4 tent over the 2x2 block.\n"
"float4 ssaa_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p3.zw;\n"
"    float2 o = 0.75 * p0.xy;\n"
"    float4 s = SceneTex.SampleLevel(LinSmp, uv + float2(-o.x, -o.y), 0) + SceneTex.SampleLevel(LinSmp, uv + float2(o.x, -o.y), 0) +\n"
"               SceneTex.SampleLevel(LinSmp, uv + float2(-o.x,  o.y), 0) + SceneTex.SampleLevel(LinSmp, uv + float2(o.x,  o.y), 0);\n"
"    return s * 0.25;\n"
"}\n"
"// AO uses p1=(pixel radius, directions/taps, steps, strength), p2=(bias, depth range, detail blend, 0),\n"
"// and p3.xy=the real camera half-frustum. All methods share depth/normal reconstruction, not their kernels.\n"
"float ao_hash(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }\n"
"float ao_depth(float2 uv) { return DepthTex.SampleLevel(PtSmp, uv, 0); }\n"
"// Perspective depth is 1-near/z (infinite-far approximation); near is a common scale.\n"
"float ao_view_z(float d) { return min(4096.0, 1.0 / max(0.00010, 1.0 - d)); }\n"
"float3 ao_pos(float2 uv) {\n"
"    // Point-sampled depth belongs to this texel centre, not the unsnapped kernel coordinate.\n"
"    uv = (clamp(floor(uv * p0.zw), float2(0, 0), p0.zw - 1.0) + 0.5) * p0.xy;\n"
"    float z = ao_view_z(ao_depth(uv));\n"
"    float2 n = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);\n"
"    return float3(n * p3.xy * z, z);\n"
"}\n"
"float3 ao_normal(float2 uv, float3 c) {\n"
"    float3 l = ao_pos(uv - float2(p0.x, 0)), r = ao_pos(uv + float2(p0.x, 0));\n"
"    float3 u = ao_pos(uv - float2(0, p0.y)), d = ao_pos(uv + float2(0, p0.y));\n"
"    // Choose the derivative on the same surface at depth discontinuities.\n"
"    float3 dx = uv.x <= 0.5 * p0.x ? r - c : uv.x >= 1.0 - 0.5 * p0.x ? c - l :\n"
"                abs(r.z - c.z) < abs(l.z - c.z) ? r - c : c - l;\n"
"    float3 dy = uv.y <= 0.5 * p0.y ? c - d : uv.y >= 1.0 - 0.5 * p0.y ? u - c :\n"
"                abs(u.z - c.z) < abs(d.z - c.z) ? u - c : c - d;\n"
"    float3 n = cross(dx, dy);\n"
"    float n2 = dot(n, n);\n"
"    if (n2 < 1e-12) return float3(0, 0, -1);\n"
"    n *= rsqrt(n2);\n"
"    return dot(n, c) > 0 ? -n : n;\n"
"}\n"
"float ao_range(float3 c, float3 q) {\n"
"    return saturate(1.0 - length(q - c) / max(0.01, c.z * p2.y));\n"
"}\n"
"float ao_visibility_sample(float2 uv, float3 c, float3 n) {\n"
"    if (any(uv < 0) || any(uv > 1) || ao_depth(uv) >= 0.99999) return 0.0;\n"
"    float3 q = ao_pos(uv), v = q - c;\n"
"    float v2 = dot(v, v);\n"
"    if (v2 < 1e-12) return 0.0;\n"
"    return saturate(dot(n, v * rsqrt(v2)) - p2.x) * ao_range(c, q);\n"
"}\n"
"float ao_ssao_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    if (ao_depth(uv) >= 0.99999) return 1.0;\n"
"    float3 c = ao_pos(uv), n = ao_normal(uv, c);\n"
"    float rot = ao_hash(p.xy) * 6.2831853, occ = 0.0;\n"
"    int taps = (int)p1.y;\n"
"    // Independent radial samples, with no horizon search.\n"
"    [loop] for (int i = 0; i < taps; i++) {\n"
"        float a = i * 2.39996323 + rot;\n"
"        float r = sqrt((i + 0.5) / taps) * p1.x;\n"
"        occ += ao_visibility_sample(uv + float2(cos(a), sin(a)) * r * p0.xy, c, n);\n"
"    }\n"
"    return saturate(1.0 - p1.w * occ / taps);\n"
"}\n"
"float ao_hbao_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    if (ao_depth(uv) >= 0.99999) return 1.0;\n"
"    float3 c = ao_pos(uv), n = ao_normal(uv, c);\n"
"    float rot = ao_hash(p.xy) * 6.2831853, occ = 0.0;\n"
"    int dirs = (int)p1.y, steps = (int)p1.z;\n"
"    // Walk out along each ray and integrate its largest occluding horizon.\n"
"    [loop] for (int i = 0; i < dirs; i++) {\n"
"        float a = rot + i * 6.2831853 / dirs;\n"
"        float2 dir = float2(cos(a), sin(a));\n"
"        float horizon = 0.0;\n"
"        [loop] for (int j = 0; j < steps; j++) {\n"
"            float r = (j + 1.0) * p1.x / steps;\n"
"            horizon = max(horizon, ao_visibility_sample(uv + dir * r * p0.xy, c, n));\n"
"        }\n"
"        occ += horizon;\n"
"    }\n"
"    return saturate(1.0 - p1.w * occ / dirs);\n"
"}\n"
"float ao_hbaop_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    if (ao_depth(uv) >= 0.99999) return 1.0;\n"
"    float3 c = ao_pos(uv), n = ao_normal(uv, c);\n"
"    // Interleaved 4x4 rotation, nonlinear radial sampling, and a separate near-detail term.\n"
"    int2 cell = (int2)p.xy & 3;\n"
"    float rot = (cell.x + cell.y * 4 + 0.5) * (6.2831853 / 16.0), occ = 0.0;\n"
"    int dirs = (int)p1.y, steps = (int)p1.z;\n"
"    [loop] for (int i = 0; i < dirs; i++) {\n"
"        float a = rot + i * 6.2831853 / dirs;\n"
"        float2 dir = float2(cos(a), sin(a));\n"
"        float horizon = 0.0;\n"
"        [loop] for (int j = 0; j < steps; j++) {\n"
"            float r = sqrt((j + 0.5) / steps) * p1.x;\n"
"            horizon = max(horizon, ao_visibility_sample(uv + dir * r * p0.xy, c, n));\n"
"        }\n"
"        occ += horizon;\n"
"    }\n"
"    float detail = ao_visibility_sample(uv + float2(2, 0) * p0.xy, c, n)\n"
"                 + ao_visibility_sample(uv - float2(2, 0) * p0.xy, c, n)\n"
"                 + ao_visibility_sample(uv + float2(0, 2) * p0.xy, c, n)\n"
"                 + ao_visibility_sample(uv - float2(0, 2) * p0.xy, c, n);\n"
"    return saturate(1.0 - p1.w * lerp(occ / dirs, detail * 0.25, p2.z));\n"
"}\n"
"float ao_gtao_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    if (ao_depth(uv) >= 0.99999) return 1.0;\n"
"    float3 c = ao_pos(uv), n = ao_normal(uv, c);\n"
"    float rot = ao_hash(p.xy) * 3.14159265, occ = 0.0;\n"
"    int dirs = (int)p1.y, steps = (int)p1.z;\n"
"    // Approximate slice/horizon integration in both directions of each slice.\n"
"    [loop] for (int i = 0; i < dirs; i++) {\n"
"        float a = rot + i * 3.14159265 / dirs;\n"
"        float2 dir = float2(cos(a), sin(a));\n"
"        float hp = 0.0, hn = 0.0;\n"
"        [loop] for (int j = 0; j < steps; j++) {\n"
"            float r = (j + 0.75) * p1.x / steps;\n"
"            float2 offset = dir * r * p0.xy;\n"
"            hp = max(hp, ao_visibility_sample(uv + offset, c, n));\n"
"            hn = max(hn, ao_visibility_sample(uv - offset, c, n));\n"
"        }\n"
"        // Each horizon is sin(elevation); 1-cos(elevation) is its angular integral.\n"
"        occ += (2.0 - sqrt(saturate(1.0 - hp * hp)) - sqrt(saturate(1.0 - hn * hn))) * 0.5;\n"
"    }\n"
"    return saturate(1.0 - p1.w * occ / dirs);\n"
"}\n"
"float ao_blur(float4 p, float2 axis) {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    if (ao_depth(uv) >= 0.99999) return 1.0;\n"
"    float z = ao_view_z(ao_depth(uv)), sum = 0.0, wsum = 0.0;\n"
"    [unroll] for (int i = -2; i <= 2; i++) {\n"
"        float2 q = uv + axis * i * p0.xy;\n"
"        float dz = abs(ao_view_z(ao_depth(q)) - z);\n"
"        float w = exp(-0.5 * i * i) * exp(-dz / max(0.001, z * 0.02));\n"
"        sum += CocTex.SampleLevel(PtSmp, q, 0) * w;\n"
"        wsum += w;\n"
"    }\n"
"    return sum / max(wsum, 1e-6);\n"
"}\n"
"float ao_blur_h_ps(float4 p : SV_Position) : SV_Target { return ao_blur(p, float2(1, 0)); }\n"
"float ao_blur_v_ps(float4 p : SV_Position) : SV_Target { return ao_blur(p, float2(0, 1)); }\n"
"float4 ao_composite_ps(float4 p : SV_Position) : SV_Target {\n"
"    float2 uv = p.xy * p0.xy;\n"
"    float4 c = SceneTex.SampleLevel(PtSmp, uv, 0);\n"
"    return float4(c.rgb * CocTex.SampleLevel(PtSmp, uv, 0), c.a);\n"
"}\n"
"// Screen-space reflections (kelvin_gpu_ssr.inc): p1 = (steps, reach as a multiple of the pixel's depth, thickness as a\n"
"// fraction of depth, largest gain), p2 = (screen environment share for misses, its blur level, hit blur, smallest mip),\n"
"// p3.xy = the camera half-frustum; SsrW/SsrE/SsrG are what the env-mapped draws wrote.\n"
"Texture2D<float4> SsrW : register(t4);\n"
"Texture2D<float4> SsrE : register(t5);\n"
"Texture2D<float4> SsrG : register(t6);\n"
"Texture2D<float4> SsrBlur : register(t7);   // the scene with its mip chain: a soft environment from what is on screen\n"
"Texture2D<float4> RtRefl : register(t8);    // ray-traced (p3.z set): rgb a colour, a = 0 not traced, 1 no colour, 2 a hit\n"
"float2 ssr_uv(float3 q) { float2 n = q.xy / (q.z * p3.xy); return float2(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5); }\n"
"float ssr_scene_z(float2 uv) { return ao_view_z(DepthTex.SampleLevel(PtSmp, uv, 0)); }\n"
"float4 ssr_ps(float4 pos : SV_Position) : SV_Target {\n"
"    int3 px = int3(pos.xy, 0);\n"
"    float4 base = SceneTex.Load(px);\n"
"    float4 w = SsrW.Load(px);\n"
"    if (max(w.r, max(w.g, w.b)) < 0.002) return base;\n"
"    float4 g = SsrG.Load(px);\n"
"    float z0 = ao_view_z(g.x);\n"
"    // something nearer was drawn over the reflective surface since: its reflection is not on screen any more\n"
"    if (ao_view_z(DepthTex.Load(px)) < z0 * 0.995) return base;\n"
"    float3 e = SsrE.Load(px).rgb;\n"
"    // The env maps were drawn for a lit average; the screen of a night scene is far darker. Bring the reflected scene\n"
"    // up towards that average (at most p1.w times), never down.\n"
"    float avg = dot(SsrBlur.SampleLevel(LinSmp, float2(0.5, 0.5), p2.w).rgb, float3(0.299, 0.587, 0.114));\n"
"    float gain = clamp(0.3 / max(avg, 0.01), 1.0, p1.w);\n"
"    float4 rt = p3.z > 0.5 ? RtRefl.Load(px) : float4(0.0, 0.0, 0.0, 0.0);\n"
"    // RECOMP_KGPU_RT_REFL_DEBUG: traced hits green, traced misses blue, untraced pixels red\n"
"    if (p3.w > 0.5) return float4(rt.a > 1.5 ? float3(0, 1, 0) : rt.a > 0.5 ? float3(0, 0, 1) : float3(1, 0, 0), base.a);\n"
"    // A traced hit. The map's reflection e is replaced by the traced scene's, as bright as the map made it for that much\n"
"    // light (the texel the map gave is e / w; its brightness against the traced colour's) and with part of the scene's\n"
"    // colour (brought up to that brightness it looks more saturated than the scene; less on the weapon, rt.a 3, polished\n"
"    // metal a hand from the eye that would look painted). It scales the pixel by the share e has in it: e is not taken\n"
"    // away and added back, since a bright metal pixel is clipped at white and white less a bluish e is gold.\n"
"    if (rt.a > 1.5) {\n"
"        const float3 Y = float3(0.299, 0.587, 0.114);\n"
"        float3 texel = saturate(e / max(w.rgb, 0.004)), rc = saturate(rt.rgb);\n"
"        float lt = dot(texel, Y), lr = dot(rc, Y);\n"
"        float ratio = clamp(lr / max(lt, 0.02), rt.a > 2.5 ? 0.6 : 0.25, 3.0);\n"
"        float3 hue = min(lerp(float3(lr, lr, lr), rc, rt.a > 2.5 ? 0.3 : 0.6) / max(lr, 0.02), 1.5);\n"
"        float share = saturate(dot(e, Y) / max(dot(base.rgb, Y), 0.01));\n"
"        return float4(base.rgb * lerp(float3(1.0, 1.0, 1.0), ratio * hue, share), base.a);\n"
"    }\n""    float2 uv = pos.xy * p0.xy;\n"
"    float3 P = float3(float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) * p3.xy * z0, z0);\n"
"    float3 N = normalize(g.yzw);\n"
"    float3 R = reflect(normalize(P), N);\n"
"    float reach = p1.y * z0, t0 = z0 * 0.01, dt = (reach - t0) / p1.x;\n"
"    float jitter = ao_hash(pos.xy), prev = t0, hit = -1.0;\n"
"    // a traced miss needs no march: the ray found nothing the screen could show\n"
"    [loop] for (float i = 0.0; i < (rt.a > 0.5 ? 0.0 : p1.x); i += 1.0) {\n"
"        float t = t0 + (i + jitter) * dt;\n"
"        float3 q = P + R * t;\n"
"        if (q.z <= 0.05) break;\n"
"        float2 quv = ssr_uv(q);\n"
"        if (any(quv < 0.0) || any(quv > 1.0)) break;\n"
"        float dz = q.z - ssr_scene_z(quv);\n"
"        if (dz > 0.0) { if (dz < p1.z * q.z + dt) hit = t; break; }\n"
"        prev = t;\n"
"    }\n"
"    // a miss: the screen itself as the environment, blurred and looked up by direction like a sphere map, over the\n"
"    // original map by p2.x\n"
"    float2 suv = float2(0.5 + R.x * 0.45, 0.5 - R.y * 0.45);\n"
"    float3 env = SsrBlur.SampleLevel(LinSmp, suv, p2.y).rgb;\n"
"    float3 missed = base.rgb + p2.x * (gain * w.rgb * env - e);\n"
"    if (hit < 0.0) return float4(missed, base.a);\n"
"    float lo = prev, hi = hit;\n"
"    [unroll] for (int k = 0; k < 6; k++) {\n"
"        float m = 0.5 * (lo + hi);\n"
"        float3 q = P + R * m;\n"
"        if (q.z - ssr_scene_z(ssr_uv(q)) > 0.0) hi = m; else lo = m;\n"
"    }\n"
"    float2 huv = ssr_uv(P + R * hi);\n"
"    // farther hits are softer, as a rough surface would make them\n"
"    float3 hc = SsrBlur.SampleLevel(LinSmp, huv, saturate(hi / reach) * p2.z).rgb;\n"
"    // fade out at the screen's edges, at the end of the reach and for rays turning back to the camera\n"
"    float edge = saturate(min(min(huv.x, 1.0 - huv.x), min(huv.y, 1.0 - huv.y)) * 10.0);\n"
"    float fade = edge * saturate((reach - hi) / (0.25 * reach)) * saturate(R.z * 4.0 + 0.6);\n"
"    return float4(lerp(missed, base.rgb + gain * w.rgb * hc - e, fade), base.a);\n"
"}\n";

typedef struct KgPostCb { float p0[4], p1[4], p2[4], p3[4]; } KgPostCb;

/* Screen-space reflections (kelvin_gpu_ssr.inc): the Reflections video setting, PCV_REFL_ORIGINAL (the game's own
 * environment maps) or low .. ultra. */
static int s_ssr_setting = 0;
static float s_ssr_env_mix = 1.0f;     /* RECOMP_KGPU_SSR_ENV: how much a missed ray's screen environment replaces the map */
static float s_ssr_strength = 4.0f;    /* RECOMP_KGPU_SSR_STRENGTH: the most a dark scene's reflection is brightened */
static void kg_ssr_pass(KGpuSurf *cs, KGpuSurf *zs);
static void kg_ssr_after_flip(void);

static struct {
    bool tried, ok;
    ID3D11PixelShader *ps_focus, *ps_coc, *ps_dof, *ps_fxaa, *ps_sharp, *ps_ssaa;
    bool ao_tried, ao_ok;
    ID3D11PixelShader *ps_ao_ssao, *ps_ao_hbao, *ps_ao_hbaop, *ps_ao_gtao;
    ID3D11PixelShader *ps_ao_blur_h, *ps_ao_blur_v, *ps_ao_comp;
    ID3D11Buffer *cb;
    ID3D11SamplerState *smp_lin, *smp_pt;
    unsigned int w, h;                                      /* size of the scene copy and the blur map */
    ID3D11Texture2D *copy, *coc;
    ID3D11ShaderResourceView *copy_srv, *coc_srv;
    ID3D11RenderTargetView *coc_rtv;
    ID3D11Texture2D *ao[2];                                /* full-resolution visibility mask and blur scratch */
    ID3D11ShaderResourceView *ao_srv[2];
    ID3D11RenderTargetView *ao_rtv[2];
    ID3D11Texture2D *focus[2];
    ID3D11ShaderResourceView *focus_srv[2];
    ID3D11RenderTargetView *focus_rtv[2];
    unsigned int focus_cur;
    ID3D11Texture2D *ss;                                    /* the resolved frame at the internal resolution */
    ID3D11ShaderResourceView *ss_srv;
    ID3D11RenderTargetView *ss_rtv;
    unsigned int ss_w, ss_h;
    /* the settings in force (pc_video.c), read when its generation changes */
    unsigned int gen;
    bool cfg;
    int aa, dof, sharpen, ao_method, ao_quality;
    bool scene_fx;                                          /* any effect that runs on the scene */
    /* this frame */
    unsigned int world;                                     /* depth-writing draws on the main target */
    uint64_t ao_camera;                                     /* first world draw's queued camera, fixed for this frame */
    bool ao_done;
    bool done;
    long long last_qpc;                                     /* last successful autofocus update, not merely a scene pass */
    long long focus_trace_qpc;                              /* opt-in successful-observation trace, at most once per five seconds */
    int depth_stats;                                        /* RECOMP_KGPU_DEPTH_STATS=n: depth percentiles every n frames */
    char ao_dump[MAX_PATH], ao_scene_prefix[MAX_PATH];
    uint64_t ao_dump_frame, ao_scene_frame;
    bool ao_dump_done, ao_scene_done, post_trace, ao_skip_traced, focus_skipped;
} s_post;

static void kg_post_free(void)
{
    unsigned int i;
    SAFE_RELEASE(s_post.copy_srv); SAFE_RELEASE(s_post.copy);
    SAFE_RELEASE(s_post.coc_srv); SAFE_RELEASE(s_post.coc_rtv); SAFE_RELEASE(s_post.coc);
    for (i = 0; i < 2; i++) { SAFE_RELEASE(s_post.ao_srv[i]); SAFE_RELEASE(s_post.ao_rtv[i]); SAFE_RELEASE(s_post.ao[i]); }
    for (i = 0; i < 2; i++) { SAFE_RELEASE(s_post.focus_srv[i]); SAFE_RELEASE(s_post.focus_rtv[i]); SAFE_RELEASE(s_post.focus[i]); }
    SAFE_RELEASE(s_post.ss_srv); SAFE_RELEASE(s_post.ss_rtv); SAFE_RELEASE(s_post.ss);
    s_post.w = s_post.h = s_post.ss_w = s_post.ss_h = 0;
    s_post.focus_cur = 0;
}

static ID3D11PixelShader *post_shader(const char *entry)
{
    ID3DBlob *b = compile_hlsl("", kgpu_hlsl_post, entry, "ps_5_0");
    ID3D11PixelShader *ps = NULL;
    if (!b) return NULL;
    if (FAILED(ID3D11Device_CreatePixelShader(G.dev, ID3D10Blob_GetBufferPointer(b), ID3D10Blob_GetBufferSize(b), NULL, &ps))) ps = NULL;
    ID3D10Blob_Release(b);
    return ps;
}

static bool post_init(void)
{
    D3D11_SAMPLER_DESC sd;
    if (s_post.tried) return s_post.ok;
    s_post.tried = true;
    s_post.ps_focus = post_shader("focus_ps"); s_post.ps_coc = post_shader("coc_ps"); s_post.ps_dof = post_shader("dof_ps");
    s_post.ps_fxaa = post_shader("fxaa_ps"); s_post.ps_sharp = post_shader("sharpen_ps"); s_post.ps_ssaa = post_shader("ssaa_ps");
    if (!s_post.ps_focus || !s_post.ps_coc || !s_post.ps_dof || !s_post.ps_fxaa || !s_post.ps_sharp || !s_post.ps_ssaa) {
        fprintf(stderr, "[KGPU] picture effects unavailable: their shaders did not compile\n");
        return false;
    }
    if (!make_buffer(sizeof(KgPostCb), D3D11_BIND_CONSTANT_BUFFER, &s_post.cb)) return false;
    memset(&sd, 0, sizeof(sd));
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxAnisotropy = 1; sd.ComparisonFunc = D3D11_COMPARISON_NEVER; sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(ID3D11Device_CreateSamplerState(G.dev, &sd, &s_post.smp_lin))) return false;
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    if (FAILED(ID3D11Device_CreateSamplerState(G.dev, &sd, &s_post.smp_pt))) return false;
    s_post.ok = true;
    return true;
}

/* AO has independent shader availability, so a failed AO compile cannot disable the other picture effects. */
static bool post_ao_init(void)
{
    if (s_post.ao_tried) return s_post.ao_ok;
    s_post.ao_tried = true;
    s_post.ps_ao_ssao = post_shader("ao_ssao_ps"); s_post.ps_ao_hbao = post_shader("ao_hbao_ps");
    s_post.ps_ao_hbaop = post_shader("ao_hbaop_ps"); s_post.ps_ao_gtao = post_shader("ao_gtao_ps");
    s_post.ps_ao_blur_h = post_shader("ao_blur_h_ps"); s_post.ps_ao_blur_v = post_shader("ao_blur_v_ps");
    s_post.ps_ao_comp = post_shader("ao_composite_ps");
    s_post.ao_ok = s_post.ps_ao_ssao && s_post.ps_ao_hbao && s_post.ps_ao_hbaop && s_post.ps_ao_gtao &&
                   s_post.ps_ao_blur_h && s_post.ps_ao_blur_v && s_post.ps_ao_comp;
    if (!s_post.ao_ok) fprintf(stderr, "[KGPU] ambient occlusion unavailable: its shaders did not compile\n");
    return s_post.ao_ok;
}

static bool post_texture(unsigned int w, unsigned int h, DXGI_FORMAT fmt, ID3D11Texture2D **tex, ID3D11ShaderResourceView **srv,
                         ID3D11RenderTargetView **rtv)
{
    D3D11_TEXTURE2D_DESC td;
    memset(&td, 0, sizeof(td));
    td.Width = w; td.Height = h; td.MipLevels = td.ArraySize = 1; td.Format = fmt; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_SHADER_RESOURCE | (rtv ? D3D11_BIND_RENDER_TARGET : 0);
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, tex))) return false;
    if (FAILED(ID3D11Device_CreateShaderResourceView(G.dev, (ID3D11Resource *)*tex, NULL, srv))) return false;
    if (rtv && FAILED(ID3D11Device_CreateRenderTargetView(G.dev, (ID3D11Resource *)*tex, NULL, rtv))) return false;
    return true;
}

/* The scene copy and the blur map at the size of the target, and the focus pair. */
static bool post_scene_resources(unsigned int w, unsigned int h)
{
    unsigned int i;
    if (s_post.copy && s_post.w == w && s_post.h == h) return true;
    kg_post_free();
    if (!post_texture(w, h, DXGI_FORMAT_B8G8R8A8_UNORM, &s_post.copy, &s_post.copy_srv, NULL) ||
        !post_texture(w, h, DXGI_FORMAT_R8_UNORM, &s_post.coc, &s_post.coc_srv, &s_post.coc_rtv)) { kg_post_free(); return false; }
    for (i = 0; i < 2; i++) {
        const FLOAT unset[4] = { -1.0f, -1.0f, -1.0f, -1.0f };
        if (!post_texture(1, 1, DXGI_FORMAT_R32_FLOAT, &s_post.focus[i], &s_post.focus_srv[i], &s_post.focus_rtv[i])) { kg_post_free(); return false; }
        ID3D11DeviceContext_ClearRenderTargetView(G.ctx, s_post.focus_rtv[i], unset);
    }
    s_post.w = w; s_post.h = h;
    return true;
}

static bool post_ao_resources(unsigned int w, unsigned int h)
{
    unsigned int i;
    if (s_post.ao[0] && s_post.ao[1]) return true; /* scene resource resize frees both */
    for (i = 0; i < 2; i++) {
        if (!post_texture(w, h, DXGI_FORMAT_R8_UNORM, &s_post.ao[i], &s_post.ao_srv[i], &s_post.ao_rtv[i])) {
            unsigned int j;
            for (j = 0; j < 2; j++) { SAFE_RELEASE(s_post.ao_srv[j]); SAFE_RELEASE(s_post.ao_rtv[j]); SAFE_RELEASE(s_post.ao[j]); }
            return false;
        }
    }
    return true;
}

static void post_cb(float w, float h, float taps, float radius, float focus_alpha, float depth_scale, float zone, float near_blur,
                    float sharpen, float out_w, float out_h)
{
    D3D11_MAPPED_SUBRESOURCE ms;
    KgPostCb c;
    memset(&c, 0, sizeof(c));
    c.p0[0] = 1.0f / w; c.p0[1] = 1.0f / h; c.p0[2] = w; c.p0[3] = h;
    c.p1[0] = radius; c.p1[1] = taps; c.p1[2] = focus_alpha; c.p1[3] = depth_scale;
    c.p2[0] = zone; c.p2[1] = near_blur; c.p2[2] = sharpen;
    c.p3[0] = out_w; c.p3[1] = out_h; c.p3[2] = 1.0f / out_w; c.p3[3] = 1.0f / out_h;
    if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)s_post.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        memcpy(ms.pData, &c, sizeof(c));
        ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)s_post.cb, 0);
    }
}

/* Quality changes the sampling/radius; the mask stays at scene resolution and both blur passes keep five taps. */
static bool post_ao_cb(unsigned int w, unsigned int h)
{
    static const float ssao_taps[4] = { 8, 12, 18, 24 }, ssao_radius[4] = { 5, 8, 11, 15 };
    static const float hbao_dirs[4] = { 4, 6, 8, 10 }, hbao_radius[4] = { 8, 10, 13, 16 };
    static const float hbaop_dirs[4] = { 6, 8, 10, 12 }, hbaop_radius[4] = { 10, 13, 17, 21 };
    static const float gtao_dirs[4] = { 4, 6, 8, 8 }, gtao_radius[4] = { 9, 12, 16, 20 };
    int q = s_post.ao_quality;
    D3D11_MAPPED_SUBRESOURCE ms;
    KgPostCb c;
    if (q < PCV_AOQ_LOW || q > PCV_AOQ_ULTRA) q = PCV_AOQ_HIGH;
    memset(&c, 0, sizeof(c));
    c.p0[0] = 1.0f / w; c.p0[1] = 1.0f / h; c.p0[2] = (float)w; c.p0[3] = (float)h;
    c.p1[0] = ssao_radius[q]; c.p1[1] = ssao_taps[q]; c.p1[2] = (float)(3 + q); c.p1[3] = 0.48f + 0.08f * q;
    c.p2[0] = 0.025f; c.p2[1] = 0.30f; c.p2[2] = 0.32f;
    if (s_post.ao_method == PCV_AO_HBAO) {
        c.p1[0] = hbao_radius[q]; c.p1[1] = hbao_dirs[q]; c.p1[3] = 0.70f + 0.05f * q;
    } else if (s_post.ao_method == PCV_AO_HBAO_PLUS) {
        c.p1[0] = hbaop_radius[q]; c.p1[1] = hbaop_dirs[q]; c.p1[3] = 0.68f + 0.05f * q;
    } else if (s_post.ao_method == PCV_AO_GTAO) {
        c.p1[0] = gtao_radius[q]; c.p1[1] = gtao_dirs[q]; c.p1[3] = 1.80f;
        c.p2[0] = 0.015f; c.p2[1] = 0.48f;
    }
    if (!kg_ao_camera(s_post.ao_camera, &c.p3[0], &c.p3[1])) return false;
    if (SUCCEEDED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)s_post.cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
        memcpy(ms.pData, &c, sizeof(c));
        ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)s_post.cb, 0);
        return true;
    }
    return false;
}

/* One full-screen pass: `ps` reading up to four textures, into `rtv` of size w x h. */
static void post_pass(ID3D11PixelShader *ps, ID3D11RenderTargetView *rtv, unsigned int w, unsigned int h,
                      ID3D11ShaderResourceView *t0, ID3D11ShaderResourceView *t1, ID3D11ShaderResourceView *t2,
                      ID3D11ShaderResourceView *t3)
{
    ID3D11ShaderResourceView *in[4], *none[4] = { NULL, NULL, NULL, NULL };
    ID3D11SamplerState *smp[2];
    in[0] = t0; in[1] = t1; in[2] = t2; in[3] = t3;
    smp[0] = s_post.smp_lin; smp[1] = s_post.smp_pt;
    ID3D11DeviceContext_OMSetRenderTargets(G.ctx, 1, &rtv, NULL);
    fullscreen_pass_state(w, h);
    ID3D11DeviceContext_OMSetDepthStencilState(G.ctx, G.ds_none, 0);
    ID3D11DeviceContext_PSSetShader(G.ctx, ps, NULL, 0);
    ID3D11DeviceContext_PSSetConstantBuffers(G.ctx, 0, 1, &s_post.cb);
    ID3D11DeviceContext_PSSetSamplers(G.ctx, 0, 2, smp);
    ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 4, in);
    ID3D11DeviceContext_Draw(G.ctx, 3, 0);
    ID3D11DeviceContext_PSSetShaderResources(G.ctx, 0, 4, none);
}

/* The depth of field presets: taps, largest blur radius as a fraction of the picture height. */
static void post_dof_preset(int level, float *taps, float *radius)
{
    *taps = level >= 3 ? 32.0f : level == 2 ? 20.0f : 12.0f;
    *radius = level >= 3 ? 0.030f : level == 2 ? 0.020f : 0.012f;
}

static void kg_post_depth_stats(KGpuSurf *zs)
{
    static ID3D11Texture2D *stage;
    static unsigned int sw, sh;
    D3D11_MAPPED_SUBRESOURCE ms;
    unsigned int x, y, n = 0, cap = zs->hw * zs->hh / 16 + 1, k;
    float *v;
    if (!stage || sw != zs->hw || sh != zs->hh) {
        D3D11_TEXTURE2D_DESC td;
        SAFE_RELEASE(stage);
        memset(&td, 0, sizeof(td));
        td.Width = zs->hw; td.Height = zs->hh; td.MipLevels = td.ArraySize = 1; td.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
        td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &stage))) return;
        sw = zs->hw; sh = zs->hh;
    }
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)stage, (ID3D11Resource *)zs->tex);
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)stage, 0, D3D11_MAP_READ, 0, &ms))) return;
    v = (float *)malloc(cap * sizeof(float));
    if (v) {
        for (y = 0; y < zs->hh; y += 4)
            for (x = 0; x < zs->hw; x += 4) {
                const float d = *(const float *)((const uint8_t *)ms.pData + (size_t)y * ms.RowPitch + (size_t)x * 8);
                if (n < cap && d < 0.99999f) v[n++] = d;
            }
        if (n) {
            for (x = 1; x < n; x++) { float t = v[x]; for (k = x; k > 0 && v[k - 1] > t; k--) v[k] = v[k - 1]; v[k] = t; }
            fprintf(stderr, "[KDEPTH] %u non-sky samples: min %.6f p5 %.6f p25 %.6f p50 %.6f p75 %.6f p95 %.6f max %.6f\n", n, (double)v[0],
                    (double)v[n / 20], (double)v[n / 4], (double)v[n / 2], (double)v[n * 3 / 4], (double)v[n * 19 / 20], (double)v[n - 1]);
        } else fprintf(stderr, "[KDEPTH] no non-sky samples\n");
        free(v);
    }
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)stage, 0);
}

/* Opt-in captures only read back one requested frame. No staging resources or CPU copies live between frames. */
static uint8_t *post_ao_colour_capture(ID3D11Texture2D *tex, unsigned int w, unsigned int h, const char *path)
{
    KGpuSurf cs;
    ID3D11Texture2D *stage = NULL;
    D3D11_MAPPED_SUBRESOURCE ms;
    uint8_t *pixels;
    unsigned int y;
    memset(&cs, 0, sizeof(cs));
    cs.tex = tex; cs.hw = w; cs.hh = h;
    if (!readback(&cs, &stage, &ms)) return NULL;
    pixels = (uint8_t *)malloc((size_t)w * h * 4);
    if (pixels) {
        for (y = 0; y < h; y++) memcpy(pixels + (size_t)y * w * 4, (const uint8_t *)ms.pData + (size_t)y * ms.RowPitch, (size_t)w * 4);
        if (path && *path) write_bmp(path, pixels, w, h, w * 4);
    }
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)stage, 0);
    SAFE_RELEASE(stage);
    return pixels;
}

static void post_ao_mask_capture(unsigned int w, unsigned int h, uint64_t frame)
{
    D3D11_TEXTURE2D_DESC td;
    D3D11_MAPPED_SUBRESOURCE ms;
    ID3D11Texture2D *stage = NULL;
    uint8_t *pixels;
    unsigned int x, y, minv = 255, maxv = 0;
    uint64_t sum = 0, nonwhite = 0;
    memset(&td, 0, sizeof(td));
    td.Width = w; td.Height = h; td.MipLevels = td.ArraySize = 1; td.Format = DXGI_FORMAT_R8_UNORM;
    td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_STAGING; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(ID3D11Device_CreateTexture2D(G.dev, &td, NULL, &stage))) return;
    ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)stage, (ID3D11Resource *)s_post.ao[0]);
    if (FAILED(ID3D11DeviceContext_Map(G.ctx, (ID3D11Resource *)stage, 0, D3D11_MAP_READ, 0, &ms))) { SAFE_RELEASE(stage); return; }
    pixels = (uint8_t *)malloc((size_t)w * h * 4);
    for (y = 0; y < h; y++) {
        const uint8_t *row = (const uint8_t *)ms.pData + (size_t)y * ms.RowPitch;
        for (x = 0; x < w; x++) {
            unsigned int v = row[x];
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
            sum += v; nonwhite += v < 255;
            if (pixels) {
                size_t k = ((size_t)y * w + x) * 4;
                pixels[k] = pixels[k + 1] = pixels[k + 2] = (uint8_t)v; pixels[k + 3] = 255;
            }
        }
    }
    if (pixels) { write_bmp(s_post.ao_dump, pixels, w, h, w * 4); free(pixels); }
    fprintf(stderr, "[KAO] frame %llu method %d quality %d mask min %u max %u mean %.4f occluded %llu/%llu\n",
            (unsigned long long)frame, s_post.ao_method, s_post.ao_quality, minv, maxv, (double)sum / ((double)w * h),
            (unsigned long long)nonwhite, (unsigned long long)((uint64_t)w * h));
    ID3D11DeviceContext_Unmap(G.ctx, (ID3D11Resource *)stage, 0);
    SAFE_RELEASE(stage);
}

static void post_ao_scene_compare(KGpuSurf *cs, const uint8_t *before, const char *after_path, uint64_t frame)
{
    uint8_t *after = post_ao_colour_capture(cs->tex, cs->hw, cs->hh, after_path);
    uint64_t changed = 0, sum = 0;
    unsigned int x, y, maxd = 0;
    if (before && after) {
        for (y = 0; y < cs->hh; y++) for (x = 0; x < cs->hw; x++) {
            size_t k = ((size_t)y * cs->hw + x) * 4;
            unsigned int ch, pixel_diff = 0;
            for (ch = 0; ch < 3; ch++) {
                unsigned int d = (unsigned int)abs((int)before[k + ch] - (int)after[k + ch]);
                sum += d; if (d > maxd) maxd = d; pixel_diff |= d;
            }
            changed += pixel_diff != 0;
        }
        fprintf(stderr, "[KAO] frame %llu method %d quality %d same-frame changed pixels %llu abs-sum %llu max RGB difference %u\n",
                (unsigned long long)frame, s_post.ao_method, s_post.ao_quality, (unsigned long long)changed, (unsigned long long)sum, maxd);
    } else fprintf(stderr, "[KAO] frame %llu same-frame colour readback failed\n", (unsigned long long)frame);
    free(after);
}

/* The effects on the finished scene in `cs`. AO and depth of field need its real depth buffer in `zs`. */
static void kg_post_ao_scene(KGpuSurf *cs, KGpuSurf *zs)
{
    if (s_post.ao_done) return;
    if (s_post.ao_method == PCV_AO_OFF || !cs || cs->bpp != 4 || cs->dxfmt != DXGI_FORMAT_B8G8R8A8_UNORM ||
        !post_init() || !post_scene_resources(cs->hw, cs->hh)) return;
    if (s_post.ao_method != PCV_AO_OFF && zs && zs->depth && zs->real && zs->srv && zs->hw == cs->hw && zs->hh == cs->hh && post_ao_init() &&
        post_ao_resources(cs->hw, cs->hh) && post_ao_cb(cs->hw, cs->hh)) {
        ID3D11PixelShader *ps = s_post.ps_ao_ssao;
        const uint64_t frame = G.frames + 1;
        bool scene_capture = !s_post.ao_scene_done && (s_post.post_trace || *s_post.ao_scene_prefix) && frame >= s_post.ao_scene_frame;
        bool mask_capture = !s_post.ao_dump_done && *s_post.ao_dump && frame >= s_post.ao_dump_frame;
        uint8_t *before = NULL;
        char before_path[MAX_PATH + 16], after_path[MAX_PATH + 16];
        /* A failed early attempt must allow a later draw with valid scene depth. */
        s_post.ao_done = true;
        before_path[0] = after_path[0] = 0;
        if (s_post.ao_method == PCV_AO_HBAO) ps = s_post.ps_ao_hbao;
        else if (s_post.ao_method == PCV_AO_HBAO_PLUS) ps = s_post.ps_ao_hbaop;
        else if (s_post.ao_method == PCV_AO_GTAO) ps = s_post.ps_ao_gtao;
        if (scene_capture || mask_capture) {
            float hx = 0.0f, hy = 0.0f;
            kg_ao_camera(s_post.ao_camera, &hx, &hy);
            fprintf(stderr, "[KAO] frame %llu queued world camera halfX %.9g halfY %.9g\n", (unsigned long long)frame, (double)hx, (double)hy);
        }
        if (scene_capture) {
            if (*s_post.ao_scene_prefix) {
                snprintf(before_path, sizeof(before_path), "%s_before.bmp", s_post.ao_scene_prefix);
                snprintf(after_path, sizeof(after_path), "%s_after.bmp", s_post.ao_scene_prefix);
            }
            before = post_ao_colour_capture(cs->tex, cs->hw, cs->hh, before_path);
            s_post.ao_scene_done = true;
        }
        ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_post.copy, (ID3D11Resource *)cs->tex);
        post_pass(ps, s_post.ao_rtv[0], cs->hw, cs->hh, NULL, NULL, zs->srv, NULL);
        post_pass(s_post.ps_ao_blur_h, s_post.ao_rtv[1], cs->hw, cs->hh, NULL, s_post.ao_srv[0], zs->srv, NULL);
        post_pass(s_post.ps_ao_blur_v, s_post.ao_rtv[0], cs->hw, cs->hh, NULL, s_post.ao_srv[1], zs->srv, NULL);
        post_pass(s_post.ps_ao_comp, cs->rtv, cs->hw, cs->hh, s_post.copy_srv, s_post.ao_srv[0], NULL, NULL);
        if (mask_capture) { s_post.ao_dump_done = true; post_ao_mask_capture(cs->hw, cs->hh, frame); }
        if (scene_capture) { post_ao_scene_compare(cs, before, after_path, frame); free(before); }
        kg_cache_invalidate();
    }
}

static void kg_post_scene(KGpuSurf *cs, KGpuSurf *zs)
{
    float taps, radius, depth_scale = 140.0f;
    long long now = qpc_now();
    bool dlaa, upscale, upscaled = false;
    s_post.done = true;
    if (!s_post.scene_fx || !cs || cs->bpp != 4 || cs->dxfmt != DXGI_FORMAT_B8G8R8A8_UNORM || !post_init() ||
        !post_scene_resources(cs->hw, cs->hh)) return;
    {
        const char *e = getenv("RECOMP_KGPU_DOF_SCALE");
        if (e && *e) depth_scale = (float)atof(e);
    }
    if (s_post.depth_stats > 0 && zs && (G.frames % (uint64_t)s_post.depth_stats) == 0) kg_post_depth_stats(zs);
    if (s_post.ao_method != PCV_AO_OFF && !s_post.ao_camera && !s_post.ao_skip_traced &&
        (s_post.post_trace || *s_post.ao_scene_prefix || *s_post.ao_dump)) {
        s_post.ao_skip_traced = true;
        fprintf(stderr, "[KAO] frame %llu ambient occlusion skipped: no valid queued world camera\n", (unsigned long long)(G.frames + 1));
    }
    kg_post_ao_scene(cs, zs);
    kg_ssr_pass(cs, zs);
    dlaa = s_post.aa == PCV_AA_DLAA && kg_dlss_run(cs, zs);
    upscale = s_post.aa >= PCV_AA_DLSS_QUALITY && s_post.aa <= PCV_AA_DLSS_PERFORMANCE;
    if (s_post.dof && (!zs || !zs->depth || !zs->real || !zs->srv || zs->hw != cs->hw || zs->hh != cs->hh)) {
        if (s_post.post_trace && !s_post.focus_skipped)
            fprintf(stderr, "[KPOST] frame %llu autofocus skipped: depth present=%d depth=%d real=%d sampleable=%d size=%ux%u scene=%ux%u\n",
                    (unsigned long long)(G.frames + 1), zs != NULL, zs ? (int)zs->depth : 0,
                    zs ? (int)zs->real : 0, zs && zs->srv != NULL, zs ? zs->hw : 0, zs ? zs->hh : 0, cs->hw, cs->hh);
        s_post.focus_skipped = true;
    }
    if (s_post.dof && zs && zs->depth && zs->real && zs->srv && zs->hw == cs->hw && zs->hh == cs->hh) {
        const unsigned int next = s_post.focus_cur ^ 1u;
        const double dt = s_post.last_qpc ? (double)(now - s_post.last_qpc) / (double)G.qpf : -1.0;
        const float alpha = kg_post_focus_blend(dt);
        if (s_post.post_trace && (s_post.focus_skipped || !s_post.last_qpc))
            fprintf(stderr, "[KPOST] frame %llu autofocus acquired: observation gap %.9g alpha %.9g\n",
                    (unsigned long long)(G.frames + 1), dt, (double)alpha);
        s_post.focus_skipped = false;
        post_dof_preset(s_post.dof, &taps, &radius);
        ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_post.copy, (ID3D11Resource *)cs->tex);
        post_cb((float)cs->hw, (float)cs->hh, taps, radius * (float)cs->hh, alpha, depth_scale, 0.002f, 0.0f, 0.0f, 1.0f, 1.0f);
        post_pass(s_post.ps_focus, s_post.focus_rtv[next], 1, 1, NULL, NULL, zs->srv, s_post.focus_srv[s_post.focus_cur]);
        s_post.focus_cur = next;
        post_pass(s_post.ps_coc, s_post.coc_rtv, cs->hw, cs->hh, NULL, NULL, zs->srv, s_post.focus_srv[next]);
        post_pass(s_post.ps_dof, cs->rtv, cs->hw, cs->hh, s_post.copy_srv, s_post.coc_srv, NULL, NULL);
        /* A frame without usable depth did not observe a focus surface. Keep
         * the last observation time so a long interruption actually triggers
         * kg_post_focus_blend's immediate reacquisition when depth returns. */
        s_post.last_qpc = now;
        if (s_post.post_trace && (!s_post.focus_trace_qpc ||
            (double)(now - s_post.focus_trace_qpc) / (double)G.qpf >= 5.0)) {
            fprintf(stderr, "[KPOST] frame %llu autofocus updated: observation gap %.9g alpha %.9g depth=%ux%u scene=%ux%u\n",
                    (unsigned long long)(G.frames + 1), dt, (double)alpha, zs->hw, zs->hh, cs->hw, cs->hh);
            s_post.focus_trace_qpc = now;
        }
    }
    if (upscale) upscaled = kg_dlss_run(cs, zs);   /* the frame is now the internal resolution: cs was swapped */
    if (s_post.aa == PCV_AA_FXAA || (s_post.aa == PCV_AA_DLAA && !dlaa) || (upscale && !upscaled)) {   /* no DLSS: FXAA */
        ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_post.copy, (ID3D11Resource *)cs->tex);
        post_cb((float)cs->hw, (float)cs->hh, 0, 0, 0, 0, 0, 0, 0, 1.0f, 1.0f);
        post_pass(s_post.ps_fxaa, cs->rtv, cs->hw, cs->hh, s_post.copy_srv, NULL, NULL, NULL);
    }
    if (s_post.sharpen && !upscaled) {
        ID3D11DeviceContext_CopyResource(G.ctx, (ID3D11Resource *)s_post.copy, (ID3D11Resource *)cs->tex);
        post_cb((float)cs->hw, (float)cs->hh, 0, 0, 0, 0, 0, 0, s_post.sharpen == 1 ? 0.25f : s_post.sharpen == 2 ? 0.55f : 0.9f, 1.0f, 1.0f);
        post_pass(s_post.ps_sharp, cs->rtv, cs->hw, cs->hh, s_post.copy_srv, NULL, NULL, NULL);
    }
    kg_cache_invalidate();
}

/* The target of a draw is the main full-screen target (the world, then the HUD, are drawn to it). */
static inline bool post_main_target(const KGpuSurf *cs)
{
    return cs && !cs->depth && cs->gw == 640 && cs->gh == 480;
}

/* Called for every draw once its surfaces are known, before its state is bound. */
static void kg_post_note_draw(const KPixelState *ps, const KGpuSurf *cs, KGpuSurf *zs)
{
    if (!s_post.scene_fx || !post_main_target(cs)) return;
    if (ps->depth_write && zs) {
        if (!s_post.world) s_post.ao_camera = s_ao_draw_camera;
        s_post.world++;
        return;
    }
    if (kg_post_order_should_apply_ao_before_blend(s_post.world, ps->depth_write, ps->blend, &s_post.ao_done))
        kg_post_ao_scene((KGpuSurf *)cs, zs);
    /* The HUD and menus: depth test on, always passing, nothing written. */
    if (!s_post.done && s_post.world >= 4 && ps->depth_test && ps->depth_func == 7 && !ps->depth_write)
        kg_post_scene((KGpuSurf *)cs, zs);
}

/* A change in the settings: the sampler cache, the draw state and the passes follow. */
static void kg_post_poll(void)
{
    const unsigned int gen = pc_video_generation();
    unsigned int i;
    PcVideoSettings v;
    if (s_post.cfg && gen == s_post.gen) return;
    pc_video_get(&v);
    s_post.gen = gen;
    if (!s_post.cfg) {
        const char *e = getenv("RECOMP_KGPU_DEPTH_STATS");
        s_post.depth_stats = e && *e ? atoi(e) : 0;
        e = getenv("RECOMP_KGPU_AO_DUMP");
        if (e && *e) snprintf(s_post.ao_dump, sizeof(s_post.ao_dump), "%s", e);
        e = getenv("RECOMP_KGPU_AO_DUMP_FRAME");
        s_post.ao_dump_frame = e && *e ? strtoull(e, NULL, 0) : 120;
        e = getenv("RECOMP_KGPU_AO_SCENE_DUMP_PREFIX");
        if (e && *e) snprintf(s_post.ao_scene_prefix, sizeof(s_post.ao_scene_prefix), "%s", e);
        e = getenv("RECOMP_KGPU_AO_SCENE_DUMP_FRAME");
        s_post.ao_scene_frame = e && *e ? strtoull(e, NULL, 0) : 120;
        e = getenv("RECOMP_KGPU_POST_TRACE");
        s_post.post_trace = e && *e && atoi(e) != 0;
        e = getenv("RECOMP_KGPU_SSR_ENV");
        if (e && *e) s_ssr_env_mix = (float)atof(e);
        if (!(s_ssr_env_mix >= 0.0f)) s_ssr_env_mix = 0.0f;
        if (s_ssr_env_mix > 1.0f) s_ssr_env_mix = 1.0f;
        e = getenv("RECOMP_KGPU_SSR_STRENGTH");
        if (e && *e) s_ssr_strength = (float)atof(e);
        if (!(s_ssr_strength >= 1.0f) || s_ssr_strength > 8.0f) s_ssr_strength = 4.0f;
    }
    if (s_post.cfg && !s_post.dof && v.dof && s_post.focus_rtv[0] && s_post.focus_rtv[1]) {
        const FLOAT unset[4] = { -1.0f, -1.0f, -1.0f, -1.0f };
        for (i = 0; i < 2; i++) ID3D11DeviceContext_ClearRenderTargetView(G.ctx, s_post.focus_rtv[i], unset);
        s_post.focus_cur = 0;
    }
    s_post.cfg = true;
    s_post.aa = (int)v.aa; s_post.dof = (int)v.dof; s_post.sharpen = (int)v.sharpen;
    s_post.ao_method = (int)v.ao_method; s_post.ao_quality = (int)v.ao_quality;
    s_ssr_setting = v.reflections < PCV_REFL_COUNT ? (int)v.reflections : PCV_REFL_HIGH;
    s_post.scene_fx = s_post.ao_method != PCV_AO_OFF || s_post.aa == PCV_AA_FXAA || s_post.aa >= PCV_AA_DLAA || s_post.dof || s_post.sharpen || s_post.depth_stats > 0 ||
                      s_ssr_setting > 0;
    if ((int)v.af != s_af) {
        unsigned int i;
        s_af = (int)v.af;
        for (i = 0; i < (unsigned int)G.nsampler && i < 256; i++) SAFE_RELEASE(G.sampler[i].ptr);
        G.nsampler = 0;
        kg_cache_invalidate();
    }
}

/* Before a flip: a frame with no HUD gets its effects now; the frame state starts again afterwards. */
static void kg_post_before_flip(const KelvinTarget *t, unsigned int color_format)
{
    kg_post_poll();
    if (s_post.scene_fx && !s_post.done && s_post.world >= 4) {
        KGpuSurf *cs = surf_get(t, false, color_format);
        KGpuSurf *zs = t->zeta_va ? surf_find(t->zeta_va, true) : NULL;
        if (post_main_target(cs)) kg_post_scene(cs, zs);
    }
}

static void kg_post_after_flip(void)
{
    kg_ssr_after_flip();
    kg_dlss_after_flip();
    s_post.world = 0;
    s_post.ao_camera = 0;
    s_post.ao_done = false;
    s_post.done = false;
}

/* The frame to present: the target itself, or under super-sampling that target resolved down to the internal resolution. */
/* A capture of the next finished frame as the renderer hands it to the presenter (internal resolution, before window
 * scaling, so without the presenter's menus and hints): any thread asks, the renderer writes it -- a PNG through WIC
 * for a .png path, else a BMP. kgpu_capture_result says how the last one went. */
static volatile LONG s_capture_pending, s_capture_result;
static char s_capture_path[512];

void kgpu_request_capture(const char *path)
{
    if (!path || !*path || InterlockedCompareExchange(&s_capture_pending, 0, 0)) return;
    snprintf(s_capture_path, sizeof(s_capture_path), "%s", path);
    InterlockedExchange(&s_capture_result, 0);
    InterlockedExchange(&s_capture_pending, 1);
}

int kgpu_capture_result(void) { return (int)InterlockedCompareExchange(&s_capture_result, 0, 0); }

/* 24-bit RGB (the target's alpha is not a picture's), UTF-8 path. */
static int write_png(const char *path, const uint8_t *bgra, unsigned int w, unsigned int h)
{
    IWICImagingFactory *factory = NULL;
    IWICStream *stream = NULL;
    IWICBitmapEncoder *encoder = NULL;
    IWICBitmapFrameEncode *frame = NULL;
    WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
    WCHAR wide[MAX_PATH];
    uint8_t *bgr = (uint8_t *)malloc((size_t)w * h * 3);
    const HRESULT com = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    HRESULT hr = bgr && MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, MAX_PATH) ? S_OK : E_FAIL;
    size_t i;
    if (SUCCEEDED(hr)) for (i = 0; i < (size_t)w * h; i++) { bgr[i * 3] = bgra[i * 4]; bgr[i * 3 + 1] = bgra[i * 4 + 1]; bgr[i * 3 + 2] = bgra[i * 4 + 2]; }
    if (SUCCEEDED(hr)) hr = CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, &IID_IWICImagingFactory, (void **)&factory);
    if (SUCCEEDED(hr)) hr = IWICImagingFactory_CreateStream(factory, &stream);
    if (SUCCEEDED(hr)) hr = IWICStream_InitializeFromFilename(stream, wide, GENERIC_WRITE);
    if (SUCCEEDED(hr)) hr = IWICImagingFactory_CreateEncoder(factory, &GUID_ContainerFormatPng, NULL, &encoder);
    if (SUCCEEDED(hr)) hr = IWICBitmapEncoder_Initialize(encoder, (IStream *)stream, WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = IWICBitmapEncoder_CreateNewFrame(encoder, &frame, NULL);
    if (SUCCEEDED(hr)) hr = IWICBitmapFrameEncode_Initialize(frame, NULL);
    if (SUCCEEDED(hr)) hr = IWICBitmapFrameEncode_SetSize(frame, w, h);
    if (SUCCEEDED(hr)) hr = IWICBitmapFrameEncode_SetPixelFormat(frame, &format);
    if (SUCCEEDED(hr) && !IsEqualGUID(&format, &GUID_WICPixelFormat24bppBGR)) hr = E_FAIL;
    if (SUCCEEDED(hr)) hr = IWICBitmapFrameEncode_WritePixels(frame, h, w * 3, w * h * 3, bgr);
    if (SUCCEEDED(hr)) hr = IWICBitmapFrameEncode_Commit(frame);
    if (SUCCEEDED(hr)) hr = IWICBitmapEncoder_Commit(encoder);
    if (frame) IWICBitmapFrameEncode_Release(frame);
    if (encoder) IWICBitmapEncoder_Release(encoder);
    if (stream) IWICStream_Release(stream);
    if (factory) IWICImagingFactory_Release(factory);
    if (FAILED(hr)) { MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, MAX_PATH); DeleteFileW(wide); }
    if (SUCCEEDED(com)) CoUninitialize();
    free(bgr);
    return SUCCEEDED(hr);
}

static ID3D11Texture2D *kg_capture(ID3D11Texture2D *tex, unsigned int w, unsigned int h)
{
    if (InterlockedCompareExchange(&s_capture_pending, 0, 0)) {
        const size_t n = strlen(s_capture_path);
        const int png = n > 4 && !_stricmp(s_capture_path + n - 4, ".png");
        uint8_t *pixels = post_ao_colour_capture(tex, w, h, png ? NULL : s_capture_path);
        const int ok = pixels && (!png || write_png(s_capture_path, pixels, w, h));
        free(pixels);
        fprintf(stderr, "[KGPU] capture %ux%u to %s: %s\n", w, h, s_capture_path, ok ? "written" : "failed");
        InterlockedExchange(&s_capture_result, ok ? 1 : -1);
        InterlockedExchange(&s_capture_pending, 0);
    }
    return tex;
}

static ID3D11Texture2D *kg_post_flip_source(KGpuSurf *cs, unsigned int *w, unsigned int *h)
{
    if (G.out_w == 0 || G.out_h == 0 || (G.out_w == cs->hw && G.out_h == cs->hh) || cs->hw % G.out_w || cs->hh % G.out_h ||
        !post_init()) { *w = cs->hw; *h = cs->hh; return kg_capture(cs->tex, cs->hw, cs->hh); }
    if (!s_post.ss || s_post.ss_w != G.out_w || s_post.ss_h != G.out_h) {
        SAFE_RELEASE(s_post.ss_srv); SAFE_RELEASE(s_post.ss_rtv); SAFE_RELEASE(s_post.ss);
        if (!post_texture(G.out_w, G.out_h, DXGI_FORMAT_B8G8R8A8_UNORM, &s_post.ss, &s_post.ss_srv, &s_post.ss_rtv)) {
            SAFE_RELEASE(s_post.ss_srv); SAFE_RELEASE(s_post.ss_rtv); SAFE_RELEASE(s_post.ss);
            *w = cs->hw; *h = cs->hh; return cs->tex;
        }
        s_post.ss_w = G.out_w; s_post.ss_h = G.out_h;
    }
    post_cb((float)cs->hw, (float)cs->hh, 0, 0, 0, 0, 0, 0, 0, (float)G.out_w, (float)G.out_h);
    post_pass(s_post.ps_ssaa, s_post.ss_rtv, G.out_w, G.out_h, cs->srv, NULL, NULL, NULL);
    kg_cache_invalidate();
    *w = G.out_w; *h = G.out_h;
    return kg_capture(s_post.ss, G.out_w, G.out_h);
}
