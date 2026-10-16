/*
 * Ray tracing for the Kelvin D3D11 backend. See kelvin_rt.h.
 *
 * d3d12.dll is loaded at run time, so a system without Direct3D 12 still runs the game, only without ray tracing.
 * Bring-up checks, in order: the D3D12 device on the renderer's adapter, DXR tier 1.1, a fence both devices can
 * use, and a texture written by D3D12 that D3D11 reads back as written. Any failure leaves ray tracing off.
 *
 * A frame (krt_frame): the captured triangles are unpacked from the shared texture into vertex buffers, an
 * acceleration structure is built over them -- one geometry a captured draw, so a hit's geometry index names its
 * draw -- with a single instance on top, and the passes query it with inline ray tracing (RayQuery in compute
 * shaders, kelvin/rt). Three command allocators take turns, so the CPU only waits if the GPU is three frames behind.
 */
#ifdef _WIN32
#define COBJMACROS
#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kelvin_rt.h"

#ifdef KRT_HAVE_SHADERS
#include "krt_unpack.h"
#include "krt_view.h"
#include "krt_reflect.h"
#include "krt_calibrate.h"
#include "krt_calib_update.h"
#include "krt_light.h"
#include "krt_reflect_filter.h"
#include "krt_light_filter.h"
#endif

#define KRT_RELEASE(p) do { if (p) { IUnknown_Release((IUnknown *)(p)); (p) = NULL; } } while (0)
#define KRT_SLOTS 3
#define KRT_HISTORIES 3
#define KRT_DRAWS_MAX 4096u
#define KRT_TEX_MAX 4096u
#define KRT_TEX_PENDING 1024u
#define KRT_STAMPS 7u

typedef HRESULT (WINAPI *PfnCreateDevice)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
typedef HRESULT (WINAPI *PfnGetDebug)(REFIID, void **);
typedef HRESULT (WINAPI *PfnSerializeRoot)(const D3D12_ROOT_SIGNATURE_DESC *, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob **, ID3DBlob **);

/* Descriptor heap: the unpack pass's inputs, a trace table a slot (its draw list differs), the unpack outputs, the
 * trace outputs, the game's textures. Each table is as long as the root signature's range. */
enum {
    H_UNPACK_SRV = 0,
    H_TRACE_SRV = 12,                  /* + 12 * slot */
    H_UNPACK_UAV = H_TRACE_SRV + 12 * KRT_SLOTS,
    H_TRACE_UAV = H_UNPACK_UAV + 8,
    H_TEX = H_TRACE_UAV + 8,
    H_COUNT = H_TEX + KRT_TEX_MAX
};

/* Constants a slot: the unpack pass's at 0, the frame's at 256. */
typedef struct KrtUnpackCb { uint32_t n, width, pad[2]; } KrtUnpackCb;
typedef struct KrtFrameCb {
    float inv[12], c58[4], c59[4], size[4], vp0[4], vp1[4], vp3[4], zp[4], misc[4];
    float light[4];                     /* towards the main light (xyz), shadow strength (w) */
    float prev_vp0[4], prev_vp1[4], prev_vp3[4]; /* last frame's camera, for the passes' history */
    float fx[4];                        /* occlusion strength, its reach, bounce colour strength */
} KrtFrameCb;

/* The colour factors shade learns (krt_calibrate.hlsl), rgb in 16 bytes from sums of 32: one a texture index, then
 * one a kind of lighting. */
#define KRT_FACTOR_KEYS (KRT_TEX_MAX + 3u)

static struct {
    bool ok;
    HMODULE lib;
    ID3D12Device5 *dev;
    ID3D12CommandQueue *queue;
    ID3D12CommandAllocator *alloc[KRT_SLOTS];
    uint64_t alloc_done[KRT_SLOTS];     /* the fence value that frees each allocator */
    unsigned slot;
    ID3D12GraphicsCommandList4 *list;
    ID3D12Fence *fence;                 /* the D3D11 fence below, opened on this device */
    HANDLE fence_event;
    ID3D11Device5 *dev11;
    ID3D11DeviceContext4 *ctx11;
    ID3D11Fence *fence11;
    HANDLE fence_h;
    uint64_t fence_value;

    ID3D12RootSignature *root;
    ID3D12PipelineState *pso_unpack, *pso_view, *pso_reflect, *pso_calibrate, *pso_calib_update, *pso_light, *pso_reflect_filter, *pso_light_filter;
    ID3D12Resource *sums, *factors;     /* the brightness calibration's sums and its factors */
    uint64_t frame;
    ID3D12DescriptorHeap *heap;
    UINT inc;

    ID3D12Resource *geo;                /* the captured triangles, written by D3D11 */
    unsigned geo_w, geo_h, max_verts;
    ID3D12Resource *pos, *uv, *col;     /* unpacked: 12, 16 and 4 bytes a vertex */
    bool vb_readable;                   /* in the shader-resource state (else unordered access) */
    ID3D12Resource *draws;              /* upload: KRT_SLOTS lists of each draw's first vertex, textures and flags */
    uint32_t *draws_cpu;
    ID3D12Resource *tex[KRT_TEX_MAX];   /* the game's textures, by index (krt_tex_add) */
    struct { int index; uint64_t value; } pending[KRT_TEX_PENDING]; /* removed, released once the GPU passes value
                                                                     * (UINT64_MAX: the frame being captured may use it) */
    unsigned npending;
    ID3D12Resource *consts;             /* upload: 512 bytes a slot */
    uint8_t *consts_cpu;
    ID3D12Resource *instance;           /* upload: the one instance */
    D3D12_RAYTRACING_INSTANCE_DESC *instance_cpu;
    ID3D12Resource *blas[2], *tlas, *scratch;   /* bottom levels: static, dynamic */
    UINT64 blas_size[2], tlas_size, scratch_size;
    D3D12_RAYTRACING_GEOMETRY_DESC *geoms;
    unsigned order[KRT_DRAWS_MAX];      /* geometry k is draw order[k]: the static ones, then the dynamic ones */
    float prev_vp[16];
    bool have_prev;

    ID3D12Resource *in_pic, *in_w, *in_g, *in_depth; /* the passes' inputs, written by D3D11 */
    ID3D12Resource *light_out;          /* the lighting factor, read by D3D11 */
    ID3D12Resource *history[2];         /* the lighting pass's, at the output's size */
    bool history_fresh;                 /* just made: nothing in it yet */
    ID3D12Resource *refl_raw, *refl_hist; /* the reflections before filtering, and the weapon's over frames */
    /* Split-screen: each view's own history (the lighting pass's two, the reflections') and last camera, kept here
     * while another view's frame is traced */
    struct { ID3D12Resource *t[3]; float prev_vp[16]; bool have_prev, fresh, kept; } hist[KRT_HISTORIES];
    int hist_cur;
    /* RECOMP_KGPU_RT_TIMES=1 (diagnostic): GPU time of each pass, averaged over 120 frames */
    int times;
    ID3D12QueryHeap *qheap;
    ID3D12Resource *qread;
    uint64_t *qread_cpu, qfreq;
    bool qpending[KRT_SLOTS];
    double qsum[KRT_STAMPS];
    unsigned qframes;
    ID3D12Resource *out;                /* written by the passes, read by D3D11 */
    unsigned out_w, out_h;
    bool said_frame;
} K;

static void release_inputs(void);

static D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle(unsigned i)
{
    D3D12_CPU_DESCRIPTOR_HANDLE h;
    ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(K.heap, &h);
    h.ptr += (SIZE_T)i * K.inc;
    return h;
}

static D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(unsigned i)
{
    D3D12_GPU_DESCRIPTOR_HANDLE h;
    ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(K.heap, &h);
    h.ptr += (UINT64)i * K.inc;
    return h;
}

static ID3D12Resource *make_buffer(UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES hp;
    D3D12_RESOURCE_DESC rd;
    ID3D12Resource *r = NULL;
    memset(&hp, 0, sizeof(hp));
    hp.Type = heap;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = flags;
    if (FAILED(ID3D12Device5_CreateCommittedResource(K.dev, &hp, D3D12_HEAP_FLAG_NONE, &rd, state, NULL,
                                                     &IID_ID3D12Resource, (void **)&r)))
        return NULL;
    return r;
}

static void raw_srv(unsigned slot, ID3D12Resource *r, UINT64 bytes)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d;
    memset(&d, 0, sizeof(d));
    d.Format = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.NumElements = (UINT)(bytes / 4);
    d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    ID3D12Device5_CreateShaderResourceView(K.dev, r, &d, cpu_handle(slot));
}

static void raw_uav(unsigned slot, ID3D12Resource *r, UINT64 bytes)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC d;
    memset(&d, 0, sizeof(d));
    d.Format = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements = (UINT)(bytes / 4);
    d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    ID3D12Device5_CreateUnorderedAccessView(K.dev, r, NULL, &d, cpu_handle(slot));
}

static void null_srv(unsigned slot)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d;
    memset(&d, 0, sizeof(d));
    d.Format = DXGI_FORMAT_R32_UINT;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    ID3D12Device5_CreateShaderResourceView(K.dev, NULL, &d, cpu_handle(slot));
}

static void null_tex(unsigned slot)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d;
    memset(&d, 0, sizeof(d));
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    ID3D12Device5_CreateShaderResourceView(K.dev, NULL, &d, cpu_handle(slot));
}

static void null_uav(unsigned slot)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC d;
    memset(&d, 0, sizeof(d));
    d.Format = DXGI_FORMAT_R32_UINT;
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    ID3D12Device5_CreateUnorderedAccessView(K.dev, NULL, NULL, &d, cpu_handle(slot));
}

/* The trace tables name the top-level structure by address, so they are written again when it moves. */
static void write_trace_tables(void)
{
    unsigned s, i;
    for (s = 0; s < KRT_SLOTS; s++) {
        const unsigned t = H_TRACE_SRV + 12 * s;
        D3D12_SHADER_RESOURCE_VIEW_DESC d;
        if (K.tlas) {
            memset(&d, 0, sizeof(d));
            d.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.RaytracingAccelerationStructure.Location = ID3D12Resource_GetGPUVirtualAddress(K.tlas);
            ID3D12Device5_CreateShaderResourceView(K.dev, NULL, &d, cpu_handle(t));
        } else {
            null_srv(t);
        }
        raw_srv(t + 1, K.pos, (UINT64)K.max_verts * 12);
        raw_srv(t + 2, K.uv, (UINT64)K.max_verts * 16);
        raw_srv(t + 3, K.col, (UINT64)K.max_verts * 4);
        memset(&d, 0, sizeof(d));
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.FirstElement = (UINT64)s * KRT_DRAWS_MAX;
        d.Buffer.NumElements = KRT_DRAWS_MAX;
        d.Buffer.StructureByteStride = 16;
        ID3D12Device5_CreateShaderResourceView(K.dev, K.draws, &d, cpu_handle(t + 4));
        for (i = 5; i < 8; i++) {
            static const DXGI_FORMAT fmt[3] = { DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R32G32B32A32_FLOAT };
            ID3D12Resource *r = i == 5 ? K.in_w : i == 6 ? K.in_pic : K.in_g;
            if (!r) { null_srv(t + i); continue; }
            memset(&d, 0, sizeof(d));
            d.Format = fmt[i - 5];
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture2D.MipLevels = 1;
            ID3D12Device5_CreateShaderResourceView(K.dev, r, &d, cpu_handle(t + i));
        }
        if (K.in_depth) {
            memset(&d, 0, sizeof(d));
            d.Format = DXGI_FORMAT_R32_FLOAT;
            d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Texture2D.MipLevels = 1;
            ID3D12Device5_CreateShaderResourceView(K.dev, K.in_depth, &d, cpu_handle(t + 8));
        } else {
            null_srv(t + 8);
        }
        for (i = 9; i < 12; i++) null_srv(t + i);
    }
}

bool krt_available(void) { return K.ok; }

static void release_scene(void)
{
    KRT_RELEASE(K.pos); KRT_RELEASE(K.uv); KRT_RELEASE(K.col);
    KRT_RELEASE(K.blas[0]); KRT_RELEASE(K.blas[1]); KRT_RELEASE(K.tlas); KRT_RELEASE(K.scratch);
    K.blas_size[0] = K.blas_size[1] = K.tlas_size = K.scratch_size = 0;
}

void krt_shutdown(void)
{
    unsigned i;
    release_scene();
    for (i = 0; i < KRT_TEX_MAX; i++) KRT_RELEASE(K.tex[i]);
    K.npending = 0;
    KRT_RELEASE(K.geo); KRT_RELEASE(K.out);
    release_inputs();
    KRT_RELEASE(K.draws); KRT_RELEASE(K.consts); KRT_RELEASE(K.instance);
    KRT_RELEASE(K.pso_unpack); KRT_RELEASE(K.pso_view); KRT_RELEASE(K.pso_reflect);
    KRT_RELEASE(K.pso_calibrate); KRT_RELEASE(K.pso_calib_update); KRT_RELEASE(K.pso_light); KRT_RELEASE(K.pso_reflect_filter); KRT_RELEASE(K.pso_light_filter); KRT_RELEASE(K.sums); KRT_RELEASE(K.factors); KRT_RELEASE(K.root); KRT_RELEASE(K.heap);
    KRT_RELEASE(K.list);
    for (i = 0; i < KRT_SLOTS; i++) KRT_RELEASE(K.alloc[i]);
    KRT_RELEASE(K.qread); KRT_RELEASE(K.qheap); K.times = 0;
    KRT_RELEASE(K.queue); KRT_RELEASE(K.fence); KRT_RELEASE(K.dev);
    KRT_RELEASE(K.fence11); KRT_RELEASE(K.ctx11); KRT_RELEASE(K.dev11);
    if (K.fence_h) { CloseHandle(K.fence_h); K.fence_h = NULL; }
    if (K.fence_event) { CloseHandle(K.fence_event); K.fence_event = NULL; }
    if (K.lib) { FreeLibrary(K.lib); K.lib = NULL; }
    free(K.geoms); K.geoms = NULL;
    K.ok = false;
}

static void barrier(ID3D12Resource *r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b;
    memset(&b, 0, sizeof(b));
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    ID3D12GraphicsCommandList4_ResourceBarrier(K.list, 1, &b);
}

static void uav_barrier(ID3D12Resource *r)
{
    D3D12_RESOURCE_BARRIER b;
    memset(&b, 0, sizeof(b));
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = r;
    ID3D12GraphicsCommandList4_ResourceBarrier(K.list, 1, &b);
}

static void wait_fence(uint64_t value)
{
    if (ID3D12Fence_GetCompletedValue(K.fence) >= value) return;
    ID3D12Fence_SetEventOnCompletion(K.fence, value, K.fence_event);
    WaitForSingleObject(K.fence_event, INFINITE);
}

/* Closes and runs the open command list, then signals the shared fence; the value it will reach. */
static uint64_t submit(void)
{
    ID3D12CommandList *lists[1];
    ID3D12GraphicsCommandList4_Close(K.list);
    lists[0] = (ID3D12CommandList *)K.list;
    ID3D12CommandQueue_ExecuteCommandLists(K.queue, 1, lists);
    ID3D12CommandQueue_Signal(K.queue, K.fence, ++K.fence_value);
    return K.fence_value;
}

/* The pass times this slot measured last time it ran (its GPU work is done by now). */
static void collect_times(void)
{
    static const char *const names[KRT_STAMPS - 1] = { "unpack", "build", "calibrate", "lighting", "reflections", "view" };
    const uint64_t *q = K.qread_cpu + (size_t)K.slot * 16;
    unsigned i;
    if (!K.times || !K.qpending[K.slot]) return;
    K.qpending[K.slot] = false;
    for (i = 1; i < KRT_STAMPS; i++) K.qsum[i - 1] += (double)(q[i] - q[i - 1]) * 1000.0 / (double)K.qfreq;
    if (++K.qframes == 120) {
        double total = 0.0;
        fprintf(stderr, "[KRT] GPU ms a frame:");
        for (i = 0; i + 1 < KRT_STAMPS; i++) { fprintf(stderr, " %s %.2f", names[i], K.qsum[i] / 120.0); total += K.qsum[i]; K.qsum[i] = 0.0; }
        fprintf(stderr, " = %.2f\n", total / 120.0);
        K.qframes = 0;
    }
}

static void stamp(unsigned i)
{
    if (K.times) ID3D12GraphicsCommandList4_EndQuery(K.list, K.qheap, D3D12_QUERY_TYPE_TIMESTAMP, K.slot * 16 + i);
}

/* Opens the command list on the next allocator, once the GPU is done with that allocator's last frame. */
static bool open_list(void)
{
    K.slot = (K.slot + 1) % KRT_SLOTS;
    wait_fence(K.alloc_done[K.slot]);
    collect_times();
    return SUCCEEDED(ID3D12CommandAllocator_Reset(K.alloc[K.slot])) &&
           SUCCEEDED(ID3D12GraphicsCommandList4_Reset(K.list, K.alloc[K.slot], NULL));
}

/* D3D12 copies a pattern into a texture D3D11 created; D3D11 waits on the shared fence and reads it back. */
static bool interop_check(ID3D11Device *dev, ID3D11DeviceContext *ctx)
{
    enum { W = 64, H = 64 };
    D3D11_TEXTURE2D_DESC td;
    ID3D11Texture2D *tex = NULL, *stage = NULL;
    IDXGIResource1 *r1 = NULL;
    HANDLE h = NULL;
    ID3D12Resource *shared = NULL, *upload = NULL;
    D3D12_TEXTURE_COPY_LOCATION dst, src;
    D3D11_MAPPED_SUBRESOURCE ms;
    uint32_t *p;
    uint64_t value;
    bool good = false;
    unsigned x, y;

    memset(&td, 0, sizeof(td));
    td.Width = W; td.Height = H; td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    if (FAILED(ID3D11Device_CreateTexture2D(dev, &td, NULL, &tex)) ||
        FAILED(ID3D11Texture2D_QueryInterface(tex, &IID_IDXGIResource1, (void **)&r1)) ||
        FAILED(IDXGIResource1_CreateSharedHandle(r1, NULL, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, NULL, &h)) ||
        FAILED(ID3D12Device5_OpenSharedHandle(K.dev, h, &IID_ID3D12Resource, (void **)&shared))) {
        fprintf(stderr, "[KRT] interop: D3D12 cannot open a D3D11 texture\n");
        goto done;
    }
    upload = make_buffer(W * H * 4, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!upload || FAILED(ID3D12Resource_Map(upload, 0, NULL, (void **)&p))) goto done;
    for (y = 0; y < H; y++) for (x = 0; x < W; x++) p[y * W + x] = 0xFF000000u | (y << 8) | x;
    ID3D12Resource_Unmap(upload, 0, NULL);

    if (!open_list()) goto done;
    memset(&dst, 0, sizeof(dst)); memset(&src, 0, sizeof(src));
    dst.pResource = shared; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = upload; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src.PlacedFootprint.Footprint.Width = W; src.PlacedFootprint.Footprint.Height = H;
    src.PlacedFootprint.Footprint.Depth = 1; src.PlacedFootprint.Footprint.RowPitch = W * 4;
    barrier(shared, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12GraphicsCommandList4_CopyTextureRegion(K.list, &dst, 0, 0, 0, &src, NULL);
    barrier(shared, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    value = submit();
    K.alloc_done[K.slot] = value;

    td.Usage = D3D11_USAGE_STAGING; td.BindFlags = 0; td.CPUAccessFlags = D3D11_CPU_ACCESS_READ; td.MiscFlags = 0;
    if (FAILED(ID3D11Device_CreateTexture2D(dev, &td, NULL, &stage))) goto done;
    ID3D11DeviceContext4_Wait(K.ctx11, K.fence11, value);
    ID3D11DeviceContext_CopyResource(ctx, (ID3D11Resource *)stage, (ID3D11Resource *)tex);
    if (FAILED(ID3D11DeviceContext_Map(ctx, (ID3D11Resource *)stage, 0, D3D11_MAP_READ, 0, &ms))) goto done;
    good = true;
    for (y = 0; y < H && good; y++) {
        const uint32_t *row = (const uint32_t *)((const uint8_t *)ms.pData + (size_t)y * ms.RowPitch);
        for (x = 0; x < W; x++) if (row[x] != (0xFF000000u | (y << 8) | x)) { good = false; break; }
    }
    ID3D11DeviceContext_Unmap(ctx, (ID3D11Resource *)stage, 0);
    if (!good) fprintf(stderr, "[KRT] interop: D3D11 did not read what D3D12 wrote\n");
    wait_fence(value);

done:
    KRT_RELEASE(stage); KRT_RELEASE(upload); KRT_RELEASE(shared); KRT_RELEASE(r1); KRT_RELEASE(tex);
    if (h) CloseHandle(h);
    return good;
}

static ID3D12PipelineState *make_pso(const void *code, size_t n)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC d;
    ID3D12PipelineState *p = NULL;
    memset(&d, 0, sizeof(d));
    d.pRootSignature = K.root;
    d.CS.pShaderBytecode = code;
    d.CS.BytecodeLength = n;
    if (FAILED(ID3D12Device5_CreateComputePipelineState(K.dev, &d, &IID_ID3D12PipelineState, (void **)&p))) return NULL;
    return p;
}

/* Root signature: b0 the constants, a table of twelve shader resources, one of eight unordered-access views, and the
 * game's textures (space 1) with a wrapping linear sampler. */
static bool make_pipelines(void)
{
#ifdef KRT_HAVE_SHADERS
    PfnSerializeRoot serialize = (PfnSerializeRoot)(void *)GetProcAddress(K.lib, "D3D12SerializeRootSignature");
    D3D12_DESCRIPTOR_RANGE ranges[3];
    D3D12_ROOT_PARAMETER params[4];
    D3D12_STATIC_SAMPLER_DESC smp;
    D3D12_ROOT_SIGNATURE_DESC rs;
    D3D12_DESCRIPTOR_HEAP_DESC hd;
    ID3DBlob *blob = NULL, *errors = NULL;
    unsigned i;
    if (!serialize) return false;
    memset(ranges, 0, sizeof(ranges)); memset(params, 0, sizeof(params)); memset(&rs, 0, sizeof(rs));
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors = 12;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors = 8;
    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[2].NumDescriptors = KRT_TEX_MAX; ranges[2].RegisterSpace = 1;
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (i = 0; i < 3; i++) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1 + i].DescriptorTable.NumDescriptorRanges = 1;
        params[1 + i].DescriptorTable.pDescriptorRanges = &ranges[i];
        params[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    memset(&smp, 0, sizeof(smp));
    smp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    smp.AddressU = smp.AddressV = smp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    smp.MaxLOD = D3D12_FLOAT32_MAX;
    smp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rs.NumParameters = 4; rs.pParameters = params;
    rs.NumStaticSamplers = 1; rs.pStaticSamplers = &smp;
    if (FAILED(serialize(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors)) ||
        FAILED(ID3D12Device5_CreateRootSignature(K.dev, 0, ID3D10Blob_GetBufferPointer(blob), ID3D10Blob_GetBufferSize(blob),
                                                 &IID_ID3D12RootSignature, (void **)&K.root))) {
        if (errors) fprintf(stderr, "[KRT] root signature: %s\n", (const char *)ID3D10Blob_GetBufferPointer(errors));
        KRT_RELEASE(blob); KRT_RELEASE(errors);
        return false;
    }
    KRT_RELEASE(blob); KRT_RELEASE(errors);
    K.pso_unpack = make_pso(krt_unpack_cs, sizeof(krt_unpack_cs));
    K.pso_view = make_pso(krt_view_cs, sizeof(krt_view_cs));
    K.pso_reflect = make_pso(krt_reflect_cs, sizeof(krt_reflect_cs));
    K.pso_calibrate = make_pso(krt_calibrate_cs, sizeof(krt_calibrate_cs));
    K.pso_calib_update = make_pso(krt_calib_update_cs, sizeof(krt_calib_update_cs));
    K.pso_light = make_pso(krt_light_cs, sizeof(krt_light_cs));
    K.pso_reflect_filter = make_pso(krt_reflect_filter_cs, sizeof(krt_reflect_filter_cs));
    K.pso_light_filter = make_pso(krt_light_filter_cs, sizeof(krt_light_filter_cs));
    if (!K.pso_unpack || !K.pso_view || !K.pso_reflect || !K.pso_calibrate || !K.pso_calib_update || !K.pso_light ||
        !K.pso_reflect_filter || !K.pso_light_filter) { fprintf(stderr, "[KRT] compute pipelines failed\n"); return false; }

    memset(&hd, 0, sizeof(hd));
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = H_COUNT;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(ID3D12Device5_CreateDescriptorHeap(K.dev, &hd, &IID_ID3D12DescriptorHeap, (void **)&K.heap))) return false;
    K.inc = ID3D12Device5_GetDescriptorHandleIncrementSize(K.dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    for (i = 0; i < H_COUNT; i++) {
        if (i >= H_UNPACK_UAV && i < H_TEX) null_uav(i);
        else if (i >= H_TEX) null_tex(i);
        else null_srv(i);
    }

    K.draws = make_buffer((UINT64)KRT_SLOTS * KRT_DRAWS_MAX * 16, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    K.consts = make_buffer((UINT64)KRT_SLOTS * 512, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    K.instance = make_buffer((UINT64)KRT_SLOTS * 2 * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!K.draws || !K.consts || !K.instance ||
        FAILED(ID3D12Resource_Map(K.draws, 0, NULL, (void **)&K.draws_cpu)) ||
        FAILED(ID3D12Resource_Map(K.consts, 0, NULL, (void **)&K.consts_cpu)) ||
        FAILED(ID3D12Resource_Map(K.instance, 0, NULL, (void **)&K.instance_cpu)))
        return false;
    K.sums = make_buffer((UINT64)KRT_FACTOR_KEYS * 32, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    K.factors = make_buffer((UINT64)KRT_FACTOR_KEYS * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!K.sums || !K.factors) return false;
    raw_uav(H_TRACE_UAV + 1, K.sums, (UINT64)KRT_FACTOR_KEYS * 32);
    raw_uav(H_TRACE_UAV + 2, K.factors, (UINT64)KRT_FACTOR_KEYS * 16);
    K.geoms = (D3D12_RAYTRACING_GEOMETRY_DESC *)calloc(KRT_DRAWS_MAX, sizeof(*K.geoms));
    return K.geoms != NULL;
#else
    fprintf(stderr, "[KRT] this build has no ray tracing shaders (dxc was not found)\n");
    return false;
#endif
}

bool krt_supported(LUID adapter_luid)
{
#ifdef KRT_HAVE_SHADERS
    HMODULE lib = LoadLibraryA("d3d12.dll");
    PfnCreateDevice create = lib ? (PfnCreateDevice)(void *)GetProcAddress(lib, "D3D12CreateDevice") : NULL;
    IDXGIFactory4 *factory = NULL;
    IDXGIAdapter1 *adapter = NULL;
    ID3D12Device5 *dev = NULL;
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5;
    bool ok = false;
    if (create && SUCCEEDED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&factory)) &&
        SUCCEEDED(IDXGIFactory4_EnumAdapterByLuid(factory, adapter_luid, &IID_IDXGIAdapter1, (void **)&adapter)) &&
        SUCCEEDED(create((IUnknown *)adapter, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device5, (void **)&dev))) {
        memset(&o5, 0, sizeof(o5));
        ok = SUCCEEDED(ID3D12Device5_CheckFeatureSupport(dev, D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) &&
             o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
    }
    KRT_RELEASE(dev); KRT_RELEASE(adapter); KRT_RELEASE(factory);
    return ok;
#else
    (void)adapter_luid;
    return false;
#endif
}

bool krt_init(ID3D11Device *dev, ID3D11DeviceContext *ctx, LUID adapter_luid)
{
    PfnCreateDevice create;
    IDXGIFactory4 *factory = NULL;
    IDXGIAdapter1 *adapter = NULL;
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5;
    D3D12_COMMAND_QUEUE_DESC qd;
    DXGI_ADAPTER_DESC1 ad;
    HRESULT hr;
    unsigned i;

    if (K.ok) return true;
    K.lib = LoadLibraryA("d3d12.dll");
    create = K.lib ? (PfnCreateDevice)(void *)GetProcAddress(K.lib, "D3D12CreateDevice") : NULL;
    if (!create) { fprintf(stderr, "[KRT] no Direct3D 12: ray tracing off\n"); krt_shutdown(); return false; }
    if (getenv("RECOMP_KGPU_RT_DEBUG")) {
        PfnGetDebug get = (PfnGetDebug)(void *)GetProcAddress(K.lib, "D3D12GetDebugInterface");
        ID3D12Debug *debug = NULL;
        if (get && SUCCEEDED(get(&IID_ID3D12Debug, (void **)&debug))) { ID3D12Debug_EnableDebugLayer(debug); ID3D12Debug_Release(debug); }
    }
    if (getenv("RECOMP_KGPU_RT_DRED")) {
        /* what the GPU was doing when the device is lost: reported by device_lost */
        PfnGetDebug get = (PfnGetDebug)(void *)GetProcAddress(K.lib, "D3D12GetDebugInterface");
        ID3D12DeviceRemovedExtendedDataSettings *dred = NULL;
        if (get && SUCCEEDED(get(&IID_ID3D12DeviceRemovedExtendedDataSettings, (void **)&dred))) {
            ID3D12DeviceRemovedExtendedDataSettings_SetAutoBreadcrumbsEnablement(dred, D3D12_DRED_ENABLEMENT_FORCED_ON);
            ID3D12DeviceRemovedExtendedDataSettings_SetPageFaultEnablement(dred, D3D12_DRED_ENABLEMENT_FORCED_ON);
            ID3D12DeviceRemovedExtendedDataSettings_Release(dred);
        }
    }
    if (FAILED(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&factory)) ||
        FAILED(IDXGIFactory4_EnumAdapterByLuid(factory, adapter_luid, &IID_IDXGIAdapter1, (void **)&adapter))) {
        fprintf(stderr, "[KRT] the renderer's adapter is not visible to DXGI: ray tracing off\n");
        KRT_RELEASE(adapter); KRT_RELEASE(factory); krt_shutdown(); return false;
    }
    IDXGIAdapter1_GetDesc1(adapter, &ad);
    hr = create((IUnknown *)adapter, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device5, (void **)&K.dev);
    KRT_RELEASE(adapter); KRT_RELEASE(factory);
    if (FAILED(hr)) { fprintf(stderr, "[KRT] D3D12 device failed 0x%08lX: ray tracing off\n", (unsigned long)hr); krt_shutdown(); return false; }

    memset(&o5, 0, sizeof(o5));
    if (FAILED(ID3D12Device5_CheckFeatureSupport(K.dev, D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) ||
        o5.RaytracingTier < D3D12_RAYTRACING_TIER_1_1) {
        fprintf(stderr, "[KRT] %ls has no DXR 1.1 (tier %d): ray tracing off\n", ad.Description, (int)o5.RaytracingTier);
        krt_shutdown(); return false;
    }

    memset(&qd, 0, sizeof(qd));
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(ID3D12Device5_CreateCommandQueue(K.dev, &qd, &IID_ID3D12CommandQueue, (void **)&K.queue))) { krt_shutdown(); return false; }
    for (i = 0; i < KRT_SLOTS; i++)
        if (FAILED(ID3D12Device5_CreateCommandAllocator(K.dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void **)&K.alloc[i]))) {
            krt_shutdown(); return false;
        }
    if (FAILED(ID3D12Device5_CreateCommandList(K.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, K.alloc[0], NULL,
                                               &IID_ID3D12GraphicsCommandList4, (void **)&K.list))) {
        fprintf(stderr, "[KRT] D3D12 command objects failed: ray tracing off\n");
        krt_shutdown(); return false;
    }
    ID3D12GraphicsCommandList4_Close(K.list);
    K.fence_event = CreateEventA(NULL, FALSE, FALSE, NULL);
    if (getenv("RECOMP_KGPU_RT_TIMES")) {
        D3D12_QUERY_HEAP_DESC qd2;
        memset(&qd2, 0, sizeof(qd2));
        qd2.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qd2.Count = KRT_SLOTS * 16;
        if (SUCCEEDED(ID3D12Device5_CreateQueryHeap(K.dev, &qd2, &IID_ID3D12QueryHeap, (void **)&K.qheap)) &&
            (K.qread = make_buffer((UINT64)KRT_SLOTS * 16 * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)) != NULL &&
            SUCCEEDED(ID3D12Resource_Map(K.qread, 0, NULL, (void **)&K.qread_cpu)) &&
            SUCCEEDED(ID3D12CommandQueue_GetTimestampFrequency(K.queue, &K.qfreq)))
            K.times = 1;
    }

    if (FAILED(ID3D11Device_QueryInterface(dev, &IID_ID3D11Device5, (void **)&K.dev11)) ||
        FAILED(ID3D11DeviceContext_QueryInterface(ctx, &IID_ID3D11DeviceContext4, (void **)&K.ctx11)) ||
        FAILED(ID3D11Device5_CreateFence(K.dev11, 0, D3D11_FENCE_FLAG_SHARED, &IID_ID3D11Fence, (void **)&K.fence11)) ||
        FAILED(ID3D11Fence_CreateSharedHandle(K.fence11, NULL, GENERIC_ALL, NULL, &K.fence_h)) ||
        FAILED(ID3D12Device5_OpenSharedHandle(K.dev, K.fence_h, &IID_ID3D12Fence, (void **)&K.fence))) {
        fprintf(stderr, "[KRT] D3D11 and D3D12 cannot share a fence: ray tracing off\n");
        krt_shutdown(); return false;
    }
    K.fence_value = 0;

    if (!interop_check(dev, ctx) || !make_pipelines()) { krt_shutdown(); return false; }
    K.ok = true;
    fprintf(stderr, "[KRT] ray tracing available: %ls, DXR tier %d.%d, D3D11 and D3D12 share textures and a fence\n",
            ad.Description, (int)o5.RaytracingTier / 10, (int)o5.RaytracingTier % 10);
    return true;
}

bool krt_set_geometry(HANDLE texture, unsigned width, unsigned height)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d;
    if (!K.ok) return false;
    KRT_RELEASE(K.geo);
    release_scene();
    if (FAILED(ID3D12Device5_OpenSharedHandle(K.dev, texture, &IID_ID3D12Resource, (void **)&K.geo))) {
        fprintf(stderr, "[KRT] D3D12 cannot open the geometry texture\n");
        return false;
    }
    K.geo_w = width; K.geo_h = height; K.max_verts = width * height / 2;
    K.pos = make_buffer((UINT64)K.max_verts * 12, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    K.uv = make_buffer((UINT64)K.max_verts * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    K.col = make_buffer((UINT64)K.max_verts * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!K.pos || !K.uv || !K.col) { fprintf(stderr, "[KRT] vertex buffers failed\n"); return false; }
    K.vb_readable = false;
    memset(&d, 0, sizeof(d));
    d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    ID3D12Device5_CreateShaderResourceView(K.dev, K.geo, &d, cpu_handle(H_UNPACK_SRV));
    raw_uav(H_UNPACK_UAV + 0, K.pos, (UINT64)K.max_verts * 12);
    raw_uav(H_UNPACK_UAV + 1, K.uv, (UINT64)K.max_verts * 16);
    raw_uav(H_UNPACK_UAV + 2, K.col, (UINT64)K.max_verts * 4);
    write_trace_tables();
    return true;
}

bool krt_set_output(HANDLE texture, unsigned width, unsigned height)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC d;
    if (!K.ok) return false;
    wait_fence(K.fence_value);
    KRT_RELEASE(K.out);
    if (FAILED(ID3D12Device5_OpenSharedHandle(K.dev, texture, &IID_ID3D12Resource, (void **)&K.out))) {
        fprintf(stderr, "[KRT] D3D12 cannot open the output texture\n");
        return false;
    }
    K.out_w = width; K.out_h = height;
    memset(&d, 0, sizeof(d));
    d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    ID3D12Device5_CreateUnorderedAccessView(K.dev, K.out, NULL, &d, cpu_handle(H_TRACE_UAV));
    return true;
}

/* Removed textures whose last frame the GPU has finished: released, and their index free again. all: every one
 * (nothing is in flight, and the frame being captured is not recorded yet). */
static void release_textures(bool all)
{
    const uint64_t done = ID3D12Fence_GetCompletedValue(K.fence);
    unsigned i = 0;
    while (i < K.npending) {
        if (!all && K.pending[i].value > done) { i++; continue; }
        KRT_RELEASE(K.tex[K.pending[i].index]);
        null_tex(H_TEX + (unsigned)K.pending[i].index);
        K.pending[i] = K.pending[--K.npending];
    }
}

int krt_tex_add(HANDLE texture)
{
    unsigned i;
    bool busy;
    if (!K.ok || !texture) return -1;
    release_textures(false);
    for (i = 0; i < KRT_TEX_MAX; i++) {
        unsigned k;
        if (K.tex[i]) continue;
        for (busy = false, k = 0; k < K.npending; k++) if (K.pending[k].index == (int)i) { busy = true; break; }
        if (!busy) break;
    }
    if (i == KRT_TEX_MAX) return -1;
    if (FAILED(ID3D12Device5_OpenSharedHandle(K.dev, texture, &IID_ID3D12Resource, (void **)&K.tex[i]))) {
        K.tex[i] = NULL;
        return -1;
    }
    ID3D12Device5_CreateShaderResourceView(K.dev, K.tex[i], NULL, cpu_handle(H_TEX + i));
    return (int)i;
}

void krt_tex_remove(int index)
{
    if (!K.ok || index < 0 || index >= (int)KRT_TEX_MAX || !K.tex[index]) return;
    if (K.npending == KRT_TEX_PENDING) { wait_fence(K.fence_value); release_textures(true); }
    K.pending[K.npending].index = index;
    K.pending[K.npending].value = UINT64_MAX;
    K.npending++;
}

static void release_inputs(void)
{
    unsigned v, i;
    KRT_RELEASE(K.in_pic); KRT_RELEASE(K.in_w); KRT_RELEASE(K.in_g); KRT_RELEASE(K.in_depth);
    KRT_RELEASE(K.light_out); KRT_RELEASE(K.history[0]); KRT_RELEASE(K.history[1]);
    KRT_RELEASE(K.refl_raw); KRT_RELEASE(K.refl_hist);
    for (v = 0; v < KRT_HISTORIES; v++) {
        for (i = 0; i < 3; i++) KRT_RELEASE(K.hist[v].t[i]);
        K.hist[v].kept = false;
    }
}

/* A history texture: the lighting pass's two at half the output's size, the reflections' at its size. */
static bool make_history(unsigned i, D3D12_RESOURCE_STATES state, ID3D12Resource **out)
{
    D3D12_HEAP_PROPERTIES hp;
    D3D12_RESOURCE_DESC rd;
    memset(&hp, 0, sizeof(hp));
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = i < 2 ? (K.out_w + 1) / 2 : K.out_w;
    rd.Height = i < 2 ? (K.out_h + 1) / 2 : K.out_h;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return SUCCEEDED(ID3D12Device5_CreateCommittedResource(K.dev, &hp, D3D12_HEAP_FLAG_NONE, &rd, state, NULL,
                                                          &IID_ID3D12Resource, (void **)out));
}

static void copy_texture(ID3D12Resource *to, D3D12_RESOURCE_STATES to_state, ID3D12Resource *from, D3D12_RESOURCE_STATES from_state)
{
    barrier(to, to_state, D3D12_RESOURCE_STATE_COPY_DEST);
    barrier(from, from_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    ID3D12GraphicsCommandList4_CopyResource(K.list, to, from);
    barrier(to, D3D12_RESOURCE_STATE_COPY_DEST, to_state);
    barrier(from, D3D12_RESOURCE_STATE_COPY_SOURCE, from_state);
}

/* Before the passes: the history in the working textures becomes `to`'s, the one they held is kept for its view. */
static void switch_history(int to)
{
    ID3D12Resource *work[3];
    int from = K.hist_cur;
    unsigned i;
    if (to < 0 || to >= KRT_HISTORIES || to == from) return;
    work[0] = K.history[0]; work[1] = K.history[1]; work[2] = K.refl_hist;
    for (i = 0; i < 3; i++) {
        if (!work[i]) return;
        if (!K.hist[from].t[i] && !make_history(i, D3D12_RESOURCE_STATE_COMMON, &K.hist[from].t[i])) return;
    }
    for (i = 0; i < 3; i++)
        copy_texture(K.hist[from].t[i], D3D12_RESOURCE_STATE_COMMON, work[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    memcpy(K.hist[from].prev_vp, K.prev_vp, sizeof(K.prev_vp));
    K.hist[from].have_prev = K.have_prev;
    K.hist[from].fresh = K.history_fresh;
    K.hist[from].kept = true;
    if (K.hist[to].kept) {
        for (i = 0; i < 3; i++)
            copy_texture(work[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, K.hist[to].t[i], D3D12_RESOURCE_STATE_COMMON);
        memcpy(K.prev_vp, K.hist[to].prev_vp, sizeof(K.prev_vp));
        K.have_prev = K.hist[to].have_prev;
        K.history_fresh = K.hist[to].fresh;
    } else {
        K.have_prev = false;
        K.history_fresh = true;
    }
    K.hist_cur = to;
}

bool krt_set_inputs(HANDLE picture, HANDLE weight, HANDLE surface, HANDLE depth, HANDLE light)
{
    D3D12_HEAP_PROPERTIES hp;
    D3D12_RESOURCE_DESC rd;
    D3D12_UNORDERED_ACCESS_VIEW_DESC u;
    unsigned i;
    if (!K.ok) return false;
    wait_fence(K.fence_value);
    release_inputs();
    if (FAILED(ID3D12Device5_OpenSharedHandle(K.dev, picture, &IID_ID3D12Resource, (void **)&K.in_pic)) ||
        FAILED(ID3D12Device5_OpenSharedHandle(K.dev, weight, &IID_ID3D12Resource, (void **)&K.in_w)) ||
        FAILED(ID3D12Device5_OpenSharedHandle(K.dev, surface, &IID_ID3D12Resource, (void **)&K.in_g)) ||
        FAILED(ID3D12Device5_OpenSharedHandle(K.dev, depth, &IID_ID3D12Resource, (void **)&K.in_depth)) ||
        FAILED(ID3D12Device5_OpenSharedHandle(K.dev, light, &IID_ID3D12Resource, (void **)&K.light_out))) {
        fprintf(stderr, "[KRT] D3D12 cannot open the passes' inputs\n");
        release_inputs();
        write_trace_tables();
        return false;
    }
    /* the lighting pass's history (at half the size, as the pass runs) and the reflections', their own */
    memset(&hp, 0, sizeof(hp));
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width = K.out_w; rd.Height = K.out_h; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    for (i = 0; i < 4; i++) {
        rd.Width = i < 2 ? (K.out_w + 1) / 2 : K.out_w;
        rd.Height = i < 2 ? (K.out_h + 1) / 2 : K.out_h;
        if (FAILED(ID3D12Device5_CreateCommittedResource(K.dev, &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, NULL,
                                                         &IID_ID3D12Resource,
                                                         (void **)(i < 2 ? &K.history[i] : i == 2 ? &K.refl_raw : &K.refl_hist)))) {
            fprintf(stderr, "[KRT] the lighting history failed\n");
            release_inputs();
            write_trace_tables();
            return false;
        }
    }
    memset(&u, 0, sizeof(u));
    u.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    ID3D12Device5_CreateUnorderedAccessView(K.dev, K.light_out, NULL, &u, cpu_handle(H_TRACE_UAV + 3));
    ID3D12Device5_CreateUnorderedAccessView(K.dev, K.history[0], NULL, &u, cpu_handle(H_TRACE_UAV + 4));
    ID3D12Device5_CreateUnorderedAccessView(K.dev, K.history[1], NULL, &u, cpu_handle(H_TRACE_UAV + 5));
    ID3D12Device5_CreateUnorderedAccessView(K.dev, K.refl_raw, NULL, &u, cpu_handle(H_TRACE_UAV + 6));
    ID3D12Device5_CreateUnorderedAccessView(K.dev, K.refl_hist, NULL, &u, cpu_handle(H_TRACE_UAV + 7));
    K.history_fresh = true;
    write_trace_tables();
    return true;
}

/* Grows a buffer for an acceleration structure (or its scratch) to at least `need`, with room to spare so a
 * slightly bigger frame does not grow it again. The GPU is idle first: a frame in flight may still use the old one. */
/* With RECOMP_KGPU_RT_DRED: the operations (D3D12_AUTO_BREADCRUMB_OP) of each command list the GPU did not finish,
 * the one it stopped at in [], and the address of a page fault. */
static void report_dred(void)
{
    ID3D12DeviceRemovedExtendedData *d = NULL;
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc;
    D3D12_DRED_PAGE_FAULT_OUTPUT pf;
    const D3D12_AUTO_BREADCRUMB_NODE *node;
    if (FAILED(ID3D12Device5_QueryInterface(K.dev, &IID_ID3D12DeviceRemovedExtendedData, (void **)&d))) return;
    if (SUCCEEDED(ID3D12DeviceRemovedExtendedData_GetAutoBreadcrumbsOutput(d, &bc)))
        for (node = bc.pHeadAutoBreadcrumbNode; node; node = node->pNext) {
            UINT32 done = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0, i;
            if (done >= node->BreadcrumbCount) continue;
            fprintf(stderr, "[KRT] DRED: a list stopped after %u of %u operations:", done, node->BreadcrumbCount);
            for (i = 0; i < node->BreadcrumbCount; i++) fprintf(stderr, i == done ? " [%d]" : " %d", (int)node->pCommandHistory[i]);
            fprintf(stderr, "\n");
        }
    if (SUCCEEDED(ID3D12DeviceRemovedExtendedData_GetPageFaultAllocationOutput(d, &pf)) && pf.PageFaultVA)
        fprintf(stderr, "[KRT] DRED: page fault at GPU address 0x%llx (%s)\n", (unsigned long long)pf.PageFaultVA,
                pf.pHeadExistingAllocationNode ? "inside a live allocation" :
                pf.pHeadRecentFreedAllocationNode ? "inside a freed allocation" : "outside any allocation");
    ID3D12DeviceRemovedExtendedData_Release(d);
}

/* The device gone (a GPU fault or reset): ray tracing turned off for the rest of the run. */
static bool device_lost(void)
{
    HRESULT hr = ID3D12Device5_GetDeviceRemovedReason(K.dev);
    if (SUCCEEDED(hr)) return false;
    fprintf(stderr, "[KRT] the D3D12 device was lost (0x%08lX): ray tracing off\n", (unsigned long)hr);
    report_dred();
    K.ok = false;
    return true;
}

static bool ensure_as(ID3D12Resource **r, UINT64 *size, UINT64 need, bool scratch)
{
    if (*r && *size >= need) return true;
    wait_fence(K.fence_value);
    KRT_RELEASE(*r);
    need += need / 4;
    need = (need + 65535) & ~(UINT64)65535;
    *r = make_buffer(need, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                     scratch ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    *size = *r ? need : 0;
    return *r != NULL;
}

bool krt_frame(const KrtFrame *f)
{
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bl[2], tl;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO bpi[2], tpi;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build;
    D3D12_GPU_VIRTUAL_ADDRESS cva;
    ID3D12DescriptorHeap *heaps[1];
    KrtUnpackCb *uc;
    KrtFrameCb *fc;
    uint64_t ready, done, scratch;
    unsigned i, n, k, ns, group[2];
    bool moved = false, wrote = false;

    if (!K.ok || device_lost() || !K.geo || !f->nverts || !f->ndraws || f->nverts > K.max_verts) return false;
    release_textures(false);
    n = f->ndraws < KRT_DRAWS_MAX ? f->ndraws : KRT_DRAWS_MAX;

    /* The scene this frame: one geometry a draw, its triangles where the unpack pass puts them; the static draws
     * (the level, drawn with the camera's matrix alone) first, then the dynamic ones (with a transform of their own:
     * characters, props, debris). Each group is a bottom level of its own under an instance of its own, so a ray can
     * look for one group only (its instance mask: 1 static, 2 dynamic), and the instance's ID is where its draws
     * start. */
    for (ns = 0, i = 0; i < n; i++) if (!(f->flags[i] & KRT_DYNAMIC)) K.order[ns++] = i;
    for (k = ns, i = 0; i < n; i++) if (f->flags[i] & KRT_DYNAMIC) K.order[k++] = i;
    for (k = 0; k < n; k++) {
        D3D12_RAYTRACING_GEOMETRY_DESC *g = &K.geoms[k];
        i = K.order[k];
        memset(g, 0, sizeof(*g));
        g->Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g->Flags = (f->flags[i] & KRT_ALPHA_TEST) ? D3D12_RAYTRACING_GEOMETRY_FLAG_NONE : D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        g->Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        g->Triangles.VertexCount = f->count[i];
        g->Triangles.VertexBuffer.StartAddress = ID3D12Resource_GetGPUVirtualAddress(K.pos) + (UINT64)f->first[i] * 12;
        g->Triangles.VertexBuffer.StrideInBytes = 12;
    }
    group[0] = ns; group[1] = n - ns;
    for (k = 0; k < 2; k++) {
        memset(&bl[k], 0, sizeof(bl[k]));
        bl[k].Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        bl[k].Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        bl[k].NumDescs = group[k];
        bl[k].DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        bl[k].pGeometryDescs = K.geoms + (k ? ns : 0);
        memset(&bpi[k], 0, sizeof(bpi[k]));
        if (group[k]) ID3D12Device5_GetRaytracingAccelerationStructurePrebuildInfo(K.dev, &bl[k], &bpi[k]);
    }
    memset(&tl, 0, sizeof(tl));
    tl.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tl.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    tl.NumDescs = 2;
    tl.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    ID3D12Device5_GetRaytracingAccelerationStructurePrebuildInfo(K.dev, &tl, &tpi);
    scratch = tpi.ScratchDataSizeInBytes;
    for (k = 0; k < 2; k++) if (bpi[k].ScratchDataSizeInBytes > scratch) scratch = bpi[k].ScratchDataSizeInBytes;
    if (!K.blas[0] || K.blas_size[0] < bpi[0].ResultDataMaxSizeInBytes || !K.blas[1] || K.blas_size[1] < bpi[1].ResultDataMaxSizeInBytes ||
        !K.tlas || K.tlas_size < tpi.ResultDataMaxSizeInBytes)
        moved = true;
    if (!ensure_as(&K.blas[0], &K.blas_size[0], bpi[0].ResultDataMaxSizeInBytes + 256, false) ||
        !ensure_as(&K.blas[1], &K.blas_size[1], bpi[1].ResultDataMaxSizeInBytes + 256, false) ||
        !ensure_as(&K.tlas, &K.tlas_size, tpi.ResultDataMaxSizeInBytes, false) ||
        !ensure_as(&K.scratch, &K.scratch_size, scratch, true)) {
        static unsigned reported;
        if (!device_lost() && reported++ < 3)
            fprintf(stderr, "[KRT] acceleration structure memory failed: %u draws, %u vertices, %llu + %llu + %llu KiB\n", n, f->nverts,
                    (unsigned long long)(bpi[0].ResultDataMaxSizeInBytes >> 10), (unsigned long long)(bpi[1].ResultDataMaxSizeInBytes >> 10),
                    (unsigned long long)(scratch >> 10));
        return false;
    }
    if (moved) write_trace_tables();

    if (!open_list()) return false;
    switch_history(f->history);
    /* What D3D11 wrote before this point -- the geometry texture -- is done once the fence passes this value. */
    ready = ++K.fence_value;
    ID3D11DeviceContext4_Signal(K.ctx11, K.fence11, ready);
    ID3D12CommandQueue_Wait(K.queue, K.fence, ready);

    uc = (KrtUnpackCb *)(K.consts_cpu + K.slot * 512);
    uc->n = f->nverts; uc->width = K.geo_w;
    fc = (KrtFrameCb *)(K.consts_cpu + K.slot * 512 + 256);
    memcpy(fc->inv, f->inv, sizeof(fc->inv));
    memcpy(fc->c58, f->c58, sizeof(fc->c58));
    memcpy(fc->c59, f->c59, sizeof(fc->c59));
    fc->size[0] = (float)K.out_w; fc->size[1] = (float)K.out_h; fc->size[2] = f->guest_w; fc->size[3] = f->guest_h;
    memcpy(fc->vp0, f->vp, sizeof(fc->vp0));
    memcpy(fc->vp1, f->vp + 4, sizeof(fc->vp1));
    memcpy(fc->vp3, f->vp + 12, sizeof(fc->vp3));
    {
        /* a perspective camera's z row is its w row scaled, plus a constant */
        const float *zr = f->vp + 8, *wr = f->vp + 12;
        int best = 0, q;
        for (q = 1; q < 3; q++) if (fabsf(wr[q]) > fabsf(wr[best])) best = q;
        fc->zp[0] = wr[best] != 0.0f ? zr[best] / wr[best] : 1.0f;
        fc->zp[1] = zr[3] - fc->zp[0] * wr[3];
        fc->zp[2] = f->zmax;
        fc->zp[3] = (float)f->view;
    }
    fc->misc[0] = (float)(K.frame++ % 4096u);
    memcpy(fc->light, f->light, sizeof(fc->light));
    memcpy(fc->prev_vp0, K.prev_vp, sizeof(fc->prev_vp0));
    memcpy(fc->prev_vp1, K.prev_vp + 4, sizeof(fc->prev_vp1));
    memcpy(fc->prev_vp3, K.prev_vp + 12, sizeof(fc->prev_vp3));
    fc->misc[1] = K.have_prev && !K.history_fresh ? 1.0f : 0.0f;
    fc->fx[0] = f->ao; fc->fx[1] = f->ao_radius > 0.0f ? f->ao_radius : 1.0f; fc->fx[2] = f->gi; fc->fx[3] = 0.0f;
    memcpy(K.prev_vp, f->vp, sizeof(K.prev_vp));
    K.have_prev = true;
    for (k = 0; k < n; k++) {
        uint32_t *d = K.draws_cpu + ((size_t)K.slot * KRT_DRAWS_MAX + k) * 4;
        i = K.order[k];
        d[0] = f->first[i]; d[1] = f->tex0[i]; d[2] = f->tex1[i]; d[3] = f->flags[i];
    }
    for (k = 0; k < 2; k++) {
        D3D12_RAYTRACING_INSTANCE_DESC *in = K.instance_cpu + K.slot * 2 + k;
        memset(in, 0, sizeof(*in));
        in->Transform[0][0] = in->Transform[1][1] = in->Transform[2][2] = 1.0f;
        in->InstanceID = k ? ns : 0;
        /* an empty group's bottom level is not built this frame: the instance is inactive (no address) */
        in->InstanceMask = group[k] ? (k ? 2 : 1) : 0;
        in->AccelerationStructure = group[k] ? ID3D12Resource_GetGPUVirtualAddress(K.blas[k]) : 0;
    }
    cva = ID3D12Resource_GetGPUVirtualAddress(K.consts) + K.slot * 512;

    heaps[0] = K.heap;
    ID3D12GraphicsCommandList4_SetDescriptorHeaps(K.list, 1, heaps);
    ID3D12GraphicsCommandList4_SetComputeRootSignature(K.list, K.root);
    ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 3, gpu_handle(H_TEX));

    /* Unpack. */
    barrier(K.geo, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (K.vb_readable) {
        barrier(K.pos, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        barrier(K.uv, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        barrier(K.col, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    stamp(0);
    ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_unpack);
    ID3D12GraphicsCommandList4_SetComputeRootConstantBufferView(K.list, 0, cva);
    ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 1, gpu_handle(H_UNPACK_SRV));
    ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 2, gpu_handle(H_UNPACK_UAV));
    ID3D12GraphicsCommandList4_Dispatch(K.list, (f->nverts + 63) / 64, 1, 1);
    barrier(K.pos, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(K.uv, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(K.col, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(K.geo, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    K.vb_readable = true;
    stamp(1);

    /* Build: the two bottom levels, one after the other through the same scratch, then the top level over them. */
    for (k = 0; k < 2; k++) {
        if (!group[k]) continue;
        memset(&build, 0, sizeof(build));
        build.Inputs = bl[k];
        build.DestAccelerationStructureData = ID3D12Resource_GetGPUVirtualAddress(K.blas[k]);
        build.ScratchAccelerationStructureData = ID3D12Resource_GetGPUVirtualAddress(K.scratch);
        ID3D12GraphicsCommandList4_BuildRaytracingAccelerationStructure(K.list, &build, 0, NULL);
        uav_barrier(K.blas[k]);
        uav_barrier(K.scratch);
    }
    memset(&build, 0, sizeof(build));
    build.Inputs = tl;
    /* this frame's instances: K.slot is this frame's only once open_list has run */
    build.Inputs.InstanceDescs = ID3D12Resource_GetGPUVirtualAddress(K.instance) + (UINT64)K.slot * 2 * sizeof(D3D12_RAYTRACING_INSTANCE_DESC);
    build.DestAccelerationStructureData = ID3D12Resource_GetGPUVirtualAddress(K.tlas);
    build.ScratchAccelerationStructureData = ID3D12Resource_GetGPUVirtualAddress(K.scratch);
    ID3D12GraphicsCommandList4_BuildRaytracingAccelerationStructure(K.list, &build, 0, NULL);
    uav_barrier(K.tlas);
    stamp(2);

    if (f->calibrate && K.out && K.in_pic) {
        /* the brightness factors, from what the camera sees of the picture this frame */
        ID3D12GraphicsCommandList4_SetComputeRootConstantBufferView(K.list, 0, cva + 256);
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 1, gpu_handle(H_TRACE_SRV + 12 * K.slot));
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 2, gpu_handle(H_TRACE_UAV));
        barrier(K.in_pic, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_calibrate);
        ID3D12GraphicsCommandList4_Dispatch(K.list, (K.out_w / 8 + 7) / 8, (K.out_h / 8 + 7) / 8, 1);
        uav_barrier(K.sums);
        ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_calib_update);
        ID3D12GraphicsCommandList4_Dispatch(K.list, (KRT_FACTOR_KEYS + 63) / 64, 1, 1);
        uav_barrier(K.factors);
        uav_barrier(K.sums);
        barrier(K.in_pic, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    stamp(3);
    if (f->lighting && K.in_depth && K.light_out && K.history[0]) {
        ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_light);
        ID3D12GraphicsCommandList4_SetComputeRootConstantBufferView(K.list, 0, cva + 256);
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 1, gpu_handle(H_TRACE_SRV + 12 * K.slot));
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 2, gpu_handle(H_TRACE_UAV));
        barrier(K.in_depth, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(K.light_out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12GraphicsCommandList4_Dispatch(K.list, ((K.out_w + 1) / 2 + 7) / 8, ((K.out_h + 1) / 2 + 7) / 8, 1);
        uav_barrier(K.history[0]);
        uav_barrier(K.history[1]);
        ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_light_filter);
        ID3D12GraphicsCommandList4_Dispatch(K.list, ((K.out_w + 1) / 2 + 7) / 8, ((K.out_h + 1) / 2 + 7) / 8, 1);
        barrier(K.light_out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        barrier(K.in_depth, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        K.history_fresh = false;
        wrote = true;
    }
    stamp(4);
    if (f->reflect && !f->view && K.out && K.in_pic && K.in_w && K.in_g && K.refl_raw) {
        ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_reflect);
        ID3D12GraphicsCommandList4_SetComputeRootConstantBufferView(K.list, 0, cva + 256);
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 1, gpu_handle(H_TRACE_SRV + 12 * K.slot));
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 2, gpu_handle(H_TRACE_UAV));
        barrier(K.in_pic, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(K.in_w, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(K.in_g, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ID3D12GraphicsCommandList4_Dispatch(K.list, (K.out_w + 7) / 8, (K.out_h + 7) / 8, 1);
        uav_barrier(K.refl_raw);
        /* one ray a pixel, filtered */
        barrier(K.out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_reflect_filter);
        ID3D12GraphicsCommandList4_Dispatch(K.list, (K.out_w + 7) / 8, (K.out_h + 7) / 8, 1);
        barrier(K.out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        barrier(K.in_g, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        barrier(K.in_w, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        barrier(K.in_pic, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        wrote = true;
    }
    stamp(5);
    if (f->view && K.out) {
        ID3D12GraphicsCommandList4_SetPipelineState(K.list, K.pso_view);
        ID3D12GraphicsCommandList4_SetComputeRootConstantBufferView(K.list, 0, cva + 256);
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 1, gpu_handle(H_TRACE_SRV + 12 * K.slot));
        ID3D12GraphicsCommandList4_SetComputeRootDescriptorTable(K.list, 2, gpu_handle(H_TRACE_UAV));
        if (K.in_pic) barrier(K.in_pic, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(K.out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12GraphicsCommandList4_Dispatch(K.list, (K.out_w + 7) / 8, (K.out_h + 7) / 8, 1);
        barrier(K.out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        if (K.in_pic) barrier(K.in_pic, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        wrote = true;
    }

    stamp(6);
    if (K.times) {
        ID3D12GraphicsCommandList4_ResolveQueryData(K.list, K.qheap, D3D12_QUERY_TYPE_TIMESTAMP, K.slot * 16, KRT_STAMPS, K.qread,
                                                    (UINT64)K.slot * 16 * 8);
        K.qpending[K.slot] = true;
    }
    done = submit();
    K.alloc_done[K.slot] = done;
    for (i = 0; i < K.npending; i++) if (K.pending[i].value == UINT64_MAX) K.pending[i].value = done;
    ID3D11DeviceContext4_Wait(K.ctx11, K.fence11, done);
    if (!K.said_frame) {
        K.said_frame = true;
        fprintf(stderr, "[KRT] first scene: %u draws (%u dynamic), %u triangles; acceleration structures %llu + %llu + %llu KiB\n",
                n, n - ns, f->nverts / 3, (unsigned long long)(K.blas_size[0] >> 10), (unsigned long long)(K.blas_size[1] >> 10),
                (unsigned long long)(K.tlas_size >> 10));
    }
    return wrote;
}
#endif
