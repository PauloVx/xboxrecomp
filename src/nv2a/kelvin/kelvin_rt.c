/*
 * Ray tracing for the Kelvin D3D11 backend. See kelvin_rt.h.
 *
 * d3d12.dll is loaded at run time, so a system without Direct3D 12 still runs the game, only without ray tracing.
 * Bring-up checks, in order: the D3D12 device on the renderer's adapter, DXR tier 1.1, a fence both devices can
 * use, and a texture written by D3D12 that D3D11 reads back as written. Any failure leaves ray tracing off.
 */
#ifdef _WIN32
#define COBJMACROS
#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kelvin_rt.h"

#define KRT_RELEASE(p) do { if (p) { IUnknown_Release((IUnknown *)(p)); (p) = NULL; } } while (0)

typedef HRESULT (WINAPI *PfnCreateDevice)(IUnknown *, D3D_FEATURE_LEVEL, REFIID, void **);
typedef HRESULT (WINAPI *PfnGetDebug)(REFIID, void **);

static struct {
    bool ok;
    HMODULE lib;
    ID3D12Device5 *dev;
    ID3D12CommandQueue *queue;
    ID3D12CommandAllocator *alloc;
    ID3D12GraphicsCommandList4 *list;
    ID3D12Fence *fence;             /* the D3D11 fence below, opened on this device */
    ID3D11Device5 *dev11;
    ID3D11DeviceContext4 *ctx11;
    ID3D11Fence *fence11;
    HANDLE fence_h;
    uint64_t fence_value;
    ID3D12Resource *geo;            /* the captured triangles, written by D3D11 */
    unsigned geo_w, geo_h;
} K;

bool krt_set_geometry(HANDLE texture, unsigned width, unsigned height)
{
    KRT_RELEASE(K.geo);
    if (!K.ok || FAILED(ID3D12Device5_OpenSharedHandle(K.dev, texture, &IID_ID3D12Resource, (void **)&K.geo))) {
        fprintf(stderr, "[KRT] D3D12 cannot open the geometry texture\n");
        return false;
    }
    K.geo_w = width; K.geo_h = height;
    return true;
}

bool krt_available(void) { return K.ok; }

void krt_shutdown(void)
{
    KRT_RELEASE(K.geo);
    KRT_RELEASE(K.list); KRT_RELEASE(K.alloc); KRT_RELEASE(K.queue); KRT_RELEASE(K.fence); KRT_RELEASE(K.dev);
    KRT_RELEASE(K.fence11); KRT_RELEASE(K.ctx11); KRT_RELEASE(K.dev11);
    if (K.fence_h) { CloseHandle(K.fence_h); K.fence_h = NULL; }
    if (K.lib) { FreeLibrary(K.lib); K.lib = NULL; }
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

static bool reset_list(void)
{
    return SUCCEEDED(ID3D12CommandAllocator_Reset(K.alloc)) &&
           SUCCEEDED(ID3D12GraphicsCommandList4_Reset(K.list, K.alloc, NULL));
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
    D3D12_HEAP_PROPERTIES hp;
    D3D12_RESOURCE_DESC rd;
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

    memset(&hp, 0, sizeof(hp));
    hp.Type = D3D12_HEAP_TYPE_UPLOAD;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = W * H * 4; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(ID3D12Device5_CreateCommittedResource(K.dev, &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     NULL, &IID_ID3D12Resource, (void **)&upload)) ||
        FAILED(ID3D12Resource_Map(upload, 0, NULL, (void **)&p))) {
        goto done;
    }
    for (y = 0; y < H; y++) for (x = 0; x < W; x++) p[y * W + x] = 0xFF000000u | (y << 8) | x;
    ID3D12Resource_Unmap(upload, 0, NULL);

    if (!reset_list()) goto done;
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

done:
    KRT_RELEASE(stage); KRT_RELEASE(upload); KRT_RELEASE(shared); KRT_RELEASE(r1); KRT_RELEASE(tex);
    if (h) CloseHandle(h);
    return good;
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

    if (K.ok) return true;
    K.lib = LoadLibraryA("d3d12.dll");
    create = K.lib ? (PfnCreateDevice)(void *)GetProcAddress(K.lib, "D3D12CreateDevice") : NULL;
    if (!create) { fprintf(stderr, "[KRT] no Direct3D 12: ray tracing off\n"); krt_shutdown(); return false; }
    if (getenv("RECOMP_KGPU_RT_DEBUG")) {
        PfnGetDebug get = (PfnGetDebug)(void *)GetProcAddress(K.lib, "D3D12GetDebugInterface");
        ID3D12Debug *debug = NULL;
        if (get && SUCCEEDED(get(&IID_ID3D12Debug, (void **)&debug))) { ID3D12Debug_EnableDebugLayer(debug); ID3D12Debug_Release(debug); }
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
    if (FAILED(ID3D12Device5_CreateCommandQueue(K.dev, &qd, &IID_ID3D12CommandQueue, (void **)&K.queue)) ||
        FAILED(ID3D12Device5_CreateCommandAllocator(K.dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void **)&K.alloc)) ||
        FAILED(ID3D12Device5_CreateCommandList(K.dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, K.alloc, NULL,
                                               &IID_ID3D12GraphicsCommandList4, (void **)&K.list))) {
        fprintf(stderr, "[KRT] D3D12 command objects failed: ray tracing off\n");
        krt_shutdown(); return false;
    }
    ID3D12GraphicsCommandList4_Close(K.list);

    if (FAILED(ID3D11Device_QueryInterface(dev, &IID_ID3D11Device5, (void **)&K.dev11)) ||
        FAILED(ID3D11DeviceContext_QueryInterface(ctx, &IID_ID3D11DeviceContext4, (void **)&K.ctx11)) ||
        FAILED(ID3D11Device5_CreateFence(K.dev11, 0, D3D11_FENCE_FLAG_SHARED, &IID_ID3D11Fence, (void **)&K.fence11)) ||
        FAILED(ID3D11Fence_CreateSharedHandle(K.fence11, NULL, GENERIC_ALL, NULL, &K.fence_h)) ||
        FAILED(ID3D12Device5_OpenSharedHandle(K.dev, K.fence_h, &IID_ID3D12Fence, (void **)&K.fence))) {
        fprintf(stderr, "[KRT] D3D11 and D3D12 cannot share a fence: ray tracing off\n");
        krt_shutdown(); return false;
    }
    K.fence_value = 0;

    if (!interop_check(dev, ctx)) { krt_shutdown(); return false; }
    K.ok = true;
    fprintf(stderr, "[KRT] ray tracing available: %ls, DXR tier %d.%d, D3D11 and D3D12 share textures and a fence\n",
            ad.Description, (int)o5.RaytracingTier / 10, (int)o5.RaytracingTier % 10);
    return true;
}
#endif
