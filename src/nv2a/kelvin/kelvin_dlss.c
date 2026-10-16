/* NVIDIA DLSS for the Kelvin D3D11 backend. See kelvin_dlss.h. */
#include "kelvin_dlss.h"

#ifdef _WIN32
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#ifdef KDLSS_HAVE_SDK
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_params.h"

/* A stable project id (NGX's custom-engine identification) and where NGX may write its logs. */
#define KDLSS_PROJECT "6d1e2b8c-5b3a-4f0e-9a51-b1ac2006e0f1"

static struct {
    int state;                          /* 0 not tried, 1 available, -1 not */
    NVSDK_NGX_Parameter *params;
    NVSDK_NGX_Handle *feature;
    unsigned in_w, in_h, out_w, out_h;
    int quality;
} D;

bool kdlss_init(ID3D11Device *dev)
{
    NVSDK_NGX_Result r;
    int available = 0, driver = 0;
    wchar_t path[MAX_PATH];
    if (D.state) return D.state > 0;
    D.state = -1;
    if (!GetModuleFileNameW(NULL, path, MAX_PATH)) wcscpy_s(path, MAX_PATH, L".");
    else { wchar_t *slash = wcsrchr(path, L'\\'); if (slash) *slash = 0; }
    r = NVSDK_NGX_D3D11_Init_with_ProjectID(KDLSS_PROJECT, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", path, dev, NULL, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) { fprintf(stderr, "[DLSS] NGX did not start (0x%08X): DLSS off\n", (unsigned)r); return false; }
    r = NVSDK_NGX_D3D11_GetCapabilityParameters(&D.params);
    if (NVSDK_NGX_FAILED(r) || !D.params) {
        fprintf(stderr, "[DLSS] NGX has no capability parameters (0x%08X): DLSS off\n", (unsigned)r);
        NVSDK_NGX_D3D11_Shutdown1(dev);
        return false;
    }
    NVSDK_NGX_Parameter_GetI(D.params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &driver);
    if (NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetI(D.params, NVSDK_NGX_Parameter_SuperSampling_Available, &available)) || !available) {
        fprintf(stderr, "[DLSS] not available on this GPU%s: DLSS off\n", driver ? " (the driver needs an update)" : "");
        NVSDK_NGX_D3D11_DestroyParameters(D.params); D.params = NULL;
        NVSDK_NGX_D3D11_Shutdown1(dev);
        return false;
    }
    D.state = 1;
    fprintf(stderr, "[DLSS] available\n");
    return true;
}

bool kdlss_available(void) { return D.state > 0; }

void kdlss_render_size(int quality, unsigned out_w, unsigned out_h, unsigned *in_w, unsigned *in_h)
{
    const double scale = quality == KDLSS_QUALITY ? 2.0 / 3.0 : quality == KDLSS_BALANCED ? 0.58 : quality == KDLSS_PERFORMANCE ? 0.5 : 1.0;
    *in_w = (unsigned)(out_w * scale + 0.5) & ~1u;
    *in_h = (unsigned)(out_h * scale + 0.5) & ~1u;
    if (*in_w < 2) *in_w = 2;
    if (*in_h < 2) *in_h = 2;
}

static NVSDK_NGX_PerfQuality_Value perf_value(int quality)
{
    switch (quality) {
    case KDLSS_QUALITY: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    case KDLSS_BALANCED: return NVSDK_NGX_PerfQuality_Value_Balanced;
    case KDLSS_PERFORMANCE: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    default: return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
}

bool kdlss_evaluate(ID3D11DeviceContext *ctx, ID3D11Resource *color, ID3D11Resource *depth, ID3D11Resource *motion,
                    ID3D11Resource *output, unsigned in_w, unsigned in_h, unsigned out_w, unsigned out_h, int quality,
                    float jitter_x, float jitter_y, bool reset)
{
    NVSDK_NGX_Result r;
    if (D.state <= 0) return false;
    if (!D.feature || D.in_w != in_w || D.in_h != in_h || D.out_w != out_w || D.out_h != out_h || D.quality != quality) {
        if (D.feature) { NVSDK_NGX_D3D11_ReleaseFeature(D.feature); D.feature = NULL; }
        NVSDK_NGX_Parameter_SetUI(D.params, NVSDK_NGX_Parameter_Width, in_w);
        NVSDK_NGX_Parameter_SetUI(D.params, NVSDK_NGX_Parameter_Height, in_h);
        NVSDK_NGX_Parameter_SetUI(D.params, NVSDK_NGX_Parameter_OutWidth, out_w);
        NVSDK_NGX_Parameter_SetUI(D.params, NVSDK_NGX_Parameter_OutHeight, out_h);
        NVSDK_NGX_Parameter_SetI(D.params, NVSDK_NGX_Parameter_PerfQualityValue, (int)perf_value(quality));
        NVSDK_NGX_Parameter_SetI(D.params, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, NVSDK_NGX_DLSS_Feature_Flags_MVLowRes);
        NVSDK_NGX_Parameter_SetI(D.params, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
        r = NVSDK_NGX_D3D11_CreateFeature(ctx, NVSDK_NGX_Feature_SuperSampling, D.params, &D.feature);
        if (NVSDK_NGX_FAILED(r) || !D.feature) {
            fprintf(stderr, "[DLSS] feature for %ux%u -> %ux%u failed (0x%08X): DLSS off\n", in_w, in_h, out_w, out_h, (unsigned)r);
            D.feature = NULL;
            D.state = -1;
            return false;
        }
        D.in_w = in_w; D.in_h = in_h; D.out_w = out_w; D.out_h = out_h; D.quality = quality;
        reset = true;
        fprintf(stderr, "[DLSS] %ux%u -> %ux%u (%s)\n", in_w, in_h, out_w, out_h,
                quality == KDLSS_DLAA ? "DLAA" : quality == KDLSS_QUALITY ? "quality" : quality == KDLSS_BALANCED ? "balanced" : "performance");
    }
    NVSDK_NGX_Parameter_SetD3d11Resource(D.params, NVSDK_NGX_Parameter_Color, color);
    NVSDK_NGX_Parameter_SetD3d11Resource(D.params, NVSDK_NGX_Parameter_Output, output);
    NVSDK_NGX_Parameter_SetD3d11Resource(D.params, NVSDK_NGX_Parameter_Depth, depth);
    NVSDK_NGX_Parameter_SetD3d11Resource(D.params, NVSDK_NGX_Parameter_MotionVectors, motion);
    NVSDK_NGX_Parameter_SetF(D.params, NVSDK_NGX_Parameter_Jitter_Offset_X, jitter_x);
    NVSDK_NGX_Parameter_SetF(D.params, NVSDK_NGX_Parameter_Jitter_Offset_Y, jitter_y);
    NVSDK_NGX_Parameter_SetF(D.params, NVSDK_NGX_Parameter_Sharpness, 0.0f);
    NVSDK_NGX_Parameter_SetI(D.params, NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
    NVSDK_NGX_Parameter_SetF(D.params, NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    NVSDK_NGX_Parameter_SetF(D.params, NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    NVSDK_NGX_Parameter_SetUI(D.params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, in_w);
    NVSDK_NGX_Parameter_SetUI(D.params, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, in_h);
    NVSDK_NGX_Parameter_SetF(D.params, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    NVSDK_NGX_Parameter_SetF(D.params, NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    r = NVSDK_NGX_D3D11_EvaluateFeature_C(ctx, D.feature, D.params, NULL);
    if (NVSDK_NGX_FAILED(r)) {
        static unsigned reported;
        if (reported++ < 3) fprintf(stderr, "[DLSS] evaluate failed (0x%08X)\n", (unsigned)r);
        return false;
    }
    return true;
}

void kdlss_shutdown(void)
{
    if (D.feature) { NVSDK_NGX_D3D11_ReleaseFeature(D.feature); D.feature = NULL; }
    if (D.params) { NVSDK_NGX_D3D11_DestroyParameters(D.params); D.params = NULL; }
    if (D.state > 0) NVSDK_NGX_D3D11_Shutdown1(NULL);
    D.state = 0;
}

#else /* no SDK at build time */
bool kdlss_init(ID3D11Device *dev) { (void)dev; return false; }
bool kdlss_available(void) { return false; }
void kdlss_render_size(int quality, unsigned out_w, unsigned out_h, unsigned *in_w, unsigned *in_h)
{ (void)quality; *in_w = out_w; *in_h = out_h; }
bool kdlss_evaluate(ID3D11DeviceContext *ctx, ID3D11Resource *color, ID3D11Resource *depth, ID3D11Resource *motion,
                    ID3D11Resource *output, unsigned in_w, unsigned in_h, unsigned out_w, unsigned out_h, int quality,
                    float jitter_x, float jitter_y, bool reset)
{
    (void)ctx; (void)color; (void)depth; (void)motion; (void)output; (void)in_w; (void)in_h; (void)out_w; (void)out_h;
    (void)quality; (void)jitter_x; (void)jitter_y; (void)reset;
    return false;
}
void kdlss_shutdown(void) {}
#endif
#endif
