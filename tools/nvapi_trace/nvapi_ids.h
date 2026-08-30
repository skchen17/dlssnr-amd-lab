// nvapi_ids.h — NVAPI interface IDs known from public sources.
//
// Policy: only IDs verified in NVIDIA's open-source nvapi.h (github.com/NVIDIA/nvapi)
// are tagged SRC_NVAPI_H. IDs cited by community interop projects (fakenvapi /
// OptiScaler ecosystem) are tagged SRC_COMMUNITY and MUST be re-verified against the
// trace itself before being used as evidence. The shim always records the raw ID, so
// unknown IDs lose nothing.

#pragma once
#include <cstdint>

struct NvApiIdName {
    uint32_t id;
    const char* name;
    const char* source;   // "nvapi.h" | "community"
};

// Verified against NVIDIA open-source nvapi.h (stable across versions).
static const NvApiIdName kNvApiKnownIds[] = {
    { 0x0150E828, "NvAPI_Initialize",                     "nvapi.h" },
    { 0xD22BDD7E, "NvAPI_Unload",                         "nvapi.h" },
    { 0x6C2D048C, "NvAPI_GetErrorMessage",                "nvapi.h" },
    { 0x01053FA5, "NvAPI_GetInterfaceVersionString",      "nvapi.h" },
    { 0x9ABDD40D, "NvAPI_EnumNvidiaDisplayHandle",        "nvapi.h" },
    { 0xE5AC921F, "NvAPI_EnumPhysicalGPUs",               "nvapi.h" },
    { 0x35C29134, "NvAPI_GPU_GetFullName",                "nvapi.h" },
    { 0xC33BAEB7, "NvAPI_GPU_GetBusId",                   "nvapi.h" },
    { 0xD8265D24, "NvAPI_GPU_GetPhysicalFrameBufferSize", "nvapi.h" },
    { 0x46FBEB03, "NvAPI_GPU_GetMemoryInfo",              "nvapi.h" },
    { 0x5F68DAAA, "NvAPI_D3D11_CreateDevice",             "nvapi.h" },
    { 0xBB939EE9, "NvAPI_D3D11_CreateDeviceAndSwapChain", "nvapi.h" },
    { 0x171C7F0A, "NvAPI_D3D11_IsNvShaderExtnOpCodeSupported", "nvapi.h" },
    { 0x208B2ABF, "NvAPI_D3D12_IsFatbinPTXSupported",     "nvapi.h" },  // name per nvapi.h ext
    { 0x2A2C79E8, "NvAPI_D3D12_CreateGraphicsPipelineState", "nvapi.h" },
};

// Community-cited IDs (fakenvapi / DLSS tooling). Treat as hypotheses:
// the trace log records raw IDs regardless, so wrong names here cannot corrupt data.
static const NvApiIdName kNvApiCommunityIds[] = {
    { 0xC7449F65, "NvAPI_D3D12_EnumCuModules?(community)",         "community" },
    { 0x162BD2E5, "NvAPI_D3D12_CreateCuModule?(community)",        "community" },
    { 0xC79680B8, "NvAPI_D3D12_CreateCuFunction?(community)",      "community" },
    { 0xD0D9F9C8, "NvAPI_D3D12_LaunchCuKernelChain?(community)",   "community" },
    { 0x4C44B0AE, "NvAPI_D3D12_DestroyCuModule?(community)",       "community" },
    { 0x47552258, "NvAPI_D3D12_SetCreateGraphicsPipelineStateFlags?(community)", "community" },
};

// NGP/DLSS host entry points exported by nvngx_dlss.dll itself (not nvapi IDs) —
// kept here so module_trace/nr_host docs share one list.
static const char* const kNgxExports[] = {
    "NVSDK_NGX_D3D12_Init", "NVSDK_NGX_D3D12_Init_Ext", "NVSDK_NGX_D3D12_Shutdown",
    "NVSDK_NGX_D3D12_CreateFeature", "NVSDK_NGX_D3D12_EvaluateFeature",
    "NVSDK_NGX_D3D12_ReleaseFeature", "NVSDK_NGX_D3D12_GetFeatureRequirements",
    "NVSDK_NGX_D3D12_GetCapabilityParameters", "NVSDK_NGX_D3D12_GetParameters",
    "NVSDK_NGX_D3D12_AllocateResource", "NVSDK_NGX_D3D12_ReleaseResource",
    "NVSDK_NGX_D3D12_GetScratchBufferSize", "NVSDK_NGX_CUDA_Init",
    "NVSDK_NGX_CUDA_GetParameters", "NVSDK_NGX_CUDA_CreateFeature",
    "NVSDK_NGX_CUDA_EvaluateFeature", "NVSDK_NGX_CUDA_ReleaseFeature",
    "NVSDK_NGX_CUDA_Shutdown", nullptr,
};
