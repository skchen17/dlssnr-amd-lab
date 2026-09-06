// module_trace — Round 2 Phase E: reliable module-load recorder.
//
// Interposes LoadLibraryExW/A + LoadLibraryW/A + GetProcAddress + FreeLibrary
// via IAT patching of every loaded module (no third-party hooking library
// needed), and logs every event as JSONL.
//
// Round 2 fixes vs. Round 1:
//   - SYNCHRONOUS startup: ModuleTrace_InitializeAndWait() lets the host wait
//     for hooks_installed=true before loading anything interesting (the old
//     async-only InitThread raced the host's first LoadLibrary).
//   - GetProcAddress is now intercepted and logged.
//   - newly loaded modules get their IAT patched immediately (inside the
//     LoadLibrary hooks), plus a best-effort ntdll
//     LdrRegisterModuleNotificationCallback registration so modules loaded
//     through other paths are patched too.
//
// Known limitation (kept, not silently ignored): direct ntdll LdrLoadDll
// callers are only observed via the module-notification callback, not logged
// as explicit load calls.
//
// Log: %MODULE_TRACE_LOG% (default: .\module_trace.log).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <winternl.h>
#include <psapi.h>
#include <share.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#pragma comment(lib, "psapi.lib")

static CRITICAL_SECTION g_cs;
static FILE* g_log = nullptr;
static LARGE_INTEGER g_freq;
static HMODULE g_self = nullptr;
static HANDLE g_readyEvt = nullptr;          // set once hooks are installed
static volatile LONG g_initStarted = 0;      // one-shot gate for DoInit
static volatile LONG g_hooksInstalled = 0;

typedef HMODULE(WINAPI* LoadLibraryExW_t)(LPCWSTR, HANDLE, DWORD);
typedef HMODULE(WINAPI* LoadLibraryExA_t)(LPCSTR, HANDLE, DWORD);
typedef HMODULE(WINAPI* LoadLibraryW_t)(LPCWSTR);
typedef HMODULE(WINAPI* LoadLibraryA_t)(LPCSTR);
typedef BOOL(WINAPI* FreeLibrary_t)(HMODULE);
typedef FARPROC(WINAPI* GetProcAddress_t)(HMODULE, LPCSTR);

static LoadLibraryExW_t g_realLoadLibraryExW = LoadLibraryExW;
static LoadLibraryExA_t g_realLoadLibraryExA = LoadLibraryExA;
static LoadLibraryW_t g_realLoadLibraryW = LoadLibraryW;
static LoadLibraryA_t g_realLoadLibraryA = LoadLibraryA;
static FreeLibrary_t g_realFreeLibrary = FreeLibrary;
static GetProcAddress_t g_realGetProcAddress = GetProcAddress;

typedef void (STDMETHODCALLTYPE* CreateShaderResourceView_t)(
    ID3D12Device*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*,
    D3D12_CPU_DESCRIPTOR_HANDLE);
typedef void (STDMETHODCALLTYPE* CreateUnorderedAccessView_t)(
    ID3D12Device*, ID3D12Resource*, ID3D12Resource*,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
typedef void (STDMETHODCALLTYPE* CreateSampler_t)(
    ID3D12Device*, const D3D12_SAMPLER_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
typedef void (STDMETHODCALLTYPE* CopyDescriptors_t)(
    ID3D12Device*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, UINT,
    const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, D3D12_DESCRIPTOR_HEAP_TYPE);
typedef void (STDMETHODCALLTYPE* CopyDescriptorsSimple_t)(
    ID3D12Device*, UINT, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_TYPE);
typedef HRESULT (STDMETHODCALLTYPE* CreateCommittedResource_t)(
    ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
    const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*,
    REFIID, void**);
typedef HRESULT (STDMETHODCALLTYPE* CreatePlacedResource_t)(
    ID3D12Device*, ID3D12Heap*, UINT64, const D3D12_RESOURCE_DESC*,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
typedef HRESULT (STDMETHODCALLTYPE* CreateReservedResource_t)(
    ID3D12Device*, const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
    const D3D12_CLEAR_VALUE*, REFIID, void**);
typedef HRESULT (STDMETHODCALLTYPE* CreateCommandQueue_t)(
    ID3D12Device*, const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
typedef HRESULT (STDMETHODCALLTYPE* CreateCommandList_t)(
    ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator*,
    ID3D12PipelineState*, REFIID, void**);
typedef HRESULT (STDMETHODCALLTYPE* CommandListClose_t)(
    ID3D12GraphicsCommandList*);
typedef void (STDMETHODCALLTYPE* ExecuteCommandLists_t)(
    ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);

static CreateShaderResourceView_t g_realCreateShaderResourceView = nullptr;
static CreateUnorderedAccessView_t g_realCreateUnorderedAccessView = nullptr;
static CreateSampler_t g_realCreateSampler = nullptr;
static CopyDescriptors_t g_realCopyDescriptors = nullptr;
static CopyDescriptorsSimple_t g_realCopyDescriptorsSimple = nullptr;
static CreateCommittedResource_t g_realCreateCommittedResource = nullptr;
static CreatePlacedResource_t g_realCreatePlacedResource = nullptr;
static CreateReservedResource_t g_realCreateReservedResource = nullptr;
static CreateCommandQueue_t g_realCreateCommandQueue = nullptr;
static CreateCommandList_t g_realCreateCommandList = nullptr;
static CommandListClose_t g_realCommandListClose = nullptr;
static ExecuteCommandLists_t g_realExecuteCommandLists = nullptr;
using ResourceBarrier_t = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*);
using ListReset_t = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12CommandAllocator*, ID3D12PipelineState*);
using CopyBuffer_t = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12Resource*, UINT64, ID3D12Resource*, UINT64, UINT64);
using CopyTexture_t = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, const D3D12_TEXTURE_COPY_LOCATION*, UINT, UINT, UINT, const D3D12_TEXTURE_COPY_LOCATION*, const D3D12_BOX*);
using CopyResource_t = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12Resource*, ID3D12Resource*);
using Dispatch_t = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
static ResourceBarrier_t g_realResourceBarrier = nullptr;
static ListReset_t g_realListReset = nullptr;
static CopyBuffer_t g_realCopyBuffer = nullptr;
static CopyTexture_t g_realCopyTexture = nullptr;
static CopyResource_t g_realCopyResource = nullptr;
static Dispatch_t g_realDispatch = nullptr;
static volatile LONG g_deviceHooksInstalled = 0;
static volatile LONG g_queueHooksInstalled = 0;
static volatile LONG g_commandListHooksInstalled = 0;
static volatile LONG g_submissionTimelineSequence = 0;

struct DescriptorResourceRecord {
    SIZE_T descriptor;
    ID3D12Resource* resource;
};
struct ObjectResourceRecord {
    uint64_t object;
    ID3D12Resource* resource;
};
struct BufferResourceRecord {
    D3D12_GPU_VIRTUAL_ADDRESS address;
    UINT64 size;
    ID3D12Resource* resource;
};
struct ResolvedBufferAddress {
    ID3D12Resource* resource;
    UINT64 offset;
    UINT64 remaining;
};
struct CommandListTimelineRecord {
    ID3D12CommandList* list;
    LONG firstLaunchCall;
    LONG lastLaunchCall;
    LONG firstFrame;
    LONG lastFrame;
    LONG firstSlot;
    LONG lastSlot;
    uint32_t launchCount;
    uint32_t closeCount;
    uint32_t executeCount;
};
struct CommandListTimelineSnapshot {
    LONG firstLaunchCall;
    LONG lastLaunchCall;
    LONG firstFrame;
    LONG lastFrame;
    LONG firstSlot;
    LONG lastSlot;
    uint32_t launchCount;
    uint32_t closeCount;
    uint32_t executeCount;
};
struct SnapshotReadback {
    ID3D12Resource* buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT rows;
    UINT64 rowSize;
    UINT64 totalBytes;
};
struct NeuralCaptureWindow {
    const char* name;
    uint32_t paramOffset;
    UINT64 weightViewOffset;
    uint64_t pointer;
    ID3D12Resource* source;
    UINT64 sourceOffset;
    UINT64 size;
    ID3D12Resource* before;
    ID3D12Resource* after;
};
struct FullGraphCaptureWindow {
    uint32_t slot;
    uint32_t paramOffset;
    uint64_t function;
    uint64_t pointer;
    ID3D12Resource* source;
    UINT64 sourceOffset;
    UINT64 size;
    ID3D12Resource* before;
    ID3D12Resource* after;
};
static constexpr size_t kFullGraphWindowCapacity = 4096;
static constexpr const char* kDownstreamWeightNames[] = {
    "slot3_weights", "slot4_weights", "slot5_weights", "slot6_weights",
    "slot7_weights", "slot8_weights", "slot9_weights", "slot10_weights",
    "slot11_weights", "slot12_weights", "slot13_weights", "slot14_weights",
    "slot15_weights", "slot16_weights", "slot17_weights", "slot18_weights",
    "slot19_weights", "slot20_weights", "slot21_weights", "slot22_weights",
    "slot23_weights"};
static constexpr UINT64 kDownstreamWeightOffsets[] = {
    0x44DA00, 0x1390400, 0x7681600, 0x86FCE00, 0x8BC7400, 0x8C8A600,
    0x8C9F000, 0x8CB0200, 0xA800, 0x3AC00, 0x6B000, 0x9B400,
    0xCB800, 0x103C00, 0x1AC200, 0x254800, 0x2FCE00, 0x3A5400,
    0x452C00, 0x4FB200, 0x5A3800};
static constexpr size_t kDownstreamWeightCount =
    sizeof(kDownstreamWeightOffsets) / sizeof(kDownstreamWeightOffsets[0]);
static_assert(kDownstreamWeightCount ==
              sizeof(kDownstreamWeightNames) / sizeof(kDownstreamWeightNames[0]));
static constexpr size_t kNeuralWindowCount = 3 + kDownstreamWeightCount;
static CRITICAL_SECTION g_resourceCs;
static DescriptorResourceRecord g_descriptorResources[4096]{};
static size_t g_descriptorResourceCount = 0;
static ObjectResourceRecord g_objectResources[2048]{};
static size_t g_objectResourceCount = 0;
static ID3D12Resource* g_copyInputResource = nullptr;
static ID3D12Resource* g_copyOutputResource = nullptr;
static ID3D12Resource* g_postTextureResource = nullptr;
static uint64_t g_postTextureObject = 0;
static SnapshotReadback g_postTexturePreLaunchReadback{};
static D3D12_RESOURCE_DESC g_postTexturePreLaunchDesc{};
static volatile LONG g_postTexturePreLaunchState = 0; // 0 untouched, 1 arming, 2 recorded, -1 failed
static LONG g_postTexturePreLaunchFrame = 0;
static SnapshotReadback g_frame1PostOutputReadback{};
static D3D12_RESOURCE_DESC g_frame1PostOutputDesc{};
static uint64_t g_frame1PostOutputObject = 0;
static uintptr_t g_frame1PostOutputResourceAddress = 0;
static volatile LONG g_frame1PostOutputState = 0; // 0 untouched, 1 arming, 2 recorded, -1 failed
static LONG g_frame1PostOutputFrame = 0;
static SnapshotReadback g_frame1PostSurfaceInitialReadback{};
static D3D12_RESOURCE_DESC g_frame1PostSurfaceInitialDesc{};
static uint64_t g_frame1PostSurfaceObject = 0;
static uintptr_t g_frame1PostSurfaceResourceAddress = 0;
static volatile LONG g_frame1PostSurfaceInitialState = 0;
static LONG g_frame1PostSurfaceInitialFrame = 0;
static ID3D12Resource* g_postActivationPreLaunchReadback = nullptr;
static UINT64 g_postActivationPreLaunchBytes = 0;
static uint64_t g_postActivationParam0 = 0;
static uint64_t g_postActivationParam8 = 0;
static uintptr_t g_postActivationResourceAddress = 0;
static volatile LONG g_postActivationPreLaunchState = 0;
static BufferResourceRecord g_bufferResources[512]{};
static size_t g_bufferResourceCount = 0;
static NeuralCaptureWindow g_neuralWindows[kNeuralWindowCount]{};
static volatile LONG g_neuralCaptureState = 0; // 0 untouched, 1 armed, 2 recorded, -1 failed
static uint32_t g_neuralPaddedHeight = 0;
static uint32_t g_neuralPaddedWidth = 0;
static FullGraphCaptureWindow g_fullGraphWindows[kFullGraphWindowCapacity]{};
static size_t g_fullGraphWindowCount = 0;
static volatile LONG g_fullGraphCaptureFailures = 0;
static CommandListTimelineRecord g_commandListTimeline[1024]{};
static size_t g_commandListTimelineCount = 0;

static bool BeginPostTexturePreLaunchCapture(
    ID3D12GraphicsCommandList* list, ID3D12Resource* source,
    uint64_t textureObject, LONG frame);
static bool BeginPostActivationPreLaunchCapture(
    ID3D12GraphicsCommandList* list, const void* parameterData,
    uint32_t parameterSize, LONG frame);
static bool BeginFrame1PostOutputCapture(
    ID3D12GraphicsCommandList* list, ID3D12Resource* source,
    uint64_t surfaceObject, LONG frame);
static bool BeginFrame1PostSurfaceInitialCapture(
    ID3D12GraphicsCommandList* list, ID3D12Resource* source,
    uint64_t surfaceObject, LONG frame);

static bool FullGraphCaptureEnabled() {
    char value[16]{};
    DWORD length = GetEnvironmentVariableA(
        "MODULE_TRACE_FULL_GRAPH_CAPTURE", value, (DWORD)sizeof(value));
    return length > 0 && length < sizeof(value) && strcmp(value, "0") != 0;
}

static bool InlineSnapshotsEnabled() {
    char value[16]{};
    DWORD length = GetEnvironmentVariableA(
        "MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS", value, (DWORD)sizeof(value));
    return !(length > 0 && length < sizeof(value) && strcmp(value, "0") != 0);
}

static UINT64 FullGraphCaptureLimit() {
    constexpr UINT64 kDefault = 8ull << 20;
    constexpr UINT64 kMaximum = 64ull << 20;
    char value[32]{};
    DWORD length = GetEnvironmentVariableA(
        "MODULE_TRACE_FULL_GRAPH_MAX_BYTES", value, (DWORD)sizeof(value));
    if (!length || length >= sizeof(value)) return kDefault;
    char* end = nullptr;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (!end || *end || !parsed) return kDefault;
    return parsed > kMaximum ? kMaximum : (UINT64)parsed;
}

static void RememberBufferResource(ID3D12Resource* resource,
                                   D3D12_GPU_VIRTUAL_ADDRESS address,
                                   UINT64 size) {
    if (!resource || !address || !size) return;
    EnterCriticalSection(&g_resourceCs);
    for (size_t i = 0; i < g_bufferResourceCount; ++i) {
        if (g_bufferResources[i].address == address) {
            if (g_bufferResources[i].resource != resource) {
                g_bufferResources[i].resource->Release();
                resource->AddRef();
                g_bufferResources[i].resource = resource;
            }
            g_bufferResources[i].size = size;
            LeaveCriticalSection(&g_resourceCs);
            return;
        }
    }
    if (g_bufferResourceCount < _countof(g_bufferResources)) {
        resource->AddRef();
        g_bufferResources[g_bufferResourceCount++] = {address, size, resource};
    }
    LeaveCriticalSection(&g_resourceCs);
}

static ResolvedBufferAddress ResolveBufferAddress(uint64_t address) {
    ResolvedBufferAddress result{};
    EnterCriticalSection(&g_resourceCs);
    for (size_t i = g_bufferResourceCount; i > 0; --i) {
        const auto& record = g_bufferResources[i - 1];
        if (address >= record.address && address - record.address < record.size) {
            result.resource = record.resource;
            result.offset = address - record.address;
            result.remaining = record.size - result.offset;
            break;
        }
    }
    LeaveCriticalSection(&g_resourceCs);
    return result;
}

static void RememberDescriptorResource(SIZE_T descriptor, ID3D12Resource* resource) {
    if (!descriptor || !resource) return; // retain the last non-null view across NVAPI's internal null clobber
    EnterCriticalSection(&g_resourceCs);
    for (size_t i = 0; i < g_descriptorResourceCount; ++i) {
        if (g_descriptorResources[i].descriptor == descriptor) {
            g_descriptorResources[i].resource = resource;
            LeaveCriticalSection(&g_resourceCs);
            return;
        }
    }
    if (g_descriptorResourceCount < _countof(g_descriptorResources)) {
        g_descriptorResources[g_descriptorResourceCount++] = {descriptor, resource};
    }
    LeaveCriticalSection(&g_resourceCs);
}

static ID3D12Resource* FindDescriptorResource(SIZE_T descriptor) {
    ID3D12Resource* result = nullptr;
    EnterCriticalSection(&g_resourceCs);
    for (size_t i = g_descriptorResourceCount; i > 0; --i) {
        if (g_descriptorResources[i - 1].descriptor == descriptor) {
            result = g_descriptorResources[i - 1].resource;
            break;
        }
    }
    LeaveCriticalSection(&g_resourceCs);
    return result;
}

static void CopyDescriptorResource(SIZE_T destination, SIZE_T source) {
    ID3D12Resource* resource = FindDescriptorResource(source);
    if (resource) RememberDescriptorResource(destination, resource);
}

static void RememberObjectResource(uint64_t object, ID3D12Resource* resource) {
    if (!object || !resource) return;
    EnterCriticalSection(&g_resourceCs);
    for (size_t i = 0; i < g_objectResourceCount; ++i) {
        if (g_objectResources[i].object == object) {
            g_objectResources[i].resource = resource;
            LeaveCriticalSection(&g_resourceCs);
            return;
        }
    }
    if (g_objectResourceCount < _countof(g_objectResources)) {
        g_objectResources[g_objectResourceCount++] = {object, resource};
    }
    LeaveCriticalSection(&g_resourceCs);
}

static ID3D12Resource* FindObjectResource(uint64_t object) {
    ID3D12Resource* result = nullptr;
    EnterCriticalSection(&g_resourceCs);
    for (size_t i = g_objectResourceCount; i > 0; --i) {
        if (g_objectResources[i - 1].object == object) {
            result = g_objectResources[i - 1].resource;
            break;
        }
    }
    LeaveCriticalSection(&g_resourceCs);
    return result;
}

static void RememberCopyResources(ID3D12Resource* input, ID3D12Resource* output) {
    if (!input || !output) return;
    EnterCriticalSection(&g_resourceCs);
    if (g_copyInputResource != input) {
        if (g_copyInputResource) g_copyInputResource->Release();
        g_copyInputResource = input;
        g_copyInputResource->AddRef();
    }
    if (g_copyOutputResource != output) {
        if (g_copyOutputResource) g_copyOutputResource->Release();
        g_copyOutputResource = output;
        g_copyOutputResource->AddRef();
    }
    LeaveCriticalSection(&g_resourceCs);
}

static void RememberPostTextureResource(uint64_t object, ID3D12Resource* resource) {
    if (!object || !resource) return;
    EnterCriticalSection(&g_resourceCs);
    if (g_postTextureResource != resource) {
        if (g_postTextureResource) g_postTextureResource->Release();
        g_postTextureResource = resource;
        g_postTextureResource->AddRef();
    }
    g_postTextureObject = object;
    LeaveCriticalSection(&g_resourceCs);
}

// nvapi_QueryInterface has a stable, public ABI.  Returning a small wrapper from
// our existing GetProcAddress hook lets us observe requested interface IDs while
// still dispatching every call to the real system nvapi64.dll.  Deliberately do
// not wrap nvapi_Direct_GetMethod here: its private ABI is not established, and a
// guessed wrapper would make the reference run less trustworthy.
typedef void* (__cdecl* NvApiQueryInterface_t)(uint32_t);
static NvApiQueryInterface_t g_realNvApiQueryInterface = nullptr;

// Official NVIDIA R610 D3D12 experimental CuModule ABI.  Opaque D3D12 and NVDX
// handles are represented as pointers here so the tracer does not need to take a
// compile-time dependency on the NVAPI SDK headers.
struct NvApiDim3 { uint32_t x, y, z; };
struct NvApiCuKernelLaunchParams {
    void* hFunction;
    NvApiDim3 gridDim;
    NvApiDim3 blockDim;
    uint32_t dynSharedMemBytes;
    const void* pParams;
    uint32_t paramSize;
};
static_assert(sizeof(NvApiCuKernelLaunchParams) == 56,
              "R610 NVAPI_CU_KERNEL_LAUNCH_PARAMS ABI mismatch");
typedef int32_t (__cdecl* NvApiCreateCuModule_t)(void*, const void*, uint32_t, void**);
typedef int32_t (__cdecl* NvApiCreateCuFunction_t)(void*, void*, const char*, void**);
typedef int32_t (__cdecl* NvApiLaunchCuKernelChain_t)(void*, const NvApiCuKernelLaunchParams*, uint32_t);
typedef int32_t (__cdecl* NvApiDestroyCuObject_t)(void*, void*);

struct NvApiMergedTextureSamplerParams {
    size_t structSizeIn;
    size_t structSizeOut;
    void* pDevice;
    uintptr_t texDesc;
    uintptr_t smpDesc;
    uint64_t textureHandle;
};
struct NvApiIndependentDescriptorParams {
    size_t structSizeIn;
    size_t structSizeOut;
    void* pDevice;
    uint32_t type;
    uint32_t padding;
    uintptr_t desc;
    uint64_t handle;
};
static_assert(sizeof(NvApiMergedTextureSamplerParams) == 48,
              "R610 merged texture/sampler ABI mismatch");
static_assert(sizeof(NvApiIndependentDescriptorParams) == 48,
              "R610 independent descriptor ABI mismatch");
typedef int32_t (__cdecl* NvApiMergedTextureSampler_t)(NvApiMergedTextureSamplerParams*);
typedef int32_t (__cdecl* NvApiIndependentDescriptor_t)(NvApiIndependentDescriptorParams*);

static NvApiCreateCuModule_t g_realCreateCuModule = nullptr;
static NvApiCreateCuFunction_t g_realCreateCuFunction = nullptr;
static NvApiLaunchCuKernelChain_t g_realLaunchCuKernelChain = nullptr;
static NvApiDestroyCuObject_t g_realDestroyCuModule = nullptr;
static NvApiDestroyCuObject_t g_realDestroyCuFunction = nullptr;
static NvApiMergedTextureSampler_t g_realMergedTextureSampler = nullptr;
static NvApiIndependentDescriptor_t g_realIndependentDescriptor = nullptr;
static volatile LONG g_launchChainCalls = 0;

static const char* NvApiCandidateName(uint32_t id) {
    switch (id) {
    case 0xAD1A677D: return "NvAPI_D3D12_CreateCuModule";
    case 0x7AB88D88: return "NvAPI_D3D12_EnumFunctionsInModule";
    case 0xE2436E22: return "NvAPI_D3D12_CreateCuFunction";
    case 0x24973538: return "NvAPI_D3D12_LaunchCuKernelChain";
    case 0x846A9BF0: return "NvAPI_D3D12_LaunchCuKernelChainEx";
    case 0x41C65285: return "NvAPI_D3D12_DestroyCuModule";
    case 0xDF295EA6: return "NvAPI_D3D12_DestroyCuFunction";
    case 0x70C07832: return "NvAPI_D3D12_IsFatbinPTXSupported";
    case 0x329FE6E0: return "NvAPI_D3D12_GetCudaMergedTextureSamplerObject";
    case 0x0DDAC234: return "NvAPI_D3D12_GetCudaIndependentDescriptorObject";
    case 0x80403FC9: return "NvAPI_D3D12_GetCudaTextureObject";
    case 0x48F5B2EE: return "NvAPI_D3D12_GetCudaSurfaceObject";
    case 0x299F5FDC: return "NvAPI_D3D12_CreateCubinComputeShaderExV2";
    case 0x3151211B: return "NvAPI_D3D12_CreateCubinComputeShaderEx";
    case 0x1DC7261F: return "NvAPI_D3D12_CreateCubinComputeShaderWithName";
    case 0x5C52BB86: return "NvAPI_D3D12_LaunchCubinShader";
    case 0x7FB785BA: return "NvAPI_D3D12_DestroyCubinComputeShader";
    default: return nullptr;
    }
}

static void PatchModuleIAT(HMODULE mod, HMODULE self);  // fwd

static double NowMs() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)g_freq.QuadPart;
}

static void Log(const char* fmt, ...) {
    if (!g_log) return;
    EnterCriticalSection(&g_cs);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    fputc('\n', g_log);
    fflush(g_log);
    LeaveCriticalSection(&g_cs);
}

static void JsonEscape(const char* input, char* output, size_t capacity) {
    if (!output || !capacity) return;
    size_t out = 0;
    const unsigned char* p = (const unsigned char*)(input ? input : "");
    while (*p && out + 1 < capacity) {
        const char* replacement = nullptr;
        switch (*p) {
        case '\\': replacement = "\\\\"; break;
        case '"': replacement = "\\\""; break;
        case '\b': replacement = "\\b"; break;
        case '\f': replacement = "\\f"; break;
        case '\n': replacement = "\\n"; break;
        case '\r': replacement = "\\r"; break;
        case '\t': replacement = "\\t"; break;
        default: break;
        }
        if (replacement) {
            size_t length = strlen(replacement);
            if (out + length >= capacity) break;
            memcpy(output + out, replacement, length);
            out += length;
        } else if (*p < 0x20) {
            if (out + 6 >= capacity) break;
            snprintf(output + out, capacity - out, "\\u%04x", (unsigned)*p);
            out += 6;
        } else {
            output[out++] = (char)*p;
        }
        ++p;
    }
    output[out] = 0;
}

#include "amd_head_recording.h"

static void LogResource(const char* method, ID3D12Resource* resource,
                        HRESULT status, UINT heapType, UINT heapFlags,
                        UINT64 heapOffset) {
    if (!resource) {
        Log("{\"ev\":\"d3d12_resource_create\",\"ts\":%.3f,\"method\":\"%s\","
            "\"status\":\"0x%08x\",\"resource\":\"0x0\"}",
            NowMs(), method, (unsigned)status);
        return;
    }
    D3D12_RESOURCE_DESC desc = resource->GetDesc();
    D3D12_GPU_VIRTUAL_ADDRESS va = desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
                                        ? resource->GetGPUVirtualAddress()
                                        : 0;
    if (va) RememberBufferResource(resource, va, desc.Width);
    Log("{\"ev\":\"d3d12_resource_create\",\"ts\":%.3f,\"method\":\"%s\","
        "\"status\":\"0x%08x\",\"resource\":\"0x%llx\","
        "\"gpu_va\":\"0x%llx\",\"dimension\":%u,\"width\":%llu,"
        "\"height\":%u,\"depth_or_array\":%u,\"mips\":%u,\"format\":%u,"
        "\"sample_count\":%u,\"layout\":%u,\"resource_flags\":%u,"
        "\"heap_type\":%u,\"heap_flags\":%u,\"heap_offset\":%llu}",
        NowMs(), method, (unsigned)status,
        (unsigned long long)(uintptr_t)resource, (unsigned long long)va,
        (unsigned)desc.Dimension, (unsigned long long)desc.Width, desc.Height,
        desc.DepthOrArraySize, desc.MipLevels, (unsigned)desc.Format,
        desc.SampleDesc.Count, (unsigned)desc.Layout, (unsigned)desc.Flags,
        heapType, heapFlags, (unsigned long long)heapOffset);
}

static ID3D12Resource* QueryReturnedResource(void** out) {
    if (!out || !*out) return nullptr;
    ID3D12Resource* resource = nullptr;
    ((IUnknown*)*out)->QueryInterface(__uuidof(ID3D12Resource), (void**)&resource);
    return resource;
}

static bool AmdInteropEnabled() {
    char value[8]{};
    DWORD length = GetEnvironmentVariableA(
        "MODULE_TRACE_AMD_INTEROP", value, (DWORD)sizeof(value));
    return length > 0 && length < sizeof(value) && strcmp(value, "0") != 0;
}

typedef int (__cdecl* NvapiAmdRegisterExternalBuffer_t)(void*, void*, uint64_t);

static bool RegisterInteropBuffer(ID3D12Device* device,
                                  ID3D12Resource* resource, UINT64 bytes) {
    if (!device || !resource || !bytes) return false;
    HMODULE backend = GetModuleHandleW(L"nvapi64.dll");
    auto registration = backend
        ? reinterpret_cast<NvapiAmdRegisterExternalBuffer_t>(
              g_realGetProcAddress(backend, "NvapiAmd_RegisterExternalBuffer"))
        : nullptr;
    if (!registration) return false;
    HANDLE shared = nullptr;
    if (FAILED(device->CreateSharedHandle(resource, nullptr, GENERIC_ALL,
                                          nullptr, &shared))) return false;
    bool registered = registration(resource, shared, bytes) != 0;
    CloseHandle(shared);
    return registered;
}

static void STDMETHODCALLTYPE HookCreateShaderResourceView(
        ID3D12Device* device, ID3D12Resource* resource,
        const D3D12_SHADER_RESOURCE_VIEW_DESC* view,
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor) {
    g_realCreateShaderResourceView(device, resource, view, descriptor);
    RememberDescriptorResource(descriptor.ptr, resource);
    D3D12_RESOURCE_DESC rd{};
    if (resource) rd = resource->GetDesc();
    Log("{\"ev\":\"d3d12_create_srv\",\"ts\":%.3f,\"device\":\"0x%llx\","
        "\"resource\":\"0x%llx\",\"descriptor\":\"0x%llx\","
        "\"resource_dimension\":%u,\"resource_width\":%llu,"
        "\"resource_height\":%u,\"resource_format\":%u,"
        "\"view_format\":%u,\"view_dimension\":%u}",
        NowMs(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)resource,
        (unsigned long long)descriptor.ptr, (unsigned)rd.Dimension,
        (unsigned long long)rd.Width, rd.Height, (unsigned)rd.Format,
        view ? (unsigned)view->Format : (unsigned)rd.Format,
        view ? (unsigned)view->ViewDimension : 0u);
}

static void STDMETHODCALLTYPE HookCreateUnorderedAccessView(
        ID3D12Device* device, ID3D12Resource* resource,
        ID3D12Resource* counterResource,
        const D3D12_UNORDERED_ACCESS_VIEW_DESC* view,
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor) {
    g_realCreateUnorderedAccessView(device, resource, counterResource, view, descriptor);
    RememberDescriptorResource(descriptor.ptr, resource);
    D3D12_RESOURCE_DESC rd{};
    if (resource) rd = resource->GetDesc();
    Log("{\"ev\":\"d3d12_create_uav\",\"ts\":%.3f,\"device\":\"0x%llx\","
        "\"resource\":\"0x%llx\",\"counter_resource\":\"0x%llx\","
        "\"descriptor\":\"0x%llx\",\"resource_dimension\":%u,"
        "\"resource_width\":%llu,\"resource_height\":%u,"
        "\"resource_format\":%u,\"view_format\":%u,\"view_dimension\":%u}",
        NowMs(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)resource,
        (unsigned long long)(uintptr_t)counterResource,
        (unsigned long long)descriptor.ptr, (unsigned)rd.Dimension,
        (unsigned long long)rd.Width, rd.Height, (unsigned)rd.Format,
        view ? (unsigned)view->Format : (unsigned)rd.Format,
        view ? (unsigned)view->ViewDimension : 0u);
}

static void STDMETHODCALLTYPE HookCreateSampler(
        ID3D12Device* device, const D3D12_SAMPLER_DESC* desc,
        D3D12_CPU_DESCRIPTOR_HANDLE descriptor) {
    g_realCreateSampler(device, desc, descriptor);
    Log("{\"ev\":\"d3d12_create_sampler\",\"ts\":%.3f,\"device\":\"0x%llx\","
        "\"descriptor\":\"0x%llx\",\"filter\":%u,\"address_u\":%u,"
        "\"address_v\":%u,\"address_w\":%u,\"comparison\":%u}",
        NowMs(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)descriptor.ptr, desc ? (unsigned)desc->Filter : 0u,
        desc ? (unsigned)desc->AddressU : 0u, desc ? (unsigned)desc->AddressV : 0u,
        desc ? (unsigned)desc->AddressW : 0u,
        desc ? (unsigned)desc->ComparisonFunc : 0u);
}

static void LogDescriptorCopy(D3D12_DESCRIPTOR_HEAP_TYPE type,
                              D3D12_CPU_DESCRIPTOR_HANDLE destination,
                              D3D12_CPU_DESCRIPTOR_HANDLE source) {
    Log("{\"ev\":\"d3d12_copy_descriptor\",\"ts\":%.3f,\"heap_type\":%u,"
        "\"destination\":\"0x%llx\",\"source\":\"0x%llx\"}",
        NowMs(), (unsigned)type, (unsigned long long)destination.ptr,
        (unsigned long long)source.ptr);
}

static void STDMETHODCALLTYPE HookCopyDescriptorsSimple(
        ID3D12Device* device, UINT count, D3D12_CPU_DESCRIPTOR_HANDLE destination,
        D3D12_CPU_DESCRIPTOR_HANDLE source, D3D12_DESCRIPTOR_HEAP_TYPE type) {
    g_realCopyDescriptorsSimple(device, count, destination, source, type);
    UINT increment = device->GetDescriptorHandleIncrementSize(type);
    for (UINT i = 0; i < count; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE dst{destination.ptr + SIZE_T(i) * increment};
        D3D12_CPU_DESCRIPTOR_HANDLE src{source.ptr + SIZE_T(i) * increment};
        CopyDescriptorResource(dst.ptr, src.ptr);
        LogDescriptorCopy(type, dst, src);
    }
}

static void STDMETHODCALLTYPE HookCopyDescriptors(
        ID3D12Device* device, UINT destinationRangeCount,
        const D3D12_CPU_DESCRIPTOR_HANDLE* destinationStarts,
        const UINT* destinationSizes, UINT sourceRangeCount,
        const D3D12_CPU_DESCRIPTOR_HANDLE* sourceStarts, const UINT* sourceSizes,
        D3D12_DESCRIPTOR_HEAP_TYPE type) {
    g_realCopyDescriptors(device, destinationRangeCount, destinationStarts,
                          destinationSizes, sourceRangeCount, sourceStarts,
                          sourceSizes, type);
    UINT increment = device->GetDescriptorHandleIncrementSize(type);
    UINT dr = 0, di = 0, sr = 0, si = 0;
    while (dr < destinationRangeCount && sr < sourceRangeCount) {
        UINT dcount = destinationSizes ? destinationSizes[dr] : 1;
        UINT scount = sourceSizes ? sourceSizes[sr] : 1;
        D3D12_CPU_DESCRIPTOR_HANDLE dst{
            destinationStarts[dr].ptr + SIZE_T(di) * increment};
        D3D12_CPU_DESCRIPTOR_HANDLE src{
            sourceStarts[sr].ptr + SIZE_T(si) * increment};
        CopyDescriptorResource(dst.ptr, src.ptr);
        LogDescriptorCopy(type, dst, src);
        if (++di == dcount) { ++dr; di = 0; }
        if (++si == scount) { ++sr; si = 0; }
    }
}

static HRESULT STDMETHODCALLTYPE HookCreateCommittedResource(
        ID3D12Device* device, const D3D12_HEAP_PROPERTIES* heapProperties,
        D3D12_HEAP_FLAGS heapFlags, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE* clearValue,
        REFIID iid, void** out) {
    const bool interopBuffer = AmdInteropEnabled() && heapProperties && desc &&
        heapProperties->Type == D3D12_HEAP_TYPE_DEFAULT &&
        desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
        (desc->Width == 27807744ull || desc->Width == 147719680ull);
    D3D12_HEAP_FLAGS effectiveFlags = interopBuffer
        ? D3D12_HEAP_FLAGS(heapFlags | D3D12_HEAP_FLAG_SHARED)
        : heapFlags;
    HRESULT status = g_realCreateCommittedResource(
        device, heapProperties, effectiveFlags, desc, initialState, clearValue,
        iid, out);
    ID3D12Resource* resource = SUCCEEDED(status) ? QueryReturnedResource(out) : nullptr;
    LogResource("committed", resource, status,
                heapProperties ? (UINT)heapProperties->Type : UINT_MAX,
                (UINT)effectiveFlags, 0);
    if (resource && interopBuffer) {
        bool registered = RegisterInteropBuffer(device, resource, desc->Width);
        Log("{\"ev\":\"amd_interop_buffer_registration\",\"ts\":%.3f,"
            "\"resource\":\"0x%llx\",\"bytes\":%llu,\"registered\":%s}",
            NowMs(), (unsigned long long)(uintptr_t)resource,
            (unsigned long long)desc->Width,
            registered ? "true" : "false");
    }
    if (resource) resource->Release();
    return status;
}

static HRESULT STDMETHODCALLTYPE HookCreatePlacedResource(
        ID3D12Device* device, ID3D12Heap* heap, UINT64 offset,
        const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES initialState,
        const D3D12_CLEAR_VALUE* clearValue, REFIID iid, void** out) {
    HRESULT status = g_realCreatePlacedResource(
        device, heap, offset, desc, initialState, clearValue, iid, out);
    ID3D12Resource* resource = SUCCEEDED(status) ? QueryReturnedResource(out) : nullptr;
    D3D12_HEAP_DESC heapDesc{};
    if (heap) heapDesc = heap->GetDesc();
    LogResource("placed", resource, status, (UINT)heapDesc.Properties.Type,
                (UINT)heapDesc.Flags, offset);
    if (resource) resource->Release();
    return status;
}

static HRESULT STDMETHODCALLTYPE HookCreateReservedResource(
        ID3D12Device* device, const D3D12_RESOURCE_DESC* desc,
        D3D12_RESOURCE_STATES initialState, const D3D12_CLEAR_VALUE* clearValue,
        REFIID iid, void** out) {
    HRESULT status = g_realCreateReservedResource(
        device, desc, initialState, clearValue, iid, out);
    ID3D12Resource* resource = SUCCEEDED(status) ? QueryReturnedResource(out) : nullptr;
    LogResource("reserved", resource, status, UINT_MAX, UINT_MAX, 0);
    if (resource) resource->Release();
    return status;
}

static CommandListTimelineRecord* FindOrAddCommandListTimelineLocked(
        ID3D12CommandList* list) {
    for (size_t i = 0; i < g_commandListTimelineCount; ++i) {
        if (g_commandListTimeline[i].list == list)
            return &g_commandListTimeline[i];
    }
    if (!list || g_commandListTimelineCount >= _countof(g_commandListTimeline))
        return nullptr;
    auto* record = &g_commandListTimeline[g_commandListTimelineCount++];
    memset(record, 0, sizeof(*record));
    record->list = list;
    record->firstLaunchCall = -1;
    record->lastLaunchCall = -1;
    record->firstFrame = -1;
    record->lastFrame = -1;
    record->firstSlot = -1;
    record->lastSlot = -1;
    return record;
}

static CommandListTimelineSnapshot SnapshotCommandListTimeline(
        ID3D12CommandList* list, bool countClose, bool countExecute) {
    CommandListTimelineSnapshot snapshot{};
    snapshot.firstLaunchCall = snapshot.lastLaunchCall = -1;
    snapshot.firstFrame = snapshot.lastFrame = -1;
    snapshot.firstSlot = snapshot.lastSlot = -1;
    EnterCriticalSection(&g_resourceCs);
    auto* record = FindOrAddCommandListTimelineLocked(list);
    if (record) {
        if (countClose) ++record->closeCount;
        if (countExecute) ++record->executeCount;
        snapshot.firstLaunchCall = record->firstLaunchCall;
        snapshot.lastLaunchCall = record->lastLaunchCall;
        snapshot.firstFrame = record->firstFrame;
        snapshot.lastFrame = record->lastFrame;
        snapshot.firstSlot = record->firstSlot;
        snapshot.lastSlot = record->lastSlot;
        snapshot.launchCount = record->launchCount;
        snapshot.closeCount = record->closeCount;
        snapshot.executeCount = record->executeCount;
    }
    LeaveCriticalSection(&g_resourceCs);
    return snapshot;
}

static LONG RememberCommandListLaunch(ID3D12CommandList* list, LONG call,
                                      LONG frame, LONG slot) {
    LONG sequence = InterlockedIncrement(&g_submissionTimelineSequence);
    EnterCriticalSection(&g_resourceCs);
    auto* record = FindOrAddCommandListTimelineLocked(list);
    if (record) {
        if (!record->launchCount) {
            record->firstLaunchCall = call;
            record->firstFrame = frame;
            record->firstSlot = slot;
        }
        record->lastLaunchCall = call;
        record->lastFrame = frame;
        record->lastSlot = slot;
        ++record->launchCount;
    }
    LeaveCriticalSection(&g_resourceCs);
    return sequence;
}

static bool PatchDeviceVtableEntry(void** vtable, size_t index, void* hook,
                                   void** original) {
    if (!vtable || !hook || !original) return false;
    *original = vtable[index];
    DWORD oldProtect = 0;
    if (!VirtualProtect(&vtable[index], sizeof(void*), PAGE_EXECUTE_READWRITE,
                        &oldProtect)) return false;
    InterlockedExchangePointer((PVOID volatile*)&vtable[index], hook);
    DWORD ignored = 0;
    VirtualProtect(&vtable[index], sizeof(void*), oldProtect, &ignored);
    FlushInstructionCache(GetCurrentProcess(), &vtable[index], sizeof(void*));
    return true;
}

static HRESULT STDMETHODCALLTYPE HookCommandListClose(
        ID3D12GraphicsCommandList* list) {
    LONG sequence = InterlockedIncrement(&g_submissionTimelineSequence);
    auto snapshot = SnapshotCommandListTimeline(list, true, false);
    Log("{\"ev\":\"d3d12_command_list_close_call\",\"ts\":%.3f,"
        "\"sequence\":%ld,\"command_list\":\"0x%llx\","
        "\"launch_count\":%u,\"first_launch_call\":%ld,"
        "\"last_launch_call\":%ld,\"first_frame\":%ld,\"last_frame\":%ld,"
        "\"first_slot\":%ld,\"last_slot\":%ld,\"close_count\":%u}",
        NowMs(), sequence, (unsigned long long)(uintptr_t)list,
        snapshot.launchCount, snapshot.firstLaunchCall, snapshot.lastLaunchCall,
        snapshot.firstFrame, snapshot.lastFrame, snapshot.firstSlot,
        snapshot.lastSlot, snapshot.closeCount);
    HRESULT status = g_realCommandListClose
        ? g_realCommandListClose(list) : E_UNEXPECTED;
    if (SUCCEEDED(status) && auto_head::Enabled()) auto_head::Closed(list);
    Log("{\"ev\":\"d3d12_command_list_close_ret\",\"ts\":%.3f,"
        "\"sequence\":%ld,\"command_list\":\"0x%llx\","
        "\"hresult\":\"0x%08X\"}", NowMs(), sequence,
        (unsigned long long)(uintptr_t)list, (unsigned)status);
    return status;
}

static void STDMETHODCALLTYPE HookExecuteCommandLists(
        ID3D12CommandQueue* queue, UINT count,
        ID3D12CommandList* const* lists) {
    LONG sequence = InterlockedIncrement(&g_submissionTimelineSequence);
    Log("{\"ev\":\"d3d12_execute_command_lists_call\",\"ts\":%.3f,"
        "\"sequence\":%ld,\"queue\":\"0x%llx\",\"count\":%u}",
        NowMs(), sequence, (unsigned long long)(uintptr_t)queue, count);
    for (UINT i = 0; lists && i < count; ++i) {
        auto snapshot = SnapshotCommandListTimeline(lists[i], false, true);
        Log("{\"ev\":\"d3d12_execute_command_list\",\"ts\":%.3f,"
            "\"sequence\":%ld,\"queue\":\"0x%llx\",\"index\":%u,"
            "\"command_list\":\"0x%llx\",\"launch_count\":%u,"
            "\"first_launch_call\":%ld,\"last_launch_call\":%ld,"
            "\"first_frame\":%ld,\"last_frame\":%ld,"
            "\"first_slot\":%ld,\"last_slot\":%ld,"
            "\"close_count\":%u,\"execute_count\":%u}",
            NowMs(), sequence, (unsigned long long)(uintptr_t)queue, i,
            (unsigned long long)(uintptr_t)lists[i], snapshot.launchCount,
            snapshot.firstLaunchCall, snapshot.lastLaunchCall,
            snapshot.firstFrame, snapshot.lastFrame, snapshot.firstSlot,
            snapshot.lastSlot, snapshot.closeCount, snapshot.executeCount);
    }
    if (!auto_head::BeforeSubmit(queue, count, lists)) return;
    if (g_realExecuteCommandLists)
        g_realExecuteCommandLists(queue, count, lists);
    auto_head::AfterSubmit(queue, count, lists);
    Log("{\"ev\":\"d3d12_execute_command_lists_ret\",\"ts\":%.3f,"
        "\"sequence\":%ld,\"queue\":\"0x%llx\"}", NowMs(), sequence,
        (unsigned long long)(uintptr_t)queue);
}

static bool InstallCommandQueueHooks(ID3D12CommandQueue* queue) {
    if (!queue) return false;
    if (InterlockedCompareExchange(&g_queueHooksInstalled, 1, 0) != 0)
        return g_queueHooksInstalled > 0;
    void** vtable = *(void***)queue;
    bool ok = PatchDeviceVtableEntry(vtable, 10,
        (void*)&HookExecuteCommandLists, (void**)&g_realExecuteCommandLists);
    if (!ok) InterlockedExchange(&g_queueHooksInstalled, -1);
    return ok;
}

static void STDMETHODCALLTYPE HookResourceBarrier(ID3D12GraphicsCommandList* list, UINT count, const D3D12_RESOURCE_BARRIER* barriers) {
    auto_head::Barriers(list, count, barriers);
    g_realResourceBarrier(list, count, barriers);
}
static HRESULT STDMETHODCALLTYPE HookListReset(ID3D12GraphicsCommandList* list, ID3D12CommandAllocator* allocator, ID3D12PipelineState* pipeline) {
    if (!auto_head::ResetAllowed(list)) return E_PENDING;
    const auto status = g_realListReset(list, allocator, pipeline);
    if (SUCCEEDED(status) && auto_head::Enabled()) auto_head::ResetStates(list);
    return status;
}
static void STDMETHODCALLTYPE HookCopyBuffer(ID3D12GraphicsCommandList* list, ID3D12Resource* dest, UINT64 dstOffset, ID3D12Resource* src, UINT64 srcOffset, UINT64 bytes) {
    auto_head::CopyUse(list, dest); auto_head::CopyUse(list, src);
    g_realCopyBuffer(list, dest, dstOffset, src, srcOffset, bytes);
}
static void STDMETHODCALLTYPE HookCopyTexture(ID3D12GraphicsCommandList* list, const D3D12_TEXTURE_COPY_LOCATION* dest, UINT x, UINT y, UINT z, const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box) {
    if (dest) auto_head::CopyUse(list, dest->pResource); if (src) auto_head::CopyUse(list, src->pResource);
    g_realCopyTexture(list, dest, x, y, z, src, box);
}
static void STDMETHODCALLTYPE HookCopyResource(ID3D12GraphicsCommandList* list, ID3D12Resource* dest, ID3D12Resource* src) {
    auto_head::CopyUse(list, dest); auto_head::CopyUse(list, src); g_realCopyResource(list, dest, src);
}
static void STDMETHODCALLTYPE HookDispatch(ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z) {
    auto_head::DispatchUse(list); g_realDispatch(list, x, y, z);
}

static bool InstallCommandListHooks(ID3D12GraphicsCommandList* list) {
    if (!list) return false;
    if (InterlockedCompareExchange(&g_commandListHooksInstalled, 1, 0) != 0)
        return g_commandListHooksInstalled > 0;
    void** vtable = *(void***)list;
    bool ok = PatchDeviceVtableEntry(vtable, 9,
        (void*)&HookCommandListClose, (void**)&g_realCommandListClose);
    ok &= PatchDeviceVtableEntry(vtable, 10, (void*)&HookListReset, (void**)&g_realListReset);
    ok &= PatchDeviceVtableEntry(vtable, 14, (void*)&HookDispatch, (void**)&g_realDispatch);
    ok &= PatchDeviceVtableEntry(vtable, 15, (void*)&HookCopyBuffer, (void**)&g_realCopyBuffer);
    ok &= PatchDeviceVtableEntry(vtable, 16, (void*)&HookCopyTexture, (void**)&g_realCopyTexture);
    ok &= PatchDeviceVtableEntry(vtable, 17, (void*)&HookCopyResource, (void**)&g_realCopyResource);
    ok &= PatchDeviceVtableEntry(vtable, 26, (void*)&HookResourceBarrier, (void**)&g_realResourceBarrier);
    if (!ok) InterlockedExchange(&g_commandListHooksInstalled, -1);
    return ok;
}

static HRESULT STDMETHODCALLTYPE HookCreateCommandQueue(
        ID3D12Device* device, const D3D12_COMMAND_QUEUE_DESC* desc,
        REFIID iid, void** out) {
    HRESULT status = g_realCreateCommandQueue
        ? g_realCreateCommandQueue(device, desc, iid, out) : E_UNEXPECTED;
    ID3D12CommandQueue* queue = nullptr;
    bool installed = false;
    if (SUCCEEDED(status) && out && *out) {
        auto* unknown = reinterpret_cast<IUnknown*>(*out);
        if (SUCCEEDED(unknown->QueryInterface(IID_PPV_ARGS(&queue))))
            installed = InstallCommandQueueHooks(queue);
    }
    Log("{\"ev\":\"d3d12_command_queue_create\",\"ts\":%.3f,"
        "\"device\":\"0x%llx\",\"queue\":\"0x%llx\",\"type\":%u,"
        "\"hresult\":\"0x%08X\",\"hooks_installed\":%s}", NowMs(),
        (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)queue,
        desc ? (unsigned)desc->Type : UINT_MAX, (unsigned)status,
        installed ? "true" : "false");
    if (queue) queue->Release();
    return status;
}

static HRESULT STDMETHODCALLTYPE HookCreateCommandList(
        ID3D12Device* device, UINT nodeMask, D3D12_COMMAND_LIST_TYPE type,
        ID3D12CommandAllocator* allocator, ID3D12PipelineState* initialState,
        REFIID iid, void** out) {
    HRESULT status = g_realCreateCommandList
        ? g_realCreateCommandList(device, nodeMask, type, allocator,
                                  initialState, iid, out) : E_UNEXPECTED;
    ID3D12GraphicsCommandList* list = nullptr;
    bool installed = false;
    if (SUCCEEDED(status) && out && *out) {
        auto* unknown = reinterpret_cast<IUnknown*>(*out);
        if (SUCCEEDED(unknown->QueryInterface(IID_PPV_ARGS(&list)))) {
            installed = InstallCommandListHooks(list);
            if (auto_head::Enabled()) auto_head::ResetStates(list);
            SnapshotCommandListTimeline(list, false, false);
        }
    }
    Log("{\"ev\":\"d3d12_command_list_create\",\"ts\":%.3f,"
        "\"device\":\"0x%llx\",\"command_list\":\"0x%llx\","
        "\"type\":%u,\"hresult\":\"0x%08X\",\"hooks_installed\":%s}",
        NowMs(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)list, (unsigned)type,
        (unsigned)status, installed ? "true" : "false");
    if (list) list->Release();
    return status;
}

static bool InstallDeviceHooks(ID3D12Device* device) {
    if (!device) return false;
    if (InterlockedCompareExchange(&g_deviceHooksInstalled, 1, 0) != 0)
        return true;
    void** vtable = *(void***)device;
    bool ok = true;
    ok &= PatchDeviceVtableEntry(vtable, 8, (void*)&HookCreateCommandQueue,
                                 (void**)&g_realCreateCommandQueue);
    ok &= PatchDeviceVtableEntry(vtable, 12, (void*)&HookCreateCommandList,
                                 (void**)&g_realCreateCommandList);
    ok &= PatchDeviceVtableEntry(vtable, 18, (void*)&HookCreateShaderResourceView,
                                 (void**)&g_realCreateShaderResourceView);
    ok &= PatchDeviceVtableEntry(vtable, 19, (void*)&HookCreateUnorderedAccessView,
                                 (void**)&g_realCreateUnorderedAccessView);
    ok &= PatchDeviceVtableEntry(vtable, 22, (void*)&HookCreateSampler,
                                 (void**)&g_realCreateSampler);
    ok &= PatchDeviceVtableEntry(vtable, 23, (void*)&HookCopyDescriptors,
                                 (void**)&g_realCopyDescriptors);
    ok &= PatchDeviceVtableEntry(vtable, 24, (void*)&HookCopyDescriptorsSimple,
                                 (void**)&g_realCopyDescriptorsSimple);
    ok &= PatchDeviceVtableEntry(vtable, 27, (void*)&HookCreateCommittedResource,
                                 (void**)&g_realCreateCommittedResource);
    ok &= PatchDeviceVtableEntry(vtable, 29, (void*)&HookCreatePlacedResource,
                                 (void**)&g_realCreatePlacedResource);
    ok &= PatchDeviceVtableEntry(vtable, 30, (void*)&HookCreateReservedResource,
                                 (void**)&g_realCreateReservedResource);
    if (!ok) InterlockedExchange(&g_deviceHooksInstalled, -1);
    Log("{\"ev\":\"d3d12_device_hooks\",\"ts\":%.3f,\"device\":\"0x%llx\","
        "\"installed\":%s}", NowMs(), (unsigned long long)(uintptr_t)device,
        ok ? "true" : "false");
    return ok;
}

static void LogWide(const char* ev, const wchar_t* wpath, HMODULE res) {
    char path[1024] = {0};
    char escaped[2048] = {0};
    if (wpath) WideCharToMultiByte(CP_UTF8, 0, wpath, -1, path, sizeof(path) - 1, nullptr, nullptr);
    JsonEscape(path, escaped, sizeof(escaped));
    Log("{\"ev\":\"%s\",\"ts\":%.3f,\"tid\":%lu,\"path\":\"%s\",\"result\":\"0x%llx\"}",
        ev, NowMs(), GetCurrentThreadId(), escaped, (unsigned long long)(uintptr_t)res);
}
static void LogNarrow(const char* ev, const char* path, HMODULE res) {
    char escaped[2048] = {0};
    JsonEscape(path, escaped, sizeof(escaped));
    Log("{\"ev\":\"%s\",\"ts\":%.3f,\"tid\":%lu,\"path\":\"%s\",\"result\":\"0x%llx\"}",
        ev, NowMs(), GetCurrentThreadId(), escaped, (unsigned long long)(uintptr_t)res);
}

static int32_t __cdecl HookCreateCuModule(void* device, const void* blob,
                                         uint32_t size, void** outModule) {
    unsigned char head[8] = {};
    if (blob && size) memcpy(head, blob, size < sizeof(head) ? size : sizeof(head));
    uint64_t fnv = 14695981039346656037ull;
    if (blob) {
        const unsigned char* bytes = (const unsigned char*)blob;
        for (uint32_t i = 0; i < size; ++i) {
            fnv ^= bytes[i];
            fnv *= 1099511628211ull;
        }
    }
    NvApiCreateCuModule_t real = g_realCreateCuModule;
    int32_t status = real ? real(device, blob, size, outModule) : -136;
    void* module = outModule ? *outModule : nullptr;
    Log("{\"ev\":\"nvapi_create_cu_module\",\"ts\":%.3f,\"tid\":%lu,\"device\":\"0x%llx\",\"blob\":\"0x%llx\",\"size\":%u,\"head\":\"%02X%02X%02X%02X%02X%02X%02X%02X\",\"fnv1a64\":\"%016llX\",\"status\":%d,\"module\":\"0x%llx\"}",
        NowMs(), GetCurrentThreadId(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)blob, size, head[0], head[1], head[2], head[3],
        head[4], head[5], head[6], head[7], (unsigned long long)fnv, status,
        (unsigned long long)(uintptr_t)module);
    return status;
}

static int32_t __cdecl HookCreateCuFunction(void* device, void* module,
                                           const char* name, void** outFunction) {
    NvApiCreateCuFunction_t real = g_realCreateCuFunction;
    int32_t status = real ? real(device, module, name, outFunction) : -136;
    void* function = outFunction ? *outFunction : nullptr;
    Log("{\"ev\":\"nvapi_create_cu_function\",\"ts\":%.3f,\"tid\":%lu,\"device\":\"0x%llx\",\"module\":\"0x%llx\",\"name\":\"%.240s\",\"status\":%d,\"function\":\"0x%llx\"}",
        NowMs(), GetCurrentThreadId(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)module, name ? name : "", status,
        (unsigned long long)(uintptr_t)function);
    return status;
}

static HRESULT CreateLinearReadback(ID3D12Device* device, UINT64 size,
                                    ID3D12Resource** output) {
    if (!device || !size || !output) return E_INVALIDARG;
    *output = nullptr;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(output));
}

static bool BeginNeuralCapture(ID3D12GraphicsCommandList* list,
                               const void* parameterData, uint32_t parameterSize) {
    if (!list || !parameterData || parameterSize < 264 ||
        InterlockedCompareExchange(&g_neuralCaptureState, 1, 0) != 0)
        return false;
    const unsigned char* data = (const unsigned char*)parameterData;
    uint64_t pointers[kNeuralWindowCount]{};
    memcpy(&pointers[0], data + 216, sizeof(uint64_t));
    memcpy(&pointers[1], data + 224, sizeof(uint64_t));
    memcpy(&pointers[2], data + 248, sizeof(uint64_t));
    for (size_t i = 0; i < kDownstreamWeightCount; ++i)
        pointers[3 + i] = pointers[1] + kDownstreamWeightOffsets[i];
    memcpy(&g_neuralPaddedHeight, data + 240, sizeof(uint32_t));
    memcpy(&g_neuralPaddedWidth, data + 244, sizeof(uint32_t));
    ResolvedBufferAddress resolved[kNeuralWindowCount]{};
    bool allResolved = true;
    for (size_t i = 0; i < kNeuralWindowCount; ++i) {
        resolved[i] = ResolveBufferAddress(pointers[i]);
        allResolved &= resolved[i].resource != nullptr;
    }
    if (!allResolved) {
        InterlockedExchange(&g_neuralCaptureState, -1);
        Log("{\"ev\":\"neural_capture_arm\",\"ts\":%.3f,\"status\":\"FAIL\","
            "\"reason\":\"unresolved_buffer_address\",\"p216\":\"0x%llx\","
            "\"p224\":\"0x%llx\",\"p248\":\"0x%llx\","
            "\"downstream_window_count\":%llu,\"first_downstream\":\"0x%llx\","
            "\"last_downstream\":\"0x%llx\"}", NowMs(),
            (unsigned long long)pointers[0], (unsigned long long)pointers[1],
            (unsigned long long)pointers[2],
            (unsigned long long)kDownstreamWeightCount,
            (unsigned long long)pointers[3],
            (unsigned long long)pointers[kNeuralWindowCount - 1]);
        return false;
    }

    UINT64 scratchSize = resolved[0].remaining < (8ull << 20)
                             ? resolved[0].remaining : (8ull << 20);
    if (resolved[0].resource == resolved[2].resource &&
        pointers[2] > pointers[0] && pointers[2] - pointers[0] <= resolved[0].remaining)
        scratchSize = pointers[2] - pointers[0];
    UINT64 weightsSize = resolved[1].remaining < 65536 ? resolved[1].remaining : 65536;
    // PTX store addressing covers padded_height * (padded_width / 2) cells,
    // with sixteen packed FP8 bytes per cell. This is a capture envelope, not
    // yet a semantic tensor-shape assertion.
    UINT64 outputEnvelope = UINT64(g_neuralPaddedHeight) *
                            UINT64(g_neuralPaddedWidth / 2) * 16;
    if (!outputEnvelope || outputEnvelope > (4ull << 20)) outputEnvelope = 2ull << 20;
    UINT64 outputSize = resolved[2].remaining < outputEnvelope
                            ? resolved[2].remaining : outputEnvelope;
    const char* names[kNeuralWindowCount]{};
    uint32_t offsets[kNeuralWindowCount]{};
    UINT64 weightViewOffsets[kNeuralWindowCount]{};
    UINT64 sizes[kNeuralWindowCount]{};
    names[0] = "scratch"; names[1] = "weights"; names[2] = "output";
    offsets[0] = 216; offsets[1] = 224; offsets[2] = 248;
    sizes[0] = scratchSize; sizes[1] = weightsSize; sizes[2] = outputSize;
    for (size_t i = 0; i < kDownstreamWeightCount; ++i) {
        names[3 + i] = kDownstreamWeightNames[i];
        offsets[3 + i] = 224;
        weightViewOffsets[3 + i] = kDownstreamWeightOffsets[i];
        constexpr UINT64 kDownstreamCaptureLimit = 1ull << 20;
        sizes[3 + i] = resolved[3 + i].remaining < kDownstreamCaptureLimit
                           ? resolved[3 + i].remaining
                           : kDownstreamCaptureLimit;
    }
    ID3D12Device* device = nullptr;
    HRESULT hr = list->GetDevice(IID_PPV_ARGS(&device));
    if (FAILED(hr) || !device) goto fail;
    for (size_t i = 0; i < kNeuralWindowCount; ++i) {
        if (resolved[i].remaining < sizes[i]) {
            hr = HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
            goto fail;
        }
        g_neuralWindows[i].name = names[i];
        g_neuralWindows[i].paramOffset = offsets[i];
        g_neuralWindows[i].weightViewOffset = weightViewOffsets[i];
        g_neuralWindows[i].pointer = pointers[i];
        g_neuralWindows[i].source = resolved[i].resource;
        g_neuralWindows[i].source->AddRef();
        g_neuralWindows[i].sourceOffset = resolved[i].offset;
        g_neuralWindows[i].size = sizes[i];
        hr = CreateLinearReadback(device, sizes[i], &g_neuralWindows[i].before);
        if (FAILED(hr)) goto fail;
        hr = CreateLinearReadback(device, sizes[i], &g_neuralWindows[i].after);
        if (FAILED(hr)) goto fail;
    }
    for (size_t i = 0; i < kNeuralWindowCount; ++i) {
        const auto& window = g_neuralWindows[i];
        list->CopyBufferRegion(window.before, 0, window.source,
                               window.sourceOffset, window.size);
    }
    device->Release();
    Log("{\"ev\":\"neural_capture_arm\",\"ts\":%.3f,\"status\":\"PASS\","
        "\"padded_height\":%u,\"padded_width\":%u,\"scratch_bytes\":%llu,"
        "\"weights_bytes\":%llu,\"output_bytes\":%llu,"
        "\"downstream_window_count\":%llu,\"downstream_window_limit_bytes\":1048576}", NowMs(),
        g_neuralPaddedHeight, g_neuralPaddedWidth,
        (unsigned long long)g_neuralWindows[0].size,
        (unsigned long long)g_neuralWindows[1].size,
        (unsigned long long)g_neuralWindows[2].size,
        (unsigned long long)kDownstreamWeightCount);
    return true;

fail:
    if (device) device->Release();
    for (auto& window : g_neuralWindows) {
        if (window.before) { window.before->Release(); window.before = nullptr; }
        if (window.after) { window.after->Release(); window.after = nullptr; }
        if (window.source) { window.source->Release(); window.source = nullptr; }
    }
    InterlockedExchange(&g_neuralCaptureState, -1);
    Log("{\"ev\":\"neural_capture_arm\",\"ts\":%.3f,\"status\":\"FAIL\","
        "\"reason\":\"readback_creation\",\"hresult\":\"0x%08X\"}",
        NowMs(), (unsigned)hr);
    return false;
}

static void EndNeuralCapture(ID3D12GraphicsCommandList* list) {
    if (!list || InterlockedCompareExchange(&g_neuralCaptureState, 2, 1) != 1)
        return;
    for (const auto& window : g_neuralWindows) {
        list->CopyBufferRegion(window.after, 0, window.source,
                               window.sourceOffset, window.size);
    }
    Log("{\"ev\":\"neural_capture_recorded\",\"ts\":%.3f,\"status\":\"PASS\"}",
        NowMs());
}

struct FullGraphCaptureSpan {
    size_t first;
    size_t count;
};

static FullGraphCaptureSpan BeginFullGraphCapture(
        ID3D12GraphicsCommandList* list, uint32_t slot, uint64_t function,
        const void* parameterData, uint32_t parameterSize) {
    FullGraphCaptureSpan span{};
    if (!list || !parameterData || parameterSize < sizeof(uint64_t)) return span;
    struct Candidate {
        uint32_t paramOffset;
        uint64_t pointer;
        ResolvedBufferAddress resolved;
    };
    Candidate candidates[40]{};
    size_t candidateCount = 0;
    const unsigned char* data = (const unsigned char*)parameterData;
    uint32_t scanBytes = parameterSize < 320 ? parameterSize : 320;
    for (uint32_t offset = 0; offset + sizeof(uint64_t) <= scanBytes;
         offset += sizeof(uint64_t)) {
        uint64_t pointer = 0;
        memcpy(&pointer, data + offset, sizeof(pointer));
        if (!pointer) continue;
        ResolvedBufferAddress resolved = ResolveBufferAddress(pointer);
        if (!resolved.resource || !resolved.remaining) continue;
        bool duplicate = false;
        for (size_t i = 0; i < candidateCount; ++i)
            duplicate |= candidates[i].pointer == pointer;
        if (duplicate || candidateCount >= _countof(candidates)) continue;
        candidates[candidateCount++] = {offset, pointer, resolved};
    }
    if (!candidateCount) {
        Log("{\"ev\":\"full_graph_capture_arm\",\"ts\":%.3f,"
            "\"status\":\"SKIP\",\"slot\":%u,\"reason\":\"no_buffer_pointer\"}",
            NowMs(), slot);
        return span;
    }
    ID3D12Device* device = nullptr;
    HRESULT hr = list->GetDevice(IID_PPV_ARGS(&device));
    if (FAILED(hr) || !device) {
        InterlockedIncrement(&g_fullGraphCaptureFailures);
        return span;
    }
    const UINT64 limit = FullGraphCaptureLimit();
    EnterCriticalSection(&g_resourceCs);
    span.first = g_fullGraphWindowCount;
    for (size_t i = 0; i < candidateCount; ++i) {
        if (g_fullGraphWindowCount >= kFullGraphWindowCapacity) {
            InterlockedIncrement(&g_fullGraphCaptureFailures);
            break;
        }
        UINT64 size = candidates[i].resolved.remaining < limit
                          ? candidates[i].resolved.remaining : limit;
        for (size_t j = 0; j < candidateCount; ++j) {
            if (candidates[j].resolved.resource == candidates[i].resolved.resource &&
                candidates[j].pointer > candidates[i].pointer) {
                UINT64 distance = candidates[j].pointer - candidates[i].pointer;
                if (distance < size) size = distance;
            }
        }
        if (!size) continue;
        FullGraphCaptureWindow window{};
        window.slot = slot;
        window.paramOffset = candidates[i].paramOffset;
        window.function = function;
        window.pointer = candidates[i].pointer;
        window.source = candidates[i].resolved.resource;
        window.source->AddRef();
        window.sourceOffset = candidates[i].resolved.offset;
        window.size = size;
        hr = CreateLinearReadback(device, size, &window.before);
        if (SUCCEEDED(hr)) hr = CreateLinearReadback(device, size, &window.after);
        if (FAILED(hr) || !window.before || !window.after) {
            if (window.before) window.before->Release();
            if (window.after) window.after->Release();
            window.source->Release();
            InterlockedIncrement(&g_fullGraphCaptureFailures);
            continue;
        }
        list->CopyBufferRegion(window.before, 0, window.source,
                               window.sourceOffset, window.size);
        g_fullGraphWindows[g_fullGraphWindowCount++] = window;
        ++span.count;
    }
    LeaveCriticalSection(&g_resourceCs);
    device->Release();
    Log("{\"ev\":\"full_graph_capture_arm\",\"ts\":%.3f,"
        "\"status\":\"%s\",\"slot\":%u,\"windows\":%llu,"
        "\"capture_limit_bytes\":%llu}", NowMs(),
        span.count ? "PASS" : "FAIL", slot,
        (unsigned long long)span.count, (unsigned long long)limit);
    return span;
}

static void EndFullGraphCapture(ID3D12GraphicsCommandList* list,
                                FullGraphCaptureSpan span) {
    if (!list || !span.count) return;
    EnterCriticalSection(&g_resourceCs);
    size_t end = span.first + span.count;
    if (end > g_fullGraphWindowCount) end = g_fullGraphWindowCount;
    for (size_t i = span.first; i < end; ++i) {
        const auto& window = g_fullGraphWindows[i];
        list->CopyBufferRegion(window.after, 0, window.source,
                               window.sourceOffset, window.size);
    }
    LeaveCriticalSection(&g_resourceCs);
}

static int32_t __cdecl HookLaunchCuKernelChain(void* commandList,
                                              const NvApiCuKernelLaunchParams* kernels,
                                              uint32_t numKernels) {
    LONG call = InterlockedIncrement(&g_launchChainCalls);
    // R-20 established one capability probe followed by exactly 156 launches per
    // frame. Capture complete frames 1/61/121/181/241, avoiding v8's 60-call
    // sampling alias while keeping the 300-frame log compact.
    LONG frame = call >= 2 ? ((call - 2) / 156) + 1 : 0;
    LONG slot = call >= 2 ? (call - 2) % 156 : -1;
    LONG sequence = RememberCommandListLaunch(
        reinterpret_cast<ID3D12CommandList*>(commandList), call, frame, slot);
    const bool inlineSnapshots = InlineSnapshotsEnabled();
    if (slot == 154 && kernels && numKernels && kernels[0].pParams &&
        kernels[0].paramSize >= 64) {
        if (inlineSnapshots) {
            BeginPostActivationPreLaunchCapture(
                (ID3D12GraphicsCommandList*)commandList, kernels[0].pParams,
                kernels[0].paramSize, frame);
        }
        uint64_t surfaceObject = 0;
        memcpy(&surfaceObject,
               (const unsigned char*)kernels[0].pParams + 16,
               sizeof(surfaceObject));
        ID3D12Resource* surfaceResource = FindObjectResource(surfaceObject);
        bool surfaceInitialCapture = inlineSnapshots && frame == 1 &&
            BeginFrame1PostSurfaceInitialCapture(
                (ID3D12GraphicsCommandList*)commandList, surfaceResource,
                surfaceObject, frame);
        Log("{\"ev\":\"post_surface_resource_bind\",\"ts\":%.3f,\"frame\":%ld,"
            "\"slot\":%ld,\"surface_object\":\"0x%016llX\","
            "\"surface_resource\":\"0x%llx\",\"initial_capture\":%s}",
            NowMs(), frame, slot, (unsigned long long)surfaceObject,
            (unsigned long long)(uintptr_t)surfaceResource,
            surfaceInitialCapture ? "true" : "false");
        uint64_t textureObject = 0;
        memcpy(&textureObject,
               (const unsigned char*)kernels[0].pParams + 56,
               sizeof(textureObject));
        ID3D12Resource* textureResource = FindObjectResource(textureObject);
        if (inlineSnapshots)
            RememberPostTextureResource(textureObject, textureResource);
        bool preLaunchCapture = inlineSnapshots && frame == 1 && BeginPostTexturePreLaunchCapture(
            (ID3D12GraphicsCommandList*)commandList, textureResource,
            textureObject, frame);
        Log("{\"ev\":\"post_texture_resource_bind\",\"ts\":%.3f,\"frame\":%ld,"
            "\"slot\":%ld,\"texture_object\":\"0x%016llX\","
            "\"texture_resource\":\"0x%llx\",\"prelaunch_capture\":%s}",
            NowMs(), frame, slot, (unsigned long long)textureObject,
            (unsigned long long)(uintptr_t)textureResource,
            preLaunchCapture ? "true" : "false");
    }
    if (slot == 155 && kernels && numKernels && kernels[0].pParams &&
        kernels[0].paramSize >= sizeof(uint64_t) * 2) {
        uint64_t inputObject = 0, outputObject = 0;
        memcpy(&inputObject, kernels[0].pParams, sizeof(inputObject));
        memcpy(&outputObject,
               (const unsigned char*)kernels[0].pParams + sizeof(inputObject),
               sizeof(outputObject));
        ID3D12Resource* input = FindObjectResource(inputObject);
        ID3D12Resource* output = FindObjectResource(outputObject);
        RememberCopyResources(input, output);
        bool frameOutputCapture = inlineSnapshots && frame == 1 && BeginFrame1PostOutputCapture(
            (ID3D12GraphicsCommandList*)commandList, input, inputObject, frame);
        Log("{\"ev\":\"copy_resource_bind\",\"ts\":%.3f,\"frame\":%ld,"
            "\"input_object\":\"0x%016llX\",\"output_object\":\"0x%016llX\","
            "\"input_resource\":\"0x%llx\",\"output_resource\":\"0x%llx\","
            "\"frame_output_capture\":%s}",
            NowMs(), frame, (unsigned long long)inputObject,
            (unsigned long long)outputObject,
            (unsigned long long)(uintptr_t)input,
            (unsigned long long)(uintptr_t)output,
            frameOutputCapture ? "true" : "false");
    }
    bool neuralCapture = inlineSnapshots && frame == 1 && slot == 1 && kernels && numKernels &&
                         kernels[0].pParams && kernels[0].paramSize >= 264 &&
                         BeginNeuralCapture((ID3D12GraphicsCommandList*)commandList,
                                            kernels[0].pParams, kernels[0].paramSize);
    FullGraphCaptureSpan fullGraphSpan{};
    // Slots 0-154 expose tracked D3D12 buffer virtual addresses. Slot 155 is
    // cg2r_copy_kernel and uses descriptor objects, so it is covered by the
    // copy_resource_bind/copy_snapshot path above instead of buffer windows.
    if (frame == 1 && slot >= 0 && slot < 155 && FullGraphCaptureEnabled() &&
        kernels && numKernels && kernels[0].pParams) {
        fullGraphSpan = BeginFullGraphCapture(
            (ID3D12GraphicsCommandList*)commandList, (uint32_t)slot,
            (uint64_t)(uintptr_t)kernels[0].hFunction,
            kernels[0].pParams, kernels[0].paramSize);
    }
    bool detail = call == 1 || (frame > 0 && ((frame - 1) % 60) == 0);
    Log("{\"ev\":\"nvapi_launch_cu_kernel_chain_call\",\"ts\":%.3f,\"tid\":%lu,\"sequence\":%ld,\"call\":%ld,\"frame\":%ld,\"slot\":%ld,\"command_list\":\"0x%llx\",\"num_kernels\":%u,\"detail\":%s}",
        NowMs(), GetCurrentThreadId(), sequence, call,
        frame, slot, (unsigned long long)(uintptr_t)commandList, numKernels,
        detail ? "true" : "false");
    if (detail && kernels) {
        uint32_t limit = numKernels < 256 ? numKernels : 256;
        for (uint32_t i = 0; i < limit; ++i) {
            const auto& k = kernels[i];
            char paramHex[641] = {};
            uint32_t captured = k.paramSize < 320 ? k.paramSize : 320;
            static const char digits[] = "0123456789ABCDEF";
            if (k.pParams) {
                const unsigned char* p = (const unsigned char*)k.pParams;
                for (uint32_t j = 0; j < captured; ++j) {
                    paramHex[j * 2] = digits[p[j] >> 4];
                    paramHex[j * 2 + 1] = digits[p[j] & 15];
                }
            }
            Log("{\"ev\":\"nvapi_launch_cu_kernel\",\"ts\":%.3f,\"tid\":%lu,\"call\":%ld,\"frame\":%ld,\"slot\":%ld,\"index\":%u,\"function\":\"0x%llx\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"dynamic_shared\":%u,\"params\":\"0x%llx\",\"param_size\":%u,\"param_captured\":%u,\"param_hex\":\"%s\",\"param_truncated\":%s}",
                NowMs(), GetCurrentThreadId(), call, frame, slot, i,
                (unsigned long long)(uintptr_t)k.hFunction,
                k.gridDim.x, k.gridDim.y, k.gridDim.z,
                k.blockDim.x, k.blockDim.y, k.blockDim.z,
                k.dynSharedMemBytes, (unsigned long long)(uintptr_t)k.pParams,
                k.paramSize, captured, paramHex,
                captured < k.paramSize ? "true" : "false");
        }
    }
    NvApiLaunchCuKernelChain_t real = g_realLaunchCuKernelChain;
    int32_t status = -136;
    bool containsHeadContract = false;
    if (auto_head::Enabled() && kernels)
        for (uint32_t i = 0; i < numKernels; ++i) containsHeadContract |= kernels[i].paramSize == 184;
    if (containsHeadContract) {
        if (numKernels == 1 && auto_head::Prepare(static_cast<ID3D12GraphicsCommandList*>(commandList), kernels[0].pParams, kernels[0].paramSize, reinterpret_cast<void*>(real))) {
            auto_head::InternalScope suppress;
            status = real ? real(commandList, kernels, numKernels) : -136;
        } else status = -5; // NVAPI_INVALID_ARGUMENT: no speculative HIP fallback.
    } else status = real ? real(commandList, kernels, numKernels) : -136;
    EndFullGraphCapture((ID3D12GraphicsCommandList*)commandList, fullGraphSpan);
    if (neuralCapture) EndNeuralCapture((ID3D12GraphicsCommandList*)commandList);
    Log("{\"ev\":\"nvapi_launch_cu_kernel_chain_ret\",\"ts\":%.3f,\"tid\":%lu,\"call\":%ld,\"status\":%d}",
        NowMs(), GetCurrentThreadId(), call, status);
    return status;
}

static int32_t __cdecl HookDestroyCuModule(void* device, void* module) {
    NvApiDestroyCuObject_t real = g_realDestroyCuModule;
    int32_t status = real ? real(device, module) : -136;
    Log("{\"ev\":\"nvapi_destroy_cu_module\",\"ts\":%.3f,\"tid\":%lu,\"device\":\"0x%llx\",\"module\":\"0x%llx\",\"status\":%d}",
        NowMs(), GetCurrentThreadId(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)module, status);
    return status;
}

static int32_t __cdecl HookDestroyCuFunction(void* device, void* function) {
    NvApiDestroyCuObject_t real = g_realDestroyCuFunction;
    int32_t status = real ? real(device, function) : -136;
    Log("{\"ev\":\"nvapi_destroy_cu_function\",\"ts\":%.3f,\"tid\":%lu,\"device\":\"0x%llx\",\"function\":\"0x%llx\",\"status\":%d}",
        NowMs(), GetCurrentThreadId(), (unsigned long long)(uintptr_t)device,
        (unsigned long long)(uintptr_t)function, status);
    return status;
}

typedef int (__cdecl* NvapiAmdRegisterDescriptorResource_t)(uint64_t, void*);

static bool RegisterDescriptorResourceWithBackend(void* backendFunction,
                                                  uintptr_t descriptor,
                                                  ID3D12Resource* resource) {
    if (!backendFunction || !descriptor || !resource) return false;
    HMODULE backend = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(backendFunction),
                            &backend) || !backend) return false;
    auto registration = reinterpret_cast<NvapiAmdRegisterDescriptorResource_t>(
        GetProcAddress(backend, "NvapiAmd_RegisterDescriptorResource"));
    return registration && registration(descriptor, resource) != 0;
}

static int32_t __cdecl HookMergedTextureSampler(
        NvApiMergedTextureSamplerParams* params) {
    ID3D12Resource* resource = params ? FindDescriptorResource(params->texDesc) : nullptr;
    bool registered = params && RegisterDescriptorResourceWithBackend(
        reinterpret_cast<void*>(g_realMergedTextureSampler), params->texDesc,
        resource);
    Log("{\"ev\":\"nvapi_get_cuda_merged_texture_sampler_call\",\"ts\":%.3f,"
        "\"texture_descriptor\":\"0x%llx\",\"descriptor_resource\":\"0x%llx\","
        "\"backend_registered\":%s}",
        NowMs(), (unsigned long long)(params ? params->texDesc : 0),
        (unsigned long long)(uintptr_t)resource, registered ? "true" : "false");
    NvApiMergedTextureSampler_t real = g_realMergedTextureSampler;
    int32_t status = real ? real(params) : -136;
    if (status == 0 && params) RememberObjectResource(params->textureHandle, resource);
    Log("{\"ev\":\"nvapi_get_cuda_merged_texture_sampler\",\"ts\":%.3f,\"tid\":%lu,\"params\":\"0x%llx\",\"struct_size_in\":%llu,\"struct_size_out\":%llu,\"device\":\"0x%llx\",\"texture_descriptor\":\"0x%llx\",\"sampler_descriptor\":\"0x%llx\",\"status\":%d,\"texture_handle\":\"0x%016llX\"}",
        NowMs(), GetCurrentThreadId(),
        (unsigned long long)(uintptr_t)params,
        (unsigned long long)(params ? params->structSizeIn : 0),
        (unsigned long long)(params ? params->structSizeOut : 0),
        (unsigned long long)(uintptr_t)(params ? params->pDevice : nullptr),
        (unsigned long long)(params ? params->texDesc : 0),
        (unsigned long long)(params ? params->smpDesc : 0), status,
        (unsigned long long)(params ? params->textureHandle : 0));
    return status;
}

static int32_t __cdecl HookIndependentDescriptor(
        NvApiIndependentDescriptorParams* params) {
    ID3D12Resource* resource = params ? FindDescriptorResource(params->desc) : nullptr;
    bool registered = params && RegisterDescriptorResourceWithBackend(
        reinterpret_cast<void*>(g_realIndependentDescriptor), params->desc,
        resource);
    Log("{\"ev\":\"nvapi_get_cuda_independent_descriptor_call\",\"ts\":%.3f,"
        "\"type\":%u,\"descriptor\":\"0x%llx\",\"descriptor_resource\":\"0x%llx\","
        "\"backend_registered\":%s}",
        NowMs(), params ? params->type : UINT32_MAX,
        (unsigned long long)(params ? params->desc : 0),
        (unsigned long long)(uintptr_t)resource, registered ? "true" : "false");
    NvApiIndependentDescriptor_t real = g_realIndependentDescriptor;
    int32_t status = real ? real(params) : -136;
    if (status == 0 && params) RememberObjectResource(params->handle, resource);
    Log("{\"ev\":\"nvapi_get_cuda_independent_descriptor\",\"ts\":%.3f,\"tid\":%lu,\"params\":\"0x%llx\",\"struct_size_in\":%llu,\"struct_size_out\":%llu,\"device\":\"0x%llx\",\"type\":%u,\"descriptor\":\"0x%llx\",\"status\":%d,\"handle\":\"0x%016llX\"}",
        NowMs(), GetCurrentThreadId(),
        (unsigned long long)(uintptr_t)params,
        (unsigned long long)(params ? params->structSizeIn : 0),
        (unsigned long long)(params ? params->structSizeOut : 0),
        (unsigned long long)(uintptr_t)(params ? params->pDevice : nullptr),
        params ? params->type : UINT32_MAX,
        (unsigned long long)(params ? params->desc : 0), status,
        (unsigned long long)(params ? params->handle : 0));
    return status;
}

static void* __cdecl HookNvApiQueryInterface(uint32_t id) {
    NvApiQueryInterface_t real = g_realNvApiQueryInterface;
    void* result = real ? real(id) : nullptr;
    const char* candidate = NvApiCandidateName(id);
    Log("{\"ev\":\"nvapi_query_interface\",\"ts\":%.3f,\"tid\":%lu,\"id\":\"0x%08X\",\"candidate\":\"%s\",\"result\":\"0x%llx\"}",
        NowMs(), GetCurrentThreadId(), id, candidate ? candidate : "unknown",
        (unsigned long long)(uintptr_t)result);
    if (!result) return nullptr;
    void* wrapper = nullptr;
    switch (id) {
    case 0xAD1A677D:
        InterlockedExchangePointer((PVOID volatile*)&g_realCreateCuModule, result);
        wrapper = (void*)&HookCreateCuModule;
        break;
    case 0xE2436E22:
        InterlockedExchangePointer((PVOID volatile*)&g_realCreateCuFunction, result);
        wrapper = (void*)&HookCreateCuFunction;
        break;
    case 0x24973538:
        InterlockedExchangePointer((PVOID volatile*)&g_realLaunchCuKernelChain, result);
        wrapper = (void*)&HookLaunchCuKernelChain;
        break;
    case 0x41C65285:
        InterlockedExchangePointer((PVOID volatile*)&g_realDestroyCuModule, result);
        wrapper = (void*)&HookDestroyCuModule;
        break;
    case 0xDF295EA6:
        InterlockedExchangePointer((PVOID volatile*)&g_realDestroyCuFunction, result);
        wrapper = (void*)&HookDestroyCuFunction;
        break;
    case 0x329FE6E0:
        InterlockedExchangePointer((PVOID volatile*)&g_realMergedTextureSampler,
                                   result);
        wrapper = (void*)&HookMergedTextureSampler;
        break;
    case 0x0DDAC234:
        InterlockedExchangePointer((PVOID volatile*)&g_realIndependentDescriptor,
                                   result);
        wrapper = (void*)&HookIndependentDescriptor;
        break;
    default:
        break;
    }
    if (wrapper) {
        Log("{\"ev\":\"nvapi_call_wrapper_armed\",\"ts\":%.3f,\"tid\":%lu,\"id\":\"0x%08X\",\"name\":\"%s\",\"real\":\"0x%llx\",\"wrapper\":\"0x%llx\"}",
            NowMs(), GetCurrentThreadId(), id, candidate ? candidate : "unknown",
            (unsigned long long)(uintptr_t)result,
            (unsigned long long)(uintptr_t)wrapper);
        return wrapper;
    }
    return result;
}

// ---------------------------------------------------------------------------
// hooks
// ---------------------------------------------------------------------------

static HMODULE WINAPI HookLoadLibraryExW(LPCWSTR p, HANDLE f, DWORD flags) {
    HMODULE r = g_realLoadLibraryExW(p, f, flags);
    LogWide("load", p, r);
    if (r && g_hooksInstalled) PatchModuleIAT(r, g_self);   // patch immediately
    return r;
}
static HMODULE WINAPI HookLoadLibraryExA(LPCSTR p, HANDLE f, DWORD flags) {
    HMODULE r = g_realLoadLibraryExA(p, f, flags);
    LogNarrow("load", p, r);
    if (r && g_hooksInstalled) PatchModuleIAT(r, g_self);
    return r;
}
static HMODULE WINAPI HookLoadLibraryW(LPCWSTR p) {
    HMODULE r = g_realLoadLibraryW(p);
    LogWide("load", p, r);
    if (r && g_hooksInstalled) PatchModuleIAT(r, g_self);
    return r;
}
static HMODULE WINAPI HookLoadLibraryA(LPCSTR p) {
    HMODULE r = g_realLoadLibraryA(p);
    LogNarrow("load", p, r);
    if (r && g_hooksInstalled) PatchModuleIAT(r, g_self);
    return r;
}
static BOOL WINAPI HookFreeLibrary(HMODULE m) {
    char name[512] = "?";
    char escaped[1024] = {0};
    GetModuleFileNameA(m, name, sizeof(name));
    JsonEscape(name, escaped, sizeof(escaped));
    BOOL r = g_realFreeLibrary(m);
    Log("{\"ev\":\"free\",\"ts\":%.3f,\"tid\":%lu,\"path\":\"%s\",\"ok\":%s}",
        NowMs(), GetCurrentThreadId(), escaped, r ? "true" : "false");
    return r;
}
static FARPROC WINAPI HookGetProcAddress(HMODULE m, LPCSTR name) {
    FARPROC r = g_realGetProcAddress(m, name);
    if (name && !HIWORD((uintptr_t)name)) {
        Log("{\"ev\":\"getprocaddr\",\"ts\":%.3f,\"tid\":%lu,\"mod\":\"0x%llx\",\"ordinal\":%u,\"result\":\"0x%llx\"}",
            NowMs(), GetCurrentThreadId(), (unsigned long long)(uintptr_t)m,
            LOWORD((uintptr_t)name), (unsigned long long)(uintptr_t)r);
    } else {
        char mod[256] = "?";
        char escapedMod[512] = {0};
        char escapedName[512] = {0};
        GetModuleFileNameA(m, mod, sizeof(mod));
        JsonEscape(mod, escapedMod, sizeof(escapedMod));
        JsonEscape(name, escapedName, sizeof(escapedName));
        Log("{\"ev\":\"getprocaddr\",\"ts\":%.3f,\"tid\":%lu,\"mod\":\"%s\",\"name\":\"%s\",\"result\":\"0x%llx\"}",
            NowMs(), GetCurrentThreadId(), escapedMod, escapedName,
            (unsigned long long)(uintptr_t)r);

        // Restrict wrapping to the system-facing nvapi64.dll module.  This
        // excludes nvapi64_impl.dll and avoids accidentally changing which real
        // implementation an earlier caller reaches.
        const char* base = strrchr(mod, '\\');
        base = base ? base + 1 : mod;
        if (r && name && !_stricmp(base, "nvapi64.dll") &&
            !strcmp(name, "nvapi_QueryInterface")) {
            InterlockedExchangePointer((PVOID volatile*)&g_realNvApiQueryInterface,
                                       (PVOID)r);
            Log("{\"ev\":\"nvapi_query_wrapper_armed\",\"ts\":%.3f,\"tid\":%lu,\"module\":\"%s\",\"real\":\"0x%llx\"}",
                NowMs(), GetCurrentThreadId(), escapedMod,
                (unsigned long long)(uintptr_t)r);
            return (FARPROC)&HookNvApiQueryInterface;
        }
    }
    return r;
}

// ---------------------------------------------------------------------------
// IAT patching
// ---------------------------------------------------------------------------

// Patch one module's IAT entries that point at the loader APIs.
static void PatchModuleIAT(HMODULE mod, HMODULE self) {
    if (!mod || mod == self) return;
    // guard against non-image memory (SEH is unreliable in this toolchain)
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery((LPCVOID)mod, &mbi, sizeof(mbi)) < sizeof(mbi) ||
        mbi.State != MEM_COMMIT) return;
    auto dos = (IMAGE_DOS_HEADER*)mod;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    auto nt = (IMAGE_NT_HEADERS*)((BYTE*)mod + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;
    auto dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir->VirtualAddress) return;
    auto imp = (IMAGE_IMPORT_DESCRIPTOR*)((BYTE*)mod + dir->VirtualAddress);
    for (; imp->Name; ++imp) {
        const char* dllName = (const char*)((BYTE*)mod + imp->Name);
        bool isKernel = _stricmp(dllName, "kernel32.dll") == 0 ||
                        _stricmp(dllName, "KERNELBASE.dll") == 0;
        if (!isKernel) continue;
        auto thunk = (IMAGE_THUNK_DATA*)((BYTE*)mod +
                       (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        auto iat = (IMAGE_THUNK_DATA*)((BYTE*)mod + imp->FirstThunk);
        for (; thunk->u1.AddressOfData; ++thunk, ++iat) {
            if (thunk->u1.Ordinal & IMAGE_ORDINAL_FLAG) continue;
            auto byName = (IMAGE_IMPORT_BY_NAME*)((BYTE*)mod + thunk->u1.AddressOfData);
            void* target = nullptr;
            if (!strcmp(byName->Name, "LoadLibraryExW")) target = (void*)HookLoadLibraryExW;
            else if (!strcmp(byName->Name, "LoadLibraryExA")) target = (void*)HookLoadLibraryExA;
            else if (!strcmp(byName->Name, "LoadLibraryW")) target = (void*)HookLoadLibraryW;
            else if (!strcmp(byName->Name, "LoadLibraryA")) target = (void*)HookLoadLibraryA;
            else if (!strcmp(byName->Name, "FreeLibrary")) target = (void*)HookFreeLibrary;
            else if (!strcmp(byName->Name, "GetProcAddress")) target = (void*)HookGetProcAddress;
            if (!target) continue;
            DWORD old = 0;
            if (VirtualProtect(&iat->u1.Function, sizeof(void*), PAGE_READWRITE, &old)) {
                iat->u1.Function = (ULONGLONG)(uintptr_t)target;
                VirtualProtect(&iat->u1.Function, sizeof(void*), old, &old);
            }
        }
    }
}

static void PatchAllModules(HMODULE self) {
    HMODULE mods[1024];
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed)) return;
    int n = needed / sizeof(HMODULE);
    for (int i = 0; i < n; ++i) PatchModuleIAT(mods[i], self);
}

// ---------------------------------------------------------------------------
// best-effort module-load notification (private ntdll API, defensive usage)
// ---------------------------------------------------------------------------

// PRIVATE WIN32 ABI — inferred: ntdll!LdrRegisterModuleNotificationCallback.
// The callback context is a node whose layout is (LIST_ENTRY, reason, data*).
struct LdrModuleNotificationNode {
    LIST_ENTRY Entry;
    ULONG Reason;                 // 1 = loaded, 2 = unloaded
    struct LdrModuleNotificationData* Data;
};
struct LdrModuleNotificationData {
    ULONG Flags;
    UNICODE_STRING* FullDllName;
    UNICODE_STRING* BaseDllName;
    PVOID DllBase;
    ULONG SizeOfImage;
};
typedef void (CALLBACK* LdrModuleNotificationCallback)(PVOID Context);
typedef NTSTATUS (NTAPI* LdrRegisterModuleNotificationCallback_t)(LdrModuleNotificationCallback, PVOID*);

// Open %MODULE_TRACE_LOG% (Win32 env lookup; CRT getenv can miss variables
// the host set right before LoadLibrary). Default: .\module_trace.log.
static void OpenLog() {
    char logPath[1024] = "module_trace.log";
    DWORD n = GetEnvironmentVariableA("MODULE_TRACE_LOG", logPath, sizeof(logPath));
    if (n == 0 || n >= sizeof(logPath)) strcpy(logPath, "module_trace.log");
    EnterCriticalSection(&g_cs);
    FILE* fresh = _fsopen(logPath, "a", _SH_DENYNO);   // shared: readers may tail
    if (fresh) {
        if (g_log && g_log != fresh) fclose(g_log);
        g_log = fresh;
    }
    LeaveCriticalSection(&g_cs);
}

static void CALLBACK OnModuleNotification(PVOID ctx) {
    // defensive: validate the node before touching it (loader lock context)
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(ctx, &mbi, sizeof(mbi)) < sizeof(mbi) || mbi.State != MEM_COMMIT) return;
    auto node = (LdrModuleNotificationNode*)ctx;
    if (node->Reason != 1 || !node->Data) return;
    auto data = node->Data;
    if (VirtualQuery(data, &mbi, sizeof(mbi)) < sizeof(mbi) || mbi.State != MEM_COMMIT) return;
    if (data->DllBase && data->DllBase != g_self)   // never patch ourselves (reentrancy)
        PatchModuleIAT((HMODULE)data->DllBase, g_self);
    char name[256] = "?";
    char escaped[512] = {0};
    if (data->BaseDllName && data->BaseDllName->Buffer && data->BaseDllName->Length < sizeof(name) * 2)
        WideCharToMultiByte(CP_UTF8, 0, data->BaseDllName->Buffer, data->BaseDllName->Length / 2,
                            name, sizeof(name) - 1, nullptr, nullptr);
    JsonEscape(name, escaped, sizeof(escaped));
    Log("{\"ev\":\"ldr_notify_load\",\"ts\":%.3f,\"path\":\"%s\",\"base\":\"0x%llx\"}",
        NowMs(), escaped, (unsigned long long)(uintptr_t)data->DllBase);
}

static void RegisterModuleNotifications() {
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll) return;
    auto reg = (LdrRegisterModuleNotificationCallback_t)
        g_realGetProcAddress(ntdll, "LdrRegisterModuleNotificationCallback");
    if (!reg) {
        Log("{\"ev\":\"ldr_notify\",\"ts\":%.3f,\"status\":\"API_absent\"}", NowMs());
        return;
    }
    PVOID cookie = nullptr;
    NTSTATUS st = reg(OnModuleNotification, &cookie);
    Log("{\"ev\":\"ldr_notify\",\"ts\":%.3f,\"status\":\"%s\",\"ntstatus\":\"0x%lx\"}",
        NowMs(), st >= 0 ? "registered" : "failed", (unsigned long)st);
}

// ---------------------------------------------------------------------------
// initialization (one-shot; sync via ModuleTrace_InitializeAndWait)
// ---------------------------------------------------------------------------

static void DoInit() {
    OpenLog();   // honor MODULE_TRACE_LOG even on the auto-init thread path
    // capture already-loaded modules as baseline
    HMODULE mods[1024];
    DWORD needed = 0;
    if (EnumProcessModules(GetCurrentProcess(), mods, sizeof(mods), &needed)) {
        int n = needed / sizeof(HMODULE);
        for (int i = 0; i < n; ++i) {
            char name[1024] = "?";
            char escaped[2048] = {0};
            GetModuleFileNameA(mods[i], name, sizeof(name));
            JsonEscape(name, escaped, sizeof(escaped));
            Log("{\"ev\":\"baseline\",\"ts\":%.3f,\"path\":\"%s\"}", NowMs(), escaped);
        }
    }
    PatchAllModules(g_self);   // skips g_self: patching our own IAT would make
                               // our internal GetModuleFileNameA calls recurse
                               // into HookGetProcAddress (loader-lock deadlock)
    RegisterModuleNotifications();
    InterlockedExchange(&g_hooksInstalled, 1);
    Log("{\"ev\":\"hooks_installed\",\"ts\":%.3f,\"pid\":%lu}", NowMs(), GetCurrentProcessId());
    if (g_readyEvt) SetEvent(g_readyEvt);
}

static DWORD WINAPI InitThread(LPVOID) {
    if (InterlockedCompareExchange(&g_initStarted, 1, 0) == 0) DoInit();
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls((HMODULE)inst);
        InitializeCriticalSection(&g_cs);
        InitializeCriticalSection(&g_resourceCs);
        QueryPerformanceFrequency(&g_freq);
        g_self = (HMODULE)inst;
        g_readyEvt = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        OpenLog();
        Log("{\"ev\":\"attach\",\"ts\":%.3f,\"pid\":%lu}", NowMs(), GetCurrentProcessId());
        // heavy work stays out of DllMain; hosts that need determinism call
        // ModuleTrace_InitializeAndWait() right after LoadLibrary.
        HANDLE t = CreateThread(nullptr, 0, InitThread, nullptr, 0, nullptr);
        if (t) CloseHandle(t);
    }
    return TRUE;
}

// ---------------------------------------------------------------------------
// exports
// ---------------------------------------------------------------------------

static HRESULT CreateSnapshotReadback(ID3D12Device* device,
                                      ID3D12Resource* source,
                                      SnapshotReadback* result) {
    if (!device || !source || !result) return E_INVALIDARG;
    memset(result, 0, sizeof(*result));
    D3D12_RESOURCE_DESC sourceDesc = source->GetDesc();
    device->GetCopyableFootprints(&sourceDesc, 0, 1, 0, &result->footprint,
                                  &result->rows, &result->rowSize,
                                  &result->totalBytes);
    if (!result->totalBytes || !result->rows || !result->rowSize) return E_FAIL;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bufferDesc{};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = result->totalBytes;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&result->buffer));
}

static bool BeginPostTexturePreLaunchCapture(
        ID3D12GraphicsCommandList* list, ID3D12Resource* source,
        uint64_t textureObject, LONG frame) {
    if (!list || !source || !textureObject) return false;
    if (InterlockedCompareExchange(&g_postTexturePreLaunchState, 1, 0) != 0)
        return false;
    ID3D12Device* device = nullptr;
    SnapshotReadback readback{};
    D3D12_RESOURCE_DESC desc = source->GetDesc();
    HRESULT hr = list->GetDevice(IID_PPV_ARGS(&device));
    if (SUCCEEDED(hr) && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) hr = CreateSnapshotReadback(device, source, &readback);
    if (SUCCEEDED(hr)) {
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readback.buffer;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = readback.footprint;
        D3D12_TEXTURE_COPY_LOCATION origin{};
        origin.pResource = source;
        origin.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        origin.SubresourceIndex = 0;
        // This is inserted on the exact command list immediately before the
        // slot-154 NVAPI launch. COMMON resources use implicit COPY_SOURCE
        // promotion, preserving ordering without a guessed prior state.
        list->CopyTextureRegion(&destination, 0, 0, 0, &origin, nullptr);
        EnterCriticalSection(&g_resourceCs);
        g_postTexturePreLaunchReadback = readback;
        g_postTexturePreLaunchDesc = desc;
        g_postTexturePreLaunchFrame = frame;
        LeaveCriticalSection(&g_resourceCs);
        InterlockedExchange(&g_postTexturePreLaunchState, 2);
    } else {
        if (readback.buffer) readback.buffer->Release();
        InterlockedExchange(&g_postTexturePreLaunchState, -1);
    }
    if (device) device->Release();
    Log("{\"ev\":\"post_texture_prelaunch_capture_arm\",\"ts\":%.3f,"
        "\"status\":\"%s\",\"frame\":%ld,\"texture_object\":\"0x%016llX\","
        "\"texture_resource\":\"0x%llx\",\"hresult\":\"0x%08X\"}",
        NowMs(), SUCCEEDED(hr) ? "PASS" : "FAIL", frame,
        (unsigned long long)textureObject,
        (unsigned long long)(uintptr_t)source, (unsigned)hr);
    return SUCCEEDED(hr);
}

static bool BeginFrame1PostOutputCapture(
        ID3D12GraphicsCommandList* list, ID3D12Resource* source,
        uint64_t surfaceObject, LONG frame) {
    if (!list || !source || !surfaceObject || frame != 1) return false;
    if (InterlockedCompareExchange(&g_frame1PostOutputState, 1, 0) != 0)
        return false;
    ID3D12Device* device = nullptr;
    SnapshotReadback readback{};
    D3D12_RESOURCE_DESC desc = source->GetDesc();
    HRESULT hr = list->GetDevice(IID_PPV_ARGS(&device));
    if (SUCCEEDED(hr) && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) hr = CreateSnapshotReadback(device, source, &readback);
    if (SUCCEEDED(hr)) {
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readback.buffer;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = readback.footprint;
        D3D12_TEXTURE_COPY_LOCATION origin{};
        origin.pResource = source;
        origin.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        origin.SubresourceIndex = 0;
        // Slot 155 consumes slot 154's surface. Capturing on this exact command
        // list immediately before slot 155 provides the missing frame-aligned
        // oracle instead of the unrelated texture contents after frame 300.
        list->CopyTextureRegion(&destination, 0, 0, 0, &origin, nullptr);
        EnterCriticalSection(&g_resourceCs);
        g_frame1PostOutputReadback = readback;
        g_frame1PostOutputDesc = desc;
        g_frame1PostOutputObject = surfaceObject;
        g_frame1PostOutputResourceAddress = (uintptr_t)source;
        g_frame1PostOutputFrame = frame;
        LeaveCriticalSection(&g_resourceCs);
        InterlockedExchange(&g_frame1PostOutputState, 2);
    } else {
        if (readback.buffer) readback.buffer->Release();
        InterlockedExchange(&g_frame1PostOutputState, -1);
    }
    if (device) device->Release();
    Log("{\"ev\":\"frame1_post_output_capture_arm\",\"ts\":%.3f,"
        "\"status\":\"%s\",\"frame\":%ld,\"surface_object\":\"0x%016llX\","
        "\"surface_resource\":\"0x%llx\",\"hresult\":\"0x%08X\"}",
        NowMs(), SUCCEEDED(hr) ? "PASS" : "FAIL", frame,
        (unsigned long long)surfaceObject,
        (unsigned long long)(uintptr_t)source, (unsigned)hr);
    return SUCCEEDED(hr);
}

static bool BeginFrame1PostSurfaceInitialCapture(
        ID3D12GraphicsCommandList* list, ID3D12Resource* source,
        uint64_t surfaceObject, LONG frame) {
    if (!list || !source || !surfaceObject || frame != 1) return false;
    if (InterlockedCompareExchange(&g_frame1PostSurfaceInitialState, 1, 0) != 0)
        return false;
    ID3D12Device* device = nullptr;
    SnapshotReadback readback{};
    D3D12_RESOURCE_DESC desc = source->GetDesc();
    HRESULT hr = list->GetDevice(IID_PPV_ARGS(&device));
    if (SUCCEEDED(hr) && desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
        hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) hr = CreateSnapshotReadback(device, source, &readback);
    if (SUCCEEDED(hr)) {
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readback.buffer;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = readback.footprint;
        D3D12_TEXTURE_COPY_LOCATION origin{};
        origin.pResource = source;
        origin.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        origin.SubresourceIndex = 0;
        // Preserve the destination's prior contents as well as the post-kernel
        // image: slot 154 may conditionally update pixels, so a zero-initialized
        // standalone CUDA surface is not yet a justified contract assumption.
        list->CopyTextureRegion(&destination, 0, 0, 0, &origin, nullptr);
        EnterCriticalSection(&g_resourceCs);
        g_frame1PostSurfaceInitialReadback = readback;
        g_frame1PostSurfaceInitialDesc = desc;
        g_frame1PostSurfaceObject = surfaceObject;
        g_frame1PostSurfaceResourceAddress = (uintptr_t)source;
        g_frame1PostSurfaceInitialFrame = frame;
        LeaveCriticalSection(&g_resourceCs);
        InterlockedExchange(&g_frame1PostSurfaceInitialState, 2);
    } else {
        if (readback.buffer) readback.buffer->Release();
        InterlockedExchange(&g_frame1PostSurfaceInitialState, -1);
    }
    if (device) device->Release();
    Log("{\"ev\":\"frame1_post_surface_initial_capture_arm\",\"ts\":%.3f,"
        "\"status\":\"%s\",\"frame\":%ld,\"surface_object\":\"0x%016llX\","
        "\"surface_resource\":\"0x%llx\",\"hresult\":\"0x%08X\"}",
        NowMs(), SUCCEEDED(hr) ? "PASS" : "FAIL", frame,
        (unsigned long long)surfaceObject,
        (unsigned long long)(uintptr_t)source, (unsigned)hr);
    return SUCCEEDED(hr);
}

static bool BeginPostActivationPreLaunchCapture(
        ID3D12GraphicsCommandList* list, const void* parameterData,
        uint32_t parameterSize, LONG frame) {
    if (!list || !parameterData || parameterSize < 16 || frame != 1) return false;
    if (InterlockedCompareExchange(&g_postActivationPreLaunchState, 1, 0) != 0)
        return false;
    uint64_t pointer0 = 0, pointer8 = 0;
    memcpy(&pointer0, parameterData, sizeof(pointer0));
    memcpy(&pointer8, (const unsigned char*)parameterData + 8, sizeof(pointer8));
    ResolvedBufferAddress resolved0 = ResolveBufferAddress(pointer0);
    ResolvedBufferAddress resolved8 = ResolveBufferAddress(pointer8);
    ID3D12Device* device = nullptr;
    ID3D12Resource* readback = nullptr;
    HRESULT hr = list->GetDevice(IID_PPV_ARGS(&device));
    D3D12_RESOURCE_DESC desc{};
    if (SUCCEEDED(hr) && (!resolved0.resource ||
                         resolved0.resource != resolved8.resource))
        hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) {
        desc = resolved0.resource->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
            !desc.Width || desc.Width > (64ull << 20))
            hr = E_INVALIDARG;
    }
    if (SUCCEEDED(hr)) hr = CreateLinearReadback(device, desc.Width, &readback);
    if (SUCCEEDED(hr)) {
        list->CopyBufferRegion(readback, 0, resolved0.resource, 0, desc.Width);
        EnterCriticalSection(&g_resourceCs);
        g_postActivationPreLaunchReadback = readback;
        g_postActivationPreLaunchBytes = desc.Width;
        g_postActivationParam0 = pointer0;
        g_postActivationParam8 = pointer8;
        g_postActivationResourceAddress = (uintptr_t)resolved0.resource;
        LeaveCriticalSection(&g_resourceCs);
        InterlockedExchange(&g_postActivationPreLaunchState, 2);
    } else {
        if (readback) readback->Release();
        InterlockedExchange(&g_postActivationPreLaunchState, -1);
    }
    if (device) device->Release();
    Log("{\"ev\":\"post_activation_prelaunch_capture_arm\",\"ts\":%.3f,"
        "\"status\":\"%s\",\"frame\":%ld,\"resource\":\"0x%llx\","
        "\"bytes\":%llu,\"param0\":\"0x%016llX\","
        "\"param8\":\"0x%016llX\",\"offset0\":%llu,\"offset8\":%llu,"
        "\"hresult\":\"0x%08X\"}", NowMs(),
        SUCCEEDED(hr) ? "PASS" : "FAIL", frame,
        (unsigned long long)(uintptr_t)resolved0.resource,
        (unsigned long long)desc.Width,
        (unsigned long long)pointer0, (unsigned long long)pointer8,
        (unsigned long long)resolved0.offset,
        (unsigned long long)resolved8.offset, (unsigned)hr);
    return SUCCEEDED(hr);
}

static bool WriteCompactSnapshot(const char* path, SnapshotReadback* readback,
                                 uint64_t* hashOut, UINT64* bytesOut) {
    if (!path || !readback || !readback->buffer) return false;
    void* mapped = nullptr;
    D3D12_RANGE range{0, (SIZE_T)readback->totalBytes};
    if (FAILED(readback->buffer->Map(0, &range, &mapped)) || !mapped) return false;
    FILE* file = nullptr;
    fopen_s(&file, path, "wb");
    uint64_t hash = 1469598103934665603ull;
    UINT64 written = 0;
    bool ok = file != nullptr;
    const unsigned char* base = (const unsigned char*)mapped + readback->footprint.Offset;
    for (UINT row = 0; ok && row < readback->rows; ++row) {
        const unsigned char* data = base + UINT64(row) * readback->footprint.Footprint.RowPitch;
        if (fwrite(data, 1, (size_t)readback->rowSize, file) != readback->rowSize) {
            ok = false;
            break;
        }
        for (UINT64 i = 0; i < readback->rowSize; ++i) {
            hash ^= data[i];
            hash *= 1099511628211ull;
        }
        written += readback->rowSize;
    }
    if (file && fclose(file) != 0) ok = false;
    D3D12_RANGE noWrite{0, 0};
    readback->buffer->Unmap(0, &noWrite);
    if (hashOut) *hashOut = hash;
    if (bytesOut) *bytesOut = written;
    return ok;
}

static bool WriteLinearSnapshot(const char* path, ID3D12Resource* readback,
                                UINT64 size, uint64_t* hashOut) {
    if (!path || !readback || !size) return false;
    void* mapped = nullptr;
    D3D12_RANGE range{0, (SIZE_T)size};
    if (FAILED(readback->Map(0, &range, &mapped)) || !mapped) return false;
    FILE* file = nullptr;
    fopen_s(&file, path, "wb");
    bool ok = file && fwrite(mapped, 1, (size_t)size, file) == size;
    if (file && fclose(file) != 0) ok = false;
    uint64_t hash = 1469598103934665603ull;
    const unsigned char* bytes = (const unsigned char*)mapped;
    for (UINT64 i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    D3D12_RANGE noWrite{0, 0};
    readback->Unmap(0, &noWrite);
    if (hashOut) *hashOut = hash;
    return ok;
}

static bool CompareLinearSnapshots(ID3D12Resource* before,
                                   ID3D12Resource* after, UINT64 size,
                                   UINT64* changed, UINT64* first,
                                   UINT64* last) {
    if (!before || !after || !size) return false;
    void* beforeMapped = nullptr;
    void* afterMapped = nullptr;
    D3D12_RANGE range{0, (SIZE_T)size};
    if (FAILED(before->Map(0, &range, &beforeMapped)) || !beforeMapped) return false;
    if (FAILED(after->Map(0, &range, &afterMapped)) || !afterMapped) {
        D3D12_RANGE noWrite{0, 0};
        before->Unmap(0, &noWrite);
        return false;
    }
    UINT64 count = 0, firstValue = UINT64_MAX, lastValue = UINT64_MAX;
    const unsigned char* a = (const unsigned char*)beforeMapped;
    const unsigned char* b = (const unsigned char*)afterMapped;
    for (UINT64 i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            if (!count) firstValue = i;
            lastValue = i;
            ++count;
        }
    }
    D3D12_RANGE noWrite{0, 0};
    before->Unmap(0, &noWrite);
    after->Unmap(0, &noWrite);
    if (changed) *changed = count;
    if (first) *first = firstValue;
    if (last) *last = lastValue;
    return true;
}

static bool DumpFullGraphCaptureFiles(const char* directory) {
    if (!directory || !g_fullGraphWindowCount) return false;
    char fullDirectory[MAX_PATH * 4]{};
    char metadataPath[MAX_PATH * 4]{};
    snprintf(fullDirectory, sizeof(fullDirectory), "%s\\full_graph", directory);
    snprintf(metadataPath, sizeof(metadataPath), "%s\\full_graph_capture.json", directory);
    CreateDirectoryA(directory, nullptr);
    CreateDirectoryA(fullDirectory, nullptr);
    FILE* metadata = nullptr;
    fopen_s(&metadata, metadataPath, "wb");
    if (!metadata) return false;
    bool ok = InterlockedCompareExchange(&g_fullGraphCaptureFailures, 0, 0) == 0;
    // Buffer-window coverage is slots 0-154. The final descriptor-object copy
    // (slot 155) is reported independently in copy_snapshot.json.
    bool slots[155]{};
    size_t capturedSlots = 0;
    for (size_t i = 0; i < g_fullGraphWindowCount; ++i) {
        if (g_fullGraphWindows[i].slot < _countof(slots) &&
            !slots[g_fullGraphWindows[i].slot]) {
            slots[g_fullGraphWindows[i].slot] = true;
            ++capturedSlots;
        }
    }
    fprintf(metadata,
        "{\n  \"schema\": 1,\n"
        "  \"classification\": \"FULL_GRAPH_BOUNDED_BUFFER_PRE_POST_WINDOWS\",\n"
        "  \"frame\": 1,\n  \"capture_limit_bytes\": %llu,\n"
        "  \"window_count\": %llu,\n  \"captured_slot_count\": %llu,\n"
        "  \"capture_failures\": %ld,\n  \"windows\": [\n",
        (unsigned long long)FullGraphCaptureLimit(),
        (unsigned long long)g_fullGraphWindowCount,
        (unsigned long long)capturedSlots,
        InterlockedCompareExchange(&g_fullGraphCaptureFailures, 0, 0));
    for (size_t i = 0; i < g_fullGraphWindowCount; ++i) {
        const auto& window = g_fullGraphWindows[i];
        char beforeName[160]{}, afterName[160]{};
        char beforePath[MAX_PATH * 4]{}, afterPath[MAX_PATH * 4]{};
        snprintf(beforeName, sizeof(beforeName),
                 "slot_%03u_param_%03u_window_%04llu_before.raw",
                 window.slot, window.paramOffset, (unsigned long long)i);
        snprintf(afterName, sizeof(afterName),
                 "slot_%03u_param_%03u_window_%04llu_after.raw",
                 window.slot, window.paramOffset, (unsigned long long)i);
        snprintf(beforePath, sizeof(beforePath), "%s\\%s", fullDirectory, beforeName);
        snprintf(afterPath, sizeof(afterPath), "%s\\%s", fullDirectory, afterName);
        uint64_t beforeHash = 0, afterHash = 0;
        UINT64 changed = 0, first = UINT64_MAX, last = UINT64_MAX;
        bool itemOk =
            WriteLinearSnapshot(beforePath, window.before, window.size, &beforeHash) &&
            WriteLinearSnapshot(afterPath, window.after, window.size, &afterHash) &&
            CompareLinearSnapshots(window.before, window.after, window.size,
                                   &changed, &first, &last);
        ok &= itemOk;
        fprintf(metadata,
            "    {\"slot\":%u,\"param_offset\":%u,"
            "\"function\":\"0x%llX\",\"pointer\":\"0x%llX\","
            "\"resource\":\"0x%llX\",\"resource_offset\":%llu,"
            "\"capture_bytes\":%llu,\"before_file\":\"full_graph/%s\","
            "\"after_file\":\"full_graph/%s\","
            "\"before_fnv1a64\":\"0x%016llX\","
            "\"after_fnv1a64\":\"0x%016llX\","
            "\"changed_bytes\":%llu,\"first_changed_offset\":%lld,"
            "\"last_changed_offset\":%lld,\"write_ok\":%s}%s\n",
            window.slot, window.paramOffset,
            (unsigned long long)window.function,
            (unsigned long long)window.pointer,
            (unsigned long long)(uintptr_t)window.source,
            (unsigned long long)window.sourceOffset,
            (unsigned long long)window.size, beforeName, afterName,
            (unsigned long long)beforeHash, (unsigned long long)afterHash,
            (unsigned long long)changed,
            first == UINT64_MAX ? -1ll : (long long)first,
            last == UINT64_MAX ? -1ll : (long long)last,
            itemOk ? "true" : "false",
            i + 1 == g_fullGraphWindowCount ? "" : ",");
    }
    fprintf(metadata,
        "  ],\n  \"status\": \"%s\",\n  \"limitations\": ["
        "\"only 64-bit parameter values resolving into tracked D3D12 buffers are captured\","
        "\"texture and descriptor-object inputs remain covered by the separate descriptor/copy trace\","
        "\"windows are bounded by the next pointer in the same resource or the configured byte limit\"]\n}\n",
        ok ? "PASS" : "FAIL");
    fclose(metadata);
    Log("{\"ev\":\"full_graph_capture_dump\",\"ts\":%.3f,"
        "\"status\":\"%s\",\"windows\":%llu,\"captured_slots\":%llu,"
        "\"failures\":%ld}", NowMs(), ok ? "PASS" : "FAIL",
        (unsigned long long)g_fullGraphWindowCount,
        (unsigned long long)capturedSlots,
        InterlockedCompareExchange(&g_fullGraphCaptureFailures, 0, 0));
    return ok;
}

static bool DumpNeuralCaptureFiles(const char* directory) {
    if (!directory || InterlockedCompareExchange(&g_neuralCaptureState, 2, 2) != 2)
        return false;
    char metadataPath[MAX_PATH * 4]{};
    snprintf(metadataPath, sizeof(metadataPath), "%s\\n0_preblock_capture.json", directory);
    uint64_t beforeHashes[kNeuralWindowCount]{}, afterHashes[kNeuralWindowCount]{};
    UINT64 changed[kNeuralWindowCount]{}, first[kNeuralWindowCount]{};
    UINT64 last[kNeuralWindowCount]{};
    char beforePaths[kNeuralWindowCount][MAX_PATH * 4]{};
    char afterPaths[kNeuralWindowCount][MAX_PATH * 4]{};
    bool ok = true;
    for (size_t i = 0; i < kNeuralWindowCount; ++i) {
        first[i] = UINT64_MAX;
        last[i] = UINT64_MAX;
        snprintf(beforePaths[i], sizeof(beforePaths[i]), "%s\\n0_%s_before.raw",
                 directory, g_neuralWindows[i].name);
        snprintf(afterPaths[i], sizeof(afterPaths[i]), "%s\\n0_%s_after.raw",
                 directory, g_neuralWindows[i].name);
        ok &= WriteLinearSnapshot(beforePaths[i], g_neuralWindows[i].before,
                                  g_neuralWindows[i].size, &beforeHashes[i]);
        ok &= WriteLinearSnapshot(afterPaths[i], g_neuralWindows[i].after,
                                  g_neuralWindows[i].size, &afterHashes[i]);
        ok &= CompareLinearSnapshots(g_neuralWindows[i].before,
                                     g_neuralWindows[i].after,
                                     g_neuralWindows[i].size, &changed[i],
                                     &first[i], &last[i]);
    }
    FILE* metadata = nullptr;
    fopen_s(&metadata, metadataPath, "wb");
    if (!metadata) ok = false;
    if (metadata) {
        fprintf(metadata,
            "{\n  \"schema\": 1,\n  \"status\": \"%s\",\n"
            "  \"function\": \"cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8\",\n"
            "  \"frame\": 1,\n  \"slot\": 1,\n"
            "  \"capture_classification\": \"BOUNDED_PRE_POST_RAW_WINDOWS\",\n"
            "  \"padded_height\": %u,\n  \"padded_width\": %u,\n"
            "  \"windows\": [\n",
            ok ? "PASS" : "FAIL", g_neuralPaddedHeight, g_neuralPaddedWidth);
        for (size_t i = 0; i < kNeuralWindowCount; ++i) {
            const auto& window = g_neuralWindows[i];
            fprintf(metadata,
                "    {\"name\":\"%s\",\"param_offset\":%u,"
                "\"weight_view_offset\":%llu,"
                "\"pointer\":\"0x%llx\",\"resource\":\"0x%llx\","
                "\"resource_offset\":%llu,\"capture_bytes\":%llu,"
                "\"before_file\":\"n0_%s_before.raw\","
                "\"after_file\":\"n0_%s_after.raw\","
                "\"before_fnv1a64\":\"0x%016llX\","
                "\"after_fnv1a64\":\"0x%016llX\","
                "\"changed_bytes\":%llu,\"first_changed_offset\":%lld,"
                "\"last_changed_offset\":%lld}%s\n",
                window.name, window.paramOffset,
                (unsigned long long)window.weightViewOffset,
                (unsigned long long)window.pointer,
                (unsigned long long)(uintptr_t)window.source,
                (unsigned long long)window.sourceOffset,
                (unsigned long long)window.size, window.name, window.name,
                (unsigned long long)beforeHashes[i],
                (unsigned long long)afterHashes[i],
                (unsigned long long)changed[i],
                first[i] == UINT64_MAX ? -1ll : (long long)first[i],
                last[i] == UINT64_MAX ? -1ll : (long long)last[i],
                i + 1 == kNeuralWindowCount ? "" : ",");
        }
        fprintf(metadata,
            "  ],\n  \"limitations\": ["
            "\"capture sizes are PTX-derived envelopes, not final tensor shape claims\","
            "\"downstream weight captures extend up to 1 MiB at graph-observed resource offsets and are clamped to the resource boundary\"]\n}\n");
        fclose(metadata);
    }
    Log("{\"ev\":\"neural_capture_dump\",\"ts\":%.3f,\"status\":\"%s\","
        "\"scratch_changed\":%llu,\"weights_changed\":%llu,"
        "\"output_changed\":%llu}", NowMs(), ok ? "PASS" : "FAIL",
        (unsigned long long)changed[0], (unsigned long long)changed[1],
        (unsigned long long)changed[2]);
    return ok;
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_DumpCopyResources(
        ID3D12CommandQueue* queue, const char* directory) {
    ID3D12Resource* input = nullptr;
    ID3D12Resource* output = nullptr;
    ID3D12Resource* postTexture = nullptr;
    uint64_t postTextureObject = 0;
    SnapshotReadback postTextureReadback{};
    D3D12_RESOURCE_DESC postTextureDesc{};
    LONG postTextureCaptureState = 0;
    LONG postTextureCaptureFrame = 0;
    SnapshotReadback frame1PostOutputReadback{};
    D3D12_RESOURCE_DESC frame1PostOutputDesc{};
    uint64_t frame1PostOutputObject = 0;
    uintptr_t frame1PostOutputResourceAddress = 0;
    LONG frame1PostOutputState = 0;
    LONG frame1PostOutputFrame = 0;
    SnapshotReadback frame1PostSurfaceInitialReadback{};
    D3D12_RESOURCE_DESC frame1PostSurfaceInitialDesc{};
    uint64_t frame1PostSurfaceObject = 0;
    uintptr_t frame1PostSurfaceResourceAddress = 0;
    LONG frame1PostSurfaceInitialState = 0;
    LONG frame1PostSurfaceInitialFrame = 0;
    ID3D12Resource* postActivationReadback = nullptr;
    UINT64 postActivationBytes = 0;
    uint64_t postActivationParam0 = 0, postActivationParam8 = 0;
    uintptr_t postActivationResourceAddress = 0;
    LONG postActivationCaptureState = 0;
    EnterCriticalSection(&g_resourceCs);
    input = g_copyInputResource;
    output = g_copyOutputResource;
    postTexture = g_postTextureResource;
    postTextureObject = g_postTextureObject;
    postTextureReadback = g_postTexturePreLaunchReadback;
    postTextureDesc = g_postTexturePreLaunchDesc;
    postTextureCaptureState = g_postTexturePreLaunchState;
    postTextureCaptureFrame = g_postTexturePreLaunchFrame;
    frame1PostOutputReadback = g_frame1PostOutputReadback;
    frame1PostOutputDesc = g_frame1PostOutputDesc;
    frame1PostOutputObject = g_frame1PostOutputObject;
    frame1PostOutputResourceAddress = g_frame1PostOutputResourceAddress;
    frame1PostOutputState = g_frame1PostOutputState;
    frame1PostOutputFrame = g_frame1PostOutputFrame;
    frame1PostSurfaceInitialReadback = g_frame1PostSurfaceInitialReadback;
    frame1PostSurfaceInitialDesc = g_frame1PostSurfaceInitialDesc;
    frame1PostSurfaceObject = g_frame1PostSurfaceObject;
    frame1PostSurfaceResourceAddress = g_frame1PostSurfaceResourceAddress;
    frame1PostSurfaceInitialState = g_frame1PostSurfaceInitialState;
    frame1PostSurfaceInitialFrame = g_frame1PostSurfaceInitialFrame;
    postActivationReadback = g_postActivationPreLaunchReadback;
    postActivationBytes = g_postActivationPreLaunchBytes;
    postActivationParam0 = g_postActivationParam0;
    postActivationParam8 = g_postActivationParam8;
    postActivationResourceAddress = g_postActivationResourceAddress;
    postActivationCaptureState = g_postActivationPreLaunchState;
    if (input) input->AddRef();
    if (output) output->AddRef();
    if (postTexture) postTexture->AddRef();
    if (postTextureReadback.buffer) postTextureReadback.buffer->AddRef();
    if (frame1PostOutputReadback.buffer) frame1PostOutputReadback.buffer->AddRef();
    if (frame1PostSurfaceInitialReadback.buffer)
        frame1PostSurfaceInitialReadback.buffer->AddRef();
    if (postActivationReadback) postActivationReadback->AddRef();
    LeaveCriticalSection(&g_resourceCs);

    HRESULT hr = S_OK;
    int result = 0;
    ID3D12Device* device = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE eventHandle = nullptr;
    SnapshotReadback inputReadback{}, outputReadback{};
    uint64_t inputHash = 0, outputHash = 0, postTextureHash = 0;
    uint64_t frame1PostOutputHash = 0;
    uint64_t frame1PostSurfaceInitialHash = 0;
    uint64_t postActivationHash = 0;
    UINT64 inputBytes = 0, outputBytes = 0, postTextureBytes = 0;
    UINT64 frame1PostOutputBytes = 0;
    UINT64 frame1PostSurfaceInitialBytes = 0;
    D3D12_RESOURCE_DESC inputDesc{}, outputDesc{};
    char inputPath[MAX_PATH * 4]{};
    char outputPath[MAX_PATH * 4]{};
    char metadataPath[MAX_PATH * 4]{};
    char postTexturePath[MAX_PATH * 4]{};
    char postTextureMetadataPath[MAX_PATH * 4]{};
    char frame1PostOutputPath[MAX_PATH * 4]{};
    char frame1PostOutputMetadataPath[MAX_PATH * 4]{};
    char frame1PostSurfaceInitialPath[MAX_PATH * 4]{};
    char frame1PostSurfaceInitialMetadataPath[MAX_PATH * 4]{};
    char postActivationPath[MAX_PATH * 4]{};
    char postActivationMetadataPath[MAX_PATH * 4]{};

    if (!queue || !directory || !*directory || !input || !output) {
        hr = E_INVALIDARG;
        goto cleanup;
    }
    CreateDirectoryA(directory, nullptr);
    snprintf(inputPath, sizeof(inputPath), "%s\\copy_input.raw", directory);
    snprintf(outputPath, sizeof(outputPath), "%s\\copy_output.raw", directory);
    snprintf(metadataPath, sizeof(metadataPath), "%s\\copy_snapshot.json", directory);
    if (postTexture) {
        snprintf(postTexturePath, sizeof(postTexturePath),
                 "%s\\post_texture_input.raw", directory);
        snprintf(postTextureMetadataPath, sizeof(postTextureMetadataPath),
                 "%s\\post_texture_snapshot.json", directory);
    }
    if (frame1PostOutputReadback.buffer) {
        snprintf(frame1PostOutputPath, sizeof(frame1PostOutputPath),
                 "%s\\frame1_post_output_pre_copy.raw", directory);
        snprintf(frame1PostOutputMetadataPath,
                 sizeof(frame1PostOutputMetadataPath),
                 "%s\\frame1_post_output_pre_copy.json", directory);
    }
    if (frame1PostSurfaceInitialReadback.buffer) {
        snprintf(frame1PostSurfaceInitialPath,
                 sizeof(frame1PostSurfaceInitialPath),
                 "%s\\frame1_post_surface_pre_slot154.raw", directory);
        snprintf(frame1PostSurfaceInitialMetadataPath,
                 sizeof(frame1PostSurfaceInitialMetadataPath),
                 "%s\\frame1_post_surface_pre_slot154.json", directory);
    }
    if (postActivationReadback) {
        snprintf(postActivationPath, sizeof(postActivationPath),
                 "%s\\post_activation_arena_prelaunch.raw", directory);
        snprintf(postActivationMetadataPath, sizeof(postActivationMetadataPath),
                 "%s\\post_activation_arena_prelaunch.json", directory);
    }
    inputDesc = input->GetDesc();
    outputDesc = output->GetDesc();
    if (inputDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        outputDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        inputDesc.Width != outputDesc.Width || inputDesc.Height != outputDesc.Height ||
        inputDesc.Format != outputDesc.Format) {
        hr = E_INVALIDARG;
        goto cleanup;
    }
    if (postTexture) {
        if (postTextureCaptureState != 2 || !postTextureReadback.buffer ||
            postTextureDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            hr = E_INVALIDARG;
            goto cleanup;
        }
    }
    if (frame1PostOutputState != 0 &&
        (frame1PostOutputState != 2 || !frame1PostOutputReadback.buffer ||
         frame1PostOutputDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)) {
        hr = E_INVALIDARG;
        goto cleanup;
    }
    if (frame1PostSurfaceInitialState != 0 &&
        (frame1PostSurfaceInitialState != 2 ||
         !frame1PostSurfaceInitialReadback.buffer ||
         frame1PostSurfaceInitialDesc.Dimension !=
             D3D12_RESOURCE_DIMENSION_TEXTURE2D)) {
        hr = E_INVALIDARG;
        goto cleanup;
    }
    if (postActivationCaptureState != 0 &&
        (postActivationCaptureState != 2 || !postActivationReadback ||
         !postActivationBytes)) {
        hr = E_INVALIDARG;
        goto cleanup;
    }
    hr = input->GetDevice(IID_PPV_ARGS(&device));
    if (FAILED(hr)) goto cleanup;
    hr = CreateSnapshotReadback(device, input, &inputReadback);
    if (FAILED(hr)) goto cleanup;
    hr = CreateSnapshotReadback(device, output, &outputReadback);
    if (FAILED(hr)) goto cleanup;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                        IID_PPV_ARGS(&allocator));
    if (FAILED(hr)) goto cleanup;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator,
                                   nullptr, IID_PPV_ARGS(&list));
    if (FAILED(hr)) goto cleanup;
    {
        D3D12_TEXTURE_COPY_LOCATION inputDst{};
        inputDst.pResource = inputReadback.buffer;
        inputDst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        inputDst.PlacedFootprint = inputReadback.footprint;
        D3D12_TEXTURE_COPY_LOCATION inputSrc{};
        inputSrc.pResource = input;
        inputSrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        inputSrc.SubresourceIndex = 0;
        list->CopyTextureRegion(&inputDst, 0, 0, 0, &inputSrc, nullptr);

        D3D12_TEXTURE_COPY_LOCATION outputDst{};
        outputDst.pResource = outputReadback.buffer;
        outputDst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        outputDst.PlacedFootprint = outputReadback.footprint;
        D3D12_TEXTURE_COPY_LOCATION outputSrc{};
        outputSrc.pResource = output;
        outputSrc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        outputSrc.SubresourceIndex = 0;
        list->CopyTextureRegion(&outputDst, 0, 0, 0, &outputSrc, nullptr);

    }
    hr = list->Close();
    if (FAILED(hr)) goto cleanup;
    {
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
    }
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) goto cleanup;
    hr = queue->Signal(fence, 1);
    if (FAILED(hr)) goto cleanup;
    eventHandle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!eventHandle) { hr = HRESULT_FROM_WIN32(GetLastError()); goto cleanup; }
    hr = fence->SetEventOnCompletion(1, eventHandle);
    if (FAILED(hr)) goto cleanup;
    if (WaitForSingleObject(eventHandle, 15000) != WAIT_OBJECT_0) {
        hr = HRESULT_FROM_WIN32(ERROR_TIMEOUT);
        goto cleanup;
    }
    if (!WriteCompactSnapshot(inputPath, &inputReadback, &inputHash, &inputBytes) ||
        !WriteCompactSnapshot(outputPath, &outputReadback, &outputHash, &outputBytes)) {
        hr = E_FAIL;
        goto cleanup;
    }
    if (postTexture && !WriteCompactSnapshot(
            postTexturePath, &postTextureReadback,
            &postTextureHash, &postTextureBytes)) {
        hr = E_FAIL;
        goto cleanup;
    }
    if (frame1PostOutputReadback.buffer && !WriteCompactSnapshot(
            frame1PostOutputPath, &frame1PostOutputReadback,
            &frame1PostOutputHash, &frame1PostOutputBytes)) {
        hr = E_FAIL;
        goto cleanup;
    }
    if (frame1PostSurfaceInitialReadback.buffer && !WriteCompactSnapshot(
            frame1PostSurfaceInitialPath, &frame1PostSurfaceInitialReadback,
            &frame1PostSurfaceInitialHash, &frame1PostSurfaceInitialBytes)) {
        hr = E_FAIL;
        goto cleanup;
    }
    if (postActivationReadback &&
        !WriteLinearSnapshot(postActivationPath, postActivationReadback,
                             postActivationBytes, &postActivationHash)) {
        hr = E_FAIL;
        goto cleanup;
    }
    if (g_fullGraphWindowCount) DumpFullGraphCaptureFiles(directory);
    DumpNeuralCaptureFiles(directory);
    result = 1;

cleanup:
    if (frame1PostSurfaceInitialMetadataPath[0]) {
        FILE* initialMetadata = nullptr;
        fopen_s(&initialMetadata, frame1PostSurfaceInitialMetadataPath, "wb");
        if (initialMetadata) {
            fprintf(initialMetadata,
                "{\n  \"schema\": 1,\n  \"status\": \"%s\",\n"
                "  \"classification\": \"FRAME1_SLOT154_OUTPUT_SURFACE_INITIAL\",\n"
                "  \"capture_timing\": \"same command list immediately before frame-1 slot-154 launch\",\n"
                "  \"capture_frame\": %ld,\n  \"hresult\": \"0x%08X\",\n"
                "  \"resource_state_assumption\": \"COMMON implicit COPY_SOURCE promotion after queue ordering\",\n"
                "  \"surface_object\": \"0x%016llX\",\n"
                "  \"surface_resource\": \"0x%llX\",\n"
                "  \"width\": %llu,\n  \"height\": %u,\n  \"format\": %u,\n"
                "  \"row_size\": %llu,\n  \"num_rows\": %u,\n"
                "  \"bytes\": %llu,\n  \"fnv1a64\": \"0x%016llX\",\n"
                "  \"file\": \"frame1_post_surface_pre_slot154.raw\"\n}\n",
                result ? "PASS" : "FAIL", frame1PostSurfaceInitialFrame,
                (unsigned)hr, (unsigned long long)frame1PostSurfaceObject,
                (unsigned long long)frame1PostSurfaceResourceAddress,
                (unsigned long long)frame1PostSurfaceInitialDesc.Width,
                frame1PostSurfaceInitialDesc.Height,
                (unsigned)frame1PostSurfaceInitialDesc.Format,
                (unsigned long long)frame1PostSurfaceInitialReadback.rowSize,
                frame1PostSurfaceInitialReadback.rows,
                (unsigned long long)frame1PostSurfaceInitialBytes,
                (unsigned long long)frame1PostSurfaceInitialHash);
            fclose(initialMetadata);
        }
    }
    if (frame1PostOutputMetadataPath[0]) {
        FILE* frameMetadata = nullptr;
        fopen_s(&frameMetadata, frame1PostOutputMetadataPath, "wb");
        if (frameMetadata) {
            fprintf(frameMetadata,
                "{\n  \"schema\": 1,\n  \"status\": \"%s\",\n"
                "  \"classification\": \"FRAME1_SLOT154_OUTPUT_BEFORE_SLOT155_COPY\",\n"
                "  \"capture_timing\": \"same command list immediately before frame-1 slot-155 launch\",\n"
                "  \"capture_frame\": %ld,\n  \"hresult\": \"0x%08X\",\n"
                "  \"resource_state_assumption\": \"COMMON implicit COPY_SOURCE promotion after queue ordering\",\n"
                "  \"surface_object\": \"0x%016llX\",\n"
                "  \"surface_resource\": \"0x%llX\",\n"
                "  \"width\": %llu,\n  \"height\": %u,\n  \"format\": %u,\n"
                "  \"row_size\": %llu,\n  \"num_rows\": %u,\n"
                "  \"bytes\": %llu,\n  \"fnv1a64\": \"0x%016llX\",\n"
                "  \"file\": \"frame1_post_output_pre_copy.raw\"\n}\n",
                result ? "PASS" : "FAIL", frame1PostOutputFrame, (unsigned)hr,
                (unsigned long long)frame1PostOutputObject,
                (unsigned long long)frame1PostOutputResourceAddress,
                (unsigned long long)frame1PostOutputDesc.Width,
                frame1PostOutputDesc.Height, (unsigned)frame1PostOutputDesc.Format,
                (unsigned long long)frame1PostOutputReadback.rowSize,
                frame1PostOutputReadback.rows,
                (unsigned long long)frame1PostOutputBytes,
                (unsigned long long)frame1PostOutputHash);
            fclose(frameMetadata);
        }
    }
    if (postActivationMetadataPath[0]) {
        FILE* activationMetadata = nullptr;
        fopen_s(&activationMetadata, postActivationMetadataPath, "wb");
        if (activationMetadata) {
            fprintf(activationMetadata,
                "{\n  \"schema\": 1,\n  \"status\": \"%s\",\n"
                "  \"classification\": \"SLOT154_PRELAUNCH_FULL_ACTIVATION_BUFFER\",\n"
                "  \"capture_timing\": \"same command list immediately before frame-1 slot-154 launch\",\n"
                "  \"frame\": 1,\n  \"resource\": \"0x%llX\",\n"
                "  \"bytes\": %llu,\n  \"param0\": \"0x%016llX\",\n"
                "  \"param8\": \"0x%016llX\",\n"
                "  \"fnv1a64\": \"0x%016llX\",\n"
                "  \"file\": \"post_activation_arena_prelaunch.raw\"\n}\n",
                result ? "PASS" : "FAIL",
                (unsigned long long)postActivationResourceAddress,
                (unsigned long long)postActivationBytes,
                (unsigned long long)postActivationParam0,
                (unsigned long long)postActivationParam8,
                (unsigned long long)postActivationHash);
            fclose(activationMetadata);
        }
    }
    if (postTextureMetadataPath[0]) {
        FILE* postMetadata = nullptr;
        fopen_s(&postMetadata, postTextureMetadataPath, "wb");
        if (postMetadata) {
            fprintf(postMetadata,
                "{\n  \"schema\": 1,\n  \"status\": \"%s\",\n"
                "  \"classification\": \"SLOT154_PRELAUNCH_TEXTURE_INPUT\",\n"
                "  \"capture_timing\": \"same command list immediately before frame-1 slot-154 launch\",\n"
                "  \"capture_frame\": %ld,\n"
                "  \"hresult\": \"0x%08X\",\n"
                "  \"resource_state_assumption\": \"COMMON implicit COPY_SOURCE promotion after queue ordering\",\n"
                "  \"texture_object\": \"0x%016llX\",\n"
                "  \"texture_resource\": \"0x%llX\",\n"
                "  \"width\": %llu,\n  \"height\": %u,\n  \"format\": %u,\n"
                "  \"row_size\": %llu,\n  \"num_rows\": %u,\n"
                "  \"bytes\": %llu,\n  \"fnv1a64\": \"0x%016llX\",\n"
                "  \"file\": \"post_texture_input.raw\"\n}\n",
                result ? "PASS" : "FAIL", postTextureCaptureFrame, (unsigned)hr,
                (unsigned long long)postTextureObject,
                (unsigned long long)(uintptr_t)postTexture,
                (unsigned long long)postTextureDesc.Width,
                postTextureDesc.Height, (unsigned)postTextureDesc.Format,
                (unsigned long long)postTextureReadback.rowSize,
                postTextureReadback.rows,
                (unsigned long long)postTextureBytes,
                (unsigned long long)postTextureHash);
            fclose(postMetadata);
        }
    }
    if (metadataPath[0]) {
        FILE* metadata = nullptr;
        fopen_s(&metadata, metadataPath, "wb");
        if (metadata) {
            fprintf(metadata,
                "{\n  \"schema\": 1,\n  \"status\": \"%s\",\n"
                "  \"hresult\": \"0x%08X\",\n"
                "  \"resource_state_assumption\": \"COMMON implicit COPY_SOURCE promotion after queue ordering\",\n"
                "  \"width\": %llu,\n  \"height\": %u,\n  \"format\": %u,\n"
                "  \"row_size\": %llu,\n  \"num_rows\": %u,\n"
                "  \"input_bytes\": %llu,\n  \"output_bytes\": %llu,\n"
                "  \"input_resource\": \"0x%llx\",\n"
                "  \"output_resource\": \"0x%llx\",\n"
                "  \"input_fnv1a64\": \"0x%016llX\",\n"
                "  \"output_fnv1a64\": \"0x%016llX\",\n"
                "  \"input_file\": \"copy_input.raw\",\n"
                "  \"output_file\": \"copy_output.raw\"\n}\n",
                result ? "PASS" : "FAIL", (unsigned)hr,
                (unsigned long long)inputDesc.Width, inputDesc.Height,
                (unsigned)inputDesc.Format,
                (unsigned long long)inputReadback.rowSize, inputReadback.rows,
                (unsigned long long)inputBytes, (unsigned long long)outputBytes,
                (unsigned long long)(uintptr_t)input,
                (unsigned long long)(uintptr_t)output,
                (unsigned long long)inputHash, (unsigned long long)outputHash);
            fclose(metadata);
        }
    }
    Log("{\"ev\":\"copy_snapshot\",\"ts\":%.3f,\"status\":\"%s\","
        "\"hresult\":\"0x%08X\",\"input_resource\":\"0x%llx\","
        "\"output_resource\":\"0x%llx\",\"input_bytes\":%llu,"
        "\"output_bytes\":%llu}", NowMs(), result ? "PASS" : "FAIL",
        (unsigned)hr, (unsigned long long)(uintptr_t)input,
        (unsigned long long)(uintptr_t)output,
        (unsigned long long)inputBytes, (unsigned long long)outputBytes);
    if (eventHandle) CloseHandle(eventHandle);
    if (fence) fence->Release();
    if (list) list->Release();
    if (allocator) allocator->Release();
    if (inputReadback.buffer) inputReadback.buffer->Release();
    if (outputReadback.buffer) outputReadback.buffer->Release();
    if (postTextureReadback.buffer) postTextureReadback.buffer->Release();
    if (frame1PostOutputReadback.buffer) frame1PostOutputReadback.buffer->Release();
    if (frame1PostSurfaceInitialReadback.buffer)
        frame1PostSurfaceInitialReadback.buffer->Release();
    if (postActivationReadback) postActivationReadback->Release();
    if (device) device->Release();
    if (input) input->Release();
    if (output) output->Release();
    if (postTexture) postTexture->Release();
    return result;
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_TestBindCopyResources(
        ID3D12Resource* input, ID3D12Resource* output) {
    RememberCopyResources(input, output);
    return input && output ? 1 : 0;
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_TestBeginPostTextureCapture(
        ID3D12GraphicsCommandList* list, ID3D12Resource* texture,
        uint64_t textureObject) {
    RememberPostTextureResource(textureObject, texture);
    return BeginPostTexturePreLaunchCapture(
        list, texture, textureObject, 1) ? 1 : 0;
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_TestBeginFrame1PostOutputCapture(
        ID3D12GraphicsCommandList* list, ID3D12Resource* texture,
        uint64_t surfaceObject) {
    return BeginFrame1PostOutputCapture(
        list, texture, surfaceObject, 1) ? 1 : 0;
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_TestBeginFrame1PostSurfaceInitialCapture(
        ID3D12GraphicsCommandList* list, ID3D12Resource* texture,
        uint64_t surfaceObject) {
    return BeginFrame1PostSurfaceInitialCapture(
        list, texture, surfaceObject, 1) ? 1 : 0;
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_TestBeginNeuralCapture(
        ID3D12GraphicsCommandList* list, const void* parameters, uint32_t size) {
    return BeginNeuralCapture(list, parameters, size) ? 1 : 0;
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_TestEndNeuralCapture(
        ID3D12GraphicsCommandList* list) {
    if (InterlockedCompareExchange(&g_neuralCaptureState, 1, 1) != 1) return 0;
    EndNeuralCapture(list);
    return InterlockedCompareExchange(&g_neuralCaptureState, 2, 2) == 2 ? 1 : 0;
}

extern "C" __declspec(dllexport) uint64_t WINAPI ModuleTrace_TestBeginFullGraphCapture(
        ID3D12GraphicsCommandList* list, uint32_t slot, uint64_t function,
        const void* parameters, uint32_t size) {
    FullGraphCaptureSpan span = BeginFullGraphCapture(
        list, slot, function, parameters, size);
    return (uint64_t(span.first) << 32) | uint64_t(span.count);
}

extern "C" __declspec(dllexport) int WINAPI ModuleTrace_TestEndFullGraphCapture(
        ID3D12GraphicsCommandList* list, uint64_t encodedSpan) {
    FullGraphCaptureSpan span{
        size_t(encodedSpan >> 32), size_t(encodedSpan & 0xFFFFFFFFull)};
    EndFullGraphCapture(list, span);
    return span.count ? 1 : 0;
}

extern "C" __declspec(dllexport) const char* module_trace_version() {
    return "dlssnr-amd-lab module_trace 3.5 (D3D12 submission timeline)";
}

extern "C" __declspec(dllexport) int ModuleTrace_RegisterD3D12Device(
        ID3D12Device* device) {
    return InstallDeviceHooks(device) ? 1 : 0;
}

// Narrow self-test seam: exercise the same independent-descriptor wrapper used
// by production nvapi_QueryInterface interception without relying on the host
// compiler's loader-import call shape.
extern "C" __declspec(dllexport) void* ModuleTrace_TestWrapIndependentDescriptor(
        void* realFunction) {
    if (!realFunction) return nullptr;
    InterlockedExchangePointer((PVOID volatile*)&g_realIndependentDescriptor,
                               realFunction);
    return reinterpret_cast<void*>(&HookIndependentDescriptor);
}

extern "C" __declspec(dllexport) void* ModuleTrace_TestWrapNvapiQuery(void* realFunction) {
    if (!realFunction) return nullptr;
    InterlockedExchangePointer((PVOID volatile*)&g_realNvApiQueryInterface, realFunction);
    return reinterpret_cast<void*>(&HookNvApiQueryInterface);
}
// Call at host shutdown after normal queue completion. Normal launch/reset/submit
// hooks also collect completed sessions; this drains the final idle submission.
extern "C" __declspec(dllexport) int ModuleTrace_CollectAmdHeads() {
    auto_head::Collect();
    std::lock_guard<std::recursive_mutex> guard(auto_head::mutex);
    return static_cast<int>(auto_head::pending.size());
}

// Synchronous initialization gate. Returns 1 once hooks_installed=true.
// Safe to call multiple times / from multiple threads (one-shot + event wait).
extern "C" __declspec(dllexport) int ModuleTrace_InitializeAndWait(DWORD timeoutMs) {
    // hosts may set MODULE_TRACE_LOG after LoadLibrary but before this call;
    // honor it here (keeps attach events in the default log, everything
    // interesting in the requested log)
    OpenLog();
    if (InterlockedCompareExchange(&g_initStarted, 1, 0) == 0) DoInit();
    if (!g_readyEvt) return g_hooksInstalled;
    return WaitForSingleObject(g_readyEvt, timeoutMs ? timeoutMs : 10000) == WAIT_OBJECT_0
               && g_hooksInstalled ? 1 : 0;
}
