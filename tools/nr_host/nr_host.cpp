// nr_host — Round 2 Phase C: a real NGX host (D3D12 x64), official ABI only.
//
// Pipeline (R2/R3):
//   deterministic synthetic inputs (Color RGBA8 / Depth R32F / MV R16G16F)
//   -> upload -> NVSDK_NGX_D3D12_Init (loader ABI, ProjectID fallback)
//   -> GetCapabilityParameters / AllocateParameters (official lifecycle)
//   -> CreateFeature(SuperSampling, DLAA contract per DLSS5-Feeder)
//   -> record command list -> EvaluateFeature -> ExecuteCommandLists -> fence
//   -> readback + FNV-1a hash + BMP dump -> ReleaseFeature -> Shutdown1
//
// Every step has its own machine-readable status; nothing is fabricated:
// without a real NVIDIA driver stack the loader's Init fails and we record
// exactly that. Feature id 18 (Neural Rendering) is PRIVATE ABI and lives
// only in private_ngx_compat.h — Stage A here uses public DLSS/DLAA only.
//
// Usage:
//   nr_host.exe --frames N --width W --height H [--input <raw color rgba8>]
//               [--output <path>] [--trace] [--json <path>] [--force-load]
//
// Environment:
//   DLSS_DLL_PATH    feature DLL (its directory is added to NGX PathListInfo)
//   DLSSNR_DLL_PATH  NR runtime DLL (loaded for trace observation only)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// PUBLIC ABI — verified (third_party/nvidia-dlss, see third_party/PROVENANCE.md)
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
// PRIVATE ABI — inferred (feature 18 only, reserved for Stage B)
#include "private_ngx_compat.h"

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------

struct Options {
    int frames = 8;
    int width = 512;
    int height = 512;
    bool trace = false;
    bool forceLoad = false;   // run on the first hardware adapter even without NVIDIA
    const char* input = nullptr;
    const char* output = nullptr;
    const char* json = nullptr;
};

static uint32_t g_lcg = 0x2545F491u;
static inline uint32_t Lcg() { g_lcg = g_lcg * 1664525u + 1013904223u; return g_lcg; }

// Deterministic synthetic inputs (byte-identical to the reference package).
static void GenColor(std::vector<uint8_t>& out, int w, int h) {
    out.resize((size_t)w * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t* p = &out[((size_t)y * w + x) * 4];
            p[0] = (uint8_t)((x * 255 / (w - 1 ? w - 1 : 1)));
            p[1] = (uint8_t)((y * 255 / (h - 1 ? h - 1 : 1)));
            p[2] = (uint8_t)(((x ^ y) & 0xFF));
            p[3] = (uint8_t)(Lcg() & 0xFF);
        }
}
static void GenDepth(std::vector<float>& out, int w, int h) {
    out.resize((size_t)w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            out[(size_t)y * w + x] = 0.001f + 0.999f * ((float)y / (float)(h - 1 ? h - 1 : 1));
}
// Motion vectors in pixels, deterministic ±0.5 px (R16G16_FLOAT on the GPU).
static void GenMotionF(std::vector<float>& out, int w, int h) {
    out.resize((size_t)w * h * 2);
    for (size_t i = 0; i < out.size(); i += 2) {
        out[i]     = ((float)(Lcg() % 1024) / 1024.0f - 0.5f);
        out[i + 1] = ((float)(Lcg() % 1024) / 1024.0f - 0.5f);
    }
}

static uint16_t F32ToF16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0) return (uint16_t)sign;           // 0 / subnormal -> 0
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);              // inf
    if (exp <= 0)  return (uint16_t)(sign | (mant >> 13));        // flush small values
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static uint64_t Fnv1a(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ull; }
    return h;
}

static void WriteRaw(const char* path, const void* data, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(data, 1, bytes, f); fclose(f); }
}

// Minimal 24-bit BMP writer for quick visual inspection of the output.
static void WriteBmp(const char* path, const uint8_t* rgba, int w, int h) {
    int rowBytes = w * 3, padded = (rowBytes + 3) & ~3;
    std::vector<uint8_t> img((size_t)padded * h, 0);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = rgba + (size_t)(h - 1 - y) * w * 4;  // BMP is bottom-up
        uint8_t* dst = &img[(size_t)y * padded];
        for (int x = 0; x < w; ++x) {
            dst[x * 3 + 0] = src[x * 4 + 2];
            dst[x * 3 + 1] = src[x * 4 + 1];
            dst[x * 3 + 2] = src[x * 4 + 0];
        }
    }
    BITMAPFILEHEADER fh{}; BITMAPINFOHEADER ih{};
    fh.bfType = 0x4D42; fh.bfOffBits = 54; fh.bfSize = 54 + (DWORD)img.size();
    ih.biSize = sizeof(ih); ih.biWidth = w; ih.biHeight = h; ih.biPlanes = 1; ih.biBitCount = 24;
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(&fh, 1, sizeof(fh), f); fwrite(&ih, 1, sizeof(ih), f);
    fwrite(img.data(), 1, img.size(), f); fclose(f);
}

struct StepStatus {
    const char* name;
    const char* status;   // PASS / FAIL / SKIPPED / BLOCKED_*
    char detail[160];
};
static std::vector<StepStatus> g_steps;
static void Step(const char* name, const char* status, const char* fmt, ...) {
    StepStatus s{}; s.name = name; s.status = status;
    va_list ap; va_start(ap, fmt);
    vsnprintf(s.detail, sizeof(s.detail), fmt, ap);
    va_end(ap);
    g_steps.push_back(s);
    printf("[%-16s] %-8s %s\n", name, status, s.detail);
}

static const char* NgxName(NVSDK_NGX_Result r) {
    switch ((unsigned)r) {
    case 0x1: return "Success";
    case 0xBAD00001: return "FAIL_FeatureNotSupported";
    case 0xBAD00002: return "FAIL_PlatformError";
    case 0xBAD00005: return "FAIL_InvalidParameter";
    case 0xBAD00007: return "FAIL_NotInitialized";
    case 0xBAD00008: return "FAIL_UnsupportedInputFormat";
    case 0xBAD0000A: return "FAIL_MissingInput";
    case 0xBAD0000B: return "FAIL_UnableToInitializeFeature";
    case 0xBAD0000C: return "FAIL_OutOfDate";
    case 0xBAD0000D: return "FAIL_OutOfGPUMemory";
    case 0xBAD0000E: return "FAIL_UnsupportedFormat";
    case 0xBAD00012: return "FAIL_NotImplemented";
    default: return "?";
    }
}

// ---------------------------------------------------------------------------
// SEH wrappers: NGX calls are guarded exactly like DLSS5-Feeder does; a faulted
// NGX must never take down the host or produce fake evidence.
// The wrappers must not contain C++ objects needing unwinding.

static NVSDK_NGX_Result SafeNgxInit(unsigned long long appId, const wchar_t* path,
                                    ID3D12Device* dev, const NVSDK_NGX_FeatureCommonInfo* info,
                                    DWORD* code) {
    *code = 0;
    __try { return NVSDK_NGX_D3D12_Init(appId, path, dev, info, NVSDK_NGX_Version_API); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return NVSDK_NGX_Result_Fail; }
}
static NVSDK_NGX_Result SafeNgxInitProjectID(const char* pid, const wchar_t* path,
                                              ID3D12Device* dev, const NVSDK_NGX_FeatureCommonInfo* info,
                                              DWORD* code) {
    *code = 0;
    __try {
        return NVSDK_NGX_D3D12_Init_with_ProjectID(pid, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
                                                   path, dev, info, NVSDK_NGX_Version_API);
    } __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return NVSDK_NGX_Result_Fail; }
}
static NVSDK_NGX_Result SafeCreateDlss(ID3D12GraphicsCommandList* list, NVSDK_NGX_Handle** out,
                                       NVSDK_NGX_Parameter* params, NVSDK_NGX_DLSS_Create_Params* cp,
                                       DWORD* code) {
    *code = 0;
    __try { return NGX_D3D12_CREATE_DLSS_EXT(list, 1, 1, out, params, cp); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return NVSDK_NGX_Result_Fail; }
}
static NVSDK_NGX_Result SafeEvaluateDlss(ID3D12GraphicsCommandList* list, NVSDK_NGX_Handle* h,
                                         NVSDK_NGX_Parameter* params, NVSDK_NGX_D3D12_DLSS_Eval_Params* ep,
                                         DWORD* code) {
    *code = 0;
    __try { return NGX_D3D12_EVALUATE_DLSS_EXT(list, h, params, ep); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); return NVSDK_NGX_Result_Fail; }
}
static void SafeReleaseFeature(NVSDK_NGX_Handle* h, DWORD* code) {
    *code = 0;
    __try { NVSDK_NGX_D3D12_ReleaseFeature(h); }
    __except (EXCEPTION_EXECUTE_HANDLER) { *code = GetExceptionCode(); }
}

// ---------------------------------------------------------------------------

static ComPtr<ID3D12Resource> MakeTex(ID3D12Device* dev, UINT w, UINT h, DXGI_FORMAT fmt, bool uav) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w; rd.Height = h; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.Format = fmt; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> t;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                 nullptr, IID_PPV_ARGS(&t));
    return t;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) o.frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width") && i + 1 < argc) o.width = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i + 1 < argc) o.height = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--input") && i + 1 < argc) o.input = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) o.output = argv[++i];
        else if (!strcmp(argv[i], "--json") && i + 1 < argc) o.json = argv[++i];
        else if (!strcmp(argv[i], "--trace")) o.trace = true;
        else if (!strcmp(argv[i], "--force-load")) o.forceLoad = true;
    }
    if (o.width < 8 || o.height < 8 || o.frames < 1) { printf("invalid arguments\n"); return 2; }

    printf("=== nr_host (dlssnr-amd-lab Round 3: complete DLSS evaluate contract) ===\n");
    if (o.trace) {
        printf("trace mode: set these env vars and rerun under the shims:\n");
        printf("  MODULE_TRACE_LOG=<path>  (inject build\\module_trace.dll)\n");
        printf("  NVAPI_TRACE_LOG=<path>   (place build\\nvapi64.dll as the game's nvapi64.dll)\n");
    }

    // ---- deterministic inputs ----
    g_lcg = 0x2545F491u;
    std::vector<uint8_t> color; std::vector<float> depth; std::vector<float> motion;
    GenColor(color, o.width, o.height);
    GenDepth(depth, o.width, o.height);
    GenMotionF(motion, o.width, o.height);
    if (o.input) {
        FILE* f = fopen(o.input, "rb");
        if (f) { size_t n = fread(color.data(), 1, color.size(), f); fclose(f);
                 printf("loaded external color input: %zu bytes\n", n); }
        else printf("WARNING: cannot open --input %s; using synthetic color\n", o.input);
    }
    printf("inputs: color=%zuB depth=%zuB motion=%zuB (deterministic LCG 0x2545F491)\n",
           color.size(), depth.size() * 4, motion.size() * 4);

    // ---- reference package (always produced; byte-identical for the RTX machine) ----
    char ts[32]; SYSTEMTIME st; GetLocalTime(&st);
    snprintf(ts, sizeof(ts), "%04u%02u%02u_%02u%02u%02u", st.wYear, st.wMonth, st.wDay,
             st.wHour, st.wMinute, st.wSecond);
    char pkgDir[512];
    snprintf(pkgDir, sizeof(pkgDir), "results\\%s_reference_package", ts);
    CreateDirectoryA("results", nullptr);
    CreateDirectoryA(pkgDir, nullptr);
    WriteRaw((std::string(pkgDir) + "\\color_rgba8.bin").c_str(), color.data(), color.size());
    WriteRaw((std::string(pkgDir) + "\\depth_r32f.bin").c_str(), depth.data(), depth.size() * 4);
    WriteRaw((std::string(pkgDir) + "\\motion_rg16f.bin").c_str(), motion.data(), motion.size() * 4);
    printf("reference package written to %s\\\n", pkgDir);

    // ---- environment / prerequisites ----
    const char* dlssPath = getenv("DLSS_DLL_PATH");
    const char* nrPath = getenv("DLSSNR_DLL_PATH");
    bool hasDlss = dlssPath && *dlssPath && GetFileAttributesA(dlssPath) != INVALID_FILE_ATTRIBUTES;
    if (!hasDlss) {
        Step("prereq", "BLOCKED_MISSING_PREREQUISITE", "nvngx_dlss.dll missing (set DLSS_DLL_PATH)");
        printf("{\"status\": \"BLOCKED_MISSING_PREREQUISITE\"}\n");
        return 3;
    }

    bool hasNvidia = false;
    ComPtr<IDXGIFactory7> factory;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> nvAdapter, anyAdapter;
    if (factory) {
        ComPtr<IDXGIAdapter1> a;
        for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i, a.Reset()) {
            DXGI_ADAPTER_DESC1 d{}; a->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (!anyAdapter) anyAdapter = a;
            if (d.VendorId == 0x10DE && !nvAdapter) nvAdapter = a;
        }
    }
    hasNvidia = nvAdapter != nullptr;
    if (!hasNvidia && !o.forceLoad) {
        Step("prereq", "BLOCKED_NO_NVIDIA_GPU",
             "no NVIDIA adapter; reference run impossible here, package emitted, nothing fabricated");
        printf("{\"status\": \"BLOCKED_NO_NVIDIA_GPU\"}\n");
        return 4;
    }
    ComPtr<IDXGIAdapter1> useAdapter = hasNvidia ? nvAdapter : anyAdapter;
    DXGI_ADAPTER_DESC1 usedDesc{};
    if (useAdapter) useAdapter->GetDesc1(&usedDesc);
    printf("adapter: vendor=0x%04x device=0x%04x (0x1002=AMD, 0x10DE=NVIDIA)%s\n",
           usedDesc.VendorId, usedDesc.DeviceId,
           (!hasNvidia && o.forceLoad) ? " [--force-load experiment on non-NVIDIA]" : "");

    // ---- device + direct queue + command infra ----
    ComPtr<ID3D12Device> device;
    if (!useAdapter || FAILED(D3D12CreateDevice(useAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
        Step("device", "FAIL", "D3D12CreateDevice failed");
        printf("{\"status\": \"BLOCKED_DEVICE_CREATE_FAILED\"}\n");
        return 7;
    }
    Step("device", "PASS", "D3D12 device created");
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fenceValue = 0;
    if (!queue || !alloc || !list || !fence) {
        Step("device", "FAIL", "queue/allocator/list/fence creation failed");
        return 7;
    }
    auto Submit = [&]() -> bool {
        list->Close();
        ID3D12CommandList* lists[] = { list.Get() };
        queue->ExecuteCommandLists(1, lists);
        UINT64 v = ++fenceValue;
        queue->Signal(fence.Get(), v);
        if (fence->GetCompletedValue() < v) {
            fence->SetEventOnCompletion(v, fenceEvent);
            if (WaitForSingleObject(fenceEvent, 10000) != WAIT_OBJECT_0) return false;
        }
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        return true;
    };

    if (o.trace) {
        HMODULE mt = LoadLibraryA("module_trace.dll");
        if (!mt) {
            printf("module_trace self-inject FAILED (lasterr=%lu)\n", GetLastError());
            Step("module_trace", "FAIL", "LoadLibrary(module_trace.dll) failed; trace results incomplete");
        } else {
            // Round 2: wait for hooks_installed=true BEFORE loading anything else
            typedef int(WINAPI* InitAndWait_t)(DWORD);
            auto initAndWait = (InitAndWait_t)GetProcAddress(mt, "ModuleTrace_InitializeAndWait");
            int ok = initAndWait ? initAndWait(10000) : 0;
            printf("module_trace self-inject: %p hooks_installed=%s\n", (void*)mt, ok ? "true" : "false");
            Step("module_trace", ok ? "PASS" : "FAIL",
                 "hooks_installed=%s (%s)", ok ? "true" : "false",
                 initAndWait ? "ModuleTrace_InitializeAndWait" : "export missing (old dll?)");
        }
    }
    if (nrPath && *nrPath) {
        HMODULE hNr = LoadLibraryA(nrPath);
        Step("nr_dll", hNr ? "PASS" : "FAIL", "%s -> %p (loaded for observation only)", nrPath, (void*)hNr);
    }

    int exitCode = 0;
    NVSDK_NGX_Parameter* params = nullptr;
    NVSDK_NGX_Parameter* caps = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    bool ngxInited = false;

    // ---- NGX Init (loader ABI; feature DLL dir via PathListInfo) ----
    wchar_t appDataPath[MAX_PATH] = {};
    GetCurrentDirectoryW(MAX_PATH, appDataPath);
    std::wstring wdir = appDataPath;

    std::wstring wDllDir;
    const wchar_t* pathPtrs[1] = { nullptr };
    NVSDK_NGX_FeatureCommonInfo info{};
    if (dlssPath) {
        std::string s = dlssPath;
        size_t slash = s.find_last_of("\\/");
        std::string dir = (slash == std::string::npos) ? "." : s.substr(0, slash);
        int n = MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, nullptr, 0);
        wDllDir.resize(n - 1);
        MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, &wDllDir[0], n);
        pathPtrs[0] = wDllDir.c_str();
        info.PathListInfo.Path = pathPtrs;
        info.PathListInfo.Length = 1;
    }

    DWORD seCode = 0;
    // Match the public DLSS5-Feeder reference contract exactly. The previous
    // lab-only identifiers (0x4C424C21 / 4c424c21-...) returned
    // FAIL_FeatureNotSupported on both AMD and an RTX 5060, so they were not a
    // valid positive-control initialization contract.
    constexpr unsigned long long kFeederAppId = 0x1000000ull;
    constexpr const char* kFeederProjectId = "a0f57b54-1daf-4934-90ae-c4035c19df04";
    NVSDK_NGX_Result r = SafeNgxInit(kFeederAppId, wdir.c_str(), device.Get(), &info, &seCode);
    printf("NGX Init (official loader ABI, DLSS5-Feeder AppId=0x%llx): 0x%08x (%s)%s\n",
           kFeederAppId,
           (unsigned)r, NgxName(r), seCode ? " [SEH exception caught]" : "");
    if (NVSDK_NGX_FAILED(r)) {
        // Same fallback identifier used by the known-working DLSS5-Feeder host.
        DWORD seCode2 = 0;
        NVSDK_NGX_Result r2 = SafeNgxInitProjectID(kFeederProjectId,
                                                   wdir.c_str(), device.Get(), &info, &seCode2);
        printf("NGX Init_with_ProjectID fallback (%s): 0x%08x (%s)%s\n",
               kFeederProjectId, (unsigned)r2, NgxName(r2),
               seCode2 ? " [SEH exception caught]" : "");
        if (NVSDK_NGX_SUCCEED(r2)) { r = r2; seCode = seCode2; }
    }
    if (NVSDK_NGX_FAILED(r) || seCode) {
        Step("ngx_init", "FAIL", "Init 0x%08x (%s) vendor=0x%04x%s", (unsigned)r, NgxName(r),
             usedDesc.VendorId, seCode ? " SEH-exception" : "");
        printf("{\"status\": \"NGX_INIT_FAILED\", \"result\": \"0x%08x\", \"abi\": \"official_loader\"}\n", (unsigned)r);
        return 6;
    }
    ngxInited = true;
    Step("ngx_init", "PASS", "official loader ABI, 0x%08x", (unsigned)r);

    // ---- capability parameters (read-only query of SuperSampling availability) ----
    r = NVSDK_NGX_D3D12_GetCapabilityParameters(&caps);
    int ssAvail = -1;
    if (NVSDK_NGX_SUCCEED(r) && caps) {
        caps->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &ssAvail);
        Step("capabilities", ssAvail ? "PASS" : "FAIL", "SuperSampling.Available=%d", ssAvail);
    } else {
        Step("capabilities", "FAIL", "GetCapabilityParameters 0x%08x", (unsigned)r);
    }

    // ---- parameter block (official lifecycle: allocate -> use -> destroy) ----
    r = NVSDK_NGX_D3D12_AllocateParameters(&params);
    if (NVSDK_NGX_FAILED(r) || !params) {
        Step("params", "FAIL", "AllocateParameters 0x%08x (%s)", (unsigned)r, NgxName(r));
        printf("{\"status\": \"PARAMS_ALLOC_FAILED\", \"result\": \"0x%08x\"}\n", (unsigned)r);
        NVSDK_NGX_D3D12_Shutdown1(device.Get());
        return 6;
    }
    Step("params", "PASS", "AllocateParameters (official lifecycle)");

    size_t scratch = 0;
    NVSDK_NGX_Result rs = NVSDK_NGX_D3D12_GetScratchBufferSize(NVSDK_NGX_Feature_SuperSampling, params, &scratch);
    Step("scratch", NVSDK_NGX_SUCCEED(rs) ? "PASS" : "FAIL", "GetScratchBufferSize=%zu (0x%08x)", scratch, (unsigned)rs);

    // ---- textures ----
    UINT W = (UINT)o.width, H = (UINT)o.height;
    list->Reset(alloc.Get(), nullptr);
    ComPtr<ID3D12Resource> texColor = MakeTex(device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, false);
    ComPtr<ID3D12Resource> texDepth = MakeTex(device.Get(), W, H, DXGI_FORMAT_R32_FLOAT, false);
    ComPtr<ID3D12Resource> texMv    = MakeTex(device.Get(), W, H, DXGI_FORMAT_R16G16_FLOAT, false);
    ComPtr<ID3D12Resource> texOut   = MakeTex(device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, true);
    if (!texColor || !texDepth || !texMv || !texOut) {
        Step("textures", "FAIL", "texture creation failed");
        return 7;
    }
    // half-encode motion vectors on CPU
    std::vector<uint16_t> mvHalf(motion.size());
    for (size_t i = 0; i < motion.size(); ++i) mvHalf[i] = F32ToF16(motion[i]);
    // staging buffers must live until submit; keep them in a vector
    std::vector<ComPtr<ID3D12Resource>> staging;
    auto Up = [&](ID3D12Resource* tex, const void* data, UINT px, DXGI_FORMAT fmt) {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT64 total = 0;
        D3D12_RESOURCE_DESC texd = tex->GetDesc();
        device->GetCopyableFootprints(&texd, 0, 1, 0, &fp, nullptr, nullptr, &total);
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> sb;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&sb))))
            return false;
        uint8_t* dst = nullptr;
        if (FAILED(sb->Map(0, nullptr, (void**)&dst))) return false;
        const uint8_t* src = (const uint8_t*)data;
        for (UINT y = 0; y < H; ++y)
            memcpy(dst + (size_t)y * fp.Footprint.RowPitch, src + (size_t)y * W * px, (size_t)W * px);
        sb->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION s{}, d{};
        s.pResource = sb.Get(); s.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; s.PlacedFootprint = fp;
        d.pResource = tex; d.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
        staging.push_back(sb);
        return true;
    };
    bool up = Up(texColor.Get(), color.data(), 4, DXGI_FORMAT_R8G8B8A8_UNORM)
           && Up(texDepth.Get(), depth.data(), 4, DXGI_FORMAT_R32_FLOAT)
           && Up(texMv.Get(), mvHalf.data(), 4, DXGI_FORMAT_R16G16_FLOAT);
    if (!up) { Step("upload", "FAIL", "staging/copy setup failed"); return 7; }
    if (!Submit()) { Step("upload", "FAIL", "upload submit timed out"); return 7; }
    Step("upload", "PASS", "color+depth+mv uploaded (%ux%u)", W, H);

    // ---- CreateFeature: DLAA contract (render size == output size), public feature ----
    NVSDK_NGX_DLSS_Create_Params cp{};
    cp.Feature.InWidth = W; cp.Feature.InHeight = H;
    cp.Feature.InTargetWidth = W; cp.Feature.InTargetHeight = H;   // DLAA: no upscale
    cp.Feature.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_DLAA;
    // proven DLSS5-Feeder --test flags; depth is synthetic 0..1 near->far
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                              NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    cp.InEnableOutputSubrects = false;

    DWORD ccode = 0;
    NVSDK_NGX_Result rf = SafeCreateDlss(list.Get(), &feature, params, &cp, &ccode);
    if (ccode) {
        list->Close();  // never execute a list NGX faulted in
        Step("create_feature", "FAIL", "CreateFeature raised SEH 0x%08lx (list discarded)", ccode);
        printf("{\"status\": \"CREATE_FEATURE_SEH\"}\n");
        return 8;
    }
    if (!Submit()) { Step("create_feature", "FAIL", "create submit timed out"); return 8; }
    if (NVSDK_NGX_FAILED(rf) || !feature) {
        Step("create_feature", "FAIL", "CreateFeature(SuperSampling/DLAA) 0x%08x (%s)", (unsigned)rf, NgxName(rf));
        printf("{\"status\": \"CREATE_FEATURE_FAILED\", \"result\": \"0x%08x\"}\n", (unsigned)rf);
        NVSDK_NGX_D3D12_DestroyParameters(params);
        NVSDK_NGX_D3D12_Shutdown1(device.Get());
        return 8;
    }
    Step("create_feature", "PASS", "DLAA %ux%u handle=%p (public feature SuperSampling=1)", W, H, (void*)feature);

    // ---- Evaluate loop: record -> execute -> fence, per frame ----
    int evalOk = 0;
    unsigned firstFail = 0;
    for (int i = 0; i < o.frames; ++i) {
        NVSDK_NGX_D3D12_DLSS_Eval_Params ep{};
        ep.Feature.pInColor = texColor.Get();
        ep.Feature.pInOutput = texOut.Get();
        ep.Feature.InSharpness = 0.0f;
        ep.pInDepth = texDepth.Get();
        ep.pInMotionVectors = texMv.Get();
        ep.InJitterOffsetX = 0.0f;
        ep.InJitterOffsetY = 0.0f;
        // NGX validates these dimensions even when the render size equals the
        // output size (DLAA).  The v2 RTX 5060 run omitted them and every frame
        // returned FAIL_InvalidParameter after CreateFeature had succeeded.
        ep.InRenderSubrectDimensions.Width = W;
        ep.InRenderSubrectDimensions.Height = H;
        ep.InReset = (i == 0) ? 1 : 0;
        ep.InMVScaleX = 1.0f; ep.InMVScaleY = 1.0f;
        ep.InPreExposure = 1.0f; ep.InExposureScale = 1.0f;
        DWORD ecode = 0;
        NVSDK_NGX_Result re = SafeEvaluateDlss(list.Get(), feature, params, &ep, &ecode);
        if (ecode) {
            list->Close();
            Step("evaluate", "FAIL", "frame %d: EvaluateFeature raised SEH 0x%08lx", i, ecode);
            printf("{\"status\": \"EVALUATE_SEH\", \"frame\": %d}\n", i);
            return 9;
        }
        if (!Submit()) { Step("evaluate", "FAIL", "frame %d: submit timed out", i); return 9; }
        if (NVSDK_NGX_FAILED(re)) {
            if (!firstFail) firstFail = (unsigned)re;
            printf("frame %d evaluate: 0x%08x (%s)\n", i, (unsigned)re, NgxName(re));
        } else ++evalOk;
    }
    if (evalOk == o.frames)
        Step("evaluate", "PASS", "%d/%d frames (real GPU command-list recording + execution)", evalOk, o.frames);
    else {
        Step("evaluate", "FAIL", "%d/%d frames (first fail 0x%08x)", evalOk, o.frames, firstFail);
        exitCode = 9;
    }

    // ---- readback + evidence ----
    uint64_t outHash = 0;
    std::vector<uint8_t> outPix;
    if (evalOk > 0) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = texOut.Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        list->ResourceBarrier(1, &b);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{}; UINT64 total = 0;
        D3D12_RESOURCE_DESC outd = texOut->GetDesc();
        device->GetCopyableFootprints(&outd, 0, 1, 0, &fp, nullptr, nullptr, &total);
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
        bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> rb;
        if (SUCCEEDED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb)))) {
            D3D12_TEXTURE_COPY_LOCATION s{}, d{};
            s.pResource = texOut.Get(); s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            d.pResource = rb.Get(); d.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; d.PlacedFootprint = fp;
            list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
            if (Submit()) {
                uint8_t* mapped = nullptr;
                if (SUCCEEDED(rb->Map(0, nullptr, (void**)&mapped))) {
                    outPix.resize((size_t)W * H * 4);
                    for (UINT y = 0; y < H; ++y)
                        memcpy(&outPix[(size_t)y * W * 4], mapped + (size_t)y * fp.Footprint.RowPitch, (size_t)W * 4);
                    rb->Unmap(0, nullptr);
                    outHash = Fnv1a(outPix.data(), outPix.size());
                }
            }
        }
        if (!outPix.empty()) {
            Step("readback", "PASS", "fnv1a=%016llx (%u bytes)", (unsigned long long)outHash, (unsigned)outPix.size());
            char outBin[600], outBmp[600];
            if (o.output && *o.output) snprintf(outBin, sizeof(outBin), "%s", o.output);
            else snprintf(outBin, sizeof(outBin), "results\\%s_output_rgba8.bin", ts);
            WriteRaw(outBin, outPix.data(), outPix.size());
            snprintf(outBmp, sizeof(outBmp), "%.*s.bmp", (int)sizeof(outBmp) - 8, outBin);
            size_t blen = strlen(outBmp);
            if (blen > 4 && !strcmp(outBmp + blen - 8, ".bin.bmp")) strcpy(outBmp + blen - 8, ".bmp");
            WriteBmp(outBmp, outPix.data(), (int)W, (int)H);
            printf("output written: %s + bmp\n", outBin);
        } else {
            Step("readback", "FAIL", "readback mapping failed");
            if (!exitCode) exitCode = 10;
        }
    }

    // ---- teardown: official lifecycle order ----
    if (feature) {
        DWORD relCode = 0;
        SafeReleaseFeature(feature, &relCode);
        Step("release_feature", relCode ? "FAIL" : "PASS", "ReleaseFeature%s",
             relCode ? " (SEH caught)" : "");
    }
    if (params) NVSDK_NGX_D3D12_DestroyParameters(params);
    if (caps) NVSDK_NGX_D3D12_DestroyParameters(caps);
    if (ngxInited) NVSDK_NGX_D3D12_Shutdown1(device.Get());
    Step("shutdown", "PASS", "DestroyParameters + Shutdown1");

    // ---- final JSON ----
    printf("\n{\"status\": \"%s\", \"frames\": %d, \"frames_ok\": %d, \"width\": %u, \"height\": %u, "
           "\"output_fnv1a\": \"%016llx\", \"abi\": \"official_loader\", \"feature\": \"SuperSampling/DLAA\"}\n",
           exitCode == 0 ? "PASS" : "PARTIAL", o.frames, evalOk, W, H, (unsigned long long)outHash);
    if (o.json) {
        FILE* f = fopen(o.json, "w");
        if (f) {
            fprintf(f, "{\n  \"tool\": \"nr_host\", \"round\": 2, \"abi\": \"official_loader\",\n");
            fprintf(f, "  \"width\": %u, \"height\": %u, \"frames\": %d, \"frames_ok\": %d,\n", W, H, o.frames, evalOk);
            fprintf(f, "  \"adapter_vendor\": \"0x%04x\", \"output_fnv1a\": \"%016llx\",\n",
                    usedDesc.VendorId, (unsigned long long)outHash);
            fprintf(f, "  \"steps\": [\n");
            for (size_t i = 0; i < g_steps.size(); ++i) {
                fprintf(f, "    {\"name\": \"%s\", \"status\": \"%s\", \"detail\": \"%s\"}%s\n",
                        g_steps[i].name, g_steps[i].status, g_steps[i].detail,
                        i + 1 < g_steps.size() ? "," : "");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
            printf("json written: %s\n", o.json);
        }
    }
    return exitCode;
}
