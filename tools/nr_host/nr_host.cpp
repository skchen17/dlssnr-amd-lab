// nr_host — Phase 5: standalone Neural Rendering host skeleton (D3D12 x64).
//
// Pipeline (as far as prerequisites allow):
//   deterministic synthetic inputs (Color / Depth / MotionVectors / Exposure)
//   -> NGX init -> feature create (id 18, Neural Rendering) -> evaluate N frames
//   -> readback + SHA-less checksum -> output file.
//
// This machine has NO NVIDIA GPU and NO proprietary DLLs, therefore by design:
//   - we never fabricate a reference; instead a "reference package" (inputs +
//     manifest + run instructions) is emitted so any NVIDIA machine can produce
//     the real reference output;
//   - without DLSS_DLL_PATH / DLSSNR_DLL_PATH the run is marked blocked with a
//     machine-readable status (exit 3), skeleton still delivered.
//
// The NGX declarations below are RECONSTRUCTED from public documentation and the
// DLSS5-Feeder project (MIT). They are NOT the official SDK headers and any
// execution through them is explicitly flagged as such in the output.
//
// Usage:
//   nr_host.exe --frames N --width W --height H [--input <raw color rgba8>]
//               [--output <path>] [--trace] [--json <path>]

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// RECONSTRUCTED NGX API surface (public docs / DLSS5-Feeder). Handle/opaque
// types are pointer-sized tokens; calling convention is cdecl per public usage.
// MARKED RECONSTRUCTED — verify against official NGX SDK before trusting results.

typedef unsigned int NVSDK_NGX_Result;
#define NVSDK_NGX_Result_Success 1u
struct NVSDK_NGX_Parameter;   // opaque (C++ vtable object in the real SDK)
struct NVSDK_NGX_Handle;      // opaque feature handle

typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_Init)(
    unsigned int appId, const wchar_t* appDataPath, ID3D12Device* device,
    NVSDK_NGX_Parameter** outParams);
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_Shutdown)(ID3D12Device* device);
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_CreateFeature)(
    ID3D12GraphicsCommandList* cmd, unsigned int featureId,
    NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** outHandle);
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_EvaluateFeature)(
    ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Handle* handle,
    NVSDK_NGX_Parameter* params, void* progressCallback);
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_ReleaseFeature)(
    ID3D12Device* device, NVSDK_NGX_Handle* handle);

static const unsigned int NGX_FEATURE_ID_NEURAL_RENDERING = 18;  // per DLSS5-Feeder

// ---------------------------------------------------------------------------

struct Options {
    int frames = 1;
    int width = 512;
    int height = 512;
    bool trace = false;
    bool forceLoad = false;   // AMD-side experiment: load the DLLs even without an NVIDIA GPU
    const char* input = nullptr;
    const char* output = nullptr;
    const char* json = nullptr;
};

static uint32_t g_lcg = 0x2545F491u;
static inline uint32_t Lcg() { g_lcg = g_lcg * 1664525u + 1013904223u; return g_lcg; }

// Deterministic synthetic inputs. Layout constants live in the manifest so the
// NVIDIA-side reference run uses byte-identical data.
static void GenColor(std::vector<uint8_t>& out, int w, int h) {
    out.resize((size_t)w * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t* p = &out[((size_t)y * w + x) * 4];
            // smooth gradients + hash noise: structured enough for NR features
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
static void GenMotion(std::vector<int16_t>& out, int w, int h) {
    out.resize((size_t)w * h * 2);
    for (size_t i = 0; i < out.size(); i += 2) {
        out[i] = (int16_t)((int)(Lcg() % 5) - 2);       // ~[-2,2] subpixel
        out[i + 1] = (int16_t)((int)(Lcg() % 5) - 2);
    }
}

static void WriteRaw(const char* path, const void* data, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(data, 1, bytes, f); fclose(f); }
}

static std::string Status(const char* s, const Options& o, const char* detail) {
    char b[512];
    snprintf(b, sizeof(b),
             "{\"status\": \"%s\", \"frames\": %d, \"width\": %d, \"height\": %d, \"detail\": \"%s\"}",
             s, o.frames, o.width, o.height, detail ? detail : "");
    return b;
}

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
    if (o.width < 8 || o.height < 8 || o.frames < 1) {
        printf("invalid arguments\n");
        return 2;
    }

    printf("=== nr_host (dlssnr-amd-lab Phase 5 skeleton) ===\n");
    if (o.trace) {
        printf("trace mode: set these env vars and rerun under the shims:\n");
        printf("  MODULE_TRACE_LOG=<path>  (inject build\\module_trace.dll)\n");
        printf("  NVAPI_TRACE_LOG=<path>   (place build\\nvapi64.dll as the game's nvapi64.dll)\n");
    }

    // ---- deterministic inputs ----
    g_lcg = 0x2545F491u;
    std::vector<uint8_t> color;
    std::vector<float> depth;
    std::vector<int16_t> motion;
    GenColor(color, o.width, o.height);
    GenDepth(depth, o.width, o.height);
    GenMotion(motion, o.width, o.height);
    float exposure = 0.18f;
    printf("inputs: color=%zuB depth=%zuB motion=%zuB exposure=%.3f (deterministic LCG 0x2545F491)\n",
           color.size(), depth.size() * sizeof(float), motion.size() * sizeof(int16_t), exposure);

    if (o.input) {
        FILE* f = fopen(o.input, "rb");
        if (f) {
            size_t n = fread(color.data(), 1, color.size(), f);
            fclose(f);
            printf("loaded external color input: %zu bytes\n", n);
        } else {
            printf("WARNING: cannot open --input %s; using synthetic color\n", o.input);
        }
    }

    // ---- prerequisite checks ----
    const char* dlssPath = getenv("DLSS_DLL_PATH");
    const char* nrPath = getenv("DLSSNR_DLL_PATH");
    bool hasDlss = dlssPath && *dlssPath && GetFileAttributesA(dlssPath) != INVALID_FILE_ATTRIBUTES;
    bool hasNr = nrPath && *nrPath && GetFileAttributesA(nrPath) != INVALID_FILE_ATTRIBUTES;

    // reference package (always produced: deterministic inputs + manifest)
    char ts[32];
    SYSTEMTIME st;
    GetLocalTime(&st);
    snprintf(ts, sizeof(ts), "%04u%02u%02u_%02u%02u%02u", st.wYear, st.wMonth, st.wDay,
             st.wHour, st.wMinute, st.wSecond);
    char pkgDir[512];
    snprintf(pkgDir, sizeof(pkgDir), "results\\%s_reference_package", ts);
    CreateDirectoryA("results", nullptr);
    CreateDirectoryA(pkgDir, nullptr);
    {
        std::string p;
        p = std::string(pkgDir) + "\\color_rgba8.bin"; WriteRaw(p.c_str(), color.data(), color.size());
        p = std::string(pkgDir) + "\\depth_r32f.bin"; WriteRaw(p.c_str(), depth.data(), depth.size() * 4);
        p = std::string(pkgDir) + "\\motion_rg16s.bin"; WriteRaw(p.c_str(), motion.data(), motion.size() * 2);
        p = std::string(pkgDir) + "\\manifest.json";
        FILE* f = fopen(p.c_str(), "w");
        if (f) {
            fprintf(f, "{\n  \"generator\": \"nr_host (dlssnr-amd-lab)\",\n");
            fprintf(f, "  \"width\": %d, \"height\": %d, \"frames\": %d, \"exposure\": %.3f,\n",
                    o.width, o.height, o.frames, exposure);
            fprintf(f, "  \"color\": {\"file\": \"color_rgba8.bin\", \"format\": \"R8G8B8A8_UNORM\"},\n");
            fprintf(f, "  \"depth\": {\"file\": \"depth_r32f.bin\", \"format\": \"R32_FLOAT\"},\n");
            fprintf(f, "  \"motion\": {\"file\": \"motion_rg16s.bin\", \"format\": \"R16G16_SINT (1/4px)\"},\n");
            fprintf(f, "  \"instructions\": \"Run nr_host.exe with these inputs on an RTX (Blackwell for leaked NR build) machine to produce the reference output. Do NOT approximate on AMD/CPU — this package exists so the reference is real.\"\n}\n");
            fclose(f);
        }
        printf("reference package written to %s\\ (copy to an NVIDIA machine)\n", pkgDir);
    }

    // no NVIDIA GPU on this box -> reference run impossible here, never fake it
    bool hasNvidia = false;
    {
        ComPtr<IDXGIFactory7> factory;
        if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
            ComPtr<IDXGIAdapter1> a;
            for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i, a.Reset()) {
                DXGI_ADAPTER_DESC1 d{};
                a->GetDesc1(&d);
                if (d.VendorId == 0x10DE && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { hasNvidia = true; break; }
            }
        }
    }

    std::string status;
    int exitCode = 0;
    if (!hasDlss || !hasNr) {
        status = Status("BLOCKED_MISSING_PREREQUISITE", o,
                        hasDlss ? "nvngx_dlssnr.dll missing (set DLSSNR_DLL_PATH)"
                                : "nvngx_dlss.dll missing (set DLSS_DLL_PATH)");
        exitCode = 3;
    } else if (!hasNvidia && !o.forceLoad) {
        status = Status("BLOCKED_NO_NVIDIA_GPU", o,
                        "DLLs present but no NVIDIA adapter; real NGX execution impossible here. "
                        "Reference package emitted; no fabricated results.");
        exitCode = 4;
    } else {
        // ---- real attempt (NVIDIA machine with legal DLLs, or AMD --force-load experiment) ----
        if (o.trace) {
            // self-inject the module tracer BEFORE any NVIDIA module loads;
            // module_trace.dll lives next to this exe (build\) which is first in search order
            HMODULE mt = LoadLibraryA("module_trace.dll");
            printf("module_trace self-inject: %p (lasterr=%lu)\n", (void*)mt, mt ? 0ul : GetLastError());
        }
        HMODULE hDlss = LoadLibraryA(dlssPath);
        HMODULE hNr = LoadLibraryA(nrPath);   // NR addon must be loadable next to the host
        printf("load: dlss=%p nr=%p (lasterr dlss=%lu nr=%lu)\n", (void*)hDlss, (void*)hNr,
               hDlss ? 0ul : GetLastError(), hNr ? 0ul : GetLastError());
        auto pInit = hDlss ? (PFN_NVSDK_NGX_D3D12_Init)GetProcAddress(hDlss, "NVSDK_NGX_D3D12_Init") : nullptr;
        auto pShutdown = hDlss ? (PFN_NVSDK_NGX_D3D12_Shutdown)GetProcAddress(hDlss, "NVSDK_NGX_D3D12_Shutdown") : nullptr;
        auto pCreate = hDlss ? (PFN_NVSDK_NGX_D3D12_CreateFeature)GetProcAddress(hDlss, "NVSDK_NGX_D3D12_CreateFeature") : nullptr;
        auto pEval = hDlss ? (PFN_NVSDK_NGX_D3D12_EvaluateFeature)GetProcAddress(hDlss, "NVSDK_NGX_D3D12_EvaluateFeature") : nullptr;
        auto pRelease = hDlss ? (PFN_NVSDK_NGX_D3D12_ReleaseFeature)GetProcAddress(hDlss, "NVSDK_NGX_D3D12_ReleaseFeature") : nullptr;
        printf("NGX exports (RECONSTRUCTED signatures): init=%d shutdown=%d create=%d eval=%d release=%d; nr_dll=%s\n",
               !!pInit, !!pShutdown, !!pCreate, !!pEval, !!pRelease, hNr ? "loaded" : "FAILED");

        if (!pInit || !pCreate || !pEval) {
            status = Status("BLOCKED_API_MISMATCH", o,
                            "expected NGX exports missing; signatures need refresh from official SDK");
            exitCode = 5;
        } else {
            ComPtr<IDXGIFactory7> factory;
            CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
            ComPtr<ID3D12Device> device;
            DXGI_ADAPTER_DESC1 usedDesc{};
            ComPtr<IDXGIAdapter1> a;
            if (hasNvidia && !o.forceLoad) {
                for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i, a.Reset()) {
                    DXGI_ADAPTER_DESC1 d{};
                    a->GetDesc1(&d);
                    if (d.VendorId == 0x10DE && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                        D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
                        usedDesc = d;
                        break;
                    }
                }
            } else {
                // AMD-side experiment: build the device on the first hardware adapter (AMD)
                for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i, a.Reset()) {
                    DXGI_ADAPTER_DESC1 d{};
                    a->GetDesc1(&d);
                    if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                        D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
                        usedDesc = d;
                        break;
                    }
                }
            }
            printf("adapter used for NGX init: vendor=0x%04x device=0x%04x (0x1002=AMD, 0x10DE=NVIDIA)\n",
                   usedDesc.VendorId, usedDesc.DeviceId);
            if (!device) {
                status = Status("BLOCKED_DEVICE_CREATE_FAILED", o, "D3D12CreateDevice failed");
                exitCode = 7;
            } else {
                NVSDK_NGX_Parameter* params = nullptr;
                NVSDK_NGX_Result r = pInit(0x4C424C21u, L".", device.Get(), &params);
                printf("NGX init (reconstructed): 0x%08x params=%p\n", r, (void*)params);
                if (r != NVSDK_NGX_Result_Success) {
                    char b[192];
                    snprintf(b, sizeof(b),
                             "NGX init returned 0x%08x on vendor=0x%04x adapter (reconstructed signature)",
                             r, usedDesc.VendorId);
                    status = Status("NGX_INIT_FAILED", o, b);
                    exitCode = 6;
                } else {
                    // create + evaluate loop would go here; full parameter-block construction
                    // requires the official SDK (B-4). Recorded as the next concrete step.
                    status = Status("PARTIAL_NGX_INIT_OK", o,
                                    "init succeeded; create/evaluate need official NGX SDK parameter blocks (B-4)");
                    exitCode = 0;
                    if (pShutdown) pShutdown(device.Get());
                }
            }
        }
    }

    printf("\n%s\n", status.c_str());
    if (o.json) {
        FILE* f = fopen(o.json, "w");
        if (f) { fprintf(f, "%s\n", status.c_str()); fclose(f); }
    }
    if (o.output && exitCode == 0) {
        printf("note: --output %s will receive readback bytes once evaluate is wired\n", o.output);
    }
    return exitCode;
}
