#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = std::uint64_t;
using CUarray = void*;
using CUtexObject = std::uint64_t;
using CUsurfObject = std::uint64_t;

enum CUmemorytype : unsigned int { CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_ARRAY = 3 };
enum CUarray_format : unsigned int { CU_AD_FORMAT_HALF = 16 };
enum CUresourcetype : unsigned int { CU_RESOURCE_TYPE_ARRAY = 0 };
enum CUaddress_mode : unsigned int {
    CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_BORDER = 3,
};
enum CUfilter_mode : unsigned int {
    CU_TR_FILTER_MODE_POINT = 0,
    CU_TR_FILTER_MODE_LINEAR = 1,
};
constexpr unsigned int CU_TRSF_NORMALIZED_COORDINATES = 2;

struct CUDA_ARRAY_DESCRIPTOR {
    std::size_t Width;
    std::size_t Height;
    CUarray_format Format;
    unsigned int NumChannels;
};

struct CUDA_MEMCPY2D {
    std::size_t srcXInBytes, srcY;
    CUmemorytype srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    std::size_t srcPitch;
    std::size_t dstXInBytes, dstY;
    CUmemorytype dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    std::size_t dstPitch;
    std::size_t WidthInBytes, Height;
};

struct CUDA_RESOURCE_DESC {
    CUresourcetype resType;
    unsigned int padding;
    union {
        struct { CUarray hArray; } array;
        int reserved[32];
    } res;
    unsigned int flags;
};

struct CUDA_TEXTURE_DESC {
    CUaddress_mode addressMode[3];
    CUfilter_mode filterMode;
    unsigned int flags;
    unsigned int maxAnisotropy;
    CUfilter_mode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};

struct CUDA_RESOURCE_VIEW_DESC { int reserved[16]; };

constexpr CUresult CUDA_SUCCESS = 0;

using PFN_cuInit = CUresult(__stdcall*)(unsigned int);
using PFN_cuDeviceGetCount = CUresult(__stdcall*)(int*);
using PFN_cuDeviceGet = CUresult(__stdcall*)(CUdevice*, int);
using PFN_cuDeviceGetName = CUresult(__stdcall*)(char*, int, CUdevice);
using PFN_cuCtxCreate_v2 = CUresult(__stdcall*)(CUcontext*, unsigned int, CUdevice);
using PFN_cuCtxDestroy_v2 = CUresult(__stdcall*)(CUcontext);
using PFN_cuModuleLoadData = CUresult(__stdcall*)(CUmodule*, const void*);
using PFN_cuModuleGetFunction = CUresult(__stdcall*)(CUfunction*, CUmodule, const char*);
using PFN_cuModuleUnload = CUresult(__stdcall*)(CUmodule);
using PFN_cuMemAlloc_v2 = CUresult(__stdcall*)(CUdeviceptr*, std::size_t);
using PFN_cuMemFree_v2 = CUresult(__stdcall*)(CUdeviceptr);
using PFN_cuMemsetD32_v2 = CUresult(__stdcall*)(CUdeviceptr, unsigned int,
                                                std::size_t);
using PFN_cuMemcpyDtoH_v2 = CUresult(__stdcall*)(void*, CUdeviceptr, std::size_t);
using PFN_cuMemcpyHtoD_v2 = CUresult(__stdcall*)(CUdeviceptr, const void*, std::size_t);
using PFN_cuArrayCreate_v2 = CUresult(__stdcall*)(CUarray*, const CUDA_ARRAY_DESCRIPTOR*);
using PFN_cuArrayDestroy = CUresult(__stdcall*)(CUarray);
using PFN_cuMemcpy2D_v2 = CUresult(__stdcall*)(const CUDA_MEMCPY2D*);
using PFN_cuTexObjectCreate = CUresult(__stdcall*)(CUtexObject*, const CUDA_RESOURCE_DESC*,
                                                   const CUDA_TEXTURE_DESC*,
                                                   const CUDA_RESOURCE_VIEW_DESC*);
using PFN_cuTexObjectDestroy = CUresult(__stdcall*)(CUtexObject);
using PFN_cuSurfObjectCreate = CUresult(__stdcall*)(CUsurfObject*,
                                                     const CUDA_RESOURCE_DESC*);
using PFN_cuSurfObjectDestroy = CUresult(__stdcall*)(CUsurfObject);
using PFN_cuLaunchKernel = CUresult(__stdcall*)(
    CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
using PFN_cuCtxSynchronize = CUresult(__stdcall*)();
using PFN_cuGetErrorName = CUresult(__stdcall*)(CUresult, const char**);
using PFN_cuGetErrorString = CUresult(__stdcall*)(CUresult, const char**);

struct Api {
    HMODULE library = nullptr;
    PFN_cuInit cuInit = nullptr;
    PFN_cuDeviceGetCount cuDeviceGetCount = nullptr;
    PFN_cuDeviceGet cuDeviceGet = nullptr;
    PFN_cuDeviceGetName cuDeviceGetName = nullptr;
    PFN_cuCtxCreate_v2 cuCtxCreate_v2 = nullptr;
    PFN_cuCtxDestroy_v2 cuCtxDestroy_v2 = nullptr;
    PFN_cuModuleLoadData cuModuleLoadData = nullptr;
    PFN_cuModuleGetFunction cuModuleGetFunction = nullptr;
    PFN_cuModuleUnload cuModuleUnload = nullptr;
    PFN_cuMemAlloc_v2 cuMemAlloc_v2 = nullptr;
    PFN_cuMemFree_v2 cuMemFree_v2 = nullptr;
    PFN_cuMemsetD32_v2 cuMemsetD32_v2 = nullptr;
    PFN_cuMemcpyDtoH_v2 cuMemcpyDtoH_v2 = nullptr;
    PFN_cuMemcpyHtoD_v2 cuMemcpyHtoD_v2 = nullptr;
    PFN_cuArrayCreate_v2 cuArrayCreate_v2 = nullptr;
    PFN_cuArrayDestroy cuArrayDestroy = nullptr;
    PFN_cuMemcpy2D_v2 cuMemcpy2D_v2 = nullptr;
    PFN_cuTexObjectCreate cuTexObjectCreate = nullptr;
    PFN_cuTexObjectDestroy cuTexObjectDestroy = nullptr;
    PFN_cuSurfObjectCreate cuSurfObjectCreate = nullptr;
    PFN_cuSurfObjectDestroy cuSurfObjectDestroy = nullptr;
    PFN_cuLaunchKernel cuLaunchKernel = nullptr;
    PFN_cuCtxSynchronize cuCtxSynchronize = nullptr;
    PFN_cuGetErrorName cuGetErrorName = nullptr;
    PFN_cuGetErrorString cuGetErrorString = nullptr;
};

struct Step {
    std::string name;
    CUresult code = -1;
    double milliseconds = 0.0;
};

static std::string JsonEscape(const std::string& input) {
    std::string output;
    output.reserve(input.size() + 16);
    for (unsigned char c : input) {
        switch (c) {
        case '\\': output += "\\\\"; break;
        case '"': output += "\\\""; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (c < 0x20) {
                char escaped[7] = {};
                std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
                output += escaped;
            } else {
                output += static_cast<char>(c);
            }
        }
    }
    return output;
}

static std::string WideToUtf8(const std::wstring& input) {
    if (input.empty()) return {};
    int needed = WideCharToMultiByte(CP_UTF8, 0, input.data(),
                                     static_cast<int>(input.size()), nullptr, 0,
                                     nullptr, nullptr);
    std::string output(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, input.data(), static_cast<int>(input.size()),
                        output.data(), needed, nullptr, nullptr);
    return output;
}

static std::string LastWin32Error() {
    DWORD error = GetLastError();
    wchar_t* message = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, error, 0, reinterpret_cast<wchar_t*>(&message), 0,
                   nullptr);
    std::wstring text = message ? message : L"unknown Win32 error";
    if (message) LocalFree(message);
    while (!text.empty() && (text.back() == L'\r' || text.back() == L'\n')) {
        text.pop_back();
    }
    return std::to_string(error) + ": " + WideToUtf8(text);
}

template <typename T>
static bool Resolve(HMODULE library, const char* name, T& target,
                    std::string& missing) {
    target = reinterpret_cast<T>(GetProcAddress(library, name));
    if (!target) {
        if (!missing.empty()) missing += ", ";
        missing += name;
        return false;
    }
    return true;
}

static std::string ErrorName(const Api& api, CUresult result) {
    const char* name = nullptr;
    if (api.cuGetErrorName && api.cuGetErrorName(result, &name) == CUDA_SUCCESS &&
        name) {
        return name;
    }
    return result == CUDA_SUCCESS ? "CUDA_SUCCESS" : "CUDA_ERROR_UNKNOWN";
}

static std::string ErrorText(const Api& api, CUresult result) {
    const char* text = nullptr;
    if (api.cuGetErrorString &&
        api.cuGetErrorString(result, &text) == CUDA_SUCCESS && text) {
        return text;
    }
    return result == CUDA_SUCCESS ? "success" : "unknown CUDA driver error";
}

template <typename Fn>
static CUresult TimedStep(std::vector<Step>& steps, const char* name, Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    const CUresult code = fn();
    const auto end = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    steps.push_back({name, code, ms});
    return code;
}

static bool ReadFile(const std::wstring& path, std::vector<char>& data,
                     std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "could not open PTX file";
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0) {
        error = "PTX file is empty";
        return false;
    }
    data.resize(static_cast<size_t>(size) + 1);
    file.read(data.data(), size);
    if (!file) {
        error = "could not read complete PTX file";
        return false;
    }
    data.back() = '\0';
    return true;
}

static bool ReadBytes(const std::wstring& path, std::vector<std::uint8_t>& data,
                      std::string& error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { error = "could not open binary input"; return false; }
    const std::streamoff size = file.tellg();
    if (size < 0) { error = "could not query binary input size"; return false; }
    file.seekg(0);
    data.resize(static_cast<std::size_t>(size));
    if (size && !file.read(reinterpret_cast<char*>(data.data()), size)) {
        error = "could not read complete binary input";
        return false;
    }
    return true;
}

static bool WriteBytes(const std::wstring& path, const std::vector<std::uint8_t>& data,
                       std::string& error) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) { error = "could not create binary output"; return false; }
    if (!data.empty()) file.write(reinterpret_cast<const char*>(data.data()),
                                  static_cast<std::streamsize>(data.size()));
    if (!file) { error = "could not write complete binary output"; return false; }
    return true;
}

static bool WriteJson(const std::wstring& path, const std::string& json,
                      std::string& error) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        error = "could not create JSON output";
        return false;
    }
    file.write(json.data(), static_cast<std::streamsize>(json.size()));
    if (!file) {
        error = "could not write complete JSON output";
        return false;
    }
    return true;
}

static void Usage(const wchar_t* executable) {
    std::fwprintf(stderr,
                  L"usage: %ls --nvcuda <nvcuda.dll> --ptx <module.ptx> "
                  L"--function <entry> --json <result.json> "
                  L"[--clear-words <count>] "
                  L"[--n0-input <rgba16f.raw> --n0-weights <weights.raw> "
                  L"--n0-params <params.raw> --n0-scratch-out <scratch.raw> "
                  L"--n0-output <output.raw> [--n0-grid-x N --n0-grid-y N] "
                  L"[--n0-scratch-extra-bytes N]] "
                  L"[--n1-input <input.raw> --n1-weights <weights.raw> "
                  L"--n1-params <params.raw> --n1-output <output.raw> "
                  L"--n1-sync-out <sync.raw> [--n1-wait-ready] "
                  L"[--n1-no-release] "
                  L"[--n1-output-initial <before.raw>] "
                  L"[--n1-rgba16f-surface-param-offset N] "
                  L"[--n1-rgba16f-surface-initial <rgba16f.raw>] "
                  L"[--n1-rgba16f-zero-texture-param-offset N] "
                  L"[--n1-arena <activation.raw>] "
                  L"[--n1-arena-out <activation-after.raw>] "
                  L"[--n1-arena-param-view <param-offset>:<arena-offset>] "
                  L"[--n1-weight-param-view <param-offset>:<weight-offset>] "
                  L"[--n1-arena-input-offset N] [--n1-arena-input2-offset N] "
                  L"[--n1-arena-output-offset N] [--n1-arena-extra-offset N] "
                  L"[--n1-input2 <input.raw>] "
                  L"[--n1-sync-initial <before.raw>] "
                  L"[--n1-wait-sync-initial <before.raw>] "
                  L"[--n1-extra-output <output.raw>] [--n1-weight-view-offset N] "
                  L"[--n1-extra-output-initial <before.raw>] "
                  L"[--n1-wait-param-offset N] [--n1-release-param-offset N] "
                  L"[--n1-no-weights-param] "
                  L"[--n1-extra-param-offset N] "
                  L"[--n1-input-param-offset N] [--n1-input2-param-offset N] "
                  L"[--n1-output-param-offset N] [--n1-weights-param-offset N] "
                  L"[--n1-grid-x N --n1-grid-y N --n1-grid-z N] "
                  L"[--n1-block-x N --n1-block-y N --n1-block-z N] "
                  L"[--n1-expected-releases N]]\n",
                  executable);
}

int wmain(int argc, wchar_t** argv) {
    std::wstring nvcuda_path;
    std::wstring ptx_path;
    std::wstring json_path;
    std::wstring n0_input_path;
    std::wstring n0_weights_path;
    std::wstring n0_params_path;
    std::wstring n0_scratch_output_path;
    std::wstring n0_output_path;
    std::wstring n1_input_path;
    std::wstring n1_input2_path;
    std::wstring n1_arena_path;
    std::wstring n1_arena_output_path;
    std::wstring n1_weights_path;
    std::wstring n1_params_path;
    std::wstring n1_output_path;
    std::wstring n1_sync_output_path;
    std::wstring n1_extra_output_path;
    std::wstring n1_output_initial_path;
    std::wstring n1_rgba16f_surface_initial_path;
    std::wstring n1_sync_initial_path;
    std::wstring n1_wait_sync_initial_path;
    std::wstring n1_extra_output_initial_path;
    std::string function_name;
    std::uint32_t clear_words = 0;
    std::uint32_t n0_grid_x = 1;
    std::uint32_t n0_grid_y = 1;
    std::uint32_t n0_scratch_extra_bytes = 0;
    std::uint32_t n1_grid_x = 40;
    std::uint32_t n1_grid_y = 24;
    std::uint32_t n1_grid_z = 1;
    std::uint32_t n1_block_x = 32;
    std::uint32_t n1_block_y = 1;
    std::uint32_t n1_block_z = 1;
    std::uint32_t n1_weight_view_offset = 0x5600;
    std::uint32_t n1_arena_input_offset = 0;
    std::uint32_t n1_arena_input2_offset = 0;
    std::uint32_t n1_arena_output_offset = 0;
    std::uint32_t n1_arena_extra_offset = 0;
    std::uint32_t n1_input_param_offset = 0;
    std::uint32_t n1_input2_param_offset = 8;
    std::uint32_t n1_output_param_offset = 8;
    std::uint32_t n1_weights_param_offset = 16;
    std::uint32_t n1_wait_param_offset = 40;
    std::uint32_t n1_release_param_offset = 56;
    std::uint32_t n1_extra_param_offset = 64;
    std::uint32_t n1_expected_releases = 40 * 24;
    std::uint32_t n1_rgba16f_surface_param_offset = 0;
    std::uint32_t n1_rgba16f_zero_texture_param_offset = 0;
    bool n1_rgba16f_surface = false;
    bool n1_rgba16f_zero_texture = false;
    bool n1_wait_ready = false;
    bool n1_patch_release = true;
    bool n1_patch_weights = true;
    bool n1_arena_input2_set = false;
    bool n1_arena_extra_set = false;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> n1_arena_param_views;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> n1_weight_param_views;

    for (int i = 1; i < argc; ++i) {
        auto take_wide = [&](std::wstring& target) -> bool {
            if (i + 1 >= argc) return false;
            target = argv[++i];
            return true;
        };
        if (std::wcscmp(argv[i], L"--nvcuda") == 0) {
            if (!take_wide(nvcuda_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--ptx") == 0) {
            if (!take_wide(ptx_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--json") == 0) {
            if (!take_wide(json_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--function") == 0) {
            std::wstring value;
            if (!take_wide(value)) { Usage(argv[0]); return 2; }
            function_name = WideToUtf8(value);
        } else if (std::wcscmp(argv[i], L"--clear-words") == 0) {
            std::wstring value;
            if (!take_wide(value)) { Usage(argv[0]); return 2; }
            wchar_t* end = nullptr;
            unsigned long parsed = std::wcstoul(value.c_str(), &end, 10);
            if (!end || *end != L'\0' || parsed == 0 || parsed > UINT32_MAX) {
                std::fwprintf(stderr, L"invalid --clear-words value: %ls\n",
                              value.c_str());
                return 2;
            }
            clear_words = static_cast<std::uint32_t>(parsed);
        } else if (std::wcscmp(argv[i], L"--n0-input") == 0) {
            if (!take_wide(n0_input_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n0-weights") == 0) {
            if (!take_wide(n0_weights_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n0-params") == 0) {
            if (!take_wide(n0_params_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n0-scratch-out") == 0) {
            if (!take_wide(n0_scratch_output_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n0-output") == 0) {
            if (!take_wide(n0_output_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-input") == 0) {
            if (!take_wide(n1_input_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-input2") == 0) {
            if (!take_wide(n1_input2_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-arena") == 0) {
            if (!take_wide(n1_arena_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-arena-out") == 0) {
            if (!take_wide(n1_arena_output_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-arena-param-view") == 0) {
            std::wstring value;
            if (!take_wide(value)) { Usage(argv[0]); return 2; }
            const std::size_t separator = value.find(L':');
            if (separator == std::wstring::npos) return 2;
            const std::wstring param_text = value.substr(0, separator);
            const std::wstring arena_text = value.substr(separator + 1);
            wchar_t* param_end = nullptr;
            wchar_t* arena_end = nullptr;
            const unsigned long param_offset =
                std::wcstoul(param_text.c_str(), &param_end, 0);
            const unsigned long arena_offset =
                std::wcstoul(arena_text.c_str(), &arena_end, 0);
            if (!param_end || *param_end != L'\0' ||
                !arena_end || *arena_end != L'\0' ||
                param_offset > UINT32_MAX || arena_offset > UINT32_MAX)
                return 2;
            n1_arena_param_views.emplace_back(
                static_cast<std::uint32_t>(param_offset),
                static_cast<std::uint32_t>(arena_offset));
        } else if (std::wcscmp(argv[i], L"--n1-weight-param-view") == 0) {
            std::wstring value;
            if (!take_wide(value)) { Usage(argv[0]); return 2; }
            const std::size_t separator = value.find(L':');
            if (separator == std::wstring::npos) return 2;
            const std::wstring param_text = value.substr(0, separator);
            const std::wstring weight_text = value.substr(separator + 1);
            wchar_t* param_end = nullptr;
            wchar_t* weight_end = nullptr;
            const unsigned long param_offset =
                std::wcstoul(param_text.c_str(), &param_end, 0);
            const unsigned long weight_offset =
                std::wcstoul(weight_text.c_str(), &weight_end, 0);
            if (!param_end || *param_end != L'\0' ||
                !weight_end || *weight_end != L'\0' ||
                param_offset > UINT32_MAX || weight_offset > UINT32_MAX)
                return 2;
            n1_weight_param_views.emplace_back(
                static_cast<std::uint32_t>(param_offset),
                static_cast<std::uint32_t>(weight_offset));
        } else if (std::wcscmp(argv[i], L"--n1-weights") == 0) {
            if (!take_wide(n1_weights_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-params") == 0) {
            if (!take_wide(n1_params_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-output") == 0) {
            if (!take_wide(n1_output_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-sync-out") == 0) {
            if (!take_wide(n1_sync_output_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-extra-output") == 0) {
            if (!take_wide(n1_extra_output_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-output-initial") == 0) {
            if (!take_wide(n1_output_initial_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-rgba16f-surface-initial") == 0) {
            if (!take_wide(n1_rgba16f_surface_initial_path)) {
                Usage(argv[0]); return 2;
            }
        } else if (std::wcscmp(argv[i], L"--n1-sync-initial") == 0) {
            if (!take_wide(n1_sync_initial_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-wait-sync-initial") == 0) {
            if (!take_wide(n1_wait_sync_initial_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-extra-output-initial") == 0) {
            if (!take_wide(n1_extra_output_initial_path)) { Usage(argv[0]); return 2; }
        } else if (std::wcscmp(argv[i], L"--n1-wait-ready") == 0) {
            n1_wait_ready = true;
        } else if (std::wcscmp(argv[i], L"--n1-no-release") == 0) {
            n1_patch_release = false;
        } else if (std::wcscmp(argv[i], L"--n1-no-weights-param") == 0) {
            n1_patch_weights = false;
        } else if (std::wcscmp(argv[i], L"--n1-weight-view-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-arena-input-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-arena-input2-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-arena-output-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-arena-extra-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-input-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-input2-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-output-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-weights-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-wait-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-release-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-extra-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-rgba16f-surface-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-rgba16f-zero-texture-param-offset") == 0 ||
                   std::wcscmp(argv[i], L"--n1-grid-x") == 0 ||
                   std::wcscmp(argv[i], L"--n1-grid-y") == 0 ||
                   std::wcscmp(argv[i], L"--n1-grid-z") == 0 ||
                   std::wcscmp(argv[i], L"--n1-block-x") == 0 ||
                   std::wcscmp(argv[i], L"--n1-block-y") == 0 ||
                   std::wcscmp(argv[i], L"--n1-block-z") == 0 ||
                   std::wcscmp(argv[i], L"--n1-expected-releases") == 0) {
            const bool is_weight = std::wcscmp(argv[i], L"--n1-weight-view-offset") == 0;
            const bool is_arena_input = std::wcscmp(argv[i], L"--n1-arena-input-offset") == 0;
            const bool is_arena_input2 = std::wcscmp(argv[i], L"--n1-arena-input2-offset") == 0;
            const bool is_arena_output = std::wcscmp(argv[i], L"--n1-arena-output-offset") == 0;
            const bool is_arena_extra = std::wcscmp(argv[i], L"--n1-arena-extra-offset") == 0;
            const bool is_input_param = std::wcscmp(argv[i], L"--n1-input-param-offset") == 0;
            const bool is_input2_param = std::wcscmp(argv[i], L"--n1-input2-param-offset") == 0;
            const bool is_output_param = std::wcscmp(argv[i], L"--n1-output-param-offset") == 0;
            const bool is_weights_param = std::wcscmp(argv[i], L"--n1-weights-param-offset") == 0;
            const bool is_wait_offset = std::wcscmp(argv[i], L"--n1-wait-param-offset") == 0;
            const bool is_release_offset = std::wcscmp(argv[i], L"--n1-release-param-offset") == 0;
            const bool is_extra_offset = std::wcscmp(argv[i], L"--n1-extra-param-offset") == 0;
            const bool is_rgba_surface =
                std::wcscmp(argv[i], L"--n1-rgba16f-surface-param-offset") == 0;
            const bool is_rgba_texture =
                std::wcscmp(argv[i], L"--n1-rgba16f-zero-texture-param-offset") == 0;
            const bool is_grid_x = std::wcscmp(argv[i], L"--n1-grid-x") == 0;
            const bool is_grid_y = std::wcscmp(argv[i], L"--n1-grid-y") == 0;
            const bool is_grid_z = std::wcscmp(argv[i], L"--n1-grid-z") == 0;
            const bool is_block_x = std::wcscmp(argv[i], L"--n1-block-x") == 0;
            const bool is_block_y = std::wcscmp(argv[i], L"--n1-block-y") == 0;
            const bool is_block_z = std::wcscmp(argv[i], L"--n1-block-z") == 0;
            std::wstring value;
            if (!take_wide(value)) { Usage(argv[0]); return 2; }
            wchar_t* end = nullptr;
            const unsigned long parsed = std::wcstoul(value.c_str(), &end, 0);
            if (!end || *end != L'\0' || parsed > UINT32_MAX ||
                ((is_grid_x || is_grid_y || is_grid_z || is_block_x ||
                  is_block_y || is_block_z) &&
                 (parsed == 0 || parsed > 1024))) return 2;
            if (is_weight) n1_weight_view_offset = static_cast<std::uint32_t>(parsed);
            else if (is_arena_input) n1_arena_input_offset = static_cast<std::uint32_t>(parsed);
            else if (is_arena_input2) {
                n1_arena_input2_offset = static_cast<std::uint32_t>(parsed);
                n1_arena_input2_set = true;
            }
            else if (is_arena_output) n1_arena_output_offset = static_cast<std::uint32_t>(parsed);
            else if (is_arena_extra) {
                n1_arena_extra_offset = static_cast<std::uint32_t>(parsed);
                n1_arena_extra_set = true;
            }
            else if (is_input_param) n1_input_param_offset = static_cast<std::uint32_t>(parsed);
            else if (is_input2_param) n1_input2_param_offset = static_cast<std::uint32_t>(parsed);
            else if (is_output_param) n1_output_param_offset = static_cast<std::uint32_t>(parsed);
            else if (is_weights_param) n1_weights_param_offset = static_cast<std::uint32_t>(parsed);
            else if (is_wait_offset) n1_wait_param_offset = static_cast<std::uint32_t>(parsed);
            else if (is_release_offset) n1_release_param_offset = static_cast<std::uint32_t>(parsed);
            else if (is_extra_offset) n1_extra_param_offset = static_cast<std::uint32_t>(parsed);
            else if (is_rgba_surface) {
                n1_rgba16f_surface_param_offset = static_cast<std::uint32_t>(parsed);
                n1_rgba16f_surface = true;
            }
            else if (is_rgba_texture) {
                n1_rgba16f_zero_texture_param_offset = static_cast<std::uint32_t>(parsed);
                n1_rgba16f_zero_texture = true;
            }
            else if (is_grid_x) n1_grid_x = static_cast<std::uint32_t>(parsed);
            else if (is_grid_y) n1_grid_y = static_cast<std::uint32_t>(parsed);
            else if (is_grid_z) n1_grid_z = static_cast<std::uint32_t>(parsed);
            else if (is_block_x) n1_block_x = static_cast<std::uint32_t>(parsed);
            else if (is_block_y) n1_block_y = static_cast<std::uint32_t>(parsed);
            else if (is_block_z) n1_block_z = static_cast<std::uint32_t>(parsed);
            else n1_expected_releases = static_cast<std::uint32_t>(parsed);
        } else if (std::wcscmp(argv[i], L"--n0-scratch-extra-bytes") == 0) {
            std::wstring value;
            if (!take_wide(value)) { Usage(argv[0]); return 2; }
            wchar_t* end = nullptr;
            const unsigned long parsed = std::wcstoul(value.c_str(), &end, 10);
            if (!end || *end != L'\0' || parsed > 64ul * 1024ul * 1024ul ||
                (parsed & 3ul) != 0) return 2;
            n0_scratch_extra_bytes = static_cast<std::uint32_t>(parsed);
        } else if (std::wcscmp(argv[i], L"--n0-grid-x") == 0 ||
                   std::wcscmp(argv[i], L"--n0-grid-y") == 0) {
            const bool is_x = std::wcscmp(argv[i], L"--n0-grid-x") == 0;
            std::wstring value;
            if (!take_wide(value)) { Usage(argv[0]); return 2; }
            wchar_t* end = nullptr;
            const unsigned long parsed = std::wcstoul(value.c_str(), &end, 10);
            if (!end || *end != L'\0' || parsed == 0 || parsed > 80) return 2;
            (is_x ? n0_grid_x : n0_grid_y) = static_cast<std::uint32_t>(parsed);
        } else {
            Usage(argv[0]);
            return 2;
        }
    }
    if (nvcuda_path.empty() || ptx_path.empty() || json_path.empty() ||
        function_name.empty()) {
        Usage(argv[0]);
        return 2;
    }
    const bool n0_mode = !n0_input_path.empty();
    const bool n1_mode = !n1_input_path.empty() || !n1_arena_path.empty();
    if (n0_mode && (n0_weights_path.empty() || n0_params_path.empty() ||
                    n0_scratch_output_path.empty() || n0_output_path.empty() ||
                    clear_words != 0 || n0_grid_x > 80 || n0_grid_y > 48)) {
        Usage(argv[0]);
        return 2;
    }
    if (n1_mode && (n1_weights_path.empty() || n1_params_path.empty() ||
                    n1_output_path.empty() || n1_sync_output_path.empty() ||
                    clear_words != 0 || n0_mode ||
                    n1_rgba16f_surface != n1_rgba16f_zero_texture ||
                    (!n1_arena_path.empty() && !n1_input_path.empty()) ||
                    (n1_arena_path.empty() && n1_input_path.empty()))) {
        Usage(argv[0]);
        return 2;
    }

    std::vector<char> ptx;
    std::string fatal_error;
    if (!ReadFile(ptx_path, ptx, fatal_error)) {
        std::fprintf(stderr, "%s: %s\n", fatal_error.c_str(),
                     WideToUtf8(ptx_path).c_str());
        return 3;
    }

    Api api;
    api.library = LoadLibraryExW(nvcuda_path.c_str(), nullptr,
                                 LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                     LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!api.library) {
        std::fprintf(stderr, "LoadLibraryExW failed for %s: %s\n",
                     WideToUtf8(nvcuda_path).c_str(), LastWin32Error().c_str());
        return 4;
    }

    std::string missing;
    Resolve(api.library, "cuInit", api.cuInit, missing);
    Resolve(api.library, "cuDeviceGetCount", api.cuDeviceGetCount, missing);
    Resolve(api.library, "cuDeviceGet", api.cuDeviceGet, missing);
    Resolve(api.library, "cuDeviceGetName", api.cuDeviceGetName, missing);
    Resolve(api.library, "cuCtxCreate_v2", api.cuCtxCreate_v2, missing);
    Resolve(api.library, "cuCtxDestroy_v2", api.cuCtxDestroy_v2, missing);
    Resolve(api.library, "cuModuleLoadData", api.cuModuleLoadData, missing);
    Resolve(api.library, "cuModuleGetFunction", api.cuModuleGetFunction, missing);
    Resolve(api.library, "cuModuleUnload", api.cuModuleUnload, missing);
    Resolve(api.library, "cuMemAlloc_v2", api.cuMemAlloc_v2, missing);
    Resolve(api.library, "cuMemFree_v2", api.cuMemFree_v2, missing);
    Resolve(api.library, "cuMemsetD32_v2", api.cuMemsetD32_v2, missing);
    Resolve(api.library, "cuMemcpyDtoH_v2", api.cuMemcpyDtoH_v2, missing);
    Resolve(api.library, "cuMemcpyHtoD_v2", api.cuMemcpyHtoD_v2, missing);
    Resolve(api.library, "cuArrayCreate_v2", api.cuArrayCreate_v2, missing);
    Resolve(api.library, "cuArrayDestroy", api.cuArrayDestroy, missing);
    Resolve(api.library, "cuMemcpy2D_v2", api.cuMemcpy2D_v2, missing);
    Resolve(api.library, "cuTexObjectCreate", api.cuTexObjectCreate, missing);
    Resolve(api.library, "cuTexObjectDestroy", api.cuTexObjectDestroy, missing);
    if (n1_rgba16f_surface) {
        Resolve(api.library, "cuSurfObjectCreate", api.cuSurfObjectCreate, missing);
        Resolve(api.library, "cuSurfObjectDestroy", api.cuSurfObjectDestroy, missing);
    }
    Resolve(api.library, "cuLaunchKernel", api.cuLaunchKernel, missing);
    Resolve(api.library, "cuCtxSynchronize", api.cuCtxSynchronize, missing);
    Resolve(api.library, "cuGetErrorName", api.cuGetErrorName, missing);
    Resolve(api.library, "cuGetErrorString", api.cuGetErrorString, missing);
    if (!missing.empty()) {
        std::fprintf(stderr, "missing nvcuda exports: %s\n", missing.c_str());
        FreeLibrary(api.library);
        return 5;
    }

    std::vector<Step> steps;
    int device_count = 0;
    CUdevice device = 0;
    char device_name[256] = {};
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUdeviceptr clear_buffer = 0;
    CUdeviceptr n0_scratch = 0;
    CUdeviceptr n0_weights = 0;
    CUdeviceptr n0_output = 0;
    CUarray n0_array = nullptr;
    CUtexObject n0_texture = 0;
    CUarray n1_surface_array = nullptr;
    CUsurfObject n1_surface = 0;
    CUarray n1_texture_array = nullptr;
    CUtexObject n1_texture = 0;
    CUdeviceptr n1_input = 0;
    CUdeviceptr n1_input2 = 0;
    CUdeviceptr n1_arena = 0;
    CUdeviceptr n1_weights = 0;
    CUdeviceptr n1_output = 0;
    CUdeviceptr n1_sync = 0;
    CUdeviceptr n1_wait_sync = 0;
    CUdeviceptr n1_extra_output = 0;
    std::uint64_t clear_mismatches = 0;
    std::uint64_t n0_scratch_nonzero = 0;
    std::uint64_t n0_output_nonzero = 0;
    std::uint64_t n1_output_nonzero = 0;
    std::uint64_t n1_sync_nonzero = 0;
    std::uint64_t n1_sync_zero_words = 0;
    std::uint64_t n1_extra_output_nonzero = 0;
    std::uint64_t n1_weights_input_bytes = 0;
    bool kernel_launched = false;
    bool execution_verified = false;

    CUresult result = TimedStep(steps, "cuInit", [&] { return api.cuInit(0); });
    bool initialized = result == CUDA_SUCCESS;
    if (initialized) {
        result = TimedStep(steps, "cuDeviceGetCount",
                           [&] { return api.cuDeviceGetCount(&device_count); });
    }
    bool have_device = result == CUDA_SUCCESS && device_count > 0;
    if (have_device) {
        result = TimedStep(steps, "cuDeviceGet",
                           [&] { return api.cuDeviceGet(&device, 0); });
        have_device = result == CUDA_SUCCESS;
    }
    if (have_device) {
        result = TimedStep(steps, "cuDeviceGetName", [&] {
            return api.cuDeviceGetName(device_name, sizeof(device_name), device);
        });
        have_device = result == CUDA_SUCCESS;
    }
    bool have_context = false;
    if (have_device) {
        result = TimedStep(steps, "cuCtxCreate_v2",
                           [&] { return api.cuCtxCreate_v2(&context, 0, device); });
        have_context = result == CUDA_SUCCESS && context;
    }
    bool module_loaded = false;
    if (have_context) {
        result = TimedStep(steps, "cuModuleLoadData",
                           [&] { return api.cuModuleLoadData(&module, ptx.data()); });
        module_loaded = result == CUDA_SUCCESS && module;
    }
    bool function_resolved = false;
    if (module_loaded) {
        result = TimedStep(steps, "cuModuleGetFunction", [&] {
            return api.cuModuleGetFunction(&function, module,
                                           function_name.c_str());
        });
        function_resolved = result == CUDA_SUCCESS && function;
    }

    if (function_resolved && clear_words > 0) {
        const std::size_t clear_bytes =
            static_cast<std::size_t>(clear_words) * sizeof(std::uint32_t);
        result = TimedStep(steps, "cuMemAlloc_v2", [&] {
            return api.cuMemAlloc_v2(&clear_buffer, clear_bytes);
        });
        if (result == CUDA_SUCCESS && clear_buffer) {
            result = TimedStep(steps, "cuMemsetD32_v2", [&] {
                return api.cuMemsetD32_v2(clear_buffer, 0u, clear_words);
            });
        }
        if (result == CUDA_SUCCESS && clear_buffer) {
            struct alignas(8) ClearParams {
                std::uint64_t pointer;
                std::uint32_t count;
                std::uint32_t ignored;
            } params{clear_buffer, clear_words, 0x21au};
            static_assert(sizeof(ClearParams) == 16);
            void* kernel_params[] = {&params};
            const unsigned int block_x = 256;
            const unsigned int grid_x = (clear_words + block_x - 1) / block_x;
            result = TimedStep(steps, "cuLaunchKernel", [&] {
                return api.cuLaunchKernel(function, grid_x, 1, 1, block_x, 1, 1,
                                          0, nullptr, kernel_params, nullptr);
            });
            kernel_launched = result == CUDA_SUCCESS;
        }
        if (kernel_launched) {
            result = TimedStep(steps, "cuCtxSynchronize",
                               [&] { return api.cuCtxSynchronize(); });
        }
        if (result == CUDA_SUCCESS && clear_buffer) {
            std::vector<std::uint32_t> readback(clear_words);
            result = TimedStep(steps, "cuMemcpyDtoH_v2", [&] {
                return api.cuMemcpyDtoH_v2(readback.data(), clear_buffer,
                                           clear_bytes);
            });
            if (result == CUDA_SUCCESS) {
                for (std::uint32_t value : readback) {
                    if (value != UINT32_MAX) ++clear_mismatches;
                }
                execution_verified = clear_mismatches == 0;
            }
        }
    }

    if (function_resolved && n0_mode) {
        constexpr std::size_t input_bytes = 640ull * 360ull * 4ull * 2ull;
        constexpr std::size_t base_scratch_bytes = 384ull * 640ull * 32ull;
        const std::size_t scratch_bytes =
            base_scratch_bytes + static_cast<std::size_t>(n0_scratch_extra_bytes);
        constexpr std::size_t output_bytes = 192ull * 320ull * 32ull;
        std::vector<std::uint8_t> input, weights, params;
        const bool files_ok = ReadBytes(n0_input_path, input, fatal_error) &&
                              ReadBytes(n0_weights_path, weights, fatal_error) &&
                              ReadBytes(n0_params_path, params, fatal_error) &&
                              input.size() == input_bytes && weights.size() == 65536 &&
                              params.size() == 264;
        if (!files_ok) {
            std::fprintf(stderr, "invalid N0 input files: %s\n", fatal_error.c_str());
            result = 1;
        } else {
            result = TimedStep(steps, "cuMemAlloc_n0_scratch", [&] {
                return api.cuMemAlloc_v2(&n0_scratch, scratch_bytes);
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemAlloc_n0_weights", [&] {
                return api.cuMemAlloc_v2(&n0_weights, weights.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemAlloc_n0_output", [&] {
                return api.cuMemAlloc_v2(&n0_output, output_bytes);
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemcpyHtoD_n0_weights", [&] {
                return api.cuMemcpyHtoD_v2(n0_weights, weights.data(), weights.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemset_n0_scratch", [&] {
                return api.cuMemsetD32_v2(n0_scratch, 0, scratch_bytes / 4);
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemset_n0_output", [&] {
                return api.cuMemsetD32_v2(n0_output, 0, output_bytes / 4);
            });
        }
        CUDA_ARRAY_DESCRIPTOR array_desc{640, 360, CU_AD_FORMAT_HALF, 4};
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuArrayCreate_n0_input", [&] {
                return api.cuArrayCreate_v2(&n0_array, &array_desc);
            });
        }
        CUDA_MEMCPY2D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = input.data();
        copy.srcPitch = 640 * 4 * 2;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = n0_array;
        copy.WidthInBytes = 640 * 4 * 2;
        copy.Height = 360;
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemcpy2D_n0_input", [&] {
                return api.cuMemcpy2D_v2(&copy);
            });
        }
        CUDA_RESOURCE_DESC resource{};
        resource.resType = CU_RESOURCE_TYPE_ARRAY;
        resource.res.array.hArray = n0_array;
        CUDA_TEXTURE_DESC texture{};
        texture.addressMode[0] = texture.addressMode[1] = texture.addressMode[2] =
            CU_TR_ADDRESS_MODE_CLAMP;
        texture.filterMode = CU_TR_FILTER_MODE_LINEAR;
        texture.flags = CU_TRSF_NORMALIZED_COORDINATES;
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuTexObjectCreate_n0_input", [&] {
                return api.cuTexObjectCreate(&n0_texture, &resource, &texture, nullptr);
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            const std::uint64_t zero = 0;
            for (std::size_t offset : {8ull, 16ull, 24ull, 32ull})
                std::memcpy(params.data() + offset, &zero, sizeof(zero));
            std::memcpy(params.data(), &n0_texture, sizeof(n0_texture));
            std::memcpy(params.data() + 216, &n0_scratch, sizeof(n0_scratch));
            std::memcpy(params.data() + 224, &n0_weights, sizeof(n0_weights));
            std::memcpy(params.data() + 248, &n0_output, sizeof(n0_output));
            void* kernel_params[] = {params.data()};
            result = TimedStep(steps, "cuLaunchKernel_n0", [&] {
                return api.cuLaunchKernel(function, n0_grid_x, n0_grid_y, 1,
                                          32, 1, 1, 0, nullptr,
                                          kernel_params, nullptr);
            });
            kernel_launched = result == CUDA_SUCCESS;
        }
        if (kernel_launched) {
            result = TimedStep(steps, "cuCtxSynchronize_n0", [&] {
                return api.cuCtxSynchronize();
            });
        }
        std::vector<std::uint8_t> scratch_readback(scratch_bytes);
        std::vector<std::uint8_t> output_readback(output_bytes);
        if (result == CUDA_SUCCESS && kernel_launched) {
            result = TimedStep(steps, "cuMemcpyDtoH_n0_scratch", [&] {
                return api.cuMemcpyDtoH_v2(scratch_readback.data(), n0_scratch,
                                           scratch_readback.size());
            });
        }
        if (result == CUDA_SUCCESS && kernel_launched) {
            result = TimedStep(steps, "cuMemcpyDtoH_n0_output", [&] {
                return api.cuMemcpyDtoH_v2(output_readback.data(), n0_output,
                                           output_readback.size());
            });
        }
        if (result == CUDA_SUCCESS && kernel_launched) {
            for (std::uint8_t value : scratch_readback) n0_scratch_nonzero += value != 0;
            for (std::uint8_t value : output_readback) n0_output_nonzero += value != 0;
            execution_verified = n0_scratch_nonzero > 0 && n0_output_nonzero > 0;
            if (!WriteBytes(n0_scratch_output_path, scratch_readback, fatal_error) ||
                !WriteBytes(n0_output_path, output_readback, fatal_error)) {
                std::fprintf(stderr, "N0 output write failed: %s\n", fatal_error.c_str());
                execution_verified = false;
            }
        }
    }

    if (function_resolved && n1_mode) {
        constexpr std::size_t tensor_bytes = 192ull * 320ull * 32ull;
        constexpr std::size_t minimum_weights_bytes = 65536;
        constexpr std::size_t sync_bytes = 27648ull * sizeof(std::uint32_t);
        std::vector<std::uint8_t> input, input2, arena, weights, params;
        std::vector<std::uint8_t> output_initial, surface_initial, sync_initial;
        std::vector<std::uint8_t> wait_sync_initial, extra_output_initial;
        const bool arena_mode = !n1_arena_path.empty();
        bool files_ok = (arena_mode
                                  ? ReadBytes(n1_arena_path, arena, fatal_error)
                                  : ReadBytes(n1_input_path, input, fatal_error)) &&
                              (arena_mode || n1_input2_path.empty() ||
                               (ReadBytes(n1_input2_path, input2, fatal_error) &&
                                input2.size() == tensor_bytes)) &&
                              ReadBytes(n1_weights_path, weights, fatal_error) &&
                              ReadBytes(n1_params_path, params, fatal_error) &&
                              (n1_output_initial_path.empty() ||
                               (ReadBytes(n1_output_initial_path, output_initial, fatal_error) &&
                                output_initial.size() == tensor_bytes)) &&
                              (n1_rgba16f_surface_initial_path.empty() ||
                               (ReadBytes(n1_rgba16f_surface_initial_path,
                                          surface_initial, fatal_error) &&
                                surface_initial.size() == 640ull * 360ull * 8ull)) &&
                              (n1_sync_initial_path.empty() ||
                               (ReadBytes(n1_sync_initial_path, sync_initial, fatal_error) &&
                                sync_initial.size() == sync_bytes)) &&
                              (n1_wait_sync_initial_path.empty() ||
                               (ReadBytes(n1_wait_sync_initial_path, wait_sync_initial, fatal_error) &&
                                wait_sync_initial.size() == sync_bytes)) &&
                              (n1_extra_output_initial_path.empty() ||
                               (ReadBytes(n1_extra_output_initial_path,
                                          extra_output_initial, fatal_error) &&
                                extra_output_initial.size() == tensor_bytes)) &&
                              (arena_mode
                                  ? (n1_arena_input_offset < arena.size() &&
                                     n1_arena_output_offset < arena.size() &&
                                     (!n1_arena_input2_set ||
                                      n1_arena_input2_offset < arena.size()) &&
                                     (!n1_arena_extra_set ||
                                      n1_arena_extra_offset < arena.size()))
                                  : input.size() == tensor_bytes) &&
                              weights.size() >= minimum_weights_bytes &&
                              n1_input_param_offset + sizeof(CUdeviceptr) <= params.size() &&
                              (n1_input2_path.empty() ||
                               n1_input2_param_offset + sizeof(CUdeviceptr) <= params.size()) &&
                              n1_output_param_offset + sizeof(CUdeviceptr) <= params.size() &&
                              (!n1_rgba16f_surface ||
                               (n1_rgba16f_surface_param_offset + sizeof(CUsurfObject) <= params.size() &&
                                n1_rgba16f_zero_texture_param_offset + sizeof(CUtexObject) <= params.size())) &&
                              (!n1_patch_weights ||
                               n1_weights_param_offset + sizeof(CUdeviceptr) <= params.size()) &&
                              (!n1_patch_release ||
                               n1_release_param_offset + sizeof(CUdeviceptr) <= params.size()) &&
                              (!n1_wait_ready || n1_wait_param_offset + sizeof(CUdeviceptr) <= params.size()) &&
                              (n1_extra_output_path.empty() ||
                               n1_extra_param_offset + sizeof(CUdeviceptr) <= params.size()) &&
                              (n1_arena_output_path.empty() || arena_mode);
        for (const auto& view : n1_arena_param_views) {
            files_ok &= arena_mode &&
                        view.first + sizeof(CUdeviceptr) <= params.size() &&
                        view.second < arena.size();
        }
        for (const auto& view : n1_weight_param_views) {
            files_ok &= view.first + sizeof(CUdeviceptr) <= params.size() &&
                        view.second < weights.size();
        }
        if (!files_ok) {
            std::fprintf(stderr, "invalid N1 input files: %s\n", fatal_error.c_str());
            result = 1;
        } else {
            n1_weights_input_bytes = weights.size();
            if (arena_mode) {
                result = TimedStep(steps, "cuMemAlloc_n1_arena", [&] {
                    return api.cuMemAlloc_v2(&n1_arena, arena.size());
                });
                if (result == CUDA_SUCCESS) {
                    n1_input = n1_arena + n1_arena_input_offset;
                    n1_output = n1_arena + n1_arena_output_offset;
                    if (n1_arena_input2_set)
                        n1_input2 = n1_arena + n1_arena_input2_offset;
                    if (n1_arena_extra_set)
                        n1_extra_output = n1_arena + n1_arena_extra_offset;
                }
            } else {
                result = TimedStep(steps, "cuMemAlloc_n1_input", [&] {
                    return api.cuMemAlloc_v2(&n1_input, tensor_bytes);
                });
            }
        }
        if (result == CUDA_SUCCESS && files_ok && !arena_mode) {
            if (!n1_input2_path.empty()) {
                result = TimedStep(steps, "cuMemAlloc_n1_input2", [&] {
                    return api.cuMemAlloc_v2(&n1_input2, tensor_bytes);
                });
            }
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemAlloc_n1_weights", [&] {
                return api.cuMemAlloc_v2(&n1_weights, weights.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok && !arena_mode) {
            result = TimedStep(steps, "cuMemAlloc_n1_output", [&] {
                return api.cuMemAlloc_v2(&n1_output, tensor_bytes);
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemAlloc_n1_sync", [&] {
                return api.cuMemAlloc_v2(&n1_sync, sync_bytes);
            });
        }
        if (result == CUDA_SUCCESS && files_ok && n1_wait_ready) {
            result = TimedStep(steps, "cuMemAlloc_n1_wait_sync", [&] {
                return api.cuMemAlloc_v2(&n1_wait_sync, sync_bytes);
            });
        }
        if (result == CUDA_SUCCESS && files_ok && !arena_mode &&
            !n1_extra_output_path.empty()) {
            result = TimedStep(steps, "cuMemAlloc_n1_extra_output", [&] {
                return api.cuMemAlloc_v2(&n1_extra_output, tensor_bytes);
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            if (arena_mode) {
                result = TimedStep(steps, "cuMemcpyHtoD_n1_arena", [&] {
                    return api.cuMemcpyHtoD_v2(n1_arena, arena.data(), arena.size());
                });
            } else {
                result = TimedStep(steps, "cuMemcpyHtoD_n1_input", [&] {
                    return api.cuMemcpyHtoD_v2(n1_input, input.data(), input.size());
                });
            }
        }
        if (result == CUDA_SUCCESS && files_ok && !arena_mode) {
            if (n1_input2) {
                result = TimedStep(steps, "cuMemcpyHtoD_n1_input2", [&] {
                    return api.cuMemcpyHtoD_v2(n1_input2, input2.data(), input2.size());
                });
            }
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps, "cuMemcpyHtoD_n1_weights", [&] {
                return api.cuMemcpyHtoD_v2(n1_weights, weights.data(), weights.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok && !arena_mode) {
            result = TimedStep(steps,
                output_initial.empty() ? "cuMemset_n1_output" :
                                         "cuMemcpyHtoD_n1_output_initial", [&] {
                return output_initial.empty()
                    ? api.cuMemsetD32_v2(n1_output, 0, tensor_bytes / 4)
                    : api.cuMemcpyHtoD_v2(n1_output, output_initial.data(),
                                          output_initial.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok) {
            result = TimedStep(steps,
                sync_initial.empty() ? "cuMemset_n1_sync" :
                                       "cuMemcpyHtoD_n1_sync_initial", [&] {
                return sync_initial.empty()
                    ? api.cuMemsetD32_v2(n1_sync, UINT32_MAX, sync_bytes / 4)
                    : api.cuMemcpyHtoD_v2(n1_sync, sync_initial.data(),
                                          sync_initial.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok && n1_wait_ready) {
            result = TimedStep(steps,
                wait_sync_initial.empty() ? "cuMemset_n1_wait_ready" :
                                            "cuMemcpyHtoD_n1_wait_sync_initial", [&] {
                return wait_sync_initial.empty()
                    ? api.cuMemsetD32_v2(n1_wait_sync, 0, sync_bytes / 4)
                    : api.cuMemcpyHtoD_v2(n1_wait_sync, wait_sync_initial.data(),
                                          wait_sync_initial.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok && n1_extra_output && !arena_mode) {
            result = TimedStep(steps,
                extra_output_initial.empty() ? "cuMemset_n1_extra_output" :
                                               "cuMemcpyHtoD_n1_extra_output_initial", [&] {
                return extra_output_initial.empty()
                    ? api.cuMemsetD32_v2(n1_extra_output, 0, tensor_bytes / 4)
                    : api.cuMemcpyHtoD_v2(n1_extra_output,
                                          extra_output_initial.data(),
                                          extra_output_initial.size());
            });
        }
        if (result == CUDA_SUCCESS && files_ok && n1_rgba16f_surface) {
            constexpr std::size_t width = 640, height = 360, pixel_bytes = 8;
            CUDA_ARRAY_DESCRIPTOR desc{width, height, CU_AD_FORMAT_HALF, 4};
            result = TimedStep(steps, "cuArrayCreate_n1_surface", [&] {
                return api.cuArrayCreate_v2(&n1_surface_array, &desc);
            });
            std::vector<std::uint8_t> zero(width * height * pixel_bytes);
            const std::vector<std::uint8_t>& surface_source =
                surface_initial.empty() ? zero : surface_initial;
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = surface_source.data();
            copy.srcPitch = width * pixel_bytes;
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = n1_surface_array;
            copy.WidthInBytes = width * pixel_bytes;
            copy.Height = height;
            if (result == CUDA_SUCCESS) {
                result = TimedStep(steps,
                    surface_initial.empty() ? "cuMemcpy2D_n1_surface_clear" :
                                              "cuMemcpy2D_n1_surface_initial", [&] {
                    return api.cuMemcpy2D_v2(&copy);
                });
            }
            CUDA_RESOURCE_DESC surface_resource{};
            surface_resource.resType = CU_RESOURCE_TYPE_ARRAY;
            surface_resource.res.array.hArray = n1_surface_array;
            if (result == CUDA_SUCCESS) {
                result = TimedStep(steps, "cuSurfObjectCreate_n1_output", [&] {
                    return api.cuSurfObjectCreate(&n1_surface, &surface_resource);
                });
            }
            if (result == CUDA_SUCCESS) {
                result = TimedStep(steps, "cuArrayCreate_n1_zero_texture", [&] {
                    return api.cuArrayCreate_v2(&n1_texture_array, &desc);
                });
            }
            copy.srcHost = zero.data();
            copy.dstArray = n1_texture_array;
            if (result == CUDA_SUCCESS) {
                result = TimedStep(steps, "cuMemcpy2D_n1_zero_texture", [&] {
                    return api.cuMemcpy2D_v2(&copy);
                });
            }
            CUDA_RESOURCE_DESC texture_resource{};
            texture_resource.resType = CU_RESOURCE_TYPE_ARRAY;
            texture_resource.res.array.hArray = n1_texture_array;
            CUDA_TEXTURE_DESC texture_desc{};
            texture_desc.addressMode[0] = texture_desc.addressMode[1] =
                texture_desc.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
            texture_desc.filterMode = CU_TR_FILTER_MODE_POINT;
            texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
            if (result == CUDA_SUCCESS) {
                result = TimedStep(steps, "cuTexObjectCreate_n1_zero_texture", [&] {
                    return api.cuTexObjectCreate(&n1_texture, &texture_resource,
                                                 &texture_desc, nullptr);
                });
            }
        }
        if (result == CUDA_SUCCESS && files_ok) {
            const CUdeviceptr weight_view = n1_weights + n1_weight_view_offset;
            std::memcpy(params.data() + n1_input_param_offset, &n1_input,
                        sizeof(n1_input));
            if (n1_input2)
                std::memcpy(params.data() + n1_input2_param_offset, &n1_input2,
                            sizeof(n1_input2));
            std::memcpy(params.data() + n1_output_param_offset, &n1_output,
                        sizeof(n1_output));
            if (n1_patch_weights)
                std::memcpy(params.data() + n1_weights_param_offset, &weight_view,
                            sizeof(weight_view));
            if (n1_wait_ready)
                std::memcpy(params.data() + n1_wait_param_offset, &n1_wait_sync,
                            sizeof(n1_wait_sync));
            if (n1_patch_release)
                std::memcpy(params.data() + n1_release_param_offset, &n1_sync,
                            sizeof(n1_sync));
            if (n1_extra_output)
                std::memcpy(params.data() + n1_extra_param_offset, &n1_extra_output,
                            sizeof(n1_extra_output));
            for (const auto& view : n1_arena_param_views) {
                const CUdeviceptr pointer = n1_arena + view.second;
                std::memcpy(params.data() + view.first, &pointer, sizeof(pointer));
            }
            for (const auto& view : n1_weight_param_views) {
                const CUdeviceptr pointer = n1_weights + view.second;
                std::memcpy(params.data() + view.first, &pointer, sizeof(pointer));
            }
            if (n1_rgba16f_surface) {
                std::memcpy(params.data() + n1_rgba16f_surface_param_offset,
                            &n1_surface, sizeof(n1_surface));
                std::memcpy(params.data() + n1_rgba16f_zero_texture_param_offset,
                            &n1_texture, sizeof(n1_texture));
            }
            void* kernel_params[] = {params.data()};
            result = TimedStep(steps, "cuLaunchKernel_n1", [&] {
                return api.cuLaunchKernel(function, n1_grid_x, n1_grid_y, n1_grid_z,
                                          n1_block_x, n1_block_y, n1_block_z, 0,
                                          nullptr, kernel_params, nullptr);
            });
            kernel_launched = result == CUDA_SUCCESS;
        }
        if (kernel_launched) {
            result = TimedStep(steps, "cuCtxSynchronize_n1", [&] {
                return api.cuCtxSynchronize();
            });
        }
        const std::size_t output_readback_bytes = n1_rgba16f_surface
            ? 640ull * 360ull * 8ull
            : (arena_mode
                ? (std::min)(tensor_bytes, arena.size() - n1_arena_output_offset)
                : tensor_bytes);
        std::vector<std::uint8_t> output_readback(output_readback_bytes);
        std::vector<std::uint8_t> sync_readback(sync_bytes);
        const std::size_t extra_output_readback_bytes =
            n1_extra_output
                ? (arena_mode
                    ? (std::min)(tensor_bytes, arena.size() - n1_arena_extra_offset)
                    : tensor_bytes)
                : 0;
        std::vector<std::uint8_t> extra_output_readback(extra_output_readback_bytes);
        std::vector<std::uint8_t> arena_readback(
            n1_arena_output_path.empty() ? 0 : arena.size());
        if (result == CUDA_SUCCESS && kernel_launched) {
            if (n1_rgba16f_surface) {
                CUDA_MEMCPY2D copy{};
                copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                copy.srcArray = n1_surface_array;
                copy.dstMemoryType = CU_MEMORYTYPE_HOST;
                copy.dstHost = output_readback.data();
                copy.dstPitch = 640 * 8;
                copy.WidthInBytes = 640 * 8;
                copy.Height = 360;
                result = TimedStep(steps, "cuMemcpy2D_n1_surface_output", [&] {
                    return api.cuMemcpy2D_v2(&copy);
                });
            } else {
                result = TimedStep(steps, "cuMemcpyDtoH_n1_output", [&] {
                    return api.cuMemcpyDtoH_v2(output_readback.data(), n1_output,
                                               output_readback.size());
                });
            }
        }
        if (result == CUDA_SUCCESS && kernel_launched) {
            result = TimedStep(steps, "cuMemcpyDtoH_n1_sync", [&] {
                return api.cuMemcpyDtoH_v2(sync_readback.data(), n1_sync,
                                           sync_readback.size());
            });
        }
        if (result == CUDA_SUCCESS && kernel_launched && n1_extra_output) {
            result = TimedStep(steps, "cuMemcpyDtoH_n1_extra_output", [&] {
                return api.cuMemcpyDtoH_v2(extra_output_readback.data(), n1_extra_output,
                                           extra_output_readback.size());
            });
        }
        if (result == CUDA_SUCCESS && kernel_launched && !arena_readback.empty()) {
            result = TimedStep(steps, "cuMemcpyDtoH_n1_arena", [&] {
                return api.cuMemcpyDtoH_v2(arena_readback.data(), n1_arena,
                                           arena_readback.size());
            });
        }
        if (result == CUDA_SUCCESS && kernel_launched) {
            for (std::uint8_t value : output_readback) n1_output_nonzero += value != 0;
            for (std::uint8_t value : extra_output_readback)
                n1_extra_output_nonzero += value != 0;
            for (std::uint8_t value : sync_readback) n1_sync_nonzero += value != 0;
            for (std::size_t offset = 0; offset < sync_readback.size(); offset += 4) {
                if (sync_readback[offset] == 0 && sync_readback[offset + 1] == 0 &&
                    sync_readback[offset + 2] == 0 && sync_readback[offset + 3] == 0) {
                    ++n1_sync_zero_words;
                }
            }
            execution_verified = n1_output_nonzero > 0 &&
                                 n1_sync_zero_words == n1_expected_releases &&
                                 (!n1_extra_output || n1_extra_output_nonzero > 0);
            if (!WriteBytes(n1_output_path, output_readback, fatal_error) ||
                !WriteBytes(n1_sync_output_path, sync_readback, fatal_error)) {
                std::fprintf(stderr, "N1 output write failed: %s\n", fatal_error.c_str());
                execution_verified = false;
            }
            if (n1_extra_output &&
                !WriteBytes(n1_extra_output_path, extra_output_readback, fatal_error)) {
                std::fprintf(stderr, "N1 extra output write failed: %s\n", fatal_error.c_str());
                execution_verified = false;
            }
            if (!arena_readback.empty() &&
                !WriteBytes(n1_arena_output_path, arena_readback, fatal_error)) {
                std::fprintf(stderr, "N1 arena output write failed: %s\n",
                             fatal_error.c_str());
                execution_verified = false;
            }
        }
    }

    const bool base_pass = initialized && have_device && have_context &&
                           module_loaded && function_resolved;
    const bool pass = base_pass &&
                      (clear_words == 0 || (kernel_launched && execution_verified)) &&
                      (!n0_mode || (kernel_launched && execution_verified)) &&
                      (!n1_mode || (kernel_launched && execution_verified));

    std::ostringstream json;
    json << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"classification\": \"PTX_TRANSLATION_PROBE\",\n"
         << "  \"counts_as_s6\": false,\n"
         << "  \"pass\": " << (pass ? "true" : "false") << ",\n"
         << "  \"nvcuda_path\": \"" << JsonEscape(WideToUtf8(nvcuda_path))
         << "\",\n"
         << "  \"ptx_path\": \"" << JsonEscape(WideToUtf8(ptx_path)) << "\",\n"
         << "  \"ptx_bytes\": " << (ptx.size() - 1) << ",\n"
         << "  \"function\": \"" << JsonEscape(function_name) << "\",\n"
         << "  \"device_count\": " << device_count << ",\n"
         << "  \"device_index\": 0,\n"
         << "  \"device_name\": \"" << JsonEscape(device_name) << "\",\n"
         << "  \"module_loaded\": " << (module_loaded ? "true" : "false")
         << ",\n"
         << "  \"function_resolved\": "
         << (function_resolved ? "true" : "false") << ",\n"
         << "  \"execution_mode\": \""
         << (n0_mode ? "n0_synthetic" :
             (n1_mode ? (n1_wait_ready ? "neural_dependency" : "n1_slot2") :
              (clear_words > 0 ? "cc_cb_clear" : "none")))
         << "\",\n"
         << "  \"clear_words\": " << clear_words << ",\n"
         << "  \"n0_grid\": [" << n0_grid_x << ", " << n0_grid_y << ", 1],\n"
         << "  \"n0_scratch_extra_bytes\": " << n0_scratch_extra_bytes << ",\n"
         << "  \"kernel_launched\": "
         << (kernel_launched ? "true" : "false") << ",\n"
         << "  \"execution_verified\": "
         << (execution_verified ? "true" : "false") << ",\n"
         << "  \"clear_mismatches\": " << clear_mismatches << ",\n"
         << "  \"n0_scratch_nonzero_bytes\": " << n0_scratch_nonzero << ",\n"
         << "  \"n0_output_nonzero_bytes\": " << n0_output_nonzero << ",\n"
         << "  \"n1_output_nonzero_bytes\": " << n1_output_nonzero << ",\n"
         << "  \"n1_sync_nonzero_bytes\": " << n1_sync_nonzero << ",\n"
         << "  \"n1_sync_zero_words\": " << n1_sync_zero_words << ",\n"
         << "  \"n1_extra_output_nonzero_bytes\": " << n1_extra_output_nonzero << ",\n"
         << "  \"n1_weights_input_bytes\": " << n1_weights_input_bytes << ",\n"
         << "  \"n1_input2_loaded\": "
         << ((!n1_input2_path.empty() || n1_arena_input2_set) ? "true" : "false") << ",\n"
         << "  \"n1_arena_loaded\": "
         << (!n1_arena_path.empty() ? "true" : "false") << ",\n"
         << "  \"n1_arena_bytes\": "
         << (!n1_arena_path.empty() ? std::filesystem::file_size(n1_arena_path) : 0) << ",\n"
         << "  \"n1_arena_output_written\": "
         << (!n1_arena_output_path.empty() ? "true" : "false") << ",\n"
         << "  \"n1_arena_param_view_count\": "
         << n1_arena_param_views.size() << ",\n"
         << "  \"n1_weight_param_view_count\": "
         << n1_weight_param_views.size() << ",\n"
         << "  \"n1_output_initial_loaded\": "
         << ((!n1_output_initial_path.empty() || !n1_arena_path.empty()) ? "true" : "false") << ",\n"
         << "  \"n1_rgba16f_surface\": "
         << (n1_rgba16f_surface ? "true" : "false") << ",\n"
         << "  \"n1_rgba16f_surface_param_offset\": "
         << n1_rgba16f_surface_param_offset << ",\n"
         << "  \"n1_rgba16f_surface_initial_loaded\": "
         << (!n1_rgba16f_surface_initial_path.empty() ? "true" : "false") << ",\n"
         << "  \"n1_rgba16f_zero_texture_param_offset\": "
         << n1_rgba16f_zero_texture_param_offset << ",\n"
         << "  \"n1_sync_initial_loaded\": "
         << (!n1_sync_initial_path.empty() ? "true" : "false") << ",\n"
         << "  \"n1_wait_sync_initial_loaded\": "
         << (!n1_wait_sync_initial_path.empty() ? "true" : "false") << ",\n"
         << "  \"n1_extra_output_initial_loaded\": "
         << ((!n1_extra_output_initial_path.empty() || n1_arena_extra_set) ? "true" : "false") << ",\n"
         << "  \"n1_grid\": [" << n1_grid_x << ", " << n1_grid_y << ", "
         << n1_grid_z << "],\n"
         << "  \"n1_block\": [" << n1_block_x << ", " << n1_block_y << ", "
         << n1_block_z << "],\n"
         << "  \"n1_weight_view_offset\": " << n1_weight_view_offset << ",\n"
         << "  \"n1_input_param_offset\": " << n1_input_param_offset << ",\n"
         << "  \"n1_input2_param_offset\": " << n1_input2_param_offset << ",\n"
         << "  \"n1_output_param_offset\": " << n1_output_param_offset << ",\n"
         << "  \"n1_weights_param_offset\": " << n1_weights_param_offset << ",\n"
         << "  \"n1_patch_weights\": " << (n1_patch_weights ? "true" : "false") << ",\n"
         << "  \"n1_arena_input_offset\": " << n1_arena_input_offset << ",\n"
         << "  \"n1_arena_input2_offset\": " << n1_arena_input2_offset << ",\n"
         << "  \"n1_arena_output_offset\": " << n1_arena_output_offset << ",\n"
         << "  \"n1_arena_extra_offset\": " << n1_arena_extra_offset << ",\n"
         << "  \"n1_wait_param_offset\": " << n1_wait_param_offset << ",\n"
         << "  \"n1_release_param_offset\": " << n1_release_param_offset << ",\n"
         << "  \"n1_patch_release\": " << (n1_patch_release ? "true" : "false") << ",\n"
         << "  \"n1_extra_param_offset\": " << n1_extra_param_offset << ",\n"
         << "  \"n1_expected_releases\": " << n1_expected_releases << ",\n"
         << "  \"steps\": [\n";
    for (size_t i = 0; i < steps.size(); ++i) {
        const Step& step = steps[i];
        json << "    {\"name\": \"" << JsonEscape(step.name) << "\", \"code\": "
             << step.code << ", \"error_name\": \""
             << JsonEscape(ErrorName(api, step.code)) << "\", \"error_text\": \""
             << JsonEscape(ErrorText(api, step.code)) << "\", \"milliseconds\": "
             << step.milliseconds << "}";
        if (i + 1 != steps.size()) json << ',';
        json << '\n';
    }
    json << "  ]\n}\n";

    if (n1_texture) api.cuTexObjectDestroy(n1_texture);
    if (n1_surface) api.cuSurfObjectDestroy(n1_surface);
    if (n1_texture_array) api.cuArrayDestroy(n1_texture_array);
    if (n1_surface_array) api.cuArrayDestroy(n1_surface_array);
    if (n0_texture) api.cuTexObjectDestroy(n0_texture);
    if (n0_array) api.cuArrayDestroy(n0_array);
    if (n0_output) api.cuMemFree_v2(n0_output);
    if (n0_weights) api.cuMemFree_v2(n0_weights);
    if (n0_scratch) api.cuMemFree_v2(n0_scratch);
    if (n1_extra_output && !n1_arena) api.cuMemFree_v2(n1_extra_output);
    if (n1_wait_sync) api.cuMemFree_v2(n1_wait_sync);
    if (n1_sync) api.cuMemFree_v2(n1_sync);
    if (n1_output && !n1_arena) api.cuMemFree_v2(n1_output);
    if (n1_weights) api.cuMemFree_v2(n1_weights);
    if (n1_input2 && !n1_arena) api.cuMemFree_v2(n1_input2);
    if (n1_input && !n1_arena) api.cuMemFree_v2(n1_input);
    if (n1_arena) api.cuMemFree_v2(n1_arena);
    if (clear_buffer) api.cuMemFree_v2(clear_buffer);
    if (module) api.cuModuleUnload(module);
    if (context) api.cuCtxDestroy_v2(context);
    FreeLibrary(api.library);

    if (!WriteJson(json_path, json.str(), fatal_error)) {
        std::fprintf(stderr, "%s: %s\n", fatal_error.c_str(),
                     WideToUtf8(json_path).c_str());
        return 6;
    }

    std::printf("device=%s module_loaded=%s function_resolved=%s "
                "kernel_launched=%s execution_verified=%s result=%s\n",
                device_name, module_loaded ? "true" : "false",
                function_resolved ? "true" : "false",
                kernel_launched ? "true" : "false",
                execution_verified ? "true" : "false",
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
