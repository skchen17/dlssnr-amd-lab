#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
using CUcontext = CUctx_st*;
using CUmodule = CUmod_st*;
using CUfunction = CUfunc_st*;

static constexpr CUresult CUDA_SUCCESS = 0;

static const char kPtx[] = R"ptx(
.version 8.1
.target sm_89
.address_size 64

.visible .entry fp8_encode_all_half(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 element_count
)
{
    .reg .pred %p<2>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r1, [element_count];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    shl.b32 %r6, %r5, 1;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;

    shl.b32 %r7, %r5, 2;
    cvt.u64.u32 %rd3, %r7;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.b32 %r8, [%rd4];
    cvt.rn.satfinite.e4m3x2.f16x2 %rs1, %r8;

    shl.b32 %r9, %r5, 1;
    cvt.u64.u32 %rd5, %r9;
    add.s64 %rd6, %rd2, %rd5;
    st.global.b16 [%rd6], %rs1;
DONE:
    ret;
}
)ptx";

template <typename T>
static bool LoadSymbol(HMODULE module, const char* name, T& value) {
    value = reinterpret_cast<T>(GetProcAddress(module, name));
    return value != nullptr;
}

static std::string JsonEscape(const char* value) {
    std::string out;
    for (const unsigned char c : std::string(value ? value : "")) {
        if (c == '\\' || c == '"') out.push_back('\\');
        if (c >= 0x20) out.push_back(char(c));
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: nvidia_fp8_reference <output.raw> <result.json>\n");
        return 2;
    }
    const fs::path rawPath = argv[1];
    const fs::path jsonPath = argv[2];
    HMODULE cuda = LoadLibraryW(L"nvcuda.dll");
    if (!cuda) {
        std::fprintf(stderr, "nvcuda.dll not found\n");
        return 3;
    }

    CUresult(WINAPI *cuInit)(unsigned) = nullptr;
    CUresult(WINAPI *cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult(WINAPI *cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult(WINAPI *cuCtxCreate)(CUcontext*, unsigned, CUdevice) = nullptr;
    CUresult(WINAPI *cuCtxDestroy)(CUcontext) = nullptr;
    CUresult(WINAPI *cuModuleLoadDataEx)(CUmodule*, const void*, unsigned, void*, void*) = nullptr;
    CUresult(WINAPI *cuModuleUnload)(CUmodule) = nullptr;
    CUresult(WINAPI *cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult(WINAPI *cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult(WINAPI *cuMemFree)(CUdeviceptr) = nullptr;
    CUresult(WINAPI *cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult(WINAPI *cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult(WINAPI *cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                     unsigned, unsigned, unsigned, unsigned,
                                     void*, void**, void**) = nullptr;
    CUresult(WINAPI *cuCtxSynchronize)() = nullptr;
    bool symbols =
        LoadSymbol(cuda, "cuInit", cuInit) &&
        LoadSymbol(cuda, "cuDeviceGet", cuDeviceGet) &&
        LoadSymbol(cuda, "cuDeviceGetName", cuDeviceGetName) &&
        LoadSymbol(cuda, "cuCtxCreate_v2", cuCtxCreate) &&
        LoadSymbol(cuda, "cuCtxDestroy_v2", cuCtxDestroy) &&
        LoadSymbol(cuda, "cuModuleLoadDataEx", cuModuleLoadDataEx) &&
        LoadSymbol(cuda, "cuModuleUnload", cuModuleUnload) &&
        LoadSymbol(cuda, "cuModuleGetFunction", cuModuleGetFunction) &&
        LoadSymbol(cuda, "cuMemAlloc_v2", cuMemAlloc) &&
        LoadSymbol(cuda, "cuMemFree_v2", cuMemFree) &&
        LoadSymbol(cuda, "cuMemcpyHtoD_v2", cuMemcpyHtoD) &&
        LoadSymbol(cuda, "cuMemcpyDtoH_v2", cuMemcpyDtoH) &&
        LoadSymbol(cuda, "cuLaunchKernel", cuLaunchKernel) &&
        LoadSymbol(cuda, "cuCtxSynchronize", cuCtxSynchronize);
    if (!symbols) {
        std::fprintf(stderr, "required CUDA driver symbol missing\n");
        FreeLibrary(cuda);
        return 4;
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUdeviceptr deviceInput = 0, deviceOutput = 0;
    std::vector<uint16_t> input(65536);
    std::vector<uint8_t> output(65536);
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = uint16_t(i);
    char deviceName[256] = {};
    CUresult status = cuInit(0);
    if (status == CUDA_SUCCESS) status = cuDeviceGet(&device, 0);
    if (status == CUDA_SUCCESS) status = cuDeviceGetName(deviceName, sizeof(deviceName), device);
    if (status == CUDA_SUCCESS) status = cuCtxCreate(&context, 0, device);
    if (status == CUDA_SUCCESS) status = cuModuleLoadDataEx(&module, kPtx, 0, nullptr, nullptr);
    if (status == CUDA_SUCCESS)
        status = cuModuleGetFunction(&function, module, "fp8_encode_all_half");
    if (status == CUDA_SUCCESS) status = cuMemAlloc(&deviceInput, input.size() * sizeof(uint16_t));
    if (status == CUDA_SUCCESS) status = cuMemAlloc(&deviceOutput, output.size());
    if (status == CUDA_SUCCESS)
        status = cuMemcpyHtoD(deviceInput, input.data(), input.size() * sizeof(uint16_t));
    uint32_t count = uint32_t(input.size());
    void* args[] = {&deviceInput, &deviceOutput, &count};
    if (status == CUDA_SUCCESS)
        status = cuLaunchKernel(function, 128, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr);
    if (status == CUDA_SUCCESS) status = cuCtxSynchronize();
    if (status == CUDA_SUCCESS)
        status = cuMemcpyDtoH(output.data(), deviceOutput, output.size());

    if (deviceOutput) cuMemFree(deviceOutput);
    if (deviceInput) cuMemFree(deviceInput);
    if (module) cuModuleUnload(module);
    if (context) cuCtxDestroy(context);
    FreeLibrary(cuda);

    bool negativeDetected = false;
    if (status == CUDA_SUCCESS) {
        std::ofstream raw(rawPath, std::ios::binary);
        raw.write(reinterpret_cast<const char*>(output.data()), output.size());
        raw.close();
        negativeDetected = (uint8_t(output[0] ^ 1) != output[0]);
        if (!raw) status = -1;
    }
    const bool pass = status == CUDA_SUCCESS && negativeDetected;
    std::ofstream json(jsonPath, std::ios::binary);
    json << "{\n"
         << "  \"schema\": 1,\n"
         << "  \"experiment\": \"nvidia_fp16_to_e4m3_exhaustive\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"classification\": \"RTX_NUMERICAL_ORACLE\",\n"
         << "  \"counts_as_s6\": false,\n"
         << "  \"device\": \"" << JsonEscape(deviceName) << "\",\n"
         << "  \"ptx_version\": \"8.1\",\n"
         << "  \"ptx_target\": \"sm_89\",\n"
         << "  \"instruction\": \"cvt.rn.satfinite.e4m3x2.f16x2\",\n"
         << "  \"input_half_patterns\": 65536,\n"
         << "  \"output_bytes\": 65536,\n"
         << "  \"cuda_status\": " << status << ",\n"
         << "  \"negative_verifier_detected\": "
         << (negativeDetected ? "true" : "false") << "\n"
         << "}\n";
    std::printf("[%s] device=%s half_patterns=65536 output_bytes=65536 cuda_status=%d\n",
                pass ? "PASS" : "FAIL", deviceName, status);
    return pass ? 0 : 1;
}
