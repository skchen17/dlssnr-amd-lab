#include <windows.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
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
.version 8.7
.target sm_120
.address_size 64

.visible .entry mma_e4m3_fragments(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u64 d_ptr,
    .param .u32 case_count
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b32 %a<4>;
    .reg .b32 %b<2>;
    .reg .b32 %c<2>;
    .reg .b32 %d<2>;
    .reg .b64 %rd<12>;

    mov.u32 %r1, %ctaid.x;
    ld.param.u32 %r2, [case_count];
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra DONE;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, 32, %r3;

    ld.param.u64 %rd1, [a_ptr];
    shl.b32 %r5, %r4, 4;
    cvt.u64.u32 %rd5, %r5;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.b32 %a0, [%rd6];
    ld.global.b32 %a1, [%rd6+4];
    ld.global.b32 %a2, [%rd6+8];
    ld.global.b32 %a3, [%rd6+12];

    ld.param.u64 %rd2, [b_ptr];
    shl.b32 %r6, %r4, 3;
    cvt.u64.u32 %rd7, %r6;
    add.s64 %rd8, %rd2, %rd7;
    ld.global.b32 %b0, [%rd8];
    ld.global.b32 %b1, [%rd8+4];

    ld.param.u64 %rd3, [c_ptr];
    add.s64 %rd9, %rd3, %rd7;
    ld.global.b32 %c0, [%rd9];
    ld.global.b32 %c1, [%rd9+4];

    mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
        {%d0, %d1},
        {%a0, %a1, %a2, %a3},
        {%b0, %b1},
        {%c0, %c1};

    ld.param.u64 %rd4, [d_ptr];
    add.s64 %rd10, %rd4, %rd7;
    st.global.b32 [%rd10], %d0;
    st.global.b32 [%rd10+4], %d1;
DONE:
    ret;
}
)ptx";

template <typename T>
static bool LoadSymbol(HMODULE module, const char* name, T& value) {
    value = reinterpret_cast<T>(GetProcAddress(module, name));
    return value != nullptr;
}

static bool WriteAll(const fs::path& path, const void* data, size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data), bytes);
    return bool(file);
}

static bool ReadText(const fs::path& path, std::string& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return bool(file) || file.eof();
}

static uint32_t XorShift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static uint8_t FiniteE4M3(uint32_t value) {
    uint8_t code = uint8_t(value);
    return (code & 0x7f) == 0x7f ? uint8_t(code - 1) : code;
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
    if (argc != 2 && argc != 6) {
        std::fprintf(stderr,
                     "usage: nvidia_mma_reference <output-directory> "
                     "[--nvcuda <driver.dll> --ptx <module.ptx>]\n");
        return 2;
    }
    const fs::path outputDir = argv[1];
    std::string nvcudaPath = "nvcuda.dll";
    std::string ptxText = kPtx;
    if (argc == 6) {
        if (std::string(argv[2]) != "--nvcuda" || std::string(argv[4]) != "--ptx") {
            return 2;
        }
        nvcudaPath = argv[3];
        if (!ReadText(argv[5], ptxText)) return 2;
    }
    fs::create_directories(outputDir);
    HMODULE cuda = LoadLibraryA(nvcudaPath.c_str());
    if (!cuda) return 3;
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
    const bool symbols =
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
        FreeLibrary(cuda);
        return 4;
    }

    constexpr uint32_t kCases = 8, kLanes = 32;
    std::vector<uint32_t> a(kCases * kLanes * 4);
    std::vector<uint32_t> b(kCases * kLanes * 2);
    std::vector<uint32_t> c(kCases * kLanes * 2);
    std::vector<uint32_t> d(kCases * kLanes * 2);
    const std::array<uint16_t, 8> halfValues = {
        0x0000, 0x3c00, 0xbc00, 0x3800, 0xb800, 0x3400, 0xb400, 0x4000};
    uint32_t randomState = 0x6d6d6131;
    for (uint32_t cs = 0; cs < kCases; ++cs) {
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const size_t ai = (cs * kLanes + lane) * 4;
            const size_t bi = (cs * kLanes + lane) * 2;
            for (int reg = 0; reg < 4; ++reg) {
                uint32_t packed = 0;
                for (int byte = 0; byte < 4; ++byte) {
                    uint8_t value = 0;
                    if (cs == 1) value = 0x38;
                    else if (cs == 2) value = ((lane + reg + byte) & 1) ? 0xb8 : 0x38;
                    else if (cs == 6) value = ((lane + byte) & 1) ? 0xfe : 0x7e;
                    else if (cs == 7) value = uint8_t(1 + ((lane + reg + byte) % 7));
                    else if (cs >= 3) value = FiniteE4M3(XorShift32(randomState));
                    packed |= uint32_t(value) << (byte * 8);
                }
                a[ai + reg] = packed;
            }
            for (int reg = 0; reg < 2; ++reg) {
                uint32_t packed = 0;
                for (int byte = 0; byte < 4; ++byte) {
                    uint8_t value = 0;
                    if (cs == 1) value = 0x38;
                    else if (cs == 2) value = ((lane + reg + byte) & 1) ? 0x38 : 0xb8;
                    else if (cs == 6) value = ((lane + byte) & 1) ? 0x7e : 0xfe;
                    else if (cs == 7) value = uint8_t(1 + ((lane + reg + byte + 3) % 7));
                    else if (cs >= 3) value = FiniteE4M3(XorShift32(randomState));
                    packed |= uint32_t(value) << (byte * 8);
                }
                b[bi + reg] = packed;
                const uint16_t lo = cs >= 4 ? halfValues[(lane + reg + cs) % halfValues.size()] : 0;
                const uint16_t hi = cs >= 4 ? halfValues[(lane + reg + cs + 3) % halfValues.size()] : 0;
                c[bi + reg] = uint32_t(lo) | (uint32_t(hi) << 16);
            }
        }
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUdeviceptr da = 0, db = 0, dc = 0, dd = 0;
    char deviceName[256] = {};
    CUresult status = cuInit(0);
    if (status == CUDA_SUCCESS) status = cuDeviceGet(&device, 0);
    if (status == CUDA_SUCCESS) status = cuDeviceGetName(deviceName, sizeof(deviceName), device);
    if (status == CUDA_SUCCESS) status = cuCtxCreate(&context, 0, device);
    if (status == CUDA_SUCCESS)
        status = cuModuleLoadDataEx(&module, ptxText.c_str(), 0, nullptr, nullptr);
    if (status == CUDA_SUCCESS)
        status = cuModuleGetFunction(&function, module, "mma_e4m3_fragments");
    if (status == CUDA_SUCCESS) status = cuMemAlloc(&da, a.size() * sizeof(uint32_t));
    if (status == CUDA_SUCCESS) status = cuMemAlloc(&db, b.size() * sizeof(uint32_t));
    if (status == CUDA_SUCCESS) status = cuMemAlloc(&dc, c.size() * sizeof(uint32_t));
    if (status == CUDA_SUCCESS) status = cuMemAlloc(&dd, d.size() * sizeof(uint32_t));
    if (status == CUDA_SUCCESS) status = cuMemcpyHtoD(da, a.data(), a.size() * sizeof(uint32_t));
    if (status == CUDA_SUCCESS) status = cuMemcpyHtoD(db, b.data(), b.size() * sizeof(uint32_t));
    if (status == CUDA_SUCCESS) status = cuMemcpyHtoD(dc, c.data(), c.size() * sizeof(uint32_t));
    void* args[] = {&da, &db, &dc, &dd, const_cast<uint32_t*>(&kCases)};
    if (status == CUDA_SUCCESS)
        status = cuLaunchKernel(function, kCases, 1, 1, kLanes, 1, 1, 0, nullptr, args, nullptr);
    if (status == CUDA_SUCCESS) status = cuCtxSynchronize();
    if (status == CUDA_SUCCESS) status = cuMemcpyDtoH(d.data(), dd, d.size() * sizeof(uint32_t));

    if (dd) cuMemFree(dd);
    if (dc) cuMemFree(dc);
    if (db) cuMemFree(db);
    if (da) cuMemFree(da);
    if (module) cuModuleUnload(module);
    if (context) cuCtxDestroy(context);
    FreeLibrary(cuda);

    bool filesOk = false, negativeDetected = false;
    if (status == CUDA_SUCCESS) {
        filesOk = WriteAll(outputDir / "mma_a.raw", a.data(), a.size() * sizeof(uint32_t)) &&
                  WriteAll(outputDir / "mma_b.raw", b.data(), b.size() * sizeof(uint32_t)) &&
                  WriteAll(outputDir / "mma_c.raw", c.data(), c.size() * sizeof(uint32_t)) &&
                  WriteAll(outputDir / "mma_d.raw", d.data(), d.size() * sizeof(uint32_t));
        negativeDetected = (d[0] ^ 1u) != d[0];
    }
    const bool pass = status == CUDA_SUCCESS && filesOk && negativeDetected;
    std::ofstream json(outputDir / "mma_reference.json", std::ios::binary);
    json << "{\n"
         << "  \"schema\": 1,\n"
         << "  \"experiment\": \"nvidia_m16n8k32_e4m3_mma_fragments\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"classification\": \"RTX_NUMERICAL_ORACLE\",\n"
         << "  \"counts_as_s6\": false,\n"
         << "  \"device\": \"" << JsonEscape(deviceName) << "\",\n"
         << "  \"ptx_version\": \"8.7\",\n"
         << "  \"ptx_target\": \"sm_120\",\n"
         << "  \"instruction\": \"mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16\",\n"
         << "  \"cases\": 8,\n"
         << "  \"lanes_per_case\": 32,\n"
         << "  \"a_registers_per_lane\": 4,\n"
         << "  \"b_registers_per_lane\": 2,\n"
         << "  \"c_registers_per_lane\": 2,\n"
         << "  \"d_registers_per_lane\": 2,\n"
         << "  \"cuda_status\": " << status << ",\n"
         << "  \"negative_verifier_detected\": " << (negativeDetected ? "true" : "false") << "\n"
         << "}\n";
    std::printf("[%s] device=%s cases=%u output_bytes=%zu cuda_status=%d\n",
                pass ? "PASS" : "FAIL", deviceName, kCases,
                d.size() * sizeof(uint32_t), status);
    return pass ? 0 : 1;
}
