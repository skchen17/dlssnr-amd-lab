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

.visible .entry mma_f16_fragments(
    .param .u64 a_ptr, .param .u64 b_ptr, .param .u64 c_ptr,
    .param .u64 d_ptr, .param .u32 case_count)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>, %a<4>, %b<2>, %c<2>, %d<2>;
    .reg .b64 %rd<12>;
    mov.u32 %r1, %ctaid.x;
    ld.param.u32 %r2, [case_count];
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra F16_DONE;
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
    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
        {%d0, %d1}, {%a0, %a1, %a2, %a3}, {%b0, %b1}, {%c0, %c1};
    ld.param.u64 %rd4, [d_ptr];
    add.s64 %rd10, %rd4, %rd7;
    st.global.b32 [%rd10], %d0;
    st.global.b32 [%rd10+4], %d1;
F16_DONE:
    ret;
}

.visible .entry movmatrix_fragments(
    .param .u64 input_ptr, .param .u64 output_ptr, .param .u32 case_count)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>, %source, %dest;
    .reg .b64 %rd<8>;
    mov.u32 %r1, %ctaid.x;
    ld.param.u32 %r2, [case_count];
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra MOV_DONE;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, 32, %r3;
    shl.b32 %r5, %r4, 2;
    cvt.u64.u32 %rd3, %r5;
    ld.param.u64 %rd1, [input_ptr];
    add.s64 %rd4, %rd1, %rd3;
    ld.global.b32 %source, [%rd4];
    movmatrix.sync.trans.aligned.m8n8.b16 %dest, %source;
    ld.param.u64 %rd2, [output_ptr];
    add.s64 %rd5, %rd2, %rd3;
    st.global.b32 [%rd5], %dest;
MOV_DONE:
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

static uint16_t FiniteHalf(uint32_t value) {
    uint16_t bits = uint16_t(value);
    if ((bits & 0x7c00) == 0x7c00) bits = uint16_t((bits & 0x83ff) | 0x7800);
    return bits;
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
    if (argc != 2 && argc != 6) return 2;
    const fs::path outputDir = argv[1];
    std::string nvcudaPath = "nvcuda.dll";
    std::string ptxText = kPtx;
    if (argc == 6) {
        if (std::string(argv[2]) != "--nvcuda" || std::string(argv[4]) != "--ptx") return 2;
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
    const bool symbols = LoadSymbol(cuda, "cuInit", cuInit) &&
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
    if (!symbols) { FreeLibrary(cuda); return 4; }

    constexpr uint32_t kCases = 8, kLanes = 32;
    std::vector<uint32_t> a(kCases * kLanes * 4), b(kCases * kLanes * 2);
    std::vector<uint32_t> c(kCases * kLanes * 2), d(kCases * kLanes * 2);
    std::vector<uint32_t> movIn(kCases * kLanes), movOut(kCases * kLanes);
    const std::array<uint16_t, 8> values = {
        0x0000, 0x3c00, 0xbc00, 0x3800, 0xb800, 0x3400, 0xb400, 0x4000};
    uint32_t state = 0x66313631;
    for (uint32_t cs = 0; cs < kCases; ++cs) {
        for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const size_t ai = (cs * kLanes + lane) * 4;
            const size_t bi = (cs * kLanes + lane) * 2;
            for (int reg = 0; reg < 4; ++reg) {
                uint16_t lo = 0, hi = 0;
                if (cs == 1) lo = hi = 0x3c00;
                else if (cs == 2) { lo = ((lane + reg) & 1) ? 0xbc00 : 0x3c00; hi = uint16_t(lo ^ 0x8000); }
                else if (cs == 6) { lo = 0x7bff; hi = 0xfbff; }
                else if (cs == 7) { lo = uint16_t(1 + ((lane + reg) % 31)); hi = uint16_t(1 + ((lane + reg + 7) % 31)); }
                else if (cs >= 3) { lo = FiniteHalf(XorShift32(state)); hi = FiniteHalf(XorShift32(state)); }
                a[ai + reg] = uint32_t(lo) | (uint32_t(hi) << 16);
            }
            for (int reg = 0; reg < 2; ++reg) {
                uint16_t lo = 0, hi = 0;
                if (cs == 1) lo = hi = 0x3c00;
                else if (cs == 2) { lo = ((lane + reg) & 1) ? 0x3c00 : 0xbc00; hi = uint16_t(lo ^ 0x8000); }
                else if (cs == 6) { lo = 0x4000; hi = 0xc000; }
                else if (cs == 7) { lo = uint16_t(1 + ((lane + reg + 3) % 31)); hi = uint16_t(1 + ((lane + reg + 11) % 31)); }
                else if (cs >= 3) { lo = FiniteHalf(XorShift32(state)); hi = FiniteHalf(XorShift32(state)); }
                b[bi + reg] = uint32_t(lo) | (uint32_t(hi) << 16);
                const uint16_t clo = cs >= 4 ? values[(lane + reg + cs) % values.size()] : 0;
                const uint16_t chi = cs >= 4 ? values[(lane + reg + cs + 3) % values.size()] : 0;
                c[bi + reg] = uint32_t(clo) | (uint32_t(chi) << 16);
            }
            movIn[cs * kLanes + lane] =
                (uint32_t(uint16_t(cs * 0x101 + lane)) << 16) |
                uint16_t(0x8000 + cs * 0x101 + lane * 3);
        }
    }

    CUdevice device = 0; CUcontext context = nullptr; CUmodule module = nullptr;
    CUfunction f16Function = nullptr, movFunction = nullptr;
    CUdeviceptr da=0, db=0, dc=0, dd=0, dmi=0, dmo=0;
    char deviceName[256] = {};
    CUresult status = cuInit(0);
    if(status==0) status=cuDeviceGet(&device,0);
    if(status==0) status=cuDeviceGetName(deviceName,sizeof(deviceName),device);
    if(status==0) status=cuCtxCreate(&context,0,device);
    if(status==0) status=cuModuleLoadDataEx(&module,ptxText.c_str(),0,nullptr,nullptr);
    if(status==0) status=cuModuleGetFunction(&f16Function,module,"mma_f16_fragments");
    if(status==0) status=cuModuleGetFunction(&movFunction,module,"movmatrix_fragments");
    auto alloc = [&](CUdeviceptr& ptr,size_t bytes){ if(status==0) status=cuMemAlloc(&ptr,bytes); };
    alloc(da,a.size()*4); alloc(db,b.size()*4); alloc(dc,c.size()*4); alloc(dd,d.size()*4);
    alloc(dmi,movIn.size()*4); alloc(dmo,movOut.size()*4);
    auto h2d = [&](CUdeviceptr ptr,const void* data,size_t bytes){if(status==0)status=cuMemcpyHtoD(ptr,data,bytes);};
    h2d(da,a.data(),a.size()*4); h2d(db,b.data(),b.size()*4); h2d(dc,c.data(),c.size()*4);
    h2d(dmi,movIn.data(),movIn.size()*4);
    uint32_t caseCount=kCases;
    void* f16Args[]={&da,&db,&dc,&dd,&caseCount};
    if(status==0) status=cuLaunchKernel(f16Function,kCases,1,1,kLanes,1,1,0,nullptr,f16Args,nullptr);
    void* movArgs[]={&dmi,&dmo,&caseCount};
    if(status==0) status=cuLaunchKernel(movFunction,kCases,1,1,kLanes,1,1,0,nullptr,movArgs,nullptr);
    if(status==0) status=cuCtxSynchronize();
    if(status==0) status=cuMemcpyDtoH(d.data(),dd,d.size()*4);
    if(status==0) status=cuMemcpyDtoH(movOut.data(),dmo,movOut.size()*4);
    for(CUdeviceptr ptr:{dmo,dmi,dd,dc,db,da}) if(ptr) cuMemFree(ptr);
    if(module)cuModuleUnload(module); if(context)cuCtxDestroy(context); FreeLibrary(cuda);

    bool filesOk=false, negative=false;
    if(status==0) {
        filesOk=WriteAll(outputDir/"f16_a.raw",a.data(),a.size()*4)&&
            WriteAll(outputDir/"f16_b.raw",b.data(),b.size()*4)&&
            WriteAll(outputDir/"f16_c.raw",c.data(),c.size()*4)&&
            WriteAll(outputDir/"f16_d.raw",d.data(),d.size()*4)&&
            WriteAll(outputDir/"mov_input.raw",movIn.data(),movIn.size()*4)&&
            WriteAll(outputDir/"mov_output.raw",movOut.data(),movOut.size()*4);
        negative=((d[0]^1u)!=d[0])&&((movOut[0]^1u)!=movOut[0]);
    }
    const bool pass=status==0&&filesOk&&negative;
    std::ofstream json(outputDir/"remaining_reference.json",std::ios::binary);
    json<<"{\n  \"schema\": 1,\n  \"experiment\": \"nvidia_remaining_n0_primitives\",\n"
        <<"  \"status\": \""<<(pass?"PASS":"FAIL")<<"\",\n"
        <<"  \"classification\": \"RTX_NUMERICAL_ORACLE\",\n  \"counts_as_s6\": false,\n"
        <<"  \"device\": \""<<JsonEscape(deviceName)<<"\",\n"
        <<"  \"ptx_version\": \"8.7\",\n  \"ptx_target\": \"sm_120\",\n"
        <<"  \"cases\": 8,\n  \"lanes_per_case\": 32,\n"
        <<"  \"f16_instruction\": \"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16\",\n"
        <<"  \"mov_instruction\": \"movmatrix.sync.trans.aligned.m8n8.b16\",\n"
        <<"  \"cuda_status\": "<<status<<",\n  \"negative_verifier_detected\": "<<(negative?"true":"false")<<"\n}\n";
    std::printf("[%s] device=%s cases=8 f16_d_bytes=%zu mov_bytes=%zu cuda_status=%d\n",
        pass?"PASS":"FAIL",deviceName,d.size()*4,movOut.size()*4,status);
    return pass?0:1;
}
