#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t e_ = (expr); \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#expr) + ": " + hipGetErrorString(e_)); \
} while (0)

constexpr uint32_t kGridWidth = 81, kGridHeight = 49, kGroups = 4;
constexpr size_t kCtas = size_t(kGridWidth) * kGridHeight;
constexpr size_t kInputValuesPerCta = 4 * 16 * 32;
constexpr size_t kInputValues = kCtas * kInputValuesPerCta;
constexpr uint32_t kMmas = 48, kRows = 16, kColumns = 8;
constexpr size_t kOutputValuesPerCta = size_t(kMmas) * kRows * kColumns;
constexpr size_t kOutputValues = kCtas * kOutputValuesPerCta;
constexpr size_t kHeadOffset = 147429888, kHeadBytes = 21808;
constexpr size_t kSelectedCta = size_t(26) * kGridWidth + 70;

__device__ static float DecodeE4M3(uint8_t bits) {
    const uint8_t magnitude = bits & 0x7f;
    const int exponent = magnitude >> 3, mantissa = magnitude & 7;
    float value;
    if (exponent == 0) value = ldexpf(float(mantissa), -9);
    else if (magnitude == 0x7f) value = nanf("");
    else value = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    return (bits & 0x80) ? -value : value;
}

__device__ static uint32_t RoundShift(uint32_t value, uint32_t shift) {
    const uint32_t base = value >> shift, remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    return base + uint32_t(remainder > halfway || (remainder == halfway && (base & 1u)));
}

__device__ static uint8_t EncodeHalfBitsE4M3(uint16_t bits) {
    const uint8_t sign = uint8_t((bits >> 8) & 0x80);
    const uint32_t exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    if (exponent == 31 && mantissa) return 0x7f;
    uint32_t code;
    if (exponent < 9) code = min(RoundShift(1024 + mantissa, 16 - exponent), 8u);
    else {
        uint32_t rounded = RoundShift(mantissa, 7); const bool carry = rounded >= 8;
        code = ((exponent - 8 + uint32_t(carry)) << 3) | (carry ? 0 : rounded);
        code = min(code, 0x7eu);
    }
    return sign | uint8_t(code);
}

__device__ static size_t PackedAIndex(uint32_t group, uint32_t row, uint32_t k) {
    const uint32_t lane = (row % 8) * 4 + (k % 16) / 4;
    const uint32_t element = (k % 4) + (row >= 8 ? 4 : 0) + (k >= 16 ? 8 : 0);
    return size_t(group) * 512 + lane * 16 + element;
}

__device__ static size_t PackedBIndex(size_t tile, uint32_t k, uint32_t column) {
    const uint32_t lane = column * 4 + (k % 16) / 4;
    const uint32_t element = (k % 4) + (k >= 16 ? 4 : 0);
    return tile + lane * 16 + element;
}

__global__ static void Pack(const uint16_t* input, uint8_t* packed) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kInputValues) return;
    size_t split = index;
    const uint32_t column = uint32_t(split % 32); split /= 32;
    const uint32_t row = uint32_t(split % 16); split /= 16;
    const uint32_t group = uint32_t(split % 4);
    const size_t cta = split / 4;
    const uint32_t mmaInGroup = (column / 16) * 2 + (column % 16) / 8;
    const uint32_t columnWithinEight = column % 8;
    const uint32_t lane = (row % 8) * 4 + columnWithinEight / 2;
    const uint32_t dElement = (row >= 8 ? 2 : 0) + (columnWithinEight & 1);
    const uint32_t conversion = (mmaInGroup / 2) * 4 + (mmaInGroup % 2) +
                                (dElement / 2) * 2;
    const uint32_t packedElement = conversion * 2 + (dElement & 1);
    packed[cta * kInputValuesPerCta + size_t(group) * 512 + lane * 16 + packedElement] =
        EncodeHalfBitsE4M3(input[index]);
}

__global__ static void Project(const uint8_t* packed, const uint8_t* head, uint16_t* output) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputValues) return;
    size_t split = index;
    const uint32_t column = uint32_t(split % 8); split /= 8;
    const uint32_t row = uint32_t(split % 16); split /= 16;
    const uint32_t mmaLocal = uint32_t(split % kMmas);
    const size_t cta = split / kMmas;
    const uint32_t group = mmaLocal / 12;
    const uint32_t within = mmaLocal % 12;
    const uint8_t* a = packed + cta * kInputValuesPerCta;
    const size_t tile = 8400 + size_t(within / 2) * 512;
    const uint32_t outputColumn = (within & 1) * 8 + column;
    float accumulator = 0;
    for (uint32_t k = 0; k < 32; ++k)
        accumulator += DecodeE4M3(a[PackedAIndex(group, row, k)]) *
                       DecodeE4M3(head[PackedBIndex(tile, k, outputColumn % 8) +
                                          (outputColumn / 8) * 8]);
    output[index] = __half_as_ushort(__float2half_rn(accumulator));
}

static std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const size_t bytes = size_t(stream.tellg()); stream.seekg(0); std::vector<uint8_t> out(bytes);
    if (!stream.read(reinterpret_cast<char*>(out.data()), std::streamsize(bytes))) throw std::runtime_error("read failed");
    return out;
}

static std::vector<uint8_t> ReadSlice(const std::filesystem::path& path, size_t offset, size_t bytes) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const size_t size = size_t(stream.tellg()); if (offset > size || bytes > size - offset) throw std::runtime_error("slice exceeds file");
    stream.seekg(std::streamoff(offset)); std::vector<uint8_t> out(bytes);
    if (!stream.read(reinterpret_cast<char*>(out.data()), std::streamsize(bytes))) throw std::runtime_error("read failed");
    return out;
}

static void WriteFile(const std::filesystem::path& path, const void* data, size_t bytes) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(static_cast<const char*>(data), std::streamsize(bytes))) throw std::runtime_error("write failed");
}

static uint16_t TraceD(const std::vector<uint8_t>& trace, uint32_t mma, uint32_t row, uint32_t column) {
    const uint32_t lane = (row % 8) * 4 + column / 2;
    const uint32_t element = (row >= 8 ? 2 : 0) + (column & 1);
    const size_t o = (size_t(mma) * 32 + lane) * 40 + 32 + (element / 2) * 4;
    const uint32_t word = uint32_t(trace[o]) | uint32_t(trace[o+1])<<8 | uint32_t(trace[o+2])<<16 | uint32_t(trace[o+3])<<24;
    return uint16_t(word >> (16 * (element & 1)));
}

static uint64_t Fnv1a64(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data); uint64_t h = 1469598103934665603ull;
    for (size_t i=0;i<bytes;++i) { h ^= p[i]; h *= 1099511628211ull; } return h;
}

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 7) {
            std::fprintf(stderr, "usage: output_head_mma128_175 <first128_fp16.raw> <model_arena.raw> <rtx_trace.raw> <output.raw> <result.json> [iterations]\n");
            return 2;
        }
        const uint32_t iterations = argc == 7 ? uint32_t(std::stoul(argv[6])) : 10;
        const auto inputBytes = ReadFile(argv[1]); const auto head = ReadSlice(argv[2], kHeadOffset, kHeadBytes); const auto trace = ReadFile(argv[3]);
        if (inputBytes.size() != kInputValues * 2 || trace.size() < 176*32*40) throw std::runtime_error("unexpected input size");
        uint16_t *dInput=nullptr,*dOutput=nullptr; uint8_t *dPacked=nullptr,*dHead=nullptr;
        HIP_CHECK(hipMalloc(&dInput,inputBytes.size())); HIP_CHECK(hipMalloc(&dPacked,kInputValues));
        HIP_CHECK(hipMalloc(&dHead,head.size())); HIP_CHECK(hipMalloc(&dOutput,kOutputValues*2));
        HIP_CHECK(hipMemcpy(dInput,inputBytes.data(),inputBytes.size(),hipMemcpyHostToDevice)); HIP_CHECK(hipMemcpy(dHead,head.data(),head.size(),hipMemcpyHostToDevice));
        const dim3 block(256), packGrid(uint32_t((kInputValues+255)/256)), outGrid(uint32_t((kOutputValues+255)/256));
        auto launch=[&](){ Pack<<<packGrid,block>>>(dInput,dPacked); Project<<<outGrid,block>>>(dPacked,dHead,dOutput); };
        launch(); HIP_CHECK(hipDeviceSynchronize()); std::vector<uint16_t> output(kOutputValues),repeat(kOutputValues);
        HIP_CHECK(hipMemcpy(output.data(),dOutput,output.size()*2,hipMemcpyDeviceToHost)); launch(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(repeat.data(),dOutput,repeat.size()*2,hipMemcpyDeviceToHost));
        hipEvent_t start{},stop{}; HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); HIP_CHECK(hipEventRecord(start));
        for(uint32_t i=0;i<iterations;++i)launch(); HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop)); float totalMs=0; HIP_CHECK(hipEventElapsedTime(&totalMs,start,stop));
        uint64_t mismatches=0; int64_t first=-1;
        for(uint32_t m=0;m<48;++m)for(uint32_t r=0;r<16;++r)for(uint32_t c=0;c<8;++c){
            size_t i=(kSelectedCta*kOutputValuesPerCta+size_t(m)*128+size_t(r)*8+c); uint16_t expected=TraceD(trace,128+m,r,c);
            if(output[i]!=expected){if(first<0)first=int64_t(m*128+r*8+c);++mismatches;}}
        uint64_t nonfinite=0,nonzero=0; float lo=std::numeric_limits<float>::infinity(),hi=-lo;
        for(uint16_t bits:output){float v=__half2float(__ushort_as_half(bits));nonfinite+=!std::isfinite(v);nonzero+=v!=0;if(std::isfinite(v)){lo=std::min(lo,v);hi=std::max(hi,v);}}
        const bool deterministic=output==repeat, pass=deterministic&&mismatches==0&&nonfinite==0&&nonzero;
        WriteFile(argv[4],output.data(),output.size()*2); char hash[32]{}; std::snprintf(hash,sizeof(hash),"%016llX",(unsigned long long)Fnv1a64(output.data(),output.size()*2));
        hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop,0));
        std::string json="{\n  \"schema\": 1,\n  \"experiment\": \"amd_output_head_mma128_175\",\n  \"status\": \""+std::string(pass?"PASS":"FAIL")+"\",\n  \"device\": \""+std::string(prop.name)+"\",\n  \"output_shape\": [49, 81, 48, 16, 8],\n  \"selected_cta_rtx_half_values\": 6144,\n  \"selected_cta_rtx_mismatches\": "+std::to_string(mismatches)+",\n  \"first_selected_cta_mismatch\": "+std::to_string(first)+",\n  \"deterministic_repeat\": "+std::string(deterministic?"true":"false")+",\n  \"non_finite_values\": "+std::to_string(nonfinite)+",\n  \"nonzero_values\": "+std::to_string(nonzero)+",\n  \"minimum\": "+std::to_string(lo)+",\n  \"maximum\": "+std::to_string(hi)+",\n  \"iterations\": "+std::to_string(iterations)+",\n  \"average_gpu_ms\": "+std::to_string(totalMs/float(iterations))+",\n  \"output_fnv1a64\": \""+hash+"\",\n  \"covered_original_mmas\": [128, 175],\n  \"next_gate\": \"Recover the dynamic activation operands and FP16 seeds for MMAs 176 through 255\"\n}\n";
        WriteFile(argv[5],json.data(),json.size());
        HIP_CHECK(hipEventDestroy(start));HIP_CHECK(hipEventDestroy(stop));HIP_CHECK(hipFree(dOutput));HIP_CHECK(hipFree(dHead));HIP_CHECK(hipFree(dPacked));HIP_CHECK(hipFree(dInput));
        std::printf("[%s] %s MMAs 128-175: mismatches=%llu, %.6f ms\n",pass?"PASS":"FAIL",prop.name,(unsigned long long)mismatches,totalMs/float(iterations)); return pass?0:1;
    } catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
