#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

__device__ static uint16_t FragmentHalf(const uint32_t* data, size_t laneBase,
                                        uint32_t element) {
    return uint16_t(data[laneBase + element / 2] >> (16 * (element & 1)));
}

__global__ static void EmulateF16M16N8K16(const uint32_t* a, const uint32_t* b,
                                          const uint16_t* c, uint16_t* d,
                                          uint32_t cases) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(cases) * 32 * 4) return;
    const uint32_t cs = uint32_t(index / 128);
    const uint32_t local = uint32_t(index % 128);
    const uint32_t lane = local / 4, ci = local & 3;
    const uint32_t group = lane >> 2, threadInGroup = lane & 3;
    const uint32_t row = group + (ci >= 2 ? 8 : 0);
    const uint32_t col = threadInGroup * 2 + (ci & 1);
    float accumulator = __half2float(__ushort_as_half(c[index]));
    for (uint32_t k = 0; k < 16; ++k) {
        const uint32_t aLane = group * 4 + (k & 3);
        const uint32_t ai = 2 * (k >> 2) + (row >= 8 ? 1 : 0);
        const size_t aBase = (size_t(cs) * 32 + aLane) * 4;
        const uint32_t bLane = col * 4 + ((k & 7) >> 1);
        const uint32_t bi = (k & 1) + (k >= 8 ? 2 : 0);
        const size_t bBase = (size_t(cs) * 32 + bLane) * 2;
        const float av = __half2float(__ushort_as_half(FragmentHalf(a, aBase, ai)));
        const float bv = __half2float(__ushort_as_half(FragmentHalf(b, bBase, bi)));
        accumulator += av * bv;
    }
    d[index] = __half_as_ushort(__float2half_rn(accumulator));
}

__global__ static void EmulateMovMatrix(const uint32_t* input, uint32_t* output,
                                        uint32_t cases) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(cases) * 32) return;
    const uint32_t cs = uint32_t(index / 32), lane = uint32_t(index % 32);
    const uint32_t col = lane >> 2, rowPair = lane & 3;
    uint32_t packed = 0;
    for (uint32_t element = 0; element < 2; ++element) {
        const uint32_t row = rowPair * 2 + element;
        const uint32_t sourceLane = row * 4 + (col >> 1);
        const uint32_t source = input[cs * 32 + sourceLane];
        const uint16_t value = uint16_t(source >> (16 * (col & 1)));
        packed |= uint32_t(value) << (16 * element);
    }
    output[index] = packed;
}

template <typename T>
static std::vector<T> ReadVector(const fs::path& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || size_t(file.tellg()) != count * sizeof(T)) return {};
    file.seekg(0);
    std::vector<T> result(count);
    if (!file.read(reinterpret_cast<char*>(result.data()),
                   std::streamsize(count * sizeof(T)))) return {};
    return result;
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: remaining_primitives_probe <f16-a> <f16-b> "
            "<f16-c> <f16-d> <mov-in> <mov-out> <result.json>\n");
        return 2;
    }
    constexpr uint32_t kCases = 8;
    const auto a = ReadVector<uint32_t>(argv[1], 1024);
    const auto b = ReadVector<uint32_t>(argv[2], 512);
    const auto c = ReadVector<uint16_t>(argv[3], 1024);
    const auto f16Reference = ReadVector<uint16_t>(argv[4], 1024);
    const auto movInput = ReadVector<uint32_t>(argv[5], 256);
    const auto movReference = ReadVector<uint32_t>(argv[6], 256);
    if (a.empty() || b.empty() || c.empty() || f16Reference.empty() ||
        movInput.empty() || movReference.empty()) return 3;
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 4;
    uint32_t *da=nullptr,*db=nullptr,*dmi=nullptr,*dmo=nullptr;
    uint16_t *dc=nullptr,*dd=nullptr;
    if (hipMalloc(&da,a.size()*4)!=hipSuccess || hipMalloc(&db,b.size()*4)!=hipSuccess ||
        hipMalloc(&dc,c.size()*2)!=hipSuccess || hipMalloc(&dd,f16Reference.size()*2)!=hipSuccess ||
        hipMalloc(&dmi,movInput.size()*4)!=hipSuccess || hipMalloc(&dmo,movReference.size()*4)!=hipSuccess) return 5;
    if (hipMemcpy(da,a.data(),a.size()*4,hipMemcpyHostToDevice)!=hipSuccess ||
        hipMemcpy(db,b.data(),b.size()*4,hipMemcpyHostToDevice)!=hipSuccess ||
        hipMemcpy(dc,c.data(),c.size()*2,hipMemcpyHostToDevice)!=hipSuccess ||
        hipMemcpy(dmi,movInput.data(),movInput.size()*4,hipMemcpyHostToDevice)!=hipSuccess) return 6;
    hipLaunchKernelGGL(EmulateF16M16N8K16,dim3(4),dim3(256),0,0,da,db,dc,dd,kCases);
    hipLaunchKernelGGL(EmulateMovMatrix,dim3(1),dim3(256),0,0,dmi,dmo,kCases);
    if(hipGetLastError()!=hipSuccess||hipDeviceSynchronize()!=hipSuccess)return 7;
    std::vector<uint16_t> f16Output(1024);std::vector<uint32_t> movOutput(256);
    if(hipMemcpy(f16Output.data(),dd,f16Output.size()*2,hipMemcpyDeviceToHost)!=hipSuccess||
       hipMemcpy(movOutput.data(),dmo,movOutput.size()*4,hipMemcpyDeviceToHost)!=hipSuccess)return 8;
    for(void* ptr:{(void*)dmo,(void*)dmi,(void*)dd,(void*)dc,(void*)db,(void*)da})(void)hipFree(ptr);

    uint32_t f16Case[8]={};uint64_t f16Mismatch=0,movMismatch=0;
    for(size_t i=0;i<f16Reference.size();++i)if(f16Output[i]!=f16Reference[i]){++f16Mismatch;++f16Case[i/128];}
    for(size_t i=0;i<movReference.size();++i)if(movOutput[i]!=movReference[i])++movMismatch;
    const uint64_t stableMismatch=f16Case[0]+f16Case[1]+f16Case[2]+f16Case[7];
    const uint64_t stressMismatch=f16Case[3]+f16Case[4]+f16Case[5]+f16Case[6];
    const bool negative=(uint16_t(f16Output[0]^1)!=f16Reference[0])&&((movOutput[0]^1)!=movReference[0]);
    const bool pass=stableMismatch==0&&movMismatch==0&&negative;
    std::printf("[%s] device=%s f16_stable_mismatches=%llu f16_stress_mismatches=%llu mov_mismatches=%llu\n",
        pass?"PASS":"FAIL",properties.name,(unsigned long long)stableMismatch,
        (unsigned long long)stressMismatch,(unsigned long long)movMismatch);
    std::ofstream json(argv[7],std::ios::binary);
    json<<"{\n  \"schema\": 1,\n  \"experiment\": \"amd_remaining_n0_primitives_rtx_oracle\",\n"
        <<"  \"status\": \""<<(pass?"PASS":"FAIL")<<"\",\n  \"classification\": \"NUMERICAL_PRIMITIVE\",\n"
        <<"  \"counts_as_s6\": false,\n  \"hip_device\": \""<<properties.name<<"\",\n"
        <<"  \"f16_total_mismatches\": "<<f16Mismatch<<",\n  \"f16_stable_mismatches\": "<<stableMismatch<<",\n"
        <<"  \"f16_overflow_stress_mismatches\": "<<stressMismatch<<",\n  \"f16_case_mismatches\": [";
    for(int i=0;i<8;++i){if(i)json<<", ";json<<f16Case[i];}
    json<<"],\n  \"f16_stable_cases\": [0, 1, 2, 7],\n"
        <<"  \"overflow_stress_policy\": \"reported_not_gated_ptx_accumulation_order_unspecified\",\n"
        <<"  \"movmatrix_words\": 256,\n  \"movmatrix_mismatches\": "<<movMismatch<<",\n"
        <<"  \"negative_verifier_detected\": "<<(negative?"true":"false")<<"\n}\n";
    return pass?0:1;
}
