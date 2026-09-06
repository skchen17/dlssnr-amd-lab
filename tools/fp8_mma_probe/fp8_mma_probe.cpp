#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

__device__ static float DecodeE4M3(uint8_t bits) {
    const uint8_t magnitude = bits & 0x7f;
    const int exponent = magnitude >> 3;
    const int mantissa = magnitude & 7;
    float value;
    if (exponent == 0) value = ldexpf(float(mantissa), -9);
    else if (magnitude == 0x7f) value = nanf("");
    else value = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    return (bits & 0x80) ? -value : value;
}

__device__ static uint8_t FragmentByte(const uint32_t* data, size_t laneBase,
                                       int element) {
    return uint8_t(data[laneBase + size_t(element / 4)] >> ((element & 3) * 8));
}

__global__ static void EmulateM16N8K32(const uint32_t* a, const uint32_t* b,
                                       const uint16_t* c, uint16_t* d,
                                       uint32_t cases) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(cases) * 32 * 4) return;
    const uint32_t cs = uint32_t(index / (32 * 4));
    const uint32_t local = uint32_t(index % (32 * 4));
    const uint32_t lane = local / 4;
    const uint32_t ci = local & 3;
    const uint32_t group = lane >> 2;
    const uint32_t threadInGroup = lane & 3;
    const uint32_t row = group + (ci >= 2 ? 8 : 0);
    const uint32_t col = threadInGroup * 2 + (ci & 1);
    float accumulator = __half2float(__ushort_as_half(c[index]));
    for (uint32_t k = 0; k < 32; ++k) {
        const uint32_t aThread = (k & 15) >> 2;
        const uint32_t aLane = group * 4 + aThread;
        const uint32_t ai = (k & 3) + (row >= 8 ? 4 : 0) + (k >= 16 ? 8 : 0);
        const size_t aBase = (size_t(cs) * 32 + aLane) * 4;
        const uint32_t bThread = (k & 15) >> 2;
        const uint32_t bLane = col * 4 + bThread;
        const uint32_t bi = (k & 3) + (k >= 16 ? 4 : 0);
        const size_t bBase = (size_t(cs) * 32 + bLane) * 2;
        accumulator += DecodeE4M3(FragmentByte(a, aBase, ai)) *
                       DecodeE4M3(FragmentByte(b, bBase, bi));
    }
    d[index] = __half_as_ushort(__float2half_rn(accumulator));
}

template <typename T>
static std::vector<T> ReadVector(const fs::path& path, size_t expectedCount) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || size_t(file.tellg()) != expectedCount * sizeof(T)) return {};
    file.seekg(0);
    std::vector<T> result(expectedCount);
    if (!file.read(reinterpret_cast<char*>(result.data()),
                   std::streamsize(result.size() * sizeof(T))))
        return {};
    return result;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: fp8_mma_probe <a.raw> <b.raw> <c.raw> "
                             "<d-reference.raw> <result.json>\n");
        return 2;
    }
    constexpr uint32_t kCases = 8;
    const std::vector<uint32_t> a = ReadVector<uint32_t>(argv[1], kCases * 32 * 4);
    const std::vector<uint32_t> b = ReadVector<uint32_t>(argv[2], kCases * 32 * 2);
    const std::vector<uint16_t> c = ReadVector<uint16_t>(argv[3], kCases * 32 * 4);
    const std::vector<uint16_t> reference = ReadVector<uint16_t>(argv[4], kCases * 32 * 4);
    if (a.empty() || b.empty() || c.empty() || reference.empty()) return 3;
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 4;

    uint32_t *deviceA = nullptr, *deviceB = nullptr;
    uint16_t *deviceC = nullptr, *deviceD = nullptr;
    if (hipMalloc(&deviceA, a.size() * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&deviceB, b.size() * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&deviceC, c.size() * sizeof(uint16_t)) != hipSuccess ||
        hipMalloc(&deviceD, reference.size() * sizeof(uint16_t)) != hipSuccess)
        return 5;
    if (hipMemcpy(deviceA, a.data(), a.size() * sizeof(uint32_t), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(deviceB, b.data(), b.size() * sizeof(uint32_t), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(deviceC, c.data(), c.size() * sizeof(uint16_t), hipMemcpyHostToDevice) != hipSuccess)
        return 6;
    hipLaunchKernelGGL(EmulateM16N8K32, dim3(4), dim3(256), 0, 0,
                       deviceA, deviceB, deviceC, deviceD, kCases);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        return 7;
    std::vector<uint16_t> output(reference.size());
    if (hipMemcpy(output.data(), deviceD, output.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return 8;
    (void)hipFree(deviceD);
    (void)hipFree(deviceC);
    (void)hipFree(deviceB);
    (void)hipFree(deviceA);

    uint64_t mismatches = 0;
    int64_t firstMismatch = -1, lastMismatch = -1;
    uint32_t caseMismatches[kCases] = {};
    for (size_t i = 0; i < reference.size(); ++i) {
        if (output[i] == reference[i]) continue;
        if (firstMismatch < 0) firstMismatch = int64_t(i);
        lastMismatch = int64_t(i);
        ++mismatches;
        ++caseMismatches[i / (32 * 4)];
    }
    const bool negativeDetected = (uint16_t(output[0] ^ 1) != reference[0]);
    const bool pass = mismatches == 0 && negativeDetected;
    std::printf("[%s] device=%s cases=8 half_outputs=%zu mismatches=%llu\n",
                pass ? "PASS" : "FAIL", properties.name, output.size(),
                (unsigned long long)mismatches);
    std::ofstream json(argv[5], std::ios::binary);
    json << "{\n"
         << "  \"schema\": 1,\n"
         << "  \"experiment\": \"amd_m16n8k32_e4m3_mma_rtx_oracle\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"classification\": \"MMA_NUMERICAL_PRIMITIVE\",\n"
         << "  \"counts_as_s6\": false,\n"
         << "  \"hip_device\": \"" << properties.name << "\",\n"
         << "  \"cases\": 8,\n"
         << "  \"half_outputs\": " << output.size() << ",\n"
         << "  \"mismatches\": " << mismatches << ",\n"
         << "  \"first_mismatch\": " << firstMismatch << ",\n"
         << "  \"last_mismatch\": " << lastMismatch << ",\n"
         << "  \"case_mismatches\": [";
    for (uint32_t cs = 0; cs < kCases; ++cs) {
        if (cs) json << ", ";
        json << caseMismatches[cs];
    }
    json << "],\n"
         << "  \"accumulation\": \"fp32_sequential_then_fp16_rn\",\n"
         << "  \"negative_verifier_detected\": "
         << (negativeDetected ? "true" : "false") << "\n"
         << "}\n";
    return pass ? 0 : 1;
}
