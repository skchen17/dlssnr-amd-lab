#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

namespace fs = std::filesystem;

constexpr size_t kSkipArenaOffset = 110592;
constexpr size_t kHeadArenaOffset = 147429888;
constexpr uint32_t kWidth = 160;
constexpr uint32_t kHeight = 96;
constexpr uint32_t kInputChannels = 512;
constexpr uint32_t kOutputChannels = 16;
constexpr size_t kSkipBytes = size_t(kWidth) * kHeight * kInputChannels;
constexpr size_t kWeightBytes = size_t(kInputChannels) * kOutputChannels;

__host__ __device__ static float DecodeE4M3(uint8_t bits) {
    const uint8_t magnitude = bits & 0x7f;
    const int exponent = magnitude >> 3;
    const int mantissa = magnitude & 7;
    float value;
    if (exponent == 0) value = ldexpf(float(mantissa), -9);
    else if (magnitude == 0x7f) value = NAN;
    else value = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    return (bits & 0x80) ? -value : value;
}

// Exact inverse of the PTX m16n8k32 B-fragment mapping. Each K=32,N=16 tile
// is stored as 32 lanes x 16 bytes (two adjacent N=8 fragments).
__host__ __device__ static size_t PackedWeightIndex(uint32_t inputChannel,
                                                     uint32_t outputChannel) {
    const uint32_t kBlock = inputChannel / 32;
    const uint32_t row = inputChannel % 32;
    const uint32_t column = outputChannel % 8;
    const uint32_t lane = column * 4 + (row % 16) / 4;
    const uint32_t element = (row % 4) + (row >= 16 ? 4 : 0);
    const uint32_t nHalf = outputChannel / 8;
    return size_t(kBlock) * 512 + size_t(lane) * 16 + nHalf * 8 + element;
}

__global__ static void Project512To16(const uint8_t* input, const uint8_t* weights,
                                      uint16_t* output, size_t values) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= values) return;
    const size_t position = index / kOutputChannels;
    const uint32_t outputChannel = uint32_t(index % kOutputChannels);
    float accumulator = 0.0f;
    for (uint32_t inputChannel = 0; inputChannel < kInputChannels; ++inputChannel) {
        const float a = DecodeE4M3(input[position * kInputChannels + inputChannel]);
        const float b = DecodeE4M3(weights[PackedWeightIndex(inputChannel, outputChannel)]);
        accumulator += a * b;
    }
    output[index] = __half_as_ushort(__float2half_rn(accumulator));
}

static std::vector<uint8_t> ReadSlice(const fs::path& path, size_t offset, size_t bytes) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const size_t size = size_t(stream.tellg());
    if (offset > size || bytes > size - offset)
        throw std::runtime_error("slice exceeds " + path.string());
    stream.seekg(std::streamoff(offset));
    std::vector<uint8_t> result(bytes);
    if (!stream.read(reinterpret_cast<char*>(result.data()), std::streamsize(bytes)))
        throw std::runtime_error("cannot read " + path.string());
    return result;
}

static void WriteFile(const fs::path& path, const void* data, size_t bytes) {
    if (path.has_parent_path()) fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(static_cast<const char*>(data), std::streamsize(bytes)))
        throw std::runtime_error("cannot write " + path.string());
}

static std::string Escape(const std::string& value) {
    std::string result;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        result.push_back(ch);
    }
    return result;
}

static uint64_t Fnv1a64(const void* data, size_t bytes) {
    const auto* raw = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= raw[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 6) {
            std::fprintf(stderr, "usage: output_head_projection <activation_arena.raw> "
                                 "<model_arena.raw> <output.raw> <result.json> [iterations]\n");
            return 2;
        }
        const fs::path activationPath = argv[1];
        const fs::path modelPath = argv[2];
        const fs::path outputPath = argv[3];
        const fs::path jsonPath = argv[4];
        const uint32_t iterations = argc == 6 ? uint32_t(std::stoul(argv[5])) : 50;
        if (!iterations) throw std::runtime_error("iterations must be positive");

        const std::vector<uint8_t> input = ReadSlice(activationPath, kSkipArenaOffset, kSkipBytes);
        const std::vector<uint8_t> weights = ReadSlice(modelPath, kHeadArenaOffset, kWeightBytes);
        const size_t outputValues = size_t(kWidth) * kHeight * kOutputChannels;
        const size_t outputBytes = outputValues * sizeof(uint16_t);
        std::vector<uint16_t> output(outputValues), repeat(outputValues);

        hipDeviceProp_t properties{};
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        uint8_t *deviceInput = nullptr, *deviceWeights = nullptr;
        uint16_t* deviceOutput = nullptr;
        HIP_CHECK(hipMalloc(&deviceInput, input.size()));
        HIP_CHECK(hipMalloc(&deviceWeights, weights.size()));
        HIP_CHECK(hipMalloc(&deviceOutput, outputBytes));
        HIP_CHECK(hipMemcpy(deviceInput, input.data(), input.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(deviceWeights, weights.data(), weights.size(), hipMemcpyHostToDevice));
        const dim3 block(256);
        const dim3 grid(uint32_t((outputValues + block.x - 1) / block.x));

        Project512To16<<<grid, block>>>(deviceInput, deviceWeights, deviceOutput, outputValues);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), deviceOutput, outputBytes, hipMemcpyDeviceToHost));
        Project512To16<<<grid, block>>>(deviceInput, deviceWeights, deviceOutput, outputValues);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(repeat.data(), deviceOutput, outputBytes, hipMemcpyDeviceToHost));
        const bool deterministic = output == repeat;

        hipEvent_t start{}, stop{};
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (uint32_t i = 0; i < iterations; ++i)
            Project512To16<<<grid, block>>>(deviceInput, deviceWeights, deviceOutput, outputValues);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float totalMs = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&totalMs, start, stop));

        size_t nonFinite = 0, zeros = 0;
        double sum = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint16_t bits : output) {
            const float value = __half2float(__ushort_as_half(bits));
            if (!std::isfinite(value)) { ++nonFinite; continue; }
            if (value == 0.0f) ++zeros;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            sum += value;
        }

        size_t sampleMismatches = 0;
        float maximumSampleError = 0.0f;
        constexpr size_t kSamplePositions = 16;
        for (size_t position = 0; position < kSamplePositions; ++position) {
            for (uint32_t outputChannel = 0; outputChannel < kOutputChannels; ++outputChannel) {
                float expected = 0.0f;
                for (uint32_t inputChannel = 0; inputChannel < kInputChannels; ++inputChannel) {
                    expected += DecodeE4M3(input[position * kInputChannels + inputChannel]) *
                                DecodeE4M3(weights[PackedWeightIndex(inputChannel, outputChannel)]);
                }
                const float actual = __half2float(__ushort_as_half(output[position * kOutputChannels + outputChannel]));
                const float error = std::fabs(actual - expected);
                maximumSampleError = std::max(maximumSampleError, error);
                if (error > std::max(0.01f, std::fabs(expected) * 0.005f)) ++sampleMismatches;
            }
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(deviceOutput));
        HIP_CHECK(hipFree(deviceWeights));
        HIP_CHECK(hipFree(deviceInput));

        WriteFile(outputPath, output.data(), outputBytes);
        const bool pass = deterministic && nonFinite == 0 && zeros < outputValues && sampleMismatches == 0;
        char hashText[32]{};
        std::snprintf(hashText, sizeof(hashText), "%016llX",
                      static_cast<unsigned long long>(Fnv1a64(output.data(), outputBytes)));
        const double mean = sum / double(outputValues - nonFinite);
        const float averageMs = totalMs / float(iterations);
        const std::string json =
            "{\n"
            "  \"schema\": 1,\n"
            "  \"experiment\": \"amd_output_head_candidate_512_to_16_projection\",\n"
            "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
            "  \"classification\": \"NATIVE_PACKED_WEIGHT_PROJECTION_NOT_FULL_HEAD_PARITY\",\n"
            "  \"device\": \"" + Escape(properties.name) + "\",\n"
            "  \"input_shape\": [96, 160, 512],\n"
            "  \"output_shape\": [96, 160, 16],\n"
            "  \"weight_shape\": [512, 16],\n"
            "  \"weight_layout\": \"ptx_b_fragment_packed_k32_n16_exact\",\n"
            "  \"input_layout\": \"linear_512_candidate_before_head_normalization\",\n"
            "  \"output_storage\": \"fp16\",\n"
            "  \"iterations\": " + std::to_string(iterations) + ",\n"
            "  \"average_gpu_ms\": " + std::to_string(averageMs) + ",\n"
            "  \"deterministic_repeat\": " + std::string(deterministic ? "true" : "false") + ",\n"
            "  \"non_finite_values\": " + std::to_string(nonFinite) + ",\n"
            "  \"zero_values\": " + std::to_string(zeros) + ",\n"
            "  \"minimum\": " + std::to_string(minimum) + ",\n"
            "  \"maximum\": " + std::to_string(maximum) + ",\n"
            "  \"mean\": " + std::to_string(mean) + ",\n"
            "  \"cpu_sample_values\": 256,\n"
            "  \"cpu_sample_mismatches\": " + std::to_string(sampleMismatches) + ",\n"
            "  \"maximum_cpu_sample_absolute_error\": " + std::to_string(maximumSampleError) + ",\n"
            "  \"output_fnv1a64\": \"" + hashText + "\",\n"
            "  \"full_head_parity_claimed\": false,\n"
            "  \"next_gate\": \"Reconstruct the pre-MMA normalization and compare the first RTX MMA inputs\"\n"
            "}\n";
        WriteFile(jsonPath, json.data(), json.size());
        std::printf("[%s] %s candidate 512->16 %.6f ms, range=[%.6f, %.6f], cpu_mismatches=%zu\n",
                    pass ? "PASS" : "FAIL", properties.name, averageMs, minimum, maximum,
                    sampleMismatches);
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
