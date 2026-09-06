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

constexpr uint32_t kGridWidth = 81, kGridHeight = 49;
constexpr uint32_t kGroups = 4, kWeightBlocks = 2, kRows = 16, kColumns = 16;
constexpr size_t kValuesPerCta = size_t(kGroups) * kWeightBlocks * kRows * kColumns;
constexpr size_t kCtas = size_t(kGridWidth) * kGridHeight;
constexpr size_t kInputBytes = kCtas * 4 * 32 * 16;
constexpr size_t kOutputValues = kCtas * kValuesPerCta;
constexpr size_t kHeadOffset = 147429888;
constexpr size_t kProjectionWeightBytes = 1024;

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

__device__ static size_t PackedAIndex(uint32_t group, uint32_t row, uint32_t k) {
    const uint32_t lane = (row % 8) * 4 + (k % 16) / 4;
    const uint32_t element = (k % 4) + (row >= 8 ? 4 : 0) + (k >= 16 ? 8 : 0);
    return size_t(group) * 512 + lane * 16 + element;
}

__device__ static size_t PackedBIndex(uint32_t weightBlock, uint32_t k, uint32_t column) {
    const uint32_t lane = (column % 8) * 4 + (k % 16) / 4;
    const uint32_t element = (k % 4) + (k >= 16 ? 4 : 0);
    return size_t(weightBlock) * 512 + lane * 16 + (column / 8) * 8 + element;
}

__global__ static void Project(const uint8_t* packedA, const uint8_t* packedB,
                               uint16_t* output) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputValues) return;
    const size_t cta = index / kValuesPerCta;
    uint32_t local = uint32_t(index % kValuesPerCta);
    const uint32_t column = local % kColumns; local /= kColumns;
    const uint32_t row = local % kRows; local /= kRows;
    const uint32_t weightBlock = local % kWeightBlocks;
    const uint32_t group = local / kWeightBlocks;
    const uint8_t* a = packedA + cta * 2048;
    float accumulator = 0.0f;
    for (uint32_t k = 0; k < 32; ++k)
        accumulator += DecodeE4M3(a[PackedAIndex(group, row, k)]) *
                       DecodeE4M3(packedB[PackedBIndex(weightBlock, k, column)]);
    output[index] = __half_as_ushort(__float2half_rn(accumulator));
}

static std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const size_t bytes = size_t(stream.tellg()); stream.seekg(0);
    std::vector<uint8_t> result(bytes);
    if (!stream.read(reinterpret_cast<char*>(result.data()), std::streamsize(bytes)))
        throw std::runtime_error("cannot read " + path.string());
    return result;
}

static std::vector<uint8_t> ReadSlice(const std::filesystem::path& path,
                                      size_t offset, size_t bytes) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const size_t size = size_t(stream.tellg());
    if (offset > size || bytes > size - offset) throw std::runtime_error("slice exceeds file");
    stream.seekg(std::streamoff(offset));
    std::vector<uint8_t> result(bytes);
    if (!stream.read(reinterpret_cast<char*>(result.data()), std::streamsize(bytes)))
        throw std::runtime_error("cannot read " + path.string());
    return result;
}

static void WriteFile(const std::filesystem::path& path, const void* data, size_t bytes) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
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
    for (size_t i = 0; i < bytes; ++i) { hash ^= raw[i]; hash *= 1099511628211ull; }
    return hash;
}

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 7) {
            std::fprintf(stderr, "usage: output_head_first_projection <full_grid_a_e4m3.raw> "
                                 "<model_arena.raw> <rtx_mma_trace.raw> <output_fp16.raw> "
                                 "<result.json> [iterations]\n");
            return 2;
        }
        const uint32_t iterations = argc == 7 ? uint32_t(std::stoul(argv[6])) : 100;
        if (!iterations) throw std::runtime_error("iterations must be positive");
        const auto input = ReadFile(argv[1]);
        if (input.size() != kInputBytes) throw std::runtime_error("unexpected full-grid A size");
        const auto weights = ReadSlice(argv[2], kHeadOffset, kProjectionWeightBytes);
        const auto trace = ReadFile(argv[3]);
        if (trace.size() < 16 * 32 * 40) throw std::runtime_error("RTX trace is too short");
        std::vector<uint16_t> output(kOutputValues), repeat(kOutputValues);

        uint8_t *dInput = nullptr, *dWeights = nullptr;
        uint16_t* dOutput = nullptr;
        HIP_CHECK(hipMalloc(&dInput, input.size()));
        HIP_CHECK(hipMalloc(&dWeights, weights.size()));
        HIP_CHECK(hipMalloc(&dOutput, output.size() * 2));
        HIP_CHECK(hipMemcpy(dInput, input.data(), input.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dWeights, weights.data(), weights.size(), hipMemcpyHostToDevice));
        const dim3 block(256), grid(uint32_t((kOutputValues + 255) / 256));
        Project<<<grid, block>>>(dInput, dWeights, dOutput);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), dOutput, output.size() * 2, hipMemcpyDeviceToHost));
        Project<<<grid, block>>>(dInput, dWeights, dOutput);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(repeat.data(), dOutput, repeat.size() * 2, hipMemcpyDeviceToHost));

        hipEvent_t start{}, stop{}; HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (uint32_t i = 0; i < iterations; ++i) Project<<<grid, block>>>(dInput, dWeights, dOutput);
        HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
        float totalMs = 0.0f; HIP_CHECK(hipEventElapsedTime(&totalMs, start, stop));

        constexpr size_t selectedCta = size_t(26) * kGridWidth + 70;
        uint64_t selectedMismatches = 0;
        int64_t firstSelectedMismatch = -1;
        for (uint32_t group = 0; group < 4; ++group) {
            for (uint32_t weightBlock = 0; weightBlock < 2; ++weightBlock) {
                for (uint32_t row = 0; row < 16; ++row) {
                    for (uint32_t column = 0; column < 16; ++column) {
                        const uint32_t mma = group * 4 + weightBlock * 2 + column / 8;
                        const uint32_t lane = (row % 8) * 4 + (column % 8) / 2;
                        const uint32_t element = (row >= 8 ? 2 : 0) + (column & 1);
                        const size_t traceRecord = (size_t(mma) * 32 + lane) * 40;
                        const uint32_t word = uint32_t(trace[traceRecord + 32 + (element / 2) * 4]) |
                            (uint32_t(trace[traceRecord + 33 + (element / 2) * 4]) << 8) |
                            (uint32_t(trace[traceRecord + 34 + (element / 2) * 4]) << 16) |
                            (uint32_t(trace[traceRecord + 35 + (element / 2) * 4]) << 24);
                        const uint16_t expected = uint16_t(word >> (16 * (element & 1)));
                        const size_t local = (((size_t(group) * 2 + weightBlock) * 16 + row) * 16 + column);
                        const size_t outputIndex = selectedCta * kValuesPerCta + local;
                        if (output[outputIndex] != expected) {
                            if (firstSelectedMismatch < 0) firstSelectedMismatch = int64_t(local);
                            ++selectedMismatches;
                        }
                    }
                }
            }
        }
        uint64_t nonFinite = 0, nonzero = 0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint16_t bits : output) {
            const float value = __half2float(__ushort_as_half(bits));
            nonFinite += !std::isfinite(value); nonzero += value != 0.0f;
            if (std::isfinite(value)) { minimum = std::min(minimum, value); maximum = std::max(maximum, value); }
        }
        const bool deterministic = output == repeat;
        const bool pass = deterministic && selectedMismatches == 0 && nonFinite == 0 && nonzero != 0;
        WriteFile(argv[4], output.data(), output.size() * 2);
        char hashText[32]{}; std::snprintf(hashText, sizeof(hashText), "%016llX",
            static_cast<unsigned long long>(Fnv1a64(output.data(), output.size() * 2)));
        hipDeviceProp_t properties{}; HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(dOutput)); HIP_CHECK(hipFree(dWeights)); HIP_CHECK(hipFree(dInput));
        const std::string json =
            "{\n"
            "  \"schema\": 1,\n"
            "  \"experiment\": \"amd_output_head_first_projection_full_grid\",\n"
            "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
            "  \"classification\": \"NATIVE_RAW_ACTIVATION_TO_FIRST_FP8_PROJECTION\",\n"
            "  \"device\": \"" + Escape(properties.name) + "\",\n"
            "  \"input_shape\": [49, 81, 4, 32, 16],\n"
            "  \"output_shape\": [49, 81, 4, 2, 16, 16],\n"
            "  \"output_storage\": \"fp16\",\n"
            "  \"output_values\": " + std::to_string(kOutputValues) + ",\n"
            "  \"selected_cta\": [70, 26, 0],\n"
            "  \"selected_cta_rtx_half_values\": 2048,\n"
            "  \"selected_cta_rtx_mismatches\": " + std::to_string(selectedMismatches) + ",\n"
            "  \"first_selected_cta_mismatch\": " + std::to_string(firstSelectedMismatch) + ",\n"
            "  \"deterministic_repeat\": " + std::string(deterministic ? "true" : "false") + ",\n"
            "  \"non_finite_values\": " + std::to_string(nonFinite) + ",\n"
            "  \"nonzero_values\": " + std::to_string(nonzero) + ",\n"
            "  \"minimum\": " + std::to_string(minimum) + ",\n"
            "  \"maximum\": " + std::to_string(maximum) + ",\n"
            "  \"iterations\": " + std::to_string(iterations) + ",\n"
            "  \"average_gpu_ms\": " + std::to_string(totalMs / float(iterations)) + ",\n"
            "  \"output_fnv1a64\": \"" + hashText + "\",\n"
            "  \"covered_original_mmas\": [0, 15],\n"
            "  \"next_gate\": \"Reconstruct the normalization and accumulator inputs for FP8 MMAs 16 through 127\"\n"
            "}\n";
        WriteFile(argv[5], json.data(), json.size());
        std::printf("[%s] %s full-grid first projection: selected mismatches=%llu, %.6f ms\n",
                    pass ? "PASS" : "FAIL", properties.name,
                    static_cast<unsigned long long>(selectedMismatches), totalMs / float(iterations));
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
