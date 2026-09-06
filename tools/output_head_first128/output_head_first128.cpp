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
constexpr uint32_t kRows = 16, kColumns = 32, kHiddenPasses = 4;
constexpr size_t kCtas = size_t(kGridWidth) * kGridHeight;
constexpr size_t kValuesPerCta = size_t(kGroups) * kRows * kColumns;
constexpr size_t kRawValues = kCtas * kValuesPerCta;
constexpr size_t kHiddenValues = kRawValues * kHiddenPasses;
constexpr size_t kHeadOffset = 147429888, kHeadBytes = 21808;
constexpr size_t kSelectedCta = size_t(26) * kGridWidth + 70;
constexpr size_t kSelectedStageValues = kHiddenPasses * 2 * kValuesPerCta;

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

__device__ static uint32_t RoundShift(uint32_t value, uint32_t shift) {
    const uint32_t base = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    return base + uint32_t(remainder > halfway || (remainder == halfway && (base & 1u)));
}

__device__ static uint8_t EncodeHalfBitsE4M3(uint16_t bits) {
    const uint8_t sign = uint8_t((bits >> 8) & 0x80);
    const uint32_t exponent = (bits >> 10) & 31;
    const uint32_t mantissa = bits & 1023;
    if (exponent == 31 && mantissa) return 0x7f;
    uint32_t code;
    if (exponent < 9) code = min(RoundShift(1024 + mantissa, 16 - exponent), 8u);
    else {
        uint32_t rounded = RoundShift(mantissa, 7);
        const bool carry = rounded >= 8;
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

__device__ static size_t PackedBIndex(size_t tileOffset, uint32_t k, uint32_t column) {
    const uint32_t lane = (column % 8) * 4 + (k % 16) / 4;
    const uint32_t element = (k % 4) + (k >= 16 ? 4 : 0);
    return tileOffset + lane * 16 + (column / 8) * 8 + element;
}

__device__ static uint16_t HalfRound(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__device__ static float HalfValue(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ static uint16_t Activate(uint16_t xBits) {
    const float x = HalfValue(xBits);
    const float clamped = fmaxf(-4.0f, fminf(x, 4.0f));
    const float absolute = fabsf(clamped);
    const float first = HalfValue(HalfRound(fmaf(-0.055908203125f, absolute, 0.447265625f)));
    const float second = HalfValue(HalfRound(fmaf(clamped, first, 0.89453125f)));
    return HalfRound(x * second);
}

__global__ static void Hidden(const uint8_t* rawA, const uint8_t* head,
                              uint8_t* hiddenE4, uint16_t* selectedStages) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kHiddenValues) return;
    size_t split = index;
    const uint32_t column = uint32_t(split % kColumns); split /= kColumns;
    const uint32_t row = uint32_t(split % kRows); split /= kRows;
    const uint32_t group = uint32_t(split % kGroups); split /= kGroups;
    const uint32_t pass = uint32_t(split % kHiddenPasses);
    const size_t cta = split / kHiddenPasses;
    const uint8_t* a = rawA + cta * kValuesPerCta;
    const size_t tile = size_t(pass) * 1024 + (column / 16) * 512;
    float accumulator = 0.0f;
    for (uint32_t k = 0; k < 32; ++k)
        accumulator += DecodeE4M3(a[PackedAIndex(group, row, k)]) *
                       DecodeE4M3(head[PackedBIndex(tile, k, column % 16)]);
    const uint16_t hidden = HalfRound(accumulator);
    const uint8_t activated = EncodeHalfBitsE4M3(Activate(hidden));
    const uint32_t mmaInGroup = (column / 16) * 2 + (column % 16) / 8;
    const uint32_t columnWithinEight = column % 8;
    const uint32_t lane = (row % 8) * 4 + columnWithinEight / 2;
    const uint32_t dElement = (row >= 8 ? 2 : 0) + (columnWithinEight & 1);
    const uint32_t conversion = (mmaInGroup / 2) * 4 + (mmaInGroup % 2) +
                                (dElement / 2) * 2;
    const uint32_t packedElement = conversion * 2 + (dElement & 1);
    const size_t packedLocal = size_t(group) * 512 + lane * 16 + packedElement;
    hiddenE4[(cta * kHiddenPasses + pass) * kValuesPerCta + packedLocal] = activated;
    if (cta == kSelectedCta) {
        const size_t local = (size_t(group) * kRows + row) * kColumns + column;
        selectedStages[(size_t(pass) * 2) * kValuesPerCta + local] = hidden;
    }
}

__device__ static uint16_t ResidualSeed(const uint16_t* rawHalf, const uint8_t* head,
                                        uint32_t group, uint32_t row, uint32_t column) {
    const uint32_t mmaInGroup = (column / 16) * 2 + (column % 16) / 8;
    const uint32_t operationPairs[4][2] = {{0, 2}, {1, 3}, {4, 6}, {5, 7}};
    const uint32_t lane = (row % 8) * 4 + (column % 8) / 2;
    const uint32_t element = (row >= 8 ? 2 : 0) + (column & 1);
    const uint32_t operation = operationPairs[mmaInGroup][element / 2];
    const uint32_t pairHalf = element & 1;
    const size_t rawIndex = size_t(group) * 512 + lane * 16 + operation * 2 + pairHalf;
    const uint32_t scaleWord = mmaInGroup * 4 + lane % 4;
    const size_t scaleIndex = 8208 + scaleWord * 4 + pairHalf * 2;
    const uint16_t scaleBits = uint16_t(head[scaleIndex]) | (uint16_t(head[scaleIndex + 1]) << 8);
    return HalfRound(HalfValue(rawHalf[rawIndex]) * HalfValue(scaleBits));
}

__global__ static void Project(const uint16_t* rawHalf, const uint8_t* hiddenE4,
                               const uint8_t* head, uint16_t* output,
                               uint16_t* selectedStages) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kRawValues) return;
    size_t split = index;
    const uint32_t column = uint32_t(split % kColumns); split /= kColumns;
    const uint32_t row = uint32_t(split % kRows); split /= kRows;
    const uint32_t group = uint32_t(split % kGroups);
    const size_t cta = split / kGroups;
    uint16_t accumulatorBits = ResidualSeed(rawHalf + cta * kValuesPerCta,
                                            head, group, row, column);
    const size_t local = (size_t(group) * kRows + row) * kColumns + column;
    for (uint32_t pass = 0; pass < kHiddenPasses; ++pass) {
        float accumulator = HalfValue(accumulatorBits);
        const size_t hiddenBase = (cta * kHiddenPasses + pass) * kValuesPerCta;
        const size_t tile = 4096 + size_t(pass) * 1024 + (column / 16) * 512;
        for (uint32_t k = 0; k < 32; ++k) {
            const size_t hiddenIndex = hiddenBase + PackedAIndex(group, row, k);
            accumulator += DecodeE4M3(hiddenE4[hiddenIndex]) *
                           DecodeE4M3(head[PackedBIndex(tile, k, column % 16)]);
        }
        accumulatorBits = HalfRound(accumulator);
        if (cta == kSelectedCta)
            selectedStages[(size_t(pass) * 2 + 1) * kValuesPerCta + local] = accumulatorBits;
    }
    output[index] = accumulatorBits;
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

static std::vector<uint8_t> ReadSlice(const std::filesystem::path& path, size_t offset, size_t bytes) {
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
    for (char ch : value) { if (ch == '\\' || ch == '"') result.push_back('\\'); result.push_back(ch); }
    return result;
}

static uint64_t Fnv1a64(const void* data, size_t bytes) {
    const auto* raw = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) { hash ^= raw[i]; hash *= 1099511628211ull; }
    return hash;
}

static uint16_t TraceD(const std::vector<uint8_t>& trace, uint32_t mma,
                       uint32_t row, uint32_t columnWithinEight) {
    const uint32_t lane = (row % 8) * 4 + columnWithinEight / 2;
    const uint32_t element = (row >= 8 ? 2 : 0) + (columnWithinEight & 1);
    const size_t offset = (size_t(mma) * 32 + lane) * 40 + 32 + (element / 2) * 4;
    const uint32_t word = uint32_t(trace[offset]) | (uint32_t(trace[offset + 1]) << 8) |
        (uint32_t(trace[offset + 2]) << 16) | (uint32_t(trace[offset + 3]) << 24);
    return uint16_t(word >> (16 * (element & 1)));
}

int main(int argc, char** argv) {
    try {
        if (argc < 7 || argc > 8) {
            std::fprintf(stderr, "usage: output_head_first128 <full_grid_a_e4m3.raw> "
                "<full_grid_a_fp16.raw> <model_arena.raw> <rtx_mma_trace.raw> "
                "<output_fp16.raw> <result.json> [iterations]\n");
            return 2;
        }
        const uint32_t iterations = argc == 8 ? uint32_t(std::stoul(argv[7])) : 20;
        const auto rawA = ReadFile(argv[1]);
        const auto rawHalfBytes = ReadFile(argv[2]);
        const auto head = ReadSlice(argv[3], kHeadOffset, kHeadBytes);
        const auto trace = ReadFile(argv[4]);
        if (rawA.size() != kRawValues || rawHalfBytes.size() != kRawValues * 2)
            throw std::runtime_error("unexpected full-grid activation size");
        if (trace.size() < 128 * 32 * 40) throw std::runtime_error("RTX trace is too short");
        if (!iterations) throw std::runtime_error("iterations must be positive");

        uint8_t *dRawA = nullptr, *dHead = nullptr, *dHidden = nullptr;
        uint16_t *dRawHalf = nullptr, *dOutput = nullptr, *dSelected = nullptr;
        HIP_CHECK(hipMalloc(&dRawA, rawA.size()));
        HIP_CHECK(hipMalloc(&dRawHalf, rawHalfBytes.size()));
        HIP_CHECK(hipMalloc(&dHead, head.size()));
        HIP_CHECK(hipMalloc(&dHidden, kHiddenValues));
        HIP_CHECK(hipMalloc(&dOutput, kRawValues * 2));
        HIP_CHECK(hipMalloc(&dSelected, kSelectedStageValues * 2));
        HIP_CHECK(hipMemcpy(dRawA, rawA.data(), rawA.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRawHalf, rawHalfBytes.data(), rawHalfBytes.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dHead, head.data(), head.size(), hipMemcpyHostToDevice));
        const dim3 block(256), hiddenGrid(uint32_t((kHiddenValues + 255) / 256));
        const dim3 outputGrid(uint32_t((kRawValues + 255) / 256));
        auto launch = [&]() {
            Hidden<<<hiddenGrid, block>>>(dRawA, dHead, dHidden, dSelected);
            Project<<<outputGrid, block>>>(dRawHalf, dHidden, dHead, dOutput, dSelected);
        };
        launch(); HIP_CHECK(hipDeviceSynchronize());
        std::vector<uint16_t> output(kRawValues), repeat(kRawValues), selected(kSelectedStageValues);
        HIP_CHECK(hipMemcpy(output.data(), dOutput, output.size() * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(selected.data(), dSelected, selected.size() * 2, hipMemcpyDeviceToHost));
        launch(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(repeat.data(), dOutput, repeat.size() * 2, hipMemcpyDeviceToHost));
        hipEvent_t start{}, stop{}; HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (uint32_t i = 0; i < iterations; ++i) launch();
        HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
        float totalMs = 0; HIP_CHECK(hipEventElapsedTime(&totalMs, start, stop));

        uint64_t mismatches = 0, stageMismatches[8]{}; int64_t firstMismatch = -1;
        for (uint32_t pass = 0; pass < 4; ++pass)
            for (uint32_t stage = 0; stage < 2; ++stage)
                for (uint32_t group = 0; group < 4; ++group)
                    for (uint32_t row = 0; row < 16; ++row)
                        for (uint32_t column = 0; column < 32; ++column) {
                            const uint32_t base = stage ? 16 + pass * 32 : pass * 32;
                            const uint32_t mma = base + group * 4 + (column / 16) * 2 + (column % 16) / 8;
                            const uint16_t expected = TraceD(trace, mma, row, column % 8);
                            const size_t local = (size_t(group) * 16 + row) * 32 + column;
                            const size_t index = (size_t(pass) * 2 + stage) * kValuesPerCta + local;
                            if (selected[index] != expected) {
                                if (firstMismatch < 0) firstMismatch = int64_t(index);
                                ++mismatches; ++stageMismatches[pass * 2 + stage];
                            }
                        }
        uint64_t nonFinite = 0, nonzero = 0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (uint16_t bits : output) {
            const float value = __half2float(__ushort_as_half(bits));
            nonFinite += !std::isfinite(value); nonzero += value != 0;
            if (std::isfinite(value)) { minimum = std::min(minimum, value); maximum = std::max(maximum, value); }
        }
        const bool deterministic = output == repeat;
        const bool pass = deterministic && mismatches == 0 && nonFinite == 0 && nonzero != 0;
        WriteFile(argv[5], output.data(), output.size() * 2);
        char hashText[32]{}; std::snprintf(hashText, sizeof(hashText), "%016llX",
            static_cast<unsigned long long>(Fnv1a64(output.data(), output.size() * 2)));
        hipDeviceProp_t properties{}; HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        const std::string json =
            "{\n  \"schema\": 1,\n  \"experiment\": \"amd_output_head_first_128_mmas\",\n"
            "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
            "  \"classification\": \"NATIVE_RAW_ACTIVATION_FIRST_128_MMA_PIPELINE\",\n"
            "  \"device\": \"" + Escape(properties.name) + "\",\n"
            "  \"output_shape\": [49, 81, 4, 16, 32],\n"
            "  \"output_storage\": \"fp16\",\n"
            "  \"selected_cta_rtx_half_values\": " + std::to_string(kSelectedStageValues) + ",\n"
            "  \"selected_cta_rtx_mismatches\": " + std::to_string(mismatches) + ",\n"
            "  \"selected_stage_mismatches\": [" +
                std::to_string(stageMismatches[0]) + ", " + std::to_string(stageMismatches[1]) + ", " +
                std::to_string(stageMismatches[2]) + ", " + std::to_string(stageMismatches[3]) + ", " +
                std::to_string(stageMismatches[4]) + ", " + std::to_string(stageMismatches[5]) + ", " +
                std::to_string(stageMismatches[6]) + ", " + std::to_string(stageMismatches[7]) + "],\n"
            "  \"first_selected_cta_mismatch\": " + std::to_string(firstMismatch) + ",\n"
            "  \"deterministic_repeat\": " + std::string(deterministic ? "true" : "false") + ",\n"
            "  \"non_finite_values\": " + std::to_string(nonFinite) + ",\n"
            "  \"nonzero_values\": " + std::to_string(nonzero) + ",\n"
            "  \"minimum\": " + std::to_string(minimum) + ",\n"
            "  \"maximum\": " + std::to_string(maximum) + ",\n"
            "  \"iterations\": " + std::to_string(iterations) + ",\n"
            "  \"average_gpu_ms\": " + std::to_string(totalMs / float(iterations)) + ",\n"
            "  \"output_fnv1a64\": \"" + hashText + "\",\n"
            "  \"covered_original_mmas\": [0, 127],\n"
            "  \"next_gate\": \"Reconstruct FP8 MMAs 128 through 255 and the final FP16 projection\"\n}\n";
        WriteFile(argv[6], json.data(), json.size());
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(dSelected)); HIP_CHECK(hipFree(dOutput)); HIP_CHECK(hipFree(dHidden));
        HIP_CHECK(hipFree(dHead)); HIP_CHECK(hipFree(dRawHalf)); HIP_CHECK(hipFree(dRawA));
        std::printf("[%s] %s first 128 MMAs: selected mismatches=%llu, %.6f ms\n",
            pass ? "PASS" : "FAIL", properties.name,
            static_cast<unsigned long long>(mismatches), totalMs / float(iterations));
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
