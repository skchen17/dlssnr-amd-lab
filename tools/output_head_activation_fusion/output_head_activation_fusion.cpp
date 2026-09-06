#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t e_ = (expr); \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#expr) + ": " + hipGetErrorString(e_)); \
} while (0)

constexpr size_t kMainOffset = 13873152;
constexpr size_t kSkipOffset = 110592;
constexpr size_t kHeadOffset = 147429888;
constexpr size_t kMainBytes = 2 * 192 * 320 * 16;
constexpr size_t kSkipBytes = 96 * 160 * 512;
constexpr size_t kHeadBytes = 21808;
constexpr uint32_t kTargetCtaX = 70;
constexpr uint32_t kTargetCtaY = 26;
constexpr uint32_t kOperations = 32;
constexpr uint32_t kLanes = 32;
constexpr uint32_t kHalves = 2;
constexpr size_t kValues = size_t(kOperations) * kLanes * kHalves;
constexpr uint32_t kGridWidth = 81;
constexpr uint32_t kGridHeight = 49;
constexpr size_t kGridValues = size_t(kGridWidth) * kGridHeight * kValues;

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
    if (exponent < 9) {
        code = min(RoundShift(1024 + mantissa, 16 - exponent), 8u);
    } else {
        uint32_t rounded = RoundShift(mantissa, 7);
        const bool carry = rounded >= 8;
        code = ((exponent - 8 + uint32_t(carry)) << 3) | (carry ? 0 : rounded);
        code = min(code, 0x7eu);
    }
    return sign | uint8_t(code);
}

__device__ static uint32_t ShuffledSourceLane(uint32_t lane, uint32_t output) {
    const uint32_t q = (lane & 19u) | ((lane << 1) & 8u) | ((lane >> 1) & 4u);
    uint32_t sources[4] = {q, q ^ 8u, q ^ 16u, q ^ 24u};
    uint32_t middle[4];
    if (lane & 16u) {
        middle[0] = sources[2]; middle[1] = sources[3];
        middle[2] = sources[0]; middle[3] = sources[1];
    } else {
        for (uint32_t i = 0; i < 4; ++i) middle[i] = sources[i];
    }
    if (!(lane & 4u)) return middle[output];
    const uint32_t swapped[4] = {middle[1], middle[0], middle[3], middle[2]};
    return swapped[output];
}

__device__ static uint16_t FuseValue(const uint8_t* mainFeature,
                                     const uint8_t* skipFeature,
                                     const uint8_t* head,
                                     uint32_t ctaX, uint32_t ctaY,
                                     uint32_t operation, uint32_t lane,
                                     uint32_t pairHalf) {
    const uint32_t cell = operation / 8;
    const uint32_t within = operation % 8;

    const uint32_t sourceSet = operation < 16 ? 0 : 1;
    const uint32_t sourceBases[2][4] = {{0, 1, 4, 5}, {2, 3, 6, 7}};
    const uint32_t source = sourceBases[sourceSet][within / 2];
    const uint32_t parity = (operation % 16) / 8;
    const uint32_t shuffleOutput = parity + (within & 1 ? 2 : 0);
    const uint32_t sourceLane = ShuffledSourceLane(lane, shuffleOutput);
    const uint32_t load = source / 2;
    const uint32_t plane = load / 2;
    const int32_t mainBaseY = (-4 + int32_t(ctaY) * 8) / 2;
    const int32_t mainBaseX = (-4 + int32_t(ctaX) * 8) / 2;
    const int32_t y = mainBaseY + int32_t(sourceLane / 16) + (load & 1 ? 2 : 0);
    const int32_t x = mainBaseX + int32_t((sourceLane / 4) % 4);
    const uint32_t channelGroup = sourceLane % 4;
    uint8_t mainCode = 0;
    if (y >= 0 && y < 192 && x >= 0 && x < 320) {
        const size_t mainIndex =
            ((size_t(plane) * 192 + uint32_t(y)) * 320 + uint32_t(x)) * 16 +
            channelGroup * 4 + (source & 1) * 2 + pairHalf;
        mainCode = mainFeature[mainIndex];
    }

    const uint32_t byteOrder[8] = {0, 4, 2, 6, 8, 12, 10, 14};
    const int32_t skipBaseY = (-4 + int32_t(ctaY) * 8) / 4;
    const int32_t skipBaseX = (-4 + int32_t(ctaX) * 8) / 4;
    const int32_t skipY = skipBaseY + int32_t(cell / 2);
    const int32_t skipX = skipBaseX + int32_t(cell % 2);
    uint8_t skipCode = 0;
    if (skipY >= 0 && skipY < 96 && skipX >= 0 && skipX < 160) {
        const size_t skipIndex =
            (size_t(skipY) * 160 + uint32_t(skipX)) * 512 + lane * 16 +
            byteOrder[within] + pairHalf;
        skipCode = skipFeature[skipIndex];
    }

    const size_t scaleOffset = size_t((operation / 2) % 4) * 16 + (lane % 4) * 4 + pairHalf * 2;
    const uint16_t mainScaleBits = uint16_t(head[8272 + scaleOffset]) |
                                   (uint16_t(head[8273 + scaleOffset]) << 8);
    const uint16_t skipScaleBits = uint16_t(head[8336 + scaleOffset]) |
                                   (uint16_t(head[8337 + scaleOffset]) << 8);
    const float mainValue = DecodeE4M3(mainCode);
    const float skipValue = DecodeE4M3(skipCode);
    const float mainScale = __half2float(__ushort_as_half(mainScaleBits));
    const float skipScale = __half2float(__ushort_as_half(skipScaleBits));

    // This is the output-preserving RTX contract observed in the original
    // kernel: round the main product, then fuse skip*scale + rounded_main.
    const float roundedMain = __half2float(__float2half_rn(mainValue * mainScale));
    return __half_as_ushort(
        __float2half_rn(fmaf(skipValue, skipScale, roundedMain)));
}

__global__ static void Fuse(const uint8_t* mainFeature, const uint8_t* skipFeature,
                            const uint8_t* head, uint16_t* fusedHalf, uint8_t* fusedE4) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kValues) return;
    const uint32_t pairHalf = uint32_t(index & 1);
    const uint32_t record = uint32_t(index >> 1);
    const uint32_t lane = record % kLanes;
    const uint32_t operation = record / kLanes;
    const uint16_t result = FuseValue(mainFeature, skipFeature, head,
                                      kTargetCtaX, kTargetCtaY,
                                      operation, lane, pairHalf);
    fusedHalf[index] = result;
    fusedE4[index] = EncodeHalfBitsE4M3(result);
}

__global__ static void FuseFullGrid(const uint8_t* mainFeature,
                                    const uint8_t* skipFeature,
                                    const uint8_t* head, uint8_t* packedE4,
                                    uint16_t* packedHalf) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kGridValues) return;
    const size_t ctaIndex = index / kValues;
    const uint32_t ctaX = uint32_t(ctaIndex % kGridWidth);
    const uint32_t ctaY = uint32_t(ctaIndex / kGridWidth);
    const uint32_t local = uint32_t(index % kValues);
    const uint32_t pairHalf = local & 1;
    const uint32_t packedWithin = (local >> 1) % 8;
    const uint32_t lane = (local >> 4) % kLanes;
    const uint32_t cell = local / (kLanes * 16);
    // Store in the byte order consumed by the PTX A fragment, not register
    // declaration order. The permutation is self-inverse.
    const uint32_t conversionOrder[8] = {0, 2, 1, 3, 4, 6, 5, 7};
    const uint32_t operation = cell * 8 + conversionOrder[packedWithin];
    const uint16_t result = FuseValue(mainFeature, skipFeature, head,
                                      ctaX, ctaY, operation, lane, pairHalf);
    packedE4[index] = EncodeHalfBitsE4M3(result);
    if (packedHalf) packedHalf[index] = result;
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

static std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
    return ReadSlice(path, 0, std::filesystem::file_size(path));
}

static void WriteText(const std::filesystem::path& path, const std::string& text) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(text.data(), std::streamsize(text.size())))
        throw std::runtime_error("cannot write " + path.string());
}

static void WriteBinary(const std::filesystem::path& path, const void* data, size_t bytes) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(static_cast<const char*>(data), std::streamsize(bytes)))
        throw std::runtime_error("cannot write " + path.string());
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

static std::string Escape(const std::string& value) {
    std::string result;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        result.push_back(ch);
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 8) {
            std::fprintf(stderr, "usage: output_head_activation_fusion <activation_arena.raw> "
                                 "<model_arena.raw> <rtx_e4_trace.raw> <result.json> "
                                 "[iterations] [full_grid_a_e4m3.raw] "
                                 "[full_grid_a_fp16.raw]\n");
            return 2;
        }
        const uint32_t iterations = argc >= 6 ? uint32_t(std::stoul(argv[5])) : 1000;
        if (!iterations) throw std::runtime_error("iterations must be positive");
        const auto mainFeature = ReadSlice(argv[1], kMainOffset, kMainBytes);
        const auto skipFeature = ReadSlice(argv[1], kSkipOffset, kSkipBytes);
        const auto head = ReadSlice(argv[2], kHeadOffset, kHeadBytes);
        const auto trace = ReadFile(argv[3]);
        if (trace.size() != 107520) throw std::runtime_error("unexpected E4 trace size");

        uint8_t *dMain = nullptr, *dSkip = nullptr, *dHead = nullptr, *dE4 = nullptr;
        uint16_t* dHalf = nullptr;
        HIP_CHECK(hipMalloc(&dMain, mainFeature.size()));
        HIP_CHECK(hipMalloc(&dSkip, skipFeature.size()));
        HIP_CHECK(hipMalloc(&dHead, head.size()));
        HIP_CHECK(hipMalloc(&dHalf, kValues * 2));
        HIP_CHECK(hipMalloc(&dE4, kValues));
        HIP_CHECK(hipMemcpy(dMain, mainFeature.data(), mainFeature.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dSkip, skipFeature.data(), skipFeature.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dHead, head.data(), head.size(), hipMemcpyHostToDevice));
        const dim3 block(256), grid(uint32_t((kValues + 255) / 256));
        Fuse<<<grid, block>>>(dMain, dSkip, dHead, dHalf, dE4);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<uint16_t> half(kValues), repeatHalf(kValues);
        std::vector<uint8_t> e4(kValues), repeatE4(kValues);
        HIP_CHECK(hipMemcpy(half.data(), dHalf, kValues * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(e4.data(), dE4, kValues, hipMemcpyDeviceToHost));
        Fuse<<<grid, block>>>(dMain, dSkip, dHead, dHalf, dE4);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(repeatHalf.data(), dHalf, kValues * 2, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(repeatE4.data(), dE4, kValues, hipMemcpyDeviceToHost));

        hipEvent_t start{}, stop{};
        HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (uint32_t i = 0; i < iterations; ++i) Fuse<<<grid, block>>>(dMain, dSkip, dHead, dHalf, dE4);
        HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
        float totalMs = 0.0f; HIP_CHECK(hipEventElapsedTime(&totalMs, start, stop));

        const uint32_t conversionOrder[8] = {0, 2, 1, 3, 4, 6, 5, 7};
        uint64_t halfMismatches = 0, e4Mismatches = 0;
        int64_t firstHalf = -1, firstE4 = -1;
        for (uint32_t operation = 0; operation < kOperations; ++operation) {
            const uint32_t cell = operation / 8;
            const uint32_t traceOperation = 4 + cell * 8 + conversionOrder[operation % 8];
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                const size_t traceRecord = (size_t(traceOperation) * kLanes + lane) * 8;
                for (uint32_t pairHalf = 0; pairHalf < 2; ++pairHalf) {
                    const size_t index = (size_t(operation) * kLanes + lane) * 2 + pairHalf;
                    const uint16_t expectedHalf = uint16_t(trace[traceRecord + pairHalf * 2]) |
                                                  (uint16_t(trace[traceRecord + pairHalf * 2 + 1]) << 8);
                    const uint8_t expectedE4 = trace[traceRecord + 4 + pairHalf];
                    if (half[index] != expectedHalf) {
                        if (firstHalf < 0) firstHalf = int64_t(index);
                        ++halfMismatches;
                    }
                    if (e4[index] != expectedE4) {
                        if (firstE4 < 0) firstE4 = int64_t(index);
                        ++e4Mismatches;
                    }
                }
            }
        }
        const bool deterministic = half == repeatHalf && e4 == repeatE4;
        bool fullGridExecuted = argc >= 7;
        bool fullGridDeterministic = false, selectedGridMatch = false;
        uint64_t fullGridNonzero = 0, fullGridNanCodes = 0, fullGridHash = 0;
        float fullGridAverageMs = 0.0f;
        if (fullGridExecuted) {
            uint8_t* dGrid = nullptr;
            uint16_t* dGridHalf = nullptr;
            HIP_CHECK(hipMalloc(&dGrid, kGridValues));
            if (argc == 8) HIP_CHECK(hipMalloc(&dGridHalf, kGridValues * 2));
            const dim3 fullGrid(uint32_t((kGridValues + 255) / 256));
            FuseFullGrid<<<fullGrid, block>>>(dMain, dSkip, dHead, dGrid, dGridHalf);
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<uint8_t> gridOutput(kGridValues), gridRepeat(kGridValues);
            HIP_CHECK(hipMemcpy(gridOutput.data(), dGrid, kGridValues, hipMemcpyDeviceToHost));
            FuseFullGrid<<<fullGrid, block>>>(dMain, dSkip, dHead, dGrid, dGridHalf);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(gridRepeat.data(), dGrid, kGridValues, hipMemcpyDeviceToHost));
            fullGridDeterministic = gridOutput == gridRepeat;
            HIP_CHECK(hipEventRecord(start));
            for (uint32_t i = 0; i < iterations; ++i)
                FuseFullGrid<<<fullGrid, block>>>(dMain, dSkip, dHead, dGrid, dGridHalf);
            HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
            float gridTotalMs = 0.0f; HIP_CHECK(hipEventElapsedTime(&gridTotalMs, start, stop));
            fullGridAverageMs = gridTotalMs / float(iterations);
            for (uint8_t code : gridOutput) {
                fullGridNonzero += code != 0;
                fullGridNanCodes += (code & 0x7f) == 0x7f;
            }
            selectedGridMatch = true;
            const size_t selectedBase = (size_t(kTargetCtaY) * kGridWidth + kTargetCtaX) * kValues;
            const uint32_t conversionOrder[8] = {0, 2, 1, 3, 4, 6, 5, 7};
            for (uint32_t group = 0; group < 4; ++group) {
                for (uint32_t packedWithin = 0; packedWithin < 8; ++packedWithin) {
                const uint32_t operation = group * 8 + conversionOrder[packedWithin];
                for (uint32_t lane = 0; lane < kLanes; ++lane)
                    for (uint32_t pairHalf = 0; pairHalf < 2; ++pairHalf) {
                        const size_t packedIndex = selectedBase +
                            size_t(group) * 512 + lane * 16 + packedWithin * 2 + pairHalf;
                        const size_t selectedIndex =
                            (size_t(operation) * kLanes + lane) * 2 + pairHalf;
                        selectedGridMatch &= gridOutput[packedIndex] == e4[selectedIndex];
                    }
                }
            }
            fullGridHash = Fnv1a64(gridOutput.data(), gridOutput.size());
            WriteBinary(argv[6], gridOutput.data(), gridOutput.size());
            if (argc == 8) {
                std::vector<uint16_t> gridHalf(kGridValues);
                HIP_CHECK(hipMemcpy(gridHalf.data(), dGridHalf, kGridValues * 2,
                                    hipMemcpyDeviceToHost));
                WriteBinary(argv[7], gridHalf.data(), gridHalf.size() * 2);
            }
            if (dGridHalf) HIP_CHECK(hipFree(dGridHalf));
            HIP_CHECK(hipFree(dGrid));
        }
        const bool fullGridPass = !fullGridExecuted ||
            (fullGridDeterministic && selectedGridMatch && fullGridNanCodes == 0 &&
             fullGridNonzero != 0);
        const bool pass = deterministic && halfMismatches == 0 && e4Mismatches == 0 && fullGridPass;
        hipDeviceProp_t properties{}; HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(dE4)); HIP_CHECK(hipFree(dHalf)); HIP_CHECK(hipFree(dHead));
        HIP_CHECK(hipFree(dSkip)); HIP_CHECK(hipFree(dMain));
        const float averageMs = totalMs / float(iterations);
        char fullGridHashText[32]{};
        std::snprintf(fullGridHashText, sizeof(fullGridHashText), "%016llX",
                      static_cast<unsigned long long>(fullGridHash));
        const std::string json =
            "{\n"
            "  \"schema\": 1,\n"
            "  \"experiment\": \"amd_output_head_activation_fusion\",\n"
            "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
            "  \"classification\": \"RAW_ACTIVATION_TO_FP8_MMA_A_FRAGMENT\",\n"
            "  \"device\": \"" + Escape(properties.name) + "\",\n"
            "  \"target_cta\": [70, 26, 0],\n"
            "  \"source_half_values\": 2048,\n"
            "  \"source_half_mismatches_vs_rtx\": " + std::to_string(halfMismatches) + ",\n"
            "  \"first_source_half_mismatch\": " + std::to_string(firstHalf) + ",\n"
            "  \"e4m3_values\": 2048,\n"
            "  \"e4m3_mismatches_vs_rtx\": " + std::to_string(e4Mismatches) + ",\n"
            "  \"first_e4m3_mismatch\": " + std::to_string(firstE4) + ",\n"
            "  \"deterministic_repeat\": " + std::string(deterministic ? "true" : "false") + ",\n"
            "  \"fusion_contract\": \"fp16_rn(main*scale_main), then fp16_rn(fma(skip,scale_skip,rounded_main))\",\n"
            "  \"iterations\": " + std::to_string(iterations) + ",\n"
            "  \"average_gpu_ms\": " + std::to_string(averageMs) + ",\n"
            "  \"full_grid_executed\": " + std::string(fullGridExecuted ? "true" : "false") + ",\n"
            "  \"full_grid_shape\": [49, 81, 4, 32, 16],\n"
            "  \"full_grid_values\": " + std::to_string(kGridValues) + ",\n"
            "  \"full_grid_storage\": \"e4m3_ptx_a_fragment_byte_order\",\n"
            "  \"full_grid_deterministic_repeat\": " + std::string(fullGridDeterministic ? "true" : "false") + ",\n"
            "  \"selected_cta_matches_trace_verified_path\": " + std::string(selectedGridMatch ? "true" : "false") + ",\n"
            "  \"full_grid_nonzero_values\": " + std::to_string(fullGridNonzero) + ",\n"
            "  \"full_grid_nan_codes\": " + std::to_string(fullGridNanCodes) + ",\n"
            "  \"full_grid_average_gpu_ms\": " + std::to_string(fullGridAverageMs) + ",\n"
            "  \"full_grid_fnv1a64\": \"" + fullGridHashText + "\",\n"
            "  \"boundary_policy\": \"zero_pad_outside_main_192x320_and_skip_96x160\",\n"
            "  \"boundary_policy_rtx_oracle_verified\": false,\n"
            "  \"next_gate\": \"Feed full-grid A fragments through the first packed projection and validate boundary CTAs\"\n"
            "}\n";
        WriteText(argv[4], json);
        std::printf("[%s] %s raw activation -> A fragment: half=%llu e4=%llu, %.6f ms\n",
                    pass ? "PASS" : "FAIL", properties.name,
                    static_cast<unsigned long long>(halfMismatches),
                    static_cast<unsigned long long>(e4Mismatches), averageMs);
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
