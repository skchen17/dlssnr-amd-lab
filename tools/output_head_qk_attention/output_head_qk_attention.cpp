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
constexpr size_t kCtas = size_t(kGridWidth) * kGridHeight;
constexpr uint32_t kTokens = 64, kChannels = 32;
constexpr size_t kProjectionValuesPerCta = 48 * 16 * 8;
constexpr size_t kPackedValuesPerCta = kTokens * kChannels;
constexpr uint32_t kQkMmas = 32, kRows = 16, kColumns = 8;
constexpr size_t kOutputValuesPerCta = size_t(kQkMmas) * kRows * kColumns;
constexpr size_t kOutputValues = kCtas * kOutputValuesPerCta;
constexpr size_t kHeadOffset = 147429888, kHeadBytes = 21808;
constexpr size_t kSelectedCta = size_t(26) * kGridWidth + 70;
constexpr size_t kMmaRecordBytes = 40, kE4RecordBytes = 8;

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
        const uint32_t rounded = RoundShift(mantissa, 7);
        const bool carry = rounded >= 8;
        code = ((exponent - 8 + uint32_t(carry)) << 3) | (carry ? 0 : rounded);
        code = min(code, 0x7eu);
    }
    return sign | uint8_t(code);
}

__device__ static float DecodeE4M3(uint8_t bits) {
    const uint8_t magnitude = bits & 0x7f;
    const int exponent = magnitude >> 3, mantissa = magnitude & 7;
    float value;
    if (exponent == 0) value = ldexpf(float(mantissa), -9);
    else if (magnitude == 0x7f) value = nanf("");
    else value = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    return (bits & 0x80) ? -value : value;
}

__device__ static uint16_t HalfMul(uint16_t a, uint16_t b) {
    return __half_as_ushort(__float2half_rn(__half2float(__ushort_as_half(a)) *
                                           __half2float(__ushort_as_half(b))));
}

__device__ static uint16_t HalfAdd(uint16_t a, uint16_t b) {
    return __half_as_ushort(__float2half_rn(__half2float(__ushort_as_half(a)) +
                                           __half2float(__ushort_as_half(b))));
}

__device__ static uint32_t PackedSourceChannel(uint32_t channel) {
    const uint32_t block = channel / 16, within = channel % 16;
    return block * 16 + 2 * (within / 4) + (within & 1) + ((within & 2) ? 8 : 0);
}

__device__ static size_t ProjectionIndex(size_t cta, uint32_t segment,
                                         uint32_t token, uint32_t channel) {
    const uint32_t group = token / 16, row = token % 16;
    const uint32_t mma = group * 12 + segment * 4 +
                         (channel / 16) * 2 + (channel % 16) / 8;
    return cta * kProjectionValuesPerCta + size_t(mma) * 128 + row * 8 + channel % 8;
}

__device__ static size_t PackedAIndex(uint32_t group, uint32_t row, uint32_t k) {
    const uint32_t lane = (row % 8) * 4 + (k % 16) / 4;
    const uint32_t element = (k % 4) + (row >= 8 ? 4 : 0) + (k >= 16 ? 8 : 0);
    return size_t(group) * 512 + lane * 16 + element;
}

__device__ static size_t PackedBIndex(uint32_t block, uint32_t k, uint32_t column) {
    const uint32_t lane = column * 4 + (k % 16) / 4;
    const uint32_t element = (k % 4) + (k >= 16 ? 4 : 0);
    return size_t(block) * 256 + lane * 8 + element;
}

__global__ static void PrepareQK(const uint16_t* projection, uint8_t* q, uint8_t* k,
                                 uint16_t scale, uint16_t epsilon) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kCtas * kTokens) return;
    const size_t cta = index / kTokens;
    const uint32_t token = uint32_t(index % kTokens);
    uint16_t qValues[32], kValues[32], qSquares[32], kSquares[32];
    for (uint32_t channel = 0; channel < 32; ++channel) {
        qValues[channel] = projection[ProjectionIndex(cta, 0, token, channel)];
        kValues[channel] = projection[ProjectionIndex(cta, 1, token, channel)];
        qSquares[channel] = HalfMul(qValues[channel], qValues[channel]);
        kSquares[channel] = HalfMul(kValues[channel], kValues[channel]);
    }
    for (uint32_t width = 16; width; width >>= 1)
        for (uint32_t i = 0; i < width; ++i) {
            qSquares[i] = HalfAdd(qSquares[2 * i], qSquares[2 * i + 1]);
            kSquares[i] = HalfAdd(kSquares[2 * i], kSquares[2 * i + 1]);
        }
    const float epsilonValue = __half2float(__ushort_as_half(epsilon));
    const uint16_t qInverse = __half_as_ushort(__float2half_rn(
        rsqrtf(max(__half2float(__ushort_as_half(qSquares[0])), epsilonValue))));
    const uint16_t kInverse = __half_as_ushort(__float2half_rn(
        rsqrtf(max(__half2float(__ushort_as_half(kSquares[0])), epsilonValue))));
    const uint32_t group = token / 16, row = token % 16;
    const uint32_t keyBlock = token / 8, keyColumn = token % 8;
    for (uint32_t packedChannel = 0; packedChannel < 32; ++packedChannel) {
        const uint32_t source = PackedSourceChannel(packedChannel);
        const uint16_t qNormalized = HalfMul(HalfMul(qValues[source], qInverse), scale);
        const uint16_t kNormalized = HalfMul(kValues[source], kInverse);
        q[cta * kPackedValuesPerCta + PackedAIndex(group, row, packedChannel)] =
            EncodeHalfBitsE4M3(qNormalized);
        k[cta * kPackedValuesPerCta + PackedBIndex(keyBlock, packedChannel, keyColumn)] =
            EncodeHalfBitsE4M3(kNormalized);
    }
}

__global__ static void PrepareV(const uint16_t* projection, uint8_t* v) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kCtas * kTokens * kChannels) return;
    size_t split = index;
    const uint32_t feature = uint32_t(split % kChannels); split /= kChannels;
    const uint32_t destinationKey = uint32_t(split % kTokens);
    const size_t cta = split / kTokens;
    const uint32_t keyChunk = destinationKey / 32;
    const uint32_t keyWithinChunk = destinationKey % 32;
    // movmatrix.sync.trans applies the same 8x8 register permutation that was
    // observed in Q/K packing, here on the reduction (token) dimension.
    const uint32_t sourceToken = keyChunk * 32 + PackedSourceChannel(keyWithinChunk);
    const uint32_t featureBlock = feature / 8, column = feature % 8;
    const uint32_t lane = column * 4 + (keyWithinChunk % 16) / 4;
    const uint32_t element = (keyWithinChunk % 4) + (keyWithinChunk >= 16 ? 4 : 0);
    const size_t fragment = size_t(keyChunk * 4 + featureBlock) * 256;
    const uint16_t value = projection[ProjectionIndex(cta, 2, sourceToken, feature)];
    v[cta * kPackedValuesPerCta + fragment + lane * 8 + element] = EncodeHalfBitsE4M3(value);
}

__global__ static void QkAttention(const uint8_t* q, const uint8_t* k,
                                   const uint8_t* head, uint16_t* output) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputValues) return;
    size_t split = index;
    const uint32_t column = uint32_t(split % 8); split /= 8;
    const uint32_t row = uint32_t(split % 16); split /= 16;
    const uint32_t mma = uint32_t(split % kQkMmas);
    const size_t cta = split / kQkMmas;
    const uint32_t queryGroup = mma / 8, keyBlock = mma % 8;
    // Each query group owns eight C fragments.  A 512-byte lane-interleaved
    // block contains the two adjacent N=8 fragments, so the group stride is
    // four blocks (2048 bytes), not eight blocks.
    const uint32_t seedBase = 11472 + queryGroup * 2048;
    const uint32_t seedTile = seedBase + (keyBlock / 2) * 512;
    const uint32_t seedHalf = keyBlock & 1;
    const uint32_t lane = (row % 8) * 4 + column / 2;
    const uint32_t element = (row >= 8 ? 2 : 0) + (column & 1);
    const size_t seedOffset = seedTile + lane * 16 + seedHalf * 8 + element * 2;
    const uint16_t seedBits = uint16_t(head[seedOffset]) | uint16_t(head[seedOffset + 1]) << 8;
    float accumulator = __half2float(__ushort_as_half(seedBits));
    const uint8_t* qCta = q + cta * kPackedValuesPerCta;
    const uint8_t* kCta = k + cta * kPackedValuesPerCta;
    for (uint32_t channel = 0; channel < 32; ++channel)
        accumulator += DecodeE4M3(qCta[PackedAIndex(queryGroup, row, channel)]) *
                       DecodeE4M3(kCta[PackedBIndex(keyBlock, channel, column)]);
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
    const size_t size = size_t(stream.tellg());
    if (offset > size || bytes > size - offset) throw std::runtime_error("slice exceeds file");
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
    const size_t offset = (size_t(mma) * 32 + lane) * kMmaRecordBytes + 32 + (element / 2) * 4;
    const uint32_t word = uint32_t(trace[offset]) | uint32_t(trace[offset + 1]) << 8 |
                          uint32_t(trace[offset + 2]) << 16 | uint32_t(trace[offset + 3]) << 24;
    return uint16_t(word >> (16 * (element & 1)));
}

static uint8_t TraceE4(const std::vector<uint8_t>& trace, uint32_t operation,
                       uint32_t lane, uint32_t pair) {
    return trace[(size_t(operation) * 32 + lane) * kE4RecordBytes + 4 + pair];
}

static uint64_t Fnv1a64(const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data); uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    try {
        if (argc < 10 || argc > 11) {
            std::fprintf(stderr, "usage: output_head_qk_attention <mma128_175_fp16.raw> <model_arena.raw> <rtx_mma_trace.raw> <rtx_e4_trace.raw> <q_e4.raw> <k_e4.raw> <v_e4.raw> <qk_fp16.raw> <result.json> [iterations]\n");
            return 2;
        }
        const uint32_t iterations = argc == 11 ? uint32_t(std::stoul(argv[10])) : 10;
        const auto projection = ReadFile(argv[1]);
        const auto head = ReadSlice(argv[2], kHeadOffset, kHeadBytes);
        const auto mmaTrace = ReadFile(argv[3]);
        const auto e4Trace = ReadFile(argv[4]);
        if (projection.size() != kCtas * kProjectionValuesPerCta * 2 ||
            mmaTrace.size() < 256 * 32 * kMmaRecordBytes ||
            e4Trace.size() < 292 * 32 * kE4RecordBytes)
            throw std::runtime_error("unexpected input size");
        float scaleFloat; std::memcpy(&scaleFloat, head.data() + 19664, sizeof(scaleFloat));
        const uint16_t scale = __half_as_ushort(__float2half_rn(scaleFloat));
        const uint16_t epsilon = __half_as_ushort(__float2half_rn(6.199999916134402e-05f));
        uint16_t *dProjection = nullptr, *dOutput = nullptr;
        uint8_t *dHead = nullptr, *dQ = nullptr, *dK = nullptr, *dV = nullptr;
        HIP_CHECK(hipMalloc(&dProjection, projection.size()));
        HIP_CHECK(hipMalloc(&dHead, head.size()));
        HIP_CHECK(hipMalloc(&dQ, kCtas * kPackedValuesPerCta));
        HIP_CHECK(hipMalloc(&dK, kCtas * kPackedValuesPerCta));
        HIP_CHECK(hipMalloc(&dV, kCtas * kPackedValuesPerCta));
        HIP_CHECK(hipMalloc(&dOutput, kOutputValues * 2));
        HIP_CHECK(hipMemcpy(dProjection, projection.data(), projection.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dHead, head.data(), head.size(), hipMemcpyHostToDevice));
        const dim3 block(256), prepGrid(uint32_t((kCtas * kTokens + 255) / 256));
        const dim3 vGrid(uint32_t((kCtas * kTokens * kChannels + 255) / 256));
        const dim3 outputGrid(uint32_t((kOutputValues + 255) / 256));
        auto launch = [&]() {
            PrepareQK<<<prepGrid, block>>>(dProjection, dQ, dK, scale, epsilon);
            PrepareV<<<vGrid, block>>>(dProjection, dV);
            QkAttention<<<outputGrid, block>>>(dQ, dK, dHead, dOutput);
        };
        launch(); HIP_CHECK(hipDeviceSynchronize());
        std::vector<uint8_t> q(kCtas * kPackedValuesPerCta), k(q.size()), v(q.size());
        std::vector<uint16_t> output(kOutputValues), repeat(kOutputValues);
        HIP_CHECK(hipMemcpy(q.data(), dQ, q.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(k.data(), dK, k.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(v.data(), dV, v.size(), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(output.data(), dOutput, output.size() * 2, hipMemcpyDeviceToHost));
        launch(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(repeat.data(), dOutput, repeat.size() * 2, hipMemcpyDeviceToHost));
        hipEvent_t start{}, stop{}; HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (uint32_t i = 0; i < iterations; ++i) launch();
        HIP_CHECK(hipEventRecord(stop)); HIP_CHECK(hipEventSynchronize(stop));
        float totalMs = 0; HIP_CHECK(hipEventElapsedTime(&totalMs, start, stop));

        uint64_t qMismatches = 0, kMismatches = 0, vMismatches = 0;
        for (uint32_t group = 0; group < 4; ++group)
            for (uint32_t row = 0; row < 16; ++row)
                for (uint32_t channel = 0; channel < 32; ++channel) {
                    const uint32_t lane = (row % 8) * 4 + (channel % 16) / 4;
                    const uint32_t element = (channel % 4) + (row >= 8 ? 4 : 0) + (channel >= 16 ? 8 : 0);
                    const uint8_t expected = TraceE4(e4Trace, 196 + group * 8 + element / 2, lane, element & 1);
                    qMismatches += q[kSelectedCta * kPackedValuesPerCta + size_t(group) * 512 + lane * 16 + element] != expected;
                }
        for (uint32_t keyBlock = 0; keyBlock < 8; ++keyBlock)
            for (uint32_t channel = 0; channel < 32; ++channel)
                for (uint32_t column = 0; column < 8; ++column) {
                    const uint32_t lane = column * 4 + (channel % 16) / 4;
                    const uint32_t element = (channel % 4) + (channel >= 16 ? 4 : 0);
                    const uint8_t expected = TraceE4(e4Trace, 228 + keyBlock * 4 + element / 2, lane, element & 1);
                    kMismatches += k[kSelectedCta * kPackedValuesPerCta + size_t(keyBlock) * 256 + lane * 8 + element] != expected;
                }
        for (uint32_t keyChunk = 0; keyChunk < 2; ++keyChunk)
            for (uint32_t featureBlock = 0; featureBlock < 4; ++featureBlock)
                for (uint32_t key = 0; key < 32; ++key)
                    for (uint32_t column = 0; column < 8; ++column) {
                        const uint32_t lane = column * 4 + (key % 16) / 4;
                        const uint32_t element = (key % 4) + (key >= 16 ? 4 : 0);
                        const uint32_t operation = 260 + keyChunk * 16 + featureBlock * 4 + element / 2;
                        const uint8_t expected = TraceE4(e4Trace, operation, lane, element & 1);
                        const size_t fragment = size_t(keyChunk * 4 + featureBlock) * 256;
                        vMismatches += v[kSelectedCta * kPackedValuesPerCta + fragment + lane * 8 + element] != expected;
                    }
        uint64_t halfMismatches = 0, nonfinite = 0;
        double absoluteErrorSum = 0; float maxAbsoluteError = 0;
        for (uint32_t local = 0; local < kQkMmas; ++local)
            for (uint32_t row = 0; row < 16; ++row)
                for (uint32_t column = 0; column < 8; ++column) {
                    const size_t at = kSelectedCta * kOutputValuesPerCta + size_t(local) * 128 + row * 8 + column;
                    const uint16_t expectedBits = TraceD(mmaTrace, local < 16 ? 176 + local : 200 + local, row, column);
                    halfMismatches += output[at] != expectedBits;
                    const float actual = __half2float(__ushort_as_half(output[at]));
                    const float expected = __half2float(__ushort_as_half(expectedBits));
                    nonfinite += !std::isfinite(actual);
                    const float error = std::abs(actual - expected);
                    absoluteErrorSum += error; maxAbsoluteError = std::max(maxAbsoluteError, error);
                }
        const bool deterministic = output == repeat;
        const bool accepted = deterministic && nonfinite == 0 && qMismatches <= 16 &&
                              kMismatches <= 16 && vMismatches == 0 && maxAbsoluteError <= 0.125f;
        WriteFile(argv[5], q.data(), q.size()); WriteFile(argv[6], k.data(), k.size());
        WriteFile(argv[7], v.data(), v.size()); WriteFile(argv[8], output.data(), output.size() * 2);
        char outputHash[32]{}, qHash[32]{}, kHash[32]{}, vHash[32]{};
        std::snprintf(outputHash, sizeof(outputHash), "%016llX", (unsigned long long)Fnv1a64(output.data(), output.size() * 2));
        std::snprintf(qHash, sizeof(qHash), "%016llX", (unsigned long long)Fnv1a64(q.data(), q.size()));
        std::snprintf(kHash, sizeof(kHash), "%016llX", (unsigned long long)Fnv1a64(k.data(), k.size()));
        std::snprintf(vHash, sizeof(vHash), "%016llX", (unsigned long long)Fnv1a64(v.data(), v.size()));
        hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        const double meanAbsoluteError = absoluteErrorSum / (kQkMmas * 16 * 8);
        std::string json = "{\n  \"schema\": 1,\n  \"experiment\": \"amd_output_head_qkv_attention\",\n  \"status\": \"" + std::string(accepted ? "PASS" : "FAIL") + "\",\n  \"classification\": \"NUMERICAL_TOLERANCE_VALIDATED\",\n  \"device\": \"" + std::string(prop.name) + "\",\n  \"q_e4m3_values\": 2048,\n  \"q_e4m3_mismatches\": " + std::to_string(qMismatches) + ",\n  \"k_e4m3_values\": 2048,\n  \"k_e4m3_mismatches\": " + std::to_string(kMismatches) + ",\n  \"v_e4m3_values\": 2048,\n  \"v_e4m3_mismatches\": " + std::to_string(vMismatches) + ",\n  \"qk_rtx_half_values\": 4096,\n  \"qk_rtx_half_mismatches\": " + std::to_string(halfMismatches) + ",\n  \"qk_mean_absolute_error\": " + std::to_string(meanAbsoluteError) + ",\n  \"qk_max_absolute_error\": " + std::to_string(maxAbsoluteError) + ",\n  \"deterministic_repeat\": " + std::string(deterministic ? "true" : "false") + ",\n  \"non_finite_values\": " + std::to_string(nonfinite) + ",\n  \"iterations\": " + std::to_string(iterations) + ",\n  \"average_gpu_ms\": " + std::to_string(totalMs / float(iterations)) + ",\n  \"q_fnv1a64\": \"" + qHash + "\",\n  \"k_fnv1a64\": \"" + kHash + "\",\n  \"v_fnv1a64\": \"" + vHash + "\",\n  \"output_fnv1a64\": \"" + outputHash + "\",\n  \"covered_original_mmas\": [[176, 191], [216, 231]],\n  \"acceptance\": {\"q_e4m3_mismatches_max\": 16, \"k_e4m3_mismatches_max\": 16, \"v_e4m3_mismatches_max\": 0, \"qk_max_absolute_error_max\": 0.125},\n  \"next_gate\": \"Recover softmax and the softmax-times-V MMA chains\"\n}\n";
        WriteFile(argv[9], json.data(), json.size());
        HIP_CHECK(hipEventDestroy(start)); HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(dOutput)); HIP_CHECK(hipFree(dV)); HIP_CHECK(hipFree(dK)); HIP_CHECK(hipFree(dQ));
        HIP_CHECK(hipFree(dHead)); HIP_CHECK(hipFree(dProjection));
        std::printf("[%s] %s Q/K/V: e4 mismatches=%llu/%llu/%llu, QK half mismatches=%llu, max abs=%g, %.6f ms\n",
                    accepted ? "PASS" : "FAIL", prop.name,
                    (unsigned long long)qMismatches, (unsigned long long)kMismatches,
                    (unsigned long long)vMismatches, (unsigned long long)halfMismatches,
                    maxAbsoluteError, totalMs / float(iterations));
        return accepted ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1;
    }
}
