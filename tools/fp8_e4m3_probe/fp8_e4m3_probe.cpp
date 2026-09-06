#include <hip/hip_runtime.h>

#include <cmath>
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
    if (exponent == 0) {
        value = ldexpf(float(mantissa), -9);
    } else if (magnitude == 0x7f) {
        value = nanf("");
    } else {
        value = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    }
    return (bits & 0x80) ? -value : value;
}

__device__ static int RoundNearestEvenPositive(float value) {
    float baseFloat = floorf(value);
    int base = int(baseFloat);
    float fraction = value - baseFloat;
    if (fraction > 0.5f || (fraction == 0.5f && (base & 1))) ++base;
    return base;
}

__device__ static uint8_t EncodeE4M3Satfinite(float value) {
    const uint8_t sign = signbit(value) ? 0x80 : 0;
    // NVIDIA PTX canonicalizes both positive and negative NaNs to 0x7f.
    if (isnan(value)) return 0x7f;
    float magnitude = fabsf(value);
    if (magnitude >= 448.0f) return sign | 0x7e;
    if (magnitude < 0.015625f) {
        int mantissa = RoundNearestEvenPositive(magnitude * 512.0f);
        if (mantissa >= 8) return sign | 0x08;
        return sign | uint8_t(mantissa);
    }
    int exponentPlusOne = 0;
    float fraction = frexpf(magnitude, &exponentPlusOne);
    int encodedExponent = exponentPlusOne - 1 + 7;
    int mantissa = RoundNearestEvenPositive((fraction * 2.0f - 1.0f) * 8.0f);
    if (mantissa == 8) {
        mantissa = 0;
        ++encodedExponent;
    }
    if (encodedExponent > 15 || (encodedExponent == 15 && mantissa >= 7))
        return sign | 0x7e;
    return sign | uint8_t((encodedExponent << 3) | mantissa);
}

__device__ static float DecodeHalfBits(uint16_t bits) {
    const uint16_t magnitude = bits & 0x7fff;
    const int exponent = (magnitude >> 10) & 31;
    const int mantissa = magnitude & 1023;
    float value;
    if (exponent == 0) {
        value = ldexpf(float(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa ? nanf("") : INFINITY;
    } else {
        value = ldexpf(1.0f + float(mantissa) / 1024.0f, exponent - 15);
    }
    return (bits & 0x8000) ? -value : value;
}

__global__ static void RoundTripE4M3(const uint8_t* input, uint8_t* output,
                                    float* decoded, size_t count) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = DecodeE4M3(input[index]);
    decoded[index] = value;
    output[index] = EncodeE4M3Satfinite(value);
}

__global__ static void ConvertAllHalf(const uint16_t* input, uint8_t* output,
                                      size_t count) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = EncodeE4M3Satfinite(DecodeHalfBits(input[index]));
}

static std::vector<uint8_t> ReadAll(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size && !file.read(reinterpret_cast<char*>(data.data()), size)) return {};
    return data;
}

static int RunHalfOracle(const fs::path& oraclePath, const fs::path& jsonPath,
                         const hipDeviceProp_t& properties) {
    std::vector<uint8_t> oracle = ReadAll(oraclePath);
    if (oracle.size() != 65536) return 20;
    std::vector<uint16_t> input(65536);
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = uint16_t(i);
    uint16_t* deviceInput = nullptr;
    uint8_t* deviceOutput = nullptr;
    if (hipMalloc(&deviceInput, input.size() * sizeof(uint16_t)) != hipSuccess ||
        hipMalloc(&deviceOutput, oracle.size()) != hipSuccess)
        return 21;
    if (hipMemcpy(deviceInput, input.data(), input.size() * sizeof(uint16_t),
                  hipMemcpyHostToDevice) != hipSuccess)
        return 22;
    hipLaunchKernelGGL(ConvertAllHalf, dim3(256), dim3(256), 0, 0,
                       deviceInput, deviceOutput, input.size());
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        return 23;
    std::vector<uint8_t> output(oracle.size());
    if (hipMemcpy(output.data(), deviceOutput, output.size(),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return 24;
    (void)hipFree(deviceOutput);
    (void)hipFree(deviceInput);

    uint64_t mismatches = 0, finiteMismatches = 0;
    uint64_t infinityMismatches = 0, nanMismatches = 0;
    int64_t firstMismatch = -1, lastMismatch = -1;
    for (size_t i = 0; i < oracle.size(); ++i) {
        if (output[i] == oracle[i]) continue;
        if (firstMismatch < 0) firstMismatch = int64_t(i);
        lastMismatch = int64_t(i);
        ++mismatches;
        const int exponent = (int(i) >> 10) & 31;
        const int mantissa = int(i) & 1023;
        if (exponent != 31) ++finiteMismatches;
        else if (mantissa == 0) ++infinityMismatches;
        else ++nanMismatches;
    }
    const bool negativeDetected = (uint8_t(output[0] ^ 1) != oracle[0]);
    const bool pass = mismatches == 0 && negativeDetected;
    std::printf("[%s] device=%s half_patterns=65536 mismatches=%llu "
                "finite=%llu infinity=%llu nan=%llu\n",
                pass ? "PASS" : "FAIL", properties.name,
                (unsigned long long)mismatches,
                (unsigned long long)finiteMismatches,
                (unsigned long long)infinityMismatches,
                (unsigned long long)nanMismatches);
    std::ofstream json(jsonPath, std::ios::binary);
    json << "{\n"
         << "  \"schema\": 1,\n"
         << "  \"experiment\": \"amd_fp16_to_e4m3_rtx_oracle\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"classification\": \"FP8_NUMERICAL_PRIMITIVE\",\n"
         << "  \"counts_as_s6\": false,\n"
         << "  \"hip_device\": \"" << properties.name << "\",\n"
         << "  \"input_half_patterns\": 65536,\n"
         << "  \"oracle_bytes\": 65536,\n"
         << "  \"mismatches\": " << mismatches << ",\n"
         << "  \"finite_mismatches\": " << finiteMismatches << ",\n"
         << "  \"infinity_mismatches\": " << infinityMismatches << ",\n"
         << "  \"nan_mismatches\": " << nanMismatches << ",\n"
         << "  \"first_mismatch\": " << firstMismatch << ",\n"
         << "  \"last_mismatch\": " << lastMismatch << ",\n"
         << "  \"nan_canonical_code\": 127,\n"
         << "  \"packed_lane_order\": \"low_f16_to_low_e4m3\",\n"
         << "  \"negative_verifier_detected\": "
         << (negativeDetected ? "true" : "false") << "\n"
         << "}\n";
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    const bool halfOracle = argc == 4 && std::string(argv[1]) == "--half-oracle";
    if ((!halfOracle && argc != 3) || (halfOracle && argc != 4)) {
        std::fprintf(stderr, "usage: fp8_e4m3_probe [--half-oracle] "
                             "<RTX-raw> <result.json>\n");
        return 2;
    }
    fs::path inputPath = argv[halfOracle ? 2 : 1];
    fs::path jsonPath = argv[halfOracle ? 3 : 2];
    std::vector<uint8_t> input = ReadAll(inputPath);
    if (input.empty()) return 3;
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess) return 4;
    if (halfOracle) return RunHalfOracle(inputPath, jsonPath, properties);

    std::vector<uint8_t> exhaustive(256);
    for (int i = 0; i < 256; ++i) exhaustive[i] = uint8_t(i);
    std::vector<uint8_t> combined = input;
    combined.insert(combined.end(), exhaustive.begin(), exhaustive.end());
    uint8_t *deviceInput = nullptr, *deviceOutput = nullptr;
    float* deviceDecoded = nullptr;
    const size_t count = combined.size();
    if (hipMalloc(&deviceInput, count) != hipSuccess ||
        hipMalloc(&deviceOutput, count) != hipSuccess ||
        hipMalloc(&deviceDecoded, count * sizeof(float)) != hipSuccess)
        return 5;
    if (hipMemcpy(deviceInput, combined.data(), count, hipMemcpyHostToDevice) != hipSuccess)
        return 6;
    hipLaunchKernelGGL(RoundTripE4M3, dim3((count + 255) / 256), dim3(256),
                       0, 0, deviceInput, deviceOutput, deviceDecoded, count);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess)
        return 7;
    std::vector<uint8_t> output(count);
    std::vector<float> decoded(count);
    if (hipMemcpy(output.data(), deviceOutput, count, hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(decoded.data(), deviceDecoded, count * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return 8;
    (void)hipFree(deviceDecoded);
    (void)hipFree(deviceOutput);
    (void)hipFree(deviceInput);

    uint64_t captureMismatches = 0, finiteCodeMismatches = 0;
    uint64_t nanBytes = 0, nonfiniteDecoded = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        if (output[i] != input[i]) ++captureMismatches;
        if ((input[i] & 0x7f) == 0x7f) ++nanBytes;
        if (!std::isfinite(decoded[i])) ++nonfiniteDecoded;
    }
    for (size_t i = 0; i < 256; ++i) {
        const uint8_t code = uint8_t(i);
        if ((code & 0x7f) != 0x7f && output[input.size() + i] != code)
            ++finiteCodeMismatches;
    }
    bool negativeDetected = false;
    if (!input.empty()) {
        uint8_t corrupted = output[0] ^ 1;
        negativeDetected = corrupted != input[0];
    }
    bool pass = captureMismatches == 0 && finiteCodeMismatches == 0 &&
                nonfiniteDecoded == nanBytes && negativeDetected;
    std::printf("[%s] device=%s bytes=%zu capture_mismatches=%llu "
                "finite_code_mismatches=%llu nan_bytes=%llu\n",
                pass ? "PASS" : "FAIL", properties.name, input.size(),
                (unsigned long long)captureMismatches,
                (unsigned long long)finiteCodeMismatches,
                (unsigned long long)nanBytes);
    std::ofstream json(jsonPath, std::ios::binary);
    json << "{\n"
         << "  \"schema\": 1,\n"
         << "  \"experiment\": \"amd_e4m3_roundtrip_rtx_oracle\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"classification\": \"FP8_NUMERICAL_PRIMITIVE\",\n"
         << "  \"counts_as_s6\": false,\n"
         << "  \"hip_device\": \"" << properties.name << "\",\n"
         << "  \"capture_bytes\": " << input.size() << ",\n"
         << "  \"capture_mismatches\": " << captureMismatches << ",\n"
         << "  \"finite_code_mismatches\": " << finiteCodeMismatches << ",\n"
         << "  \"nan_bytes\": " << nanBytes << ",\n"
         << "  \"decoded_nonfinite\": " << nonfiniteDecoded << ",\n"
         << "  \"negative_verifier_detected\": "
         << (negativeDetected ? "true" : "false") << "\n"
         << "}\n";
    return pass ? 0 : 1;
}
