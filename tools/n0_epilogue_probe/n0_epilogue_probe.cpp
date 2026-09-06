#include <hip/hip_fp16.h>
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
    if (exponent == 0) value = ldexpf(float(mantissa), -9);
    else if (magnitude == 0x7f) value = nanf("");
    else value = ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    return (bits & 0x80) ? -value : value;
}

__device__ static int RoundNearestEvenPositive(float value) {
    const float floorValue = floorf(value);
    int result = int(floorValue);
    const float fraction = value - floorValue;
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1))) ++result;
    return result;
}

__device__ static uint8_t EncodeE4M3(float value) {
    const uint8_t sign = signbit(value) ? 0x80 : 0;
    if (isnan(value)) return 0x7f;
    const float magnitude = fabsf(value);
    if (magnitude >= 448.0f) return sign | 0x7e;
    if (magnitude < 0.015625f) {
        const int mantissa = RoundNearestEvenPositive(magnitude * 512.0f);
        return sign | uint8_t(mantissa >= 8 ? 8 : mantissa);
    }
    int exponentPlusOne = 0;
    const float fraction = frexpf(magnitude, &exponentPlusOne);
    int encodedExponent = exponentPlusOne - 1 + 7;
    int mantissa = RoundNearestEvenPositive((fraction * 2.0f - 1.0f) * 8.0f);
    if (mantissa == 8) { mantissa = 0; ++encodedExponent; }
    if (encodedExponent > 15 || (encodedExponent == 15 && mantissa >= 7)) return sign | 0x7e;
    return sign | uint8_t((encodedExponent << 3) | mantissa);
}

__device__ static uint8_t AverageFour(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    const __half ha = __float2half(DecodeE4M3(a));
    const __half hb = __float2half(DecodeE4M3(b));
    const __half hc = __float2half(DecodeE4M3(c));
    const __half hd = __float2half(DecodeE4M3(d));
    const __half sum = __hadd(__hadd(ha, hb), __hadd(hc, hd));
    return EncodeE4M3(__half2float(__hmul(sum, __float2half(0.25f))));
}

__device__ static uint8_t WarpAverageComponent(const uint8_t* tileA, const uint8_t* tileB,
                                                int firstPosition, int secondPosition,
                                                int component, int lane) {
    const uint8_t a0 = tileA[lane * 16 + firstPosition + component];
    const uint8_t b0 = tileB[lane * 16 + firstPosition + component];
    const uint8_t a1 = tileA[lane * 16 + secondPosition + component];
    const uint8_t b1 = tileB[lane * 16 + secondPosition + component];
    uint32_t first = (lane & 4) ? b0 : a0;
    uint32_t second = (lane & 4) ? a0 : b0;
    uint32_t third = (lane & 4) ? b1 : a1;
    uint32_t fourth = (lane & 4) ? a1 : b1;
    if (lane & 16) {
        const uint32_t savedFirst = first, savedSecond = second;
        first = third; second = fourth; third = savedFirst; fourth = savedSecond;
    }
    const int source0 = (lane & 19) | ((lane >> 1) & 4) | ((lane << 1) & 8);
    return AverageFour(
        uint8_t(__shfl(first, source0, 32)),
        uint8_t(__shfl(second, source0 ^ 4, 32)),
        uint8_t(__shfl(third, source0 ^ 16, 32)),
        uint8_t(__shfl(fourth, source0 ^ 20, 32)));
}

__global__ static void N0Epilogue(const uint8_t* scratch, uint8_t* output,
                                  int height, int width) {
    const int lane = int(threadIdx.x);
    const int ctaX = int(blockIdx.x), ctaY = int(blockIdx.y);
    const int tileColumns = width / 4;
    const int tileAIndex = (ctaY * 2) * tileColumns + ctaX * 2;
    const uint8_t* tileA = scratch + tileAIndex * 512;
    const uint8_t* tileB = tileA + 512;
    const uint8_t* tileC = tileA + tileColumns * 512;
    const uint8_t* tileD = tileC + 512;
    uint8_t values[16];
    const uint8_t* leftTiles[2] = {tileA, tileC};
    const uint8_t* rightTiles[2] = {tileB, tileD};
    const int positionPairs[4][2] = {{0, 4}, {2, 6}, {8, 12}, {10, 14}};
    for (int rowGroup = 0; rowGroup < 2; ++rowGroup) {
        for (int pairIndex = 0; pairIndex < 4; ++pairIndex) {
            for (int component = 0; component < 2; ++component) {
                values[rowGroup * 8 + pairIndex * 2 + component] = WarpAverageComponent(
                    leftTiles[rowGroup], rightTiles[rowGroup],
                    positionPairs[pairIndex][0], positionPairs[pairIndex][1], component, lane);
            }
        }
    }
    const int outputHeight = height / 2, outputWidth = width / 2;
    const size_t planeBytes = size_t(outputHeight) * outputWidth * 16;
    const int x = ctaX * 4 + (lane / 4) % 4;
    const int group = lane % 4;
    for (int rowGroup = 0; rowGroup < 2; ++rowGroup) {
        const int y = ctaY * 4 + rowGroup * 2 + lane / 16;
        const size_t low = (size_t(y) * outputWidth + x) * 16 + group * 4;
        output[low] = values[rowGroup * 8 + 0];
        output[low + 1] = values[rowGroup * 8 + 1];
        output[low + 2] = values[rowGroup * 8 + 2];
        output[low + 3] = values[rowGroup * 8 + 3];
        const size_t high = planeBytes + low;
        output[high] = values[rowGroup * 8 + 4];
        output[high + 1] = values[rowGroup * 8 + 5];
        output[high + 2] = values[rowGroup * 8 + 6];
        output[high + 3] = values[rowGroup * 8 + 7];
    }
}

static std::vector<uint8_t> ReadAll(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = file.tellg(); file.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size > 0 && !file.read(reinterpret_cast<char*>(data.data()), size)) return {};
    return data;
}

static bool WriteAll(const fs::path& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    return bool(file.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size())));
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: n0_epilogue_probe scratch.raw rtx_output.raw amd_output.raw comparison.json\n");
        return 2;
    }
    constexpr int height = 384, width = 640;
    const auto scratch = ReadAll(argv[1]);
    const auto reference = ReadAll(argv[2]);
    if (scratch.size() != size_t(height) * width * 32 ||
        reference.size() != size_t(height / 2) * (width / 2) * 32) return 3;
    hipDeviceProp_t properties{};
    if (hipGetDeviceProperties(&properties, 0) != hipSuccess || properties.warpSize != 32) return 4;
    uint8_t *deviceScratch = nullptr, *deviceOutput = nullptr;
    if (hipMalloc(&deviceScratch, scratch.size()) != hipSuccess ||
        hipMalloc(&deviceOutput, reference.size()) != hipSuccess) return 5;
    if (hipMemcpy(deviceScratch, scratch.data(), scratch.size(), hipMemcpyHostToDevice) != hipSuccess) return 6;
    hipLaunchKernelGGL(N0Epilogue, dim3(width / 8, height / 8), dim3(32), 0, 0,
                       deviceScratch, deviceOutput, height, width);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) return 7;
    std::vector<uint8_t> output(reference.size());
    if (hipMemcpy(output.data(), deviceOutput, output.size(), hipMemcpyDeviceToHost) != hipSuccess) return 8;
    const hipError_t freeScratch = hipFree(deviceScratch);
    const hipError_t freeOutput = hipFree(deviceOutput);
    if (freeScratch != hipSuccess || freeOutput != hipSuccess) return 9;
    if (!WriteAll(argv[3], output)) return 9;
    size_t exact = 0, adjacent = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (output[i] == reference[i]) { ++exact; continue; }
        if ((output[i] >> 7) == (reference[i] >> 7) &&
            std::abs(int(output[i] & 0x7f) - int(reference[i] & 0x7f)) == 1) ++adjacent;
    }
    const double tolerantFraction = double(exact + adjacent) / output.size();
    const bool pass = tolerantFraction > 0.90;
    std::ofstream json(argv[4]);
    json << "{\n  \"schema\": 1,\n  \"experiment\": \"amd_n0_tiled_epilogue\",\n"
         << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"classification\": \"REAL_AMD_N0_DATAFLOW_STAGE\",\n"
         << "  \"counts_as_s6\": false,\n  \"device_name\": \"" << properties.name << "\",\n"
         << "  \"warp_size\": " << properties.warpSize << ",\n  \"kernel_launched\": true,\n"
         << "  \"elements\": " << output.size() << ",\n  \"exact_matches\": " << exact << ",\n"
         << "  \"adjacent_same_sign_e4m3_codes\": " << adjacent << ",\n"
         << "  \"exact_fraction\": " << double(exact) / output.size() << ",\n"
         << "  \"exact_or_adjacent_fraction\": " << tolerantFraction << "\n}\n";
    std::printf("device=%s exact=%zu adjacent=%zu tolerant_fraction=%.9f status=%s\n",
                properties.name, exact, adjacent, tolerantFraction, pass ? "PASS" : "FAIL");
    return pass ? 0 : 10;
}
