#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

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

constexpr uint32_t kMmas = 256;
constexpr uint32_t kLanes = 32;
constexpr uint32_t kWordsPerRecord = 10;
constexpr uint32_t kHalfOutputsPerLane = 4;
constexpr size_t kFp8TraceBytes = size_t(kMmas) * kLanes * kWordsPerRecord * 4;

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

__device__ static uint8_t FragmentByte(uint32_t word, uint32_t element) {
    return uint8_t(word >> ((element & 3) * 8));
}

__global__ static void Replay(const uint32_t* trace, uint16_t* output) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = size_t(kMmas) * kLanes * kHalfOutputsPerLane;
    if (index >= total) return;
    const uint32_t mma = uint32_t(index / (kLanes * kHalfOutputsPerLane));
    const uint32_t local = uint32_t(index % (kLanes * kHalfOutputsPerLane));
    const uint32_t lane = local / kHalfOutputsPerLane;
    const uint32_t element = local % kHalfOutputsPerLane;
    const uint32_t group = lane >> 2;
    const uint32_t threadInGroup = lane & 3;
    const uint32_t row = group + (element >= 2 ? 8 : 0);
    const uint32_t column = threadInGroup * 2 + (element & 1);
    const size_t record = (size_t(mma) * kLanes + lane) * kWordsPerRecord;
    const uint32_t cword = trace[record + 6 + element / 2];
    float accumulator = __half2float(__ushort_as_half(uint16_t(cword >> (16 * (element & 1)))));

    for (uint32_t k = 0; k < 32; ++k) {
        const uint32_t aThread = (k & 15) >> 2;
        const uint32_t aLane = group * 4 + aThread;
        const uint32_t ai = (k & 3) + (row >= 8 ? 4 : 0) + (k >= 16 ? 8 : 0);
        const size_t aRecord = (size_t(mma) * kLanes + aLane) * kWordsPerRecord;
        const uint8_t a = FragmentByte(trace[aRecord + ai / 4], ai);

        const uint32_t bThread = (k & 15) >> 2;
        const uint32_t bLane = column * 4 + bThread;
        const uint32_t bi = (k & 3) + (k >= 16 ? 4 : 0);
        const size_t bRecord = (size_t(mma) * kLanes + bLane) * kWordsPerRecord;
        const uint8_t b = FragmentByte(trace[bRecord + 4 + bi / 4], bi);
        accumulator += DecodeE4M3(a) * DecodeE4M3(b);
    }
    output[index] = __half_as_ushort(__float2half_rn(accumulator));
}

static std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const size_t bytes = size_t(stream.tellg());
    stream.seekg(0);
    std::vector<uint8_t> result(bytes);
    if (!stream.read(reinterpret_cast<char*>(result.data()), std::streamsize(bytes)))
        throw std::runtime_error("cannot read " + path.string());
    return result;
}

static void WriteText(const std::filesystem::path& path, const std::string& text) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(text.data(), std::streamsize(text.size())))
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

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            std::fprintf(stderr, "usage: output_head_mma_replay <rtx_trace.raw> <result.json> [iterations]\n");
            return 2;
        }
        const uint32_t iterations = argc == 4 ? uint32_t(std::stoul(argv[3])) : 1000;
        if (!iterations) throw std::runtime_error("iterations must be positive");
        const std::vector<uint8_t> bytes = ReadFile(argv[1]);
        if (bytes.size() < kFp8TraceBytes) throw std::runtime_error("trace is too short");
        const auto* trace = reinterpret_cast<const uint32_t*>(bytes.data());
        const size_t outputCount = size_t(kMmas) * kLanes * kHalfOutputsPerLane;
        std::vector<uint16_t> output(outputCount), repeat(outputCount), reference(outputCount);
        for (uint32_t mma = 0; mma < kMmas; ++mma) {
            for (uint32_t lane = 0; lane < kLanes; ++lane) {
                const size_t record = (size_t(mma) * kLanes + lane) * kWordsPerRecord;
                for (uint32_t element = 0; element < 4; ++element) {
                    const uint32_t word = trace[record + 8 + element / 2];
                    reference[(size_t(mma) * kLanes + lane) * 4 + element] =
                        uint16_t(word >> (16 * (element & 1)));
                }
            }
        }

        hipDeviceProp_t properties{};
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipGetDeviceProperties(&properties, 0));
        uint32_t* deviceTrace = nullptr;
        uint16_t* deviceOutput = nullptr;
        HIP_CHECK(hipMalloc(&deviceTrace, kFp8TraceBytes));
        HIP_CHECK(hipMalloc(&deviceOutput, outputCount * sizeof(uint16_t)));
        HIP_CHECK(hipMemcpy(deviceTrace, bytes.data(), kFp8TraceBytes, hipMemcpyHostToDevice));
        const dim3 block(256);
        const dim3 grid(uint32_t((outputCount + block.x - 1) / block.x));
        Replay<<<grid, block>>>(deviceTrace, deviceOutput);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), deviceOutput, output.size() * 2, hipMemcpyDeviceToHost));
        Replay<<<grid, block>>>(deviceTrace, deviceOutput);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(repeat.data(), deviceOutput, repeat.size() * 2, hipMemcpyDeviceToHost));

        hipEvent_t start{}, stop{};
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (uint32_t i = 0; i < iterations; ++i) Replay<<<grid, block>>>(deviceTrace, deviceOutput);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float totalMs = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&totalMs, start, stop));

        uint64_t mismatches = 0;
        int64_t firstMismatch = -1;
        for (size_t i = 0; i < outputCount; ++i) {
            if (output[i] == reference[i]) continue;
            if (firstMismatch < 0) firstMismatch = int64_t(i);
            ++mismatches;
        }
        const bool deterministic = output == repeat;
        const bool pass = deterministic && mismatches == 0;
        const float averageMs = totalMs / float(iterations);
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(deviceOutput));
        HIP_CHECK(hipFree(deviceTrace));

        const std::string json =
            "{\n"
            "  \"schema\": 1,\n"
            "  \"experiment\": \"amd_output_head_fp8_mma_replay\",\n"
            "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
            "  \"classification\": \"CAPTURED_OPERAND_NATIVE_ARITHMETIC_REPLAY\",\n"
            "  \"device\": \"" + Escape(properties.name) + "\",\n"
            "  \"fp8_mma_count\": 256,\n"
            "  \"half_output_count\": " + std::to_string(outputCount) + ",\n"
            "  \"half_mismatches_vs_rtx\": " + std::to_string(mismatches) + ",\n"
            "  \"first_mismatch\": " + std::to_string(firstMismatch) + ",\n"
            "  \"deterministic_repeat\": " + std::string(deterministic ? "true" : "false") + ",\n"
            "  \"iterations\": " + std::to_string(iterations) + ",\n"
            "  \"average_gpu_ms\": " + std::to_string(averageMs) + ",\n"
            "  \"captured_operands_used\": true,\n"
            "  \"upstream_activation_path_reconstructed\": false,\n"
            "  \"next_gate\": \"Generate the same A fragments from activation_arena.raw on AMD\"\n"
            "}\n";
        WriteText(argv[2], json);
        std::printf("[%s] %s replayed 256 FP8 MMAs: mismatches=%llu, %.6f ms\n",
                    pass ? "PASS" : "FAIL", properties.name,
                    static_cast<unsigned long long>(mismatches), averageMs);
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
