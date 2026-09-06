#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { \
    hipError_t e_ = (expr); \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#expr) + ": " + hipGetErrorString(e_)); \
} while (0)

struct alignas(8) Pixel { __half r, g, b, a; };

__global__ void Compose(const Pixel* base, const Pixel* residual, Pixel* output,
                        size_t pixels, float strength) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    const Pixel b = base[index];
    const Pixel d = residual[index];
    Pixel o;
    o.r = __float2half_rn(__half2float(b.r) + strength * __half2float(d.r));
    o.g = __float2half_rn(__half2float(b.g) + strength * __half2float(d.g));
    o.b = __float2half_rn(__half2float(b.b) + strength * __half2float(d.b));
    o.a = b.a;
    output[index] = o;
}

static void WriteFile(const std::filesystem::path& path, const void* data, size_t bytes) {
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

int main(int argc, char** argv) {
    try {
        uint32_t width = 1920, height = 1080, iterations = 500;
        std::filesystem::path jsonPath;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto take = [&](const char* name) -> std::string {
                if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
                return argv[i];
            };
            if (arg == "--width") width = uint32_t(std::stoul(take("--width")));
            else if (arg == "--height") height = uint32_t(std::stoul(take("--height")));
            else if (arg == "--iterations") iterations = uint32_t(std::stoul(take("--iterations")));
            else if (arg == "--json") jsonPath = take("--json");
            else throw std::runtime_error("unknown argument: " + arg);
        }
        if (!width || !height || !iterations) throw std::runtime_error("dimensions and iterations must be positive");

        int device = 0;
        hipDeviceProp_t props{};
        HIP_CHECK(hipSetDevice(device));
        HIP_CHECK(hipGetDeviceProperties(&props, device));
        const size_t count = size_t(width) * height;
        const size_t bytes = count * sizeof(Pixel);
        std::vector<Pixel> base(count), residual(count), output(count);
        for (size_t i = 0; i < count; ++i) {
            const bool alternate = (i & 1) != 0;
            base[i] = {__float2half(alternate ? 0.5f : 0.25f), __float2half(0.5f),
                       __float2half(0.75f), __float2half(1.0f)};
            residual[i] = {__float2half(0.125f), __float2half(-0.125f),
                           __float2half(alternate ? -0.25f : 0.25f), __float2half(0.0f)};
        }

        Pixel *dBase = nullptr, *dResidual = nullptr, *dOutput = nullptr;
        HIP_CHECK(hipMalloc(&dBase, bytes));
        HIP_CHECK(hipMalloc(&dResidual, bytes));
        HIP_CHECK(hipMalloc(&dOutput, bytes));
        HIP_CHECK(hipMemcpy(dBase, base.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dResidual, residual.data(), bytes, hipMemcpyHostToDevice));
        const dim3 block(256);
        const dim3 grid(uint32_t((count + block.x - 1) / block.x));

        Compose<<<grid, block>>>(dBase, dResidual, dOutput, count, 0.0f);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(output.data(), dOutput, bytes, hipMemcpyDeviceToHost));
        const bool zeroStrengthExact = std::equal(
            base.begin(), base.end(), output.begin(),
            [](const Pixel& a, const Pixel& b) { return std::memcmp(&a, &b, sizeof(Pixel)) == 0; });

        hipEvent_t start{}, stop{};
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));
        for (uint32_t i = 0; i < iterations; ++i)
            Compose<<<grid, block>>>(dBase, dResidual, dOutput, count, 1.0f);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float totalMs = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&totalMs, start, stop));
        HIP_CHECK(hipMemcpy(output.data(), dOutput, bytes, hipMemcpyDeviceToHost));

        uint64_t mismatches = 0;
        for (size_t i = 0; i < count; ++i) {
            const float expected[4] = {
                (i & 1) ? 0.625f : 0.375f, 0.375f, (i & 1) ? 0.5f : 1.0f, 1.0f};
            const float got[4] = {__half2float(output[i].r), __half2float(output[i].g),
                                  __half2float(output[i].b), __half2float(output[i].a)};
            for (int c = 0; c < 4; ++c) if (got[c] != expected[c]) ++mismatches;
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(dBase));
        HIP_CHECK(hipFree(dResidual));
        HIP_CHECK(hipFree(dOutput));

        const float averageMs = totalMs / float(iterations);
        const bool pass = zeroStrengthExact && mismatches == 0;
        std::string json =
            "{\n"
            "  \"schema\": 1,\n"
            "  \"experiment\": \"amd_fused_residual_complete_frame_output\",\n"
            "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
            "  \"device\": \"" + Escape(props.name) + "\",\n"
            "  \"width\": " + std::to_string(width) + ",\n"
            "  \"height\": " + std::to_string(height) + ",\n"
            "  \"iterations\": " + std::to_string(iterations) + ",\n"
            "  \"average_gpu_ms\": " + std::to_string(averageMs) + ",\n"
            "  \"zero_strength_identity_exact\": " + std::string(zeroStrengthExact ? "true" : "false") + ",\n"
            "  \"component_mismatches\": " + std::to_string(mismatches) + ",\n"
            "  \"output_contract\": \"complete_rgba16f_frame\"\n"
            "}\n";
        if (!jsonPath.empty()) WriteFile(jsonPath, json.data(), json.size());
        std::printf("[%s] %s %ux%u %.6f ms, mismatches=%llu\n", pass ? "PASS" : "FAIL",
                    props.name, width, height, averageMs,
                    static_cast<unsigned long long>(mismatches));
        return pass ? 0 : 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
