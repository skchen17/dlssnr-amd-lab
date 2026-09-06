#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) \
  throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); } while (0)

namespace {
constexpr uint32_t kGridWidth = 81;
constexpr uint32_t kGridHeight = 49;
constexpr uint32_t kTokens = 64;
constexpr uint32_t kChannels = 4;
constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 360;
constexpr uint32_t kPaddedHeight = 384;
constexpr uint32_t kSelectedX = 70;
constexpr uint32_t kSelectedY = 26;
constexpr size_t kSelectedCta = size_t(kSelectedY) * kGridWidth + kSelectedX;
constexpr uint32_t kFp8Mmas = 256;
constexpr uint32_t kMmaRecordBytes = 40;
constexpr uint32_t kStoreRecordBytes = 32;
constexpr uint32_t kStoreThreadsPerSite = 127008;

std::vector<uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  const size_t bytes = size_t(stream.tellg());
  stream.seekg(0);
  std::vector<uint8_t> result(bytes);
  if (!stream.read(reinterpret_cast<char*>(result.data()), std::streamsize(bytes)))
    throw std::runtime_error("read failed: " + path.string());
  return result;
}

void Write(const std::filesystem::path& path, const void* data, size_t bytes) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream || !stream.write(reinterpret_cast<const char*>(data), std::streamsize(bytes)))
    throw std::runtime_error("write failed: " + path.string());
}

uint32_t LoadU32(const std::vector<uint8_t>& data, size_t offset) {
  if (offset + 4 > data.size()) throw std::runtime_error("trace read out of range");
  return uint32_t(data[offset]) | uint32_t(data[offset + 1]) << 8 |
         uint32_t(data[offset + 2]) << 16 | uint32_t(data[offset + 3]) << 24;
}

float LoadF32(const std::vector<uint8_t>& data, size_t offset) {
  const uint32_t bits = LoadU32(data, offset);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t TraceD(const std::vector<uint8_t>& trace, uint32_t mma, uint32_t row,
                uint32_t column) {
  const uint32_t lane = (row % 8) * 4 + column / 2;
  const uint32_t element = (row >= 8 ? 2 : 0) + (column & 1);
  const size_t offset = (size_t(kFp8Mmas + mma) * 32 + lane) * kMmaRecordBytes +
                        32 + (element / 2) * 4;
  return uint16_t(LoadU32(trace, offset) >> (16 * (element & 1)));
}

float HalfToFloat(uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}

uint64_t Hash(const void* data, size_t bytes) {
  const auto* cursor = static_cast<const uint8_t*>(data);
  uint64_t hash = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    hash ^= cursor[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

__global__ void Compose(const uint16_t* residual, const uint16_t* base,
                        uint16_t* output, uint32_t surface_height) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = size_t(kGridWidth) * kGridHeight * kTokens;
  if (index >= count) return;

  const uint32_t token = uint32_t(index % kTokens);
  const size_t cta = index / kTokens;
  const uint32_t cta_x = uint32_t(cta % kGridWidth);
  const uint32_t cta_y = uint32_t(cta / kGridWidth);
  const uint32_t group = token / 16;
  const uint32_t within = token % 16;
  const int32_t x = int32_t(cta_x * 8) - 4 + int32_t((group & 1) * 4 + within % 4);
  const int32_t y = int32_t(cta_y * 8) - 4 + int32_t((group >> 1) * 4 + within / 4);
  if (x < 0 || y < 0 || x >= int32_t(kWidth) || y >= int32_t(surface_height)) return;

  const size_t source = index * kChannels;
  const size_t destination = (size_t(y) * kWidth + uint32_t(x)) * kChannels;
  for (uint32_t channel = 0; channel < 3; ++channel) {
    const float b = base ? __half2float(__ushort_as_half(base[destination + channel])) : 0.0f;
    const float r = __half2float(__ushort_as_half(residual[source + channel]));
    output[destination + channel] = __half_as_ushort(__float2half_rn(
        fminf(1.0f, fmaxf(0.0f, b + 0.25f * r))));
  }
  output[destination + 3] = __half_as_ushort(__float2half_rn(1.0f));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 7 || argc > 8) {
      std::fprintf(stderr,
          "usage: output_head_surface_store <rgba_residual_fp16.raw> "
          "<base_rgba16f.raw|-> <rtx_mma_trace.raw> <rtx_store_trace.raw> "
          "<output_rgba16f.raw> <result.json> [iterations]\n");
      return 2;
    }
    const uint32_t iterations = argc == 8 ? uint32_t(std::stoul(argv[7])) : 20;
    const auto residual = Read(argv[1]);
    const bool has_base = std::string(argv[2]) != "-";
    const auto base = has_base ? Read(argv[2]) : std::vector<uint8_t>{};
    const auto mma_trace = Read(argv[3]);
    const auto store_trace = Read(argv[4]);
    const size_t residual_bytes = size_t(kGridWidth) * kGridHeight * kTokens * kChannels * 2;
    const size_t surface_bytes = size_t(kWidth) * kHeight * kChannels * 2;
    if (residual.size() != residual_bytes) throw std::runtime_error("unexpected residual size");
    if (has_base && base.size() != surface_bytes) throw std::runtime_error("unexpected base size");
    if (mma_trace.size() < size_t(kFp8Mmas + 16) * 32 * kMmaRecordBytes)
      throw std::runtime_error("unexpected MMA trace size");
    if (store_trace.size() != size_t(2) * kStoreThreadsPerSite * kStoreRecordBytes)
      throw std::runtime_error("unexpected store trace size");

    uint16_t* device_residual = nullptr;
    uint16_t* device_base = nullptr;
    uint16_t* device_output = nullptr;
    HIP_CHECK(hipMalloc(&device_residual, residual_bytes));
    HIP_CHECK(hipMalloc(&device_output, surface_bytes));
    HIP_CHECK(hipMemcpy(device_residual, residual.data(), residual_bytes, hipMemcpyHostToDevice));
    if (has_base) {
      HIP_CHECK(hipMalloc(&device_base, surface_bytes));
      HIP_CHECK(hipMemcpy(device_base, base.data(), surface_bytes, hipMemcpyHostToDevice));
    }
    auto launch = [&]() {
      HIP_CHECK(hipMemset(device_output, 0, surface_bytes));
      const size_t count = size_t(kGridWidth) * kGridHeight * kTokens;
      Compose<<<uint32_t((count + 255) / 256), 256>>>(
          device_residual, device_base, device_output, kHeight);
    };
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> output(surface_bytes / 2), repeat(output.size());
    HIP_CHECK(hipMemcpy(output.data(), device_output, surface_bytes, hipMemcpyDeviceToHost));
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(repeat.data(), device_output, surface_bytes, hipMemcpyDeviceToHost));

    hipEvent_t start{}, stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (uint32_t index = 0; index < iterations; ++index) launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

    uint64_t coordinate_mismatches = 0;
    uint64_t oracle_formula_mismatches = 0;
    uint64_t finite_failures = 0;
    double absolute_error_sum = 0.0;
    float maximum_error = 0.0f;
    uint64_t compared = 0;
    for (uint32_t token = 0; token < kTokens; ++token) {
      const uint32_t group = token / 16;
      const uint32_t row = token % 16;
      const uint32_t site = token / 32;
      const uint32_t lane = token % 32;
      const size_t record_index = size_t(site) * kStoreThreadsPerSite + kSelectedCta * 32 + lane;
      const size_t record_offset = record_index * kStoreRecordBytes;
      const uint32_t expected_x = kSelectedX * 8 - 4 + (group & 1) * 4 + row % 4;
      const uint32_t expected_y = kSelectedY * 8 - 4 + (group >> 1) * 4 + row / 4;
      coordinate_mismatches += LoadU32(store_trace, record_offset + 16) != expected_x;
      coordinate_mismatches += LoadU32(store_trace, record_offset + 20) != expected_y;
      coordinate_mismatches += LoadU32(store_trace, record_offset + 24) != site + 1;
      const uint32_t final_mma = group * 4 + 2;
      for (uint32_t channel = 0; channel < 3; ++channel) {
        const float rtx_residual = HalfToFloat(TraceD(mma_trace, final_mma, row, channel));
        // Preserve the two fused operations used by the captured PTX.  Although
        // their constants cancel algebraically for a zero source texture, two
        // tiny values differ from a single 0.25*r multiply by one f32 quantum.
        const float decoded = std::fma(rtx_residual, 0.03125f, -0.0625f);
        const float expected_store = std::min(1.0f, std::max(
            0.0f, std::fma(decoded, 8.0f, 0.5f)));
        const float observed_store = LoadF32(store_trace, record_offset + channel * 4);
        oracle_formula_mismatches += expected_store != observed_store;
        const float amd_store = HalfToFloat(output[(size_t(expected_y) * kWidth + expected_x) * 4 + channel]);
        finite_failures += !std::isfinite(amd_store);
        const float error = std::abs(amd_store - observed_store);
        absolute_error_sum += error;
        maximum_error = std::max(maximum_error, error);
        ++compared;
      }
    }

    Write(argv[5], output.data(), surface_bytes);
    char hash[32]{};
    std::snprintf(hash, sizeof(hash), "%016llX",
                  static_cast<unsigned long long>(Hash(output.data(), surface_bytes)));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    const bool deterministic = output == repeat;
    const double mean_error = absolute_error_sum / double(compared);
    const bool pass = deterministic && coordinate_mismatches == 0 &&
                      oracle_formula_mismatches == 0 && finite_failures == 0 &&
                      mean_error <= 0.01 && maximum_error <= 0.2;
    const std::string json =
        "{\n"
        "  \"schema\": 1,\n"
        "  \"experiment\": \"amd_output_head_surface_store\",\n"
        "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
        "  \"classification\": \"AMD_NATIVE_OUTPUT_HEAD_SPATIAL_STORE_VALIDATED\",\n"
        "  \"device\": \"" + std::string(properties.name) + "\",\n"
        "  \"surface_shape\": [360, 640, 4],\n"
        "  \"logical_cta_shape\": [8, 8],\n"
        "  \"cta_origin\": [-4, -4],\n"
        "  \"base_source\": \"" + std::string(has_base ? "rgba16f_file" : "zero_capture") + "\",\n"
        "  \"composition\": \"clamp(base_rgb + 0.25 * residual_rgb, 0, 1); alpha=1\",\n"
        "  \"selected_cta\": [70, 26, 0],\n"
        "  \"coordinate_or_marker_mismatches\": " + std::to_string(coordinate_mismatches) + ",\n"
        "  \"rtx_formula_float_mismatches\": " + std::to_string(oracle_formula_mismatches) + ",\n"
        "  \"rtx_rgb_values_compared\": " + std::to_string(compared) + ",\n"
        "  \"amd_vs_rtx_mean_absolute_error\": " + std::to_string(mean_error) + ",\n"
        "  \"amd_vs_rtx_max_absolute_error\": " + std::to_string(maximum_error) + ",\n"
        "  \"deterministic_repeat\": " + std::string(deterministic ? "true" : "false") + ",\n"
        "  \"non_finite_values\": " + std::to_string(finite_failures) + ",\n"
        "  \"iterations\": " + std::to_string(iterations) + ",\n"
        "  \"average_gpu_ms\": " + std::to_string(elapsed_ms / float(iterations)) + ",\n"
        "  \"output_fnv1a64\": \"" + std::string(hash) + "\",\n"
        "  \"next_gate\": \"Bind the game color resource as base input through D3D12-HIP interop\"\n"
        "}\n";
    Write(argv[6], json.data(), json.size());

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(device_output));
    if (device_base) HIP_CHECK(hipFree(device_base));
    HIP_CHECK(hipFree(device_residual));
    std::printf("[%s] %s surface store: map=%llu formula=%llu mean/max=%g/%g, %.6f ms\n",
                pass ? "PASS" : "FAIL", properties.name,
                static_cast<unsigned long long>(coordinate_mismatches),
                static_cast<unsigned long long>(oracle_formula_mismatches), mean_error,
                maximum_error, elapsed_ms / float(iterations));
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
