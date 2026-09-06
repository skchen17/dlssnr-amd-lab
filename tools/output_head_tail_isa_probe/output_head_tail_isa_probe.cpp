#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t error_ = (call);                                                 \
    if (error_ != hipSuccess)                                                   \
      throw std::runtime_error(std::string(#call) + ": " +                    \
                               hipGetErrorString(error_));                      \
  } while (0)

namespace {
constexpr uint32_t kGridWidth = 81;
constexpr uint32_t kGridHeight = 49;
constexpr uint32_t kTokens = 64;
constexpr uint32_t kInputChannels = 32;
constexpr uint32_t kOutputChannels = 4;
constexpr uint32_t kTraceStages = 7;
constexpr size_t kCtaCount = size_t(kGridWidth) * kGridHeight;
constexpr size_t kOutputCount = kCtaCount * kTokens * kOutputChannels;
constexpr size_t kHeadOffset = 147429888;
constexpr size_t kHeadBytes = 21808;

__device__ uint16_t Weight(const uint8_t *head, uint32_t k, uint32_t channel) {
  const uint32_t tile = 20784 + (k / 16) * 512;
  const uint32_t kk = k % 16;
  const uint32_t lane = channel * 4 + (kk % 8) / 2;
  const uint32_t element = (kk % 2) + (kk >= 8 ? 2 : 0);
  const size_t offset = tile + lane * 16 + element * 2;
  return uint16_t(head[offset]) | uint16_t(head[offset + 1]) << 8;
}

__device__ _Float16 AsFloat16(uint16_t bits) {
  union {
    uint16_t bits;
    _Float16 value;
  } converted{bits};
  return converted.value;
}

__device__ uint32_t FloatBits(float value) {
  union {
    float value;
    uint32_t bits;
  } converted{value};
  return converted.bits;
}

__global__ void Probe(const uint16_t *input, const uint8_t *head,
                      uint32_t *trace, uint16_t *output) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= kOutputCount)
    return;
  size_t index = i;
  const uint32_t channel = uint32_t(index % kOutputChannels);
  index /= kOutputChannels;
  const uint32_t token = uint32_t(index % kTokens);
  const size_t cta = index / kTokens;
  const size_t base = (cta * kTokens + token) * kInputChannels;

  float accumulator = 0.0f;
  for (uint32_t k = 0; k < 16; ++k)
    accumulator += __half2float(__ushort_as_half(input[base + k])) *
                   __half2float(__ushort_as_half(Weight(head, k, channel)));
  accumulator = __half2float(__float2half_rn(accumulator));
  for (uint32_t k = 16; k < 20; k += 2) {
    accumulator = __fmaf_rn(
        __half2float(__ushort_as_half(input[base + k + 1])),
        __half2float(__ushort_as_half(Weight(head, k + 1, channel))),
        accumulator);
    accumulator = __fmaf_rn(
        __half2float(__ushort_as_half(input[base + k])),
        __half2float(__ushort_as_half(Weight(head, k, channel))), accumulator);
  }
  trace[i * kTraceStages] = FloatBits(accumulator);

  for (uint32_t pair = 0; pair < 6; ++pair) {
    const uint32_t k = 20 + pair * 2;
    const _Float16_2 inputPair{AsFloat16(input[base + k]),
                               AsFloat16(input[base + k + 1])};
    const _Float16_2 weightPair{AsFloat16(Weight(head, k, channel)),
                                AsFloat16(Weight(head, k + 1, channel))};
    accumulator =
        __builtin_amdgcn_fdot2(inputPair, weightPair, accumulator, false);
    trace[i * kTraceStages + pair + 1] = FloatBits(accumulator);
  }
  output[i] = __half_as_ushort(__float2half_rn(accumulator));
}

std::vector<uint8_t> Read(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path.string());
  const size_t size = size_t(stream.tellg());
  stream.seekg(0);
  std::vector<uint8_t> bytes(size);
  if (!stream.read(reinterpret_cast<char *>(bytes.data()),
                   std::streamsize(size)))
    throw std::runtime_error("read failed: " + path.string());
  return bytes;
}

std::vector<uint8_t> Slice(const std::filesystem::path &path, size_t offset,
                           size_t size) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path.string());
  const size_t total = size_t(stream.tellg());
  if (offset > total || size > total - offset)
    throw std::runtime_error("slice exceeds file");
  stream.seekg(std::streamoff(offset));
  std::vector<uint8_t> bytes(size);
  if (!stream.read(reinterpret_cast<char *>(bytes.data()),
                   std::streamsize(size)))
    throw std::runtime_error("read failed: " + path.string());
  return bytes;
}

void Write(const std::filesystem::path &path, const void *data, size_t size) {
  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream || !stream.write(reinterpret_cast<const char *>(data),
                               std::streamsize(size)))
    throw std::runtime_error("write failed: " + path.string());
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::fprintf(stderr,
                   "usage: output_head_tail_isa_probe <projected.raw> "
                   "<model.raw> <trace.raw> <output.raw>\n");
      return 2;
    }
    auto input = Read(argv[1]);
    auto head = Slice(argv[2], kHeadOffset, kHeadBytes);
    if (input.size() != kCtaCount * kTokens * kInputChannels * 2)
      throw std::runtime_error("unexpected projected input size");

    uint16_t *deviceInput = nullptr;
    uint8_t *deviceHead = nullptr;
    uint32_t *deviceTrace = nullptr;
    uint16_t *deviceOutput = nullptr;
    HIP_CHECK(hipMalloc(&deviceInput, input.size()));
    HIP_CHECK(hipMalloc(&deviceHead, head.size()));
    HIP_CHECK(hipMalloc(&deviceTrace,
                        kOutputCount * kTraceStages * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&deviceOutput, kOutputCount * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(deviceInput, input.data(), input.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(deviceHead, head.data(), head.size(),
                        hipMemcpyHostToDevice));

    Probe<<<uint32_t((kOutputCount + 255) / 256), 256>>>(
        deviceInput, deviceHead, deviceTrace, deviceOutput);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> trace(kOutputCount * kTraceStages);
    std::vector<uint16_t> output(kOutputCount);
    HIP_CHECK(hipMemcpy(trace.data(), deviceTrace, trace.size() * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(output.data(), deviceOutput,
                        output.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost));
    Write(argv[3], trace.data(), trace.size() * sizeof(uint32_t));
    Write(argv[4], output.data(), output.size() * sizeof(uint16_t));

    HIP_CHECK(hipFree(deviceOutput));
    HIP_CHECK(hipFree(deviceTrace));
    HIP_CHECK(hipFree(deviceHead));
    HIP_CHECK(hipFree(deviceInput));
    std::printf("[PASS] wrote %zu output-tail ISA traces\n", kOutputCount);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
