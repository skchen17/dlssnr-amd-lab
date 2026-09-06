#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

#define HR_CHECK(call)                                                         \
  do {                                                                         \
    HRESULT result_ = (call);                                                   \
    if (FAILED(result_)) {                                                      \
      char message_[192];                                                       \
      std::snprintf(message_, sizeof(message_), #call " failed: 0x%08lX",     \
                    static_cast<unsigned long>(result_));                       \
      throw std::runtime_error(message_);                                       \
    }                                                                           \
  } while (0)

namespace {
constexpr uint64_t kProjectedBytes = uint64_t(81) * 49 * 64 * 32 * 2;
constexpr uint64_t kResidualBytes = uint64_t(81) * 49 * 64 * 4 * 2;
constexpr uint64_t kSurfaceBytes = uint64_t(640) * 360 * 4 * 2;
constexpr uint64_t kHeadOffset = 147429888;
constexpr uint64_t kHeadBytes = 21808;
constexpr uint32_t kWorkItems = 81 * 49 * 64;

std::vector<uint8_t> Read(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path.string());
  const size_t size = size_t(stream.tellg());
  stream.seekg(0);
  std::vector<uint8_t> data(size);
  if (size && !stream.read(reinterpret_cast<char *>(data.data()),
                           std::streamsize(size)))
    throw std::runtime_error("read failed: " + path.string());
  return data;
}

std::vector<uint8_t> ReadSlice(const fs::path &path, uint64_t offset,
                               uint64_t size) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path.string());
  const uint64_t total = uint64_t(stream.tellg());
  if (offset > total || size > total - offset)
    throw std::runtime_error("slice exceeds " + path.string());
  stream.seekg(std::streamoff(offset));
  auto data = std::vector<uint8_t>(static_cast<size_t>(size));
  if (!stream.read(reinterpret_cast<char *>(data.data()), std::streamsize(size)))
    throw std::runtime_error("slice read failed: " + path.string());
  return data;
}

void Write(const fs::path &path, const void *data, size_t size) {
  if (path.has_parent_path())
    fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream || !stream.write(reinterpret_cast<const char *>(data),
                               std::streamsize(size)))
    throw std::runtime_error("write failed: " + path.string());
}

uint64_t Hash(const void *data, size_t size) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  uint64_t hash = 14695981039346656037ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string FormatNumber(double value) {
  char text[48]{};
  std::snprintf(text, sizeof(text), "%.9g", value);
  return text;
}

uint32_t OrderedHalf(uint16_t bits) {
  return bits & 0x8000u ? 0xffffu - bits : uint32_t(bits) + 0x8000u;
}

bool HalfFinite(uint16_t bits) { return (bits & 0x7c00u) != 0x7c00u; }

float HalfToFloat(uint16_t bits) {
  const float sign = bits & 0x8000u ? -1.0f : 1.0f;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x3ffu;
  if (exponent == 0)
    return sign * std::ldexp(float(mantissa), -24);
  if (exponent == 31)
    return mantissa ? std::numeric_limits<float>::quiet_NaN()
                    : sign * std::numeric_limits<float>::infinity();
  return sign * std::ldexp(1.0f + float(mantissa) / 1024.0f,
                           int(exponent) - 15);
}

D3D12_RESOURCE_DESC BufferDesc(
    uint64_t size, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = flags;
  return desc;
}

ComPtr<ID3D12Resource> Upload(ID3D12Device *device,
                              const std::vector<uint8_t> &data) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_UPLOAD;
  const auto desc = BufferDesc(data.size());
  ComPtr<ID3D12Resource> resource;
  HR_CHECK(device->CreateCommittedResource(
      &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
      nullptr, IID_PPV_ARGS(&resource)));
  void *mapped = nullptr;
  HR_CHECK(resource->Map(0, nullptr, &mapped));
  std::memcpy(mapped, data.data(), data.size());
  resource->Unmap(0, nullptr);
  return resource;
}

ComPtr<ID3D12Resource> DefaultBuffer(ID3D12Device *device, uint64_t bytes) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  const auto desc = BufferDesc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  ComPtr<ID3D12Resource> resource;
  HR_CHECK(device->CreateCommittedResource(
      &heap, D3D12_HEAP_FLAG_NONE, &desc,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
      IID_PPV_ARGS(&resource)));
  return resource;
}

ComPtr<ID3D12Resource> Readback(ID3D12Device *device, uint64_t bytes) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_READBACK;
  const auto desc = BufferDesc(bytes);
  ComPtr<ID3D12Resource> resource;
  HR_CHECK(device->CreateCommittedResource(
      &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
      nullptr, IID_PPV_ARGS(&resource)));
  return resource;
}

ComPtr<ID3D12PipelineState> Pipeline(ID3D12Device *device,
                                     ID3D12RootSignature *root,
                                     const std::vector<uint8_t> &dxil) {
  D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
  desc.pRootSignature = root;
  desc.CS = {dxil.data(), dxil.size()};
  ComPtr<ID3D12PipelineState> pipeline;
  HR_CHECK(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipeline)));
  return pipeline;
}

void Transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  list->ResourceBarrier(1, &barrier);
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 9) {
      std::fprintf(stderr,
                   "usage: output_head_tail_surface_d3d12 <projected.raw> "
                   "<model.raw> <base.raw|-> <expected_residual.raw> "
                   "<expected_surface.raw> <residual_out.raw> "
                   "<surface_out.raw> <manifest.json>\n");
      return 2;
    }
    auto projected = Read(argv[1]);
    auto head = ReadSlice(argv[2], kHeadOffset, kHeadBytes);
    auto base = std::string(argv[3]) == "-"
                    ? std::vector<uint8_t>(size_t(kSurfaceBytes), 0)
                    : Read(argv[3]);
    auto expectedResidual = Read(argv[4]);
    auto expectedSurface = Read(argv[5]);
    if (projected.size() != kProjectedBytes || head.size() != kHeadBytes ||
        base.size() != kSurfaceBytes ||
        expectedResidual.size() != kResidualBytes ||
        expectedSurface.size() != kSurfaceBytes)
      throw std::runtime_error("unexpected input size");

    ComPtr<IDXGIFactory7> factory;
    HR_CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 adapterDesc{};
    bool found = false;
    for (UINT i = 0;
         factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
         ++i, adapter.Reset()) {
      HR_CHECK(adapter->GetDesc1(&adapterDesc));
      if (!(adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
          adapterDesc.VendorId == 0x1002) {
        found = true;
        break;
      }
    }
    if (!found)
      throw std::runtime_error("AMD D3D12 adapter not found");
    ComPtr<ID3D12Device> device;
    HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                              IID_PPV_ARGS(&device)));

    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].Descriptor.ShaderRegister = 0;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 1;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    for (auto &param : params)
      param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = 3;
    rootDesc.pParameters = params;
    ComPtr<ID3DBlob> rootBlob, errors;
    HR_CHECK(D3D12SerializeRootSignature(
        &rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errors));
    ComPtr<ID3D12RootSignature> root;
    HR_CHECK(device->CreateRootSignature(0, rootBlob->GetBufferPointer(),
                                        rootBlob->GetBufferSize(),
                                        IID_PPV_ARGS(&root)));

    const fs::path build = fs::absolute(argv[0]).parent_path();
    auto tailPipeline = Pipeline(device.Get(), root.Get(),
                                 Read(build / "output_head_tail_d3d12.dxil"));
    auto surfacePipeline =
        Pipeline(device.Get(), root.Get(),
                 Read(build / "output_head_surface_d3d12.dxil"));
    auto projectedUpload = Upload(device.Get(), projected);
    auto headUpload = Upload(device.Get(), head);
    auto baseUpload = Upload(device.Get(), base);
    auto residual = DefaultBuffer(device.Get(), kResidualBytes);
    auto surface = DefaultBuffer(device.Get(), kSurfaceBytes);
    auto residualReadback = Readback(device.Get(), kResidualBytes);
    auto surfaceReadback = Readback(device.Get(), kSurfaceBytes);

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    HR_CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator;
    HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list;
    HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      allocator.Get(), tailPipeline.Get(),
                                      IID_PPV_ARGS(&list)));
    list->SetComputeRootSignature(root.Get());
    list->SetPipelineState(tailPipeline.Get());
    list->SetComputeRootShaderResourceView(
        0, projectedUpload->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(1, headUpload->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(2,
                                            residual->GetGPUVirtualAddress());
    list->Dispatch((kWorkItems + 255) / 256, 1, 1);
    Transition(list.Get(), residual.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->SetPipelineState(surfacePipeline.Get());
    list->SetComputeRootShaderResourceView(0, residual->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(1, baseUpload->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(2, surface->GetGPUVirtualAddress());
    list->Dispatch((kWorkItems + 255) / 256, 1, 1);
    Transition(list.Get(), residual.Get(),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(list.Get(), surface.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(residualReadback.Get(), 0, residual.Get(), 0,
                           kResidualBytes);
    list->CopyBufferRegion(surfaceReadback.Get(), 0, surface.Get(), 0,
                           kSurfaceBytes);
    HR_CHECK(list->Close());
    ID3D12CommandList *lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(), 1));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event)
      throw std::runtime_error("CreateEvent failed");
    if (fence->GetCompletedValue() < 1) {
      HR_CHECK(fence->SetEventOnCompletion(1, event));
      WaitForSingleObject(event, INFINITE);
    }
    CloseHandle(event);

    auto actualResidual =
        std::vector<uint8_t>(static_cast<size_t>(kResidualBytes));
    auto actualSurface =
        std::vector<uint8_t>(static_cast<size_t>(kSurfaceBytes));
    void *mapped = nullptr;
    D3D12_RANGE residualRange{0, SIZE_T(kResidualBytes)};
    HR_CHECK(residualReadback->Map(0, &residualRange, &mapped));
    std::memcpy(actualResidual.data(), mapped, actualResidual.size());
    residualReadback->Unmap(0, nullptr);
    D3D12_RANGE surfaceRange{0, SIZE_T(kSurfaceBytes)};
    HR_CHECK(surfaceReadback->Map(0, &surfaceRange, &mapped));
    std::memcpy(actualSurface.data(), mapped, actualSurface.size());
    surfaceReadback->Unmap(0, nullptr);

    const auto *residualWords =
        reinterpret_cast<const uint16_t *>(actualResidual.data());
    const auto *expectedResidualWords =
        reinterpret_cast<const uint16_t *>(expectedResidual.data());
    const auto *surfaceWords =
        reinterpret_cast<const uint16_t *>(actualSurface.data());
    const auto *expectedSurfaceWords =
        reinterpret_cast<const uint16_t *>(expectedSurface.data());
    uint64_t residualMismatches = 0, surfaceMismatches = 0, nonFinite = 0;
    uint32_t maxResidualUlp = 0, maxSurfaceUlp = 0;
    double residualAbsoluteSum = 0.0, surfaceAbsoluteSum = 0.0;
    float maxResidualAbsolute = 0.0f, maxSurfaceAbsolute = 0.0f;
    for (size_t i = 0; i < kResidualBytes / 2; ++i) {
      residualMismatches += residualWords[i] != expectedResidualWords[i];
      maxResidualUlp = std::max(
          maxResidualUlp,
          uint32_t(std::abs(int32_t(OrderedHalf(residualWords[i])) -
                            int32_t(OrderedHalf(expectedResidualWords[i])))));
      const float absolute = std::abs(HalfToFloat(residualWords[i]) -
                                      HalfToFloat(expectedResidualWords[i]));
      residualAbsoluteSum += absolute;
      maxResidualAbsolute = std::max(maxResidualAbsolute, absolute);
      nonFinite += !HalfFinite(residualWords[i]);
    }
    for (size_t i = 0; i < kSurfaceBytes / 2; ++i) {
      surfaceMismatches += surfaceWords[i] != expectedSurfaceWords[i];
      maxSurfaceUlp = std::max(
          maxSurfaceUlp,
          uint32_t(std::abs(int32_t(OrderedHalf(surfaceWords[i])) -
                            int32_t(OrderedHalf(expectedSurfaceWords[i])))));
      const float absolute = std::abs(HalfToFloat(surfaceWords[i]) -
                                      HalfToFloat(expectedSurfaceWords[i]));
      surfaceAbsoluteSum += absolute;
      maxSurfaceAbsolute = std::max(maxSurfaceAbsolute, absolute);
      nonFinite += !HalfFinite(surfaceWords[i]);
    }
    const double residualRatio =
        double(residualMismatches) / double(kResidualBytes / 2);
    const double surfaceRatio =
        double(surfaceMismatches) / double(kSurfaceBytes / 2);
    const double residualMeanAbsolute =
        residualAbsoluteSum / double(kResidualBytes / 2);
    const double surfaceMeanAbsolute =
        surfaceAbsoluteSum / double(kSurfaceBytes / 2);
    const bool pass = nonFinite == 0 && residualRatio <= 0.005 &&
                      surfaceRatio <= 0.005 && maxResidualAbsolute <= 0.01f &&
                      maxSurfaceAbsolute <= 0.0001f;
    Write(argv[6], actualResidual.data(), actualResidual.size());
    Write(argv[7], actualSurface.data(), actualSurface.size());
    char residualHash[32]{}, surfaceHash[32]{};
    std::snprintf(residualHash, sizeof(residualHash), "%016llX",
                  (unsigned long long)Hash(actualResidual.data(),
                                           actualResidual.size()));
    std::snprintf(surfaceHash, sizeof(surfaceHash), "%016llX",
                  (unsigned long long)Hash(actualSurface.data(),
                                           actualSurface.size()));
    const std::string manifest =
        "{\n  \"schema\": 1,\n  \"experiment\": "
        "\"output_head_tail_surface_d3d12\",\n  \"status\": \"" +
        std::string(pass ? "PASS" : "FAIL") +
        "\",\n  \"classification\": "
        "\"SAME_COMMAND_LIST_D3D12_TAIL_AND_SURFACE\",\n  "
        "\"d3d12_vendor_id\": \"0x1002\",\n  \"single_command_list\": "
        "true,\n  \"hip_runtime_dependency\": false,\n  "
        "\"residual_half_mismatches\": " +
        std::to_string(residualMismatches) +
        ",\n  \"residual_mismatch_ratio\": " + FormatNumber(residualRatio) +
        ",\n  \"max_residual_half_ulp\": " + std::to_string(maxResidualUlp) +
        ",\n  \"residual_mean_absolute_error\": " +
        FormatNumber(residualMeanAbsolute) +
        ",\n  \"residual_max_absolute_error\": " +
        FormatNumber(maxResidualAbsolute) +
        ",\n  \"surface_half_mismatches\": " +
        std::to_string(surfaceMismatches) +
        ",\n  \"surface_mismatch_ratio\": " + FormatNumber(surfaceRatio) +
        ",\n  \"max_surface_half_ulp\": " + std::to_string(maxSurfaceUlp) +
        ",\n  \"surface_mean_absolute_error\": " +
        FormatNumber(surfaceMeanAbsolute) +
        ",\n  \"surface_max_absolute_error\": " +
        FormatNumber(maxSurfaceAbsolute) +
        ",\n  \"non_finite_values\": " + std::to_string(nonFinite) +
        ",\n  \"residual_fnv1a64\": \"" + residualHash +
        "\",\n  \"surface_fnv1a64\": \"" + surfaceHash +
        "\",\n  \"game_runtime_ready\": false,\n  \"next_gate\": "
        "\"Port the final projection producer into the same D3D12 command "
        "list\"\n}\n";
    Write(argv[8], manifest.data(), manifest.size());
    std::printf("[%s] D3D12 tail+surface: residual=%llu surface=%llu "
                "surface_max_ulp=%u\n",
                pass ? "PASS" : "FAIL",
                (unsigned long long)residualMismatches,
                (unsigned long long)surfaceMismatches, maxSurfaceUlp);
    return pass ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
