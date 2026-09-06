#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

#define HR_CHECK(x) do { HRESULT hr_=(x); if (FAILED(hr_)) { char message_[192]; \
  std::snprintf(message_,sizeof(message_),#x " failed: 0x%08lX",(unsigned long)hr_); \
  throw std::runtime_error(message_); } } while (0)

namespace {
constexpr uint64_t kResidualBytes = uint64_t(81) * 49 * 64 * 4 * 2;
constexpr uint64_t kSurfaceBytes = uint64_t(640) * 360 * 4 * 2;
constexpr uint64_t kProjectedBytes = uint64_t(81) * 49 * 64 * 32 * 2;
constexpr uint64_t kQkBytes = uint64_t(81) * 49 * 32 * 16 * 8 * 2;
constexpr uint64_t kPackedVBytes = uint64_t(81) * 49 * 64 * 32;
constexpr uint64_t kHeadOffset = 147429888;
constexpr uint64_t kHeadBytes = 21808;
constexpr uint32_t kWorkItems = 81 * 49 * 64;

std::vector<uint8_t> Read(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  size_t size = size_t(stream.tellg()); stream.seekg(0);
  std::vector<uint8_t> data(size);
  if (size && !stream.read(reinterpret_cast<char*>(data.data()), std::streamsize(size)))
    throw std::runtime_error("read failed: " + path.string());
  return data;
}

std::vector<uint8_t> ReadSlice(const fs::path& path, uint64_t offset, uint64_t size) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  uint64_t total = uint64_t(stream.tellg());
  if (offset > total || size > total - offset) throw std::runtime_error("slice exceeds " + path.string());
  stream.seekg(std::streamoff(offset)); auto data = std::vector<uint8_t>(static_cast<size_t>(size));
  if (!stream.read(reinterpret_cast<char*>(data.data()), std::streamsize(size)))
    throw std::runtime_error("slice read failed: " + path.string());
  return data;
}

void Write(const fs::path& path, const void* data, size_t size) {
  if (path.has_parent_path()) fs::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream || !stream.write(reinterpret_cast<const char*>(data), std::streamsize(size)))
    throw std::runtime_error("write failed: " + path.string());
}

uint64_t Hash(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  uint64_t hash = 14695981039346656037ull;
  for (size_t i = 0; i < size; ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
  return hash;
}

float HalfToFloat(uint16_t bits) {
  const float sign = bits & 0x8000u ? -1.0f : 1.0f;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x3ffu;
  if (exponent == 0) return sign * std::ldexp(float(mantissa), -24);
  if (exponent == 31) return mantissa ? NAN : sign * INFINITY;
  return sign * std::ldexp(1.0f + float(mantissa) / 1024.0f, int(exponent) - 15);
}

std::string FormatNumber(double value) {
  char text[48]{}; std::snprintf(text, sizeof(text), "%.9g", value); return text;
}

D3D12_RESOURCE_DESC BufferDesc(uint64_t size, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
  D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = size; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
  desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
  return desc;
}

ComPtr<ID3D12Resource> Upload(ID3D12Device* device, const std::vector<uint8_t>& data) {
  D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_UPLOAD;
  auto desc = BufferDesc(data.size()); ComPtr<ID3D12Resource> resource;
  HR_CHECK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&resource)));
  void* mapped = nullptr; HR_CHECK(resource->Map(0, nullptr, &mapped));
  std::memcpy(mapped, data.data(), data.size()); resource->Unmap(0, nullptr);
  return resource;
}
}

int main(int argc, char** argv) { try {
  const std::string executable = fs::path(argv[0]).stem().string();
  const bool softmaxMode = executable.find("softmax_v") != std::string::npos;
  const bool finalMode = executable.find("final_projection") != std::string::npos;
  const bool tailMode = !softmaxMode && !finalMode && executable.find("tail") != std::string::npos;
  if (argc != (finalMode ? 7 : 6)) {
    std::fprintf(stderr, "usage: output_head_{surface,tail}_d3d12 <input0.raw> <input1.raw|-> <expected.raw> <output.raw> <manifest.json>\n"
                         "   or: output_head_softmax_v_d3d12 <qk.raw> <v.raw> <expected.raw> <output.raw> <manifest.json>\n"
                         "   or: output_head_final_projection_d3d12 <attention.raw> <residual.raw> <model.raw> <expected.raw> <output.raw> <manifest.json>\n");
    return 2;
  }
  const uint64_t input0Bytes = softmaxMode ? kQkBytes : ((tailMode || finalMode) ? kProjectedBytes : kResidualBytes);
  const uint64_t input1Bytes = softmaxMode ? kPackedVBytes : (finalMode ? kProjectedBytes : (tailMode ? kHeadBytes : kSurfaceBytes));
  const uint64_t outputBytes = (softmaxMode || finalMode) ? kProjectedBytes : (tailMode ? kResidualBytes : kSurfaceBytes);
  auto input0 = Read(argv[1]);
  auto input1 = (softmaxMode || finalMode) ? Read(argv[2]) : (tailMode ? ReadSlice(argv[2], kHeadOffset, kHeadBytes) :
      (std::string(argv[2]) == "-" ? std::vector<uint8_t>(kSurfaceBytes, 0) : Read(argv[2])));
  auto input2 = finalMode ? ReadSlice(argv[3], kHeadOffset, kHeadBytes) : std::vector<uint8_t>{};
  const int expectedIndex = finalMode ? 4 : 3;
  const int outputIndex = finalMode ? 5 : 4;
  const int manifestIndex = finalMode ? 6 : 5;
  auto expected = Read(argv[expectedIndex]);
  if (input0.size() != input0Bytes || input1.size() != input1Bytes ||
      (finalMode && input2.size() != kHeadBytes) || expected.size() != outputBytes)
    throw std::runtime_error("unexpected input size");

  ComPtr<IDXGIFactory7> factory; HR_CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
  ComPtr<IDXGIAdapter1> adapter; DXGI_ADAPTER_DESC1 adapterDesc{}; bool found = false;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
    HR_CHECK(adapter->GetDesc1(&adapterDesc));
    if (!(adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && adapterDesc.VendorId == 0x1002) { found = true; break; }
  }
  if (!found) throw std::runtime_error("AMD D3D12 adapter not found");
  ComPtr<ID3D12Device> device; HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

  D3D12_ROOT_PARAMETER params[4]{};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; params[0].Descriptor.ShaderRegister = 0;
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; params[1].Descriptor.ShaderRegister = 1;
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; params[2].Descriptor.ShaderRegister = 2;
  params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; params[3].Descriptor.ShaderRegister = 0;
  for (auto& param : params) param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
  D3D12_ROOT_SIGNATURE_DESC rootDesc{}; rootDesc.NumParameters = 4; rootDesc.pParameters = params;
  ComPtr<ID3DBlob> rootBlob, errors;
  HR_CHECK(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errors));
  ComPtr<ID3D12RootSignature> root; HR_CHECK(device->CreateRootSignature(0,
      rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&root)));

  fs::path dxilPath = fs::absolute(argv[0]).parent_path() /
      (softmaxMode ? "output_head_softmax_v_d3d12.dxil" : (finalMode ? "output_head_final_projection_d3d12.dxil" :
       (tailMode ? "output_head_tail_d3d12.dxil" : "output_head_surface_d3d12.dxil")));
  auto dxil = Read(dxilPath); D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};
  pipelineDesc.pRootSignature = root.Get(); pipelineDesc.CS = {dxil.data(), dxil.size()};
  ComPtr<ID3D12PipelineState> pipeline; HR_CHECK(device->CreateComputePipelineState(&pipelineDesc, IID_PPV_ARGS(&pipeline)));

  auto input0Upload = Upload(device.Get(), input0); auto input1Upload = Upload(device.Get(), input1);
  auto input2Upload = finalMode ? Upload(device.Get(), input2) : ComPtr<ID3D12Resource>{};
  D3D12_HEAP_PROPERTIES defaultHeap{}; defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
  auto outputDesc = BufferDesc(outputBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  ComPtr<ID3D12Resource> output; HR_CHECK(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE,
      &outputDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&output)));
  D3D12_HEAP_PROPERTIES readbackHeap{}; readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;
  auto readbackDesc = BufferDesc(outputBytes); ComPtr<ID3D12Resource> readback;
  HR_CHECK(device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &readbackDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)));

  D3D12_COMMAND_QUEUE_DESC queueDesc{}; queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ComPtr<ID3D12CommandQueue> queue; HR_CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
  ComPtr<ID3D12CommandAllocator> allocator; HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
  ComPtr<ID3D12GraphicsCommandList> list; HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
      allocator.Get(), pipeline.Get(), IID_PPV_ARGS(&list)));
  list->SetComputeRootSignature(root.Get()); list->SetPipelineState(pipeline.Get());
  list->SetComputeRootShaderResourceView(0, input0Upload->GetGPUVirtualAddress());
  list->SetComputeRootShaderResourceView(1, input1Upload->GetGPUVirtualAddress());
  if (finalMode) list->SetComputeRootShaderResourceView(2, input2Upload->GetGPUVirtualAddress());
  list->SetComputeRootUnorderedAccessView(3, output->GetGPUVirtualAddress());
  const uint32_t dispatchItems = finalMode ? kWorkItems * 32 : kWorkItems;
  const uint32_t threadsPerGroup = softmaxMode ? 64 : 256;
  list->Dispatch((dispatchItems + threadsPerGroup - 1) / threadsPerGroup, 1, 1);
  D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = output.Get(); barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE; barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  list->ResourceBarrier(1, &barrier); list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, outputBytes);
  HR_CHECK(list->Close()); ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
  ComPtr<ID3D12Fence> fence; HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
  HR_CHECK(queue->Signal(fence.Get(), 1)); HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!event) throw std::runtime_error("CreateEvent failed");
  if (fence->GetCompletedValue() < 1) { HR_CHECK(fence->SetEventOnCompletion(1, event)); WaitForSingleObject(event, INFINITE); }
  CloseHandle(event);

  void* mapped = nullptr; D3D12_RANGE range{0, SIZE_T(outputBytes)}; HR_CHECK(readback->Map(0, &range, &mapped));
  auto actual = std::vector<uint8_t>(static_cast<size_t>(outputBytes)); std::memcpy(actual.data(), mapped, actual.size()); readback->Unmap(0, nullptr);
  uint64_t mismatches = 0; for (size_t i = 0; i < actual.size(); ++i) mismatches += actual[i] != expected[i];
  uint64_t halfMismatches = 0, nonFinite = 0; double absoluteSum = 0.0; float maxAbsolute = 0.0f;
  if (softmaxMode || tailMode || finalMode) {
    const auto* actualHalf = reinterpret_cast<const uint16_t*>(actual.data());
    const auto* expectedHalf = reinterpret_cast<const uint16_t*>(expected.data());
    for (size_t i = 0; i < actual.size() / 2; ++i) {
      halfMismatches += actualHalf[i] != expectedHalf[i];
      const float value = HalfToFloat(actualHalf[i]);
      const float difference = std::abs(value - HalfToFloat(expectedHalf[i]));
      nonFinite += !std::isfinite(value); absoluteSum += difference;
      maxAbsolute = std::max(maxAbsolute, difference);
    }
  }
  const double halfMismatchRatio = (softmaxMode || tailMode || finalMode) ? double(halfMismatches) / double(actual.size() / 2) : 0.0;
  const double meanAbsolute = (softmaxMode || tailMode || finalMode) ? absoluteSum / double(actual.size() / 2) : 0.0;
  const bool pass = softmaxMode ? nonFinite == 0 && meanAbsolute <= 0.1 && maxAbsolute <= 0.5f :
      (finalMode ? nonFinite == 0 && meanAbsolute <= 0.25 && maxAbsolute <= 2.0f :
      (tailMode ? nonFinite == 0 && halfMismatchRatio <= 0.005 && maxAbsolute <= 0.01f : mismatches == 0));
  Write(argv[outputIndex], actual.data(), actual.size());
  char hash[32]{}; std::snprintf(hash, sizeof(hash), "%016llX", (unsigned long long)Hash(actual.data(), actual.size()));
  const std::string experiment = softmaxMode ? "output_head_softmax_v_d3d12" : (finalMode ? "output_head_final_projection_d3d12" :
      (tailMode ? "output_head_tail_d3d12" : "output_head_surface_d3d12"));
  const std::string classification = softmaxMode ? "SAME_COMMAND_LIST_D3D12_COMPUTE_SOFTMAX_V" : (finalMode ? "SAME_COMMAND_LIST_D3D12_COMPUTE_FINAL_PROJECTION" :
      (tailMode ? "SAME_COMMAND_LIST_D3D12_COMPUTE_FP16_TAIL" : "SAME_COMMAND_LIST_D3D12_COMPUTE_SURFACE"));
  std::string manifest = "{\n  \"schema\": 1,\n  \"experiment\": \"" + experiment + "\",\n  \"status\": \"" +
      std::string(pass ? "PASS" : "FAIL") + "\",\n  \"classification\": \"" + classification + "\",\n" +
      "  \"d3d12_vendor_id\": \"0x1002\",\n  \"input0_bytes\": " + std::to_string(input0.size()) +
      ",\n  \"output_bytes\": " + std::to_string(actual.size()) + ",\n  \"byte_mismatches\": " + std::to_string(mismatches) +
      ",\n  \"half_mismatches\": " + std::to_string(halfMismatches) +
      ",\n  \"half_mismatch_ratio\": " + FormatNumber(halfMismatchRatio) +
      ",\n  \"mean_absolute_error\": " + FormatNumber(meanAbsolute) +
      ",\n  \"max_absolute_error\": " + FormatNumber(maxAbsolute) +
      ",\n  \"non_finite_values\": " + std::to_string(nonFinite) +
      ",\n  \"output_fnv1a64\": \"" + hash + "\",\n  \"same_command_list_compute\": true,\n  \"game_runtime_ready\": false\n}\n";
  Write(argv[manifestIndex], manifest.data(), manifest.size());
  std::printf("[%s] D3D12 output %s byte mismatches=%llu\n", pass ? "PASS" : "FAIL",
      softmaxMode ? "softmax/V" : (finalMode ? "final projection" : (tailMode ? "FP16 tail" : "surface")), (unsigned long long)mismatches);
  return pass ? 0 : 1;
} catch (const std::exception& error) { std::fprintf(stderr, "ERROR: %s\n", error.what()); return 1; } }
