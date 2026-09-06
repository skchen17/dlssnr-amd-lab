// End-to-end D3D12 texture -> linear shared heap -> HIP residual composition
// -> D3D12 texture bridge.  This deliberately stages textures because direct
// raw HIP views of AMD D3D12 textures use a vendor-tiled layout.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) \
  throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); } while (0)
#define HR_CHECK(x) do { HRESULT h_ = (x); if (FAILED(h_)) { char b_[128]; \
  std::snprintf(b_, sizeof(b_), #x " failed: 0x%08lX", (unsigned long)h_); \
  throw std::runtime_error(b_); } } while (0)

namespace {
constexpr uint32_t kGridWidth = 81, kGridHeight = 49, kTokens = 64;
constexpr uint32_t kWidth = 640, kHeight = 360, kChannels = 4;
constexpr size_t kResidualBytes = size_t(kGridWidth) * kGridHeight * kTokens * kChannels * 2;
constexpr size_t kSurfaceBytes = size_t(kWidth) * kHeight * kChannels * 2;

size_t Align(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

std::vector<uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path.string());
  const size_t size = size_t(stream.tellg());
  stream.seekg(0);
  std::vector<uint8_t> data(size);
  if (!stream.read(reinterpret_cast<char*>(data.data()), std::streamsize(size)))
    throw std::runtime_error("read failed: " + path.string());
  return data;
}

void Write(const std::filesystem::path& path, const void* data, size_t size) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary);
  if (!stream || !stream.write(reinterpret_cast<const char*>(data), std::streamsize(size)))
    throw std::runtime_error("write failed: " + path.string());
}

uint64_t Hash(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  uint64_t hash = 1469598103934665603ull;
  for (size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

__global__ void ComposeBridge(const uint16_t* base, const uint16_t* residual,
                              uint16_t* output) {
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
  if (x < 0 || y < 0 || x >= int32_t(kWidth) || y >= int32_t(kHeight)) return;
  const size_t source = index * kChannels;
  const size_t destination = (size_t(y) * kWidth + uint32_t(x)) * kChannels;
  for (uint32_t channel = 0; channel < 3; ++channel) {
    const float b = __half2float(__ushort_as_half(base[destination + channel]));
    const float r = __half2float(__ushort_as_half(residual[source + channel]));
    output[destination + channel] = __half_as_ushort(__float2half_rn(
        fminf(1.0f, fmaxf(0.0f, b + 0.25f * r))));
  }
  output[destination + 3] = __half_as_ushort(__float2half_rn(1.0f));
}

D3D12_RESOURCE_DESC BufferDesc(UINT64 bytes) {
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = bytes;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = DXGI_FORMAT_UNKNOWN;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  return desc;
}

D3D12_RESOURCE_DESC TextureDesc() {
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = kWidth;
  desc.Height = kHeight;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  return desc;
}

void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  list->ResourceBarrier(1, &barrier);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 5) {
      std::fprintf(stderr, "usage: d3d12_residual_bridge <rgba_residual_fp16.raw> "
                           "<output_rgba16f.raw> <result.json> [iterations]\n");
      return 2;
    }
    const uint32_t iterations = argc == 5 ? uint32_t(std::stoul(argv[4])) : 100;
    const auto residual = Read(argv[1]);
    if (residual.size() != kResidualBytes) throw std::runtime_error("unexpected residual size");

    ComPtr<IDXGIFactory7> factory;
    HR_CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    bool found = false;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND;
         ++index, adapter.Reset()) {
      HR_CHECK(adapter->GetDesc1(&adapter_desc));
      if (!(adapter_desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && adapter_desc.VendorId == 0x1002) {
        found = true;
        break;
      }
    }
    if (!found) throw std::runtime_error("AMD D3D12 adapter not found");
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t hip_properties{};
    HIP_CHECK(hipGetDeviceProperties(&hip_properties, 0));

    ComPtr<ID3D12Device> device;
    HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                               IID_PPV_ARGS(&device)));
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    HR_CHECK(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&allocator)));
    HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                       nullptr, IID_PPV_ARGS(&list)));
    HR_CHECK(list->Close());

    ComPtr<ID3D12Fence> shared_fence, cpu_fence;
    HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&shared_fence)));
    HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cpu_fence)));
    UINT64 cpu_fence_value = 0;
    auto ExecuteAndWait = [&]() {
      ID3D12CommandList* lists[] = {list.Get()};
      queue->ExecuteCommandLists(1, lists);
      HR_CHECK(queue->Signal(cpu_fence.Get(), ++cpu_fence_value));
      while (cpu_fence->GetCompletedValue() < cpu_fence_value) Sleep(0);
    };

    const auto texture_desc = TextureDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, texture_copy_bytes = 0;
    device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, &rows,
                                  &row_bytes, &texture_copy_bytes);
    if (row_bytes != kWidth * kChannels * 2 || rows != kHeight)
      throw std::runtime_error("unexpected RGBA16F footprint");

    const size_t base_offset = 0;
    const size_t residual_offset = Align(texture_copy_bytes, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    const size_t output_offset = Align(residual_offset + kResidualBytes,
                                       D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    const size_t heap_bytes = Align(output_offset + texture_copy_bytes,
                                    D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    D3D12_HEAP_DESC heap_desc{};
    heap_desc.SizeInBytes = heap_bytes;
    heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_desc.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ComPtr<ID3D12Heap> shared_heap;
    HR_CHECK(device->CreateHeap(&heap_desc, IID_PPV_ARGS(&shared_heap)));
    const auto shared_desc = BufferDesc(heap_bytes);
    ComPtr<ID3D12Resource> shared_buffer;
    HR_CHECK(device->CreatePlacedResource(shared_heap.Get(), 0, &shared_desc,
                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                           IID_PPV_ARGS(&shared_buffer)));

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> base_texture, output_texture;
    HR_CHECK(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE,
        &texture_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&base_texture)));
    HR_CHECK(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE,
        &texture_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&output_texture)));

    D3D12_HEAP_PROPERTIES upload_heap{};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES readback_heap{};
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> base_upload, residual_upload, readback;
    auto texture_buffer_desc = BufferDesc(texture_copy_bytes);
    auto residual_buffer_desc = BufferDesc(kResidualBytes);
    HR_CHECK(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE,
        &texture_buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&base_upload)));
    HR_CHECK(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE,
        &residual_buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&residual_upload)));
    HR_CHECK(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE,
        &texture_buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback)));

    std::vector<uint16_t> tight_base(kSurfaceBytes / 2);
    for (size_t pixel = 0; pixel < size_t(kWidth) * kHeight; ++pixel) {
      const uint32_t x = uint32_t(pixel % kWidth), y = uint32_t(pixel / kWidth);
      const float values[4] = {
          float((x % 17) + 2) / 32.0f,
          float((y % 13) + 3) / 32.0f,
          float(((x + y) % 19) + 4) / 32.0f,
          1.0f};
      for (uint32_t channel = 0; channel < 4; ++channel)
        tight_base[pixel * 4 + channel] = __half_as_ushort(__float2half_rn(values[channel]));
    }
    void* mapped = nullptr;
    HR_CHECK(base_upload->Map(0, nullptr, &mapped));
    std::memset(mapped, 0, size_t(texture_copy_bytes));
    for (uint32_t row = 0; row < kHeight; ++row)
      std::memcpy(static_cast<uint8_t*>(mapped) + footprint.Footprint.RowPitch * row,
                  tight_base.data() + size_t(row) * kWidth * 4, kWidth * 8);
    base_upload->Unmap(0, nullptr);
    HR_CHECK(residual_upload->Map(0, nullptr, &mapped));
    std::memcpy(mapped, residual.data(), kResidualBytes);
    residual_upload->Unmap(0, nullptr);

    HR_CHECK(allocator->Reset());
    HR_CHECK(list->Reset(allocator.Get(), nullptr));
    Transition(list.Get(), base_texture.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION texture_destination{}, upload_source{};
    texture_destination.pResource = base_texture.Get();
    texture_destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    upload_source.pResource = base_upload.Get();
    upload_source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    upload_source.PlacedFootprint = footprint;
    list->CopyTextureRegion(&texture_destination, 0, 0, 0, &upload_source, nullptr);
    Transition(list.Get(), base_texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(list.Get(), shared_buffer.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION shared_base_destination{}, base_source{};
    shared_base_destination.pResource = shared_buffer.Get();
    shared_base_destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    shared_base_destination.PlacedFootprint = footprint;
    shared_base_destination.PlacedFootprint.Offset = base_offset;
    base_source.pResource = base_texture.Get();
    base_source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&shared_base_destination, 0, 0, 0, &base_source, nullptr);
    list->CopyBufferRegion(shared_buffer.Get(), residual_offset, residual_upload.Get(), 0,
                           kResidualBytes);
    Transition(list.Get(), shared_buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_COMMON);
    Transition(list.Get(), base_texture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    HR_CHECK(list->Close());
    ExecuteAndWait();
    HR_CHECK(queue->Signal(shared_fence.Get(), 1));

    HANDLE heap_handle = nullptr, fence_handle = nullptr;
    HR_CHECK(device->CreateSharedHandle(shared_heap.Get(), nullptr, GENERIC_ALL, nullptr,
                                        &heap_handle));
    HR_CHECK(device->CreateSharedHandle(shared_fence.Get(), nullptr, GENERIC_ALL, nullptr,
                                        &fence_handle));
    hipExternalMemoryHandleDesc memory_desc{};
    memory_desc.type = hipExternalMemoryHandleTypeD3D12Heap;
    memory_desc.handle.win32.handle = heap_handle;
    memory_desc.size = heap_bytes;
    hipExternalMemory_t external_memory = nullptr;
    HIP_CHECK(hipImportExternalMemory(&external_memory, &memory_desc));
    hipExternalMemoryBufferDesc mapping_desc{};
    mapping_desc.size = heap_bytes;
    void* shared_pointer = nullptr;
    HIP_CHECK(hipExternalMemoryGetMappedBuffer(&shared_pointer, external_memory, &mapping_desc));
    hipExternalSemaphoreHandleDesc semaphore_desc{};
    semaphore_desc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
    semaphore_desc.handle.win32.handle = fence_handle;
    hipExternalSemaphore_t external_fence = nullptr;
    HIP_CHECK(hipImportExternalSemaphore(&external_fence, &semaphore_desc));
    CloseHandle(heap_handle);
    CloseHandle(fence_handle);

    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    hipExternalSemaphoreWaitParams wait_params{};
    wait_params.params.fence.value = 1;
    HIP_CHECK(hipWaitExternalSemaphoresAsync(&external_fence, &wait_params, 1, stream));
    auto* shared_bytes = static_cast<uint8_t*>(shared_pointer);
    const auto* device_base = reinterpret_cast<const uint16_t*>(shared_bytes + base_offset);
    const auto* device_residual = reinterpret_cast<const uint16_t*>(shared_bytes + residual_offset);
    auto* device_output = reinterpret_cast<uint16_t*>(shared_bytes + output_offset);
    const size_t logical_count = size_t(kGridWidth) * kGridHeight * kTokens;
    HIP_CHECK(hipMemsetAsync(device_output, 0, texture_copy_bytes, stream));
    hipEvent_t start{}, stop{};
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, stream));
    for (uint32_t index = 0; index < iterations; ++index)
      hipLaunchKernelGGL(ComposeBridge, dim3((logical_count + 255) / 256), dim3(256),
                         0, stream, device_base, device_residual, device_output);
    HIP_CHECK(hipEventRecord(stop, stream));
    hipExternalSemaphoreSignalParams signal_params{};
    signal_params.params.fence.value = 2;
    HIP_CHECK(hipSignalExternalSemaphoresAsync(&external_fence, &signal_params, 1, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float kernel_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));

    HR_CHECK(queue->Wait(shared_fence.Get(), 2));
    HR_CHECK(allocator->Reset());
    HR_CHECK(list->Reset(allocator.Get(), nullptr));
    Transition(list.Get(), shared_buffer.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(list.Get(), output_texture.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION output_destination{}, shared_output_source{};
    output_destination.pResource = output_texture.Get();
    output_destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    shared_output_source.pResource = shared_buffer.Get();
    shared_output_source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    shared_output_source.PlacedFootprint = footprint;
    shared_output_source.PlacedFootprint.Offset = output_offset;
    list->CopyTextureRegion(&output_destination, 0, 0, 0, &shared_output_source, nullptr);
    Transition(list.Get(), output_texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION readback_destination{}, output_source{};
    readback_destination.pResource = readback.Get();
    readback_destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    readback_destination.PlacedFootprint = footprint;
    output_source.pResource = output_texture.Get();
    output_source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&readback_destination, 0, 0, 0, &output_source, nullptr);
    Transition(list.Get(), output_texture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    Transition(list.Get(), shared_buffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    HR_CHECK(list->Close());
    ExecuteAndWait();

    std::vector<uint16_t> tight_output(kSurfaceBytes / 2);
    HR_CHECK(readback->Map(0, nullptr, &mapped));
    for (uint32_t row = 0; row < kHeight; ++row)
      std::memcpy(tight_output.data() + size_t(row) * kWidth * 4,
                  static_cast<const uint8_t*>(mapped) + footprint.Footprint.RowPitch * row,
                  kWidth * 8);
    readback->Unmap(0, nullptr);

    uint64_t mismatches = 0, changed_rgb = 0, nonfinite = 0;
    std::vector<uint16_t> expected(kSurfaceBytes / 2);
    for (size_t cta = 0; cta < size_t(kGridWidth) * kGridHeight; ++cta) {
      for (uint32_t token = 0; token < kTokens; ++token) {
        const uint32_t group = token / 16, within = token % 16;
        const int32_t x = int32_t((cta % kGridWidth) * 8) - 4 +
                          int32_t((group & 1) * 4 + within % 4);
        const int32_t y = int32_t((cta / kGridWidth) * 8) - 4 +
                          int32_t((group >> 1) * 4 + within / 4);
        if (x < 0 || y < 0 || x >= int32_t(kWidth) || y >= int32_t(kHeight)) continue;
        const size_t source = (cta * kTokens + token) * 4;
        const size_t destination = (size_t(y) * kWidth + uint32_t(x)) * 4;
        for (uint32_t channel = 0; channel < 3; ++channel) {
          const float b = __half2float(__ushort_as_half(tight_base[destination + channel]));
          const float r = __half2float(__ushort_as_half(
              reinterpret_cast<const uint16_t*>(residual.data())[source + channel]));
          expected[destination + channel] = __half_as_ushort(__float2half_rn(
              std::min(1.0f, std::max(0.0f, b + 0.25f * r))));
          changed_rgb += expected[destination + channel] != tight_base[destination + channel];
        }
        expected[destination + 3] = __half_as_ushort(__float2half_rn(1.0f));
      }
    }
    for (size_t index = 0; index < expected.size(); ++index) {
      mismatches += expected[index] != tight_output[index];
      nonfinite += !std::isfinite(__half2float(__ushort_as_half(tight_output[index])));
    }
    Write(argv[2], tight_output.data(), kSurfaceBytes);
    char hash[32]{};
    std::snprintf(hash, sizeof(hash), "%016llX",
                  static_cast<unsigned long long>(Hash(tight_output.data(), kSurfaceBytes)));
    const bool pass = mismatches == 0 && changed_rgb != 0 && nonfinite == 0;
    const std::string json =
        "{\n"
        "  \"schema\": 1,\n"
        "  \"experiment\": \"d3d12_hip_residual_texture_bridge\",\n"
        "  \"status\": \"" + std::string(pass ? "PASS" : "FAIL") + "\",\n"
        "  \"classification\": \"AMD_NATIVE_GAME_TEXTURE_STAGING_GATE\",\n"
        "  \"d3d12_vendor_id\": \"0x1002\",\n"
        "  \"d3d12_device_id\": " + std::to_string(adapter_desc.DeviceId) + ",\n"
        "  \"hip_device\": \"" + std::string(hip_properties.name) + "\",\n"
        "  \"surface_shape\": [360, 640, 4],\n"
        "  \"format\": \"DXGI_FORMAT_R16G16B16A16_FLOAT\",\n"
        "  \"path\": \"D3D12 texture -> shared linear heap -> HIP -> shared linear heap -> D3D12 texture\",\n"
        "  \"external_fence_wait_signal\": true,\n"
        "  \"component_mismatches\": " + std::to_string(mismatches) + ",\n"
        "  \"changed_rgb_components\": " + std::to_string(changed_rgb) + ",\n"
        "  \"non_finite_components\": " + std::to_string(nonfinite) + ",\n"
        "  \"iterations\": " + std::to_string(iterations) + ",\n"
        "  \"average_hip_kernel_ms\": " + std::to_string(kernel_ms / float(iterations)) + ",\n"
        "  \"output_fnv1a64\": \"" + std::string(hash) + "\",\n"
        "  \"next_gate\": \"Wire this bridge to the feature-18 evaluate resource registry\"\n"
        "}\n";
    Write(argv[3], json.data(), json.size());

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipDestroyExternalSemaphore(external_fence));
    HIP_CHECK(hipDestroyExternalMemory(external_memory));
    std::printf("[%s] %s D3D12 residual bridge: mismatches=%llu changed=%llu %.6f ms\n",
                pass ? "PASS" : "FAIL", hip_properties.name,
                static_cast<unsigned long long>(mismatches),
                static_cast<unsigned long long>(changed_rgb), kernel_ms / float(iterations));
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
