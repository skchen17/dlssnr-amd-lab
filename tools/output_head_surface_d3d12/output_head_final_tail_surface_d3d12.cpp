#define main output_head_tail_surface_file_cli
#include "output_head_tail_surface_d3d12.cpp"
#undef main

namespace {
struct Metrics {
  uint64_t mismatches = 0;
  uint64_t nonFinite = 0;
  double meanAbsolute = 0.0;
  float maxAbsolute = 0.0f;
};

Metrics CompareHalf(const std::vector<uint8_t> &actual,
                    const std::vector<uint8_t> &expected) {
  Metrics metrics{};
  const auto *a = reinterpret_cast<const uint16_t *>(actual.data());
  const auto *e = reinterpret_cast<const uint16_t *>(expected.data());
  const size_t count = actual.size() / 2;
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    metrics.mismatches += a[i] != e[i];
    const float value = HalfToFloat(a[i]);
    const float difference = std::abs(value - HalfToFloat(e[i]));
    metrics.nonFinite += !std::isfinite(value);
    sum += difference;
    metrics.maxAbsolute = std::max(metrics.maxAbsolute, difference);
  }
  metrics.meanAbsolute = sum / double(count);
  return metrics;
}

std::vector<uint8_t> Download(ID3D12Resource *readback, uint64_t bytes) {
  std::vector<uint8_t> output(static_cast<size_t>(bytes));
  void *mapped = nullptr;
  D3D12_RANGE range{0, SIZE_T(bytes)};
  HR_CHECK(readback->Map(0, &range, &mapped));
  std::memcpy(output.data(), mapped, output.size());
  readback->Unmap(0, nullptr);
  return output;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 12) {
      std::fprintf(stderr,
                   "usage: output_head_final_tail_surface_d3d12 "
                   "<attention.raw> <first128.raw> <model.raw> <base.raw|-> "
                   "<expected_projected.raw> <expected_residual.raw> "
                   "<expected_surface.raw> <projected_out.raw> "
                   "<residual_out.raw> <surface_out.raw> <manifest.json>\n");
      return 2;
    }
    auto attention = Read(argv[1]);
    auto first128 = Read(argv[2]);
    auto head = ReadSlice(argv[3], kHeadOffset, kHeadBytes);
    auto base = std::string(argv[4]) == "-"
                    ? std::vector<uint8_t>(size_t(kSurfaceBytes), 0)
                    : Read(argv[4]);
    auto expectedProjected = Read(argv[5]);
    auto expectedResidual = Read(argv[6]);
    auto expectedSurface = Read(argv[7]);
    if (attention.size() != kProjectedBytes ||
        first128.size() != kProjectedBytes || head.size() != kHeadBytes ||
        base.size() != kSurfaceBytes ||
        expectedProjected.size() != kProjectedBytes ||
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

    D3D12_ROOT_PARAMETER params[4]{};
    for (uint32_t i = 0; i < 3; ++i) {
      params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
      params[i].Descriptor.ShaderRegister = i;
      params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor.ShaderRegister = 0;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = 4;
    rootDesc.pParameters = params;
    ComPtr<ID3DBlob> rootBlob, errors;
    HR_CHECK(D3D12SerializeRootSignature(
        &rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errors));
    ComPtr<ID3D12RootSignature> root;
    HR_CHECK(device->CreateRootSignature(0, rootBlob->GetBufferPointer(),
                                        rootBlob->GetBufferSize(),
                                        IID_PPV_ARGS(&root)));

    const fs::path build = fs::absolute(argv[0]).parent_path();
    auto finalPipeline = Pipeline(
        device.Get(), root.Get(),
        Read(build / "output_head_final_projection_d3d12.dxil"));
    auto tailPipeline = Pipeline(device.Get(), root.Get(),
                                 Read(build / "output_head_tail_d3d12.dxil"));
    auto surfacePipeline = Pipeline(
        device.Get(), root.Get(),
        Read(build / "output_head_surface_d3d12.dxil"));
    auto attentionUpload = Upload(device.Get(), attention);
    auto first128Upload = Upload(device.Get(), first128);
    auto headUpload = Upload(device.Get(), head);
    auto baseUpload = Upload(device.Get(), base);
    auto projected = DefaultBuffer(device.Get(), kProjectedBytes);
    auto residual = DefaultBuffer(device.Get(), kResidualBytes);
    auto surface = DefaultBuffer(device.Get(), kSurfaceBytes);
    auto projectedReadback = Readback(device.Get(), kProjectedBytes);
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
                                      allocator.Get(), finalPipeline.Get(),
                                      IID_PPV_ARGS(&list)));
    list->SetComputeRootSignature(root.Get());

    list->SetPipelineState(finalPipeline.Get());
    list->SetComputeRootShaderResourceView(
        0, attentionUpload->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(
        1, first128Upload->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(2, headUpload->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(3,
                                            projected->GetGPUVirtualAddress());
    list->Dispatch((kWorkItems * 32 + 255) / 256, 1, 1);
    Transition(list.Get(), projected.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    list->SetPipelineState(tailPipeline.Get());
    list->SetComputeRootShaderResourceView(0, projected->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(1, headUpload->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(3,
                                            residual->GetGPUVirtualAddress());
    list->Dispatch((kWorkItems + 255) / 256, 1, 1);
    Transition(list.Get(), residual.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    list->SetPipelineState(surfacePipeline.Get());
    list->SetComputeRootShaderResourceView(0, residual->GetGPUVirtualAddress());
    list->SetComputeRootShaderResourceView(1, baseUpload->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(3, surface->GetGPUVirtualAddress());
    list->Dispatch((kWorkItems + 255) / 256, 1, 1);

    Transition(list.Get(), projected.Get(),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(list.Get(), residual.Get(),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(list.Get(), surface.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(projectedReadback.Get(), 0, projected.Get(), 0,
                           kProjectedBytes);
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

    auto actualProjected = Download(projectedReadback.Get(), kProjectedBytes);
    auto actualResidual = Download(residualReadback.Get(), kResidualBytes);
    auto actualSurface = Download(surfaceReadback.Get(), kSurfaceBytes);
    const Metrics projectedMetrics =
        CompareHalf(actualProjected, expectedProjected);
    const Metrics residualMetrics = CompareHalf(actualResidual, expectedResidual);
    const Metrics surfaceMetrics = CompareHalf(actualSurface, expectedSurface);
    const double residualRatio =
        double(residualMetrics.mismatches) / double(kResidualBytes / 2);
    const double surfaceRatio =
        double(surfaceMetrics.mismatches) / double(kSurfaceBytes / 2);
    const bool pass = projectedMetrics.mismatches == 0 &&
                      residualMetrics.nonFinite == 0 &&
                      surfaceMetrics.nonFinite == 0 && residualRatio <= 0.005 &&
                      surfaceRatio <= 0.005 &&
                      residualMetrics.maxAbsolute <= 0.01f &&
                      surfaceMetrics.maxAbsolute <= 0.0001f;
    Write(argv[8], actualProjected.data(), actualProjected.size());
    Write(argv[9], actualResidual.data(), actualResidual.size());
    Write(argv[10], actualSurface.data(), actualSurface.size());
    const std::string manifest =
        "{\n  \"schema\": 1,\n  \"experiment\": "
        "\"output_head_final_tail_surface_d3d12\",\n  \"status\": \"" +
        std::string(pass ? "PASS" : "FAIL") +
        "\",\n  \"classification\": "
        "\"SAME_COMMAND_LIST_D3D12_FINAL_TAIL_SURFACE\",\n  "
        "\"d3d12_vendor_id\": \"0x1002\",\n  \"single_command_list\": "
        "true,\n  \"hip_runtime_dependency\": false,\n  "
        "\"projected_half_mismatches\": " +
        std::to_string(projectedMetrics.mismatches) +
        ",\n  \"projected_mean_absolute_error\": " +
        FormatNumber(projectedMetrics.meanAbsolute) +
        ",\n  \"projected_max_absolute_error\": " +
        FormatNumber(projectedMetrics.maxAbsolute) +
        ",\n  \"residual_half_mismatches\": " +
        std::to_string(residualMetrics.mismatches) +
        ",\n  \"residual_mismatch_ratio\": " + FormatNumber(residualRatio) +
        ",\n  \"residual_mean_absolute_error\": " +
        FormatNumber(residualMetrics.meanAbsolute) +
        ",\n  \"residual_max_absolute_error\": " +
        FormatNumber(residualMetrics.maxAbsolute) +
        ",\n  \"surface_half_mismatches\": " +
        std::to_string(surfaceMetrics.mismatches) +
        ",\n  \"surface_mismatch_ratio\": " + FormatNumber(surfaceRatio) +
        ",\n  \"surface_mean_absolute_error\": " +
        FormatNumber(surfaceMetrics.meanAbsolute) +
        ",\n  \"surface_max_absolute_error\": " +
        FormatNumber(surfaceMetrics.maxAbsolute) +
        ",\n  \"non_finite_values\": " +
        std::to_string(projectedMetrics.nonFinite +
                       residualMetrics.nonFinite + surfaceMetrics.nonFinite) +
        ",\n  \"game_runtime_ready\": false,\n  \"next_gate\": "
        "\"Port the attention producer into the same D3D12 command list\"\n}\n";
    Write(argv[11], manifest.data(), manifest.size());
    std::printf("[%s] D3D12 final+tail+surface: projected=%llu residual=%llu "
                "surface=%llu\n",
                pass ? "PASS" : "FAIL",
                (unsigned long long)projectedMetrics.mismatches,
                (unsigned long long)residualMetrics.mismatches,
                (unsigned long long)surfaceMetrics.mismatches);
    return pass ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
  }
}
