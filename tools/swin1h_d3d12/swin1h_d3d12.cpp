#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"

// Operator-family prototype. Matrix/attention shaders are shared with the
// reconstructed output head, not translated PTX. No reference data is consumed.
int main(int argc, char** argv) { try {
    if (argc != 8) throw std::invalid_argument("usage: swin1h_d3d12 <input.e4> <weights.raw> <width> <height> <originX> <originY> <output_directory>");
    unsigned width = std::stoul(argv[3]), height = std::stoul(argv[4]);
    int originX = std::stoi(argv[5]), originY = std::stoi(argv[6]);
    if (!width || width > 640 || !height || height > 384 || width % 8 || height % 8 ||
        (originX != 0 && originX != -4) || (originY != 0 && originY != -4))
        throw std::invalid_argument("unsupported tiled geometry");
    unsigned gx = (width - originX + 7) / 8, gy = (height - originY + 7) / 8;
    const uint64_t cells = uint64_t(gx) * gy, e4Bytes = cells * 2048, halfBytes = e4Bytes * 2;
    auto input = Read(argv[1]), weights = ReadSlice(argv[2], 0, 20672);
    if (input.size() != uint64_t(width) * height * 32 || weights.size() != 20672)
        throw std::invalid_argument("expected tiled E4 input and 20672-byte chained weights");
    fs::path outDir = argv[7]; fs::create_directories(outDir);
    ComPtr<ID3D12Debug> debug; bool debugEnabled = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    if (debugEnabled) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory; HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; DXGI_ADAPTER_DESC1 ad{}; bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
        HR_CHECK(adapter->GetDesc1(&ad));
        if (!(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && ad.VendorId == 0x1002) { found = true; break; }
    }
    if (!found) throw std::runtime_error("AMD adapter not found");
    ComPtr<ID3D12Device> device; HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    ComPtr<ID3D12InfoQueue> info; device.As(&info);
    D3D12_ROOT_PARAMETER params[6]{};
    for (unsigned i = 0; i < 5; ++i) {
        params[i].ParameterType = i < 3 ? D3D12_ROOT_PARAMETER_TYPE_SRV : D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[i].Descriptor.ShaderRegister = i < 3 ? i : i - 3;
    }
    params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[5].Constants.Num32BitValues = 6;
    D3D12_ROOT_SIGNATURE_DESC rd{}; rd.NumParameters = 6; rd.pParameters = params;
    ComPtr<ID3DBlob> blob, errors; HR_CHECK(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
    ComPtr<ID3D12RootSignature> root;
    HR_CHECK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
    const char* names[] = {"swin1h_gather", "swin1h_first128_hidden", "swin1h_first128_project",
        "swin1h_mma128_175", "swin1h_prepare_qk", "swin1h_prepare_v", "swin1h_qk",
        "swin1h_softmax_v", "swin1h_final_projection", "swin1h_scatter"};
    std::array<ComPtr<ID3D12PipelineState>, 10> pipelines;
    auto directory = fs::absolute(argv[0]).parent_path();
    for (unsigned i = 0; i < 10; ++i) pipelines[i] = Pipeline(device.Get(), root.Get(), Read(directory / (std::string(names[i]) + ".dxil")));
    const uint64_t sizes[] = {e4Bytes, halfBytes, e4Bytes * 4, halfBytes, halfBytes * 3,
        e4Bytes, e4Bytes, e4Bytes, halfBytes * 2, halfBytes, halfBytes};
    std::array<ComPtr<ID3D12Resource>, 11> scratch;
    for (unsigned i = 0; i < 11; ++i) scratch[i] = DefaultBuffer(device.Get(), sizes[i]);
    auto dInput = DefaultBuffer(device.Get(), input.size()), dWeights = DefaultBuffer(device.Get(), weights.size());
    auto dOutput = DefaultBuffer(device.Get(), input.size());
    auto uInput = Upload(device.Get(), input), uWeights = Upload(device.Get(), weights);
    auto uZero = Upload(device.Get(), std::vector<uint8_t>(input.size(), 0));
    auto rb0 = Readback(device.Get(), input.size()), rb1 = Readback(device.Get(), input.size());
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue; HR_CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator; HR_CHECK(device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list; HR_CHECK(device->CreateCommandList(0, qd.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
    auto upload = [&](ID3D12Resource* dst, ID3D12Resource* src, uint64_t size, D3D12_RESOURCE_STATES finalState) {
        Recorder::Transition(list.Get(), dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(dst, 0, src, 0, size);
        Recorder::Transition(list.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST, finalState);
    };
    upload(dInput.Get(), uInput.Get(), input.size(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    upload(dWeights.Get(), uWeights.Get(), weights.size(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    uint32_t config[] = {width, height, gx, gy, uint32_t(originX), uint32_t(originY)};
    list->SetComputeRootSignature(root.Get()); list->SetComputeRoot32BitConstants(5, 6, config, 0);
    auto dispatch = [&](unsigned p, ID3D12Resource* a, ID3D12Resource* b, ID3D12Resource* c,
                        ID3D12Resource* dst, uint64_t groups, ID3D12Resource* second = nullptr) {
        list->SetPipelineState(pipelines[p].Get());
        list->SetComputeRootShaderResourceView(0, a->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(1, (b ? b : a)->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(2, (c ? c : a)->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(3, dst->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(4, (second ? second : dst)->GetGPUVirtualAddress());
        list->Dispatch(UINT(groups), 1, 1);
    };
    auto s = [&](unsigned i) { return scratch[i].Get(); };
    auto ready = [&](unsigned i) { Recorder::Transition(list.Get(), s(i), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); };
    for (unsigned run = 0; run < 2; ++run) {
        upload(dOutput.Get(), uZero.Get(), input.size(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(0, dInput.Get(), nullptr, nullptr, s(0), e4Bytes / 1024, s(1)); ready(0); ready(1);
        dispatch(1, s(0), dWeights.Get(), nullptr, s(2), e4Bytes / 256); ready(2);
        dispatch(2, s(1), s(2), dWeights.Get(), s(3), halfBytes / 1024); ready(3);
        dispatch(3, s(3), dWeights.Get(), nullptr, s(4), halfBytes * 3 / 1024); ready(4);
        dispatch(4, s(4), dWeights.Get(), nullptr, s(5), cells, s(6)); ready(5); ready(6);
        dispatch(5, s(4), nullptr, nullptr, s(7), e4Bytes / 1024); ready(7);
        dispatch(6, s(5), s(6), dWeights.Get(), s(8), halfBytes * 2 / 1024); ready(8);
        dispatch(7, s(8), s(7), nullptr, s(9), cells); ready(9);
        dispatch(8, s(9), s(3), dWeights.Get(), s(10), cells * 8); ready(10);
        dispatch(9, s(10), nullptr, nullptr, dOutput.Get(), e4Bytes / 1024);
        Recorder::Transition(list.Get(), dOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(run ? rb1.Get() : rb0.Get(), 0, dOutput.Get(), 0, input.size());
        Recorder::Transition(list.Get(), dOutput.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (unsigned i = 0; i < 11; ++i) Recorder::Transition(list.Get(), s(i), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    HR_CHECK(list->Close()); ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence; HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(), 1)); HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("event creation failed");
    HR_CHECK(fence->SetEventOnCompletion(1, event)); auto wait = WaitForSingleObject(event, 60000); CloseHandle(event);
    if (wait != WAIT_OBJECT_0) throw std::runtime_error("GPU wait failed"); HR_CHECK(device->GetDeviceRemovedReason());
    auto actual = ReadGpu(rb0.Get(), input.size()), repeat = ReadGpu(rb1.Get(), input.size());
    uint64_t debugErrors = 0, nonfinite = 0;
    if (info) for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T bytes = 0; info->GetMessage(i, nullptr, &bytes); std::vector<uint8_t> message(bytes);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(message.data()); HR_CHECK(info->GetMessage(i, m, &bytes));
        if (m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) { ++debugErrors; std::fprintf(stderr, "%s\n", m->pDescription); }
    }
    for (auto v : actual) nonfinite += (v & 127) == 127;
    bool pass = actual == repeat && !nonfinite && !debugErrors;
    Write(outDir / "output.e4", actual.data(), actual.size());
    std::ostringstream json;
    json << "{\n  \"status\": \"" << (pass ? "EXECUTION_PASS" : "FAIL") << "\",\n"
         << "  \"backend\": \"D3D12_DXIL\",\n  \"dispatches_per_run\": 10,\n"
         << "  \"repeat_exact\": " << (actual == repeat ? "true" : "false") << ",\n"
         << "  \"nonfinite\": " << nonfinite << ",\n  \"debug_errors\": " << debugErrors << ",\n"
         << "  \"oracle_inputs\": false,\n  \"native_graph_complete\": false\n}\n";
    auto manifest = json.str(); Write(outDir / "manifest.json", manifest.data(), manifest.size());
    std::printf("%s", manifest.c_str()); return pass ? 0 : 1;
} catch (const std::exception& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; } }
