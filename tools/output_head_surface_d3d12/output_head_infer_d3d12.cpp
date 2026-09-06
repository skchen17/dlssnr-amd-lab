#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "output_head_recording_selftest.cpp"

// Inference CLI: no RTX or AMD intermediate/output oracle is read. The caller
// provides current activation/model/base; comparisons belong in the experiment
// coordinator. This is also usable with an AMD-generated pre-head arena.
int main(int argc, char** argv) { try {
    if (argc != 6) throw std::invalid_argument("usage: output_head_infer_d3d12 <activation.raw> <model.raw> <base.raw|-> <output.raw> <manifest.json>");
    auto main = ReadSlice(argv[1], 13873152, output_head_dx12::MainBytes);
    auto skip = ReadSlice(argv[1], 110592, output_head_dx12::SkipBytes);
    auto head = ReadSlice(argv[2], kHeadOffset, kHeadBytes);
    auto baseData = std::string(argv[3]) == "-" ? std::vector<uint8_t>(size_t(kSurfaceBytes), 0) : Read(argv[3]);
    if (baseData.size() != kSurfaceBytes) throw std::invalid_argument("base must be 640x360 RGBA16F");
    ComPtr<IDXGIFactory6> factory; HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; DXGI_ADAPTER_DESC1 desc{}; bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
        HR_CHECK(adapter->GetDesc1(&desc));
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && desc.VendorId == 0x1002) { found = true; break; }
    }
    if (!found) throw std::runtime_error("AMD adapter not found");
    ComPtr<ID3D12Device> device; HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    Recorder::ShaderSet shaders; auto directory = fs::absolute(argv[0]).parent_path();
    for (size_t i = 0; i < shaders.size(); ++i) shaders[i] = Read(directory / (std::string(output_head_dx12::ShaderNames[i]) + ".dxil"));
    Recorder recorder(device.Get(), shaders);
    auto uMain = Upload(device.Get(), main), uSkip = Upload(device.Get(), skip), uHead = Upload(device.Get(), head), uBase = Upload(device.Get(), baseData);
    auto dMain = DefaultBuffer(device.Get(), main.size()), dSkip = DefaultBuffer(device.Get(), skip.size()), dHead = DefaultBuffer(device.Get(), head.size());
    auto base = Texture(device.Get()), output = Texture(device.Get());
    auto rb0 = Readback(device.Get(), kSurfaceBytes), rb1 = Readback(device.Get(), kSurfaceBytes), times = Readback(device.Get(), 32);
    D3D12_QUERY_HEAP_DESC qhd{}; qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qhd.Count = 4;
    ComPtr<ID3D12QueryHeap> queries; HR_CHECK(device->CreateQueryHeap(&qhd, IID_PPV_ARGS(&queries)));
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue; HR_CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator; HR_CHECK(device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list; HR_CHECK(device->CreateCommandList(0, qd.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
    auto copy = [&](ID3D12Resource* dst, ID3D12Resource* src, size_t bytes) {
        Recorder::Transition(list.Get(), dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(dst, 0, src, 0, bytes);
        Recorder::Transition(list.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    };
    copy(dMain.Get(), uMain.Get(), main.size()); copy(dSkip.Get(), uSkip.Get(), skip.size()); copy(dHead.Get(), uHead.Get(), head.size());
    Recorder::Transition(list.Get(), base.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    auto src = Recorder::LinearLocation(uBase.Get()), dst = Recorder::TextureLocation(base.Get());
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Recorder::Transition(list.Get(), base.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    output_head_dx12::Inputs inputs{{dMain.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
        {dSkip.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS}, {dHead.Get(), 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
        {base.Get(), D3D12_RESOURCE_STATE_COMMON}, {output.Get(), D3D12_RESOURCE_STATE_COMMON}};
    for (unsigned i = 0; i < 2; ++i) {
        list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2);
        recorder.Record(list.Get(), inputs);
        list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, i * 2 + 1);
        Recorder::Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        src = Recorder::TextureLocation(output.Get()); dst = Recorder::LinearLocation(i ? rb1.Get() : rb0.Get());
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Recorder::Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4, times.Get(), 0);
    HR_CHECK(list->Close()); ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence; HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(), 1)); HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("event creation failed");
    HR_CHECK(fence->SetEventOnCompletion(1, event)); DWORD wait = WaitForSingleObject(event, 60000); CloseHandle(event);
    if (wait != WAIT_OBJECT_0) throw std::runtime_error("GPU wait failed"); HR_CHECK(device->GetDeviceRemovedReason());
    auto actual = ReadGpu(rb0.Get(), size_t(kSurfaceBytes)), repeat = ReadGpu(rb1.Get(), size_t(kSurfaceBytes));
    auto timing = ReadGpu(times.Get(), 32); auto* ticks = reinterpret_cast<const uint64_t*>(timing.data());
    UINT64 frequency = 0; HR_CHECK(queue->GetTimestampFrequency(&frequency));
    const auto* values = reinterpret_cast<const uint16_t*>(actual.data()); uint64_t nonFinite = 0;
    for (size_t i = 0; i < kSurfaceBytes / 2; ++i) nonFinite += !HalfFinite(values[i]);
    bool pass = actual == repeat && nonFinite == 0;
    Write(argv[4], actual.data(), actual.size());
    std::ostringstream json;
    json << "{\n  \"experiment\": \"output_head_d3d12_inference\",\n  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"gpu_ms\": [" << double(ticks[1]-ticks[0])*1000/frequency << ", " << double(ticks[3]-ticks[2])*1000/frequency << "],\n"
         << "  \"non_finite_values\": " << nonFinite << ",\n  \"repeat_exact\": " << (actual == repeat ? "true" : "false") << ",\n"
         << "  \"oracle_inputs\": false,\n  \"texture_output\": true,\n  \"game_runtime_ready\": false\n}\n";
    auto manifest = json.str(); Write(argv[5], manifest.data(), manifest.size()); std::printf("%s", manifest.c_str()); return pass ? 0 : 1;
} catch (const std::exception& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; } }
