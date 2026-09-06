#define main output_head_tail_surface_unused_cli
#include "output_head_tail_surface_d3d12.cpp"
#undef main
#include "output_head_d3d12_runtime.h"
#include <d3d12sdklayers.h>
#include <sstream>

namespace {
using Recorder = output_head_dx12::Executor;
ComPtr<ID3D12Resource> Texture(ID3D12Device* device) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 640; desc.Height = 360; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> resource;
    HR_CHECK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource)));
    return resource;
}
std::vector<uint8_t> ReadGpu(ID3D12Resource* resource, size_t bytes) {
    std::vector<uint8_t> data(bytes); void* p = nullptr;
    D3D12_RANGE range{0, bytes}; HR_CHECK(resource->Map(0, &range, &p));
    std::memcpy(data.data(), p, bytes); D3D12_RANGE written{0,0}; resource->Unmap(0, &written);
    return data;
}
// Independent CPU nearest-even half encoder for finite clamped output values.
uint16_t UnitHalf(float v) {
    uint32_t low = 0, high = 0x3c00;
    while (low < high) {
        uint32_t middle = (low + high) / 2;
        if (HalfToFloat(uint16_t(middle)) < v) low = middle + 1; else high = middle;
    }
    if (!low) return 0;
    float upper = HalfToFloat(uint16_t(low)), lower = HalfToFloat(uint16_t(low - 1));
    float du = upper - v, dl = v - lower;
    return uint16_t(dl < du || (dl == du && (low & 1)) ? low - 1 : low);
}
std::vector<uint8_t> Compose(const std::vector<uint8_t>& residual, const std::vector<uint8_t>& base) {
    std::vector<uint8_t> result(static_cast<size_t>(kSurfaceBytes));
    const auto* r = reinterpret_cast<const uint16_t*>(residual.data());
    const auto* b = reinterpret_cast<const uint16_t*>(base.data());
    auto* o = reinterpret_cast<uint16_t*>(result.data());
    for (unsigned cta = 0; cta < 81 * 49; ++cta) for (unsigned token = 0; token < 64; ++token) {
        unsigned group = token / 16, inner = token % 16;
        int x = int(cta % 81) * 8 - 4 + int((group & 1) * 4 + inner % 4);
        int y = int(cta / 81) * 8 - 4 + int((group >> 1) * 4 + inner / 4);
        if (x < 0 || y < 0 || x >= 640 || y >= 360) continue;
        size_t pixel = (size_t(y) * 640 + x) * 4, source = (cta * 64 + token) * 4;
        for (unsigned c = 0; c < 3; ++c)
            o[pixel + c] = UnitHalf(std::clamp(HalfToFloat(b[pixel + c]) + 0.25f * HalfToFloat(r[source + c]), 0.0f, 1.0f));
        o[pixel + 3] = 0x3c00;
    }
    return result;
}
}

#ifndef OUTPUT_HEAD_RECORDING_HELPERS_ONLY
int main(int argc, char** argv) { try {
    if (argc != 5) throw std::invalid_argument("usage: output_head_recording_selftest <activation.raw> <model.raw> <standalone_output.raw> <result_directory>");
    const auto main = ReadSlice(argv[1], 13873152, output_head_dx12::MainBytes);
    const auto skip = ReadSlice(argv[1], 110592, output_head_dx12::SkipBytes);
    const auto head = ReadSlice(argv[2], kHeadOffset, kHeadBytes);
    const auto expected = Read(argv[3]);
    if (expected.size() != kSurfaceBytes) throw std::invalid_argument("invalid reference surface");
    ComPtr<ID3D12Debug> debug;
    const bool debugEnabled = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    if (debugEnabled) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory7> factory; HR_CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; DXGI_ADAPTER_DESC1 ad{}; bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
        HR_CHECK(adapter->GetDesc1(&ad));
        if (!(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && ad.VendorId == 0x1002) { found = true; break; }
    }
    if (!found) throw std::runtime_error("AMD adapter not found");
    ComPtr<ID3D12Device> device; HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    ComPtr<ID3D12InfoQueue> info; device.As(&info);
    Recorder::ShaderSet shaders; fs::path build = fs::absolute(argv[0]).parent_path();
    for (size_t i = 0; i < shaders.size(); ++i) shaders[i] = Read(build / (std::string(output_head_dx12::ShaderNames[i]) + ".dxil"));
    Recorder recorder(device.Get(), shaders);
    // Main and skip share one arena, with real nonzero captured offsets.
    auto arena = DefaultBuffer(device.Get(), 27807744), model = DefaultBuffer(device.Get(), 147719680);
    auto uMain = Upload(device.Get(), main), uSkip = Upload(device.Get(), skip), uHead = Upload(device.Get(), head);
    auto uZeroMain = Upload(device.Get(), std::vector<uint8_t>(main.size(), 0));
    auto baseTexture = Texture(device.Get()), outputTexture = Texture(device.Get());
    constexpr unsigned frameCount = 4;
    std::array<std::vector<uint8_t>, frameCount> bases;
    std::array<ComPtr<ID3D12Resource>, frameCount> uploads, readbacks, residualReadbacks;
    for (unsigned frame = 0; frame < frameCount; ++frame) {
        bases[frame].resize(size_t(kSurfaceBytes), 0);
        if (frame == 1) {
            auto* b = reinterpret_cast<uint16_t*>(bases[frame].data());
            for (size_t i = 0; i < kSurfaceBytes / 2; ++i) b[i] = uint16_t((i % 4 == 3) ? 0x3c00 : ((i / 4) % 2 ? 0x3400 : 0x3800));
        }
        uploads[frame] = Upload(device.Get(), bases[frame]);
        readbacks[frame] = Readback(device.Get(), kSurfaceBytes);
        residualReadbacks[frame] = Readback(device.Get(), kResidualBytes);
    }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue; HR_CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12CommandAllocator> allocator; HR_CHECK(device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list; HR_CHECK(device->CreateCommandList(0, qd.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
    output_head_dx12::Inputs inputs{
        {arena.Get(), 13873152, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
        {arena.Get(), 110592, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
        {model.Get(), kHeadOffset, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
        {baseTexture.Get(), D3D12_RESOURCE_STATE_COMMON},
        {outputTexture.Get(), D3D12_RESOURCE_STATE_COMMON}
    };
    unsigned rejected = 0;
    auto reject = [&](const output_head_dx12::Inputs& bad) {
        try { recorder.Record(list.Get(), bad); } catch (const std::invalid_argument&) { ++rejected; }
    };
    auto bad = inputs; bad.main.offset = 27807744; reject(bad);
    bad = inputs; bad.skip.state = D3D12_RESOURCE_STATE_COMMON; reject(bad);
    bad = inputs; bad.output.resource = baseTexture.Get(); reject(bad);
    bad = inputs; bad.head.resource = nullptr; reject(bad);
    D3D12_QUERY_HEAP_DESC queryDesc{}; queryDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; queryDesc.Count = frameCount * 2;
    ComPtr<ID3D12QueryHeap> queries; HR_CHECK(device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&queries)));
    auto times = Readback(device.Get(), frameCount * 2 * sizeof(uint64_t));
    // These producer copies are not submitted until after all calls below.
    Recorder::Transition(list.Get(), arena.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(arena.Get(), 13873152, uMain.Get(), 0, main.size());
    list->CopyBufferRegion(arena.Get(), 110592, uSkip.Get(), 0, skip.size());
    Recorder::Transition(list.Get(), arena.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Recorder::Transition(list.Get(), model.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(model.Get(), kHeadOffset, uHead.Get(), 0, head.size());
    Recorder::Transition(list.Get(), model.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for (unsigned frame = 0; frame < frameCount; ++frame) {
        if (frame >= 2) {
            Recorder::Transition(list.Get(), arena.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyBufferRegion(arena.Get(), 13873152, frame == 2 ? uZeroMain.Get() : uMain.Get(), 0, main.size());
            Recorder::Transition(list.Get(), arena.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        Recorder::Transition(list.Get(), baseTexture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        auto src = Recorder::LinearLocation(uploads[frame].Get()), dst = Recorder::TextureLocation(baseTexture.Get());
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Recorder::Transition(list.Get(), baseTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame * 2);
        recorder.Record(list.Get(), inputs, residualReadbacks[frame].Get());
        list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, frame * 2 + 1);
        Recorder::Transition(list.Get(), outputTexture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        src = Recorder::TextureLocation(outputTexture.Get()); dst = Recorder::LinearLocation(readbacks[frame].Get());
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Recorder::Transition(list.Get(), outputTexture.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    }
    list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, frameCount * 2, times.Get(), 0);
    HR_CHECK(list->Close()); ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence; HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(), 1)); HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent failed");
    HR_CHECK(fence->SetEventOnCompletion(1, event)); const DWORD wait = WaitForSingleObject(event, 60000); CloseHandle(event);
    if (wait != WAIT_OBJECT_0) throw std::runtime_error("GPU completion timed out");
    HR_CHECK(device->GetDeviceRemovedReason());
    std::array<std::vector<uint8_t>, frameCount> firstSubmission;
    for (unsigned frame = 0; frame < frameCount; ++frame)
        firstSubmission[frame] = ReadGpu(readbacks[frame].Get(), size_t(kSurfaceBytes));
    // Replay the closed caller-owned list after fence completion. This also
    // exercises scratch states and resource reuse across queue submissions.
    queue->ExecuteCommandLists(1, lists);
    HR_CHECK(queue->Signal(fence.Get(), 2)); event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent failed");
    HR_CHECK(fence->SetEventOnCompletion(2, event)); const DWORD repeatWait = WaitForSingleObject(event, 60000); CloseHandle(event);
    if (repeatWait != WAIT_OBJECT_0) throw std::runtime_error("repeat GPU completion timed out");
    HR_CHECK(device->GetDeviceRemovedReason());
    uint64_t debugErrors = 0;
    if (info) for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T bytes = 0; info->GetMessage(i, nullptr, &bytes); std::vector<uint8_t> storage(bytes);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(storage.data()); HR_CHECK(info->GetMessage(i, m, &bytes));
        if (m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) { ++debugErrors; std::fprintf(stderr, "%s\n", m->pDescription); }
    }
    UINT64 frequency = 0; HR_CHECK(queue->GetTimestampFrequency(&frequency));
    const auto timeBytes = ReadGpu(times.Get(), frameCount * 2 * sizeof(uint64_t)); const auto* ticks = reinterpret_cast<const uint64_t*>(timeBytes.data());
    std::array<std::vector<uint8_t>, frameCount> outputs, residuals;
    uint64_t compositionMismatches = 0, nonFinite = 0;
    bool submissionRepeat = true;
    std::ostringstream json; json << "{\n  \"experiment\": \"output_head_recording_selftest\",\n  \"frames\": 8,\n  \"submissions\": 2,\n  \"gpu_ms_second_submission_including_diagnostic_copy\": [";
    for (unsigned frame = 0; frame < frameCount; ++frame) {
        outputs[frame] = ReadGpu(readbacks[frame].Get(), size_t(kSurfaceBytes));
        submissionRepeat &= outputs[frame] == firstSubmission[frame];
        residuals[frame] = ReadGpu(residualReadbacks[frame].Get(), size_t(kResidualBytes));
        const auto composed = Compose(residuals[frame], bases[frame]);
        const auto* rh = reinterpret_cast<const uint16_t*>(residuals[frame].data());
        for (size_t i = 0; i < kResidualBytes / 2; ++i) nonFinite += !HalfFinite(rh[i]);
        for (size_t i = 0; i < composed.size(); ++i) compositionMismatches += composed[i] != outputs[frame][i];
        const auto* h = reinterpret_cast<const uint16_t*>(outputs[frame].data());
        for (size_t i = 0; i < kSurfaceBytes / 2; ++i) nonFinite += !HalfFinite(h[i]);
        json << (frame ? ", " : "") << double(ticks[frame*2+1] - ticks[frame*2]) * 1000.0 / frequency;
        Write(fs::path(argv[4]) / ("frame" + std::to_string(frame) + "_rgba16f.raw"), outputs[frame].data(), outputs[frame].size());
    }
    bool exact = outputs[0] == expected, repeat = outputs[0] == outputs[3], changed = outputs[0] != outputs[1];
    bool activationChanged = residuals[0] != residuals[2] && outputs[0] != outputs[2];
    bool residualRepeat = residuals[0] == residuals[1] && residuals[0] == residuals[3];
    bool pass = exact && repeat && submissionRepeat && changed && activationChanged && residualRepeat && !compositionMismatches && !nonFinite && !debugErrors && rejected == 4;
    json << "],\n  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"standalone_byte_exact\": " << (exact ? "true" : "false") << ",\n"
         << "  \"a_b_c_a_repeat_exact\": " << (repeat ? "true" : "false") << ",\n"
         << "  \"activation_update_changes_output\": " << (activationChanged ? "true" : "false") << ",\n"
         << "  \"residual_repeat_exact\": " << (residualRepeat ? "true" : "false") << ",\n"
         << "  \"submission_repeat_exact\": " << (submissionRepeat ? "true" : "false") << ",\n"
         << "  \"nonzero_base_changes_output\": " << (changed ? "true" : "false") << ",\n"
         << "  \"cpu_composition_byte_mismatches\": " << compositionMismatches << ",\n"
         << "  \"non_finite_values\": " << nonFinite << ",\n  \"rejected_invalid_bindings\": " << rejected << ",\n"
         << "  \"debug_layer_enabled\": " << (debugEnabled ? "true" : "false") << ",\n  \"debug_errors\": " << debugErrors << ",\n"
         << "  \"caller_owned_resources\": true,\n  \"single_command_list\": true,\n"
         << "  \"captured_activation_input\": true,\n  \"game_runtime_ready\": false\n}\n";
    const std::string manifest = json.str(); Write(fs::path(argv[4]) / "manifest.json", manifest.data(), manifest.size());
    std::printf("%s", manifest.c_str()); return pass ? 0 : 1;
} catch (const std::exception& error) { std::fprintf(stderr, "ERROR: %s\n", error.what()); return 1; } }
#endif
