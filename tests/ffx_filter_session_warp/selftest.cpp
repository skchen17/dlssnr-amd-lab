#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../../tools/output_head_surface_d3d12/output_head_recording_selftest.cpp"
#include "../../third_party/fidelityfx-api-1.1.3/ffx_upscale.h"
#include "../../third_party/fidelityfx-api-1.1.3/dx12/ffx_api_dx12.h"
#include "../../tools/ffx_observer/capture_session.h"
#include <fstream>

namespace {
using CreateFn = ffxReturnCode_t (*)(ffxContext*, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*);
using DestroyFn = ffxReturnCode_t (*)(ffxContext*, const ffxAllocationCallbacks*);
using DispatchFn = ffxReturnCode_t (*)(ffxContext*, const ffxDispatchDescHeader*);
using InitFn = DWORD (WINAPI*)(void*);
using CommandFn = DWORD (WINAPI*)(void*);
constexpr D3D12_RESOURCE_STATES kReadable = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

void Wait(ID3D12Device* device, ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Fence> fence;
    HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(), 1));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("event");
    HR_CHECK(fence->SetEventOnCompletion(1, event));
    const auto status = WaitForSingleObject(event, 30000);
    CloseHandle(event);
    if (status != WAIT_OBJECT_0) throw std::runtime_error("queue timeout");
    HR_CHECK(device->GetDeviceRemovedReason());
}

std::vector<uint8_t> ReadRows(ID3D12Resource* resource,
                              const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint,
                              UINT rows, UINT64 rowBytes, UINT64 totalBytes) {
    std::vector<uint8_t> result(size_t(rows * rowBytes));
    void* data = nullptr;
    D3D12_RANGE range{0, SIZE_T(totalBytes)};
    HR_CHECK(resource->Map(0, &range, &data));
    for (UINT y = 0; y < rows; ++y)
        std::memcpy(result.data() + size_t(y * rowBytes),
                    static_cast<uint8_t*>(data) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
                    size_t(rowBytes));
    D3D12_RANGE none{0, 0};
    resource->Unmap(0, &none);
    return result;
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2 && argc != 3) return 2;
    const bool network = argc == 3 && !wcscmp(argv[2], L"--network");
    if (argc == 3 && !network) return 2;
    const auto root = fs::absolute(argv[1]);
    if (fs::exists(root)) return 2;
    fs::create_directories(root);
    try {
        wchar_t executable[32768]{};
        if (!GetModuleFileNameW(nullptr, executable, 32768)) throw std::runtime_error("module path");
        const auto bin = fs::path(executable).parent_path();
        HMODULE fixture = LoadLibraryExW((bin / L"ffx_filter_session_fixture.dll").c_str(), nullptr,
                                         LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        HMODULE session = LoadLibraryExW((bin / L"ffx_capture_session.dll").c_str(), nullptr,
                                         LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!fixture || !session) throw std::runtime_error("module load");
        auto originalCreate = reinterpret_cast<CreateFn>(GetProcAddress(fixture, "ffxCreateContext"));
        auto originalDestroy = reinterpret_cast<DestroyFn>(GetProcAddress(fixture, "ffxDestroyContext"));
        auto originalDispatch = reinterpret_cast<DispatchFn>(GetProcAddress(fixture, "ffxDispatch"));
        auto initialize = reinterpret_cast<InitFn>(GetProcAddress(session, "FfxSession_TestInitialize"));
        auto create = reinterpret_cast<CreateFn>(GetProcAddress(session, "FfxSession_TestCreate"));
        auto destroy = reinterpret_cast<DestroyFn>(GetProcAddress(session, "FfxSession_TestDestroy"));
        auto dispatch = reinterpret_cast<DispatchFn>(GetProcAddress(session, "FfxSession_TestDispatch"));
        auto command = reinterpret_cast<CommandFn>(GetProcAddress(session, "FfxSession_Command"));
        if (!originalCreate || !originalDestroy || !originalDispatch || !initialize || !create || !destroy || !dispatch || !command)
            throw std::runtime_error("exports");

        ComPtr<ID3D12Debug> debug;
        HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;
        HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter> warp;
        HR_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));
        ComPtr<ID3D12Device> device;
        HR_CHECK(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        ComPtr<ID3D12InfoQueue> info;
        HR_CHECK(device.As(&info));

        FfxSessionTestConfigV1 config{};
        config.common.size = sizeof(config.common);
        config.common.version = 2;
        config.common.sample_limit = 4;
        config.common.capture_enabled = 0;
        wcscpy_s(config.common.log_path, (root / L"session.jsonl").c_str());
        config.original_create = reinterpret_cast<uintptr_t>(originalCreate);
        config.original_destroy = reinterpret_cast<uintptr_t>(originalDestroy);
        config.original_dispatch = reinterpret_cast<uintptr_t>(originalDispatch);
        if (initialize(&config) != 1) throw std::runtime_error("session initialize");

        ffxOverrideVersion version{};
        version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        version.versionId = 1;
        ffxCreateBackendDX12Desc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.header.pNext = &version.header;
        backend.device = device.Get();
        ffxCreateContextDescUpscale createDesc{};
        createDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        createDesc.header.pNext = &backend.header;
        createDesc.maxRenderSize = {160, 90};
        createDesc.maxUpscaleSize = {320, 180};
        ffxContext context = nullptr;
        if (create(&context, &createDesc.header, nullptr) != 0 || !context) throw std::runtime_error("context create");
        FfxSessionCommandV1 arm{sizeof(arm), network ? 13u : 11u};
        if (command(&arm) != 1) throw std::runtime_error("filter/network arm");

        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        ComPtr<ID3D12CommandQueue> queue;
        HR_CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
        D3D12_RESOURCE_DESC textureDesc{};
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.Width = 320;
        textureDesc.Height = 180;
        textureDesc.DepthOrArraySize = 1;
        textureDesc.MipLevels = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES local{};
        local.Type = D3D12_HEAP_TYPE_DEFAULT;
        ComPtr<ID3D12Resource> output;
        HR_CHECK(device->CreateCommittedResource(&local, D3D12_HEAP_FLAG_NONE, &textureDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&output)));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows = 0;
        UINT64 rowBytes = 0, totalBytes = 0;
        device->GetCopyableFootprints(&textureDesc, 0, 1, 0, &footprint, &rows, &rowBytes, &totalBytes);
        std::vector<uint8_t> expected(size_t(320) * 180 * 8), packed(size_t(totalBytes), 0);
        for (UINT y = 0; y < 180; ++y) for (UINT x = 0; x < 320; ++x) {
            uint16_t pixel[4] = {uint16_t(0x3000 + x % 128), uint16_t(0x3200 + y % 64),
                                 uint16_t(0x3400 + (x + y) % 64), 0x3c00};
            std::memcpy(expected.data() + (size_t(y) * 320 + x) * 8, pixel, 8);
        }
        for (UINT y = 0; y < 180; ++y)
            std::memcpy(packed.data() + footprint.Offset + y * footprint.Footprint.RowPitch,
                        expected.data() + size_t(y) * rowBytes, size_t(rowBytes));
        auto upload = Upload(device.Get(), packed);

        ComPtr<ID3D12CommandAllocator> allocator, suffixAllocator, suffix2Allocator;
        ComPtr<ID3D12GraphicsCommandList> list, suffix, suffix2;
        HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        HR_CHECK(list->Close());
        HR_CHECK(list->Reset(allocator.Get(), nullptr));
        D3D12_TEXTURE_COPY_LOCATION uploadLocation{};
        uploadLocation.pResource = upload.Get();
        uploadLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        uploadLocation.PlacedFootprint = footprint;
        auto outputLocation = Recorder::TextureLocation(output.Get());
        list->CopyTextureRegion(&outputLocation, 0, 0, 0, &uploadLocation, nullptr);
        Recorder::Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ffxDispatchDescUpscale dispatchDesc{};
        dispatchDesc.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dispatchDesc.commandList = list.Get();
        dispatchDesc.output = ffxApiGetResourceDX12(output.Get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatchDesc.renderSize = {160, 90};
        dispatchDesc.upscaleSize = {320, 180};
        if (dispatch(&context, &dispatchDesc.header) != 0) throw std::runtime_error("dispatch");
        D3D12_RESOURCE_BARRIER boundary{};
        boundary.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        boundary.Transition = {output.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kReadable};
        list->ResourceBarrier(1, &boundary);
        HR_CHECK(list->Close());

        auto downstream = Readback(device.Get(), totalBytes);
        HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&suffixAllocator)));
        HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, suffixAllocator.Get(), nullptr, IID_PPV_ARGS(&suffix)));
        Recorder::Transition(suffix.Get(), output.Get(), kReadable, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION downstreamLocation{};
        downstreamLocation.pResource = downstream.Get();
        downstreamLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        downstreamLocation.PlacedFootprint = footprint;
        suffix->CopyTextureRegion(&downstreamLocation, 0, 0, 0, &outputLocation, nullptr);
        HR_CHECK(suffix->Close());
        HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&suffix2Allocator)));
        HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, suffix2Allocator.Get(), nullptr, IID_PPV_ARGS(&suffix2)));
        HR_CHECK(suffix2->Close());
        ID3D12CommandList* batch[] = {list.Get(), suffix.Get(), suffix2.Get()};
        queue->ExecuteCommandLists(3, batch);
        Wait(device.Get(), queue.Get());
        FfxSessionCommandV1 poll{sizeof(poll), network ? 14u : 12u};
        if (command(&poll) != 1) throw std::runtime_error("filter/network poll");
        const auto observed = ReadRows(downstream.Get(), footprint, rows, rowBytes, totalBytes);
        const auto outputRoot = root / (network ? L"output_network" : L"output_filter");
        const auto before = Read(outputRoot / L"boundary_before.raw");
        const auto after = Read(outputRoot / L"boundary_output.raw");
        if (destroy(&context, nullptr) != 0 || context) throw std::runtime_error("destroy");

        std::ifstream log(root / L"session.jsonl");
        const std::string events((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
        UINT errors = 0, warnings = 0;
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<uint8_t> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            HR_CHECK(info->GetMessage(i, message, &size));
            errors += message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR;
            warnings += message->Severity == D3D12_MESSAGE_SEVERITY_WARNING;
        }
        const bool pass = before == expected && observed == after && before != after &&
            events.find("\"internal_hook_bypass\":true") != std::string::npos &&
            events.find("\"submission_batch_count\":3") != std::string::npos &&
            events.find("\"submission_batch_index\":0") != std::string::npos &&
            events.find("\"submission_suffix_lists\":2") != std::string::npos &&
            events.find("\"submission_split_performed\":true") != std::string::npos &&
            events.find("\"event\":\"failure\"") == std::string::npos && !errors && !warnings;
        std::ostringstream summary;
        summary << "{\"status\":\"" << (pass ? (network ? "WARP_SESSION_NETWORK_PASS" : "WARP_SESSION_FILTER_PASS") : "FAIL")
                << "\",\"hardware_gpu_used\":false,\"queue_wait_gate_used\":false"
                << ",\"internal_hook_bypass\":true,\"submission_batch_count\":3"
                << ",\"submission_batch_index\":0,\"submission_suffix_lists\":2"
                << ",\"downstream_exact\":" << (observed == after ? "true" : "false")
                << ",\"d3d12_errors\":" << errors << ",\"d3d12_warnings\":" << warnings
                << ",\"external_weights\":" << (network ? "true" : "false")
                << ",\"trained_weights\":false,\"dlss_nr_verified\":false}\n";
        const auto text = summary.str();
        Write(root / L"summary.json", text.data(), text.size());
        std::printf("%s", text.c_str());
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "WARP session filter test failed: %s\n", error.what());
        return 1;
    }
}
