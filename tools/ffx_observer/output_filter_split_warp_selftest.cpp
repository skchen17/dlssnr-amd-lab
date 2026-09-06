#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../output_head_surface_d3d12/output_head_recording_selftest.cpp"
#include "output_boundary_filter.h"

namespace {
constexpr D3D12_RESOURCE_STATES kReadable =
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

void Wait(ID3D12Device* device, ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Fence> fence;
    HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(), 1));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("event creation");
    HR_CHECK(fence->SetEventOnCompletion(1, event));
    const auto status = WaitForSingleObject(event, 30000);
    CloseHandle(event);
    if (status != WAIT_OBJECT_0) throw std::runtime_error("WARP queue timeout");
    HR_CHECK(device->GetDeviceRemovedReason());
}

std::vector<uint8_t> UnpackRows(ID3D12Resource* resource,
                                const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint,
                                UINT rows, UINT64 rowBytes, UINT64 totalBytes) {
    std::vector<uint8_t> result(size_t(rows * rowBytes));
    void* mapped = nullptr;
    D3D12_RANGE range{0, SIZE_T(totalBytes)};
    HR_CHECK(resource->Map(0, &range, &mapped));
    for (UINT y = 0; y < rows; ++y)
        std::memcpy(result.data() + size_t(y * rowBytes),
                    static_cast<uint8_t*>(mapped) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
                    size_t(rowBytes));
    D3D12_RANGE none{0, 0};
    resource->Unmap(0, &none);
    return result;
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 3) return 2;
    const auto shader = Read(fs::absolute(argv[1]));
    const auto root = fs::absolute(argv[2]);
    if (fs::exists(root)) return 2;
    fs::create_directories(root);
    try {
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
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        ComPtr<ID3D12CommandQueue> queue;
        HR_CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));

        constexpr UINT width = 320, height = 180;
        D3D12_RESOURCE_DESC textureDesc{};
        textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.Width = width;
        textureDesc.Height = height;
        textureDesc.DepthOrArraySize = 1;
        textureDesc.MipLevels = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES local{};
        local.Type = D3D12_HEAP_TYPE_DEFAULT;
        ComPtr<ID3D12Resource> texture;
        HR_CHECK(device->CreateCommittedResource(&local, D3D12_HEAP_FLAG_NONE, &textureDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture)));

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT rows = 0;
        UINT64 rowBytes = 0, totalBytes = 0;
        device->GetCopyableFootprints(&textureDesc, 0, 1, 0, &footprint, &rows, &rowBytes, &totalBytes);
        std::vector<uint8_t> expected(size_t(width) * height * 8);
        std::vector<uint8_t> packed(size_t(totalBytes), 0);
        for (UINT y = 0; y < height; ++y) for (UINT x = 0; x < width; ++x) {
            const uint16_t pixel[4] = {
                uint16_t(0x3000 + (x % 256)), uint16_t(0x3200 + (y % 128)),
                uint16_t(0x3400 + ((x + y) % 128)), 0x3c00};
            std::memcpy(expected.data() + (size_t(y) * width + x) * 8, pixel, 8);
        }
        for (UINT y = 0; y < height; ++y)
            std::memcpy(packed.data() + footprint.Offset + y * footprint.Footprint.RowPitch,
                        expected.data() + size_t(y) * rowBytes, size_t(rowBytes));
        auto upload = Upload(device.Get(), packed);

        ComPtr<ID3D12CommandAllocator> producerAllocator;
        ComPtr<ID3D12GraphicsCommandList> producer;
        HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&producerAllocator)));
        HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, producerAllocator.Get(), nullptr,
                                           IID_PPV_ARGS(&producer)));
        D3D12_TEXTURE_COPY_LOCATION uploadLocation{};
        uploadLocation.pResource = upload.Get();
        uploadLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        uploadLocation.PlacedFootprint = footprint;
        auto textureLocation = Recorder::TextureLocation(texture.Get());
        producer->CopyTextureRegion(&textureLocation, 0, 0, 0, &uploadLocation, nullptr);
        Recorder::Transition(producer.Get(), texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_RESOURCE_BARRIER boundary{};
        boundary.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        boundary.Transition = {texture.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, kReadable};
        producer->ResourceBarrier(1, &boundary);
        HR_CHECK(producer->Close());

        ffx_boundary::OutputFilter filter(device.Get(), texture.Get(), shader);
        filter.Record(boundary);

        auto downstream = Readback(device.Get(), totalBytes);
        ComPtr<ID3D12CommandAllocator> suffixAllocator, suffix2Allocator;
        ComPtr<ID3D12GraphicsCommandList> suffix, suffix2;
        HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&suffixAllocator)));
        HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, suffixAllocator.Get(), nullptr,
                                           IID_PPV_ARGS(&suffix)));
        Recorder::Transition(suffix.Get(), texture.Get(), kReadable, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION downstreamLocation{};
        downstreamLocation.pResource = downstream.Get();
        downstreamLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        downstreamLocation.PlacedFootprint = footprint;
        suffix->CopyTextureRegion(&downstreamLocation, 0, 0, 0, &textureLocation, nullptr);
        HR_CHECK(suffix->Close());
        HR_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&suffix2Allocator)));
        HR_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, suffix2Allocator.Get(), nullptr,
                                           IID_PPV_ARGS(&suffix2)));
        HR_CHECK(suffix2->Close());

        ID3D12CommandList* prefix[] = {producer.Get()};
        ID3D12CommandList* inserted[] = {filter.CommandList()};
        ID3D12CommandList* suffixes[] = {suffix.Get(), suffix2.Get()};
        queue->ExecuteCommandLists(1, prefix);
        queue->ExecuteCommandLists(1, inserted);
        queue->ExecuteCommandLists(2, suffixes);
        filter.Submitted(queue.Get());
        Wait(device.Get(), queue.Get());

        std::vector<uint8_t> before, after;
        if (!filter.Poll(before, after)) throw std::runtime_error("filter fence incomplete");
        auto observed = UnpackRows(downstream.Get(), footprint, rows, rowBytes, totalBytes);
        uint64_t changedPixels = 0;
        bool alphaExact = true;
        for (size_t i = 0; i < after.size(); i += 8) {
            changedPixels += std::memcmp(before.data() + i, after.data() + i, 8) != 0;
            alphaExact &= std::memcmp(before.data() + i + 6, after.data() + i + 6, 2) == 0;
        }
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
        const bool pass = before == expected && observed == after && changedPixels && alphaExact && !errors && !warnings;
        std::ostringstream summary;
        summary << "{\"status\":\"" << (pass ? "WARP_FILTER_SPLIT_PASS" : "FAIL")
                << "\",\"adapter\":\"WARP\",\"hardware_gpu_used\":false,\"queue_wait_gate_used\":false"
                << ",\"prefix_lists\":1,\"inserted_lists\":1,\"suffix_lists\":2"
                << ",\"before_exact\":" << (before == expected ? "true" : "false")
                << ",\"downstream_exact\":" << (observed == after ? "true" : "false")
                << ",\"changed_pixels\":" << changedPixels << ",\"alpha_exact\":" << (alphaExact ? "true" : "false")
                << ",\"d3d12_errors\":" << errors << ",\"d3d12_warnings\":" << warnings
                << ",\"trained_weights\":false,\"dlss_nr_verified\":false}\n";
        const auto text = summary.str();
        Write(root / "summary.json", text.data(), text.size());
        std::printf("%s", text.c_str());
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "WARP split test failed: %s\n", error.what());
        return 1;
    }
}
