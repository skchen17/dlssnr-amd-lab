#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "resolution_plan.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;
using full_graph_dx12::CompatibilityTileHeight;
using full_graph_dx12::CompatibilityTileWidth;
using full_graph_dx12::CompatibilityVerticalHalo;
using full_graph_dx12::CompatibilityWorkHeight;
using full_graph_dx12::CompatibilityResidentTiles;
using full_graph_dx12::ResolutionPlan;
using full_graph_dx12::TileDescriptor;

#define HR_CHECK(call) do { HRESULT hr_ = (call); if (FAILED(hr_)) { \
    char text_[192]; std::snprintf(text_, sizeof(text_), #call " failed: 0x%08lX", \
    static_cast<unsigned long>(hr_)); throw std::runtime_error(text_); } } while (0)

namespace {

std::vector<uint8_t> Read(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const size_t bytes = size_t(stream.tellg()); stream.seekg(0);
    std::vector<uint8_t> result(bytes);
    if (bytes && !stream.read(reinterpret_cast<char*>(result.data()), std::streamsize(bytes)))
        throw std::runtime_error("read failed " + path.string());
    return result;
}

ComPtr<ID3D12Resource> Buffer(ID3D12Device* device, uint64_t bytes,
                              D3D12_HEAP_TYPE heapType,
                              D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = heapType;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes; desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> result;
    HR_CHECK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        state, nullptr, IID_PPV_ARGS(&result)));
    return result;
}

ComPtr<ID3D12Resource> Texture(ID3D12Device* device, uint32_t width,
                               uint32_t height, uint16_t slices,
                               D3D12_RESOURCE_FLAGS flags,
                               D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.DepthOrArraySize = slices;
    desc.MipLevels = 1; desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1; desc.Flags = flags;
    ComPtr<ID3D12Resource> result;
    HR_CHECK(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        state, nullptr, IID_PPV_ARGS(&result)));
    return result;
}

void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after) return;
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}

void UavBarrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource; list->ResourceBarrier(1, &barrier);
}

ComPtr<ID3D12PipelineState> Pipeline(ID3D12Device* device, ID3D12RootSignature* root,
                                     const std::vector<uint8_t>& shader) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{}; desc.pRootSignature = root;
    desc.CS = {shader.data(), shader.size()}; ComPtr<ID3D12PipelineState> result;
    HR_CHECK(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&result)));
    return result;
}

D3D12_CPU_DESCRIPTOR_HANDLE Cpu(ID3D12DescriptorHeap* heap, uint32_t index,
                                 uint32_t stride) {
    auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += SIZE_T(index) * stride; return handle;
}
D3D12_GPU_DESCRIPTOR_HANDLE Gpu(ID3D12DescriptorHeap* heap, uint32_t index,
                                 uint32_t stride) {
    auto handle = heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += UINT64(index) * stride; return handle;
}

void TextureSrv(ID3D12Device* device, ID3D12Resource* resource,
                D3D12_CPU_DESCRIPTOR_HANDLE handle, bool array) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{}; desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    desc.ViewDimension = array ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY : D3D12_SRV_DIMENSION_TEXTURE2D;
    if (array) { desc.Texture2DArray.MipLevels = 1; desc.Texture2DArray.ArraySize = resource->GetDesc().DepthOrArraySize; }
    else desc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(resource, &desc, handle);
}

void TextureUav(ID3D12Device* device, ID3D12Resource* resource,
                D3D12_CPU_DESCRIPTOR_HANDLE handle, bool array) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC desc{}; desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.ViewDimension = array ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY : D3D12_UAV_DIMENSION_TEXTURE2D;
    if (array) desc.Texture2DArray.ArraySize = resource->GetDesc().DepthOrArraySize;
    device->CreateUnorderedAccessView(resource, nullptr, &desc, handle);
}

std::vector<uint16_t> Pattern(uint32_t width, uint32_t height) {
    constexpr uint16_t values[] = {0x0000, 0x3000, 0x3400, 0x3800, 0x3a00, 0x3c00};
    std::vector<uint16_t> result(uint64_t(width) * height * 4);
    for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x)
        for (uint32_t c = 0; c < 4; ++c)
            result[(uint64_t(y) * width + x) * 4 + c] = values[(x * 3 + y * 5 + c) % 6];
    return result;
}

struct CaseResult { uint32_t width, height, tileCount, batchCount; uint64_t mismatches; };

CaseResult RunCase(ID3D12Device* device, ID3D12CommandQueue* queue,
                   ID3D12RootSignature* root, ID3D12PipelineState* pack,
                   ID3D12PipelineState* identity, ID3D12PipelineState* unpack,
                   uint32_t width, uint32_t height) {
    const auto plan = ResolutionPlan::Build(width, height);
    const auto inputData = Pattern(width, height);
    auto input = Texture(device, width, height, 1, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_COPY_DEST);
    const uint16_t residentTiles = uint16_t(std::min<size_t>(CompatibilityResidentTiles, plan.tiles.size()));
    auto work = Texture(device, CompatibilityTileWidth, CompatibilityWorkHeight,
                        residentTiles, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto tileOutput = Texture(device, CompatibilityTileWidth, CompatibilityTileHeight,
                              residentTiles, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto output = Texture(device, width, height, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    D3D12_RESOURCE_DESC inputDesc = input->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputFoot{}; UINT rows = 0; UINT64 rowBytes = 0, inputBytes = 0;
    device->GetCopyableFootprints(&inputDesc, 0, 1, 0, &inputFoot, &rows, &rowBytes, &inputBytes);
    auto upload = Buffer(device, inputBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr; HR_CHECK(upload->Map(0, nullptr, &mapped));
    for (uint32_t y = 0; y < height; ++y)
        std::memcpy(static_cast<uint8_t*>(mapped) + inputFoot.Offset + uint64_t(y) * inputFoot.Footprint.RowPitch,
                    inputData.data() + uint64_t(y) * width * 4, uint64_t(width) * 8);
    upload->Unmap(0, nullptr);

    auto descriptors = Buffer(device, plan.tiles.size() * sizeof(TileDescriptor),
                              D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    HR_CHECK(descriptors->Map(0, nullptr, &mapped));
    std::memcpy(mapped, plan.tiles.data(), plan.tiles.size() * sizeof(TileDescriptor));
    descriptors->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{}; heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 9; heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap; HR_CHECK(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)));
    const uint32_t stride = device->GetDescriptorHandleIncrementSize(heapDesc.Type);
    for (uint32_t base : {0u, 3u, 6u}) {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{}; desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Buffer.NumElements = UINT(plan.tiles.size()); desc.Buffer.StructureByteStride = sizeof(TileDescriptor);
        device->CreateShaderResourceView(descriptors.Get(), &desc, Cpu(heap.Get(), base, stride));
    }
    TextureSrv(device, input.Get(), Cpu(heap.Get(), 1, stride), false);
    TextureUav(device, work.Get(), Cpu(heap.Get(), 2, stride), true);
    TextureSrv(device, work.Get(), Cpu(heap.Get(), 4, stride), true);
    TextureUav(device, tileOutput.Get(), Cpu(heap.Get(), 5, stride), true);
    TextureSrv(device, tileOutput.Get(), Cpu(heap.Get(), 7, stride), true);
    TextureUav(device, output.Get(), Cpu(heap.Get(), 8, stride), false);

    D3D12_RESOURCE_DESC outputDesc = output->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT outputFoot{}; UINT64 outputBytes = 0;
    device->GetCopyableFootprints(&outputDesc, 0, 1, 0, &outputFoot, &rows, &rowBytes, &outputBytes);
    auto readback = Buffer(device, outputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_COMMAND_QUEUE_DESC queueDesc{}; queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandAllocator> allocator; HR_CHECK(device->CreateCommandAllocator(queueDesc.Type, IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> list; HR_CHECK(device->CreateCommandList(0, queueDesc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
    D3D12_TEXTURE_COPY_LOCATION source{}; source.pResource = upload.Get(); source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; source.PlacedFootprint = inputFoot;
    D3D12_TEXTURE_COPY_LOCATION destination{}; destination.pResource = input.Get(); destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    Transition(list.Get(), input.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* heaps[] = {heap.Get()}; list->SetDescriptorHeaps(1, heaps); list->SetComputeRootSignature(root);
    auto dispatch = [&](ID3D12PipelineState* pipeline, uint32_t descriptorBase, uint32_t groupsY,
                        const uint32_t* constants) {
        list->SetPipelineState(pipeline); list->SetComputeRoot32BitConstants(0, 5, constants, 0);
        list->SetComputeRootDescriptorTable(1, Gpu(heap.Get(), descriptorBase, stride));
        list->SetComputeRootDescriptorTable(2, Gpu(heap.Get(), descriptorBase + 2, stride));
        list->Dispatch(CompatibilityTileWidth / 8, groupsY, constants[2]);
    };
    for (uint32_t tileBase = 0; tileBase < plan.tiles.size(); tileBase += CompatibilityResidentTiles) {
        if (tileBase) {
            Transition(list.Get(), work.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Transition(list.Get(), tileOutput.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        const uint32_t batchTiles = std::min<uint32_t>(CompatibilityResidentTiles, uint32_t(plan.tiles.size()) - tileBase);
        const uint32_t constants[] = {width, height, batchTiles, CompatibilityVerticalHalo, tileBase};
        dispatch(pack, 0, CompatibilityWorkHeight / 8, constants); UavBarrier(list.Get(), work.Get());
        Transition(list.Get(), work.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dispatch(identity, 3, CompatibilityTileHeight / 8, constants); UavBarrier(list.Get(), tileOutput.Get());
        Transition(list.Get(), tileOutput.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dispatch(unpack, 6, CompatibilityTileHeight / 8, constants); UavBarrier(list.Get(), output.Get());
    }
    Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    source = {}; source.pResource = output.Get(); source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination = {}; destination.pResource = readback.Get(); destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; destination.PlacedFootprint = outputFoot;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr); HR_CHECK(list->Close());
    ID3D12CommandList* lists[] = {list.Get()}; queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence; HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HR_CHECK(queue->Signal(fence.Get(), 1)); HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent failed"); HR_CHECK(fence->SetEventOnCompletion(1, event));
    const DWORD wait = WaitForSingleObject(event, 60000); CloseHandle(event);
    if (wait != WAIT_OBJECT_0) throw std::runtime_error("resolution GPU test timeout");
    HR_CHECK(device->GetDeviceRemovedReason());
    D3D12_RANGE range{0, SIZE_T(outputBytes)}; HR_CHECK(readback->Map(0, &range, &mapped));
    uint64_t mismatches = 0;
    for (uint32_t y = 0; y < height; ++y) {
        const auto* actual = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(mapped) + outputFoot.Offset + uint64_t(y) * outputFoot.Footprint.RowPitch);
        const auto* expected = inputData.data() + uint64_t(y) * width * 4;
        for (uint64_t i = 0; i < uint64_t(width) * 4; ++i) mismatches += actual[i] != expected[i];
    }
    D3D12_RANGE written{0, 0}; readback->Unmap(0, &written);
    return {width, height, uint32_t(plan.tiles.size()), plan.BatchCount(), mismatches};
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 5) throw std::invalid_argument("usage: resolution_gpu_selftest <pack.dxil> <identity.dxil> <unpack.dxil> <output.json>");
    ComPtr<ID3D12Debug> debug; HR_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))); debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory; HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; HR_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
    ComPtr<ID3D12Device> device; HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    ComPtr<ID3D12InfoQueue> info; HR_CHECK(device.As(&info));
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3]{}; params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 5}; params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &ranges[0]}; params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &ranges[1]};
    D3D12_ROOT_SIGNATURE_DESC rootDesc{}; rootDesc.NumParameters = 3; rootDesc.pParameters = params;
    ComPtr<ID3DBlob> blob, error; HR_CHECK(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
    ComPtr<ID3D12RootSignature> root; HR_CHECK(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
    auto pack = Pipeline(device.Get(), root.Get(), Read(argv[1]));
    auto identity = Pipeline(device.Get(), root.Get(), Read(argv[2]));
    auto unpack = Pipeline(device.Get(), root.Get(), Read(argv[3]));
    D3D12_COMMAND_QUEUE_DESC queueDesc{}; ComPtr<ID3D12CommandQueue> queue;
    HR_CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
    const std::array<std::pair<uint32_t, uint32_t>, 5> dimensions{{
        {320, 180}, {641, 361}, {1920, 1080}, {2342, 1317}, {3840, 2160}}};
    std::vector<CaseResult> cases; uint64_t mismatches = 0;
    for (auto [width, height] : dimensions) { cases.push_back(RunCase(device.Get(), queue.Get(), root.Get(), pack.Get(), identity.Get(), unpack.Get(), width, height)); mismatches += cases.back().mismatches; }
    uint64_t errors = 0, warnings = 0; std::ostringstream messages;
    for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) { SIZE_T bytes = 0; info->GetMessage(i, nullptr, &bytes); std::vector<uint8_t> storage(bytes); auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data()); HR_CHECK(info->GetMessage(i, message, &bytes)); errors += message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR; warnings += message->Severity == D3D12_MESSAGE_SEVERITY_WARNING; if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) messages << message->pDescription << '\n'; }
    const bool pass = !mismatches && !errors && !warnings; std::ostringstream json;
    json << "{\n  \"status\": \"" << (pass ? "FULL_GRAPH_ARBITRARY_RESOLUTION_GPU_PASS" : "FAIL") << "\",\n  \"adapter\": \"WARP\",\n  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) { const auto& c = cases[i]; json << "    {\"resolution\": [" << c.width << ", " << c.height << "], \"tiles\": " << c.tileCount << ", \"batches\": " << c.batchCount << ", \"half_mismatches\": " << c.mismatches << "}" << (i + 1 == cases.size() ? "\n" : ",\n"); }
    json << "  ],\n  \"resident_tile_limit\": " << CompatibilityResidentTiles << ",\n  \"total_half_mismatches\": " << mismatches << ",\n  \"d3d12_errors\": " << errors << ",\n  \"d3d12_warnings\": " << warnings << ",\n  \"bounded_compatibility_memory\": true,\n  \"complete_network\": false\n}\n";
    const fs::path output = fs::absolute(argv[4]); fs::create_directories(output.parent_path());
    std::ofstream stream(output, std::ios::binary); stream << json.str();
    std::ofstream messageStream(output.parent_path() / "d3d12_messages.log", std::ios::binary); messageStream << messages.str();
    std::cout << json.str(); return pass ? 0 : 1;
} catch (const std::exception& error) { std::fprintf(stderr, "resolution GPU test failed: %s\n", error.what()); return 1; }
