#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>

using InitTrace_t = int(WINAPI*)(DWORD);
using RegisterDevice_t = int(WINAPI*)(ID3D12Device*);
using BindCopyResources_t = int(WINAPI*)(ID3D12Resource*, ID3D12Resource*);
using BeginPostTextureCapture_t = int(WINAPI*)(
    ID3D12GraphicsCommandList*, ID3D12Resource*, uint64_t);
using BeginFrame1PostOutputCapture_t = int(WINAPI*)(
    ID3D12GraphicsCommandList*, ID3D12Resource*, uint64_t);
using BeginFrame1PostSurfaceInitialCapture_t = int(WINAPI*)(
    ID3D12GraphicsCommandList*, ID3D12Resource*, uint64_t);
using DumpCopyResources_t = int(WINAPI*)(ID3D12CommandQueue*, const char*);
using BeginNeuralCapture_t = int(WINAPI*)(ID3D12GraphicsCommandList*, const void*, uint32_t);
using EndNeuralCapture_t = int(WINAPI*)(ID3D12GraphicsCommandList*);
using BeginFullGraphCapture_t = uint64_t(WINAPI*)(
    ID3D12GraphicsCommandList*, uint32_t, uint64_t, const void*, uint32_t);
using EndFullGraphCapture_t = int(WINAPI*)(ID3D12GraphicsCommandList*, uint64_t);

template <typename T> void Release(T*& value) {
    if (value) value->Release();
    value = nullptr;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: module_trace_d3d12_selftest <module_trace.dll> <output-dir>\n");
        return 2;
    }
    HMODULE trace = LoadLibraryW(std::filesystem::absolute(argv[1]).wstring().c_str());
    if (!trace) return 3;
    auto init = reinterpret_cast<InitTrace_t>(
        GetProcAddress(trace, "ModuleTrace_InitializeAndWait"));
    auto registerDevice = reinterpret_cast<RegisterDevice_t>(
        GetProcAddress(trace, "ModuleTrace_RegisterD3D12Device"));
    auto bindCopyResources = reinterpret_cast<BindCopyResources_t>(
        GetProcAddress(trace, "ModuleTrace_TestBindCopyResources"));
    auto dumpCopyResources = reinterpret_cast<DumpCopyResources_t>(
        GetProcAddress(trace, "ModuleTrace_DumpCopyResources"));
    auto beginPostTextureCapture = reinterpret_cast<BeginPostTextureCapture_t>(
        GetProcAddress(trace, "ModuleTrace_TestBeginPostTextureCapture"));
    auto beginFrame1PostOutputCapture =
        reinterpret_cast<BeginFrame1PostOutputCapture_t>(GetProcAddress(
            trace, "ModuleTrace_TestBeginFrame1PostOutputCapture"));
    auto beginFrame1PostSurfaceInitialCapture =
        reinterpret_cast<BeginFrame1PostSurfaceInitialCapture_t>(GetProcAddress(
            trace, "ModuleTrace_TestBeginFrame1PostSurfaceInitialCapture"));
    auto beginNeuralCapture = reinterpret_cast<BeginNeuralCapture_t>(
        GetProcAddress(trace, "ModuleTrace_TestBeginNeuralCapture"));
    auto endNeuralCapture = reinterpret_cast<EndNeuralCapture_t>(
        GetProcAddress(trace, "ModuleTrace_TestEndNeuralCapture"));
    auto beginFullGraphCapture = reinterpret_cast<BeginFullGraphCapture_t>(
        GetProcAddress(trace, "ModuleTrace_TestBeginFullGraphCapture"));
    auto endFullGraphCapture = reinterpret_cast<EndFullGraphCapture_t>(
        GetProcAddress(trace, "ModuleTrace_TestEndFullGraphCapture"));
    if (!init || !registerDevice || !bindCopyResources || !dumpCopyResources ||
        !beginPostTextureCapture || !beginFrame1PostOutputCapture ||
        !beginFrame1PostSurfaceInitialCapture ||
        !beginNeuralCapture || !endNeuralCapture || !beginFullGraphCapture ||
        !endFullGraphCapture || !init(10000)) return 4;

    ID3D12Device* device = nullptr;
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&device));
    if (FAILED(hr) || !device) return 5;
    if (!registerDevice(device)) return 6;
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)))) return 13;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bufferDesc{};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    // Cover the real model resource offsets used by slots 3-5, including the
    // last 64 KiB view at 0x7681600.
    // Covers the highest graph-observed downstream weight view (slot 10 at
    // 0x8CB0200) plus its 64 KiB immutable capture window.
    bufferDesc.Width = 160ull << 20;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* buffer = nullptr;
    hr = device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer));
    if (FAILED(hr)) return 7;

    bufferDesc.Width = 12ull << 20;
    ID3D12Resource* neuralBuffer = nullptr;
    hr = device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&neuralBuffer));
    if (FAILED(hr)) return 15;
    D3D12_HEAP_PROPERTIES uploadHeap{};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    bufferDesc.Width = 256;
    ID3D12Resource* upload = nullptr;
    hr = device->CreateCommittedResource(
        &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
    if (FAILED(hr)) return 16;
    void* uploadData = nullptr;
    if (FAILED(upload->Map(0, nullptr, &uploadData)) || !uploadData) return 17;
    for (unsigned i = 0; i < 256; ++i)
        static_cast<unsigned char*>(uploadData)[i] = static_cast<unsigned char>(0xA5 ^ i);
    upload->Unmap(0, nullptr);

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = 64;
    textureDesc.Height = 32;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* texture = nullptr;
    hr = device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &textureDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&texture));
    if (FAILED(hr)) return 8;

    D3D12_DESCRIPTOR_HEAP_DESC viewHeapDesc{};
    viewHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    viewHeapDesc.NumDescriptors = 2;
    ID3D12DescriptorHeap* cpuViews = nullptr;
    if (FAILED(device->CreateDescriptorHeap(&viewHeapDesc, IID_PPV_ARGS(&cpuViews))))
        return 9;
    viewHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ID3D12DescriptorHeap* gpuViews = nullptr;
    if (FAILED(device->CreateDescriptorHeap(&viewHeapDesc, IID_PPV_ARGS(&gpuViews))))
        return 10;
    UINT viewIncrement = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto srv = cpuViews->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE uav{srv.ptr + viewIncrement};
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = textureDesc.Format;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(texture, &srvDesc, srv);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
    uavDesc.Format = textureDesc.Format;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(texture, nullptr, &uavDesc, uav);
    device->CopyDescriptorsSimple(
        2, gpuViews->GetCPUDescriptorHandleForHeapStart(), srv,
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc{};
    samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerHeapDesc.NumDescriptors = 1;
    ID3D12DescriptorHeap* cpuSamplers = nullptr;
    if (FAILED(device->CreateDescriptorHeap(&samplerHeapDesc,
                                             IID_PPV_ARGS(&cpuSamplers))))
        return 11;
    samplerHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ID3D12DescriptorHeap* gpuSamplers = nullptr;
    if (FAILED(device->CreateDescriptorHeap(&samplerHeapDesc,
                                             IID_PPV_ARGS(&gpuSamplers))))
        return 12;
    D3D12_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW =
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    auto samplerCpu = cpuSamplers->GetCPUDescriptorHandleForHeapStart();
    device->CreateSampler(&sampler, samplerCpu);
    device->CopyDescriptorsSimple(
        1, gpuSamplers->GetCPUDescriptorHandleForHeapStart(), samplerCpu,
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    ID3D12CommandAllocator* neuralAllocator = nullptr;
    ID3D12GraphicsCommandList* neuralList = nullptr;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                               IID_PPV_ARGS(&neuralAllocator))))
        return 18;
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          neuralAllocator, nullptr,
                                          IID_PPV_ARGS(&neuralList))))
        return 19;
    unsigned char parameters[264]{};
    uint64_t scratchPointer = neuralBuffer->GetGPUVirtualAddress();
    uint64_t weightsPointer = buffer->GetGPUVirtualAddress();
    uint64_t outputPointer = scratchPointer + (8ull << 20);
    uint32_t paddedHeight = 384, paddedWidth = 640;
    std::memcpy(parameters + 216, &scratchPointer, sizeof(scratchPointer));
    std::memcpy(parameters + 224, &weightsPointer, sizeof(weightsPointer));
    std::memcpy(parameters + 240, &paddedHeight, sizeof(paddedHeight));
    std::memcpy(parameters + 244, &paddedWidth, sizeof(paddedWidth));
    std::memcpy(parameters + 248, &outputPointer, sizeof(outputPointer));
    if (!beginNeuralCapture(neuralList, parameters, sizeof(parameters))) return 20;
    unsigned char fullGraphParameters[24]{};
    std::memcpy(fullGraphParameters + 0, &scratchPointer, sizeof(scratchPointer));
    std::memcpy(fullGraphParameters + 8, &outputPointer, sizeof(outputPointer));
    std::memcpy(fullGraphParameters + 16, &weightsPointer, sizeof(weightsPointer));
    uint64_t fullGraphSpan = beginFullGraphCapture(
        neuralList, 42, 0x12345678, fullGraphParameters,
        static_cast<uint32_t>(sizeof(fullGraphParameters)));
    if (!(fullGraphSpan & 0xFFFFFFFFull)) return 22;
    if (!beginPostTextureCapture(neuralList, texture, 0x123456789ABCDEF0ull))
        return 23;
    if (!beginFrame1PostOutputCapture(neuralList, texture, 0x0FEDCBA987654321ull))
        return 24;
    if (!beginFrame1PostSurfaceInitialCapture(
            neuralList, texture, 0x1111222233334444ull))
        return 25;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = neuralBuffer;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    neuralList->ResourceBarrier(1, &barrier);
    neuralList->CopyBufferRegion(neuralBuffer, 8ull << 20, upload, 0, 256);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    neuralList->ResourceBarrier(1, &barrier);
    if (!endFullGraphCapture(neuralList, fullGraphSpan) ||
        !endNeuralCapture(neuralList) || FAILED(neuralList->Close())) return 21;
    ID3D12CommandList* neuralLists[] = {neuralList};
    queue->ExecuteCommandLists(1, neuralLists);

    // Binding the same COMMON-state texture at both ends gives a deterministic
    // bitwise identity oracle while exercising the complete GPU readback path.
    if (!bindCopyResources(texture, texture) ||
        !dumpCopyResources(queue, std::filesystem::absolute(argv[2]).string().c_str()))
        return 14;

    std::printf("[PASS] D3D12 hooks + copy snapshot + N0 + full-graph windows\n");
    Release(gpuSamplers);
    Release(cpuSamplers);
    Release(gpuViews);
    Release(cpuViews);
    Release(texture);
    Release(neuralList);
    Release(neuralAllocator);
    Release(upload);
    Release(neuralBuffer);
    Release(buffer);
    Release(queue);
    Release(device);
    return 0;
}
