#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "../../third_party/nvapi/nvapi.h"
#include "../../tools/nvapi_amd/nvapi_amd_diag.h"

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

using Query_t = void*(__cdecl*)(uint32_t);
using CreateModule_t = NvAPI_Status(__cdecl*)(ID3D12Device*, const void*, NvU32,
                                              NVDX_ObjectHandle*);
using CreateFunction_t = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle,
                                                const char*, NVDX_ObjectHandle*);
using Launch_t = NvAPI_Status(__cdecl*)(ID3D12GraphicsCommandList*,
                                        const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
using Destroy_t = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle);
using Merged_t = NvAPI_Status(__cdecl*)(
    NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS*);
using Independent_t = NvAPI_Status(__cdecl*)(
    NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS*);

static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* resource,
                                         D3D12_RESOURCE_STATES before,
                                         D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

int main(int argc, char** argv) {
    fs::path dll = argc > 1 ? argv[1] : "nvapi64_amd.dll";
    fs::path jsonPath = argc > 2 ? argv[2] : fs::path{};
    fs::path inputPath = argc > 3 ? argv[3] : fs::path{};
    fs::path expectedPath = argc > 4 ? argv[4] : fs::path{};
    const bool externalOracle = !inputPath.empty() && !expectedPath.empty();
    const UINT width = argc > 5 ? static_cast<UINT>(std::stoul(argv[5])) : 64;
    const UINT height = argc > 6 ? static_cast<UINT>(std::stoul(argv[6])) : 32;
    const DXGI_FORMAT format = argc > 7
        ? static_cast<DXGI_FORMAT>(std::stoul(argv[7]))
        : DXGI_FORMAT_R32G32B32A32_FLOAT;
    const size_t bytesPerPixel = format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8
                               : format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 16
                               : 0;
    if (!width || !height || !bytesPerPixel || (inputPath.empty() != expectedPath.empty()))
        return 1;
    HMODULE library = LoadLibraryW(dll.wstring().c_str());
    if (!library) return 2;
    auto query = reinterpret_cast<Query_t>(GetProcAddress(library, "nvapi_QueryInterface"));
    auto getDiag = reinterpret_cast<NvapiAmdGetDiagnostics_t>(
        GetProcAddress(library, "NvapiAmd_GetDiagnostics"));
    auto resetDiag = reinterpret_cast<NvapiAmdResetDiagnostics_t>(
        GetProcAddress(library, "NvapiAmd_ResetDiagnostics"));
    if (!query || !getDiag || !resetDiag || !resetDiag()) return 3;

    auto createModule = reinterpret_cast<CreateModule_t>(query(0xAD1A677D));
    auto createFunction = reinterpret_cast<CreateFunction_t>(query(0xE2436E22));
    auto launch = reinterpret_cast<Launch_t>(query(0x24973538));
    auto destroyFunction = reinterpret_cast<Destroy_t>(query(0xDF295EA6));
    auto destroyModule = reinterpret_cast<Destroy_t>(query(0x41C65285));
    auto getMerged = reinterpret_cast<Merged_t>(query(0x329FE6E0));
    auto getIndependent = reinterpret_cast<Independent_t>(query(0x0DDAC234));
    if (!createModule || !createFunction || !launch || !destroyFunction ||
        !destroyModule || !getMerged || !getIndependent) return 4;

    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 adapterDesc{};
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 5;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> candidate;
        if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
        candidate->GetDesc1(&adapterDesc);
        if (adapterDesc.VendorId == 0x1002 &&
            !(adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            adapter = candidate;
            break;
        }
    }
    ComPtr<ID3D12Device> device;
    if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                             IID_PPV_ARGS(&device)))) return 5;
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)))) return 5;
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = format;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> source, destination;
    if (FAILED(device->CreateCommittedResource(
            &defaultHeap, D3D12_HEAP_FLAG_NONE, &textureDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&source))) ||
        FAILED(device->CreateCommittedResource(
            &defaultHeap, D3D12_HEAP_FLAG_NONE, &textureDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&destination)))) return 6;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 rowBytes = 0, totalBytes = 0;
    device->GetCopyableFootprints(&textureDesc, 0, 1, 0, &footprint, &rows,
                                  &rowBytes, &totalBytes);
    D3D12_RESOURCE_DESC bufferDesc{};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = totalBytes;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES uploadHeap{}, readbackHeap{};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> upload, readback;
    if (FAILED(device->CreateCommittedResource(
            &uploadHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))) ||
        FAILED(device->CreateCommittedResource(
            &readbackHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))))
        return 6;

    const size_t compactBytes = size_t(width) * height * bytesPerPixel;
    std::vector<uint8_t> inputBytes(compactBytes), expectedBytes(compactBytes);
    std::vector<float> expectedFloats;
    if (externalOracle) {
        std::ifstream input(inputPath, std::ios::binary);
        std::ifstream expectedFile(expectedPath, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(inputBytes.data()), compactBytes) ||
            input.peek() != std::ifstream::traits_type::eof() ||
            !expectedFile.read(reinterpret_cast<char*>(expectedBytes.data()), compactBytes) ||
            expectedFile.peek() != std::ifstream::traits_type::eof())
            return 10;
    } else {
        expectedFloats.resize(size_t(width) * height * 4);
        for (UINT y = 0; y < height; ++y) {
            for (UINT x = 0; x < width; ++x) {
                size_t index = (size_t(y) * width + x) * 4;
                expectedFloats[index + 0] = float(x) + 0.25f;
                expectedFloats[index + 1] = float(y) + 0.5f;
                expectedFloats[index + 2] = float(x ^ y) + 0.75f;
                expectedFloats[index + 3] = 1.0f;
            }
        }
        std::memcpy(inputBytes.data(), expectedFloats.data(), compactBytes);
        expectedBytes = inputBytes;
    }
    uint8_t* uploadData = nullptr;
    upload->Map(0, nullptr, reinterpret_cast<void**>(&uploadData));
    for (UINT y = 0; y < height; ++y) {
        std::memcpy(uploadData + footprint.Offset +
                        size_t(y) * footprint.Footprint.RowPitch,
                    inputBytes.data() + size_t(y) * width * bytesPerPixel,
                    size_t(width) * bytesPerPixel);
    }
    upload->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC viewHeapDesc{};
    viewHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    viewHeapDesc.NumDescriptors = 2;
    ComPtr<ID3D12DescriptorHeap> views;
    device->CreateDescriptorHeap(&viewHeapDesc, IID_PPV_ARGS(&views));
    D3D12_DESCRIPTOR_HEAP_DESC samplerHeapDesc{};
    samplerHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    samplerHeapDesc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> samplers;
    device->CreateDescriptorHeap(&samplerHeapDesc, IID_PPV_ARGS(&samplers));
    UINT viewStride = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE srv = views->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE uav{srv.ptr + viewStride};
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = format;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(source.Get(), &srvDesc, srv);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
    uavDesc.Format = format;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(destination.Get(), nullptr, &uavDesc, uav);
    D3D12_CPU_DESCRIPTOR_HANDLE sampler =
        samplers->GetCPUDescriptorHandleForHeapStart();
    D3D12_SAMPLER_DESC samplerDesc{};
    samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samplerDesc.AddressU = samplerDesc.AddressV = samplerDesc.AddressW =
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
    device->CreateSampler(&samplerDesc, sampler);

    NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS merged{};
    merged.structSizeIn = merged.structSizeOut = sizeof(merged);
    merged.pDevice = device.Get();
    merged.texDesc = srv;
    merged.smpDesc = sampler;
    NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS independent{};
    independent.structSizeIn = independent.structSizeOut = sizeof(independent);
    independent.pDevice = device.Get();
    independent.type =
        NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SURFACE;
    independent.desc = uav;
    if (getMerged(&merged) != NVAPI_OK || !merged.textureHandle ||
        getIndependent(&independent) != NVAPI_OK || !independent.handle)
        return 7;

    uint64_t blob[2] = {0xBA55ED5000100001ull, 14};
    NVDX_ObjectHandle module = nullptr, function = nullptr;
    if (createModule(device.Get(), blob, sizeof(blob), &module) != NVAPI_OK ||
        createFunction(device.Get(), module, "cg2r_copy_kernel", &function) !=
            NVAPI_OK) return 8;

    alignas(8) uint8_t params[72]{};
    std::memcpy(params, &merged.textureHandle, 8);
    std::memcpy(params + 8, &independent.handle, 8);
    float transforms[12] = {0.0f, 0.0f, float(width), float(height),
                            1.0f / width, 1.0f / height, 0.0f, 0.0f,
                            float(width), float(height),
                            1.0f / width, 1.0f / height};
    std::memcpy(params + 16, transforms, sizeof(transforms));
    std::memcpy(params + 64, &width, 4);
    std::memcpy(params + 68, &height, 4);

    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                   IID_PPV_ARGS(&allocator));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                              nullptr, IID_PPV_ARGS(&list));
    D3D12_TEXTURE_COPY_LOCATION sourceCopy{};
    sourceCopy.pResource = upload.Get();
    sourceCopy.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    sourceCopy.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION sourceTexture{};
    sourceTexture.pResource = source.Get();
    sourceTexture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&sourceTexture, 0, 0, 0, &sourceCopy, nullptr);
    auto sourceBarrier = Transition(source.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &sourceBarrier);
    NVAPI_CU_KERNEL_LAUNCH_PARAMS launchParams{};
    launchParams.hFunction = function;
    launchParams.gridDim = {(width + 15) / 16, (height + 15) / 16, 1};
    launchParams.blockDim = {16, 16, 1};
    launchParams.pParams = params;
    launchParams.paramSize = sizeof(params);
    if (launch(list.Get(), &launchParams, 1) != NVAPI_OK) return 9;
    D3D12_RESOURCE_BARRIER uavBarrier{};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = destination.Get();
    list->ResourceBarrier(1, &uavBarrier);
    auto destinationBarrier = Transition(destination.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &destinationBarrier);
    D3D12_TEXTURE_COPY_LOCATION destinationTexture{};
    destinationTexture.pResource = destination.Get();
    destinationTexture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION readbackCopy{};
    readbackCopy.pResource = readback.Get();
    readbackCopy.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    readbackCopy.PlacedFootprint = footprint;
    list->CopyTextureRegion(&readbackCopy, 0, 0, 0, &destinationTexture, nullptr);
    list->Close();
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.Get(), 1);
    while (fence->GetCompletedValue() < 1) Sleep(0);

    const uint8_t* readbackData = nullptr;
    readback->Map(0, nullptr, reinterpret_cast<void**>(
        const_cast<uint8_t**>(&readbackData)));
    uint64_t byteMismatches = 0;
    uint64_t mismatches = 0;
    float maxAbsError = 0.0f;
    for (UINT y = 0; y < height; ++y) {
        const uint8_t* row = readbackData + footprint.Offset +
                             size_t(y) * footprint.Footprint.RowPitch;
        const uint8_t* expectedRow = expectedBytes.data() +
                                     size_t(y) * width * bytesPerPixel;
        for (size_t x = 0; x < size_t(width) * bytesPerPixel; ++x) {
            if (row[x] != expectedRow[x]) ++byteMismatches;
        }
        if (format == DXGI_FORMAT_R32G32B32A32_FLOAT) {
            const float* floatRow = reinterpret_cast<const float*>(row);
            const float* expectedRowFloat = reinterpret_cast<const float*>(expectedRow);
            for (UINT x = 0; x < width * 4; ++x) {
                float error = std::fabs(floatRow[x] - expectedRowFloat[x]);
                if (error != 0.0f) ++mismatches;
                if (error > maxAbsError) maxAbsError = error;
            }
        } else {
            for (size_t x = 0; x < size_t(width) * 4; ++x) {
                if (std::memcmp(row + x * 2, expectedRow + x * 2, 2) != 0)
                    ++mismatches;
            }
        }
    }
    readback->Unmap(0, nullptr);
    destroyFunction(device.Get(), function);
    destroyModule(device.Get(), module);

    NvapiAmdDiagnosticsV1 diag{};
    diag.struct_size = sizeof(diag);
    bool diagOk = getDiag(&diag) != 0;
    bool pass = byteMismatches == 0 && mismatches == 0 && diagOk &&
                diag.merged_texture_sampler_calls == 1 &&
                diag.independent_descriptor_calls == 1 &&
                diag.d3d12_copy_dispatches == 1 &&
                diag.pipeline_failures == 0 &&
                diag.neural_math_executed == 0;
    std::printf("[%s] copy=%ux%u format=%u byte_mismatches=%llu "
                "component_mismatches=%llu max_abs_error=%g "
                "handles=0x%llx/0x%llx dispatches=%u\n",
                pass ? "PASS" : "FAIL", width, height, unsigned(format),
                static_cast<unsigned long long>(byteMismatches),
                static_cast<unsigned long long>(mismatches), maxAbsError,
                static_cast<unsigned long long>(merged.textureHandle),
                static_cast<unsigned long long>(independent.handle),
                diag.d3d12_copy_dispatches);
    if (!jsonPath.empty()) {
        std::ofstream json(jsonPath, std::ios::binary);
        json << "{\n"
             << "  \"schema\": 1,\n"
             << "  \"experiment\": \"nvapi_amd_copy_selftest\",\n"
             << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
             << "  \"classification\": \"BOUNDARY_OPERATION_ONLY\",\n"
             << "  \"oracle_mode\": \""
             << (externalOracle ? "RTX_CAPTURE" : "SYNTHETIC") << "\",\n"
             << "  \"counts_as_s6\": false,\n"
             << "  \"neural_math_executed\": false,\n"
             << "  \"dxgi_vendor\": \"0x" << std::hex
             << adapterDesc.VendorId << "\",\n" << std::dec
             << "  \"width\": " << width << ",\n"
             << "  \"height\": " << height << ",\n"
             << "  \"dxgi_format\": " << unsigned(format) << ",\n"
             << "  \"byte_mismatches\": " << byteMismatches << ",\n"
             << "  \"component_mismatches\": " << mismatches << ",\n"
             << "  \"max_abs_error\": " << maxAbsError << ",\n"
             << "  \"merged_texture_sampler_calls\": "
             << diag.merged_texture_sampler_calls << ",\n"
             << "  \"independent_descriptor_calls\": "
             << diag.independent_descriptor_calls << ",\n"
             << "  \"d3d12_copy_dispatches\": "
             << diag.d3d12_copy_dispatches << ",\n"
             << "  \"pipeline_failures\": " << diag.pipeline_failures << "\n"
             << "}\n";
    }
    return pass ? 0 : 1;
}
