#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../../third_party/nvapi/nvapi.h"
#include "../../tools/nvapi_amd/nvapi_amd_diag.h"

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

using QueryInterface_t = void*(__cdecl*)(uint32_t);
using CreateModule_t = NvAPI_Status(__cdecl*)(ID3D12Device*, const void*, NvU32,
                                              NVDX_ObjectHandle*);
using CreateFunction_t = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle,
                                                const char*, NVDX_ObjectHandle*);
using Launch_t = NvAPI_Status(__cdecl*)(ID3D12GraphicsCommandList*,
                                        const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
using Destroy_t = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle);

int main(int argc, char** argv) {
    fs::path dll = "nvapi64_amd.dll";
    fs::path jsonPath;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dll" && i + 1 < argc) dll = argv[++i];
        else if (std::string(argv[i]) == "--json" && i + 1 < argc) jsonPath = argv[++i];
        else if (i == 1) dll = argv[i];
    }
    HMODULE library = LoadLibraryW(dll.wstring().c_str());
    if (!library) { printf("FAIL load dll error=%lu\n", GetLastError()); return 2; }
    auto query = reinterpret_cast<QueryInterface_t>(GetProcAddress(library, "nvapi_QueryInterface"));
    auto getDiag = reinterpret_cast<NvapiAmdGetDiagnostics_t>(GetProcAddress(library, "NvapiAmd_GetDiagnostics"));
    auto resetDiag = reinterpret_cast<NvapiAmdResetDiagnostics_t>(GetProcAddress(library, "NvapiAmd_ResetDiagnostics"));
    auto registerBuffer = reinterpret_cast<NvapiAmdRegisterExternalBuffer_t>(GetProcAddress(library, "NvapiAmd_RegisterExternalBuffer"));
    auto unregisterBuffer = reinterpret_cast<NvapiAmdUnregisterExternalBuffer_t>(GetProcAddress(library, "NvapiAmd_UnregisterExternalBuffer"));
    if (!query || !getDiag || !resetDiag || !registerBuffer || !unregisterBuffer) {
        printf("FAIL exports\n"); return 3;
    }

    auto createModule = reinterpret_cast<CreateModule_t>(query(0xAD1A677D));
    auto createFunction = reinterpret_cast<CreateFunction_t>(query(0xE2436E22));
    auto launch = reinterpret_cast<Launch_t>(query(0x24973538));
    auto destroyFunction = reinterpret_cast<Destroy_t>(query(0xDF295EA6));
    auto destroyModule = reinterpret_cast<Destroy_t>(query(0x41C65285));
    bool queryPass = createModule && createFunction && launch && destroyFunction &&
                     destroyModule && !query(0xDEADBEEF);
    if (!queryPass || !resetDiag()) { printf("FAIL query/reset\n"); return 4; }

    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc{};
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> candidate;
        if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
        candidate->GetDesc1(&desc);
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && desc.VendorId == 0x1002) {
            adapter = candidate; break;
        }
    }
    ComPtr<ID3D12Device> device;
    if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                             IID_PPV_ARGS(&device)))) {
        printf("FAIL no AMD D3D12 device\n"); return 5;
    }
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    uint64_t fenceValue = 0;
    auto submitAndWait = [&](ID3D12GraphicsCommandList* list) {
        list->Close();
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), ++fenceValue);
        while (fence->GetCompletedValue() < fenceValue) Sleep(0);
    };

    constexpr uint32_t clearCount = 27648; // R-21 slot 0 parameter + 108*256 threads
    constexpr uint64_t clearBytes = uint64_t(clearCount) * sizeof(uint32_t);
    D3D12_RESOURCE_DESC bufferDesc{};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = clearBytes;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES defaultHeap{};
    defaultHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_HEAP_PROPERTIES uploadHeap{};
    uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_HEAP_PROPERTIES readbackHeap{};
    readbackHeap.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> sharedBuffer, uploadBuffer, readbackBuffer;
    if (FAILED(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_SHARED,
                                               &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&sharedBuffer))) ||
        FAILED(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE,
                                               &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                               nullptr, IID_PPV_ARGS(&uploadBuffer))) ||
        FAILED(device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE,
                                               &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&readbackBuffer)))) {
        printf("FAIL resource creation\n"); return 6;
    }
    void* mapped = nullptr;
    uploadBuffer->Map(0, nullptr, &mapped);
    std::memset(mapped, 0, clearBytes);
    uploadBuffer->Unmap(0, nullptr);
    ComPtr<ID3D12CommandAllocator> prepAllocator;
    ComPtr<ID3D12GraphicsCommandList> prepList;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&prepAllocator));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, prepAllocator.Get(), nullptr,
                              IID_PPV_ARGS(&prepList));
    prepList->CopyResource(sharedBuffer.Get(), uploadBuffer.Get());
    D3D12_RESOURCE_BARRIER toCommon{};
    toCommon.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toCommon.Transition.pResource = sharedBuffer.Get();
    toCommon.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    toCommon.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    toCommon.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    prepList->ResourceBarrier(1, &toCommon);
    submitAndWait(prepList.Get());
    HANDLE sharedHandle = nullptr;
    if (FAILED(device->CreateSharedHandle(sharedBuffer.Get(), nullptr, GENERIC_ALL,
                                          nullptr, &sharedHandle)) ||
        !registerBuffer(sharedBuffer.Get(), sharedHandle, clearBytes)) {
        printf("FAIL resource registration\n"); return 6;
    }

    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commandList;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                              IID_PPV_ARGS(&commandList));

    std::vector<NVDX_ObjectHandle> modules(9);
    for (uint32_t i = 0; i < modules.size(); ++i) {
        uint64_t blob[2] = {0xBA55ED5000100001ull, i};
        if (createModule(device.Get(), blob, sizeof(blob), &modules[i]) != NVAPI_OK) {
            printf("FAIL create module %u\n", i); return 6;
        }
    }
    std::vector<NVDX_ObjectHandle> functions(96);
    std::vector<std::string> names;
    names.reserve(functions.size());
    for (uint32_t i = 0; i < functions.size(); ++i) {
        names.push_back(i == 0 ? "cc_cb_clear" :
                        "lab_transport_function_" + std::to_string(i));
        if (createFunction(device.Get(), modules[i % modules.size()], names.back().c_str(),
                           &functions[i]) != NVAPI_OK) {
            printf("FAIL create function %u\n", i); return 7;
        }
    }
    for (uint32_t slot = 0; slot < 156; ++slot) {
        uint8_t params[16]{};
        if (slot == 0) {
            uint64_t address = sharedBuffer->GetGPUVirtualAddress();
            uint32_t ignoredR21Field = 0x21A;
            std::memcpy(params, &address, sizeof(address));
            std::memcpy(params + 8, &clearCount, sizeof(clearCount));
            std::memcpy(params + 12, &ignoredR21Field, sizeof(ignoredR21Field));
        } else {
            for (uint32_t i = 0; i < sizeof(params); ++i) params[i] = uint8_t(slot + i * 17);
        }
        NVAPI_CU_KERNEL_LAUNCH_PARAMS p{};
        p.hFunction = functions[slot == 0 ? 0 : 1 + ((slot - 1) % 95)];
        p.gridDim = slot == 0 ? NVAPI_DIM3{108, 1, 1} : NVAPI_DIM3{1 + slot % 4, 1, 1};
        p.blockDim = slot == 0 ? NVAPI_DIM3{256, 1, 1} : NVAPI_DIM3{32, 1, 1};
        p.pParams = params;
        p.paramSize = sizeof(params);
        if (launch(commandList.Get(), &p, 1) != NVAPI_OK) {
            printf("FAIL launch slot %u\n", slot); return 8;
        }
    }

    // Negative address test: a pointer exactly beyond the registered interval
    // must fail instead of being passed to HIP.
    uint8_t badParams[16]{};
    uint64_t badAddress = sharedBuffer->GetGPUVirtualAddress() + clearBytes;
    uint32_t badCount = 1;
    std::memcpy(badParams, &badAddress, sizeof(badAddress));
    std::memcpy(badParams + 8, &badCount, sizeof(badCount));
    NVAPI_CU_KERNEL_LAUNCH_PARAMS badLaunch{};
    badLaunch.hFunction = functions[0];
    badLaunch.gridDim = {1, 1, 1};
    badLaunch.blockDim = {1, 1, 1};
    badLaunch.pParams = badParams;
    badLaunch.paramSize = sizeof(badParams);
    bool negativeAddressPass = launch(commandList.Get(), &badLaunch, 1) == NVAPI_INVALID_POINTER;

    ComPtr<ID3D12CommandAllocator> readAllocator;
    ComPtr<ID3D12GraphicsCommandList> readList;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&readAllocator));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, readAllocator.Get(), nullptr,
                              IID_PPV_ARGS(&readList));
    D3D12_RESOURCE_BARRIER toCopy{};
    toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toCopy.Transition.pResource = sharedBuffer.Get();
    toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    readList->ResourceBarrier(1, &toCopy);
    readList->CopyResource(readbackBuffer.Get(), sharedBuffer.Get());
    submitAndWait(readList.Get());
    readbackBuffer->Map(0, nullptr, &mapped);
    uint32_t clearMismatches = 0;
    const auto* words = static_cast<const uint32_t*>(mapped);
    for (uint32_t i = 0; i < clearCount; ++i) {
        if (words[i] != 0xFFFFFFFFu) ++clearMismatches;
    }
    readbackBuffer->Unmap(0, nullptr);
    bool unregisterPass = unregisterBuffer(sharedBuffer.Get()) != 0;
    CloseHandle(sharedHandle);
    for (auto function : functions) {
        if (destroyFunction(device.Get(), function) != NVAPI_OK) return 9;
    }
    for (auto module : modules) {
        if (destroyModule(device.Get(), module) != NVAPI_OK) return 10;
    }

    NvapiAmdDiagnosticsV1 diag{};
    diag.struct_size = sizeof(diag);
    if (!getDiag(&diag)) return 11;
    bool pass = desc.VendorId == 0x1002 && diag.module_creates == 9 &&
                diag.function_creates == 96 && diag.launch_chain_calls == 157 &&
                diag.kernels_submitted == 156 && diag.gpu_markers == 156 &&
                diag.validation_mismatches == 0 && diag.function_destroys == 96 &&
                diag.module_destroys == 9 && diag.live_functions == 0 &&
                diag.live_modules == 0 && diag.hip_device_count > 0 &&
                diag.resource_registrations == 1 && diag.resource_unregistrations == 1 &&
                diag.live_resources == 0 && diag.address_translations == 1 &&
                diag.translation_failures == 1 && diag.boundary_kernels == 1 &&
                clearMismatches == 0 && negativeAddressPass && unregisterPass &&
                diag.counts_as_s6 == 0 && diag.neural_math_executed == 0;
    printf("[%s] dxgi=0x%04x:0x%04x hip=%s arch=%s modules=%u/%u functions=%u/%u launches=%u markers=%u mismatches=%u clear_mismatches=%u translations=%u/%u negative_address=%s classification=%s counts_as_s6=%u\n",
           pass ? "PASS" : "FAIL", desc.VendorId, desc.DeviceId, diag.hip_device,
           diag.hip_arch, diag.module_creates, diag.module_destroys,
           diag.function_creates, diag.function_destroys, diag.launch_chain_calls,
           diag.gpu_markers, diag.validation_mismatches, clearMismatches,
           diag.address_translations, diag.translation_failures,
           negativeAddressPass ? "PASS" : "FAIL", diag.classification,
           diag.counts_as_s6);
    if (!jsonPath.empty()) {
        std::ofstream json(jsonPath, std::ios::binary);
        if (!json) return 12;
        json << "{\n"
             << "  \"schema\": 1,\n"
             << "  \"experiment\": \"nvapi_amd_selftest\",\n"
             << "  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
             << "  \"classification\": \"LAB_TRANSPORT_ONLY\",\n"
             << "  \"counts_as_s6\": false,\n"
             << "  \"neural_math_executed\": false,\n"
             << "  \"dxgi_vendor\": \"0x" << std::hex << desc.VendorId << "\",\n"
             << "  \"dxgi_device\": \"0x" << desc.DeviceId << "\",\n" << std::dec
             << "  \"hip_device\": \"" << diag.hip_device << "\",\n"
             << "  \"hip_arch\": \"" << diag.hip_arch << "\",\n"
             << "  \"modules_created\": " << diag.module_creates << ",\n"
             << "  \"modules_destroyed\": " << diag.module_destroys << ",\n"
             << "  \"functions_created\": " << diag.function_creates << ",\n"
             << "  \"functions_destroyed\": " << diag.function_destroys << ",\n"
             << "  \"launch_chain_calls\": " << diag.launch_chain_calls << ",\n"
             << "  \"gpu_markers\": " << diag.gpu_markers << ",\n"
             << "  \"validation_mismatches\": " << diag.validation_mismatches << ",\n"
             << "  \"boundary_kernels\": " << diag.boundary_kernels << ",\n"
             << "  \"clear_word_count\": " << clearCount << ",\n"
             << "  \"clear_mismatches\": " << clearMismatches << ",\n"
             << "  \"resource_registrations\": " << diag.resource_registrations << ",\n"
             << "  \"resource_unregistrations\": " << diag.resource_unregistrations << ",\n"
             << "  \"address_translations\": " << diag.address_translations << ",\n"
             << "  \"translation_failures\": " << diag.translation_failures << ",\n"
             << "  \"negative_address_test\": " << (negativeAddressPass ? "true" : "false") << ",\n"
             << "  \"live_modules\": " << diag.live_modules << ",\n"
             << "  \"live_functions\": " << diag.live_functions << ",\n"
             << "  \"live_resources\": " << diag.live_resources << "\n"
             << "}\n";
    }
    return pass ? 0 : 1;
}
