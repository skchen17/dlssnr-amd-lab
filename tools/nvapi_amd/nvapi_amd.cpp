// nvapi_amd — D3D12/NVAPI compatibility backend for the DLSSNR launch surface.
//
// It implements the five R610 CuModule APIs plus the two descriptor-object APIs
// observed on the RTX reference path. Boundary operations do not count as S6.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#include <d3d12.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../third_party/nvapi/nvapi.h"
#include "nvapi_amd_diag.h"
#include "nvapi_amd_d3d12_head.h"
#include "../output_head_surface_d3d12/output_head_d3d12_runtime.h"

#define OUTPUT_HEAD_RESIDENT_KERNELS_ONLY
#include "../output_head_resident/output_head_resident.cpp"
#undef OUTPUT_HEAD_RESIDENT_KERNELS_ONLY

static_assert(sizeof(NVAPI_CU_KERNEL_LAUNCH_PARAMS) == 56,
              "R610 NVAPI_CU_KERNEL_LAUNCH_PARAMS ABI mismatch");

namespace {

constexpr uint32_t kCreateCuModule = 0xAD1A677D;
constexpr uint32_t kCreateCuFunction = 0xE2436E22;
constexpr uint32_t kLaunchCuKernelChain = 0x24973538;
constexpr uint32_t kDestroyCuFunction = 0xDF295EA6;
constexpr uint32_t kDestroyCuModule = 0x41C65285;
constexpr uint32_t kGetMergedTextureSampler = 0x329FE6E0;
constexpr uint32_t kGetIndependentDescriptor = 0x0DDAC234;
constexpr uint32_t kModuleMagic = 0xA6D64D4Fu;
constexpr uint32_t kFunctionMagic = 0xA6D6464Eu;
constexpr uint32_t kMarkerMagic = 0xA6D60002u;

struct ModuleHandle {
    uint32_t magic = kModuleMagic;
    bool alive = true;
    uint32_t blob_size = 0;
    uint32_t live_functions = 0;
    uint64_t blob_hash = 0;
};

struct FunctionHandle {
    uint32_t magic = kFunctionMagic;
    bool alive = true;
    ModuleHandle* module = nullptr;
    std::string name;
    uint64_t name_hash = 0;
};

struct GpuMarker {
    uint32_t magic;
    uint32_t param_size;
    uint64_t function_hash;
    uint64_t param_hash;
};

struct ResourceMapping {
    ID3D12Resource* resource = nullptr;
    uint64_t d3d12_begin = 0;
    uint64_t byte_size = 0;
    uint8_t* hip_base = nullptr;
    hipExternalMemory_t external_memory = nullptr;
};

struct DeviceState {
    ID3D12Device* device = nullptr;
    ID3D12DescriptorHeap* view_heap = nullptr;
    ID3D12DescriptorHeap* sampler_heap = nullptr;
    ID3D12RootSignature* copy_root_signature = nullptr;
    ID3D12PipelineState* copy_pipeline = nullptr;
    ID3D12Resource* output_head_staging = nullptr;
    hipExternalMemory_t output_head_external = nullptr;
    uint16_t* output_head_hip = nullptr;
    UINT view_increment = 0;
    UINT sampler_increment = 0;
    UINT next_view = 0;
    UINT next_sampler = 0;
};

struct OutputHeadState {
    uint8_t* activation_e4 = nullptr;
    uint16_t* activation_half = nullptr;
    uint8_t* hidden = nullptr;
    uint16_t* first128 = nullptr;
    uint16_t* projection176 = nullptr;
    uint8_t* q = nullptr;
    uint8_t* k = nullptr;
    uint8_t* v = nullptr;
    uint16_t* qk = nullptr;
    uint8_t* softmax = nullptr;
    uint16_t* attention = nullptr;
    uint16_t* projected = nullptr;
    uint16_t* residual = nullptr;
    bool ready = false;
};

enum class DescriptorKind : uint32_t {
    MergedTextureSampler,
    Surface,
    Texture,
    Sampler,
};

struct DescriptorObject {
    DescriptorKind kind{};
    ID3D12Device* device = nullptr;
    D3D12_GPU_DESCRIPTOR_HANDLE texture{};
    D3D12_GPU_DESCRIPTOR_HANDLE sampler{};
    D3D12_GPU_DESCRIPTOR_HANDLE surface{};
    ID3D12Resource* resource = nullptr;
};

std::mutex g_lock;
std::unordered_map<void*, ModuleHandle*> g_modules;
std::unordered_map<void*, FunctionHandle*> g_functions;
std::unordered_map<void*, ResourceMapping> g_resources;
std::unordered_map<ID3D12Device*, DeviceState*> g_device_states;
std::unordered_map<uint64_t, DescriptorObject> g_descriptor_objects;
std::unordered_map<SIZE_T, ID3D12Resource*> g_descriptor_resources;
std::atomic<uint64_t> g_next_descriptor_handle{0xA6D0000000000001ull};
NvapiAmdDiagnosticsV1 g_diag{};
bool g_hip_initialized = false;
OutputHeadState g_output_head;
std::mutex g_output_head_lock;

struct HeadResource {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state;
};
struct HeadSession {
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    uint64_t completion = 0;
    bool sealed = false;
    std::vector<HeadResource> resources;
    std::unique_ptr<output_head_dx12::Executor> executor;
};
std::unordered_map<ID3D12GraphicsCommandList*, std::unique_ptr<HeadSession>> g_head_sessions;
NvapiAmdHeadDiagnosticsV1 g_head_diag{sizeof(NvapiAmdHeadDiagnosticsV1)};

output_head_dx12::Executor::ShaderSet ReadHeadShaders() {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&ReadHeadShaders), &module))
        throw std::runtime_error("cannot locate head shader module");
    wchar_t path[32768]{};
    DWORD length = GetModuleFileNameW(module, path, 32768);
    if (!length || length >= 32768) throw std::runtime_error("invalid head shader path");
    const auto directory = std::filesystem::path(path).parent_path();
    output_head_dx12::Executor::ShaderSet shaders;
    for (size_t i = 0; i < shaders.size(); ++i) {
        std::ifstream stream(directory / (std::string(output_head_dx12::ShaderNames[i]) + ".dxil"), std::ios::binary);
        shaders[i].assign(std::istreambuf_iterator<char>(stream), {});
        if (shaders[i].empty()) throw std::runtime_error("missing head shader");
    }
    return shaders;
}

bool HeadParamsSupported(const uint8_t* bytes) {
    // All non-relocatable words of the accepted frame1 ABI. Later history-bearing
    // frames and alternative image/scale contracts fail closed until implemented.
    static constexpr uint32_t words[] = {
        0x180, 0x280, 0xfffffffc, 0xfffffffc, 0x3d000000, 1, 0, 0,
        0, 0, 0x44200000, 0x43b40000, 0x3acccccd, 0x3b360b61, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x3acccccd, 0x3b360b61, 0x280, 0x168, 0
    };
    for (size_t i = 0; i < std::size(words); ++i) {
        const size_t offset = 32 + i * 4;
        if (offset == 56 || offset == 60 || offset == 104 || offset == 108) continue;
        uint32_t word; std::memcpy(&word, bytes + offset, 4);
        if (word != words[i]) return false;
    }
    return true;
}

NvAPI_Status RecordHeadLocked(HeadSession& session, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& launch) {
    try {
        if (launch.paramSize != 184 || launch.gridDim.x != 81 || launch.gridDim.y != 49 || launch.gridDim.z != 1 ||
            launch.blockDim.x != 32 || launch.blockDim.y != 1 || launch.blockDim.z != 1 || launch.dynSharedMemBytes != 0 ||
            !HeadParamsSupported(static_cast<const uint8_t*>(launch.pParams)))
            throw std::invalid_argument("unsupported output-head contract");
        if (session.sealed || (session.fence && session.fence->GetCompletedValue() >= session.completion))
            throw std::invalid_argument("completed session must be released");
        auto pointer = [&](size_t offset) { uint64_t v; std::memcpy(&v, static_cast<const uint8_t*>(launch.pParams) + offset, 8); return v; };
        auto buffer = [&](uint64_t address, uint64_t bytes) -> output_head_dx12::BufferView {
            for (auto& binding : session.resources) {
                auto* r = binding.resource.Get(); const auto d = r->GetDesc();
                if (d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) continue;
                const uint64_t begin = r->GetGPUVirtualAddress();
                if (address >= begin && address - begin <= d.Width && bytes <= d.Width - (address - begin))
                    return {r, address - begin, binding.state};
            }
            throw std::invalid_argument("head address is not registered in this session");
        };
        auto texture = [&](uint64_t token, bool surface) -> output_head_dx12::TextureView {
            const auto found = g_descriptor_objects.find(token);
            if (found == g_descriptor_objects.end() ||
                (surface ? found->second.kind != DescriptorKind::Surface : found->second.kind != DescriptorKind::MergedTextureSampler))
                throw std::invalid_argument("unsupported head descriptor token");
            for (auto& binding : session.resources)
                if (binding.resource.Get() == found->second.resource)
                    return {binding.resource.Get(), binding.state};
            throw std::invalid_argument("head texture is not registered in this session");
        };
        output_head_dx12::Inputs inputs{buffer(pointer(0), output_head_dx12::MainBytes),
            buffer(pointer(8), output_head_dx12::SkipBytes), buffer(pointer(24), output_head_dx12::HeadBytes),
            texture(pointer(56), false), texture(pointer(16), true)};
        const auto blend = buffer(pointer(104), 2);
        if (inputs.main.resource != inputs.skip.resource || inputs.main.offset != 13873152 || inputs.skip.offset != 110592 ||
            inputs.head.offset != 147429888 || blend.resource != inputs.head.resource || blend.offset != 147429376)
            throw std::invalid_argument("unsupported model/activation layout");
        session.executor->Record(session.list.Get(), inputs);
        ++g_head_diag.head_records;
        ++g_diag.kernels_submitted;
        ++g_diag.neural_math_executed;
        g_diag.address_translations += 4;
        ++g_diag.d3d12_copy_dispatches;
        strncpy_s(g_diag.classification, "D3D12_RECORDED_OUTPUT_HEAD", _TRUNCATE);
        return NVAPI_OK;
    } catch (const std::invalid_argument&) {
        ++g_head_diag.rejected_records;
        return NVAPI_INVALID_ARGUMENT;
    } catch (...) {
        ++g_head_diag.rejected_records;
        return NVAPI_ERROR;
    }
}

uint64_t Fnv1a(const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

__global__ void TransportMarker(GpuMarker* out, const uint8_t* params,
                                uint32_t paramSize, uint64_t functionHash) {
    if (blockIdx.x || blockIdx.y || blockIdx.z ||
        threadIdx.x || threadIdx.y || threadIdx.z) return;
    uint64_t hash = 14695981039346656037ull;
    for (uint32_t i = 0; i < paramSize; ++i) {
        hash ^= params[i];
        hash *= 1099511628211ull;
    }
    out->magic = kMarkerMagic;
    out->param_size = paramSize;
    out->function_hash = functionHash;
    out->param_hash = hash;
}

__global__ void ClearControlBuffer(uint32_t* output, uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = 0xFFFFFFFFu;
}

void ResetDiagLocked() {
    std::memset(&g_diag, 0, sizeof(g_diag));
    g_diag.struct_size = sizeof(g_diag);
    g_diag.schema = 1;
    strcpy_s(g_diag.classification, "LAB_TRANSPORT_ONLY");
    if (!g_hip_initialized) {
        g_hip_initialized = true;
        int count = 0;
        if (hipGetDeviceCount(&count) == hipSuccess && count > 0) {
            g_diag.hip_device_count = static_cast<uint32_t>(count);
            hipDeviceProp_t prop{};
            if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
                strncpy_s(g_diag.hip_device, prop.name, _TRUNCATE);
                strncpy_s(g_diag.hip_arch, prop.gcnArchName, _TRUNCATE);
            }
            hipSetDevice(0);
        }
    } else {
        int count = 0;
        if (hipGetDeviceCount(&count) == hipSuccess) g_diag.hip_device_count = count;
        if (count > 0) {
            hipDeviceProp_t prop{};
            if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
                strncpy_s(g_diag.hip_device, prop.name, _TRUNCATE);
                strncpy_s(g_diag.hip_arch, prop.gcnArchName, _TRUNCATE);
            }
        }
    }
}

void EnsureDiagLocked() {
    if (!g_diag.struct_size) ResetDiagLocked();
}

ModuleHandle* FindModule(NVDX_ObjectHandle handle) {
    auto it = g_modules.find(handle);
    return it == g_modules.end() ? nullptr : it->second;
}

FunctionHandle* FindFunction(NVDX_ObjectHandle handle) {
    auto it = g_functions.find(handle);
    return it == g_functions.end() ? nullptr : it->second;
}

uint8_t* TranslateAddressLocked(uint64_t d3d12Address, uint64_t byteSize) {
    for (auto& [key, mapping] : g_resources) {
        if (d3d12Address >= mapping.d3d12_begin &&
            d3d12Address - mapping.d3d12_begin <= mapping.byte_size &&
            byteSize <= mapping.byte_size - (d3d12Address - mapping.d3d12_begin)) {
            ++g_diag.address_translations;
            return mapping.hip_base + (d3d12Address - mapping.d3d12_begin);
        }
    }
    ++g_diag.translation_failures;
    return nullptr;
}

bool EnsureOutputHeadState() {
    if (g_output_head.ready) return true;
    auto allocate = [](auto** pointer, size_t bytes) {
        return hipMalloc(reinterpret_cast<void**>(pointer), bytes) == hipSuccess;
    };
    if (!allocate(&g_output_head.activation_e4, A_BYTES) ||
        !allocate(&g_output_head.activation_half, H_BYTES) ||
        !allocate(&g_output_head.hidden, HIDDEN_BYTES) ||
        !allocate(&g_output_head.first128, H_BYTES) ||
        !allocate(&g_output_head.projection176, PROJ176_VALUES * 2) ||
        !allocate(&g_output_head.q, A_BYTES) ||
        !allocate(&g_output_head.k, A_BYTES) ||
        !allocate(&g_output_head.v, A_BYTES) ||
        !allocate(&g_output_head.qk, QK_VALUES * 2) ||
        !allocate(&g_output_head.softmax, SOFT_BYTES) ||
        !allocate(&g_output_head.attention, H_BYTES) ||
        !allocate(&g_output_head.projected, H_BYTES) ||
        !allocate(&g_output_head.residual, RESIDUAL_VALUES * 2)) return false;
    g_output_head.ready = true;
    return true;
}

hipError_t DispatchOutputHead(const uint8_t* mainFeature, const uint8_t* skipFeature,
                              const uint8_t* head, uint16_t* output) {
    std::lock_guard<std::mutex> executionGuard(g_output_head_lock);
    if (!EnsureOutputHeadState()) return hipErrorOutOfMemory;
    float scaleFloat = 0.0f;
    hipError_t error = hipMemcpy(&scaleFloat, head + 19664, sizeof(scaleFloat),
                                 hipMemcpyDeviceToHost);
    if (error != hipSuccess) return error;
    const uint16_t scale = __half_as_ushort(__float2half_rn(scaleFloat));
    const uint16_t epsilon = __half_as_ushort(
        __float2half_rn(6.199999916134402e-05f));
    auto grid = [](size_t count) { return dim3(uint32_t((count + 255) / 256)); };
    const dim3 block(256);
    auto& state = g_output_head;
    Activation<<<grid(VALUES), block>>>(mainFeature, skipFeature, head,
                                        state.activation_e4,
                                        state.activation_half);
    Hidden<<<grid(VALUES * 4), block>>>(state.activation_e4, head, state.hidden);
    First128<<<grid(VALUES), block>>>(state.activation_half, state.hidden, head,
                                      state.first128);
    Pack176<<<grid(VALUES), block>>>(state.first128, state.activation_e4);
    Project176<<<grid(PROJ176_VALUES), block>>>(state.activation_e4, head,
                                                state.projection176);
    PrepQK<<<grid(CTAS * 64), block>>>(state.projection176, state.q, state.k,
                                       scale, epsilon);
    PrepV<<<grid(VALUES), block>>>(state.projection176, state.v);
    QK<<<grid(QK_VALUES), block>>>(state.q, state.k, head, state.qk);
    Softmax<<<grid(CTAS * 64), block>>>(state.qk, state.softmax);
    Attention<<<grid(VALUES), block>>>(state.softmax, state.v, state.attention);
    FinalPack<<<grid(VALUES), block>>>(state.attention, state.q);
    FinalProject<<<grid(VALUES), block>>>(state.q, state.first128, head,
                                          state.projected);
    Tail<<<grid(RESIDUAL_VALUES), block>>>(state.projected, head, state.residual);
    Surface<<<grid(CTAS * 64), block>>>(state.residual, nullptr, output);
    error = hipGetLastError();
    if (error == hipSuccess) error = hipDeviceSynchronize();
    return error;
}

DeviceState* GetOrCreateDeviceStateLocked(ID3D12Device* device) {
    auto found = g_device_states.find(device);
    if (found != g_device_states.end()) return found->second;

    auto* state = new DeviceState;
    state->device = device;
    device->AddRef();
    auto fail = [&]() -> DeviceState* {
        if (state->output_head_external) hipDestroyExternalMemory(state->output_head_external);
        if (state->output_head_staging) state->output_head_staging->Release();
        if (state->copy_pipeline) state->copy_pipeline->Release();
        if (state->copy_root_signature) state->copy_root_signature->Release();
        if (state->sampler_heap) state->sampler_heap->Release();
        if (state->view_heap) state->view_heap->Release();
        state->device->Release();
        delete state;
        ++g_diag.pipeline_failures;
        return nullptr;
    };
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 4096;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc,
                                            IID_PPV_ARGS(&state->view_heap)))) {
        return fail();
    }
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    heapDesc.NumDescriptors = 2048;
    if (FAILED(device->CreateDescriptorHeap(&heapDesc,
                                            IID_PPV_ARGS(&state->sampler_heap)))) {
        return fail();
    }
    state->view_increment = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    state->sampler_increment = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

    D3D12_DESCRIPTOR_RANGE ranges[3]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
    ranges[2] = {D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER rootParams[4]{};
    for (UINT i = 0; i < 3; ++i) {
        rootParams[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParams[i].DescriptorTable.NumDescriptorRanges = 1;
        rootParams[i].DescriptorTable.pDescriptorRanges = &ranges[i];
        rootParams[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rootParams[3].Constants.ShaderRegister = 0;
    rootParams[3].Constants.RegisterSpace = 0;
    rootParams[3].Constants.Num32BitValues = 10;
    rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = 4;
    rootDesc.pParameters = rootParams;
    ID3DBlob* rootBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    HMODULE d3d12Module = LoadLibraryExW(L"d3d12.dll", nullptr,
                                         LOAD_LIBRARY_SEARCH_SYSTEM32);
    auto serializeRootSignature = d3d12Module
        ? reinterpret_cast<PFN_D3D12_SERIALIZE_ROOT_SIGNATURE>(
              GetProcAddress(d3d12Module, "D3D12SerializeRootSignature"))
        : nullptr;
    HRESULT hr = serializeRootSignature
        ? serializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                 &rootBlob, &errorBlob)
        : E_NOINTERFACE;
    if (errorBlob) errorBlob->Release();
    if (FAILED(hr) || !rootBlob) {
        if (d3d12Module) FreeLibrary(d3d12Module);
        return fail();
    }
    hr = device->CreateRootSignature(0, rootBlob->GetBufferPointer(),
                                     rootBlob->GetBufferSize(),
                                     IID_PPV_ARGS(&state->copy_root_signature));
    rootBlob->Release();
    if (d3d12Module) FreeLibrary(d3d12Module);
    if (FAILED(hr)) return fail();

    HMODULE selfModule = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&GetOrCreateDeviceStateLocked),
                       &selfModule);
    wchar_t modulePath[MAX_PATH]{};
    GetModuleFileNameW(selfModule, modulePath, MAX_PATH);
    std::wstring dxilPath = modulePath;
    size_t slash = dxilPath.find_last_of(L"\\/");
    dxilPath.resize(slash == std::wstring::npos ? 0 : slash + 1);
    dxilPath += L"cg2r_copy.dxil";
    std::ifstream dxilFile(dxilPath, std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(dxilFile)),
                              std::istreambuf_iterator<char>());
    if (dxil.empty()) return fail();
    D3D12_COMPUTE_PIPELINE_STATE_DESC pipelineDesc{};
    pipelineDesc.pRootSignature = state->copy_root_signature;
    pipelineDesc.CS = {dxil.data(), dxil.size()};
    if (FAILED(device->CreateComputePipelineState(
            &pipelineDesc, IID_PPV_ARGS(&state->copy_pipeline)))) {
        return fail();
    }
    D3D12_HEAP_PROPERTIES stagingHeap{};
    stagingHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC stagingDesc{};
    stagingDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    stagingDesc.Width = SURFACE_VALUES * 2;
    stagingDesc.Height = 1;
    stagingDesc.DepthOrArraySize = 1;
    stagingDesc.MipLevels = 1;
    stagingDesc.SampleDesc.Count = 1;
    stagingDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->CreateCommittedResource(
            &stagingHeap, D3D12_HEAP_FLAG_SHARED, &stagingDesc,
            D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&state->output_head_staging)))) return fail();
    HANDLE stagingHandle = nullptr;
    if (FAILED(device->CreateSharedHandle(state->output_head_staging, nullptr,
                                           GENERIC_ALL, nullptr, &stagingHandle))) return fail();
    hipExternalMemoryHandleDesc externalDesc{};
    externalDesc.type = hipExternalMemoryHandleTypeD3D12Resource;
    externalDesc.handle.win32.handle = stagingHandle;
    externalDesc.size = SURFACE_VALUES * 2;
    hipError_t stagingError = hipImportExternalMemory(
        &state->output_head_external, &externalDesc);
    CloseHandle(stagingHandle);
    if (stagingError != hipSuccess) return fail();
    hipExternalMemoryBufferDesc mappedDesc{};
    mappedDesc.size = SURFACE_VALUES * 2;
    if (hipExternalMemoryGetMappedBuffer(
            reinterpret_cast<void**>(&state->output_head_hip),
            state->output_head_external, &mappedDesc) != hipSuccess) return fail();
    g_device_states[device] = state;
    return state;
}

D3D12_CPU_DESCRIPTOR_HANDLE AllocateCpuDescriptor(DeviceState* state,
                                                   bool sampler,
                                                   UINT* indexOut) {
    UINT index = sampler ? state->next_sampler++ : state->next_view++;
    D3D12_CPU_DESCRIPTOR_HANDLE handle = sampler
        ? state->sampler_heap->GetCPUDescriptorHandleForHeapStart()
        : state->view_heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += SIZE_T(index) * (sampler ? state->sampler_increment
                                           : state->view_increment);
    *indexOut = index;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE GpuDescriptor(DeviceState* state, bool sampler,
                                          UINT index) {
    D3D12_GPU_DESCRIPTOR_HANDLE handle = sampler
        ? state->sampler_heap->GetGPUDescriptorHandleForHeapStart()
        : state->view_heap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += UINT64(index) * (sampler ? state->sampler_increment
                                           : state->view_increment);
    return handle;
}

NvAPI_Status __cdecl AmdGetMergedTextureSampler(
        NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS* params) {
    if (!params || !params->pDevice || !params->texDesc.ptr ||
        !params->smpDesc.ptr) return NVAPI_INVALID_POINTER;
    std::lock_guard<std::mutex> guard(g_lock);
    EnsureDiagLocked();
    DeviceState* state = GetOrCreateDeviceStateLocked(params->pDevice);
    if (!state || state->next_view >= 4096 || state->next_sampler >= 2048)
        return NVAPI_OUT_OF_MEMORY;
    UINT viewIndex = 0, samplerIndex = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE viewDst =
        AllocateCpuDescriptor(state, false, &viewIndex);
    D3D12_CPU_DESCRIPTOR_HANDLE samplerDst =
        AllocateCpuDescriptor(state, true, &samplerIndex);
    params->pDevice->CopyDescriptorsSimple(1, viewDst, params->texDesc,
                                           D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    params->pDevice->CopyDescriptorsSimple(1, samplerDst, params->smpDesc,
                                           D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    uint64_t handle = g_next_descriptor_handle.fetch_add(1);
    DescriptorObject object{};
    object.kind = DescriptorKind::MergedTextureSampler;
    object.device = params->pDevice;
    auto registered = g_descriptor_resources.find(params->texDesc.ptr);
    if (registered != g_descriptor_resources.end()) object.resource = registered->second;
    object.texture = GpuDescriptor(state, false, viewIndex);
    object.sampler = GpuDescriptor(state, true, samplerIndex);
    g_descriptor_objects[handle] = object;
    params->textureHandle = handle;
    ++g_diag.merged_texture_sampler_calls;
    ++g_diag.live_descriptor_objects;
    return NVAPI_OK;
}

NvAPI_Status __cdecl AmdGetIndependentDescriptor(
        NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS* params) {
    if (!params || !params->pDevice || !params->desc.ptr)
        return NVAPI_INVALID_POINTER;
    bool sampler = params->type ==
        NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SAMPLER;
    if (params->type >
        NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SAMPLER)
        return NVAPI_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(g_lock);
    EnsureDiagLocked();
    DeviceState* state = GetOrCreateDeviceStateLocked(params->pDevice);
    if (!state || (!sampler && state->next_view >= 4096) ||
        (sampler && state->next_sampler >= 2048)) return NVAPI_OUT_OF_MEMORY;
    UINT index = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE destination =
        AllocateCpuDescriptor(state, sampler, &index);
    params->pDevice->CopyDescriptorsSimple(
        1, destination, params->desc,
        sampler ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
                : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    uint64_t handle = g_next_descriptor_handle.fetch_add(1);
    DescriptorObject object{};
    object.device = params->pDevice;
    auto registered = g_descriptor_resources.find(params->desc.ptr);
    if (registered != g_descriptor_resources.end()) object.resource = registered->second;
    if (params->type ==
        NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SURFACE) {
        object.kind = DescriptorKind::Surface;
        object.surface = GpuDescriptor(state, false, index);
    } else if (params->type ==
               NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_TEXTURE) {
        object.kind = DescriptorKind::Texture;
        object.texture = GpuDescriptor(state, false, index);
    } else {
        object.kind = DescriptorKind::Sampler;
        object.sampler = GpuDescriptor(state, true, index);
    }
    g_descriptor_objects[handle] = object;
    params->handle = handle;
    ++g_diag.independent_descriptor_calls;
    ++g_diag.live_descriptor_objects;
    return NVAPI_OK;
}

NvAPI_Status __cdecl AmdCreateCuModule(ID3D12Device* device, const void* blob,
                                       NvU32 size, NVDX_ObjectHandle* outModule) {
    if (!device || !blob || !size || !outModule) return NVAPI_INVALID_POINTER;
    auto* module = new ModuleHandle;
    module->blob_size = size;
    module->blob_hash = Fnv1a(blob, size);
    std::lock_guard<std::mutex> guard(g_lock);
    EnsureDiagLocked();
    g_modules[module] = module;
    *outModule = reinterpret_cast<NVDX_ObjectHandle>(module);
    ++g_diag.module_creates;
    ++g_diag.live_modules;
    return NVAPI_OK;
}

NvAPI_Status __cdecl AmdCreateCuFunction(ID3D12Device* device,
                                         NVDX_ObjectHandle moduleHandle,
                                         const char* name,
                                         NVDX_ObjectHandle* outFunction) {
    if (!device || !moduleHandle || !name || !outFunction) return NVAPI_INVALID_POINTER;
    std::lock_guard<std::mutex> guard(g_lock);
    EnsureDiagLocked();
    ModuleHandle* module = FindModule(moduleHandle);
    if (!module || !module->alive) return NVAPI_INVALID_HANDLE;
    auto* function = new FunctionHandle;
    function->module = module;
    function->name = name;
    function->name_hash = Fnv1a(name, std::strlen(name));
    g_functions[function] = function;
    *outFunction = reinterpret_cast<NVDX_ObjectHandle>(function);
    ++module->live_functions;
    ++g_diag.function_creates;
    ++g_diag.live_functions;
    return NVAPI_OK;
}

NvAPI_Status __cdecl AmdLaunchCuKernelChain(
        ID3D12GraphicsCommandList* commandList,
        const NVAPI_CU_KERNEL_LAUNCH_PARAMS* kernels, NvU32 numKernels) {
    if (!commandList || !kernels || !numKernels) return NVAPI_INVALID_POINTER;
    {
        std::lock_guard<std::mutex> guard(g_lock);
        EnsureDiagLocked();
        ++g_diag.launch_chain_calls;
    }

    for (NvU32 i = 0; i < numKernels; ++i) {
        const auto& launch = kernels[i];
        FunctionHandle* function = nullptr;
        {
            std::lock_guard<std::mutex> guard(g_lock);
            function = FindFunction(launch.hFunction);
            if (!function || !function->alive) return NVAPI_INVALID_HANDLE;
        }
        if (!launch.pParams || !launch.paramSize ||
            !launch.gridDim.x || !launch.gridDim.y || !launch.gridDim.z ||
            !launch.blockDim.x || !launch.blockDim.y || !launch.blockDim.z) {
            return NVAPI_INVALID_ARGUMENT;
        }

        {
            std::lock_guard<std::mutex> guard(g_lock);
            const auto session = g_head_sessions.find(commandList);
            if (session != g_head_sessions.end()) {
                if (function->name != "cc_tinlayout_fused_post_block_swin_1h_32_fp8") {
                    ++g_head_diag.rejected_records;
                    return NVAPI_NO_IMPLEMENTATION;
                }
                const auto status = RecordHeadLocked(*session->second, launch);
                if (status != NVAPI_OK) return status;
                continue;
            }
        }

        if (function->name == "cc_cb_clear") {
            if (launch.paramSize != 16) return NVAPI_INVALID_ARGUMENT;
            uint64_t d3d12Address = 0;
            uint32_t count = 0;
            std::memcpy(&d3d12Address, launch.pParams, sizeof(d3d12Address));
            std::memcpy(&count, static_cast<const uint8_t*>(launch.pParams) + 8,
                        sizeof(count));
            uint8_t* translated = nullptr;
            {
                std::lock_guard<std::mutex> guard(g_lock);
                translated = TranslateAddressLocked(d3d12Address,
                                                    uint64_t(count) * sizeof(uint32_t));
            }
            if (!translated) return NVAPI_INVALID_POINTER;
            hipLaunchKernelGGL(ClearControlBuffer,
                               dim3(launch.gridDim.x, launch.gridDim.y, launch.gridDim.z),
                               dim3(launch.blockDim.x, launch.blockDim.y, launch.blockDim.z),
                               launch.dynSharedMemBytes, 0,
                               reinterpret_cast<uint32_t*>(translated), count);
            hipError_t clearError = hipGetLastError();
            if (clearError == hipSuccess) clearError = hipDeviceSynchronize();
            if (clearError != hipSuccess) return NVAPI_ERROR;
            std::lock_guard<std::mutex> guard(g_lock);
            ++g_diag.kernels_submitted;
            ++g_diag.gpu_markers;
            ++g_diag.boundary_kernels;
            continue;
        }

        if (function->name == "cg2r_copy_kernel") {
            if (launch.paramSize != 72) return NVAPI_INVALID_ARGUMENT;
            const auto* bytes = static_cast<const uint8_t*>(launch.pParams);
            uint64_t textureHandle = 0, surfaceHandle = 0;
            std::memcpy(&textureHandle, bytes, sizeof(textureHandle));
            std::memcpy(&surfaceHandle, bytes + 8, sizeof(surfaceHandle));
            DescriptorObject texture{}, surface{};
            DeviceState* state = nullptr;
            {
                std::lock_guard<std::mutex> guard(g_lock);
                auto textureIt = g_descriptor_objects.find(textureHandle);
                auto surfaceIt = g_descriptor_objects.find(surfaceHandle);
                if (textureIt == g_descriptor_objects.end() ||
                    surfaceIt == g_descriptor_objects.end())
                    return NVAPI_INVALID_HANDLE;
                texture = textureIt->second;
                surface = surfaceIt->second;
                if (texture.kind != DescriptorKind::MergedTextureSampler ||
                    surface.kind != DescriptorKind::Surface ||
                    texture.device != surface.device)
                    return NVAPI_INVALID_ARGUMENT;
                auto stateIt = g_device_states.find(texture.device);
                if (stateIt == g_device_states.end()) return NVAPI_INVALID_HANDLE;
                state = stateIt->second;
            }
            ID3D12Device* commandDevice = nullptr;
            if (FAILED(commandList->GetDevice(IID_PPV_ARGS(&commandDevice))) ||
                commandDevice != texture.device) {
                if (commandDevice) commandDevice->Release();
                return NVAPI_INVALID_ARGUMENT;
            }
            commandDevice->Release();

            uint32_t constants[10]{};
            std::memcpy(constants, bytes + 16, 8 * sizeof(uint32_t));
            std::memcpy(constants + 8, bytes + 64, 2 * sizeof(uint32_t));
            ID3D12DescriptorHeap* heaps[] = {state->view_heap,
                                             state->sampler_heap};
            commandList->SetDescriptorHeaps(2, heaps);
            commandList->SetComputeRootSignature(state->copy_root_signature);
            commandList->SetPipelineState(state->copy_pipeline);
            commandList->SetComputeRootDescriptorTable(0, texture.texture);
            commandList->SetComputeRootDescriptorTable(1, surface.surface);
            commandList->SetComputeRootDescriptorTable(2, texture.sampler);
            commandList->SetComputeRoot32BitConstants(3, 10, constants, 0);
            commandList->Dispatch(launch.gridDim.x, launch.gridDim.y,
                                  launch.gridDim.z);
            std::lock_guard<std::mutex> guard(g_lock);
            ++g_diag.kernels_submitted;
            ++g_diag.boundary_kernels;
            ++g_diag.d3d12_copy_dispatches;
            continue;
        }

        if (function->name == "cc_tinlayout_fused_post_block_swin_1h_32_fp8") {
            if (launch.paramSize != 184 || launch.gridDim.x != GW ||
                launch.gridDim.y != GH || launch.gridDim.z != 1 ||
                launch.blockDim.x != 32 || launch.blockDim.y != 1 ||
                launch.blockDim.z != 1) return NVAPI_INVALID_ARGUMENT;
            const auto* bytes = static_cast<const uint8_t*>(launch.pParams);
            uint64_t mainAddress = 0, skipAddress = 0, outputToken = 0;
            uint64_t headAddress = 0, blendAddress = 0;
            std::memcpy(&mainAddress, bytes, 8);
            std::memcpy(&skipAddress, bytes + 8, 8);
            std::memcpy(&outputToken, bytes + 16, 8);
            std::memcpy(&headAddress, bytes + 24, 8);
            std::memcpy(&blendAddress, bytes + 104, 8);
            uint8_t *mainFeature = nullptr, *skipFeature = nullptr;
            uint8_t *head = nullptr, *blend = nullptr, *output = nullptr;
            ID3D12Resource* outputResource = nullptr;
            ID3D12Resource* stagingResource = nullptr;
            ID3D12Device* outputDevice = nullptr;
            {
                std::lock_guard<std::mutex> guard(g_lock);
                mainFeature = TranslateAddressLocked(mainAddress, MAIN_BYTES);
                skipFeature = TranslateAddressLocked(skipAddress, SKIP_BYTES);
                head = TranslateAddressLocked(headAddress, HEAD_BYTES);
                blend = TranslateAddressLocked(blendAddress, 2);
                auto descriptor = g_descriptor_objects.find(outputToken);
                if (descriptor != g_descriptor_objects.end() &&
                    descriptor->second.kind == DescriptorKind::Surface &&
                    descriptor->second.resource) {
                    auto state = g_device_states.find(descriptor->second.device);
                    if (state != g_device_states.end()) {
                        output = reinterpret_cast<uint8_t*>(
                            state->second->output_head_hip);
                        outputResource = descriptor->second.resource;
                        stagingResource = state->second->output_head_staging;
                        outputDevice = descriptor->second.device;
                    }
                } else {
                    // Retain the address-lowered laboratory path for regression
                    // tests. Captured feature-18 traffic uses a surface object.
                    output = TranslateAddressLocked(outputToken,
                                                    SURFACE_VALUES * 2);
                }
            }
            if (!mainFeature || !skipFeature || !output || !head || !blend)
                return NVAPI_INVALID_POINTER;
            if (DispatchOutputHead(mainFeature, skipFeature, head,
                                   reinterpret_cast<uint16_t*>(output)) != hipSuccess)
                return NVAPI_ERROR;
            if (outputResource) {
                D3D12_RESOURCE_DESC outputDesc = outputResource->GetDesc();
                if (!stagingResource ||
                    outputDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                    outputDesc.Width != WIDTH || outputDesc.Height != HEIGHT ||
                    outputDesc.DepthOrArraySize != 1 || outputDesc.MipLevels != 1 ||
                    outputDesc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT)
                    return NVAPI_INVALID_ARGUMENT;
                ID3D12Device* commandDevice = nullptr;
                if (FAILED(commandList->GetDevice(IID_PPV_ARGS(&commandDevice))) ||
                    commandDevice != outputDevice) {
                    if (commandDevice) commandDevice->Release();
                    return NVAPI_INVALID_ARGUMENT;
                }
                commandDevice->Release();

                D3D12_RESOURCE_BARRIER barriers[2]{};
                barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[0].Transition.pResource = stagingResource;
                barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[1].Transition.pResource = outputResource;
                barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
                barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                commandList->ResourceBarrier(2, barriers);

                D3D12_TEXTURE_COPY_LOCATION source{};
                source.pResource = stagingResource;
                source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                source.PlacedFootprint.Footprint.Format =
                    DXGI_FORMAT_R16G16B16A16_FLOAT;
                source.PlacedFootprint.Footprint.Width = WIDTH;
                source.PlacedFootprint.Footprint.Height = HEIGHT;
                source.PlacedFootprint.Footprint.Depth = 1;
                source.PlacedFootprint.Footprint.RowPitch = WIDTH * 8;
                D3D12_TEXTURE_COPY_LOCATION destination{};
                destination.pResource = outputResource;
                destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                destination.SubresourceIndex = 0;
                commandList->CopyTextureRegion(&destination, 0, 0, 0,
                                               &source, nullptr);

                std::swap(barriers[0].Transition.StateBefore,
                          barriers[0].Transition.StateAfter);
                std::swap(barriers[1].Transition.StateBefore,
                          barriers[1].Transition.StateAfter);
                commandList->ResourceBarrier(2, barriers);
            }
            std::lock_guard<std::mutex> guard(g_lock);
            ++g_diag.kernels_submitted;
            ++g_diag.neural_math_executed;
            if (outputResource) ++g_diag.d3d12_copy_dispatches;
            strncpy_s(g_diag.classification,
                      "PARTIAL_NEURAL_OUTPUT_HEAD", _TRUNCATE);
            continue;
        }

        uint8_t* deviceParams = nullptr;
        GpuMarker* deviceMarker = nullptr;
        hipError_t error = hipMalloc(&deviceParams, launch.paramSize);
        if (error == hipSuccess) error = hipMalloc(&deviceMarker, sizeof(GpuMarker));
        if (error == hipSuccess) {
            error = hipMemcpy(deviceParams, launch.pParams, launch.paramSize,
                              hipMemcpyHostToDevice);
        }
        if (error == hipSuccess) {
            hipLaunchKernelGGL(TransportMarker,
                               dim3(launch.gridDim.x, launch.gridDim.y, launch.gridDim.z),
                               dim3(launch.blockDim.x, launch.blockDim.y, launch.blockDim.z),
                               launch.dynSharedMemBytes, 0, deviceMarker, deviceParams,
                               launch.paramSize, function->name_hash);
            error = hipGetLastError();
        }
        GpuMarker marker{};
        if (error == hipSuccess) {
            error = hipMemcpy(&marker, deviceMarker, sizeof(marker), hipMemcpyDeviceToHost);
        }
        if (deviceMarker) hipFree(deviceMarker);
        if (deviceParams) hipFree(deviceParams);
        if (error != hipSuccess) return NVAPI_ERROR;

        const uint64_t hostHash = Fnv1a(launch.pParams, launch.paramSize);
        const bool valid = marker.magic == kMarkerMagic &&
                           marker.param_size == launch.paramSize &&
                           marker.function_hash == function->name_hash &&
                           marker.param_hash == hostHash;
        std::lock_guard<std::mutex> guard(g_lock);
        ++g_diag.kernels_submitted;
        if (valid) ++g_diag.gpu_markers;
        else ++g_diag.validation_mismatches;
    }
    return NVAPI_OK;
}

NvAPI_Status __cdecl AmdDestroyCuFunction(ID3D12Device* device,
                                          NVDX_ObjectHandle functionHandle) {
    if (!device || !functionHandle) return NVAPI_INVALID_POINTER;
    std::lock_guard<std::mutex> guard(g_lock);
    FunctionHandle* function = FindFunction(functionHandle);
    if (!function || !function->alive) return NVAPI_INVALID_HANDLE;
    function->alive = false;
    --function->module->live_functions;
    --g_diag.live_functions;
    ++g_diag.function_destroys;
    return NVAPI_OK;
}

NvAPI_Status __cdecl AmdDestroyCuModule(ID3D12Device* device,
                                        NVDX_ObjectHandle moduleHandle) {
    if (!device || !moduleHandle) return NVAPI_INVALID_POINTER;
    std::lock_guard<std::mutex> guard(g_lock);
    ModuleHandle* module = FindModule(moduleHandle);
    if (!module || !module->alive) return NVAPI_INVALID_HANDLE;
    if (module->live_functions) return NVAPI_HANDLE_INVALIDATED;
    module->alive = false;
    --g_diag.live_modules;
    ++g_diag.module_destroys;
    return NVAPI_OK;
}

} // namespace

extern "C" __declspec(dllexport) void* __cdecl nvapi_QueryInterface(uint32_t id) {
    switch (id) {
    case kCreateCuModule: return reinterpret_cast<void*>(&AmdCreateCuModule);
    case kCreateCuFunction: return reinterpret_cast<void*>(&AmdCreateCuFunction);
    case kLaunchCuKernelChain: return reinterpret_cast<void*>(&AmdLaunchCuKernelChain);
    case kDestroyCuFunction: return reinterpret_cast<void*>(&AmdDestroyCuFunction);
    case kDestroyCuModule: return reinterpret_cast<void*>(&AmdDestroyCuModule);
    case kGetMergedTextureSampler:
        return reinterpret_cast<void*>(&AmdGetMergedTextureSampler);
    case kGetIndependentDescriptor:
        return reinterpret_cast<void*>(&AmdGetIndependentDescriptor);
    default: return nullptr;
    }
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_GetDiagnostics(
        NvapiAmdDiagnosticsV1* out) {
    if (!out || out->struct_size != sizeof(NvapiAmdDiagnosticsV1)) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    EnsureDiagLocked();
    *out = g_diag;
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_ResetDiagnostics() {
    std::lock_guard<std::mutex> guard(g_lock);
    if (g_diag.live_modules || g_diag.live_functions || g_diag.live_resources || !g_head_sessions.empty()) return 0;
    ResetDiagLocked();
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_BeginHeadD3D12(const NvapiAmdHeadSessionV1* config) {
    if (!config || config->struct_size != sizeof(*config) || config->contract != 1 ||
        !config->command_list ||
        !config->resources || config->resource_count < 4 || config->resource_count > 16) return 0;
    try {
        const bool deferred = !config->queue && !config->completion_fence && !config->completion_value;
        if (!deferred && (!config->queue || !config->completion_fence || !config->completion_value)) return 0;
        auto session = std::make_unique<HeadSession>();
        session->list = static_cast<ID3D12GraphicsCommandList*>(config->command_list);
        session->queue = static_cast<ID3D12CommandQueue*>(config->queue);
        session->fence = static_cast<ID3D12Fence*>(config->completion_fence);
        session->completion = config->completion_value;
        if (session->list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return 0;
        Microsoft::WRL::ComPtr<ID3D12Device> device, queueDevice, fenceDevice;
        output_head_dx12::Check(session->list->GetDevice(IID_PPV_ARGS(&device)));
        if (!deferred) {
            if (session->fence->GetCompletedValue() >= session->completion ||
                session->queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT) return 0;
            output_head_dx12::Check(session->queue->GetDevice(IID_PPV_ARGS(&queueDevice)));
            output_head_dx12::Check(session->fence->GetDevice(IID_PPV_ARGS(&fenceDevice)));
            if (device.Get() != queueDevice.Get() || device.Get() != fenceDevice.Get()) return 0;
        }
        for (uint32_t i = 0; i < config->resource_count; ++i) {
            const auto& binding = config->resources[i];
            if (!binding.resource) return 0;
            auto* resource = static_cast<ID3D12Resource*>(binding.resource);
            Microsoft::WRL::ComPtr<ID3D12Device> resourceDevice;
            output_head_dx12::Check(resource->GetDevice(IID_PPV_ARGS(&resourceDevice)));
            if (resourceDevice.Get() != device.Get()) return 0;
            for (const auto& previous : session->resources)
                if (previous.resource.Get() == resource) return 0;
            HeadResource entry;
            entry.resource = resource;
            entry.state = static_cast<D3D12_RESOURCE_STATES>(binding.state);
            session->resources.push_back(std::move(entry));
        }
        std::lock_guard<std::mutex> guard(g_lock);
        if (g_head_sessions.count(session->list.Get())) return 0;
        session->executor = std::make_unique<output_head_dx12::Executor>(device.Get(), ReadHeadShaders());
        g_head_sessions.emplace(session->list.Get(), std::move(session));
        ++g_head_diag.active_sessions;
        ++g_head_diag.sessions_created;
        return 1;
    } catch (...) { return 0; }
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_ReleaseHeadD3D12(void* commandList) {
    std::lock_guard<std::mutex> guard(g_lock);
    auto found = g_head_sessions.find(static_cast<ID3D12GraphicsCommandList*>(commandList));
    if (found == g_head_sessions.end()) return 0;
    if (!found->second->fence) return 0;
    const uint64_t completed = found->second->fence->GetCompletedValue();
    // UINT64_MAX means device removal, not successful completion.
    if (completed == UINT64_MAX || completed < found->second->completion) return 0;
    g_head_sessions.erase(found);
    --g_head_diag.active_sessions;
    ++g_head_diag.sessions_released;
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_SealHeadD3D12(
        void* commandList, void* commandQueue, void* completionFence, uint64_t value) {
    if (!commandList || !commandQueue || !completionFence || !value) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    const auto found = g_head_sessions.find(static_cast<ID3D12GraphicsCommandList*>(commandList));
    if (found == g_head_sessions.end() || found->second->sealed || found->second->fence) return 0;
    auto* queue = static_cast<ID3D12CommandQueue*>(commandQueue);
    auto* fence = static_cast<ID3D12Fence*>(completionFence);
    if (queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT || fence->GetCompletedValue() >= value) return 0;
    Microsoft::WRL::ComPtr<ID3D12Device> a, b, c;
    if (FAILED(found->second->list->GetDevice(IID_PPV_ARGS(&a))) || FAILED(queue->GetDevice(IID_PPV_ARGS(&b))) ||
        FAILED(fence->GetDevice(IID_PPV_ARGS(&c))) || a.Get() != b.Get() || a.Get() != c.Get()) return 0;
    found->second->queue = queue; found->second->fence = fence;
    found->second->completion = value; found->second->sealed = true;
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_GetHeadD3D12Diagnostics(NvapiAmdHeadDiagnosticsV1* result) {
    if (!result || result->struct_size != sizeof(*result)) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    *result = g_head_diag;
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_RegisterExternalBuffer(
        void* d3d12Resource, void* sharedHandle, uint64_t byteSize) {
    if (!d3d12Resource || !sharedHandle || !byteSize) return 0;
    auto* resource = static_cast<ID3D12Resource*>(d3d12Resource);
    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.handle = static_cast<HANDLE>(sharedHandle);
    desc.size = byteSize;
    hipExternalMemory_t externalMemory = nullptr;
    if (hipImportExternalMemory(&externalMemory, &desc) != hipSuccess) return 0;
    hipExternalMemoryBufferDesc bufferDesc{};
    bufferDesc.offset = 0;
    bufferDesc.size = byteSize;
    uint8_t* hipBase = nullptr;
    if (hipExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&hipBase),
                                         externalMemory, &bufferDesc) != hipSuccess) {
        hipDestroyExternalMemory(externalMemory);
        return 0;
    }
    ResourceMapping mapping{};
    mapping.resource = resource;
    mapping.d3d12_begin = resource->GetGPUVirtualAddress();
    mapping.byte_size = byteSize;
    mapping.hip_base = hipBase;
    mapping.external_memory = externalMemory;
    std::lock_guard<std::mutex> guard(g_lock);
    EnsureDiagLocked();
    if (g_resources.count(resource)) {
        hipDestroyExternalMemory(externalMemory);
        return 0;
    }
    g_resources[resource] = mapping;
    ++g_diag.resource_registrations;
    ++g_diag.live_resources;
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_RegisterDescriptorResource(
        uint64_t descriptorCpu, void* d3d12Resource) {
    if (!descriptorCpu || !d3d12Resource) return 0;
    std::lock_guard<std::mutex> guard(g_lock);
    g_descriptor_resources[static_cast<SIZE_T>(descriptorCpu)] =
        static_cast<ID3D12Resource*>(d3d12Resource);
    return 1;
}

extern "C" __declspec(dllexport) int __cdecl NvapiAmd_UnregisterExternalBuffer(
        void* d3d12Resource) {
    if (!d3d12Resource) return 0;
    hipExternalMemory_t externalMemory = nullptr;
    {
        std::lock_guard<std::mutex> guard(g_lock);
        auto it = g_resources.find(d3d12Resource);
        if (it == g_resources.end()) return 0;
        externalMemory = it->second.external_memory;
        g_resources.erase(it);
        ++g_diag.resource_unregistrations;
        --g_diag.live_resources;
    }
    return hipDestroyExternalMemory(externalMemory) == hipSuccess ? 1 : 0;
}
