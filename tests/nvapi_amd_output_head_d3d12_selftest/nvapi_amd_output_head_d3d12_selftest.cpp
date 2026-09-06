#define OUTPUT_HEAD_RECORDING_HELPERS_ONLY
#include "../../tools/output_head_surface_d3d12/output_head_recording_selftest.cpp"
#include "../../third_party/nvapi/nvapi.h"
#include "../../tools/nvapi_amd/nvapi_amd_diag.h"
#include "../../tools/nvapi_amd/nvapi_amd_d3d12_head.h"

using Query = void*(__cdecl*)(uint32_t);
using CreateModule = NvAPI_Status(__cdecl*)(ID3D12Device*, const void*, NvU32, NVDX_ObjectHandle*);
using CreateFunction = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle, const char*, NVDX_ObjectHandle*);
using Launch = NvAPI_Status(__cdecl*)(ID3D12GraphicsCommandList*, const NVAPI_CU_KERNEL_LAUNCH_PARAMS*, NvU32);
using Destroy = NvAPI_Status(__cdecl*)(ID3D12Device*, NVDX_ObjectHandle);
using Independent = NvAPI_Status(__cdecl*)(NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS*);
using Merged = NvAPI_Status(__cdecl*)(NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS*);
void Require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

int main(int argc, char** argv) { try {
    Require(argc == 7, "usage: nvapi_amd_output_head_d3d12_selftest <backend.dll> <activation.raw> <model.raw> <params.raw> <recording_reference_directory> <result_directory>");
    auto main = ReadSlice(argv[2], 13873152, output_head_dx12::MainBytes);
    auto skip = ReadSlice(argv[2], 110592, output_head_dx12::SkipBytes);
    auto head = ReadSlice(argv[3], kHeadOffset, kHeadBytes);
    auto blend = ReadSlice(argv[3], 147429376, 4);
    auto captured = Read(argv[4]); Require(captured.size() == 184, "bad parameter size");
    std::array<std::vector<uint8_t>, 4> references;
    for (unsigned i = 0; i < 4; ++i) {
        references[i] = Read(fs::path(argv[5]) / ("frame" + std::to_string(i) + "_rgba16f.raw"));
        Require(references[i].size() == kSurfaceBytes, "bad reference size");
    }
    ComPtr<ID3D12Debug> debug;
    const bool debugEnabled = SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
    if (debugEnabled) debug->EnableDebugLayer();
    char autoValue[8]{};
    const bool autoMode = GetEnvironmentVariableA("MODULE_TRACE_AMD_D3D12_HEAD", autoValue, 8) && strcmp(autoValue, "1") == 0;
    char batchValue[16]{};
    unsigned batches = GetEnvironmentVariableA("DLSSNR_HEAD_TEST_BATCHES", batchValue, 16) ? unsigned(std::stoul(batchValue)) : 2u;
    Require(batches >= 2 && batches <= 250, "batches must be in [2,250]");
    using InitTrace = int(*)(DWORD);
    using RegisterTrace = int(*)(ID3D12Device*);
    using WrapQuery = void*(*)(void*);
    using CollectTrace = int(*)();
    RegisterTrace registerTrace = nullptr; WrapQuery wrapQuery = nullptr; CollectTrace collectTrace = nullptr;
    if (autoMode) {
        auto trace = LoadLibraryW((fs::path(argv[1]).parent_path() / "module_trace.dll").wstring().c_str());
        Require(trace != nullptr, "trace load failed");
        auto init = reinterpret_cast<InitTrace>(GetProcAddress(trace, "ModuleTrace_InitializeAndWait"));
        registerTrace = reinterpret_cast<RegisterTrace>(GetProcAddress(trace, "ModuleTrace_RegisterD3D12Device"));
        wrapQuery = reinterpret_cast<WrapQuery>(GetProcAddress(trace, "ModuleTrace_TestWrapNvapiQuery"));
        collectTrace = reinterpret_cast<CollectTrace>(GetProcAddress(trace, "ModuleTrace_CollectAmdHeads"));
        Require(init && registerTrace && wrapQuery && collectTrace && init(10000), "trace init failed");
    }
    HMODULE library = LoadLibraryW(fs::path(argv[1]).wstring().c_str()); Require(library != nullptr, "backend load failed");
    auto query = reinterpret_cast<Query>(GetProcAddress(library, "nvapi_QueryInterface"));
    if (autoMode) query = reinterpret_cast<Query>(wrapQuery(reinterpret_cast<void*>(query)));
    auto begin = reinterpret_cast<NvapiAmdBeginHeadD3D12_t>(GetProcAddress(library, "NvapiAmd_BeginHeadD3D12"));
    auto release = reinterpret_cast<NvapiAmdReleaseHeadD3D12_t>(GetProcAddress(library, "NvapiAmd_ReleaseHeadD3D12"));
    auto headDiag = reinterpret_cast<NvapiAmdGetHeadD3D12Diagnostics_t>(GetProcAddress(library, "NvapiAmd_GetHeadD3D12Diagnostics"));
    auto diag = reinterpret_cast<NvapiAmdGetDiagnostics_t>(GetProcAddress(library, "NvapiAmd_GetDiagnostics"));
    auto registerDescriptor = reinterpret_cast<NvapiAmdRegisterDescriptorResource_t>(GetProcAddress(library, "NvapiAmd_RegisterDescriptorResource"));
    Require(query && begin && release && headDiag && diag && registerDescriptor, "missing backend exports");
    auto createModule = reinterpret_cast<CreateModule>(query(0xAD1A677D));
    auto createFunction = reinterpret_cast<CreateFunction>(query(0xE2436E22));
    auto launch = reinterpret_cast<Launch>(query(0x24973538));
    auto destroyFunction = reinterpret_cast<Destroy>(query(0xDF295EA6));
    auto destroyModule = reinterpret_cast<Destroy>(query(0x41C65285));
    auto independent = reinterpret_cast<Independent>(query(0x0DDAC234));
    auto merged = reinterpret_cast<Merged>(query(0x329FE6E0));
    Require(createModule && createFunction && launch && destroyFunction && destroyModule && independent && merged, "missing NVAPI interface");
    ComPtr<IDXGIFactory6> factory; HR_CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter; DXGI_ADAPTER_DESC1 ad{}; bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
        HR_CHECK(adapter->GetDesc1(&ad));
        if (!(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && ad.VendorId == 0x1002) { found = true; break; }
    }
    Require(found, "AMD adapter not found");
    ComPtr<ID3D12Device> device; HR_CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    if (autoMode) Require(registerTrace(device.Get()), "trace device registration failed");
    ComPtr<ID3D12InfoQueue> info; device.As(&info);
    auto arena = DefaultBuffer(device.Get(), 27807744), model = DefaultBuffer(device.Get(), 147719680);
    auto base = Texture(device.Get()), output = Texture(device.Get());
    auto uMain = Upload(device.Get(), main), uSkip = Upload(device.Get(), skip), uHead = Upload(device.Get(), head);
    auto uBlend = Upload(device.Get(), blend), uZero = Upload(device.Get(), std::vector<uint8_t>(main.size(), 0));
    std::array<ComPtr<ID3D12Resource>, 4> bases, readbacks;
    for (unsigned f = 0; f < 4; ++f) {
        std::vector<uint8_t> bytes(static_cast<size_t>(kSurfaceBytes), 0);
        if (f == 1) {
            auto* h = reinterpret_cast<uint16_t*>(bytes.data());
            for (size_t i = 0; i < kSurfaceBytes / 2; ++i) h[i] = uint16_t(i % 4 == 3 ? 0x3c00 : ((i / 4) % 2 ? 0x3400 : 0x3800));
        }
        bases[f] = Upload(device.Get(), bytes); readbacks[f] = Readback(device.Get(), kSurfaceBytes);
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 2;
    ComPtr<ID3D12DescriptorHeap> views, samplers; HR_CHECK(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&views)));
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER; hd.NumDescriptors = 1; HR_CHECK(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&samplers)));
    auto outputDescriptor = views->GetCPUDescriptorHandleForHeapStart(), baseDescriptor = outputDescriptor;
    baseDescriptor.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.Format = DXGI_FORMAT_R16G16B16A16_FLOAT; ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(output.Get(), nullptr, &ud, outputDescriptor);
    D3D12_SHADER_RESOURCE_VIEW_DESC sd{}; sd.Format = ud.Format; sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sd.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(base.Get(), &sd, baseDescriptor);
    D3D12_SAMPLER_DESC sampler{}; sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    sampler.MaxLOD = D3D12_FLOAT32_MAX; sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    auto samplerDescriptor = samplers->GetCPUDescriptorHandleForHeapStart(); device->CreateSampler(&sampler, samplerDescriptor);
    if (!autoMode) Require(registerDescriptor(outputDescriptor.ptr, output.Get()) && registerDescriptor(baseDescriptor.ptr, base.Get()), "descriptor registration failed");
    NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_PARAMS surface{};
    surface.structSizeIn = surface.structSizeOut = sizeof(surface); surface.pDevice = device.Get();
    surface.type = NVAPI_D3D12_GET_CUDA_INDEPENDENT_DESCRIPTOR_OBJECT_SURFACE; surface.desc = outputDescriptor;
    Require(independent(&surface) == NVAPI_OK, "surface object failed");
    NVAPI_D3D12_GET_CUDA_MERGED_TEXTURE_SAMPLER_OBJECT_PARAMS texture{};
    texture.structSizeIn = texture.structSizeOut = sizeof(texture); texture.pDevice = device.Get();
    texture.texDesc = baseDescriptor; texture.smpDesc = samplerDescriptor;
    Require(merged(&texture) == NVAPI_OK, "base object failed");
    uint64_t blob = 0x4845414454455354; NVDX_ObjectHandle module = nullptr, function = nullptr, unsupported = nullptr;
    Require(createModule(device.Get(), &blob, sizeof(blob), &module) == NVAPI_OK, "module create failed");
    Require(createFunction(device.Get(), module, "cc_tinlayout_fused_post_block_swin_1h_32_fp8", &function) == NVAPI_OK, "function create failed");
    Require(createFunction(device.Get(), module, "cc_cb_clear", &unsupported) == NVAPI_OK, "control function create failed");
    auto params = captured;
    auto put = [&](size_t offset, uint64_t value) { std::memcpy(params.data() + offset, &value, 8); };
    put(0, arena->GetGPUVirtualAddress() + 13873152); put(8, arena->GetGPUVirtualAddress() + 110592);
    put(16, surface.handle); put(24, model->GetGPUVirtualAddress() + kHeadOffset);
    put(56, texture.textureHandle); put(104, model->GetGPUVirtualAddress() + 147429376);
    NVAPI_CU_KERNEL_LAUNCH_PARAMS call{}; call.hFunction = function; call.gridDim = {81,49,1}; call.blockDim = {32,1,1};
    call.pParams = params.data(); call.paramSize = 184;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue; HR_CHECK(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    ComPtr<ID3D12Fence> fence; HR_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    uint64_t mismatches = 0; unsigned rejected = 0, earlyReleaseRejected = 0, released = 0;
    NvapiAmdHeadResourceV1 resources[] = {{arena.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
        {model.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS}, {base.Get(), D3D12_RESOURCE_STATE_COMMON}, {output.Get(), D3D12_RESOURCE_STATE_COMMON}};
    for (unsigned batch = 0; batch < batches; ++batch) {
        ComPtr<ID3D12CommandAllocator> allocator; ComPtr<ID3D12GraphicsCommandList> list;
        HR_CHECK(device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&allocator)));
        HR_CHECK(device->CreateCommandList(0, qd.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)));
        NvapiAmdHeadSessionV1 config{sizeof(config), 1, list.Get(), queue.Get(), fence.Get(), batch + 1ull, resources, 4};
        if (!autoMode) {
          Require(begin(&config) != 0, "session begin failed");
          Require(begin(&config) == 0, "duplicate session accepted");
          earlyReleaseRejected += release(list.Get()) == 0;
          for (size_t offset : {size_t(0), size_t(16), size_t(32), size_t(88)}) {
            auto corrupted = params; corrupted[offset] ^= 0x10;
            auto bad = call; bad.pParams = corrupted.data();
            rejected += launch(list.Get(), &bad, 1) == NVAPI_INVALID_ARGUMENT;
          }
          auto bad = call; bad.hFunction = unsupported;
          rejected += launch(list.Get(), &bad, 1) == NVAPI_NO_IMPLEMENTATION;
        } else {
          // No observed barriers yet: an automatic session must not guess states.
          rejected += launch(list.Get(), &call, 1) == NVAPI_INVALID_ARGUMENT;
          Recorder::Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
          Recorder::Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        }
        auto copy = [&](ID3D12Resource* target, uint64_t offset, ID3D12Resource* source, uint64_t bytes) {
            Recorder::Transition(list.Get(), target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyBufferRegion(target, offset, source, 0, bytes);
            Recorder::Transition(list.Get(), target, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        };
        copy(arena.Get(), 110592, uSkip.Get(), skip.size()); copy(model.Get(), kHeadOffset, uHead.Get(), head.size());
        copy(model.Get(), 147429376, uBlend.Get(), blend.size());
        for (unsigned frame = 0; frame < 4; ++frame) {
            copy(arena.Get(), 13873152, frame == 2 ? uZero.Get() : uMain.Get(), main.size());
            Recorder::Transition(list.Get(), base.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            auto source = Recorder::LinearLocation(bases[frame].Get()), dest = Recorder::TextureLocation(base.Get());
            list->CopyTextureRegion(&dest, 0, 0, 0, &source, nullptr);
            Recorder::Transition(list.Get(), base.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            Require(launch(list.Get(), &call, 1) == NVAPI_OK, "NVAPI DXIL record failed");
            if (autoMode && frame == 0) {
                earlyReleaseRejected += release(list.Get()) == 0;
                auto changedHistory = params; changedHistory[88] = 1;
                auto bad = call; bad.pParams = changedHistory.data();
                rejected += launch(list.Get(), &bad, 1) == NVAPI_INVALID_ARGUMENT;
                NVAPI_CU_KERNEL_LAUNCH_PARAMS chain[] = {call, call};
                rejected += launch(list.Get(), chain, 2) == NVAPI_INVALID_ARGUMENT;
            }
            Recorder::Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            source = Recorder::TextureLocation(output.Get()); dest = Recorder::LinearLocation(readbacks[frame].Get());
            list->CopyTextureRegion(&dest, 0, 0, 0, &source, nullptr);
            Recorder::Transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        }
        Require(fence->GetCompletedValue() == batch, "recording changed completion fence");
        HR_CHECK(list->Close()); ID3D12CommandList* lists[] = {list.Get()};
        if (autoMode) {
            // Rejection must not seal the first occurrence or poison its later
            // legitimate submission. The hook suppresses this invalid batch.
            ID3D12CommandList* duplicate[] = {list.Get(), list.Get()};
            queue->ExecuteCommandLists(2, duplicate);
        }
        queue->ExecuteCommandLists(1, lists);
        HR_CHECK(queue->Signal(fence.Get(), batch + 1));
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr); Require(event != nullptr, "event failed");
        HR_CHECK(fence->SetEventOnCompletion(batch + 1, event)); auto wait = WaitForSingleObject(event, 60000); CloseHandle(event);
        Require(wait == WAIT_OBJECT_0, "GPU timed out"); HR_CHECK(device->GetDeviceRemovedReason());
        // Completed sessions refuse further recording, without touching the closed list.
        rejected += launch(list.Get(), &call, 1) == NVAPI_INVALID_ARGUMENT;
        for (unsigned frame = 0; frame < 4; ++frame) {
            auto actual = ReadGpu(readbacks[frame].Get(), size_t(kSurfaceBytes));
            for (size_t i = 0; i < actual.size(); ++i) mismatches += actual[i] != references[frame][i];
            if (batch + 1 == batches)
                Write(fs::path(argv[6]) / ("final_frame" + std::to_string(frame) + ".raw"), actual.data(), actual.size());
        }
        if (autoMode) Require(collectTrace() == 0, "automatic retirement incomplete");
        else { released += release(list.Get()) != 0; Require(release(list.Get()) == 0, "duplicate release accepted"); }
        if ((batch + 1) % 25 == 0) { std::printf("Validated %u/%u head calls\n", (batch+1)*4, batches*4); std::fflush(stdout); }
    }
    uint64_t debugErrors = 0;
    if (info) for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T n = 0; info->GetMessage(i, nullptr, &n); std::vector<uint8_t> bytes(n);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(bytes.data()); HR_CHECK(info->GetMessage(i, m, &n));
        if (m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) { ++debugErrors; std::fprintf(stderr, "%s\n", m->pDescription); }
    }
    NvapiAmdHeadDiagnosticsV1 hdg{sizeof(hdg)}; Require(headDiag(&hdg), "head diagnostics failed");
    if (autoMode) released = unsigned(hdg.sessions_released);
    NvapiAmdDiagnosticsV1 dg{}; dg.struct_size = sizeof(dg); Require(diag(&dg), "backend diagnostics failed");
    Require(destroyFunction(device.Get(), unsupported) == NVAPI_OK && destroyFunction(device.Get(), function) == NVAPI_OK && destroyModule(device.Get(), module) == NVAPI_OK, "module cleanup failed");
    bool pass = !mismatches && !debugErrors && rejected == (autoMode ? 4u : 6u) * batches && earlyReleaseRejected == batches && released == batches &&
        hdg.active_sessions == 0 && hdg.sessions_created == batches && hdg.sessions_released == batches && hdg.head_records == 4 * batches && hdg.rejected_records == (autoMode ? 1u : 6u) * batches &&
        dg.gpu_markers == 0 && dg.resource_registrations == 0 && dg.kernels_submitted == 4 * batches && dg.address_translations == 16 * batches;
    std::ostringstream json;
    json << "{\n  \"experiment\": \"nvapi_output_head_d3d12_dispatch\",\n  \"status\": \"" << (pass ? "PASS" : "FAIL") << "\",\n"
         << "  \"head_records\": " << hdg.head_records << ",\n  \"submissions\": " << batches << ",\n  \"surface_byte_mismatches\": " << mismatches << ",\n"
         << "  \"rejected_calls\": " << rejected << ",\n  \"early_release_rejected\": " << earlyReleaseRejected << ",\n"
         << "  \"sessions_released\": " << released << ",\n  \"active_sessions\": " << hdg.active_sessions << ",\n"
         << "  \"debug_layer_enabled\": " << (debugEnabled ? "true" : "false") << ",\n  \"debug_errors\": " << debugErrors << ",\n"
         << "  \"hip_activation_model_imports\": " << dg.resource_registrations << ",\n  \"gpu_markers\": " << dg.gpu_markers << ",\n"
         << "  \"binding_mode\": \"" << (autoMode ? "AUTOMATIC_TRACE_STATES_AND_SUBMISSION_FENCE" : "EXPLICIT_HOST_STATES_AND_FENCE") << "\",\n  \"captured_activation_input\": true,\n  \"game_runtime_ready\": false\n}\n";
    auto result = json.str(); Write(fs::path(argv[6]) / "manifest.json", result.data(), result.size());
    std::printf("%s", result.c_str()); return pass ? 0 : 1;
} catch (const std::exception& error) { std::fprintf(stderr, "ERROR: %s\n", error.what()); return 1; } }
