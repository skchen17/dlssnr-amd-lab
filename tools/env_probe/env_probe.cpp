// env_probe: record DXGI adapters (vendor/device/LUID/VRAM) and D3D12 feature level
// of the AMD compute adapter. Phase 0 companion to scripts/collect_environment.ps1.
//
// Output: JSON to stdout (--json). Exit code 0 on success.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdio>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

static std::string WideToUtf8Print(const std::wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

int main() {
    // ---- DXGI adapter enumeration ----
    ComPtr<IDXGIFactory7> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        printf("{\"error\": \"CreateDXGIFactory2 failed\"}\n");
        return 1;
    }

    printf("{\"dxgi_adapters\": [");
    ComPtr<IDXGIAdapter1> adapter;
    bool first = true;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 d{};
        adapter->GetDesc1(&d);
        if (!first) printf(",");
        first = false;
        printf("\n  {\"index\": %u, \"description\": \"%s\", \"vendor\": \"0x%04x\", \"device\": \"0x%04x\", "
               "\"subsys\": \"0x%08x\", \"revision\": \"0x%04x\", "
               "\"luid_low\": %lu, \"luid_high\": %ld, "
               "\"dedicated_vram\": %llu, \"flags_softwared_only\": %s}",
               i, WideToUtf8Print(d.Description).c_str(), d.VendorId, d.DeviceId,
               d.SubSysId, d.Revision,
               d.AdapterLuid.LowPart, d.AdapterLuid.HighPart,
               (unsigned long long)d.DedicatedVideoMemory,
               (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ? "true" : "false");
    }
    printf("],\n");

    // ---- D3D12 device + feature level on the first hardware (non-software) adapter ----
    printf("\"d3d12\": {");
    ComPtr<IDXGIAdapter1> hwAdapter;
    int hwIndex = -1;
    for (UINT i = 0; factory->EnumAdapters1(i, &hwAdapter) != DXGI_ERROR_NOT_FOUND; ++i, hwAdapter.Reset()) {
        DXGI_ADAPTER_DESC1 d{};
        hwAdapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { hwIndex = (int)i; break; }
    }
    if (hwIndex < 0) {
        printf("\"error\": \"no hardware adapter\"}\n}\n");
        return 1;
    }

    ComPtr<ID3D12Device> device;
    D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1,
                                  D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_11_1};
    HRESULT hr = D3D12CreateDevice(hwAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        printf("\"error\": \"D3D12CreateDevice failed 0x%08lx\"}\n}\n", (unsigned long)hr);
        return 1;
    }

    D3D12_FEATURE_DATA_FEATURE_LEVELS fl{};
    fl.NumFeatureLevels = _countof(levels);
    fl.pFeatureLevelsRequested = levels;
    device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &fl, sizeof(fl));

    // Shader model support
    D3D12_FEATURE_DATA_SHADER_MODEL sm{};
    sm.HighestShaderModel = D3D_SHADER_MODEL_6_9;
    while (sm.HighestShaderModel > D3D_SHADER_MODEL_5_1 &&
           FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm)))) {
        sm.HighestShaderModel = (D3D_SHADER_MODEL)((int)sm.HighestShaderModel - 1);
    }

    // Options relevant to interop / neural paths
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts));
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts1{};
    bool hasOpts1 = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts1, sizeof(opts1)));
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 opts9{};
    bool hasOpts9 = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &opts9, sizeof(opts9)));

    DXGI_ADAPTER_DESC1 hd{};
    hwAdapter->GetDesc1(&hd);
    printf("\"adapter_index\": %d, \"vendor\": \"0x%04x\", \"device\": \"0x%04x\", "
           "\"luid_low\": %lu, \"luid_high\": %ld, ",
           hwIndex, hd.VendorId, hd.DeviceId, hd.AdapterLuid.LowPart, hd.AdapterLuid.HighPart);
    printf("\"max_feature_level\": \"0x%x\", ", fl.MaxSupportedFeatureLevel);
    printf("\"highest_shader_model\": \"0x%x\", ", (int)sm.HighestShaderModel);
    printf("\"wave_ops_supported\": %s, ", (hasOpts1 && opts1.WaveOps) ? "true" : "false");
    if (hasOpts9) {
        printf("\"wave_mma_tier\": %d, ", (int)opts9.WaveMMATier);
    }
    printf("\"resource_heap_tier\": %d, \"cross_adapter\": %s}\n}\n",
           (int)opts.ResourceHeapTier, opts.CrossAdapterRowMajorTextureSupported ? "true" : "false");
    return 0;
}
