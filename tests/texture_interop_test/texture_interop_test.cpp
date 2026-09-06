// texture_interop_test — Round 2 Phase G: D3D12 <-> HIP texture interop gate.
//
// For each of RGBA8 / RGBA16F / R32F / RG16F:
//   dir1: D3D12 writes a deterministic byte pattern into a SHARED committed
//         texture -> HIP imports it (hipExternalMemoryHandleTypeD3D12Resource,
//         buffer view) -> HIP kernel verifies byte-exact contents.
//   dir2: HIP kernel modifies the imported bytes in place (XOR 0x5A) ->
//         D3D12 copies the texture back -> host compares against the expected
//         modified pattern.
//
// Reports per format: mismatch_count + max_abs_error for BOTH directions.
// dir2 is swizzle-invariant (any fixed layout permutation cancels); dir1
// additionally proves identity layout. CPU-side sync (SubmitAndWait /
// hipDeviceSynchronize) is used deliberately for determinism.
//
// Interpretation written into the JSON:
//   layout_identity = dir1 byte-exact (HIP sees D3D12 texture as linear)
//   roundtrip       = dir2 byte-exact (data path safe for kernel work even
//                     if the memory layout is swizzled)
//
// Exit 0 only if all four formats pass both directions byte-exact. No PASS is
// hardcoded: import failures or mismatches are reported as-is.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

struct FmtSpec { const char* name; DXGI_FORMAT fmt; int bpp; };
static const FmtSpec kFormats[] = {
    {"RGBA8",   DXGI_FORMAT_R8G8B8A8_UNORM,       4},
    {"RGBA16F", DXGI_FORMAT_R16G16B16A16_FLOAT,   8},
    {"R32F",    DXGI_FORMAT_R32_FLOAT,            4},
    {"RG16F",   DXGI_FORMAT_R16G16_FLOAT,         4},
};

static const int W = 64, H = 64;

__host__ __device__ static inline uint8_t Pat(uint32_t i) { return (uint8_t)(i * 7 + 13); }

__global__ static void k_verify_bytes(const uint8_t* p, uint32_t n,
                                      uint32_t* mismatch, uint32_t* maxErr) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t e = Pat(i);
    uint8_t g = p[i];
    if (g != e) atomicAdd(mismatch, 1u);
    uint32_t d = g > e ? (uint32_t)(g - e) : (uint32_t)(e - g);
    if (d) atomicMax(maxErr, d);
}

__global__ static void k_modify_bytes(uint8_t* p, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] ^= 0x5A;
}

struct FmtResult {
    std::string name;
    std::string status;          // PASS / FAIL / IMPORT_FAIL / CREATE_FAIL
    std::string detail;
    uint32_t d1Mismatch = 0, d1MaxErr = 0;
    uint32_t d2Mismatch = 0, d2MaxErr = 0;
    bool pass = false;
};

int main(int argc, char** argv) {
    const char* jsonPath = nullptr;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--json") && i + 1 < argc) jsonPath = argv[++i];

    printf("=== texture_interop_test (dlssnr-amd-lab Round 2 Phase G gate) ===\n");

    // ---- first hardware adapter (AMD required for the HIP side) ----
    ComPtr<IDXGIFactory7> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        printf("FATAL: CreateDXGIFactory2 failed\n"); return 2;
    }
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 adesc{};
    bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
        adapter->GetDesc1(&adesc);
        if (!(adesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { found = true; break; }
    }
    if (!found) { printf("FATAL: no hardware adapter\n"); return 2; }
    printf("d3d12 adapter: vendor=0x%04x device=0x%04x\n", adesc.VendorId, adesc.DeviceId);

    int hipCount = 0;
    if (hipGetDeviceCount(&hipCount) != hipSuccess || hipCount < 1) {
        printf("FATAL: no HIP devices\n"); return 2;
    }
    hipSetDevice(0);

    ComPtr<ID3D12Device> device;
    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) { printf("FATAL: D3D12CreateDevice 0x%08lx\n", (unsigned long)hr); return 2; }

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12Fence> cpuFence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cpuFence));
    UINT64 cpuFenceVal = 0;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> clist;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                              IID_PPV_ARGS(&clist));
    clist->Close();
    auto SubmitAndWait = [&]() {
        ID3D12CommandList* lists[] = { clist.Get() };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(cpuFence.Get(), ++cpuFenceVal);
        while (cpuFence->GetCompletedValue() < cpuFenceVal) Sleep(0);
    };

    uint32_t* dMismatch = nullptr;
    uint32_t* dMaxErr = nullptr;
    hipMalloc(&dMismatch, sizeof(uint32_t));
    hipMalloc(&dMaxErr, sizeof(uint32_t));

    std::vector<FmtResult> results;
    bool allPass = true;
    bool allRoundtrip = true;

    for (const auto& fmt : kFormats) {
        FmtResult r;
        r.name = fmt.name;
        const uint32_t totalBytes = (uint32_t)(W * H * fmt.bpp);

        // ---- shared texture ----
        ComPtr<ID3D12Resource> tex;
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = W; td.Height = H; td.DepthOrArraySize = 1; td.MipLevels = 1;
        td.Format = fmt.fmt; td.SampleDesc.Count = 1;
        td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; td.Flags = D3D12_RESOURCE_FLAG_NONE;
        hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &td,
                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             IID_PPV_ARGS(&tex));
        if (FAILED(hr)) {
            char b[120]; snprintf(b, sizeof(b), "CreateCommittedResource 0x%08lx", (unsigned long)hr);
            r.status = "CREATE_FAIL"; r.detail = b;
            results.push_back(std::move(r)); allPass = false; continue;
        }

        // ---- upload: fill staging buffer with the byte pattern ----
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT foot{};
        UINT numRows = 0; UINT64 rowBytes = 0, copyBytes = 0;
        device->GetCopyableFootprints(&td, 0, 1, 0, &foot, &numRows, &rowBytes, &copyBytes);

        ComPtr<ID3D12Resource> upBuf, rbBuf;
        D3D12_HEAP_PROPERTIES upHp{}; upHp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = copyBytes; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
        bd.Format = DXGI_FORMAT_UNKNOWN; bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        device->CreateCommittedResource(&upHp, D3D12_HEAP_FLAG_NONE, &bd,
                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                        IID_PPV_ARGS(&upBuf));
        D3D12_HEAP_PROPERTIES rbHp{}; rbHp.Type = D3D12_HEAP_TYPE_READBACK;
        device->CreateCommittedResource(&rbHp, D3D12_HEAP_FLAG_NONE, &bd,
                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                        IID_PPV_ARGS(&rbBuf));

        void* mapped = nullptr;
        upBuf->Map(0, nullptr, &mapped);
        for (UINT row = 0; row < numRows; ++row) {
            uint8_t* dst = (uint8_t*)mapped + foot.Offset + row * foot.Footprint.RowPitch;
            for (int x = 0; x < W * fmt.bpp; ++x) {
                uint32_t li = (row * W * fmt.bpp) + (uint32_t)x;   // linear byte index
                dst[x] = Pat(li);
            }
        }
        upBuf->Unmap(0, nullptr);

        // ---- D3D12 writes the texture ----
        clist->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource = tex.Get();
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        clist->ResourceBarrier(1, &bar);
        D3D12_TEXTURE_COPY_LOCATION dstLoc{}, srcLoc{};
        dstLoc.pResource = tex.Get(); dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = 0;
        srcLoc.pResource = upBuf.Get(); srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLoc.PlacedFootprint = foot;
        clist->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        std::swap(bar.Transition.StateBefore, bar.Transition.StateAfter);
        clist->ResourceBarrier(1, &bar);   // back to COMMON (cross-API handoff state)
        clist->Close();
        SubmitAndWait();

        // ---- HIP import ----
        HANDLE sh = nullptr;
        hr = device->CreateSharedHandle(tex.Get(), nullptr, GENERIC_ALL, nullptr, &sh);
        if (FAILED(hr)) {
            char b[120]; snprintf(b, sizeof(b), "CreateSharedHandle 0x%08lx", (unsigned long)hr);
            r.status = "CREATE_FAIL"; r.detail = b;
            results.push_back(std::move(r)); allPass = false; continue;
        }
        hipExternalMemory_t extMem = nullptr;
        hipExternalMemoryHandleDesc md{};
        md.type = hipExternalMemoryHandleTypeD3D12Resource;
        md.handle.win32.handle = sh;
        md.size = totalBytes;
        hipError_t e = hipImportExternalMemory(&extMem, &md);
        void* devPtr = nullptr;
        if (e == hipSuccess) {
            hipExternalMemoryBufferDesc mbd{};
            mbd.offset = 0; mbd.size = totalBytes;
            e = hipExternalMemoryGetMappedBuffer(&devPtr, extMem, &mbd);
        }
        if (e != hipSuccess || !devPtr) {
            char b[200]; snprintf(b, sizeof(b), "hipImport/GetMappedBuffer -> %d (%s)",
                                  (int)e, hipGetErrorString(e));
            r.status = "IMPORT_FAIL"; r.detail = b;
            if (extMem) hipDestroyExternalMemory(extMem);
            CloseHandle(sh);
            results.push_back(std::move(r)); allPass = false; continue;
        }

        // ---- dir1: HIP verifies the D3D12-written pattern ----
        hipMemset(dMismatch, 0, sizeof(uint32_t));
        hipMemset(dMaxErr, 0, sizeof(uint32_t));
        uint32_t blocks = (totalBytes + 255) / 256;
        hipLaunchKernelGGL(k_verify_bytes, dim3(blocks), dim3(256), 0, nullptr,
                           (const uint8_t*)devPtr, totalBytes, dMismatch, dMaxErr);
        hipDeviceSynchronize();
        hipMemcpy(&r.d1Mismatch, dMismatch, sizeof(uint32_t), hipMemcpyDeviceToHost);
        hipMemcpy(&r.d1MaxErr, dMaxErr, sizeof(uint32_t), hipMemcpyDeviceToHost);

        // ---- dir2: HIP modifies, D3D12 reads back ----
        hipLaunchKernelGGL(k_modify_bytes, dim3(blocks), dim3(256), 0, nullptr,
                           (uint8_t*)devPtr, totalBytes);
        hipDeviceSynchronize();

        clist->Reset(alloc.Get(), nullptr);
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        clist->ResourceBarrier(1, &bar);
        D3D12_TEXTURE_COPY_LOCATION dst2{}, src2{};
        dst2.pResource = rbBuf.Get(); dst2.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst2.PlacedFootprint = foot;
        src2.pResource = tex.Get(); src2.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src2.SubresourceIndex = 0;
        clist->CopyTextureRegion(&dst2, 0, 0, 0, &src2, nullptr);
        std::swap(bar.Transition.StateBefore, bar.Transition.StateAfter);
        clist->ResourceBarrier(1, &bar);
        clist->Close();
        SubmitAndWait();

        rbBuf->Map(0, nullptr, &mapped);
        r.d2Mismatch = 0; r.d2MaxErr = 0;
        for (UINT row = 0; row < numRows; ++row) {
            const uint8_t* src = (const uint8_t*)mapped + foot.Offset + row * foot.Footprint.RowPitch;
            for (int x = 0; x < W * fmt.bpp; ++x) {
                uint32_t li = row * W * fmt.bpp + (uint32_t)x;
                uint8_t expect = (uint8_t)(Pat(li) ^ 0x5A);
                uint8_t got = src[x];
                if (got != expect) {
                    r.d2Mismatch++;
                    uint32_t d = got > expect ? got - expect : expect - got;
                    if (d > r.d2MaxErr) r.d2MaxErr = d;
                }
            }
        }
        rbBuf->Unmap(0, nullptr);

        hipDestroyExternalMemory(extMem);
        CloseHandle(sh);

        r.pass = (r.d1Mismatch == 0 && r.d2Mismatch == 0);
        r.status = r.pass ? "PASS" : (r.d2Mismatch == 0 ? "ROUNDTRIP_ONLY" : "FAIL");
        char b[260];
        snprintf(b, sizeof(b),
                 "dir1 mismatch=%u max_abs_error=%u; dir2 mismatch=%u max_abs_error=%u",
                 r.d1Mismatch, r.d1MaxErr, r.d2Mismatch, r.d2MaxErr);
        r.detail = b;
        printf("[%s] %-8s %s\n",
               r.pass ? "PASS" : (r.d2Mismatch == 0 ? "RTOK" : "FAIL"),
               r.name.c_str(), r.detail.c_str());
        if (!r.pass) allPass = false;
        if (r.d2Mismatch != 0) allRoundtrip = false;
        results.push_back(std::move(r));
    }

    hipFree(dMismatch);
    hipFree(dMaxErr);

    printf("\ntexture interop gate: %s\n", allPass ? "PASS (all formats byte-exact both ways)"
        : (allRoundtrip ? "PARTIAL: roundtrip byte-exact, but layout is NOT identity "
                          "(dir1 mismatch => AMD texture memory is swizzled; HIP kernels "
                          "must use swizzle-aware addressing on raw buffer views)"
                        : "NOT PASSED (see per-format detail)"));

    if (jsonPath) {
        FILE* f = strcmp(jsonPath, "-") ? fopen(jsonPath, "w") : stdout;
        if (f) {
            fprintf(f, "{\n  \"w\": %d, \"h\": %d,\n  \"formats\": [", W, H);
            bool first = true;
            for (auto& t : results) {
                if (!first) fprintf(f, ",");
                first = false;
                fprintf(f,
                        "\n    {\"format\": \"%s\", \"status\": \"%s\", "
                        "\"dir1_mismatch_count\": %u, \"dir1_max_abs_error\": %u, "
                        "\"dir2_mismatch_count\": %u, \"dir2_max_abs_error\": %u, "
                        "\"pass\": %s, \"detail\": \"%s\"}",
                        t.name.c_str(), t.status.c_str(), t.d1Mismatch, t.d1MaxErr,
                        t.d2Mismatch, t.d2MaxErr, t.pass ? "true" : "false", t.detail.c_str());
            }
            fprintf(f, "\n  ],\n  \"gate_pass\": %s,\n  \"roundtrip_pass\": %s,\n"
                       "  \"layout_identity\": %s\n}\n",
                    allPass ? "true" : "false", allRoundtrip ? "true" : "false",
                    allPass ? "true" : "false");
            if (f != stdout) fclose(f);
        }
    }
    return allPass ? 0 : 1;
}
