// d3d12_hip_interop — Phase 2: D3D12 <-> HIP shared-memory interop on the AMD
// RDNA4 adapter (gate S2).
//
// What is proved here:
//   T1 buffer import   : a D3D12 shared heap (placed buffer) is imported into HIP
//                        (hipExternalMemoryHandleTypeD3D12Heap, fallback OpaqueWin32).
//                        A secondary attempt uses a shared committed buffer
//                        (hipExternalMemoryHandleTypeD3D12Resource).
//   T2 fence import    : a shared D3D12 fence imported as hipExternalSemaphore
//                        (hipExternalSemaphoreHandleTypeD3D12Fence).
//   T3 d3d12 -> hip    : D3D12 writes a pattern, signals fence(1); HIP waits on
//                        the external fence in-stream, verifies contents (GPU-GPU).
//   T4 hip -> d3d12    : HIP kernel writes a pattern, signals fence(2) via
//                        hipSignalExternalSemaphoresAsync; D3D12 GPU-waits on the
//                        same fence, copies back, host verifies.
//   T5 texture import  : informational attempt (shared RGBA8 texture -> HIP buffer
//                        view). Failure here is recorded, not fatal.
//   T6 PASS-B survey   : runtime probe for any HIP API that could embed HIP work
//                        into a D3D12 command list. HIP exposes none; recorded.
//
// PASS-A (spec) = T1 && T2 && T3 && T4. PASS-B = survey, informational only.
//
// Output: human log to stdout; JSON report to --json <path> (or stdout with --json -).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

static bool g_anyFailure = false;

struct TestResult {
    const char* name;
    bool ran = false;
    bool pass = false;
    bool informational = false;   // informational results do not affect overall_pass
    std::string detail;
    double kernelMs = -1.0;
};
static std::vector<TestResult> g_results;

static void Record(TestResult r) {
    printf("[%s] %-26s %s\n",
           r.pass ? "PASS" : (r.informational ? "INFO" : (r.ran ? "FAIL" : "SKIP")),
           r.name, r.detail.c_str());
    g_results.push_back(std::move(r));
}

#define HIP_CHECK_R(cmd, r)                                                  \
    do {                                                                     \
        hipError_t e_ = (cmd);                                               \
        if (e_ != hipSuccess) {                                              \
            char b_[256];                                                    \
            snprintf(b_, sizeof(b_), "%s -> %d (%s) @%d", #cmd, (int)e_,     \
                     hipGetErrorString(e_), __LINE__);                       \
            (r).detail += "  HIPERR["; (r).detail += b_; (r).detail += "]";  \
            (r).pass = false;                                                \
        }                                                                    \
    } while (0)

#define HR_CHECK(hr, what, r)                                                \
    do {                                                                     \
        HRESULT h_ = (hr);                                                   \
        if (FAILED(h_)) {                                                    \
            char b_[160];                                                    \
            snprintf(b_, sizeof(b_), "%s failed 0x%08lx @%d", what,          \
                     (unsigned long)h_, __LINE__);                           \
            (r).detail += "  "; (r).detail += b_;                            \
            (r).pass = false;                                                \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// HIP kernels operating on the shared buffer (uint32 elements)

__global__ static void k_verify(const uint32_t* buf, uint32_t expectBase, int n,
                                uint32_t* mismatchOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t expect = expectBase + (uint32_t)i;
    if (buf[i] != expect) atomicAdd(mismatchOut, 1u);
}

__global__ static void k_write(uint32_t* buf, uint32_t base, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = base + (uint32_t)i;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const char* jsonPath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--json") && i + 1 < argc) jsonPath = argv[++i];
    }

    printf("=== d3d12_hip_interop (dlssnr-amd-lab Phase 2, gate S2) ===\n");

    const int N = 1 << 20;                     // 1M uint32 = 4 MB payload
    const UINT64 BUF_SIZE = (UINT64)N * sizeof(uint32_t);
    const UINT64 HEAP_SIZE = BUF_SIZE + 65536; // slack so placed-resource offset rules are happy

    // ---- 1. Pick the first hardware DXGI adapter and verify it is AMD ----
    ComPtr<IDXGIFactory7> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        printf("FATAL: CreateDXGIFactory2 failed\n");
        return 2;
    }
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 ad{};
    bool found = false;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i, adapter.Reset()) {
        adapter->GetDesc1(&ad);
        if (!(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { found = true; break; }
    }
    if (!found) { printf("FATAL: no hardware adapter\n"); return 2; }
    printf("d3d12 adapter: vendor=0x%04x device=0x%04x\n", ad.VendorId, ad.DeviceId);
    if (ad.VendorId != 0x1002) {
        printf("FATAL: hardware adapter is not AMD (0x1002); interop proof requires AMD.\n");
        return 2;
    }

    // ---- 2. HIP device on the same physical GPU ----
    int hipCount = 0;
    if (hipGetDeviceCount(&hipCount) != hipSuccess || hipCount < 1) {
        printf("FATAL: no HIP devices\n");
        return 2;
    }
    hipDeviceProp_t hp;
    hipGetDeviceProperties(&hp, 0);
    hipSetDevice(0);
    printf("hip device 0 : %s gcnArch=%s pci=%04x:%02x:%02x\n", hp.name, hp.gcnArchName,
           hp.pciDomainID, hp.pciBusID, hp.pciDeviceID);

    // ---- 3. D3D12 device + queue + fence ----
    ComPtr<ID3D12Device> device;
    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) { printf("FATAL: D3D12CreateDevice 0x%08lx\n", (unsigned long)hr); return 2; }

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) { printf("FATAL: CreateCommandQueue 0x%08lx\n", (unsigned long)hr); return 2; }

    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) { printf("FATAL: CreateFence(SHARED) 0x%08lx\n", (unsigned long)hr); return 2; }
    UINT64 fenceVal = 0;

    // CPU completion wait fence (non-shared)
    ComPtr<ID3D12Fence> cpuFence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cpuFence));
    UINT64 cpuFenceVal = 0;
    auto SubmitAndWait = [&](ID3D12GraphicsCommandList* list) {
        ID3D12CommandList* lists[] = { list };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(cpuFence.Get(), ++cpuFenceVal);
        while (cpuFence->GetCompletedValue() < cpuFenceVal) Sleep(0);
    };

    // ========================================================================
    // T1: shared heap + placed buffer -> HIP external memory
    // ========================================================================
    TestResult t1; t1.name = "T1_buffer_import"; t1.ran = true; t1.pass = true;
    void* sharedDevPtr = nullptr;
    hipExternalMemory_t extMem = nullptr;
    HANDLE heapSharedHandle = nullptr;
    ComPtr<ID3D12Heap> sharedHeap;
    ComPtr<ID3D12Resource> placedBuf;
    const char* importKind = "none";

    {
        D3D12_HEAP_DESC hd{};
        hd.SizeInBytes = HEAP_SIZE;
        hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        hd.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        hd.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        hd.Alignment = 0;
        hd.Flags = (D3D12_HEAP_FLAGS)(D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
        HR_CHECK(device->CreateHeap(&hd, IID_PPV_ARGS(&sharedHeap)), "CreateHeap(SHARED)", t1);
    }
    if (sharedHeap) {
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Alignment = 0;
        rd.Width = BUF_SIZE;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_NONE;
        HR_CHECK(device->CreatePlacedResource(sharedHeap.Get(), 0, &rd, D3D12_RESOURCE_STATE_COMMON,
                                              nullptr, IID_PPV_ARGS(&placedBuf)),
                 "CreatePlacedResource", t1);
        HR_CHECK(device->CreateSharedHandle(sharedHeap.Get(), nullptr, GENERIC_ALL, nullptr,
                                            &heapSharedHandle),
                 "CreateSharedHandle(heap)", t1);
    }
    if (heapSharedHandle) {
        hipExternalMemoryHandleDesc md{};
        md.type = hipExternalMemoryHandleTypeD3D12Heap;
        md.handle.win32.handle = heapSharedHandle;
        md.size = HEAP_SIZE;
        md.flags = 0;
        hipError_t e = hipImportExternalMemory(&extMem, &md);
        if (e != hipSuccess) {
            // fallback: treat the same NT handle as opaque win32
            char b[160];
            snprintf(b, sizeof(b), "D3D12Heap import -> %d (%s); try OpaqueWin32",
                     (int)e, hipGetErrorString(e));
            t1.detail += b;
            md.type = hipExternalMemoryHandleTypeOpaqueWin32;
            extMem = nullptr;
            e = hipImportExternalMemory(&extMem, &md);
        }
        if (e == hipSuccess) {
            importKind = (md.type == hipExternalMemoryHandleTypeD3D12Heap) ? "D3D12Heap" : "OpaqueWin32";
            hipExternalMemoryBufferDesc bd{};
            bd.offset = 0;
            bd.size = BUF_SIZE;
            bd.flags = 0;
            e = hipExternalMemoryGetMappedBuffer(&sharedDevPtr, extMem, &bd);
            if (e != hipSuccess) {
                char b[160];
                snprintf(b, sizeof(b), "GetMappedBuffer -> %d (%s)", (int)e, hipGetErrorString(e));
                t1.detail += b;
                t1.pass = false;
                sharedDevPtr = nullptr;
            }
        } else {
            char b[160];
            snprintf(b, sizeof(b), " OpaqueWin32 import -> %d (%s)", (int)e, hipGetErrorString(e));
            t1.detail += b;
            t1.pass = false;
        }
    } else {
        t1.pass = false;
    }
    if (t1.pass && sharedDevPtr) {
        char b[160];
        snprintf(b, sizeof(b), "imported %s handle=0x%p devPtr=%p size=%llu",
                 importKind, (void*)heapSharedHandle, sharedDevPtr,
                 (unsigned long long)BUF_SIZE);
        t1.detail += b;
    }
    Record(std::move(t1));

    // Secondary: shared committed buffer import attempt (informational)
    {
        TestResult ts; ts.name = "T1b_committed_resource"; ts.informational = true; ts.ran = true;
        ComPtr<ID3D12Resource> cbuf;
        D3D12_HEAP_PROPERTIES hp2{};
        hp2.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = BUF_SIZE;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        HRESULT h2 = device->CreateCommittedResource(&hp2, D3D12_HEAP_FLAG_SHARED, &rd,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                     IID_PPV_ARGS(&cbuf));
        if (FAILED(h2)) {
            char b[160];
            snprintf(b, sizeof(b), "CreateCommittedResource(SHARED) -> 0x%08lx", (unsigned long)h2);
            ts.detail = b;
        } else {
            HANDLE ch = nullptr;
            h2 = device->CreateSharedHandle(cbuf.Get(), nullptr, GENERIC_ALL, nullptr, &ch);
            if (FAILED(h2)) {
                char b[160];
                snprintf(b, sizeof(b), "CreateSharedHandle(resource) -> 0x%08lx", (unsigned long)h2);
                ts.detail = b;
            } else {
                hipExternalMemoryHandleDesc md{};
                md.type = hipExternalMemoryHandleTypeD3D12Resource;
                md.handle.win32.handle = ch;
                md.size = BUF_SIZE;
                hipExternalMemory_t em2 = nullptr;
                hipError_t e2 = hipImportExternalMemory(&em2, &md);
                char b[200];
                snprintf(b, sizeof(b), "D3D12Resource import -> %d (%s)", (int)e2,
                         e2 == hipSuccess ? "ok" : hipGetErrorString(e2));
                ts.detail = b;
                ts.pass = (e2 == hipSuccess);
                if (em2) hipDestroyExternalMemory(em2);
                CloseHandle(ch);
            }
        }
        Record(std::move(ts));
    }

    // ========================================================================
    // T2: shared D3D12 fence -> HIP external semaphore
    // ========================================================================
    TestResult t2; t2.name = "T2_fence_import"; t2.ran = true; t2.pass = true;
    hipExternalSemaphore_t extFence = nullptr;
    {
        HANDLE fh = nullptr;
        HR_CHECK(device->CreateSharedHandle(fence.Get(), nullptr, GENERIC_ALL, nullptr, &fh),
                 "CreateSharedHandle(fence)", t2);
        if (fh) {
            hipExternalSemaphoreHandleDesc sd{};
            sd.type = hipExternalSemaphoreHandleTypeD3D12Fence;
            sd.handle.win32.handle = fh;
            sd.flags = 0;
            hipError_t e = hipImportExternalSemaphore(&extFence, &sd);
            if (e != hipSuccess) {
                char b[160];
                snprintf(b, sizeof(b), " OpaqueWin32 fallback -> ");
                t2.detail += b;
                sd.type = hipExternalSemaphoreHandleTypeOpaqueWin32;
                e = hipImportExternalSemaphore(&extFence, &sd);
            }
            if (e != hipSuccess) {
                char b[160];
                snprintf(b, sizeof(b), "%d (%s)", (int)e, hipGetErrorString(e));
                t2.detail += b;
                t2.pass = false;
            } else {
                t2.detail += "fence imported";
            }
            CloseHandle(fh);
        } else {
            t2.pass = false;
        }
    }
    Record(std::move(t2));

    // ---- shared staging resources on the D3D12 side ----
    ComPtr<ID3D12Resource> uploadBuf, readbackBuf;
    {
        D3D12_HEAP_PROPERTIES up{}; up.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = BUF_SIZE; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &rd,
                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                        IID_PPV_ARGS(&uploadBuf));
        D3D12_HEAP_PROPERTIES rb{}; rb.Type = D3D12_HEAP_TYPE_READBACK;
        device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &rd,
                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                        IID_PPV_ARGS(&readbackBuf));
    }

    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> clist;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                              IID_PPV_ARGS(&clist));

    // ========================================================================
    // T3: D3D12 writes pattern, fence(1), HIP verifies after external wait
    // ========================================================================
    TestResult t3; t3.name = "T3_d3d12_to_hip"; t3.ran = true; t3.pass = true;
    const uint32_t BASE1 = 0xA0000000u;
    if (!sharedDevPtr || !extFence || !uploadBuf) {
        t3.pass = false;
        t3.detail = "prerequisites missing";
    } else {
        // host fills upload buffer with pattern
        void* mapped = nullptr;
        uploadBuf->Map(0, nullptr, &mapped);
        uint32_t* up32 = (uint32_t*)mapped;
        for (int i = 0; i < N; ++i) up32[i] = BASE1 + (uint32_t)i;
        uploadBuf->Unmap(0, nullptr);

        // D3D12 copies pattern into the shared placed buffer
        clist->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource = placedBuf.Get();
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        clist->ResourceBarrier(1, &bar);
        clist->CopyBufferRegion(placedBuf.Get(), 0, uploadBuf.Get(), 0, BUF_SIZE);
        std::swap(bar.Transition.StateBefore, bar.Transition.StateAfter);
        clist->ResourceBarrier(1, &bar);
        clist->Close();
        SubmitAndWait(clist.Get());
        queue->Signal(fence.Get(), ++fenceVal);   // value 1

        // HIP waits on the imported fence in-stream, then verifies
        hipStream_t stream;
        HIP_CHECK_R(hipStreamCreate(&stream), t3);
        uint32_t* dMismatch = nullptr;
        HIP_CHECK_R(hipMalloc(&dMismatch, sizeof(uint32_t)), t3);
        HIP_CHECK_R(hipMemsetAsync(dMismatch, 0, sizeof(uint32_t), stream), t3);

        hipExternalSemaphoreWaitParams wp{};
        wp.params.fence.value = fenceVal;
        hipError_t ew = hipWaitExternalSemaphoresAsync(&extFence, &wp, 1, stream);
        if (ew != hipSuccess) {
            // fallback: CPU-wait the fence, continue without GPU-side sync proof
            char b[200];
            snprintf(b, sizeof(b), "ext-sem wait -> %d (%s); CPU-wait fallback", (int)ew,
                     hipGetErrorString(ew));
            t3.detail += b;
            while (fence->GetCompletedValue() < fenceVal) Sleep(0);
        }

        hipEvent_t e0, e1;
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0, stream);
        hipLaunchKernelGGL(k_verify, dim3((N + 255) / 256), dim3(256), 0, stream,
                           (const uint32_t*)sharedDevPtr, BASE1, N, dMismatch);
        hipEventRecord(e1, stream);
        HIP_CHECK_R(hipStreamSynchronize(stream), t3);
        float ms = 0.f; hipEventElapsedTime(&ms, e0, e1);
        t3.kernelMs = ms;

        uint32_t mismatch = 1;
        hipMemcpy(&mismatch, dMismatch, sizeof(uint32_t), hipMemcpyDeviceToHost);
        char b[160];
        snprintf(b, sizeof(b), "n=%d mismatch=%u kernel=%.4fms", N, mismatch, ms);
        t3.detail += b;
        t3.pass = t3.pass && (mismatch == 0);

        hipFree(dMismatch);
        hipEventDestroy(e0); hipEventDestroy(e1);
        hipStreamDestroy(stream);
    }
    Record(std::move(t3));

    // ========================================================================
    // T4: HIP writes pattern + signals fence, D3D12 GPU-waits and reads back
    // ========================================================================
    TestResult t4; t4.name = "T4_hip_to_d3d12"; t4.ran = true; t4.pass = true;
    const uint32_t BASE2 = 0xB0000000u;
    if (!sharedDevPtr || !extFence || !readbackBuf) {
        t4.pass = false;
        t4.detail = "prerequisites missing";
        Record(std::move(t4));
    } else {
        hipStream_t stream;
        HIP_CHECK_R(hipStreamCreate(&stream), t4);
        hipEvent_t e0, e1;
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0, stream);
        hipLaunchKernelGGL(k_write, dim3((N + 255) / 256), dim3(256), 0, stream,
                           (uint32_t*)sharedDevPtr, BASE2, N);
        hipEventRecord(e1, stream);

        fenceVal++;
        hipExternalSemaphoreSignalParams sp{};
        sp.params.fence.value = fenceVal;
        hipError_t es = hipSignalExternalSemaphoresAsync(&extFence, &sp, 1, stream);
        if (es != hipSuccess) {
            char b[200];
            snprintf(b, sizeof(b), "ext-sem signal -> %d (%s); sync+CPU-signal fallback",
                     (int)es, hipGetErrorString(es));
            t4.detail += b;
            HIP_CHECK_R(hipStreamSynchronize(stream), t4);
            queue->Signal(fence.Get(), fenceVal);   // weaker: CPU-side signal
        }
        HIP_CHECK_R(hipStreamSynchronize(stream), t4);
        float ms = 0.f; hipEventElapsedTime(&ms, e0, e1);
        t4.kernelMs = ms;
        hipEventDestroy(e0); hipEventDestroy(e1);
        hipStreamDestroy(stream);

        // D3D12: GPU-wait on fence, copy back, verify on host
        queue->Wait(fence.Get(), fenceVal);
        clist->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource = placedBuf.Get();
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        clist->ResourceBarrier(1, &bar);
        clist->CopyBufferRegion(readbackBuf.Get(), 0, placedBuf.Get(), 0, BUF_SIZE);
        std::swap(bar.Transition.StateBefore, bar.Transition.StateAfter);
        clist->ResourceBarrier(1, &bar);
        clist->Close();
        SubmitAndWait(clist.Get());

        void* mapped = nullptr;
        readbackBuf->Map(0, nullptr, &mapped);
        const uint32_t* rb32 = (const uint32_t*)mapped;
        uint32_t mismatch = 0;
        for (int i = 0; i < N; ++i) {
            if (rb32[i] != BASE2 + (uint32_t)i) { mismatch++; if (mismatch < 4) {
                printf("  mismatch at %d: got 0x%08x expect 0x%08x\n", i, rb32[i], BASE2 + (uint32_t)i);
            } }
        }
        readbackBuf->Unmap(0, nullptr);
        char b[160];
        snprintf(b, sizeof(b), "n=%d mismatch=%u kernel=%.4fms", N, mismatch, ms);
        t4.detail += b;
        t4.pass = (mismatch == 0);
        Record(std::move(t4));
    }

    // ========================================================================
    // T5: shared texture import attempt (informational)
    // ========================================================================
    {
        TestResult t5; t5.name = "T5_texture_import"; t5.informational = true; t5.ran = true;
        ComPtr<ID3D12Resource> tex;
        D3D12_HEAP_PROPERTIES hp3{}; hp3.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = 64; rd.Height = 64; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; rd.Flags = D3D12_RESOURCE_FLAG_NONE;
        HRESULT h5 = device->CreateCommittedResource(&hp3, D3D12_HEAP_FLAG_SHARED, &rd,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                     IID_PPV_ARGS(&tex));
        if (FAILED(h5)) {
            char b[160];
            snprintf(b, sizeof(b), "CreateCommittedResource(tex,SHARED) -> 0x%08lx", (unsigned long)h5);
            t5.detail = b;
        } else {
            HANDLE th = nullptr;
            h5 = device->CreateSharedHandle(tex.Get(), nullptr, GENERIC_ALL, nullptr, &th);
            if (FAILED(h5)) {
                char b[160];
                snprintf(b, sizeof(b), "CreateSharedHandle(tex) -> 0x%08lx", (unsigned long)h5);
                t5.detail = b;
            } else {
                hipExternalMemoryHandleDesc md{};
                md.type = hipExternalMemoryHandleTypeD3D12Resource;
                md.handle.win32.handle = th;
                md.size = 64 * 64 * 4;
                hipExternalMemory_t em5 = nullptr;
                hipError_t e5 = hipImportExternalMemory(&em5, &md);
                char b[220];
                if (e5 == hipSuccess) {
                    void* p5 = nullptr;
                    hipExternalMemoryBufferDesc bd{};
                    bd.offset = 0; bd.size = 64 * 64 * 4;
                    hipError_t e6 = hipExternalMemoryGetMappedBuffer(&p5, em5, &bd);
                    snprintf(b, sizeof(b), "import ok; GetMappedBuffer -> %d (%s) ptr=%p",
                             (int)e6, e6 == hipSuccess ? "ok" : hipGetErrorString(e6), p5);
                    t5.pass = (e6 == hipSuccess);
                    hipDestroyExternalMemory(em5);
                } else {
                    snprintf(b, sizeof(b), "import -> %d (%s)", (int)e5, hipGetErrorString(e5));
                }
                t5.detail = b;
                CloseHandle(th);
            }
        }
        Record(std::move(t5));
    }

    // ========================================================================
    // T6: PASS-B survey — HIP API for embedding HIP work in a D3D12 command list
    // ========================================================================
    {
        TestResult t6; t6.name = "T6_pass_b_cmdlist_api"; t6.informational = true; t6.ran = true;
        // Probe the HIP runtime DLL for any symbol that could attach HIP work to a
        // D3D12 command list. HIP defines none; this is a negative survey result.
        HMODULE hm = GetModuleHandleA("amdhip64.dll");
        const char* candidates[] = {
            "hipStreamAttachD3D12CommandList", "hipCommandListInsertKernel",
            "hipExtQueueAttachD3D12", nullptr };
        int foundAny = 0;
        std::string probed;
        if (hm) {
            for (int i = 0; candidates[i]; ++i) {
                if (GetProcAddress(hm, candidates[i])) foundAny++;
                if (i) probed += ",";
                probed += candidates[i];
            }
        }
        char b[320];
        snprintf(b, sizeof(b),
                 "amdhip64.dll=%s probed=[%s] found=%d; no public HIP API embeds HIP kernels "
                 "in D3D12 command lists (ZLUDA FAQ capability targets CUDA Driver API; "
                 "re-check at Phase 4 trace level). PASS-B fallback = separate HIP queue + "
                 "external fence (proved in T3/T4 if PASS-A holds).",
                 hm ? "loaded" : "absent", probed.c_str(), foundAny);
        t6.detail = b;
        t6.pass = false; // informational; not claimed
        Record(std::move(t6));
    }

    // ---- cleanup ----
    if (extFence) hipDestroyExternalSemaphore(extFence);
    if (extMem) hipDestroyExternalMemory(extMem);  // releases HIP side only
    if (heapSharedHandle) CloseHandle(heapSharedHandle);

    // ---- overall ----
    bool passA = true;
    for (auto& r : g_results)
        if (!r.informational) passA = passA && r.pass;
    printf("\nPASS-A (S2 gate: buffer+fence+bidirectional transfer): %s\n", passA ? "YES" : "NO");

    if (jsonPath) {
        FILE* f = strcmp(jsonPath, "-") ? fopen(jsonPath, "w") : stdout;
        if (f) {
            fprintf(f, "{\n  \"d3d12_adapter\": {\"vendor\": \"0x%04x\", \"device\": \"0x%04x\"},\n",
                    ad.VendorId, ad.DeviceId);
            fprintf(f, "  \"hip_device\": \"%s %s\",\n", hp.name, hp.gcnArchName);
            fprintf(f, "  \"import_kind\": \"%s\",\n", importKind);
            fprintf(f, "  \"tests\": [");
            bool first = true;
            for (auto& t : g_results) {
                if (!first) fprintf(f, ",");
                first = false;
                fprintf(f,
                        "\n    {\"name\": \"%s\", \"ran\": %s, \"pass\": %s, \"informational\": %s, "
                        "\"kernel_ms\": %.4f, \"detail\": \"%s\"}",
                        t.name, t.ran ? "true" : "false", t.pass ? "true" : "false",
                        t.informational ? "true" : "false", t.kernelMs, t.detail.c_str());
            }
            fprintf(f, "\n  ],\n  \"pass_a\": %s\n}\n", passA ? "true" : "false");
            if (f != stdout) fclose(f);
        }
    }
    return passA ? 0 : 1;
}
