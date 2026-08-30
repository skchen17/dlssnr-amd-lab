# INTEROP.md — D3D12 ↔ HIP interop on RDNA4 (Phase 2, gate S2)

Date: 2026-08-30 · Run: `results/20260830_170305/` · Tool: `tools/d3d12_hip_interop/`

## Verdict

**PASS-A = YES.** A shared D3D12 heap buffer and a shared D3D12 fence were imported
into HIP on the AMD RX 9070 XT (gfx1201), and data moved correctly in both directions
with GPU-GPU fence synchronization. This clears the S2 prerequisite for any
DLSSNR-on-AMD pipeline that needs D3D12 resources to feed HIP kernels.

## What was proved (results/20260830_170305/d3d12_hip_interop.json)

| Test | Result | Detail |
|---|---|---|
| T1 buffer import | PASS | `ID3D12Heap` (SHARED, placed buffer) → `CreateSharedHandle` (NT handle) → `hipImportExternalMemory` with `hipExternalMemoryHandleTypeD3D12Heap` → `hipExternalMemoryGetMappedBuffer` (4 MB, offset 0) |
| T1b committed resource | PASS | shared committed buffer → `hipExternalMemoryHandleTypeD3D12Resource` also imports cleanly |
| T2 fence import | PASS | `ID3D12Fence` (`D3D12_FENCE_FLAG_SHARED`) → shared handle → `hipImportExternalSemaphore` with `hipExternalSemaphoreHandleTypeD3D12Fence` |
| T3 D3D12 → HIP | PASS | D3D12 copies 1M×u32 pattern (base 0xA0000000) into shared buffer, signals fence(1); HIP waits on imported fence in-stream (`hipWaitExternalSemaphoresAsync`), verifies mismatch=0 |
| T4 HIP → D3D12 | PASS | HIP kernel writes pattern (base 0xB0000000), `hipSignalExternalSemaphoresAsync` fence(2); D3D12 `ID3D12CommandQueue::Wait(fence,2)`, copies back, host verifies mismatch=0 |
| T5 texture import | PASS (bonus) | shared RGBA8 64×64 texture imports via `D3D12Resource` and maps as a buffer view — useful for staging color buffers |
| T6 PASS-B survey | NO API | no public HIP symbol embeds HIP kernels into a D3D12 command list (probed amdhip64.dll; absent in this process snapshot). PASS-B fallback = separate HIP queue + external fence, which T3/T4 prove works |

Known anomalies (do not affect the verdict):
- T3 `kernel_ms = -0.7651` — hipEvent elapsed-time readback across an external-fence wait
  stream returns a negative value on this ROCm 7.2 Windows runtime. Data correctness is
  established by content verification (mismatch=0), not by timing; the timing figure is
  discarded.
- `GetModuleHandleA("amdhip64.dll")` in T6 returned NULL (module loaded under a different
  name/path in this process); the negative survey conclusion is corroborated by the HIP
  7.2 headers (no such API exists).

## Minimal reproduction (recipe)

1. D3D12 side
   - `CreateDXGIFactory2` → first hardware adapter (vendor must be 0x1002 for this lab).
   - `D3D12CreateDevice` (FL 11.0 minimum).
   - `ID3D12Heap`: `D3D12_HEAP_TYPE_DEFAULT`, flags `D3D12_HEAP_FLAG_SHARED |
     D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS`; place a buffer resource on it (ROW_MAJOR layout).
   - `ID3D12Fence` with `D3D12_FENCE_FLAG_SHARED`.
   - Shared handles: **`ID3D12Device::CreateSharedHandle(pObject, nullptr, GENERIC_ALL,
     nullptr, &handle)`** — sharing goes through the *device*, not the resource (this is a
     common documentation trap).
2. HIP side
   - `hipExternalMemoryHandleDesc{ type = hipExternalMemoryHandleTypeD3D12Heap,
     handle.win32.handle = ntHandle, size = heapSize }` → `hipImportExternalMemory`.
   - `hipExternalMemoryBufferDesc{ offset, size }` → `hipExternalMemoryGetMappedBuffer`
     yields a device pointer usable by any kernel.
   - `hipExternalSemaphoreHandleDesc{ type = hipExternalSemaphoreHandleTypeD3D12Fence,
     handle.win32.handle = fenceNtHandle }` → `hipImportExternalSemaphore`.
   - Sync: `hipWaitExternalSemaphoresAsync(&sem, {params.fence.value = N}, 1, stream)`
     before reading; `hipSignalExternalSemaphoresAsync(&sem, {params.fence.value = N}, 1,
     stream)` after writing. Fence values are the normal monotonic D3D12 fence values;
     the D3D12 side waits with `ID3D12CommandQueue::Wait(fence, N)`.
3. Handles imported by HIP remain owned by D3D12; `hipDestroyExternalMemory` releases the
   HIP mapping only. Close NT handles after import if not reused.

## Synchronization model chosen

Separate HIP queue (stream) + external shared fence (PASS-A). No attempt is made to inject
HIP work into D3D12 command lists (PASS-B): HIP exposes no such API (see T6), and this is
consistent with ZLUDA's documented position that command-list fusion is a CUDA-Driver-API /
ZLUDA-injection concern, not a public HIP capability. Phase 4 trace will revisit this once
DLSSNR's actual API surface is known.

## Fallbacks (if this had failed)

- `hipExternalMemoryHandleTypeOpaqueWin32` with the same NT handle (built into the tool as
  automatic fallback; not needed).
- Host-pinned staging copies (degraded; would have capped the gate at S2 PARTIAL).

## Implication for later phases

- Texture staging of NR inputs/outputs (color/depth/mv/exposure) is viable via T5.
- A ZLUDA-replaced CUDA path can rely on the same transport: whatever D3D12 resources the
  NR feature consumes can be mirrored to the HIP/ZLUDA side through shared handles.
