# ARCHITECTURE.md

## System overview

```
                +-----------------------------+
                | nr_host.exe (D3D12 x64)     |  genuine DLSS/DLAA contract:
                |  official-ABI NGX pipeline  |  Color + Depth + MV + Exposure
                +--------------+--------------+
                               |
            +------------------+------------------+
            |                                     |
   NVIDIA path (reference)                AMD path (experiment)
            |                                     |
    nvngx.dll -> dlssnr.dll                module_trace + nvapi_trace shim
            |                                     |
   NVAPI CUBIN / CUDA Driver API          nvapi_amd / ZLUDA compatibility layer
            |                                     |
   NVIDIA driver dispatch                  HIP -> RDNA4 dispatch (gfx1200/1201)
```

## Components

### tools/hip_probe
Standalone HIP test suite. No DLSS involvement. Gate S1.
Tests: device count, alloc, FP32, FP16 matmul, FP8 E4M3/E5M2, WMMA (guarded by
`__gfx12__` / WMMA intrinsic availability at compile time). Emits JSON.

### tools/d3d12_hip_interop
Two separate experiments:
- **PASS-A**: independent HIP queue + shared fence/semaphore synchronization around a
  D3D12-created shareable buffer (NT handle export -> `hipImportExternalMemory`).
- **PASS-B**: investigate enqueueing HIP work inside a D3D12 command list context
  (ZLUDA FAQ claim — treated as unverified).

### tools/d3d12_residual_bridge
Production-shape bridge for the operator reconstruction route. It stages a
`640x360` D3D12 RGBA16F texture into a shared linear heap, waits through an
imported D3D12 fence, maps the output-head's `81x49x64x4` logical residual to
pixels in HIP, computes `clamp(base + 0.25*residual, 0, 1)`, signals the fence,
and copies the result back to a D3D12 texture. This explicit staging is required
because direct raw views of AMD D3D12 textures are tiled rather than linear.

### tools/output_head_*
Clean-room AMD-native implementation of slot 154's output head: activation
fusion, 256 FP8 projection/attention MMAs, stable softmax, 16 FP16 tail MMAs,
8x8 spatial shuffle and RGBA16F store. `scripts/run_output_head_pipeline.ps1`
runs its nine validated stages in dependency order. RTX traces are validation
oracles only and are not runtime inputs to the production math path.

`output_head_resident` executes the same graph with fourteen launches in one
process and retains every intermediate on the GPU. `output_head_resident_d3d12`
adds a shared D3D12 heap and imported-fence wait/signal around a production-shape
RGBA16F texture. Both match the modular residual bitwise; the D3D12 variant also
matches all texture components after readback. These are output-head integration
harnesses, not yet the feature-18 resource-registry hook.

### tools/full_graph_d3d12

Native same-command-stream migration of the 156-slot network. Slot 0 currently
implements the captured synchronization clear. The resolution adapter around
the graph accepts any nonzero D3D12 texture size up to the API dimension limit.
Operators may consume an eight-pixel-aligned dynamic working shape; operators
that retain the captured shape use overlapping 640x360 output tiles backed by
640x384 work tensors. Midpoint ownership makes assembly single-writer and exact
in extent. A 16-tile resident limit batches large frames, keeping scratch memory
bounded. WARP tests validate half-bit-exact pack/unpack through 4K. The test-only
identity stage must be replaced by slots 1-154; resolution coverage alone is not
neural reconstruction evidence.

### tools/binary_probe
Read-only PE + CUDA-payload inspector. Enumerates imports, sections, fatbin containers,
PTX/cubin entries, SM targets, kernel names, weights regions. Outputs
`binary_manifest.json` (metadata only: size/hash/magic — never binary content).

### tools/module_trace
Injected or launcher-based tracing of LoadLibraryA/W/ExA/ExW, GetProcAddress and
FreeLibrary to capture module load order (nvapi64, nvcuda, nvngx*, etc.).
Round 2 v2: synchronous `ModuleTrace_InitializeAndWait` gate (host waits for
hooks_installed=true), immediate IAT patching of newly loaded modules, best-effort
ntdll module-load notification. Readiness gated by tests/module_trace_selftest.

### tools/nvapi_trace
`nvapi64.dll` shim exporting `nvapi_QueryInterface`; decodes ordinals to function
names, logs every call; binary payload pointers reduced to a readability-checked
magic peek. Round 2: the forwarding trampoline was re-proven byte-exact for 1..10
args (tests/nvapi_trampoline_test) after being rewritten as a log-then-pure-jmp
passthrough; return values are no longer logged (passthrough correctness first).

### tools/nr_host
Round 2 genuine NGX host: built against the OFFICIAL ABI (third_party/nvidia-dlss
headers + nvsdk_ngx_d.lib; private/inferred ABI strictly isolated in
private_ngx_compat.h). Pipeline: device → Init/Init_with_ProjectID → official
parameter allocation → DLSS/DLAA create (DLSS5-Feeder DLAA contract) → command-list
recorded EvaluateFeature + ExecuteCommandLists → readback → ReleaseFeature →
Shutdown1, every NGX call SEH-wrapped, every step machine-readable (--json).
Flags: `--trace --frames N --input --output --width --height --json --force-load`.

### tools/ngx_abi_probe (Round 2)
ABI regression tests: compile-time static_asserts over the official headers
(Result enum, struct sizes, function-pointer prototypes, API version macro) plus a
runtime export-table check of the DLL at DLSS_DLL_PATH.

### tools/amd_graph_replay
Track-B AMD transport scaffold. It consumes the R-21 structured maps, rebuilds a
9-module/96-function synthetic handle registry, submits the captured 156-slot
launch envelope on `gfx1201`, uploads every captured parameter block and validates
it with GPU-side hashes. Its HIP kernel is lab-authored and its JSON hard-codes
`LAB_TRANSPORT_ONLY`, `counts_as_s6=false`; it is not a neural implementation.

### tools/nvapi_amd
Drop-in implementation of the five official NVIDIA R610 D3D12 CuModule APIs
observed in R-19..R-21 plus the merged texture/sampler and independent descriptor
object calls. It owns module/function handles, validates chain calls, maps
registered D3D12 GPU-VA intervals to imported HIP pointers, and dispatches AMD
kernels. The current tests reconstruct 9/96/156 lifetimes, execute the exact
`cc_cb_clear` boundary contract, and lower the decoded `cg2r_copy_kernel` ABI to
a native D3D12 compute dispatch. Neural dispatch remains unimplemented.

### tools/zluda_ptx_probe + scripts/extract_ptx_entry.py
Loads ZLUDA's `nvcuda.dll` explicitly on RX 9070 XT, creates a real context,
loads one extracted PTX function, resolves it and optionally launches/readbacks
the clear contract. Function isolation prevents an unused unsupported entry from
invalidating an otherwise translatable kernel. Translator trace logs preserve
the exact unsupported statement for each failing function.

### tests/ (Round 2)
- module_trace_selftest: exe + 2 fixture DLLs asserting the full traced chain.
- nvapi_trampoline_test: 1..10-arg byte-exact trampoline self-proof.
- texture_interop_test: RGBA8/RGBA16F/R32F/RG16F D3D12↔HIP texture gate
  (mismatch_count + max_abs_error per direction).

## Key design rules

1. Proprietary binaries referenced via env vars only; never copied into the repo.
2. Every GPU-success claim carries adapter proof (vendor 0x1002, device id, LUID).
3. Numeric validation uses defined tolerances (FP8/FP16 legal deviation), never bit-exact
   requirement, never hardcoded outputs.
4. Fault tolerance for closed-source addons: SEH wrapping, discard faulted command lists,
   re-init after repeated failures (pattern from DLSS5-Feeder).
5. PUBLIC vs PRIVATE ABI separation: nr_host uses only official NGX headers; anything
   inferred (e.g. feature 18) lives in tools/nr_host/private_ngx_compat.h and every
   artifact touching it is labeled PRIVATE_ABI.
6. Transport markers, placeholder kernels and reconstructed boundary operations
   never count as S6; only a validated DLSSNR-derived neural operation can advance
   that gate.
