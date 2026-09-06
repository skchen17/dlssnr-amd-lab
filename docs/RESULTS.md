# RESULTS.md

Experiment results registry. One section per completed experiment.

Template:

```
## R-<n> <title>
- Date / run id:
- Command:
- Input spec: (dimensions, formats, seed/hash)
- Metrics: max abs err / mean abs err / RMSE / PSNR / SSIM / timing
- Verdict: PASS / FAIL / PARTIAL
- Raw data: results/<ts>/...
```

## Numerical tolerance policy

- FP32: max abs err <= 1e-5 relative to reference magnitude.
- FP16: max abs err <= 1e-2; RMSE tracked separately.
- FP8 (E4M3/E5M2): compare against FP16 upcast reference with documented tolerance
  per test; NaN/Inf = automatic FAIL.
- Bit-exact is NOT required (legitimate arch/precision differences); semantic agreement
  IS required (no passthrough, no NaN, structure preserved).

## R-0 environment capture (S0)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `powershell -File scripts\collect_environment.ps1`
- Input spec: n/a (host enumeration only)
- Metrics: RX 9070 XT vendor=0x1002 device=0x7550, driver 32.0.31041.1004, Win11 26200.9168,
  HIP SDK 7.2.0 (_rocm_sdk_core), MSVC 14.50.35717, no CMake, no NVIDIA GPU, no proprietary DLLs.
- Verdict: PASS
- Raw data: results/20260830_170305/environment.json

## R-1 hip_probe RDNA4 compute baseline (S1)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\build_all.ps1 -Only hip_probe` then `build\hip_probe.exe --json results\20260830_170305\hip_probe.json`
- Input spec: deterministic synthetic data (sinf/cosf fp32 vectors N=2^20; LCG seed 1234 fp16 GEMM 64x64;
  LCG seeds 42/777 fp8 N=4096); CPU references computed in fp64/fp32 on host.
- Metrics:
  - A device enum: gfx1201, 1 device, 17.10 GB
  - B alloc roundtrip: exact
  - C fp32 vadd: maxErr=5.96e-08, kernel=0.064 ms
  - D fp16 GEMM: maxErr=3.26e-03, rmse=5.37e-04, kernel=1.650 ms
  - E fp8 e4m3/e5m2: quantMismatch=0, dotRelErr=0.0 (exact vs host conversion logic)
  - F WMMA: UNVERIFIED — not compiled in default build; `-DWMMA_TEST=1` repro hits backend
    "Cannot select %llvm.amdgcn.wmma.f32.16x16x16.f16" (see BLOCKERS S-B4, wmma_diag.log)
- Verdict: PASS (overall_pass=true; S1 judged on A–E per spec "if intrinsics available" clause)
- Raw data: results/20260830_170305/hip_probe.json, wmma_diag.log

## R-2 d3d12_hip_interop shared-resource gate (S2)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\build_all.ps1 -Only d3d12_hip_interop` then
  `build\d3d12_hip_interop.exe --json results\20260830_170305\d3d12_hip_interop.json`
- Input spec: 1M×uint32 deterministic patterns (base 0xA0000000 D3D12→HIP, base 0xB0000000
  HIP→D3D12) through a 4 MB shared placed buffer on a shared D3D12 heap; shared D3D12
  fence for GPU-GPU ordering; shared 64×64 RGBA8 texture import attempt.
- Metrics:
  - T1 buffer import via hipExternalMemoryHandleTypeD3D12Heap: ok (devPtr mapped)
  - T1b committed-resource import via D3D12Resource: ok
  - T2 fence import via hipExternalSemaphoreHandleTypeD3D12Fence: ok
  - T3 D3D12→HIP: mismatch=0 (in-stream external-fence wait worked)
  - T4 HIP→D3D12: mismatch=0 (external-fence signal + ID3D12CommandQueue::Wait worked)
  - T5 texture import + buffer map: ok
  - T6 PASS-B survey: no public HIP API for D3D12 command-list fusion (informational)
  - Anomaly: T3 hipEvent elapsed time negative (-0.765 ms) on this ROCm 7.2 Windows
    runtime; timing discarded, correctness rests on content verification.
- Verdict: PASS (pass_a=true → S2 achieved)
- Raw data: results/20260830_170305/d3d12_hip_interop.json, interop_build.log; see docs/INTEROP.md

## R-3 binary_probe tool self-test (Phase 3 tooling delivery)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\build_all.ps1 -Only binary_probe`; then
  `build\binary_probe.exe <dll> --json <manifest>`
- Input spec: two legal non-proprietary PE files: C:\Windows\System32\dxgi.dll (1.29 MB),
  amdhip64_7.dll (14.79 MB).
- Metrics: PE header/sections/imports + delay-imports parsed correctly (36 delay entries
  decoded on dxgi.dll); 554 ELF headers detected inside amdhip64_7.dll; PTX marker scan
  active; entropy-run detector active; SHA-256 computed; MISSING_PREREQUISITE path returns
  exit 3 with manifest stub.
- Verdict: PASS (tool delivered; nvngx_dlssnr.dll analysis remains MISSING_PREREQUISITE, see B-1)
- Raw data: results/20260830_170305/selftest_dxgi.json, selftest_amdhip.json

## R-4 nr_host skeleton delivery + deterministic-input check (Phase 5)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\build_all.ps1 -Only nr_host`; then
  `build\nr_host.exe --frames 1 --width 512 --height 512 --trace --json results\20260830_170305\nr_host.json`
  (wrappers: `scripts\run_amd.ps1`, `scripts\run_reference.ps1`)
- Input spec: deterministic synthetic Color RGBA8 / Depth R32F / Motion RG16S, 512×512,
  LCG seed 0x2545F491; exposure 0.18.
- Metrics:
  - build: PASS (nr_host.exe, see results/20260830_170305/nr_host_build.log)
  - status machine: `BLOCKED_MISSING_PREREQUISITE` + exit 3 with DLSS_DLL_PATH unset
    (nr_host_run.log, nr_host.json) — no fabricated NGX execution
  - reference package emitted: results/20260830_180553_reference_package\
    (color_rgba8.bin 1,048,576 B, depth_r32f.bin 1,048,576 B, motion_rg16s.bin 1,048,576 B,
    manifest.json with formats + NVIDIA-machine instructions)
  - determinism: two independent runs produce identical color SHA-256
    CD71008B8F88F22E63494C3B3F097C0618CEE431062B02D42D66E7EE23460043 (scripts\_determinism_check.ps1)
- Verdict: PASS (skeleton delivered; real NGX runs remain blocked per B-1/B-3/B-4)
- Raw data: results/20260830_170305/nr_host_run.log, nr_host.json, nr_host_build.log;
  results/20260830_180553_reference_package/

## R-5 Initial nvngx_dlssnr.dll / nvngx_dlss.dll static scan (SUPERSEDED in part by R-24)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\_probe_now.ps1` (binary_probe on both DLLs); `scripts\_carve_cubin.ps1`
  + `scripts\_readobj_cubin.ps1` (llvm-objdump on carved CUBINs); `scripts\_scan_sm_targets.ps1`
- Input spec: user-supplied legally owned nvngx_dlssnr.dll 310.8.0.0 (165.8 MB) and
  nvngx_dlss.dll 310.7.0.0 (59.0 MB); ledgered in docs/PROPRIETARY_FILES.md.
- Metrics/findings:
  - NR payload: 15 CUBIN ELF representations (e_machine=190) inside .rsrc (147 MB);
    this run found zero *plaintext* PTX markers
  - Target arch: **sm_120 only** (15 markers, one per module) — Blackwell-only, no fallback
  - Kernel names + per-kernel .nv.info parameter metadata fully preserved
    (fused Swin-Transformer backbone, FP16+FP8 sibling variants)
  - No nvcuda.dll / nvapi64.dll imports (static or delay) in either DLL
  - SR runtime: 670 ELFs + PTX 8.7 for sm_89 (contrast case)
- Historical verdict: GATE-0 PASS. The GATE-1 FAIL / PTX-blocked conclusion is
  **superseded by R-24**: decompression proves PTX 9.4 in all 15 containers.
- Raw data: results/20260830_170305/probe_dlssnr_stdout.log, probe_dlss_stdout.log,
  binary_manifest_dlssnr.json, binary_manifest_dlss.json, carved_cubin_{0,1}.readobj.log,
  sm_targets_scan.log; docs/BINARY_ANALYSIS.md

## R-6 first real load of the NVIDIA runtime on AMD
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\_forcload_run.ps1` (= build nr_host, set DLL paths, run
  `build\nr_host.exe --frames 1 --width 512 --height 512 --trace --force-load`)
- Input spec: unmodified DLLs, no spoofing; D3D12 device on vendor=0x1002 device=0x7550;
  module_trace self-injected; nvapi shim armed (unused — init failed first).
- Metrics/findings:
  - Both DLLs load cleanly on AMD (dlss@0x7FFB9D5E0000, nr@0x7FFB93790000, lasterr=0)
  - The NGX exports the Round 1 reconstructed ABI declared did resolve (that ABI was
    later audited as invalid — docs/NGX_ABI_AUDIT.md E1/E3)
  - `NVSDK_NGX_D3D12_Init` returned **0xbad00001** = `NVSDK_NGX_Result_FAIL_FeatureNotSupported`
    on the AMD adapter; process stable, clean rejection
  - No NVIDIA support modules (nvapi64/nvcuda) observed loading before the rejection
    (weak evidence — Round 1 tracer pre-dated the Phase E selftest gate)
- Verdict: PARTIAL (S5 blocked; the load itself is the first genuine runtime interaction
  with the leaked DLL on AMD). Later controls supersede the original interpretation:
  R-14 found an invalid host-ID confound and R-15 proved corrected IDs succeed on RTX
  while the AMD initialization still rejects them.
- Raw data: results/20260830_170305/nr_host_forceload_stdout.log, nr_host_forceload.json,
  forcload_20260830_182045_module_trace.log

## R-7 (Round 2) official-ABI audit + 0xBAD00001 reproduction
- Date: 2026-08-30 (Round 2 Phase A). Command: see docs/NGX_ABI_AUDIT.md +
  `scripts\run_ngx_abi_probe.ps1`.
- Findings: Round 1's reconstructed ABI was invalid in 6 audited ways (E1–E6); the
  shipped nvngx_dlss.dll is a *snippet* surface (E7). Under the OFFICIAL ABI (static
  nvsdk_ngx_d.lib, both Init variants) the AMD adapter again returns
  0xBAD00001 = FAIL_FeatureNotSupported. Verdict: reproduction PASS. R-15 later
  supplies the corrected-ID RTX positive control and confirms the rejection is
  adapter-dependent rather than an App/Project-ID artifact.

## R-8 (Round 2) ABI regression tests — tools/ngx_abi_probe
- Command: `scripts\run_ngx_abi_probe.ps1`. Compile-time static_asserts (Result enum,
  sizeofs, function-pointer prototypes, API version macro) + runtime export-table check
  of DLSS_DLL_PATH. Verdict: PASS (results/<ts>/ngx_abi_test.json); wired into
  scripts/build_all.ps1.

## R-9 (Round 2) genuine NGX host (nr_host rewrite)
- Official-ABI pipeline: device → Init/Init_with_ProjectID → GetCapabilityParameters →
  AllocateParameters → DLSS/DLAA create (DLAA contract from DLSS5-Feeder, MIT) →
  command-list-recorded EvaluateFeature + ExecuteCommandLists → readback (FNV-1a + BMP)
  → ReleaseFeature → Shutdown1; SEH around every NGX call. On this AMD box the run
  terminates at Init (exit 6, 0xBAD00001); the remaining stages are the RTX-machine
  script (scripts/run_nvidia_reference.ps1). No fabricated PASS anywhere.

## R-10 (Round 2) module_trace v2 selftest — READY
- Command: `scripts\run_module_trace_selftest.ps1`. Full load→resolve→unload chain
  (baseline, hooks_installed, load×3 incl. nested DLL, getprocaddr by name+ordinal,
  free) asserted in order from the JSONL log. Verdict: READY, 0 failures
  (results/<ts>_module_trace_selftest/).

## R-11 (Round 2) nvapi trampoline ABI self-proof — PROVED after fix
- Command: `scripts\run_nvapi_trampoline_test.ps1`. Round 1's "stack args ride along"
  DISPROVED (args 5+ corrupted: the C dispatcher's frame sits between caller and
  callee). Trampoline rewritten as log-then-pure-jmp passthrough; re-test:
  1..10 args + return byte-exact (results/<ts>_nvapi_trampoline_test/).

## R-12 (Round 2) texture interop gate — PARTIAL (swizzled layout)
- Command: `scripts\run_texture_interop_test.ps1`. RGBA8/RGBA16F/R32F/RG16F, shared
  textures imported as HIP buffer views. dir2 (HIP modify → D3D12 readback):
  mismatch=0, max_abs_error=0 for ALL four formats. dir1 (D3D12 write → HIP
  linear-read): 15–30k byte mismatches — AMD texture memory is NOT linearly laid
  out. Conclusion: interop data path is sound; raw-buffer kernels must be
  swizzle-aware. Verdict: PARTIAL (results/<ts>_texture_interop/texture_interop.json).

## R-13 RTX 40 remote CUBIN compatibility control — NEGATIVE CONTROL ONLY
- Host: Ubuntu 22.04, 2× GeForce RTX 4090 D (`sm_89`), driver 550.144.03,
  CUDA 12.4.
- Tool: `tools/cubin_compat_probe`; a same-architecture `sm_89` CUBIN loaded
  successfully on both GPUs, proving the CUDA context and probe were valid.
- An `sm_90` control was rejected cleanly with `CUDA_ERROR_INVALID_SOURCE` (300).
- Both tested DLSSNR modules were structurally readable by `cuobjdump` as `sm_120`,
  but `cuModuleLoad` terminated with SIGSEGV (exit 139) on this old driver rather
  than returning a CUDA error.
- Verdict: this RTX 40 host cannot load the tested DLSSNR payload. The result is a
  useful architecture/driver negative control, not a DLSS 5 reference: driver age
  and architecture mismatch remain confounded, and Linux cannot execute the
  project's Windows D3D12/NGX/RenoDX feature-18 chain.
- Raw machine-readable result: `results/20260830_rtx40_remote/cubin_compat.json`.

## R-14 RTX 5060 Windows positive-control attempt — HOST CONTRACT FAIL
- Host: Windows 10 build 19044, GeForce RTX 5060 Blackwell (`sm_120`), driver
  591.59 / CUDA 13.1, D3D12 visible on vendor 0x10de with an 8 GB adapter.
- Runtime integrity: DLSS 310.7 and DLSSNR 310.8 hashes match the AMD lab copies;
  both Authenticode signatures are valid NVIDIA signatures. ABI probe PASS
  (59-export snippet surface); binary probe PASS (15 EM_CUDA payloads).
- Tooling: module_trace READY and NVAPI 1..10-argument trampoline PROVED.
- Stage A and observational Stage B both reached a real NVIDIA D3D12 device, then
  returned `0xBAD00001 = FAIL_FeatureNotSupported` from Init and ProjectID fallback.
  Stage B separately proved that `nvngx_dlssnr.dll` loads successfully on RTX.
- The trace contains no DLSS feature DLL, NVAPI, or CUDA module load before the
  failure. Thus the failure precedes DLSSNR kernel compatibility and cannot be
  used as evidence of an AMD vendor gate.
- Root-cause correction: comparison against the vendored, known-working
  DLSS5-Feeder source showed the lab used different self-assigned identifiers.
  `nr_host` now matches Feeder AppId `0x1000000` and ProjectID
  `a0f57b54-1daf-4934-90ae-c4035c19df04`; tested by R-15.
- Evidence: `results/20260830_230102_rtx5060/`.

## R-15 RTX 5060 Windows corrected-contract run — INIT/CREATE PASS, EVALUATE CONTRACT FAIL
- Returned archive: `dlssnr_cloud_result_20260830_231421.zip`, SHA-256
  `C0F85D186BBE67B6235396237E23F7C7DBC2AB8385BF0A07EFBC4482084B4549`.
- Adapter: GeForce RTX 5060 Blackwell `sm_120`, vendor 0x10de, device 0x2f04.
- With DLSS5-Feeder AppId `0x1000000`, NGX Init returned Success. Capability
  discovery (`SuperSampling.Available=1`), AllocateParameters, scratch query,
  deterministic 512x512 uploads, and public DLAA CreateFeature all passed.
- All 8 Evaluate calls returned `0xBAD00005 = FAIL_InvalidParameter`; therefore
  0/8 frames completed and no output image was produced.
- Direct source comparison found v2 omitted
  `InRenderSubrectDimensions.Width/Height`, which the known-working Feeder host
  supplies on every Evaluate. Round 3 now fills those fields plus the explicit
  zero jitter/sharpness members from the reference contract.
- The module trace now records the driver `_nvngx.dll`, `nvngx_dlss.dll`, system
  `nvapi64.dll`, `nvapi64_impl.dll`, `nvcuda64.dll` (`cuGetExportTable`),
  `nvdxgdmal64.dll`, and `nvobjectloader64.dll`. This is the public DLSS support
  chain, not proof of DLSSNR kernel execution.
- `nvngx_dlssnr.dll` loaded in observational Stage B, but the package has no
  ReShade/RenoDX add-on host, so private feature 18 was not created or evaluated.
- Consequence: the corrected host succeeds on RTX while AMD rejects at Init,
  confirming a real adapter-dependent NVIDIA support gate. S3B/S4 remain open
  until Round 3 Evaluate succeeds and a feature-18 host is added.
- Evidence: `results/20260830_231421_rtx5060_v2/`.

## R-16 RTX 5060 Windows Round-3 public DLAA reference — PASS
- Returned archive: `dlssnr_cloud_result_20260830_232657.zip`, SHA-256
  `CAE631C214CA2239D97664AA945CF768E7D636A0FF59671772DC510AC82954B9`.
- Adapter/runtime: RTX 5060 `sm_120`, vendor 0x10de/device 0x2f04; signed DLSS
  310.7 and DLSSNR 310.8 hashes unchanged from earlier runs.
- Stage A: Init, capabilities, parameters, scratch, upload, public DLAA
  CreateFeature, 8/8 Evaluate calls, readback, release, and shutdown all PASS.
- Traced Stage B repeats the same 8/8 PASS while observationally loading
  `nvngx_dlssnr.dll`; both stages produce an identical 1,048,576-byte output,
  SHA-256 `9B98DFC631755EED553036786D23527D7E810E1DFA204BEB802787AD394C5041`
  and FNV-1a `62e45f5a7c5735e1`.
- Output is not a copied input: 302,026/1,048,576 bytes differ (28.8034%), mean
  absolute byte difference 0.721212, maximum 153.
- The trace again records the complete public support-module chain through driver
  NGX, `nvngx_dlss.dll`, NVAPI, `nvcuda64.dll`/`cuGetExportTable`, DXG DMA, and
  NVIDIA object loader.
- Verdict: public NVIDIA reference host PASS. This removes host correctness as a
  confound from the AMD `0xBAD00001` result. It is not S3B/S4 NR success because
  no `renodx-dlss5.addon64` was supplied and feature 18 was explicitly NOT_RUN.
- Evidence: `results/20260830_232657_rtx5060_v3/`.

## R-17 RTX 5070 Feature-18 neural-rendering reference — PASS (S4)
- Returned archive: `dlssnr_feature18_result_20260830_234330.zip`, SHA-256
  `9D6CCCCBA47AFA8DE0262928EE46C90E0E1EEC834BEB6DA1EBBAE2CE1060DDA5`.
- Actual adapter from ReShade: GeForce RTX 5070, driver 610.88. This differs from
  the earlier RTX 5060 cloud host and is recorded as a distinct reference machine.
- ReShade 6.8.0.2155 registered add-on "DLSS 5 Neural Rendering"
  v0.2026.827.2036 using API version 18.
- RenoDX hooked D3D12 NGX CreateFeature/EvaluateFeature/ReleaseFeature and armed
  inline DLSS contract capture. Missing `EvaluateFeature_C` was logged as an error,
  but the applicable D3D12 EvaluateFeature path continued successfully.
- The signed DLSSNR 310.8 D3D12 runtime initialized, inline NR resources were
  created at 640x360, private feature 18 was created after DLSS/DLAA, and inline
  feature-18 evaluation succeeded at logged counts 1 and 60.
- The Feeder public host completed 300/300 DLAA evaluations and exited cleanly;
  the warm-up re-create also produced a second successful feature-18 creation.
- Verdict: **S4 PASS** and S3B PASS-HIGH-LEVEL. This is genuine private feature-18
  execution on NVIDIA, not merely DLL loading or a non-null handle. Launch-side
  NVAPI/CUDA API identification and per-kernel tracing remain pending.
- Evidence: `results/20260830_234330_rtx5070_feature18/`.

## R-18 RTX 5070 Feature-18 dynamic module/export trace — PASS
- Returned archive: `dlssnr_feature18_result_20260830_235021.zip`, SHA-256
  `8DC23B1AEFE852190AF1900C24624B9A071F3BF1A0332EC8CC335C21ECAF893E`.
- Package revision `v6_feature18_traced`; `module_trace` reported
  `hooks_installed=true` before the ReShade/NGX path started.
- The execution result reproduces R-17: host exit 0, 300/300 public DLAA
  evaluations, signed DLSSNR 310.8 initialization, two feature-18 creations, and
  inline feature-18 evaluation success (logged at counts 1 and 60).
- The private path resolves DLSSNR's D3D12 Init/Create/Evaluate/Release/Shutdown
  exports and loads system `nvapi64.dll`/`nvapi64_impl.dll`. It resolves both
  `nvapi_QueryInterface` and the private `nvapi_Direct_GetMethod` entry point.
- Crucial negative evidence: after DLSSNR enters the successful feature-18 path,
  no `nvcuda.dll`/`nvcuda64.dll`, `cuModuleLoad`, `cuLaunchKernel`, DXG DMA helper,
  or NVIDIA object-loader module is observed. The CUDA modules in R-15/R-16 are
  therefore attributable to the public DLSS chain, not automatically to NR.
- Verdict: S4 remains PASS. For this run, pure classic CUDA Driver API CASE B is
  rejected; CASE A (D3D12 + NVAPI private CuModule/CUBIN family) is the leading
  inference. Exact NVAPI method IDs/calls are still required before finalizing the
  AMD transport ABI.
- Evidence: `results/20260830_235021_rtx5070_feature18_trace/`.

## R-19 RTX 5070 Feature-18 NVAPI interface-ID trace — PASS / CASE A CONFIRMED
- Returned archive: `dlssnr_feature18_result_20260831_000136.zip`, SHA-256
  `2262A415B7E93C4DF43C52702030B157D1299D03EBA194629E848E50E16503A5`.
- Package revision `v7_feature18_nvapi_query_trace`; the narrow QueryInterface
  wrapper was armed and the execution remained healthy: host exit 0, 300/300
  public evaluations, two feature-18 creations, and inline NR evaluation success.
- Phase-difference against the pre-DLSSNR portion identifies five non-null methods
  first requested only after `nvngx_dlssnr.dll` loads:
  `0xAD1A677D CreateCuModule`, `0xE2436E22 CreateCuFunction`,
  `0x24973538 LaunchCuKernelChain`, `0xDF295EA6 DestroyCuFunction`, and
  `0x41C65285 DestroyCuModule`.
- These are not community guesses: the names/IDs are present in NVIDIA's official
  R610 open-source `nvapi_interface.h`, and their typed D3D12 ABI is published in
  `nvapi.h`. The older community-cited IDs in the lab table were stale and have
  been corrected.
- Verdict: **CASE A CONFIRMED** for this Feature-18 path. DLSSNR submits CUBIN
  modules/functions/kernel chains through D3D12 NVAPI rather than the classic
  `nvcuda` Driver API. Next capture target is actual calls: CUBIN blob sizes,
  function names, handles, launch-chain lengths, grid/block dimensions and
  parameter sizes.
- Evidence: `results/20260831_000136_rtx5070_feature18_nvapi_ids/`.

## R-20 RTX 5070 Feature-18 typed NVAPI call trace — PASS
- Returned archive: `dlssnr_feature18_result_20260831_001127.zip`, SHA-256
  `0A8C6FB6890B6DA9E512103E75B89CE335E5FF616589F8244F17337D09D06192`.
- v8 call wrappers did not disturb execution: host exit 0, 300/300 public DLAA
  calls, two private feature-18 creations, and inline evaluation success.
- Capability probes: one null/zero CreateCuModule and one zero-kernel launch each
  returned `-14 = NVAPI_INVALID_POINTER`, as expected. They are excluded below.
- Real lifecycle: 9/9 CuModules created, 96/96 CuFunctions created, 46,800/46,800
  LaunchCuKernelChain calls succeeded, then 96/96 functions and 9/9 modules were
  destroyed successfully. Every real launch chain contains exactly one kernel.
- The launch count factorizes exactly as 300 frames × 156 kernel submissions per
  frame. All module blobs share header `50 ED 55 BA 01 00 10 00`; sizes range from
  6,936 to 3,944,768 bytes.
- Created function names expose the expected FP8 Swin/ViT graph: fused pre/post,
  1/2/4/8/16-head stages, QKV, attention, projection, FFN expand/contract,
  upsample, synchronization, clear and copy variants.
- Sampled grids include 32-wide warp-oriented blocks (`32x1`, `32x4`, `32x8`)
  and a `256x1` clear kernel. Parameter blocks observed range from 16 to 264 bytes.
- Verdict: CASE A is proven at invocation level. Remaining trace gap is the exact
  156-kernel ordered frame graph and parameter bytes; v8's every-60-call detail
  cadence aliases with the 156-call period, so v9 records complete selected frames.
- Evidence: `results/20260831_001127_rtx5070_feature18_nvapi_calls/`.

## R-21 RTX 5070 Feature-18 complete 156-slot frame graph — PASS
- Returned archive: `dlssnr_feature18_result_20260831_002356.zip`, SHA-256
  `0B2DC1FCE412E5C70765DE6A7F9992A81F2329943AF6FF49867D9655B9488A65`.
- v9 preserves the positive reference result and captures all 156 slots in each
  of frames 1, 61, 121, 181 and 241 (780 detailed kernel records total).
- All five frames have identical ordered function/grid/block/parameter-size graph
  signature `be12be4011601716284d43b8053f00aa4936c13006288e001917a1119ad52064`.
- The frame uses 43 distinct functions out of 96 created variants. 153/156
  parameter blocks are byte-identical across all five frames; only the input
  pre-block, output post-block, and final copy slots vary with boundary resource
  addresses/state. Warm-up resource recreation changes addresses but not graph.
- Runtime hashes map the nine loaded modules exactly to nine of the 15 fatbin
  containers in the signed DLL, including exact file offsets and sizes. Six small
  containers are unused by this configuration.
- Verdict: the RTX architecture-capture phase is complete for the tested
  640x360/variant-02 path. Structured maps are archived as `module_map.csv`,
  `function_map.csv`, and `frame_001_sequence.csv`. Further RTX runs should be
  requirement-driven (other resolution/mode/architecture), not routine retries.
- Evidence: `results/20260831_002356_rtx5070_feature18_full_frame/`.

## R-22 RX 9070 XT AMD 156-slot graph transport replay — PASS (not S6)
- Tool: `tools/amd_graph_replay/amd_graph_replay.cpp`, built for `gfx1201` and
  run by `scripts/run_amd_graph_replay.ps1`.
- Input integrity: the manifest records SHA-256 for the R-21 module, function and
  frame-sequence CSVs plus the exact replay executable.
- Adapter proof: HIP reports AMD Radeon RX 9070 XT / `gfx1201`; DXGI independently
  reports vendor `0x1002`, device `0x7550`.
- Lifecycle reconstruction created nine synthetic module handles and 96 function
  handles from R-21 ownership data, then destroyed all 96 functions before all
  nine modules with no leaked live handles.
- The AMD default stream launched all 156 slots with the captured grid, block and
  dynamic-shared-memory dimensions. GPU-written sequence records were exactly
  0..155 and validation reported 156 markers with zero mismatches.
- All 11,624 captured parameter bytes were uploaded to AMD device memory. Each
  marker kernel computed FNV-1a over its own parameter slice on the GPU; every
  result matched the host reference.
- Verdict: **AMD graph transport scaffold PASS**. Classification is explicitly
  `LAB_TRANSPORT_ONLY`, `counts_as_s6=false`, `neural_math_executed=false`.
  It proves the scheduler/lifecycle/parameter-carriage substrate, not execution
  of DLSSNR-originated neural kernels.
- Evidence: `results/20260831_003706_amd_graph_replay/`.

## R-23 D3D12 GPU-VA versus HIP mapped-address identity — PASS / IDENTITY DISPROVED
- The existing buffer/fence interop test was extended to record both address
  spaces for the same imported 4 MiB D3D12 heap.
- RX 9070 XT values were D3D12 GPU VA `0x200c40000` and HIP mapped pointer
  `0x304000000`; `address_identity=false`. Bidirectional data tests still passed
  with zero mismatches.
- Verdict: the CASE-A backend must register D3D12 VA intervals and relocate every
  pointer-like field. Raw captured addresses cannot be passed to HIP unchanged.
- Evidence: `results/20260831_004730_interop_address_identity/`.

## R-24 Used-kernel inventory and compressed PTX recovery — PASS / GATE-1 CORRECTED
- The stable R-21 graph resolves to 9 loaded modules, 96 created functions, 43
  used functions and 156 slots: 2 boundary slots, 9 I/O-transform slots and 145
  neural-core slots. Parameter volume is 11,624 bytes/frame.
- Every one of the 15 `0xBA55ED50` runtime containers contains a Zstd frame that
  decompresses to PTX 9.4 targeting sm_120. The 15 payloads contain 231 entries.
- This supersedes R-5's “zero PTX/pure SASS” interpretation: plaintext scanning
  could not see compressed PTX. The CUBIN representation still exists and is
  sm_120-only, so the accurate classification is hybrid PTX+CUBIN.
- Evidence: `results/20260831_005343_kernel_inventory/` and
  `results/20260831_010100_all_runtime_modules/extraction_manifest.json`.

## R-25 Official five-call AMD NVAPI backend self-test — PASS (transport/boundary only)
- `tools/nvapi_amd` exports `nvapi_QueryInterface` and implements the exact
  NVIDIA R610 CreateCuModule, CreateCuFunction, LaunchCuKernelChain,
  DestroyCuFunction and DestroyCuModule IDs/signatures.
- On the real RX 9070 XT D3D12 adapter it reconstructed the 9-module,
  96-function, 156-launch lifecycle; registered a shared D3D12 buffer; translated
  a D3D12 VA interval to its HIP pointer; and rejected an intentional out-of-range
  address. No handles remained live.
- Its exact `cc_cb_clear` implementation cleared 27,648 words to `0xffffffff`;
  D3D12 readback reported zero mismatches.
- Verdict: ABI/lifecycle/resource-relocation/boundary dispatch PASS, explicitly
  `LAB_TRANSPORT_ONLY`, `counts_as_s6=false`, `neural_math=false`.
- Evidence: `results/20260831_005935_nvapi_amd_selftest/`.

## R-26 Original NVIDIA PTX translation/execution on RX 9070 XT — PARTIAL PASS (not S6)
- ZLUDA v7-preview.3 initialized `AMD Radeon RX 9070 XT [ZLUDA]`. A controlled
  simple PTX module loaded and resolved, proving the probe and driver path.
- Multi-entry modules must be split at function granularity: unsupported unused
  entries otherwise make all functions unavailable. The extractor preserves the
  original module directives/globals and selected entry without rewriting its
  instructions.
- The isolated, unchanged NVIDIA `cc_cb_clear` PTX loaded, resolved and launched.
  With the exact R-21 count 27,648, readback found zero values other than
  `0xffffffff` (`clear_mismatches=0`). This is the first original runtime PTX
  executed on AMD, but it is a boundary operation and does not count as S6.
- Isolated `cg2r_copy_kernel` fails parsing at
  `sust.p.2d.v4.b32.zero`. The first used neural pre-block fails at constructs
  including tuple-discard `mov.b64`, `mma.sync` and FP8 e4m3 conversions.
- Verdict: PTX reuse is viable per function; texture/tensor instruction lowering
  is the next engineering task. `full_translation_pass=false`,
  `neural_math_executed=false`.
- Evidence: `results/20260831_011219_zluda_ptx_probe/`.

## R-27 RX 9070 XT final texture-copy boundary implementation — PASS (not S6)
- The AMD backend now implements the official R610 merged texture/sampler and
  independent descriptor-object APIs in addition to the five lifecycle/launch
  calls. CPU descriptors are copied into backend-owned shader-visible heaps and
  represented by stable synthetic 64-bit objects.
- `cg2r_copy_kernel` is lowered to a native D3D12 compute shader with the decoded
  72-byte ABI: texture/surface objects, source and destination transforms, and
  width/height. It records a real dispatch on the caller's command list.
- On the RX 9070 XT (`DXGI vendor 0x1002`), the self-test copied a deterministic
  64x32 RGBA32F texture. All 8,192 float components matched exactly;
  `component_mismatches=0`, `max_abs_error=0`, one merged-object call, one
  independent-object call, one dispatch and zero pipeline failures.
- The older HIP clear/VA-relocation/lifecycle test was rebuilt against the new
  diagnostics ABI and re-passed: 9 modules, 96 functions, 156 markers, exact
  27,648-word clear, zero mismatches and no live module/function/resource handles.
- Verdict: the standalone AMD boundary implementation is complete. Exact RTX
  descriptor-object identity and a full-runtime numerical oracle remain pending,
  so M4 stays partial and S6 is unchanged.
- Evidence: `results/20260831_013221_nvapi_amd_copy_selftest/` and
  `results/20260831_013231_nvapi_amd_selftest/`.

## R-28 First neural pre-block parameter-load ABI — PASS static metadata (not S6)
- A reusable PTX ABI analyzer isolates one entry, follows the parameter-array base
  alias, enumerates direct `ld.param` accesses and decodes the corresponding bytes
  from the captured RTX launch without modifying proprietary PTX.
- For `cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8`, the PTX declaration and
  R-21 capture both prove a 264-byte parameter block. The entry performs 37 direct
  loads covering 252 bytes in intervals `[0,204)`, `[208,232)` and `[240,264)`.
- The remaining 12 bytes are `[204,208)` and `[232,240)`; they are not directly
  loaded by this entry and are treated as padding/reserved until dynamic evidence
  proves otherwise. Bit patterns are decoded, but pointer/tensor meanings are not
  guessed.
- Verdict: the first neural oracle can now target a bounded, offset-exact ABI.
  Tensor identity, shape/stride, before/after data and weights still require the
  targeted RTX capture; no neural execution is claimed.
- Evidence: `results/20260831_013704_preblock_param_abi/`.

## R-29 RTX Feature-18 descriptor-object identity — PASS observational (not S6)
- Returned v10 archive SHA-256 is
  `DC9093AAACE331B7B6C28A999418CB45A7ED1EC989F21101ECC4A3362EB3DE4D`.
  The positive control remained intact: host exit 0, 300/300 evaluates, feature
  18 created and inline feature-18 evaluation succeeded.
- The trace recorded 618 merged texture/sampler calls, 21 independent descriptor
  calls, 780 detailed selected-frame launches and 635 successful nonzero objects.
- All five slot-155 copy launches joined without ambiguity. Frames 1/61/121/181
  used input object `0x0000080200009803` from texture descriptor `0x70000064`
  plus sampler `0x50000045`, and output object `0x0000000000009804` from
  independent type-0 descriptor `0x70000084`. Frame 241 recreated the same roles
  as `A003/A004` backed by `0x90000064/84`. Dimensions stayed 640x360.
- In slot 1, offset +0 always resolves to the primary merged texture/sampler
  object. After warm-up, offsets +8 and +16 resolve to a persistent auxiliary
  texture and a frame-varying history texture. Integer fields prove 640x360 input,
  640x384 padded domain and 320x192 downsample domain; offset +200 is a frame index.
- The reusable join analyzer reports five copy launches, five pre-block launches
  and `copy_unresolved_object_pairs=0`. The old tracer emitted 249 malformed
  path-only JSON lines, explicitly counted and hashed; none are target NVAPI events.
- Evidence: `results/20260831_012648_rtx5070_feature18_descriptors/` and
  `results/20260831_014220_descriptor_trace_join/`.

## R-30 D3D12 resource/descriptor tracer v11 — PASS local instrumentation gate
- `module_trace` v2.4 now escapes JSON strings and can register the host's D3D12
  device before Feature-18 resource creation. It records committed/placed/reserved
  resources, GPU-VA intervals, SRV/UAV/sampler creation and descriptor copies.
- The RX 9070 XT local gate created a 64-KiB buffer and 64x32 RGBA32F texture,
  then one SRV, one UAV, one sampler and three descriptor copies. Every expected
  event was present and all JSONL records parsed (`invalid_json_lines=0`).
- Package `dlssnr-windows-feature18-resource-traced-v11-20260831_015412.zip`
  was verified against its 12-file manifest with zero mismatches. SHA-256 is
  `2D501B22A9EF1EC8484FE8953AAC73F4AD0F1381EBF19BC03CEC1DA6EC0EEA8B`.
- Verdict: instrumentation is ready for the requirement-driven RTX resource run;
  no neural execution or tensor content is claimed by this local gate.
- Evidence: `results/20260831_014841_module_trace_d3d12_selftest/`.

## R-31 RTX Feature-18 D3D12 resource identity — PASS observational (not S6)
- Returned v11 archive SHA-256 is
  `7BD023AD37C78E064FC9527513E050B39F01792DB384A7F9DC0ED8504538CA90`.
  The RTX 5070 positive control remained intact: host exit 0, 300/300 evaluates,
  feature 18 created/evaluated, strict JSONL, one successful D3D12 hook install,
  43 resource records, 1,235 SRVs, 40 UAVs, 622 samplers and 38 descriptor copies.
- The formal resource join resolves all five slot-155 launches without ambiguity.
  Source `0x29c829a98e0` and destination `0x29c829a8f50` are both 640x360
  `DXGI_FORMAT_R16G16B16A16_FLOAT` textures. All five first pre-block launches
  resolve their primary input to a third 640x360 RGBA16F texture.
- Direct pre-block pointer fields +216 and +248 fall into one 27,807,744-byte
  buffer at offsets 110,592 and 7,974,912; +224 is the base of a separate
  147,719,680-byte buffer. This proves resource ranges, not tensor semantics.
- Evidence: `results/20260831_015650_rtx5070_feature18_resources/` and
  `results/20260831_020353_d3d12_resource_trace_join/`.

## R-32 D3D12 copy-content snapshot v12 — PASS local gate, RTX run pending
- `module_trace` v2.5 preserves descriptor-to-resource identity across NVAPI's
  internal null-view clobbers, maps returned CUDA objects back to D3D12 resources,
  and binds slot 155's two object parameters to the exact input/output textures.
- Its GPU snapshot export queues two texture-to-readback copies after the workload,
  writes compact rows without D3D12 pitch padding, and records dimensions, format,
  byte counts, resource identities and FNV-1a hashes. The numerical analyzer
  supports RGBA16F and RGBA32F and requires bitwise identity for the copy oracle.
- On RX 9070 XT, the 64x32 RGBA32F self-test read back 32,768 bytes at each end;
  SHA-256 values were equal, all 8,192 scalars matched and every JSONL line parsed.
- Verified package
  `dlssnr-windows-feature18-copy-snapshot-v12-20260831_021308.zip` has SHA-256
  `1ECF387CF704535871673A24ECE98EC06352B4CF552DD3A24A558F17A7BD5DC1`.
  The RTX content result remains pending and no neural execution is claimed.
- Evidence: `results/20260831_021024_module_trace_d3d12_selftest/`.

## R-33 First neural pre-block N0 capture v13 — PASS local gate, RTX run pending
- A conservative PTX origin analyzer follows global-memory addresses back to
  64-bit parameter fields without inventing bounds for dynamic terms. In the
  isolated original pre-block, +216 reaches four writes and zero reads, +224
  reaches 109 reads and zero writes, and +248 reaches 64 writes and zero reads.
  The largest observed constant weight offset is 21,616 bytes.
- `module_trace` v2.6 resolves those three GPU addresses to live D3D12 buffer
  intervals. At frame 1 slot 1 it records bounded buffer copies before and after
  the real NVAPI neural launch on the same command list: the full gap from +216
  to +248, a 64-KiB weight window, and a 1,966,080-byte PTX-derived output
  envelope. All capture sizes remain explicitly non-semantic until RTX evidence.
- On RX 9070 XT, the injection self-test produced all six raw files and metadata,
  preserved the read-only window exactly, and detected the deliberately modified
  output at offsets 0..255 (255 changed bytes). Strict JSONL and the earlier copy
  snapshot both remained PASS.
- Verified package `dlssnr-windows-feature18-n0-snapshot-v13-20260831_023647.zip`
  has a 12-file, zero-mismatch manifest and SHA-256
  `112651D8585E6E8584E4BB41E5B921713245671AA58F4B787029AFE7256C6EB1`.
  RTX write activity is pending; this capture machinery is not AMD neural math.
- Evidence: `results/20260831_022951_preblock_global_access/` and
  `results/20260831_023543_module_trace_d3d12_selftest/`.

## R-34 Returned v13 RTX N0 oracle and same-input AMD copy — PASS (not S6)
- Returned archive SHA-256 is
  `B566AD24795050E64DA8AE59429A48ECCF46C692B7302999188E9F952F2E9936`;
  all 15 ZIP entries pass CRC. The RTX 5070 positive control remains intact:
  host exit 0, 300/300 evaluations, Feature 18 created/evaluated, D3D12 hook
  installed and zero malformed JSON lines.
- Slot 155 copied a 640x360 RGBA16F texture bitwise exactly. Both compact
  1,843,200-byte files hash to
  `CD556E0D9C2B958CF1D189412B33AA7BC52AB178369D4F5469F22E85E71FD243`;
  all 921,600 FP16 components match, with zero non-finite values and zero
  max/mean/MSE error.
- The RX 9070 XT native D3D12 lowering then consumed this exact RTX input and
  reproduced the exact expected output: zero byte/component mismatches and
  `max_abs_error=0`. This closes M4 with a same-input cross-vendor comparison.
- N0 frame-1/slot-1 activity is strict: the 7,864,320-byte scratch changed in
  7,527,456 positions, the 65,536-byte weight window was unchanged, and the
  1,966,080-byte output changed in 1,962,555 positions. Static PTX provenance and
  byte lengths give logical FP8 envelopes `[384,640,32]` and `[192,320,32]`;
  later PTX address reconstruction refines their physical storage to tiled/planar
  layouts rather than direct HWC.
- Evidence: `results/20260831_023943_rtx5070_feature18_n0/` and
  `results/20260831_024744_nvapi_amd_rtx_copy_oracle/`.

## R-35 RX 9070 XT E4M3 numerical primitive — PASS (not S6)
- A real HIP `gfx1201` kernel decoded and re-encoded every byte of the RTX N0
  output (`1,966,080` bytes, SHA-256
  `856C01903D521F90BA652ECB376A99837E62C544D434502C7F4EAEE2D7A80ED8`).
  Capture mismatches are zero; all 254 finite E4M3 encodings also round-trip
  exactly, and the negative verifier detects a deliberate one-bit mutation.
- The result establishes an AMD software implementation of the FP8 storage
  primitive used by N0. It does not yet validate arbitrary FP16-to-E4M3 rounding,
  MMA register-fragment semantics or the fused pre-block, so `counts_as_s6=false`.
- Evidence: `results/20260831_025229_amd_e4m3_rtx_oracle/`.

## R-36 Exhaustive NVIDIA FP16-to-E4M3 oracle v14 — PASS build gate, RTX run pending
- A self-contained Windows executable dynamically uses the NVIDIA CUDA driver;
  no CUDA SDK or proprietary project binary is packaged. Its embedded PTX 8.1
  kernel targets `sm_89` and executes the exact
  `cvt.rn.satfinite.e4m3x2.f16x2` instruction over all 65,536 FP16 bit patterns.
- The returned 65,536-byte table will independently validate AMD rounding,
  saturation, NaN handling and two-lane packing rather than accepting a merely
  self-consistent encoder.
- Verified build package
  `dlssnr-windows-fp8-reference-v14-20260831_025645.zip` has SHA-256
  `AF066DBF3BE5A85D49C8723CB4DCD5E048A0EE3D8C3AD7A433E32E43DD5A3817`.
  The package contains only the executable, runner and hash manifest. RTX output
  is pending and `counts_as_s6=false`.

## R-37 Exhaustive RTX-to-RX9070XT FP16-to-E4M3 comparison — PASS (not S6)
- Returned v14 archive SHA-256 is
  `523796446D1B5EAC1622C09D4A78B7B1C23698E2C40AC2928EFECCCE0C3942D3`.
  The RTX 5070 loaded PTX 8.1/`sm_89`, executed
  `cvt.rn.satfinite.e4m3x2.f16x2` with CUDA status 0, and emitted one byte for
  every one of the 65,536 binary16 bit patterns. Raw SHA-256 is
  `0212E2599ADCD3301D3BAD890A053B8B41E514049B9988DB67E77C2E21E464CE`.
- Direct analysis exposed exactly 1,023 mismatches in the initial AMD encoder,
  all negative-NaN inputs. NVIDIA canonicalizes both NaN signs to `0x7f`; the AMD
  implementation was corrected to do the same rather than masking the difference.
- A real RX 9070 XT HIP kernel then matched all 65,536 RTX bytes: zero finite,
  infinity or NaN mismatches. Packed order is `low_f16_to_low_e4m3`; the negative
  verifier detects a deliberate bit flip. The 1,966,080-byte N0 output regression
  also remains exact after the correction.
- This closes the standalone packed conversion primitive, not the fused N0
  operator; `counts_as_s6=false`.
- Evidence: `results/20260831_104457_rtx_fp16_e4m3/`,
  `results/20260831_110051_amd_fp16_e4m3_rtx_oracle/` and
  `results/20260831_110110_amd_e4m3_rtx_oracle/`.

## R-38 FP8 MMA register-fragment oracle v15 — PASS build gate, RTX run pending
- A CUDA-driver-only probe embeds PTX 8.7/`sm_120` and invokes the exact N0
  instruction
  `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`. Eight deterministic
  cases cover zeros, ones, signed inputs, seeded finite FP8 values, nonzero FP16
  accumulators, saturation-scale operands and FP8 subnormals.
- Each case records all 32 lanes: four A, two B, two C and two D registers per
  lane. This makes register-fragment placement and FP16 accumulation independently
  reproducible without exporting model weights or relying on an entire frame.
- Verified package `dlssnr-windows-mma-reference-v15-20260831_110845.zip` has
  SHA-256 `9903909F6775C07D58671D8202215682CABF0128DFAA0A6BB069150168A13C5A`;
  its three entries match the internal manifest. RTX numerical output is pending
  and `counts_as_s6=false`.

## R-39 RTX-to-RX9070XT m16n8k32 FP8 MMA — PASS bitwise (not S6)
- Returned archive SHA-256 is
  `F1D1A345A4499B7AE347836AAFBFFD0C55CFA764360F77455ED8FE2A1151D65C`.
  RTX 5070 executed the exact PTX 8.7/`sm_120` instruction with CUDA status 0;
  all seven archive entries, expected lengths and A/B/C/D SHA-256 hashes pass.
- The RX 9070 XT HIP implementation applies the documented 32-lane fragment
  formulas to reconstruct A `[16,32]`, B `[32,8]` and C/D `[16,8]`, performs
  FP32 accumulation and rounds once to FP16. Across eight cases and 1,024 FP16
  outputs, every bit matches RTX (`mismatches=0`, all per-case counts zero).
- Cases include zero/identity-like values, signs, deterministic finite FP8,
  nonzero accumulators, maximum finite operands and subnormals. A deliberate bit
  mutation is detected.
- This closes the dominant FP8 MMA numerical primitive present 256 times in the
  isolated N0 parser inventory. It does not yet reconstruct the fused kernel;
  `counts_as_s6=false`.
- Evidence: `results/20260831_112253_rtx_m16n8k32_e4m3_mma/` and
  `results/20260831_112900_amd_m16n8k32_e4m3_mma_oracle/`.

## R-40 Remaining N0 instruction oracle v16 — PASS build gate, RTX run pending
- One package combines the two remaining high-count unsupported instruction forms
  in N0: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` and
  `movmatrix.sync.trans.aligned.m8n8.b16`. Each uses eight deterministic complete
  warp cases, including signs, finite random values, extremes and subnormals.
- The output contract is 4,096/2,048/2,048/2,048 bytes for f16 A/B/C/D fragments
  plus 1,024/1,024 bytes for movmatrix input/output. No model weights or runtime
  binary are included.
- Verified package `dlssnr-windows-remaining-reference-v16-20260831_113540.zip`
  has SHA-256 `10099E9DB441579B5EB739663E3175624487FB0531DCF829BE78B7FAC2FEB82A`;
  all three entries match its manifest. RTX numerical output remains pending.

## R-41 Returned v16 and RX 9070 XT remaining N0 primitives — SCOPED PASS
- Returned archive SHA-256 is
  `A60C2F0B7FC6F8FEB2BBD6F6877D95E77DB8EAC5D97AE17BA6F8C322068F6BC0`.
  RTX 5070 executed both PTX entries with CUDA status 0; all nine entries and six
  raw-file SHA-256 values match the manifest.
- RX 9070 XT `movmatrix.m8n8.b16` reproduces all 256 destination registers
  bitwise. Its implementation follows the row-fragment to column-fragment
  transpose across the full warp, not an identity or per-lane byte swap.
- RX 9070 XT m16n8k16 f16 MMA reproduces all 512 outputs in cases 0/1/2/7
  bitwise: zeros, ones, signed cancellation and subnormal inputs. The four
  deliberately overflow-heavy cases report 337/512 differences. They are not
  hidden: PTX explicitly leaves f16 accumulation order, rounding and subnormal
  handling unspecified, so cross-architecture bit identity is not a valid gate
  for those stress inputs.
- Verdict: movmatrix is closed and f16 MMA fragment mapping plus functional-range
  math is closed. Overflow-order parity is not claimed; the next hard gate is the
  real captured N0 value range. `counts_as_s6=false`.
- Evidence: `results/20260831_114007_rtx_remaining_n0_primitives/` and
  `results/20260831_114536_amd_remaining_n0_primitives_oracle/`.

## R-42 N0 unsupported-instruction semantic coverage — PASS inventory (not S6)
- Reclassification of the original log's 740 ZLUDA parser diagnostics is exact:
  736 `Unrecognized statement` records comprise 424 E4M3 conversions, 256 FP8
  MMAs, 32 movmatrix operations, 16 f16 MMAs and eight tuple moves
  (`mov.b64`/`mov.b128`); four additional records reject the cache policy on the
  four wide stores. No diagnostic remains in an unknown category.
- Cross-vendor micro-oracles cover the numerical semantics of 728/740 diagnostics: 712
  exact and 16 f16 MMA statements under the documented functional-range scope.
  The remaining 12 diagnostics are eight tuple repacks plus four cache-hint store
  rejections, not additional neural math.
- `translator_integrated_statements` deliberately remains zero. Standalone
  semantic coverage must not be confused with lowering these operations into the
  isolated PTX or executing fused N0.
- Evidence: `results/20260831_115000_n0_lowering_coverage/`.

## R-43 N0 tuple/cache compatibility lowering — PASS translator delta (not S6)
- Four tuple-discard `mov.b64` operations now use explicit temporary registers.
  Four `mov.b128` plus `st.global.L1::no_allocate.b128` sequences now lower to
  four ordered per-thread `st.global.b32` stores; only the non-semantic L1
  allocation hint is removed.
- A real RX 9070 XT ZLUDA load of the rewritten PTX succeeds. Parser diagnostics
  fall from 740 to exactly 728: all 12 tuple/cache mechanics disappear, while the
  counts of all four numerical categories remain unchanged.
- Function resolution still fails and no N0 kernel is launched because the 728
  numerical statements have oracle implementations but are not yet integrated
  into the PTX translator. `counts_as_s6=false`.
- Evidence: `results/20260831_120000_n0_tuple_lowering/` and
  `results/20260831_120500_n0_lowering_delta/`.

## R-44 N0 tiled downsample epilogue on RX 9070 XT — PASS stage (not S6)
- PTX address equations refine scratch storage to
  `[H/4,W/4,lane32,byte16]` MMA tiles and output storage to two contiguous
  `[H/2,W/2,channel16]` planes. The earlier direct-HWC interpretation is retired.
- A lane-exact reconstruction follows the four tile addresses, PTX source-lane
  permutation, four shuffles, pairwise FP16 additions, multiplication by 0.25 and
  satfinite E4M3 encoding. Against all 1,966,080 RTX output bytes it is exact for
  1,416,816 bytes (72.06%); another 408,685 differ by one same-sign E4M3 code.
  The declared exact-or-adjacent quantization gate is 92.85% overall and 96.86%
  outside the four leading all-zero scratch tile rows retained in the capture.
- A real `gfx1201` HIP kernel executes this same tiled epilogue on
  `AMD Radeon RX 9070 XT` with wave size 32. Its 1,966,080-byte result is bitwise
  identical to the independent CPU/PTX reconstruction (SHA-256
  `7817A3984783E448E12C1584FE4865C060117C978DCFA735C4B39AC11D955E92`).
- This is the first multi-tile N0 dataflow stage, not a standalone instruction
  micro-oracle. It consumes an RTX-captured quantized intermediate rather than
  recreating N0 from textures and weights, so complete N0 and S6 remain open.
- Evidence: `results/20260831_122500_n0_tiled_epilogue/`,
  `results/20260831_121559_amd_n0_tiled_epilogue/` and
  `results/20260831_123000_n0_storage_layout/`.

## R-45 Complete isolated N0 numeric PTX lowering — PASS resolution (not S6)
- The compatibility pipeline now lowers every original parser diagnostic:
  12 tuple/cache mechanics, 424 packed FP16-to-E4M3 conversions, 32 movmatrix
  transposes, 16 m16n8k16 f16 MMAs and 256 m16n8k32 E4M3 MMAs.
- E4M3 integer conversion is exhaustive over all 65,536 half bit patterns.
  Matrix fragment source formulas are checked for every lane, output element and
  K coordinate. The f16 and FP8 accumulation models retain the previously closed
  RTX/RX 9070 XT oracle scopes.
- Real RX 9070 XT ZLUDA probes show the exact diagnostic progression
  `740 -> 728 -> 304 -> 272 -> 256 -> 0`. The final 55,752,278-byte PTX loads in
  571,481 ms and `cuModuleGetFunction` returns CUDA success for the real N0 entry.
- This closes parser and function-resolution integration, but not execution:
  the probe intentionally supplies no texture object or 264-byte launch block,
  so `kernel_launched=false`, `execution_verified=false`, and `counts_as_s6=false`.
- Evidence: `results/20260831_124500_n0_e4m3_lowering/`,
  `results/20260831_125500_n0_movmatrix_lowering/`,
  `results/20260831_130500_n0_f16_mma_lowering/` and
  `results/20260831_132000_n0_full_numeric_lowering/`.

## R-46 Full isolated N0 execution on RX 9070 XT — PASS execution (RTX parity pending)
- The probe now creates a real 640x360 RGBA16F CUDA array/texture object through
  ZLUDA, uploads the captured 65,536-byte weight window, patches the exact
  264-byte launch block to local allocations and launches the learned N0 entry.
- A one-CTA smoke test (`1x1x32`) and the complete `80x48x32` launch both return
  CUDA success at launch and synchronization on `AMD Radeon RX 9070 XT [ZLUDA]`.
  The complete launch takes 6.16 ms to enqueue and 15.46 ms to synchronize after
  an 815.63 ms cached module load.
- Independent verification, separate from the probe JSON, re-reads every byte.
  Full-grid scratch is 7,864,320 bytes with 7,863,132 nonzero bytes and SHA-256
  `A6818180420B9BDC1C3E441959C078287E26E365D793B0CCE76C1E56DA8DA715`;
  output is 1,966,080 bytes with 1,965,587 nonzero bytes and SHA-256
  `3AF3ECEE848166E02BF98D25C6B09E877D98A168F1D26DC8C0EA75BB332807B7`.
- This is real full learned-body execution, not a marker or epilogue-only kernel.
  It remains `counts_as_s6=false`: the input is the controlled synthetic
  RGBA16F resource used by this harness, and the original RTX SM120 PTX must now
  run over the exact same input/weights/parameters before numerical parity can be
  judged.
- Evidence: `results/20260831_134000_amd_n0_single_cta/` and
  `results/20260831_135000_amd_n0_full_grid/`.

## R-47 First same-input RTX N0 attempt — DRIVER VERSION BLOCK, corrected package ready
- The returned archive SHA-256 is
  `A6852EF71AA0635959115118A3B1D332BB763BCFB7F9649313BF8140977CAA24`.
  It identifies `NVIDIA GeForce RTX 5070`, validates the expected input, weights,
  parameter and PTX hashes, and reaches CUDA context creation successfully.
- `cuModuleLoadData` returns code 222,
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`: the captured text declares PTX 9.4, which
  this installed driver does not accept. No function lookup or kernel launch is
  attempted, so this result is not a numerical failure and supplies no tensor
  oracle.
- The replacement package changes exactly one directive, `.version 9.4` to
  `.version 8.7`; restoring that directive makes its PTX byte-for-byte identical
  to the captured source. The SM120 target, instructions, parameters and data are
  unchanged. PTX 8.7/SM120 was already executed successfully by the same RTX 5070
  during the FP8 MMA oracle.
- Evidence: `results/20260831_125952_rtx5070_n0_reference_fail/`. Replacement
  package: `deliverables/n0_full_reference_20260831_130219.zip`, SHA-256
  `B575F369CDB01126356DA94BD7D9D9CE18A1614B75992DB39DB45EFE4187DCC8`.

## R-48 Same-input RTX/RX 9070 XT full N0 comparison — EXECUTION PASS, PARITY FAIL
- Returned archive SHA-256
  `B4CE0B06CDBF4983FE3E1665511908C40E87719C3559BDD01FC72B9509C75862`
  proves the normalized original PTX loads, resolves and completes the full
  `80x48x32` launch on RTX 5070. RTX scratch/output sizes and all five payload
  hashes match the RX 9070 XT experiment exactly.
- Execution is closed but numerical parity is not. Scratch has 27,575 exact bytes
  (0.3506%), 1.0333% exact-or-adjacent E4M3, correlation -0.3637 and normalized
  RMSE 18.73. Output has 5,726 exact bytes (0.2912%), 0.8730%
  exact-or-adjacent, correlation -0.5426 and normalized RMSE 16.56. Both fail the
  declared parity gate by a wide margin; this is retained as a hard FAIL.
- Three independent RX 9070 XT full launches produce identical scratch and output
  SHA-256 hashes, excluding uninitialized or nondeterministic execution.
- The generated lowering itself was then executed over the earlier RTX register
  micro-oracles. FP8 MMA matches all 2,048 output bytes and movmatrix all 1,024
  bytes. f16 MMA matches cases 0/1/2/7 exactly; only the already scoped overflow
  stress cases differ. This moves the earliest-divergence search before the first
  MMA rather than blaming the dominant FP8 lowering without evidence.
- A one-CTA checkpoint now exports 3,072 bytes of A/B fragments immediately before
  the first f16 MMA on AMD. RTX package
  `deliverables/n0_checkpoint_reference_20260831_132043.zip` has SHA-256
  `A4C943997EDC7DF978C59539B9C953EE5CF817F6EB1C9905BE0DC6B41E901F85`.
- Evidence: `results/20260831_130611_rtx5070_n0_same_input/`,
  `results/20260831_132500_amd_fp8_mma_lowered_ptx/`,
  `results/20260831_133100_amd_remaining_lowered_ptx/` and
  `results/20260831_133700_amd_n0_pre_mma_checkpoint/`.

## R-49 First N0 pre-MMA checkpoint — PASS bitwise, post-MMA package ready
- Returned archive SHA-256 is
  `01627E32B61376235DC1EE58FEC32629DA7112660E1DCC9849523FF0B5DE8B70`.
  RTX 5070 executes the one-CTA prefix and exports 32 lanes x 96 bytes immediately
  before the first f16 MMA. Payload hashes match the AMD experiment.
- All 3,072 bytes are bitwise-identical to RX 9070 XT, including 2,048 bytes of
  input-feature A fragments and 1,024 bytes of weight B fragments. Both sides have
  SHA-256 `D3C7D2DE96A5169265229404A9812D0583010A127702B0F7C859924CDE8CAC82`.
- This closes texture sampling, input preprocessing, shared-memory fragment
  layout, pointer relocation and the first weight loads for CTA (0,0). The full
  tensor divergence therefore begins at or after the first matrix block.
- RX 9070 XT now executes a lowered checkpoint immediately after the first 16 f16
  MMAs and exports 4,096 D-fragment bytes. RTX comparison package
  `deliverables/n0_post_f16_reference_20260831_132945.zip` has SHA-256
  `75C29EF161108D5B2A5674F65F32E8DDBF66F131A18DA8AD47D6367AEAB2F118`.
- Evidence: `results/20260831_132553_rtx5070_n0_pre_mma_checkpoint/` and
  `results/20260831_134000_amd_n0_post_f16_checkpoint/`.

## R-50 Corrected N0 f16 fragment lowering — S6 PASS with full numerical parity
- The returned post-f16 archive has SHA-256
  `A2C316D8C43D59AC4BFFE67FC0F22F40B25ACD8A94D0810C3B4759DA34F73DE2`.
  Before correction, RTX and RX differ in all 2,048 exported FP16 values after
  the first 16 f16 MMAs, while the immediately preceding 3,072 A/B bytes are
  identical. This localizes the defect to the f16 MMA lowering itself.
- The lowering used an incorrect A-fragment source-lane formula. Correcting the
  lane, packed-register and half selection to the PTX ISA m16n8k16 fragment
  layout makes the post-MMA checkpoint bitwise exact: 0/4,096 byte and
  0/2,048 FP16 mismatches across all 16 operations.
- The corrected complete lowered N0 entry then loads, resolves and executes its
  full `80x48x32` grid on `AMD Radeon RX 9070 XT [ZLUDA]`. A cached repeat is
  bitwise deterministic for both returned tensors.
- Against RTX 5070 over the exact same input, weights and 264-byte ABI, scratch
  has correlation 0.99999852, normalized RMSE 0.001719 and 99.9873%
  exact-or-adjacent E4M3 agreement. Output has correlation 0.99999880,
  normalized RMSE 0.001552 and 99.9895% exact-or-adjacent agreement. Both pass
  the declared 0.99/0.1/0.95 numerical gate by a wide margin.
- This is the first DLSSNR-originated learned neural kernel executing on the RX
  9070 XT with an RTX same-input numerical oracle, so S6 is PASS. It is still an
  isolated controlled N0 harness: the other graph functions, real-frame resources,
  156-slot dispatch and Feature-18 integration remain required for S7/S8.
- Evidence: `results/20260831_133211_rtx5070_n0_post_f16_checkpoint/`,
  `results/20260831_134500_amd_n0_post_f16_corrected/`,
  `results/20260831_140500_amd_n0_full_grid_corrected/` and
  `results/20260831_140501_amd_n0_full_grid_corrected_repeat/`.

## R-51 First downstream Swin N1 on RX 9070 XT — AMD EXECUTION PASS, RTX oracle ready
- Captured graph slot 2 is
  `cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8`, the direct successor
  of N0. Its stable launch is `40x24x1`, block `32x1x1`, with a 96-byte ABI.
- The original isolated entry exposes 680 first-pass ZLUDA diagnostics: 256 FP8
  MMAs, 384 E4M3 conversions, 32 movmatrix operations, four wide stores and four
  cache-hint store diagnostics. After those are lowered, one previously masked
  release-store cache hint appears. Removing only `L1::no_allocate` while
  retaining `st.release.gpu.global.s32` clears the final diagnostic.
- The resulting 54,558,338-byte entry compiles and resolves on RX 9070 XT. Static
  origin analysis binds ABI +0 to read-only N0 output, +8 to N1 output, +16 to a
  read-only weight view at `weights.raw+0x5600`, and +56 to the graph sync buffer.
- A real full-grid launch over the RTX N0 same-input tensor succeeds and emits
  1,964,665 nonzero output bytes out of 1,966,080. The sync region is initialized
  to `0xffffffff`; exactly 960 CTA release locations become zero and the other
  26,688 words retain the sentinel. A repeat run is bitwise identical for both
  output and sync tensors.
- This is real downstream neural execution, but it does not yet advance S7:
  numerical correctness awaits the original-PTX RTX output. The verified package
  `deliverables/n1_slot2_reference_20260831_142143.zip` has SHA-256
  `B5F00CCC55475F453C202852D8483221FAD72EFE16A3BEAC88DE3FE6F9145EA2`.
  Its normalized PTX restores byte-for-byte to the isolated 9.4 source after
  changing only `.version 8.7` back to `.version 9.4`.
- Evidence: `results/20260831_142000_n1_slot2_baseline/`,
  `results/20260831_142500_n1_slot2_lowering/`,
  `results/20260831_143501_amd_n1_slot2/` and
  `results/20260831_143502_amd_n1_slot2_repeat/`.

## R-52 Same-input RTX/RX 9070 XT N1 slot-2 comparison — NUMERICAL PARITY PASS
- The returned RTX archive has SHA-256
  `2E3CA0DC27BCEEC9D15BA6A5603B2E5D7E172ED092BEBA150A0671F25BB38C65`.
  Its RTX 5070 manifest, probe, payload hashes, `40x24x32` launch and output hash
  all validate against the packaged oracle.
- RTX and RX emit the same 1,964,665 nonzero bytes. Of 1,966,080 E4M3 elements,
  1,965,376 (99.9642%) are byte-exact and 99.9966% are exact or one adjacent
  same-sign code apart. Pearson correlation is 0.99999888 and normalized RMSE is
  0.001495, passing the declared 0.99/0.1/0.95 gate by a wide margin.
- The complete 110,592-byte sync region is bitwise identical across vendors:
  exactly 960 release words are zero and all 26,688 untouched words retain the
  `0xffffffff` sentinel. The RX output and sync buffers also repeat bitwise.
- This closes numerical correctness for the second consecutive learned graph
  kernel (N0 then N1 slot 2). It is not S7: slots 3-155, full-frame resources and
  image-level output verification remain open.
- Evidence: `results/20260831_142631_rtx5070_n1_slot2/` and
  `results/20260831_143501_amd_n1_slot2/comparison.json`.

## R-53 Feature-18 downstream weight capture v17 — PACKAGE READY
- Static graph/resource joins locate the next weight views at model-resource
  offsets `0x44DA00` (slot 3), `0x1390400` (slot 4), and `0x7681600` (slot 5).
  The previous 64 KiB N0 capture cannot cover any of them.
- `module_trace` 2.7 now resolves all three addresses from the captured N0 weight
  base and records a 64 KiB before/after pair for each while preserving the N0
  scratch/weight/output windows. PASS requires all four weight views to be
  immutable and all twelve raw files to exist.
- A real D3D12 queue self-test resolves, copies, dumps and validates all new
  windows while retaining the output mutation negative check. The complete 46
  Python regression tests also pass.
- Package `dlssnr-windows-feature18-downstream-weights-v17-20260831_151327.zip`
  has SHA-256 `925633826900621AD46F9ADF96E1942D362BC227BF58D874264FFC967CC4F7C0`;
  all 12 manifest payload records pass streamed archive hash verification.

## R-54 Swin dependency slots 3-5 on RX 9070 XT — AMD EXECUTION PASS
- Returned v17 archive SHA-256 is
  `1C78DFA1DBEF447CD860E1C67BBD25B4955FC78F24317FC4372BBCC1EABDFC6C`.
  Its three 65,536-byte downstream weight views match the exact graph resource
  offsets, remain immutable before/after N0, and have distinct stable hashes.
- Using the real preceding AMD N1 output, captured parameters and captured weights,
  the lowered chained entry executes slot 3 at `41x25x32` and slot 4 at
  `41x24x32`. Their sentinel-backed release counts are exactly 1,025 and 984.
- The ds-wait entry then executes slot 5 at `40x25x32`; its main tensor has
  1,965,261 nonzero bytes and its additional downsample tensor has 982,699
  nonzero bytes. Every main tensor is 1,966,080 bytes.
- A separately allocated second run reproduces every slot output, sync buffer and
  slot-5 additional output bitwise. Capture, execution, dependency-chain and
  determinism gates all report PASS.
- RTX reference package
  `deliverables/swin_slots3_5_reference_20260831_153128.zip` has SHA-256
  `CEA701965DEB03EBE33E14A02FD98DF58464B9E4263EFAEFBAE6A67361FE225F`.
  All 12 payload hashes pass streamed ZIP verification and both PTX entries
  restore byte-exactly after reversing the version-only normalization.
- Evidence: `results/20260831_151635_rtx5070_feature18_v17/`,
  `results/20260831_152915_amd_swin_slots3_5/` and
  `results/20260831_152935_amd_swin_slots3_5_repeat/`.

## R-55 Swin slots 3-5 RTX/RX numerical comparison — PARITY PASS
- Returned RTX archive SHA-256 is
  `B0193CE69C31C7FF30E5FD2C3BE33A927C8EE5EB9E22E60650213A965DBA35C5`.
  The manifest is PASS on RTX 5070, all 12 payload hashes match, and observed
  release counts are exactly 1,025/984/0 for slots 3/4/5.
- E4M3 main-output correlations are 0.99999104, 0.99999944 and 0.99999973;
  normalized RMSE values are 0.004234, 0.001059 and 0.000735. Exact-or-adjacent
  rates are 99.8609%, 99.9862% and 99.9905%. Every declared gate passes.
- Slot 5's logical `96x160x32` FP16 downsample output has correlation 0.99999793,
  NRMSE 0.002033 and 99.8641% exact-or-adjacent agreement. The unused allocation
  tail is all-zero on both devices.
- All three sync buffers are bitwise identical across vendors, and every AMD
  output remains bitwise deterministic across two independent runs. This closes
  five consecutive learned graph slots after N0, but not the complete graph/S7.
- Evidence: `results/20260831_153542_rtx5070_swin_slots3_5/` and
  `results/20260831_152915_amd_swin_slots3_5/comparison.json`.

## R-56 Swin slots 3-23 weight acquisition v18 — PACKAGE READY
- Graph parameter decoding binds 21 downstream 64 KiB model views to the exact
  weight-resource offsets consumed by slots 3-23. This covers the mapped 1h/32,
  2h/64, 4h/128 and 8h/256 Swin families in one Feature-18 run.
- The expanded D3D12 capture self-test resolves and dumps all 24 windows
  (scratch/base weight/output plus 21 downstream views). All downstream views
  remain immutable, the output mutation negative check still detects 255 changed
  bytes, and all 48 Python tests pass.
- Slot 6's isolated source has 713 ZLUDA parser diagnostics. The lowering accounts
  for 368 E4M3 conversions, 304 FP8 MMAs, 32 movmatrix operations, four wide
  stores and one release-cache store. A further 28 explicit `shared::cta` scopes
  pass the parser but fail in the backend; removing that redundant qualifier
  preserves CTA-local shared-memory semantics.
- Moving the already validated scalar FP8 MMA body into one PTX device helper
  reduces the module from 64,055,996 to 4,208,001 bytes. The compact slot-6 entry
  loads and resolves on RX 9070 XT. Running the same compact lowering on slot 3
  reproduces its established scalar output and sync hashes bitwise over the full
  `41x25x32` grid, so the compression does not change observed results.
- Package `dlssnr-windows-feature18-swin-weights-v18-20260831_155721.zip` has
  SHA-256 `B155A65FBF590CDFB49285E08EE7F9E83FDE770CFF41C20D261532A3C599AE6A`.
  The returned capture will supply the real slot 6-23 weight views needed to
  launch slot 6 without repeating per-family acquisition; execution is not yet
  claimed because that real slot-6 view is not available locally.
- Evidence: `results/20260831_154500_swin2h_slot6_baseline/`,
  `results/20260831_155000_swin2h_slot6_lowering/` and
  `results/20260831_155547_module_trace_d3d12_selftest/`, plus
  `results/20260831_160900_slot3_compact_regression/`.

## R-57 Swin 2h slots 6-8 on RX 9070 XT — AMD EXECUTION PASS; v19 READY
- Returned v18 archive SHA-256 is
  `BFB607BC682910E3C645567E03AC8967A3A37895D20420C38E797A36B6CF359F`.
  Feature 18 completes 300/300 evaluations on RTX 5070; all 24 capture windows
  and 48 raw-file metadata hashes validate, and all 21 downstream views remain
  immutable. The summary retained an obsolete v17 revision label, but the window
  count/content proves the v18 instrumentation ran.
- With the real slot-6 view and slot-5 AMD downsample output, the compact entry
  executes the captured `20x12x1` grid with block `32x2x1`. It emits 982,584
  nonzero bytes in the 983,040-byte logical FP16 output and exactly 240 releases;
  the unused allocation tail is all zero.
- The lowered chained entry then executes slots 7 and 8 at `21x13x(32x2)` and
  `21x12x(32x2)`, producing exactly 273 and 252 releases. Input hashes equal the
  preceding AMD output hashes. A separately allocated repeat reproduces every
  output and sync buffer bitwise for slots 6-8.
- Static access analysis proves slot 9 can read through byte 65,843 of its weight
  view, exceeding v18's 65,536-byte capture. No out-of-bounds placeholder run is
  accepted. v19 extends each downstream capture to as much as 1 MiB, clamped to
  the real resource boundary; its D3D12 self-test and all 51 Python tests pass.
- Package `dlssnr-windows-feature18-extended-swin-weights-v19-20260831_162942.zip`
  has SHA-256 `C5392A77D570F71058272CD059382599A8D243E2D193C2B1593381D9F3F91FC5`.
- Evidence: `results/20260831_161539_rtx5070_feature18_v18/`,
  `results/20260831_163147_amd_swin2h_slots6_8_formal/`,
  `results/20260831_163153_amd_swin2h_slots6_8_formal_repeat/`, and
  `results/20260831_162926_module_trace_d3d12_selftest/`.

## R-58 Swin 2h slot 9 on RX 9070 XT — AMD FAMILY EXECUTION PASS
- Returned v19 archive SHA-256 is
  `9D4880AA3CF97D2482F59ACFF206A7203148B657C5D99E76E9A27C4FC1EFF8D5`.
  Its revision is correctly labeled v19, Feature 18 completes 300/300 calls,
  all 24 windows and 48 raw hashes validate, and every model view is immutable.
  Slot 9 now has 267,776 captured bytes, comfortably covering the statically
  proven 65,844-byte footprint.
- The original slot-9 ABI has input/output/weights at +0/+8/+16, predecessor wait
  at +48, extra downsample output at +72, and no release pointer. The harness
  therefore deliberately leaves +56 unchanged and verifies zero release writes.
- RX 9070 XT completes `20x13x(32x2)`. Its 983,040-byte logical main FP16 tensor
  has 982,840 nonzero bytes; the 491,520-byte downsample tensor has 491,404.
  Both allocation tails are all-zero. An independently allocated repeat is
  byte-identical for main output, extra output and sentinel sync storage.
- This completes deterministic AMD execution of the full slots 6-9 2h/64 family,
  but numerical parity remains pending. Verified RTX package
  `deliverables/swin2h_slots6_9_reference_20260831_165256.zip` has SHA-256
  `8174411230AB109B78334DE18D6C9ABAD6E274D9AC1DEEDACD7262C06DE9B62D`;
  all 16 payload hashes and all three reversible PTX normalizations pass.
- Evidence: `results/20260831_163553_rtx5070_feature18_v19/`,
  `results/20260831_164100_amd_swin2h_slot9/`, and
  `results/20260831_164300_amd_swin2h_slot9_repeat/`.

## R-59 Swin 2h slots 6-9 RTX/RX numerical comparison — PARITY PASS
- Returned RTX archive SHA-256 is
  `AE3AE26EFD12407DA03C070CF7031958179FBCAF089B654C161DB81222804F4E`.
  The manifest is PASS on RTX 5070, all 16 payload hashes match, all four
  original-PTX entries launched and verified, and observed release counts are
  exactly 240/273/252/0 for slots 6/7/8/9.
- Main FP16 output correlations are 0.99992376, 0.99994653, 0.99998064 and
  0.99999300; normalized RMSE values are 0.012349, 0.010341, 0.006223 and
  0.003742. Exact-or-adjacent rates are 98.8310%, 98.2815%, 98.9026% and
  99.4672%. Every declared numerical gate passes and every unused allocation
  tail is all-zero on both devices.
- Slot 9's logical `48x160x32` FP16 downsample output independently passes with
  correlation 0.99998421, NRMSE 0.005619 and 98.5099% exact-or-adjacent
  agreement. Its unused tails are also all-zero.
- All four sync buffers are bitwise identical across vendors. Every AMD main
  output, sync buffer and the slot-9 extra output remains bitwise deterministic
  across two independently allocated runs. This closes numerical parity for the
  complete 2h/64 family and advances the dependency frontier to slot 10's
  4h/128 family; it does not claim the complete graph or S7.
- Evidence: `results/20260831_170015_rtx5070_swin2h_slots6_9/` and its
  `comparison.json`, plus `results/20260831_163147_amd_swin2h_slots6_8_formal/`
  and `results/20260831_164100_amd_swin2h_slot9/`.

## R-60 Swin 4h slots 10-15 on RX 9070 XT — AMD EXECUTION PASS; RTX PACKAGE READY
- The three original 4h/128 entries have 320/324/340 lowered E4M3 conversions,
  272/272/288 lowered FP8 MMAs and 32 movmatrix operations each. Compatibility
  lowering also accounts for every wide store, cache-hint operation and explicit
  shared-CTA scope. All lowering reports pass.
- RX 9070 XT executes slots 10-15 in dependency order with captured `32x4`
  blocks and grids `10x6`, `11x7`, `11x6`, `10x7`, `10x6`, `11x7`. Observed
  release counts exactly match 60/77/66/70/60/0. Every 491,520-byte logical
  FP16 main tensor is dense and every unused allocation tail is all-zero.
- Slot 15 additionally emits a dense 245,760-byte logical FP16 downsample tensor
  with an all-zero allocation tail. Two independently allocated runs reproduce
  all six main outputs, all six sync buffers and the extra output bitwise.
- Verified RTX package
  `deliverables/swin4h_slots10_15_reference_20260831_172544.zip` has SHA-256
  `197AC89CB70E26043943261BE04F63628B504D73B956E060EC6561C83034E147`.
  All 22 payload hashes pass, all three PTX files reverse exactly after the
  version-only normalization, and the ZIP has no unsafe paths. Numerical parity
  remains pending its one RTX 5070 run; S7 is not claimed.
- Evidence: `results/20260831_172000_swin4h_slots10_15_baseline/`,
  `results/20260831_172500_swin4h_slots10_15_lowering/`,
  `results/20260831_172222_amd_swin4h_slots10_15_formal_v2/`, and
  `results/20260831_172243_amd_swin4h_slots10_15_repeat/`.

### Exact native-state closure for Swin 4h slots 10-15

- Evidence: `results/20260831_191000_swin4h_full_graph_exact_state/`.
- All six RX 9070 XT launches restore the full-graph RTX before-state allocations,
  including nonzero output/sync contents, and execute twice deterministically.
- Main FP16 correlation is at least 0.99992681 with NRMSE at most 0.012098 and
  exact-or-adjacent agreement at least 96.3549%. Slot 15's E4M3 downsample reaches
  0.99996693 correlation and 98.9750% exact-or-adjacent agreement.
- RTX after-state synchronization is bitwise exact. This supersedes the prior
  `PENDING_RTX_ORACLE` boundary for slots 10-15 and selects slots 16-23 next.

## R-61 One-shot full-graph RTX acquisition — PACKAGE READY
- Static inventory proves the captured frame contains contiguous slots 0-155,
  43 used functions and 15 PTX-bearing runtime modules; every used function name
  exists in its recovered module.
- Module tracer v3.0 scans every aligned parameter field around each frame-1
  launch, accepts only values resolving into tracked D3D12 buffer intervals, and
  records bounded pre/post windows. Window size is clamped by the next pointer in
  the same resource or 8 MiB. Existing descriptor-object, N0 and final texture
  copy evidence remains enabled for non-buffer inputs.
- The real D3D12 self-test captures three full-graph windows, detects exactly the
  mutated output window, writes all declared byte ranges, and passes content-
  addressed consolidation. The consolidation reduced 41,943,040 logical bytes
  to 16,777,216 unique bytes in the synthetic test.
- Package `deliverables/dlssnr-windows-full-graph-v20-20260831_175425.zip` is
  146,954,381 bytes with SHA-256
  `BCB81054638714A0C67B3317C5F7AA3424443311A0F8B515CC176B96F340822C`.
  All 13 manifested files pass SHA-256/size validation under Windows PowerShell
  5.1; the ZIP has 14 unique safe entries and all 55 Python tests pass.
- This is acquisition readiness, not AMD execution of the remaining graph. One
  cloud command now replaces per-family RTX package runs and returns a single
  content-addressed result archive.
- Evidence: `results/20260831_175000_full_graph_inventory/`,
  `results/20260831_175202_module_trace_d3d12_selftest/`, and the package above.

## R-62 Swin 8h slots 16-23 exact-state RTX/RX comparison — PARITY PASS
- All three 8h/256 entries pass strict compatibility, E4M3, movmatrix and
  FP8-MMA lowering. Eight graph-observed launches execute twice on RX 9070 XT
  using exact native RTX before-state buffers, ABIs, grids and `32x8` blocks.
- Main FP16 correlations range from 0.99987072 to 0.99999983 and NRMSE from
  0.000583 to 0.016082 with no non-finite pairs. Slot 17's 94.9683%
  exact-or-adjacent result is an advisory because PTX leaves f16 MMA accumulation
  order, rounding and subnormal handling unspecified; the correlation and NRMSE
  gates remain normative and pass. Slot 23's extra E4M3 tensor reaches
  0.99999781 correlation and 99.9536% exact-or-adjacent agreement.
- RTX synchronization state is bitwise exact for every slot and the two AMD runs
  reproduce every output/sync artifact bitwise. The configurable-ABI probe then
  reran all 16 AMD launches without regression.
- Evidence: `results/20260831_192000_swin8h_slots16_23_lowering/`,
  `results/20260831_193100_swin8h_full_graph_exact_state/`, and
  `results/20260831_194000_swin8h_probe_offset_regression/`. This closes slots
  1-23, not the complete frame or S7.

## R-63 Split 16h slots 24-56 exact-state RTX/RX comparison — PARITY PASS
- Eight strict lowered entries cover all mbarrier, bulk-copy, elect, compatibility,
  E4M3, movmatrix and FP8-MMA constructs and independently resolve on RX 9070 XT.
  All 33 captured launches execute twice with exact activation/model arenas,
  parameter layouts and synchronization state.
- Chained numerical oracles use the next consumer's input snapshot, not the
  asynchronous producer-after snapshot. Resource identity and byte offset are
  equal for every such edge, and the downstream wait makes that snapshot the
  settled RTX result. This correction turns the former 12 apparent failures into
  exact matches and preserves the original analysis for audit.
- 30 of 33 main logical tensors are bitwise exact. The only non-bitwise slots
  (42/50/54) still exceed 0.999984 correlation, stay below 0.005523 NRMSE and
  exceed 99.9007% exact-or-adjacent agreement. Both E4M3 terminal tensors are
  bitwise exact; every release region matches RTX and both AMD runs repeat
  bitwise.
- Evidence: `results/20260831_202000_split16h_slots24_56_lowering_v4/` and
  `results/20260831_210000_split16h_full_resource_exact_state/`. This closes
  slots 1-56; complete-frame execution and S7 remain open.

## R-64 ViT 1D slots 57-98 exact-state graph-edge comparison — PARITY PASS
- Nine final PTX entries pass 54 strict lowering reports and all independently
  resolve on RX 9070 XT. New coverage includes one release-fence strengthening
  and 48 vector FP16x2 reductions lowered to atomic-CAS loops.
- Forty-two exact-state launches run twice with all activation/sync pointers
  relocated into the original resource arena. Execution, asset integrity,
  numerical parity and complete-arena repeat determinism all pass.
- The 58 reliable producer-to-adjacent-consumer edges contain 42 bitwise-exact
  tensors. The other 16 are QKV E4M3 results with 100% exact-or-adjacent codes,
  correlation at least 0.99999088 and NRMSE at most 0.004276.
- Twenty-four side accumulators have no adjacent settled oracle and remain
  diagnostic. Evidence: `results/20260831_215000_vit1d_slots57_98_lowering_v2/`
  and `results/20260831_221000_vit1d_full_graph_exact_state/`. Reliable graph
  edges are now closed through slot 98; decoder slot 99 is next.

## R-65 Decoder slots 99-154 exact-state comparison — RGB VERDICT CORRECTED
- Nineteen final entries pass 171 count-checked lowering reports and all resolve
  on RX 9070 XT. The probe now relocates repeated model-arena views as well as
  activation-arena views, including both post-block model pointers.
- All 56 slots execute twice with exact RTX states and bitwise AMD repeat
  determinism. Every settled main output passes decoded numerical parity; 27 are
  byte-exact. Slot 132's 94.1711% E4M3 code-adjacency rate is retained as a
  failed advisory, not hidden; its normative correlation/NRMSE are
  0.99970255/0.024392.
- The post block writes an actual 640x360 RGBA16F image on RX 9070 XT. The
  originally reported 0.99999907 correlation and 0.001366 NRMSE included the
  constant alpha=1 plane. Reanalysis on RGB only gives correlation `0.421357178`
  and NRMSE `1.105763648`, so image parity fails. The execution and deterministic
  surface-write observations remain valid.
- Evidence: `results/20260831_232000_decoder_slots99_154_lowering_v4/` and
  `results/20260831_231000_decoder_full_graph_exact_state/`. This completes
  deterministic exact-state execution through slot 154, but not numerical RGB
  parity. R-110 supersedes the former pass conclusion.

## R-66 Integrated slots 0-155 on RX 9070 XT — EXECUTION PASS / PARITY FAIL
- A single process/context loads 51 strict PTX modules and launches the complete
  captured sequence twice without restoring any RTX intermediate. Both runs
  complete all 156 slots, every checkpoint repeats bitwise, the final hashes are
  identical, and the explicit slot-155 GPU copy is bitwise correct.
- Propagated numerical error first crosses the declared threshold at slot 9
  (correlation 0.99492953, NRMSE 0.10069924). It increases at slots 15 and 23,
  remains material through ViT/decoder entry, and produces final RGB values near
  0.501 rather than the RTX reference near 0.001. No visual/S7 pass is claimed.
- Evidence: `results/20260831_234000_full_graph_integrated_plan/` and
  `results/20260831_235000_full_graph_integrated_rx9070xt/`.

## R-67 Slot-3 numerical-stability sweep — BASELINE RETAINED; RTX TRACE READY
- Lower FP8 accumulator precision, alternative reduction orders, accurate
  reciprocal math, scalar packed-FP16 math and both subnormal-flush hypotheses
  fail to improve the exact-state RTX comparison. The baseline remains best at
  correlation 0.99999424 and NRMSE 0.003395 for slot 3.
- A new one-shot RTX 50 package captures all 256 MMA A/B/C/D fragments from the
  first CTA. Its local RX counterpart and automatic first-divergence analyzer are
  ready; the package hash is
  `E9AF6C29D95F1AE1D11ABBD0ED4272DA25BEECEF9E042E35082E2D071B98BB09`.
- The analyzer's current-model self-check reproduces all 32,768 AMD D fragments
  bitwise and ranks 43 accumulator hypotheses, so an RTX mismatch will not be
  confounded with trace-layout or lane-mapping errors. A new one-command result
  receiver safely validates the returned archive and emits `comparison.json`
  plus `receipt.json`; success, altered-hash and path-traversal tests are covered.
  The expanded hypotheses cover exact/FP32/tree reductions, RN/RZ mantissa
  widths, staged FP16 products/dot sums and subnormal handling as permitted by
  PTX's unspecified numerical details. All 87 tests pass.
- Evidence: `results/20260901_000000_slot3_accumulation_diagnostic/`,
  `results/20260901_003000_slot3_mma_trace/`, and
  `deliverables/slot3_mma_trace_reference_20260831_232130.zip`.

## R-68 Executable slot-3 correction paths — AMD PASS / NO NUMERICAL WIN
- The 43-model trace evaluator now feeds the PTX lowerer by candidate name. All
  candidates except CPU-only `exact_fsum` have executable rewrites, including
  previously missing RZ mantissas, pairwise reduction, FP16 products and staged
  FP16 dot accumulation.
- Four representative paths execute successfully with real computation on RX
  9070 XT using exact captured slot-3 state. Module load, function resolution,
  launch, synchronization and dense output gates all pass.
- Against the exact-state RTX output, FP16-product and pairwise outputs equal the
  existing baseline (NRMSE 0.00339520); RZ and staged-dot are worse at
  0.00343504 and 0.00979034. Therefore baseline remains selected until the RTX
  internal trace identifies a model. Evidence:
  `results/20260901_010000_slot3_candidate_rx_smoke/`. This is not S7.

## R-69 CTA `(0,0)` trace — EXACT; first divergent output localized to CTA `(2,0)`
- The returned RTX 5070 trace and the existing RX 9070 XT trace are bitwise
  identical for every A/B/C/D fragment in all 256 MMA instructions at CTA
  `(0,0)`. Both trace hashes are
  `2B2AB4CB7AC4E9E82EAA7F591ECC0643984722EA78E0FDA7C1E5467EADD2B28C`.
- Instrumentation does not alter either vendor's complete output. The full
  vendor outputs still differ at byte 1,949, which reverse maps to spatial
  coordinate `(3,0)`, channel byte 413, and producer CTA `(2,0)`. The exact CTA
  `(0,0)` is therefore a boundary case, not evidence of whole-kernel parity.
- A generalized CTA tracer and a matching RX 9070 XT trace for CTA `(2,0)` now
  pass execution/integrity gates. The new RTX one-run package is
  `deliverables/slot3_cta2_mma_trace_reference_20260901_002923.zip`, SHA-256
  `8DC7BAF79107F81D56C8D962A8949E98CBC171B2B7CC734C89CED284A164C143`.
  Its return will determine whether the first correction belongs in FP8 MMA
  semantics or the post-MMA epilogue. This is diagnostic and not S7.

## R-70 CTA `(2,0)` MMA trace — ACTIVATION INPUT DIVERGES BEFORE MMA RESULT
- The validated RTX 5070 trace first differs from RX 9070 XT at MMA 216 in one
  A-fragment byte (`0x39` versus adjacent E4M3 code `0x38`). B and C are exact;
  D remains exact through MMA 219 and first differs at MMA 220. Across the trace,
  36 A bytes and 91 D half results differ.
- The source register is `%r2106`, downstream of two packed-FP16 multiplies and
  immediately upstream of an already exhaustively validated E4M3 conversion.
  This rules out selecting an MMA accumulation candidate as the next correction.
- A new non-perturbing RX 9070 XT trace covers 28 registers across both multiply
  stages and their scale inputs. The one-run RTX package is
  `deliverables/slot3_f16_path_trace_reference_20260901_005345.zip`, SHA-256
  `AAFA3ADB563D845D86FB251C2F9D3BB83B5AD1796F8FB8189CBCDC9FC1CA287A`.
  The next return will identify the exact first packed-FP16 register divergence.
  This remains diagnostic and not S7; all 90 tests pass.

## R-71 Packed-FP16/rsqrt trace — RSQRT FP32 DELTA IS NOT EFFECTIVE ROOT CAUSE
- All sampled learned activations feeding the first packed-FP16 multiplication
  are exact. The scale from approximate reciprocal square root is the first
  different value: RTX `%r2002=0x3A553A55`, AMD `0x3A563A56` in lanes 24-27;
  `%r2004` shows corresponding one-ULP differences in 12 lanes.
- The full rsqrt return supersedes the initial primitive attribution. For all 944
  equal-input scalar cases, RTX and AMD half-rounded rsqrt outputs are identical,
  even though their intermediate FP32 approximations often differ by one ULP.
  All surviving half-scale mismatches begin with a different rsqrt input.
- A one-run RTX trace now covers the input, FP32 approximate results and half
  outputs of all 16 paired-rsqrt blocks. Package:
  `deliverables/slot3_rsqrt_trace_reference_20260901_010837.zip`, SHA-256
  `7267089E8A622E19A10C3E62AB2CF8956741D4A99369B03418157FDE560FA17F`.
  Its result redirects localization into the preceding square/reduction tree.
  This is not S7; all 93 tests pass.

## R-72 Packed-FP16 square/reduction trace — RTX PACKAGE READY
- Local PTX-semantic replay from exact activations reproduces AMD but not the RTX
  final reduction, confirming that rsqrt only exposes an upstream difference.
  The even tree differs in four lanes and the odd tree in twelve lanes.
- A new checkpoint spans every stage from eight exact activations through squares,
  pair-adds, two butterfly exchanges/adds, half swaps and the final two sums. Its
  RX 9070 XT execution is valid and non-perturbing.
- Package: `deliverables/slot3_f16_reduction_trace_reference_20260901_013136.zip`,
  SHA-256
  `81FE297A717DA2F6AA550C1DB8D0F2BEE1569438FB96287CD838F4E6C92C9F8E`.
  The return will identify the first exact instruction result that differs and
  enable a targeted lowering. This is not S7; all 96 tests pass.

## R-73 Packed-FP16 reduction return — PERTURBING TRACE / SELECTIVE CANDIDATE IMPROVES
- The returned RTX and RX checkpoint payloads are bitwise equal, but the RTX
  checkpoint build changes the complete output hash from the accepted reference
  `33FE6004...` to `87465D19...`. The receiver now records
  `instrumentation_perturbed=true` and
  `admissible_as_uninstrumented_oracle=false`; the apparent checkpoint equality
  cannot be read as equality of the original uninstrumented path.
- The perturbation supports an extra-precision/fusion difference: materializing
  intermediate packed halves makes RTX behave like AMD. A count-checked lowering
  can fuse only selected four-term local reductions in FP32 and round once back
  to packed FP16.
- Twenty-seven candidate executions on the real RX 9070 XT identify a five-node
  local optimum: `%r1855`, `%r1871`, `%r2235`, `%r2251`, `%r2283`. Against the
  exact RTX slot-3 output, NRMSE improves by 4.2359% to `0.003251384490`,
  correlation improves to `0.9999947143`, and sign mismatches decline to 281.
  The exact-match fraction is slightly worse, so this candidate is retained as
  an error-minimizing diagnostic correction and still does not count as S7.
- Evidence: `results/20260901_090100_slot3_f16_reduction_cross_vendor_corrected/`,
  `results/20260901_094000_slot3_local_reduction_ablation/`, and
  `results/20260901_101000_slot3_selective_reduction_greedy4/search_summary.json`.

## R-74 Selective slot-3 reduction in full graph — PROPAGATES / NOT RETAINED
- Two complete 156-launch RX 9070 XT runs prove that the five-node candidate is
  executable and deterministic in the integrated graph. A compact-lowering-only
  control reproduces the original full-graph hashes exactly, so the observed
  changes are attributable to the reduction rewrite.
- The candidate improves slot-5 NRMSE by about 0.0883%, then produces mixed
  downstream effects: slot 9 and several later boundaries worsen while slot 98
  improves slightly. The historical all-channel RGBA NRMSE improves by only
  0.00786%, from `1.001179508632` to `1.001100800241`; R-110 supersedes this
  alpha-contaminated metric with RGB-only evaluation.
- The selective rewrite remains diagnostic and is not adopted into the default
  full-graph plan. Since slots 6-9 already pass same-input cross-vendor tests,
  the next correction target is the earlier slot-1/2 propagated error rather
  than another 2h-family semantic rewrite. S7 remains false.

## R-75 N0 earliest-error continuation — FP8 CANDIDATES CLOSED / RTX TRACE READY
- Thirteen real full-grid N0 candidates show that FP32 accumulation ordering does
  not affect the current output and reduced-precision accumulation is worse.
  Baseline remains NRMSE `0.001551503979`; no candidate improves it.
- A new N0-specific tracer exports A/B/C/D around all 256 FP8 MMA instructions
  from CTA `(0,0)` into an unused one-CTA scratch range. The fully lowered trace
  module loads, resolves and executes on RX 9070 XT, producing a dense 327,680-
  byte checkpoint.
- The RTX package independently runs both the traced and untouched original PTX
  with identical resources. It records output preservation explicitly so a
  compiler perturbation cannot be mistaken for original-path evidence. Package:
  `deliverables/n0_fp8_mma_trace_reference_20260901_023855.zip`, SHA-256
  `04E3948A01135DB0E63FDD8B8476324EAD0781945316ED7F006F06C671038B50`.
  Its return will either identify the first divergent MMA or move localization
  into the post-MMA normalization/activation/quantization epilogue. S7 is false.

## R-76 Returned N0 all-MMA trace — FIRST DIFFERENCE IS NORMALIZED A INPUT
- Returned archive SHA-256 is
  `FC1F01F257D26004D76687F845BAE7660E87309CBC471A7A3B353627FE0A35B4`.
  The RTX 5070 trace passes payload, execution, size and hash validation; the
  instrumented and untouched outputs are identical, so it is admissible.
- FP8 MMA 0-183 have bitwise-identical A/B/C/D fragments across RTX and RX.
  The first pre-fragment difference is MMA 184 lane 7 A2: RTX `9D171712`, RX
  `9D181712`. The corresponding warp MMA changes two FP16 D values. This closes
  FP8 accumulation as the first cause and moves the boundary to the preceding
  normalization/scaling chain.
- A ten-stage trace now spans the exact `%r2164` MMA output, max inputs/result,
  rsqrt output, two FP16 scaling results and final E4M3 pack. The lowered module
  loads, resolves and executes one CTA on RX 9070 XT and exports 1,280 bytes.
- RTX package:
  `deliverables/n0_norm_path_trace_reference_20260901_025852.zip`, SHA-256
  `13CC278997F51B10DC8FD0838C913B4EFDEB60E87DED00CDA000B2DBA9E0E256`.
  It retains the untouched-output perturbation control. All 107 tests pass;
  S7 remains false.

## R-77 Returned N0 normalization path — ERROR STARTS IN r2329 REDUCTION INPUT
- Returned archive SHA-256 is
  `CED19D109CFDA75AA1051755F8375D5895AA2C848DDB17A6E7193D425A9CC509`.
  RTX 5070 output preservation passes, making the ten-stage trace admissible.
- The preceding MMA output `%r2164` is bitwise exact. `%r2329`, an input to the
  following max operation, is the first different stage: RTX/RX differ by one
  to two FP16 ULP in 16 halves. `%r2319`, the other max input, remains exact.
- `%r2329` reduces 32 squared values from four FP8 MMA outputs. None of 352
  enumerated rounding/fusion models based on visible half values or reconstructed
  scalar-FP32 MMA accumulators matches every RTX lane, so a single global
  packed-FP16 rounding change is rejected.
- The next package traces all 17 visible nodes of this exact reduction and also
  checks whether materializing them perturbs RTX output:
  `deliverables/n0_reduction_path_trace_reference_20260901_095422.zip`, SHA-256
  `92374DD4231A3BEDBF6D13BAA8400C278A5CA534D573B3BC30F380866008ACE8`.
  The corresponding RX trace already executes successfully. S7 remains false.

## R-78 Returned r2329 tree — MATERIALIZATION PROVES HIDDEN NVIDIA FUSION
- Returned archive SHA-256 is
  `C57923795C23726253C98DCE8ECACC13687EA68CB79C32449C56C1C9439B5009`.
  The 2,176-byte RTX and RX traces are bitwise identical at every one of 17
  nodes, but the instrumented RTX output hash `6FEF4A75...` differs from its
  untouched `65A4EF92...` and exactly equals RX. The trace is intentionally
  inadmissible as an uninstrumented oracle, while its perturbation conclusively
  identifies cross-node hidden precision/fusion as the behavior to emulate.
- Three scoped RX `%r2329` policies execute over the complete N0 grid. Max and
  fused-only worsen output NRMSE. Min improves it by 0.425% to
  `0.001544907342` but worsens scratch NRMSE and exact/sign metrics, so it remains
  diagnostic pending N0-to-slot2 chained propagation. No global rewrite is
  promoted and S7 remains false.

## R-79 N0 min full-graph propagation — REJECTED; ZERO-INPUT TRACE READY
- The N0 min variant executes all 156 slots twice on RX 9070 XT and is bitwise
  deterministic, but it worsens the first propagated boundaries: slot-1 NRMSE
  becomes `0.001506111734`, slot 2 `0.004088084795`, and slot 9
  `0.101137117279`. Final NRMSE also worsens slightly to `1.001208925587`.
  Mixed later improvements do not justify retention.
- The mismatch with the isolated test direction is explained by input scope:
  integrated replay uses the captured null-SRV zero texture. The next oracle
  therefore uses an all-zero 1,843,200-byte RGBA16F tensor, already executed on
  RX. RTX package:
  `deliverables/n0_norm_path_zero_trace_reference_20260901_101646.zip`, SHA-256
  `E80BF839C5B71E7487D4FCACEFE86425860B00AD42CAE24DA3449889586EF810`.
  It needs one RTX run and preserves the original-vs-instrumented output check.
  S7 remains false.

## R-80 Returned zero-input N0 path — r2329 DIFFERENCE DIES AT RSQRT
- Returned archive SHA-256 is
  `97E155ED989A04F45669A1C12B222CB84C2FFB6144F22EB7F0142BE70860F97E`;
  instrumentation preserves the RTX 5070 output under the complete graph's
  zero-texture input condition.
- `%r2329` differs by two half ULP in lanes 20-23, but rsqrt output and all later
  recorded scale/E4M3 values are bitwise exact. This proves that the earlier
  hidden-fusion difference is not the effective slot-1 error source for the
  integrated replay and closes the rejected min-candidate direction.
- A zero-input all-MMA package now scans every one of 256 FP8 operations:
  `deliverables/n0_zero_fp8_mma_trace_reference_20260901_102146.zip`, SHA-256
  `D5E81341C7944CBE2E6E07599D77EA0CB0C09A36E37193E6B58F95D5BECF4C4E`.
  The corresponding RX trace already passes. Its return will either identify
  the effective pre-MMA activation difference or move the boundary to the final
  N0 epilogue. S7 remains false.

## R-81 Returned zero-input all-MMA trace — MMA 178 B1 ACTIVATION IS FIRST EFFECTIVE DIFFERENCE
- Returned archive SHA-256 is
  `10758BA9CE506E04A706BB83CB2BBC22A19945FC359D6DE3FD5D8958C4B410C0`;
  the instrumented RTX 5070 output equals the untouched output.
- MMA 0-177 are bitwise exact in every A/B/C/D fragment. MMA 178 first differs
  at lane 27 B1 (`%r4350`: RTX `2020A69A`, RX `2021A69A`) and the collective
  instruction propagates that single FP8 byte into eight D-half differences.
  Three later operations repeat isolated one-byte pre-fragment differences.
- Def-use analysis places the mismatch in a dynamic activation normalization
  path: prior exact MMA output `%r2561`, scale `%r2803`, packed-half multiply
  `%r2842`, then E4M3 conversion `%rs266`. It is not a weight-load discrepancy
  and not evidence for changing FP8 MMA accumulation.
- The next one-run package records nine short-path boundaries under the exact
  zero-RGBA16F input:
  `deliverables/n0_mma178_b_path_trace_reference_20260901_103311.zip`, SHA-256
  `394ADADFE7F66EA16A3FD4A615AA6B72C3D6D32555F5A2A93EFDD58DD8FD5F5D`.
  The RX 9070 XT counterpart already passes and exports 1,152 bytes. All 114
  tests pass; S7 remains false.

## R-82 Returned MMA-178 short path — r2723 LOCALIZED BUT NOT KERNEL-OBSERVABLE
- Returned archive SHA-256 is
  `71CEB44993FD2C97D40BD28F71FD63DAAFE1238268E4259A2BA0B077654D9F7D`;
  the nine-stage RTX trace is non-perturbing. Exact MMA 146/147 outputs feed a
  first difference at `%r2723`: RTX `41204120`, RX `411F411F` in lanes 24-27.
  It survives rsqrt/scaling and changes one E4M3 byte in `%r4350`.
- No enumerated visible-FP16 or reconstructed-MMA global rounding model exactly
  reproduces RTX `%r2723`. More importantly, clean uninstrumented CTA `(0,0)`
  RTX/RX scratch and output hashes are already identical, proving that this
  internal difference is not the full-grid slot-1 output cause.
- The first real slot-1 byte mismatch maps to CTA `(1,0)`. The next package runs
  the complete `80x48` grid and traces all FP8 MMA fragments only for that CTA,
  using an opt-in scratch allocation extension so production state is untouched:
  `deliverables/n0_cta1_0_fp8_mma_trace_reference_20260901_104646.zip`, SHA-256
  `A97B1CFE91808F235E7433285BEB7ADC297D045B1CB8AA7C77A5B8F6B4C96669`.
  RX output preservation passes; all 116 tests pass and S7 remains false.

## R-83 Returned full-grid CTA `(1,0)` trace — FIRST REAL ERROR PRECEDES MMA 176
- Returned archive SHA-256 is
  `5A29D1A2A0B39C67FC3B8F11CE480EAF0C10DEB4280BD6FBEA142D5AB3EF0965`;
  full-grid RTX tracing preserves the untouched output.
- MMA 0-175 are bitwise exact. MMA 176 begins with two input-byte differences:
  B0 lane 13 (`90` vs `8F`) and A3 lane 16 (`22` vs `21`), then produces 18
  D-half differences. This rules out MMA accumulation as the first observable
  slot-1 cause and localizes two preceding activation-normalization branches.
- One combined package now traces both branches from their exact earlier MMA
  outputs through reduction, max, rsqrt, packed-FP16 scaling and E4M3 packing:
  `deliverables/n0_cta1_mma176_dual_path_trace_reference_20260901_105529.zip`,
  SHA-256
  `30462EE7B56DF3B834809D9BFF60B4EC9CCF237DA64CBE00361E3CBEF954512B`.
  RX full-grid execution is non-perturbing. All 116 tests pass; S7 remains false.

## R-84 Returned dual path — A REDUCTION PROPAGATES; B DIES; CTA MAP CORRECTED
- Returned archive SHA-256 is
  `F82C1FEDB86EF888D7A7C622A40979ABB0A406BD2BE9083493385D068C37E9D5`.
  A first differs at `%r2321`; B first differs at `%r2713`; both earlier MMA
  outputs are exact and the full-grid trace is non-perturbing.
- On real RX 9070 XT, exact observed-value A correction removes two slot-1 byte
  mismatches and slightly improves NRMSE. B correction changes no output byte;
  combined A+B equals A-only. This separates a propagating A normalization error
  from a dead B internal error, but the scoped constants are not a general fix.
- The first mismatch byte 515 remains, proving the earlier contiguous-tile CTA
  mapping false. The actual epilogue formula maps byte 515 to CTA `(8,0)`, lane
  0. Its full-grid all-MMA RTX package is
  `deliverables/n0_cta8_0_fp8_mma_trace_reference_20260901_110340.zip`, SHA-256
  `A6291F2A3DA1C8BE38A72CCBED2338A83D8A8EA9127014725013321984DF5394`.
  RX output preservation and all 116 tests pass; S7 remains false.

## R-85 Returned CTA `(8,0)` trace — FIRST REAL ERROR IS AN FP16-MMA PRODUCT
- Returned archive SHA-256 is
  `E0D26DECFE43C9DB118C004421C1A962C67FA87BE3D2996BAB8018362ACA276F`;
  full-grid RTX tracing preserves the untouched output.
- FP8 MMA 0 already receives one differing input byte: lane 17 A2 byte 1 is
  RTX `0x26` versus RX `0x25`. It is generated by E4M3 conversion of `%r556`,
  the output of initial FP16 MMA 2. The corresponding D halves also differ, so
  this observation supersedes FP8 accumulation-order tuning as the immediate
  correction target.
- The next one-run RTX package traces all 16 upstream FP16 MMA A/B/C/D
  fragments for the correct CTA `(8,0)`:
  `deliverables/n0_cta8_0_f16_mma_trace_reference_20260901_111609.zip`, SHA-256
  `93BC1C4B124D81EAAB93B0F7CFF15FA387762E1920A233F39792C412FDF0E999`.
  Its RX 9070 XT trace is already captured and non-perturbing. All 117 tests
  pass; S7 remains false.

## R-86 Returned FP16 trace — ROOT MOVES TO BOX-MULLER APPROXIMATE MATH
- Returned archive SHA-256 is
  `908B6DFE18D1D70079567DC0049B41C8751342FEF6C27AB7D071E52739D0B4D3`;
  full-grid RTX tracing preserves its clean output.
- FP16 MMA 0-3 all receive one identical A0 input-byte difference at lane 16
  (`0x23` vs `0x24`) before producing 16 differing output halves. Shared-memory
  address mapping identifies writer lane 8 `%rs15`, not MMA arithmetic, weights
  or texture input.
- `%rs15` is a half-rounded Box-Muller normal generated by integer hashing,
  `lg2.approx`, `sqrt.approx`, `cos.approx` and a multiply. The next one-run
  package captures the immediate result of each boundary across 64 samples:
  `deliverables/n0_cta8_0_box_muller_trace_reference_20260901_112952.zip`,
  SHA-256
  `32A4F0E060CD24E5579FBEE8B55C80DD3CC77CDDB7169780A190CB8095682EF5`.
  It is explicitly a controlled defining-instruction trace, not a downstream
  output oracle. All 118 tests pass; S7 remains false.

## R-87 Returned Box-Muller trace — `lg2.approx` IS THE FIRST CAUSAL PRIMITIVE
- Returned archive SHA-256 is
  `0EBEA3C3F4038CE94B830B5D03A57E828D85BC0ED224D96F73D788C7A5C1DC87`;
  its RTX 5070 instrumented output equals the untouched control.
- Hash and uniform stages are exact; 63/64 `lg2.approx` results differ first.
  Only sample 8 survives the downstream sqrt/cos/product FP16 boundary as RTX
  `0x3923` versus RX `0x3924`, which is the exact shared-memory input byte that
  caused FP16 MMA 0-3 and FP8 MMA 0 to diverge.
- A one-ULP directional diagnostic removes output byte 515, proving causality,
  but introduces three net exact-byte regressions elsewhere and is rejected as
  a general fix.
- The next package records 245,760 non-perturbing production-domain lg2 pairs
  to reconstruct the NVIDIA SFU curve:
  `deliverables/n0_lg2_full_grid_trace_reference_20260901_115154.zip`, SHA-256
  `FC48B9C1E6426F3DC0D5D99ABFE5C88AC4B5A84F066A836067ED6DCD6E65B9FF`.
  This revision uses explicit `mov` reads of CTA special registers for strict
  NVIDIA JIT compatibility and surfaces the original probe error before trace
  extraction. Its README contains the exact run command. All 120 tests pass;
  S7 remains false.

## R-88 Returned full-grid `lg2` trace — 64-SEGMENT CURVE IMPROVES REAL N0
- Returned archive SHA-256 is
  `D4E324FF8268D8CDF87A1F88945331BD2A5144C086B7D539D9D959B2D3FD6B0A`;
  the RTX 5070 run preserves its output. The 245,760 lg2 inputs are all bitwise
  equal to RX 9070 XT, so no upstream input distribution difference remains;
  242,124 RTX/RX lg2 results differ.
- A 64-segment quadratic fit is the first practical general replacement: it
  reduces clean slot-1 mismatching bytes from 2,555 to 2,491 and NRMSE from
  `0.001495641447` to `0.001463723639`. It fixes 78 old mismatches and adds 14.
  Direct half-boundary intercept tuning regresses to 2,512 and is rejected.
- Byte 515 is now exact. The next earliest mismatch, byte 2,243, maps to CTA
  `(35,0)`. The new one-run RTX package is
  `deliverables/n0_cta35_0_box_muller_trace_reference_20260901_120441.zip`,
  SHA-256
  `A42744F2BA5BE3C4FC2053CC989DAD9C850C44B51752DE62219D6505738790A2`.
  Its RX counterpart already runs the 64-segment candidate successfully and
  captures 2,048 bytes. The return will decide whether the first residual is
  still curve fitting or the following sqrt/cos approximation. The README has
  the exact command; S7 remains false.

## R-89 Returned CTA `(35,0)` trace — FIRST NORMAL HALF IS NOT THE RESIDUAL
- Returned archive SHA-256 is
  `6F5D7539BC57823B010189449730E996592929A3C086FD437C3BD51F04A17225`;
  the RTX 5070 run and output-preservation control pass.
- The 64-segment RX candidate still differs internally at lg2/sqrt/cos, but its
  first Box-Muller output is bitwise equal at the final FP16 boundary for all
  64 samples. These internal FP32 differences are dead for `%rs15` and cannot
  explain the new earliest clean-output byte 2,243.
- The kernel creates two additional halves: `%rs16` from the sine companion and
  `%rs17` from a second lg2/sqrt/cos path. The new package traces all three
  outputs plus their primitive inputs/results in one run:
  `deliverables/n0_cta35_0_box_muller_trace_reference_20260901_121249.zip`,
  SHA-256
  `D52B5452C2BF3AEE9ABD31F60D626EDD77834200746855AEA219D310544F0549`.
  Its RX 9070 XT counterpart has already passed. The return will identify which
  half survives and whether curve, sine/cosine or sqrt is the live cause. S7
  remains false.

## R-90 Returned full Box-Muller trace — ALL THREE HALF OUTPUTS ARE EXACT
- Returned archive SHA-256 is
  `B3EF46ABD57B199CF53E4B523597AB193096F8D7F7BC9F268931F5DCED16D380`;
  the RTX 5070 trace is valid and non-perturbing.
- Although approximate FP32 stages differ, all three values consumed by the
  network (`%rs15`, `%rs16`, `%rs17`) are bitwise exact after FP16 conversion
  for every selected-CTA sample. The 64-segment lg2 replacement has therefore
  solved the Box-Muller path at CTA `(35,0)`; tuning sqrt/sin/cos here would not
  affect the current first output mismatch.
- Byte 2,243 is the next independent error exposed after byte 515 was fixed.
  The new one-run package traces every FP8 MMA input and output for CTA `(35,0)`:
  `deliverables/n0_cta35_0_fp8_mma_trace_reference_20260901_121641.zip`,
  SHA-256
  `0FF31E1827F9D45FA7267B73F938C1B874998BEE5E0437BE05E10800C7BB0263`.
  Its RX counterpart already passes with the curve candidate. The return will
  locate the earliest MMA that receives or creates the new residual. S7 remains
  false.

## R-91 Returned CTA `(35,0)` all-MMA trace — RESIDUAL PRECEDES MMA 176
- Returned archive SHA-256 is
  `CF8898A55FCE142B14ECE8EE652C64CDE03AC2B7DB06DF4785E72CD11182E286`;
  full-grid RTX tracing is valid and non-perturbing.
- MMA 0-175 inputs and outputs are exact. MMA 176 first receives two differing
  A-fragment bytes, then produces four differing D halves. The two bytes are
  lane 0 A2 byte 3 (`1E/1D`) and lane 30 A1 byte 0 (`0E/0D`). RTX outputs are
  exactly reproduced by ordinary FP32 accumulation from RTX inputs, excluding
  MMA accumulation behavior as the cause.
- Both differing bytes come from normalization of otherwise exact earlier MMA
  outputs. The next one-run package captures both complete square-reduction,
  max, rsqrt, FP16 scaling and E4M3 packing paths:
  `deliverables/n0_cta35_0_mma176_a_path_trace_reference_20260901_123136.zip`,
  SHA-256
  `07F3C368CAA127306C03865416545D1B37BADCE75BAEF3DFAE81D6D3DF2DED9C`.
  Its RX counterpart passes and its README includes the exact command. S7
  remains false.

## R-92 Returned 45-stage path — REJECTED AS ORACLE, BUT FUSION IS EXPOSED
- Returned archive SHA-256 is
  `549F4FC1499365824176A8C7AAE6A393F5BFA034FAA0F2DA33AE4FA065615058`.
  The trace kernel executes, but its final RTX output differs from the untouched
  baseline, so the 45 checkpoints cannot be treated as observations of the
  original program.
- Under this forced materialization, all 45 RTX checkpoints become bitwise
  identical to the RX 9070 XT curve candidate. Combined with the clean all-MMA
  result, this shows the remaining difference depends on an NVIDIA JIT fusion
  spanning one or more packed-FP16 path operations; it is not an ordinary
  register-by-register discrepancy that survives every observation.
- The replacement package performs 45 independent single-store variants in one
  command, retaining the output-preservation result for each:
  `deliverables/n0_cta35_0_mma176_a_fusion_sweep_reference_20260901_124132.zip`,
  SHA-256
  `9BDD9ED1188A44F3AAC27C4EBBB9A276B68A85C552EDE30F84CC0FA95E2C74F4`.
  This will locate the minimal fusion-sensitive boundary without accepting the
  perturbed trace as evidence. S7 remains false.

## R-93 Returned fusion sweep — EXPLICIT HALF2 FMA REMOVES 90% OF RESIDUALS
- Returned archive SHA-256 is
  `4E9860BEB48849C064D861940C0D0B894B299FCD0D4FDA4A02577F940EA46E08`.
  All 45 RTX 5070 variants pass. Exactly 12 observations perturb the baseline:
  stages 4-9 and 26-31, covering both branches' square and first pair-sum
  nodes. The lane sums at stages 10 and 32 preserve output and are therefore
  valid clean oracles.
- The unique 128/128-bitwise model is a half2 FMA for the left square with the
  independently FP16-rounded right square. Neither strict FP16 at every node
  nor a full-FP32 four-square reduction matches. The behavior is now explicit
  PTX rather than an assumed driver optimization.
- Applying this rule to all 32 structurally identical N0 square-pairs on RX
  9070 XT fixes 2,250 of 2,491 remaining output bytes, adds no regressions and
  lowers NRMSE to `0.000446189948`. The original target byte 2,243 is exact.
  The next first mismatch is byte 36,306 at CTA `(7,1)`; 241 bytes remain.
  The next one-command package combines clean three-half Box-Muller boundary
  tracing with all 256 FP8 MMA fragments:
  `deliverables/n0_cta7_1_residual_bundle_reference_20260901_130330.zip`,
  SHA-256 `DCCB875A35AD0A28845923EA904E13124E7D8083DE723D2F922D3453881858DC`.
  Its README contains the exact run command. This advances numerical parity
  substantially, but S7 and game integration remain pending.

## R-94 Returned CTA `(7,1)` bundle — RESIDUAL IS A BOX-MULLER HALF BOUNDARY
- Returned archive SHA-256 is
  `55220D03C60065245E9BB8878FA503BEB4A4E5AD843A9CE348CFC83EE98FCF88`;
  both RTX 5070 cases preserve baseline output and pass integrity controls.
- Only the first consumed Box-Muller half differs, at samples 16 and 32 by one
  FP16 ULP. The other two halves are exact. MMA 16 is the first network stage
  to differ and already receives three changed C-fragment bytes, so its
  arithmetic is downstream rather than causal.
- Increasing the curve to 128 or 1024 segments, changing to cubic interpolation,
  and applying half-safe local intercept biases all cause net output regressions.
  The retained 64-segment quadratic plus explicit 32-site half2 FMA remains the
  best clean candidate at 241 differing bytes and NRMSE `0.000446189948`.
  The next diagnostic uses individually output-controlled sqrt and cos
  observations rather than another global curve replacement:
  `deliverables/n0_cta7_1_box_stage_sweep_reference_20260901_135851.zip`,
  SHA-256 `8A1CB69F7140FE8077CA760E2A27EF89B0B6DFF7924CC768549104223BBDBEB3`.
  Its README includes the exact command. S7 remains false.

## R-95 RTX 4090 D qualification — CURRENT BOX-STAGE PACKAGE CANNOT RUN
- The server is reachable again and still exposes two RTX 4090 D GPUs
  (`sm_89`), driver 550.144.03 and CUDA 12.4. The package transferred with the
  expected SHA-256.
- All three original PTX 8.7/`sm_120` modules return CUDA 222. Rewriting only
  the declarations to PTX 8.4/`sm_89` changes the failure to CUDA 218, and
  `ptxas` reports the first unsupported form at the full network's FP8 MMA.
- Therefore no requested output-preserving full-kernel trace was obtained from
  4090 D. Useful Ada data requires the actual RTX-40 kernel/CUBIN patch (or a
  separately reconstructed Ada kernel), not the current Blackwell payload with
  edited headers. Evidence is in
  `results/20260904_4090d_box_stage_compatibility/`. RTX 5070 remains the
  reference path for the pending E-92 package; S7 remains false.

## R-96 Returned clean box-stage sweep — TARGET DIVERGENCE STARTS AT COSINE
- The returned RTX 5070 archive passes integrity and both independent probes
  preserve the untouched output. Its SHA-256 is
  `E4B97CFB25C228C2071D9AD9AECCFC87785FF3DD7A5CD80F7F5DDD972B7697E4`.
- Although rounded sqrt differs elsewhere in the CTA, it is exact at target
  samples 16 and 32. Rounded cosine differs at both target samples by three and
  one FP32 representable steps respectively. Cosine is therefore the first
  clean observable associated with both remaining first-half errors.
- Stored rounded observations do not reveal possible hidden precision used by
  the downstream fused multiply/conversion, so this result is a localization
  gate rather than a complete replacement formula. The next package captures
  cosine input/output pairs for all 245,760 N0 samples and rejects any run that
  changes the baseline output:
  `deliverables/n0_cos_full_grid_trace_reference_20260904_002125.zip`, SHA-256
  `E74480B3ACB344610AF5A047384C818D9C0949899D9EC1AEB175B561C917F97D`.
  The matching RX trace is already valid and output-preserving. The accepted
  implementation remains at 241 output mismatches and NRMSE `0.000446189948`;
  S7 remains false.

## R-97 Returned full-grid cosine oracle — CAUSAL, BUT GLOBAL FIT REJECTED
- The RTX 5070 return passes all integrity and output-preservation controls;
  archive SHA-256 is
  `7C8A664490B5F1C814A06B2B8F2AF0D799F3EC5F6B5F2F33A462C618219FB305`.
  Its 245,760 inputs are bitwise identical to RX 9070 XT. Only 17.41496% of
  rounded cosine outputs are exact, directly establishing different SM120 and
  RDNA4/ZLUDA approximation behavior for the production input distribution.
- Global 1,024- and 4,096-segment quadratic corrections regress full N0 from
  241 differing bytes to 247 and 248. They are not promoted. A narrow two-angle
  diagnostic instead fixes exactly four old output differences, adds none and
  advances the first residual from offset 36306 to 79089; it is explicitly an
  overfit causal experiment, not a production implementation.
- Offset 79089 maps to CTA `(35,3)`. The next one-run RTX package captures clean,
  individually output-controlled `sqrt0` and `cos0` observations for that CTA:
  `deliverables/n0_cta35_3_box_stage_sweep_reference_20260904_004223.zip`,
  SHA-256 `8DE783F8AD97570B805BC0ABFF97041048ABFA79FAA16071B94887EDB64A4979`.
  The accepted production candidate remains the 241-byte version until a
  general correction improves it. S7 remains false.

## R-98 Returned CTA `(35,3)` clean sweep — MIXED STAGE DIVERGENCE
- The archive passes all integrity, RTX 5070, CTA and output-preservation gates;
  SHA-256 is
  `72AB4744351BC47A413350ECF8DAF136A4FC185EB6D98D82F31C589AF70D030B`.
- `sqrt0` differs in 20/64 samples and `cos0` in 52/64. This rules out declaring
  either rounded observation alone as the cause of the new first residual.
- To avoid another split round trip, the next bundle captures both the final
  three FP16 values consumed by the network and all 256 FP8 MMA A/B/C/D
  fragments. Its package is
  `deliverables/n0_cta35_3_residual_bundle_reference_20260904_004947.zip`,
  SHA-256 `9BC6ADA82639818B25473BF76EEE112EDCBF16F4594EDDE8F452B7C73D086CD0`.
  Matching RX traces already exist and preserve the accepted output. The
  production candidate remains at 241 differing bytes; S7 remains false.

## R-99 Returned CTA `(35,3)` residual bundle — COSINE CAUSALITY CONFIRMED
- The combined return passes integrity and preserves the RTX 5070 baseline in
  both instrumented runs; archive SHA-256 is
  `A6D1133EF35663E3ED1343C6D3DE73D246E8A0E8DFCA87A360A7820BD8416DE2`.
- Only `normal0_f16` sample 54 differs at the consumed Box-Muller boundary.
  MMA 12 first receives the difference in its A fragment, so MMA arithmetic is
  downstream. At the same sample sqrt is exact; the 12-step rounded cosine
  difference alone selects FP16 `0x2D76` on RTX versus `0x2D77` on RX.
- Adding that observed cosine input to the diagnostic correction fixes two more
  final bytes with no regressions: 235 differences versus the production
  candidate's 241. This remains a causal ablation, not a shippable lookup.
- The new first residual maps to CTA `(20,4)`. To avoid repeating one CTA per
  round trip, the next package captures `sqrt0` and consumed `normal0_f16` for
  the entire grid: `deliverables/n0_normal0_boundary_full_grid_trace_reference_20260904_010138.zip`,
  SHA-256 `6F27162EE6724D09AD2B3C9DC9EF7B2D7586D243C45E47B9DEDFF1B7155F181B`.
  Its matching RX trace preserves the accepted output. S7 remains false.

## R-100 Returned full-grid normal0 boundary — NEW 202-BYTE N0 CANDIDATE
- The RTX 5070 archive passes all integrity and output-preservation checks;
  SHA-256 is
  `E3612132B6BAECF32F2D017F8E87E50928C42911F8A19FE89D5C24048B66E9C3`.
- Only 356/245,760 consumed `normal0_f16` values differ despite 87,577 rounded
  sqrt differences and 202,961 rounded cosine differences. This confirms that
  optimization should target actual FP16 threshold crossings, not exact SFU
  bit patterns. Correct RTX cosine with AMD sqrt would leave 33 mismatches.
- The conservative 4,096-segment model enables 88 no-added-error segments. A
  full N0 run on RX 9070 XT improves 241 to 202 differing bytes, fixes 39 and
  adds none. NRMSE is `0.000416785661`, correlation is `0.999999913145`, and a
  second run repeats output hash `F954825E...DE0442` exactly. It is promoted as
  the new accepted N0 candidate.
- The first residual remains at CTA `(20,4)`. Its combined FP16-boundary and
  256-MMA package is
  `deliverables/n0_cta20_4_residual_bundle_reference_20260904_011215.zip`,
  SHA-256 `38DA2D0332F1C96DC2EB7F8FD7B57B4F45AAA8E227F12CCB3E2AE5E02B5B653C`.
  This result improves isolated N0 correctness but does not itself establish a
  complete-frame S7 result.

## R-101 Returned CTA `(20,4)` residual — SECOND BOX-MULLER PATH
- The returned RTX 5070 archive passes integrity, target-CTA and
  output-preservation gates; SHA-256 is
  `C577DCA9CCD324BC8C8FE04EB8CAB74A731D868AD43978A13BBEA6408321DCC9`.
- The first two consumed Box-Muller half values are exact. The third,
  `normal2_f16`, has one difference at sample 2 (`0xBC8A` RTX, `0xBC8B` RX).
  The difference is present in MMA 0's A input, proving the MMA result is a
  downstream consequence rather than an accumulation-semantics defect.
- The remaining ambiguity is whether the threshold crossing originates in
  `sqrt1` or `cos1`. To answer it without another CTA-by-CTA cycle, the next
  package captures both paths across all 245,760 samples in one invocation:
  `deliverables/n0_normal2_full_grid_traces_reference_20260904_012329.zip`,
  SHA-256 `69066751ECBB5D4A6FDCFB0E71572C08C98D3F8016BB636E1F068D304403FA2B`.
  Both matching RX traces preserve the accepted 202-byte candidate exactly;
  S7 remains false.

## R-102 Returned full-grid normal2 oracle — NEW 154-BYTE N0 CANDIDATE
- The archive passes RTX 5070, integrity, trace-shape/hash and two independent
  output-preservation controls; SHA-256 is
  `6D5CD8BD4E521E4E1990CB204C038EF5D20FA35B8B4FA814900CEB72F7D94C41`.
- All second-cosine inputs match bitwise. There are 342 consumed
  `normal2_f16` differences despite 202,606 cosine and 88,347 sqrt rounded
  differences. An RTX cosine with the AMD sqrt leaves only 29 theoretical
  boundary differences, confirming cosine as the dominant correctable term.
- Resolution search over 8,192/16,384/32,768 segments reduces full N0 to
  174/159/154 differing bytes with zero additions. The 32,768-segment result is
  repeat-deterministic, has output SHA-256 `61E14BDD...D764983`, NRMSE
  `0.000367638428` and correlation `0.999999932421`; it becomes the accepted
  isolated N0 candidate. A 65,536-segment result is rejected because it adds a
  previously correct byte and is poorly constrained despite a lower raw count.
- The new first residual maps to CTA `(8,7)`. Its next combined FP16-boundary
  and 256-MMA package is
  `deliverables/n0_cta8_7_residual_bundle_reference_20260904_014338.zip`,
  SHA-256 `A6FA7AA7D45CE0387904781F06D2F8AEAF18B694926B81F28808D0E489D98F64`.
  S7 and real-frame generalization remain open.

## R-103 Returned CTA `(8,7)` and propagation — N0 GENERALIZES; COARSE SLOT 15
- The archive passes all RTX 5070, target, integrity and perturbation controls;
  SHA-256 is
  `926BB1F01B89CFDF3FDF315F9A7BF4972032532BCF2034944702EC76B67DEF4A`.
  Its only consumed Box-Muller differences are two one-ULP `normal1_f16`
  samples. MMA 12 already receives the first different A byte, while ordinary
  FP32 scalar MMA models exactly reconstruct RTX from RTX fragments. No new MMA
  lowering defect is indicated; the remaining local source is the sine branch.
- The 154-byte zero-input candidate was run without refitting on the independent
  nonzero captured RGBA16F input. It produces 168 differing bytes, NRMSE
  `0.000383619918`, correlation `0.999999926418`, and a repeat-identical output
  hash. Relative to the fusion-only candidate it fixes 94 bytes and adds none.
  This is the first positive cross-input-domain result for the current N0
  approximation corrections.
- Replacing only slot 1 in the accepted compact integrated plan completes two
  deterministic 156-launch RX 9070 XT runs. Improvements persist through every
  recorded downstream boundary and move slot 9 from NRMSE `0.100699238` to
  `0.086658668` (pass). Among the then-recorded coarse checkpoints, slot 15 is
  the first failure at `0.190025019`. The legacy all-channel RGBA NRMSE improves
  from `1.001179509` to `1.001121844`; R-110 supersedes that alpha-contaminated
  metric with RGB-only evaluation. R-104 later
  adds aligned fine checkpoints and supersedes the first-failure target with
  slot 6; defer a full-grid sine oracle unless later N0 evidence requires it.

## R-104 Same-capture fine localization — SLOT 6 FIRST; 1H STATE IS CAUSAL
- The new checkpoint builder reads settled adjacent-consumer states directly
  from the original full-graph RTX ZIP. This caught and rejected an initially
  mixed-capture comparison before it could be promoted as evidence.
- Two native 156-launch RX runs remain deterministic. Same-capture NRMSE is
  slot 3 `0.005225894`, slot 4 `0.014557257`, slot 5 `0.021818147`, slot 6
  `0.102493965`, slot 7 `0.255073016`, slot 8 `0.318781847`, and slot 9
  `0.086658668`. Slot 6 is therefore the earliest strict failure, narrowly above
  the 0.1 gate; the apparent pass at slot 9 reflects a compressing/downsampling
  boundary rather than absence of the earlier error.
- Four deterministic, explicitly non-S7 injection experiments isolate causality.
  Exact slot-2 state only partly helps. Injection after slot 3 or 4 sharply
  reduces downstream error, while exact slot-5 state makes slot 6/7/8 all pass
  at `0.015492492`/`0.049286892`/`0.084323278`.
- The next engineering target is the slot 3-5/1h precision path, especially the
  small slot-3 residual whose downstream sensitivity is much larger than its
  local NRMSE suggests. A 2h rewrite is not currently justified. Final NRMSE
  remains about 1.001, so neither S7 nor real-game usefulness is claimed.

## R-105 Old slot-3 local optimum on the current graph — REJECTED AGAIN
- The previous five-reduction candidate was rerun against the accepted N0 and
  the new same-capture fine checkpoints, with two deterministic 156-slot runs.
- It improves slot 3 by `0.000079443` NRMSE but worsens the important downstream
  boundaries: slot 5 by `0.000222742`, slot 6 by `0.001170472`, slot 7 by
  `0.001640345`, and slot 8 by `0.000585789`. It also worsens slots 9-23.
  Slot 98 improves by `0.017213135` and final NRMSE by only `0.000078227`, which
  does not approach the 0.1 image gate.
- The production plan remains unchanged. The next candidate search must use
  slot 5/6 as the primary loss and reject any change that improves isolated
  slot 3 while increasing the first failing boundary. This is local AMD work;
  no additional RTX capture is required for that search.

## R-106 Slot-3 reduction search — CLOSED WITHOUT A PASS
- Sixteen single reductions and nine combinations were executed and ranked on
  same-capture slot 6. Best single `%r1893` reaches `0.102352458`; combinations
  are no better. No lowering change is promoted.
- Evidence: `results/20260904_400000_full_graph_slot34_single_reduction_search/`
  and `results/20260904_401000_full_graph_slot34_combo_reduction_search/`.

## R-107 Partial slot-3 state sweep — DISTRIBUTED SENSITIVITY
- Every one of eight disjoint 245,760-byte exact-state replacements makes slot 6
  pass; no one-eighth subrange of the best chunk does. The best coarse/fine slot-6
  NRMSE values are `0.097584746` and `0.101890374` respectively.
- This rejects a single-local-region model and keeps all partial injections
  explicitly outside S7. Evidence: `results/20260904_410000_full_graph_slot3_partial_injection_sweep/`
  and `results/20260904_411000_full_graph_slot3_chunk6_subrange_injection_sweep/`.

## R-108 Decoder checkpoints and late injection — POST BLOCK SEPARATED
- Slots 99-152 fluctuate above the gate, slot 153 returns to NRMSE `0.068902005`,
  and the N0 scratch is already at `0.000481114`. Replacing the visible slot-154
  main and scratch inputs, including immediately before launch, does not fix the
  output. This directs diagnosis to slot 154 execution/ABI rather than another
  upstream checkpoint campaign.
- Evidence: `results/20260904_420000_full_graph_decoder_same_capture_rx9070xt/`
  through `results/20260904_470000_full_graph_inject_n0_scratch_main_late_rx9070xt/`.

## R-109 Slot-154 parameter offset 56 — RESOURCE BRANCH RESTORED
- Even a complete exact RTX pre-slot arena reproduces the old failure when offset
  56 is cleared. Preserving its captured nonzero value removes the +0.5 RGB bias
  and restores the branch direction. R-111 supersedes the earlier scalar/ABI
  interpretation: the value is a texture object, and its samples still require
  the actual SRV contents. The deterministic hash remains `8CCF23A5...77D25`.
- Evidence: `results/20260904_480000_full_graph_inject_full_pre_slot154_arena_rx9070xt/`,
  `results/20260904_490000_full_graph_inject_full_arena_raw_resource_rx9070xt/`, and
  `results/20260904_500000_full_graph_postblock_param56_fix_rx9070xt/`.

## R-110 RGB-only reanalysis — FINAL IMAGE STILL FAILS
- Constant alpha=1 invalidated the former aggregate RGBA correlation/NRMSE gate.
  The corrected analyzer uses RGB normatively and records alpha separately.
  Exact-state slot 154 is `1.105763648` NRMSE / `0.421357178` correlation; the
  injection-free complete graph is about `1.117199` / `0.427049`. Therefore the
  post-block parameter fix is real but does not close image parity.
- The complete 156-slot graph still executes twice on RX 9070 XT without RTX
  injection and repeats bitwise. S7/S8 and live game-frame integration remain
  open. All 158 Python tests pass.

## R-111 Returned post-block trace — ARITHMETIC PASS, RESOURCE CONTROL FAIL
- The RTX 5070 trace is deterministic and complete. At both pre-store sites,
  RTX/RX RGB correlation is at least `0.999999640` and NRMSE is at most
  `0.000848467`; alpha is exact. This closes the transformed post-block arithmetic
  question but is not a final-image pass.
- The transformed RTX output fails against the original capture at correlation
  `0.421336395` / NRMSE `1.105928620`. The cause is now bounded to the resource
  model: offset 56 is a real 640x360 RGBA16F SRV with point/border sampling, not
  a null texture. Substituting final-output pixels as the texture is also rejected
  (`0.826286171` / `2.152537713`).
- Evidence: `results/20260904_530000_postblock_store_trace_cross_vendor/` and
  `results/20260904_540000_full_graph_real_post_texture_rx9070xt/`. The next RTX
  package captures the missing texture bytes directly; a strict local receiver is
  ready to validate the return. S7/S8 remain false. All 162 Python tests pass.

## R-112 Slot-154 late texture capture — REJECTED; PRE-LAUNCH RETRY READY
- The v20 RTX run is authentic and completes `300/300`, but its captured texture
  is entirely zero. Substitution into the RX 9070 XT graph has no effect: two
  complete deterministic 156-slot runs retain hash `8CCF23A5...77D25` and final
  RGB correlation/NRMSE `0.427048989`/`1.117199202`.
- The capture point, not the replay mechanism, is therefore the immediate issue.
  v21 inserts the readback immediately before frame-1 slot 154 on its command
  list and rejects zero-only evidence. S7/S8 remain false pending this return.
- Evidence: `results/20260904_550000_post_texture_snapshot_rtx5070/`,
  `results/20260904_571000_full_graph_captured_post_texture_rx9070xt/`, and
  `results/20260904_124828_module_trace_d3d12_selftest/`.

## R-113 Slot-154 pre-launch texture — VALID ZERO; LOWERING SWEEP READY

- v21 completes RTX 5070 Feature 18 `300/300` and captures the slot-154 SRV on
  the same command list immediately before its frame-1 launch. All 1,843,200
  bytes are zero, while resource creation, SRV, merged texture handle and launch
  binding form a consistent trace. The original copy input/output remain exact
  at `CD556E0D...71FD243`. Thus R-112's stale-clear explanation is superseded:
  zero is the real input, and the resource-content hypothesis is rejected.
- The next package tests seven cumulative pre-surface-lowering forms in one RTX
  run: original, compat, E4M3, movmatrix, FP8 MMA, FP16 MMA and decoder compat.
  Each runs twice with a native CUDA RGBA16F output surface and all-zero native
  texture. The original must be byte-exact; the first later output change names
  the next correction target. Package:
  `deliverables/postblock_native_stage_sweep_reference_20260904_131704.zip`,
  SHA-256 `E2B1155B4AADD6BA7F10059CFD32D863428616E66B606CC088E16F6D7510DDA7`.
- Evidence: `results/20260904_600000_post_texture_prelaunch_rtx5070/`. This does
  not close slot 6, S7/S8, multi-frame generalization or game integration. All
  164 Python tests pass.

## R-114 Native stage sweep — EARLY TRANSFORMS EXACT; BASELINE STATE INCOMPLETE

- The returned RTX 5070 sweep is deterministic. Original, compat, E4M3 and
  movmatrix all produce the identical hash `8ADB4DE9...E6E8C6`, so these stages
  do not cause the observed output change.
- The original standalone result itself fails against the live final copy at
  RGB correlation/NRMSE `0.421336375`/`1.105928784`. This invalidates the
  bounded-window activation arena as a complete native baseline and prevents
  assigning blame to a numerical lowering at this point.
- FP8 MMA and the two successor forms fail NVIDIA JIT with invalid PTX; this is
  tracked separately. v22 captures the entire pre-slot154 activation resource
  in one RTX Feature-18 run. Package:
  `deliverables/dlssnr-windows-feature18-slot154-full-activation-v22-20260904_134339.zip`,
  SHA-256 `AE02FEB20041F8D38BABF15B53BEC75218341A1C54CB475026C5F384CCD49A63`.
- Evidence: `results/20260904_610000_postblock_native_stage_sweep_rtx5070/`.
  All 165 tests pass; S7/S8 and live game integration remain open.

## R-115 Full activation replay — HYPOTHESIS REJECTED; ORACLE TIME MISALIGNED

- v22 passes strict ingestion. The complete frame-1 activation resource is
  27,807,744 bytes (`925AE733...6CD33`); it differs from the bounded reconstruction
  in 11,876 overlap bytes.
- Injecting this complete resource immediately before slot154 on RX 9070 XT does
  not repair the comparison. Both 156-slot runs are bitwise deterministic with
  final hash `EA6CF1CB...C0EEBA`, RGB correlation `0.421357178` and NRMSE
  `1.105763648`. The missing bounded bytes are not the cause.
- The reference pairing itself was invalid: graph state and activation came from
  frame 1, whereas `copy_input.raw` was read after the 300-evaluation run. Slot154
  parameters +88/+96 are zero in frame 1 but hold live texture handles in frame
  61, directly proving temporal state changes.
- v23 captures slot154's output on the same command list immediately before
  frame-1 slot155, preserving the prior frame-1 activation and zero-texture
  captures. Package:
  `deliverables/dlssnr-windows-feature18-frame-aligned-output-v23-20260904_141326.zip`,
  SHA-256 `4B03F9655FEBD24C168D6EB535F5979EA3EF72B3378269193C08E17C46619592`.
- Evidence: `results/20260904_620000_post_activation_snapshot_rtx5070/` and
  `results/20260904_630000_full_graph_live_pre_slot154_activation_rx9070xt/`.
  All 167 tests and the D3D12 tracer self-test pass. This remains diagnostic;
  S7/S8 and game integration are still open.

## R-116 Frame-aligned slot154 output — ACCEPTED; ZERO-SURFACE ASSUMPTION OPEN

- v23 passes strict ingestion. The frame-1 output has hash
  `94A5E9E6...BA5DF3`; the end-of-run output remains the historical
  `CD556E0D...71FD243`. They differ in 1,473,125/1,843,200 bytes, conclusively
  rejecting the old cross-frame comparison.
- Against the corrected frame-1 oracle, the RX 9070 XT replay with the complete
  activation buffer still fails at RGB correlation `0.338362441` / NRMSE
  `1.497234543`. This means temporal alignment was a real defect but not the
  only missing standalone contract state.
- The 184-byte slot154 parameter block matches the original full-graph capture
  in every non-address byte. Four changed fields are GPU pointers rebased by the
  runtime, so scalar drift is rejected.
- The native probe currently clears its CUDA output surface before launch. v24
  captures the corresponding D3D12 output resource immediately before slot154
  as well as immediately after it; the changed-byte map will show whether the
  kernel preserves prior pixels. Package:
  `deliverables/dlssnr-windows-feature18-post-surface-before-after-v24-20260904_143351.zip`,
  SHA-256 `A765AC98799997C476C86FB3CE049B53F0C65D621478F124A701760A3CA9282A`.
- Evidence: `results/20260904_640000_frame_aligned_post_output_rtx5070/` and
  `results/20260904_650000_frame_aligned_post_output_reanalysis/`. All 168 tests
  and the D3D12 self-test pass; S7/S8 remain open.

## R-117 Slot154 surface before/after — SUPERSEDED INLINE SNAPSHOT

- v24 passes structural ingestion. The inline surface hashes are
  `8DF6D450...E88A3` before slot154 and `E8F7E0E3...CFF24E` after it. Only
  677,929 bytes (`36.780002%`) change. E-116 later shows the after image is an
  incomplete wavefront, so this does not imply conditional stores.
- The captured before image is byte-identical to the integrated plan's existing
  surface slice at activation-arena offset 27,807,744; both are entirely zero.
  Zero initialization is therefore correct. The correlation `0.338362441` /
  NRMSE `1.497234543` comparison against v23 is withdrawn because v23 is also an
  incomplete inline snapshot.
- The reference probe now supports `--n1-rgba16f-surface-initial`. A compact RTX
  package runs original, compat, E4M3 and movmatrix twice against the exact
  same-frame contract; the original must reproduce the live after image exactly.
  Package:
  `deliverables/postblock_exact_contract_reference_20260904_144431.zip`, SHA-256
  `7B069AC7EFA20B07D7B2A431154DA2F9908FE9618BC90A0F699D7C8E1E01D7DD`.
- Evidence: `results/20260904_660000_frame_aligned_post_surface_rtx5070/`.
  All 168 tests pass and the updated probe builds successfully. S7/S8 remain open.

## R-118 Native exact contract — ARITHMETIC MATCHES INLINE WAVEFRONTS; v25 READY

- The returned exact-contract archive SHA-256 is
  `8D789F064BF8C53A80110744C1B0D952CE80BD2AB483B5C6B2EBB1CF10D953A2`.
  All four stages execute twice on RTX 5070 and produce the same deterministic
  full-surface hash `8ADB4DE9...E6E8C6`.
- `scripts/analyze_postblock_surface_contract.py` shows the native original is
  bitwise exact for all 370,428 FP16 components changed in v23 and all 389,845
  changed in v24. The mismatch is exclusively where the inline snapshot remains
  zero while the native launch has completed output.
- v23 and v24 have different four-row-aligned wavefront boundaries despite
  identical complete activation, parameters, model weights, zero texture and
  zero destination. They are classified as synchronization-perturbed partial
  readbacks, not valid final references.
- v25 disables the four inline snapshot paths, executes only frame 1, and obtains
  slot155 input/output through a new command list followed by a queue fence.
  Package:
  `deliverables/dlssnr-windows-feature18-deferred-frame1-output-v25-20260904_152127.zip`,
  SHA-256 `E97A0D7A7FCDD3D0C0C4C7E5EED4002E92C15A370526EF9E45A1201C9B8AF373`.
  Its return receiver is `scripts/process_deferred_frame1_output_result.py`.
  D3D12 self-test and all 170 Python tests pass. E-117 records the returned
  queue-complete result; S7/S8 remain open.

## R-119 Queue-complete slot154 oracle — ACCEPTED; MMA TRACE READY

- v25 passes strict ingestion. Its deferred slot155 input/output are bitwise
  identical and hash to `8ADB4DE9...E6E8C6`, exactly matching the independent
  native original-PTX result. The v23/v24 inline snapshots are definitively
  superseded.
- Reanalysis of the RX 9070 XT exact-activation output against this complete
  oracle gives RGB correlation `0.9895909773328502`, NRMSE
  `0.1442672662285069`, exact fraction `0.4637847222`, exact-or-adjacent fraction
  `0.4927546296`, zero nonfinite pairs and exact alpha. The strict S8 gate still
  fails narrowly.
- An exact-state full-grid pre-store trace reconstructs all 640x384 output pixels.
  Its visible 640x360 bytes equal the direct fully lowered AMD output exactly
  (`EA6CF1CB...C0EEBA`). This rules out surface-resource addressing, FP32-to-FP16
  rounding and writeback as the cause of the residual.
- The next package records all FP8/F16 MMA inputs and outputs for CTA `(70,26,0)`,
  selected from the largest per-CTA output RMSE. Package:
  `deliverables/postblock_mma_trace_reference_20260904_154238.zip`, SHA-256
  `136645005EA17896B7B6752CC4FAB68CA525F367F0401D5382F6CCC5AA319EDC`.
  README contains the exact command. The D3D12 self-test and all 173 Python tests
  pass. S7/S8 and live game integration remain open; slot 6 is still the
  independent earliest upstream graph failure.

## R-120 Post-block MMA trace — FP8/F16 PRIMARY DEFECT REJECTED; PRE-FP8 TRACE READY

- The returned package passes every strict control. RTX repeat traces are
  bitwise identical, and instrumentation leaves the complete output exactly at
  `8ADB4DE9...E6E8C6`.
- The first apparent result difference occurs at FP8 MMA 0, but its warp inputs
  are not identical: A0 byte 3 differs at lanes 9 and 29 (`1B` versus `1A`).
  Warp-correct analysis finds 16/256 FP8 operations with all inputs identical;
  every one has an identical output. There are no identical-input FP8 failures.
- All remaining 240 FP8 operations already receive divergent fragments. F16
  input equality is only `0.63720703125` at the local-word level and no F16 MMA
  has a fully identical input warp, so its output differences are downstream.
  The primary defect is now localized before FP8 MMA.
- Package `deliverables/postblock_e4m3_mov_trace_reference_20260904_171152.zip`
  (SHA-256 `C48903733108762C39CD04D5C67F57BA79F9BA5D27816EA689A44775DBDFF9DE`)
  captures all E4M3 conversion and movmatrix boundaries at CTA `(70,26,0)`.
  All 175 Python tests pass. S7/S8 remain open.

## R-121 Post-block E4M3/movmatrix trace — MATCHED-INPUT PARITY PASS; F16x2 TRACE READY

- The returned archive passes every strict receiver check. RTX traces repeat
  bitwise with SHA-256 `89F676EC...FD449`, and both instrumented executions
  preserve the definitive output `8ADB4DE9...E6E8C6`.
- E4M3 has 4,702 differing input words and 792 differing output words, but zero
  equal-input/unequal-output records. Movmatrix has zero unequal-output operations
  among complete matching input warps. Neither lowering is the primary defect.
- The earliest visible mismatch is E4M3 conversion 4, lane 1, where the low half
  is `BD09` on RTX and `BD08` on RX. The output code is equal at this site; later
  input differences survive quantization and feed the FP8 MMA fragment mismatch.
- A full scalar-RN lowering of all packed-F16 operations runs on RX but produces
  the unchanged direct output hash `EA6CF1CB...C0EEBA`, correlation
  `0.9895909773328502` and NRMSE `0.1442672662285069`. Thus indiscriminate
  scalarization is not promoted as a fix.
- Package `deliverables/postblock_f16x2_trace_reference_20260904_174308.zip`
  (SHA-256 `5862355F9764C6E2224DEB1A2171611B214A00AD51B54439688AA0C4061E5F77`)
  captures all 1,522 packed-F16 arithmetic boundaries at the same CTA. The RX
  counterpart already executes successfully and produces 974,080 trace bytes.
  S7/S8 and live game integration remain open; slot 6 is still the separate
  earliest upstream graph failure.

## R-122 Full F16x2 trace — REJECTED FOR PERTURBATION; r944 PATH TRACE READY

- The full 1,522-operation package executes twice on RTX 5070 and produces a
  deterministic trace, but changes the complete output from `8ADB...E6E8C6` to
  `9A3D...3B834`. It fails the predefined output-preservation control and is not
  accepted for causal inference.
- The rejected RTX trace is bitwise equal to the RX trace. This is useful as a
  warning that high-density global-store instrumentation can alter this kernel,
  but it cannot be interpreted as proof that the unmodified paths match.
- The replacement package records only `%r816`, `%r817`, and `%r944`, reducing
  trace storage from 974,080 bytes to 1,920 bytes. The RX version executes and
  retains output hash `41AE1A6A...EF1598` for the padded surface, identical to
  the previous AMD baseline.
- Package:
  `deliverables/postblock_f16x2_path_trace_reference_20260904_181851.zip`, SHA-256
  `60E49965ECCFF48824D3D04FB255B0F2DFBBD0ABD045421FC83CC5A9F168FB55`.
  Its two RTX runs must retain `8ADB...E6E8C6` before any numerical comparison
  is accepted. S7/S8 remain open.

## R-123 Three-site F16x2 path trace — REJECTED; PRE-E4 REGISTER SNAPSHOT READY

- Reducing instrumentation from 1,522 producer sites to three does not restore
  the original RTX result. Both runs return `9EC0...42530` instead of
  `8ADB...E6E8C6`, so the package is rejected under the same predefined control.
- Its trace is nevertheless deterministic and byte-identical across RTX and RX.
  Because the output contract failed, this equality is classified as probe
  perturbation and is not used to claim arithmetic parity.
- The new package moves all stores away from the packed-F16 producers and reads
  the still-live dependency registers once immediately before E4M3 conversion 4.
  It can distinguish input divergence, either multiply, the add, and conversion
  while minimally changing the already validated E4 boundary.
- Package:
  `deliverables/postblock_r944_snapshot_reference_20260904_183411.zip`, SHA-256
  `77443FB214E88D084A3E114A028A6B510FF42D12FA6B28FC3009AECA5179A5AD`.
  S7/S8 remain open.

## R-124 Pre-E4 snapshot — REJECTED TRACE; EXACT-INPUT POST-BLOCK GATE PASS

- The returned snapshot package executes twice on RTX 5070 and repeats exactly,
  but changes the output to `D823...DE1D`. It fails the mandatory preservation
  check against `8ADB...E6E8C6`; apparent RTX/RX trace equality is not accepted
  as native parity.
- The accepted original-path E4 trace already contains the target `%r944` words.
  Testing its values against the snapshot operands identifies one unique 32/32
  lane model: round `%r204*%r205` to F16, keep `%r720*%r721` in FP32, fuse the
  latter product into the add, then round the sum to F16.
- Applying this contract to the first 32 structural candidates improves the
  exact-state RX output from correlation `0.989590977` / NRMSE `0.144267266` to
  `0.999161906` / `0.040940560`. Exact fraction is `0.961141493`,
  exact-or-adjacent fraction is `0.965296586`, alpha is exact, and there are no
  nonfinite pairs. The predefined post-block parity gate passes.
- Applying the rule to all 64 candidates is slightly worse; the final promoted
  scope is therefore first 32 only. The integrated plan executes 156/156 slots
  twice on RX 9070 XT and is deterministic, but its native full-frame comparison
  still fails because divergence begins upstream (fine slot 6; coarse slot 15).
  This result closes the exact-input post-block defect, not S7/S8 or live-game
  integration.

## R-125 Slot-3 delayed normalization snapshot — RTX ORACLE READY

- Upstream work resumes at the earliest fine-checkpoint failure, slot 6. Exact
  slot-5 injection makes slots 6–8 pass, so the current target is the slot-3–5
  1h state feeding it, not a blanket rewrite of the 2h family.
- A 31-register snapshot has been moved away from packed-F16 producer sites to
  the previously output-safe E4M3 boundary. The RX 9070 XT version executes the
  full slot-3 grid and preserves the prior AMD output hash exactly.
- `deliverables/slot3_delayed_norm_snapshot_reference_20260904_185812.zip`
  contains a two-run, output-preservation-gated RTX experiment and its exact
  PowerShell command. A returned package is admissible only if both outputs stay
  at `33FE6004...50BD4F9` and the two traces repeat bit-for-bit.

## R-126 Square-pair fusion — SLOT 6 PASSES; CURRENT ROOT RETURNS TO N0

- A generalized half2 square-pair fusion is integrated into slots 2-6. The
  injection-free 156-slot RX 9070 XT replay completes twice deterministically.
  Slot 6 falls from the old `0.102493965` NRMSE failure to `0.042475154`; slot 7
  is now the first failing fine boundary at `0.110049329`. Slot 9 remains passing
  at `0.049582652`.
- This is not an empirical-only tweak. On the earlier accepted same-input
  isolated oracle, slot 2 becomes byte-exact across all 1,966,080 output bytes.
  Slot 3 independently matches all 896 traced 32-bit path values and its complete
  output byte-for-byte (`33FE6004...50BD4F9`).
- Two explicitly non-S7 causality runs inject exact RTX state after slot 1 or
  slot 2. In either case slots 2-6 become bitwise exact, slots 7-9 pass, and the
  first failure moves to slot 11. This demonstrates that the native slot-7
  residual is the remaining N0 error being amplified, not a reason to continue
  blind slot-by-slot rewrites in the 1h/2h chain.
- The critical path therefore returns to the 154-byte N0 residual. Its known
  first remaining branch is `sin0 -> normal1_f16`. The new full-grid package
  `deliverables/n0_normal1_full_grid_traces_reference_20260904_193326.zip`
  captures phase/sine and sqrt/consumed-FP16 over all 245,760 samples with
  independent output-preservation controls. SHA-256 is
  `4545FFB1F2DB418F7C15866E3C1CDE532BAF6461824E69B0A5E52BC27C9DE71A`.
  S7/S8 and live-game integration remain open.

## R-127 Normal1 full-grid correction — N0 116 BYTES; SLOT 7 AT 0.10374

- Returned archive `_n0_normal1_full_grid_traces_reference_result_20260904_194712.zip`
  has SHA-256 `5E36BD452D163365662B5125D53072FA544308BEE57B906C2F97ECC3625779CB`.
  RTX 5070 identity, package integrity, both launches, repeatability and both
  untouched-output controls pass.
- Across 245,760 samples, phase inputs are bitwise exact. RTX/RX sine differs in
  206,632 samples and sqrt in 87,577, while the consumed FP16 normal differs in
  only 380. Replacing only sine with RTX values leaves 43 theoretical mismatches;
  replacing only sqrt leaves 358, so sine is the dominant source.
- A boundary-safe 65,536-segment sine model reduces the complete RX N0 output
  from 154 to 126 differing bytes on the captured zero-input contract, and from
  168 to 126 on an independent nonzero input, adding no new differences. Sparse
  per-segment sqrt ULP shifts improve those further to 116 and 109 bytes,
  respectively, again with zero additions.
- The promoted candidate runs all 156 graph slots twice in one RX 9070 XT
  context without RTX intermediate injection and repeats exactly. Fine-boundary
  NRMSE is slot 1 `0.000290588`, slot 2 `0.000747672`, slot 5 `0.007999347`,
  slot 6 `0.039886535`, slot 7 `0.103740020`, slot 8 `0.140209500`, and slot 9
  `0.047513142`. Slot 7 remains the first strict failure, but is now only
  `0.003740020` above the threshold.
- This is a genuine numerical advance, not S7/S8 completion: final RGB remains
  outside the image gate, and live NGX game integration has not been achieved.

## R-128 Hierarchical cos1/sin0 refinement — SLOT 7 PASSES; SLOT 8 IS NEXT

- The normal1 high-resolution search was narrowed on the real RX 9070 XT from
  775 grouped overrides to the single useful segment 55169. It reduces N0 from
  116 to 108 differing bytes on zero input and from 109 to 105 on the independent
  nonzero input, with no new mismatches in either comparison.
- A separate 524,288-segment normal2/cos1 model preserves the accepted 32K
  behavior and admits only boundary-safe high-resolution overrides. With both
  corrections and the existing sqrt ULP model, N0 reaches 83 differing bytes on
  zero input and 86 on nonzero input. Relative to the segment-55169 baseline it
  fixes 25/19 old bytes and adds zero.
- The injection-free full graph completes 156/156 launches twice and is bitwise
  deterministic. Slot 6 passes at `0.034749463`; slot 7 falls from `0.103740020`
  to `0.089691090` and now passes. Slot 8 is the new first strict failure at
  `0.123923003`; slot 9 is `0.043610627` and passes.
- The final RGB comparison still fails (correlation `0.428485569`, NRMSE
  `1.102721636`). This is an upstream numerical milestone, not yet a complete
  frame or game-ready DLSS 5 implementation.

## R-129 Joint cos0/sqrt0 refinement — SLOT 8 AND SLOT 10 PASS

- The first-cosine hierarchy reduces the dual-input N0 residual from 83/86 to
  64/68 bytes without additions. Analysis of its first remaining output error
  shows a coupled rounding boundary: both cos0 and sqrt0 must move by one ULP;
  either change alone leaves the consumed FP16 value unchanged.
- A full-grid joint model evaluates both normal0 and normal1 before accepting a
  phase segment. Ten segments pass the zero-addition rule. The real RX 9070 XT
  output reaches 29 differing bytes on the zero-input contract and 46 on the
  independent nonzero capture, fixing 35/22 old bytes with no new mismatches.
- In the complete injection-free graph, slots 6/7/8/9/10 all pass at NRMSE
  `0.023386232`/`0.061686839`/`0.086983541`/`0.036550767`/`0.096071659`.
  Slot 11 is now the first failure at `0.134164531`. Both 156-launch runs repeat
  bitwise on RX 9070 XT.
- The final RGB output remains outside the image gate (correlation
  `0.435413076`, NRMSE `1.111083053`). The next target is the slot-10-to-slot-11
  transition, not further blind N0 fitting.

## R-130 Exact 4h controls — SLOT 11 IMPLEMENTATION CLEARED; NATIVE BASELINE UNCHANGED

- Exact RTX slot-10 output injection is diagnostic only and cannot count as S7.
  It makes every slot 11-15 boundary pass and moves the first failure to slot 23.
  Slot-11 NRMSE falls from `0.134164531` to `0.008314572`.
- Exact RTX slot-9 input injection leaves slot 10 and all later work on RX. It
  makes slot 10 pass at `0.012098444` and slots 11-13 pass, with the first
  failure delayed to slot 14 at `0.113606190`. This independently confirms
  that the 4h kernels have acceptable exact-input semantics and primarily
  amplify inherited state differences.
- Applying the previously proven square-pair fusion to slot 10's 32 matching
  sites does not solve the native graph. Slot 10 changes from `0.096071659` to
  `0.096188659`, slot 11 changes from `0.134164531` to `0.133910018`, and the
  first failure remains slot 11. The candidate is rejected rather than promoted
  on the basis of the tiny downstream improvement.
- The accepted injection-free result remains
  `results/20260905_043000_full_graph_fine_n0_jointulp_slots2to6_square_rx9070xt/`.
  Work now targets the sparse slot-9 input residual and an output-preserving
  slot-10 internal trace; live game integration and S7/S8 remain open.
- The first exact-input slot-10 mismatch is localized to CTA `(4,0)`, warp-y
  `1`. The new all-272-MMA tracer is AMD-output-preserving and its verified RTX
  package is
  `deliverables/swin4h_slot10_fp8_mma_trace_reference_20260904_215525.zip`
  (SHA-256 `E6953608A3D8C003F44D591A6AF462A7442DB1AE15C7C6731908D32240ECB7A5`).
  Its README includes the one-line command; the returned trace will identify
  the first A/B/C/D divergence instead of testing more global fusion guesses.

## R-131 AMD-native output head — COMPLETE SLOT-154 OPERATOR PASS

- The output head is now reconstructed end to end from the captured 27,807,744-byte
  activation arena and the user-local model arena. One command executes nine
  dependency-ordered stages on RX 9070 XT; all nine pass and repeat
  deterministically: `results/20260905_013000_output_head_pipeline_rx9070xt/`.
- The final 16 FP16 MMAs use the recovered 32x16 padded weight tile and produce
  `49x81x64x4` half residuals. Against 256 selected RTX half values, mean/max
  absolute error is `0.020706/0.371094`; all accumulator seeds chain exactly.
- Symbolic and trace validation reduce the PTX shuffle to four 4x4 tiles within
  each 8x8 CTA. The selected RTX surface trace has zero coordinate/marker
  mismatches and proves the two-fused-operation store is algebraically
  `clamp(base + 0.25*residual, 0, 1)`, alpha one. All 192 RTX RGB formula values
  reproduce exactly. Native AMD final RGB mean/max error is only
  `0.000159/0.000764`.

## R-132 D3D12/HIP game-texture bridge — PASS

- `results/20260905_012000_d3d12_residual_bridge_rx9070xt/` executes a real
  `640x360` `DXGI_FORMAT_R16G16B16A16_FLOAT` input texture through the supported
  staging path: D3D12 texture -> shared linear heap -> HIP composition -> shared
  heap -> D3D12 texture.
- The shared D3D12 fence is imported into HIP and used for GPU-side wait and
  signal. Readback has zero mismatches across all 921,600 components, no
  non-finite values, and 668,907 RGB components differ from the base image.
  The composition kernel averages `0.011183 ms` over 100 iterations on RX 9070 XT.
- This closes the standalone slot-154 output and texture-transport gate, not S7.
  The nine-stage runner still uses separate processes and disk intermediates.
  The next implementation boundary is a resident-GPU output-head pipeline,
  followed by the feature-18 resource registry and the native decoder stage that
  must produce slot 154's activation tensor without a captured input.

## R-133 Resident output head plus D3D12 texture — PASS

- `results/20260905_014000_output_head_resident_rx9070xt/` replaces the nine
  validation processes and their disk intermediates with fourteen ordered GPU
  launches in one process. All working tensors remain allocated on the GPU.
  The final `49x81x64x4` FP16 residual has zero mismatches against the accepted
  modular result and repeats bitwise. It has no runtime RTX-trace dependency and
  averages `19.828800 ms` over the final 20-iteration run on RX 9070 XT.
- `results/20260905_015000_output_head_resident_d3d12_rx9070xt/` puts that full
  resident calculation and the D3D12 bridge in the same process. The imported
  shared fence performs the D3D12-to-HIP wait and HIP-to-D3D12 signal; residual
  mismatch count is zero and the read-back RGBA16F texture matches all 921,600
  expected components. The combined GPU output-head measurement is
  `20.242809 ms` over 20 iterations.
- This removes process launch, disk-intermediate and runtime-oracle dependencies
  from the output head. It is not yet a game plugin: the validation harness still
  loads the slot-154 activation arena from disk and creates its own D3D12
  resources. The next gate is binding the feature-18 evaluate resources and
  feeding the head from the native decoder in the live resource registry.

## R-134 NVAPI slot-154 neural dispatch — ADDRESS-LOWERED PATH PASS

- `results/20260905_016000_nvapi_amd_output_head_rx9070xt/` invokes the output
  head through the actual R610 `LaunchCuKernelChain` compatibility entry using
  the real function name, launch shape and 184-byte captured parameter block.
  Five GPU virtual addresses resolve through two registered D3D12 resources;
  translation failures are zero and `neural_math_executed` advances to one.
- The produced 640x360 RGBA16F surface has zero byte mismatches against R-133
  and hash `A9CD87746E10C577`. The first host-observed call is `52.8233 ms`
  because it includes lazy allocation; it is not a steady-state GPU benchmark.
  The pre-existing NVAPI clear and texture/descriptor copy tests both remain
  green after adding the neural branch.
- This self-test deliberately replaces the captured surface token at parameter
  offset 16 with a registered linear output address. The real RTX ABI uses the
  independent surface object `0x9802` there and a merged texture/sampler object
  at offset 56. Therefore R-134 proves dispatch and buffer translation, not a
  live game evaluate. The next implementation must retain descriptor-to-resource
  identity and record the staging-to-texture copy in D3D12 queue order.

## R-135 NVAPI slot-154 real surface object — PASS

- `results/20260905_020645_nvapi_amd_output_surface/` removes the synthetic
  output address from R-134. Parameter offset 16 is now an actual independent
  surface-object handle backed by a committed 640x360
  `DXGI_FORMAT_R16G16B16A16_FLOAT` UAV.
- The test contains no direct buffer or descriptor registration calls. Under
  explicit `MODULE_TRACE_AMD_INTEROP=1`, the tracker promotes and automatically
  imports the captured-size 27,807,744-byte activation and 147,719,680-byte
  model buffers, and registers the output descriptor's resource identity with
  the AMD backend. Its trace reports two successful buffer registrations and
  `backend_registered:true` for the surface.
- Four address translations pass, neural execution advances once, and one
  D3D12 placed-footprint texture copy is recorded. The final 1,843,200 output
  bytes match R-133 exactly with hash `A9CD87746E10C577`.
- This closes the live resource-identity portion of slot 154, but not queue
  ordering or the native full network. The current HIP call is synchronous on
  the recording thread; it cannot yet consume work that appears earlier in the
  same unsubmitted D3D12 command list. Slots 0-153 also remain marker/standalone
  implementations rather than the live activation producer, so the result is
  deliberately `game_runtime_ready:false` and does not count as S7.

## R-136 D3D12 queue/list timeline — LOCAL INSTRUMENTATION PASS

- `results/20260905_022254_module_trace_d3d12_selftest/` records command-list
  creation, close and queue submission with stable object identities and a
  single sequence domain shared with neural launches.
- The test records 1 queue creation, 2 command-list creations, 2 closes, 2
  execute calls and 2 submitted list items. Existing copy identity, N0 capture,
  full-graph capture and descriptor/resource checks remain passing.
- `deliverables/dlssnr-windows-feature18-submission-topology-v26-20260905_022332.zip`
  is the required single-run RTX experiment (SHA-256
  `D4A56F156B5E44AB6C494569E1CEA36C32B9443DAF7186A225781846FDB6D1A1`).
  Its result will select between an external split/fence design and execution
  inside the host's D3D12 command list; neither design is yet claimed complete.

## R-137 RTX feature-18 submission topology — SAME-LIST PATH REQUIRED

- `results/20260905_025006_rtx_feature18_submission_topology/` is a complete
  PASS return from v26. It contains 156 unique frame-1 slots on one graphics
  command list; slots 154 and 155 are the final two launches.
- Timeline: slot154 `159` -> slot155 `160` -> list close `161` -> queue submit
  `162`. The completed 640x360 RGBA16F slot154/155 resources remain bitwise
  identical with the accepted native RTX hash `8ADB4DE9...E6E8C6`.
- This rejects the simple external-fence branch: at slot154 the D3D12 producer
  work is unsubmitted, so HIP cannot safely consume it. The implementation
  queue now targets D3D12 compute recorded into the same command list, beginning
  with the resident output-head stages and then the upstream producer families.

## R-138 D3D12-native output surface stage — PASS

- `results/20260905_025647_output_head_surface_d3d12_rx9070xt/` dispatches the
  final output-head surface/composition stage as DXIL on RX 9070 XT and reads it
  back from the same D3D12 command list.
- The 2,032,128-byte native residual input produces a 1,843,200-byte RGBA16F
  output with zero byte mismatches against R-133. This stage no longer requires
  a HIP queue, external memory import, or an external fence.
- DXIL's direct `f32tof16` disagreed at 1,709 positive subnormal components; the
  accepted shader uses explicit IEEE round-to-nearest-even conversion. The next
  migration target is the FP16 tail projection that produces this residual.

## R-139 D3D12-native FP16 tail + surface — NUMERICAL PASS

- `results/20260905_033200_output_head_tail_surface_d3d12_rx9070xt/` records the
  final two output-head stages into one RX 9070 XT D3D12 command list. A resource
  transition orders the residual UAV write before the surface SRV read; no HIP
  queue or external fence participates.
- The native tail is deterministic and finite. Against the accepted HIP
  residual it differs in 2,286/1,016,064 half values (`0.224985828%`), with
  mean/max absolute error `3.43386189e-8 / 0.00390625`.
- The resulting 640x360 RGBA16F surface differs in 1,456/921,600 half components
  (`0.157986111%`), but mean/max absolute error is only
  `1.41295863e-9 / 3.05175781e-5`; this is below the stage's practical gate.
  The result is deliberately labeled numerical rather than byte-exact.
- The next migration target is final projection. It must produce the tail's
  32-channel FP16 input on the same list before these two accepted dispatches.

## R-140 D3D12-native final projection + tail + surface — PASS

- `results/20260905_034000_output_head_final_projection_d3d12_rx9070xt/`
  validates the final projection alone: its full 16,257,024-byte output is
  byte-exact against the accepted RX 9070 XT HIP oracle.
- `results/20260905_034100_output_head_final_tail_surface_d3d12_rx9070xt/`
  then records final projection, FP16 tail, and surface composition on one D3D12
  command list, with UAV-to-SRV transitions between the stages and no HIP
  runtime dependency.
- Projection remains exact. Tail and final surface retain precisely the R-139
  numerical result (2,286 and 1,456 half mismatches respectively), so command-
  list fusion introduces no new arithmetic or ordering difference.
- The accepted same-list suffix now begins at the attention tensor. The next
  implementation target is the softmax/V attention producer; live game readiness
  stays false until the remaining upstream graph and slot interception use the
  same ordered D3D12 path.

## R-141 D3D12-native attention suffix — PASS

- `results/20260905_035100_output_head_softmax_v_d3d12_rx9070xt/` validates the
  full-grid softmax/V stage: 96/8,128,512 attention half values differ from the
  HIP oracle, while mean/max absolute error is only
  `6.14968557e-8 / 0.0234375`.
- `results/20260905_035300_output_head_attention_suffix_d3d12_rx9070xt/` chains
  softmax/V, final projection, FP16 tail and surface composition in one D3D12
  command list with explicit resource transitions and no HIP dependency.
- The four-stage output remains bounded: 54 projected, 2,293 residual and 1,461
  surface half mismatches. Final surface mean/max absolute error is
  `1.89543546e-9 / 0.000163555145`, with no non-finite values.
- The migration frontier moves to Q/K/V preparation and QK scoring. This is a
  stronger same-list operator-chain result, but `game_runtime_ready` remains
  false until that producer and the earlier graph are live on the same path.

## R-142 Complete D3D12 output head — PASS

- `results/20260905_041000_output_head_full_d3d12_rx9070xt/` executes the whole
  reconstructed output head from activation-arena input to RGBA16F output using
  original weights, 11 ordered DXIL dispatches and one command list. HIP and RTX
  trace inputs are absent from the runtime path.
- The early first128 and MMA128-175 checkpoints are exact. Numerical differences
  begin at Q/K normalization and QK scoring, but the final surface mean/max error
  remains `3.4076811e-6 / 0.000862598419`, with no non-finite output.
- `results/20260905_041100_output_head_full_d3d12_rx9070xt_repeat/` reproduces all
  mismatch counts and metrics exactly. The complete Python regression suite also
  remains green at 222 tests.
- Output-head migration is no longer a layer-by-layer blocker. The active task is
  to record this complete executor against the real slot-154 activation, model,
  base texture and UAV resources supplied by the intercepted game command list.

## R-143 Caller-owned D3D12 output-head recording — PASS

- `results/20260905_042914_output_head_recording_selftest/` runs A/B/C/A twice:
  original captured main/zero base, patterned base, zeroed main/zero base, then
  restored original inputs. Eight invocations use one caller-owned command list,
  replayed in two fence-ordered submissions, with the same scratch allocations.
- Main/skip share the activation buffer at real nonzero offsets. All activation,
  model and base producer copies are queued ahead of consumers on that list;
  inputs are not submitted before the output head is recorded. The reusable
  component restores external resource states and writes a real RGBA16F texture.
- The original and restored outputs match the prior standalone DXIL bytes
  exactly. Changed main and base inputs change the output; submission repeats
  and residual repeats are exact. CPU scatter/composition from each computed
  residual has zero byte mismatches; all residual/output values are finite.
  Four invalid bindings are rejected before recording and the enabled D3D12
  debug layer reports zero errors.
- Second-submission timestamp durations are 44.9135, 44.1035, 41.3804 and
  39.6348 ms. These include base/output staging and diagnostic residual copies,
  with debug validation enabled; they are not a release benchmark or game FPS.
- This closes the reusable recorder/controlled texture-reuse test. The NVAPI
  slot154 branch remains HIP, and upstream native producers plus live multi-frame
  quality/performance are still outstanding. `game_runtime_ready:false`.

## R-145 Automatic head submission/lifetime controls — PASS, head only

See E-142 and `results/20260905_050734_nvapi_output_head_d3d12/`: 1,000 head calls,
250 retired sessions, no byte mismatches, debug errors or remaining sessions.
Additional chain/batch rejection checks pass in `20260905_051316...`.
This is not a game-frame stability pass.

## R-146 Actual AMD upstream to DXIL head — execution PASS, image FAIL

See E-143 and `results/20260905_050700_hybrid_full_frame_validation/`.
Same-frame RGB correlation 0.683468, NRMSE 0.812474; exact repeated output.
The near-black reference makes unit-range PSNR unsuitable as the sole gate.

## R-147 Native 1h/32 family — local PASS, integrated candidate REJECTED

See E-144. Four encoder/decoder cases pass at NRMSE 0.01769-0.02580. However,
the injection-free integrated variant plus DXIL head regresses to correlation
0.638833 / NRMSE 0.854350. It remains opt-in and does not replace the baseline.
Native graph coverage, image quality, temporal modes and game readiness remain
incomplete. The next work must retain a full-frame gate, not promote local passes.

## R-144 NVAPI slot154 dispatch records the complete DXIL head — PASS

- `results/20260905_044156_nvapi_output_head_d3d12/` exercises the real NVAPI
  query/launch ABI with relocated 184-byte captured parameters. An explicit host
  session provides actual resource states and a future queue-completion fence.
- Two distinct lists/sessions each record original, nonzero-base, zeroed-main
  and restored-input cases. All eight final textures match the standalone
  recorder byte-for-byte. Input copies remain pending on the same lists until
  the caller submits; the NVAPI head does not execute synchronous HIP math.
- Twelve unsupported/invalid calls and two premature releases are rejected.
  Both completed sessions release their held resources/scratch with zero active
  sessions. D3D12 debug errors and marker counts are zero. Activation/model
  buffers require no HIP import; the legacy DLL still depends on HIP.
- The original automatic module-trace/HIP output-head self-test still passes
  with zero mismatches. All 222 Python tests pass (44.73 seconds).
- This is explicit-host integration, not a live game run. State/fence registration
  is supplied by the test, and the registered head-only session rejects other
  functions and history-bearing modes. Upstream neural producers and full-frame
  quality/performance remain outstanding; `game_runtime_ready:false`.

