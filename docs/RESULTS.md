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

## R-5 nvngx_dlssnr.dll / nvngx_dlss.dll static analysis (S3 static, GATE-0/1)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\_probe_now.ps1` (binary_probe on both DLLs); `scripts\_carve_cubin.ps1`
  + `scripts\_readobj_cubin.ps1` (llvm-objdump on carved CUBINs); `scripts\_scan_sm_targets.ps1`
- Input spec: user-supplied legally owned nvngx_dlssnr.dll 310.8.0.0 (165.8 MB) and
  nvngx_dlss.dll 310.7.0.0 (59.0 MB); ledgered in docs/PROPRIETARY_FILES.md.
- Metrics/findings:
  - NR payload: 15 CUBIN ELF modules (e_machine=190) inside .rsrc (147 MB); **zero PTX**
  - Target arch: **sm_120 only** (15 markers, one per module) — Blackwell-only, no fallback
  - Kernel names + per-kernel .nv.info parameter metadata fully preserved
    (fused Swin-Transformer backbone, FP16+FP8 sibling variants)
  - No nvcuda.dll / nvapi64.dll imports (static or delay) in either DLL
  - SR runtime: 670 ELFs + PTX 8.7 for sm_89 (contrast case)
- Verdict: PASS — GATE-0 PASS, GATE-1 FAIL → **PTX path = BLOCKED**, Track B per spec §12
- Raw data: results/20260830_170305/probe_dlssnr_stdout.log, probe_dlss_stdout.log,
  binary_manifest_dlssnr.json, binary_manifest_dlss.json, carved_cubin_{0,1}.readobj.log,
  sm_targets_scan.log; docs/BINARY_ANALYSIS.md

## R-6 first real load of the NVIDIA runtime on AMD (vendor gate)
- Date / run id: 2026-08-30 / results/20260830_170305
- Command: `scripts\_forcload_run.ps1` (= build nr_host, set DLL paths, run
  `build\nr_host.exe --frames 1 --width 512 --height 512 --trace --force-load`)
- Input spec: unmodified DLLs, no spoofing; D3D12 device on vendor=0x1002 device=0x7550;
  module_trace self-injected; nvapi shim armed (unused — gate failed first).
- Metrics/findings:
  - Both DLLs load cleanly on AMD (dlss@0x7FFB9D5E0000, nr@0x7FFB93790000, lasterr=0)
  - All 5 reconstructed NGX exports resolve in nvngx_dlss.dll (signatures correct)
  - `NVSDK_NGX_D3D12_Init` executes and returns **0xbad00001** on the AMD adapter —
    a real hardware/vendor capability check, clean rejection, process stable
  - No NVIDIA support modules (nvapi64/nvcuda) loaded before the rejection
- Verdict: PARTIAL (S5 blocked at vendor gate = BLOCKERS S-B5; the load itself is the
  first genuine runtime interaction with the leaked DLL on AMD)
- Raw data: results/20260830_170305/nr_host_forceload_stdout.log, nr_host_forceload.json,
  forcload_20260830_182045_module_trace.log

