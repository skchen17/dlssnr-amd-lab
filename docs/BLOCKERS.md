# BLOCKERS.md

## Hard blockers (require user action)

### B-1 MISSING_PREREQUISITE: nvngx_dlssnr.dll — RESOLVED 2026-08-30
- User supplied a legally obtained copy; ledgered in docs/PROPRIETARY_FILES.md
  (v310.8.0.0, SHA-256 E16BCF15..., Authenticode Valid NVIDIA Corporation).

### B-2 MISSING_PREREQUISITE: nvngx_dlss.dll — RESOLVED 2026-08-30
- User supplied; ledgered (v310.7.0.0, SHA-256 BE6E434A..., Authenticode Valid).
- NR addon requires the DLSS SR runtime next to the host.

### B-3 NO NVIDIA GPU on this machine
- Blocks: S4 reference run, NVIDIA-side intermediate tensor capture (Track B step 5).
- Mitigation: build the reference package (nr_host + fixed inputs) to be copied to any
  NVIDIA machine; proceed with static analysis meanwhile. Do NOT fabricate reference data.

### B-4 OPTIONAL: NVIDIA NGX SDK
- Needed to compile nr_host against official NGX headers/libs (`nvsdk_ngx.lib`).
- Source: developer.nvidia.com (free account, requires user registration/license acceptance).
- Interim mitigation: nr_host skeleton builds against a minimal local NGX API declaration
  reconstructed from public docs/DLSS5-Feeder usage, clearly marked as reconstructed.

### B-5 HIP SDK for Windows (to be confirmed by Phase 0)
- If not installed: hip_probe / d3d12_hip_interop cannot compile or run.
- Source: amd.com HIP SDK installer (user installs, may require reboot).

### B-6 git identity not configured
- `git commit` fails with "Author identity unknown". User should run:
  `git config user.name "..."; git config user.email "..."` (repo-local, inside dlssnr-amd-lab).
- Milestone commits are queued until then.

## Soft blockers (tracked, non-fatal)

### S-B1: HIP Windows external memory D3D12 NT-handle support — RESOLVED 2026-08-30
- Experiment d3d12_hip_interop PASS-A succeeded: `hipExternalMemoryHandleTypeD3D12Heap`
  (and `D3D12Resource`) accept D3D12 shared NT handles; `D3D12Fence` semaphore import and
  in-stream wait/signal work. See docs/INTEROP.md. No fallback needed.

### S-B2: ZLUDA RDNA4/ROCm version skew
- Community reports: ROCm >= 6.4.x required for RX 9070 XT; ROCm 7.1 broke one consumer app.
- Action: pin exact HIP SDK version in environment.json; keep ZLUDA and HIP SDK versions
  aligned per ZLUDA release notes.

### S-B3: WMMA output lane mapping documentation gap (ROCm issue #6025)
- Action: hip_probe validates WMMA by numeric ground truth, not docs.

### S-B4: WMMA f32 builtin ISel defect in this ROCm 7.2 LLVM (recorded 2026-08-30)
- Repro: compile tools/hip_probe/hip_probe.cpp with `-DWMMA_TEST=1 --offload-arch=gfx1201`
  (AMD clang 22.0.0git, ROCm/llvm-project 6602f325c + patch 93c451b46cc0).
- Result: `fatal error: error in backend: Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16.f16`
  (pass: AMDGPU DAG->DAG Pattern Instruction Selection on k_wmma_f16_ones_identity).
- Full log: results/20260830_170305/wmma_diag.log (exit 1).
- Consequence: spec section 6 test F is gated on "if the current HIP compiler provides
  suitable intrinsics" — it does not. S1 is judged on tests A–E (all PASS); WMMA stays
  opt-in (`-DWMMA_TEST=1`) and is recorded as UNVERIFIED, not as S-level evidence.
- Follow-up options (next round): f16-accumulator variant
  `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`, inline asm `v_wmma_f32_16x16x16_f16`,
  or a newer ROCm LLVM.

### S-B5: NGX vendor gate (recorded 2026-08-30) — current first dynamic blocker
- Fact: unmodified nvngx_dlss.dll loaded on AMD and NVSDK_NGX_D3D12_Init returned
  0xbad00001 on the vendor=0x1002 device, before loading nvapi64/nvcuda or issuing
  any observed NVAPI/CUDA call (results/20260830_170305/nr_host_forceload_stdout.log).
- Consequence: S5+ and launch-side trace require getting init past this gate.
- Constraint: per spec, "spoofing the vendor check" is NEVER success evidence.
  A spoof may only be used as a LABELED trace-aid to OBSERVE the downstream call
  chain (module loads, nvapi IDs), and every such run must say so in RESULTS.md.
- Candidate trace-aid routes (next round, all labeled): DXGI adapter desc proxy,
  OptiScaler-style FakeNVAPI only for observation, or running the same host on an
  NVIDIA box to capture the clean chain and diffing.

### S-B6: NR payload is sm_120-only SASS (recorded 2026-08-30)
- No PTX in nvngx_dlssnr.dll (verified container-aware, not ASCII-grep), so ZLUDA
  JIT is impossible for NR kernels. Track B with the named-kernel inventory
  (docs/BINARY_ANALYSIS.md) plus a CASE A transport shim is the working plan.
