# BLOCKERS.md

## Hard blockers (require user action)

### B-1 MISSING_PREREQUISITE: nvngx_dlssnr.dll — RESOLVED 2026-08-30
- User supplied a legally obtained copy; ledgered in docs/PROPRIETARY_FILES.md
  (v310.8.0.0, SHA-256 E16BCF15..., Authenticode Valid NVIDIA Corporation).

### B-2 MISSING_PREREQUISITE: nvngx_dlss.dll — RESOLVED 2026-08-30
- User supplied; ledgered (v310.7.0.0, SHA-256 BE6E434A..., Authenticode Valid).
- NR addon requires the DLSS SR runtime next to the host.

### B-3 NO NVIDIA GPU on this machine
- Blocks: S4 reference run, NVIDIA-side intermediate tensor capture (Track B step 5),
  and the positive control that would confirm/deny the vendor-gate reading of
  0xBAD00001 (R4–R7 = BLOCKED_EXTERNAL_HARDWARE).
- Mitigation: Round 2 delivered scripts/run_nvidia_reference.ps1 + reference_bundle/
  (environment + runtime hashes, no proprietary DLLs) so an RTX machine run is
  one command. Do NOT fabricate reference data.
- Update 2026-08-30: an external Windows RTX 5060 (`sm_120`) is now available.
  The first run reproduced `0xBAD00001` under the old lab-only App/Project IDs,
  disproving the vendor-gate interpretation of that code and moving the blocker
  to the host initialization contract. The corrected v2 run then passed Init and
  CreateFeature, and Round 3 completed 8/8 Evaluate/readback frames, removing B-3
  for public-host positive controls. Private feature 18 is still blocked on a real
  ReShade/RenoDX add-on host.

### B-7 RENO DX FEATURE-18 HOST COMPONENTS — RESOLVED 2026-08-30
- The proven standalone host intentionally did not load a ReShade runtime or
  `renodx-dlss5.addon64`; returned summary marks feature 18 as NOT_RUN.
- The vendored DLSS5-Feeder supplies a purpose-built `dlss5-feed-host64 --test`
  route, but its runtime directory must also contain a 64-bit ReShade `dxgi.dll`,
  `renodx-dlss5.addon64`, `nvngx_dlss.dll`, and `nvngx_dlssnr.dll`.
- Resolved by R-17/E-13: ReShade registered the add-on, signed DLSSNR initialized,
  feature 18 was created, and inline evaluation succeeded on RTX 5070. S4 PASS.
- R-19/E-15 captured the exact official NVAPI IDs and confirms CASE A:
  CreateCuModule/CreateCuFunction/LaunchCuKernelChain/DestroyCuFunction/
  DestroyCuModule. R-20/E-16 then proves the typed calls: 9 modules, 96 functions
  and 300×156 successful launches. R-21/E-17 captures the complete stable graph,
  parameter bytes and exact DLL payload offsets. The RTX reference blocker is
  closed for this configuration; remaining work is AMD-side graph/kernel
  reconstruction and the NVAPI-to-HIP transport implementation.

### B-4 OPTIONAL: NVIDIA NGX SDK — PARTIALLY RESOLVED (Round 2)
- Round 2 sourced the official public headers from the NVIDIA/DLSS GitHub repo
  (third_party/nvidia-dlss, commit recorded in third_party/PROVENANCE.md) plus the
  redistributable nvsdk_ngx_d.lib; nr_host now builds and links against the OFFICIAL
  ABI. If a fuller SDK (additional private headers) is ever needed, it requires user
  license acceptance at developer.nvidia.com and is supplied via NGX_SDK_PATH.

### B-5 HIP SDK for Windows (to be confirmed by Phase 0)
- If not installed: hip_probe / d3d12_hip_interop cannot compile or run.
- Source: amd.com HIP SDK installer (user installs, may require reboot).

### B-6 git identity not configured — RESOLVED (Round 2)
- Resolved without touching git config: every commit uses per-command
  `-c user.name=... -c user.email=...` (identity = Round 1's
  `skchen17 <skchen17@users.noreply.github.com>`); `-c safe.directory=...`
  also needed on this box (dubious ownership).
- Round 1 + 7 Round 2 milestone commits exist locally. Push to
  `skchen17/dlssnr-amd-lab` is DEFERRED at user's choice: git direct/explicit
  proxy remain blocked (curl 000), the verified WinHTTP Git Data API channel is
  ready and needs a user-provided PAT to complete the push.

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

### S-B5: NGX init rejection on AMD (recorded 2026-08-30, RTX-controlled in Round 3) — current AMD blocker
- Fact: unmodified nvngx_dlss.dll loaded on AMD and NVSDK_NGX_D3D12_Init returned
  0xbad00001 = NVSDK_NGX_Result_FAIL_FeatureNotSupported on the vendor=0x1002 device,
  before any observed nvapi64/nvcuda load or NVAPI/CUDA call. Round 1 produced this
  under a reconstructed ABI; Round 2 reproduced it under the OFFICIAL ABI
  (docs/NGX_ABI_AUDIT.md Post-script).
- Classification: corrected Feeder identifiers return Success on the RTX 5060
  (R-15) but `0xBAD00001` on AMD. This confirms an adapter-dependent NVIDIA
  hardware support gate. It does not by itself identify the exact internal check.
- Consequence: S5+ and launch-side trace require getting init past this rejection.
- Constraint: per spec, "spoofing the vendor check" is NEVER success evidence.
  A spoof may only be used as a LABELED trace-aid to OBSERVE the downstream call
  chain (module loads, nvapi IDs), and every such run must say so in RESULTS.md.
  Round 2 rule: trace-aid runs are only allowed after official ABI + NVIDIA reference
  + validated tracers are in place, and must set TRACE_AID_VENDOR_PROXY=true /
  RESULT_NOT_COUNTED_AS_S5_SUCCESS=true.
- Candidate trace-aid routes (all labeled): DXGI adapter desc proxy,
  OptiScaler-style FakeNVAPI only for observation, or running the same host on an
  NVIDIA box to capture the clean chain and diffing.

### S-B6: Unsupported PTX 9.4 instructions in the available translator (corrected 2026-08-31)
- The old “no PTX” blocker is superseded. All 15 runtime containers carry
  compressed PTX 9.4/sm_120 plus CUBIN. Function-isolated `cc_cb_clear` compiles
  and executes on RX 9070 XT through ZLUDA v7-preview.3.
- Current translator failures are instruction-specific: the copy boundary's
  `sust.p.2d.v4.b32.zero` now has a validated native D3D12 lowering; neural
  entries still use unsupported tuple-discard moves, `mma.sync` variants, FP8
  conversions, asynchronous barriers/copies and cache policies. These require
  translator extensions, PTX lowering, or per-function HIP implementations.

### S-B7: AMD D3D12 shared-texture memory is swizzled (recorded Round 2, Phase G)
- tests/texture_interop_test: HIP can import shared D3D12 textures (RGBA8/RGBA16F/
  R32F/RG16F) and the modify→readback roundtrip is byte-exact, but D3D12-written
  pixels are NOT at their linear byte offsets when read through the buffer view.
- Consequence: any HIP kernel that touches imported texture memory must implement
  swizzle-aware addressing (or the pipeline must stage through buffers); treat
  "texture interop proven" claims from Round 1 as superseded.

### S-B8: Neural instruction lowering, parameter semantics and weights — current S6 blocker
- R-22/E-18 proves that the complete 156-slot envelope, parameter bytes and handle
  lifetimes can be carried by HIP on RX 9070 XT, but its marker kernel contains no
  DLSSNR neural math and intentionally does not count as S6.
- The payload has PTX, but ZLUDA cannot yet translate the tensor/FP8 instruction
  set used by the tested neural entry. The remaining route is hybrid PTX lowering
  and clean-room HIP reconstruction, validated per function.
- The 16-byte `cc_cb_clear` boundary ABI (slot 0) is decoded and executed from
  original PTX on AMD. The final `cg2r_copy_kernel` texture/surface contract now
  passes exact standalone AMD readback; its RTX descriptor trace and bounded
  reference footprint remain to be joined. The first pre-block neural operator
  is the next implementation target.
  These boundary kernels validate dispatch/resource plumbing but still do not
  advance S6; the first neural target follows only after its tensor shape, weights
  and numerical reference can be measured.
