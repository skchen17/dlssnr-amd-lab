# EVIDENCE.md

Every "success" claim must link to a concrete artifact: command, log path, hash.

## E-174 Original-weight dynamic game output visibly reaches Present (debug only)

- Date: 2026-09-05. Game PID 4592, RX 9070 XT, SDR R10G10B10A2, 2342x1382.
- Commands: `start_gowr_resident_present.ps1`, `control_gowr_resident_present.ps1`,
  `analyze_resident_present.py`; exact examples in `docs/RESIDENT_PRESENT_DEBUG.md`.
- Session: `results/20260905_192944_171_gowr_resident_present/`.
- Analysis: `results/20260905_present_game_analysis_v3/summary.json` reports
  `SAME_BUFFER_INFERENCE_PRESENT_PASS`, 12 distinct current inputs, 16 complete
  156-slot tiles/frame. Original model SHA is
  `A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`.
- No captured activation injection, stale overlay, resizing of the whole input
  or substitute trained toy CNN. First two packed buffers change 3,236,010 and
  3,235,959 pixels. Finite, alpha-preserving, exact independent packed conversion
  and GPU writeback. Each corresponding Present returns S_OK.
- `screen_verification.json` binds `resident_running.png` (3 frames completed)
  and `resident_running_late.png` (11): visible blockwise brightness seams track
  the newly generated frames. Automatic 12-frame stop returns to original
  rendering (`resident_stopped.png`). Physical image evidence is visual, not
  pixel-exact screenshot equivalence.
- Worker times 5,081.9–5,201.2 ms/frame. This is a low-FPS executable experiment,
  NOT playable realtime, not DLSS5 parity, not correct pre-HUD/exposure/temporal
  semantics, and not arbitrary-resolution full-frame attention quality.
- Original game HDR was ON (brightness 80, UI 50); first live attempt at PID 9696
  was safely blocked at zero frames. v3 logs show color space 12 (PQ/BT.2020),
  then 0 after temporarily turning HDR OFF in the game's own calibration page.
  No Windows HDR setting was changed. `hdr_restoration.json` and
  `hdr_restored.png` verify game HDR ON / brightness 80 / UI 50 restored in
  PID 18768 without the resident network, followed by normal exit.
- `after_run_audit.json` and `after_exit_audit.json`: responsive before normal
  exit, four original image hashes unchanged, no new System 4101/41/6008 events.
- Tested v3 DLL SHA:
  `C5AE681477AF01D9243E95C9E812FA346192580E2A11189768D021AA61FA9128`,
  bound in the session's `provenance.json` and staged `present_validation.json`.
- WARP/AMD Present + Present1, TEST bypass, stop restoration, resize, packed
  conversions and four IPC failure controls pass on v3. Python suite: 533 pass.

## E-172 Original-weight persistent full graph on RX 9070 XT

- Date: 2026-09-05.
- Command: `scripts/validate_resident_full_graph.py`; full command in
  `docs/RESIDENT_GAME_BRINGUP.md`.
- Evidence: `results/20260905_resident_graph_v1/validation.json`.
- Six interleaved input/control executions, all 156 slots, 51 retained modules,
  one original 147,719,680-byte model upload; no RTX intermediate injection.
  E-170 real-input and zero-preblock controls are byte-exact, including across
  sync-each-slot and ordinary event-completion execution. Model readback is
  immutable. Normal 640x360 host inference about 308–311 ms.
- Scope: true image-driven resident inference, not smooth realtime or quality.

## E-173 Continuous live game texture loop; final presentation NOT accepted

- Date: 2026-09-05.
- Sessions: `results/20260905_184745_733_gowr_capture_session` and
  `results/20260905_190329_410_gowr_capture_session`.
- Analysis: `results/20260905_resident_game_analysis_v2/summary.json` (old name,
  superseded scope), `results/20260905_resident_game_presentcheck_analysis/summary.json`.
- Each completed 12 distinct 2342x1317 frames, 16 independent full-network tiles
  per frame. First two input/output payloads are byte-identical across IPC and
  game readback, all values finite and alpha preserved. About 5.1–5.3 seconds
  per whole surface. Neither shrinking the full frame nor captured activation
  substitution is used.
- Negative evidence: `resident_running.png` (5 completed frames) and
  `resident_running_late.png` (10) in the second session do NOT visibly contain
  the strong tile seams seen in the network raw output. End-of-list injection
  does not establish that game consumers have not already run inside the list.
- The correct status is `GAME_TEXTURE_INFERENCE_LOOP_PASS`, with explicit
  `final_presentation_verified: false`, not a visible game integration pass.
- Both after-run/after-exit audits: responsive before normal exit, no deployed
  game binary, four game image hashes unchanged, no new System 4101/41/6008 event.
- Separate WARP/AMD bridge and installed-hook controls, stop restoration,
  20 FSR3/FSR4 regressions, and four hostile IPC responses also pass; paths in
  `docs/RESIDENT_GAME_BRINGUP.md`. These do not override negative screen evidence.

Format per entry:

```
## E-<n> <claim>
- Date:
- S-level touched:
- Command:
- Artifact(s): results/<ts>/... (JSON/log path)
- Adapter proof: vendor=0x1002 device=0x____ LUID=____
- Verdict: PASS / FAIL / PARTIAL
```

## E-1 RDNA4 compute baseline is numerically correct (S1)
- Date: 2026-08-30
- S-level touched: S1
- Command: `scripts\build_all.ps1 -Only hip_probe` then `build\hip_probe.exe --json results\20260830_170305\hip_probe.json`
- Artifact(s): results/20260830_170305/hip_probe.json (overall_pass=true; A–E PASS; F UNVERIFIED per S-B4)
- Adapter proof: vendor=0x1002 device=0x7550 LUID=recorded in environment.json (gfx1201 reported by HIP)
- Verdict: PASS

## E-2 D3D12 resources processed by HIP kernels with zero-content mismatch (S2)
- Date: 2026-08-30
- S-level touched: S2
- Command: `scripts\build_all.ps1 -Only d3d12_hip_interop` then `build\d3d12_hip_interop.exe --json results\20260830_170305\d3d12_hip_interop.json`
- Artifact(s): results/20260830_170305/d3d12_hip_interop.json (pass_a=true; T3/T4 mismatch=0; T5 texture import ok), docs/INTEROP.md
- Adapter proof: D3D12 device created on AMD adapter vendor=0x1002 device=0x7550; HIP device 0 = same physical GPU (single-GPU box; LUID match recorded in INTEROP.md)
- Verdict: PASS

## E-3 Trace shims actually intercept (tooling readiness for S5)
- Date: 2026-08-30
- S-level touched: none yet (pre-S5 tooling)
- Command: `C:\Users\20426\Documents\ComfyUI\.venv\Scripts\python.exe scripts\smoke_trace.py`
- Artifact(s): results/20260830_170305/nvapi_trace_selftest.jsonl (call/ret events incl. arg1 peek + 0xFFFFFFFF sentinel), results/20260830_170305/module_trace_selftest.log (IAT patches + hooks_installed)
- Adapter proof: n/a (CPU-side interception test)
- Verdict: PASS

## E-4 nr_host refuses to fabricate execution and emits deterministic reference package
- Date: 2026-08-30
- S-level touched: none (blocked by design — no fabricated success)
- Command: `build\nr_host.exe --frames 1 --width 512 --height 512 --trace --json results\20260830_170305\nr_host.json`
- Artifact(s): results/20260830_170305/nr_host_run.log + nr_host.json (`BLOCKED_MISSING_PREREQUISITE`, exit 3); results/20260830_180553_reference_package/ (SHA-256 reproducible across runs: CD71008B...60043)
- Adapter proof: n/a (no GPU work executed)
- Verdict: PASS (correct refusal is itself the evidence)

## E-5 Initial plaintext/CUBIN scan (SUPERSEDED by E-20)
- Date: 2026-08-30
- S-level touched: S3
- Command: `scripts\_probe_now.ps1`; `scripts\_carve_cubin.ps1`; `scripts\_readobj_cubin.ps1`; `scripts\_scan_sm_targets.ps1`
- Artifact(s): results/20260830_170305/binary_manifest_dlssnr.json (15 ELF markers), sm_targets_scan.log (15× sm_120), carved_cubin_0.readobj.log (kernel names + .nv.info), probe_dlssnr_stdout.log (zero PTX markers)
- Adapter proof: n/a (static analysis)
- Verdict: PASS for locating the 15 sm_120 CUBIN representations and preserving
  names/metadata, but the “no PTX/pure SASS” inference is **RETRACTED**. PTX was
  Zstd-compressed inside the runtime containers; E-20 is authoritative.

## E-6 unmodified DLSSNR runtime loads on AMD; NGX init rejected (first dynamic contact)
- Date: 2026-08-30
- S-level touched: S5 attempt (blocked)
- Command: `scripts\_forcload_run.ps1`
- Artifact(s): results/20260830_170305/nr_host_forceload_stdout.log (load OK, init=0xbad00001), forcload_20260830_182045_module_trace.log (no NVIDIA support modules observed)
- Adapter proof: D3D12 device created on vendor=0x1002 device=0x7550 (printed by nr_host)
- Verdict: PARTIAL — `0xBAD00001 = FAIL_FeatureNotSupported` under a reconstructed ABI (superseded by E-7); vendor-gate classification UNCONFIRMED; NOT counted as any execution success

## E-7 (Round 2) official-ABI rerun reproduces 0xBAD00001 on AMD
- Date: 2026-08-30 (Round 2)
- S-level touched: S5 (still blocked)
- Command: build nr_host against third_party/nvidia-dlss (official headers + nvsdk_ngx_d.lib); run with DLSS_DLL_PATH set
- Artifact(s): docs/NGX_ABI_AUDIT.md (E1–E7 + Post-script), results/*_host_r2*.json
- Adapter proof: vendor=0x1002 device=0x7550 printed by nr_host
- Verdict: reproduction PASS. `0xBAD00001 = NVSDK_NGX_Result_FAIL_FeatureNotSupported` under the OFFICIAL ABI (both snippet and loader Init variants). R-15/E-11 later provides the corrected-ID RTX positive control.

## E-8 (Round 2) tracer/tooling self-tests
- module_trace v2: `scripts\run_module_trace_selftest.ps1` → READY, 0 failures; full load→resolve→unload chain asserted in order (results/<ts>_module_trace_selftest/).
- nvapi trampoline: `scripts\run_nvapi_trampoline_test.ps1` → 1..10 args + return byte-exact after the pure-jmp rewrite; the Round 1 dispatcher was DISPROVED for args 5+ by this same test (results/<ts>_nvapi_trampoline_test/).
- ABI regression: `scripts\run_ngx_abi_probe.ps1` → compile-time asserts + export-table check PASS (results/<ts>/ngx_abi_test.json).

## E-9 (Round 2) texture interop gate (S2 texture path)
- Date: 2026-08-30 (Round 2)
- S-level touched: S2 (texture refinement)
- Command: `scripts\run_texture_interop_test.ps1`
- Artifact(s): results/<ts>_texture_interop/texture_interop.json (per-format mismatch_count + max_abs_error, both directions)
- Adapter proof: D3D12 vendor=0x1002 device=0x7550 + HIP device 0
- Verdict: PARTIAL — roundtrip byte-exact for RGBA8/RGBA16F/R32F/RG16F; layout is NOT identity (swizzled), so linear-addressing assumptions on imported texture memory are false.

## E-10 RTX 5060 control localizes the init failure to the host contract
- Date: 2026-08-30
- Artifact(s): `results/20260830_230102_rtx5060/`
- Adapter proof: RTX 5060 Blackwell `sm_120`, vendor 0x10de, driver 591.59;
  D3D12 device creation PASS.
- Integrity: DLSS 310.7 and DLSSNR 310.8 hashes match the lab copies and have
  valid NVIDIA signatures; ABI and binary probes PASS.
- Observation: old nr_host identifiers produce `0xBAD00001` on RTX before DLSS,
  NVAPI, or CUDA module load; DLSSNR itself loads for observation.
- Verdict: PASS as a diagnostic control. It disproves classification of this
  return code as AMD vendor-gate evidence and requires a corrected-host rerun.

## E-11 RTX 5060 corrected-ID public DLAA chain
- Date: 2026-08-30
- Artifact(s): `results/20260830_231421_rtx5060_v2/`; returned ZIP SHA-256
  `C0F85D186BBE67B6235396237E23F7C7DBC2AB8385BF0A07EFBC4482084B4549`.
- Adapter proof: RTX 5060 Blackwell `sm_120`, vendor 0x10de, device 0x2f04;
  D3D12 device creation PASS.
- Positive-control milestones: NGX Init Success, SuperSampling available,
  parameters/scratch/upload PASS, public DLAA CreateFeature PASS.
- Dynamic modules: driver `_nvngx.dll`, `nvngx_dlss.dll`, NVAPI, `nvcuda64.dll`,
  `nvdxgdmal64.dll`, and `nvobjectloader64.dll` observed by the proved tracer.
- Failure boundary: 8/8 Evaluate calls returned
  `0xBAD00005 = FAIL_InvalidParameter`; no output and no DLSSNR kernel execution.
  Source comparison identifies omitted render-subrect dimensions; Round 3 fixes it.
- Verdict: PASS as the corrected public-host positive control through
  CreateFeature; NOT S3B/S4 because feature 18 was not hosted and no NR launch ran.

## E-12 RTX 5060 public DLAA end-to-end positive control
- Date: 2026-08-30
- Artifact(s): `results/20260830_232657_rtx5060_v3/`; returned ZIP SHA-256
  `CAE631C214CA2239D97664AA945CF768E7D636A0FF59671772DC510AC82954B9`.
- Adapter proof: RTX 5060 `sm_120`, vendor 0x10de/device 0x2f04.
- Execution proof: Stage A and traced Stage B each report 8/8 successful Evaluate
  calls followed by GPU fence completion and readback; both outputs are byte-equal.
- Output proof: 1,048,576 bytes, SHA-256
  `9B98DFC631755EED553036786D23527D7E810E1DFA204BEB802787AD394C5041`;
  302,026 bytes differ from the deterministic color input.
- Verdict: PASS for the public NVIDIA NGX/DLAA host. Feature 18 remains NOT_RUN,
  so this is deliberately not counted as DLSSNR S3B/S4 evidence.

## E-13 RTX 5070 signed DLSSNR / feature-18 execution
- Date: 2026-08-30
- Artifact(s): `results/20260830_234330_rtx5070_feature18/`; returned ZIP SHA-256
  `9D6CCCCBA47AFA8DE0262928EE46C90E0E1EEC834BEB6DA1EBBAE2CE1060DDA5`.
- Adapter proof: ReShade reports NVIDIA GeForce RTX 5070, driver 610.88.
- Private-path proof: add-on registration via ReShade API 18; signed DLSSNR 310.8
  initialization; feature 18 creation; inline feature-18 evaluation success at
  counts 1 and 60; second creation after the host warm-up rebuild.
- Public-contract proof: DLSS5-Feeder host reports 300/300 successful evaluations.
- Verdict: **S4 PASS**, S3B PASS-HIGH-LEVEL. Launch API/per-kernel identity is not
  yet proven and must not be inferred from these logs.

## E-14 RTX 5070 feature-18 lower-path module/export discrimination
- Date: 2026-08-30
- Artifact(s): `results/20260830_235021_rtx5070_feature18_trace/`; returned ZIP
  SHA-256 `8DC23B1AEFE852190AF1900C24624B9A071F3BF1A0332EC8CC335C21ECAF893E`.
- Instrumentation proof: package revision `v6_feature18_traced`; host stdout says
  `hooks_installed=true` before ReShade and NGX initialization.
- Positive execution proof: 300/300 public evaluations, private feature 18 created
  twice, and private inline evaluation succeeded.
- Lower-path observation: DLSSNR uses its D3D12 NGX exports and resolves system
  NVAPI's `nvapi_QueryInterface` plus `nvapi_Direct_GetMethod`; no classic CUDA
  Driver API load/export resolution appears on the successful private path.
- Verdict: PASS as dynamic case discrimination. CASE A is the leading inference;
  exact private IDs and the per-kernel sequence remain unproven.

## E-15 RTX 5070 official NVAPI CuModule/CuFunction/launch-chain IDs
- Date: 2026-08-31
- Artifact(s): `results/20260831_000136_rtx5070_feature18_nvapi_ids/`; returned
  ZIP SHA-256 `2262A415B7E93C4DF43C52702030B157D1299D03EBA194629E848E50E16503A5`.
- Non-interference proof: v7 returns PASS with the same RTX 5070/610.88 reference,
  300/300 public evaluates and successful private feature-18 execution.
- NR-only QueryInterface IDs: `AD1A677D`, `E2436E22`, `24973538`, `DF295EA6`,
  `41C65285`; every resolution returned a non-null driver pointer.
- Identity proof: NVIDIA's official R610 `nvapi_interface.h` maps them to D3D12
  CreateCuModule, CreateCuFunction, LaunchCuKernelChain, DestroyCuFunction and
  DestroyCuModule respectively; official `nvapi.h` publishes their signatures.
- Verdict: **CASE A CONFIRMED**. Exact invocation data/per-kernel order remains a
  separate instrumentation task and is not inferred from pointer resolution.

## E-16 RTX 5070 typed CASE-A invocation proof
- Date: 2026-08-31
- Artifact(s): `results/20260831_001127_rtx5070_feature18_nvapi_calls/`; returned
  ZIP SHA-256 `0A8C6FB6890B6DA9E512103E75B89CE335E5FF616589F8244F17337D09D06192`.
- Non-interference: reference remains PASS (300/300 public calls plus successful
  private creation/evaluation) under the typed wrappers.
- Lifecycle proof excluding one deliberate null probe: nine modules and 96 named
  functions created; 46,800 single-kernel chains returned NVAPI_OK; all 96
  functions and all nine modules destroyed with NVAPI_OK.
- Frame-graph proof: 46,800 = 300 × 156, establishing 156 ordered single-kernel
  submissions per synthetic 640x360 frame for this configuration.
- Verdict: CASE A proven at actual-call level. Exact full-frame order/parameter
  bytes remain pending a non-aliasing detailed capture.

## E-17 Complete and stable Feature-18 frame graph
- Date: 2026-08-31
- Artifact(s): `results/20260831_002356_rtx5070_feature18_full_frame/`; returned
  ZIP SHA-256 `0B2DC1FCE412E5C70765DE6A7F9992A81F2329943AF6FF49867D9655B9488A65`.
- Completeness: five selected frames × 156 ordered slots = 780 detailed events.
- Stability: identical structural SHA-256 for all five frames; zero function,
  order, grid, block or parameter-size mismatches.
- Parameter stability: 153 slots byte-identical; three boundary I/O/state slots
  vary. Resource recreation changes addresses without changing the graph.
- Static/dynamic join: all nine runtime module hashes match exact fatbin offsets
  in the signed DLL; 96 function names map to module ownership and the tested
  frame selects 43 of them.
- Verdict: PASS-GRAPH for this configuration. RTX reference acquisition can stop
  unless another resolution/mode/architecture is specifically needed.

## E-18 AMD graph transport and parameter-carriage proof
- Date: 2026-08-31
- Artifact(s): `results/20260831_003706_amd_graph_replay/` (`amd_graph_replay.json`,
  `manifest.json`, `stdout.log`, `SOURCE.md`).
- Adapter proof: HIP `AMD Radeon RX 9070 XT`, arch `gfx1201`; DXGI vendor/device
  `0x1002:0x7550`.
- Input proof: manifest SHA-256 binds the replay to R-21's module/function/156-slot
  CSVs and to the exact executable used.
- GPU proof: 156 marker records were written on AMD in strict slot order with the
  captured launch geometry; mismatch count is zero.
- Parameter proof: 11,624 raw captured bytes were copied to device memory and
  hashed per slot by GPU code; every GPU hash matched its host-side reference.
- Lifecycle proof: 9/9 modules and 96/96 functions reconstructed, ownership
  checked, functions destroyed before modules, no residual live handles.
- Boundary: only lab-authored marker math ran. This is transport evidence and is
  deliberately excluded from S6/S7.

## E-19 D3D12/HIP address relocation requirement
- Date: 2026-08-31
- Artifact(s): `results/20260831_004730_interop_address_identity/`.
- Same-allocation proof: D3D12 heap import passes T1/T3/T4 with zero transfer
  mismatches while D3D12 VA `0x200c40000` differs from HIP pointer `0x304000000`.
- Verdict: explicit interval registration/translation is mandatory for CASE A.

## E-20 Runtime PTX recovery and used-function inventory
- Date: 2026-08-31
- Artifact(s): `results/20260831_005343_kernel_inventory/` and
  `results/20260831_010100_all_runtime_modules/extraction_manifest.json`.
- Completeness: all 15 runtime containers decompress; all identify as PTX 9.4,
  target sm_120; entry counts total 231. The R-21 path selects 43 functions and
  156 slots from 9 of these modules.
- Verdict: the earlier pure-SASS/no-PTX evidence was a compressed-data false
  negative and is superseded. Proprietary extracted PTX files remain ignored;
  only hashes, names and metadata are recorded.

## E-21 Official NVAPI AMD backend and translated address self-test
- Date: 2026-08-31
- Artifact(s): `results/20260831_005935_nvapi_amd_selftest/`.
- ABI proof: exact official R610 five-call interface IDs; 9/96/156 lifecycle on
  DXGI `0x1002:0x7550`; intentional invalid address rejected; no live handles.
- GPU/readback proof: registered D3D12 buffer is mapped into HIP, exact clear
  contract writes 27,648 `0xffffffff` words, D3D12 readback mismatch count zero.
- Boundary: transport and clear only; `counts_as_s6=false`.

## E-22 Original NVIDIA PTX executes on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_011219_zluda_ptx_probe/`.
- Provenance: `clear_extraction.json` binds the isolated function to module 6;
  hashes bind every probed PTX input and ZLUDA `nvcuda.dll` v7-preview.3.
- Execution proof: real ZLUDA AMD device/context, module load, function resolve,
  memory allocation, launch, synchronization and device-to-host copy all return
  success; 27,648/27,648 words match `0xffffffff`.
- Negative precision: translator logs name exact unsupported statements for copy
  and neural entries. `full_translation_pass=false`, neural math did not execute,
  and S6 is intentionally unchanged.

## E-23 Native D3D12 lowering of the final copy boundary on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_013221_nvapi_amd_copy_selftest/` and
  `results/20260831_013231_nvapi_amd_selftest/`.
- ABI proof: the official merged texture/sampler and independent descriptor APIs
  generated distinct stable objects consumed by the decoded 72-byte copy launch.
- GPU/readback proof: one real D3D12 compute dispatch copied 64x32 RGBA32F on
  DXGI vendor `0x1002`; all 8,192 components were bit-exact and pipeline failures
  were zero.
- Regression proof: the expanded seven-interface backend retained the prior
  9/96/156 lifecycle, D3D12-VA relocation and 27,648-word clear results with zero
  mismatches after both test and DLL were rebuilt from the same diagnostics ABI.
- Boundary: this is an independently lowered I/O operation, not neural math;
  `counts_as_s6=false`.

## E-24 Offset-exact ABI for the first neural pre-block
- Date: 2026-08-31
- Artifact(s): `results/20260831_013704_preblock_param_abi/`.
- Provenance: manifest hashes bind the analysis to the recovered module-0 PTX,
  the stable R-21 frame CSV and the analyzer source; proprietary PTX stays ignored.
- Static proof: declared size 264 equals the captured launch size; 37 direct loads
  cover 252 bytes across three intervals, leaving only two explicit gaps totaling
  12 bytes.
- Boundary: access offsets and captured bit patterns are evidence; semantic field
  names and tensor layouts remain unproven pending dynamic RTX resource snapshots.

## E-25 RTX descriptor objects joined to copy and pre-block launches
- Date: 2026-08-31
- Artifact(s): `results/20260831_012648_rtx5070_feature18_descriptors/` and
  `results/20260831_014220_descriptor_trace_join/`.
- Positive reference: 300/300 host evaluations plus successful private feature-18
  creation/evaluation under the descriptor wrappers.
- Copy proof: five of five selected slot-155 launches resolve both object handles
  to the exact successful NVAPI calls; unresolved pairs are zero. Descriptor
  recreation at frame 241 changes handles but preserves roles and dimensions.
- Neural-boundary proof: slot-1 offsets +0/+8/+16 are joined to primary,
  auxiliary and history merged texture/sampler objects where nonzero; dimensions
  and frame counter are decoded from PTX-proven direct parameter loads.
- Limit: object-to-resource format/content is not inferred and is the v11 target.

## E-26 D3D12 resource tracer readiness on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_014841_module_trace_d3d12_selftest/`.
- Real-device proof: the registered AMD D3D12 device emitted two resource records,
  one SRV, one UAV, one sampler and three descriptor-copy records.
- Integrity proof: the self-test exited 0 and every JSONL line parsed. The v11 RTX
  package has a 12-file zero-mismatch manifest and fixed SHA-256 recorded in R-30.
- Boundary: this proves observation fidelity, not any RTX tensor values or AMD
  neural output.

## E-27 RTX object-to-resource identity and allocation ranges
- Date: 2026-08-31
- Artifact(s): `results/20260831_015650_rtx5070_feature18_resources/` and
  `results/20260831_020353_d3d12_resource_trace_join/`.
- Positive reference: 300/300 evaluations and private feature-18 creation and
  evaluation remained successful under the D3D12 hooks; invalid JSON lines are zero.
- Boundary proof: five of five slot-155 launches resolve to distinct 640x360
  RGBA16F source/destination textures. Five of five selected first pre-block
  launches resolve the primary RGBA16F texture and all three direct GPU-VA fields
  resolve to exact allocation intervals and offsets.
- Boundary: resource identity, dimensions and intervals are proven; texture/tensor
  values still require v12 and neural output still requires N0.

## E-28 Copy-content snapshot pipeline on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_021024_module_trace_d3d12_selftest/`.
- Real-device proof: two ordered D3D12 texture copies ran on AMD, readback emitted
  compact raw rows, and input/output SHA-256 hashes matched for 32,768 bytes.
- Numerical proof: 8,192 RGBA32F scalars have zero mismatches and zero absolute/MSE
  error; metadata and JSONL both report PASS with zero malformed lines.
- Package proof: v12's archive passed ZIP integrity and contains all 12 manifested
  files. Its fixed archive SHA-256 is recorded in R-32.
- Boundary: the local identity fixture validates capture fidelity. The actual RTX
  640x360 RGBA16F copy result is the remaining M4 content gate.

## E-29 N0 pre-block memory roles and ordered capture pipeline
- Date: 2026-08-31
- Artifact(s): `results/20260831_022951_preblock_global_access/` and
  `results/20260831_023543_module_trace_d3d12_selftest/`.
- Static provenance: analysis is bound to original isolated PTX SHA-256
  `4000374C4EBE9CCB92780386D016F66C4C7C7CB3F5B1361E4A7F98B202196332`.
  It proves +216/+248 are write-only candidates and +224 is read-only across all
  177 traced global operations from these pointer origins.
- Real-device proof: RX 9070 XT recorded before/after copies in one ordered D3D12
  command list, exported 20,840,448 bytes across six windows, preserved the
  weight fixture and detected the exact intentional output mutation.
- Negative/strict gate: the returned RTX run is accepted only if scratch and
  output both change, weights do not, all file lengths and hashes match metadata,
  Feature 18 remains successful and JSONL has no malformed lines.
- Boundary: this proves capture fidelity and static roles only; `counts_as_s6=false`.

## E-30 Returned RTX N0 tensor oracle and M4 cross-vendor closure
- Date: 2026-08-31
- Artifact(s): `results/20260831_023943_rtx5070_feature18_n0/` and
  `results/20260831_024744_nvapi_amd_rtx_copy_oracle/`.
- RTX integrity: the returned archive passes CRC and its SHA-256 is
  `B566AD24795050E64DA8AE59429A48ECCF46C692B7302999188E9F952F2E9936`.
  Formal copy and N0 analyzers both report PASS under the positive Feature-18 run.
- Cross-vendor proof: RTX source/destination and RX 9070 XT native-copy output are
  the same 1,843,200 bytes, with zero byte and FP16-component mismatches.
- Neural observation: scratch/output changed substantially while the bounded
  weight window was bitwise stable; hashes, roles, byte lengths, FP8 dtype and
  exact logical shape envelopes are recorded in `n0_analysis.json` and
  `n0_tensor_layout.json`; physical layout is refined by E-40.
- Boundary: M4 is complete. N0 is an RTX oracle, not AMD neural execution.

## E-31 RX 9070 XT E4M3 storage primitive
- Date: 2026-08-31
- Artifact(s): `results/20260831_025229_amd_e4m3_rtx_oracle/`.
- Real-device proof: a HIP kernel identifies `AMD Radeon RX 9070 XT` and processes
  all 1,966,080 bytes from the RTX N0 output plus the exhaustive 256-code domain.
- Numerical/negative proof: all capture bytes and 254 finite codes round-trip
  exactly; decoded non-finite count equals the capture NaN-code count; an injected
  one-bit difference is detected.
- Boundary: this validates representable E4M3 storage values only. A separate RTX
  conversion oracle is still required for arbitrary FP16 rounding and packed-lane
  order; MMA and fused-operator validation remain open. `counts_as_s6=false`.

## E-32 Exhaustive RTX conversion-oracle package gate
- Date: 2026-08-31
- Artifact(s):
  `deliverables/dlssnr-windows-fp8-reference-v14-20260831_025645.zip`.
- Build proof: the host-only executable builds without CUDA SDK headers/libraries
  and loads CUDA driver entry points dynamically. The package manifest hashes all
  three files, and archive SHA-256 is
  `AF066DBF3BE5A85D49C8723CB4DCD5E048A0EE3D8C3AD7A433E32E43DD5A3817`.
- Scope: the planned RTX output is exactly one byte for each of the 65,536 binary16
  bit patterns, produced by NVIDIA PTX `cvt.rn.satfinite.e4m3x2.f16x2`.
- Boundary: this is a local build/package gate only; PASS numerical evidence needs
  the returned RTX result archive.

## E-33 Exhaustive cross-vendor FP16-to-E4M3 proof
- Date: 2026-08-31
- Artifact(s): `results/20260831_104457_rtx_fp16_e4m3/` and
  `results/20260831_110051_amd_fp16_e4m3_rtx_oracle/`.
- RTX proof: CUDA status is zero on an RTX 5070, the exact packed PTX instruction
  processed all 65,536 binary16 inputs and produced the manifested 65,536-byte
  oracle with SHA-256 `0212E259...E464CE`.
- Discrepancy/repair proof: the pre-fix implementation disagreed only for 1,023
  negative NaNs. Canonicalizing every NaN to `0x7f` reduced total, finite,
  infinity and NaN mismatches to zero on RX 9070 XT.
- Negative/regression proof: a deliberate bit mutation is detected and the full
  captured N0 output still round-trips exactly after the semantic correction.
- Boundary: conversion semantics and lane order are closed. MMA accumulation and
  full operator layout remain open; `counts_as_s6=false`.

## E-34 FP8 MMA oracle package gate
- Date: 2026-08-31
- Artifact(s):
  `deliverables/dlssnr-windows-mma-reference-v15-20260831_110845.zip`.
- Build/integrity proof: the CUDA-driver probe builds without CUDA SDK linkage;
  all packaged executable/runner hashes match the manifest and archive SHA-256 is
  `9903909F6775C07D58671D8202215682CABF0128DFAA0A6BB069150168A13C5A`.
- Coverage: eight complete 32-lane register-fragment cases will emit 4,096 bytes
  of A and 2,048 bytes each of B/C/D. The exact PTX instruction is the dominant
  unsupported FP8 MMA form found in N0.
- Boundary: local packaging only. RTX CUDA status and D-fragment bytes are pending.

## E-35 Cross-vendor m16n8k32 FP8 MMA fragment/math proof
- Date: 2026-08-31
- Artifact(s): `results/20260831_112253_rtx_m16n8k32_e4m3_mma/` and
  `results/20260831_112900_amd_m16n8k32_e4m3_mma_oracle/`.
- RTX proof: CUDA status 0 on RTX 5070; manifested A/B/C/D registers and exact PTX
  metadata are intact. D raw SHA-256 is `2BD19DAF...18B07D`.
- AMD numerical proof: RX 9070 XT computes all 1,024 half outputs from the same
  register fragments with zero bit mismatches across all eight cases.
- Mapping proof: implementation uses the PTX per-lane row/column formulas rather
  than treating fragments as linear matrices; an injected output bit is detected.
- Boundary: standalone MMA is closed, while f16 MMA, movmatrix, memory movement
  and the full N0 instruction/dataflow remain open. `counts_as_s6=false`.

## E-36 Remaining N0 primitive package gate
- Date: 2026-08-31
- Artifact(s):
  `deliverables/dlssnr-windows-remaining-reference-v16-20260831_113540.zip`.
- Integrity: host probe and runner build successfully; package entries pass their
  internal SHA-256 manifest and archive SHA-256 is
  `10099E9DB441579B5EB739663E3175624487FB0531DCF829BE78B7FAC2FEB82A`.
- Scope: exact f16 MMA register fragments and m8n8 b16 transpose fragments, eight
  full warps each. This consolidates the next two RTX requirements into one run.
- Boundary: numerical evidence awaits the returned RTX archive; no S6 claim.

## E-37 Cross-vendor f16 MMA and movmatrix proof
- Date: 2026-08-31
- Artifact(s): `results/20260831_114007_rtx_remaining_n0_primitives/` and
  `results/20260831_114536_amd_remaining_n0_primitives_oracle/`.
- RTX proof: both exact PTX entries execute on RTX 5070 with CUDA status zero;
  the returned archive and all six raw hashes pass.
- Exact AMD proof: RX 9070 XT has zero movmatrix word mismatches and zero f16 MMA
  mismatches over the four functional-range cases (512 half outputs total).
- Stress disclosure: 337 differences across 512 intentionally overflowing f16
  outputs are recorded separately. The gate follows PTX's stated unspecified
  accumulation order rather than treating architecture-dependent Inf formation
  as a portable bitwise contract.
- Boundary: isolated primitives only; complete N0 input/dataflow/output remains
  open and `counts_as_s6=false`.

## E-38 N0 lowering coverage audit
- Date: 2026-08-31
- Artifact(s): `results/20260831_115000_n0_lowering_coverage/`.
- Completeness: all 740 parser diagnostics fall into seven expected categories:
  736 unrecognized statements plus four cache-policy store rejections. Unknown
  category count is zero and category totals reproduce the original log exactly.
- Coverage: 728 numerical statements have RTX/AMD semantic evidence; eight tuple
  moves and four cache-hint store mechanics remain at this audit point. Integrated
  translator count is explicitly zero.
- Boundary: this is a static readiness audit, not fused execution or S6 evidence.

## E-39 N0 tuple/cache lowering delta on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_120000_n0_tuple_lowering/` and
  `results/20260831_120500_n0_lowering_delta/`.
- Transformation: four discard tuple moves and four b128/cache-policy stores are
  rewritten without changing the per-thread value or address sequence.
- AMD proof: ZLUDA loads the rewritten module on `AMD Radeon RX 9070 XT`; the
  original 740 diagnostics become exactly 728, with all 12 mechanical diagnostics
  removed and no numerical category lost or added.
- Boundary: function resolution remains false and kernel launch remains false;
  numerical lowering and fused N0 execution are still open, so this is not S6.

## E-40 N0 tiled epilogue execution on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_122500_n0_tiled_epilogue/`,
  `results/20260831_121559_amd_n0_tiled_epilogue/` and
  `results/20260831_123000_n0_storage_layout/`.
- Storage proof: scratch is addressed as 4x4x32 MMA fragment tiles; output is
  addressed as two HxWx16 planes. These layouts reproduce the PTX byte addresses
  and supersede the earlier envelope-only HWC candidate.
- Cross-vendor numerical proof: the RX 9070 XT HIP kernel launches over an 80x48
  grid of 32-thread waves and produces the exact same raw output as the independent
  tiled PTX reconstruction. Versus RTX, 1,416,816 bytes are exact and 408,685 more
  are one same-sign E4M3 code away, yielding a 92.85% declared quantization gate.
- Boundary: RTX scratch is already E4M3-quantized whereas original N0 averages the
  unquantized FP16 registers. This stage is real AMD neural dataflow evidence but
  does not yet reproduce texture preprocessing, the learned MMA body or full N0.

## E-41 Full isolated N0 translator resolution on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_124500_n0_e4m3_lowering/`,
  `results/20260831_125500_n0_movmatrix_lowering/`,
  `results/20260831_130500_n0_f16_mma_lowering/` and
  `results/20260831_132000_n0_full_numeric_lowering/`.
- Completeness proof: staged parser counts are exactly 740, 728, 304, 272, 256
  and zero. All 740 original diagnostics therefore have an explicit lowering;
  none disappear through entry deletion or an unknown-category filter.
- AMD proof: on `AMD Radeon RX 9070 XT [ZLUDA]`, the final module load and real
  N0 function lookup both return CUDA success. The 571-second first compile is
  recorded rather than omitted.
- Numerical provenance: E4M3 covers all half bit patterns; movmatrix/f16 MMA/FP8
  MMA use the fragment mappings and accumulation models already compared against
  RTX, including the documented f16 overflow-order boundary.
- Boundary: this proves translator completeness and function resolution only.
  A valid texture object, captured weights and pointer parameters were not passed,
  so no complete N0 launch or output comparison is claimed.

## E-42 Full-grid learned N0 execution on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_134000_amd_n0_single_cta/` and
  `results/20260831_135000_amd_n0_full_grid/`.
- Resource/ABI proof: the harness creates and uploads a real normalized-linear
  640x360 RGBA16F texture, allocates the exact scratch/weight/output sizes and
  relocates all captured handles and pointers in the 264-byte launch block.
- AMD execution proof: both `1x1x32` and `80x48x32` launches return success from
  `cuLaunchKernel` and `cuCtxSynchronize` on `AMD Radeon RX 9070 XT [ZLUDA]`.
  Full-grid raw readback is 99.9849% nonzero for scratch and 99.9749% nonzero for
  output; an independent analyzer confirms the recorded counts, byte sizes and
  SHA-256 hashes directly from both raw files.
- Boundary: this closes local resource provisioning and complete N0 execution,
  not output correctness. The next gate is an RTX 50 run of the original PTX over
  the identical five hashed inputs, followed by scratch/output error analysis.
  Until that cross-vendor oracle passes, `counts_as_s6=false`.

## E-43 RTX N0 PTX-version rejection and controlled header normalization
- Date: 2026-08-31
- Artifact(s): `results/20260831_125952_rtx5070_n0_reference_fail/` and
  `deliverables/n0_full_reference_20260831_130219.zip`.
- Failure proof: RTX 5070 initializes CUDA and creates a context, then returns
  code 222 specifically at `cuModuleLoadData` for PTX 9.4. `kernel_launched` and
  `execution_verified` are false, and no scratch/output files are claimed.
- Correction integrity: the replacement PTX differs from source SHA-256
  `4000374C...196332` in exactly one directive. Replacing its `.version 8.7` line
  with `.version 9.4` reproduces the source byte-for-byte; the package manifest
  records both versions, hashes and reason.
- Boundary: this is a driver-compatibility packaging correction only. Same-input
  RTX numerical evidence remains pending and S6 is unchanged.

## E-44 Same-input full N0 comparison and earliest-divergence isolation
- Date: 2026-08-31
- Artifact(s): `results/20260831_130611_rtx5070_n0_same_input/`,
  `results/20260831_132500_amd_fp8_mma_lowered_ptx/`,
  `results/20260831_133100_amd_remaining_lowered_ptx/` and
  `results/20260831_133700_amd_n0_pre_mma_checkpoint/`.
- Cross-vendor execution proof: RTX 5070 and RX 9070 XT both complete the identical
  full grid with the same input, weights and scalar ABI. Raw sizes, nonzero counts
  and hashes are independently verified.
- Negative numerical proof: scratch/output correlation is -0.3637/-0.5426 and
  exact-or-adjacent E4M3 agreement is only 1.0333%/0.8730%; the analyzer reports
  FAIL rather than promoting dense output to success. AMD repeats are bitwise
  deterministic across three launches.
- Lowering isolation proof: executing the emitted scalar PTX, rather than only a
  CPU/HIP model of it, reproduces RTX FP8 MMA and movmatrix output bitwise and all
  functional f16 MMA cases bitwise. Known overflow-stress differences remain
  explicitly scoped.
- Next boundary: compare 32 lanes x 96 bytes of shared-input/weight fragments
  immediately before the first f16 MMA. This determines whether divergence starts
  in texture/preprocessing or after the first matrix stage. S6 remains open.

## E-45 Bitwise N0 pre-MMA input/weight checkpoint
- Date: 2026-08-31
- Artifact(s): `results/20260831_132553_rtx5070_n0_pre_mma_checkpoint/` and
  `results/20260831_133700_amd_n0_pre_mma_checkpoint/`.
- RTX/AMD proof: identical PTX prefix, texture bytes, weight bytes and scalar ABI
  produce a byte-identical 3,072-byte export across all 32 lanes. Input A has
  0/2,048 mismatches and weight B has 0/1,024 mismatches.
- Localization proof: this checkpoint occurs immediately before the first of 16
  f16 MMAs. Therefore the large final tensor difference cannot be attributed to
  texture sampling, prefix arithmetic, shared-memory layout or initial weight
  addressing for CTA (0,0).
- Boundary: only the first CTA and pre-MMA point are closed. The next checkpoint
  compares all 4,096 D-fragment bytes immediately after those 16 MMAs; S6 remains
  open.

## E-46 Corrected learned N0 execution and RTX numerical parity on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_133211_rtx5070_n0_post_f16_checkpoint/`,
  `results/20260831_134500_amd_n0_post_f16_corrected/`,
  `results/20260831_140500_amd_n0_full_grid_corrected/` and
  `results/20260831_140501_amd_n0_full_grid_corrected_repeat/`.
- Defect localization: the pre-MMA A/B export is bitwise exact, but the old
  post-MMA export differs in all 2,048 half values. Correcting the A-fragment
  lane/register/half formula makes all 4,096 post-MMA bytes bitwise exact.
- AMD execution proof: the corrected 55,752,278-byte learned entry loads and
  resolves on `AMD Radeon RX 9070 XT [ZLUDA]`; the complete `80x48x32` launch,
  synchronization and raw tensor readback all return CUDA success.
- Same-input proof: input, weight and parameter hashes equal the RTX manifest.
  Scratch/output correlations are 0.99999852/0.99999880, normalized RMSE values
  are 0.001719/0.001552, and exact-or-adjacent E4M3 rates are
  99.9873%/99.9895%. Both pass the declared numerical gate; an AMD repeat has
  identical scratch/output SHA-256 values.
- Verdict: **S6 PASS** for the first isolated DLSSNR-originated neural kernel on
  AMD. This does not claim a complete frame, Feature-18 integration or optimized
  performance; those remain S7+ requirements.

## E-47 First N0-dependent Swin entry executes on RX 9070 XT
- Date: 2026-08-31
- Artifact(s): `results/20260831_142000_n1_slot2_baseline/`,
  `results/20260831_142500_n1_slot2_lowering/`,
  `results/20260831_143501_amd_n1_slot2/` and
  `results/20260831_143502_amd_n1_slot2_repeat/`.
- Translation proof: all 680 original diagnostics and the one masked release-cache
  diagnostic are accounted for. The release rewrite preserves GPU release
  ordering and removes only `L1::no_allocate`; the function then resolves on the
  real RX 9070 XT ZLUDA device.
- ABI/resource proof: the 96-byte captured slot block is hash-bound to the stable
  frame sequence. Static access origins classify input/weights read-only and
  output/sync write-only; the weight pointer is relocated to the captured
  `+0x5600` view inside the same 65,536-byte window used by N0.
- Execution proof: launch and synchronization return success for `40x24x32`.
  Output has 1,964,665 nonzero bytes; exactly 960 sync words change from the
  `0xffffffff` negative-test sentinel to the PTX-authored release value zero,
  with no unexpected words. A second run has identical output and sync hashes.
- RTX gate: `deliverables/n1_slot2_reference_20260831_142143.zip`, SHA-256
  `B5F00CCC55475F453C202852D8483221FAD72EFE16A3BEAC88DE3FE6F9145EA2`,
  has zero payload hash mismatches and a byte-restorable version-only PTX change.
- Verdict: AMD execution PASS, cross-vendor numerical parity pending. This does
  not count as S7 or a complete frame.

## E-48 N1 slot-2 RTX/RX 9070 XT same-input parity
- Date: 2026-08-31
- Artifact(s): `results/20260831_142631_rtx5070_n1_slot2/` and
  `results/20260831_143501_amd_n1_slot2/comparison.json`.
- Integrity proof: returned archive SHA-256 is
  `2E3CA0DC27BCEEC9D15BA6A5603B2E5D7E172ED092BEBA150A0671F25BB38C65`;
  input, weight and parameter hashes match the verified reference payload.
- Numerical proof: 1,965,376/1,966,080 E4M3 elements are byte-exact, 99.9966%
  are exact-or-adjacent, correlation is 0.99999888 and NRMSE is 0.001495. All
  thresholds pass and the RX repeat is bitwise deterministic.
- Synchronization proof: RTX and RX sync buffers have the identical SHA-256
  `E83A50D11453E82BFA718569877C5D2F2C5746E7082F69893E53C822CB84C8F8`,
  with exactly 960 releases and zero unexpected words.
- Verdict: N1 slot 2 parity PASS. S7 remains open until the dependency chain and
  complete frame execute with real captured resources on AMD.

## E-49 Downstream slot 3-5 weight capture readiness
- Date: 2026-08-31
- Artifact(s): `results/20260831_151119_module_trace_d3d12_selftest/` and
  `../deliverables/dlssnr-windows-feature18-downstream-weights-v17-20260831_151327.zip`
  (workspace deliverables directory).
- Address proof: graph parameters and resource tracing bind slots 3/4/5 to
  weight-resource offsets `0x44DA00`, `0x1390400`, and `0x7681600`.
- Harness proof: the D3D12 self-test reports PASS with one arm/record/dump cycle,
  zero changed bytes in the base plus three downstream weight windows, and the
  existing 255-byte output mutation still detected.
- Package proof: SHA-256 is
  `925633826900621AD46F9ADF96E1942D362BC227BF58D874264FFC967CC4F7C0`;
  streamed ZIP validation finds all 12 manifest records with matching sizes and
  hashes. This is capture readiness, not AMD execution or S7.

## E-50 Deterministic AMD execution of Swin slots 3-5
- Date: 2026-08-31
- Artifact(s): `results/20260831_151635_rtx5070_feature18_v17/`,
  `results/20260831_152915_amd_swin_slots3_5/` and
  `results/20260831_152935_amd_swin_slots3_5_repeat/`.
- Resource proof: the v17 result is PASS and supplies immutable, address-matched
  64 KiB weight views for all three slots.
- Dependency proof: each slot input hash equals the preceding AMD slot output
  hash. Wait buffers are initialized to the real predecessor-ready value, while
  distinct sentinel-filled release buffers prove exactly 1,025 and 984 chained
  publications. Slot 5 produces both its main and downsample outputs.
- Determinism proof: outputs and sync buffers for slots 3/4/5, plus the slot-5
  additional output, are bitwise identical across two independently allocated
  runs. Formal capture/execution/chain/determinism gates all pass.
- Boundary: original-PTX RTX numerical outputs are packaged but pending. This is
  real AMD neural execution beyond N1, not yet cross-vendor parity or S7.

## E-51 Cross-vendor parity for Swin slots 3-5
- Date: 2026-08-31
- Artifact(s): `results/20260831_153542_rtx5070_swin_slots3_5/` and
  `results/20260831_152915_amd_swin_slots3_5/comparison.json`.
- Integrity/execution proof: the returned original-PTX RTX 5070 result and every
  payload hash pass; launch grids, output sizes and release counts match the
  captured graph and AMD executions.
- Numerical proof: all three E4M3 main outputs exceed 0.99999 correlation and
  remain below 0.00424 NRMSE; exact-or-adjacent rates exceed 99.86%. The slot-5
  FP16 downsample output independently passes at 0.99999793 correlation and
  0.002033 NRMSE.
- Synchronization proof: each full sentinel-backed sync readback is byte-exact
  across RTX/RX. AMD repeats reproduce all outputs and sync buffers bitwise.
- Verdict: slots 3-5 numerical parity PASS. S7 remains open because this covers
  only the initial learned dependency chain, not all 156 graph slots.

## E-52 Slots 3-23 weight-capture readiness and slot-6 translation boundary
- Date: 2026-08-31
- Artifact(s): `results/20260831_154500_swin2h_slot6_baseline/`,
  `results/20260831_155000_swin2h_slot6_lowering/`,
  `results/20260831_155547_module_trace_d3d12_selftest/`, and
  `../deliverables/dlssnr-windows-feature18-swin-weights-v18-20260831_155721.zip`.
- Address proof: captured graph ABI bytes decode slots 3-23 to 21 explicit offsets
  inside the same model resource; every requested 64 KiB range is bounds-checked.
- Harness proof: the D3D12 self-test reports 21/21 present and immutable downstream
  windows, a successful arm/record/dump cycle, and the retained output mutation
  negative check. Package SHA-256 is
  `B155A65FBF590CDFB49285E08EE7F9E83FDE770CFF41C20D261532A3C599AE6A`.
- Translation proof: every one of slot 6's 713 first-pass diagnostics plus 28
  backend-only redundant shared-memory scopes is covered. A 4.2 MB helper-based
  compact form loads and resolves on RX 9070 XT, and the same transformation
  reproduces the already accepted slot-3 output and sync buffers bitwise.
- Boundary: the real slot-6 weight view is pending from v18, so slot-6 launch,
  RTX comparison and S7 are not claimed.

## E-53 Deterministic AMD execution of Swin 2h slots 6-8
- Date: 2026-08-31
- Artifact(s): `results/20260831_161539_rtx5070_feature18_v18/`,
  `results/20260831_163147_amd_swin2h_slots6_8_formal/` and
  `results/20260831_163153_amd_swin2h_slots6_8_formal_repeat/`.
- Capture proof: all v18 raw files match their recorded SHA-256 values; slot 6-8
  views are 65,536-byte immutable ranges at the graph-decoded model offsets.
- Execution proof: all three entries load, resolve, launch and synchronize on RX
  9070 XT with their captured `32x2` block geometry. Release counts are exactly
  240/273/252, and each slot consumes the previous AMD output hash.
- Determinism proof: two independently allocated dependency-chain runs have
  byte-identical output and sync hashes for all three slots.
- Boundary/negative gate: slot 9's statically proven 65,844-byte footprint exceeds
  the returned 65,536-byte view. It is deliberately not launched until v19
  returns extended data. Slots 6-8 also await original-PTX RTX numerical outputs,
  so this evidence is AMD execution PASS, not cross-vendor parity or S7.

## E-54 Deterministic AMD execution of Swin 2h slot 9
- Date: 2026-08-31
- Artifact(s): `results/20260831_163553_rtx5070_feature18_v19/`,
  `results/20260831_164100_amd_swin2h_slot9/` and
  `results/20260831_164300_amd_swin2h_slot9_repeat/`.
- Resource proof: v19 supplies a hash-validated immutable 267,776-byte slot-9
  view; the probe uploads the full range rather than truncating to 64 KiB.
- ABI/execution proof: captured block/grid are `32x2x1` and `20x13x1`; +48 is
  initialized as predecessor-ready, +72 receives the extra output, and no release
  pointer is patched. Module load, resolution, launch, synchronization and all
  three readbacks return success on RX 9070 XT.
- Content proof: logical main/extra tails are zero, both tensors are dense, and a
  second allocation reproduces their hashes plus the untouched sync hash exactly.
- Boundary: slots 6-9 now pass AMD dependency execution and determinism, while
  the packaged original-PTX RTX outputs are still required for numerical parity.

## E-55 Cross-vendor parity for Swin 2h slots 6-9
- Date: 2026-08-31
- Artifact(s): `results/20260831_170015_rtx5070_swin2h_slots6_9/`,
  `results/20260831_163147_amd_swin2h_slots6_8_formal/`,
  `results/20260831_163153_amd_swin2h_slots6_8_formal_repeat/`,
  `results/20260831_164100_amd_swin2h_slot9/`, and
  `results/20260831_164300_amd_swin2h_slot9_repeat/`.
- Integrity/execution proof: returned archive SHA-256 is
  `AE3AE26EFD12407DA03C070CF7031958179FBCAF089B654C161DB81222804F4E`;
  its RTX 5070 manifest, all 16 packaged payload hashes, four launches, captured
  grids and release counts pass. AMD and RTX consume the same captured inputs,
  parameters and relevant weight ranges.
- Numerical proof: slot 6-9 main FP16 tensors have correlations from
  0.99992376 to 0.99999300, NRMSE from 0.003742 to 0.012349, and at least
  98.2815% exact-or-adjacent agreement. Slot 9's extra FP16 tensor passes at
  0.99998421 correlation, 0.005619 NRMSE and 98.5099% exact-or-adjacent.
  All logical-allocation tails contain zero nonzero bytes on both vendors.
- Synchronization/determinism proof: all four full sync readbacks are byte-exact
  across RTX/RX. Independent AMD runs reproduce every output, extra output and
  sync buffer bitwise.
- Verdict: the complete 2h/64 family has cross-vendor numerical parity. Slots
  1-9 are now closed in dependency order, while slot 10 onward and complete-frame
  integration remain required for S7.

## E-56 Deterministic AMD execution of Swin 4h slots 10-15
- Date: 2026-08-31
- Artifact(s): `results/20260831_172000_swin4h_slots10_15_baseline/`,
  `results/20260831_172500_swin4h_slots10_15_lowering/`,
  `results/20260831_172222_amd_swin4h_slots10_15_formal_v2/`, and
  `results/20260831_172243_amd_swin4h_slots10_15_repeat/`.
- Translation/resource proof: all three isolated entries pass complete
  compatibility, E4M3, movmatrix and FP8-MMA lowering reports. Slot 10's
  197,632-byte capture reaches beyond the largest statically observed constant
  weight offset, while slots 11-15 use their full captured 1 MiB views.
- Execution proof: all six modules load, functions resolve, kernels launch and
  synchronize on RX 9070 XT. Captured `32x4` geometry and release counts
  60/77/66/70/60/0 match the graph; each slot input hash is the preceding AMD
  output hash.
- Content/determinism proof: every main and extra logical tensor is dense, all
  allocation tails are zero, and independent runs reproduce all 13 output/sync
  artifacts bitwise.
- Boundary: this is real deterministic AMD execution of the complete 4h/128
  family. Original-PTX RTX numerical parity is packaged but pending, so slots
  10-15 are not yet cross-vendor closed and S7 remains open.

## E-57 Full-graph one-shot RTX acquisition readiness
- Date: 2026-08-31
- Artifact(s): `results/20260831_175000_full_graph_inventory/`,
  `results/20260831_175202_module_trace_d3d12_selftest/`, and
  `deliverables/dlssnr-windows-full-graph-v20-20260831_175425.zip`.
- Coverage proof: static inventory contains all 156 ordered slots, all 43 used
  functions and all 15 runtime modules, with every used PTX entry present.
- Capture proof: the D3D12 self-test verifies pointer-to-resource interval
  resolution, pre/post GPU ordering, declared sizes, mutation detection, metadata
  emission and content-addressed deduplication. Texture/descriptor evidence is
  preserved separately instead of being falsely treated as a linear buffer.
- Package proof: SHA-256 is
  `BCB81054638714A0C67B3317C5F7AA3424443311A0F8B515CC176B96F340822C`;
  manifest validation, Windows PowerShell 5.1 parsing, ZIP path safety and the
  55-test suite pass.
- Boundary: the package must still run on the RTX 5070. Its readiness does not
  establish remaining-slot numerical parity or complete-frame AMD execution.

## E-58 Full-graph RTX acquisition returned and verified
- Date: 2026-08-31
- Artifact(s): external private archive
  `_full_graph_reference_result_20260831_181952(1).zip` and text-only analysis
  `results/20260831_183500_full_graph_reference_analysis/`.
- Archive proof: the 4,065,752,706-byte ZIP has SHA-256
  `A7C3B7AB3DFDE02DEBCF05665557D1FD48BB0E473C3AA09B7463969463362CFB`;
  all 962 content-addressed blobs plus 60 ordinary return files pass their
  declared sizes and SHA-256 values (1,022 records, 5,840,841,641 bytes).
- Runtime proof: Feature 18 is created and evaluated successfully while the host
  completes 300/300 evaluations. The bounded capture writes 881/881 windows with
  no capture failure, and the final 640x360 RGBA16F copy has identical input and
  output content.
- Coverage proof: slots 0-154 provide bounded buffer pre/post windows and slot
  155 (`cg2r_copy_kernel`) provides its descriptor-object final-copy snapshot.
  Together they cover all 156 static graph slots. Of the buffer-backed slots,
  153 have at least one changed window; slots 131 and 154 have valid but unchanged
  tracked windows. There are 370 windows at the 8 MiB bound.
- Verdict/boundary: RTX acquisition is complete and ready for offline kernel
  reconstruction. Bounded samples are not themselves AMD implementations, so S7
  and complete-frame parity remain open.

## E-59 Exact-state cross-vendor parity for Swin 4h slots 10-15
- Date: 2026-08-31
- Artifact(s): `results/20260831_191000_swin4h_full_graph_exact_state/`.
- State proof: each isolated RX 9070 XT launch consumes the exact RTX native
  before-state input, weight window, output allocation, wait/release synchronization
  region and frame-1 parameter ABI extracted from the hash-validated full-graph
  archive. The extended probe records that every requested initial state was
  uploaded rather than replaced with a zero/sentinel approximation.
- Execution proof: all six functions load, resolve, launch and synchronize twice
  on `AMD Radeon RX 9070 XT [ZLUDA]`. Grids/blocks and release counts match the
  graph, and slots 10-14 reproduce their native RTX after-state sync regions
  bitwise.
- Numerical proof: all six 245,760-element FP16 main tensors pass. Correlations
  range from 0.99992681 to 0.99999340, NRMSE from 0.003634 to 0.012098, and
  exact-or-adjacent fractions from 96.3550% to 98.3415%. Slot 15's 245,760-element
  E4M3 downsample passes at 0.99996693 correlation, 0.008133 NRMSE and 98.9750%
  exact-or-adjacent. Every uncaptured-write tail is bitwise identical to RTX.
- Determinism/verdict: the two AMD runs are bitwise identical for all outputs,
  downsample and sync artifacts. Slots 10-15 are now cross-vendor closed; slots
  1-15 are numerically closed in dependency order. This remains isolated replay,
  not S7 complete-frame execution.

## E-60 Exact-state cross-vendor parity for Swin 8h slots 16-23
- Date: 2026-08-31
- Artifact(s): `results/20260831_192000_swin8h_slots16_23_lowering/`,
  `results/20260831_193100_swin8h_full_graph_exact_state/`, and probe-offset
  regression `results/20260831_194000_swin8h_probe_offset_regression/`.
- Translation proof: the inpview, chained and ds-wait entries pass complete
  compatibility, E4M3, movmatrix and FP8-MMA lowering reports. All eight exact
  native-state launches load, resolve, execute and synchronize twice on RX 9070
  XT; release/wait regions reproduce RTX after-state bytes exactly and both AMD
  runs are bitwise deterministic.
- Numerical proof: the eight 122,880-element FP16 tensors have correlations from
  0.99987072 to 0.99999983 and NRMSE from 0.000583 to 0.016082. Their
  exact-or-adjacent fractions span 94.9683% to 99.9471%. Slot 17 is 94.9683%,
  reported as an advisory rather than a failure because PTX specifies that f16
  MMA accumulation order, rounding and subnormal handling are unspecified; its
  normative correlation/NRMSE/non-finite gates all pass. Slot 23's E4M3 extra
  tensor independently passes at 0.99999781 correlation, 0.002093 NRMSE and
  99.9536% exact-or-adjacent.
- Verdict/boundary: slots 1-23 are now numerically closed in dependency order.
  This is still exact-state isolated replay, not a complete 156-slot AMD frame;
  S7 remains open and the next contiguous family is Split 16h slots 24-56.

## E-61 Exact-state cross-vendor parity for Split 16h slots 24-56
- Date: 2026-08-31
- Artifact(s): `results/20260831_202000_split16h_slots24_56_lowering_v4/`
  and `results/20260831_210000_split16h_full_resource_exact_state/`.
- State/execution proof: all 33 graph-observed launches restore the exact
  27,807,744-byte activation arena, immutable 147,719,680-byte model arena,
  parameter ABI, wait/release state, grids and `32x8` blocks from the verified
  RTX archive. Every module loads, resolves, launches and synchronizes twice on
  RX 9070 XT; synchronization bytes match RTX and both AMD runs are bitwise
  deterministic.
- Oracle correction: the tracer's producer `after` snapshots are asynchronous
  and can be stale or partial. For slots 24-54, the output allocation is exactly
  aliased by slot N+1's input; that consumer snapshot occurs after the graph wait
  dependency and is the synchronization-settled RTX oracle. Slot 55's E4M3 pool
  output is likewise aliased by slot 56's input. The original immediate-after
  manifest is retained as `manifest_immediate_after_oracle.json`.
- Numerical proof: 30/33 main tensors are bitwise exact over their logical
  ranges. Slots 42, 50 and 54 pass with correlations 0.99998477, 0.99999317 and
  0.99999996; NRMSE 0.005522, 0.003696 and 0.000297; and at least 99.9007%
  exact-or-adjacent agreement. Slot 55's E4M3 pool tensor and slot 56's terminal
  E4M3 output are bitwise exact.
- Verdict/boundary: slots 1-56 are numerically closed in dependency order. This
  remains function-isolated exact-state replay rather than one integrated
  156-slot RX frame; S7 remains open and slots 57-104 are next.

## E-62 Exact-state cross-vendor parity for ViT 1D slots 57-98
- Date: 2026-08-31
- Artifact(s): `results/20260831_215000_vit1d_slots57_98_lowering_v2/`,
  `results/20260831_220100_vit1d_full_graph_exact_state/cases/`, and
  `results/20260831_221000_vit1d_full_graph_exact_state/`.
- Translation proof: nine ViT entries produce 54 strict PASS reports. In
  addition to established mbarrier/bulk-copy, E4M3, movmatrix and FP8-MMA
  lowering, `fence.release.gpu` is strengthened to `membar.gl` and 48 vector
  FP16x2 reductions are represented by packed FP16x2 atomic-CAS loops. All nine
  final entries independently load and resolve on RX 9070 XT.
- ABI/execution proof: a generalized resource-arena relocation map patches every
  graph-observed activation, synchronization and side-output pointer at its exact
  parameter and resource offset. All 42 slots execute twice from exact RTX
  activation/model arenas; every full post-launch arena repeats bitwise.
- Numerical proof: all 58 pointer-identical adjacent-consumer graph edges pass.
  Forty-two are bitwise exact. The 16 non-bitwise edges are QKV E4M3 tensors and
  all have 100% exact-or-adjacent codes; their worst correlation is 0.99999088
  and worst NRMSE is 0.004276.
- Boundary: 24 side accumulator outputs lack an adjacent pointer-identical
  consumer snapshot and remain explicitly diagnostic rather than being accepted
  against an asynchronous producer-after capture. Slots 1-98 are closed on
  their reliable dependency edges, but decoder slots 99-154 and integrated S7
  remain open.

## E-63 Exact-state decoder study through slot 154 — RGB conclusion corrected
- Date: 2026-08-31
- Artifact(s): `results/20260831_232000_decoder_slots99_154_lowering_v4/`,
  `results/20260831_230000_decoder_full_graph_exact_state/cases/`, and
  `results/20260831_231000_decoder_full_graph_exact_state/`.
- Translation/execution proof: 19 decoder entries pass 171 strict lowering
  reports and independently resolve on RX 9070 XT. All 56 captured slots restore
  exact RTX activation/model arenas and parameter ABIs, execute twice on the AMD
  adapter, and reproduce the complete post-launch arena bitwise between runs.
- Resource ABI proof: two unsupported formatted surface writes are represented
  by explicit linear `R16G16B16A16_FLOAT` addressing. Fourteen texture samples
  are zeroed only because the captured descriptor evidence proves the nonzero
  handle is a D3D12 null SRV and the other texture parameters are zero. The
  first attempt without the texture rewrite is retained and fails with CUDA 719.
- Historical numerical result: all 56 settled main edges appeared to pass decoded correlation/NRMSE and
  non-finite gates; 27 are bitwise exact. Slot 132 retains an explicit 94.1711%
  E4M3 exact-or-adjacent advisory, while its correlation is 0.99970255 and NRMSE
  is 0.024392. The reported final correlation 0.99999907 and NRMSE 0.001366
  included constant alpha=1 and are not a valid RGB image gate.
- Correction (2026-09-04): the same bytes evaluated on RGB only have correlation
  `0.421357178` and NRMSE `1.105763648`; alpha is exactly 1 but is excluded from
  the normative result. Execution/determinism evidence remains valid, while the
  claim that slot-154 numerical parity was closed is superseded by E-108.

## E-64 Literal integrated 156-slot RX frame — execution PASS, numerical FAIL
- Date: 2026-08-31
- Artifact(s): `results/20260831_234000_full_graph_integrated_plan/` and
  `results/20260831_235000_full_graph_integrated_rx9070xt/`.
- State/integration proof: the plan contains every contiguous slot 0-155, exact
  captured parameter blocks and pointer views, one 29,773,824-byte activation
  arena and one 147,719,680-byte model arena. Only slot 0's frame-start snapshot
  seeds activation memory; no later RTX intermediate is injected. The captured
  null SRV is represented by a valid zero CUDA texture, post output uses the
  explicit linear RGBA16F ABI, and slot 155 is a real GPU linear copy.
- AMD execution proof: 51 modules load and all 156 kernels launch/synchronize in
  one ZLUDA context, twice, on `AMD Radeon RX 9070 XT [ZLUDA]`. Both final images
  have identical SHA-256
  `886F32BA5CC1FCA7CE49FA38B851B74AEE741CF90FCFDDBDBF54EFAC16BE2A94`;
  every boundary checkpoint repeats bitwise and slot 155 equals slot 154 exactly.
- Honest numerical failure: slots 1/2 pass at NRMSE 0.001496/0.004044 and slot 5
  passes at 0.034495. Slot 9 is the first declared failure at NRMSE 0.100699,
  followed by slots 15/23/56/98 at 0.210452/0.272778/0.159568/0.195182. The
  final RGB channels are centered near 0.501 instead of the RTX reference near
  0.001; the high aggregate correlation is therefore not accepted. S7 remains
  false despite complete deterministic execution.

## E-65 Slot-3 internal arithmetic search and one-run RTX trace package
- Date: 2026-08-31
- Artifact(s): `results/20260901_000000_slot3_accumulation_diagnostic/`,
  `results/20260901_003000_slot3_mma_trace/`, and
  `deliverables/slot3_mma_trace_reference_20260831_232130.zip`.
- Negative search proof: exact-state slot 3 was rerun with FP8 MMA intermediate
  mantissas 10/12/14/16/18/20/22, FP16 accumulation, C-first/C-last, forward,
  reverse and 2/4/8-way reductions. The 18-22-bit and all reduction-order
  variants equal the baseline byte-for-byte; lower precision is worse. Accurate
  reciprocal math and scalarized packed-FP16 arithmetic also equal baseline.
  Flushing E4M3 inputs or f16-subnormal products is substantially worse.
- Remaining localization: the instrumented original slot-3 PTX exports A/B/C/D
  register fragments for all 256 FP8 MMAs in CTA (0,0), 327,680 bytes total,
  through unused parameter +64. The corresponding RX trace exists with SHA-256
  `2B2AB4CB7AC4E9E82EAA7F591ECC0643984722EA78E0FDA7C1E5467EADD2B28C`.
- Analyzer validation: replaying the current scalar FP8 model over that RX trace
  reproduces all 32,768 half-precision D results bitwise. This validates the
  fragment layout, lane mapping and arithmetic simulator before any RTX result
  is classified; the self-test report is
  `results/20260901_003000_slot3_mma_trace/amd_model_selftest_v3.json`.
- Standards basis: the original instruction is
  `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` targeting `sm_120`.
  NVIDIA PTX ISA 9.3 states that MMA accumulation order, rounding and subnormal
  handling are unspecified, while NVIDIA Tile IR 13.3 documents architecture-
  dependent FP8 internal precision and promotion. The analyzer therefore covers
  43 distinguishable models: FP32/exact/tree orders, RN and RZ mantissas 10-22,
  staged FP16 product/dot narrowing and subnormal flush behavior. Sources:
  `https://docs.nvidia.com/cuda/parallel-thread-execution/` and
  `https://docs.nvidia.com/cuda/tile-ir/13.3/sections/operations.html`.
- Cloud package: the 113,442,439-byte ZIP has SHA-256
  `E9AF6C29D95F1AE1D11ABBD0ED4272DA25BEECEF9E042E35082E2D071B98BB09`.
  It needs one RTX 50 run, not interactive debugging. The prepared analyzer will
  identify the first MMA whose bitwise-identical A/B/C fragments produce a
  different D fragment and rank 43 candidate accumulator models. The result
  receiver validates the NVIDIA execution manifest, trace size/nonzero count and
  SHA-256 without extracting untrusted members, then writes a content-addressed
  receipt and comparison. It rejects altered hashes, duplicate/unsafe paths and
  path traversal. All 87 Python regression tests pass.

## E-66 Executable accumulator candidates on RX 9070 XT
- Date: 2026-09-01
- Artifact(s): `results/20260901_010000_slot3_candidate_rx_smoke/`,
  `scripts/run_slot3_candidate_rx.py`, and
  `scripts/analyze_slot3_candidate_rx.py`.
- Rewrite coverage: 42 of the 43 trace-analysis candidates map directly to a
  selectable PTX lowering through `--candidate-model`; only CPU-diagnostic
  infinite-precision `exact_fsum` has no finite PTX implementation. Coverage
  includes every RN/RZ mantissa 10-22, sequential/reverse/interleaved/pairwise
  orders, FP16 product and staged-dot narrowing, and subnormal policies.
- Real AMD execution: representative new paths `mantissa17_rz_each_fma`,
  `f16_products_f32_acc`, `f32_pairwise_c_first`, and
  `f16_dot_then_f32_add_c` all load, resolve, launch and synchronize on
  `AMD Radeon RX 9070 XT [ZLUDA]` with the exact slot-3 activation/model arenas,
  ABI and `41x25x1` grid. Each writes about 1.964 million nonzero output bytes.
- RTX numerical comparison: FP16-product and pairwise variants equal the current
  baseline byte-for-byte at NRMSE 0.00339520. RZ is slightly worse at 0.00343504
  and staged-dot is worse at 0.00979034. No speculative candidate is promoted;
  the internal RTX trace remains the required model selector. This evidence is
  diagnostic and does not count as S7. All 87 Python tests pass.

## E-67 CTA-localized slot-3 RTX/RX MMA follow-up
- Date: 2026-09-01
- Artifact(s): `results/20260901_020000_slot3_mma_trace_cross_vendor/`,
  `results/20260901_023000_slot3_cta2_mma_trace/`, and
  `deliverables/slot3_cta2_mma_trace_reference_20260901_002923.zip`.
- Returned RTX proof: the validated RTX 5070 CTA `(0,0)` trace has SHA-256
  `2B2AB4CB7AC4E9E82EAA7F591ECC0643984722EA78E0FDA7C1E5467EADD2B28C`,
  exactly equal to the RX 9070 XT trace. A/B/C inputs and all 32,768 half D
  results across 256 MMA instructions are bitwise equal.
- Non-perturbation proof: the instrumented RTX and AMD complete outputs equal
  their respective original baselines byte-for-byte, while the two vendors still
  differ. The full comparison remains correlation 0.999994236, NRMSE
  0.003395203, 99.425252% exact bytes and 99.847972% exact-or-adjacent E4M3
  codes. Thus CTA `(0,0)` is genuinely equal but does not explain the full-grid
  divergence.
- Address localization: the first differing output byte is offset 1,949,
  corresponding to `(x=3,y=0)` and channel byte 413 in the 48x80x512 E4M3
  tensor. The original PTX's output-address calculation maps this element to CTA
  `(2,0)`; CTA `(0,0)` is a boundary tile.
- Targeted AMD proof: the generalized tracer records arbitrary CTA coordinates.
  CTA `(2,0)` loads, resolves, launches and synchronizes on a real
  `AMD Radeon RX 9070 XT [ZLUDA]`, exports 327,680 bytes with SHA-256
  `D532FFAFF64A260C3172EFE361EBB3308D793495ADF625F58F2749FE31209C3C`,
  and leaves the full AMD output at the established baseline hash.
- Targeted RTX package: the 113,442,573-byte ZIP explicitly records CTA
  `[2,0,0]` and has SHA-256
  `8DC7BAF79107F81D56C8D962A8949E98CBC171B2B7CC734C89CED284A164C143`.
  One RTX run is now required to distinguish MMA-internal divergence from the
  post-MMA epilogue. All 87 Python regression tests pass; S7 remains false.

## E-68 CTA `(2,0)` activation-first divergence and packed-FP16 trace
- Date: 2026-09-01
- Artifact(s): `results/20260901_030000_slot3_cta2_mma_trace_cross_vendor/`,
  `results/20260901_040000_slot3_cta2_f16_path_trace/`, and
  `deliverables/slot3_f16_path_trace_reference_20260901_005345.zip`.
- Returned-oracle integrity: the 10,824,101-byte RTX result ZIP has SHA-256
  `8665AB002125FB9F0D85FCC118BA7741B8D57AAF550339E43E745DE52BDD4B39`.
  Its manifest proves an RTX 5070 launch of CTA `(2,0)` and a 327,680-byte
  checkpoint with SHA-256
  `E660B8024D17CC3050695783AE906CD8BEA2AEE32525DFE216B51DB3C889CAF3`.
- First divergence: MMA inputs remain equal through instruction 215. At MMA 216,
  only A differs: lane 24, A byte 7 is RTX E4M3 `0x39` versus AMD `0x38`.
  B weights and C accumulators remain bitwise exact. The first D mismatch occurs
  only at MMA 220; total differences are 36 A bytes and 91 D half values.
- Source localization: the first A byte is the upper E4M3 code produced by
  `cvt.rn.satfinite.e4m3x2.f16x2 %rs220, %r2106`; `%r2106` is produced by a
  second packed-FP16 multiply from `%r2062` and `%r2024`, while `%r2062` is a
  first packed-FP16 product. Exhaustive half-to-E4M3 conversion was already exact,
  so changing MMA accumulation or E4M3 conversion is not justified.
- Targeted AMD proof: a 28-register path checkpoint executes on a real RX 9070 XT,
  exports 3,584 bytes with SHA-256
  `DD1EE39B9B3055B3E28BEFC1474B060D79F372E8DB3562423D4503B7E4449AA7`,
  and preserves the established full AMD output bitwise. The matching RTX package
  is 113,434,645 bytes with SHA-256
  `AAFA3ADB563D845D86FB251C2F9D3BB83B5AD1796F8FB8189CBCDC9FC1CA287A`.
  The safe receiver validates and compares every lane/register. All 90 tests pass;
  S7 remains false.

## E-69 Rsqrt trace separates harmless FP32 approximation from divergent input
- Date: 2026-09-01
- Artifact(s): `results/20260901_050000_slot3_f16_path_cross_vendor/`,
  `results/20260901_060000_slot3_cta2_rsqrt_trace/`, and
  `deliverables/slot3_rsqrt_trace_reference_20260901_010837.zip`.
- Returned-oracle integrity: the 10,627,844-byte RTX result ZIP has SHA-256
  `AA5EF926F3EB67482EF65712B83727BB7C6959CBE8FFAE1BC1A1C20E8019CDD7`.
  Its RTX 5070 trace has SHA-256
  `6814154F59064EB40A99F18E7F4ED0392A681DFF2410404E98B329D5AD238999`.
- First-register result: all eight sampled activation inputs `%r1751..%r1765`
  are bitwise exact. The first mismatch is `%r2002` lane 24: RTX packed half
  `0x3A553A55`, AMD `0x3A563A56`; `%r2004` has the same +1-half-ULP pattern in
  12 lanes. Downstream packed-FP16 products inherit the difference.
- Follow-up correction: the broad rsqrt return proves that, among 944 scalar
  samples with bitwise-identical half input, RTX and AMD produce the same final
  half in 944/944 cases. Their FP32 approximate values commonly differ by one ULP,
  but that difference is erased by half rounding. Every scale mismatch that
  survives instead has an already-different packed-half rsqrt input. Rsqrt is
  therefore diagnostic noise at the effective precision, not the root cause.
- Broad oracle: a new checkpoint records all 16 paired-rsqrt blocks, with packed
  input, both FP32 approximate outputs and packed half output per lane. Its real
  RX 9070 XT trace is 8,192 bytes, SHA-256
  `A6FB49DF07CCC58EC4E1FC5805ACD07A79F10266C061D19AB85C938550E5E729`,
  and does not perturb the full output. The matching RTX package is 113,434,679
  bytes, SHA-256
  `7267089E8A622E19A10C3E62AB2CF8956741D4A99369B03418157FDE560FA17F`.
  All 93 tests pass; S7 remains false.

## E-70 Packed-FP16 square/reduction tree isolated after rsqrt correction
- Date: 2026-09-01
- Artifact(s): `results/20260901_070000_slot3_rsqrt_cross_vendor/`,
  `results/20260901_080000_slot3_cta2_f16_reduction_trace/`, and
  `deliverables/slot3_f16_reduction_trace_reference_20260901_013136.zip`.
- Returned-oracle integrity: the 10,623,158-byte RTX ZIP has SHA-256
  `B0BA7B04BA3905C850073856D85360F2EB5083DDF6A84058E771427DE025F369`;
  its 8,192-byte RTX trace has SHA-256
  `43A15AB1FC94F6566FF35BACF787BC42D99C2EC47F41C856387E6EE9DB57C8F3`.
- Effective-precision result: for equal input, RTX-vs-AMD FP32 rsqrt deltas are
  mostly -1/0/+1 ULP, yet all 944 half outputs are identical. Mismatching half
  scales occur only in blocks whose half input differs before rsqrt.
- Local replay: starting from the bitwise-exact `%r1751..%r1765` activations,
  IEEE half RN square/add plus the captured butterfly tree reproduces AMD, not
  RTX. The final even reduction differs in lanes 24-27 and the odd reduction in
  12 lanes. This places the unresolved semantic difference within square,
  packed-half add, shuffle selection or their reduction ordering.
- New oracle: 34 registers cover activation inputs, eight squares, pair sums,
  both butterfly stages, half swaps and final sums. The real RX 9070 XT trace is
  4,352 bytes, SHA-256
  `45B68F0F23339A252B0E40E4707507D19A1F531FC903A01893A06ED0FE520C6D`,
  with unchanged full output. The 113,434,548-byte RTX package has SHA-256
  `81FE297A717DA2F6AA550C1DB8D0F2BEE1569438FB96287CD838F4E6C92C9F8E`.
  All 96 tests pass; S7 remains false.

## E-71 Reduction trace perturbation and selective RX correction search
- Date: 2026-09-01
- Artifact(s): `results/20260901_090100_slot3_f16_reduction_cross_vendor_corrected/`,
  `results/20260901_094000_slot3_local_reduction_ablation/`, and
  `results/20260901_101000_slot3_selective_reduction_greedy4/search_summary.json`.
- Returned-oracle integrity: the 10,628,081-byte RTX result ZIP has SHA-256
  `AEB57307DAEA65C0C1E249F2D4FBFBE03CF195E8930CF0A706A1FC416B09BC33`.
  Its 4,352-byte trace is bitwise equal to AMD, SHA-256
  `45B68F0F23339A252B0E40E4707507D19A1F531FC903A01893A06ED0FE520C6D`.
- Perturbation finding: the instrumented RTX full output is
  `87465D1987DC0FA003A3447F113885B6E1AFFBD0226CE202F950AAAD60539625`,
  not the uninstrumented `33FE600487C7CF89F8D8F238999D8E0C4602A865E33802CAB09C5D1DC50BD4F9`.
  The receiver therefore marks this trace as instrumentation-perturbed and not
  admissible as a literal oracle for the uninstrumented instruction stream.
- Interpretation: forcing stores at intermediate packed-half values makes RTX
  agree with strict AMD half materialization while changing RTX's final output.
  This is evidence for NVIDIA JIT retention/fusion of extra precision across the
  local reduction, but not proof that every local add must be fused.
- Real-RX search: targeted FP32 four-term reductions were tested in 27 real RX
  9070 XT executions. The best local set is `%r1855`, `%r1871`, `%r2235`,
  `%r2251`, `%r2283`. NRMSE falls from `0.003395202652` to `0.003251384490`
  (4.2359% relative), correlation rises from `0.9999942363` to `0.9999947143`,
  and sign mismatches fall from 290 to 281. Exact equality decreases from
  0.9942525 to 0.9938914, so this is a numerical-error improvement rather than
  bitwise parity. It remains exact-state slot-3 evidence, not S7.

## E-72 Selective reduction propagated through the complete 156-slot graph
- Date: 2026-09-01
- Artifact(s): `results/20260901_103000_full_graph_slot3_selective_rx9070xt/`,
  `results/20260901_104500_full_graph_slot3_compact_baseline_rx9070xt/`, and
  `results/20260901_103000_full_graph_slot3_selective_rx9070xt/variant_delta.json`.
- Execution proof: both the compact control and selective variant execute all 156
  slots twice on RX 9070 XT in one context with shared state. Each variant is
  internally bitwise deterministic and reaches the explicit slot-155 GPU copy.
- Confounder control: compact FP8 lowering without the selective reduction is
  bitwise identical to the prior expanded-lowering full graph at every captured
  boundary and final output (`886F32BA...`). The candidate delta therefore comes
  from the five selected reduction nodes.
- Propagation: slot-5 NRMSE improves from `0.034494942449` to `0.034464470895`
  (about 0.0883%), but slot 9 changes from `0.100699238487` to
  `0.100708414336` and slots 15/23/56 also worsen. Slot 98 improves slightly.
  Final NRMSE decreases only from `1.001179508632` to `1.001100800241`, a
  0.00786% relative improvement; the full-frame gate still fails.
- Verdict: the candidate is a real propagated numerical change, but the tiny and
  mixed full-graph effect does not justify promoting it to the production plan.
  The next root-cause target remains earlier slot-1/2 error amplification. S7 is
  false.

## E-73 N0 FP8 accumulation sweep and non-perturbing trace package
- Date: 2026-09-01
- Artifact(s): `results/20260901_110000_n0_fp8_candidate_search/`,
  `results/20260901_113000_n0_fp8_mma_trace/`, and
  `deliverables/n0_fp8_mma_trace_reference_20260901_023855.zip`.
- Candidate proof: thirteen complete `80x48` N0 executions run on RX 9070 XT.
  Nine FP32 accumulation orders and FP16-rounded products reproduce the current
  baseline bitwise; per-FMA FP16 rounding worsens output NRMSE to `0.0148333`,
  and staged FP16 dot models worsen it to `0.0067003`. No FP8 accumulation model
  improves the current `0.0015515` output NRMSE.
- Localization consequence: simple FP8 MMA accumulation order is ruled out as
  the remaining N0 correction. The first 16 FP16 MMA outputs were already
  bitwise exact, so the next boundary is all 256 FP8 MMA A/B/C/D fragments for
  CTA `(0,0)` followed by the post-MMA epilogue if those are exact.
- AMD trace proof: the instrumented and fully lowered entry executes a real
  one-CTA RX 9070 XT launch and exports 327,680 trace bytes from a bounded unused
  scratch window. All compatibility/lowering count gates pass.
- RTX package: the 1,623,066-byte archive has SHA-256
  `04E3948A01135DB0E63FDD8B8476324EAD0781945316ED7F006F06C671038B50`.
  It runs both instrumented and uninstrumented original PTX, records whether the
  complete one-CTA output is preserved, and only then exposes the trace as an
  uninstrumented-path oracle. The safe receiver validates archive layout,
  payload metadata, trace size/hash and perturbation fields. All 104 tests pass;
  S7 remains false.

## E-74 N0 first divergent FP8 activation localized before MMA 184
- Date: 2026-09-01
- Artifact(s): `results/20260901_120000_n0_fp8_mma_cross_vendor/`,
  `results/20260901_123000_n0_norm_path_trace/`, and
  `deliverables/n0_norm_path_trace_reference_20260901_025852.zip`.
- Oracle admissibility: RTX 5070 executes both the instrumented and untouched
  N0 PTX with identical one-CTA output SHA-256, so the 327,680-byte register
  trace is not rejected as a compiler-perturbed path.
- Localization: MMA operations 0-183 have bitwise-identical A/B/C/D fragments.
  At MMA 184, lane 7 A2 changes from RTX `9D171712` to RX `9D181712`; the
  collective operation then changes two output halves. The first difference is
  therefore an already-quantized activation input, not FP8 MMA accumulation.
- Path target: that byte comes from `%r2517 -> %rs237 -> %r2960`, with `%r2517`
  produced by the N0 max/rsqrt and two packed-FP16 scaling stages. A ten-stage,
  1,280-byte path tracer now executes successfully on the real RX 9070 XT.
  Its RTX package again includes an untouched-output perturbation control.
- Package: 1,598,634 bytes, SHA-256
  `13CC278997F51B10DC8FD0838C913B4EFDEB60E87DED00CDA000B2DBA9E0E256`.
  All 107 tests pass. This is numerical localization evidence, not S7.

## E-75 N0 normalization return and hidden-precision reduction hypothesis
- Date: 2026-09-01
- Artifact(s): `results/20260901_130000_n0_norm_path_cross_vendor/`,
  `results/20260901_133000_n0_reduction_path_trace/`, and
  `deliverables/n0_reduction_path_trace_reference_20260901_095422.zip`.
- Returned archive SHA-256 is
  `CED19D109CFDA75AA1051755F8375D5895AA2C848DDB17A6E7193D425A9CC509`.
  RTX instrumentation preserves the untouched one-CTA output, so the trace is
  admissible. `%r2164` is bitwise exact, while the next recorded max input
  `%r2329` first differs in 16 halves across lanes 4-7 and 20-23. Max, rsqrt and
  both scaling stages only propagate this earlier reduction error.
- `%r2329` is the final result of a 32-value square/reduction tree fed by four
  FP8 MMA outputs. Exhaustive enumeration of 352 visible-FP16 and reconstructed
  FP32-MMA rounding-boundary models finds no single exact RTX model. This rules
  out a uniform packed-add rounding switch and supports hidden precision/fusion
  spanning the MMA-to-reduction path.
- A 17-stage tracer records all four MMA outputs, four squares, pair/lane sums,
  shuffle reductions, half swap and final `%r2329`. Its fully lowered form
  executes successfully on RX 9070 XT and exports 2,176 bytes. The RTX package
  is  `n0_reduction_path_trace_reference_20260901_095422.zip`, SHA-256
  `92374DD4231A3BEDBF6D13BAA8400C278A5CA534D573B3BC30F380866008ACE8`.
  It retains the untouched-output perturbation control. S7 remains false.

## E-76 Materialization makes RTX exactly equal RX; scoped fusion ablation
- Date: 2026-09-01
- Artifact(s): `results/20260901_140000_n0_reduction_path_cross_vendor/`,
  `results/20260901_143000_n0_r2329_upper_fused_candidate/`,
  `results/20260901_144000_n0_r2329_fused/`,
  `results/20260901_144000_n0_r2329_min/`, and
  `results/20260901_144000_n0_r2329_policy_ablation.json`.
- Returned archive SHA-256 is
  `C57923795C23726253C98DCE8ECACC13687EA68CB79C32449C56C1C9439B5009`.
  All 17 visible reduction nodes are bitwise equal between RTX and RX, but the
  instrumented RTX output changes from untouched `65A4EF92...` to
  `6FEF4A75...`, exactly the RX output hash. This is direct evidence that
  materializing the packed-FP16 nodes removes NVIDIA's hidden fusion/precision.
- A scoped `%r2329` candidate computes a square/pair-rounded FP32 lane/xor2
  reduction and selects max, fused-only or min against the current staged result.
  All three complete `80x48` N0 launches execute on RX 9070 XT. Max and fused
  worsen output NRMSE by 1.12%/0.70%. Min improves output NRMSE by 0.425%, from
  `0.001551503979` to `0.001544907342`, but worsens scratch NRMSE by 1.25% and
  slightly reduces exactness. It is retained only for chained slot-2 testing,
  not promoted to the default lowering. S7 remains false.

## E-77 N0 min candidate rejected by complete graph; zero-input oracle package
- Date: 2026-09-01
- Artifact(s): `results/20260901_150000_full_graph_n0_r2329_min_rx9070xt/`,
  `results/20260901_153000_n0_zero_input/`, and
  `deliverables/n0_norm_path_zero_trace_reference_20260901_101646.zip`.
- Scope control: the variant plan changes only slot 1 from the accepted compact
  baseline. Two runs each execute all 156 slots in one RX 9070 XT context and
  produce identical final hashes, so execution and determinism pass.
- Rejection: compared with compact baseline, NRMSE worsens at slot 1
  (`0.001495641447 -> 0.001506111734`), slot 2
  (`0.004043566573 -> 0.004088084795`), slot 5, slot 9 and slot 15. Later mixed
  improvements do not survive the final boundary, which worsens from
  `1.001179508632` to `1.001208925587`. The isolated-input min candidate is not
  promoted.
- Cause of generalization failure: isolated N0 used the captured RGBA16F input,
  while the integrated replay models the captured null SRV with a zero RGBA16F
  texture. A zero-input N0 normalization trace now executes locally on RX and
  exports 1,280 bytes. RTX package SHA-256 is
  `E80BF839C5B71E7487D4FCACEFE86425860B00AD42CAE24DA3449889586EF810`.
  The package is 274,621 bytes and retains the untouched-output perturbation
  control. All 111 tests pass; S7 remains false.

## E-78 Zero-input normalization closes r2329 as an effective error source
- Date: 2026-09-01
- Artifact(s): `results/20260901_160000_n0_zero_norm_path_cross_vendor/`,
  `results/20260901_163000_n0_zero_fp8_mma_trace/`, and
  `deliverables/n0_zero_fp8_mma_trace_reference_20260901_102146.zip`.
- Returned archive SHA-256 is
  `97E155ED989A04F45669A1C12B222CB84C2FFB6144F22EB7F0142BE70860F97E`.
  RTX output preservation passes under the exact full-graph zero RGBA16F input.
- `%r2329` still differs, but only in lanes 20-23 (`RTX 3F5A`, `RX 3F58`).
  The following rsqrt rounds to bitwise-identical `%r2409`; both packed-FP16
  scales and final E4M3 pack are also exact. Thus the hidden reduction fusion is
  real but has no effective propagation through this traced zero-input branch.
- The next bounded gate is all 256 FP8 MMA A/B/C/D records under zero input. Its
  RX trace executes successfully and exports 327,680 bytes. RTX package size is
  299,056 bytes, SHA-256
  `D5E81341C7944CBE2E6E07599D77EA0CB0C09A36E37193E6B58F95D5BECF4C4E`.
  It retains the untouched-output perturbation control. All 111 tests pass;
  S7 remains false.

## E-79 Zero-input all-MMA return localizes the first effective error to MMA 178 B1
- Date: 2026-09-01
- Artifact(s): `results/20260901_170000_n0_zero_fp8_mma_cross_vendor/`,
  `results/20260901_173000_n0_zero_mma178_b_path_trace/`, and
  `deliverables/n0_mma178_b_path_trace_reference_20260901_103311.zip`.
- Returned archive SHA-256 is
  `10758BA9CE506E04A706BB83CB2BBC22A19945FC359D6DE3FD5D8958C4B410C0`.
  RTX 5070 execution, payload integrity and untouched-output preservation all
  pass, so the 327,680-byte trace is admissible.
- MMA 0-177 A/B/C/D records are bitwise equal. The first pre-fragment mismatch
  is MMA 178 lane 27 B1: RTX `%r4350=2020A69A`, RX
  `%r4350=2021A69A`. The collective MMA produces eight differing output halves.
  Only MMA 178, 186, 218 and 226 have pre-fragment byte differences (one each).
- PTX def-use proves `%r4350` is not a static weight fragment. Its differing
  `%rs266` byte is converted from `%r2842 = %r2561 * %r2803`; `%r2561` is the
  exact output of earlier MMA 146 and `%r2803` is the packed-half rsqrt scale of
  a subsequent square/reduction/max path.
- A nine-stage 1,152-byte short-path tracer now executes successfully on the
  real RX 9070 XT. The RTX package is 274,802 bytes, SHA-256
  `394ADADFE7F66EA16A3FD4A615AA6B72C3D6D32555F5A2A93EFDD58DD8FD5F5D`.
  It brackets MMA outputs, final reduction, max, rsqrt, multiply and E4M3 pack,
  with an untouched-output perturbation control. All 114 tests pass; S7 remains
  false.

## E-80 CTA `(0,0)` r2723 is internal-only; first observable mismatch CTA package
- Date: 2026-09-01
- Artifact(s): `results/20260901_180000_n0_mma178_b_path_cross_vendor/`,
  `results/20260901_183000_n0_r2723_observed_ablation/`,
  `results/20260901_190000_n0_cta1_0_fp8_mma_trace/`, and
  `deliverables/n0_cta1_0_fp8_mma_trace_reference_20260901_104646.zip`.
- Returned short-path archive SHA-256 is
  `71CEB44993FD2C97D40BD28F71FD63DAAFE1238268E4259A2BA0B077654D9F7D`.
  RTX output preservation passes. MMA 146/147 outputs are exact; `%r2723` first
  differs in lanes 24-27 (`RTX 41204120`, `RX 411F411F`), then propagates through
  max, rsqrt and scaling to the single MMA-178 B1 byte.
- None of 352 visible-half/reconstructed-MMA rounding-boundary models reproduces
  the complete RTX `%r2723` vector. A scoped observed-value ablation confirms
  that CTA `(0,0)` uninstrumented scratch and output were already bitwise equal
  across RTX/RX, so this internal divergence dies before the kernel boundary and
  must not be promoted as the slot-1 correction.
- The actual slot-1 boundary has 2,555 byte mismatches across 1,190 CTAs; the
  first is output byte 515, mapping to CTA `(1,0)`, tile byte 3. A new tracer
  launches the full `80x48` grid but predicates all trace stores on CTA `(1,0)`.
  The probe now supports an opt-in 327,680-byte scratch extension, placing trace
  data after the untouched 7,864,320-byte production scratch region.
- Real RX 9070 XT full-grid execution passes and the traced output is bitwise
  equal to its clean control. RTX package size is 311,262 bytes, SHA-256
  `A97B1CFE91808F235E7433285BEB7ADC297D045B1CB8AA7C77A5B8F6B4C96669`.
  All 116 tests pass; S7 remains false.

## E-81 First observable slot-1 CTA diverges before MMA 176
- Date: 2026-09-01
- Artifact(s): `results/20260901_200000_n0_cta1_0_fp8_mma_cross_vendor/`,
  `results/20260901_203000_n0_cta1_mma176_dual_path_trace/`, and
  `deliverables/n0_cta1_mma176_dual_path_trace_reference_20260901_105529.zip`.
- Returned archive SHA-256 is
  `5A29D1A2A0B39C67FC3B8F11CE480EAF0C10DEB4280BD6FBEA142D5AB3EF0965`.
  RTX 5070 full-grid execution and untouched-output preservation pass.
- CTA `(1,0)` MMA 0-175 A/B/C/D fragments are bitwise exact. MMA 176 first
  differs in two already-quantized inputs: lane 13 B0 byte 0 (`90` vs `8F`)
  and lane 16 A3 byte 2 (`22` vs `21`). The collective operation immediately
  produces 18 differing D halves. Later operations contain 65 pre-fragment byte
  and 190 D-half differences in total.
- The two bytes are dynamic activations, not weights. A3 comes from
  `%r2150 -> %r2401 -> %r2455 -> %r2511 -> %r2941`; B0 comes from
  `%r2537 -> %r2713 -> %r2753 -> %r2793 -> %r2832 -> %r4337`. Earlier MMA
  sources are exact, so both candidates lie in reduction/normalization paths.
- A combined 16-stage, 2,048-byte tracer covers both chains in one full-grid
  CTA-conditioned run. Its RX 9070 XT output equals the clean control. RTX
  package size is 275,651 bytes, SHA-256
  `30462EE7B56DF3B834809D9BFF60B4EC9CCF237DA64CBE00361E3CBEF954512B`.
  All 116 tests pass; S7 remains false.

## E-82 Dual reduction return, causal ablation, and corrected CTA mapping
- Date: 2026-09-01
- Artifact(s): `results/20260901_210000_n0_cta1_mma176_dual_path_cross_vendor/`,
  `results/20260901_213000_n0_cta1_reduction_ablation/`,
  `results/20260901_220000_n0_cta8_0_fp8_mma_trace/`, and
  `deliverables/n0_cta8_0_fp8_mma_trace_reference_20260901_110340.zip`.
- Returned archive SHA-256 is
  `F82C1FEDB86EF888D7A7C622A40979ABB0A406BD2BE9083493385D068C37E9D5`;
  RTX output preservation passes. The A path first differs at `%r2321` and the
  B path at `%r2713`; their preceding MMA outputs `%r2150/%r2537` are exact.
- Scoped real-RX causal ablations show A-only removes two slot-1 mismatch bytes
  and improves NRMSE from `0.001495641447` to `0.001495577493`. B-only is
  bitwise identical to baseline, so the observed B0 difference dies before the
  kernel output. A+B equals A-only. These value-specific rewrites are diagnostic
  and are not promoted.
- The original CTA mapping assumed contiguous 512-byte tiles and was rejected.
  Static evaluation of the actual four epilogue address formulas with dimensions
  `384x640 -> 192x320` proves output byte 515 is CTA `(8,0)`, lane 0, first
  4-byte store. CTA `(1,0)` instead accounts for two later bytes.
- A CTA `(8,0)` all-MMA tracer now passes on RX 9070 XT with full-grid output
  preservation. RTX package size is 311,274 bytes, SHA-256
  `A6291F2A3DA1C8BE38A72CCBED2338A83D8A8EA9127014725013321984DF5394`.
  All 116 tests pass; S7 remains false.

## E-83 CTA `(8,0)` begins upstream of FP8 MMA; FP16 producer package
- Date: 2026-09-01
- Artifact(s): `results/20260901_230000_n0_cta8_0_fp8_mma_cross_vendor/`,
  `results/20260901_233000_n0_cta8_0_f16_mma_trace/`, and
  `deliverables/n0_cta8_0_f16_mma_trace_reference_20260901_111609.zip`.
- Returned CTA `(8,0)` archive SHA-256 is
  `E0D26DECFE43C9DB118C004421C1A962C67FA87BE3D2996BAB8018362ACA276F`;
  RTX 5070 execution and untouched-output preservation pass. Across the 256
  FP8 MMAs there are 1,298 pre-fragment byte differences and 1,610 differing D
  halves, beginning at MMA 0.
- The first pre-fragment difference is lane 17, A2 byte 1: RTX `0x26`, RX
  `0x25`. A2 is `%r645`; the byte is the upper E4M3 element of `%rs36`, produced
  by `cvt.rn.satfinite.e4m3x2.f16x2 %rs36, %r556`. `%r556` is D0 of the third
  initial `m16n8k16` FP16 MMA. Therefore the first real slot-1 mismatch already
  precedes FP8 MMA arithmetic; changing the FP8 accumulation model cannot fix
  this cause.
- A selected-CTA tracer now records A/B/C/D fragments for all 16 preceding FP16
  MMAs during the complete `80x48` launch. Its 20,480-byte RX 9070 XT trace has
  SHA-256 `AF414C1B1766E5CE1F7BF0933180792ACB114DE9030FEE7C0949885953F0FBD6`.
  The traced output remains bitwise equal to the clean output
  (`8E2383A2721DFD10BC1996AD577DC460D3FCAC3CB793A4FFF2BE41853F916189`).
  RTX package SHA-256 is
  `93BC1C4B124D81EAAB93B0F7CFF15FA387762E1920A233F39792C412FDF0E999`.
  All 117 tests pass; S7 remains false.

## E-84 FP16 producer return localizes shared-memory Box-Muller feature
- Date: 2026-09-01
- Artifact(s): `results/20260901_235000_n0_cta8_0_f16_mma_cross_vendor/`,
  `results/20260902_001000_n0_cta8_0_box_muller_trace/`, and
  `deliverables/n0_cta8_0_box_muller_trace_reference_20260901_112952.zip`.
- Returned FP16 archive SHA-256 is
  `908B6DFE18D1D70079567DC0049B41C8751342FEF6C27AB7D071E52739D0B4D3`;
  RTX 5070 full-grid execution and untouched-output preservation pass.
- FP16 MMAs 0-3 share the same sole pre-fragment difference: lane 16 A0 byte 0
  is RTX `0x23`, RX `0x24`. They produce 16 differing D halves. The A0 load is
  `%r487` from shared offset 128, written by lane 8 `%r444` byte 0, which is the
  low byte of `%rs15`.
- `%rs15` is the FP16 conversion of Box-Muller normal `%r149`, computed from an
  exact integer hash followed by `lg2.approx`, `sqrt.approx`, `cos.approx` and
  FP32 multiplication. Thus the first real slot-1 error precedes both FP16 and
  FP8 MMA and is neither a texture nor weight discrepancy.
- The next package captures eight immediate post-definition boundaries for all
  64 dynamic samples. This controlled diagnostic deliberately admits only the
  captured defining-instruction results; its downstream instrumented output is
  excluded from evidence. Package SHA-256 is
  `32A4F0E060CD24E5579FBEE8B55C80DD3CC77CDDB7169780A190CB8095682EF5`.
  All 118 tests pass; S7 remains false.

## E-85 Box-Muller return proves `lg2.approx`; full-grid curve package
- Date: 2026-09-01
- Artifact(s): `results/20260902_010000_n0_cta8_0_box_muller_cross_vendor/`,
  `results/20260902_013000_n0_lg2_bias_u1/`,
  `results/20260902_020000_n0_lg2_full_grid_trace/`, and
  `deliverables/n0_lg2_full_grid_trace_reference_20260901_115154.zip`.
- Returned archive SHA-256 is
  `0EBEA3C3F4038CE94B830B5D03A57E828D85BC0ED224D96F73D788C7A5C1DC87`;
  RTX 5070 execution and full-output preservation pass.
- Integer hash and uniform inputs are bitwise exact for all 64 samples. The
  first difference is `lg2.approx.ftz.f32`: 63/64 outputs differ. Later mismatch
  counts are 48 at sqrt, 57 at cos, 59 at the FP32 normal, but only sample 8
  survives FP16 rounding (`0x3923` RTX, `0x3924` RX), exactly matching the
  previously localized shared-memory byte.
- A one-ULP RX lg2 bias toward positive infinity removes the first real output
  mismatch byte 515 in a clean full-grid run. It slightly worsens aggregate N0
  parity, so a uniform bias is causal evidence only and is not promoted.
- The next non-perturbing tracer captures 245,760 production-domain lg2
  input/output pairs (1,966,080 bytes) across all `80x48` CTAs. RX trace SHA-256
  is `895FDAE296741465BB41CA3223C7C53888C5119FB88ECD1F55B957421122E6F3`;
  its full output remains bitwise baseline. RTX package SHA-256 is
  `FC48B9C1E6426F3DC0D5D99ABFE5C88AC4B5A84F066A836067ED6DCD6E65B9FF`.
  The package README includes the complete PowerShell command and returned ZIP
  pattern. This revision first moves `%ctaid.x/y` into ordinary registers for
  strict NVIDIA PTX JIT compatibility and reports the original probe log before
  any missing-trace extraction error. All 120 tests pass; S7 remains false.

## E-86 Full-grid `lg2` return reconstructs a useful RTX 5070 SFU curve
- Date: 2026-09-01
- Artifact(s): `results/20260902_030000_n0_lg2_full_grid_cross_vendor/`,
  `models/nvidia_sm120_lg2_64seg.json`,
  `results/20260902_040000_n0_lg2_curve_candidate/`, and
  `deliverables/n0_cta35_0_box_muller_trace_reference_20260901_120441.zip`.
- Returned archive SHA-256 is
  `D4E324FF8268D8CDF87A1F88945331BD2A5144C086B7D539D9D959B2D3FD6B0A`;
  RTX 5070 execution and full-output preservation pass. All 245,760 production
  inputs are bitwise equal across RTX/RX, while 242,124 lg2 outputs differ.
- Piecewise-quadratic fitting has a clear accuracy knee at 64 mantissa segments.
  The untuned 64-segment PTX lowering improves clean slot-1 output mismatches
  from 2,555 to 2,491, fixes 78 old bytes while introducing 14, and improves
  RMSE from `0.000621048648` to `0.000607795130`. Intercept tuning for direct
  FP16 boundaries worsens the real output to 2,512 mismatches and is rejected.
- The original first mismatch byte 515 is removed. The new first mismatch byte
  2,243 maps through the actual epilogue stores to CTA `(35,0)`. An RX 9070 XT
  trace of the curve-lowered CTA passes; the paired RTX package records
  hash/uniform/lg2/sqrt/phase/cos/normal/FP16 boundaries. Package SHA-256 is
  `A42744F2BA5BE3C4FC2053CC989DAD9C850C44B51752DE62219D6505738790A2`.
  Its README contains the full PowerShell command. S7 remains false.

## E-87 CTA `(35,0)` shows the first traced half output is already exact
- Date: 2026-09-01
- Artifact(s): `results/20260902_060000_n0_cta35_0_box_muller_curve_cross_vendor/`,
  `results/20260902_070000_n0_cta35_0_box_muller_full_curve_trace/`, and
  `deliverables/n0_cta35_0_box_muller_trace_reference_20260901_121249.zip`.
- Returned archive SHA-256 is
  `6F5D7539BC57823B010189449730E996592929A3C086FD437C3BD51F04A17225`;
  RTX 5070 execution, payload integrity and untouched-output preservation pass.
- With the 64-segment curve on RX, hash/uniform remain exact. Intermediate
  mismatches are 35/64 at lg2, 33/64 at sqrt, 55/64 at cos and 55/64 at the
  first FP32 normal, but all 64 values are exact after `%rs15` FP16 conversion.
  Therefore clean output byte 2,243 is not caused by the previously traced
  first normal output.
- The follow-up trace covers all three generated normal halves (`%rs15-17`) and
  the inputs/results of both lg2, both sqrt, sin and both cos branches: 22 stages,
  5,632 bytes. RX 9070 XT execution passes. RTX package SHA-256 is
  `D52B5452C2BF3AEE9ABD31F60D626EDD77834200746855AEA219D310544F0549`;
  its README contains the exact PowerShell command. S7 remains false.

## E-88 All CTA `(35,0)` Box-Muller halves are exact after the curve fix
- Date: 2026-09-01
- Artifact(s): `results/20260902_080000_n0_cta35_0_box_muller_full_curve_cross_vendor/`,
  `results/20260902_090000_n0_cta35_0_fp8_mma_curve_trace/`, and
  `deliverables/n0_cta35_0_fp8_mma_trace_reference_20260901_121641.zip`.
- Returned archive SHA-256 is
  `B3EF46ABD57B199CF53E4B523597AB193096F8D7F7BC9F268931F5DCED16D380`;
  RTX 5070 execution, payload integrity and output preservation pass.
- The 64-segment RX curve has intermediate mismatches in both lg2/sqrt paths,
  sine and both cosine paths, but `%rs15`, `%rs16` and `%rs17` are each exact
  for all 64 samples. Hence every Box-Muller feature actually written to shared
  memory is exact at CTA `(35,0)`; none causes clean output byte 2,243.
- The first remaining clean-output difference is an independent later error.
  A new full-grid selected-CTA tracer captures A/B/C/D fragments for all 256
  FP8 MMAs under the curve candidate. RX 9070 XT execution passes and exports
  327,680 bytes. RTX package SHA-256 is
  `0FF31E1827F9D45FA7267B73F938C1B874998BEE5E0437BE05E10800C7BB0263`.
  Its README contains the complete PowerShell commands. S7 remains false.

## E-89 CTA `(35,0)` residual first enters FP8 MMA 176 through A
- Date: 2026-09-01
- Artifact(s): `results/20260902_100000_n0_cta35_0_fp8_mma_curve_cross_vendor/`,
  `results/20260902_110000_n0_cta35_0_mma176_a_path_trace/`, and
  `deliverables/n0_cta35_0_mma176_a_path_trace_reference_20260901_123136.zip`.
- Returned archive SHA-256 is
  `CF8898A55FCE142B14ECE8EE652C64CDE03AC2B7DB06DF4785E72CD11182E286`;
  RTX 5070 tracing preserves the clean output. MMAs 0-175 are fully bitwise
  exact. MMA 176 first receives two differing A bytes: lane 0 A2 byte 3
  (`0x1E` RTX, `0x1D` RX) and lane 30 A1 byte 0 (`0x0E` RTX, `0x0D` RX).
- The scalar FP32 MMA model exactly reproduces all RTX D halves from the RTX
  inputs, so accumulation is not the cause. The two bytes are E4M3 conversions
  of `%r2509` and `%r2506`, built from earlier exact MMA 128/131 outputs after
  two packed-FP16 square-reduction/rsqrt/scale paths.
- One follow-up package records every visible node in both complete reductions,
  45 stages and 5,760 bytes, rather than requiring a second reduction round.
  RX 9070 XT execution passes. RTX package SHA-256 is
  `07F3C368CAA127306C03865416545D1B37BADCE75BAEF3DFAE81D6D3DF2DED9C`;
  its README contains the exact PowerShell command. S7 remains false.

## E-90 Combined 45-stage trace exposes hidden RTX fusion; single-store sweep
- Date: 2026-09-01
- Artifact(s): `results/20260902_120000_n0_cta35_0_mma176_a_path_cross_vendor/`,
  `results/20260902_130000_n0_cta35_0_mma176_a_fusion_sweep/`, and
  `deliverables/n0_cta35_0_mma176_a_fusion_sweep_reference_20260901_124132.zip`.
- Returned archive SHA-256 is
  `549F4FC1499365824176A8C7AAE6A393F5BFA034FAA0F2DA33AE4FA065615058`.
  Payload integrity and both executions pass, but the 45-store RTX output hash
  changes from baseline `856C0190...80ED8` to `AB45C458...86BCC`; the result is
  formally `REJECTED_PERTURBED` and is not used as a clean numerical oracle.
- The rejected RTX trace nevertheless becomes bitwise identical to the
  output-preserving AMD trace (SHA-256 `EDD707F1...B97B`). This is diagnostic
  evidence that materializing the PTX intermediates suppresses an NVIDIA hidden
  fusion, exactly the behavior the next experiment must localize.
- One package now runs a baseline plus 45 variants, each containing exactly one
  observation store. It returns a per-stage output-preservation map and 128-byte
  trace for each stage in one cloud command. Package SHA-256 is
  `9BDD9ED1188A44F3AAC27C4EBBB9A276B68A85C552EDE30F84CC0FA95E2C74F4`;
  its README contains the exact command. S7 remains false.

## E-91 Single-store sweep identifies exact half2 fusion semantics
- Date: 2026-09-01
- Artifact(s): `results/20260902_140000_n0_cta35_0_mma176_a_fusion_sweep_return/`
  and `results/20260902_150000_n0_square_pair_fusion_candidate/`.
- Returned archive SHA-256 is
  `4E9860BEB48849C064D861940C0D0B894B299FCD0D4FDA4A02577F940EA46E08`;
  payload integrity, RTX 5070 execution and all 45 probes pass. Thirty-three
  variants preserve the baseline output. The only 12 perturbing observations
  are the four square and two first pair-sum registers in each of the two
  MMA-176 A normalization branches. The first clean RTX/RX divergence is the
  following lane sum: stage 10 `%r2252`, then stage 32 `%r2262`.
- Seven candidate rounding models were fitted against both clean RTX lane sums.
  Exactly one is bitwise exact for all 128 finite packed-half samples:
  `round_f16(x_left*x_left + round_f16(x_right*x_right))` for each pair,
  followed by the ordinary FP16 lane add. Strict per-operation FP16 matches
  only 110/128 samples; full-FP32 fusion matches 100/128.
- Replacing all 32 N0 add-of-two-square pairs with explicit
  `fma.rn.f16x2(left,left,rounded_right_square)` on RX 9070 XT reduces clean
  output mismatches from 2,491 to 241 bytes and NRMSE from `0.001463723639` to
  `0.000446189948`. It fixes 2,250 old mismatches and introduces zero new ones.
  The first remaining byte is offset 36,306, which the verified epilogue map
  assigns to CTA `(7,1)`. A single-command follow-up package captures the three
  consumed Box-Muller FP16 values and all 256 FP8 MMA fragments in separate,
  output-controlled runs: `deliverables/n0_cta7_1_residual_bundle_reference_20260901_130330.zip`,
  SHA-256 `DCCB875A35AD0A28845923EA904E13124E7D8083DE723D2F922D3453881858DC`.
  Both corresponding RX traces preserve the clean candidate output. This is a
  strong numerical result but not S7 or game integration evidence.

## E-92 CTA `(7,1)` residual begins at the first Box-Muller half
- Date: 2026-09-01
- Artifact(s): `results/20260902_170000_n0_cta7_1_residual_cross_vendor/`,
  `results/20260902_180000_n0_lg2_model_search/`, and
  `results/20260902_185000_n0_lg2_1024seg_candidate/`.
- Returned archive SHA-256 is
  `55220D03C60065245E9BB8878FA503BEB4A4E5AD843A9CE348CFC83EE98FCF88`.
  Payload integrity, RTX 5070 execution and output preservation pass for both
  controlled runs. Of the three Box-Muller halves, only `%rs15` differs: sample
  16 is RTX `0x33B3` versus RX `0x33B2`, and sample 32 is RTX `0xB960` versus
  RX `0xB95F`; `%rs16` and `%rs17` are fully exact.
- The first network divergence is MMA 16, whose C fragments already contain
  three differing bytes. This excludes a new MMA arithmetic or half2-fusion
  cause and links the current first output residual to the consumed random
  feature boundary.
- Broader curve replacements were tested and rejected on clean RX output:
  128-segment quadratic gives 262 mismatches, 64-segment cubic gives 261, and
  1024-segment quadratic gives 272, versus 241 for the retained 64-segment
  quadratic. Half-safe biases to segments 20 and 22 also become net regressions
  once large enough to alter the target halves. The existing model remains the
  accepted candidate. The next package independently observes only `sqrt0` and
  `cos0`, both already verified non-perturbing on RX:
  `deliverables/n0_cta7_1_box_stage_sweep_reference_20260901_135851.zip`,
  SHA-256 `8A1CB69F7140FE8077CA760E2A27EF89B0B6DFF7924CC768549104223BBDBEB3`.
  S7 and game integration remain pending.

## E-93 RTX 4090 D cannot execute the current Blackwell trace package
- Date: 2026-09-04
- Artifact(s): `results/20260904_4090d_box_stage_compatibility/`.
- Host proof: Ubuntu 22.04, 2x RTX 4090 D (`sm_89`), driver 550.144.03 and
  CUDA 12.4. The tested package hash matches E-92.
- The untouched PTX 8.7/`sm_120` baseline and both stage variants all stop at
  `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (222), before any kernel execution.
- A declaration-only PTX 8.4/`sm_89` adaptation advances into JIT parsing, but
  all three modules fail with `CUDA_ERROR_INVALID_PTX` (218). `ptxas` locates
  the first blocker at the network's
  `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` instruction.
- Verdict: this server can still supply Ada-specific scalar/control data, but it
  cannot supply the current full-kernel `sqrt0`/`cos0` traces without an
  Ada-specific kernel patch. Retargeting declarations is not such a patch; RTX
  5070 remains the admissible oracle for the current package. S7 remains false.

## E-94 Clean CTA `(7,1)` observations first diverge at cosine
- Date: 2026-09-04
- Artifact(s): `results/20260904_003000_n0_cta7_1_box_stage_cross_vendor/`
  and `results/20260904_010000_n0_cos_full_grid_trace/`.
- Returned archive SHA-256 is
  `E4B97CFB25C228C2071D9AD9AECCFC87785FF3DD7A5CD80F7F5DDD972B7697E4`.
  Payload integrity, RTX 5070 execution and untouched-output preservation pass
  independently for both observations; both outputs retain baseline hash
  `856C0190...80ED8`.
- Across the selected CTA, rounded `sqrt0` differs in 20/64 samples and rounded
  `cos0` in 54/64. At the only two samples whose consumed first half differed
  in E-92, `sqrt0` is bitwise exact. `cos0` differs at both: sample 16 is RTX
  `0x3E8C0CBF` versus RX `0x3E8C0CBC`, and sample 32 is RTX `0xBF3CF511`
  versus RX `0xBF3CF510`.
- This establishes cosine as the first clean observable target divergence. It
  does not claim that rounded stored values expose hidden precision consumed by
  a fused downstream path. A new output-controlled package captures all 245,760
  cosine input/output pairs:
  `deliverables/n0_cos_full_grid_trace_reference_20260904_002125.zip`, SHA-256
  `E74480B3ACB344610AF5A047384C818D9C0949899D9EC1AEB175B561C917F97D`.
  Its RX 9070 XT counterpart preserves the accepted candidate output. S7 and
  game integration remain pending.

## E-95 Full-grid cosine oracle confirms architecture-specific SFU behavior
- Date: 2026-09-04
- Artifact(s): `results/20260904_020000_n0_cos_full_grid_cross_vendor/`,
  `results/20260904_030000_n0_cos_correction_candidates/`, and
  `results/20260904_040000_n0_cos_target_segment_ablation/`.
- Returned archive SHA-256 is
  `7C8A664490B5F1C814A06B2B8F2AF0D799F3EC5F6B5F2F33A462C618219FB305`.
  Its RTX 5070 manifest, payload hashes, launch geometry and untouched-output
  control all pass. All 245,760 cosine inputs match RX bitwise; 202,961 rounded
  outputs differ. Mean absolute error over mismatches is
  `1.2701491624636053e-07`, with maximum `5.513429641723633e-07`.
- Two additive full-domain correction fits do not improve the retained N0:
  1,024 segments produce 247 differing output bytes and 4,096 segments produce
  248, versus the accepted 241. Both are rejected.
- A deliberately overfit two-bin causal ablation at the two E-94 angles reduces
  the real RX output from 241 to 237 differing bytes, fixes four old differences
  and introduces none. Widths 12/14/16/18 all produce the same output. This
  proves that the observed cosine error reaches final N0 E4M3 output, but the
  ablation is not accepted as a general lowering.
- The new first output difference is offset 79089, mapped through the captured
  output layout to CTA `(35,3)`. Its matching clean RTX stage package is
  `deliverables/n0_cta35_3_box_stage_sweep_reference_20260904_004223.zip`,
  SHA-256 `8DE783F8AD97570B805BC0ABFF97041048ABFA79FAA16071B94887EDB64A4979`.
  S7 and game integration remain pending.

## E-96 CTA `(35,3)` clean stage sweep is valid but not yet causal
- Date: 2026-09-04
- Artifact(s): `results/20260904_060000_n0_cta35_3_box_stage_cross_vendor/`
  and `results/20260904_070000_n0_cta35_3_residual_bundle/`.
- Returned archive SHA-256 is
  `72AB4744351BC47A413350ECF8DAF136A4FC185EB6D98D82F31C589AF70D030B`.
  Its RTX 5070 identity, CTA `(35,3)`, payload integrity, both launches and
  untouched-output controls pass.
- Rounded `sqrt0` differs in 20/64 samples and rounded `cos0` in 52/64. Since
  both stages differ within the CTA, these independent observations do not by
  themselves identify which consumed FP16 value causes output offset 79089.
- Matching output-preserving RX traces for the three consumed FP16 Box-Muller
  values and all A/B/C/D fragments of all 256 FP8 MMAs have been captured. The
  corresponding one-run RTX bundle is
  `deliverables/n0_cta35_3_residual_bundle_reference_20260904_004947.zip`,
  SHA-256 `9BC6ADA82639818B25473BF76EEE112EDCBF16F4594EDDE8F452B7C73D086CD0`.
  S7 remains false.

## E-97 CTA `(35,3)` residual is a cosine-driven consumed-FP16 boundary
- Date: 2026-09-04
- Artifact(s): `results/20260904_080000_n0_cta35_3_residual_cross_vendor/`,
  `results/20260904_090000_n0_cos_observed_input_ablation/`, and
  `results/20260904_100000_n0_normal0_boundary_full_grid_trace/`.
- Returned archive SHA-256 is
  `A6D1133EF35663E3ED1343C6D3DE73D246E8A0E8DFCA87A360A7820BD8416DE2`.
  RTX 5070 identity, target CTA, payload integrity, all three launches and both
  instrumented-output preservation controls pass.
- Of the three FP16 Box-Muller values consumed by the network, only
  `normal0_f16` sample 54 differs: RTX `0x2D76`, RX `0x2D77`. MMA 12 first
  receives one different A-fragment byte (`0xAA` versus `0xA9`); its B/C inputs
  are not the origin, and the subsequent D differences are downstream.
- At sample 54, sqrt is bitwise exact (`0x3F47C3F5`) while cosine is RTX
  `0x3DE005CC` versus RX `0x3DE005D8`. Cross multiplication proves that replacing
  cosine alone changes the consumed half from `0x2D77` to the RTX `0x2D76`;
  replacing sqrt does not. An exact-input diagnostic fixes output offsets 79089
  and 1062143 with no additions, taking the diagnostic residual from 237 to 235.
  It remains overfit and is not promoted.
- Instead of continuing CTA-by-CTA at the new first residual `(20,4)`, the next
  one-run package captures sqrt and consumed normal0 FP16 over all 245,760
  samples: `deliverables/n0_normal0_boundary_full_grid_trace_reference_20260904_010138.zip`,
  SHA-256 `6F27162EE6724D09AD2B3C9DC9EF7B2D7586D243C45E47B9DEDFF1B7155F181B`.
  S7 remains false.

## E-98 Full-grid consumed-FP16 analysis yields a no-regression cosine candidate
- Date: 2026-09-04
- Artifact(s): `results/20260904_110000_n0_normal0_boundary_full_grid_cross_vendor/`
  and `results/20260904_120000_n0_cos_boundary_safe_candidates/`.
- Returned archive SHA-256 is
  `E3612132B6BAECF32F2D017F8E87E50928C42911F8A19FE89D5C24048B66E9C3`.
  RTX 5070 identity, payload integrity, full-grid shape and untouched-output
  controls pass; the trace hash is
  `1B421897AC4BE5C5DA4B0868DADC7EEE539E1241D6CF96666A672F14F8BDBEDA`.
- Across 245,760 samples, rounded sqrt differs in 87,577 cases, but consumed
  `normal0_f16` differs in only 356 (99.855143% exact). Cosine inputs are bitwise
  identical. Of the 356 boundary differences, 345 have different rounded
  cosine outputs; using the RTX cosine with the AMD sqrt leaves a theoretical
  33 boundary differences, quantifying the separate sqrt floor.
- A conservative 4,096-segment quadratic candidate enables only 88 segments
  that fixed at least one observed boundary and added none. On the complete RX
  N0 output it fixes 39 old differences, adds zero, and reduces 241 to 202 bytes.
  NRMSE improves to `0.000416785661`; two RX runs are byte-identical with SHA-256
  `F954825E...DE0442`. This becomes the new accepted N0 candidate, not S7.
- The first remaining output difference is offset 83218 in CTA `(20,4)`. A
  matching output-preserving RX residual trace exists, and the next RTX bundle
  is `deliverables/n0_cta20_4_residual_bundle_reference_20260904_011215.zip`,
  SHA-256 `38DA2D0332F1C96DC2EB7F8FD7B57B4F45AAA8E227F12CCB3E2AE5E02B5B653C`.

## E-99 CTA `(20,4)` residual begins at the second Box-Muller path
- Date: 2026-09-04
- Artifact(s): `results/20260904_150000_n0_cta20_4_residual_cross_vendor/`
  and `results/20260904_160000_n0_normal2_full_grid_traces/`.
- Returned archive SHA-256 is
  `C577DCA9CCD324BC8C8FE04EB8CAB74A731D868AD43978A13BBEA6408321DCC9`.
  The RTX 5070 identity, payload integrity, CTA coordinates, both launches and
  untouched-output controls pass.
- `normal0_f16` and `normal1_f16` are bitwise exact for all 64 samples. Only
  `normal2_f16` differs, at sample 2: RTX `0xBC8A` versus RX `0xBC8B`.
  MMA 0 already receives the changed value in A (lane 10, register 0, byte 2,
  RTX `0x21` versus RX `0x22`), so MMA is downstream rather than the root.
- Two output-preserving RX full-grid traces now cover the second Box-Muller
  path: phase/cosine and sqrt/consumed-FP16 boundary. Both variants reproduce
  the accepted output SHA-256 `F954825E...DE0442`. The matching one-run RTX
  package is
  `deliverables/n0_normal2_full_grid_traces_reference_20260904_012329.zip`,
  SHA-256 `69066751ECBB5D4A6FDCFB0E71572C08C98D3F8016BB636E1F068D304403FA2B`.
  The accepted candidate remains at 202 differing bytes; S7 remains false.

## E-100 Full-grid normal2 oracle yields a 154-byte no-regression candidate
- Date: 2026-09-04
- Artifact(s): `results/20260904_170000_n0_normal2_full_grid_cross_vendor/`,
  `results/20260904_190000_n0_cos1_high_resolution_candidates/`, and
  `results/20260904_210000_n0_cos1_ultrahigh_resolution_candidates/`.
- Returned archive SHA-256 is
  `6D5CD8BD4E521E4E1990CB204C038EF5D20FA35B8B4FA814900CEB72F7D94C41`.
  RTX 5070 identity, package integrity, both trace hashes and both independent
  untouched-output controls pass.
- Across 245,760 samples, phase inputs are bitwise identical. Rounded `cos1`
  differs in 202,606 samples, rounded `sqrt1` in 88,347, but consumed
  `normal2_f16` in only 342. Replacing only cosine with RTX values leaves 29
  theoretical boundary differences; replacing only sqrt leaves 309.
- A boundary-safe 32,768-segment second-cosine model enables 147 observed-safe
  segments. The complete RX N0 output fixes 48 old differences, adds zero and
  improves 202 to 154 bytes, with NRMSE `0.000367638428` and repeat hash
  `61E14BDD...D764983`. The 65,536-segment alternative reaches 145 but adds one
  new output error and is rejected as underconstrained.
- The new first residual is offset 143935, CTA `(8,7)`. Matching RX consumed
  FP16 and all-MMA traces preserve the accepted output. The next RTX package is
  `deliverables/n0_cta8_7_residual_bundle_reference_20260904_014338.zip`,
  SHA-256 `A6FA7AA7D45CE0387904781F06D2F8AEAF18B694926B81F28808D0E489D98F64`.
  This remains isolated zero-input N0 evidence, not S7 or broad input-domain
  validation.

## E-101 CTA `(8,7)` closes MMA and the accepted N0 propagates through the graph
- Date: 2026-09-04
- Artifact(s): `results/20260904_230000_n0_cta8_7_residual_cross_vendor/`,
  `results/20260904_240000_n0_nonzero_captured_frame_candidate/`, and
  `results/20260904_250000_full_graph_n0_cos01_accepted_rx9070xt/`.
- Returned archive SHA-256 is
  `926BB1F01B89CFDF3FDF315F9A7BF4972032532BCF2034944702EC76B67DEF4A`.
  RTX 5070 identity, archive/payload integrity, CTA `(8,7)`, both launches and
  both untouched-output controls pass.
- `normal0_f16` and `normal2_f16` are exact across all 64 samples. Only
  `normal1_f16` differs at samples 30 and 44, each by one FP16 ULP. MMA 12 first
  receives one changed A-fragment byte. Nineteen scalar accumulation models,
  including ordinary FP32 sequential/pairwise orders, exactly reproduce RTX D
  when supplied RTX fragments; the observed AMD D differences are downstream
  from the already changed input, not a new MMA arithmetic defect.
- The accepted zero-input candidate executes on the independent nonzero captured
  RGBA16F input twice with identical SHA-256 `E0D85EDA...97F05`. Against its RTX
  reference, 168/1,966,080 bytes differ, exactness is 99.991455%, NRMSE is
  `0.000383619918`, and correlation is `0.999999926418`. Relative to the
  square-pair-fusion baseline, the two cosine corrections fix 94 bytes and add
  zero, establishing initial cross-input generalization rather than coordinate
  memorization.
- A scope-controlled plan replaces only slot 1 in the accepted compact 156-slot
  replay. Both RX runs execute all 156 launches in one context, inject no RTX
  intermediate state and produce final hash `6899A43F...A5E26`. NRMSE improves
  versus compact baseline at every recorded learned boundary: slot 1 by 75.42%,
  slot 2 by 58.88%, slot 5 by 36.75%, slot 9 by 13.94%, slot 15 by 9.71%, slot
  23 by 8.61%, slot 56 by 6.19% and slot 98 by 1.33%. Slot 9 crosses below the
  strict 0.1 threshold; among those coarse checkpoints, slot 15 is the first
  failure at `0.190025019`. E-102 adds same-capture fine checkpoints and
  supersedes the localization target with slot 6.
  The legacy all-channel RGBA NRMSE improves from `1.001179509` to
  `1.001121844`; E-108 supersedes that alpha-contaminated metric with RGB-only
  evaluation. This remains propagation evidence, not S7 or visual parity.

## E-102 Same-capture early-Swin checkpoints and causal state sensitivity
- Date: 2026-09-04
- Artifact(s):
  `results/20260904_320000_full_graph_early_swin_same_capture_rx9070xt/`,
  `results/20260904_330000_full_graph_inject_slot2_same_capture_rx9070xt/`,
  `results/20260904_350000_full_graph_inject_slot3_same_capture_rx9070xt/`,
  `results/20260904_360000_full_graph_inject_slot4_same_capture_rx9070xt/`,
  `results/20260904_340000_full_graph_inject_slot5_same_capture_rx9070xt/`, and
  `results/20260904_370000_full_graph_early_swin_sensitivity/analysis.json`.
- Reference control: slot 3/4/6/7/8 settled states are extracted from adjacent
  consumer `before` blobs in the same full-graph RTX archive used by the
  integrated plan. An earlier exploratory comparison against function-isolated
  outputs was rejected after their input-state hashes failed to match this
  capture; the affected result directories carry explicit invalidity markers.
- Native AMD result: two injection-free 156-slot runs are bitwise deterministic
  and retain final SHA-256 `6899A43F...A5E26`. Slot 3/4/5 NRMSE is
  `0.005225894`/`0.014557257`/`0.021818147` (all pass). Slot 6 is the first
  strict failure at `0.102493965`; slot 7/8 reaches
  `0.255073016`/`0.318781847`, then slot 9's downsampled E4M3 boundary returns
  to `0.086658668` (pass).
- Causal control: RTX state is injected only after one selected boundary and is
  explicitly excluded from S7. Exact slot-2 state reduces slot 6 to
  `0.093327669` but leaves slot 7/8 failing. Exact slot-3 and slot-4 states reduce
  slot 6 to `0.042745668` and `0.024215944`. Exact slot-5 state makes all 2h
  slots 6/7/8 pass at `0.015492492`/`0.049286892`/`0.084323278`.
- Verdict: the 2h family is largely correct for exact upstream state; it amplifies
  small, structured errors accumulated by the 1h slot 3-5 chain. Correct slot
  3-5 propagation before revisiting 2h. Injection runs are causal diagnostics,
  never deployment or S7 evidence. Final image parity and live game integration
  remain open.

## E-103 Previous slot-3 precision candidate fails the new causal objective
- Date: 2026-09-04
- Artifact(s):
  `results/20260904_380000_full_graph_n0_cos01_slot34_selective_rx9070xt/` and
  `results/20260904_380000_full_graph_n0_cos01_slot34_selective_rx9070xt/delta_vs_native.json`.
- Scope control: only slots 3 and 4 use the previous five-node selective FP32
  reduction PTX; N0 remains the accepted 154-byte candidate. Two complete runs
  execute all 156 slots, are bitwise deterministic, and inject no RTX state.
- Result: slot-3 NRMSE improves from `0.005225894` to `0.005146451`, but slot 4,
  slot 5 and the first failing slot 6 worsen to `0.014578233`, `0.022040888` and
  `0.103664437`. The candidate worsens every checkpoint from slot 4 through slot
  23. It improves slot 56/98 and final NRMSE, but the final change is only
  `1.001121844 -> 1.001043618` and remains far from the image gate.
- Verdict: reject this candidate again. Isolated slot-3 NRMSE is not a reliable
  optimization objective; the next search must rank changes by same-capture
  slot-5 and slot-6 propagation while preserving slot 4. S7 remains false.

## E-104 Slot-3 reduction search does not close the slot-6 boundary
- Date: 2026-09-04
- Artifact(s): `results/20260904_400000_full_graph_slot34_single_reduction_search/`
  and `results/20260904_401000_full_graph_slot34_combo_reduction_search/`.
- Sixteen single-node and nine combination variants were scored by the causal
  slot-6 objective in complete deterministic graph runs. The best single change,
  `%r1893`, reaches NRMSE `0.102352458`; the best tested three-node combination
  reaches `0.102685665`. Neither crosses the 0.1 gate and no candidate is promoted.

## E-105 Slot-3 error sensitivity is distributed across the output tensor
- Date: 2026-09-04
- Artifact(s): `results/20260904_410000_full_graph_slot3_partial_injection_sweep/`
  and `results/20260904_411000_full_graph_slot3_chunk6_subrange_injection_sweep/`.
- Eight disjoint 245,760-byte replacements from the same-capture RTX slot-3 state
  each independently make slot 6 pass. The best is chunk 6 at arena-relative
  offset 1,474,560, yielding slot-6 NRMSE `0.097584746`.
- Splitting that chunk into eight 30,720-byte ranges no longer makes slot 6 pass;
  the best subrange replaces 569 differing bytes and reaches `0.101890374`.
  Sensitivity is therefore distributed and nonlinear, not one corrupt row/channel
  or one reduction instruction. The tiled physical layout is not relabeled as an
  image/channel map without further proof. These RTX injections are non-S7.

## E-106 Same-capture decoder checkpoints show non-monotonic propagation
- Date: 2026-09-04
- Artifact(s): `results/20260904_420000_full_graph_decoder_same_capture_rx9070xt/`,
  `results/20260904_430000_full_graph_inject_slot153_same_capture_rx9070xt/`, and
  `results/20260904_440000_full_graph_inject_n0_scratch_rx9070xt/` through
  `results/20260904_470000_full_graph_inject_n0_scratch_main_late_rx9070xt/`.
- Same-capture slots 99-152 generally remain at NRMSE 0.11-0.34, while slot 153
  compresses back to `0.068902005` and passes. The N0 scratch consumed by slot 154
  is already close to RTX (`0.000481114`). Exact main/scratch replacements, both
  early and immediately before slot 154, do not remove the former final failure.
  This separates the post-block problem from those two visible input tensors.

## E-107 Full-arena and parameter-56 controls expose a post-block resource branch
- Date: 2026-09-04
- Artifact(s): `results/20260904_480000_full_graph_inject_full_pre_slot154_arena_rx9070xt/`,
  `results/20260904_490000_full_graph_inject_full_arena_raw_resource_rx9070xt/`, and
  `results/20260904_500000_full_graph_postblock_param56_fix_rx9070xt/`.
- Restoring all 29,773,824 activation-arena bytes from the exact same RTX state
  before slot 154 leaves the old biased output unchanged. Changing only the replay
  policy so captured parameter offset 56 (`0x0000080200009801`) is preserved instead
  of forced to zero removes the roughly +0.5 RGB bias. With an exact arena the old
  aggregate NRMSE becomes `0.001365541`; without any RTX injection it is
  `0.001379663`. Both executions repeat bitwise.
- The runner now preserves that parameter and a regression test prevents its
  accidental clearing. E-109 supersedes the earlier scalar/ABI interpretation:
  offset 56 is a merged texture/sampler handle. Preserving its nonzero value keeps
  the correct branch direction, but zero-lowered samples are still not the real
  resource contents.

## E-108 Constant alpha had hidden the remaining post-block RGB failure
- Date: 2026-09-04
- Artifact(s): `results/20260831_231000_decoder_full_graph_exact_state/analysis_rgb.json`,
  `results/20260904_490000_full_graph_inject_full_arena_raw_resource_rx9070xt/analysis_rgb.json`,
  and `results/20260904_500000_full_graph_postblock_param56_fix_rx9070xt/analysis_rgb.json`.
- RGBA16F analysis now excludes alpha from normative correlation/NRMSE and reports
  alpha exactness separately. The prior near-perfect aggregate result was dominated
  by alpha=1. Exact-state slot 154 actually has RGB NRMSE/correlation
  `1.105763648`/`0.421357178`; the injection-free complete graph has
  `1.117199`/`0.427049`. Its RGB RMSE is `0.000689036`, but the captured reference
  itself is near black, so low absolute error cannot establish real-frame
  visual generalization.
- Verdict: the parameter fix is retained, 156-launch AMD execution remains real
  and deterministic, but S7/S8 stay false. The next correctness target is
  the slot-154 resource model; slot 6 remains the earliest strict intermediate failure.
  All 158 Python regression tests pass.

## E-109 Post-block pre-store parity passes; zero-texture model is disproved
- Date: 2026-09-04
- Artifact(s): `results/20260904_530000_postblock_store_trace_cross_vendor/`
  and `results/20260904_540000_full_graph_real_post_texture_rx9070xt/`.
- Returned archive SHA-256 is
  `9ABB2D9DAFBC48BE26EB52F958C579D4104F5C45AE34470A94562FB430D1467A`.
  Its RTX 5070 execution, payload integrity, nonzero 8,128,512-byte trace and
  repeat hash all validate. Both sites cover 122,880 active records; their union
  covers every 640x384 padded coordinate exactly once and reconstructs the RTX
  transformed output byte-for-byte.
- RTX/RX f32 values immediately before both stores pass every component gate.
  RGB correlations range from `0.999999640068` to `0.999999762846`; NRMSE ranges
  from `0.000688708943` to `0.000848467409`; alpha is bitwise exact. Thus the
  tested transformed post-block neural arithmetic is no longer the dominant
  final-frame blocker.
- The returned runner correctly reports FAIL for its independent original-output
  control. Transformed RTX versus captured original RGB is correlation
  `0.421336395`, NRMSE `1.105928620`. Descriptor/resource tracing resolves the
  nonzero offset-56 handle to a real 640x360 `DXGI_FORMAT_R16G16B16A16_FLOAT`
  SRV with point filtering and border addressing, contradicting the old
  zero-texture report. A deterministic RX trial using the final image as a
  guessed texture reaches correlation `0.826286171` but NRMSE `2.152537713` and
  is rejected. Actual texture pixels are required.
- The prepared acquisition package is
  `deliverables/dlssnr-windows-feature18-slot154-texture-v20-20260904_122414.zip`,
  SHA-256 `70A78BD6191C6B5C74286059BB71CF93B669AC22F873331AA931902B5D31226E`.
  It adds an ordered D3D12 readback of `post_texture_input.raw`; this is pending
  external RTX execution and does not count as S7. The dedicated receiver
  validates revision, Feature-18 execution, resource binding, dimensions, format,
  byte count and SHA-256 before admitting the texture. All 162 Python regression
  tests pass.

## E-110 End-of-run texture readback is stale; pre-launch capture required
- Date: 2026-09-04
- Artifact(s): `results/20260904_550000_post_texture_snapshot_rtx5070/` and
  `results/20260904_571000_full_graph_captured_post_texture_rx9070xt/`.
- The v20 RTX return has archive SHA-256
  `A6604AF3984FA92562870A292BE072747E504BA363B0E83630977C9DA3EF0BD0`
  and valid `300/300` Feature-18 execution. Its 640x360 RGBA16F snapshot hash is
  `8DF6D450B5A7CB358B9E8373AF9FD9304E5912389C644F6C4BC66068380E88A3`,
  but every byte/value is zero.
- RX 9070 XT replay with that snapshot executes all 156 launches twice and is
  bitwise deterministic, but produces the exact old zero-texture output hash
  `8CCF23A57196BEC8D6DAFBFDC8867078EBC100A75ABC2863EC27230980877D25`.
  Final RGB remains correlation `0.427048989` / NRMSE `1.117199202`; therefore
  the late snapshot is rejected as stale and contributes no S7/S8 evidence.
- v21 records the copy immediately before frame-1 slot 154 on the same command
  list. D3D12 self-test passes, the receiver requires timing metadata and nonzero
  contents, and the package is
  `deliverables/dlssnr-windows-feature18-slot154-prelaunch-texture-v21-20260904_124948.zip`
  with SHA-256 `2E0B016B0660DC002E995291F7D8D9D0B80A320512BA04FCE603836C7F4794DC`.

## E-111 Pre-launch texture is genuinely zero; resource-content hypothesis rejected

- Date: 2026-09-04
- Artifact(s): `results/20260904_600000_post_texture_prelaunch_rtx5070/` and
  `deliverables/postblock_native_stage_sweep_reference_20260904_131704.zip`.
- The returned v21 archive SHA-256 is
  `111D7B337FD630C7B1C92D96A280F4A7F331FB6FE0210CC25F481D57CDE7BCE9`.
  RTX 5070 Feature 18 completes `300/300`; the frame-1 snapshot is taken on the
  same D3D12 command list immediately before slot 154 and contains zero nonzero
  bytes. The snapshot hash is the RGBA16F all-zero hash
  `8DF6D450B5A7CB358B9E8373AF9FD9304E5912389C644F6C4BC66068380E88A3`.
- Module events link the 640x360 format-10 resource creation to its SRV, the
  merged CUDA texture handle, capture arm and frame-1 slot-154 bind. Independent
  copy input/output readbacks are equal and match the original reference hash
  `CD556E0D9C2B958CF1D189412B33AA7BC52AB178369D4F5469F22E85E71FD243`.
  The v21 runner's FAIL is only its obsolete nonzero-data assertion; the capture
  itself passes the corrected strict receiver.
- Verdict: v20 was not demonstrably stale; v21 proves zero is the live input.
  Missing texture content cannot explain why the transformed post-block output
  differs from the original. The next package runs seven cumulative PTX stages
  twice on RTX with a native CUDA RGBA16F surface and zero point/border texture.
  Its ZIP SHA-256 is
  `E2B1155B4AADD6BA7F10059CFD32D863428616E66B606CC088E16F6D7510DDA7`.
  This remains diagnostic evidence, not S7/S8. All 164 tests pass.

## E-112 Native post-block sweep rejects early lowerings and exposes incomplete state

- Date: 2026-09-04
- Artifact(s): `results/20260904_610000_postblock_native_stage_sweep_rtx5070/`
  and `deliverables/dlssnr-windows-feature18-slot154-full-activation-v22-20260904_134339.zip`.
- Returned archive SHA-256 is
  `E811AE239817C6ECB4A0880415087F57184E5C26A5DBB3700229A4F8735BF259`.
  RTX 5070 runs original, compatibility, E4M3 and movmatrix stages twice. All
  eight successful outputs are byte-identical with SHA-256
  `8ADB4DE9E238DDA7A1155BD817E22236963A067C52C58A182AF50A09BAE6E8C6`.
  Thus those three cumulative rewrites are exonerated for this input.
- Even the original native PTX standalone result differs from live copy output:
  RGB correlation is `0.421336375` and NRMSE `1.105928784`, essentially the same
  result as the earlier transformed replay. No lowering can be assigned while
  the original baseline fails. The prior arena was synthesized from bounded
  pointer windows and zero-filled gaps; that reconstruction is now the primary
  missing state candidate.
- FP8-MMA, FP16-MMA and decoder-compat files fail module load twice with code 218
  (`CUDA_ERROR_INVALID_PTX`). This is an independent generated-PTX defect, not a
  measured numerical divergence.
- v22 captures the complete D3D12 buffer containing slot154 parameter offsets 0
  and 8 on the same command list before launch. Package SHA-256 is
  `AE02FEB20041F8D38BABF15B53BEC75218341A1C54CB475026C5F384CCD49A63`.
  The D3D12 tracer self-test and all 165 Python tests pass. S7/S8 remain false.

## E-113 Complete activation rejects the arena hypothesis; frame oracle was misaligned

- Date: 2026-09-04
- Artifact(s): `results/20260904_620000_post_activation_snapshot_rtx5070/`,
  `results/20260904_630000_full_graph_live_pre_slot154_activation_rx9070xt/`,
  and `deliverables/dlssnr-windows-feature18-frame-aligned-output-v23-20260904_141326.zip`.
- Returned v22 archive SHA-256 is
  `D76AB57978EF1E35B8C0C9601255BE8FE47E8061DA0E295200C8B582052EA8FB`.
  Its 27,807,744-byte frame-1 activation buffer has SHA-256
  `925AE733BEF13D46BADB79341372985C74C8A164E4624424A4B51AD713F6CD33`.
  All Feature-18, capture-ordering, D3D12-resource, zero-texture and final-copy
  controls pass.
- Injecting that complete buffer before slot154 on RX 9070 XT executes all 156
  slots twice and repeats exactly, but the end-of-run comparison remains at RGB
  correlation `0.4213571776971408` / NRMSE `1.1057636484104916`. The 11,876
  bytes that differed from the bounded reconstruction are therefore not the
  relevant missing post-block input.
- The captured launch parameters identify the real experimental defect. In
  frame 1, slot154 texture-object fields +88 and +96 are zero; by frame 61 they
  are nonzero. Previous standalone inputs were from frame 1 while
  `copy_input.raw` was read after all 300 evaluations. That output is not a
  same-frame oracle, so its mismatch cannot be attributed to AMD arithmetic.
- v23 adds an ordered D3D12 readback of slot155's input immediately before the
  frame-1 slot155 launch, i.e. slot154's just-written frame-1 output. Package
  SHA-256 is
  `4B03F9655FEBD24C168D6EB535F5979EA3EF72B3378269193C08E17C46619592`.
  The D3D12 self-test and all 167 Python tests pass. S7/S8 remain open pending
  return and replay against this aligned oracle.

## E-114 Inline frame-aligned snapshot accepted structurally — SUPERSEDED BY E-116

- Date: 2026-09-04
- Artifact(s): `results/20260904_640000_frame_aligned_post_output_rtx5070/`,
  `results/20260904_650000_frame_aligned_post_output_reanalysis/`, and
  `deliverables/dlssnr-windows-feature18-post-surface-before-after-v24-20260904_143351.zip`.
- Returned v23 archive SHA-256 is
  `DAA579C3CA2F5C70D09CA87F57A7E4B1568E39D8EB7088DAB7CCFCA64EE4977C`.
  It passes package, Feature-18, trace, activation, texture, D3D12-resource and
  copy controls. The captured frame-1 output is 1,843,200 bytes with SHA-256
  `94A5E9E672FAD2E139F610472687FCA72992C0564E10004806216BCBEEBA5DF3`.
- Frame 1 and the end-of-run texture differ in 1,473,125 bytes, proving temporal
  misalignment. Correcting only the oracle does not close parity: the RX 9070 XT
  complete-activation replay scored correlation `0.338362441` / NRMSE
  `1.497234543`. E-116 later invalidates that numerical comparison because the
  v23 inline image is only a partial wavefront; zero initialization is correct.
- The v23 and original full-graph slot154 parameter blocks differ only at pointer
  offsets 0, 8, 24 and 104; all non-address bytes are identical. v24 captures
  the same output surface immediately before slot154 and immediately before
  slot155, testing whether conditional stores require preloading destination
  pixels. Package SHA-256 is
  `A765AC98799997C476C86FB3CE049B53F0C65D621478F124A701760A3CA9282A`.
  S7/S8 remain open.

## E-115 Slot154 inline before/after capture — SUPERSEDED AS NUMERICAL ORACLE

- Date: 2026-09-04
- Artifact(s): `results/20260904_660000_frame_aligned_post_surface_rtx5070/`
  and `deliverables/postblock_exact_contract_reference_20260904_144431.zip`.
- Returned v24 archive SHA-256 is
  `DA9292BC46297016B754FEFF80446E0F1BC97CD57E266C1E538AAE8FF1394491`.
  The before/after snapshots resolve to the same frame-1 D3D12 resource and pass
  all Feature-18, activation, texture, trace and final-copy controls.
- The inline after snapshot differs from the before snapshot in 677,929 of
  1,843,200 bytes (`36.780002%`). Later E-116 proves this is an incomplete
  wavefront, not a conditional-store contract.
- The captured initial surface SHA-256 is
  `8DF6D450B5A7CB358B9E8373AF9FD9304E5912389C644F6C4BC66068380E88A3`.
  It is the hash of 1,843,200 zero bytes and is byte-identical to the existing
  integrated-plan surface slice at arena offset 27,807,744. Thus zero
  initialization was already correct.
- The exact-contract RTX package uses the complete activation, exact parameter
  block, initial surface, zero texture and after-slot154 oracle. It runs original,
  compat, E4M3 and movmatrix twice. Package SHA-256 is
  `7B069AC7EFA20B07D7B2A431154DA2F9908FE9618BC90A0F699D7C8E1E01D7DD`.
  E-116 supersedes the package's v24-after reference and its derived AMD metric.

## E-116 Exact-contract RTX replay exposes incomplete inline snapshots

- Date: 2026-09-04
- Artifact(s): `results/20260904_670000_postblock_exact_contract_rtx5070/`,
  `results/20260904_680000_postblock_surface_contract_analysis/`, and
  `deliverables/dlssnr-windows-feature18-deferred-frame1-output-v25-20260904_152127.zip`.
- Returned archive SHA-256 is
  `8D789F064BF8C53A80110744C1B0D952CE80BD2AB483B5C6B2EBB1CF10D953A2`.
  Original, compat, E4M3 and movmatrix each execute twice on RTX 5070, repeat
  bitwise, and all produce
  `8ADB4DE9E238DDA7A1155BD817E22236963A067C52C58A182AF50A09BAE6E8C6`.
- Relative to the exact zero initial surface, v23 changes 370,428 FP16
  components across 94,912 pixels. Every one of those components equals the
  standalone original PTX output bit-for-bit. v24 changes 389,845 components
  across 100,000 pixels, again with 100% equality wherever it changed.
- The spatial footprints differ: v23 is empty through row 199, then expands to
  full width by row 220; v24 begins at row 188 and is full width by row 212.
  Activation SHA-256 `925AE733...6CD33`, the 184-byte parameter block, weights,
  zero texture and zero initial surface are otherwise identical. This is strong
  evidence that the in-command-list readbacks observe perturbed/incomplete GPU
  progress rather than two legitimate full kernel outputs.
- The earlier AMD comparisons against v23/v24 are therefore withdrawn. v25 runs
  one evaluation with `MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS=1`, then copies and
  fences slot155 input/output after queue completion. Package SHA-256 is
  `E97A0D7A7FCDD3D0C0C4C7E5EED4002E92C15A370526EF9E45A1201C9B8AF373`.
  E-117 records the accepted queue-complete oracle; S7/S8 remain open.

## E-117 Queue-complete frame-1 oracle and exact-state post-block localization

- Date: 2026-09-04
- Artifact(s): `results/20260904_690000_deferred_frame1_output_rtx5070/`,
  `results/20260904_700000_queue_complete_post_output_reanalysis/`,
  `results/20260904_740000_postblock_exact_store_trace_rx9070xt/`, and
  `deliverables/postblock_mma_trace_reference_20260904_154238.zip`.
- Returned v25 archive SHA-256 is
  `7231729846D7117DF9185D64111625A912521443717CD59F498C101073C1BE2E`.
  It executes exactly one Feature-18 evaluation with inline snapshots disabled.
  After queue completion, slot155 input/output are bitwise identical and both
  hash to `8ADB4DE9E238DDA7A1155BD817E22236963A067C52C58A182AF50A09BAE6E8C6`.
  This is also the independent RTX original-PTX replay hash from E-116, making it
  the definitive complete frame-1 slot154 oracle.
- The exact-activation RX 9070 XT output hashes to
  `EA6CF1CB1CE94856F6746014622BC99E5054756882C36B5E240FA9F9C2C0EEBA`.
  Its RGB correlation is `0.9895909773328502`, NRMSE is
  `0.1442672662285069`, exact fraction is `0.4637847222`, and alpha is exact.
  It remains outside the strict gate but supersedes all earlier final metrics
  derived from cross-frame or incomplete inline images.
- The exact-state RX post-block trace has 245,760 active records and covers every
  coordinate of the padded 640x384 surface exactly once. Rounding its traced
  pre-store FP32 values to RGBA16F reconstructs the visible AMD output hash
  `EA6CF1CB...C0EEBA` exactly. The residual therefore precedes surface addressing,
  conversion and storage; the remaining candidates are FP8/F16 MMA scalarization.
- The next package captures the eight input words and two output words for all
  256 FP8 and 16 F16 MMA operations at CTA `(70,26,0)`, while requiring the
  instrumented RTX output to remain exactly `8ADB...E6E8C6`. Package SHA-256 is
  `136645005EA17896B7B6752CC4FAB68CA525F367F0401D5382F6CCC5AA319EDC`.
  This is diagnostic and does not count as S7; slot 6 remains a separate upstream
  correction target.

## E-118 Exact-state post-block MMA trace moves first divergence before FP8 MMA

- Date: 2026-09-04
- Artifact(s): `results/20260904_761000_postblock_exact_mma_trace_cross_vendor_corrected/`,
  `results/20260904_760000_postblock_exact_mma_trace_cross_vendor/fp8_model_analysis.json`,
  and `deliverables/postblock_e4m3_mov_trace_reference_20260904_171152.zip`.
- Returned archive SHA-256 is
  `8E28085FC22D2D033E85D9B69D126EA61E38C462F1A2CBA5B7E401706F5AC5AE`.
  Both RTX 5070 runs produce the same trace hash
  `933EB603A1CA6F3E50A2C20E7461CD96B3800B82D93A7F0AAFA876893768C594`
  and preserve the definitive output hash `8ADB...E6E8C6` exactly.
- MMA fragment inputs are warp-distributed, so equality of one lane's eight local
  words is insufficient to call its result an identical-input arithmetic test.
  The corrected comparison groups all 32 lanes. Of 256 FP8 MMAs, 16 have complete
  bitwise-identical warp inputs and all 16 produce bitwise-identical outputs.
  The other 240 have both input and output differences; there are zero cases of
  exact warp inputs with a divergent FP8 result.
- The first pre-FP8 difference is MMA 0 A-fragment register 0, byte 3, at lanes 9
  and 29: RTX has `0x1B`, RX has `0x1A`. F16 has no complete matching-input warp
  in this CTA and therefore only demonstrates propagated error, not a primary
  F16 lowering defect. CPU reconstruction from RTX operands also shows the current
  FP32 scalar MMA model rounds to every RTX FP16 result in the trace.
- The next package captures all 388 E4M3 conversions and 32 movmatrix operations
  at the same CTA. Its SHA-256 is
  `C48903733108762C39CD04D5C67F57BA79F9BA5D27816EA689A44775DBDFF9DE`.
  It distinguishes quantization input/output error from transpose/shuffle error.

## E-119 Exact-state E4M3/movmatrix trace moves divergence to packed-F16 input

- Date: 2026-09-04
- Artifact(s): `results/20260904_780000_postblock_exact_e4m3_mov_cross_vendor/`,
  `results/20260904_790000_postblock_f16x2_scalar_candidate/`, and
  `deliverables/postblock_f16x2_trace_reference_20260904_174308.zip`.
- Returned archive SHA-256 is
  `BC0F6D60D9A40083252F274CF2BF943AA9FCFA0DD0B78E6AD23109F51F9967B4`.
  Both RTX runs produce trace SHA-256
  `89F676EC25B5B4BD6CDDD879F6E9EBCAD54C8D66110047E3DC1737F41A1FD449`
  and retain the queue-complete output `8ADB...E6E8C6` exactly.
- For all 388 E4M3 conversions, there are zero equal-input/unequal-output
  records. Their inputs differ in 4,702 words and outputs in 792 words, proving
  that the output residual is fed into quantization rather than created by the
  validated integer lowering.
- Movmatrix is warp-distributed. Although 191 lane-local records have the same
  local word and a different output word, zero operations have an unequal output
  with all 32 input lanes equal. Therefore no valid matched-input movmatrix
  failure is observed.
- The first traced input mismatch is E4M3 operation 4, lane 1:
  `0xB32EBD09` on RTX versus `0xB32EBD08` on RX. The conversion output is equal,
  so this particular one-ULP half difference is absorbed by quantization. The
  source register is produced by packed-F16 add `%r944 = %r816 + %r817`.
- Replacing all 1,522 packed-F16 arithmetic operations with scalar RN-F32 then
  F16 packing executes successfully on RX 9070 XT but leaves its final output
  hash and metrics unchanged. This rejects blanket scalarization as a fix; it
  does not yet prove each native packed operation because their operands may
  already differ.
- The prepared RTX package captures inputs and outputs for all 1,522 operations
  at CTA `(70,26,0)`. It is
  `deliverables/postblock_f16x2_trace_reference_20260904_174308.zip`, SHA-256
  `5862355F9764C6E2224DEB1A2171611B214A00AD51B54439688AA0C4061E5F77`.
  The matching RX trace is deterministic and its instrumented output is unchanged.

## E-120 Full packed-F16 trace is repeatable but rejected for probe perturbation

- Date: 2026-09-04
- Artifact(s): source archive
  `_postblock_f16x2_trace_reference_result_20260904_181133.zip`,
  `results/20260904_820000_postblock_exact_f16x2_path_trace/`, and
  `deliverables/postblock_f16x2_path_trace_reference_20260904_181851.zip`.
- Returned archive SHA-256 is
  `C08B3467C5879F248A88DCE1384942278CC182AA3CB57E7C7FE0BBEFBEA9FCFB`.
  Both RTX 5070 executions load and run, the 974,080-byte traces repeat exactly,
  and package payload integrity passes.
- Both instrumented RTX outputs hash to
  `9A3D8A1D248DD15DEC1241C92485A14FEA781A6BD9B864A726915D689A13B834`,
  rather than the mandatory uninstrumented reference
  `8ADB4DE9E238DDA7A1155BD817E22236963A067C52C58A182AF50A09BAE6E8C6`.
  The result is therefore rejected as an original-path causal oracle.
- The rejected RTX trace hash is
  `205A235109D0046D6241166C77EC458B81E8AD48E4F8A97F15A8BC838766D79C`,
  byte-identical to the independently collected RX trace. This does not prove
  native parity; it shows that full instrumentation changes the NVIDIA result
  and makes the traced boundaries converge to the scalarized RX path.
- The low-perturbation replacement records only source-operation indices 0, 32
  and 64: `mul.f16x2 %r816`, `mul.f16x2 %r817`, and
  `add.f16x2 %r944`. Its RX execution passes and leaves the established AMD
  output unchanged. Package SHA-256 is
  `60E49965ECCFF48824D3D04FB255B0F2DFBBD0ABD045421FC83CC5A9F168FB55`.

## E-121 Producer-adjacent three-operation trace also perturbs RTX output

- Date: 2026-09-04
- Artifact(s): source archive
  `_postblock_f16x2_path_trace_reference_result_20260904_182824.zip`,
  `results/20260904_840000_postblock_exact_r944_snapshot/`, and
  `deliverables/postblock_r944_snapshot_reference_20260904_183411.zip`.
- Returned archive SHA-256 is
  `2E33A03B40B3DA5AF87C5A836D3E726E777C1B077EF17831466074ED0B80A3CA`.
  Both RTX runs execute and repeat exactly, but their output is
  `9EC0AACCE21C841BBC3F3F55AAB764901DCA53649180ACDCBCD3890D40442530`,
  so strict output preservation fails.
- The three-operation RTX and RX traces both hash to
  `FD1AABA00B49527E2CC02CAAC9374410E27AB48FE22BB8E452548DD7D17AF026`.
  This repeats the full-trace perturbation with only three sites and points to
  producer-adjacent instrumentation/JIT effects rather than trace-buffer volume.
- The replacement takes a single 1,024-byte snapshot at E4M3 conversion 4. It
  reads `%r204`, `%r205`, `%r720`, `%r721`, `%r816`, `%r817`, `%r944` and `%rs53`
  without inserting code around the packed-F16 producer instructions. The RX
  form executes and preserves its established output. Package SHA-256 is
  `77443FB214E88D084A3E114A028A6B510FF42D12FA6B28FC3009AECA5179A5AD`.

## E-122 Original-path model identifies the post-block fusion contract

- Date: 2026-09-04
- Artifact(s): source archive
  `_postblock_r944_snapshot_reference_result_20260904_183759.zip`,
  `results/20260904_780000_postblock_exact_e4m3_mov_cross_vendor/`,
  `results/20260904_880000_postblock_mul_add_fusion_first32_candidate/`, and
  `results/20260904_890000_full_graph_n0_cos01_postblock_fusion_first32_rx9070xt/`.
- The returned archive SHA-256 is
  `B77A5010D2A1AA7A85D95A2A46BFA4FCAA4D788D9F905405B8353C45ED60603A`.
  Both RTX 5070 runs execute and repeat, but their instrumented output is
  `D82309DE0FC794794994145DEA1DDC74531C09F6C3E8DA8598163B42DABEDE1D`,
  not the required `8ADB...E6E8C6`. The package is therefore rejected as a
  native-path causal oracle. Its RTX/RX trace equality is classified as probe
  perturbation, consistent with E-120 and E-121.
- Combining the four captured source operands with `%r944` from the accepted,
  output-preserving E4M3 trace permits an offline test of four rounding/fusion
  models. Separate operations, both-fused and second-rounded match 24/32,
  24/32 and 20/32 lanes respectively. Only first-product-rounded plus
  second-product-fused matches all 32/32 original RTX lanes.
- `scripts/lower_postblock_mul_add_fusion.py` implements that contract. Applying
  it to the first 32 of 64 structurally matching sites gives visible output hash
  `B98793AD...7815`, correlation `0.9991619063056614`, NRMSE
  `0.04094056022426491`, exact fraction `0.9611414930555555`, exact-or-adjacent
  fraction `0.9652965856481481`, zero nonfinite pairs and exact alpha. The
  predefined exact-input post-block numerical gate passes.
- Rewriting all 64 sites also passes but is worse (correlation `0.998763071`,
  NRMSE `0.049737473`), so the second 32 are not promoted. The first-32 variant
  completes all 156 graph launches twice and repeats bitwise. Its final image
  remains outside the gate because it receives already-diverged upstream state:
  fine checkpoints first fail at slot 6, while the integrated coarse checkpoint
  set first reports slot 15.

## E-123 Delayed slot-3 normalization snapshot is AMD-output-preserving

- Date: 2026-09-04
- Artifact(s): `results/20260904_900000_slot3_delayed_norm_snapshot/` and
  `deliverables/slot3_delayed_norm_snapshot_reference_20260904_185812.zip`.
- Same-capture injection evidence in E-102 shows that exact slot-5 state makes
  slots 6–8 pass, so diagnosis returns to the slot-3–5 1h chain. The earlier
  packed-F16 reduction trace is not reusable because it changed the RTX output.
- The new instrument records 31 registers spanning both packed-square partial
  groups, both shuffle reduction trees, completed sums, clamp inputs and packed
  rsqrt outputs. All stores occur at the later E4M3 conversion-224 boundary
  already shown to preserve the original RTX output in the prior path trace.
- On RX 9070 XT the full `41x25x1` launch succeeds and produces 3,968 bytes of
  nonempty trace. Output hash `C05D1463...94D581` equals the previous output-safe
  AMD path trace exactly, establishing that the new delayed snapshot does not
  alter the AMD result.
- Package SHA-256 is
  `9CC85E2D2F613940B032FFDB2E7CCA5FFE811305E33D88D25404DC2C715E3BFD`.
  The included runner performs two independent RTX executions and accepts only
  if both retain original output `33FE6004...50BD4F9` and their traces are
  bitwise identical. The package is an oracle request, not S7 evidence.

## E-124 Square-pair fusion closes isolated slot 2/3 and moves the full-graph frontier

- Date: 2026-09-04
- Artifact(s):
  `results/20260904_999100_full_graph_fine_slots2to6_all_square_compact_rx9070xt/`,
  `results/20260904_999300_slot3_all_square_fusion_outputsafe_path_trace/`,
  `results/20260904_999500_slot2_all_square_fuse_left_old_oracle_rx9070xt/`,
  `results/20260904_999400_full_graph_diag_exact_slot2_slots2to6_square_rx9070xt/`,
  `results/20260904_999600_full_graph_diag_exact_slot1_slots2to6_square_rx9070xt/`,
  and `results/20260904_999700_n0_normal1_full_grid_traces/`.
- The inferred packed-F16 contract is generalized to every structurally matched
  add of two square terms: retain the right square rounded to F16, then fuse the
  left square into the add with `fma.rn.f16x2`. Scoped candidates cover the
  slot-2 inpview, shared slot-3/4 chained, slot-5 downsample and slot-6 inpview
  functions. No RTX intermediate is injected in the primary run.
- The primary RX 9070 XT run executes all 156 slots twice and repeats bitwise.
  Same-capture NRMSE is slot 2 `0.000877122`, slot 3 `0.002420872`, slot 4
  `0.006707575`, slot 5 `0.008784803`, slot 6 `0.042475154`, slot 7
  `0.110049329`, slot 8 `0.149466126`, and slot 9 `0.049582652`. This moves the
  earliest strict failure from slot 6 to slot 7.
- On the original accepted isolated slot-2 payload, the new candidate's
  1,966,080-byte output has SHA-256 `29546A51...6E390E9`, exactly equal to the
  RTX 5070 output. For slot 3, the output-safe path experiment matches all 28
  registers x 32 lanes (896 words, zero mismatches), and RX output SHA-256
  `33FE6004...50BD4F9` is bitwise equal to RTX. Thus these are cross-vendor
  operator closures, not merely improvements in an aggregate metric.
- Exact slot1 or slot2 state is then injected only as a causal diagnostic and is
  explicitly excluded from S7. Both controls make slot 2/3/4/5/6 bitwise exact;
  slot 7/8/9 pass at NRMSE `0.009966183`/`0.031927418`/`0.028862428`, and the
  first failure moves to slot 11 (`0.126677794`). Therefore the current native
  slot-7 failure is amplification of the accepted N0 candidate's 154 differing
  E4M3 bytes, not an independent slot-2-through-slot-6 arithmetic defect.
- The next oracle covers all 245,760 first Box-Muller sine-path samples. Both RX
  trace variants preserve accepted output `61E14BDD...D764983`. Package
  `deliverables/n0_normal1_full_grid_traces_reference_20260904_193326.zip` has
  SHA-256 `4545FFB1F2DB418F7C15866E3C1CDE532BAF6461824E69B0A5E52BC27C9DE71A`.
  This is still numerical localization, not S7/S8 or game-integration evidence.

## E-125 Normal1 full-grid oracle reduces N0 and narrows slot-7 failure

- Date: 2026-09-04
- Artifact(s): `results/20260904_999800_n0_normal1_full_grid_cross_vendor/`,
  `results/20260904_999920_n0_sin0_nonzero_validation/`,
  `results/20260904_999980_n0_sqrt0_ulp_zero_validation/`,
  `results/20260904_999981_n0_sqrt0_ulp_nonzero_validation/`, and
  `results/20260904_999990_full_graph_fine_n0_sin0_sqrt0ulp_slots2to6_square_rx9070xt/`.
- Returned archive SHA-256 is
  `5E36BD452D163365662B5125D53072FA544308BEE57B906C2F97ECC3625779CB`.
  RTX 5070 identity, archive/payload integrity, both instrumented launches,
  repeatability and untouched-output controls pass.
- Over all 245,760 samples, phase is bitwise equal. Sine differs in 206,632,
  sqrt in 87,577 and consumed `normal1_f16` in 380. Category counts are 145 with
  both inputs different, 5 sqrt-only, 229 sine-only and 1 where rounded inputs
  match. Ideal RTX sine leaves 43 boundary mismatches; ideal RTX sqrt leaves 358.
- The accepted empirical correction uses a conservative 65,536-segment sine
  table followed by sparse, positive-sqrt bit-level ULP shifts. Complete N0
  output differs from RTX in 116/1,966,080 bytes on zero input and 109 bytes on
  the independent nonzero capture. Relative to the previous accepted candidate,
  both runs add zero new mismatches. The random-path trace hashes are identical
  under the two image inputs, confirming this Box-Muller dataset is input-
  invariant for the captured launch contract.
- The primary full-graph run executes 156 launches twice on RX 9070 XT, is
  bitwise deterministic and injects no RTX intermediate state. Slot 6 passes at
  NRMSE `0.039886535`; slot 7 improves from `0.110049329` to `0.103740020` but
  remains the first strict failure. S7/S8 and live-game integration remain open.

## E-126 Hierarchical normal2 correction makes slot 7 pass

- Date: 2026-09-05
- Artifact(s): `results/20260905_018000_n0_sin0_segment55169_nonzero_validation/`,
  `results/20260905_023000_n0_cos1_hierarchical_sin55169_sqrtulp_zero_validation/`,
  `results/20260905_024000_n0_cos1_hierarchical_sin55169_sqrtulp_nonzero_validation/`,
  and
  `results/20260905_025000_full_graph_fine_n0_cos1hier_sin55169_slots2to6_square_rx9070xt/`.
- A 524,288-segment normal1 search was reduced by GPU group, subgroup and
  individual-segment ablation. Only segment 55169 changes the zero-input N0
  result: it fixes eight old bytes and adds none. On the independent nonzero
  input it fixes four old bytes and likewise adds none.
- The accepted 32,768-segment `cos1` behavior was flattened into a single-add
  524,288-segment table, then only trace-domain normal2 overrides with zero
  observed boundary additions were retained. Combined with segment 55169 and
  the prior sqrt ULP correction, complete N0 differs from RTX in 83/1,966,080
  bytes on zero input and 86 bytes on the independent nonzero capture. Relative
  to the preceding accepted candidate it fixes 25 and 19 bytes respectively,
  with zero additions in both cases.
- The primary RX 9070 XT replay executes all 156 launches twice in one context,
  repeats bitwise and injects no RTX intermediate. Fine-boundary NRMSE is slot 6
  `0.034749463`, slot 7 `0.089691090` and slot 8 `0.123923003`. Slot 7 therefore
  passes the strict `0.1` gate for the first time, and slot 8 becomes the first
  failing boundary. Final RGB remains outside the image gate, so S7/S8 and live
  game integration are still open.

## E-127 Joint normal0/normal1 correction makes slots 8 and 10 pass

- Date: 2026-09-05
- Artifact(s): `results/20260905_028000_n0_cos01_hierarchical_sin55169_sqrtulp_zero_validation/`,
  `results/20260905_029000_n0_cos01_hierarchical_sin55169_sqrtulp_nonzero_validation/`,
  `results/20260905_040000_n0_cos0_sqrt0_joint_ulp_candidate/`,
  `results/20260905_041000_n0_jointulp_zero_validation/`,
  `results/20260905_042000_n0_jointulp_nonzero_validation/`, and
  `results/20260905_043000_full_graph_fine_n0_jointulp_slots2to6_square_rx9070xt/`.
- Extending the first cosine to the same hierarchical 524,288-segment policy
  reduces N0 to 64 bytes on zero input and 68 on nonzero input, fixing 19/18 old
  bytes with no additions. The then-first residual, output offset 323622, maps
  to CTA `(16,15)` and normal0 sample 38. Its cosine and sqrt are each one FP32
  ULP above RTX, while changing either alone is absorbed by FP16 rounding.
- A full-grid joint search tests small raw-cos0 and post-correction-sqrt0 ULP
  shifts per phase segment against both consumed normal0 and normal1 boundaries.
  Ten segments fix observed boundary differences with zero additions. On real RX
  9070 XT execution the promoted candidate reduces complete N0 to 29/1,966,080
  differing bytes on zero input and 46 on nonzero input. Relative to the prior
  64/68 candidate it fixes 35/22 bytes and adds none.
- The single-context, injection-free graph executes 156 launches twice and is
  bitwise deterministic. NRMSE is slot 6 `0.023386232`, slot 7 `0.061686839`,
  slot 8 `0.086983541`, slot 9 `0.036550767`, and slot 10 `0.096071659`; all
  pass. Slot 11 becomes the first strict failure at `0.134164531`. This closes
  the current N0-amplified frontier through slot 10, but final image parity and
  live game integration remain open.

## E-128 Exact 4h checkpoints localize the next failure to propagated state

- Date: 2026-09-05
- Artifact(s):
  `results/20260905_044000_full_graph_diag_exact_slot10_tensor_n0_jointulp_rx9070xt/`,
  `results/20260905_045000_slot10_square_pair_candidate/`,
  `results/20260905_046000_full_graph_fine_n0_jointulp_slot10_square_rx9070xt/`,
  and
  `results/20260905_047000_full_graph_diag_exact_slot9_tensor_n0_jointulp_rx9070xt/`.
- Injecting only the 491,520-byte native RTX slot-10 output is an explicitly
  non-S7 causal control. Slots 11-15 then pass at NRMSE
  `0.008314572`/`0.037572101`/`0.060818466`/`0.081675242`/`0.051178800`,
  and the first strict failure moves from slot 11 to slot 23 (`0.104075188`).
  Thus the slot-11 implementation is not the source of its native failure.
- A second non-S7 control injects only the exact RTX slot-9 tensor, immediately
  before slot 10. Slot 10 then passes at `0.012098444`, followed by slots 11-13
  at `0.034723807`/`0.083353930`/`0.095236783`; slot 14 becomes the first
  failure at `0.113606190`. Both 156-launch runs are bitwise deterministic.
  This shows that the 4h implementation is accurate from exact input but that
  its residual is amplified across repeated chained entries.
- Generalizing the accepted 1h square-pair fusion to all 32 structural sites in
  slot 10 is rejected. In the injection-free graph it slightly worsens slot 10
  from `0.096071659` to `0.096188659` and only marginally improves slot 11 from
  `0.134164531` to `0.133910018`; the first failing boundary remains slot 11.
  The accepted primary baseline therefore remains E-127, with no RTX state
  injection and the original slot-10 lowering.
- The next causal target is the slot-9-to-slot-10 state path: either reduce the
  sparse upstream differences before the 4h entry or identify an output-safe
  slot-10 internal rounding contract. Blind rewrites of slot 11 are ruled out.
  The first exact-input slot-10 difference maps to CTA `(4,0)`, warp-y `1`.
  A 272-MMA A/B/C/D tracer for that warp writes 348,160 bytes on RX 9070 XT
  while preserving the complete AMD output hash exactly. Package
  `deliverables/swin4h_slot10_fp8_mma_trace_reference_20260904_215525.zip`
  (SHA-256 `E6953608A3D8C003F44D591A6AF462A7442DB1AE15C7C6731908D32240ECB7A5`)
  runs an uninstrumented RTX control plus two traced runs and documents the
  exact PowerShell command.

## E-129 AMD-native output head reaches a real D3D12 texture

- Date: 2026-09-05
- Artifact(s):
  `results/20260905_010000_output_head_fp16_tail_rx9070xt/`,
  `results/20260905_011000_output_head_surface_store_rx9070xt/`,
  `results/20260905_012000_d3d12_residual_bridge_rx9070xt/`, and
  `results/20260905_013000_output_head_pipeline_rx9070xt/`.
- All sixteen FP16 tail MMAs execute on RX 9070 XT and preserve every observed
  accumulator-chain boundary. The four-group MMA result maps directly to four
  4x4 pixel tiles inside each 8x8 CTA; all 64 selected coordinates and store-site
  markers match the RTX trace.
- The reconstructed two-FMA store contract matches all 192 selected RTX RGB
  floats exactly. With native residuals, RGB mean/max absolute error at the
  final store is `0.000159/0.000764`, with deterministic output and no NaN/Inf.
- A separate D3D12/HIP integration gate copies a production-format RGBA16F
  texture into a shared linear heap, composes the actual native output-head
  residual, and copies the result back to a D3D12 texture after an imported-fence
  signal. Readback is bitwise exact against the host contract for all 921,600
  components and changes 668,907 RGB components. This proves the output can
  affect a game-shaped D3D12 resource; it does not prove live feature-18/game
  interception or a complete native network.

## E-130 Resident output head and in-process D3D12 interop pass

- Date: 2026-09-05
- Artifact(s):
  `results/20260905_014000_output_head_resident_rx9070xt/` and
  `results/20260905_015000_output_head_resident_d3d12_rx9070xt/`.
- Fourteen dependency-ordered kernels now execute the reconstructed output head
  in one process while all intermediate tensors remain in GPU memory. Its
  1,016,064 FP16 residual values match the accepted modular pipeline bitwise,
  repeat deterministically, and require no RTX trace during execution.
- In the D3D12-integrated run, the same process imports a shared D3D12 heap and
  fence, waits for a staged RGBA16F texture, computes the complete output head,
  signals D3D12, and copies the result into a D3D12 texture. Readback has zero
  mismatches across 921,600 components, no non-finite values, and 667,197 RGB
  components change from the zero base. This closes the resident output-head
  transport gate, not live feature-18 integration or the native upstream graph.

## E-131 NVAPI compatibility entry executes the resident output head

- Date: 2026-09-05
- Artifact(s): `results/20260905_016000_nvapi_amd_output_head_rx9070xt/`,
  `results/20260905_014119_nvapi_amd_selftest/`, and
  `results/20260905_014134_nvapi_amd_copy_selftest/`.
- The R610-compatible launch interface recognizes the real slot-154 function,
  validates the captured launch contract, translates five registered GPU
  addresses and executes real neural math on RX 9070 XT. Its linearized output
  is byte-exact to E-130. Negative-address, clear, descriptor and texture-copy
  regressions remain passing.
- The original surface-object ABI is not silently reclassified as a GPU address.
  Captured offset 16 is handle `0x9802`; resolving that handle to the D3D12 UAV
  resource and preserving command-queue ordering is explicitly still open.

## E-132 Real surface-object ABI and automatic live-resource registration pass

- Date: 2026-09-05
- Artifact(s): `results/20260905_020645_nvapi_amd_output_surface/`,
  `results/20260905_015320_nvapi_amd_selftest/`,
  `results/20260905_015335_nvapi_amd_copy_selftest/`, and
  `results/20260905_020557_module_trace_d3d12_selftest/`.
- The slot-154 launch retains the captured parameter ABI: offset 16 is an
  independent surface object, not a linear GPU address. `module_trace` observes
  the UAV creation, resolves its `ID3D12Resource`, and logs
  `backend_registered:true` before the AMD descriptor call.
- With `MODULE_TRACE_AMD_INTEROP=1`, buffers requested with the same non-shared
  flags used by the captured application are selectively recreated as shared
  resources only for the proven activation/model sizes. Both automatic HIP
  imports report `registered:true`; the four actual buffer pointers translate
  with zero failures.
- The resident output head writes an internal shared linear staging allocation;
  the original command list receives a placed-footprint copy into the real
  640x360 RGBA16F UAV. Readback has zero mismatches across 1,843,200 bytes and
  hash `A9CD87746E10C577`.
- This does not establish queue correctness in an unmodified game. HIP work is
  still executed while the D3D12 command list is being recorded, so earlier
  commands in that list may not yet have produced the inputs. The native
  slots 0-153 producer path is also not integrated; S7/S8 remain open.

## E-133 D3D12 submission-topology instrumentation passes locally

- Date: 2026-09-05
- Artifact: `results/20260905_022254_module_trace_d3d12_selftest/`.
- The tracker hooks queue and graphics-command-list creation, list `Close`, and
  `ExecuteCommandLists`. Its monotonic sequence joins those events to every
  observed feature-18 neural launch and retains command-list identity.
- The regression observes one hooked queue, two hooked lists, two closes and two
  submitted list items. JSONL is strict, the resource/descriptor hooks remain
  active, and all copy/N0/full-graph snapshot checks still pass.
- RTX package v26 packages the same tracer for one reference evaluation. This
  local result validates instrumentation safety only; it does not prove whether
  real slots 0-155 share one command list or establish an AMD synchronization
  design before the RTX result returns.

## E-134 RTX frame-1 topology requires same-list execution

- Date: 2026-09-05
- Artifact: `results/20260905_025006_rtx_feature18_submission_topology/`;
  returned archive SHA-256
  `D4A6A7F427FBB9D4BA79D95FEE1A8777B9E3B6B3C46024A24903CD7B4E3407AB`.
- The v26 package passes every integrity and runtime gate: Feature 18 creates
  and evaluates once, JSONL has zero invalid lines, and the completed slot154
  input and slot155 output are bitwise identical at 1,843,200 bytes with hash
  `8ADB4DE9...E6E8C6`.
- All 156 frame-1 launches (unique slots 0-155) use command list
  `0x1535d5a8d00`. Slot154 is sequence 159, slot155 is 160, that list closes at
  161 and is submitted on queue `0x153544446d0` at 162.
- There is no D3D12 submission boundary between the upstream producer and
  output head. A HIP launch performed synchronously inside the NVAPI wrapper
  cannot observe earlier commands that are still only recorded. An external
  fence at the current call point is therefore insufficient. The selected
  production architecture is same-command-list D3D12 compute (or an equivalent
  higher-level whole-list rewrite); S7 remains open until real slots 0-154 use
  that ordered path.

## E-135 First same-list D3D12 output-head stage is byte-exact on RX 9070 XT

- Date: 2026-09-05
- Artifact: `results/20260905_025647_output_head_surface_d3d12_rx9070xt/`.
- A native DXIL compute shader now performs the output head's spatial scatter,
  `clamp(base + 0.25 * residual, 0, 1)` composition, FP32-to-FP16 conversion,
  alpha write, and linear RGBA16F output entirely as commands recorded on one
  D3D12 graphics command list. It launches on adapter vendor `0x1002`.
- Across the full 640x360 RGBA16F surface, all 1,843,200 output bytes match the
  accepted resident output-head oracle; mismatch count is zero and SHA-256 is
  `3007949000DA625643310A1F1F3B24596765F7C8ABD239F21301DD6171BEFEC6`.
- Native HLSL `f32tof16` was initially one ULP low in 1,709 positive subnormal
  components. An explicit round-to-nearest-even conversion removes every
  mismatch. This establishes both the D3D12 execution mechanism and a required
  numerical-porting rule; the preceding learned stages remain to be migrated.

## E-136 Same-list D3D12 FP16 tail and surface pipeline passes on RX 9070 XT

- Date: 2026-09-05
- Artifact: `results/20260905_033200_output_head_tail_surface_d3d12_rx9070xt/`.
- Two native DXIL compute dispatches now run in one D3D12 graphics command
  list: the 32-to-4 FP16 projection writes the residual buffer, a D3D12
  transition makes it an SRV, and the surface shader scatters/composes it into
  the 640x360 RGBA16F output. This path has no HIP runtime, imported allocation,
  external semaphore, CPU wait, or disk boundary between the two stages.
- Disassembly of the RX 9070 XT HIP oracle identifies the actual tail lowering:
  `v_fma_mix_f32` through k14, `v_fma_mixlo_f16` at k15, four further mix-FMAs,
  then six `v_dot2_f32_f16` operations. The D3D12 tail differs in 2,286 of
  1,016,064 residual half values (`0.224985828%`); mean/max absolute error is
  `3.43386189e-8 / 0.00390625`, with no non-finite output.
- After surface conversion the difference contracts to 1,456 of 921,600 half
  components (`0.157986111%`), mean/max absolute error
  `1.41295863e-9 / 3.05175781e-5`. This passes the practical numerical gate;
  the large half-ULP count occurs only near zero and is not an image-scale
  error. Bitwise parity remains documented as a non-blocking arithmetic
  difference rather than misrepresented as exact.
- The next backward migration boundary is the final-projection producer of the
  32-channel FP16 tensor. Live game readiness remains false until all upstream
  producers are recorded into this same command list.

## E-137 Same-list D3D12 final projection is byte-exact on RX 9070 XT

- Date: 2026-09-05
- Artifacts: `results/20260905_034000_output_head_final_projection_d3d12_rx9070xt/`
  and `results/20260905_034100_output_head_final_tail_surface_d3d12_rx9070xt/`.
- The clean-room FP8 final projection now runs as native DXIL. It performs the
  exact half-to-E4M3 packing, permuted-channel weight addressing, E4M3 decode,
  residual seed rounding, and 32-term projection directly from the captured
  attention and first-128 tensors.
- In isolation, all 16,257,024 projected bytes match the accepted HIP oracle;
  byte and half mismatch counts are both zero. The same exact result is retained
  when final projection, FP16 tail, and surface composition are dispatched in
  order on one D3D12 command list.
- The combined path has no HIP runtime dependency and reproduces the previously
  accepted tail/surface bounds exactly: 2,286 residual half mismatches with
  mean/max error `3.43386189e-8 / 0.00390625`, followed by 1,456 surface half
  mismatches with mean/max error `1.41295863e-9 / 3.05175781e-5`. No non-finite
  value is produced.
- The backward migration boundary is now the attention producer (softmax/V and
  its Q/K/V inputs). This remains an operator-chain result, not a live game
  integration claim; `game_runtime_ready` is intentionally false.

## E-138 Same-list D3D12 attention-to-surface suffix passes on RX 9070 XT

- Date: 2026-09-05
- Artifacts: `results/20260905_035100_output_head_softmax_v_d3d12_rx9070xt/`
  and `results/20260905_035300_output_head_attention_suffix_d3d12_rx9070xt/`.
- A native DXIL softmax/V stage consumes the complete 32,514,048-byte QK tensor
  and 8,128,512-byte packed-V tensor. It preserves the observed per-32-key
  `Permute32` layout and E4M3 quantization boundary. Against the accepted HIP
  attention tensor it differs in only 96 of 8,128,512 half values
  (`0.00118102797%`), with mean/max error `6.14968557e-8 / 0.0234375` and no
  non-finite output.
- Softmax/V, final projection, FP16 tail, and surface composition then execute
  as four ordered dispatches on one D3D12 command list. Attention differences
  contract to 54 projected half differences. The final surface differs in
  1,461/921,600 half components, with mean/max absolute error
  `1.89543546e-9 / 0.000163555145`; no instability or command-list ordering
  regression appears.
- The current backward migration boundary is the Q/K/V preparation and QK-score
  producer. The suffix is ready for same-list integration, but upstream graph
  production and live slot interception remain required before a game claim.

## E-139 Complete output head executes on one D3D12 list with original weights

- Date: 2026-09-05
- Artifacts: `results/20260905_041000_output_head_full_d3d12_rx9070xt/`
  and deterministic repeat
  `results/20260905_041100_output_head_full_d3d12_rx9070xt_repeat/`.
- One native executable now consumes the captured activation arena and original
  model arena, records 11 DXIL compute dispatches on one D3D12 command list, and
  produces the 640x360 RGBA16F surface. It covers activation fusion, first-128
  hidden/projection work, MMAs 128-175, Q/K normalization, V preparation, QK,
  softmax/V, final projection, FP16 tail, and surface composition. It has no HIP
  runtime dependency and does not load an RTX trace during execution.
- The first128 and MMA128-175 full-grid checkpoints are byte-exact against the
  accepted HIP pipeline. QK and later stages use a practical numerical contract;
  the final surface differs in 32,218/921,600 half components but has mean/max
  absolute error only `3.4076811e-6 / 0.000862598419`, with zero non-finite
  values. Its SHA-256 is
  `8D91F1E7176A25C6B9DC76D7863BB2222B11E6415DC7C4ECA0E6F51624399C2C`.
- The repeat run produces identical checkpoint counts and final metrics. This
  closes the standalone complete-output-head D3D12 migration gate without
  per-stage RTX experiments. The next gate is live slot-154 D3D12 resource and
  command-list binding; `game_runtime_ready` remains false until that succeeds.

## E-140 Reusable output-head recorder with caller-owned textures

- Date: 2026-09-05
- Artifact: `results/20260905_042914_output_head_recording_selftest/manifest.json`.
- `output_head_d3d12_runtime.h` accepts caller-provided buffer slices, texture
  resources and command list, and records the complete 11-dispatch output head.
  It does not read captured files, map tensors, submit queues or wait. The test
  harness supplies captured activations through earlier commands on the same list.
- Eight invocations across two submissions prove original-output byte parity,
  changed main/base sensitivity, restoration and submission determinism. CPU
  composition matches all returned surface bytes; no residual/output NaN/Inf or
  D3D12 debug errors occur. Four invalid binding cases reject before recording.
- GPU timestamps observe 39.6-44.9 ms per head on the second submission with
  debug layer and diagnostic copies enabled. No performance gate is claimed.
- Coverage excludes live NVAPI routing, upstream native computation, additional
  real scenes, temporal history and dynamic resolution. The existing HIP slot154
  dispatch is unchanged. E-140 is a resource/scheduling component gate, not S7/S8.

## E-141 Real NVAPI launch entry selects D3D12 output-head recording

- Date: 2026-09-05
- Artifact: `results/20260905_044156_nvapi_output_head_d3d12/manifest.json`.
- The slot154 branch resolves actual parameter addresses and surface/base tokens
  against explicitly registered, COM-retained host resources, then records the
  complete DXIL executor on the original command list. Session ownership is
  retained until the registered completion fence passes.
- Eight output comparisons across two lists/sessions have zero byte mismatches
  against E-140's recorder. Input changes and restoration are exercised. Twelve
  invalid/unsupported calls, premature release and duplicate session operations
  reject. No D3D12 debug errors or GPU markers; zero remaining head sessions.
- No activation/model HIP imports are needed. The DLL retains its legacy HIP
  dependency/descriptor staging; this result proves a D3D12 head computation,
  not a HIP-free backend. Previous HIP self-test and 222 Python tests pass.
- Host state/fence bindings are explicit, not automatically collected from a
  game. The contract is the captured 640x360 model layout without history. Other
  kernels in a registered head-only session are rejected. This is not S7/S8,
  multi-frame scene generalization or a completed feature18 evaluation.

## E-142 Automatic head recording and queue retirement

- `results/20260905_050734_nvapi_output_head_d3d12/`: 1,000 real head records,
  250 submissions/sessions, zero byte mismatches/debug errors/active sessions.
  No activation/model HIP import or GPU marker substitutes. These are repeated
  A/B/C/A tensor controls, not a 1,000-frame scene or S9 pass.
- `results/20260905_051316_nvapi_output_head_d3d12/`: additional mixed kernel-chain
  and duplicate command-list batch checks pass without poisoning later valid
  submissions. Automatic binding trusts observed whole-resource legacy barriers;
  unknown/implicit/subresource states fail closed. One head list per submit only.
- The hook begins a deferred session at NVAPI launch, seals it with the actual
  submission queue, signals an owned fence after execution, and retires resources
  after completion. `ModuleTrace_CollectAmdHeads` allows explicit final draining.
  No live-game or general device-loss/unload support is implied.

## E-143 AMD upstream feeds the native output head; image gate fails

- `results/20260905_050400_upstream_head_export/` executes all 156 slots twice,
  exports state before slot154, and never injects RTX intermediates.
- `results/20260905_050700_hybrid_full_frame_validation/` feeds that actual AMD
  state to the DXIL head. Outputs repeat exactly but RGB correlation/NRMSE against
  canonical frame1 `8ADB4DE9...E6E8C6` are 0.683468/0.812474. `IMAGE_GATE_FAIL`.
- The old all-translated final output, rechecked against the same canonical
  reference, is 0.682751/0.810331, not the old stale-reference 0.435/1.111 claim.
- Reference RGB maximum is only 0.005913. Unit-range PSNR is not sufficient;
  normalized RGB and alpha checks are mandatory. A useful real-scene/full-sequence
  acceptance set remains absent from this experiment.

## E-144 Native chained 1h/32 family passes locally, regresses globally

- `results/20260905_052500_swin1h_native_family/`: shared FFN/QKV/attention/
  projection math plus native tiled gather/scatter runs as 10 DXIL dispatches.
  No PTX, HIP math or oracle file is consumed by the operator executable.
- Slots 3/4/151/152 use original checked weights, two window origins and encoder/
  decoder input distributions. Correlations are 0.999773/0.999844/0.999722/0.999667;
  NRMSE 0.021310/0.017686/0.023610/0.025798. Every repeat is exact and debug errors
  and nonfinite counts are zero. Intermediate bitwise equality is not required.
- `results/20260905_053100_native_swin_full_graph/` replaces those four actual
  upstream producers in two complete 156-slot executions, not with captured
  tensors. This diagnostic crosses APIs through CPU copies; it is not resident
  full-graph integration. Other operators still use ZLUDA.
- `results/20260905_053200_native_swin_head_full_frame/` then uses the native head.
  Correlation falls to 0.638833 and NRMSE rises to 0.854350. **Do not promote** the
  family variant as a quality improvement. Baseline remains unchanged by default.
- Shared-shader default output-head regression still passes in
  `results/20260905_052433_output_head_full_d3d12_rx9070xt/` with unchanged errors.

Integration-only evidence (2026-09-05):
`results/20260905_103732_540_ffx_dispatch/` contains actual AMD FFX `3.1.0` and
`4.1.1 *` version-override context creation, four reset upscale dispatches/fences
each, poisoned-output overwrite, independent A/A/B/A checks, and unchanged signed
game-library provenance. Both return zero debug errors/warnings. This validates
an isolated FFX data path, **not** an NR network, game scene or temporal history.

## E-145 Bounded texture capture preserves real FFX output in a controlled host

- Evidence: `results/20260905_104954_587_ffx_dispatch/`; command:
  `scripts/run_ffx_dispatch_probe.ps1 -CaptureValidation`.
- Two explicitly version-selected providers, `3.1.0` and `4.1.1 *`, each execute
  uninstrumented and captured four-frame A/A/B/A controls on AMD device 0x7550.
- Three captured frames contain color, depth, motion, exposure and output:
  15 resources / 8,294,412 raw bytes per provider. All bytes match independently
  known producer data or the normal output readback. Padded 1x1 exposure copies
  round-trip correctly. Every original output byte is unchanged by collection.
- A host-controlled queue wait deliberately holds execution; collector polling
  before and during this wait yields no data. Actual collector fence completion
  retires all jobs. Debug-layer errors and warnings are zero in both modes.
- Eight invalid/duplicate descriptor tests, cumulative byte budget and frame cap
  reject collection without stopping the fourth original rendering dispatch.
- Limits: explicit caller-owned single direct queue and verified live pointers/
  states; no automatic hooking, abandoned-list recovery or in-game bootstrap.
  All frames reset history and are synthetic. This is not a DLSS-NR image pass.
- Regression after collector work: 257 Python tests pass (45.72 seconds).

## E-146 Live game metadata and queue/resource contract inspection

- Reader sharing fixed/tested in `results/20260905_110638_418_ffx_bootstrap_test/`.
  PID 17484 exited normally; header log parsed. Replacement metadata snapshot
  `results/20260905_110900_gowr_metadata_snapshot/` has 9,997 decoded dispatches,
  all successful, but does not establish COM/state/pixel validity.
- Inspector controls `results/20260905_112352_927_ffx_dispatch/` preserve all
  output bytes and correlate all four calls for each real FSR3/FSR4 provider.
- Same running game PID 10968: `results/20260905_112424_948_gowr_live_inspection/`
  correlates 16/16 FFX calls to one direct queue. Whole-resource depth validation
  correctly fails: actual D32_FLOAT_S8X24_UINT versus depth-only FFX usage.
  Current dimensions differ from earlier metadata; no fixed-size contract claim.
- `results/20260905_112656_305_ffx_depth_plane/` separately proves both planes
  read back exact patterned clears at three sizes, zero debug errors, three
  intentional clear-value performance warnings. Game plane states are unproven.
- No game GPU copies or output replacement, no game-file deployment, unchanged
  audited image hashes. See `tools/ffx_observer/LIVE_INSPECTION.md`. This is an
  integration-contract milestone, not full native NR rendering or quality.

## E-147 Explicit depth-plane collector and game-size controls

- `results/20260905_113938_561_ffx_dispatch/depth_capture_analysis.json`: FSR3
  observed-size synthetic capture passes, 15 exact textures / 143,046,480 bytes,
  unchanged output/stencil, zero debug errors/warnings. Aggregate remains FAIL
  because FSR4 baseline repeatability fails before collection is enabled.
- Fresh contexts do not eliminate that FSR4 bottom-right edge difference:
  `results/20260905_114148_633_ffx_depth_fresh_context/`. Root cause unproven.
- `results/20260905_114847_910_ffx_dispatch/`: BOTH providers pass separate
  aligned-size depth capture on/off controls and legacy capture regression.
  Each large capture saves 139,345,920 exact bytes, no output/stencil changes.
- `results/20260905_115055_125_ffx_capture_plane_state/`: independent stencil
  DEPTH_WRITE versus depth shader-read states remain distinct through collector
  copies; exact bytes, zero debug errors. This host does not invoke FFX.
- Current game resample `results/20260905_114603_696_gowr_live_inspection/`
  confirms renderSize 1552x872, default upscaleSize [0,0], 16/16 queue matches.
  No game copies/file replacement/settings changes. Game context maximum and
  per-plane states remain unproven; native NR quality remains failed.
- Implementation/test details and reproduction: `tools/ffx_observer/DEPTH_CAPTURE.md`.

## E-148 Automatic capture sessions and bounded state diagnostics

- `results/20260905_121214_603_ffx_dispatch/`: independent FSR3/FSR4 automatic
  capture controls pass. Five textures / 2,764,804 exact bytes each; all original
  outputs preserved. This is not a game frame.
- `results/20260905_121239_377_gowr_capture_session/`: same live PID 10968,
  264 candidates, zero ready, zero copies. Epoch 4 context maximum now verified;
  declared versus local recorded state gaps remain. Provider ID unknown.
- `results/20260905_122303_633_ffx_state_trace_stage/controls/`: new separately
  staged state recorder passes both providers' observe-only controls (68 events
  each, outputs unchanged) and capture regression. Not loaded in the game.
- Menu observation confirms AMD FSR 3.1 / Quality, frame generation off. This
  names the UI option, not the selected backend. No settings were changed by
  the assistant in this follow-up. See `tools/ffx_observer/CAPTURE_SESSION.md`.

## E-149 Restarted game: actual bounded state-transition evidence

- Rebuilt exact binary controls pass in `results/20260905_122935_190_ffx_dispatch/`.
  Prior loaded session binary preserved; user exited normally before replacement.
- New PID 15544, bootstrap `results/20260905_122951_145_gowr_ffx_observation/`;
  session `results/20260905_123013_029_gowr_capture_session/` attached successfully.
  No game-file deployment; audited hashes unchanged.
- `boundary_analysis.json` has 34 complete menu-stage FFX call windows inside
  a truncated 1,024-event trace. All show depth transitions on other lists and
  post-call output StateBefore 64. No watched legacy resource transitions are
  observed inside those FFX calls. This is not proof of no GPU work, and is not
  complete cross-queue or game-scene state evidence.
- Continue loaded the existing save after this trace budget had been exhausted.
  New-process context remains unknown because creation preceded session attach.
  No capture arming, no game pixels or NR quality pass. Scope/next work and
  reproduction are in `tools/ffx_observer/CAPTURE_SESSION.md`.

## E-150 Deferred observation windows and two-list batch controls

- `results/20260905_124552_868_ffx_observe_window_stage/controls/`: exact FSR3/
  FSR4 baseline, observe-only, deferred-window and capture regressions all pass.
- Window control has two idle frames and two separately triggered frames;
  depth/motion producer and FSR use separate lists in the same two-list batch.
  Both providers preserve every independent baseline output byte, with zero
  D3D12 errors/warnings. 42 trace events/provider; zero observation-mode copies.
- Busy/restart/8-window/invalid-command controls pass. Observe/stop reject an
  armed capture. Window IDs prevent stale callback/list associations and batch
  IDs replace log adjacency as the correlation key. This is not complete GPU
  state tracking, nor a real game capture.
- 346 Python tests pass in 59.98 seconds. Bundle is staged, NOT loaded in current
  PID 15544. No current game restart, UI operation or game-file deployment.
  Reproduction and next attach: `tools/ffx_observer/OBSERVATION_WINDOWS.md`.

## E-151 Playable-scene retrigger, batch-prefix evidence and read-only entry audit

- User exited PID 15544 normally. New PID 23192 started 12:50:20 via
  `results/20260905_125020_962_gowr_ffx_observation/`; deferred staged session
  attach succeeded in `results/20260905_125046_331_gowr_capture_session/`.
  Exact loaded session SHA256 is
  `11683891D49A6332F2A01F9820C8A5FDC70576D7C0189E55640C6F66BCC67DA0`.
  Audited game files unchanged; no deployed/replaced game DLLs or game UI input.
- Two explicitly triggered playable-scene windows, stopped after sampling:
  128 candidates / 2,048 capped trace events / 50 complete FFX call windows /
  96 complete batches, 1 incomplete, 0 invalid. Zero ready/failure/copy events.
- `prefix_analysis_v2.json`: 48 unique same-batch call associations. Most last
  observed prefix states are 64 vs declared inputs 192/output 8. All 50 first
  observed post-call output transitions start at 64. First sample has missing
  reset coverage and depth chain gaps; no row is a complete GPU state proof.
- `live_entry_audit.json`: only QUERY_INFORMATION + VM_READ, 20,592 bytes,
  zero writes. Four FFX export entry prefixes equal disk bytes; two sampled
  command-list barrier slots still point to the correct current session hook.
  No recognized entry jump found. This rules out those narrow snapshot
  discrepancies, not internal redirection or alternate command/barrier paths.
- New offline analyzer tests cover batch-vs-CPU ordering, missing/reset/window/
  generation coverage, plane/split/alias uncertainty, chain gaps, duplicate and
  incomplete batches; entry decoder tests cover supported jumps and unknowns.
  Full regression: 366 passed in 63.33 seconds.
  Commands/details: `tools/ffx_observer/OBSERVATION_WINDOWS.md`.
- No arm was sent. Six observation windows remain. Current session context and
  selected backend remain unverified; no game pixels or native NR quality pass.

## E-152 Internal FSR4 route and real-game command-path observations

- Read-only `results/20260905_125046_331_gowr_capture_session/provider_metadata_audit.json`
  identifies the exact-image context forwarding route to AMD driver
  `amdxcffx64.dll + 0x199d40`. Private ID 17700776142811697153 / name `4.1.1 *`
  match the independent FSR4 provider control. This is not a public context
  query, lifetime proof or an explanation of why the FSR3.1 UI label differs.
- `results/20260905_131819_826_ffx_command_path_stage/controls/` passes exact
  FSR3/FSR4 output equality, capture regression, deferred windows and new
  positive compute/legacy/enhanced hook controls. Zero D3D12 errors/warnings;
  enhanced global barriers inserted outside FFX do not become inside-call counts.
  390 Python tests pass in 46.81 seconds. Counts allow wrapper double-entry and
  never certify unique GPU work or authorize collection.
- User exited PID 23192 normally; new PID 24336 bootstrap at 13:20:38 and staged
  attach succeed. Current evidence: `results/20260905_132114_421_gowr_capture_session/`.
  64 FFX calls each have 28 Dispatch / 16 legacy-barrier callbacks (26 barriers),
  zero enhanced/indirect/other-interface calls and all-zero original statuses.
  25 complete state call windows and 48 complete batches fit the capped trace.
  No watched texture transitions inside FFX; post-call output begins at 64.
- Private FSR4 route/metadata reproduced in the new process. State/resource
  identity and output-boundary coverage are the next diagnostic target, not
  another assumption of no GPU work. Session context still predates attach.
  Window 1 stopped, seven remain; no arm, zero ready/failures/game copies,
  unchanged game files. Full NR quality and runtime readiness remain false.
- Reproduction, exact hashes, limitations: `tools/ffx_observer/COMMAND_PATH.md`.

## E-153 Game output UAV resource identity reproduced without capture

- New independently validated resource-identity stage:
  `results/20260905_133303_873_ffx_resource_identity_stage/`. Session SHA256
  `325123ED040C3E7FF35E485696A1813599DF6CC394611241100A4DB6D41DA5E0`.
  All eight FSR3/FSR4 native controls pass, byte-identical on/off outputs and
  zero D3D12 errors/warnings. Five control reports pass; 403 Python tests pass.
- User exited PID 24336 normally. New PID 30652 starts 13:38:00, deferred
  session at `results/20260905_133833_941_gowr_capture_session/`. Two stopped
  playable-scene windows contain 128 paired calls, all original statuses zero.
- `resource_path_v2.json`: eight complete detailed calls, 224 uncapped detail
  events. Each call has 18 UAV / eight transition barriers, two UAV barriers
  matching exact game output raw pointer AND canonical IUnknown identity, zero
  identity-only matches, and unchanged descriptor fields. No watched transition
  matches. This corrects the earlier broad zero-matching-barriers statement;
  old diagnostics only detailed watched state transitions, not UAV identities.
- `state_v2.json`: 2,048 capped events, 50 complete calls, 98 complete batches,
  zero incomplete/invalid batches. First post-call output transition starts at
  64 in all 50 calls. `prefix_v2.json`: 49 partial same-batch associations.
  Neither is full state/lifetime proof; session context creation is unobserved.
- Private FSR4 4.1.1 * route reproduced by read-only audit, 20,901 bytes read.
  Game files and loaded stage hash unchanged. Computer-use only selected,
  observed and activated the window, with no settings/save/movement/combat input.
  Six windows remain; no arm, zero ready/failure/copy events. Full NR quality,
  real-game frame capture and runtime readiness remain false.
- Details and running commands: `tools/ffx_observer/RESOURCE_IDENTITY.md`.

## E-154 Post-FFX output-copy primitive and independent provider controls

- `results/20260905_135025_947_ffx_output_boundary_controls/`: four synthetic
  output-copy cases at 640x360 and 2342x1317, both source states 8/64 -> 192.
  53,037,024 bytes equal the CPU reference and downstream readback exactly;
  32 negative checks, eight early-poll checks, zero D3D12 errors/warnings.
- Separate real FSR3/FSR4 baseline and boundary-copy runs pass all four frames
  at 640x360. Saved boundary data and downstream output equal independent
  baseline byte-for-byte. `controls/analysis.json` passes without an NR claim.
- Existing PID 30652 evidence yields seven structural output-boundary matches
  among eight detailed calls. One is excluded for missing reset coverage.
  No new live observation/attach/copy was performed; six windows remain.
- Current trace records before ResourceBarrier forwarding and omits callback
  batch identity. Therefore it does not prove the final state after the whole
  callback. Live hook integration, queue/lifecycle handling and its independent
  adversarial controls remain required. The primitive is isolated-only, with
  fail-stop protection against releasing unretired work; not drop-in game code.
- 426 Python tests pass in 63.71 seconds. Existing capture gates, private-context
  limitations, actual-size FSR4 repeatability failure and full NR quality failure
  are not waived. No input/temporal dataset or real-game NR output was produced.
- Exact hashes, boundaries, remaining requirements and run commands:
  `tools/ffx_observer/OUTPUT_BOUNDARY.md`.

## E-155 Whole-callback return diagnostics and isolated hook-copy execution

- `results/20260905_140838_246_ffx_boundary_hook_stage/` passes 12 native runs
  and six analysis reports. Ten normal runs: zero D3D12 errors/warnings. Two
  negative runs: exactly two checked inefficiency warnings each, zero errors;
  multiple/split output transitions correctly rejected, outputs unchanged.
- Session SHA256 `5EB4374459655A00198E6380112A47C40BDD56E498BFDAAD6FCE6E1A47FE982F`.
  Output-return event is emitted after original whole-callback forwarding, with
  a consumed pending target preventing double selection under reentry. Target
  generation/reset/open state and one FFX call per generation are checked.
- Isolated version 3 really records a copy from inside the hook. Each provider
  writes one 1,843,200-byte output equal to baseline; all downstream outputs
  remain equal. A blocked GPU queue prevents early readback. Exact event order
  and recorded/closed/submitted/fenced artifact checks pass independently.
- Live Attach explicitly rejects version 3. No game output-copy mode was added,
  no live DLL replaced, no game files deployed. PID 30652 still has the older
  stopped session (6/8 windows remain), pending the requested normal user exit.
- 443 Python tests pass in 64.26 seconds. Live cross-queue/lifecycle adversarial
  coverage, game frame capture, NR quality, actual-size FSR4 repeatability and
  runtime readiness are not claimed. Next attach is diagnostic-only.
- Reproduction commands, exact stage and safety limits: `tools/ffx_observer/BOUNDARY_CALLBACK.md`.

## E-156 Real-game callback convergence and bounded live-output controls

- GoWR PID 31044 loaded the validated E-155 diagnostic stage through the fresh
  bootstrap/inspector/contract/session chain. Two playable-scene observation
  windows produced eight post-forward callbacks: seven supported, one startup
  sample rejected only because reset was not observed. Window two is 4/4.
- Every callback has one transition of the exact output resource, no alias or
  unknown barriers, state 64 -> 192 over all subresources, unchanged FFX resource
  fields, successful original return and exactly one forward call. There were
  no failures or copies. Session: `results/20260905_142028_834_gowr_capture_session/`.
- `results/20260905_142713_314_ffx_live_output_stage/` adds separate commands 5/6
  (`output-arm`/`output-poll`). These cannot arm the multi-input collector and
  accept only one already-proven boundary before stopping observation. The copy
  remains on the original list/generation, is associated by COM identity with
  exactly one submission, and is readable only after its collector-owned fence.
- Both FSR3 and FSR4 controls exercise the same version-2 command path used by
  live attach. Output equals independent baseline byte-for-byte, downstream
  output remains unchanged, an intentionally blocked queue returns incomplete,
  and exact lifecycle/artifact metadata checks pass. Isolated records explicitly
  retain `game_frame_capture_verified:false`.
- Six native analysis reports pass and 455 Python tests pass in 62.73 seconds.
  A new live artifact analyzer rejects bad ordering/provenance, truncation,
  non-finite half data, blank frames, format/metadata mismatch and changed game
  files. It cannot turn an FSR frame into NR evidence.
- The user reported exit, but PID 31044 remains responsive and owns a game
  window. Therefore the new stage was not stacked or hot-replaced. A real-game
  frame, NR execution, teacher data, output replacement and runtime readiness
  remain unverified until a genuinely fresh process is available.

## E-157 First live copy fail-stop and canonical-list correction

- Fresh GoWR PID 30212 loaded the `142713` stage through exact bootstrap,
  observer, contract and deferred-session provenance. Command `output-arm` was
  accepted; no multi-input arm or UI input was sent.
- Sample 1 was structurally supported but correctly rejected because list reset
  was not observed. Sample 2 passed the whole callback and lifecycle gate; the
  copy was recorded after the original 64 -> 192 transition and restored 192.
- Queue submission then surfaced a different graphics-list interface pointer
  than the boundary callback. The pre-identity raw-map close check failed, so
  the session emitted `failure`, stopped, retained the uncertain job and wrote
  no artifact. The game remained responsive. This is a fail-stop result, not a
  capture or NR pass. Evidence:
  `results/20260905_143334_141_gowr_capture_session/session.jsonl`.
- `results/20260905_143600_233_ffx_canonical_output_stage/` fixes ordering of
  proof: exactly one submitted canonical IUnknown identity must match; the
  original recording entry must retain reset/generation; and at least one
  same-identity/same-generation alias must have observed Close. Pointer equality
  is preserved as a diagnostic boolean rather than an authorization gate.
- FSR3/FSR4 baseline, observation, window, capture, negative-boundary and live-
  command controls all pass, including fence blocking and byte equality. Six
  reports pass; full regression 455 passed in 61.66 seconds. This revision is
  not loaded into PID 30212. A normal exit/fresh process is required.

## E-158 Canonical submission passes live; launch frame is not scene data

- After normal exit, fresh GoWR PID 31864 loaded the canonical stage through
  bootstrap, live-inspector, contract-inspector and deferred-session checks.
  `output-arm` and one `output-poll` both returned 1; no input capture, output
  replacement, setting change, game input or game-file deployment occurred.
- The successful boundary is 64 -> 192, whole resource, no aliases/unknowns.
  Recording temporarily transitions 192 -> COPY_SOURCE -> 192. Submission is
  accepted by one canonical COM identity and the collector-owned fence completes.
- `results/20260905_143913_151_gowr_capture_session/output_boundary/boundary_output.raw`
  is 24,675,312 bytes, exactly 2342*1317*8, RGBA16F, finite, SHA256
  `FC17DAE7604FC5286AD2802381F01CC7461C561CA19CC21E2D4F824A73E9E5CA`.
  The exact game image hashes remain equal to the static audit.
- The sample was captured immediately during launch. Every sampled channel has
  only eight half values in 10.7578125..10.8125 and ~0.006 standard deviation.
  Revised report `game_output_analysis_v2.json` therefore separates transport
  proof (`game_frame_capture_verified:true`) from usable content
  (`scene_content_verified:false`, `dataset_usable:false`). It is not accepted
  as gameplay data, an NR teacher or visual-quality evidence.
- The analyzer now reports per-channel variability and is covered by the full
  456-test regression (61.26 seconds). A fresh process and user-confirmed
  playable scene are required for the next one-shot capture.

## E-159 Usable playable-scene FFX output captured from GoWR

- After a fresh bootstrap, PID 2920 loaded the exact validated canonical-output
  stage and stayed idle until the user explicitly confirmed a playable scene.
  No game UI input was generated by the tooling.
- The one-shot command records sample 2 after rejecting the expected unwitnessed-
  reset warm-up sample. Whole-callback conditions pass; submission reports one
  canonical identity, one alias, one closed same-generation alias and one raw
  pointer match. Fence completion is explicit; no failure event exists.
- `results/20260905_144814_826_gowr_capture_session/output_boundary/boundary_output.raw`
  is 2342x1317 RGBA16F, 24,675,312 finite bytes, SHA256
  `79DBAF343981F9695365AD690D16E3B1309E76D6F31211C9EC1F6BD20DBD8603`.
  All four audited game-file hashes still match.
- Per-channel sampled unique values are 7,165 / 7,391 / 7,161 / 7,165; RGB
  maxima are 5.78125 / 4.66015625 / 3.357421875. The analyzer marks
  `scene_content_verified:true` and `dataset_usable:true`.
- `game_output_preview.png` (SHA256
  `C08B8C98166B5C3CCAF1A089DB2C2432444E053EDCCB208DB142A667BAE9DA94`)
  is a deterministic Reinhard-plus-gamma preview and visibly shows the intended
  playable animal-pen scene. It is a view transform, not comparison evidence.
- This proves transport of the existing game FFX output only. No output was
  replaced; it is not an original DLSS-NR result or teacher (`false` in the
  signed analysis report). Input/temporal acquisition and safe reconstructed-
  network execution/write-back remain open.

## E-160 Exact output roundtrip passes FSR3/FSR4 controls

- `output_boundary_roundtrip.h` records an exact no-op GPU route on one direct
  command list: proven readable output -> copy source, output to private texture
  and readback, output -> copy destination, private texture back to output, then
  restore the original readable state. It accepts only the already-proven whole
  RGBA16F boundary and same-device queue identity.
- Direct primitive tests cover 640x360 and the observed 2342x1317 game size from
  both accepted source states. Eight cases pass with 106,074,048 exact bytes,
  64 negative checks, 16 blocked early polls and zero D3D12 errors/warnings.
- The real FSR3 and FSR4 isolated session hooks each produce one complete
  arm/return/record/submit/fence chain. Both write-back outputs are byte-identical
  to independent provider baselines, and both copy-only controls remain exact.
- Fresh live stage:
  `results/20260905_150435_007_ffx_output_roundtrip_live_stage/`. Session DLL
  SHA256 `793826DE8C0F0428E1A6EE09847858ACCCB8303A1F381E2FA4310E81EF99B50C`;
  attach client `BE97E7CCFC42198699C1E4749E31025A68ACD584D1C51FF4474ADE0F95750485`;
  command client `48020D8948E11FD1BD95C78169020662E9399A09F10B30DAC0377C242919DC23`;
  probe `5B464E203496D52D7A204CBB17609731B26A5547E6FF49C75DA4FA8608C9D21C`.
  Its provenance requires the roundtrip control at live attach. Full regression:
  471 passed in 55.53 seconds.
- `analyze_ffx_game_roundtrip.py` independently rejects missing/duplicate or
  reordered lifecycle events, unproven canonical submission, absent write-back,
  claimed replacement/NR, incomplete fences, malformed/non-finite/constant data
  and changed game files. A pass means the existing game output survived the
  write-back route; it still does not mean reconstructed pixels were generated.

## E-161 Exact output roundtrip passes in a playable GoWR scene

- Fresh PID 23500 loaded the immutable `150435` stage through bootstrap, live
  inspection, contract inspection and deferred-session provenance. The user
  confirmed a playable scene before `roundtrip-arm`; arm and poll each returned
  1. No input collector or game UI automation was used.
- Session `results/20260905_151638_018_gowr_capture_session/` contains exactly one
  accepted lifecycle and no failure. The canonical command-list identity,
  generation, Close and exact-once queue submission checks pass; the owned fence
  completes.
- `output_roundtrip/boundary_output.raw` is 2342x1317 RGBA16F, 24,675,312 finite
  bytes, SHA256
  `2AD69C0E4F8DC93C11E4A0E75FBE110FE320A64B37DF80F3575B20D8C5E632E7`.
  Sampled RGB has 7,314 / 7,500 / 7,222 unique half values and maxima
  7.1484375 / 6.22265625 / 5.01953125. Scene-content and dataset checks pass.
- `game_roundtrip_analysis.json` marks game frame, write-back and fence true,
  while `replacement_pixels_supplied`, output replacement, DLSS-NR and teacher
  remain false. All audited game binaries still match their original hashes.

## E-162 Supplied-pixel replacement passes isolated FSR3/FSR4 hooks

- `output_boundary_patch.h` records two full GPU readbacks around one 8x8
  RGBA16F upload patch at the same post-forward output boundary. It restores the
  readable state and uses the same canonical queue association and owned fence.
- FSR3 and FSR4 each accept one patch lifecycle after a deliberately blocked
  early poll. Before bytes equal the independent provider baseline; after bytes
  equal the downstream output. Exactly 64 target pixels change and every pixel
  outside the 8x8 rectangle remains byte-identical.
- Immutable live stage:
  `results/20260905_152548_455_ffx_supplied_patch_live_stage/`. Session DLL
  SHA256 `0E79E8988252FB07D815098B07156336EDB9E2A3D5F491A40E77C2C6BEECA2BE`;
  attach client `6FD299CB6078ACC007F51A1D151B5C3A30D38AF4237F88BE8C3F9AA310673B2A`;
  command client `9E5A5C87E4000D97CB089EE8F532951C42FF22849F173AFA16D446144EBA8D1F`;
  probe `ED54214DA90943786847BA3117AB86DC43A4C8A785754D42E8651192713F491C`.
  Stage provenance requires copy, roundtrip and supplied-patch controls. Full
  regression is 478 passed in 55.73 seconds.
- This is replacement-transport evidence, not reconstructed-network evidence.
  The patch is intentionally one frame and 8x8 to bound visual impact. A fresh
  game process is required before live validation; it is not loaded in PID 23500.

## E-163 Supplied-pixel replacement passes in a playable GoWR scene

- Fresh PID 29836 loaded the exact `152548` stage through the complete bootstrap,
  inspector, contract and deferred-session chain. The user confirmed gameplay
  before one `patch-arm`; arm and fenced poll each returned 1. No UI input or
  game-file deployment was performed by the tool.
- Session `results/20260905_152923_020_gowr_capture_session/` contains one
  accepted whole-resource boundary and one canonical-identity submission. The
  before and after files are each 24,675,312 bytes at 2342x1317 RGBA16F.
- Before SHA256 is
  `528CF76DB1674B2375347AF36DC7D0A2BF793217EEE356942B8B8671EE6FEE43`;
  after SHA256 is
  `581E68782A1444E9CE30E5178DD76C38DF13065F6D241EE4046DECC2EB9EEAC6`.
  Exactly 64 pixels / 256 components change to the supplied value. Every pixel
  outside `[0,0,8,8]` is byte-identical.
- `game_patch_analysis.json` reports `GAME_OUTPUT_PATCH_PASS`, game write-back,
  output replacement and supplied pixels true. It explicitly reports
  `reconstructed_network_output:false` and `dlss_nr_verified:false`. This closes
  the final texture-injection transport gate, not the neural model itself.

## E-164 Dynamic filter queue insertion failure and fail-safe redesign

- The first two `session-output-filter` development attempts never produced an
  accepted artifact. Windows recorded unexpected restart events (Kernel-Power
  41) at 15:48:16 and 15:51:33 after the machine became unresponsive.
- The second session log preserves the exact software failure:
  `output boundary return diagnostic: resource deadlock would occur`. The
  private filter constructor called the globally hooked command-list Close while
  the ResourceBarrier callback held the non-recursive session mutex.
- The filter path was stopped and not retried in-game. The correction moves all
  private object construction and recording outside the session lock and wraps
  observer-owned Reset/Close/Barrier/Dispatch calls in an internal hook-bypass
  scope. The filter fixture no longer uses an unsignaled GPU queue Wait.
- This is recorded as a system-level failure, not hidden as an incomplete test.
  Development artifacts:
  `results/20260905_154518_233_ffx_filter_queue_dev/` and
  `results/20260905_154900_filter_diag/`.

## E-165 Corrected dynamic filter passes software and hardware controls

- WARP-only primitive split test:
  `results/20260905_160500_filter_warp_build/warp_result/`. It executes one
  producer list, one inserted private compute list and two downstream lists;
  downstream bytes equal the filtered output, alpha is exact and D3D12 reports
  zero errors/warnings. Hardware GPU use and queue-wait gating are both false.
- WARP full-session interception test:
  `results/20260905_161800_filter_session_warp_build/warp_session_result/`.
  It exercises the actual global hooks, internal bypass, canonical identity,
  batch split and fenced poll. Status is `WARP_SESSION_FILTER_PASS` with batch
  count 3, boundary index 0 and two suffix lists.
- One-frame 640x360 RX 9070 XT tests then passed independently for FSR3 3.1.0
  and FSR4 4.1.1 *, with no queue Wait gate and zero D3D12 messages. The
  complete immutable stage is
  `results/20260905_164000_ffx_dynamic_filter_live_stage/`; all 18 provider/mode
  runs and every control analyzer pass. Full Python regression: 485 passed in
  55.71 seconds.
- Stage SHA256: session DLL
  `644537F209FE9700E4D0FA2AE6C91FFBF4C891F22BE121E9B7FDD1CC93427271`,
  attach client
  `D21B75D0A6DD86E0BC790566B2DCCBC51C094B4307B0B9BDB73077CB5FEB1673`,
  command client
  `ED4BC6E1BDEF365D10D9EE1560BD384F8BBB007E3BEB4F2FC9B6EAFEF2465D07`,
  probe `5BFB80E9F64C74515C8511309617F62D4CCFFFA8DA5A6857D4B7F6BE7DF01610`,
  shader `BE951A5B24C3263EB31D5229CD2EADE0868A6C8FDC5A973F61EF877F537C0A36`.

## E-166 Dynamic-resolution compute replacement passes in GoWR

- Fresh GoWR PID 9372 loaded the exact E-165 stage through bootstrap, live
  inspector, contract inspector and deferred-session provenance. The user
  confirmed a playable scene before one `filter-arm`; arm and fenced poll each
  returned 1. No game file was deployed or changed.
- Session `results/20260905_160743_488_gowr_capture_session/` records one
  accepted output boundary and one canonical submission. The real game batch
  contains 36 original lists; the boundary is index 33 and the two original
  downstream consumer lists remain after the inserted private compute list.
- Before and after are 2342x1317 RGBA16F (24,675,312 bytes each), SHA256
  `C2C59E56E3F9DE4488D827685D058564E66F89944948520EB8D8A085C0B8858C`
  and `0BB5E0F6C55A76CFF6BB0A0811815C8210828A233DC495A66AF95170ABCA9B10`.
  3,060,032 pixels and 8,727,182 components change; alpha is bitwise exact,
  mean/max absolute RGB change is 0.00174455 / 0.47265625, and output is finite.
- `output_filter_analysis.json` reports `GAME_GPU_RESIDUAL_FILTER_PASS`; the
  game remains responsive and the recent System log contains no display-reset
  event. This proves dynamic GPU-generated replacement reaches the real game.
- The shader is a hand-weighted five-tap residual filter. Therefore
  `trained_weights`, `reconstructed_network_output` and `dlss_nr_verified` are
  deliberately false. E-166 is the integration transport milestone, not the
  final DLSS-5/Ray-Reconstruction quality claim.

## E-167 Externally weighted residual CNN passes all pre-live gates

- `output_residual_network.hlsl` implements a dynamic 3x3 RGB -> 8 ReLU
  feature -> RGB residual network. `output_boundary_network.h` uploads an
  independently parsed 256-float model constant buffer and records the network
  on the same private-list insertion path. The sparse calibration model is
  `models/ffx_dynamic_residual_v1.weights`.
- WARP kernel and full global-hook session tests pass with zero D3D12
  errors/warnings, internal-hook bypass, one inserted list and two preserved
  suffix lists. The legacy filter session also passes in the same stage.
- RX 9070 XT kernel tests pass at 640x360 and 2342x1317. Both output files are
  byte-exact with the earlier five-tap filter outputs, proving that external
  model parsing and constant-buffer inference preserve the accepted calibration
  behavior.
- Immutable stage:
  `results/20260905_180000_weighted_network_live_stage/`. FSR3 and FSR4 each
  pass all ten modes. Their network output is byte-exact with their filter
  control output and visible to the downstream suffix. Full Python regression:
  490 passed in 61.40 seconds.
- SHA256: session DLL
  `30C158DD95705B67969484BC22CF2A9F6225A768ED6D0015EF0209F989D8F716`,
  network DXIL
  `A7D94FF84756B3310FEDB7B872AFFABA91282346EB7EA5C09BF9C12F90978244`,
  calibration weights
  `723691E51B8AA5A5B8CAC1D37676A80C5E3BB0D9B67E25F38E1900A73B5717CF`.

## E-168 Externally weighted CNN passes in a playable GoWR scene

- Fresh PID 21584 loaded the exact E-167 stage through bootstrap, live
  inspector, contract inspector and deferred session. The user confirmed a
  playable scene before one `network-arm`; arm and fenced poll returned 1.
- Session `results/20260905_163007_931_gowr_capture_session/` records one
  accepted live output boundary. The original batch has 36 lists, the selected
  boundary is index 34, and the final original consumer remains after the
  inserted private network list.
- Before and after are 2342x1317 RGBA16F, 24,675,312 bytes each. SHA256 changes
  from `AA8E0F16E765564D5EBB2489653E0EDB7CA8C68379BEC1D308E3D631B059ED86`
  to `E834265FAD80C10C651A1703B6048B55787EEE05F5F1D4E14656C29F03DEF29F`.
  3,058,969 pixels / 8,730,452 components change, alpha is bitwise exact,
  mean/max absolute RGB change is 0.00178033 / 0.49609375, and output is finite.
- `output_network_analysis.json` reports `GAME_GPU_WEIGHTED_NETWORK_PASS` with
  game capture, write-back, replacement, external weights and reconstructed
  network output true. The game remains responsive and the recent System log
  contains no GPU reset or unexpected restart.
- The current weights are calibration weights, not learned teacher weights.
  `trained_weights=false` and `dlss_nr_verified=false` remain required. This
  closes the weighted-network integration gate, not the final image-quality,
  temporal-stability or DLSS-5 equivalence gates.

## E-169 Arbitrary-resolution full-graph boundary passes WARP

- `tools/full_graph_d3d12/resolution_plan.h` defines a dynamic eight-pixel-
  aligned work geometry and an overlapping compatibility-tile fallback around
  the captured 640x384 graph. Midpoint crop ownership guarantees exactly one
  writer for every final output pixel.
- The compatibility route retains at most 16 work/output tiles. Larger frames
  advance through batches using an explicit descriptor base, so scratch memory
  is bounded instead of scaling with all tiles in the frame.
- CPU artifact `results/20260905_190000_arbitrary_resolution_plan/summary.json`
  passes 524 deterministic and random plans (26,672 tiles) through 7680x4320.
  The live 2342x1317 frame maps to a 2344x1344 native work shape or 16
  compatibility tiles.
- WARP artifact `gpu_summary.json` in the same directory executes the real
  D3D12 pack and unpack shaders at 320x180, 641x361, 1920x1080, 2342x1317 and
  3840x2160. All RGBA16F components return bitwise exact; the 4K case uses 49
  tiles in four batches. Debug-layer errors and warnings are both zero.
- The middle shader is an identity fixture. This proves arbitrary-resolution
  transport and assembly, not the unported slots 1-154, image quality or DLSS
  Ray Reconstruction.

## E-170 Original-weight translated graph responds to real game color on AMD

- Date: 2026-09-05; S7/S8 remain open. Verdict: PASS for offline color response;
  UNVERIFIED for DLSS5 quality, temporal semantics and game readiness.
- Adapter: execution reports `AMD Radeon RX 9070 XT [ZLUDA]`, gated by device
  name. User confirmed normal game exit; no game injection or launch this round.
- Commands and reproduction: `docs/REAL_COLOR_REPLAY.md`.
- Artifacts: `results/20260905_real_color_probe/analysis/summary.json`,
  `controls_summary.json`, all three repeated execution directories, raw outputs
  and `native_head/manifest.json`. All 500 Python tests pass.
- A byte-exact 640x360 crop from the verified 2342x1317 playable-game FFX capture
  is bound to preblock and texture-preserving postblock. All 156 slots execute,
  with original model SHA256
  `A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`.
- Both real-color runs produce
  `E92278B73ACC06B4C78C5E01370726B5583869542718EEE1EF5EE0178DBB02E6`.
  Clearing the initial arena gives the exact same complete pre-head arena and
  output. There are no explicit RTX intermediate injections.
- Zeroing only neural color, with the same real postblock base and zero initial
  arena, changes 26,735,093 pre-head bytes and 656,711 final RGB components.
  This rejects a base-copy-only explanation for this input.
- Native D3D12 output-head comparison: RGB MAE 0.000443146/max 0.0883789.
  This is internal AMD backend disagreement, not an RTX accuracy measurement;
  native execution PASS does not accept numerical equivalence.
- Initial real-color launch+sync totals: 428.2/406.8 ms at 640x360; not realtime
  or pure GPU timing. HDR range, color contract and temporal binding remain open.
- Provenance safeguard: rejected stale slot-154 PTX/hash before any graph launch;
  used audited texture-preserving `07_surface.ptx` in a new plan, verified all
  51 module hashes, retained the failed run and original plan unchanged.

## E-171 Original-network offline result displayed as a labeled static game preview

- Date: 2026-09-05. PASS for static display, input/output switching and manual
  stop; NOT live full-network inference or quality equivalence.
- Session: `results/20260905_175035_151_gowr_capture_session/`, PID 9524.
  `preview_network.png`, `preview_input.png` and `preview_stopped.png` show the
  labeled central panel and its removal. `manual_stop_audit.json` binds their
  hashes, process identity, unchanged game images and system-event query.
- Source: E-170 zero-initial-arena complete inference. Output SHA256
  `E92278B73ACC06B4C78C5E01370726B5583869542718EEE1EF5EE0178DBB02E6`.
  `preview_manifest.json` in `results/20260905_static_preview_stage_v2/`
  binds the original input, model and two immutable labeled panel payloads.
- Panel: 644x384, including an unmodified 640x360 half-float image, 20-row
  STATIC / NOT LIVE INFERENCE label and 2-pixel frame. Only this central region
  is copied after the exact FFX output readable transition; no GPU waits or
  network inference occur in the hook. Uploads stay resident until process exit
  so stopping cannot free data still referenced by previously recorded lists.
- Independent WARP/AMD hook tests: 9 cases each, all full-surface bytes exact,
  zero D3D12 errors/warnings, including baseline, repeated output, input switch,
  stop/next-frame restoration and re-arm. `preview_gate/gate.json` binds tested
  files before/after execution. Original FSR3/FSR4 regression and 502 Python
  tests pass.
- First live window: 6,682 recorded region copies, manual stop `enabled=false`,
  `failed=false`, process responding. No matching System 4101/41/6008 events
  since game start; this is a limited observation, not a long-term safety proof.
- Second window: 5,853 copies over the 60-second deadline, then automatic
  `enabled=false`, `failed=false`. `preview_timeout.png` confirms removal;
  `automatic_timeout_audit.json` records the same successful process/hash/event
  checks. A final explicit stop clears the idle preview mode; game left running.
- Method/quality were restored to AMD FSR 3.1 / Quality; frame generation stayed
  off. No game DLL or save file was deployed/edited. Instructions are in
  `docs/STATIC_GAME_PREVIEW.md`.

## ROCm native tensor implementation batch (2026-09-05; partial)

- `results/20260905_rocm_native_baseline/operator_probe.json`: isolated gfx1201
  GEMM, LayerNorm, SDPA, FP8 conversion and backward capability PASS in a dedicated
  Python 3.12/ROCm environment. Not a full-network or steady-state performance gate.
- `results/20260905_rocm_swin_four_cases/manifest.json`: four original-weight
  Swin1h/32 isolated controls PASS against RTX; bytes match the existing DXIL
  candidate, which is still rejected for integrated whole-frame quality.
- `results/20260905_rocm_head_v1/manifest.json`: output-head ROCm port versus the
  prior native head control; fused FP16 values exact, SDR-clamped surface error
  within the port gate. Residual maximum error remains 0.3555; no HDR or RTX
  full-frame claim is made from the clamped near-black surface.
- `results/20260905_rocm_native_baseline/gradients.json`: finite nonzero GPU
  gradients for selected input/scales/tail, no optimizer steps or training.
- `results/20260905_rocm_native_baseline/teacher_server.json`: existing 4090D
  server read only; not accepted as a teacher, no remote files created.
- `deliverables/rocm_teacher_sequence_20260905_v2/package_manifest.json`: built
  candidate collection harness, NOT RTX-validated, no private vendor payloads
  or real sequences included. Missing input contracts are not synthesized.
- `results/20260905_rocm_native_baseline/final_audit.json`: frozen assets/game
  hashes unchanged and no queried recent display-reset/restart event.

## Forbidden "evidence" (never accepted)

Bootstrap-only progress (2026-09-05): fixture
`results/20260905_105834_540_ffx_bootstrap_test/` passes exact FFX forwarding and
scoped child cleanup; actual `results/20260905_105936_344_gowr_ffx_observation/`
reaches the GoWR main menu with header-only observation and unchanged audited
images. Its log was later parsed after normal exit. Neither launching nor
loading the observer is accepted as proof of NR GPU execution or game quality.

- "DLSS 5 menu appeared"
- "Game launched"
- "FakeNVAPI passed vendor check"
- "NGX returned success"
- "Feature handle non-null"
- "ReShade overlay shows DLSS 5"
- "Output looks prettier"
- "FSR4 was called through DLSS API"

Real success requires GPU execution evidence on the AMD adapter (vendor 0x1002).
