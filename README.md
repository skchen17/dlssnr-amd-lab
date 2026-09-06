# dlssnr-amd-lab

Experimental GPU interoperability / reverse-engineering project.

## Latest checkpoint — 2026-09-06

The current delivery track is an original-weight, AMD-native ROCm **single-color
tensor candidate**, not a complete or quality-verified DLSS5 replacement.
The 71-stage candidate runs whole frames with padding/cropping, without NVIDIA
DLL/PTX/ZLUDA or CPU neural fallback in its forward path. Its current output uses
an explicitly diagnostic SDR residual composition; full temporal/HDR/HUD and
real-time game acceptance remain open.

The opt-in `native_opt3` schedule completed twelve 3840×2160 synthetic-input
repetitions with identical output: hot staged median **2515.27 ms**, allocator
reserved peak **4.687 GB**, device-wide stage samples up to **5.123 GB**.
These are offline diagnostic measurements, **not game FPS or 4K60**. Independent
D3D12/ROCm cross-process 640×360 alternating-input tests also pass.
See [latest optimization and commands](docs/NATIVE_OPT3.md),
[status](docs/STATUS.md), and [native route](docs/ROCM_NATIVE_IMPLEMENTATION.md).

This repository publishes source, tests and research documentation only. Original
weights, teacher sequences, captures, game images, generated binaries and new raw
experiment reports remain local/private. Historical document links to `results/`
identify local evidence and are not all downloadable from GitHub. Reproduction
requiring those assets needs separately provisioned local files; a source checkout
alone is not a ready-to-run game plugin. Recorded test results were obtained in
the documented local environment, not a clean checkout lacking private fixtures.
Fetch the pinned public NVAPI dependency with `git submodule update --init third_party/nvapi`.
For the read-only teacher environment probe, pass your own SSH host explicitly:
`python scripts/probe_teacher_server.py --host YOUR_SSH_ALIAS --output results/teacher_probe_NEW.json`.

The DLL-loading requirements and S-level status below describe the **historical
runtime-transplant track**, not acceptance requirements for the new native route.

## AMD-native operator reconstruction track

The delivery-oriented track now reconstructs the model at tensor/operator level
with user-local weights. It does not require NVIDIA instruction semantics or
bitwise-identical intermediates. The older PTX parity work remains a reference
oracle. See `docs/OPERATOR_RECONSTRUCTION.md` for the local-only model-pack and
complete-frame output gates.

**Goal:** determine whether the leaked NVIDIA DLSS Neural Rendering (DLSS 5) runtime
(`nvngx_dlssnr.dll`) can have its neural kernels *actually executed* on an AMD RDNA4 GPU
(RX 9070 / RX 9070 XT, gfx1200/gfx1201) — not faked, not replaced by FSR/XeSS.

"Real execution" requires ALL of:

1. The unmodified `nvngx_dlssnr.dll` is actually loaded.
2. Its Neural Rendering feature is actually created.
3. The runtime's GPU neural kernels actually execute.
4. The GPU work lands on the AMD adapter (vendor ID 0x1002) with evidence.
5. No FSR/XeSS/OptiScaler replacement, no CPU fake, no "NGX returned success" as proof.

## Status scale (see docs/STATUS.md)

Current state: **S6 PASS**; **S7 remains open**. The new clean-room route now runs
the complete slot-154 output head on RX 9070 XT: 256 FP8 MMAs, stable attention,
16 FP16 tail MMAs, exact 8x8 spatial mapping, residual composition and a real
D3D12 RGBA16F texture roundtrip pass as one 9/9 pipeline. Against the selected
RTX store oracle, final RGB mean/max absolute error is `0.000159/0.000764`.
This is a genuine AMD-native neural operator, but it currently starts from the
captured slot-154 activation. S7 still requires preceding native decoder stages
to supply that tensor without capture/injection, followed by multi-frame and
live NGX/game integration. The older 156-slot PTX replay remains a useful
diagnostic baseline, not the delivery architecture.

| Level | Meaning |
|---|---|
| S0 | environment recorded |
| S1 | AMD HIP + RDNA4 compute works |
| S2 | D3D12/HIP interop works |
| S3 | DLSSNR binary/call graph understood |
| S4 | reference NR host works on NVIDIA |
| S5 | AMD passes NGX/NVAPI initialization |
| S6 | first validated DLSSNR-originated neural kernel runs on AMD |
| S7 | complete single NR frame on AMD |
| S8 | AMD output matches reference numerically/visually |
| S9 | stable 1000-frame run |
| S10 | performance optimization |

## Layout

```
docs/        STATUS.md EVIDENCE.md CALLGRAPH.md ARCHITECTURE.md RESULTS.md BLOCKERS.md
             RESEARCH.md BINARY_ANALYSIS.md NGX_ABI_AUDIT.md INTEROP.md
tools/
  hip_probe/           Phase 1: RDNA4 compute baseline (FP32/FP16/FP8/WMMA)
  d3d12_hip_interop/   Phase 2: D3D12 <-> HIP resource interop probe
  d3d12_residual_bridge/ D3D12 RGBA16F <-> HIP output-head composition bridge
  output_head_resident/ Single-process, GPU-resident native output head
  output_head_resident_d3d12/ Resident output head with in-process D3D12 texture/fence bridge
  binary_probe/        Phase 3: PE/fatbin/CUBIN/PTX static analysis
  module_trace/        Phase 4/round-2 E: LoadLibrary/GetProcAddress tracing (v2: sync gate)
  nvapi_trace/         Phase 4/round-2 F: nvapi_QueryInterface shim (pure-jmp trampolines)
  nr_host/             Phase 5/round-2 C: genuine DLSS/DLAA NGX host (official ABI)
  ngx_abi_probe/       Round 2 B: official-ABI regression test (static_assert + exports)
  amd_graph_replay/    Track B: 156-slot AMD transport replay (marker-only, not S6)
  output_head_*/       AMD-native slot-154 operator pipeline and RTX-oracle gates
  nvapi_amd/           Official seven-interface NVAPI-compatible AMD backend
  zluda_ptx_probe/     RX 9070 XT PTX module/function/launch verification
tests/
  module_trace_selftest/   Round 2 E: load->resolve->unload chain self-test
  nvapi_trampoline_test/   Round 2 F: trampoline byte-exactness, 1..10 args
  texture_interop_test/    Round 2 G: RGBA8/RGBA16F/R32F/RG16F interop gate
scripts/     collect_environment.ps1 build_all.ps1 run_reference.ps1 run_amd.ps1
             run_nvidia_reference.ps1 run_ngx_abi_probe.ps1 run_module_trace_selftest.ps1
             run_nvapi_trampoline_test.ps1 run_texture_interop_test.ps1
results/     text logs / JSON / CSV only (no binaries)
reference_bundle/  one-command evidence package for an RTX machine (no proprietary DLLs)
third_party/ nvidia-dlss (official headers+loader lib, NVIDIA/DLSS @ a291cc7d2cc6),
             DLSS5-Feeder (MIT, round-2 D)
```

## Proprietary file policy

This repository MUST NOT contain:

- `nvngx_dlssnr.dll`, `nvngx_dlss.dll`, or any NVIDIA proprietary binary/model
- game files
- any DLL without redistribution rights

Such files are supplied by the user locally, referenced through environment variables:

```
DLSSNR_DLL_PATH           # path to nvngx_dlssnr.dll
DLSS_DLL_PATH             # path to nvngx_dlss.dll
RENODX_DLSS5_ADDON_PATH   # path to renodx-dlss5 addon
```

If these are absent, experiments that need them are marked `MISSING_PREREQUISITE`
in docs/BLOCKERS.md and everything else proceeds.

## Quick start

No CMake on this box — build via PowerShell (HIP SDK LLVM: clang-cl for host code,
clang++ -x hip for device code; see scripts/build_common.ps1 for toolchain resolution):

```powershell
# Phase 0: record the machine
powershell -ExecutionPolicy Bypass -File scripts\collect_environment.ps1

# Build everything (or: -Only hip_probe,d3d12_hip_interop,nr_host)
powershell -ExecutionPolicy Bypass -File scripts\build_all.ps1

# Phase 1: RDNA4 compute baseline
.\build\hip_probe.exe --json results\<ts>\hip_probe.json

# Phase 2: interop gate
.\build\d3d12_hip_interop.exe --json results\<ts>\d3d12_hip_interop.json

# Phase 3: static analysis (proprietary DLL via env var, read-only)
.\build\binary_probe.exe "$env:DLSSNR_DLL_PATH" --json results\<ts>\binary_manifest.json

# Phase 5 / round-2 C: genuine NGX host (official ABI; on AMD this box reaches the
# init rejection and exits 6 — the full create/evaluate/release pipeline needs RTX)
.\build\nr_host.exe --frames 1 --width 512 --height 512 --trace

# Round 2 self-tests
powershell -ExecutionPolicy Bypass -File scripts\run_ngx_abi_probe.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_module_trace_selftest.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_nvapi_trampoline_test.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_texture_interop_test.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_nvapi_amd_output_head_selftest.ps1

# Track-B AMD graph transport (marker-only; explicitly not S6)
powershell -ExecutionPolicy Bypass -File scripts\run_amd_graph_replay.ps1

# Resident output-head validation (captured activation input; not game-ready)
.\build\output_head_resident.exe `
  deliverables\postblock_mma_trace_reference_20260904_154238\payload\activation_arena.raw `
  local_models\decoded_310_8\model_arena.raw - `
  results\20260905_013000_output_head_pipeline_rx9070xt\rgba_residual_fp16.raw `
  results\resident\rgba_residual_fp16.raw results\resident\output_rgba16f.raw `
  results\resident\manifest.json 20

# Extract used PTX functions and test real translation/execution on RX 9070 XT
powershell -ExecutionPolicy Bypass -File scripts\run_zluda_ptx_probe.ps1

# One-command NVIDIA reference pipeline (run on an RTX machine)
powershell -ExecutionPolicy Bypass -File scripts\run_nvidia_reference.ps1
```

The output-head self-test enables `MODULE_TRACE_AMD_INTEROP=1` internally and
validates automatic activation/model buffer import plus surface descriptor
binding. It writes a real RGBA16F texture and compares every byte, but it is not
a game-readiness claim: command-list/HIP queue ordering and the native slots
0-153 producer path are still open.

Run wrappers: `scripts\run_amd.ps1` (AMD side, trace shims) and
`scripts\run_nvidia_reference.ps1` (NVIDIA machine only — the reference must be real;
this box emits a `reference_bundle/` marked BLOCKED_EXTERNAL_HARDWARE).
Docs: docs/STATUS.md (S-level), docs/NGX_ABI_AUDIT.md (official ABI audit),
docs/INTEROP.md (S2 recipe + texture-layout caveat), docs/CALLGRAPH.md
(CASE A–D decision tree + PTX go/no-go gates), docs/RESULTS.md (all experiments).

## Legal

Experiments use only binaries the user legally owns. No anti-cheat online games.
Standalone host first. Reverse-engineering is performed for interoperability research;
no redistribution of proprietary code or data.
