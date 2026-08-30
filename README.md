# dlssnr-amd-lab

Experimental GPU interoperability / reverse-engineering project.

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

| Level | Meaning |
|---|---|
| S0 | environment recorded |
| S1 | AMD HIP + RDNA4 compute works |
| S2 | D3D12/HIP interop works |
| S3 | DLSSNR binary/call graph understood |
| S4 | reference NR host works on NVIDIA |
| S5 | AMD passes NGX/NVAPI initialization |
| S6 | first DLSSNR-originated GPU kernel runs on AMD |
| S7 | complete single NR frame on AMD |
| S8 | AMD output matches reference numerically/visually |
| S9 | stable 1000-frame run |
| S10 | performance optimization |

## Layout

```
docs/        STATUS.md EVIDENCE.md CALLGRAPH.md ARCHITECTURE.md RESULTS.md BLOCKERS.md RESEARCH.md BINARY_ANALYSIS.md
tools/
  hip_probe/           Phase 1: RDNA4 compute baseline (FP32/FP16/FP8/WMMA)
  d3d12_hip_interop/   Phase 2: D3D12 <-> HIP resource interop probe
  binary_probe/        Phase 3: PE/fatbin/CUBIN/PTX static analysis
  module_trace/        Phase 4: LoadLibrary/GetProcAddress tracing layer
  nvapi_trace/         Phase 4: nvapi_QueryInterface / NVAPI-CUDA shim
  nr_host/             Phase 5: standalone D3D12 NGX NR host
scripts/     collect_environment.ps1 probe_dlssnr.ps1 run_reference.ps1 run_amd.ps1
results/     text logs / JSON / CSV only (no binaries)
third_party/ open-source references via git submodule
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

# Phase 5: NR host skeleton (blocked run emits a reference package)
.\build\nr_host.exe --frames 1 --width 512 --height 512 --trace
```

Run wrappers: `scripts\run_amd.ps1` (AMD side, trace shims) and
`scripts\run_reference.ps1` (NVIDIA machine only — the reference must be real).
Docs: docs/STATUS.md (S-level), docs/INTEROP.md (S2 recipe), docs/CALLGRAPH.md
(CASE A–D decision tree + PTX go/no-go gates), docs/RESULTS.md (all experiments).

## Legal

Experiments use only binaries the user legally owns. No anti-cheat online games.
Standalone host first. Reverse-engineering is performed for interoperability research;
no redistribution of proprietary code or data.
