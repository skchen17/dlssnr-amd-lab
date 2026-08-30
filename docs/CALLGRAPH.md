# CALLGRAPH.md

DLSSNR real call chain — decision tree for Phase 6 (case selection) and Phase 7
(go/no-go). Updated 2026-08-30 after proprietary files became available:
static payload analysis is complete (docs/BINARY_ANALYSIS.md); dynamic trace
progressed to the first real load on AMD, which stopped at the NGX vendor gate
(0xbad00001) — so launch-side API evidence is still pending.

## 1. Hypothesized chain (pre-trace, for reference only)

```
nr_host (our D3D12 host, tools/nr_host)
  -> NVSDK_NGX_D3D12_Init / CreateFeature (feature id 18 = Neural Rendering,
     RECONSTRUCTED signatures — see nr_host.cpp header)
  -> nvngx.dll (driver-side NGX loader)
     -> nvngx_dlssnr.dll (feature plugin)
        -> ??? one of CASE A / B / C / D (Section 2)
        -> GPU binary (fatbin / CUBIN / PTX?) — to be located by binary_probe
           -> NVIDIA driver kernel dispatch
```

## 2. Case discrimination (Phase 6 decision tree)

Evidence sources, in priority order:

1. `module_trace.log` — which modules the runtime loads (nvapi64.dll? nvcuda.dll?
   nvngx*.dll? d3d12 MetaCommand-only?).
2. `nvapi_trace.jsonl` — which `nvapi_QueryInterface` IDs are resolved
   (decoded against tools/nvapi_trace/nvapi_ids.h; unknown IDs logged raw).
3. `binary_manifest.json` (binary_probe) — presence of fatbin containers
   (magic B1 43 62 46), CUBIN ELFs (e_machine=190), PTX text markers
   (`.target sm_`), entropy runs (weights).
4. D3D12-side observation — `ID3D12GraphicsCommandList5::DispatchRays` /
   MetaCommand usage, Cooperative Vector API calls.

```
Q1: does module_trace show nvcuda.dll load OR cuda_driver imports?
  YES -> candidate CASE B
    Q2: does nvapi_trace show Cubin/CuModule family IDs at all?
      YES -> CASE C (mixed): list both call sets; replacement must cover both.
      NO  -> CASE B: pure CUDA Driver API. Replacement = ZLUDA nvcuda shim.
  NO ->
Q3: does nvapi_trace show NvAPI D3D12 Cubin/CuModule/LaunchCuKernelChain IDs
    (community IDs in nvapi_ids.h carry '?' — verify each against the live trace)?
  YES -> CASE A: NVAPI-mediated CUBIN launch.
    Replacement = nvapi_amd shim that translates those calls to HIP/HSACO dispatch.
  NO ->
Q4: any D3D12 MetaCommand / new neural-API evidence (DispatchRays-only graph,
    untracked vendor capability bits)?
  YES -> CASE D: vendor-neutral or new D3D12 path.
    Replacement = D3D12 stays native on AMD; intercept only the unsupported
    capability negotiation. This is the cheapest case IF real.
  NO  -> CASE X (unknown): stop, capture raw traces, escalate. Never guess.
```

Decision record (2026-08-30, from first real load + static analysis):

| Question | Evidence file | Result |
|---|---|---|
| Q1 nvcuda load / cuda imports | binary_manifest_dlssnr.json + forcload module_trace | **NO**: no nvcuda import (static/delay); no nvcuda load observed. Post-gate unknown (init died before). |
| Q2 nvapi Cubin family IDs | nvapi_trace | not reached — no nvapi call before vendor gate |
| Q3 nvapi launch-chain IDs | nvapi_trace | not reached — same |
| Q4 MetaCommand / new API | D3D12 debug layer + module_trace | not reached — same |
| **Selected case** | static payload evidence | **LEANING CASE A** (CUBIN module payload + no CUDA-driver imports + RTX-40 patch = CUBIN-swap precedent); runtime confirmation PENDING vendor-gate passage |

First dynamic fact (AMD side, unmodified DLLs, no spoofing):
`nr_host --force-load` loaded both DLLs (dlss@0x7FFB9D5E0000, nr@0x7FFB93790000),
resolved all 5 reconstructed NGX exports, created a D3D12 device on vendor=0x1002
and called `NVSDK_NGX_D3D12_Init`, which returned **0xbad00001** (hardware/vendor
not supported) — the runtime executed a REAL capability check and rejected the AMD
adapter cleanly, loading no NVIDIA support modules beforehand.
Evidence: results/20260830_170305/nr_host_forceload_stdout.log,
forcload_20260830_182045_module_trace.log.

## 3. Evidence-driven replacement points

| Link | Replace with (AMD) | Status | Evidence |
|---|---|---|---|
| nvapi64.dll CUBIN APIs (CASE A) | nvapi_amd shim -> HIP/HSACO dispatch; reuse build\nvapi64.dll trace shim as skeleton | not started | trace pending |
| nvcuda.dll Driver API (CASE B/C) | ZLUDA nvcuda replacement (active project; known RDNA4+Windows issues, see RESEARCH.md) | not started | trace pending |
| GPU binary payload | PTX -> ZLUDA -> LLVM AMDGPU -> HSACO (gate in Section 4) | **gated** | binary_probe pending |
| D3D12 <-> compute sync | HIP external semaphore / fence interop | **PROVEN (S2)** | results/20260830_170305/d3d12_hip_interop.json, docs/INTEROP.md |
| D3D12 shared buffers/textures | hipImportExternalMemory (D3D12Heap / D3D12Resource) | **PROVEN (S2)** | docs/INTEROP.md |
| Weights transfer | plain host->device copies via HIP (format from binary_probe entropy analysis) | tooling ready | tools/binary_probe |
| WMMA/MMA fast paths | plain FP16 GEMM first (S1 test D passes); WMMA blocked by S-B4 | partial | hip_probe.json, BLOCKERS.md |

## 4. PTX go/no-go gate (Phase 7 framework)

Prerequisite: binary_probe manifest for nvngx_dlssnr.dll (currently MISSING_PREREQUISITE, B-1).

```
GATE-0  binary_probe finds fatbin container(s)?
        NO  -> kernel extraction BLOCKED at container level; go to Track B.
        YES -> enumerate embedded payloads, continue.
GATE-1  any embedded payload contains PTX text markers (.target sm_*)?
        NO  -> PTX path = BLOCKED (SASS-only). Go to Track B. Do NOT emulate SASS.
        YES -> extract PTX, record sm target(s), continue.
GATE-2  PTX sm target supported by the available toolchain?
        (ZLUDA + LLVM AMDGPU; check RESEARCH.md for current supported ceiling)
        NO  -> PTX path = BLOCKED at toolchain level; Track B.
        YES -> continue.
GATE-3  single-kernel compile: PTX -> ZLUDA/LLVM -> HSACO for gfx1201 succeeds?
        NO  -> log exact LLVM error (cf. S-B4-style backend gaps); per-kernel
               manual rewrite decision. Track B for failing kernels.
        YES -> continue.
GATE-4  kernel_lab numerical compare (build tools/kernel_lab):
        same deterministic input as reference package (results/*_reference_package),
        tolerances per docs/RESULTS.md policy (fp32 <=1e-5, fp16 <=1e-2, fp8 vs
        fp16-upcast reference, NaN/Inf = FAIL).
        PASS -> GO: proceed to CASE replacement wiring (Section 3).
        FAIL -> per-kernel: diagnose (unsupported op / denormals / ordering).
                Fixable -> iterate; unfixable -> Track B for that kernel.
```

Track B (reconstruction, only if PTX path blocked): identify kernel math from
weights shapes + I/O tensors + published architecture descriptions; re-implement
as HIP kernels with numerical validation against the NVIDIA reference output
(from results/*_reference_package run on an RTX machine). Explicitly: Track B is
a REIMPLEMENTATION and must be labeled as such in every result — it is never
presented as "executing the leaked kernels".

Current gate status (2026-08-30, files now available):
- GATE-0 **PASS** — 15 CUBIN ELF containers located in .rsrc (binary_manifest_dlssnr.json).
- GATE-1 **FAIL** — zero PTX markers in nvngx_dlssnr.dll → **PTX path = BLOCKED**.
  Per spec section 12: do NOT emulate SASS, do NOT claim ZLUDA can execute it.
- Track B is the mandated path, with one material upgrade: because kernel NAMES and
  per-kernel `.nv.info` parameter metadata survive intact, Track B's "recover the
  compute graph" step starts from a named, structured kernel inventory (Swin-Transformer
  backbone with fused pre/post blocks and FP8 variants) instead of black-box SASS.
- The CASE A machinery (NVAPI CuModule/Cubin shim → HIP/HSACO dispatch) remains the
  intended transport layer once the runtime's launch API set is confirmed post vendor gate.

## 5. Trace artifacts (templates)

- LoadLibrary/GetProcAddress trace: `results/<ts>/module_trace.log`
  (inject build\module_trace.dll before NGX init)
- nvapi_QueryInterface decode: `results/<ts>/nvapi_trace.jsonl`
  (place build\nvapi64.dll where the target resolves nvapi64.dll; set NVAPI_TRACE_LOG)
- CUDA Driver API calls (if any): covered by module_trace load events + a future
  nvcuda shim; no separate tracer yet (not needed until CASE B/C is confirmed)
- Run wrappers: `scripts\run_amd.ps1` (AMD side, shims on), `scripts\run_reference.ps1`
  (NVIDIA machine only — reference must be real)

(Trace data: first load run captured module loads up to the vendor gate — see
Section 2 first-dynamic-fact. Launch-side traces (nvapi IDs, CUDA driver calls)
will be appended here after the vendor gate is passed; any gate-passage mechanism
used purely to OBSERVE the call chain must be labeled trace-aid and never counted
as execution evidence.)
