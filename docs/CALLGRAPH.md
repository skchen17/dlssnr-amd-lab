# CALLGRAPH.md

DLSSNR real call chain — decision tree for Phase 6 (case selection) and Phase 7
(go/no-go). Updated 2026-08-30 after proprietary files became available:
static payload analysis is complete (docs/BINARY_ANALYSIS.md); dynamic trace
progressed to real loads on AMD and an RTX 5060. With corrected DLSS5-Feeder IDs,
RTX passes NGX Init and the complete public DLAA path while AMD still returns
`0xBAD00001`. Round 3 completed 8/8 Evaluate calls plus readback and observes the
public support chain through driver NGX, DLSS, NVAPI, and CUDA modules. A later
RTX 5070 reference run proves the private high-level chain: RenoDX hooks NGX
Create/Evaluate, initializes signed DLSSNR 310.8, creates feature 18, and evaluates
it inline. R-18 traces that successful private path and observes D3D12 plus system
NVAPI (`nvapi_QueryInterface` and `nvapi_Direct_GetMethod`) but no classic CUDA
Driver API module/export resolution after DLSSNR loads. R-19 then captures the
exact official CreateCuModule/CreateCuFunction/LaunchCuKernelChain/Destroy IDs.
R-20 wraps those typed interfaces and records nine real modules, 96 named
functions and 46,800 successful single-kernel chains (300 frames × 156 calls),
followed by clean destruction. CASE A is now proven at invocation level; the
complete 156-call order and parameter bytes were captured by R-21 across five
structurally identical frames. The reference graph is closed for the tested
640x360/variant-02 configuration. R-22 replays that complete launch envelope on
RX 9070 XT with a lab HIP marker backend and validates device-side parameter
carriage; this proves the AMD transport scaffold, not neural execution.

## 1. Hypothesized chain (pre-trace, for reference only)

```
nr_host (our D3D12 host, tools/nr_host)
  -> NVSDK_NGX_D3D12_Init / CreateFeature (public DLSS/DLAA host contract;
     feature id 18 remains PRIVATE ABI and is not called directly by nr_host)
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

Decision record (2026-08-30, updated by the successful R-18 private-path trace):

| Question | Evidence file | Result |
|---|---|---|
| Q1 nvcuda load / cuda imports | binary manifest + R-16/R-18 traces | Public DLSS loads `nvcuda64.dll`/`cuGetExportTable`; the isolated successful Feature-18 path in R-18 does **not**. DLSSNR also has no static/delay nvcuda import. Pure CASE B is rejected for this run. |
| Q2 NVAPI Cubin family IDs | R-18 module/export trace | System `nvapi64.dll` is active and both `nvapi_QueryInterface` and private `nvapi_Direct_GetMethod` are resolved. Individual IDs were not wrapped by v6. |
| Q3 NVAPI launch-chain IDs | R-19 QueryInterface-ID trace + NVIDIA R610 headers | **YES** — `AD1A677D` CreateCuModule, `E2436E22` CreateCuFunction, `24973538` LaunchCuKernelChain, `DF295EA6` DestroyCuFunction, `41C65285` DestroyCuModule. |
| Q4 MetaCommand / new API | D3D12 debug layer + R-18 module trace | no positive MetaCommand evidence; ordinary D3D12 dependencies are present. |
| **Selected case** | R-21/E-17 | **CASE A PROVEN AT GRAPH LEVEL** — typed D3D12 NVAPI calls execute a stable 156-slot frame; module/function/order/geometry/parameter bytes are captured and classic CUDA is absent. |

First dynamic fact (AMD side, unmodified DLLs, no spoofing):
`nr_host --force-load` loaded both DLLs (dlss@0x7FFB9D5E0000, nr@0x7FFB93790000),
resolved the NGX exports, created a D3D12 device on vendor=0x1002 and called
`NVSDK_NGX_D3D12_Init`, which returned **0xbad00001** =
`NVSDK_NGX_Result_FAIL_FeatureNotSupported`. Round 1 made this call through a
RECONSTRUCTED ABI (later audited as invalid — docs/NGX_ABI_AUDIT.md E1/E3);
Round 2 re-ran it under the official ABI and got the same result. R-15 then used
the same corrected Feeder identifiers on RTX and reached Init/CreateFeature
success, confirming an adapter-dependent NVIDIA support gate on AMD. No NVIDIA
support modules load on AMD before that rejection; on RTX the public DLSS path
loads NGX, DLSS, NVAPI, and CUDA support modules before Evaluate validation fails.
Evidence: results/20260830_170305/nr_host_forceload_stdout.log,
forcload_20260830_182045_module_trace.log,
results/20260830_231421_rtx5060_v2/, docs/NGX_ABI_AUDIT.md (Post-script).

## 3. Evidence-driven replacement points

| Link | Replace with (AMD) | Status | Evidence |
|---|---|---|---|
| nvapi64.dll CUBIN APIs (CASE A) | nvapi_amd shim -> HIP/D3D12 dispatch | **seven-interface backend and boundary registry PASS standalone; Feature-18 injection and neural registry pending** | R-21/E-17, R-22/E-18, R-25..R-27 + NVIDIA R610 headers |
| nvcuda.dll Driver API (CASE B/C) | ZLUDA nvcuda replacement (active project; known RDNA4+Windows issues, see RESEARCH.md) | **not selected for observed NR path** | R-18 has no private-path CUDA load |
| GPU binary payload | function-isolated PTX -> ZLUDA/targeted lowering -> AMD code | **PARTIAL**: original clear executes; native copy lowering passes; neural instructions pending | R-24/R-26/R-27 |
| D3D12 <-> compute sync | HIP external semaphore / fence interop | **PASS (S2 PASS-A)** | results/20260830_170305/d3d12_hip_interop.json, docs/INTEROP.md |
| D3D12 shared buffers | hipImportExternalMemory (D3D12Heap / D3D12Resource) | **PASS (S2 PASS-A)** | docs/INTEROP.md |
| D3D12 shared textures | hipImportExternalMemory buffer view | **PARTIAL (Phase G)**: roundtrip byte-exact for RGBA8/RGBA16F/R32F/RG16F, but memory layout is NOT identity (swizzled) — kernels need swizzle-aware addressing | results/*_texture_interop/texture_interop.json |
| Weights transfer | plain host->device copies via HIP (format from binary_probe entropy analysis) | tooling ready | tools/binary_probe |
| WMMA/MMA fast paths | plain FP16 GEMM first (S1 test D passes); WMMA blocked by S-B4 | partial | hip_probe.json, BLOCKERS.md |

## 4. PTX go/no-go gate (Phase 7 framework)

Prerequisite satisfied: the user-supplied runtime has been mapped and all 15
compressed PTX payloads have been extracted locally with metadata-only manifests.

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

Track B (per-function fallback when PTX translation is blocked): identify kernel math from
weights shapes + I/O tensors + published architecture descriptions; re-implement
as HIP kernels with numerical validation against the NVIDIA reference output
(from results/*_reference_package run on an RTX machine). Explicitly: Track B is
a REIMPLEMENTATION and must be labeled as such in every result — it is never
presented as "executing the leaked kernels".

Current gate status (corrected 2026-08-31):
- GATE-0 **PASS** — 15 `0xBA55ED50` runtime containers located in `.rsrc`.
- GATE-1 **PASS** — all 15 contain Zstd-compressed PTX 9.4/sm_120; 231 entries.
  The previous plaintext-marker FAIL was a compressed-data false negative.
- GATE-2/3 **PARTIAL** — ZLUDA v7-preview.3 on RX 9070 XT compiles and executes
  function-isolated `cc_cb_clear` with zero readback mismatches. It cannot parse
  the copy kernel's surface store or the tested neural kernel's MMA/FP8 family.
- GATE-4 is pending a neural operator and targeted RTX tensor oracle.
- CASE A is confirmed by R-19..R-21. `nvapi_amd` implements the five-call
  transport plus both descriptor-object calls; original PTX, targeted D3D12/HIP
  lowering or a clean-room fallback supplies each real kernel implementation.

## 5. Trace artifacts (templates)

- LoadLibrary/GetProcAddress trace: `results/<ts>/module_trace.log`
  (inject build\module_trace.dll before NGX init; Round 2 v2 exposes
  ModuleTrace_InitializeAndWait so the host can wait for hooks_installed=true;
  tests/module_trace_selftest gates readiness)
- nvapi_QueryInterface decode: `results/<ts>/nvapi_trace.jsonl`
  (place build\nvapi64.dll where the target resolves nvapi64.dll; set NVAPI_TRACE_LOG)
- CUDA Driver API calls (if any): covered by module_trace load events + a future
  nvcuda shim; no separate tracer yet (not needed until CASE B/C is confirmed)
- Run wrappers: `scripts\run_amd.ps1` (AMD side, shims on), `scripts\run_reference.ps1`
  (NVIDIA machine only — reference must be real)

(Trace data: R-18 captures a naturally successful NVIDIA Feature-18 execution,
not a vendor-spoof trace-aid. R-19's narrow QueryInterface wrapper identifies the
complete official CASE-A lifecycle without replacing system NVAPI or touching
Direct_GetMethod. The next tracer wraps only those typed official return pointers
to record actual module/function/launch arguments. R-20 proves those calls and
establishes the 156-launch frame length; the next capture uses that period to
avoid v8's 60-call sampling alias. R-21 completes that capture: five identical
156-slot graphs, 153 stable parameter blocks, three boundary-state variations,
and exact mapping of nine runtime blobs to signed-DLL offsets.)
