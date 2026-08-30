# BINARY_ANALYSIS.md — nvngx_dlssnr.dll static analysis

**State: ANALYZED (read-only) 2026-08-30.** Files legally supplied by the user
(see docs/PROPRIETARY_FILES.md for ledger: version / SHA-256 / Authenticode).
Binaries stay outside the repo; only metadata is recorded here.

## Key facts

| Fact | nvngx_dlssnr.dll (NR) | nvngx_dlss.dll (SR runtime) |
|---|---|---|
| Size | 165,840,496 B | 58,977,904 B |
| FileVersion | 310.8.0.0 | 310.7.0.0 |
| Payload location | `.rsrc` raw size 147,696,792 B | `.data` |
| CUDA modules | **15 CUBIN ELF payloads, all e_machine=190 (EM_CUDA)** | 670 ELF payloads |
| Target arch | **sm_120 (consumer Blackwell, RTX 50) — 15 markers, exactly 1 per module** | sm_89 (Ada, 163 markers) + sm_80 (6 markers) |
| PTX present | **NO — zero `.target`/`.version`/`.visible .entry` markers in the whole file** | YES — `.target sm_89` ×16, `.version 8.7` ×8 |
| Static imports | VERSION, ADVAPI32, USER32, KERNEL32 only | same set |
| nvcuda.dll import | **NO (static or delay)** | NO |
| nvapi64.dll import | **NO (static or delay)** | NO |
| Weights candidates | 6 high-entropy runs ≥256 KB inside `.data` (0.5–1.1 MB each, maxEnt 7.995) | none ≥256 KB (low-entropy fp16/fp8 tables) |

## Kernel-level findings (carved CUBIN inspection, llvm-objdump)

- Kernel names are FULLY preserved. Module 0 exposes a fused Swin-Transformer
  attention backbone with explicit FP8 variants, e.g.:
  `cc_tinlayout_fused_swin_1h_32_1_{ds,upsample,inpview,outview,wait,tilesync,chained}[_fp8]`,
  `cc_tinlayout_fused_pre_block_swin_1h_32_1[_fp8]`,
  `cc_tinlayout_fused_post_block_swin_1h_32[_fp8]_{control_mask,simple_blend}[_full_rect]`.
- Per-kernel parameter metadata EXISTS: one `.nv.info.<kernel>` section per kernel
  (sizes 0x1e4–0x354 B each), plus `.nv.compat`, `.nv.callgraph`, `.note.nv.tkinfo`.
- The FP16 and FP8 variants are sibling kernels of the same op — Tensor-Core-class
  (FP8 e4m3/e5m2) instruction dependency is structural, not incidental.
- Extracted copies (metadata work, never committed): results/20260830_170305/
  carved_cubin_{0,1}.elf + .readobj.log; carve script: scripts/_carve_cubin.ps1.

## Checklist (the 18 questions from the experiment spec) — ANSWERED

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | PE imports | VERSION(3), ADVAPI32(3), USER32(1), KERNEL32(120); nothing else | probe_dlssnr_stdout.log |
| 2 | Direct nvcuda.dll import? | **NO** | binary_manifest_dlssnr.json |
| 3 | Loads nvapi64.dll? | Not statically imported; during our run no nvapi64 load occurred before the vendor-gate failure. Post-gate behavior still to be traced. | forcload_*_module_trace.log |
| 4 | nvapi_QueryInterface dispatch? | Not observed yet — init dies at vendor gate (0xbad00001) before any nvapi call. Trace pending gate passage. | nr_host_forceload_stdout.log |
| 5 | .nv_fatbin / CUDA payload? | No fatbin container magic; payload = 15 raw CUBIN ELFs inside .rsrc | probe manifest markers |
| 6 | GPU module count | **15** (NR); 670 in the SR runtime | binary_manifest_*.json |
| 7 | Target SM | **sm_120 only** (15 markers = 15 modules) | sm_targets_scan.log |
| 8 | fatbin / PTX / CUBIN / mixed | **Pure CUBIN (SASS)** for NR; SR runtime is CUBIN+PTX mixed | binary_probe scans |
| 9 | PTX fallback? | **NO** — zero PTX markers in the NR DLL | binary_probe PTX scan |
| 10 | PTX version | n/a for NR (absent); SR runtime ships PTX 8.7 | probe logs |
| 11 | Blackwell-only SASS? | **YES** — sm_120 exclusively, no sm_89/sm_90 fallback | sm_targets_scan.log |
| 12 | Ada patch payload class | Consistent with community evidence: RTX-40 patch = CUDA binary (CUBIN) swap | RESEARCH.md + this analysis |
| 13 | Tensor Core / FP8 dependency | **YES** — explicit `_fp8` kernel variants throughout | carved_cubin_0.readobj.log |
| 14 | Kernel names preserved? | **YES, fully** (see list above) | carved_cubin_*.readobj.log |
| 15 | Kernel parameter metadata? | **YES** — `.nv.info.<kernel>` sections per kernel | carved_cubin_*.readobj.log |
| 16 | Where are weights? | .data high-entropy runs (6 × 0.5–1.1 MB) + bulk of the 147 MB .rsrc | probe entropy report |
| 17 | Weights vs code separation | Separable: code = 15 ELFs; weights/constants = entropy regions (exact split pending dynamic trace) | probe manifest |
| 18 | Compression / encryption | PE + ELF parse cleanly (no code-layer encryption); entropy runs look like dense/compressed weight data, not a wrapper | probe entropy report |

## Go/no-go consequence (Phase 7 GATE-0/1 verdict)

- GATE-0 PASS: payload containers located (15 CUBIN ELFs in .rsrc).
- GATE-1 **FAIL for the PTX route**: no PTX anywhere in nvngx_dlssnr.dll →
  **PTX path = BLOCKED** per spec section 12. No ZLUDA JIT of NR kernels is possible.
- However: the payload is a CUBIN *module set with preserved names and metadata*, and
  the RTX-40 Ada patch demonstrates NVIDIA's own route is CUBIN-binary swap. The viable
  replacement point is therefore the module-load/launch API surface (likely the NVAPI
  CuModule/Cubin family — to be confirmed by trace after the vendor gate), substituting
  HSACO for CUBIN, i.e. CASE A machinery + Track-B-style kernel provision.

## Methodology requirements (met)

- Read-only on the original files; all analysis reproducible via scripts/_probe_now.ps1,
  _carve_cubin.ps1, _readobj_cubin.ps1, _scan_sm_targets.ps1.
- CUDA-binary-aware tooling used: PE parser + ELF payload enumeration + llvm-objdump on
  carved CUBINs — NOT just ASCII grep (the PTX-absence conclusion rests on binary_probe's
  container/marker scan across all 165 MB plus ELF inspection of both large modules).
- Machine-readable manifests: results/20260830_170305/binary_manifest_{dlssnr,dlss}.json
  (metadata only).
