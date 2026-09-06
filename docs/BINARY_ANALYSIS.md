# BINARY_ANALYSIS.md — nvngx_dlssnr.dll static analysis

**State: ANALYZED (read-only), corrected 2026-08-31 after runtime-container
decompression.** Files legally supplied by the user
(see docs/PROPRIETARY_FILES.md for ledger: version / SHA-256 / Authenticode).
Binaries stay outside the repo; only metadata is recorded here.

## Key facts

| Fact | nvngx_dlssnr.dll (NR) | nvngx_dlss.dll (SR runtime) |
|---|---|---|
| Size | 165,840,496 B | 58,977,904 B |
| FileVersion | 310.8.0.0 | 310.7.0.0 |
| Payload location | `.rsrc` raw size 147,696,792 B | `.data` |
| CUDA modules | **15 hybrid runtime containers (`0xBA55ED50`), each carrying compressed PTX plus an sm_120 CUBIN/ELF representation** | 670 ELF payloads |
| Target arch | **sm_120 (consumer Blackwell, RTX 50) — 15 markers, exactly 1 per module** | sm_89 (Ada, 163 markers) + sm_80 (6 markers) |
| PTX present | **YES — 15/15 Zstd payloads decode to PTX 9.4, target sm_120; 231 entries total** | YES — `.target sm_89` ×16, `.version 8.7` ×8 |
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
- The original whole-file ASCII scan missed PTX because it is Zstd-compressed
  inside the `0xBA55ED50` runtime containers. `scripts/extract_runtime_modules.py`
  locates the Zstd frame, decompresses it without changing the proprietary input,
  and records a metadata-only manifest. All 15 containers decode successfully.

## Checklist (the 18 questions from the experiment spec) — ANSWERED

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | PE imports | VERSION(3), ADVAPI32(3), USER32(1), KERNEL32(120); nothing else | probe_dlssnr_stdout.log |
| 2 | Direct nvcuda.dll import? | **NO** | binary_manifest_dlssnr.json |
| 3 | Loads nvapi64.dll? | Not statically imported; during our run no nvapi64 load occurred before the vendor-gate failure. Post-gate behavior still to be traced. | forcload_*_module_trace.log |
| 4 | nvapi_QueryInterface dispatch? | Not observed yet — init dies at vendor gate (0xbad00001) before any nvapi call. Trace pending gate passage. | nr_host_forceload_stdout.log |
| 5 | .nv_fatbin / CUDA payload? | 15 NVIDIA runtime containers (`0xBA55ED50`) in `.rsrc`; each has compressed PTX and an sm_120 CUBIN/ELF representation | extraction manifest + ELF markers |
| 6 | GPU module count | **15** (NR); 670 in the SR runtime | binary_manifest_*.json |
| 7 | Target SM | **sm_120 only** (15 markers = 15 modules) | sm_targets_scan.log |
| 8 | fatbin / PTX / CUBIN / mixed | **Mixed compressed PTX + CUBIN** for NR | extraction manifest + ELF scans |
| 9 | PTX fallback? | **YES as a representation** — PTX is present in all 15 containers; translator compatibility is instruction-dependent | extraction manifest + ZLUDA probes |
| 10 | PTX version | **9.4**, target sm_120; SR runtime separately ships PTX 8.7 | extraction manifests |
| 11 | Blackwell-only SASS? | CUBIN is sm_120-only, but it is no longer the only available representation because PTX 9.4 is present | sm_targets_scan.log + extraction manifest |
| 12 | Ada patch payload class | Consistent with community evidence: RTX-40 patch = CUDA binary (CUBIN) swap | RESEARCH.md + this analysis |
| 13 | Tensor Core / FP8 dependency | **YES** — explicit `_fp8` kernel variants throughout | carved_cubin_0.readobj.log |
| 14 | Kernel names preserved? | **YES, fully** (see list above) | carved_cubin_*.readobj.log |
| 15 | Kernel parameter metadata? | **YES** — `.nv.info.<kernel>` sections per kernel | carved_cubin_*.readobj.log |
| 16 | Where are weights? | .data high-entropy runs (6 × 0.5–1.1 MB) + bulk of the 147 MB .rsrc | probe entropy report |
| 17 | Weights vs code separation | Separable: code = 15 ELFs; weights/constants = entropy regions (exact split pending dynamic trace) | probe manifest |
| 18 | Compression / encryption | PE + ELF parse cleanly (no code-layer encryption); entropy runs look like dense/compressed weight data, not a wrapper | probe entropy report |

## Go/no-go consequence (corrected GATE verdict)

- GATE-0 PASS: 15 runtime containers located in `.rsrc`.
- GATE-1 **PASS**: all 15 expose compressed PTX 9.4 / sm_120, totaling 231
  entries. The previous FAIL was a false negative caused by scanning compressed
  data for plaintext markers.
- GATE-2 is **PARTIAL** on RX 9070 XT with ZLUDA v7-preview.3. Function-isolated
  `cc_cb_clear` compiles and executes byte-exactly (27,648 words changed to
  `0xffffffff`, zero mismatches). `cg2r_copy_kernel` is rejected at
  `sust.p.2d.v4.b32.zero`; the first neural entry is rejected at PTX constructs
  including tuple-discard `mov`, `mma.sync`, and FP8 conversions.
- The viable implementation route is now hybrid: reuse function-isolated PTX
  where the translator accepts it; lower unsupported texture/tensor instructions
  to HIP/LLVM equivalents; retain clean-room HIP reconstruction as the fallback
  per function rather than the default for all 43 used kernels.

## Methodology requirements (met)

- Read-only on the original files; all analysis reproducible via scripts/_probe_now.ps1,
  _carve_cubin.ps1, _readobj_cubin.ps1, _scan_sm_targets.ps1.
- CUDA-binary-aware tooling used: PE parser, runtime-container/Zstd extraction,
  ELF payload enumeration, llvm-objdump on carved CUBINs, and live function-level
  `cuModuleLoadData`/`cuModuleGetFunction`/launch probes through ZLUDA.
- Machine-readable manifests: results/20260830_170305/binary_manifest_{dlssnr,dlss}.json
  plus `results/20260831_010100_all_runtime_modules/extraction_manifest.json`
  and `results/20260831_011219_zluda_ptx_probe/manifest.json` (metadata only;
  extracted proprietary PTX remains ignored).
