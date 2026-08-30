# STATUS.md

Current highest achieved S-level: **S3 ACHIEVED** (DLSSNR binary fully understood statically: 15 sm_120 pure-SASS CUBIN modules, no PTX, names+metadata preserved; runtime launch chain pending vendor gate)

| Level | Meaning | State | Evidence |
|---|---|---|---|
| S0 | environment recorded | **achieved** | results/20260830_170305/environment.json |
| S1 | AMD HIP + RDNA4 compute works | **achieved** | results/20260830_170305/hip_probe.json (overall_pass=true, 6/6 tests as designed; F opt-in) |
| S2 | D3D12/HIP interop works | **achieved** | results/20260830_170305/d3d12_hip_interop.json (pass_a=true), docs/INTEROP.md |
| S3 | DLSSNR binary/call graph understood | **achieved (static)** — binary fully mapped; runtime call chain stops at vendor gate | docs/BINARY_ANALYSIS.md, binary_manifest_dlssnr.json, docs/CALLGRAPH.md |
| S4 | reference NR host works on NVIDIA | blocked (no NVIDIA GPU on this machine) | reference package ready (results/20260830_*_reference_package) |
| S5 | AMD passes NGX/NVAPI init | blocked at vendor gate (0xbad00001 on 0x1002) | nr_host_forceload_stdout.log, BLOCKERS S-B5 |
| S6 | first DLSSNR-originated kernel on AMD | not started | launch log + adapter evidence |
| S7 | complete single NR frame on AMD | not started | output frame hash |
| S8 | output matches reference | not started | docs/RESULTS.md metrics |
| S9 | stable 1000-frame run | not started | docs/RESULTS.md |
| S10 | performance optimization | not started | docs/RESULTS.md |

## Log

- 2026-08-30: repo scaffolded; docs/RESEARCH.md written (13 topics verified online).
- 2026-08-30: S0 achieved — environment.json recorded (RX 9070 XT 0x1002/0x7550, HIP 7.2.0, FL 12.2, SM 6.8).
- 2026-08-30: S1 achieved — hip_probe A–E all PASS on gfx1201 (fp32 maxErr 5.96e-08, fp16 GEMM maxErr 3.26e-03, fp8 e4m3/e5m2 exact). WMMA f32 builtin does not compile in this ROCm 7.2 LLVM (S-B4); test F recorded UNVERIFIED, not counted.
- 2026-08-30: S2 achieved — d3d12_hip_interop PASS-A: D3D12Heap buffer + D3D12Fence imported into HIP; T3/T4 bidirectional transfers mismatch=0; texture import also works (T5). PASS-B (command-list fusion) = no HIP API, recorded as survey (T6).
- 2026-08-30: Phase 3/4 tools delivered — binary_probe self-tested (R-3); module_trace.dll + nvapi64.dll shim smoke-tested (smoke_trace.py SELFTEST OK).
- 2026-08-30: Phase 5 delivered — nr_host skeleton builds and runs; without proprietary DLLs it exits 3 BLOCKED_MISSING_PREREQUISITE (R-4), emits deterministic reference package (two runs, identical SHA-256). No fabricated execution.
- 2026-08-30: Phase 6/7 framework delivered — docs/CALLGRAPH.md now holds the complete CASE A/B/C/D decision tree and the GATE-0..4 PTX go/no-go chain; all evidence slots PENDING_PREREQUISITE (B-1/B-2).
- 2026-08-30 (evening): B-1/B-2 resolved — user supplied legally owned nvngx_dlss.dll (310.7) and nvngx_dlssnr.dll (310.8); ledgered with SHA-256/Authenticode (docs/PROPRIETARY_FILES.md).
- 2026-08-30 (evening): S3 achieved (static) — binary_probe + ELF carving + llvm-objdump + sm-target scan: NR payload = 15 pure-SASS CUBIN modules targeting sm_120 ONLY, zero PTX (GATE-1 FAIL → PTX path BLOCKED, Track B), kernel names and .nv.info metadata fully preserved (fused Swin-Transformer backbone with FP8 variants). No nvcuda/nvapi static imports.
- 2026-08-30 (evening): first REAL load on AMD — both DLLs load cleanly, all 5 reconstructed NGX exports resolve, NGX init executes and returns 0xbad00001 on vendor=0x1002 (real vendor gate; no crash, no spoofing). S5 now blocked at S-B5.
