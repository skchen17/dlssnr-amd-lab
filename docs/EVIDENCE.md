# EVIDENCE.md

Every "success" claim must link to a concrete artifact: command, log path, hash.

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

## E-5 NR payload is 15 sm_120 pure-SASS CUBIN modules with no PTX (S3 static)
- Date: 2026-08-30
- S-level touched: S3
- Command: `scripts\_probe_now.ps1`; `scripts\_carve_cubin.ps1`; `scripts\_readobj_cubin.ps1`; `scripts\_scan_sm_targets.ps1`
- Artifact(s): results/20260830_170305/binary_manifest_dlssnr.json (15 ELF markers), sm_targets_scan.log (15× sm_120), carved_cubin_0.readobj.log (kernel names + .nv.info), probe_dlssnr_stdout.log (zero PTX markers)
- Adapter proof: n/a (static analysis)
- Verdict: PASS

## E-6 unmodified DLSSNR runtime loads on AMD and runs a real vendor check (first dynamic contact)
- Date: 2026-08-30
- S-level touched: S5 attempt (blocked at gate)
- Command: `scripts\_forcload_run.ps1`
- Artifact(s): results/20260830_170305/nr_host_forceload_stdout.log (load OK, exports OK, init=0xbad00001), forcload_20260830_182045_module_trace.log (no NVIDIA support modules loaded)
- Adapter proof: D3D12 device created on vendor=0x1002 device=0x7550 (printed by nr_host)
- Verdict: PARTIAL — vendor gate is real; NOT counted as any execution success

## Forbidden "evidence" (never accepted)

- "DLSS 5 menu appeared"
- "Game launched"
- "FakeNVAPI passed vendor check"
- "NGX returned success"
- "Feature handle non-null"
- "ReShade overlay shows DLSS 5"
- "Output looks prettier"
- "FSR4 was called through DLSS API"

Real success requires GPU execution evidence on the AMD adapter (vendor 0x1002).
