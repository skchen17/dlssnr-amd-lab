# N0 normalization-path trace reference

This package runs one N0 CTA on an NVIDIA RTX GPU and records ten registers
along the normalization path feeding the first divergent FP8 MMA. It also runs
the untouched PTX and verifies that instrumentation preserves the output.

Run in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_norm_path_trace.ps1
```

Return `_n0_norm_path_trace_reference_result_*.zip` without renaming or repacking it.
