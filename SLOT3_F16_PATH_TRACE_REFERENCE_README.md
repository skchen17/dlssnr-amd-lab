# RTX 50 slot-3 packed-FP16 path trace

This package runs one diagnostic launch on an RTX 50-series Windows machine.
It captures 28 registers per lane in CTA `(2,0)` along the first activation path
known to differ before FP8 MMA 216: first-stage packed-FP16 inputs/scales/products,
the second-stage scale, and the products immediately before E4M3 conversion.

Run from a 64-bit PowerShell prompt:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_slot3_f16_path_trace.ps1
```

Return the generated `_slot3_f16_path_trace_reference_result_*.zip`.

