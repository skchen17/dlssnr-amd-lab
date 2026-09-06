# N0 r2329 reduction-path trace

This package records every visible packed-FP16 node from four FP8 MMA outputs
through the square/reduction tree that first differs between RTX and RX.
It also compares the instrumented output against untouched PTX.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_reduction_path_trace.ps1
```

Return `_n0_reduction_path_trace_reference_result_*.zip` without repacking it.
