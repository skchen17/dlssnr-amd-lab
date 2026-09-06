# N0 full-grid selected-CTA FP8 MMA reference

This package runs the complete N0 `80x48` grid and records all 256 FP8 MMA
A/B/C/D fragments only for the CTA listed in `manifest.json`. Trace storage uses
an explicitly extended scratch allocation.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_fp8_mma_cta_trace.ps1
```

Return `_n0_cta*_fp8_mma_trace_reference_result_*.zip`. The return archive is
compact; the larger `_work_` directory can remain on the cloud host until the
result has been accepted.
