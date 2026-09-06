# N0 CTA `(1,0)` MMA-176 dual activation-path trace

Run once on the RTX host:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_cta1_mma176_dual_path_trace.ps1
```

Return `_n0_cta1_mma176_dual_path_trace_reference_result_*.zip`. The run traces
both A3 and B0 producer paths in one full-grid experiment and compares its final
output against an untouched baseline.
