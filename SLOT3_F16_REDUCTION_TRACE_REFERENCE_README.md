# RTX 50 slot-3 packed-FP16 reduction trace

This package captures CTA `(2,0)` registers across the square, pair-add,
`shfl.bfly`, half-swap and final-add reduction that feeds the first divergent
normalization input.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_slot3_f16_reduction_trace.ps1
```

Return `_slot3_f16_reduction_trace_reference_result_*.zip`.

