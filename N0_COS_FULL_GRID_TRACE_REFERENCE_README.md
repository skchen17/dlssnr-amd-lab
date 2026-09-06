# N0 cosine full-grid reference trace

This package captures the input and rounded output of the first Box-Muller
`cos.approx.ftz.f32` for all 245,760 N0 samples. The runner also executes the
untouched baseline and accepts the trace only if the network output is preserved.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_cos_full_grid_trace.ps1
```

Return `_n0_cos_full_grid_trace_reference_result_*.zip`.
