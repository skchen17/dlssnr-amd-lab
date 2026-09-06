# N0 normal0 boundary full-grid reference trace

This package captures `sqrt0` and the consumed `normal0_f16` value for every
one of the 245,760 N0 samples. The runner executes an untouched baseline and
accepts the trace only when instrumentation preserves the final network output.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_normal0_boundary_full_grid_trace.ps1
```

Return `_n0_normal0_boundary_full_grid_trace_reference_result_*.zip`. Keep the
`_work_` directory until the result is accepted.
