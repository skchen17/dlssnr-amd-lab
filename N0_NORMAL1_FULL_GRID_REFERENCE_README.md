# N0 normal1 full-grid reference traces

This package runs an untouched baseline plus two controlled full-grid traces:

- the first Box-Muller phase input and rounded sine output;
- `sqrt0` and the consumed `normal1_f16` boundary.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\run_nvidia_n0_normal1_full_grid_traces.ps1
```

Return `_n0_normal1_full_grid_traces_reference_result_*.zip`. Keep the `_work_`
directory until the result is accepted.
