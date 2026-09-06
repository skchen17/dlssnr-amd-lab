# N0 normal2 full-grid reference traces

This package runs an untouched baseline plus two controlled full-grid traces:

- the second Box-Muller phase input and rounded cosine output;
- `sqrt1` and the consumed `normal2_f16` boundary.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_normal2_full_grid_traces.ps1
```

Return `_n0_normal2_full_grid_traces_reference_result_*.zip`. Keep the `_work_`
directory until the result is accepted.
