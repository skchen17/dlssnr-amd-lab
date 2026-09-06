# N0 full-grid NVIDIA `lg2.approx` curve dataset

## Run command

Open Windows PowerShell in the extracted package directory and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_lg2_full_grid_trace.ps1
```

The script captures 245,760 real Box-Muller `lg2.approx.ftz.f32` input/output pairs from the complete `80x48` launch. Return the generated file matching:

```text
_n0_lg2_full_grid_trace_reference_result_*.zip
```

The captured values are immediate post-definition diagnostics used to reconstruct the NVIDIA SFU approximation curve; this package does not count as the AMD S7 end-to-end gate.

## If the run fails

The script now prints the original probe/JIT error before stopping. To display the latest probe log again, run this from the extracted package directory:

```powershell
$log = Get-ChildItem -Recurse -Filter stdout.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content -Raw $log.FullName
```
