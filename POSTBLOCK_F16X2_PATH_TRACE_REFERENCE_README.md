# Post-block low-perturbation packed-F16 path trace (RTX)

The earlier all-operation trace was deterministic but changed the RTX output,
so it is not a valid causal oracle. This replacement records only `%r816`,
`%r817`, and `%r944`: the two packed multiplies and packed add that directly
produce the first differing E4M3 input. Acceptance requires the complete output
to remain bitwise equal to the queue-complete reference.

Run from an elevated PowerShell prompt in the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\run_nvidia_postblock_mma_trace.ps1
```

Return `_postblock_f16x2_path_trace_reference_result_*.zip`.
