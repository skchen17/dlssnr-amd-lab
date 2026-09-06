# Post-block exact-state MMA trace (RTX)

This package captures every FP8 and F16 MMA operand/result register for post-block
CTA `(70,26,0)` using the exact frame-1 slot-154 activation. It also checks that
instrumentation preserves the known-good complete RTX output.

Run from an elevated PowerShell prompt in the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\run_nvidia_postblock_mma_trace.ps1
```

Return the generated `_postblock_mma_trace_reference_result_*.zip`. Keep the
matching `_postblock_mma_trace_work_*` directory until the result is accepted.

