# Post-block exact-state packed-F16 trace (RTX)

This package captures the inputs and output of all 1,522 packed-F16 arithmetic
operations at post-block CTA `(70,26,0)`. It locates the first divergence that
feeds the already observed E4M3 input mismatch.

Run from an elevated PowerShell prompt in the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\run_nvidia_postblock_mma_trace.ps1
```

Return `_postblock_f16x2_trace_reference_result_*.zip`. Keep the matching work
directory until the result has been accepted.
