# Post-block exact-state E4M3/movmatrix trace (RTX)

This package captures every E4M3 conversion and movmatrix input/output at
post-block CTA `(70,26,0)`. It determines whether the FP8 MMA input mismatch is
created by quantization or fragment transposition.

Run from an elevated PowerShell prompt in the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\run_nvidia_postblock_mma_trace.ps1
```

Return `_postblock_e4m3_mov_trace_reference_result_*.zip`. Keep the matching
work directory until the result has been accepted.

