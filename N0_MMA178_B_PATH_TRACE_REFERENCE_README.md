# N0 zero-input MMA 178 B1 path reference

This one-run package records nine 32-lane register checkpoints on an NVIDIA RTX
GPU.  They bracket the exact activation path that first differs between RTX 5070
and RX 9070 XT at FP8 MMA 178 under the full graph's zero-RGBA16F input.

Run from an elevated or normal 64-bit PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_mma178_b_path_trace.ps1
```

Return the generated `_n0_mma178_b_path_trace_reference_result_*.zip` file.
The script runs both instrumented and untouched PTX and records whether tracing
changed the complete one-CTA output.  The result is diagnostic evidence and does
not by itself satisfy the final RX 9070 XT game-integration gate.
