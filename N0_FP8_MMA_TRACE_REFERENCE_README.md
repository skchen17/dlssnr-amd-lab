# N0 FP8 MMA trace reference

This package runs one original-PTX N0 CTA on an NVIDIA RTX GPU and exports the
A/B/C/D fragments around all 256 FP8 MMA instructions. It also runs the
uninstrumented original PTX once and records whether instrumentation preserved
the complete one-CTA output. It is diagnostic and does not count as S7.

Run from an elevated or ordinary PowerShell prompt on the RTX machine:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_fp8_mma_trace.ps1
```

Return the generated `_n0_fp8_mma_trace_reference_result_*.zip`. Do not rename,
edit, or extract/repack the result archive.
