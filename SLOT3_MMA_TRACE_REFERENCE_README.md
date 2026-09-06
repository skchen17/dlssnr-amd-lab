# RTX 50 slot-3 FP8 MMA trace

This package runs one diagnostic launch on an RTX 50-series Windows machine. It
uses the exact slot-3 activation arena, model arena, parameter block, grid and
block captured from the full graph. The instrumented original PTX exports the
A/B/C/D register fragments for all 256 FP8 MMA instructions in the selected CTA
recorded by `manifest.json`.

Run from an elevated or normal 64-bit PowerShell prompt:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_slot3_mma_trace.ps1
```

Return the generated `_slot3_mma_trace_reference_result_*.zip`. This is a
single-run internal numerical oracle; no repeated cloud debugging is required.
