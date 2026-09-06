# Post-block pre-surface-store RTX reference

This package traces both full-grid output vectors immediately before the two
slot-154 formatted-surface stores. It runs the trace twice, checks deterministic
trace bytes, and verifies that the logical 640x360 RGBA16F output still matches
the captured RTX reference.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_postblock_store_trace.ps1
```

Return `_postblock_store_trace_reference_result_*.zip`. Keep the matching
`_postblock_store_trace_work_*` directory until the result has been accepted.
