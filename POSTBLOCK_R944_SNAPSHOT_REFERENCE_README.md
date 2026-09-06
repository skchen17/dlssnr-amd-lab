# Post-block non-producer-site r944 dependency snapshot (RTX)

This package avoids inserting stores next to packed-F16 producer instructions.
At the already validated E4M3 conversion-4 boundary, it snapshots the inputs and
outputs of the two multiplies and add that produced `%r944`: `%r204`, `%r205`,
`%r720`, `%r721`, `%r816`, `%r817`, `%r944`, and the E4M3 result `%rs53`.

Run from an elevated PowerShell prompt in the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\run_nvidia_postblock_mma_trace.ps1
```

Return `_postblock_r944_snapshot_reference_result_*.zip`.
