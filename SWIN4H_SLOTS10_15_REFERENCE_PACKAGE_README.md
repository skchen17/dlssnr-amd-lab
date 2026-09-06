# RTX 5070 Swin 4h slots 10-15 reference

Run this package once on the Windows RTX 5070 host from an ordinary PowerShell
prompt. No CUDA toolkit or Python installation is required.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_nvidia_swin4h_slots10_15_reference.ps1 -PackageRoot .
```

Return the generated `_swin4h_slots10_15_reference_result_*.zip` file. The
runner checks all payload hashes, executes the original slot 10-15 PTX entries
over the exact AMD inputs, weights and captured parameters, records outputs and
sync buffers, and packages only generated experimental results.
