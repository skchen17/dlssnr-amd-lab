# RTX 50 N0 pre-MMA checkpoint package

This package executes only the captured N0 prefix through the point immediately
before its first f16 MMA. One CTA exports all 32 lanes of the 16 input-feature A
registers and eight weight B registers (96 bytes per lane, 3,072 bytes total).
The checkpoint uses the same input texture, weights and scalar parameters as the
successful full RTX/RX 9070 XT comparison.

Requirements: Windows, RTX 50-series GPU and NVIDIA driver.

From PowerShell in the extracted package directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_checkpoint_reference.ps1
```

Return `_n0_checkpoint_reference_result_*.zip`. This is private user-owned
interoperability material; do not redistribute it.
