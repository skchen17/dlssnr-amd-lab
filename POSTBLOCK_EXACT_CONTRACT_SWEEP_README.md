# RTX slot154 exact frame-1 contract sweep

This package runs the original slot154 PTX plus the first three cumulative
compatibility transformations against one internally consistent RTX 5070 frame.
It uses the complete frame-1 activation buffer, the exact frame-1 parameter
block, the destination surface contents from immediately before slot154, and the
unperturbed v23 reference surface from immediately after slot154. The probe uses
a dedicated zero buffer for the texture at parameter offset 56; v1 accidentally
uploaded the initial destination surface into that texture and is superseded.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_postblock_exact_contract_sweep.ps1
```

Return `_postblock_exact_contract_reference_result_<timestamp>.zip`.

This is a diagnostic reference experiment, not S7. It contains no proprietary
NGX binaries. Success requires the original NVIDIA PTX output to match the
same-frame live output byte-for-byte and all four stages to repeat exactly.
