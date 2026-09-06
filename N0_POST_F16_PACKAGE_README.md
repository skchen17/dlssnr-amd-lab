# RTX 50 N0 post-f16-MMA checkpoint package

This package executes one CTA of the captured N0 prefix through its first 16
`mma.sync.m16n8k16.f16` operations. It exports the 32 D registers per lane
(128 bytes per lane, 4,096 bytes total) immediately after that matrix block.
Input, weights and parameters are identical to the bitwise-matched pre-MMA
checkpoint.

Run from the extracted package directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_post_f16_reference.ps1
```

Return `_n0_post_f16_reference_result_*.zip`. Keep this user-owned
interoperability material private.
