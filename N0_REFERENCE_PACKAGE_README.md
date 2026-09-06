# RTX 50 N0 same-input reference package

This package runs the captured SM120 N0 PTX over the exact synthetic RGBA16F
input, weight window and scalar parameter block already used by the RX 9070 XT
full-grid experiment. The module body is unchanged; only its `.version` directive
is normalized from 9.4 to 8.7 because the RTX 5070 driver rejected the newer text
header with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The package manifest records
both hashes and the single-directive transformation.

Requirements: Windows, an RTX 50-series GPU, and a current NVIDIA driver. Do not
run it on RTX 40: the original module targets SM120.

From PowerShell in the extracted package directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_synthetic_reference.ps1
```

Return the printed `_n0_full_reference_result_*.zip`. A successful run contains
`scratch.raw` (7,864,320 bytes), `output.raw` (1,966,080 bytes), `probe.json` and
hashes. The package contains interoperability test material from the user's own
runtime/capture; keep it private and do not redistribute it.
