# RTX N1 slot-2 reference package

This package runs the original isolated DLSSNR slot-2 PTX on an RTX 50-series
GPU over the exact input tensor, weight window and 96-byte launch block already
used by the RX 9070 XT experiment.

Run from an ordinary PowerShell window:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n1_slot2_reference.ps1
```

Return the generated `_n1_slot2_reference_result_*.zip`. The package does not
contain NVIDIA DLLs or model binaries; `nvcuda.dll` is loaded from the installed
driver. The PTX declaration is normalized from 9.4 to 8.7 for driver acceptance;
the manifest records both hashes and no instruction/body change is made.
