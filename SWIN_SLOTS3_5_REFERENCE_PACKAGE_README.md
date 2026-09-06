# RTX Swin slots 3-5 numerical reference

This package runs three independent original-PTX numerical references on an RTX
50-series machine. Each case uses the exact input, captured 64 KiB weight view,
and stable parameter block already executed on RX 9070 XT. The only PTX change
is the driver-compatibility declaration `.version 9.4` to `.version 8.7`.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_swin_slots3_5_reference.ps1
```

Return `_swin_slots3_5_reference_result_<timestamp>.zip`. A PASS means all three
kernels launched and returned their raw tensors; cross-vendor numerical parity
is evaluated in the lab after the archive is returned. It does not claim a full
156-slot frame.
