# RTX Swin 2-head slots 6-9 numerical reference

This package runs the original isolated NVIDIA PTX for graph slots 6-9 over the
exact input tensors, captured parameters and extended weight views already used
by the RX 9070 XT runs. Only the PTX declaration is normalized from 9.4 to 8.7;
the package manifest binds the original and payload hashes.

Run on the RTX 5070 machine from the extracted directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_swin2h_slots6_9_reference.ps1
```

Return `_swin2h_slots6_9_reference_result_<timestamp>.zip`. Slots 6-8 verify
their exact publication counts; slot 9 has no release output and returns its
additional downsample tensor.
