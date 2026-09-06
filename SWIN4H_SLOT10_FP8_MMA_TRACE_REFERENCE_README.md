# Swin 4h slot-10 FP8 MMA reference experiment

This package traces all 272 FP8 MMA instructions for CTA `(4,0)`, warp-y `1`
in the original NVIDIA slot-10 kernel. It uses the already captured exact RTX
slot-10 input, weights, parameters, output initialization and synchronization
state. The runner also executes the uninstrumented PTX as a control.

The result is accepted only when the uninstrumented run and both instrumented
runs execute on an NVIDIA GPU, reproduce the captured output and sync hashes,
and the two 348,160-byte traces are bitwise identical. This is a numerical
oracle and does not count as an AMD full-frame or game-integration result.

Run from the extracted package directory in Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_swin4h_slot10_fp8_mma_trace.ps1
```

Return the generated file named:

```text
_swin4h_slot10_fp8_mma_trace_reference_result_YYYYMMDD_HHMMSS.zip
```
