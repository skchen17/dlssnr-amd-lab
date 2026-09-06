# RTX 50 slot-3 approximate-rsqrt trace

This package runs one diagnostic launch on an RTX 50-series Windows machine.
For all 16 paired `rsqrt.approx.ftz.f32` blocks in CTA `(2,0)`, it captures the
packed-FP16 input, both FP32 approximate results, and the packed-FP16 output.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_slot3_rsqrt_trace.ps1
```

Return `_slot3_rsqrt_trace_reference_result_*.zip`.

