# N0 selected-CTA FP16 MMA reference trace

Run `scripts\run_nvidia_n0_f16_mma_cta_trace.ps1` in Windows PowerShell on the RTX 50-series host. The script performs one instrumented full-grid run plus one untouched control run, verifies that tracing did not change the output, and creates `_n0_cta8_0_f16_mma_trace_reference_result_*.zip` beside the extracted package.

Return that ZIP without modifying its contents. This is diagnostic evidence only and does not count as the AMD end-to-end S7 gate.
