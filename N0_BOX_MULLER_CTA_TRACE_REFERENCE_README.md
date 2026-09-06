# N0 selected-CTA Box-Muller path trace

Open Windows PowerShell in the extracted package directory on the RTX 50-series host, then run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_box_muller_cta_trace.ps1
```

The script captures the immediate post-definition checkpoints listed in `manifest.json` for 64 dynamic positional-feature samples in the selected CTA. Depending on the package, these cover either one diagnostic output or all three Box-Muller outputs through their final FP16 conversions. It returns `_n0_ctaX_Y_box_muller_trace_reference_result_*.zip` beside the extracted package directory.

Return the ZIP unchanged. Stores are placed immediately after each defining instruction, so only those captured values are admissible; the downstream instrumented output is retained solely as a perturbation control. This diagnostic package does not count as the AMD S7 end-to-end gate.
