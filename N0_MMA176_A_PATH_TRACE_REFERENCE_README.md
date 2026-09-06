# N0 selected-CTA MMA-176 A normalization trace

Open Windows PowerShell in the extracted package directory on the RTX host and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_mma176_a_path_trace.ps1
```

The run launches the full `80x48` N0 grid and records both complete packed-FP16 square-reduction paths that produce the first two differing MMA-176 A-fragment bytes in the CTA listed in `manifest.json`. It also runs an untouched baseline and requires identical final output.

Return `_n0_ctaX_Y_mma176_a_path_trace_reference_result_*.zip` unchanged. The larger `_work_` directory can remain on the RTX host.
