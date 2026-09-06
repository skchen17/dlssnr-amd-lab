# N0 MMA-176 A-path single-store fusion sweep

Open Windows PowerShell in the extracted package directory on the RTX host and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nvidia_n0_mma176_a_fusion_sweep.ps1
```

The command runs one baseline and 45 small variants. Each variant inserts exactly one selected-CTA trace store after one path node, allowing the result to identify which single observation breaks NVIDIA's hidden fusion while keeping the whole sweep in one cloud-host command.

Return `_n0_ctaX_Y_mma176_a_fusion_sweep_reference_result_*.zip` unchanged. The `_work_` directory can remain on the RTX host.
