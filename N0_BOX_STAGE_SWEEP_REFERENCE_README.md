# N0 Box-Muller clean stage sweep

This package runs the untouched baseline and independent `sqrt0` and `cos0`
single-observation variants. The target CTA is recorded in `manifest.json`.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_box_stage_sweep.ps1
```

Return `_n0_cta*_box_stage_sweep_reference_result_*.zip`.
