# N0 selected-CTA residual bundle

This package performs one baseline run plus two controlled RTX runs for the CTA
listed in `manifest.json`:

- three final FP16 Box-Muller values consumed by the network;
- all A/B/C/D fragments for all 256 FP8 MMAs.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_n0_cta_residual_bundle.ps1
```

Return `_n0_cta*_residual_bundle_reference_result_*.zip`. Keep the `_work_`
directory on the cloud machine until the result has been accepted.
