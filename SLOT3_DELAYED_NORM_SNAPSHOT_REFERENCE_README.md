# Slot-3 delayed normalization snapshot (RTX)

This package snapshots the packed-F16 square/reduction tree, clamp inputs and
rsqrt outputs that feed the first known slot-3 divergence. All trace stores are
delayed until the previously output-preserving E4M3 boundary; no stores are
inserted beside the producer instructions. Two independent RTX runs must retain
the original output exactly and repeat the trace bit-for-bit.

Run from an elevated PowerShell prompt in the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\run_nvidia_slot3_delayed_norm_snapshot.ps1
```

Return `_slot3_delayed_norm_snapshot_reference_result_*.zip`. Do not return the
large extracted work directory.
