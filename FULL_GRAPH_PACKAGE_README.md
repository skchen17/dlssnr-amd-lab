# DLSSNR full-graph RTX reference package v20

This package replaces the per-family RTX reference packages. It runs the proven
Feature-18 path once, captures bounded pre/post buffer windows for buffer-backed
slots 0-154, preserves the descriptor/texture final-copy evidence for slot 155,
then content-addresses duplicate raw tensors and creates one result ZIP. These
two evidence paths together cover the complete 156-slot graph.

Requirements:

- Windows RTX 50-series host with the same working driver/environment as the
  earlier Feature-18 experiments.
- At least 30 GB free disk space; 50 GB or more is recommended.
- Do not redistribute the package because it contains user-supplied NVIDIA and
  ReShade/RenoDX binaries.

Run from an ordinary PowerShell prompt in the extracted package directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_full_cloud_workflow.ps1
```

The run can take from tens of minutes to several hours depending on disk and
compression speed. Return the single file printed at the end:

```text
return_to_lab\_full_graph_reference_result_*.zip
```

Return that ZIP even if the script reports `PARTIAL`; it contains the logs and
integrity metadata needed to diagnose the missing capture without another blind
rerun. The default per-window bound is 8 MiB. Do not increase it unless asked,
because result size and readback memory grow quickly.
