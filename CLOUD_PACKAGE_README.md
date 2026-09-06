# Windows cloud experiment package

Package revision: **v3 / Round 3**. This build adds the render-subrect dimensions
required by the DLSS Evaluate contract after the v2 RTX 5060 run reached
CreateFeature but returned `FAIL_InvalidParameter` on every frame.

## Requirements

- Windows 10/11 or Windows Server with a visible NVIDIA GPU and current NVIDIA driver.
- A working D3D12 display adapter. A compute-only/TCC cloud GPU may fail this gate.
- Microsoft Visual C++ 2015–2022 x64 runtime if the packaged executables do not start.
- Legally obtained local copies of `nvngx_dlss.dll` and, optionally,
  `nvngx_dlssnr.dll` and the RenoDX DLSS5 add-on.

The package deliberately contains none of those proprietary runtime files.

## Run

Open PowerShell in the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_cloud_experiment.ps1 `
  -DlssDll "D:\runtime\nvngx_dlss.dll" `
  -DlssNrDll "D:\runtime\nvngx_dlssnr.dll" `
  -RenoDxAddon "D:\runtime\renodx-dlss5.addon64" `
  -Frames 8 -Width 512 -Height 512
```

Only `-DlssDll` is required. If it is omitted, the script asks for it.

When the run finishes, send back:

```text
return_to_lab\dlssnr_cloud_result_<timestamp>.zip
```

Do not add the proprietary DLLs to that ZIP. The runner records their filename,
version, signature status, size, and SHA-256 without copying their contents.

## Scope

The package can validate the public NGX ABI, vanilla DLSS/DLAA, tracer readiness,
DLSSNR binary metadata, and observational DLSSNR loading. It does **not** host the
RenoDX ReShade add-on, so a successful Stage B run is not evidence that private
feature 18 or DLSS 5 Neural Rendering executed.
