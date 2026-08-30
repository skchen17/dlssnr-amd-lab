# Run binary_probe on the two legally supplied NVIDIA DLLs (read-only).
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repo 'results\20260830_170305'
Push-Location $repo
$out = & (Join-Path $repo 'build\binary_probe.exe') '..\file\nvngx_dlssnr.dll' --json (Join-Path $runDir 'binary_manifest_dlssnr.json') 2>&1 | Out-String
Set-Content (Join-Path $runDir 'probe_dlssnr_stdout.log') $out
Write-Host "dlssnr probe exit=$LASTEXITCODE"
$out2 = & (Join-Path $repo 'build\binary_probe.exe') '..\file\nvngx_dlss.dll' --json (Join-Path $runDir 'binary_manifest_dlss.json') 2>&1 | Out-String
Set-Content (Join-Path $runDir 'probe_dlss_stdout.log') $out2
Write-Host "dlss probe exit=$LASTEXITCODE"
Pop-Location
