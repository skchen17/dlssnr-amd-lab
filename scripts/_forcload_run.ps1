# AMD-side first real load of the NVIDIA DLLs under trace shims.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repo 'results\20260830_170305'

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only nr_host 2>&1 | Out-String | Set-Content (Join-Path $runDir 'nr_host_build2.log')
Write-Host "build exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit 1 }

$env:DLSS_DLL_PATH   = 'C:\DATA\Python_File\Qoder_workspace\file\nvngx_dlss.dll'
$env:DLSSNR_DLL_PATH = 'C:\DATA\Python_File\Qoder_workspace\file\nvngx_dlssnr.dll'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$env:MODULE_TRACE_LOG = Join-Path $runDir "forcload_$ts`_module_trace.log"
$env:NVAPI_TRACE_LOG  = Join-Path $runDir "forcload_$ts`_nvapi_trace.jsonl"

Push-Location $repo
$out = & (Join-Path $repo 'build\nr_host.exe') --frames 1 --width 512 --height 512 --trace --force-load --json (Join-Path $runDir 'nr_host_forceload.json') 2>&1 | Out-String
Set-Content (Join-Path $runDir 'nr_host_forceload_stdout.log') $out
Write-Host "run exit=$LASTEXITCODE"
Pop-Location
Write-Host "module log: $env:MODULE_TRACE_LOG"
Write-Host "nvapi  log: $env:NVAPI_TRACE_LOG"
