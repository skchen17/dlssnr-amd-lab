# run_amd.ps1 — AMD-side experiment run: loads the same (unmodified) DLLs under
# the trace shims and the HIP/ZLUDA replacement layer. With no proprietary files
# yet, this script documents the intended invocation and exits blocked.
param([int]$Frames = 1, [int]$Width = 512, [int]$Height = 512)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\nr_host.exe'
if (-not (Test-Path $exe)) { & (Join-Path $PSScriptRoot 'build_all.ps1') -Only nr_host }

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$env:MODULE_TRACE_LOG = Join-Path $repo "results\$ts`_module_trace.log"
$env:NVAPI_TRACE_LOG  = Join-Path $repo "results\$ts`_nvapi_trace.jsonl"
Write-Host "shim logs: $env:MODULE_TRACE_LOG / $env:NVAPI_TRACE_LOG"
Write-Host "note: place build\nvapi64.dll where the target resolves nvapi64.dll,"
Write-Host "      and inject build\module_trace.dll before NGX init."

if (-not ($env:DLSS_DLL_PATH -and $env:DLSSNR_DLL_PATH)) {
    Write-Host "MISSING_PREREQUISITE: DLSS_DLL_PATH / DLSSNR_DLL_PATH not set." -ForegroundColor Yellow
    Write-Host "Experiments remain blocked; skeleton run only:"
    & $exe --frames $Frames --width $Width --height $Height --trace
    exit $LASTEXITCODE
}
& $exe --frames $Frames --width $Width --height $Height --trace --force-load --json (Join-Path $repo "results\$ts`_nr_host.json")
exit $LASTEXITCODE
