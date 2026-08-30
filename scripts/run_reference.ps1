# run_reference.ps1 — produce the NVIDIA-side reference output.
# Run this ON AN NVIDIA MACHINE (RTX; Blackwell for the leaked NR build) with legal
# nvngx DLLs. It never runs here on AMD — the reference must be real.
param(
    [int]$Frames = 1,
    [int]$Width = 512,
    [int]$Height = 512,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\nr_host.exe'
if (-not (Test-Path $exe)) { & (Join-Path $PSScriptRoot 'build_all.ps1') -Only nr_host }
if (-not $env:DLSS_DLL_PATH)  { Write-Host 'set DLSS_DLL_PATH to nvngx_dlss.dll first' -ForegroundColor Red; exit 2 }
if (-not $env:DLSSNR_DLL_PATH){ Write-Host 'set DLSSNR_DLL_PATH to nvngx_dlssnr.dll first' -ForegroundColor Red; exit 2 }
if (-not $Output) { $Output = Join-Path $repo ("results\reference_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".bin") }
& $exe --frames $Frames --width $Width --height $Height --output $Output --json ($Output + '.json')
Write-Host "exit=$LASTEXITCODE output=$Output"
exit $LASTEXITCODE
