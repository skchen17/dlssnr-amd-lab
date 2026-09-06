# run_reference.ps1 — Stage A launcher: vanilla DLSS/DLAA reference run.
#
# Stage A (Round 2): only nvngx_dlss.dll, official loader ABI, public feature
# (SuperSampling / DLAA contract). Prove that evaluate produces valid GPU output
# and store it as the reference. No NR DLL, no addon, no feature 18 here.
#
# Run this ON AN NVIDIA MACHINE (RTX) with a legally obtained nvngx_dlss.dll.
# On this AMD lab box it exits BLOCKED_EXTERNAL_HARDWARE.
param(
    [int]$Frames = 8,
    [int]$Width = 512,
    [int]$Height = 512,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\nr_host.exe'
if (-not (Test-Path $exe)) { & (Join-Path $PSScriptRoot 'build_all.ps1') -Only nr_host }

if (-not $env:DLSS_DLL_PATH) { Write-Host 'set DLSS_DLL_PATH to nvngx_dlss.dll first' -ForegroundColor Red; exit 2 }
# Stage A must run WITHOUT the NR runtime in the loop
$env:DLSSNR_DLL_PATH = $null

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $Output) { $Output = Join-Path $repo "results\${ts}_stage_a\output_rgba8.bin" }
New-Item -ItemType Directory -Force -Path (Split-Path $Output) | Out-Null

Push-Location $repo
try {
    Write-Host '=== Stage A: vanilla DLSS/DLAA evaluate (official ABI) ==='
    & $exe --frames $Frames --width $Width --height $Height --output $Output `
           --json ($Output + '.json')
} finally { Pop-Location }
$code = $LASTEXITCODE
Write-Host "stage_a exit=$code output=$Output"
exit $code
