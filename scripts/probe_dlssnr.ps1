# probe_dlssnr.ps1 — run binary_probe over every proprietary runtime the user
# has provided via environment variables. Never downloads anything.
#
#   DLSSNR_DLL_PATH          nvngx_dlssnr.dll (extracted from a legally owned game)
#   DLSS_DLL_PATH            nvngx_dlss.dll   (same game / NVIDIA DLSS update)
#   RENODX_DLSS5_ADDON_PATH  RenoDX DLSS5 addon dll (open-source release artifact)
#
# Output: results/<timestamp>/binary_manifest_<name>.json + console summary.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\binary_probe.exe'
if (-not (Test-Path $exe)) {
    Write-Host "binary_probe.exe not found; building..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'build_all.ps1') -Only binary_probe
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$outDir = Join-Path $repo "results\$ts"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$targets = [ordered]@{
    'dlssnr'       = $env:DLSSNR_DLL_PATH
    'dlss'         = $env:DLSS_DLL_PATH
    'renodx_addon' = $env:RENODX_DLSS5_ADDON_PATH
}

$any = $false
foreach ($k in $targets.Keys) {
    $p = $targets[$k]
    $summary = Join-Path $outDir "probe_$k.txt"
    if (-not $p) {
        Write-Host "[$k] MISSING_PREREQUISITE (env var not set)" -ForegroundColor Yellow
        "MISSING_PREREQUISITE" | Out-File -Encoding utf8 $summary
        continue
    }
    if (-not (Test-Path $p)) {
        Write-Host "[$k] path does not exist: $p" -ForegroundColor Red
        "PATH_NOT_FOUND: $p" | Out-File -Encoding utf8 $summary
        continue
    }
    $any = $true
    $json = Join-Path $outDir "binary_manifest_$k.json"
    Write-Host "`n[$k] probing $p" -ForegroundColor Cyan
    & $exe $p --json $json | Tee-Object -FilePath $summary
    if ($LASTEXITCODE -ne 0) { Write-Host "[$k] binary_probe exit=$LASTEXITCODE" -ForegroundColor Red }
}

if (-not $any) {
    Write-Host "`nNo proprietary files available. Static tool is built and ready;" -ForegroundColor Yellow
    Write-Host "all runtime-dependent phases remain MISSING_PREREQUISITE (see docs/BLOCKERS.md)."
}
Write-Host "`nOutputs: $outDir"
