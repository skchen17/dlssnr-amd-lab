# Determinism check: second nr_host run must produce byte-identical inputs.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repo 'results\20260830_170305'
Push-Location $repo
& (Join-Path $repo 'build\nr_host.exe') --frames 1 --width 512 --height 512 --json (Join-Path $runDir 'nr_host_run2.json') | Out-Null
Pop-Location
$first  = 'results\20260830_180553_reference_package\color_rgba8.bin'
$pkg2 = Get-ChildItem (Join-Path $repo 'results') -Directory -Filter '*reference_package*' | Sort-Object Name | Select-Object -Last 1
$h1 = (Get-FileHash (Join-Path $repo $first)).Hash
$h2 = (Get-FileHash (Join-Path $pkg2.FullName 'color_rgba8.bin')).Hash
Write-Host "pkg1=$first"
Write-Host "pkg2=$($pkg2.Name)"
Write-Host "hash1=$h1"
Write-Host "hash2=$h2"
if ($h1 -eq $h2) { Write-Host 'DETERMINISM OK' } else { Write-Host 'DETERMINISM FAILED' }
