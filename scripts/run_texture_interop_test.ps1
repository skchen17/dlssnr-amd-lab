# run_texture_interop_test.ps1 — Phase G texture interop gate runner.
# RGBA8/RGBA16F/R32F/RG16F, both directions, mismatch_count + max_abs_error.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\texture_interop_test.exe'
if (-not (Test-Path $exe)) { & (Join-Path $PSScriptRoot 'build_all.ps1') -Only texture_interop_test }

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$dir = Join-Path $repo "results\${ts}_texture_interop"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
& $exe --json (Join-Path $dir 'texture_interop.json')
$code = $LASTEXITCODE
Write-Host "texture interop gate exit=$code json=$dir\texture_interop.json"
exit $code
