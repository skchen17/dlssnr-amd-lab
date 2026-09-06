# run_nvapi_trampoline_test.ps1 — Phase F ABI self-proof gate.
# Validates the nvapi_trace forwarding trampoline for 1..10 arguments
# (byte-exact args + return). >=8 and 10-arg cases are the mandatory ones.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\nvapi_trampoline_test.exe'
if (-not (Test-Path $exe)) { & (Join-Path $PSScriptRoot 'build_all.ps1') -Only nvapi_trampoline_test,nvapi_trace }

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$dir = Join-Path $repo "results\${ts}_nvapi_trampoline_test"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Push-Location $dir
try {
    & $exe (Join-Path $repo 'build\nvapi64.dll')
} finally { Pop-Location }
$code = $LASTEXITCODE
Write-Host "nvapi trampoline test exit=$code artifacts=$dir"
exit $code
