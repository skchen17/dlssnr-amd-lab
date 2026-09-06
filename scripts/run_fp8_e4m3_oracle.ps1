[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RawPath
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$raw = (Resolve-Path -LiteralPath $RawPath).Path
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only fp8_e4m3_probe
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_amd_e4m3_rtx_oracle"
New-Item -ItemType Directory -Path $result | Out-Null
$exe = Join-Path $repo 'build\fp8_e4m3_probe.exe'
$json = Join-Path $result 'fp8_e4m3_oracle.json'
& $exe $raw $json *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$run = if (Test-Path -LiteralPath $json) {
    Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
} else { $null }
$pass = $exitCode -eq 0 -and $run -and $run.status -eq 'PASS' `
    -and $run.capture_mismatches -eq 0 -and $run.finite_code_mismatches -eq 0 `
    -and $run.negative_verifier_detected
$manifest = [ordered]@{
    schema = 1
    experiment = 'amd_e4m3_rtx_oracle'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'FP8_NUMERICAL_PRIMITIVE'
    counts_as_s6 = $false
    exit_code = $exitCode
    rtx_raw_path = $raw
    rtx_raw_sha256 = (Get-FileHash -LiteralPath $raw -Algorithm SHA256).Hash
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $result 'manifest.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Write-Host ($manifest | ConvertTo-Json -Compress)
Write-Host "Result: $result"
exit $(if ($pass) { 0 } else { 1 })
