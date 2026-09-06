[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_fp8_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$exe = Join-Path $root 'nvidia_fp8_reference.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "missing $exe" }
$raw = Join-Path $result 'fp16_to_e4m3.raw'
$json = Join-Path $result 'fp16_to_e4m3.json'
& $exe $raw $json *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$run = if (Test-Path -LiteralPath $json) {
    Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
} else { $null }
$pass = $exitCode -eq 0 -and $run -and $run.status -eq 'PASS' `
    -and $run.input_half_patterns -eq 65536 -and $run.output_bytes -eq 65536 `
    -and $run.negative_verifier_detected -and (Test-Path -LiteralPath $raw) `
    -and (Get-Item -LiteralPath $raw).Length -eq 65536
$manifest = [ordered]@{
    schema = 1
    package_revision = 'v14_rtx_fp16_e4m3_exhaustive'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_NUMERICAL_ORACLE'
    counts_as_s6 = $false
    exit_code = $exitCode
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    raw_bytes = if (Test-Path -LiteralPath $raw) { (Get-Item -LiteralPath $raw).Length } else { 0 }
    raw_sha256 = if (Test-Path -LiteralPath $raw) { (Get-FileHash -LiteralPath $raw -Algorithm SHA256).Hash } else { $null }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $result 'manifest.json') -Encoding utf8
$zip = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $zip -CompressionLevel Optimal -Force
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Write-Host ($manifest | ConvertTo-Json -Compress)
Write-Host "Result: $zip"
exit $(if ($pass) { 0 } else { 1 })
