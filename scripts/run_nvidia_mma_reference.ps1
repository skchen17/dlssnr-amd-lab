[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_mma_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$exe = Join-Path $root 'nvidia_mma_reference.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "missing $exe" }
& $exe $result *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$jsonPath = Join-Path $result 'mma_reference.json'
$run = if (Test-Path -LiteralPath $jsonPath) {
    Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
} else { $null }
$expected = [ordered]@{
    'mma_a.raw' = 4096
    'mma_b.raw' = 2048
    'mma_c.raw' = 2048
    'mma_d.raw' = 2048
}
$filesPass = $true
$hashes = [ordered]@{}
foreach($name in $expected.Keys) {
    $path = Join-Path $result $name
    if(-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -ne $expected[$name]) {
        $filesPass = $false
    } else {
        $hashes[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    }
}
$pass = $exitCode -eq 0 -and $run -and $run.status -eq 'PASS' `
    -and $run.cases -eq 8 -and $run.lanes_per_case -eq 32 `
    -and $run.negative_verifier_detected -and $filesPass
$manifest = [ordered]@{
    schema = 1
    package_revision = 'v15_rtx_m16n8k32_e4m3_mma'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_NUMERICAL_ORACLE'
    counts_as_s6 = $false
    exit_code = $exitCode
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    raw_sha256 = $hashes
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $result 'manifest.json') -Encoding utf8
$zip = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $zip -CompressionLevel Optimal -Force
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Write-Host ($manifest | ConvertTo-Json -Compress)
Write-Host "Result: $zip"
exit $(if ($pass) { 0 } else { 1 })
