[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ReferenceDirectory
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$reference = (Resolve-Path -LiteralPath $ReferenceDirectory).Path
$paths = [ordered]@{}
foreach($name in 'mma_a.raw','mma_b.raw','mma_c.raw','mma_d.raw') {
    $paths[$name] = Join-Path $reference $name
    if(-not(Test-Path -LiteralPath $paths[$name])) { throw "missing $($paths[$name])" }
}
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only fp8_mma_probe
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_amd_m16n8k32_e4m3_mma_oracle"
New-Item -ItemType Directory -Path $result | Out-Null
$exe = Join-Path $repo 'build\fp8_mma_probe.exe'
$json = Join-Path $result 'mma_comparison.json'
& $exe $paths['mma_a.raw'] $paths['mma_b.raw'] $paths['mma_c.raw'] `
    $paths['mma_d.raw'] $json *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$run = if(Test-Path -LiteralPath $json) {
    Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
} else { $null }
$pass = $exitCode -eq 0 -and $run -and $run.status -eq 'PASS' `
    -and $run.half_outputs -eq 1024 -and $run.mismatches -eq 0 `
    -and @($run.case_mismatches | Where-Object { $_ -ne 0 }).Count -eq 0 `
    -and $run.negative_verifier_detected
$rawHashes = [ordered]@{}
foreach($name in $paths.Keys) {
    $rawHashes[$name] = (Get-FileHash -LiteralPath $paths[$name] -Algorithm SHA256).Hash
}
[ordered]@{
    schema = 1
    experiment = 'amd_m16n8k32_e4m3_mma_rtx_oracle'
    status = if($pass){'PASS'}else{'FAIL'}
    classification = 'MMA_NUMERICAL_PRIMITIVE'
    counts_as_s6 = $false
    exit_code = $exitCode
    reference_directory = $reference
    reference_sha256 = $rawHashes
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $result 'manifest.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Get-Content -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $result"
exit $(if($pass){0}else{1})
