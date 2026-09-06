[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ReferenceDir
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$reference = (Resolve-Path -LiteralPath $ReferenceDir).Path
$inputRaw = Join-Path $reference 'copy_input.raw'
$expectedRaw = Join-Path $reference 'copy_output.raw'
$metadataPath = Join-Path $reference 'copy_snapshot.json'
foreach ($path in @($inputRaw,$expectedRaw,$metadataPath)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Reference artifact missing: $path" }
}
$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if ($metadata.status -ne 'PASS' -or $metadata.format -ne 10) {
    throw 'Reference must be a PASS DXGI_FORMAT_R16G16B16A16_FLOAT snapshot'
}

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only nvapi_amd,nvapi_amd_copy_selftest
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_nvapi_amd_rtx_copy_oracle"
New-Item -ItemType Directory -Path $result | Out-Null
$dll = Join-Path $repo 'build\nvapi64_amd.dll'
$dxil = Join-Path $repo 'build\cg2r_copy.dxil'
$exe = Join-Path $repo 'build\nvapi_amd_copy_selftest.exe'
$json = Join-Path $result 'nvapi_amd_rtx_copy_oracle.json'
& $exe $dll $json $inputRaw $expectedRaw $metadata.width $metadata.height $metadata.format `
    *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$run = if (Test-Path -LiteralPath $json) {
    Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
} else { $null }
$pass = $exitCode -eq 0 -and $run -and $run.status -eq 'PASS' `
    -and $run.oracle_mode -eq 'RTX_CAPTURE' -and $run.byte_mismatches -eq 0 `
    -and $run.component_mismatches -eq 0
$manifest = [ordered]@{
    schema = 1
    experiment = 'nvapi_amd_rtx_copy_oracle'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'SAME_INPUT_RTX_AMD_BOUNDARY_COMPARISON'
    counts_as_s6 = $false
    exit_code = $exitCode
    reference_directory = $reference
    reference_metadata_sha256 = (Get-FileHash -LiteralPath $metadataPath -Algorithm SHA256).Hash
    reference_input_sha256 = (Get-FileHash -LiteralPath $inputRaw -Algorithm SHA256).Hash
    reference_expected_sha256 = (Get-FileHash -LiteralPath $expectedRaw -Algorithm SHA256).Hash
    dll_sha256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
    dxil_sha256 = (Get-FileHash -LiteralPath $dxil -Algorithm SHA256).Hash
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $result 'manifest.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Write-Host ($manifest | ConvertTo-Json -Compress)
Write-Host "Result: $result"
exit $(if ($pass) { 0 } else { 1 })
