$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only nvapi_amd,nvapi_amd_copy_selftest
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_nvapi_amd_copy_selftest"
New-Item -ItemType Directory -Force -Path $result | Out-Null
$dll = Join-Path $repo 'build\nvapi64_amd.dll'
$dxil = Join-Path $repo 'build\cg2r_copy.dxil'
$exe = Join-Path $repo 'build\nvapi_amd_copy_selftest.exe'
$json = Join-Path $result 'nvapi_amd_copy_selftest.json'
& $exe $dll $json *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$manifest = [ordered]@{
    schema = 1
    experiment = 'nvapi_amd_copy_selftest'
    classification = 'BOUNDARY_OPERATION_ONLY'
    counts_as_s6 = $false
    exit_code = $exitCode
    dll_sha256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
    dxil_sha256 = (Get-FileHash -LiteralPath $dxil -Algorithm SHA256).Hash
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $result 'manifest.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Write-Host "Result: $result"
exit $exitCode
