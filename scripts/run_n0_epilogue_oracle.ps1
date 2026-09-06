[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$CaptureDirectory)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$capture = (Resolve-Path -LiteralPath $CaptureDirectory).Path
$scratch = Join-Path $capture 'n0_scratch_after.raw'
$reference = Join-Path $capture 'n0_output_after.raw'
if (-not (Test-Path -LiteralPath $scratch) -or -not (Test-Path -LiteralPath $reference)) {
    throw 'capture must contain n0_scratch_after.raw and n0_output_after.raw'
}
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only n0_epilogue_probe
if ($LASTEXITCODE -ne 0) { throw 'n0_epilogue_probe build failed' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_amd_n0_tiled_epilogue"
New-Item -ItemType Directory -Path $result | Out-Null
$exe = Join-Path $repo 'build\n0_epilogue_probe.exe'
$amdOutput = Join-Path $result 'amd_output.raw'
$comparison = Join-Path $result 'comparison.json'
& $exe $scratch $reference $amdOutput $comparison *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$run = if (Test-Path -LiteralPath $comparison) { Get-Content -LiteralPath $comparison -Raw | ConvertFrom-Json } else { $null }
$pass = $exitCode -eq 0 -and $run -and $run.status -eq 'PASS' -and $run.kernel_launched
[ordered]@{
    schema = 1
    experiment = 'amd_n0_tiled_epilogue_rtx_capture_oracle'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'REAL_AMD_N0_DATAFLOW_STAGE'
    counts_as_s6 = $false
    exit_code = $exitCode
    capture_directory = $capture
    scratch_sha256 = (Get-FileHash -LiteralPath $scratch -Algorithm SHA256).Hash
    rtx_output_sha256 = (Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash
    amd_output_sha256 = if (Test-Path -LiteralPath $amdOutput) { (Get-FileHash -LiteralPath $amdOutput -Algorithm SHA256).Hash } else { $null }
    executable_sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Get-Content -LiteralPath $comparison
Get-Content -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $result"
exit $(if ($pass) { 0 } else { 1 })
