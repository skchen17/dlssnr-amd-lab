[CmdletBinding()]
param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$scriptDirectory = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDirectory)) {
    $scriptPath = $MyInvocation.MyCommand.Path
    if ([string]::IsNullOrWhiteSpace($scriptPath)) {
        throw 'Unable to locate the runner script directory. Invoke this file with PowerShell -File.'
    }
    $scriptDirectory = Split-Path -Parent $scriptPath
}
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $scriptDirectory }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }
$required = @('n0_reference_probe.exe','n0_post_f16.ptx','input_rgba16f.raw','weights.raw','params.raw')
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $payload $name))) { throw "missing payload: $name" }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_n0_post_f16_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$probeJson = Join-Path $result 'probe.json'
$scratch = Join-Path $result 'scratch.raw'
$output = Join-Path $result 'output.raw'
& (Join-Path $payload 'n0_reference_probe.exe') `
    --nvcuda $nvcuda --ptx (Join-Path $payload 'n0_post_f16.ptx') `
    --function 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8' --json $probeJson `
    --n0-input (Join-Path $payload 'input_rgba16f.raw') `
    --n0-weights (Join-Path $payload 'weights.raw') --n0-params (Join-Path $payload 'params.raw') `
    --n0-scratch-out $scratch --n0-output $output --n0-grid-x 1 --n0-grid-y 1 `
    *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$probe = if (Test-Path -LiteralPath $probeJson) { Get-Content -LiteralPath $probeJson -Raw | ConvertFrom-Json } else { $null }
$fragmentPath = Join-Path $result 'post_f16_fragments.raw'
if (Test-Path -LiteralPath $scratch) {
    $scratchBytes = [System.IO.File]::ReadAllBytes($scratch)
    if ($scratchBytes.Length -ge 4096) {
        $fragments = [byte[]]::new(4096)
        [System.Array]::Copy($scratchBytes, $fragments, 4096)
        [System.IO.File]::WriteAllBytes($fragmentPath, $fragments)
    }
}
$pass = $exitCode -eq 0 -and $probe -and $probe.pass -and $probe.kernel_launched -and `
    $probe.execution_verified -and $probe.n0_grid[0] -eq 1 -and $probe.n0_grid[1] -eq 1 -and `
    (Test-Path -LiteralPath $fragmentPath) -and (Get-Item -LiteralPath $fragmentPath).Length -eq 4096
$hashes = [ordered]@{}
foreach ($name in $required) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
[ordered]@{
    schema = 1
    experiment = 'rtx_n0_post_f16_fragment_checkpoint'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_CONTROLLED_INTERMEDIATE_REFERENCE'
    counts_as_s6 = $false
    exit_code = $exitCode
    device_name = if ($probe) { $probe.device_name } else { $null }
    grid = if ($probe) { $probe.n0_grid } else { $null }
    kernel_launched = if ($probe) { $probe.kernel_launched } else { $false }
    mma_operations = 16
    bytes_per_lane = 128
    lanes = 32
    payload_sha256 = $hashes
    fragments_sha256 = if (Test-Path -LiteralPath $fragmentPath) { (Get-FileHash -LiteralPath $fragmentPath -Algorithm SHA256).Hash } else { $null }
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$archive = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Get-Content -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $archive"
exit $(if ($pass) { 0 } else { 1 })
