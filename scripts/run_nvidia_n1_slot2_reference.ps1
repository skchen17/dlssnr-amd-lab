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
$required = @('n1_reference_probe.exe','n1_slot2.ptx','input.raw','weights.raw','params.raw')
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $payload $name))) { throw "missing payload: $name" }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_n1_slot2_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$probeJson = Join-Path $result 'probe.json'
$output = Join-Path $result 'output.raw'
$sync = Join-Path $result 'sync.raw'
& (Join-Path $payload 'n1_reference_probe.exe') `
    --nvcuda $nvcuda --ptx (Join-Path $payload 'n1_slot2.ptx') `
    --function 'cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8' --json $probeJson `
    --n1-input (Join-Path $payload 'input.raw') --n1-weights (Join-Path $payload 'weights.raw') `
    --n1-params (Join-Path $payload 'params.raw') --n1-output $output --n1-sync-out $sync `
    *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$probe = if (Test-Path -LiteralPath $probeJson) { Get-Content -LiteralPath $probeJson -Raw | ConvertFrom-Json } else { $null }
$zeroWords = 0
$sentinelWords = 0
$unexpectedWords = 0
if (Test-Path -LiteralPath $sync) {
    $bytes = [System.IO.File]::ReadAllBytes($sync)
    if ($bytes.Length -eq 110592) {
        for ($offset = 0; $offset -lt $bytes.Length; $offset += 4) {
            $value = [BitConverter]::ToUInt32($bytes, $offset)
            if ($value -eq 0) { $zeroWords++ }
            elseif ($value -eq [uint32]::MaxValue) { $sentinelWords++ }
            else { $unexpectedWords++ }
        }
    }
}
$pass = $exitCode -eq 0 -and $probe -and $probe.pass -and $probe.kernel_launched -and `
    $probe.execution_verified -and (Test-Path -LiteralPath $output) -and `
    (Get-Item -LiteralPath $output).Length -eq 1966080 -and $zeroWords -eq 960 -and `
    $sentinelWords -eq 26688 -and $unexpectedWords -eq 0
$hashes = [ordered]@{}
foreach ($name in $required) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
[ordered]@{
    schema = 1
    experiment = 'rtx_n1_slot2_same_input_reference'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_ORIGINAL_PTX_NUMERICAL_REFERENCE'
    counts_as_s7 = $false
    exit_code = $exitCode
    device_name = if ($probe) { $probe.device_name } else { $null }
    grid = @(40,24,1)
    block = @(32,1,1)
    kernel_launched = if ($probe) { $probe.kernel_launched } else { $false }
    payload_sha256 = $hashes
    output_bytes = if (Test-Path -LiteralPath $output) { (Get-Item -LiteralPath $output).Length } else { 0 }
    output_sha256 = if (Test-Path -LiteralPath $output) { (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash } else { $null }
    sync = [ordered]@{ zero_words = $zeroWords; sentinel_words = $sentinelWords; unexpected_words = $unexpectedWords }
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$archive = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content -LiteralPath (Join-Path $result 'stdout.log')
Get-Content -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $archive"
exit $(if ($pass) { 0 } else { 1 })
