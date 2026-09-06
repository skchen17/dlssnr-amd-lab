[CmdletBinding()]
param([string]$PackageRoot)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$scriptDirectory = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDirectory)) { $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path }
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $scriptDirectory }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$packageManifest = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }

$integrity = $true
$actualHashes = [ordered]@{}
foreach ($property in $packageManifest.payload_sha256.psobject.Properties) {
    $name = $property.Name
    $actual = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash
    $actualHashes[$name] = $actual
    $integrity = $integrity -and $actual -eq [string]$property.Value
}
if (-not $integrity) { throw 'payload hash verification failed' }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_slot3_f16_path_trace_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$arenaAfter = Join-Path $result 'arena_after.raw'
$output = Join-Path $result 'output.raw'
$sync = Join-Path $result 'sync.raw'
$probeJson = Join-Path $result 'probe.json'
$arguments = @(
    '--nvcuda',$nvcuda,
    '--ptx',(Join-Path $payload 'slot3_f16_path_trace.ptx'),
    '--function','cc_tinlayout_fused_swin_1h_32_1_chained_fp8',
    '--json',$probeJson,
    '--n1-arena',(Join-Path $payload 'activation_arena_before.raw'),
    '--n1-arena-out',$arenaAfter,
    '--n1-arena-input-offset','9940992','--n1-arena-output-offset','11907072',
    '--n1-weights',(Join-Path $payload 'model_arena.raw'),
    '--n1-params',(Join-Path $payload 'params.raw'),
    '--n1-output',$output,'--n1-sync-out',$sync,
    '--n1-input-param-offset','0','--n1-output-param-offset','8',
    '--n1-no-weights-param','--n1-no-release','--n1-expected-releases','0',
    '--n1-grid-x','41','--n1-grid-y','25','--n1-grid-z','1',
    '--n1-block-x','32','--n1-block-y','1','--n1-block-z','1',
    '--n1-arena-param-view','0:9940992','--n1-arena-param-view','8:11907072',
    '--n1-arena-param-view','40:0','--n1-arena-param-view','56:4096',
    '--n1-arena-param-view','64:27807744','--n1-weight-param-view','16:4512256'
)
& (Join-Path $payload 'neural_reference_probe.exe') @arguments *> (Join-Path $result 'stdout.log')
$probeExit = $LASTEXITCODE
$probe = if (Test-Path -LiteralPath $probeJson) { Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json } else { $null }

$checkpoint = Join-Path $result 'f16_path_trace.raw'
if (Test-Path -LiteralPath $arenaAfter) {
    $buffer = New-Object byte[] ([int]$packageManifest.checkpoint_bytes)
    $stream = [IO.File]::OpenRead($arenaAfter)
    try {
        $null = $stream.Seek([int64]$packageManifest.checkpoint_arena_offset, [IO.SeekOrigin]::Begin)
        $read = $stream.Read($buffer, 0, $buffer.Length)
        if ($read -ne $buffer.Length) { throw "checkpoint short read: $read" }
    } finally { $stream.Dispose() }
    [IO.File]::WriteAllBytes($checkpoint, $buffer)
}
$checkpointBytes = if (Test-Path -LiteralPath $checkpoint) { (Get-Item -LiteralPath $checkpoint).Length } else { 0 }
$checkpointNonzero = if (Test-Path -LiteralPath $checkpoint) {
    ([IO.File]::ReadAllBytes($checkpoint) | Where-Object { $_ -ne 0 }).Count
} else { 0 }
$pass = ($probeExit -eq 0 -and $probe -and $probe.pass -eq $true -and
         $probe.device_name -match 'NVIDIA' -and
         $checkpointBytes -eq [int64]$packageManifest.checkpoint_bytes -and
         $checkpointNonzero -gt 0)
[ordered]@{
    schema = 1
    experiment = 'rtx5070_slot3_f16_path_register_trace'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_FIRST_DIVERGENT_ACTIVATION_PATH_ORACLE'
    counts_as_s7 = $false
    payload_integrity = $integrity
    payload_sha256 = $actualHashes
    probe_exit = $probeExit
    probe_pass = [bool]($probe -and $probe.pass)
    device_name = if ($probe) { $probe.device_name } else { $null }
    cta = @($packageManifest.cta)
    registers = @($packageManifest.registers)
    kernel_launched = [bool]($probe -and $probe.kernel_launched)
    checkpoint_bytes = $checkpointBytes
    checkpoint_nonzero_bytes = $checkpointNonzero
    checkpoint_sha256 = if ($checkpointBytes) { (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash } else { $null }
    output_sha256 = if (Test-Path -LiteralPath $output) { (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash } else { $null }
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8

$archive = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $archive"
exit $(if ($pass) { 0 } else { 1 })

