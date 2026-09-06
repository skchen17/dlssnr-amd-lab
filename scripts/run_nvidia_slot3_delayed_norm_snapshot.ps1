[CmdletBinding()]
param([string]$PackageRoot)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$scriptDirectory = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDirectory)) { $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path }
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $scriptDirectory }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$package = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }

$integrity = $true
$actualHashes = [ordered]@{}
foreach ($property in $package.payload_sha256.psobject.Properties) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $payload $property.Name) -Algorithm SHA256).Hash
    $actualHashes[$property.Name] = $actual
    $integrity = $integrity -and $actual -eq [string]$property.Value
}
if (-not $integrity) { throw 'payload hash verification failed' }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_slot3_delayed_norm_snapshot_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$records = @()
for ($run = 1; $run -le 2; $run++) {
    $prefix = "run$run"
    $arena = Join-Path $result "$prefix.arena.raw"
    $output = Join-Path $result "$prefix.output.raw"
    $sync = Join-Path $result "$prefix.sync.raw"
    $probeJson = Join-Path $result "$prefix.probe.json"
    $arguments = @(
        '--nvcuda',$nvcuda,'--ptx',(Join-Path $payload 'slot3_delayed_norm_snapshot.ptx'),
        '--function','cc_tinlayout_fused_swin_1h_32_1_chained_fp8','--json',$probeJson,
        '--n1-arena',(Join-Path $payload 'activation_arena_before.raw'),'--n1-arena-out',$arena,
        '--n1-arena-input-offset','9940992','--n1-arena-output-offset','11907072',
        '--n1-weights',(Join-Path $payload 'model_arena.raw'),'--n1-params',(Join-Path $payload 'params.raw'),
        '--n1-output',$output,'--n1-sync-out',$sync,
        '--n1-input-param-offset','0','--n1-output-param-offset','8',
        '--n1-no-weights-param','--n1-no-release','--n1-expected-releases','0',
        '--n1-grid-x','41','--n1-grid-y','25','--n1-grid-z','1',
        '--n1-block-x','32','--n1-block-y','1','--n1-block-z','1',
        '--n1-arena-param-view','0:9940992','--n1-arena-param-view','8:11907072',
        '--n1-arena-param-view','40:0','--n1-arena-param-view','56:4096',
        '--n1-arena-param-view','64:27807744','--n1-weight-param-view','16:4512256'
    )
    & (Join-Path $payload 'neural_reference_probe.exe') @arguments *> (Join-Path $result "$prefix.stdout.log")
    $probeExit = $LASTEXITCODE
    $probe = if (Test-Path -LiteralPath $probeJson) { Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json } else { $null }
    $trace = Join-Path $result "$prefix.trace.raw"
    if (Test-Path -LiteralPath $arena) {
        $buffer = New-Object byte[] ([int]$package.trace_bytes)
        $stream = [IO.File]::OpenRead($arena)
        try {
            $null = $stream.Seek([int64]$package.trace_arena_offset, [IO.SeekOrigin]::Begin)
            $read = $stream.Read($buffer, 0, $buffer.Length)
            if ($read -ne $buffer.Length) { throw "trace short read: $read" }
        } finally { $stream.Dispose() }
        [IO.File]::WriteAllBytes($trace, $buffer)
    }
    $traceBytes = if (Test-Path -LiteralPath $trace) { (Get-Item -LiteralPath $trace).Length } else { 0 }
    $traceNonzero = if ($traceBytes) { ([IO.File]::ReadAllBytes($trace) | Where-Object { $_ -ne 0 }).Count } else { 0 }
    $outputHash = if (Test-Path -LiteralPath $output) { (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash } else { $null }
    $records += [ordered]@{
        run = $run; probe_exit = $probeExit; probe_pass = [bool]($probe -and $probe.pass)
        execution_verified = [bool]($probe -and $probe.execution_verified)
        device_name = if ($probe) { $probe.device_name } else { $null }
        trace_bytes = $traceBytes; trace_nonzero_bytes = $traceNonzero
        trace_sha256 = if ($traceBytes) { (Get-FileHash -LiteralPath $trace -Algorithm SHA256).Hash } else { $null }
        output_sha256 = $outputHash
        output_reference_exact = [bool]($outputHash -eq [string]$package.output_reference_sha256)
    }
}
$repeat = $records.Count -eq 2 -and $records[0].trace_sha256 -eq $records[1].trace_sha256
$pass = $integrity -and $repeat -and ($records | Where-Object {
    $_.probe_exit -ne 0 -or -not $_.probe_pass -or -not $_.execution_verified -or
    $_.device_name -notmatch 'NVIDIA' -or $_.trace_bytes -ne [int]$package.trace_bytes -or
    $_.trace_nonzero_bytes -le 0 -or -not $_.output_reference_exact
}).Count -eq 0
[ordered]@{
    schema = 1; package_revision = [string]$package.package_revision
    experiment = 'rtx5070_slot3_delayed_norm_snapshot'; status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_SLOT3_DELAYED_NORMALIZATION_SNAPSHOT_ORACLE'; counts_as_s7 = $false
    payload_integrity = $integrity; payload_sha256 = $actualHashes; target_cta = @($package.cta)
    registers = @($package.registers); lanes = [int]$package.lanes
    bytes_per_lane = [int]$package.bytes_per_lane; trace_bytes = [int]$package.trace_bytes
    output_reference_sha256 = [string]$package.output_reference_sha256
    trace_repeat_bitwise_exact = $repeat; runs = $records
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8

$archive = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $archive"
exit $(if ($pass) { 0 } else { 1 })
