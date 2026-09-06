[CmdletBinding()]
param([string]$PackageRoot)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
    $PackageRoot = Split-Path -Parent $PSScriptRoot
}
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$package = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }

$actual = [ordered]@{}
$integrity = $true
foreach ($property in $package.payload_sha256.PSObject.Properties) {
    $path = Join-Path $payload $property.Name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing payload: $($property.Name)" }
    $actual[$property.Name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $integrity = $integrity -and $actual[$property.Name] -eq $property.Value
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$work = Join-Path $root "_postblock_store_trace_work_$stamp"
$result = Join-Path $root "_postblock_store_trace_reference_result_$stamp"
New-Item -ItemType Directory -Path $work,$result | Out-Null
$runs = @()
foreach ($index in 1..2) {
    $probeJson = Join-Path $work "probe_run$index.json"
    $arenaAfter = Join-Path $work "arena_after_run$index.raw"
    $output = Join-Path $work "output_run$index.raw"
    $sync = Join-Path $work "sync_run$index.raw"
    $log = Join-Path $work "stdout_run$index.log"
    $arguments = @(
        '--nvcuda',$nvcuda,
        '--ptx',(Join-Path $payload 'post_block_store_trace_rtx.ptx'),
        '--function',$package.function,
        '--json',$probeJson,
        '--n1-arena',(Join-Path $payload 'activation_arena_trace.raw'),
        '--n1-arena-out',$arenaAfter,
        '--n1-arena-input-offset','13873152',
        '--n1-arena-output-offset','27807744',
        '--n1-weights',(Join-Path $payload 'model_arena.raw'),
        '--n1-params',(Join-Path $payload 'params_trace.raw'),
        '--n1-output',$output,
        '--n1-sync-out',$sync,
        '--n1-input-param-offset','0',
        '--n1-output-param-offset','16',
        '--n1-no-weights-param','--n1-no-release','--n1-expected-releases','0',
        '--n1-grid-x','81','--n1-grid-y','49','--n1-grid-z','1',
        '--n1-block-x','32','--n1-block-y','1','--n1-block-z','1',
        '--n1-arena-param-view','0:13873152',
        '--n1-arena-param-view','8:110592',
        '--n1-arena-param-view','16:27807744',
        '--n1-arena-param-view',"184:$($package.trace_offset)",
        '--n1-weight-param-view','24:147429888',
        '--n1-weight-param-view','104:147429376'
    )
    & (Join-Path $payload 'neural_reference_probe.exe') @arguments *> $log
    $probeExit = $LASTEXITCODE
    $probe = if (Test-Path -LiteralPath $probeJson) {
        Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json
    } else { $null }

    $arenaStream = [IO.File]::OpenRead($arenaAfter)
    try {
        [void]$arenaStream.Seek([int64]$package.trace_offset,[IO.SeekOrigin]::Begin)
        $trace = [byte[]]::new([int]$package.trace_bytes)
        $read = 0
        while ($read -lt $trace.Length) {
            $count = $arenaStream.Read($trace,$read,$trace.Length-$read)
            if ($count -eq 0) { break }
            $read += $count
        }
        if ($read -ne $trace.Length) { throw "Trace short read: $read/$($trace.Length)" }
    } finally { $arenaStream.Dispose() }
    $tracePath = Join-Path $result "trace_run$index.raw"
    [IO.File]::WriteAllBytes($tracePath,$trace)

    $outputBytes = [IO.File]::ReadAllBytes($output)
    if ($outputBytes.Length -lt [int]$package.logical_output_bytes) {
        throw "Output short read: $($outputBytes.Length)"
    }
    $logical = [byte[]]::new([int]$package.logical_output_bytes)
    [Array]::Copy($outputBytes,$logical,$logical.Length)
    $logicalPath = Join-Path $result "output_run$index.raw"
    [IO.File]::WriteAllBytes($logicalPath,$logical)
    Copy-Item -LiteralPath $probeJson,$log -Destination $result
    $runs += [ordered]@{
        run = $index
        probe_exit = $probeExit
        probe_pass = [bool]($probe -and $probe.pass)
        execution_verified = [bool]($probe -and $probe.execution_verified)
        device_name = if ($probe) { $probe.device_name } else { $null }
        trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash
        output_sha256 = (Get-FileHash -LiteralPath $logicalPath -Algorithm SHA256).Hash
    }
}

$deterministic = $runs[0].trace_sha256 -eq $runs[1].trace_sha256
$traceNonzero = $runs[0].trace_sha256 -ne $package.zero_trace_sha256
$outputPreserved = $runs[0].output_sha256 -eq $package.output_reference_sha256 -and
                   $runs[1].output_sha256 -eq $package.output_reference_sha256
$runsPass = @($runs | Where-Object {
    $_.probe_exit -ne 0 -or -not $_.probe_pass -or -not $_.execution_verified -or
    $_.device_name -notmatch 'NVIDIA'
}).Count -eq 0
$pass = $integrity -and $runsPass -and $deterministic -and $traceNonzero -and $outputPreserved
[ordered]@{
    schema = 1
    experiment = 'rtx_postblock_pre_surface_store_full_grid_trace'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_POSTBLOCK_RGB_FIRST_DIVERGENCE_ORACLE'
    counts_as_s7 = $false
    payload_integrity = $integrity
    payload_sha256 = $actual
    grid = @(81,49,1)
    block = @(32,1,1)
    record_bytes = [int]$package.record_bytes
    threads_per_site = [int]$package.threads_per_site
    trace_bytes = [int]$package.trace_bytes
    trace_nonzero = $traceNonzero
    trace_repeat_bitwise_exact = $deterministic
    output_reference_preserved = $outputPreserved
    runs = $runs
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$zip = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $zip -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $zip"
exit $(if ($pass) { 0 } else { 1 })
