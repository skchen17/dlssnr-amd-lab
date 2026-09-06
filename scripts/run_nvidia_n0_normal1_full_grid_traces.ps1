[CmdletBinding()]
param([string]$PackageRoot)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $PSScriptRoot }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$packageManifest = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }

$integrity = $true
foreach ($property in $packageManifest.base_payload_sha256.psobject.Properties) {
    $path = Join-Path $payload $property.Name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing payload: $($property.Name)" }
    $integrity = $integrity -and ((Get-FileHash -LiteralPath $path).Hash -eq $property.Value)
}
foreach ($property in $packageManifest.variant_payload_sha256.psobject.Properties) {
    $path = Join-Path $payload (Join-Path 'variants' $property.Name)
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing variant: $($property.Name)" }
    $integrity = $integrity -and ((Get-FileHash -LiteralPath $path).Hash -eq $property.Value)
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$work = Join-Path $root "_n0_normal1_full_grid_traces_work_$stamp"
$result = Join-Path $root "_n0_normal1_full_grid_traces_reference_result_$stamp"
New-Item -ItemType Directory -Path $work, $result | Out-Null

function Invoke-Probe([string]$Ptx, [string]$Tag, [int]$ExtraBytes) {
    $json = Join-Path $work "$Tag.json"
    $scratch = Join-Path $work "$Tag.scratch.raw"
    $output = Join-Path $work "$Tag.output.raw"
    $arguments = @(
        '--nvcuda', $nvcuda, '--ptx', $Ptx, '--function', $packageManifest.function,
        '--json', $json, '--n0-input', (Join-Path $payload 'input_rgba16f.raw'),
        '--n0-weights', (Join-Path $payload 'weights.raw'), '--n0-params', (Join-Path $payload 'params.raw'),
        '--n0-scratch-out', $scratch, '--n0-output', $output,
        '--n0-grid-x', ([string]$packageManifest.grid[0]), '--n0-grid-y', ([string]$packageManifest.grid[1])
    )
    if ($ExtraBytes -gt 0) { $arguments += @('--n0-scratch-extra-bytes', [string]$ExtraBytes) }
    & (Join-Path $payload 'neural_reference_probe.exe') @arguments *> (Join-Path $work "$Tag.log")
    return [ordered]@{ exit = $LASTEXITCODE; json = $json; scratch = $scratch; output = $output }
}

$baseline = Invoke-Probe (Join-Path $payload 'n0_original.ptx') 'baseline' 0
if (-not (Test-Path -LiteralPath $baseline.json)) { throw 'Baseline did not produce JSON' }
$baselineProbe = Get-Content -Raw -LiteralPath $baseline.json | ConvertFrom-Json
$baselineHash = (Get-FileHash -LiteralPath $baseline.output).Hash
$rows = @()
$allRunsPass = ($baseline.exit -eq 0) -and [bool]$baselineProbe.pass

foreach ($variant in $packageManifest.variants) {
    $run = Invoke-Probe (Join-Path $payload (Join-Path 'variants' $variant.filename)) $variant.name ([int]$packageManifest.scratch_extra_bytes)
    if (-not (Test-Path -LiteralPath $run.json)) { throw "Variant did not produce JSON: $($variant.name)" }
    $probe = Get-Content -Raw -LiteralPath $run.json | ConvertFrom-Json
    $traceBytes = [int]$packageManifest.trace_bytes_per_variant
    $buffer = [byte[]]::new($traceBytes)
    $stream = [IO.File]::OpenRead($run.scratch)
    try {
        [void]$stream.Seek([int64]$packageManifest.trace_offset, [IO.SeekOrigin]::Begin)
        $read = 0
        while ($read -lt $traceBytes) {
            $count = $stream.Read($buffer, $read, $traceBytes - $read)
            if ($count -eq 0) { break }
            $read += $count
        }
        if ($read -ne $traceBytes) { throw "Short trace for $($variant.name): $read / $traceBytes" }
    }
    finally { $stream.Dispose() }
    $tracePath = Join-Path $result "$($variant.name)_trace.raw"
    [IO.File]::WriteAllBytes($tracePath, $buffer)
    $outputHash = (Get-FileHash -LiteralPath $run.output).Hash
    $preserved = $outputHash -eq $baselineHash
    $rows += [ordered]@{
        index = [int]$variant.index; name = [string]$variant.name; fields = @($variant.fields)
        probe_exit = [int]$run.exit; probe_pass = [bool]$probe.pass
        execution_verified = [bool]$probe.execution_verified; output_sha256 = $outputHash
        reference_output_preserved = $preserved; trace_filename = (Split-Path -Leaf $tracePath)
        trace_sha256 = (Get-FileHash -LiteralPath $tracePath).Hash
    }
    $allRunsPass = $allRunsPass -and ($run.exit -eq 0) -and [bool]$probe.pass -and [bool]$probe.execution_verified -and $preserved
}

$pass = $integrity -and $allRunsPass
[ordered]@{
    schema = 1; experiment = 'rtx_n0_normal1_full_grid_traces'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'N0_FIRST_BOX_MULLER_SINE_PATH_FULL_GRID_DATASET'
    counts_as_s7 = $false; payload_integrity = $integrity
    device_name = [string]$baselineProbe.device_name; grid = @($packageManifest.grid)
    block = @($packageManifest.block); sample_count = [int]$packageManifest.sample_count
    bytes_per_sample = [int]$packageManifest.bytes_per_sample
    trace_offset = [int64]$packageManifest.trace_offset
    trace_bytes_per_variant = [int]$packageManifest.trace_bytes_per_variant
    baseline_output_sha256 = $baselineHash; variants = $rows
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$zip = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $zip -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $zip"
exit $(if ($pass) { 0 } else { 1 })
