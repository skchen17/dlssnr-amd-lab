[CmdletBinding()]
param(
    [string]$TraceRoot = 'results\20260904_999700_n0_normal1_full_grid_traces',
    [Parameter(Mandatory)][string]$InputPath,
    [Parameter(Mandatory)][string]$ResultDir
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$root = if ([IO.Path]::IsPathRooted($TraceRoot)) {
    (Resolve-Path -LiteralPath $TraceRoot).Path
} else { (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path }
$resolvedInput = if ([IO.Path]::IsPathRooted($InputPath)) {
    (Resolve-Path -LiteralPath $InputPath).Path
} else { (Resolve-Path -LiteralPath (Join-Path $repo $InputPath)).Path }
$result = [IO.Path]::GetFullPath((Join-Path $repo $ResultDir))
if (Test-Path -LiteralPath $result) { throw "Result directory already exists: $result" }
New-Item -ItemType Directory -Path $result | Out-Null

$meta = Get-Content -Raw -LiteralPath (Join-Path $root 'instrumentation_amd.json') | ConvertFrom-Json
$zluda = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
$env:PATH = "$zluda;$env:PATH"
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
$payload = Join-Path $repo 'deliverables\n0_full_reference_20260831_130219\payload'
$baselinePtx = Join-Path $repo 'results\20260904_210000_n0_cos1_ultrahigh_resolution_candidates\seg32768_deg2\candidate.ptx'

function Invoke-N0([string]$Ptx, [string]$Tag, [int]$ExtraBytes) {
    $json = Join-Path $result "$Tag.json"
    $scratch = Join-Path $result "$Tag.scratch.raw"
    $output = Join-Path $result "$Tag.output.raw"
    $arguments = @(
        '--nvcuda', (Join-Path $zluda 'nvcuda.dll'), '--ptx', $Ptx,
        '--function', 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8',
        '--json', $json, '--n0-input', $resolvedInput,
        '--n0-weights', (Join-Path $payload 'weights.raw'),
        '--n0-params', (Join-Path $payload 'params.raw'),
        '--n0-scratch-out', $scratch, '--n0-output', $output,
        '--n0-grid-x', '80', '--n0-grid-y', '48'
    )
    if ($ExtraBytes -gt 0) { $arguments += @('--n0-scratch-extra-bytes', [string]$ExtraBytes) }
    & $probe @arguments *> (Join-Path $result "$Tag.log")
    [ordered]@{ exit = $LASTEXITCODE; json = $json; scratch = $scratch; output = $output }
}

$baseline = Invoke-N0 $baselinePtx 'baseline' 0
$baselineProbe = Get-Content -Raw -LiteralPath $baseline.json | ConvertFrom-Json
$baselineHash = (Get-FileHash -LiteralPath $baseline.output).Hash
$rows = @()
$pass = ($baseline.exit -eq 0) -and [bool]$baselineProbe.pass
foreach ($variant in $meta.variants) {
    $run = Invoke-N0 (Join-Path $root (Join-Path 'amd_variants' $variant.filename)) $variant.name ([int]$meta.scratch_extra_bytes)
    $probeJson = Get-Content -Raw -LiteralPath $run.json | ConvertFrom-Json
    $buffer = [byte[]]::new([int]$meta.trace_bytes_per_variant)
    $stream = [IO.File]::OpenRead($run.scratch)
    try {
        [void]$stream.Seek([int64]$meta.trace_offset, [IO.SeekOrigin]::Begin)
        $read = 0
        while ($read -lt $buffer.Length) {
            $count = $stream.Read($buffer, $read, $buffer.Length - $read)
            if ($count -eq 0) { break }
            $read += $count
        }
        if ($read -ne $buffer.Length) { throw "Short trace: $read / $($buffer.Length)" }
    } finally { $stream.Dispose() }
    $trace = Join-Path $result "amd_$($variant.name)_trace.raw"
    [IO.File]::WriteAllBytes($trace, $buffer)
    $outputHash = (Get-FileHash -LiteralPath $run.output).Hash
    $preserved = $outputHash -eq $baselineHash
    $rows += [ordered]@{ name = $variant.name; probe_pass = [bool]$probeJson.pass;
        output_preserved = $preserved; trace_sha256 = (Get-FileHash -LiteralPath $trace).Hash }
    $pass = $pass -and ($run.exit -eq 0) -and [bool]$probeJson.execution_verified -and $preserved
}
[ordered]@{ schema = 1; experiment = 'rx9070xt_n0_normal1_full_grid_traces';
    status = if ($pass) {'PASS'} else {'FAIL'}; device_name = $baselineProbe.device_name;
    input_sha256 = (Get-FileHash -LiteralPath $resolvedInput).Hash; baseline_output_sha256 = $baselineHash;
    sample_count = [int]$meta.sample_count; variants = $rows } |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $result 'amd_manifest.json') -Encoding utf8
Get-Content -Raw -LiteralPath (Join-Path $result 'amd_manifest.json')
exit $(if ($pass) {0} else {1})
