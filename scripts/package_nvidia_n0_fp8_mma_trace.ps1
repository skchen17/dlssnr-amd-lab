[CmdletBinding()]
param(
    [string]$TraceRoot = 'results\20260901_113000_n0_fp8_mma_trace',
    [string]$InputPath,
    [string]$InputVariant = 'captured_rgba16f',
    [string]$PackageTag = 'n0_fp8_mma_trace'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$traceRoot = if ([IO.Path]::IsPathRooted($TraceRoot)) { (Resolve-Path -LiteralPath $TraceRoot).Path } else { (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path }
$instrumentation = Get-Content -Raw -LiteralPath (Join-Path $traceRoot 'instrumentation.json') | ConvertFrom-Json
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\${PackageTag}_reference_$stamp"
$payload = Join-Path $stage 'payload'; $scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
$source = Join-Path $traceRoot 'n0_mma_trace_original.ptx'
$text = [IO.File]::ReadAllText($source)
$normalized = $text -replace '(?m)^\.version 9\.4$', '.version 8.7'
if ($normalized -eq $text) { throw 'PTX 9.4 version declaration not found' }
[IO.File]::WriteAllText((Join-Path $payload 'n0_fp8_mma_trace.ptx'), $normalized, [Text.UTF8Encoding]::new($false))
$originalSource = Join-Path $repo 'results\20260831_011219_zluda_ptx_probe\neural_isolated.ptx'
$originalText = [IO.File]::ReadAllText($originalSource)
$originalNormalized = $originalText -replace '(?m)^\.version 9\.4$', '.version 8.7'
if ($originalNormalized -eq $originalText) { throw 'Original PTX 9.4 version declaration not found' }
[IO.File]::WriteAllText((Join-Path $payload 'n0_original.ptx'), $originalNormalized, [Text.UTF8Encoding]::new($false))
$basePayload = Join-Path $repo 'deliverables\n0_full_reference_20260831_130219\payload'
$resolvedInput = if ([string]::IsNullOrWhiteSpace($InputPath)) { Join-Path $basePayload 'input_rgba16f.raw' } elseif ([IO.Path]::IsPathRooted($InputPath)) { (Resolve-Path -LiteralPath $InputPath).Path } else { (Resolve-Path -LiteralPath (Join-Path $repo $InputPath)).Path }
Copy-Item -LiteralPath $resolvedInput -Destination (Join-Path $payload 'input_rgba16f.raw')
foreach ($name in @('weights.raw','params.raw')) { Copy-Item -LiteralPath (Join-Path $basePayload $name) -Destination (Join-Path $payload $name) }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_n0_fp8_mma_trace.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'N0_FP8_MMA_TRACE_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')
$required = @('neural_reference_probe.exe','n0_fp8_mma_trace.ptx','n0_original.ptx','input_rgba16f.raw','weights.raw','params.raw')
$hashes = [ordered]@{}
foreach ($name in $required) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
[ordered]@{
    schema=1;experiment='rtx5070_n0_all_fp8_mma_register_trace_package'
    source_instrumented_ptx_sha256=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    source_original_ptx_sha256=(Get-FileHash -LiteralPath $originalSource -Algorithm SHA256).Hash
    normalization='.version 9.4 -> .version 8.7 only';function='cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'
    input_variant=$InputVariant;input_source_sha256=(Get-FileHash -LiteralPath $resolvedInput -Algorithm SHA256).Hash
    grid=@(1,1,1);block=@(32,1,1);mma_count=[int]$instrumentation.mma_count
    lanes=[int]$instrumentation.lanes;bytes_per_lane_mma=[int]$instrumentation.bytes_per_lane_mma
    trace_offset=[int64]$instrumentation.trace_offset;trace_bytes=[int]$instrumentation.trace_bytes
    payload_sha256=$hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
