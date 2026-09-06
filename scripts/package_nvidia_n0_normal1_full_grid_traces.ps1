[CmdletBinding()]
param(
    [string]$TraceRoot = 'results\20260904_999700_n0_normal1_full_grid_traces',
    [string]$InputPath = 'results\20260901_153000_n0_zero_input\input_rgba16f_zero.raw'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw 'Probe build failed' }

$traceRootPath = if ([IO.Path]::IsPathRooted($TraceRoot)) {
    (Resolve-Path -LiteralPath $TraceRoot).Path
}
else {
    (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path
}
$instrumentation = Get-Content -Raw -LiteralPath (Join-Path $traceRootPath 'instrumentation_rtx.json') | ConvertFrom-Json
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\n0_normal1_full_grid_traces_reference_$stamp"
$payload = Join-Path $stage 'payload'
$variantsDir = Join-Path $payload 'variants'
$scriptsDir = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload, $variantsDir, $scriptsDir | Out-Null

Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
foreach ($variant in $instrumentation.variants) {
    $source = Join-Path $traceRootPath (Join-Path 'rtx_variants' $variant.filename)
    $text = [IO.File]::ReadAllText($source)
    $normalized = $text -replace '(?m)^\.version 9\.4$', '.version 8.7'
    if ($normalized -eq $text) { throw "PTX version not found: $source" }
    [IO.File]::WriteAllText((Join-Path $variantsDir $variant.filename), $normalized, [Text.UTF8Encoding]::new($false))
}
$original = Join-Path $repo 'results\20260831_011219_zluda_ptx_probe\neural_isolated.ptx'
$originalText = [IO.File]::ReadAllText($original)
$originalNormalized = $originalText -replace '(?m)^\.version 9\.4$', '.version 8.7'
if ($originalNormalized -eq $originalText) { throw "PTX version not found: $original" }
[IO.File]::WriteAllText((Join-Path $payload 'n0_original.ptx'), $originalNormalized, [Text.UTF8Encoding]::new($false))

$resolvedInput = if ([IO.Path]::IsPathRooted($InputPath)) {
    (Resolve-Path -LiteralPath $InputPath).Path
}
else {
    (Resolve-Path -LiteralPath (Join-Path $repo $InputPath)).Path
}
Copy-Item -LiteralPath $resolvedInput -Destination (Join-Path $payload 'input_rgba16f.raw')
$basePayload = Join-Path $repo 'deliverables\n0_full_reference_20260831_130219\payload'
Copy-Item -LiteralPath (Join-Path $basePayload 'weights.raw'), (Join-Path $basePayload 'params.raw') -Destination $payload
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_n0_normal1_full_grid_traces.ps1') -Destination $scriptsDir
Copy-Item -LiteralPath (Join-Path $repo 'N0_NORMAL1_FULL_GRID_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')

$baseHashes = [ordered]@{}
foreach ($name in @('neural_reference_probe.exe', 'n0_original.ptx', 'input_rgba16f.raw', 'weights.raw', 'params.raw')) {
    $baseHashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name)).Hash
}
$variantHashes = [ordered]@{}
foreach ($variant in $instrumentation.variants) {
    $variantHashes[$variant.filename] = (Get-FileHash -LiteralPath (Join-Path $variantsDir $variant.filename)).Hash
}
[ordered]@{
    schema = 1
    experiment = 'rtx_n0_normal1_full_grid_traces_package'
    function = 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'
    grid = @($instrumentation.grid)
    block = @($instrumentation.block)
    sample_count = [int]$instrumentation.sample_count
    bytes_per_sample = [int]$instrumentation.bytes_per_sample
    trace_offset = [int64]$instrumentation.trace_offset
    trace_bytes_per_variant = [int]$instrumentation.trace_bytes_per_variant
    scratch_extra_bytes = [int]$instrumentation.scratch_extra_bytes
    input_source = $resolvedInput
    variant_count = @($instrumentation.variants).Count
    variants = @($instrumentation.variants)
    base_payload_sha256 = $baseHashes
    variant_payload_sha256 = $variantHashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$zip = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip).Hash)"
