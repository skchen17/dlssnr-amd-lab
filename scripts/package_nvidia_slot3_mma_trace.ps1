[CmdletBinding()]
param(
    [string]$TraceRoot,
    [string]$PackageTag = 'slot3_mma_trace_reference'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\${PackageTag}_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')

if ([string]::IsNullOrWhiteSpace($TraceRoot)) {
    $trace = Join-Path $repo 'results\20260901_003000_slot3_mma_trace'
} elseif ([IO.Path]::IsPathRooted($TraceRoot)) {
    $trace = (Resolve-Path -LiteralPath $TraceRoot).Path
} else {
    $trace = (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path
}
$case = Join-Path $repo 'results\20260901_000000_slot3_accumulation_diagnostic\case'
$plan = Join-Path $repo 'results\20260831_234000_full_graph_integrated_plan'
$sourcePtx = Join-Path $trace 'slot3_instrumented_original.ptx'
$ptxText = [IO.File]::ReadAllText($sourcePtx)
$normalized = $ptxText -replace '(?m)^\.version 9\.4$', '.version 8.7'
if ($normalized -eq $ptxText) { throw 'instrumented PTX 9.4 declaration not found' }
[IO.File]::WriteAllText((Join-Path $payload 'slot3_mma_trace.ptx'), $normalized, [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $case 'activation_arena_before.raw') -Destination (Join-Path $payload 'activation_arena_before.raw')
Copy-Item -LiteralPath (Join-Path $case 'output_reference.raw') -Destination (Join-Path $payload 'output_reference.raw')
Copy-Item -LiteralPath (Join-Path $plan 'model_arena.raw') -Destination (Join-Path $payload 'model_arena.raw')
Copy-Item -LiteralPath (Join-Path $plan 'params\slot3.raw') -Destination (Join-Path $payload 'params.raw')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_slot3_mma_trace.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'SLOT3_MMA_TRACE_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')

$required = @('neural_reference_probe.exe','slot3_mma_trace.ptx','activation_arena_before.raw',
              'output_reference.raw','model_arena.raw','params.raw')
$hashes = [ordered]@{}
foreach ($name in $required) {
    $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash
}
$instrumentation = Get-Content -Raw -LiteralPath (Join-Path $trace 'instrumentation.json') | ConvertFrom-Json
[ordered]@{
    schema = 1
    experiment = 'rtx5070_slot3_all_fp8_mma_register_trace_package'
    normalization = '.version 9.4 -> .version 8.7 only after instrumentation'
    source_instrumented_ptx_sha256 = (Get-FileHash -LiteralPath $sourcePtx -Algorithm SHA256).Hash
    mma_count = [int]$instrumentation.mma_count
    cta = @($instrumentation.cta)
    checkpoint_bytes = [int]$instrumentation.checkpoint_bytes
    checkpoint_param_offset = 64
    checkpoint_arena_offset = 27807744
    function = 'cc_tinlayout_fused_swin_1h_32_1_chained_fp8'
    grid = @(41,25,1)
    block = @(32,1,1)
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8

$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
