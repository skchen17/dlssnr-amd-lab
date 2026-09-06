[CmdletBinding()]
param([string]$TraceRoot = 'results\20260904_900000_slot3_delayed_norm_snapshot')

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$trace = if ([IO.Path]::IsPathRooted($TraceRoot)) {
    (Resolve-Path -LiteralPath $TraceRoot).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\slot3_delayed_norm_snapshot_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null

Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
$source = Join-Path $trace 'slot3_delayed_norm_original.ptx'
$text = [IO.File]::ReadAllText($source)
$normalized = $text -replace '(?m)^\.version 9\.4$', '.version 8.7'
if ($normalized -eq $text) { throw 'instrumented PTX 9.4 declaration not found' }
[IO.File]::WriteAllText((Join-Path $payload 'slot3_delayed_norm_snapshot.ptx'), $normalized, [Text.UTF8Encoding]::new($false))

$case = Join-Path $repo 'results\20260901_000000_slot3_accumulation_diagnostic\case'
$plan = Join-Path $repo 'results\20260831_234000_full_graph_integrated_plan'
Copy-Item -LiteralPath (Join-Path $case 'activation_arena_before.raw') -Destination (Join-Path $payload 'activation_arena_before.raw')
Copy-Item -LiteralPath (Join-Path $case 'output_reference.raw') -Destination (Join-Path $payload 'output_reference.raw')
Copy-Item -LiteralPath (Join-Path $plan 'model_arena.raw') -Destination (Join-Path $payload 'model_arena.raw')
Copy-Item -LiteralPath (Join-Path $plan 'params\slot3.raw') -Destination (Join-Path $payload 'params.raw')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_slot3_delayed_norm_snapshot.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'SLOT3_DELAYED_NORM_SNAPSHOT_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')

$required = @('neural_reference_probe.exe','slot3_delayed_norm_snapshot.ptx','activation_arena_before.raw','output_reference.raw','model_arena.raw','params.raw')
$hashes = [ordered]@{}
foreach ($name in $required) {
    $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash
}
$instrumentation = Get-Content -Raw -LiteralPath (Join-Path $trace 'instrumentation.json') | ConvertFrom-Json
[ordered]@{
    schema = 1
    package_revision = 'v1_slot3_same_input_delayed_norm_snapshot'
    experiment = 'rtx5070_slot3_delayed_norm_snapshot_package'
    source_instrumented_ptx_sha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    normalization = '.version 9.4 -> .version 8.7 only'
    function = 'cc_tinlayout_fused_swin_1h_32_1_chained_fp8'
    cta = @($instrumentation.cta)
    registers = @($instrumentation.registers)
    lanes = [int]$instrumentation.lanes
    bytes_per_lane = [int]$instrumentation.bytes_per_lane
    trace_bytes = [int]$instrumentation.checkpoint_bytes
    trace_arena_offset = 27807744
    grid = @(41,25,1)
    block = @(32,1,1)
    output_reference_sha256 = $hashes['output_reference.raw']
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8

$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
