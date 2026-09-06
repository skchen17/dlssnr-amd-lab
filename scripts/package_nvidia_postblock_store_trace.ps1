[CmdletBinding()]
param([string]$TraceRoot = 'results\20260904_510000_postblock_store_trace')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$traceRootPath = if ([IO.Path]::IsPathRooted($TraceRoot)) {
    (Resolve-Path -LiteralPath $TraceRoot).Path
} else { (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path }
$instrumentation = Get-Content -Raw -LiteralPath (Join-Path $traceRootPath 'instrumentation.json') | ConvertFrom-Json
$case = Join-Path $repo 'results\20260831_230000_decoder_full_graph_exact_state\cases\slot154'
$cases = Split-Path -Parent $case
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\postblock_store_trace_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null

Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
$ptxText = [IO.File]::ReadAllText((Join-Path $traceRootPath 'post_block_trace_rtx.ptx'))
$normalized = $ptxText -replace '(?m)^\.version 9\.4$','.version 8.7'
if ($normalized -eq $ptxText) { throw 'PTX version declaration not found' }
[IO.File]::WriteAllText((Join-Path $payload 'post_block_store_trace_rtx.ptx'),$normalized,[Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $cases 'model_arena.raw') -Destination (Join-Path $payload 'model_arena.raw')
Copy-Item -LiteralPath (Join-Path $case 'output0_reference.raw') -Destination (Join-Path $payload 'output_reference.raw')

$arena = [IO.File]::ReadAllBytes((Join-Path $case 'activation_arena_initial.raw'))
$arenaExtended = [byte[]]::new($arena.Length + [int]$instrumentation.trace_bytes)
[Array]::Copy($arena,$arenaExtended,$arena.Length)
[IO.File]::WriteAllBytes((Join-Path $payload 'activation_arena_trace.raw'),$arenaExtended)
$params = [IO.File]::ReadAllBytes((Join-Path $case 'params.raw'))
$paramsExtended = [byte[]]::new(192)
[Array]::Copy($params,$paramsExtended,$params.Length)
[IO.File]::WriteAllBytes((Join-Path $payload 'params_trace.raw'),$paramsExtended)

Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_postblock_store_trace.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'POSTBLOCK_STORE_TRACE_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')
$required = @('neural_reference_probe.exe','post_block_store_trace_rtx.ptx',
    'activation_arena_trace.raw','model_arena.raw','params_trace.raw','output_reference.raw')
$hashes = [ordered]@{}
foreach ($name in $required) {
    $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash
}
$zeroTrace = [byte[]]::new([int]$instrumentation.trace_bytes)
$zeroPath = Join-Path $stage 'zero_trace.tmp'
[IO.File]::WriteAllBytes($zeroPath,$zeroTrace)
$zeroHash = (Get-FileHash -LiteralPath $zeroPath -Algorithm SHA256).Hash
Remove-Item -LiteralPath $zeroPath
[ordered]@{
    schema = 1
    experiment = 'rtx_postblock_pre_surface_store_full_grid_trace_package'
    function = 'cc_tinlayout_fused_post_block_swin_1h_32_fp8'
    grid = @(81,49,1)
    block = @(32,1,1)
    trace_offset = 29773824
    trace_bytes = [int]$instrumentation.trace_bytes
    record_bytes = [int]$instrumentation.record_bytes
    threads_per_site = [int]$instrumentation.threads_per_site
    logical_output_bytes = 1843200
    zero_trace_sha256 = $zeroHash
    output_reference_sha256 = $hashes['output_reference.raw']
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$zip = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
