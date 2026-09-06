[CmdletBinding()]
param([string]$TraceRoot = 'results\20260905_048000_slot10_cta4_0_warp1_fp8_mma_trace')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw 'probe build failed' }
$trace = if ([IO.Path]::IsPathRooted($TraceRoot)) {
    (Resolve-Path -LiteralPath $TraceRoot).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path
}
$instrumentation = Get-Content -Raw -LiteralPath (Join-Path $trace 'instrumentation.json') | ConvertFrom-Json
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\swin4h_slot10_fp8_mma_trace_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null

Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
foreach ($mapping in @(
    @('instrumented_original.ptx','slot10_fp8_mma_trace.ptx'),
    @('..\20260831_172000_swin4h_slots10_15_baseline\inpview_original.ptx','slot10_original.ptx')
)) {
    $source = if ($mapping[0] -like '..\*') {
        Join-Path $trace $mapping[0]
    } else {
        Join-Path $trace $mapping[0]
    }
    $text = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $source).Path)
    $normalized = $text -replace '(?m)^\.version 9\.4$', '.version 8.7'
    if ($normalized -eq $text) { throw "PTX 9.4 declaration not found: $source" }
    [IO.File]::WriteAllText((Join-Path $payload $mapping[1]), $normalized, [Text.UTF8Encoding]::new($false))
}

$case = Join-Path $repo 'results\20260831_191000_swin4h_full_graph_exact_state\cases\slot10'
foreach ($name in @('input.raw','weights.raw','params.raw','output_initial.raw','output_reference.raw','sync_initial.raw','sync_reference.raw')) {
    Copy-Item -LiteralPath (Join-Path $case $name) -Destination (Join-Path $payload $name)
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_swin4h_slot10_fp8_mma_trace.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'SWIN4H_SLOT10_FP8_MMA_TRACE_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')

$required = @('neural_reference_probe.exe','slot10_fp8_mma_trace.ptx','slot10_original.ptx','input.raw','weights.raw','params.raw','output_initial.raw','output_reference.raw','sync_initial.raw','sync_reference.raw')
$hashes = [ordered]@{}
foreach ($name in $required) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
[ordered]@{
    schema=1;package_revision='v1_slot10_cta4_0_warp1_all_fp8_mma'
    experiment='rtx_swin4h_slot10_selected_warp_fp8_mma_trace_package'
    function=[string]$instrumentation.function;grid=@(10,6,1);block=@(32,4,1)
    target_cta=@($instrumentation.target_cta);target_warp_y=[int]$instrumentation.target_warp_y
    first_exact_input_output_mismatch=@{byte_offset=4292;half_index=2146;rtx_bits='0xBAC3';rx_bits='0xBBC2'}
    mma_count=[int]$instrumentation.mma_count;lanes=[int]$instrumentation.lanes
    bytes_per_lane_mma=[int]$instrumentation.bytes_per_lane_mma
    trace_bytes=[int]$instrumentation.trace_bytes;trace_param_offset=72
    output_reference_sha256=$hashes['output_reference.raw']
    sync_reference_sha256=$hashes['sync_reference.raw']
    normalization='.version 9.4 -> .version 8.7 only';payload_sha256=$hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8

$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
