[CmdletBinding()]
param([string]$TraceRoot = 'results\20260904_820000_postblock_exact_f16x2_path_trace')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$trace = if ([IO.Path]::IsPathRooted($TraceRoot)) {
    (Resolve-Path -LiteralPath $TraceRoot).Path
} else { (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path }
$capture = Join-Path $repo 'results\20260904_660000_frame_aligned_post_surface_rtx5070'
$reference = Join-Path $repo 'results\20260904_690000_deferred_frame1_output_rtx5070'
$plan = Join-Path $repo 'results\20260831_234000_full_graph_integrated_plan'
$instrumentation = Get-Content -Raw -LiteralPath (Join-Path $trace 'instrumentation.json') | ConvertFrom-Json
if ($instrumentation.status -ne 'PASS') { throw 'instrumentation did not pass' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\postblock_f16x2_path_trace_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
Copy-Item -LiteralPath (Join-Path $trace 'post_block_f16x2_path_trace_rtx_v87.ptx') -Destination (Join-Path $payload 'post_block_f16x2_path_trace.ptx')
Copy-Item -LiteralPath (Join-Path $trace 'instrumentation.json') -Destination $payload
Copy-Item -LiteralPath (Join-Path $capture 'post_activation_arena_prelaunch.raw') -Destination (Join-Path $payload 'activation_arena.raw')
Copy-Item -LiteralPath (Join-Path $capture 'frame1_post_surface_pre_slot154.raw') -Destination (Join-Path $payload 'surface_initial.raw')
Copy-Item -LiteralPath (Join-Path $reference 'copy_input.raw') -Destination (Join-Path $payload 'output_reference.raw')
Copy-Item -LiteralPath (Join-Path $plan 'model_arena.raw') -Destination $payload
$params = [IO.File]::ReadAllBytes((Join-Path $plan 'params\slot154.raw'))
$paramsTrace = [byte[]]::new(192)
[Array]::Copy($params,$paramsTrace,$params.Length)
[IO.File]::WriteAllBytes((Join-Path $payload 'params_trace.raw'),$paramsTrace)
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_postblock_mma_trace.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'POSTBLOCK_F16X2_PATH_TRACE_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')
$hashes = [ordered]@{}
foreach ($file in Get-ChildItem -LiteralPath $payload -File | Sort-Object Name) {
    $hashes[$file.Name] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
}
[ordered]@{
    schema = 1
    package_revision = 'v1_postblock_exact_frame1_selected_cta_f16x2_r944_path_trace'
    experiment = 'rtx_postblock_exact_frame1_selected_cta_f16x2_r944_path_trace_package'
    result_stem = 'postblock_f16x2_path_trace'
    result_experiment = 'rtx_postblock_exact_frame1_selected_cta_f16x2_r944_path_trace'
    result_classification = 'RTX_POSTBLOCK_F16X2_R944_PATH_ORACLE'
    ptx_file = 'post_block_f16x2_path_trace.ptx'
    function = 'cc_tinlayout_fused_post_block_swin_1h_32_fp8'
    grid = @(81,49,1); block = @(32,1,1)
    target_cta = $instrumentation.target_cta
    trace_bytes = [int]$instrumentation.trace_bytes
    record_bytes = [int]$instrumentation.record_bytes
    selected_operations = $instrumentation.selected_operations
    output_reference_sha256 = $hashes['output_reference.raw']
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$zip = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
