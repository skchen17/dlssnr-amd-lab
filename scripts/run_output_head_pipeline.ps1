param(
    [string]$ActivationArena = 'deliverables\postblock_mma_trace_reference_20260904_154238\payload\activation_arena.raw',
    [string]$ModelArena = 'local_models\decoded_310_8\model_arena.raw',
    [string]$RtxMmaTrace = 'results\20260904_761000_postblock_exact_mma_trace_cross_vendor_corrected\rtx_trace.raw',
    [string]$RtxE4Trace = 'results\20260904_780000_postblock_exact_e4m3_mov_cross_vendor\rtx_trace.raw',
    [string]$RtxStoreTrace = 'results\20260904_530000_postblock_store_trace_cross_vendor\rtx_postblock_store_trace.raw',
    [string]$OutputDirectory = '',
    [int]$Iterations = 20,
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repo ("results\{0}_output_head_pipeline_rx9070xt" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
} elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repo $OutputDirectory
}

function Resolve-Input([string]$Path) {
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repo $Path }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required input not found: $Path" }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Invoke-Stage([string]$Name, [string[]]$Arguments) {
    $exe = Join-Path $repo "build\$Name.exe"
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Executable not found: $exe" }
    Write-Host "`n== $Name =="
    & $exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

$ActivationArena = Resolve-Input $ActivationArena
$ModelArena = Resolve-Input $ModelArena
$RtxMmaTrace = Resolve-Input $RtxMmaTrace
$RtxE4Trace = Resolve-Input $RtxE4Trace
$RtxStoreTrace = Resolve-Input $RtxStoreTrace
if ($Iterations -le 0) { throw 'Iterations must be positive' }

$targets = @(
    'output_head_activation_fusion', 'output_head_first128',
    'output_head_mma128_175', 'output_head_qk_attention',
    'output_head_softmax_v', 'output_head_final_projection',
    'output_head_fp16_tail', 'output_head_surface_store',
    'd3d12_residual_bridge'
)
if ($Build) {
    & (Join-Path $PSScriptRoot 'build_all.ps1') -Only $targets
    if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$aE4 = Join-Path $OutputDirectory 'activation_e4m3.raw'
$aHalf = Join-Path $OutputDirectory 'activation_fp16.raw'
$first128 = Join-Path $OutputDirectory 'first128_fp16.raw'
$mma176 = Join-Path $OutputDirectory 'mma128_175_fp16.raw'
$q = Join-Path $OutputDirectory 'q_e4m3.raw'
$k = Join-Path $OutputDirectory 'k_e4m3.raw'
$v = Join-Path $OutputDirectory 'v_e4m3.raw'
$qk = Join-Path $OutputDirectory 'qk_fp16.raw'
$softmax = Join-Path $OutputDirectory 'softmax_e4m3.raw'
$attention = Join-Path $OutputDirectory 'attention_fp16.raw'
$projected = Join-Path $OutputDirectory 'attention_projected_fp16.raw'
$residual = Join-Path $OutputDirectory 'rgba_residual_fp16.raw'
$zeroSurface = Join-Path $OutputDirectory 'zero_base_output_rgba16f.raw'
$d3dSurface = Join-Path $OutputDirectory 'd3d12_output_rgba16f.raw'

$manifests = [ordered]@{
    activation = Join-Path $OutputDirectory '01_activation.json'
    first128 = Join-Path $OutputDirectory '02_first128.json'
    mma128_175 = Join-Path $OutputDirectory '03_mma128_175.json'
    qk_attention = Join-Path $OutputDirectory '04_qk_attention.json'
    softmax_v = Join-Path $OutputDirectory '05_softmax_v.json'
    final_projection = Join-Path $OutputDirectory '06_final_projection.json'
    fp16_tail = Join-Path $OutputDirectory '07_fp16_tail.json'
    surface_store = Join-Path $OutputDirectory '08_surface_store.json'
    d3d12_bridge = Join-Path $OutputDirectory '09_d3d12_bridge.json'
}

Invoke-Stage 'output_head_activation_fusion' @(
    $ActivationArena, $ModelArena, $RtxE4Trace, $manifests.activation,
    "$Iterations", $aE4, $aHalf)
Invoke-Stage 'output_head_first128' @(
    $aE4, $aHalf, $ModelArena, $RtxMmaTrace, $first128,
    $manifests.first128, "$Iterations")
Invoke-Stage 'output_head_mma128_175' @(
    $first128, $ModelArena, $RtxMmaTrace, $mma176,
    $manifests.mma128_175, "$Iterations")
Invoke-Stage 'output_head_qk_attention' @(
    $mma176, $ModelArena, $RtxMmaTrace, $RtxE4Trace,
    $q, $k, $v, $qk, $manifests.qk_attention, "$Iterations")
Invoke-Stage 'output_head_softmax_v' @(
    $qk, $v, $RtxMmaTrace, $RtxE4Trace, $softmax, $attention,
    $manifests.softmax_v, "$Iterations")
Invoke-Stage 'output_head_final_projection' @(
    $attention, $first128, $ModelArena, $RtxMmaTrace, $RtxE4Trace,
    $projected, $manifests.final_projection, "$Iterations")
Invoke-Stage 'output_head_fp16_tail' @(
    $projected, $ModelArena, $RtxMmaTrace, $residual,
    $manifests.fp16_tail, "$Iterations")
Invoke-Stage 'output_head_surface_store' @(
    $residual, '-', $RtxMmaTrace, $RtxStoreTrace, $zeroSurface,
    $manifests.surface_store, "$Iterations")
Invoke-Stage 'd3d12_residual_bridge' @(
    $residual, $d3dSurface, $manifests.d3d12_bridge, "$Iterations")

$stageReports = [ordered]@{}
$allPass = $true
foreach ($entry in $manifests.GetEnumerator()) {
    $report = Get-Content -LiteralPath $entry.Value -Raw | ConvertFrom-Json
    $stageReports[$entry.Key] = [ordered]@{
        status = $report.status
        experiment = $report.experiment
        manifest = [IO.Path]::GetFileName($entry.Value)
    }
    if ($report.status -ne 'PASS') { $allPass = $false }
}
$summary = [ordered]@{
    schema = 1
    experiment = 'amd_native_output_head_pipeline'
    status = $(if ($allPass) { 'PASS' } else { 'FAIL' })
    device = 'AMD Radeon RX 9070 XT'
    stages_passed = @($stageReports.Values | Where-Object status -eq 'PASS').Count
    stages_total = $stageReports.Count
    output_contract = 'D3D12 RGBA16F texture with clamp(base_rgb + 0.25 * residual_rgb, 0, 1)'
    oracle_dependency = 'Validation only; production execution consumes local activations and weights'
    execution_model = 'Validation harness: nine processes with disk intermediates'
    game_runtime_ready = $false
    next_gate = 'Fuse stages into one resident GPU process, remove oracle arguments, then connect feature-18 resources'
    stages = $stageReports
}
$summaryPath = Join-Path $OutputDirectory 'manifest.json'
[IO.File]::WriteAllText($summaryPath, (($summary | ConvertTo-Json -Depth 6) + "`n"))
Write-Host "`n[$($summary.status)] output-head pipeline $($summary.stages_passed)/$($summary.stages_total)"
Write-Host "Manifest: $summaryPath"
if (-not $allPass) { exit 1 }
