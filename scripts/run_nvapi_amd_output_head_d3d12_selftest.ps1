param(
    [string]$ReferenceDirectory = 'results\20260905_042914_output_head_recording_selftest',
    [string]$OutputDirectory = '',
    [switch]$Automatic,
    [ValidateRange(2,250)][int]$Batches = 2,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathRooted($ReferenceDirectory)) { $ReferenceDirectory = Join-Path $repo $ReferenceDirectory }
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo ("results\{0}_nvapi_output_head_d3d12" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
foreach ($frame in 0..3) {
    if (-not (Test-Path -LiteralPath (Join-Path $ReferenceDirectory "frame${frame}_rgba16f.raw"))) { throw "Missing frame $frame in $ReferenceDirectory" }
}
if (-not $SkipBuild) {
    & (Join-Path $PSScriptRoot 'build_all.ps1') -Only module_trace,nvapi_amd,nvapi_amd_output_head_d3d12_selftest
    if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
}
$previousAuto = $env:MODULE_TRACE_AMD_D3D12_HEAD
$previousInterop = $env:MODULE_TRACE_AMD_INTEROP
$previousInline = $env:MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS
$previousLog = $env:MODULE_TRACE_LOG
$previousBatches = $env:DLSSNR_HEAD_TEST_BATCHES
try {
    $env:MODULE_TRACE_AMD_D3D12_HEAD = if ($Automatic) { '1' } else { '0' }
    $env:MODULE_TRACE_AMD_INTEROP = '0'
    $env:MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS = '1'
    $env:DLSSNR_HEAD_TEST_BATCHES = [string]$Batches
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $env:MODULE_TRACE_LOG = Join-Path $OutputDirectory 'module_trace.log'
    & (Join-Path $repo 'build\nvapi_amd_output_head_d3d12_selftest.exe') `
  (Join-Path $repo 'build\nvapi64_amd.dll') `
  (Join-Path $repo 'deliverables\postblock_mma_trace_reference_20260904_154238\payload\activation_arena.raw') `
  (Join-Path $repo 'local_models\decoded_310_8\model_arena.raw') `
  (Join-Path $repo 'results\20260831_234000_full_graph_integrated_plan\params\slot154.raw') `
  $ReferenceDirectory $OutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "NVAPI D3D12 head validation failed: $LASTEXITCODE" }
} finally {
    $env:MODULE_TRACE_AMD_D3D12_HEAD = $previousAuto
    $env:MODULE_TRACE_AMD_INTEROP = $previousInterop
    $env:MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS = $previousInline
    $env:MODULE_TRACE_LOG = $previousLog
    $env:DLSSNR_HEAD_TEST_BATCHES = $previousBatches
}
Write-Host "Result: $OutputDirectory"
