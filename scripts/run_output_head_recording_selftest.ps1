param(
    [string]$ActivationArena = 'deliverables\postblock_mma_trace_reference_20260904_154238\payload\activation_arena.raw',
    [string]$Model = 'local_models\decoded_310_8\model_arena.raw',
    [string]$Reference = 'results\20260905_041000_output_head_full_d3d12_rx9070xt\output_rgba16f.raw',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
foreach ($name in @('ActivationArena', 'Model', 'Reference')) {
    $value = Get-Variable -Name $name -ValueOnly
    if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $repo $value }
    if (-not (Test-Path -LiteralPath $value -PathType Leaf)) { throw "Missing $name input: $value" }
    Set-Variable -Name $name -Value (Resolve-Path -LiteralPath $value).Path
}
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo ("results\{0}_output_head_recording_selftest" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only output_head_recording_selftest
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }
& (Join-Path $repo 'build\output_head_recording_selftest.exe') $ActivationArena $Model $Reference $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw "Recording self-test failed: $LASTEXITCODE" }
Write-Host "Result: $OutputDirectory"
