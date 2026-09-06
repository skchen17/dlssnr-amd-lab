param(
    [string]$ActivationArena = 'deliverables\postblock_mma_trace_reference_20260904_154238\payload\activation_arena.raw',
    [string]$Model = 'local_models\decoded_310_8\model_arena.raw',
    [string]$Base = '-',
    [string]$OracleDirectory = 'results\20260905_013000_output_head_pipeline_rx9070xt',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo ("results\{0}_output_head_full_d3d12_rx9070xt" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
foreach ($name in @('ActivationArena','Model','OracleDirectory')) {
    $value = Get-Variable -Name $name -ValueOnly
    if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $repo $value }
    if (-not (Test-Path -LiteralPath $value)) { throw "Missing $name input: $value" }
    Set-Variable -Name $name -Value (Resolve-Path -LiteralPath $value).Path
}
if ($Base -ne '-') {
    if (-not [IO.Path]::IsPathRooted($Base)) { $Base = Join-Path $repo $Base }
    if (-not (Test-Path -LiteralPath $Base -PathType Leaf)) { throw "Missing Base input: $Base" }
    $Base = (Resolve-Path -LiteralPath $Base).Path
}
$oracleNames = @('first128_fp16.raw','mma128_175_fp16.raw','qk_fp16.raw','attention_fp16.raw','attention_projected_fp16.raw','rgba_residual_fp16.raw','zero_base_output_rgba16f.raw')
$oracles = foreach ($name in $oracleNames) {
    $path = Join-Path $OracleDirectory $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing oracle: $path" }
    (Resolve-Path -LiteralPath $path).Path
}
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only output_head_full_d3d12
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
& (Join-Path $repo 'build\output_head_full_d3d12.exe') $ActivationArena $Model $Base `
    $oracles[0] $oracles[1] $oracles[2] $oracles[3] $oracles[4] $oracles[5] $oracles[6] `
    (Join-Path $OutputDirectory 'output_rgba16f.raw') (Join-Path $OutputDirectory 'manifest.json')
if ($LASTEXITCODE -ne 0) { throw "complete D3D12 output-head validation failed: $LASTEXITCODE" }
Write-Host "Result: $OutputDirectory"
