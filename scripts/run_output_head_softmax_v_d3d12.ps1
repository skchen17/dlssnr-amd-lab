param(
    [string]$Qk = 'results\20260905_013000_output_head_pipeline_rx9070xt\qk_fp16.raw',
    [string]$V = 'results\20260905_013000_output_head_pipeline_rx9070xt\v_e4m3.raw',
    [string]$Expected = 'results\20260905_013000_output_head_pipeline_rx9070xt\attention_fp16.raw',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo ("results\{0}_output_head_softmax_v_d3d12_rx9070xt" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
foreach ($name in @('Qk','V','Expected')) {
    $value = Get-Variable -Name $name -ValueOnly
    if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $repo $value }
    if (-not (Test-Path -LiteralPath $value -PathType Leaf)) { throw "Missing $name input: $value" }
    Set-Variable -Name $name -Value (Resolve-Path -LiteralPath $value).Path
}
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only output_head_softmax_v_d3d12
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
& (Join-Path $repo 'build\output_head_softmax_v_d3d12.exe') $Qk $V $Expected `
    (Join-Path $OutputDirectory 'attention_fp16.raw') `
    (Join-Path $OutputDirectory 'manifest.json')
if ($LASTEXITCODE -ne 0) { throw "D3D12 softmax/V validation failed: $LASTEXITCODE" }
Write-Host "Result: $OutputDirectory"
