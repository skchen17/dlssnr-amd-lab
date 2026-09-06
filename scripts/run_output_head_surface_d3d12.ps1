param(
    [string]$Residual = 'results\20260905_015000_output_head_resident_d3d12_rx9070xt\rgba_residual_fp16.raw',
    [string]$Base = '-',
    [string]$Expected = 'results\20260905_015000_output_head_resident_d3d12_rx9070xt\output_rgba16f.raw',
    [string]$OutputDirectory = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo ("results\{0}_output_head_surface_d3d12_rx9070xt" -f (Get-Date -Format 'yyyyMMdd_HHmmss')) }
elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $repo $OutputDirectory }
foreach ($name in @('Residual','Expected')) {
    $value = Get-Variable -Name $name -ValueOnly
    if (-not [IO.Path]::IsPathRooted($value)) { $value = Join-Path $repo $value }
    if (-not (Test-Path -LiteralPath $value -PathType Leaf)) { throw "Missing $name input: $value" }
    Set-Variable -Name $name -Value (Resolve-Path -LiteralPath $value).Path
}
if ($Base -ne '-') {
    if (-not [IO.Path]::IsPathRooted($Base)) { $Base = Join-Path $repo $Base }
    if (-not (Test-Path -LiteralPath $Base -PathType Leaf)) { throw "Missing Base input: $Base" }
    $Base = (Resolve-Path -LiteralPath $Base).Path
}
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only output_head_surface_d3d12
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
& (Join-Path $repo 'build\output_head_surface_d3d12.exe') $Residual $Base $Expected `
    (Join-Path $OutputDirectory 'output_rgba16f.raw') (Join-Path $OutputDirectory 'manifest.json')
if ($LASTEXITCODE -ne 0) { throw "D3D12 surface validation failed: $LASTEXITCODE" }
Write-Host "Result: $OutputDirectory"
