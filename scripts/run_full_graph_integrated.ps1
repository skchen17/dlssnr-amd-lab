[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Plan,
    [Parameter(Mandatory)][string]$ResultDir
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$planPath = (Resolve-Path -LiteralPath $Plan -ErrorAction Stop).Path
$result = [IO.Path]::GetFullPath($ResultDir)
if (Test-Path -LiteralPath $result) { throw "Result directory already exists: $result" }
$zluda = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zluda;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"

python (Join-Path $PSScriptRoot 'run_full_graph_integrated.py') `
    --plan $planPath --nvcuda (Join-Path $zluda 'nvcuda.dll') --output $result
$executionExit = $LASTEXITCODE
if ($executionExit -ne 0) {
    Get-Content -Raw -LiteralPath (Join-Path $result 'execution.json')
    exit $executionExit
}

$analysis = Join-Path $result 'manifest.json'
python (Join-Path $PSScriptRoot 'analyze_full_graph_integrated.py') `
    --plan $planPath --execution $result --output $analysis
$analysisExit = $LASTEXITCODE
Get-Content -Raw -LiteralPath $analysis
Write-Host "Result: $result"
exit $analysisExit
