[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$CasesDir,
    [Parameter(Mandatory)][string]$LoweringDir,
    [Parameter(Mandatory)][string]$ResultDir,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$cases = (Resolve-Path -LiteralPath $CasesDir -ErrorAction Stop).Path
$lowering = (Resolve-Path -LiteralPath $LoweringDir -ErrorAction Stop).Path
$result = [IO.Path]::GetFullPath($ResultDir)
if (Test-Path -LiteralPath $result) {
    if (-not $Resume) { throw "Result directory already exists: $result" }
} else {
    New-Item -ItemType Directory -Path $result | Out-Null
    New-Item -ItemType Junction -Path (Join-Path $result 'cases') -Target $cases | Out-Null
}

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "zluda_ptx_probe build failed: $LASTEXITCODE" }
$zluda = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zluda;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
$manifest = Get-Content -Raw -LiteralPath (Join-Path $cases 'manifest.json') | ConvertFrom-Json

foreach ($runIndex in 1..2) {
    foreach ($case in $manifest.slots) {
        $slot = [int]$case.slot
        $caseDir = Join-Path $cases "slot$slot"
        $runDir = Join-Path $result "amd_run${runIndex}\slot$slot"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $existingProbe = Join-Path $runDir 'probe.json'
        if ($Resume -and (Test-Path -LiteralPath $existingProbe) -and
            (Test-Path -LiteralPath (Join-Path $runDir 'arena_after.raw'))) {
            $existing = Get-Content -Raw -LiteralPath $existingProbe | ConvertFrom-Json
            if ($existing.pass -eq $true) { continue }
        }
        $arguments = @(
            '--nvcuda',(Join-Path $zluda 'nvcuda.dll'),
            '--ptx',(Join-Path $lowering ($case.ptx_tag + '_full.ptx')),
            '--function',$case.function,'--json',(Join-Path $runDir 'probe.json'),
            '--n1-arena',(Join-Path $caseDir 'activation_arena_initial.raw'),
            '--n1-arena-out',(Join-Path $runDir 'arena_after.raw'),
            '--n1-arena-input-offset',[string]$case.activation_param_views[0].arena_offset,
            '--n1-arena-output-offset',[string]$case.outputs[0].arena_offset,
            '--n1-weights',(Join-Path $cases $manifest.model_arena.path),
            '--n1-params',(Join-Path $caseDir 'params.raw'),
            '--n1-output',(Join-Path $runDir 'output.raw'),
            '--n1-sync-out',(Join-Path $runDir 'dummy_sync.raw'),
            '--n1-input-param-offset',[string]$case.activation_param_views[0].param_offset,
            '--n1-output-param-offset',[string]$case.outputs[0].param_offset,
            '--n1-no-release','--n1-expected-releases','0',
            '--n1-grid-x',[string]$case.grid[0],'--n1-grid-y',[string]$case.grid[1],
            '--n1-grid-z',[string]$case.grid[2],
            '--n1-block-x',[string]$case.block[0],'--n1-block-y',[string]$case.block[1],
            '--n1-block-z',[string]$case.block[2]
        )
        if ($null -eq $case.weight_param_offset) {
            $arguments += '--n1-no-weights-param'
        } else {
            $arguments += @('--n1-weights-param-offset',[string]$case.weight_param_offset,
                '--n1-weight-view-offset',[string]$case.weight_view_offset)
        }
        foreach ($view in $case.activation_param_views) {
            $arguments += @('--n1-arena-param-view',
                "$($view.param_offset):$($view.arena_offset)")
        }
        & $probe @arguments *> (Join-Path $runDir 'stdout.log')
        if ($LASTEXITCODE -ne 0) {
            throw "RX 9070 XT ViT replay failed at run $runIndex slot $slot"
        }
    }
}

$report = Join-Path $result 'manifest.json'
python (Join-Path $PSScriptRoot 'analyze_full_graph_vit1d_replay.py') `
    --root $result --output $report
$analysisExit = $LASTEXITCODE
Get-Content -LiteralPath $report
Write-Host "Result: $result"
exit $analysisExit
