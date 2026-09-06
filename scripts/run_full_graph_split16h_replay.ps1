[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Archive,
    [Parameter(Mandatory)][string]$LoweringDir,
    [string]$ResultDir = ''
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$archiveResolved = (Resolve-Path -LiteralPath $Archive -ErrorAction Stop).Path
$lowering = (Resolve-Path -LiteralPath $LoweringDir -ErrorAction Stop).Path
if (-not $ResultDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $ResultDir = Join-Path $repo "results\${stamp}_split16h_full_graph_exact_state"
}
$result = [IO.Path]::GetFullPath($ResultDir)
if (Test-Path -LiteralPath $result) { throw "Result directory already exists: $result" }
$casesDir = Join-Path $result 'cases'
python (Join-Path $PSScriptRoot 'extract_full_graph_split16h_cases.py') $archiveResolved `
    --inventory (Join-Path $repo 'results\20260831_175000_full_graph_inventory\full_graph_inventory.json') `
    --output $casesDir
if ($LASTEXITCODE -ne 0) { throw "Exact-state extraction failed: $LASTEXITCODE" }

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "zluda_ptx_probe build failed: $LASTEXITCODE" }
$zludaRoot = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
$nvcuda = Join-Path $zludaRoot 'nvcuda.dll'
$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zludaRoot;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
$caseManifest = Get-Content -Raw -LiteralPath (Join-Path $casesDir 'manifest.json') | ConvertFrom-Json

foreach ($runIndex in 1..2) {
    foreach ($case in $caseManifest.slots) {
        $slot = [int]$case.slot
        $caseDir = Join-Path $casesDir "slot$slot"
        $runDir = Join-Path $result "amd_run${runIndex}\slot$slot"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $arguments = @(
            '--nvcuda',$nvcuda,'--ptx',(Join-Path $lowering ($case.abi.ptx_tag + '_full.ptx')),
            '--function',$case.function,'--json',(Join-Path $runDir 'probe.json'),
            '--n1-arena',(Join-Path $caseDir 'activation_arena_initial.raw'),
            '--n1-arena-input-offset',[string]$case.arena_views.input,
            '--n1-arena-output-offset',[string]$case.arena_views.output,
            '--n1-weights',(Join-Path $casesDir $caseManifest.model_arena.path),
            '--n1-params',(Join-Path $caseDir 'params.raw'),
            '--n1-output',(Join-Path $runDir 'output.raw'),
            '--n1-sync-out',(Join-Path $runDir 'sync.raw'),
            '--n1-weight-view-offset',[string]$case.weight_view_offset,
            '--n1-input-param-offset',[string]$case.abi.input,
            '--n1-output-param-offset',[string]$case.abi.output,
            '--n1-weights-param-offset',[string]$case.abi.weights,
            '--n1-grid-x',[string]$case.grid[0],'--n1-grid-y',[string]$case.grid[1],
            '--n1-grid-z',[string]$case.grid[2],
            '--n1-block-x',[string]$case.block[0],'--n1-block-y',[string]$case.block[1],
            '--n1-block-z',[string]$case.block[2],
            '--n1-expected-releases',[string]$case.expected_releases
        )
        if ($null -ne $case.abi.input2) {
            $arguments += @('--n1-arena-input2-offset',[string]$case.arena_views.input2,
                '--n1-input2-param-offset',[string]$case.abi.input2)
        }
        if ($null -ne $case.abi.wait) {
            $arguments += @('--n1-wait-ready','--n1-wait-param-offset',[string]$case.abi.wait,
                '--n1-wait-sync-initial',(Join-Path $caseDir 'wait_sync_initial.raw'))
        }
        if ($null -ne $case.abi.release) {
            $arguments += @('--n1-release-param-offset',[string]$case.abi.release,
                '--n1-sync-initial',(Join-Path $caseDir 'sync_initial.raw'))
        } else {
            $arguments += '--n1-no-release'
        }
        if ($null -ne $case.abi.extra) {
            $arguments += @('--n1-extra-param-offset',[string]$case.abi.extra,
                '--n1-arena-extra-offset',[string]$case.arena_views.extra,
                '--n1-extra-output',(Join-Path $runDir 'extra_output.raw'),
                '--n1-extra-output-initial',(Join-Path $caseDir 'extra_output_initial.raw'))
        }
        & $probe @arguments *> (Join-Path $runDir 'stdout.log')
        if ($LASTEXITCODE -ne 0) {
            throw "RX 9070 XT replay failed at run $runIndex slot $slot"
        }
    }
}

$report = Join-Path $result 'manifest.json'
python (Join-Path $PSScriptRoot 'analyze_full_graph_swin4h_replay.py') --root $result --output $report
$analysisExit = $LASTEXITCODE
Get-Content -LiteralPath $report
Write-Host "Result: $result"
exit $analysisExit
