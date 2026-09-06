[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Archive,
    [string]$ResultDir = ''
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$archiveResolved = (Resolve-Path -LiteralPath $Archive -ErrorAction Stop).Path
if (-not $ResultDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $ResultDir = Join-Path $repo "results\${stamp}_swin4h_full_graph_exact_state"
}
$result = [IO.Path]::GetFullPath($ResultDir)
if (Test-Path -LiteralPath $result) { throw "Result directory already exists: $result" }
$casesDir = Join-Path $result 'cases'

python (Join-Path $PSScriptRoot 'extract_full_graph_swin4h_cases.py') $archiveResolved `
    --inventory (Join-Path $repo 'results\20260831_175000_full_graph_inventory\full_graph_inventory.json') `
    --params-root (Join-Path $repo 'results\20260831_172222_amd_swin4h_slots10_15_formal_v2') `
    --output $casesDir
if ($LASTEXITCODE -ne 0) { throw "Exact-state extraction failed: $LASTEXITCODE" }

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "zluda_ptx_probe build failed: $LASTEXITCODE" }

$zludaRoot = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
$nvcuda = Join-Path $zludaRoot 'nvcuda.dll'
$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zludaRoot;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
$lowering = Join-Path $repo 'results\20260831_172500_swin4h_slots10_15_lowering'
$cases = @(
    [ordered]@{slot=10;function='cc_tinlayout_fused_swin_4h_128_4_inpview_tilesync_fp8';ptx='inpview_full.ptx';grid=@(10,6);releases=60;wait=$false;release=$true;extra=$false},
    [ordered]@{slot=11;function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';ptx='chained_full.ptx';grid=@(11,7);releases=77;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=12;function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';ptx='chained_full.ptx';grid=@(11,6);releases=66;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=13;function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';ptx='chained_full.ptx';grid=@(10,7);releases=70;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=14;function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';ptx='chained_full.ptx';grid=@(10,6);releases=60;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=15;function='cc_tinlayout_fused_swin_4h_128_4_ds_wait_fp8';ptx='ds_wait_full.ptx';grid=@(11,7);releases=0;wait=$true;release=$false;extra=$true}
)

foreach ($runIndex in 1..2) {
    foreach ($case in $cases) {
        $caseDir = Join-Path $casesDir "slot$($case.slot)"
        $runDir = Join-Path $result "amd_run${runIndex}\slot$($case.slot)"
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $arguments = @(
            '--nvcuda',$nvcuda,'--ptx',(Join-Path $lowering $case.ptx),
            '--function',$case.function,'--json',(Join-Path $runDir 'probe.json'),
            '--n1-input',(Join-Path $caseDir 'input.raw'),
            '--n1-weights',(Join-Path $caseDir 'weights.raw'),
            '--n1-params',(Join-Path $caseDir 'params.raw'),
            '--n1-output',(Join-Path $runDir 'output.raw'),
            '--n1-sync-out',(Join-Path $runDir 'sync.raw'),
            '--n1-output-initial',(Join-Path $caseDir 'output_initial.raw'),
            '--n1-weight-view-offset','0','--n1-wait-param-offset','48',
            '--n1-release-param-offset','64','--n1-extra-param-offset','72',
            '--n1-grid-x',[string]$case.grid[0],'--n1-grid-y',[string]$case.grid[1],
            '--n1-block-x','32','--n1-block-y','4',
            '--n1-expected-releases',[string]$case.releases
        )
        if ($case.wait) {
            $arguments += @('--n1-wait-ready','--n1-wait-sync-initial',
                (Join-Path $caseDir 'wait_sync_initial.raw'))
        }
        if ($case.release) {
            $arguments += @('--n1-sync-initial',(Join-Path $caseDir 'sync_initial.raw'))
        } else {
            $arguments += '--n1-no-release'
        }
        if ($case.extra) {
            $arguments += @('--n1-extra-output',(Join-Path $runDir 'extra_output.raw'),
                '--n1-extra-output-initial',(Join-Path $caseDir 'extra_output_initial.raw'))
        }
        & $probe @arguments *> (Join-Path $runDir 'stdout.log')
        if ($LASTEXITCODE -ne 0) {
            throw "RX 9070 XT replay failed at run $runIndex slot $($case.slot)"
        }
    }
}

$report = Join-Path $result 'manifest.json'
python (Join-Path $PSScriptRoot 'analyze_full_graph_swin4h_replay.py') `
    --root $result --output $report
$analysisExit = $LASTEXITCODE
Get-Content -LiteralPath $report
Write-Host "Result: $result"
exit $analysisExit
