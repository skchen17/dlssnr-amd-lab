[CmdletBinding()]
param(
    [string]$InitialInput = '',
    [string]$WeightDir = '',
    [string]$ResultSuffix = 'amd_swin2h_slots6_8'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
if (-not $InitialInput) { $InitialInput = Join-Path $repo 'results\20260831_152915_amd_swin_slots3_5\slot5\extra_output.raw' }
if (-not $WeightDir) { $WeightDir = Join-Path $repo 'results\20260831_161539_rtx5070_feature18_v18' }
$InitialInput = (Resolve-Path -LiteralPath $InitialInput).Path
$WeightDir = (Resolve-Path -LiteralPath $WeightDir).Path

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "zluda_ptx_probe build failed: $LASTEXITCODE" }

$zludaRoot = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
$nvcuda = Join-Path $zludaRoot 'nvcuda.dll'
$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zludaRoot;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
$csv = Join-Path $repo 'results\20260831_002356_rtx5070_feature18_full_frame\frame_001_sequence.csv'
$slot6Ptx = Join-Path $repo 'results\20260831_155000_swin2h_slot6_lowering\slot6_compact_full.ptx'
$chainedPtx = Join-Path $repo 'results\20260831_163500_swin2h_slots7_9_lowering\chained_full.ptx'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_$ResultSuffix"
New-Item -ItemType Directory -Path $result | Out-Null

$cases = @(
    [ordered]@{slot=6; function='cc_tinlayout_fused_swin_2h_64_2_inpview_tilesync_fp8'; ptx=$slot6Ptx; grid=@(20,12,1); releases=240; wait=$false},
    [ordered]@{slot=7; function='cc_tinlayout_fused_swin_2h_64_2_chained_fp8'; ptx=$chainedPtx; grid=@(21,13,1); releases=273; wait=$true},
    [ordered]@{slot=8; function='cc_tinlayout_fused_swin_2h_64_2_chained_fp8'; ptx=$chainedPtx; grid=@(21,12,1); releases=252; wait=$true}
)

$input = $InitialInput
$records = @()
foreach ($case in $cases) {
    $caseDir = Join-Path $result "slot$($case.slot)"
    New-Item -ItemType Directory -Path $caseDir | Out-Null
    $params = Join-Path $caseDir 'params.raw'
    python (Join-Path $PSScriptRoot 'extract_graph_slot_params.py') $csv --slot $case.slot `
        --function $case.function --output $params --report (Join-Path $caseDir 'params.json')
    if ($LASTEXITCODE -ne 0) { throw "parameter extraction failed at slot $($case.slot)" }
    $weights = Join-Path $WeightDir "n0_slot$($case.slot)_weights_before.raw"
    $output = Join-Path $caseDir 'output.raw'
    $sync = Join-Path $caseDir 'sync.raw'
    $json = Join-Path $caseDir 'probe.json'
    $args = @(
        '--nvcuda',$nvcuda,'--ptx',$case.ptx,'--function',$case.function,'--json',$json,
        '--n1-input',$input,'--n1-weights',$weights,'--n1-params',$params,
        '--n1-output',$output,'--n1-sync-out',$sync,'--n1-weight-view-offset','0',
        '--n1-wait-param-offset','48','--n1-release-param-offset','64',
        '--n1-grid-x',[string]$case.grid[0],'--n1-grid-y',[string]$case.grid[1],
        '--n1-block-x','32','--n1-block-y','2',
        '--n1-expected-releases',[string]$case.releases
    )
    if ($case.wait) { $args += '--n1-wait-ready' }
    & $probe @args *> (Join-Path $caseDir 'stdout.log')
    $exitCode = $LASTEXITCODE
    $probeResult = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    $record = [ordered]@{
        slot=$case.slot; function=$case.function; grid=$case.grid; block=@(32,2,1)
        exit_code=$exitCode; pass=[bool]$probeResult.pass
        kernel_launched=[bool]$probeResult.kernel_launched
        execution_verified=[bool]$probeResult.execution_verified
        expected_releases=$case.releases; observed_releases=[uint64]$probeResult.n1_sync_zero_words
        input_sha256=(Get-FileHash -LiteralPath $input -Algorithm SHA256).Hash
        weights_sha256=(Get-FileHash -LiteralPath $weights -Algorithm SHA256).Hash
        params_sha256=(Get-FileHash -LiteralPath $params -Algorithm SHA256).Hash
        output_bytes=(Get-Item -LiteralPath $output).Length
        logical_output_bytes=983040
        output_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
        output_nonzero_bytes=[uint64]$probeResult.n1_output_nonzero_bytes
        sync_sha256=(Get-FileHash -LiteralPath $sync -Algorithm SHA256).Hash
    }
    $records += $record
    if ($exitCode -ne 0 -or -not $probeResult.pass) { throw "AMD Swin 2h execution failed at slot $($case.slot): $result" }
    $input = $output
}

$manifest = [ordered]@{
    schema=1; experiment='amd_swin2h_slots6_8_dependency_order'; status='PASS'
    classification='REAL_AMD_NEURAL_DEPENDENCY_EXECUTION_PENDING_RTX_ORACLE'
    counts_as_s7=$false; device='AMD Radeon RX 9070 XT [ZLUDA]'
    initial_input_sha256=(Get-FileHash -LiteralPath $InitialInput -Algorithm SHA256).Hash
    captured_weight_source=$WeightDir; slots=$records
    next_gate='capture an extended slot 9 weight view, execute slot 9, then run original slots 6-9 on RTX'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Write-Host ($manifest | ConvertTo-Json -Depth 8 -Compress)
Write-Host "Result: $result"
