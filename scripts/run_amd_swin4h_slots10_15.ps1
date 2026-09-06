[CmdletBinding()]
param(
    [string]$InitialInput = '',
    [string]$WeightDir = '',
    [string]$ResultSuffix = 'amd_swin4h_slots10_15'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

function Get-NonzeroByteCount([string]$Path, [int]$Offset) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    [uint64]$count = 0
    for ($index = $Offset; $index -lt $bytes.Length; $index++) {
        if ($bytes[$index] -ne 0) { $count++ }
    }
    return $count
}

$repo = Split-Path -Parent $PSScriptRoot
if (-not $InitialInput) {
    $InitialInput = Join-Path $repo 'results\20260831_164100_amd_swin2h_slot9\extra_output.raw'
}
if (-not $WeightDir) {
    $WeightDir = Join-Path $repo 'results\20260831_163553_rtx5070_feature18_v19'
}
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
$lowering = Join-Path $repo 'results\20260831_172500_swin4h_slots10_15_lowering'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_$ResultSuffix"
New-Item -ItemType Directory -Path $result | Out-Null

$cases = @(
    [ordered]@{slot=10; function='cc_tinlayout_fused_swin_4h_128_4_inpview_tilesync_fp8'; ptx='inpview_full.ptx'; grid=@(10,6,1); releases=60; wait=$false; extra=$false},
    [ordered]@{slot=11; function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8'; ptx='chained_full.ptx'; grid=@(11,7,1); releases=77; wait=$true; extra=$false},
    [ordered]@{slot=12; function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8'; ptx='chained_full.ptx'; grid=@(11,6,1); releases=66; wait=$true; extra=$false},
    [ordered]@{slot=13; function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8'; ptx='chained_full.ptx'; grid=@(10,7,1); releases=70; wait=$true; extra=$false},
    [ordered]@{slot=14; function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8'; ptx='chained_full.ptx'; grid=@(10,6,1); releases=60; wait=$true; extra=$false},
    [ordered]@{slot=15; function='cc_tinlayout_fused_swin_4h_128_4_ds_wait_fp8'; ptx='ds_wait_full.ptx'; grid=@(11,7,1); releases=0; wait=$true; extra=$true}
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
        '--nvcuda',$nvcuda,'--ptx',(Join-Path $lowering $case.ptx),
        '--function',$case.function,'--json',$json,
        '--n1-input',$input,'--n1-weights',$weights,'--n1-params',$params,
        '--n1-output',$output,'--n1-sync-out',$sync,'--n1-weight-view-offset','0',
        '--n1-wait-param-offset','48','--n1-release-param-offset','64',
        '--n1-grid-x',[string]$case.grid[0],'--n1-grid-y',[string]$case.grid[1],
        '--n1-block-x','32','--n1-block-y','4',
        '--n1-expected-releases',[string]$case.releases
    )
    if ($case.wait) { $args += '--n1-wait-ready' }
    if ($case.extra) {
        $extraOutput = Join-Path $caseDir 'extra_output.raw'
        $args += @('--n1-no-release','--n1-extra-param-offset','72','--n1-extra-output',$extraOutput)
    }
    & $probe @args *> (Join-Path $caseDir 'stdout.log')
    $exitCode = $LASTEXITCODE
    $probeResult = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    $record = [ordered]@{
        slot=$case.slot; function=$case.function; grid=$case.grid; block=@(32,4,1)
        exit_code=$exitCode; pass=[bool]$probeResult.pass
        kernel_launched=[bool]$probeResult.kernel_launched
        execution_verified=[bool]$probeResult.execution_verified
        expected_releases=$case.releases; observed_releases=[uint64]$probeResult.n1_sync_zero_words
        input_sha256=(Get-FileHash -LiteralPath $input -Algorithm SHA256).Hash
        weights_bytes=(Get-Item -LiteralPath $weights).Length
        weights_sha256=(Get-FileHash -LiteralPath $weights -Algorithm SHA256).Hash
        params_sha256=(Get-FileHash -LiteralPath $params -Algorithm SHA256).Hash
        output_bytes=(Get-Item -LiteralPath $output).Length
        logical_output_bytes=491520
        output_sha256=(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
        output_nonzero_bytes=[uint64]$probeResult.n1_output_nonzero_bytes
        output_tail_nonzero_bytes=(Get-NonzeroByteCount $output 491520)
        sync_sha256=(Get-FileHash -LiteralPath $sync -Algorithm SHA256).Hash
    }
    if ($case.extra) {
        $record.extra_output_bytes = (Get-Item -LiteralPath $extraOutput).Length
        $record.logical_extra_output_bytes = 245760
        $record.extra_output_sha256 = (Get-FileHash -LiteralPath $extraOutput -Algorithm SHA256).Hash
        $record.extra_output_nonzero_bytes = [uint64]$probeResult.n1_extra_output_nonzero_bytes
        $record.extra_output_tail_nonzero_bytes = Get-NonzeroByteCount $extraOutput 245760
    }
    $records += $record
    if ($exitCode -ne 0 -or -not $probeResult.pass) {
        throw "AMD Swin 4h execution failed at slot $($case.slot): $result"
    }
    $input = $output
}

$manifest = [ordered]@{
    schema=1; experiment='amd_swin4h_slots10_15_dependency_order'; status='PASS'
    classification='REAL_AMD_NEURAL_DEPENDENCY_EXECUTION_PENDING_RTX_ORACLE'
    counts_as_s7=$false; device='AMD Radeon RX 9070 XT [ZLUDA]'
    initial_input_sha256=(Get-FileHash -LiteralPath $InitialInput -Algorithm SHA256).Hash
    captured_weight_source=$WeightDir; slots=$records
    next_gate='repeat for determinism, then run original slots 10-15 on RTX'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Write-Host ($manifest | ConvertTo-Json -Depth 8 -Compress)
Write-Host "Result: $result"
