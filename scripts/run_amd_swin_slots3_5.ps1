[CmdletBinding()]
param(
    [string]$N1Input = '',
    [string]$WeightDir = '',
    [string]$ResultSuffix = 'amd_swin_slots3_5'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
if (-not $N1Input) { $N1Input = Join-Path $repo 'results\20260831_143501_amd_n1_slot2\output.raw' }
if (-not $WeightDir) { $WeightDir = Join-Path $repo 'results\20260831_151635_rtx5070_feature18_v17' }
$N1Input = (Resolve-Path -LiteralPath $N1Input).Path
$WeightDir = (Resolve-Path -LiteralPath $WeightDir).Path

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "zluda_ptx_probe build failed: $LASTEXITCODE" }

$zludaRoot = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
$nvcuda = Join-Path $zludaRoot 'nvcuda.dll'
$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zludaRoot;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
$csv = Join-Path $repo 'results\20260831_002356_rtx5070_feature18_full_frame\frame_001_sequence.csv'
$lowered = Join-Path $repo 'results\20260831_144500_swin1h_slots3_5_lowering'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_$ResultSuffix"
New-Item -ItemType Directory -Path $result | Out-Null

$cases = @(
    [ordered]@{
        name = 'slot3'; slot = 3
        function = 'cc_tinlayout_fused_swin_1h_32_1_chained_fp8'
        ptx = Join-Path $lowered 'chained_full.ptx'
        weights = Join-Path $WeightDir 'n0_slot3_weights_before.raw'
        grid_x = 41; grid_y = 25; releases = 1025; extra = $false
    },
    [ordered]@{
        name = 'slot4'; slot = 4
        function = 'cc_tinlayout_fused_swin_1h_32_1_chained_fp8'
        ptx = Join-Path $lowered 'chained_full.ptx'
        weights = Join-Path $WeightDir 'n0_slot4_weights_before.raw'
        grid_x = 41; grid_y = 24; releases = 984; extra = $false
    },
    [ordered]@{
        name = 'slot5'; slot = 5
        function = 'cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8'
        ptx = Join-Path $lowered 'ds_wait_full.ptx'
        weights = Join-Path $WeightDir 'n0_slot5_weights_before.raw'
        grid_x = 40; grid_y = 25; releases = 0; extra = $true
    }
)

$input = $N1Input
$records = @()
foreach ($case in $cases) {
    $caseDir = Join-Path $result $case.name
    New-Item -ItemType Directory -Path $caseDir | Out-Null
    $params = Join-Path $caseDir 'params.raw'
    $paramsReport = Join-Path $caseDir 'params.json'
    python (Join-Path $PSScriptRoot 'extract_graph_slot_params.py') $csv `
        --slot $case.slot --function $case.function --output $params --report $paramsReport
    if ($LASTEXITCODE -ne 0) { throw "parameter extraction failed: $($case.name)" }

    $output = Join-Path $caseDir 'output.raw'
    $sync = Join-Path $caseDir 'sync.raw'
    $json = Join-Path $caseDir 'probe.json'
    $stdout = Join-Path $caseDir 'stdout.log'
    $args = @(
        '--nvcuda', $nvcuda, '--ptx', $case.ptx, '--function', $case.function,
        '--json', $json, '--n1-input', $input, '--n1-weights', $case.weights,
        '--n1-params', $params, '--n1-output', $output, '--n1-sync-out', $sync,
        '--n1-wait-ready', '--n1-weight-view-offset', '0',
        '--n1-grid-x', [string]$case.grid_x, '--n1-grid-y', [string]$case.grid_y,
        '--n1-expected-releases', [string]$case.releases
    )
    $extraOutput = Join-Path $caseDir 'extra_output.raw'
    if ($case.extra) { $args += @('--n1-extra-output', $extraOutput) }
    & $probe @args *> $stdout
    $exitCode = $LASTEXITCODE
    $probeResult = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    $record = [ordered]@{
        slot = $case.slot
        function = $case.function
        grid = @($case.grid_x,$case.grid_y,1)
        exit_code = $exitCode
        pass = [bool]$probeResult.pass
        kernel_launched = [bool]$probeResult.kernel_launched
        execution_verified = [bool]$probeResult.execution_verified
        expected_releases = $case.releases
        observed_releases = [uint64]$probeResult.n1_sync_zero_words
        input_sha256 = (Get-FileHash -LiteralPath $input -Algorithm SHA256).Hash
        weights_sha256 = (Get-FileHash -LiteralPath $case.weights -Algorithm SHA256).Hash
        params_sha256 = (Get-FileHash -LiteralPath $params -Algorithm SHA256).Hash
        output_bytes = if (Test-Path -LiteralPath $output) { (Get-Item -LiteralPath $output).Length } else { 0 }
        output_sha256 = if (Test-Path -LiteralPath $output) { (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash } else { $null }
        output_nonzero_bytes = [uint64]$probeResult.n1_output_nonzero_bytes
        extra_output_bytes = if ($case.extra -and (Test-Path -LiteralPath $extraOutput)) { (Get-Item -LiteralPath $extraOutput).Length } else { 0 }
        extra_output_sha256 = if ($case.extra -and (Test-Path -LiteralPath $extraOutput)) { (Get-FileHash -LiteralPath $extraOutput -Algorithm SHA256).Hash } else { $null }
        extra_output_nonzero_bytes = [uint64]$probeResult.n1_extra_output_nonzero_bytes
        sync_sha256 = if (Test-Path -LiteralPath $sync) { (Get-FileHash -LiteralPath $sync -Algorithm SHA256).Hash } else { $null }
    }
    $records += $record
    if ($exitCode -ne 0 -or -not $probeResult.pass) {
        $manifest = [ordered]@{schema=1;status='FAIL';failed_slot=$case.slot;slots=$records}
        $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
        throw "AMD Swin dependency execution failed at slot $($case.slot): $result"
    }
    $input = $output
}

$manifest = [ordered]@{
    schema = 1
    experiment = 'amd_swin_slots3_5_dependency_order'
    status = 'PASS'
    classification = 'REAL_AMD_NEURAL_DEPENDENCY_EXECUTION_PENDING_RTX_ORACLE'
    counts_as_s7 = $false
    device = 'AMD Radeon RX 9070 XT [ZLUDA]'
    initial_input_sha256 = (Get-FileHash -LiteralPath $N1Input -Algorithm SHA256).Hash
    captured_weight_source = $WeightDir
    slots = $records
    next_gate = 'repeat for determinism, then run original slots 3-5 over identical payloads on RTX 5070'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Write-Host ($manifest | ConvertTo-Json -Depth 8 -Compress)
Write-Host "Result: $result"
