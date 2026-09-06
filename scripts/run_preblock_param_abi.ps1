$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$entry = 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'
$frameCsv = Join-Path $repo 'results\20260831_002356_rtx5070_feature18_full_frame\frame_001_sequence.csv'
$ptx = Join-Path $repo 'results\20260831_010100_all_runtime_modules\module_00_000DF0E0.ptx'
if (-not (Test-Path -LiteralPath $ptx)) {
    throw "Local ignored PTX extraction is missing: $ptx"
}
$row = Import-Csv -LiteralPath $frameCsv | Where-Object { $_.function_name -eq $entry }
if (@($row).Count -ne 1) { throw "Expected exactly one frame row for $entry" }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_preblock_param_abi"
New-Item -ItemType Directory -Path $result | Out-Null
$json = Join-Path $result 'preblock_param_abi.json'
& python (Join-Path $PSScriptRoot 'analyze_ptx_param_abi.py') `
    --ptx $ptx --entry $entry --param-hex $row.frame1_param_hex --output $json
if ($LASTEXITCODE -ne 0) { throw "parameter ABI analysis failed: $LASTEXITCODE" }

$manifest = [ordered]@{
    schema = 1
    experiment = 'preblock_param_abi'
    classification = 'STATIC_ABI_METADATA_ONLY'
    counts_as_s6 = $false
    frame_csv_sha256 = (Get-FileHash -LiteralPath $frameCsv -Algorithm SHA256).Hash
    ptx_sha256 = (Get-FileHash -LiteralPath $ptx -Algorithm SHA256).Hash
    analyzer_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'analyze_ptx_param_abi.py') -Algorithm SHA256).Hash
    output_sha256 = (Get-FileHash -LiteralPath $json -Algorithm SHA256).Hash
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Write-Host "Result: $result"
