[CmdletBinding()]
param(
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputDir = Join-Path $repo "results\${stamp}_split16h_slots24_56_lowering"
}
$output = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $output) { throw "Output directory already exists: $output" }
New-Item -ItemType Directory -Path $output | Out-Null
$sourceRoot = Join-Path $repo 'results\20260831_175500_full_graph_ptx_access'

$entries = @(
    [ordered]@{tag='ffwd';name='cc_split_swin_16h_ffwd_512_chained_fp8';mbarrier=2;bulk=2;wide=4;release=1;relaxed=2;shared=10;e4m3=82;mov=0;fp8=96},
    [ordered]@{tag='ffwd_inpview';name='cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8';mbarrier=2;bulk=0;wide=8;release=1;relaxed=0;shared=16;e4m3=160;mov=0;fp8=192},
    [ordered]@{tag='ffwd_proj';name='cc_split_swin_16h_ffwd_proj_512_chained_fp8';mbarrier=3;bulk=8;wide=8;release=1;relaxed=2;shared=16;e4m3=80;mov=0;fp8=64},
    [ordered]@{tag='ffwd_proj_inpview';name='cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8';mbarrier=3;bulk=8;wide=8;release=1;relaxed=2;shared=16;e4m3=72;mov=0;fp8=64},
    [ordered]@{tag='final_head';name='cc_split_swin_16h_final_head_512_wait_fp8';mbarrier=3;bulk=4;wide=4;release=0;relaxed=2;shared=8;e4m3=36;mov=0;fp8=32},
    [ordered]@{tag='proj';name='cc_split_swin_16h_proj_512_chained_fp8';mbarrier=3;bulk=4;wide=4;release=1;relaxed=3;shared=8;e4m3=40;mov=0;fp8=32},
    [ordered]@{tag='proj_pool';name='cc_split_swin_16h_proj_pool_512_chained_fp8';mbarrier=2;bulk=4;wide=10;release=1;relaxed=2;shared=20;e4m3=92;mov=0;fp8=128},
    [ordered]@{tag='qkv';name='cc_split_swin_16h_qkv_512_chained_fp8';mbarrier=2;bulk=4;wide=4;release=1;relaxed=2;shared=20;e4m3=196;mov=32;fp8=256}
)

foreach ($entry in $entries) {
    $source = Join-Path $sourceRoot ($entry.name + '.ptx')
    $barrier = Join-Path $output ($entry.tag + '_mbarrier.ptx')
    $compat = Join-Path $output ($entry.tag + '_compat.ptx')
    $e4m3 = Join-Path $output ($entry.tag + '_e4m3.ptx')
    $mov = Join-Path $output ($entry.tag + '_movmatrix.ptx')
    $full = Join-Path $output ($entry.tag + '_full.ptx')
    python (Join-Path $PSScriptRoot 'lower_ptx_mbarrier.py') $source $barrier `
        (Join-Path $output ($entry.tag + '_mbarrier.json')) `
        --expected-init $entry.mbarrier --expected-bulk $entry.bulk `
        --expected-elect $entry.bulk `
        --expected-arrive 2 --expected-try-wait 2
    if ($LASTEXITCODE -ne 0) { throw "mbarrier lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_zluda_compat.py') $barrier $compat `
        (Join-Path $output ($entry.tag + '_compat.json')) `
        --expected-discard 0 --expected-wide-store $entry.wide `
        --expected-release-store $entry.release --expected-relaxed-load $entry.relaxed `
        --expected-shared-cta-scope $entry.shared
    if ($LASTEXITCODE -ne 0) { throw "Compatibility lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_e4m3.py') $compat $e4m3 `
        (Join-Path $output ($entry.tag + '_e4m3.json')) --expected-count $entry.e4m3
    if ($LASTEXITCODE -ne 0) { throw "E4M3 lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_movmatrix.py') $e4m3 $mov `
        (Join-Path $output ($entry.tag + '_movmatrix.json')) --expected-count $entry.mov
    if ($LASTEXITCODE -ne 0) { throw "movmatrix lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_fp8_mma.py') $mov $full `
        (Join-Path $output ($entry.tag + '_fp8.json')) --expected-count $entry.fp8 --compact
    if ($LASTEXITCODE -ne 0) { throw "FP8 MMA lowering failed: $($entry.name)" }
}

$reports = Get-ChildItem -LiteralPath $output -Filter '*.json' | ForEach-Object {
    Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
}
$manifest = [ordered]@{
    schema = 1
    experiment = 'split16h_slots24_56_strict_lowering'
    status = if (($reports | Where-Object {$_.status -ne 'PASS'}).Count -eq 0 -and $reports.Count -eq 40) {'PASS'} else {'FAIL'}
    entry_count = $entries.Count
    report_count = $reports.Count
    source_module_fnv1a64 = '19B0B8C47C5A7007'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $output 'manifest.json')
Write-Host "Result: $output"
if ($manifest.status -ne 'PASS') { exit 1 }
