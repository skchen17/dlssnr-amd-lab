[CmdletBinding()]
param(
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputDir = Join-Path $repo "results\${stamp}_vit1d_slots57_98_lowering"
}
$output = [IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $output) { throw "Output directory already exists: $output" }
New-Item -ItemType Directory -Path $output | Out-Null
$sourceRoot = Join-Path $repo 'results\20260831_175500_full_graph_ptx_access'

$entries = @(
    [ordered]@{tag='attention';name='cc_vit_1d_attention_chained_fp8';init=2;bulk=4;wide=4;release=2;relaxed=3;shared=12;e4m3=104;mov=0;fp8=64;vred=0;fence=0},
    [ordered]@{tag='ffn_contract';name='cc_vit_1d_ffn_contract_chained_fp8';init=2;bulk=12;wide=24;release=2;relaxed=3;shared=24;e4m3=80;mov=0;fp8=128;vred=16;fence=0},
    [ordered]@{tag='ffn_expand';name='cc_vit_1d_ffn_expand_chained_fp8';init=3;bulk=24;wide=8;release=1;relaxed=3;shared=24;e4m3=80;mov=0;fp8=64;vred=0;fence=0},
    [ordered]@{tag='ffn_expand_publish';name='cc_vit_1d_ffn_expand_publish_fp8';init=3;bulk=24;wide=8;release=1;relaxed=0;shared=24;e4m3=80;mov=0;fp8=64;vred=0;fence=0},
    [ordered]@{tag='projection';name='cc_vit_1d_projection_chained_fp8';init=2;bulk=4;wide=24;release=2;relaxed=3;shared=12;e4m3=76;mov=0;fp8=64;vred=16;fence=0},
    [ordered]@{tag='projection_wait';name='cc_vit_1d_projection_wait_fp8';init=2;bulk=4;wide=24;release=1;relaxed=3;shared=12;e4m3=76;mov=0;fp8=64;vred=16;fence=0},
    [ordered]@{tag='qkv';name='cc_vit_1d_qkv_chained_fp8';init=2;bulk=4;wide=36;release=2;relaxed=3;shared=12;e4m3=100;mov=32;fp8=96;vred=0;fence=1},
    [ordered]@{tag='repack_1d_to_2d';name='cc_vit_1d_repack_1d_to_2d_fp8';init=0;bulk=0;wide=0;release=0;relaxed=0;shared=0;e4m3=0;mov=0;fp8=0;vred=0;fence=0},
    [ordered]@{tag='repack_2d_to_1d';name='cc_vit_1d_repack_2d_to_1d_fp8';init=0;bulk=0;wide=0;release=0;relaxed=0;shared=0;e4m3=0;mov=0;fp8=0;vred=0;fence=0}
)

foreach ($entry in $entries) {
    $source = Join-Path $sourceRoot ($entry.name + '.ptx')
    $barrier = Join-Path $output ($entry.tag + '_mbarrier.ptx')
    $compat = Join-Path $output ($entry.tag + '_compat.ptx')
    $e4m3 = Join-Path $output ($entry.tag + '_e4m3.ptx')
    $mov = Join-Path $output ($entry.tag + '_movmatrix.ptx')
    $numeric = Join-Path $output ($entry.tag + '_numeric.ptx')
    $full = Join-Path $output ($entry.tag + '_full.ptx')
    python (Join-Path $PSScriptRoot 'lower_ptx_mbarrier.py') $source $barrier `
        (Join-Path $output ($entry.tag + '_mbarrier.json')) `
        --expected-init $entry.init --expected-bulk $entry.bulk `
        --expected-elect $entry.bulk --expected-arrive $(if ($entry.init) {2} else {0}) `
        --expected-try-wait $(if ($entry.init) {2} else {0})
    if ($LASTEXITCODE -ne 0) { throw "mbarrier lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_zluda_compat.py') $barrier $compat `
        (Join-Path $output ($entry.tag + '_compat.json')) `
        --expected-discard 0 --expected-wide-store $entry.wide `
        --expected-release-store $entry.release --expected-relaxed-load $entry.relaxed `
        --expected-shared-cta-scope $entry.shared
    if ($LASTEXITCODE -ne 0) { throw "compatibility lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_e4m3.py') $compat $e4m3 `
        (Join-Path $output ($entry.tag + '_e4m3.json')) --expected-count $entry.e4m3
    if ($LASTEXITCODE -ne 0) { throw "E4M3 lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_movmatrix.py') $e4m3 $mov `
        (Join-Path $output ($entry.tag + '_movmatrix.json')) --expected-count $entry.mov
    if ($LASTEXITCODE -ne 0) { throw "movmatrix lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_fp8_mma.py') $mov $numeric `
        (Join-Path $output ($entry.tag + '_fp8.json')) --expected-count $entry.fp8 --compact
    if ($LASTEXITCODE -ne 0) { throw "FP8 MMA lowering failed: $($entry.name)" }
    python (Join-Path $PSScriptRoot 'lower_ptx_vit_compat.py') $numeric $full `
        (Join-Path $output ($entry.tag + '_vit_compat.json')) `
        --expected-vector-red $entry.vred --expected-release-fence $entry.fence
    if ($LASTEXITCODE -ne 0) { throw "ViT compatibility lowering failed: $($entry.name)" }
}

$reports = Get-ChildItem -LiteralPath $output -Filter '*.json' | ForEach-Object {
    Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
}
$manifest = [ordered]@{
    schema = 1
    experiment = 'vit1d_slots57_98_strict_lowering'
    status = if (($reports | Where-Object {$_.status -ne 'PASS'}).Count -eq 0 -and $reports.Count -eq 54) {'PASS'} else {'FAIL'}
    entry_count = $entries.Count
    report_count = $reports.Count
    source_module_fnv1a64 = 'DA569AC83D180C34'
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $output 'manifest.json')
Write-Host "Result: $output"
if ($manifest.status -ne 'PASS') { exit 1 }
