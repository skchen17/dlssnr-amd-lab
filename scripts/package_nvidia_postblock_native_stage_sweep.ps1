[CmdletBinding()]
param([string]$LoweringRoot = 'results\20260831_232000_decoder_slots99_154_lowering_v4')

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$lowering = if ([IO.Path]::IsPathRooted($LoweringRoot)) {
    (Resolve-Path -LiteralPath $LoweringRoot).Path
} else { (Resolve-Path -LiteralPath (Join-Path $repo $LoweringRoot)).Path }
$case = Join-Path $repo 'results\20260831_230000_decoder_full_graph_exact_state\cases\slot154'
$cases = Split-Path -Parent $case
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\postblock_native_stage_sweep_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null

Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
$stageSpecs = @(
    [ordered]@{name='original'; order=0; source='post_block_mbarrier.ptx'},
    [ordered]@{name='compat'; order=1; source='post_block_compat.ptx'},
    [ordered]@{name='e4m3'; order=2; source='post_block_e4m3.ptx'},
    [ordered]@{name='movmatrix'; order=3; source='post_block_mov.ptx'},
    [ordered]@{name='fp8_mma'; order=4; source='post_block_fp8.ptx'},
    [ordered]@{name='f16_mma'; order=5; source='post_block_f16.ptx'},
    [ordered]@{name='decoder_compat'; order=6; source='post_block_decoder_compat.ptx'}
)
foreach ($item in $stageSpecs) {
    $item.file = "$($item.name).ptx"
    $text = [IO.File]::ReadAllText((Join-Path $lowering $item.source))
    $normalized = $text -replace '(?m)^\.version 9\.4$','.version 8.7'
    if ($normalized -eq $text) { throw "PTX version declaration not found: $($item.source)" }
    [IO.File]::WriteAllText((Join-Path $payload $item.file),$normalized,[Text.UTF8Encoding]::new($false))
}
Copy-Item -LiteralPath (Join-Path $case 'activation_arena_initial.raw') -Destination $payload
Copy-Item -LiteralPath (Join-Path $cases 'model_arena.raw') -Destination $payload
Copy-Item -LiteralPath (Join-Path $case 'params.raw') -Destination $payload
Copy-Item -LiteralPath (Join-Path $case 'output0_reference.raw') -Destination (Join-Path $payload 'output_reference.raw')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_postblock_native_stage_sweep.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'POSTBLOCK_NATIVE_STAGE_SWEEP_README.md') -Destination (Join-Path $stage 'README.md')

$hashes = [ordered]@{}
foreach ($file in Get-ChildItem -LiteralPath $payload -File | Sort-Object Name) {
    $hashes[$file.Name] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
}
[ordered]@{
    schema = 1
    experiment = 'rtx_postblock_native_resource_lowering_stage_sweep_package'
    function = 'cc_tinlayout_fused_post_block_swin_1h_32_fp8'
    logical_output_bytes = 1843200
    output_reference_sha256 = $hashes['output_reference.raw']
    stages = @($stageSpecs | ForEach-Object {
        [ordered]@{name=$_.name; order=$_.order; file=$_.file; sha256=$hashes[$_.file]}
    })
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$zip = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
