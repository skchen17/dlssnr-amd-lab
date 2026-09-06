[CmdletBinding()]
param(
    [string]$CaptureRoot = 'results\20260904_660000_frame_aligned_post_surface_rtx5070',
    [string]$ReferenceRoot = 'results\20260904_640000_frame_aligned_post_output_rtx5070',
    [string]$LoweringRoot = 'results\20260831_232000_decoder_slots99_154_lowering_v4'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$capture = (Resolve-Path -LiteralPath (Join-Path $repo $CaptureRoot)).Path
$reference = (Resolve-Path -LiteralPath (Join-Path $repo $ReferenceRoot)).Path
$lowering = (Resolve-Path -LiteralPath (Join-Path $repo $LoweringRoot)).Path
$plan = Join-Path $repo 'results\20260831_234000_full_graph_integrated_plan'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\postblock_exact_contract_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null

Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') `
    -Destination (Join-Path $payload 'neural_reference_probe.exe')
$stageSpecs = @(
    [ordered]@{name='original'; order=0; source='post_block_mbarrier.ptx'},
    [ordered]@{name='compat'; order=1; source='post_block_compat.ptx'},
    [ordered]@{name='e4m3'; order=2; source='post_block_e4m3.ptx'},
    [ordered]@{name='movmatrix'; order=3; source='post_block_mov.ptx'}
)
foreach ($item in $stageSpecs) {
    $item.file = "$($item.name).ptx"
    $text = [IO.File]::ReadAllText((Join-Path $lowering $item.source))
    $normalized = $text -replace '(?m)^\.version 9\.4$','.version 8.7'
    if ($normalized -eq $text) { throw "PTX version declaration not found: $($item.source)" }
    [IO.File]::WriteAllText((Join-Path $payload $item.file),$normalized,
        [Text.UTF8Encoding]::new($false))
}
Copy-Item -LiteralPath (Join-Path $capture 'post_activation_arena_prelaunch.raw') `
    -Destination (Join-Path $payload 'activation_arena.raw')
Copy-Item -LiteralPath (Join-Path $capture 'frame1_post_surface_pre_slot154.raw') `
    -Destination (Join-Path $payload 'surface_initial.raw')
Copy-Item -LiteralPath (Join-Path $reference 'frame1_post_output_pre_copy.raw') `
    -Destination (Join-Path $payload 'output_reference.raw')
Copy-Item -LiteralPath (Join-Path $capture 'post_texture_input.raw') `
    -Destination (Join-Path $payload 'texture_input.raw')
Copy-Item -LiteralPath (Join-Path $plan 'model_arena.raw') -Destination $payload

$launch = Get-Content -LiteralPath (Join-Path $capture 'module_trace.jsonl') |
    ForEach-Object { try { $_ | ConvertFrom-Json -ErrorAction Stop } catch {} } |
    Where-Object { $_.ev -eq 'nvapi_launch_cu_kernel' -and $_.frame -eq 1 -and $_.slot -eq 154 } |
    Select-Object -First 1
if (-not $launch -or $launch.param_captured -ne 184 -or $launch.param_hex.Length -ne 368) {
    throw 'exact frame-1 slot154 parameter block missing from capture'
}
[byte[]]$params = for ($index=0; $index -lt $launch.param_hex.Length; $index+=2) {
    [Convert]::ToByte($launch.param_hex.Substring($index,2),16)
}
[IO.File]::WriteAllBytes((Join-Path $payload 'params.raw'),$params)
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_postblock_exact_contract_sweep.ps1') `
    -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'POSTBLOCK_EXACT_CONTRACT_SWEEP_README.md') `
    -Destination (Join-Path $stage 'README.md')

$hashes = [ordered]@{}
foreach ($file in Get-ChildItem -LiteralPath $payload -File | Sort-Object Name) {
    $hashes[$file.Name] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
}
[ordered]@{
    schema = 1
    package_revision = 'v2_postblock_exact_frame1_contract_zero_texture_fix'
    experiment = 'rtx_postblock_exact_frame1_contract_stage_sweep_package'
    function = 'cc_tinlayout_fused_post_block_swin_1h_32_fp8'
    grid = @(81,49,1)
    block = @(32,1,1)
    activation_bytes = (Get-Item (Join-Path $payload 'activation_arena.raw')).Length
    logical_output_bytes = 1843200
    surface_initial_sha256 = $hashes['surface_initial.raw']
    output_reference_sha256 = $hashes['output_reference.raw']
    texture_input_sha256 = $hashes['texture_input.raw']
    texture_upload_source = 'dedicated_zero_buffer'
    reference_capture = 'v23_unperturbed_post_slot154'
    stages = @($stageSpecs | ForEach-Object {
        [ordered]@{name=$_.name; order=$_.order; file=$_.file; sha256=$hashes[$_.file]}
    })
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$zip = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
