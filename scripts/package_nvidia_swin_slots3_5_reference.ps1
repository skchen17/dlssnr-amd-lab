[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\swin_slots3_5_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')

$baseline = Join-Path $repo 'results\20260831_144000_swin1h_slots3_5_baseline'
$sources = [ordered]@{
    'chained.ptx' = Join-Path $baseline 'chained_original.ptx'
    'ds_wait.ptx' = Join-Path $baseline 'ds_wait_original.ptx'
}
foreach ($item in $sources.GetEnumerator()) {
    $text = [IO.File]::ReadAllText($item.Value)
    $normalized = $text -replace '(?m)^\.version 9\.4$', '.version 8.7'
    if ($normalized -eq $text) { throw "PTX 9.4 declaration not found: $($item.Value)" }
    [IO.File]::WriteAllText((Join-Path $payload $item.Key), $normalized, [Text.UTF8Encoding]::new($false))
}

$amd = Join-Path $repo 'results\20260831_152915_amd_swin_slots3_5'
$weights = Join-Path $repo 'results\20260831_151635_rtx5070_feature18_v17'
Copy-Item -LiteralPath (Join-Path $repo 'results\20260831_143501_amd_n1_slot2\output.raw') -Destination (Join-Path $payload 'slot3_input.raw')
Copy-Item -LiteralPath (Join-Path $amd 'slot3\output.raw') -Destination (Join-Path $payload 'slot4_input.raw')
Copy-Item -LiteralPath (Join-Path $amd 'slot4\output.raw') -Destination (Join-Path $payload 'slot5_input.raw')
foreach ($slot in 3..5) {
    Copy-Item -LiteralPath (Join-Path $weights "n0_slot${slot}_weights_before.raw") -Destination (Join-Path $payload "slot${slot}_weights.raw")
    Copy-Item -LiteralPath (Join-Path $amd "slot${slot}\params.raw") -Destination (Join-Path $payload "slot${slot}_params.raw")
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_swin_slots3_5_reference.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'SWIN_SLOTS3_5_REFERENCE_PACKAGE_README.md') -Destination (Join-Path $stage 'README.md')

$required = @('neural_reference_probe.exe','chained.ptx','ds_wait.ptx')
foreach ($slot in 3..5) { $required += @("slot${slot}_input.raw","slot${slot}_weights.raw","slot${slot}_params.raw") }
$hashes = [ordered]@{}
foreach ($name in $required) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
$sourceHashes = [ordered]@{}
foreach ($item in $sources.GetEnumerator()) { $sourceHashes[$item.Key] = (Get-FileHash -LiteralPath $item.Value -Algorithm SHA256).Hash }
[ordered]@{
    schema = 1
    experiment = 'rtx_swin_slots3_5_same_input_reference_package'
    normalization = '.version 9.4 -> .version 8.7 only'
    source_ptx_sha256 = $sourceHashes
    cases = @(
        [ordered]@{slot=3;entry='cc_tinlayout_fused_swin_1h_32_1_chained_fp8';grid=@(41,25,1);expected_releases=1025},
        [ordered]@{slot=4;entry='cc_tinlayout_fused_swin_1h_32_1_chained_fp8';grid=@(41,24,1);expected_releases=984},
        [ordered]@{slot=5;entry='cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8';grid=@(40,25,1);expected_releases=0;extra_output=$true}
    )
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
