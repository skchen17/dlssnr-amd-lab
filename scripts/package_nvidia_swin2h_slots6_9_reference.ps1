[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\swin2h_slots6_9_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
Copy-Item (Join-Path $repo 'build\zluda_ptx_probe.exe') (Join-Path $payload 'neural_reference_probe.exe')
$sources = [ordered]@{
    'inpview.ptx' = Join-Path $repo 'results\20260831_154500_swin2h_slot6_baseline\slot6_original.ptx'
    'chained.ptx' = Join-Path $repo 'results\20260831_163000_swin2h_slots7_9_baseline\chained_original.ptx'
    'ds_wait.ptx' = Join-Path $repo 'results\20260831_163000_swin2h_slots7_9_baseline\ds_wait_original.ptx'
}
foreach ($item in $sources.GetEnumerator()) {
    $text = [IO.File]::ReadAllText($item.Value)
    $normalized = $text -replace '(?m)^\.version 9\.4$', '.version 8.7'
    if ($normalized -eq $text) { throw "PTX version declaration not found: $($item.Value)" }
    [IO.File]::WriteAllText((Join-Path $payload $item.Key),$normalized,[Text.UTF8Encoding]::new($false))
}
$amd = Join-Path $repo 'results\20260831_163147_amd_swin2h_slots6_8_formal'
$slot9 = Join-Path $repo 'results\20260831_164100_amd_swin2h_slot9'
$weights = Join-Path $repo 'results\20260831_163553_rtx5070_feature18_v19'
Copy-Item (Join-Path $repo 'results\20260831_152915_amd_swin_slots3_5\slot5\extra_output.raw') (Join-Path $payload 'slot6_input.raw')
Copy-Item (Join-Path $amd 'slot6\output.raw') (Join-Path $payload 'slot7_input.raw')
Copy-Item (Join-Path $amd 'slot7\output.raw') (Join-Path $payload 'slot8_input.raw')
Copy-Item (Join-Path $amd 'slot8\output.raw') (Join-Path $payload 'slot9_input.raw')
foreach ($slot in 6..9) {
    Copy-Item (Join-Path $weights "n0_slot${slot}_weights_before.raw") (Join-Path $payload "slot${slot}_weights.raw")
    $paramSource = if ($slot -eq 9) { Join-Path $slot9 'params.raw' } else { Join-Path $amd "slot${slot}\params.raw" }
    Copy-Item $paramSource (Join-Path $payload "slot${slot}_params.raw")
}
Copy-Item (Join-Path $PSScriptRoot 'run_nvidia_swin2h_slots6_9_reference.ps1') $scripts
Copy-Item (Join-Path $repo 'SWIN2H_SLOTS6_9_REFERENCE_PACKAGE_README.md') (Join-Path $stage 'README.md')
$required = @('neural_reference_probe.exe','inpview.ptx','chained.ptx','ds_wait.ptx')
foreach($slot in 6..9){$required += @("slot${slot}_input.raw","slot${slot}_weights.raw","slot${slot}_params.raw")}
$hashes=[ordered]@{};foreach($name in $required){$hashes[$name]=(Get-FileHash (Join-Path $payload $name)-Algorithm SHA256).Hash}
$sourceHashes=[ordered]@{};foreach($item in $sources.GetEnumerator()){$sourceHashes[$item.Key]=(Get-FileHash $item.Value -Algorithm SHA256).Hash}
[ordered]@{schema=1;experiment='rtx_swin2h_slots6_9_same_input_reference_package';normalization='.version 9.4 -> .version 8.7 only';source_ptx_sha256=$sourceHashes;cases=@([ordered]@{slot=6;grid=@(20,12,1);block=@(32,2,1);expected_releases=240},[ordered]@{slot=7;grid=@(21,13,1);block=@(32,2,1);expected_releases=273},[ordered]@{slot=8;grid=@(21,12,1);block=@(32,2,1);expected_releases=252},[ordered]@{slot=9;grid=@(20,13,1);block=@(32,2,1);expected_releases=0;extra_output=$true});payload_sha256=$hashes} |
    ConvertTo-Json -Depth 8 | Set-Content (Join-Path $stage 'manifest.json') -Encoding utf8
$archive="$stage.zip";Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive";Write-Host "SHA256: $((Get-FileHash $archive -Algorithm SHA256).Hash)"
