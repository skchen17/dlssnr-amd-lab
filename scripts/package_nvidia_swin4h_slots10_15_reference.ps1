[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "probe build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\swin4h_slots10_15_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
Copy-Item (Join-Path $repo 'build\zluda_ptx_probe.exe') (Join-Path $payload 'neural_reference_probe.exe')
$sources = [ordered]@{
    'inpview.ptx' = Join-Path $repo 'results\20260831_172000_swin4h_slots10_15_baseline\inpview_original.ptx'
    'chained.ptx' = Join-Path $repo 'results\20260831_172000_swin4h_slots10_15_baseline\chained_original.ptx'
    'ds_wait.ptx' = Join-Path $repo 'results\20260831_172000_swin4h_slots10_15_baseline\ds_wait_original.ptx'
}
foreach ($item in $sources.GetEnumerator()) {
    $text = [IO.File]::ReadAllText($item.Value)
    $normalized = $text -replace '(?m)^\.version 9\.4$', '.version 8.7'
    if ($normalized -eq $text) { throw "PTX version declaration not found: $($item.Value)" }
    [IO.File]::WriteAllText((Join-Path $payload $item.Key),$normalized,[Text.UTF8Encoding]::new($false))
}
$amd = Join-Path $repo 'results\20260831_172222_amd_swin4h_slots10_15_formal_v2'
$weights = Join-Path $repo 'results\20260831_163553_rtx5070_feature18_v19'
Copy-Item (Join-Path $repo 'results\20260831_164100_amd_swin2h_slot9\extra_output.raw') (Join-Path $payload 'slot10_input.raw')
foreach ($slot in 11..15) {
    $previous = $slot - 1
    Copy-Item (Join-Path $amd "slot${previous}\output.raw") (Join-Path $payload "slot${slot}_input.raw")
}
foreach ($slot in 10..15) {
    Copy-Item (Join-Path $weights "n0_slot${slot}_weights_before.raw") (Join-Path $payload "slot${slot}_weights.raw")
    Copy-Item (Join-Path $amd "slot${slot}\params.raw") (Join-Path $payload "slot${slot}_params.raw")
}
Copy-Item (Join-Path $PSScriptRoot 'run_nvidia_swin4h_slots10_15_reference.ps1') $scripts
Copy-Item (Join-Path $repo 'SWIN4H_SLOTS10_15_REFERENCE_PACKAGE_README.md') (Join-Path $stage 'README.md')
$required = @('neural_reference_probe.exe','inpview.ptx','chained.ptx','ds_wait.ptx')
foreach($slot in 10..15){$required += @("slot${slot}_input.raw","slot${slot}_weights.raw","slot${slot}_params.raw")}
$hashes=[ordered]@{};foreach($name in $required){$hashes[$name]=(Get-FileHash (Join-Path $payload $name)-Algorithm SHA256).Hash}
$sourceHashes=[ordered]@{};foreach($item in $sources.GetEnumerator()){$sourceHashes[$item.Key]=(Get-FileHash $item.Value -Algorithm SHA256).Hash}
[ordered]@{schema=1;experiment='rtx_swin4h_slots10_15_same_input_reference_package';normalization='.version 9.4 -> .version 8.7 only';source_ptx_sha256=$sourceHashes;cases=@([ordered]@{slot=10;grid=@(10,6,1);block=@(32,4,1);expected_releases=60},[ordered]@{slot=11;grid=@(11,7,1);block=@(32,4,1);expected_releases=77},[ordered]@{slot=12;grid=@(11,6,1);block=@(32,4,1);expected_releases=66},[ordered]@{slot=13;grid=@(10,7,1);block=@(32,4,1);expected_releases=70},[ordered]@{slot=14;grid=@(10,6,1);block=@(32,4,1);expected_releases=60},[ordered]@{slot=15;grid=@(11,7,1);block=@(32,4,1);expected_releases=0;extra_output=$true});payload_sha256=$hashes} |
    ConvertTo-Json -Depth 8 | Set-Content (Join-Path $stage 'manifest.json') -Encoding utf8
$archive="$stage.zip";Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive";Write-Host "SHA256: $((Get-FileHash $archive -Algorithm SHA256).Hash)"
