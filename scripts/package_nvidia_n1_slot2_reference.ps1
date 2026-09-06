[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\n1_slot2_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Force -Path $payload,$scripts | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'n1_reference_probe.exe')
$sourcePtx = Join-Path $repo 'results\20260831_142000_n1_slot2_baseline\n1_original.ptx'
$ptxText = [System.IO.File]::ReadAllText($sourcePtx)
$normalized = $ptxText -replace '(?m)^\.version 9\.4$', '.version 8.7'
if ($normalized -eq $ptxText) { throw 'PTX 9.4 declaration was not found' }
[System.IO.File]::WriteAllText((Join-Path $payload 'n1_slot2.ptx'), $normalized, [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $repo 'results\20260831_130611_rtx5070_n0_same_input\output.raw') -Destination (Join-Path $payload 'input.raw')
Copy-Item -LiteralPath (Join-Path $repo 'deliverables\n0_full_reference_20260831_130219\payload\weights.raw') -Destination (Join-Path $payload 'weights.raw')
Copy-Item -LiteralPath (Join-Path $repo 'results\20260831_143501_amd_n1_slot2\params.raw') -Destination (Join-Path $payload 'params.raw')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_n1_slot2_reference.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'N1_REFERENCE_PACKAGE_README.md') -Destination (Join-Path $stage 'README.md')
$required = @('n1_reference_probe.exe','n1_slot2.ptx','input.raw','weights.raw','params.raw')
$hashes = [ordered]@{}
foreach ($name in $required) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
[ordered]@{
    schema = 1
    experiment = 'rtx_n1_slot2_reference_package'
    source_entry = 'cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8'
    source_ptx_sha256 = (Get-FileHash -LiteralPath $sourcePtx -Algorithm SHA256).Hash
    normalization = '.version 9.4 -> .version 8.7 only'
    grid = @(40,24,1)
    block = @(32,1,1)
    weights_view_offset = 22016
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
