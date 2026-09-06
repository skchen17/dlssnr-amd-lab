[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only nvidia_remaining_reference
if($LASTEXITCODE-ne 0){throw "build failed: $LASTEXITCODE"}
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss';$stage=Join-Path $repo "build\remaining_reference_v16_$stamp"
New-Item -ItemType Directory -Path $stage|Out-Null
Copy-Item (Join-Path $repo 'build\nvidia_remaining_reference.exe') (Join-Path $stage 'nvidia_remaining_reference.exe')
Copy-Item (Join-Path $PSScriptRoot 'run_nvidia_remaining_reference.ps1') (Join-Path $stage 'run_nvidia_remaining_reference.ps1')
$files=Get-ChildItem $stage -File|Sort-Object Name|ForEach-Object{[ordered]@{name=$_.Name;bytes=$_.Length;sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash}}
[ordered]@{schema=1;package_revision='v16_rtx_remaining_n0_primitives';files=@($files)}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $stage 'package_manifest.json') -Encoding utf8
$deliverables=Join-Path $repo 'deliverables';New-Item -ItemType Directory -Force -Path $deliverables|Out-Null
$zip=Join-Path $deliverables "dlssnr-windows-remaining-reference-v16-$stamp.zip";Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal -Force
Write-Host "Package: $zip";Write-Host "SHA256: $((Get-FileHash $zip -Algorithm SHA256).Hash)"
