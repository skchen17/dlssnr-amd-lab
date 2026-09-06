[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only nvidia_fp8_reference
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "build\fp8_reference_v14_$stamp"
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\nvidia_fp8_reference.exe') `
    -Destination (Join-Path $stage 'nvidia_fp8_reference.exe')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_fp8_reference.ps1') `
    -Destination (Join-Path $stage 'run_nvidia_fp8_reference.ps1')
$files = Get-ChildItem -LiteralPath $stage -File | Sort-Object Name | ForEach-Object {
    [ordered]@{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
}
$packageManifest = [ordered]@{
    schema = 1
    package_revision = 'v14_rtx_fp16_e4m3_exhaustive'
    files = @($files)
}
$packageManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $stage 'package_manifest.json') -Encoding utf8
$deliverables = Join-Path $repo 'deliverables'
New-Item -ItemType Directory -Force -Path $deliverables | Out-Null
$zip = Join-Path $deliverables "dlssnr-windows-fp8-reference-v14-$stamp.zip"
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal -Force
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
