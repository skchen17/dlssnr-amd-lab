[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only nvidia_mma_reference
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "build\mma_reference_v15_$stamp"
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\nvidia_mma_reference.exe') `
    -Destination (Join-Path $stage 'nvidia_mma_reference.exe')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_mma_reference.ps1') `
    -Destination (Join-Path $stage 'run_nvidia_mma_reference.ps1')
$files = Get-ChildItem -LiteralPath $stage -File | Sort-Object Name | ForEach-Object {
    [ordered]@{ name = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
}
[ordered]@{
    schema = 1
    package_revision = 'v15_rtx_m16n8k32_e4m3_mma'
    files = @($files)
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath `
    (Join-Path $stage 'package_manifest.json') -Encoding utf8
$deliverables = Join-Path $repo 'deliverables'
New-Item -ItemType Directory -Force -Path $deliverables | Out-Null
$zip = Join-Path $deliverables "dlssnr-windows-mma-reference-v15-$stamp.zip"
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal -Force
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
