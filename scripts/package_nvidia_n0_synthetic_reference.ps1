[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw 'probe build failed' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\n0_full_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
$sources = [ordered]@{
    'n0_reference_probe.exe' = Join-Path $repo 'build\zluda_ptx_probe.exe'
    'input_rgba16f.raw' = Join-Path $repo 'results\20260831_023943_rtx5070_feature18_n0\copy_input.raw'
    'weights.raw' = Join-Path $repo 'results\20260831_023943_rtx5070_feature18_n0\n0_weights_after.raw'
    'params.raw' = Join-Path $repo 'results\20260831_132000_n0_full_numeric_lowering\n0_params.raw'
}
foreach ($entry in $sources.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value)) { throw "missing source: $($entry.Value)" }
    Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $payload $entry.Key)
}
$originalPtx = Join-Path $repo 'results\20260831_011219_zluda_ptx_probe\neural_isolated.ptx'
if (-not (Test-Path -LiteralPath $originalPtx)) { throw "missing source: $originalPtx" }
$ptxText = [System.IO.File]::ReadAllText($originalPtx)
$versionPattern = [regex]::new('(?m)^\.version 9\.4\r?$')
if ($versionPattern.Matches($ptxText).Count -ne 1) {
    throw 'expected exactly one PTX 9.4 version directive'
}
$normalizedPtx = $versionPattern.Replace($ptxText, '.version 8.7', 1)
$packagedPtx = Join-Path $payload 'neural_sm120_ptx87.ptx'
[System.IO.File]::WriteAllText(
    $packagedPtx,
    $normalizedPtx,
    [System.Text.UTF8Encoding]::new($false)
)
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_n0_synthetic_reference.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'N0_REFERENCE_PACKAGE_README.md') -Destination (Join-Path $stage 'README.md')
$hashes = [ordered]@{}
foreach ($name in $sources.Keys) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
$hashes['neural_sm120_ptx87.ptx'] = (Get-FileHash -LiteralPath $packagedPtx -Algorithm SHA256).Hash
[ordered]@{
    schema = 1
    experiment = 'rtx_original_n0_same_input_reference_package'
    status = 'READY'
    classification = 'USER_OWNED_INTEROPERABILITY_PACKAGE'
    counts_as_s6 = $false
    expected_gpu = 'RTX 50 series / SM120'
    grid = @(80,48,1)
    block = @(32,1,1)
    ptx_header_normalization = [ordered]@{
        source_version = '9.4'
        packaged_version = '8.7'
        changed_directives = 1
        source_sha256 = (Get-FileHash -LiteralPath $originalPtx -Algorithm SHA256).Hash
        semantic_body_unchanged = $true
        reason = 'RTX driver returned CUDA_ERROR_UNSUPPORTED_PTX_VERSION for the captured 9.4 header'
    }
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
