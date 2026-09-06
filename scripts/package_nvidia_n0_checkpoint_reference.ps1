[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw 'probe build failed' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stage = Join-Path $repo "deliverables\n0_checkpoint_reference_$stamp"
$payload = Join-Path $stage 'payload'
$scripts = Join-Path $stage 'scripts'
New-Item -ItemType Directory -Path $payload,$scripts | Out-Null
$sources = [ordered]@{
    'n0_reference_probe.exe' = Join-Path $repo 'build\zluda_ptx_probe.exe'
    'n0_pre_mma.ptx' = Join-Path $repo 'results\20260831_133700_amd_n0_pre_mma_checkpoint\n0_pre_mma.ptx'
    'input_rgba16f.raw' = Join-Path $repo 'results\20260831_023943_rtx5070_feature18_n0\copy_input.raw'
    'weights.raw' = Join-Path $repo 'results\20260831_023943_rtx5070_feature18_n0\n0_weights_after.raw'
    'params.raw' = Join-Path $repo 'results\20260831_132000_n0_full_numeric_lowering\n0_params.raw'
}
foreach ($entry in $sources.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $entry.Value)) { throw "missing source: $($entry.Value)" }
    Copy-Item -LiteralPath $entry.Value -Destination (Join-Path $payload $entry.Key)
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_n0_checkpoint_reference.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'N0_CHECKPOINT_PACKAGE_README.md') -Destination (Join-Path $stage 'README.md')
$hashes = [ordered]@{}
foreach ($name in $sources.Keys) { $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash }
[ordered]@{
    schema = 1
    experiment = 'rtx_n0_pre_mma_checkpoint_package'
    status = 'READY'
    classification = 'USER_OWNED_INTEROPERABILITY_PACKAGE'
    counts_as_s6 = $false
    expected_gpu = 'RTX 50 series / SM120'
    checkpoint = 'immediately before first f16 MMA'
    grid = @(1,1,1)
    block = @(32,1,1)
    bytes_per_lane = 96
    lanes = 32
    payload_sha256 = $hashes
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding utf8
$archive = "$stage.zip"
Compress-Archive -LiteralPath $stage -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
