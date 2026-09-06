[CmdletBinding()]
param(
    [ValidateSet('01','02','02.5')][string]$RenoDxVariant = '02',
    [ValidateRange(1048576,67108864)][uint64]$MaxCaptureBytes = 8388608,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path -Parent $PSScriptRoot
$packageManifestPath = Join-Path $root 'PACKAGE_MANIFEST.json'
$packageIntegrity = $true
$packageErrors = @()
if (-not (Test-Path -LiteralPath $packageManifestPath -PathType Leaf)) {
    throw "PACKAGE_MANIFEST.json is missing: $packageManifestPath"
}
$packageManifest = Get-Content -LiteralPath $packageManifestPath -Raw | ConvertFrom-Json
foreach ($record in $packageManifest.files) {
    $path = Join-Path $root ($record.path -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $packageIntegrity = $false
        $packageErrors += "missing: $($record.path)"
        continue
    }
    $item = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.Length -ne [int64]$record.size -or $hash -ne [string]$record.sha256) {
        $packageIntegrity = $false
        $packageErrors += "hash/size mismatch: $($record.path)"
    }
}
if (-not $packageIntegrity) {
    throw "Package integrity failed: $($packageErrors -join '; ')"
}
if ($ValidateOnly) {
    Write-Host "Package integrity: PASS ($($packageManifest.files.Count) files)"
    exit 0
}

$drive = Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))
if ($drive.Free -lt 30GB) {
    throw "At least 30 GB free space is required; available: $([math]::Round($drive.Free / 1GB, 2)) GB"
}

$resultRoot = Join-Path $root 'feature18_results'
$before = @{}
if (Test-Path -LiteralPath $resultRoot) {
    foreach ($dir in Get-ChildItem -LiteralPath $resultRoot -Directory) { $before[$dir.FullName] = $true }
}
$oldFullCapture = $env:MODULE_TRACE_FULL_GRAPH_CAPTURE
$oldCaptureLimit = $env:MODULE_TRACE_FULL_GRAPH_MAX_BYTES
$featureExit = -1
try {
    $env:MODULE_TRACE_FULL_GRAPH_CAPTURE = '1'
    $env:MODULE_TRACE_FULL_GRAPH_MAX_BYTES = [string]$MaxCaptureBytes
    & powershell -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $PSScriptRoot 'run_feature18_reference.ps1') `
        -RenoDxVariant $RenoDxVariant -SkipArchive
    $featureExit = $LASTEXITCODE
} finally {
    $env:MODULE_TRACE_FULL_GRAPH_CAPTURE = $oldFullCapture
    $env:MODULE_TRACE_FULL_GRAPH_MAX_BYTES = $oldCaptureLimit
}

$candidates = @(Get-ChildItem -LiteralPath $resultRoot -Directory | Where-Object {
    -not $before.ContainsKey($_.FullName)
} | Sort-Object LastWriteTime -Descending)
if (-not $candidates.Count) { throw 'Feature-18 run did not create a result directory' }
$result = $candidates[0].FullName

& powershell -NoProfile -ExecutionPolicy Bypass -File `
    (Join-Path $PSScriptRoot 'consolidate_full_graph_capture.ps1') -ResultDir $result
$consolidateExit = $LASTEXITCODE
$workflowSummaryPath = Join-Path $result 'full_workflow_summary.json'
$workflow = if (Test-Path -LiteralPath $workflowSummaryPath) {
    Get-Content -LiteralPath $workflowSummaryPath -Raw | ConvertFrom-Json
} else {
    [pscustomobject]@{ status='FAIL'; errors=@('consolidation summary missing') }
}
$workflow | Add-Member -NotePropertyName package_integrity -NotePropertyValue $packageIntegrity -Force
$workflow | Add-Member -NotePropertyName package_revision `
    -NotePropertyValue ([string]$packageManifest.package_revision) -Force
$workflow | Add-Member -NotePropertyName feature_runner_exit_code -NotePropertyValue $featureExit -Force
$workflow | Add-Member -NotePropertyName consolidation_exit_code -NotePropertyValue $consolidateExit -Force
$workflow | Add-Member -NotePropertyName max_capture_bytes -NotePropertyValue $MaxCaptureBytes -Force
$workflow | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $workflowSummaryPath -Encoding utf8

$forbidden = @(Get-ChildItem -LiteralPath $result -Recurse -File | Where-Object {
    $_.Name -match '^(dxgi\.dll|nvngx.*\.dll|renodx.*\.(dll|addon64))$'
})
if ($forbidden.Count) { throw "External binary leaked into result: $($forbidden.FullName -join ', ')" }

$returnFiles = @()
$resultBase = [IO.Path]::GetFullPath($result).TrimEnd('\') + '\'
foreach ($file in Get-ChildItem -LiteralPath $result -Recurse -File | Sort-Object FullName) {
    $full = [IO.Path]::GetFullPath($file.FullName)
    if (-not $full.StartsWith($resultBase, [StringComparison]::OrdinalIgnoreCase)) {
        throw "result file escapes result directory: $full"
    }
    $relative = $full.Substring($resultBase.Length).Replace('\','/')
    if ($relative.StartsWith('full_graph_blobs/')) { continue }
    $returnFiles += [ordered]@{
        path = $relative
        size = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
[ordered]@{
    schema = 1
    package_revision = 'v20_full_graph_one_shot'
    workflow_status = [string]$workflow.status
    blob_integrity_manifest = 'full_graph_blob_manifest.json'
    files = $returnFiles
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath `
    (Join-Path $result 'RETURN_MANIFEST.json') -Encoding utf8

$returnRoot = Join-Path $root 'return_to_lab'
New-Item -ItemType Directory -Force -Path $returnRoot | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$archive = Join-Path $returnRoot "_full_graph_reference_result_$stamp.zip"
Compress-Archive -Path (Join-Path $result '*') -DestinationPath $archive `
    -CompressionLevel Optimal -Force
$archiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
Write-Host "`n=== FULL CLOUD WORKFLOW COMPLETE ==="
Write-Host "Workflow status: $($workflow.status)"
Write-Host "Captured slots: $($workflow.captured_slot_count)"
Write-Host "Unique blobs: $($workflow.unique_blob_count)"
Write-Host "Return this single file: $archive"
Write-Host "SHA256: $archiveHash"
exit $(if ($workflow.status -eq 'PASS' -and $featureExit -eq 0 -and $consolidateExit -eq 0) { 0 } else { 1 })
