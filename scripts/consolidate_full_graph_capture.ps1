[CmdletBinding()]
param([Parameter(Mandatory)][string]$ResultDir)

$ErrorActionPreference = 'Stop'
$result = (Resolve-Path -LiteralPath $ResultDir -ErrorAction Stop).Path
$metadataPath = Join-Path $result 'full_graph_capture.json'
$summaryPath = Join-Path $result 'full_workflow_summary.json'
$featureSummaryPath = Join-Path $result 'summary.json'
$copySnapshotPath = Join-Path $result 'copy_snapshot.json'
$blobDir = Join-Path $result 'full_graph_blobs'
$captureDir = Join-Path $result 'full_graph'

$summary = [ordered]@{
    schema = 1
    package_revision = 'v20_full_graph_one_shot'
    status = 'FAIL'
    feature18_status = 'MISSING'
    capture_status = 'MISSING'
    capture_limit_bytes = 0
    window_count = 0
    captured_slot_count = 0
    expected_buffer_slot_count = 155
    captured_buffer_slot_count = 0
    covered_slot_count = 0
    missing_slots_0_154 = @(0..154)
    missing_buffer_slots_0_154 = @(0..154)
    missing_slots_0_155 = @(0..155)
    final_copy_slot = 155
    final_copy_status = 'MISSING'
    unique_blob_count = 0
    logical_raw_bytes = 0
    unique_blob_bytes = 0
    deduplicated_bytes = 0
    content_addressed = $false
    errors = @()
}

try {
    if (Test-Path -LiteralPath $featureSummaryPath) {
        $featureSummary = Get-Content -LiteralPath $featureSummaryPath -Raw | ConvertFrom-Json
        $summary.feature18_status = [string]$featureSummary.status
    }
    if (Test-Path -LiteralPath $copySnapshotPath -PathType Leaf) {
        $copySnapshot = Get-Content -LiteralPath $copySnapshotPath -Raw | ConvertFrom-Json
        $copyMatches = (
            [string]$copySnapshot.status -eq 'PASS' -and
            [uint64]$copySnapshot.input_bytes -eq [uint64]$copySnapshot.output_bytes -and
            [string]$copySnapshot.input_fnv1a64 -eq [string]$copySnapshot.output_fnv1a64
        )
        $summary.final_copy_status = if ($copyMatches) { 'PASS' } else { 'FAIL' }
    }
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        throw "full graph metadata is missing: $metadataPath"
    }
    $capture = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $summary.capture_status = [string]$capture.status
    $summary.capture_limit_bytes = [uint64]$capture.capture_limit_bytes
    $summary.window_count = [int]$capture.window_count
    $capturedSlots = @($capture.windows | ForEach-Object { [int]$_.slot } | Sort-Object -Unique)
    $summary.captured_slot_count = $capturedSlots.Count
    $summary.captured_buffer_slot_count = $capturedSlots.Count
    $summary.missing_slots_0_154 = @(0..154 | Where-Object { $_ -notin $capturedSlots })
    $summary.missing_buffer_slots_0_154 = @($summary.missing_slots_0_154)
    $summary.missing_slots_0_155 = @($summary.missing_buffer_slots_0_154)
    if ($summary.final_copy_status -ne 'PASS') {
        $summary.missing_slots_0_155 += 155
    }
    $summary.covered_slot_count = $summary.captured_buffer_slot_count +
        $(if ($summary.final_copy_status -eq 'PASS') { 1 } else { 0 })
    New-Item -ItemType Directory -Force -Path $blobDir | Out-Null
    $captureRoot = [IO.Path]::GetFullPath($captureDir).TrimEnd('\') + '\'
    $blobRecords = [ordered]@{}
    foreach ($window in $capture.windows) {
        foreach ($phase in @('before','after')) {
            $property = "${phase}_file"
            $relative = [string]$window.$property
            $source = [IO.Path]::GetFullPath((Join-Path $result ($relative -replace '/', '\')))
            if (-not $source.StartsWith($captureRoot, [StringComparison]::OrdinalIgnoreCase)) {
                throw "capture path escapes result directory: $relative"
            }
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
                throw "capture file missing: $relative"
            }
            $item = Get-Item -LiteralPath $source
            if ($item.Length -ne [int64]$window.capture_bytes) {
                throw "capture size mismatch: $relative"
            }
            $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
            $blobName = "$hash.raw"
            $blobPath = Join-Path $blobDir $blobName
            $summary.logical_raw_bytes += [uint64]$item.Length
            if (Test-Path -LiteralPath $blobPath -PathType Leaf) {
                if ((Get-Item -LiteralPath $blobPath).Length -ne $item.Length) {
                    throw "SHA-256 collision/size mismatch for $hash"
                }
                Remove-Item -LiteralPath $source -Force
            } else {
                Move-Item -LiteralPath $source -Destination $blobPath
                $summary.unique_blob_bytes += [uint64]$item.Length
                $blobRecords[$hash] = [ordered]@{
                    sha256 = $hash
                    bytes = [uint64]$item.Length
                    file = "full_graph_blobs/$blobName"
                }
            }
            $window | Add-Member -NotePropertyName "${phase}_sha256" -NotePropertyValue $hash -Force
            $window | Add-Member -NotePropertyName "${phase}_blob" `
                -NotePropertyValue "full_graph_blobs/$blobName" -Force
        }
    }
    if (Test-Path -LiteralPath $captureDir) {
        $remaining = @(Get-ChildItem -LiteralPath $captureDir -Force)
        if ($remaining.Count -eq 0) { Remove-Item -LiteralPath $captureDir -Force }
    }
    $capture | Add-Member -NotePropertyName content_addressed -NotePropertyValue $true -Force
    $capture | Add-Member -NotePropertyName blob_manifest -NotePropertyValue 'full_graph_blob_manifest.json' -Force
    $capture | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $metadataPath -Encoding utf8
    $blobManifest = [ordered]@{
        schema = 1
        algorithm = 'SHA256'
        unique_blob_count = $blobRecords.Count
        logical_raw_bytes = $summary.logical_raw_bytes
        unique_blob_bytes = $summary.unique_blob_bytes
        blobs = @($blobRecords.Values)
    }
    $blobManifest | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath (Join-Path $result 'full_graph_blob_manifest.json') -Encoding utf8
    $summary.unique_blob_count = $blobRecords.Count
    $summary.deduplicated_bytes = $summary.logical_raw_bytes - $summary.unique_blob_bytes
    $summary.content_addressed = $true
    $summary.status = if (
        $summary.feature18_status -eq 'PASS' -and
        $summary.capture_status -eq 'PASS' -and
        $summary.window_count -gt 0 -and
        $summary.captured_buffer_slot_count -eq $summary.expected_buffer_slot_count -and
        $summary.missing_buffer_slots_0_154.Count -eq 0 -and
        $summary.final_copy_status -eq 'PASS' -and
        $summary.covered_slot_count -eq 156
    ) { 'PASS' } else { 'PARTIAL' }
} catch {
    $summary.errors += $_.Exception.Message
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding utf8
Write-Host ($summary | ConvertTo-Json -Depth 8 -Compress)
exit $(if ($summary.status -eq 'PASS') { 0 } else { 1 })
