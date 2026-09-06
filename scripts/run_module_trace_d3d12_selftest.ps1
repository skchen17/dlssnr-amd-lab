$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only module_trace,module_trace_d3d12_selftest
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_module_trace_d3d12_selftest"
New-Item -ItemType Directory -Path $result | Out-Null
$log = Join-Path $result 'module_trace.jsonl'
$env:MODULE_TRACE_LOG = $log
$exe = Join-Path $repo 'build\module_trace_d3d12_selftest.exe'
$dll = Join-Path $repo 'build\module_trace.dll'
& $exe $dll $result 2>&1 | Tee-Object -FilePath (Join-Path $result 'stdout.log')
$code = $LASTEXITCODE
Remove-Item Env:MODULE_TRACE_LOG -ErrorAction SilentlyContinue

$events = @()
$invalid = 0
if (Test-Path -LiteralPath $log) {
    foreach ($line in Get-Content -LiteralPath $log) {
        try { $events += ($line | ConvertFrom-Json) } catch { ++$invalid }
    }
}
$summary = [ordered]@{
    schema = 1
    experiment = 'module_trace_d3d12_selftest'
    exit_code = $code
    invalid_json_lines = $invalid
    hooks_installed = @($events | Where-Object { $_.ev -eq 'd3d12_device_hooks' -and $_.installed }).Count
    resources = @($events | Where-Object { $_.ev -eq 'd3d12_resource_create' -and $_.resource -ne '0x0' }).Count
    srvs = @($events | Where-Object { $_.ev -eq 'd3d12_create_srv' }).Count
    uavs = @($events | Where-Object { $_.ev -eq 'd3d12_create_uav' }).Count
    samplers = @($events | Where-Object { $_.ev -eq 'd3d12_create_sampler' }).Count
    descriptor_copies = @($events | Where-Object { $_.ev -eq 'd3d12_copy_descriptor' }).Count
    snapshot_events = @($events | Where-Object { $_.ev -eq 'copy_snapshot' -and $_.status -eq 'PASS' }).Count
    command_queue_creates = @($events | Where-Object {
        $_.ev -eq 'd3d12_command_queue_create' -and $_.hooks_installed
    }).Count
    command_list_creates = @($events | Where-Object {
        $_.ev -eq 'd3d12_command_list_create' -and $_.hooks_installed
    }).Count
    command_list_closes = @($events | Where-Object {
        $_.ev -eq 'd3d12_command_list_close_call' -and $_.close_count -ge 1
    }).Count
    execute_calls = @($events | Where-Object {
        $_.ev -eq 'd3d12_execute_command_lists_call' -and $_.count -ge 1
    }).Count
    execute_items = @($events | Where-Object {
        $_.ev -eq 'd3d12_execute_command_list' -and $_.close_count -ge 1
    }).Count
}
$snapshotPath = Join-Path $result 'copy_snapshot.json'
$inputPath = Join-Path $result 'copy_input.raw'
$outputPath = Join-Path $result 'copy_output.raw'
$snapshot = if (Test-Path -LiteralPath $snapshotPath) {
    Get-Content -LiteralPath $snapshotPath -Raw | ConvertFrom-Json
} else { $null }
$inputHash = if (Test-Path -LiteralPath $inputPath) {
    (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash
} else { $null }
$outputHash = if (Test-Path -LiteralPath $outputPath) {
    (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
} else { $null }
$summary.copy_snapshot = [ordered]@{
    status = if ($snapshot) { $snapshot.status } else { 'MISSING' }
    width = if ($snapshot) { $snapshot.width } else { 0 }
    height = if ($snapshot) { $snapshot.height } else { 0 }
    input_bytes = if ($snapshot) { $snapshot.input_bytes } else { 0 }
    output_bytes = if ($snapshot) { $snapshot.output_bytes } else { 0 }
    input_sha256 = $inputHash
    output_sha256 = $outputHash
    bitwise_equal = $null -ne $inputHash -and $inputHash -eq $outputHash
}
$frameOutputMetadataPath = Join-Path $result 'frame1_post_output_pre_copy.json'
$frameOutputPath = Join-Path $result 'frame1_post_output_pre_copy.raw'
$frameOutput = if (Test-Path -LiteralPath $frameOutputMetadataPath) {
    Get-Content -LiteralPath $frameOutputMetadataPath -Raw | ConvertFrom-Json
} else { $null }
$summary.frame1_post_output = [ordered]@{
    status = if ($frameOutput) { $frameOutput.status } else { 'MISSING' }
    classification = if ($frameOutput) { $frameOutput.classification } else { $null }
    capture_frame = if ($frameOutput) { $frameOutput.capture_frame } else { 0 }
    bytes = if (Test-Path -LiteralPath $frameOutputPath) {
        (Get-Item -LiteralPath $frameOutputPath).Length
    } else { 0 }
    arm_events = @($events | Where-Object {
        $_.ev -eq 'frame1_post_output_capture_arm' -and $_.status -eq 'PASS'
    }).Count
}
$surfaceInitialMetadataPath = Join-Path $result 'frame1_post_surface_pre_slot154.json'
$surfaceInitialPath = Join-Path $result 'frame1_post_surface_pre_slot154.raw'
$surfaceInitial = if (Test-Path -LiteralPath $surfaceInitialMetadataPath) {
    Get-Content -LiteralPath $surfaceInitialMetadataPath -Raw | ConvertFrom-Json
} else { $null }
$summary.frame1_post_surface_initial = [ordered]@{
    status = if ($surfaceInitial) { $surfaceInitial.status } else { 'MISSING' }
    classification = if ($surfaceInitial) { $surfaceInitial.classification } else { $null }
    capture_frame = if ($surfaceInitial) { $surfaceInitial.capture_frame } else { 0 }
    bytes = if (Test-Path -LiteralPath $surfaceInitialPath) {
        (Get-Item -LiteralPath $surfaceInitialPath).Length
    } else { 0 }
    arm_events = @($events | Where-Object {
        $_.ev -eq 'frame1_post_surface_initial_capture_arm' -and $_.status -eq 'PASS'
    }).Count
}
$n0Path = Join-Path $result 'n0_preblock_capture.json'
$n0 = if (Test-Path -LiteralPath $n0Path) {
    Get-Content -LiteralPath $n0Path -Raw | ConvertFrom-Json
} else { $null }
$n0Windows = @{}
if ($n0) { foreach ($window in $n0.windows) { $n0Windows[$window.name] = $window } }
$downstreamWindowNames = @(3..23 | ForEach-Object { "slot${_}_weights" })
$summary.n0_capture = [ordered]@{
    status = if ($n0) { $n0.status } else { 'MISSING' }
    arm_events = @($events | Where-Object { $_.ev -eq 'neural_capture_arm' -and $_.status -eq 'PASS' }).Count
    recorded_events = @($events | Where-Object { $_.ev -eq 'neural_capture_recorded' -and $_.status -eq 'PASS' }).Count
    dump_events = @($events | Where-Object { $_.ev -eq 'neural_capture_dump' -and $_.status -eq 'PASS' }).Count
    scratch_changed_bytes = if ($n0Windows.ContainsKey('scratch')) { $n0Windows['scratch'].changed_bytes } else { -1 }
    weights_changed_bytes = if ($n0Windows.ContainsKey('weights')) { $n0Windows['weights'].changed_bytes } else { -1 }
    output_changed_bytes = if ($n0Windows.ContainsKey('output')) { $n0Windows['output'].changed_bytes } else { -1 }
    slot3_weights_changed_bytes = if ($n0Windows.ContainsKey('slot3_weights')) { $n0Windows['slot3_weights'].changed_bytes } else { -1 }
    slot4_weights_changed_bytes = if ($n0Windows.ContainsKey('slot4_weights')) { $n0Windows['slot4_weights'].changed_bytes } else { -1 }
    slot5_weights_changed_bytes = if ($n0Windows.ContainsKey('slot5_weights')) { $n0Windows['slot5_weights'].changed_bytes } else { -1 }
    downstream_window_count = @($downstreamWindowNames | Where-Object { $n0Windows.ContainsKey($_) }).Count
    downstream_immutable_count = @($downstreamWindowNames | Where-Object {
        $n0Windows.ContainsKey($_) -and $n0Windows[$_].changed_bytes -eq 0
    }).Count
}
$fullGraphPath = Join-Path $result 'full_graph_capture.json'
$fullGraph = if (Test-Path -LiteralPath $fullGraphPath) {
    Get-Content -LiteralPath $fullGraphPath -Raw | ConvertFrom-Json
} else { $null }
$fullGraphFilesValid = $false
if ($fullGraph) {
    $fullGraphFilesValid = @($fullGraph.windows | Where-Object {
        $before = Join-Path $result ($_.before_file -replace '/', '\')
        $after = Join-Path $result ($_.after_file -replace '/', '\')
        -not (Test-Path -LiteralPath $before -PathType Leaf) -or
        -not (Test-Path -LiteralPath $after -PathType Leaf) -or
        (Get-Item -LiteralPath $before).Length -ne $_.capture_bytes -or
        (Get-Item -LiteralPath $after).Length -ne $_.capture_bytes
    }).Count -eq 0
}
$summary.full_graph_capture = [ordered]@{
    status = if ($fullGraph) { $fullGraph.status } else { 'MISSING' }
    dump_events = @($events | Where-Object {
        $_.ev -eq 'full_graph_capture_dump' -and $_.status -eq 'PASS'
    }).Count
    window_count = if ($fullGraph) { $fullGraph.window_count } else { 0 }
    captured_slot_count = if ($fullGraph) { $fullGraph.captured_slot_count } else { 0 }
    capture_failures = if ($fullGraph) { $fullGraph.capture_failures } else { -1 }
    files_valid = $fullGraphFilesValid
    changed_window_count = if ($fullGraph) {
        @($fullGraph.windows | Where-Object { $_.changed_bytes -gt 0 }).Count
    } else { 0 }
}
$downstreamComplete = @($downstreamWindowNames | Where-Object {
    -not $n0Windows.ContainsKey($_)
}).Count -eq 0
$downstreamImmutable = @($downstreamWindowNames | Where-Object {
    -not $n0Windows.ContainsKey($_) -or $n0Windows[$_].changed_bytes -ne 0
}).Count -eq 0
$pass = $code -eq 0 -and $invalid -eq 0 -and $summary.hooks_installed -eq 1 `
    -and $summary.resources -ge 2 -and $summary.srvs -eq 1 -and $summary.uavs -eq 1 `
    -and $summary.samplers -eq 1 -and $summary.descriptor_copies -eq 3 `
    -and $summary.command_queue_creates -ge 1 `
    -and $summary.command_list_creates -ge 1 `
    -and $summary.command_list_closes -ge 1 `
    -and $summary.execute_calls -ge 1 -and $summary.execute_items -ge 1 `
    -and $summary.snapshot_events -eq 1 -and $snapshot.status -eq 'PASS' `
    -and $summary.copy_snapshot.bitwise_equal -and $n0.status -eq 'PASS' `
    -and $summary.frame1_post_output.status -eq 'PASS' `
    -and $summary.frame1_post_output.classification -eq 'FRAME1_SLOT154_OUTPUT_BEFORE_SLOT155_COPY' `
    -and $summary.frame1_post_output.capture_frame -eq 1 `
    -and $summary.frame1_post_output.bytes -eq 32768 `
    -and $summary.frame1_post_output.arm_events -eq 1 `
    -and $summary.frame1_post_surface_initial.status -eq 'PASS' `
    -and $summary.frame1_post_surface_initial.classification -eq 'FRAME1_SLOT154_OUTPUT_SURFACE_INITIAL' `
    -and $summary.frame1_post_surface_initial.capture_frame -eq 1 `
    -and $summary.frame1_post_surface_initial.bytes -eq 32768 `
    -and $summary.frame1_post_surface_initial.arm_events -eq 1 `
    -and $summary.n0_capture.arm_events -eq 1 -and $summary.n0_capture.recorded_events -eq 1 `
    -and $summary.n0_capture.dump_events -eq 1 `
    -and $summary.n0_capture.weights_changed_bytes -eq 0 `
    -and $downstreamComplete -and $downstreamImmutable `
    -and $summary.n0_capture.output_changed_bytes -eq 255 `
    -and $fullGraph.status -eq 'PASS' -and $summary.full_graph_capture.dump_events -eq 1 `
    -and $summary.full_graph_capture.window_count -eq 3 `
    -and $summary.full_graph_capture.captured_slot_count -eq 1 `
    -and $summary.full_graph_capture.capture_failures -eq 0 `
    -and $summary.full_graph_capture.files_valid `
    -and $summary.full_graph_capture.changed_window_count -eq 1
$summary.status = if ($pass) { 'PASS' } else { 'FAIL' }
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $result 'summary.json') -Encoding utf8
Write-Host ($summary | ConvertTo-Json -Compress)
Write-Host "Result: $result"
exit $(if ($pass) { 0 } else { 1 })
