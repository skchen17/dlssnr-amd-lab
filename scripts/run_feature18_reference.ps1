<#
.SYNOPSIS
  Runs the DLSS5-Feeder standalone host under ReShade/RenoDX on an NVIDIA PC.
.DESCRIPTION
  Builds the missing private-feature-18 reference evidence around a proven public
  DLAA contract. External/proprietary binaries are copied into an isolated work
  directory for the run, hashed, and never included in the returned ZIP.
#>
[CmdletBinding()]
param(
    [string]$ReShade64 = '',
    [string]$RenoDxAddon = '',
    [string]$DlssDll = '',
    [string]$DlssNrDll = '',
    [ValidateSet('01','02','02.5')][string]$RenoDxVariant = '02',
    [switch]$SkipArchive
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path -Parent $PSScriptRoot
$hostExe = Join-Path $root 'build\dlss5-feed-host64.exe'
if (-not (Test-Path -LiteralPath $hostExe)) { throw 'Packaged host missing: build\dlss5-feed-host64.exe' }
$moduleTrace = Join-Path $root 'build\module_trace.dll'
$payload = Join-Path $root 'payload'

function Resolve-File([string]$Path, [string]$Label) {
    try { return (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path }
    catch { throw "$Label not found: $Path" }
}
function File-Record([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    $sig = Get-AuthenticodeSignature -LiteralPath $Path -ErrorAction SilentlyContinue
    [ordered]@{
        filename = $item.Name
        size = $item.Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        file_version = $item.VersionInfo.FileVersion
        product_version = $item.VersionInfo.ProductVersion
        signature_status = if ($sig) { $sig.Status.ToString() } else { 'Unavailable' }
        signature_subject = if ($sig -and $sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { $null }
    }
}

if (-not $ReShade64) { $ReShade64 = Join-Path $payload 'ReShade64.dll' }
if (-not $RenoDxAddon) { $RenoDxAddon = Join-Path $payload "renodx-dlss5-$RenoDxVariant.addon64" }
if (-not $DlssDll) { $DlssDll = Join-Path $payload 'nvngx_dlss.dll' }
if (-not $DlssNrDll) { $DlssNrDll = Join-Path $payload 'nvngx_dlssnr.dll' }
$ReShade64 = Resolve-File $ReShade64 '64-bit ReShade runtime'
$RenoDxAddon = Resolve-File $RenoDxAddon "RenoDX DLSS5 variant $RenoDxVariant"
$DlssDll = Resolve-File $DlssDll 'nvngx_dlss.dll'
$DlssNrDll = Resolve-File $DlssNrDll 'nvngx_dlssnr.dll'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$workRoot = Join-Path $root 'feature18_work'
$work = Join-Path $workRoot $stamp
$resultRoot = Join-Path $root 'feature18_results'
$result = Join-Path $resultRoot $stamp
$returnRoot = Join-Path $root 'return_to_lab'
New-Item -ItemType Directory -Force -Path $work,$result,$returnRoot | Out-Null

$summary = [ordered]@{
    schema = 1
    package_revision = 'v26_feature18_submission_topology'
    timestamp_local = $stamp
    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
    adapter_test = 'DLSS5-Feeder host --test --test-frames 1; deferred queue-complete output snapshot'
    renodx_variant = $RenoDxVariant
    external_files = [ordered]@{
        reshade = File-Record $ReShade64
        renodx_addon = File-Record $RenoDxAddon
        dlss = File-Record $DlssDll
        dlssnr = File-Record $DlssNrDll
    }
    instrumentation = [ordered]@{
        module_trace = if (Test-Path -LiteralPath $moduleTrace) { File-Record $moduleTrace } else { $null }
    }
}

$oldModuleTraceLog = $env:MODULE_TRACE_LOG
$oldModuleTraceDumpDir = $env:MODULE_TRACE_DUMP_DIR
$oldDisableInlineSnapshots = $env:MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS

try {
    Copy-Item -LiteralPath $hostExe -Destination (Join-Path $work 'dlss5-feed-host64.exe')
    Copy-Item -LiteralPath $ReShade64 -Destination (Join-Path $work 'dxgi.dll')
    Copy-Item -LiteralPath $RenoDxAddon -Destination (Join-Path $work 'renodx-dlss5.addon64')
    Copy-Item -LiteralPath $DlssDll -Destination (Join-Path $work 'nvngx_dlss.dll')
    Copy-Item -LiteralPath $DlssNrDll -Destination (Join-Path $work 'nvngx_dlssnr.dll')
    if (Test-Path -LiteralPath $moduleTrace) {
        Copy-Item -LiteralPath $moduleTrace -Destination (Join-Path $work 'module_trace.dll')
        $env:MODULE_TRACE_LOG = Join-Path $result 'module_trace.jsonl'
        $env:MODULE_TRACE_DUMP_DIR = $result
        $env:MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS = '1'
    }

    Push-Location $work
    try {
        & '.\dlss5-feed-host64.exe' --test --test-frames 1 *>&1 |
            Tee-Object -FilePath (Join-Path $result 'stdout.log') |
            ForEach-Object { Write-Host $_ }
        $hostExit = if ($null -eq $LASTEXITCODE) { -1 } else { [int]$LASTEXITCODE }
    } finally { Pop-Location }

    foreach ($name in @('dlss5-feed-host.log','ReShade.log','ReShade.ini')) {
        $source = Join-Path $work $name
        if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $result $name) }
    }

    $hostLog = if (Test-Path (Join-Path $result 'dlss5-feed-host.log')) {
        Get-Content -LiteralPath (Join-Path $result 'dlss5-feed-host.log') -Raw
    } else { '' }
    $reshadeLog = if (Test-Path (Join-Path $result 'ReShade.log')) {
        Get-Content -LiteralPath (Join-Path $result 'ReShade.log') -Raw
    } else { '' }
    $m = [regex]::Match($hostLog, '--test finished:\s*(\d+)/1 evaluates succeeded', 'IgnoreCase')
    $good = if ($m.Success) { [int]$m.Groups[1].Value } else { 0 }
    $created = [regex]::IsMatch($reshadeLog, 'feature\s+18.*created', 'IgnoreCase')
    $evaluated = [regex]::IsMatch($reshadeLog, 'feature\s+18.*evaluation succeeded', 'IgnoreCase')
    $summary.host_exit = $hostExit
    $summary.evaluates_ok = $good
    $summary.feature18_created = $created
    $summary.feature18_evaluation_succeeded = $evaluated
    $tracePath = Join-Path $result 'module_trace.jsonl'
    $traceEvents = @()
    $invalidTraceLines = 0
    if (Test-Path -LiteralPath $tracePath) {
        foreach ($line in Get-Content -LiteralPath $tracePath) {
            try { $traceEvents += ($line | ConvertFrom-Json -ErrorAction Stop) }
            catch { ++$invalidTraceLines }
        }
    }
    $merged = @($traceEvents | Where-Object { $_.ev -eq 'nvapi_get_cuda_merged_texture_sampler' })
    $independent = @($traceEvents | Where-Object { $_.ev -eq 'nvapi_get_cuda_independent_descriptor' })
    $copies = @($traceEvents | Where-Object {
        $_.ev -eq 'nvapi_launch_cu_kernel' -and $_.slot -eq 155
    })
    $summary.descriptor_object_trace = [ordered]@{
        merged_texture_sampler_calls = $merged.Count
        independent_descriptor_calls = $independent.Count
        independent_types = @($independent | ForEach-Object { $_.type } | Sort-Object -Unique)
        texture_handles = @($merged | ForEach-Object { $_.texture_handle } | Sort-Object -Unique)
        independent_handles = @($independent | ForEach-Object { $_.handle } | Sort-Object -Unique)
        captured_copy_launches = $copies.Count
        copy_parameter_hex = @($copies | ForEach-Object { $_.param_hex } | Sort-Object -Unique)
    }
    $deviceHooks = @($traceEvents | Where-Object {
        $_.ev -eq 'd3d12_device_hooks' -and $_.installed -eq $true
    })
    $resourceCreates = @($traceEvents | Where-Object { $_.ev -eq 'd3d12_resource_create' })
    $srvs = @($traceEvents | Where-Object { $_.ev -eq 'd3d12_create_srv' })
    $uavs = @($traceEvents | Where-Object { $_.ev -eq 'd3d12_create_uav' })
    $samplers = @($traceEvents | Where-Object { $_.ev -eq 'd3d12_create_sampler' })
    $descriptorCopies = @($traceEvents | Where-Object { $_.ev -eq 'd3d12_copy_descriptor' })
    $postTextureBinds = @($traceEvents | Where-Object {
        $_.ev -eq 'post_texture_resource_bind' -and $_.slot -eq 154 -and
        $_.texture_object -ne '0x0000000000000000' -and $_.texture_resource -ne '0x0'
    })
    $summary.d3d12_resource_trace = [ordered]@{
        invalid_json_lines = $invalidTraceLines
        device_hook_successes = $deviceHooks.Count
        resource_creates = $resourceCreates.Count
        srv_creates = $srvs.Count
        uav_creates = $uavs.Count
        sampler_creates = $samplers.Count
        descriptor_copies = $descriptorCopies.Count
        post_texture_resource_binds = $postTextureBinds.Count
    }
    $queueCreates = @($traceEvents | Where-Object {
        $_.ev -eq 'd3d12_command_queue_create' -and $_.hooks_installed -eq $true
    })
    $listCreates = @($traceEvents | Where-Object {
        $_.ev -eq 'd3d12_command_list_create' -and $_.hooks_installed -eq $true
    })
    $closeEvents = @($traceEvents | Where-Object {
        $_.ev -eq 'd3d12_command_list_close_call'
    })
    $executeEvents = @($traceEvents | Where-Object {
        $_.ev -eq 'd3d12_execute_command_list'
    })
    $frame1Launches = @($traceEvents | Where-Object {
        $_.ev -eq 'nvapi_launch_cu_kernel_chain_call' -and $_.frame -eq 1 -and
        $_.slot -ge 0 -and $_.slot -le 155
    } | Sort-Object sequence)
    $frame1Lists = @($frame1Launches | ForEach-Object { $_.command_list } |
        Where-Object { $_ } | Sort-Object -Unique)
    $listTopology = @()
    foreach ($commandList in $frame1Lists) {
        $launches = @($frame1Launches | Where-Object {
            $_.command_list -eq $commandList
        } | Sort-Object sequence)
        $firstSequence = if ($launches.Count) { [long]$launches[0].sequence } else { -1 }
        $lastSequence = if ($launches.Count) { [long]$launches[-1].sequence } else { -1 }
        $close = @($closeEvents | Where-Object {
            $_.command_list -eq $commandList -and [long]$_.sequence -gt $lastSequence
        } | Sort-Object sequence | Select-Object -First 1)
        $closeSequence = if ($close.Count) { [long]$close[0].sequence } else { -1 }
        $execute = @($executeEvents | Where-Object {
            $_.command_list -eq $commandList -and
            ($closeSequence -lt 0 -or [long]$_.sequence -ge $closeSequence)
        } | Sort-Object sequence | Select-Object -First 1)
        $slots = @($launches | ForEach-Object { [int]$_.slot } | Sort-Object -Unique)
        $listTopology += [ordered]@{
            command_list = $commandList
            launch_count = $launches.Count
            unique_slot_count = $slots.Count
            first_slot = if ($launches.Count) { [int]$launches[0].slot } else { -1 }
            last_slot = if ($launches.Count) { [int]$launches[-1].slot } else { -1 }
            first_launch_sequence = $firstSequence
            last_launch_sequence = $lastSequence
            close_sequence = $closeSequence
            execute_sequence = if ($execute.Count) { [long]$execute[0].sequence } else { -1 }
            queue = if ($execute.Count) { $execute[0].queue } else { $null }
            close_after_last_launch = $closeSequence -gt $lastSequence
            execute_after_close = $execute.Count -gt 0 -and
                [long]$execute[0].sequence -ge $closeSequence
        }
    }
    $slot154Launch = @($frame1Launches | Where-Object { $_.slot -eq 154 } |
        Select-Object -First 1)
    $slot155Launch = @($frame1Launches | Where-Object { $_.slot -eq 155 } |
        Select-Object -First 1)
    $allFrame1ListsSubmitted = $listTopology.Count -gt 0 -and
        @($listTopology | Where-Object {
            -not $_.close_after_last_launch -or -not $_.execute_after_close
        }).Count -eq 0
    $singleFullFrameList = $frame1Lists.Count -eq 1 -and
        $frame1Launches.Count -eq 156 -and
        @($frame1Launches | ForEach-Object { [int]$_.slot } | Sort-Object -Unique).Count -eq 156
    $slot154And155SameList = $slot154Launch.Count -eq 1 -and
        $slot155Launch.Count -eq 1 -and
        $slot154Launch[0].command_list -eq $slot155Launch[0].command_list
    $topologyClassification = if ($singleFullFrameList -and $allFrame1ListsSubmitted) {
        'SINGLE_LIST_FRAME1_CLOSE_THEN_EXECUTE'
    } elseif ($frame1Launches.Count -eq 156 -and $allFrame1ListsSubmitted) {
        'MULTI_LIST_FRAME1_CLOSE_THEN_EXECUTE'
    } else {
        'INCOMPLETE_FRAME1_SUBMISSION_TRACE'
    }
    $summary.command_submission_topology = [ordered]@{
        classification = $topologyClassification
        observable = $queueCreates.Count -ge 1 -and $listCreates.Count -ge 1 -and
            $closeEvents.Count -ge 1 -and $executeEvents.Count -ge 1
        queue_create_count = $queueCreates.Count
        command_list_create_count = $listCreates.Count
        close_count = $closeEvents.Count
        execute_item_count = $executeEvents.Count
        frame1_launch_count = $frame1Launches.Count
        frame1_unique_slot_count = @($frame1Launches | ForEach-Object {
            [int]$_.slot
        } | Sort-Object -Unique).Count
        frame1_command_lists = $frame1Lists
        single_command_list_for_full_frame = $singleFullFrameList
        slot154_and_slot155_same_list = $slot154And155SameList
        all_frame1_lists_close_then_execute = $allFrame1ListsSubmitted
        lists = $listTopology
    }
    $snapshotPath = Join-Path $result 'copy_snapshot.json'
    $snapshot = if (Test-Path -LiteralPath $snapshotPath) {
        Get-Content -LiteralPath $snapshotPath -Raw | ConvertFrom-Json
    } else { $null }
    $inputRaw = Join-Path $result 'copy_input.raw'
    $outputRaw = Join-Path $result 'copy_output.raw'
    $summary.copy_content_snapshot = [ordered]@{
        status = if ($snapshot) { $snapshot.status } else { 'MISSING' }
        hresult = if ($snapshot) { $snapshot.hresult } else { $null }
        width = if ($snapshot) { $snapshot.width } else { 0 }
        height = if ($snapshot) { $snapshot.height } else { 0 }
        format = if ($snapshot) { $snapshot.format } else { 0 }
        row_size = if ($snapshot) { $snapshot.row_size } else { 0 }
        input_bytes = if ($snapshot) { $snapshot.input_bytes } else { 0 }
        output_bytes = if ($snapshot) { $snapshot.output_bytes } else { 0 }
        input_sha256 = if (Test-Path -LiteralPath $inputRaw) {
            (Get-FileHash -LiteralPath $inputRaw -Algorithm SHA256).Hash.ToLowerInvariant()
        } else { $null }
        output_sha256 = if (Test-Path -LiteralPath $outputRaw) {
            (Get-FileHash -LiteralPath $outputRaw -Algorithm SHA256).Hash.ToLowerInvariant()
        } else { $null }
    }
    $postTextureSnapshotPath = Join-Path $result 'post_texture_snapshot.json'
    $postTextureSnapshot = if (Test-Path -LiteralPath $postTextureSnapshotPath) {
        Get-Content -LiteralPath $postTextureSnapshotPath -Raw | ConvertFrom-Json
    } else { $null }
    $postTextureRaw = Join-Path $result 'post_texture_input.raw'
    $postTextureNonzeroBytes = 0L
    if (Test-Path -LiteralPath $postTextureRaw) {
        foreach ($value in [IO.File]::ReadAllBytes($postTextureRaw)) {
            if ($value -ne 0) { ++$postTextureNonzeroBytes }
        }
    }
    $summary.post_texture_snapshot = [ordered]@{
        status = if ($postTextureSnapshot) { $postTextureSnapshot.status } else { 'MISSING' }
        classification = if ($postTextureSnapshot) { $postTextureSnapshot.classification } else { $null }
        capture_timing = if ($postTextureSnapshot) { $postTextureSnapshot.capture_timing } else { $null }
        capture_frame = if ($postTextureSnapshot) { $postTextureSnapshot.capture_frame } else { 0 }
        texture_object = if ($postTextureSnapshot) { $postTextureSnapshot.texture_object } else { $null }
        texture_resource = if ($postTextureSnapshot) { $postTextureSnapshot.texture_resource } else { $null }
        width = if ($postTextureSnapshot) { $postTextureSnapshot.width } else { 0 }
        height = if ($postTextureSnapshot) { $postTextureSnapshot.height } else { 0 }
        format = if ($postTextureSnapshot) { $postTextureSnapshot.format } else { 0 }
        bytes = if ($postTextureSnapshot) { $postTextureSnapshot.bytes } else { 0 }
        nonzero_bytes = $postTextureNonzeroBytes
        sha256 = if (Test-Path -LiteralPath $postTextureRaw) {
            (Get-FileHash -LiteralPath $postTextureRaw -Algorithm SHA256).Hash.ToLowerInvariant()
        } else { $null }
    }
    $postActivationSnapshotPath = Join-Path $result 'post_activation_arena_prelaunch.json'
    $postActivationSnapshot = if (Test-Path -LiteralPath $postActivationSnapshotPath) {
        Get-Content -LiteralPath $postActivationSnapshotPath -Raw | ConvertFrom-Json
    } else { $null }
    $postActivationRaw = Join-Path $result 'post_activation_arena_prelaunch.raw'
    $summary.post_activation_snapshot = [ordered]@{
        status = if ($postActivationSnapshot) { $postActivationSnapshot.status } else { 'MISSING' }
        classification = if ($postActivationSnapshot) { $postActivationSnapshot.classification } else { $null }
        capture_timing = if ($postActivationSnapshot) { $postActivationSnapshot.capture_timing } else { $null }
        frame = if ($postActivationSnapshot) { $postActivationSnapshot.frame } else { 0 }
        resource = if ($postActivationSnapshot) { $postActivationSnapshot.resource } else { $null }
        param0 = if ($postActivationSnapshot) { $postActivationSnapshot.param0 } else { $null }
        param8 = if ($postActivationSnapshot) { $postActivationSnapshot.param8 } else { $null }
        bytes = if (Test-Path -LiteralPath $postActivationRaw) {
            (Get-Item -LiteralPath $postActivationRaw).Length
        } else { 0 }
        sha256 = if (Test-Path -LiteralPath $postActivationRaw) {
            (Get-FileHash -LiteralPath $postActivationRaw -Algorithm SHA256).Hash.ToLowerInvariant()
        } else { $null }
    }
    $frameOutputSnapshotPath = Join-Path $result 'frame1_post_output_pre_copy.json'
    $frameOutputSnapshot = if (Test-Path -LiteralPath $frameOutputSnapshotPath) {
        Get-Content -LiteralPath $frameOutputSnapshotPath -Raw | ConvertFrom-Json
    } else { $null }
    $frameOutputRaw = Join-Path $result 'frame1_post_output_pre_copy.raw'
    $frameOutputEvents = @($traceEvents | Where-Object {
        $_.ev -eq 'frame1_post_output_capture_arm' -and $_.status -eq 'PASS'
    })
    $summary.frame1_post_output_snapshot = [ordered]@{
        status = if ($frameOutputSnapshot) { $frameOutputSnapshot.status } else { 'MISSING' }
        classification = if ($frameOutputSnapshot) { $frameOutputSnapshot.classification } else { $null }
        capture_timing = if ($frameOutputSnapshot) { $frameOutputSnapshot.capture_timing } else { $null }
        capture_frame = if ($frameOutputSnapshot) { $frameOutputSnapshot.capture_frame } else { 0 }
        surface_object = if ($frameOutputSnapshot) { $frameOutputSnapshot.surface_object } else { $null }
        surface_resource = if ($frameOutputSnapshot) { $frameOutputSnapshot.surface_resource } else { $null }
        width = if ($frameOutputSnapshot) { $frameOutputSnapshot.width } else { 0 }
        height = if ($frameOutputSnapshot) { $frameOutputSnapshot.height } else { 0 }
        format = if ($frameOutputSnapshot) { $frameOutputSnapshot.format } else { 0 }
        bytes = if (Test-Path -LiteralPath $frameOutputRaw) {
            (Get-Item -LiteralPath $frameOutputRaw).Length
        } else { 0 }
        sha256 = if (Test-Path -LiteralPath $frameOutputRaw) {
            (Get-FileHash -LiteralPath $frameOutputRaw -Algorithm SHA256).Hash.ToLowerInvariant()
        } else { $null }
        capture_events = $frameOutputEvents.Count
    }
    $surfaceInitialSnapshotPath = Join-Path $result 'frame1_post_surface_pre_slot154.json'
    $surfaceInitialSnapshot = if (Test-Path -LiteralPath $surfaceInitialSnapshotPath) {
        Get-Content -LiteralPath $surfaceInitialSnapshotPath -Raw | ConvertFrom-Json
    } else { $null }
    $surfaceInitialRaw = Join-Path $result 'frame1_post_surface_pre_slot154.raw'
    $surfaceInitialEvents = @($traceEvents | Where-Object {
        $_.ev -eq 'frame1_post_surface_initial_capture_arm' -and $_.status -eq 'PASS'
    })
    $summary.frame1_post_surface_initial_snapshot = [ordered]@{
        status = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.status } else { 'MISSING' }
        classification = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.classification } else { $null }
        capture_timing = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.capture_timing } else { $null }
        capture_frame = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.capture_frame } else { 0 }
        surface_object = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.surface_object } else { $null }
        surface_resource = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.surface_resource } else { $null }
        width = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.width } else { 0 }
        height = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.height } else { 0 }
        format = if ($surfaceInitialSnapshot) { $surfaceInitialSnapshot.format } else { 0 }
        bytes = if (Test-Path -LiteralPath $surfaceInitialRaw) {
            (Get-Item -LiteralPath $surfaceInitialRaw).Length
        } else { 0 }
        sha256 = if (Test-Path -LiteralPath $surfaceInitialRaw) {
            (Get-FileHash -LiteralPath $surfaceInitialRaw -Algorithm SHA256).Hash.ToLowerInvariant()
        } else { $null }
        capture_events = $surfaceInitialEvents.Count
    }
    $n0Path = Join-Path $result 'n0_preblock_capture.json'
    $n0 = if (Test-Path -LiteralPath $n0Path) {
        Get-Content -LiteralPath $n0Path -Raw | ConvertFrom-Json
    } else { $null }
    $n0Windows = @{}
    if ($n0) { foreach ($window in $n0.windows) { $n0Windows[$window.name] = $window } }
    $n0FileRecords = @()
    $downstreamWindowNames = @(3..23 | ForEach-Object { "slot${_}_weights" })
    $n0WindowNames = @('scratch','weights','output') + $downstreamWindowNames
    foreach ($name in $n0WindowNames) {
        foreach ($phase in @('before','after')) {
            $path = Join-Path $result "n0_${name}_${phase}.raw"
            $n0FileRecords += [ordered]@{
                name = "${name}_${phase}"
                exists = Test-Path -LiteralPath $path
                size = if (Test-Path -LiteralPath $path) { (Get-Item -LiteralPath $path).Length } else { 0 }
                sha256 = if (Test-Path -LiteralPath $path) {
                    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
                } else { $null }
            }
        }
    }
    $summary.n0_preblock_snapshot = [ordered]@{
        status = if ($n0) { $n0.status } else { 'MISSING' }
        function = if ($n0) { $n0.function } else { $null }
        frame = if ($n0) { $n0.frame } else { -1 }
        slot = if ($n0) { $n0.slot } else { -1 }
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
        raw_files = $n0FileRecords
    }
    $n0FilesComplete = @($n0FileRecords | Where-Object { -not $_.exists -or $_.size -le 0 }).Count -eq 0
    $downstreamComplete = @($downstreamWindowNames | Where-Object {
        -not $n0Windows.ContainsKey($_)
    }).Count -eq 0
    $downstreamImmutable = @($downstreamWindowNames | Where-Object {
        -not $n0Windows.ContainsKey($_) -or $n0Windows[$_].changed_bytes -ne 0
    }).Count -eq 0
    $summary.deferred_frame1_output = [ordered]@{
        inline_snapshots_disabled = $true
        capture_timing = 'after the sole frame-1 evaluation and queue completion'
        slot154_input_sha256 = $summary.copy_content_snapshot.input_sha256
        slot155_output_sha256 = $summary.copy_content_snapshot.output_sha256
        bitwise_equal = $summary.copy_content_snapshot.input_sha256 -eq
            $summary.copy_content_snapshot.output_sha256
    }
    $summary.status = if ($hostExit -eq 0 -and $good -eq 1 -and $created -and $evaluated `
        -and $invalidTraceLines -eq 0 -and $deviceHooks.Count -ge 1 `
        -and $resourceCreates.Count -ge 1 -and $srvs.Count -ge 1 `
        -and $uavs.Count -ge 1 -and $descriptorCopies.Count -ge 1 `
        -and $postTextureBinds.Count -ge 1 `
        -and $summary.command_submission_topology.observable `
        -and $frame1Launches.Count -eq 156 `
        -and $summary.command_submission_topology.frame1_unique_slot_count -eq 156 `
        -and $slot154Launch.Count -eq 1 -and $slot155Launch.Count -eq 1 `
        -and $allFrame1ListsSubmitted `
        -and $snapshot -and $snapshot.status -eq 'PASS' `
        -and (Test-Path -LiteralPath $inputRaw) -and (Test-Path -LiteralPath $outputRaw) `
        -and (Get-Item -LiteralPath $inputRaw).Length -eq 1843200 `
        -and (Get-Item -LiteralPath $outputRaw).Length -eq 1843200 `
        -and $summary.deferred_frame1_output.bitwise_equal `
        -and $frameOutputEvents.Count -eq 0 -and $surfaceInitialEvents.Count -eq 0 `
        -and -not $postTextureSnapshot -and -not $postActivationSnapshot `
        -and -not $frameOutputSnapshot -and -not $surfaceInitialSnapshot) { 'PASS' } else { 'FAIL' }
} finally {
    $env:MODULE_TRACE_LOG = $oldModuleTraceLog
    $env:MODULE_TRACE_DUMP_DIR = $oldModuleTraceDumpDir
    $env:MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS = $oldDisableInlineSnapshots
    $summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $result 'summary.json') -Encoding UTF8
    # Delete only the timestamped scratch directory created above. Returned
    # evidence contains hashes/metadata and logs, never the supplied binaries.
    $resolvedWorkRoot = [IO.Path]::GetFullPath($workRoot).TrimEnd('\') + '\'
    $resolvedWork = [IO.Path]::GetFullPath($work)
    if ($resolvedWork.StartsWith($resolvedWorkRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedWork)) {
        # ReShade, the display driver, Defender, or an indexing process may hold a
        # short-lived handle after the host exits. Cleanup is best-effort and must
        # never suppress an otherwise valid result archive.
        $removed = $false
        $lastCleanupError = $null
        for ($attempt = 1; $attempt -le 8; $attempt++) {
            try {
                Remove-Item -LiteralPath $resolvedWork -Recurse -Force -ErrorAction Stop
                $removed = $true
                break
            } catch {
                $lastCleanupError = $_.Exception.Message
                Start-Sleep -Milliseconds (250 * $attempt)
            }
        }
        if (-not $removed -and (Test-Path -LiteralPath $resolvedWork)) {
            Write-Warning "Scratch cleanup deferred (result is still valid): $resolvedWork — $lastCleanupError"
        }
    }
}

$forbidden = @(Get-ChildItem -LiteralPath $result -Recurse -File | Where-Object {
    $_.Name -match '^(dxgi\.dll|nvngx.*\.dll|renodx.*\.(dll|addon64))$'
})
if ($forbidden.Count) { throw "Safety check failed: external binary in result: $($forbidden.FullName -join ', ')" }

Write-Host "`n=== COMPLETE ==="
Write-Host "Status: $($summary.status)"
Write-Host "Result directory: $result"
if (-not $SkipArchive) {
    $zip = Join-Path $returnRoot "dlssnr_feature18_result_$stamp.zip"
    Compress-Archive -Path (Join-Path $result '*') -DestinationPath $zip -CompressionLevel Optimal -Force
    Write-Host "Return this file: $zip"
    Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
}
exit 0
