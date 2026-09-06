param(
    [Parameter(Mandatory=$true)][int]$GameProcessId,
    [Parameter(Mandatory=$true)][string]$ObservationRun,
    [Parameter(Mandatory=$true)][string]$ValidationRun,
    [string]$PriorInspectionRun,
    [switch]$Session,
    [switch]$DeferObservation,
    [string]$BinaryDirectory
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
if (($DeferObservation -or $BinaryDirectory) -and -not $Session) { throw 'Deferred/staged mode requires Session' }
$sessionBin=if ($BinaryDirectory) {(Resolve-Path -LiteralPath $BinaryDirectory).Path} else {Join-Path $repo 'build'}
$observation=(Resolve-Path -LiteralPath $ObservationRun).Path
$validation=(Resolve-Path -LiteralPath $ValidationRun).Path
if ($Session) {
    if (-not $PriorInspectionRun) { throw 'Session requires prior contract inspection provenance' }
    $proof=Get-Content -LiteralPath (Join-Path $validation 'session_analysis.json') -Raw | ConvertFrom-Json
    if ($proof.status -ne 'PASS') { throw 'Session control failed' }
    foreach ($name in @('fsr3','fsr4')) { if ($proof.$name.status -ne 'SESSION_CAPTURE_PASS' -or $proof.$name.on_off_output_equal -ne $true) { throw 'Session provider on/off control incomplete' } }
    if ($DeferObservation) {
        $windowProof=Get-Content -LiteralPath (Join-Path $validation 'window_analysis.json') -Raw | ConvertFrom-Json
        if ($windowProof.status -ne 'PASS') { throw 'Window controls failed' }
        foreach ($name in @('fsr3','fsr4')) {if ($windowProof.$name.status -ne 'WINDOW_CONTROL_PASS' -or $windowProof.$name.on_off_output_equal -ne $true) {throw 'Window control incomplete'}}
    }
} else {
    $proof=Get-Content -LiteralPath (Join-Path $validation 'live_analysis.json') -Raw | ConvertFrom-Json
    if ($proof.status -ne 'LIVE_INSPECTION_PASS') { throw 'Independent on/off inspection validation required' }
    foreach ($name in @('fsr3','fsr4')) { if ($proof.providers.$name.status -ne 'LIVE_INSPECTION_PASS' -or $proof.providers.$name.on_off_output_equal -ne $true) { throw 'Provider on/off control incomplete' } }
}
$provenance=Get-Content -LiteralPath (Join-Path $observation 'provenance.json') -Raw | ConvertFrom-Json
$bootstrap=Get-Content -LiteralPath (Join-Path $observation 'bootstrap.json') -Raw | ConvertFrom-Json
if ($bootstrap.pid -ne $GameProcessId -or $bootstrap.status -ne 'BOOTSTRAP_PASS') { throw 'PID does not match known bootstrap' }
$gameDirectory=$provenance.game_directory
$exe=Join-Path $gameDirectory 'GoWR.exe'
$game=Get-Process -Id $GameProcessId
if ($game.Path -ne $exe) { throw 'Live process image mismatch' }
$observer=Join-Path $repo 'build\ffx_observer.dll'
if ((Get-FileHash -LiteralPath $observer).Hash -ne $provenance.observer_sha256) { throw 'Observer file differs from loaded bootstrap provenance' }
$before=@{}
foreach ($entry in $provenance.before.PSObject.Properties) {
    $before[$entry.Name]=(Get-FileHash -LiteralPath (Join-Path $gameDirectory $entry.Name)).Hash
    if ($before[$entry.Name] -ne $entry.Value) { throw "Audited game image changed: $($entry.Name)" }
}
$inspector=Join-Path $repo 'build\ffx_live_inspector.dll'
if ($Session) { $inspector=Join-Path $repo 'build\ffx_contract_inspector.dll' }
if ($PriorInspectionRun) {
    $prior=Get-Content -LiteralPath (Join-Path $PriorInspectionRun 'provenance.json') -Raw | ConvertFrom-Json
    if ($prior.pid -ne $GameProcessId -or $prior.exit_code -ne 0 -or $prior.inspector_sha256 -ne (Get-FileHash -LiteralPath $inspector).Hash) { throw 'Prior live inspector provenance mismatch' }
    $observer=$inspector
    $inspector=Join-Path $repo $(if ($Session) {'build\ffx_capture_session.dll'} else {'build\ffx_contract_inspector.dll'})
    if ($Session) {$inspector=Join-Path $sessionBin 'ffx_capture_session.dll'}
}
$validationProvenance=Get-Content -LiteralPath (Join-Path $validation 'provenance.json') -Raw | ConvertFrom-Json
if ($Session -and $validationProvenance.command_path_required -eq $true) {
    $pathProof=Get-Content -LiteralPath (Join-Path $validation 'command_path_analysis.json') -Raw | ConvertFrom-Json
    if ($pathProof.status -ne 'COMMAND_PATH_CONTROL_PASS' -or $pathProof.on_off_output_equal -ne $true -or $pathProof.enhanced_positive_control -ne $true) {throw 'Command-path positive/on-off control required'}
}
if ($Session -and $validationProvenance.resource_identity_required -eq $true) {
    $resourceProof=Get-Content -LiteralPath (Join-Path $validation 'resource_path_analysis.json') -Raw | ConvertFrom-Json
    if ($resourceProof.status -ne 'RESOURCE_IDENTITY_CONTROL_PASS') {throw 'Resource identity on/off control required'}
}
if ($Session -and $validationProvenance.boundary_return_required -eq $true) {
    $boundaryProof=Get-Content -LiteralPath (Join-Path $validation 'boundary_return_analysis.json') -Raw | ConvertFrom-Json
    if ($boundaryProof.status -ne 'OUTPUT_BOUNDARY_RETURN_CONTROL_PASS') {throw 'Boundary callback controls required'}
}
if ($Session -and $validationProvenance.live_output_capture_required -eq $true) {
    $boundaryProof=Get-Content -LiteralPath (Join-Path $validation 'boundary_return_analysis.json') -Raw | ConvertFrom-Json
    foreach ($name in @('fsr3','fsr4')) {
        if ($boundaryProof.providers.$name.hook_copy.status -ne 'ISOLATED_BOUNDARY_HOOK_COPY_PASS' -or
            $boundaryProof.providers.$name.hook_copy.on_off_output_equal -ne $true -or
            $boundaryProof.providers.$name.hook_copy.early_poll_blocked -ne $true -or
            $boundaryProof.providers.$name.hook_copy.game_frame_capture_verified -ne $false) {throw 'Output-only arm/poll control required'}
    }
}
if ($Session -and $validationProvenance.live_output_roundtrip_required -eq $true) {
    $boundaryProof=Get-Content -LiteralPath (Join-Path $validation 'boundary_return_analysis.json') -Raw | ConvertFrom-Json
    foreach ($name in @('fsr3','fsr4')) {
        if ($boundaryProof.providers.$name.hook_roundtrip.status -ne 'ISOLATED_BOUNDARY_HOOK_ROUNDTRIP_PASS' -or
            $boundaryProof.providers.$name.hook_roundtrip.on_off_output_equal -ne $true -or
            $boundaryProof.providers.$name.hook_roundtrip.write_back_performed -ne $true -or
            $boundaryProof.providers.$name.hook_roundtrip.early_poll_blocked -ne $true -or
            $boundaryProof.providers.$name.hook_roundtrip.game_frame_capture_verified -ne $false) {throw 'Output roundtrip arm/poll control required'}
    }
}
if ($Session -and $validationProvenance.live_output_patch_required -eq $true) {
    $boundaryProof=Get-Content -LiteralPath (Join-Path $validation 'boundary_return_analysis.json') -Raw | ConvertFrom-Json
    foreach ($name in @('fsr3','fsr4')) {
        if ($boundaryProof.providers.$name.hook_patch.status -ne 'ISOLATED_BOUNDARY_HOOK_PATCH_PASS' -or
            $boundaryProof.providers.$name.hook_patch.outside_patch_exact -ne $true -or
            $boundaryProof.providers.$name.hook_patch.supplied_pixels -ne 64 -or
            $boundaryProof.providers.$name.hook_patch.replacement_pixels_supplied -ne $true -or
            $boundaryProof.providers.$name.hook_patch.early_poll_blocked -ne $true -or
            $boundaryProof.providers.$name.hook_patch.game_frame_capture_verified -ne $false) {throw 'Supplied-pixel patch arm/poll control required'}
    }
}
if ($Session -and $validationProvenance.live_output_filter_required -eq $true) {
    $boundaryProof=Get-Content -LiteralPath (Join-Path $validation 'boundary_return_analysis.json') -Raw | ConvertFrom-Json
    foreach ($name in @('fsr3','fsr4')) {
        if ($boundaryProof.providers.$name.hook_filter.status -ne 'ISOLATED_BOUNDARY_HOOK_FILTER_PASS' -or
            $boundaryProof.providers.$name.hook_filter.baseline_before_equal -ne $true -or
            $boundaryProof.providers.$name.hook_filter.downstream_after_equal -ne $true -or
            $boundaryProof.providers.$name.hook_filter.internal_hook_bypass -ne $true -or
            $boundaryProof.providers.$name.hook_filter.submission_suffix_lists -lt 1 -or
            $boundaryProof.providers.$name.hook_filter.game_frame_capture_verified -ne $false -or
            $boundaryProof.providers.$name.hook_filter.trained_weights -ne $false) {throw 'Dynamic residual filter arm/poll control required'}
    }
}
if ($Session -and $validationProvenance.live_output_network_required -eq $true) {
    $boundaryProof=Get-Content -LiteralPath (Join-Path $validation 'boundary_return_analysis.json') -Raw | ConvertFrom-Json
    foreach ($name in @('fsr3','fsr4')) {
        if ($boundaryProof.providers.$name.hook_network.status -ne 'ISOLATED_BOUNDARY_HOOK_NETWORK_PASS' -or
            $boundaryProof.providers.$name.hook_network.baseline_before_equal -ne $true -or
            $boundaryProof.providers.$name.hook_network.downstream_after_equal -ne $true -or
            $boundaryProof.providers.$name.hook_network.calibration_filter_exact -ne $true -or
            $boundaryProof.providers.$name.hook_network.internal_hook_bypass -ne $true -or
            $boundaryProof.providers.$name.hook_network.submission_suffix_lists -lt 1 -or
            $boundaryProof.providers.$name.hook_network.external_weights -ne $true -or
            $boundaryProof.providers.$name.hook_network.game_frame_capture_verified -ne $false -or
            $boundaryProof.providers.$name.hook_network.trained_weights -ne $false) {throw 'Dynamic weighted network arm/poll control required'}
    }
}
if ($validationProvenance.inspector_sha256 -ne (Get-FileHash -LiteralPath $inspector).Hash) { throw 'Inspector differs from validated binary' }
$suffix=if ($Session) {'gowr_capture_session'} else {'gowr_live_inspection'}
$result=Join-Path $repo ('results\{0}_{1}' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'),$suffix)
New-Item -ItemType Directory -Path $result | Out-Null
$exitCode=-1
$client=if ($Session) {Join-Path $sessionBin 'ffx_live_attach.exe'} else {Join-Path $repo 'build\ffx_live_attach.exe'}
if ($DeferObservation -and $validationProvenance.client_sha256 -ne (Get-FileHash -LiteralPath $client).Hash) {throw 'Deferred client differs from control provenance'}
try {
    $logName=if ($Session) {'session.jsonl'} else {'live.jsonl'}
    $attachArguments=@($GameProcessId,$exe,$observer,$inspector,(Join-Path $result $logName))
    if ($DeferObservation) {$attachArguments+='deferred'}
    & $client @attachArguments
    $exitCode=$LASTEXITCODE
} finally {
    $after=@{}
    foreach ($entry in $before.Keys) { $after[$entry]=(Get-FileHash -LiteralPath (Join-Path $gameDirectory $entry)).Hash }
    @{pid=$GameProcessId;process_start=$game.StartTime.ToString('o');observation_run=$observation;validation_run=$validation;
      prior_inspection_run=$PriorInspectionRun;chained_module=$observer;chained_module_sha256=(Get-FileHash -LiteralPath $observer).Hash;
      inspector_sha256=(Get-FileHash -LiteralPath $inspector).Hash;observer_sha256=$provenance.observer_sha256;
      client_sha256=(Get-FileHash -LiteralPath $client).Hash;observation_deferred=[bool]$DeferObservation;
      before=$before;after=$after;exit_code=$exitCode;game_files_deployed=@();gpu_copies=0;game_runtime_ready=$false} |
      ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $result 'provenance.json') -Encoding utf8
}
Write-Host "Result: $result"
if ($exitCode -ne 0) { throw 'Live attach failed/uncertain; do not retry automatically. Game left running.' }
foreach ($entry in $before.Keys) { if ($after[$entry] -ne $before[$entry]) { throw "Audited image changed during inspection: $entry" } }
