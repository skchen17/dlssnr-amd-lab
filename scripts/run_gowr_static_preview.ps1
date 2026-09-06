param([Parameter(Mandatory=$true)][string]$StageDirectory,
      [Parameter(Mandatory=$true)][string]$SessionRun,
      [ValidateSet('network','input','stop','status')][string]$Action='status')
$ErrorActionPreference='Stop'
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
$session=(Resolve-Path -LiteralPath $SessionRun).Path
$provenance=Get-Content -LiteralPath (Join-Path $session 'provenance.json') -Raw | ConvertFrom-Json
$gate=Get-Content -LiteralPath (Join-Path $stage 'preview_gate\gate.json') -Raw | ConvertFrom-Json
if($gate.status -ne 'STATIC_PREVIEW_GATE_PASS'){throw 'Preview controls not accepted'}
foreach($entry in $gate.files.PSObject.Properties){
    if($Action -in @('stop','status') -and $entry.Name -like 'preview_*'){continue}
    if((Get-FileHash -LiteralPath (Join-Path $stage $entry.Name)).Hash -ne $entry.Value){throw "Stage file changed: $($entry.Name)"}
}
$dll=Join-Path $stage 'ffx_capture_session.dll'
if($provenance.exit_code -ne 0 -or !$provenance.observation_deferred -or $provenance.inspector_sha256 -ne (Get-FileHash -LiteralPath $dll).Hash){throw 'Session provenance mismatch'}
$game=Get-Process -Id $provenance.pid
$exe='C:\DATA\GAME\GODOFWAR\GoWR.exe'
$recordedStart=[datetime]$provenance.process_start
if($game.Path -ne $exe -or $game.StartTime.ToUniversalTime().Ticks -ne $recordedStart.ToUniversalTime().Ticks){throw 'Game process identity changed'}
if($Action -in @('input','network')) {
    $manifest=Get-Content -LiteralPath (Join-Path $stage 'preview_manifest.json') -Raw | ConvertFrom-Json
    foreach($entry in $manifest.files.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $stage $entry.Name)).Hash -ne $entry.Value){throw 'Preview payload hash mismatch'}}
}
& (Join-Path $stage 'ffx_session_control.exe') $game.Id $exe $dll "preview-$Action"
if($LASTEXITCODE -ne 0){throw 'Preview command failed; do not retry uncertain injection'}
if($Action -ne 'status') {& (Join-Path $stage 'ffx_session_control.exe') $game.Id $exe $dll 'preview-status'}
Get-Content -LiteralPath (Join-Path $session 'static_preview_status.json')
