param([Parameter(Mandatory=$true)][string]$StageDirectory,
      [Parameter(Mandatory=$true)][string]$SessionRun,
      [ValidateSet('start','stop','status')][string]$Action='status')
$ErrorActionPreference='Stop'
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
$session=(Resolve-Path -LiteralPath $SessionRun).Path
$gate=Get-Content -LiteralPath (Join-Path $stage 'resident_gate.json') -Raw | ConvertFrom-Json
if($gate.status -ne 'RESIDENT_BRINGUP_GATE_PASS'){throw 'Resident gate missing'}
foreach($entry in $gate.files.PSObject.Properties){
    if($Action -ne 'start' -and $entry.Name -eq 'worker_name.txt'){continue}
    if((Get-FileHash -LiteralPath (Join-Path $stage $entry.Name)).Hash -ne $entry.Value){throw 'Resident stage changed'}
}
$provenance=Get-Content -LiteralPath (Join-Path $session 'provenance.json') -Raw | ConvertFrom-Json
$dll=Join-Path $stage 'ffx_capture_session.dll'
if($provenance.exit_code -ne 0 -or !$provenance.observation_deferred -or $provenance.inspector_sha256 -ne (Get-FileHash -LiteralPath $dll).Hash){throw 'Session provenance mismatch'}
$game=Get-Process -Id $provenance.pid
$exe='C:\DATA\GAME\GODOFWAR\GoWR.exe'
if($game.Path -ne $exe -or $game.StartTime.ToUniversalTime().Ticks -ne ([datetime]$provenance.process_start).ToUniversalTime().Ticks){throw 'Game process changed'}
if($Action -eq 'start'){
    $worker=Get-Content -LiteralPath (Join-Path $gate.worker_run 'worker.json') -Raw | ConvertFrom-Json
    if($worker.status -ne 'READY' -or $worker.frames.Count){throw 'Worker no longer fresh/ready'}
}
& (Join-Path $stage 'ffx_session_control.exe') $game.Id $exe $dll "resident-$Action"
if($LASTEXITCODE -ne 0){throw 'Resident command rejected/failed; do not retry uncertain injection'}
if($Action -ne 'status'){& (Join-Path $stage 'ffx_session_control.exe') $game.Id $exe $dll 'resident-status'}
Get-Content -LiteralPath (Join-Path $session 'resident_status.json')
