param([Parameter(Mandatory=$true)][string]$SessionRun,
      [ValidateSet('start','stop','status')][string]$Action='status')
$ErrorActionPreference='Stop'
$session=(Resolve-Path -LiteralPath $SessionRun).Path
$p=Get-Content -LiteralPath (Join-Path $session 'provenance.json') -Raw | ConvertFrom-Json
if(!$p.display_referred_debug_route){throw 'Not a present session'}
$game=Get-Process -Id $p.pid -ErrorAction Stop
if($game.StartTime.ToUniversalTime().Ticks -ne ([datetime]$p.process_start).ToUniversalTime().Ticks){throw 'PID identity changed'}
foreach($entry in $p.files.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $p.stage $entry.Name)).Hash -ne $entry.Value){throw 'Stage binary changed'}}
if((Get-FileHash -LiteralPath (Join-Path $p.stage 'worker_name.txt')).Hash -ne $p.worker_name_sha256){throw 'Worker identity changed'}
if($Action -eq 'start'){
    $ready=Get-Content -LiteralPath (Join-Path $p.worker 'worker.json') -Raw | ConvertFrom-Json
    if($ready.status -ne 'READY' -or $ready.frames.Count -ne 0){throw 'Worker expired or already used'}
}
& (Join-Path $p.stage 'ffx_session_control.exe') $p.pid 'C:\DATA\GAME\GODOFWAR\GoWR.exe' (Join-Path $p.stage 'resident_present.dll') "resident-$Action"
if($LASTEXITCODE -ne 0){throw 'Present control rejected/failed'}
if($Action -ne 'status'){& (Join-Path $p.stage 'ffx_session_control.exe') $p.pid 'C:\DATA\GAME\GODOFWAR\GoWR.exe' (Join-Path $p.stage 'resident_present.dll') 'resident-status'}
Get-Content -LiteralPath (Join-Path $session 'resident_status.json')

