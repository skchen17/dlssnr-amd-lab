param([Parameter(Mandatory=$true)][string]$SessionRun,
      [ValidateSet('after_run','after_exit')][string]$Phase='after_run')
$ErrorActionPreference='Stop'
$session=(Resolve-Path -LiteralPath $SessionRun).Path
$target=Join-Path $session ($Phase+'_audit.json')
if(Test-Path -LiteralPath $target){throw 'Audit already exists'}
$p=Get-Content -LiteralPath (Join-Path $session 'provenance.json') -Raw | ConvertFrom-Json
$status=Get-Content -LiteralPath (Join-Path $session 'resident_status.json') -Raw | ConvertFrom-Json
$game=Get-Process -Id $p.pid -ErrorAction SilentlyContinue
if($game -and $game.StartTime.ToUniversalTime().Ticks -ne ([datetime]$p.process_start).ToUniversalTime().Ticks){throw 'PID reused'}
if($Phase -eq 'after_exit' -and $game){throw 'Game still running'}
if($Phase -eq 'after_run' -and !$game){throw 'Game absent'}
$gameFolder='C:\DATA\GAME\GODOFWAR';$hashes=@{}
foreach($entry in $p.before.PSObject.Properties){$hashes[$entry.Name]=(Get-FileHash -LiteralPath (Join-Path $gameFolder $entry.Name)).Hash;if($hashes[$entry.Name] -ne $entry.Value){throw 'Game binary changed'}}
$query='ok';$events=@()
try{$events=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=([datetime]$p.process_start);Id=4101,41,6008} -ErrorAction Stop | Select-Object TimeCreated,Id,ProviderName)}
catch{if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){$query='unavailable'}}
$screens=@{}
foreach($name in @('resident_before.png','resident_running.png','resident_running_late.png','resident_stopped.png','resident_settings.png','hdr_before.png','hdr_restored.png')){$path=Join-Path $session $name;if(Test-Path -LiteralPath $path){$screens[$name]=(Get-FileHash -LiteralPath $path).Hash}}
@{phase=$Phase;timestamp=(Get-Date).ToString('o');pid=$p.pid;process_present=[bool]$game;process_responding=$(if($game){$game.Responding}else{$null});resident_status=$status;game_hashes_unchanged=$true;game_hashes=$hashes;system_event_query=$query;display_reset_or_restart_events=$events;screenshots=$screens;dlss5_quality_verified=$false} | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $target -Encoding utf8
Get-Content -LiteralPath $target
