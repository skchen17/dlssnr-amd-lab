param([Parameter(Mandatory=$true)][string]$SessionRun,
      [ValidateSet('manual_stop','automatic_timeout')][string]$Phase)
$ErrorActionPreference='Stop'
$session=(Resolve-Path -LiteralPath $SessionRun).Path
$target=Join-Path $session ($Phase+'_audit.json')
if(Test-Path -LiteralPath $target){throw 'Audit exists; do not overwrite'}
$provenance=Get-Content (Join-Path $session 'provenance.json') -Raw | ConvertFrom-Json
$status=Get-Content (Join-Path $session 'static_preview_status.json') -Raw | ConvertFrom-Json
$game=Get-Process -Id $provenance.pid
if($game.StartTime.ToUniversalTime().Ticks -ne ([datetime]$provenance.process_start).ToUniversalTime().Ticks){throw 'PID reused'}
$hashes=@{};$gameFolder=Split-Path -Parent $game.Path
foreach($entry in $provenance.before.PSObject.Properties){
    $hashes[$entry.Name]=(Get-FileHash -LiteralPath (Join-Path $gameFolder $entry.Name)).Hash
    if($hashes[$entry.Name] -ne $entry.Value){throw 'Audited game image changed'}
}
$eventQuery='ok';$events=@()
try{$events=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$game.StartTime;Id=4101,41,6008} -ErrorAction Stop | Select-Object TimeCreated,Id,ProviderName)}
catch{if($_.FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){$eventQuery='unavailable'}}
$screens=@{}
foreach($name in @('preview_network.png','preview_input.png','preview_stopped.png','preview_timeout.png')){
    $path=Join-Path $session $name;if(Test-Path -LiteralPath $path){$screens[$name]=(Get-FileHash -LiteralPath $path).Hash}
}
@{phase=$Phase;timestamp=(Get-Date).ToString('o');pid=$game.Id;process_responding=$game.Responding;
  preview_status=$status;game_hashes_unchanged=$true;game_hashes=$hashes;
  system_event_query=$eventQuery;display_reset_or_restart_events=$events;screenshots=$screens;
  live_network_inference=$false;dlss5_quality_verified=$false} | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $target -Encoding utf8
Get-Content -LiteralPath $target
