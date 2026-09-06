param([Parameter(Mandatory=$true)][string]$StageDirectory)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Exit GoWR normally before starting a new session'}
$gate=Get-Content -LiteralPath (Join-Path $stage 'resident_gate.json') -Raw | ConvertFrom-Json
if($gate.status -ne 'RESIDENT_BRINGUP_GATE_PASS'){throw 'Stage not validated'}
foreach($entry in $gate.files.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $stage $entry.Name)).Hash -ne $entry.Value){throw 'Stage changed'}}
function Invoke-NewResult([scriptblock]$Action,[string]$Pattern){
    $before=@(Get-ChildItem -LiteralPath (Join-Path $repo 'results') -Directory -Filter $Pattern | Select-Object -ExpandProperty FullName)
    & $Action | Out-Host
    $created=@(Get-ChildItem -LiteralPath (Join-Path $repo 'results') -Directory -Filter $Pattern | Where-Object {$_.FullName -notin $before})
    if($created.Count -ne 1){throw 'Expected exactly one new validated result directory'}
    return $created[0].FullName
}
$observation=Invoke-NewResult {& (Join-Path $PSScriptRoot 'run_gowr_ffx_observation.ps1') -DecodeUpscaleMetadata} '*_gowr_ffx_observation'
$bootstrap=Get-Content -LiteralPath (Join-Path $observation 'bootstrap.json') -Raw | ConvertFrom-Json
if($bootstrap.status -ne 'BOOTSTRAP_PASS'){throw 'Bootstrap incomplete'}
$gameId=[int]$bootstrap.pid
$first=Invoke-NewResult {& (Join-Path $PSScriptRoot 'run_gowr_live_inspection.ps1') -GameProcessId $gameId -ObservationRun $observation -ValidationRun (Join-Path $repo 'results\20260905_112352_927_ffx_dispatch')} '*_gowr_live_inspection'
$second=Invoke-NewResult {& (Join-Path $PSScriptRoot 'run_gowr_live_inspection.ps1') -GameProcessId $gameId -ObservationRun $observation -ValidationRun (Join-Path $repo 'results\20260905_114528_108_ffx_dispatch') -PriorInspectionRun $first} '*_gowr_live_inspection'
$session=Invoke-NewResult {& (Join-Path $PSScriptRoot 'run_gowr_live_inspection.ps1') -GameProcessId $gameId -ObservationRun $observation -ValidationRun (Join-Path $stage 'controls') -PriorInspectionRun $second -Session -DeferObservation -BinaryDirectory $stage} '*_gowr_capture_session'
@{pid=$gameId;observation=$observation;first_inspection=$first;contract_inspection=$second;session=$session;stage=$stage;network_enabled=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $session 'resident_launch.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $session 'resident_launch.json')
