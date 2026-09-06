param([Parameter(Mandatory=$true)][string]$StageDirectory)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Normal game exit required before fresh observation'}
$proof=Get-Content -LiteralPath (Join-Path $stage 'controls\provenance.json') -Raw | ConvertFrom-Json
if($proof.session_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $stage 'ffx_capture_session.dll')).Hash -or @($proof.runs | Where-Object exit_code -ne 0).Count){throw 'Fresh staged controls not passed'}
function New-Result([scriptblock]$Action,[string]$Pattern){
    $before=@(Get-ChildItem -LiteralPath (Join-Path $repo 'results') -Directory -Filter $Pattern | Select-Object -ExpandProperty FullName)
    & $Action | Out-Host
    $created=@(Get-ChildItem -LiteralPath (Join-Path $repo 'results') -Directory -Filter $Pattern | Where-Object {$_.FullName -notin $before})
    if($created.Count -ne 1){throw 'Expected one new observation directory'}
    return $created[0].FullName
}
$observation=New-Result {& (Join-Path $PSScriptRoot 'run_gowr_ffx_observation.ps1') -DecodeUpscaleMetadata} '*_gowr_ffx_observation'
$bootstrap=Get-Content -LiteralPath (Join-Path $observation 'bootstrap.json') -Raw | ConvertFrom-Json
if($bootstrap.status -ne 'BOOTSTRAP_PASS'){throw 'Bootstrap incomplete'}
$gameId=[int]$bootstrap.pid
$first=New-Result {& (Join-Path $PSScriptRoot 'run_gowr_live_inspection.ps1') -GameProcessId $gameId -ObservationRun $observation -ValidationRun (Join-Path $repo 'results\20260905_112352_927_ffx_dispatch')} '*_gowr_live_inspection'
$second=New-Result {& (Join-Path $PSScriptRoot 'run_gowr_live_inspection.ps1') -GameProcessId $gameId -ObservationRun $observation -ValidationRun (Join-Path $repo 'results\20260905_114528_108_ffx_dispatch') -PriorInspectionRun $first} '*_gowr_live_inspection'
$session=New-Result {& (Join-Path $PSScriptRoot 'run_gowr_live_inspection.ps1') -GameProcessId $gameId -ObservationRun $observation -ValidationRun (Join-Path $stage 'controls') -PriorInspectionRun $second -Session -DeferObservation -BinaryDirectory $stage} '*_gowr_capture_session'
@{pid=$gameId;observation=$observation;session=$session;stage=$stage;network_enabled=$false;metadata_only=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $session 'boundary_launch.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $session 'boundary_launch.json')
