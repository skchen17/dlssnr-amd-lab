param([Parameter(Mandatory=$true)][string]$SessionDirectory)
$ErrorActionPreference='Stop'
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Normal game exit required before recovery'}
$root=(Resolve-Path -LiteralPath $SessionDirectory).Path
$record=Get-Content -LiteralPath (Join-Path $root 'deployment.json') -Raw | ConvertFrom-Json
$game=[IO.Path]::GetFullPath($record.game_directory)
if($game -ne 'C:\DATA\GAME\GODOFWAR'){throw 'Unexpected deployment target'}
foreach($entry in $record.original_hashes){if((Get-FileHash -LiteralPath $entry.Path).Hash -ne $entry.Hash){throw 'Original game file changed; review before recovery'}}
# ReShade.ini/log are generated at runtime; only binary identities are immutable.
foreach($entry in $record.installed | Where-Object {$_.Path -notlike '*.ini'}){if((Get-FileHash -LiteralPath $entry.Path).Hash -ne $entry.Hash){throw 'Installed binary changed; preserve it for review'}}
$recovery=Join-Path $root 'recovered_game_files'
if(Test-Path -LiteralPath $recovery){throw 'Recovery already exists'}
New-Item -ItemType Directory -Path $recovery | Out-Null
foreach($name in $record.created_files){
 if($name -notin @('dxgi.dll','native_rocm_preview.addon64','ReShade.ini','ReShade.log')){throw 'Unexpected recovery filename'}
 $target=[IO.Path]::GetFullPath((Join-Path $game $name));if((Split-Path -Parent $target) -ne $game){throw 'Recovery escaped game directory'}
 if(Test-Path -LiteralPath $target){Move-Item -LiteralPath $target -Destination (Join-Path $recovery $name)}
}
@{restored=$true;original_files_unchanged=$true;recoverable_directory=$recovery} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'restored.json')
Get-Content -LiteralPath (Join-Path $root 'restored.json')
