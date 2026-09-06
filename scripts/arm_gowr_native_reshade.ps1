param([Parameter(Mandatory=$true)][string]$SessionDirectory)
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $SessionDirectory).Path
$deployment=Get-Content -LiteralPath (Join-Path $root 'deployment.json') -Raw | ConvertFrom-Json
$launch=Get-Content -LiteralPath (Join-Path $root 'launch.json') -Raw | ConvertFrom-Json
$p=Get-Process -Id $launch.pid -ErrorAction Stop
if($p.Path -ne (Join-Path $deployment.game_directory 'GoWR.exe')){throw 'Game PID identity differs'}
foreach($entry in $deployment.source_hashes){if((Get-FileHash -LiteralPath $entry.Path).Hash -ne $entry.Hash){throw 'Source changed after deployment'}}
$events=@(Get-Content -LiteralPath (Join-Path $root 'addon.jsonl') | ForEach-Object {$_ | ConvertFrom-Json})
if(@($events | Where-Object {$_.event -in 'failure','worker_ready','frame_written'}).Count){throw 'Already started or failed; no retry'}
$target=@($events | Where-Object event -eq 'effect_target' | Select-Object -Last 1)
if($target.Count -ne 1 -or $target[0].width -ne $deployment.width -or $target[0].height -ne $deployment.height -or $target[0].format -ne $deployment.format -or $target[0].color_space -ne 1){throw 'Actual size/format/color does not match offline proof'}
# Empty marker is explicit one-shot arming; pixels do not enter the control file.
New-Item -ItemType File -Path (Join-Path $root 'ARM') -ErrorAction Stop | Out-Null
Write-Output 'Armed bounded same-frame SDR preview. Not 60 FPS or HUD-safe acceptance.'
