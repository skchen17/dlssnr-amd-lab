param([Parameter(Mandatory=$true)][string]$StageDirectory,[Parameter(Mandatory=$true)][string]$OutputDirectory,
 [ValidateSet('observe','transport','network')][string]$Mode='observe',[int]$Width=640,[int]$Height=360,[ValidateSet(1,12)][int]$Frames=1,[ValidateSet(24,28)][int]$Format=28)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'native_preview_provenance.ps1')
$sourceHashes=Get-NativePreviewSourceHashes $repo
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Exit game before independent GPU probe'}
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
$out=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $out){throw 'Fresh probe directory required'}
New-Item -ItemType Directory -Path $out | Out-Null
foreach($file in @('native_reshade_host.exe','native_rocm_preview.addon64','native_frame_bridge.dll')){Copy-Item -LiteralPath (Join-Path $stage $file) -Destination (Join-Path $out $file)}
$reshade=Join-Path (Split-Path $repo -Parent) 'dlss5\extracted_runtime\ReShade64.dll'
Copy-Item -LiteralPath $reshade -Destination (Join-Path $out 'dxgi.dll')
New-Item -ItemType Directory -Path (Join-Path $out 'Shaders'),(Join-Path $out 'Textures') | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'tools/native_reshade/NativePreview.fx') -Destination (Join-Path $out 'Shaders/NativePreview.fx')
@"
[GENERAL]
EffectSearchPaths=.\Shaders
TextureSearchPaths=.\Textures
PreprocessorDefinitions=
PerformanceMode=1
PresetPath=.\NativePreview.ini
[OVERLAY]
TutorialProgress=4
ShowClock=0
ShowFPS=0
ShowFrameTime=0
ShowScreenshotMessage=0
ShowPresetName=0
NoFontScaling=1
[SCREENSHOT]
SavePath=$out
FileFormat=1
SaveBeforeShot=0
SaveOverlayShot=0
SavePresetFile=0
"@ | Set-Content -LiteralPath (Join-Path $out 'ReShade.ini') -Encoding utf8
'Techniques=NativePreviewCopy@NativePreview.fx' | Set-Content -LiteralPath (Join-Path $out 'NativePreview.ini') -Encoding utf8
if($Mode -ne 'observe'){New-Item -ItemType File -Path (Join-Path $out 'ARM') | Out-Null}
$names=@('NR_PREVIEW_REPO','NR_PREVIEW_RUN','NR_PREVIEW_MODE','NR_PREVIEW_WIDTH','NR_PREVIEW_HEIGHT','NR_PREVIEW_FRAMES','NR_PREVIEW_AUDIT','NR_PREVIEW_SCREENSHOT','NR_PREVIEW_FORMAT')
$values=@($repo,$out,$Mode,"$Width","$Height","$Frames",'1','1',"$Format");$old=@{}
try{
 for($i=0;$i -lt $names.Count;$i++){$old[$names[$i]]=[Environment]::GetEnvironmentVariable($names[$i]);[Environment]::SetEnvironmentVariable($names[$i],$values[$i])}
 $hashes=@(Get-ChildItem -LiteralPath $out -File | Where-Object Extension -in '.exe','.dll','.addon64' | Get-FileHash)
 $p=Start-Process -FilePath (Join-Path $out 'native_reshade_host.exe') -ArgumentList @(('"'+$out+'"'),"$Width","$Height",'18',"$Format") -WorkingDirectory $out -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $out 'stdout.log') -RedirectStandardError (Join-Path $out 'stderr.log')
 $deadline=(Get-Date).AddSeconds(120)
 while(-not $p.HasExited -and (Get-Date) -lt $deadline){Start-Sleep -Milliseconds 500;$p.Refresh()}
 $timedOut=-not $p.HasExited
 @{pid=$p.Id;mode=$Mode;network_frames=$Frames;width=$Width;height=$Height;normal_exit=(-not $timedOut);exit_code=$(if($timedOut){$null}else{$p.ExitCode});host_timeout=$timedOut;automatic_kill=$false;hashes=$hashes;source_hashes=$sourceHashes} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $out 'supervisor.json')
 Assert-NativePreviewSources $sourceHashes (Get-NativePreviewSourceHashes $repo)
 if($timedOut){throw 'Host timeout, process retained; review required, no retry'}
 if($p.ExitCode -ne 0){throw 'Probe failed; review logs before another GPU run'}
 Get-Content -LiteralPath (Join-Path $out 'host.json')
 if(Test-Path -LiteralPath (Join-Path $out 'addon.jsonl')){Get-Content -LiteralPath (Join-Path $out 'addon.jsonl')}
 $events=@(Get-Content -LiteralPath (Join-Path $out 'addon.jsonl') | ForEach-Object {$_ | ConvertFrom-Json})
 if(@($events | Where-Object event -eq 'failure').Count){throw 'Addon rejected probe, not an accepted integration'}
 if($Mode -ne 'observe' -and @($events | Where-Object event -eq 'bounded_session_complete').Count -ne 1){throw 'Native session did not complete'}
}finally{foreach($n in $names){[Environment]::SetEnvironmentVariable($n,$old[$n])}}
