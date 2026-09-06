param([Parameter(Mandatory=$true)][string]$ValidatedRun,[Parameter(Mandatory=$true)][string]$OutputDirectory,[ValidateSet(1,12)][int]$Frames=1)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'native_preview_provenance.ps1')
$game='C:\DATA\GAME\GODOFWAR'
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Normal game exit required'}
$proofRoot=(Resolve-Path -LiteralPath $ValidatedRun).Path
$proof=Get-Content -LiteralPath (Join-Path $proofRoot 'assessment.json') -Raw | ConvertFrom-Json
$hostProof=Get-Content -LiteralPath (Join-Path $proofRoot 'host.json') -Raw | ConvertFrom-Json
$provenance=Get-Content -LiteralPath (Join-Path $proofRoot 'supervisor.json') -Raw | ConvertFrom-Json
if(-not $proof.pass -or $provenance.mode -ne 'network'){throw 'Same-size native ReShade proof required'}
if($hostProof.format -notin @(24,28)){throw 'Explicit validated pixel format required'}
$sourceHashes=Get-NativePreviewSourceHashes $repo
Assert-NativePreviewSources $provenance.source_hashes $sourceHashes
foreach($entry in $provenance.hashes){if((Get-FileHash -LiteralPath $entry.Path).Hash -ne $entry.Hash){throw 'Validated binary changed'}}
$out=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $out){throw 'Fresh game session required'}
$targets=@('dxgi.dll','native_rocm_preview.addon64','ReShade.ini','ReShade.log')
foreach($t in $targets){if(Test-Path -LiteralPath (Join-Path $game $t)){throw "Existing game file preserved: $t"}}
New-Item -ItemType Directory -Path $out | Out-Null
Copy-Item -LiteralPath (Join-Path $proofRoot 'native_frame_bridge.dll') -Destination (Join-Path $out 'native_frame_bridge.dll')
$originals=@('GoWR.exe','amd_fidelityfx_dx12.dll','sl.interposer.dll','version.dll')
$originalHashes=@($originals | ForEach-Object {Get-FileHash -LiteralPath (Join-Path $game $_)})
foreach($t in @('dxgi.dll','native_rocm_preview.addon64')){Copy-Item -LiteralPath (Join-Path $proofRoot $t) -Destination (Join-Path $game $t)}
'Techniques=NativePreviewCopy@NativePreview.fx' | Set-Content -LiteralPath (Join-Path $out 'NativePreview.ini') -Encoding utf8
@"
[GENERAL]
EffectSearchPaths=$repo\tools\native_reshade
TextureSearchPaths=$out
PresetPath=$out\NativePreview.ini
PerformanceMode=1
[OVERLAY]
TutorialProgress=4
ShowClock=0
ShowFPS=0
ShowFrameTime=0
[SCREENSHOT]
SavePath=$out
FileFormat=1
SaveBeforeShot=0
SaveOverlayShot=0
SavePresetFile=0
"@ | Set-Content -LiteralPath (Join-Path $game 'ReShade.ini') -Encoding utf8
$installed=@('dxgi.dll','native_rocm_preview.addon64','ReShade.ini') | ForEach-Object {Get-FileHash -LiteralPath (Join-Path $game $_)}
@{game_directory=$game;created_files=$targets;installed=$installed;original_hashes=$originalHashes;source_hashes=$sourceHashes;validated_run=$proofRoot;width=$hostProof.width;height=$hostProof.height;format=$hostProof.format;frames=$Frames;armed=$false;gpu_pixels_only=$true;diagnostic_display_screenshot=$true;hud_protected=$false;hdr_accepted=$false;realtime_accepted=$false} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $out 'deployment.json')
$names=@('NR_PREVIEW_REPO','NR_PREVIEW_RUN','NR_PREVIEW_MODE','NR_PREVIEW_WIDTH','NR_PREVIEW_HEIGHT','NR_PREVIEW_FRAMES','NR_PREVIEW_AUDIT','NR_PREVIEW_SCREENSHOT','NR_PREVIEW_FORMAT')
$values=@($repo,$out,'network',"$($hostProof.width)","$($hostProof.height)","$Frames",'0','1',"$($hostProof.format)");$old=@{}
try{
 for($i=0;$i -lt $names.Count;$i++){$old[$names[$i]]=[Environment]::GetEnvironmentVariable($names[$i]);[Environment]::SetEnvironmentVariable($names[$i],$values[$i])}
 $p=Start-Process -FilePath (Join-Path $game 'GoWR.exe') -WorkingDirectory $game -WindowStyle Hidden -PassThru
 @{pid=$p.Id;session=$out;armed=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'launch.json')
 Get-Content -LiteralPath (Join-Path $out 'launch.json')
}finally{foreach($n in $names){[Environment]::SetEnvironmentVariable($n,$old[$n])}}
