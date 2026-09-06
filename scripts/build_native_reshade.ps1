param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'build_common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$dest=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $dest){throw 'Fresh stage directory required'}
New-Item -ItemType Directory -Path $dest | Out-Null
$taskEnv=Resolve-BuildEnv
$common=@('--driver-mode=cl','/nologo','/EHsc','/std:c++20','/O2','/MD','-fuse-ld=lld-link')+(Get-SdkIncludeArgs $taskEnv)
$common+=@('/I'+(Join-Path $repo '.venv-rocm\Lib\site-packages\_rocm_sdk_core\include'))
$common+=@('/I'+(Join-Path $repo 'third_party\DLSS5-Feeder\external\reshade\include'))
$link=(Get-SdkLibArgs $taskEnv)+(Get-MsvcLibArgs $taskEnv)+@('d3d12.lib','dxgi.lib','d3dcompiler.lib','user32.lib')
foreach($job in @(@('tools/native_reshade/addon.cpp','native_rocm_preview.addon64'),@('tools/native_reshade/host.cpp','native_reshade_host.exe'),@('tools/native_frame_bridge/native_frame_bridge.cpp','native_frame_bridge.dll'))){
    $args=$common
    if($job[1] -notlike '*.exe'){$args+=@('/LD')}
    $args+=@((Join-Path $repo $job[0]),('/Fo'+(Join-Path $dest ($job[1]+'.obj'))),('/Fe:'+(Join-Path $dest $job[1])),'/link')+$link
    & $taskEnv.clangxx @args
    if($LASTEXITCODE -ne 0){throw "Failed to build $($job[1])"}
}
Get-FileHash -LiteralPath (Join-Path $dest 'native_rocm_preview.addon64'),(Join-Path $dest 'native_frame_bridge.dll'),(Join-Path $dest 'native_reshade_host.exe')
