param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'build_common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$dest=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $dest){throw 'Fresh build directory required'}
New-Item -ItemType Directory -Path $dest | Out-Null
$taskEnv=Resolve-BuildEnv
$includes=Get-SdkIncludeArgs $taskEnv
$libs=(Get-SdkLibArgs $taskEnv)+(Get-MsvcLibArgs $taskEnv)
# Headers from the PROJECT runtime; compiler installation is read-only.
$compileArgs=@('--driver-mode=cl','/nologo','/EHsc','/std:c++20','/O2','/MD','/LD','-fuse-ld=lld-link')+$includes
$compileArgs+=@('/I'+(Join-Path $repo '.venv-rocm\Lib\site-packages\_rocm_sdk_core\include'))
$compileArgs+=@((Join-Path $repo 'tools\native_frame_bridge\native_frame_bridge.cpp'),('/Fo'+(Join-Path $dest 'bridge.obj')),('/Fe:'+(Join-Path $dest 'native_frame_bridge.dll')),'/link')+$libs+@('d3d12.lib','dxgi.lib')
& $taskEnv.clangxx @compileArgs
if($LASTEXITCODE -ne 0){throw 'Native frame bridge build failed'}
Get-FileHash -LiteralPath (Join-Path $dest 'native_frame_bridge.dll')
