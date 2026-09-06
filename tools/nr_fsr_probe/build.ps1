param([Parameter(Mandatory)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'scripts/build_common.ps1')
$out=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $out){throw 'New build directory required'}
New-Item -ItemType Directory -Path $out | Out-Null
$e=Resolve-BuildEnv
$argsBuild=@('--driver-mode=cl','/nologo','/EHsc','/std:c++20','/O2','/MD','-fuse-ld=lld-link','-D_CRT_SECURE_NO_WARNINGS')
$argsBuild+=Get-SdkIncludeArgs $e
$argsBuild+=@((Join-Path $PSScriptRoot 'nr_fsr_probe.cpp'),('/Fo:'+ (Join-Path $out 'nr_fsr_probe.obj')),('/Fe:'+ (Join-Path $out 'nr_fsr_probe.exe')),'/link')
$argsBuild+=(Get-SdkLibArgs $e)+(Get-MsvcLibArgs $e)+@('dxgi.lib','d3d12.lib')
& $e.clangxx @argsBuild
if($LASTEXITCODE -ne 0){throw 'Build failed'}
