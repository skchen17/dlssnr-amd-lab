param([string]$Python='C:\DATA\Tools\ANACONDA\python.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_observer_selftest
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$result=Join-Path $repo ('results\{0}_ffx_observer_selftest' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $result | Out-Null
& (Join-Path $repo 'build\ffx_observer_selftest.exe') (Join-Path $repo 'build\ffx_observer.dll') $result
if ($LASTEXITCODE -ne 0) { throw 'Observer selftest failed' }
& $Python (Join-Path $PSScriptRoot 'analyze_ffx_observer_selftest.py') $result
if ($LASTEXITCODE -ne 0) { throw 'Observer artifact validation failed' }
Write-Host "Result: $result"
