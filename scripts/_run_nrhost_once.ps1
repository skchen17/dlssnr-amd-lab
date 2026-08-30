# One-shot wrapper: build + run nr_host, capture logs with safe error handling.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repo 'results\20260830_170305'

# 1) build
$out = & (Join-Path $PSScriptRoot 'build_all.ps1') -Only nr_host 2>&1 | Out-String
Set-Content (Join-Path $runDir 'nr_host_build.log') $out
Write-Host "build exit=$LASTEXITCODE"

# 2) run (no proprietary DLLs set -> expect BLOCKED_MISSING_PREREQUISITE, exit 3)
Push-Location $repo
$out2 = & (Join-Path $repo 'build\nr_host.exe') --frames 1 --width 512 --height 512 --trace --json (Join-Path $runDir 'nr_host.json') 2>&1 | Out-String
Set-Content (Join-Path $runDir 'nr_host_run.log') $out2
Write-Host "run exit=$LASTEXITCODE"
Pop-Location
