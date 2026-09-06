$ErrorActionPreference = 'Continue'
$repo = 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$exe = Join-Path $repo 'build\ngx_abi_probe.exe'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $repo "results\$ts"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$dll = Join-Path $repo 'third_party\nvidia-dlss\lib\Windows_x86_64\rel\nvngx_dlss.dll'
Write-Host '--- run 1: no DLL (compile-time layer only, expect BLOCKED_MISSING_PREREQUISITE exit 3)'
& $exe
Write-Host "exit=$LASTEXITCODE"
Write-Host '--- run 2: official rel DLL'
& $exe --dll $dll --json (Join-Path $out 'ngx_abi_test.json')
Write-Host "exit=$LASTEXITCODE"
Write-Host "json: $out\ngx_abi_test.json"
