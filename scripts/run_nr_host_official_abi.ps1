# Round 2 official-ABI nr_host run (persisted evidence for FINAL_REPORT chapter)
$ErrorActionPreference = 'Continue'
$repo = 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$out = Join-Path $repo "results\${ts}_nr_host_r2"
New-Item -ItemType Directory -Path $out | Out-Null
$exe = Join-Path $repo 'build\nr_host.exe'
$env:DLSS_DLL_PATH = Join-Path $repo 'third_party\nvidia-dlss\lib\Windows_x86_64\rel\nvngx_dlss.dll'
Push-Location $repo
& $exe --frames 2 --force-load --json (Join-Path $out 'nr_host_r2.json') *>&1 |
    Tee-Object -FilePath (Join-Path $out 'nr_host_r2_stdout.log')
Write-Host "exit=$LASTEXITCODE"
Pop-Location
$env:DLSS_DLL_PATH = $null
Write-Host "outdir=$out"
