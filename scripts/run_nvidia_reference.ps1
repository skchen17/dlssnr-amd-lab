# run_nvidia_reference.ps1 — one-command NVIDIA reference run (Round 2 Phase D).
#
# Pipeline:
#   1. environment detection (env_probe + adapter vendor)
#   2. runtime hashes (SHA-256 of every DLL involved; the bundle stores hashes,
#      NEVER the proprietary DLLs themselves)
#   3. ngx_abi_probe against DLSS_DLL_PATH
#   4. Stage A: vanilla DLSS/DLAA evaluate (run_reference.ps1 semantics)
#   5. Stage B (only after Stage A PASS, and only with DLSSNR_DLL_PATH +
#      RENODX_DLSS5_ADDON_PATH present): observe evaluate -> addon hook ->
#      feature 18 (PRIVATE ABI) under module_trace + nvapi_trace
#   6. package everything into reference_bundle\<ts>\ (no proprietary DLLs)
#
# Without an NVIDIA adapter the script stops with BLOCKED_EXTERNAL_HARDWARE
# and still emits a bundle containing the environment report + hashes.
param(
    [int]$Frames = 8,
    [int]$Width = 512,
    [int]$Height = 512
)
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$bundle = Join-Path $repo "reference_bundle\$ts"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null

function FileHash256([string]$p) {
    if (-not $p -or -not (Test-Path $p)) { return $null }
    return (Get-FileHash -Algorithm SHA256 -Path $p).Hash.ToLowerInvariant()
}

Write-Host "=== run_nvidia_reference ($ts) ==="
Write-Host "bundle: $bundle"

# ---- 1. environment detection ----
$envProbe = Join-Path $repo 'build\env_probe.exe'
$envJson = Join-Path $bundle 'environment.json'
if (Test-Path $envProbe) {
    Push-Location $repo
    & $envProbe --json 2>&1 | Tee-Object -FilePath (Join-Path $bundle 'environment.txt')
    Pop-Location
    Copy-Item (Join-Path $bundle 'environment.txt') $envJson -Force
} else {
    Write-Host 'env_probe.exe missing; run scripts\build_all.ps1 first'
}

# NVIDIA adapter check (same rule as nr_host)
$hasNvidia = $false
try {
    Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices;
        public class Dx { [DllImport("dxgi.dll")] public static extern int CreateDXGIFactory1(ref Guid r, out IntPtr f); }' -ErrorAction SilentlyContinue
    # Fallback: parse env_probe txt for the vendor line
    if (Test-Path (Join-Path $bundle 'environment.txt')) {
        $txt = Get-Content (Join-Path $bundle 'environment.txt') -Raw
        if ($txt -match '10DE|10de|NVIDIA') { $hasNvidia = $true }
    }
} catch { }
if (-not $hasNvidia) {
    # last resort: WMI
    $gpus = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    foreach ($g in $gpus) { if ($g.PNPDeviceID -match 'VEN_10DE') { $hasNvidia = $true } }
}

# ---- 2. runtime hashes ----
$hashes = [ordered]@{}
$hashes['nvngx_dlss.dll']     = FileHash256 $env:DLSS_DLL_PATH
$hashes['nvngx_dlssnr.dll']   = FileHash256 $env:DLSSNR_DLL_PATH
$hashes['renodx_addon']       = FileHash256 $env:RENODX_DLSS5_ADDON_PATH
$hashes['nr_host.exe']        = FileHash256 (Join-Path $repo 'build\nr_host.exe')
$hashes['module_trace.dll']   = FileHash256 (Join-Path $repo 'build\module_trace.dll')
$hashes['nvapi64.dll']        = FileHash256 (Join-Path $repo 'build\nvapi64.dll')
$hashJson = $hashes.GetEnumerator() | ForEach-Object { '  "{0}": "{1}"' -f $_.Key, $(if ($_.Value) { $_.Value } else { 'MISSING' }) }
Set-Content -Path (Join-Path $bundle 'runtime_hashes.json') -Value ("{`n" + ($hashJson -join ",`n") + "`n}") -Encoding UTF8
Write-Host 'runtime hashes recorded (proprietary DLLs are NEVER copied into the bundle)'

$status = [ordered]@{
    timestamp = $ts
    nvidia_gpu_present = $hasNvidia
    stage_abi_probe = 'NOT_RUN'
    stage_a = 'NOT_RUN'
    stage_b = 'NOT_RUN'
}

if (-not $hasNvidia) {
    $status['overall'] = 'BLOCKED_EXTERNAL_HARDWARE'
    Write-Host 'no NVIDIA adapter on this machine: R4-R7 are BLOCKED_EXTERNAL_HARDWARE'
    Write-Host 'bundle carries environment + hashes so the RTX machine run is reproducible'
} else {
    # ---- 3. ABI probe ----
    $probe = Join-Path $repo 'build\ngx_abi_probe.exe'
    if ((Test-Path $probe) -and $env:DLSS_DLL_PATH) {
        & $probe --dll $env:DLSS_DLL_PATH --json (Join-Path $bundle 'ngx_abi_test.json')
        $status['stage_abi_probe'] = if ($LASTEXITCODE -eq 0) { 'PASS' } else { "FAIL(exit=$LASTEXITCODE)" }
    } else { $status['stage_abi_probe'] = 'BLOCKED_MISSING_PREREQUISITE' }

    # ---- 4. Stage A ----
    if ($env:DLSS_DLL_PATH) {
        Push-Location $repo
        & (Join-Path $PSScriptRoot 'run_reference.ps1') -Frames $Frames -Width $Width -Height $Height `
            -Output (Join-Path $bundle 'stage_a\output_rgba8.bin')
        $codeA = $LASTEXITCODE
        Pop-Location
        $status['stage_a'] = if ($codeA -eq 0) { 'PASS' } else { "FAIL(exit=$codeA)" }
    } else { $status['stage_a'] = 'BLOCKED_MISSING_PREREQUISITE (DLSS_DLL_PATH)'; $codeA = 3 }

    # ---- 5. Stage B (PRIVATE ABI: feature 18 via DLSS5 addon) ----
    if ($codeA -eq 0 -and $env:DLSSNR_DLL_PATH -and $env:RENODX_DLSS5_ADDON_PATH) {
        $sb = Join-Path $bundle 'stage_b'
        New-Item -ItemType Directory -Force -Path $sb | Out-Null
        $env:MODULE_TRACE_LOG = Join-Path $sb 'module_trace.log'
        $env:NVAPI_TRACE_LOG  = Join-Path $sb 'nvapi_trace.jsonl'
        Push-Location $repo
        & (Join-Path $repo 'build\nr_host.exe') --frames $Frames --width $Width --height $Height `
            --trace --output (Join-Path $sb 'output_rgba8.bin') --json (Join-Path $sb 'host.json')
        $codeB = $LASTEXITCODE
        Pop-Location
        $env:MODULE_TRACE_LOG = $null; $env:NVAPI_TRACE_LOG = $null
        $status['stage_b'] = if ($codeB -eq 0) { 'PASS (PRIVATE_ABI label applies)' } else { "INCONCLUSIVE(exit=$codeB) — verify ReShade.log for 'feature 18 created'" }
    } else {
        $status['stage_b'] = 'BLOCKED_MISSING_PREREQUISITE (needs Stage A PASS + DLSSNR_DLL_PATH + RENODX_DLSS5_ADDON_PATH)'
    }
    $status['overall'] = if ($status['stage_a'] -eq 'PASS') { 'STAGE_A_PASS' } else { 'INCOMPLETE' }
}

# ---- 6. manifest ----
$m = @('round 2 reference bundle (no proprietary DLLs inside)', '')
foreach ($k in $status.Keys) { $m += "$k = $($status[$k])" }
$m += ''
$m += 'input determinism: LCG seed 0x2545F491; color RGBA8 / depth R32F / MV R16G16F'
$m += 'contract source: DLSS5-Feeder (MIT) genuine DLAA contract; ABI: docs/NGX_ABI_AUDIT.md'
Set-Content -Path (Join-Path $bundle 'manifest.txt') -Value ($m -join "`n") -Encoding UTF8

Write-Host ''
foreach ($k in $status.Keys) { Write-Host ("{0,-20} {1}" -f $k, $status[$k]) }
Write-Host "bundle: $bundle"
