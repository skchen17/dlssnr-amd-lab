<#
.SYNOPSIS
  Phase 0: record the experiment machine.
.DESCRIPTION
  Collects OS / CPU / GPU (PCI ids, driver), HIP SDK / ROCm, ZLUDA, toolchain
  (MSVC / CMake / Rust / LLVM / CUDA) and proprietary-file prerequisites.
  Writes results/<timestamp>/environment.json. Never uploads anything.
.NOTES
  The DXGI LUID / D3D12 feature level / shader model fields are filled by the
  compiled env_probe.exe when present; WMI provides PCI ids + driver version.
#>

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$outDir = Join-Path $repo "results\$ts"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$env0 = [ordered]@{
    collected_utc = (Get-Date).ToUniversalTime().ToString('o')
    collector     = 'collect_environment.ps1 v1'
}

# ---------- Windows ----------
try {
    $os = Get-CimInstance Win32_OperatingSystem
    $ubr = (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction Stop).UBR
    $env0.windows = [ordered]@{
        caption = $os.Caption
        version = $os.Version
        build   = "$($os.BuildNumber).$ubr"
    }
} catch { $env0.windows = @{ error = $_.Exception.Message } }

# ---------- CPU ----------
try {
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $env0.cpu = [ordered]@{
        name        = $cpu.Name.Trim()
        cores       = $cpu.NumberOfCores
        logical     = $cpu.NumberOfLogicalProcessors
        max_clock_mhz = $cpu.MaxClockSpeed
    }
} catch { $env0.cpu = @{ error = $_.Exception.Message } }

# ---------- GPUs via WMI (PCI ids + driver) ----------
$gpus = @()
foreach ($vc in (Get-CimInstance Win32_VideoController)) {
    $ven = $null; $dev = $null
    if ($vc.PNPDeviceID -match 'VEN_([0-9A-F]{4})') { $ven = "0x$($Matches[1].ToLower())" }
    if ($vc.PNPDeviceID -match 'DEV_([0-9A-F]{4})') { $dev = "0x$($Matches[1].ToLower())" }
    $gpus += [ordered]@{
        name          = $vc.Name
        vendor_id     = $ven
        device_id     = $dev
        pnp_device_id = $vc.PNPDeviceID
        driver_version = $vc.DriverVersion
        driver_date   = $vc.DriverDate
        adapter_ram   = $vc.AdapterRAM
        video_processor = $vc.VideoProcessor
        is_amd        = ($ven -eq '0x1002')
        is_nvidia     = ($ven -eq '0x10de')
    }
}
$env0.gpus = $gpus
$env0.amd_adapter_present = [bool]($gpus | Where-Object { $_.is_amd })
$env0.nvidia_adapter_present = [bool]($gpus | Where-Object { $_.is_nvidia })

# ---------- env_probe.exe (DXGI LUID / D3D12 feature level / shader model) ----------
$envProbeCandidates = @(
    (Join-Path $repo 'build\tools\env_probe\Release\env_probe.exe'),
    (Join-Path $repo 'build\tools\env_probe\env_probe.exe')
)
$envProbe = $envProbeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($envProbe) {
    try {
        $raw = & $envProbe --json 2>&1 | Out-String
        $env0.dxgi_d3d12 = ($raw | ConvertFrom-Json)
    } catch { $env0.dxgi_d3d12 = @{ error = $_.Exception.Message } }
} else {
    $env0.dxgi_d3d12 = @{ status = 'env_probe.exe not built yet (see README build steps)' }
}

# ---------- HIP SDK / ROCm ----------
$hip = [ordered]@{ installed = $false }
$hipPath = $env:HIP_PATH
if (-not $hipPath) {
    foreach ($cand in @('C:\Program Files\AMD\ROCm', 'C:\Program Files\AMD\HIP SDK')) {
        if (Test-Path $cand) {
            $latest = Get-ChildItem $cand -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending | Select-Object -First 1
            if ($latest) { $hipPath = $latest.FullName; break }
        }
    }
}
if ($hipPath -and (Test-Path $hipPath)) {
    $hip.installed = $true
    $hip.path = $hipPath
    $verFile = Join-Path $hipPath '.info\version'
    if (Test-Path $verFile) { $hip.rocm_version = (Get-Content $verFile -Raw).Trim() }
    $hipDll = Join-Path $hipPath 'bin\amdhip64.dll'
    if (Test-Path $hipDll) {
        $hip.runtime_dll = 'amdhip64.dll'
        $hip.runtime_file_version = (Get-Item $hipDll).VersionInfo.FileVersion
        $hip.runtime_product_version = (Get-Item $hipDll).VersionInfo.ProductVersion
    }
    $hipinfo = Join-Path $hipPath 'bin\hipinfo.exe'
    if (Test-Path $hipinfo) {
        try { $hip.hipinfo_excerpt = (& $hipinfo 2>&1 | Select-Object -First 40 | Out-String).Trim() }
        catch { $hip.hipinfo_error = $_.Exception.Message }
    }
}
$env0.hip = $hip

# ---------- ZLUDA ----------
$zluda = [ordered]@{ detected = $false }
$zludaPath = $env:ZLUDA_PATH
if ($zludaPath -and (Test-Path $zludaPath)) {
    $zluda.detected = $true
    $zluda.path = $zludaPath
    $z = Join-Path $zludaPath 'zluda.dll'
    if (Test-Path $z) { $zluda.dll_file_version = (Get-Item $z).VersionInfo.FileVersion }
} else {
    $zluda.note = 'ZLUDA_PATH not set; get release from github.com/vosen/ZLUDA (user installs)'
}
$env0.zluda = $zluda

# ---------- Toolchain ----------
function Get-ToolVersion($cmd, $arg) {
    try {
        $p = Get-Command $cmd -ErrorAction Stop
        $out = (& $cmd @arg 2>&1 | Out-String).Trim()
        return [ordered]@{ found = $true; path = $p.Source; version = ($out -split "`n")[0] }
    } catch { return [ordered]@{ found = $false } }
}
$env0.toolchain = [ordered]@{
    cmake = (Get-ToolVersion 'cmake' @('--version'))
    rustc = (Get-ToolVersion 'rustc' @('--version'))
    clang = (Get-ToolVersion 'clang' @('--version'))
    nvcc  = (Get-ToolVersion 'nvcc' @('--version'))
    git   = (Get-ToolVersion 'git' @('--version'))
}

# Visual Studio / MSVC via vswhere
try {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -property installationPath 2>$null
        $vsVer = & $vswhere -latest -products * -property installationVersion 2>$null
        $env0.toolchain.visual_studio = [ordered]@{ found = [bool]$vs; path = $vs; version = $vsVer }
        if ($vs) {
            $cl = Get-ChildItem "$vs\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending | Select-Object -First 1
            if ($cl) { $env0.toolchain.msvc_tools_version = $cl.Name }
        }
    } else { $env0.toolchain.visual_studio = [ordered]@{ found = $false } }
} catch { $env0.toolchain.visual_studio = [ordered]@{ error = $_.Exception.Message } }

# Windows SDK
try {
    $sdkKey = 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots'
    if (Test-Path $sdkKey) {
        $env0.toolchain.windows_sdk_root = (Get-ItemProperty $sdkKey).KitsRoot10
        $env0.toolchain.windows_sdk_versions = (Get-ItemProperty $sdkKey).ProductVersion
    }
} catch {}

# ---------- Proprietary prerequisites (presence only; record hash+version, never content) ----------
function Get-PropFileRecord($envName) {
    $p = [Environment]::GetEnvironmentVariable($envName)
    if (-not $p) { return [ordered]@{ present = $false; env_var = $envName } }
    if (-not (Test-Path $p)) { return [ordered]@{ present = $false; env_var = $envName; configured_path_invalid = $p } }
    $fi = Get-Item $p
    $rec = [ordered]@{
        present   = $true
        env_var   = $envName
        filename  = $fi.Name
        path      = $fi.FullName
        size      = $fi.Length
        sha256    = (Get-FileHash $p -Algorithm SHA256).Hash
        file_version = $fi.VersionInfo.FileVersion
        product_version = $fi.VersionInfo.ProductVersion
    }
    try {
        $sig = Get-AuthenticodeSignature $p
        $rec.signature_status = $sig.Status.ToString()
        $rec.signature_signer = $sig.SignerCertificate.Subject
    } catch { $rec.signature_status = 'check_failed' }
    return $rec
}
$env0.proprietary_prerequisites = [ordered]@{
    DLSSNR_DLL_PATH         = (Get-PropFileRecord 'DLSSNR_DLL_PATH')
    DLSS_DLL_PATH           = (Get-PropFileRecord 'DLSS_DLL_PATH')
    RENODX_DLSS5_ADDON_PATH = (Get-PropFileRecord 'RENODX_DLSS5_ADDON_PATH')
}

# ---------- Write JSON ----------
$outFile = Join-Path $outDir 'environment.json'
$env0 | ConvertTo-Json -Depth 10 | Set-Content -Path $outFile -Encoding utf8
Write-Host "environment.json written: $outFile"
Write-Host ("AMD adapter present : {0}" -f $env0.amd_adapter_present)
Write-Host ("NVIDIA adapter present: {0}" -f $env0.nvidia_adapter_present)
Write-Host ("HIP SDK installed   : {0}" -f $env0.hip.installed)
Write-Host ("DLSSNR dll present  : {0}" -f $env0.proprietary_prerequisites.DLSSNR_DLL_PATH.present)
Write-Host ("DLSS dll present    : {0}" -f $env0.proprietary_prerequisites.DLSS_DLL_PATH.present)
