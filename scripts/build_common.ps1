# Shared helper: resolve compiler/SDK paths used by all build_*.ps1 scripts.
$ErrorActionPreference = 'Stop'

function Resolve-BuildEnv {
    $envInfo = [ordered]@{}

    # ROCm / HIP SDK
    $hipPath = $env:HIP_PATH
    if (-not $hipPath) {
        $cand = @(
            "$env:USERPROFILE\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core",
            'C:\Program Files\AMD\ROCm'
        )
        foreach ($c in $cand) {
            if (Test-Path "$c\bin\hipcc.exe") { $hipPath = $c; break }
            if (Test-Path $c) {
                $sub = Get-ChildItem $c -Directory -ErrorAction SilentlyContinue |
                    Where-Object { Test-Path "$($_.FullName)\bin\hipcc.exe" } |
                    Sort-Object Name -Descending | Select-Object -First 1
                if ($sub) { $hipPath = $sub.FullName; break }
            }
        }
    }
    if (-not $hipPath -or -not (Test-Path "$hipPath\bin\hipcc.exe")) {
        throw 'HIP SDK not found (set HIP_PATH to the ROCm core directory containing bin\hipcc.exe)'
    }
    $envInfo.hip_path = $hipPath

    # Windows SDK
    $sdkRoot = $null
    try { $sdkRoot = (Get-ItemProperty 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots').KitsRoot10 } catch {}
    if (-not $sdkRoot) { $sdkRoot = 'C:\Program Files (x86)\Windows Kits\10\' }
    $sdkVer = (Get-ChildItem "$sdkRoot\Include" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1).Name
    if (-not $sdkVer) { throw "Windows SDK include not found under $sdkRoot\Include" }
    $envInfo.sdk_root = $sdkRoot
    $envInfo.sdk_ver  = $sdkVer

    # MSVC toolset (needed for ucrt/msvcrt headers via clang-cl? Actually clang-cl uses its own + SDK;
    # but linking needs msvc libs from VS Build Tools)
    $vsRoot = $null
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsRoot = & $vswhere -latest -products * -property installationPath 2>$null
    }
    if (-not $vsRoot) {
        foreach ($c in @('C:\Program Files\Microsoft Visual Studio\2022', 'C:\Program Files (x86)\Microsoft Visual Studio')) {
            if (Test-Path $c) { $vsRoot = (Get-ChildItem $c -Directory -ErrorAction SilentlyContinue | Select-Object -First 1).FullName }
        }
    }
    $msvcVer = $null
    if ($vsRoot) {
        $msvcVer = (Get-ChildItem "$vsRoot\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1).Name
    }
    $envInfo.vs_root  = $vsRoot
    $envInfo.msvc_ver = $msvcVer

    # Compilers
    $clang = "$($envInfo.hip_path)\lib\llvm\bin\clang++.exe"
    if (-not (Test-Path $clang)) { throw "clang++ not found at $clang" }
    $envInfo.clangxx = $clang
    $envInfo.hipcc   = "$($envInfo.hip_path)\bin\hipcc.exe"

    return $envInfo
}

function Get-SdkIncludeArgs($envInfo) {
    # Returns flat token pairs: -imsvc <path> (works for both clang-cl and clang g++ mode)
    $r = $envInfo.sdk_root; $v = $envInfo.sdk_ver
    return @(
        '-imsvc', "$r\Include\$v\ucrt",
        '-imsvc', "$r\Include\$v\shared",
        '-imsvc', "$r\Include\$v\um",
        '-imsvc', "$r\Include\$v\winrt"
    )
}

function Get-SdkLibArgs($envInfo) {
    $r = $envInfo.sdk_root; $v = $envInfo.sdk_ver
    return @("/LIBPATH:`"$r\Lib\$v\ucrt\x64`"", "/LIBPATH:`"$r\Lib\$v\um\x64`"")
}

function Get-MsvcLibArgs($envInfo) {
    if (-not $envInfo.vs_root -or -not $envInfo.msvc_ver) { return @() }
    return @("/LIBPATH:`"$($envInfo.vs_root)\VC\Tools\MSVC\$($envInfo.msvc_ver)\lib\x64`"")
}
