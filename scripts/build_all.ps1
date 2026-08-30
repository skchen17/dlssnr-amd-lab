# Builds all native tools without CMake (CMake is not installed on this box).
# Compilers come from the ROCm/HIP SDK LLVM (clang-cl for host code, hipcc for device code).
param(
    [string[]]$Only = @()   # e.g. -Only env_probe,hip_probe ; empty = all
)
$ErrorActionPreference = 'Stop'
# Compiler warnings are written to stderr; do not let them abort the build.
# Real failures are caught by the explicit $LASTEXITCODE checks below.
$PSNativeCommandUseErrorActionPreference = $false
. (Join-Path $PSScriptRoot 'build_common.ps1')

$repo = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $repo 'build'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$envInfo = Resolve-BuildEnv
Write-Host "HIP SDK  : $($envInfo.hip_path)"
Write-Host "Win SDK  : $($envInfo.sdk_root) ($($envInfo.sdk_ver))"
Write-Host "MSVC     : $($envInfo.vs_root) ($($envInfo.msvc_ver))"

$sdkInc = Get-SdkIncludeArgs $envInfo
$libArgs = (Get-SdkLibArgs $envInfo) + (Get-MsvcLibArgs $envInfo)

function Invoke-ClangCL {
    param([string]$Name, [string[]]$Sources, [string[]]$LinkLibs, [string]$OutName,
          [string[]]$ExtraDefines = @(), [string]$OutputKind = 'exe')
    Write-Host "`n== Building $Name =="
    $outPath = Join-Path $out $OutName
    $cargs = @('--driver-mode=cl', '/nologo', '/EHsc', '/std:c++20', '/O2', '/Z7', '/MD', '-fuse-ld=lld-link', '-D_CRT_SECURE_NO_WARNINGS')
    $cargs += $ExtraDefines
    $cargs += $sdkInc
    $cargs += $Sources
    $cargs += @('/Fe:' + $outPath)
    $cargs += '/link'
    $cargs += $libArgs
    $cargs += $LinkLibs
    & $envInfo.clangxx @cargs
    if ($LASTEXITCODE -ne 0) { throw "build failed: $Name (exit $LASTEXITCODE)" }
    Write-Host "  -> $outPath"
}

function Invoke-HipCC {
    param([string]$Name, [string[]]$Sources, [string[]]$LinkLibs, [string]$OutName,
          [string]$Arch = 'gfx1201', [string[]]$ExtraFlags = @(), [bool]$NeedsWindowsHeaders = $false)
    Write-Host "`n== Building $Name (HIP, $Arch) =="
    $outPath = Join-Path $out $OutName
    $rocm = $envInfo.hip_path
    $llvmBin = "$rocm\lib\llvm\bin"
    $env:ROCM_PATH = $rocm
    $env:HIP_PATH = $rocm
    $env:PATH = "$llvmBin;$env:PATH"
    $hargs = @('-x', 'hip')
    $hargs += $Sources
    # NOTE: joined form '--rocm-path=<p>' is required; clang expects <p>/amdgcn/bitcode,
    # which in this tree lives at <rocm>\lib\llvm\amdgcn\bitcode
    $hargs += @("--offload-arch=$Arch", "--rocm-path=$rocm\lib\llvm", '--hip-link', '-O2', '-std=c++20')
    $hargs += $ExtraFlags
    if ($NeedsWindowsHeaders) {
        # Get-SdkIncludeArgs returns flat pairs ['-imsvc', path]; in g++ mode (clang++ -x hip)
        # the equivalent flag is -isystem
        $inc = @(Get-SdkIncludeArgs $envInfo)
        for ($j = 0; $j + 1 -lt $inc.Count; $j += 2) {
            $hargs += @('-isystem', $inc[$j + 1])
        }
    }
    $hargs += @('-o', $outPath)
    $hargs += @('-fuse-ld=lld')
    $hargs += @('-L' + "$rocm\lib")
    foreach ($l in (Get-SdkLibArgs $envInfo)) { $hargs += (($l -replace '/LIBPATH:', '-L') -replace '"', '') }
    foreach ($l in (Get-MsvcLibArgs $envInfo)) { $hargs += (($l -replace '/LIBPATH:', '-L') -replace '"', '') }
    foreach ($l in $LinkLibs) { $hargs += '-l' + $l }
    Write-Host ("  cmd: clang++ " + ($hargs -join ' '))
    & "$llvmBin\clang++.exe" @hargs
    if ($LASTEXITCODE -ne 0) { throw "build failed: $Name (exit $LASTEXITCODE)" }
    Write-Host "  -> $outPath"
}

$Only = @($Only | ForEach-Object { $_ -split ',' }) | Where-Object { $_ }
function Want([string]$n) { ($Only.Count -eq 0) -or ($Only -contains $n) }

if (Want 'env_probe') {
    Invoke-ClangCL -Name 'env_probe' `
        -Sources @("$repo\tools\env_probe\env_probe.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'env_probe.exe'
}

if (Want 'hip_probe') {
    Invoke-HipCC -Name 'hip_probe' `
        -Sources @("$repo\tools\hip_probe\hip_probe.cpp") `
        -OutName 'hip_probe.exe'
}

if (Want 'd3d12_hip_interop') {
    Invoke-HipCC -Name 'd3d12_hip_interop' `
        -Sources @("$repo\tools\d3d12_hip_interop\d3d12_hip_interop.cpp") `
        -LinkLibs @('dxgi','d3d12','ole32','user32') `
        -OutName 'd3d12_hip_interop.exe' -NeedsWindowsHeaders $true
}

if (Want 'binary_probe') {
    Invoke-ClangCL -Name 'binary_probe' `
        -Sources @("$repo\tools\binary_probe\binary_probe.cpp") `
        -LinkLibs @('advapi32.lib','user32.lib') `
        -OutName 'binary_probe.exe'
}

if (Want 'nvapi_trace') {
    Invoke-ClangCL -Name 'nvapi_trace' `
        -Sources @("$repo\tools\nvapi_trace\nvapi_trace.cpp") `
        -LinkLibs @('user32.lib','advapi32.lib','psapi.lib','/DLL','/EXPORT:nvapi_QueryInterface,@1') `
        -OutName 'nvapi64.dll'
}

if (Want 'module_trace') {
    Invoke-ClangCL -Name 'module_trace' `
        -Sources @("$repo\tools\module_trace\module_trace.cpp") `
        -LinkLibs @('user32.lib','advapi32.lib','psapi.lib','/DLL') `
        -OutName 'module_trace.dll'
}

if (Want 'nr_host') {
    Invoke-ClangCL -Name 'nr_host' `
        -Sources @("$repo\tools\nr_host\nr_host.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'nr_host.exe'
}

Write-Host "`nAll requested builds finished."
