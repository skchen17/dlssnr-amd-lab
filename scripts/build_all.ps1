# Builds all native tools without CMake (CMake is not installed on this box).
# Compilers come from the ROCm/HIP SDK LLVM (clang-cl for host code, hipcc for device code).
param(
    [string[]]$Only = @(),   # e.g. -Only env_probe,hip_probe ; empty = all
    [string]$FreshOutputDirectory = '' # Isolated staging: must not already exist.
)
$ErrorActionPreference = 'Stop'
# Compiler warnings are written to stderr; do not let them abort the build.
# Real failures are caught by the explicit $LASTEXITCODE checks below.
$PSNativeCommandUseErrorActionPreference = $false
. (Join-Path $PSScriptRoot 'build_common.ps1')

$repo = Split-Path -Parent $PSScriptRoot
$out  = Join-Path $repo 'build'
if ($FreshOutputDirectory) {
    if (-not [IO.Path]::IsPathFullyQualified($FreshOutputDirectory)) { throw 'Fresh output path must be absolute' }
    $out=[IO.Path]::GetFullPath($FreshOutputDirectory)
    if (Test-Path -LiteralPath $out) { throw 'Fresh output directory already exists; refusing overwrite' }
}
New-Item -ItemType Directory -Force -Path $out | Out-Null
$envInfo = Resolve-BuildEnv
Write-Host "HIP SDK  : $($envInfo.hip_path)"
Write-Host "Win SDK  : $($envInfo.sdk_root) ($($envInfo.sdk_ver))"
Write-Host "MSVC     : $($envInfo.vs_root) ($($envInfo.msvc_ver))"

$sdkInc = Get-SdkIncludeArgs $envInfo
$libArgs = (Get-SdkLibArgs $envInfo) + (Get-MsvcLibArgs $envInfo)
# Official NGX public headers (NVIDIA/DLSS, see third_party/PROVENANCE.md)
$ngxIncArgs = @("/I$repo\third_party\nvidia-dlss\include")

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
    # PowerShell already passes each array element as one argument. Embedded
    # quotes become literal characters for lld-link and make this path fail.
    $ngxLibPath = '/LIBPATH:' + (Join-Path $repo 'third_party\nvidia-dlss\lib\Windows_x86_64\x64')
    Invoke-ClangCL -Name 'nr_host' `
        -Sources @("$repo\tools\nr_host\nr_host.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib','version.lib','advapi32.lib',$ngxLibPath,'nvsdk_ngx_d.lib') `
        -ExtraDefines (@("/I$repo\tools\nr_host") + $ngxIncArgs) `
        -OutName 'nr_host.exe'
}

if ((Want 'ffx_observer') -or (Want 'ffx_observer_selftest') -or (Want 'ffx_bootstrap')) {
    Invoke-ClangCL -Name 'ffx_observer' -Sources @("$repo\tools\ffx_observer\ffx_observer.cpp") `
        -LinkLibs @('user32.lib','/DLL') -OutName 'ffx_observer.dll'
}
if (Want 'ffx_live_inspector') {
    Invoke-ClangCL -Name 'ffx_live_inspector' -Sources @("$repo\tools\ffx_observer\live_inspector.cpp") `
        -LinkLibs @('d3d12.lib','dxgi.lib','/DLL') -OutName 'ffx_live_inspector.dll'
}
if (Want 'ffx_contract_inspector') {
    Invoke-ClangCL -Name 'ffx_contract_inspector' -Sources @("$repo\tools\ffx_observer\contract_inspector.cpp") `
        -LinkLibs @('d3d12.lib','dxgi.lib','/DLL') -OutName 'ffx_contract_inspector.dll'
}
if (Want 'ffx_capture_session') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out 'output_residual_filter.dxil') `
        (Join-Path $repo 'tools\ffx_observer\output_residual_filter.hlsl')
    if ($LASTEXITCODE -ne 0) { throw 'DXC build failed: output_residual_filter' }
    & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out 'output_residual_network.dxil') `
        (Join-Path $repo 'tools\ffx_observer\output_residual_network.hlsl')
    if ($LASTEXITCODE -ne 0) { throw 'DXC build failed: output_residual_network' }
    Copy-Item -LiteralPath (Join-Path $repo 'models\ffx_dynamic_residual_v1.weights') `
        -Destination (Join-Path $out 'ffx_dynamic_residual_v1.weights')
    Invoke-ClangCL -Name 'ffx_capture_session' -Sources @("$repo\tools\ffx_observer\capture_session.cpp") `
        -LinkLibs @('d3d12.lib','dxgi.lib','/DLL') -OutName 'ffx_capture_session.dll'
}
if (Want 'ffx_session_control') {
    Invoke-ClangCL -Name 'ffx_session_control' -Sources @("$repo\tools\ffx_observer\session_control.cpp") `
        -LinkLibs @('user32.lib') -OutName 'ffx_session_control.exe'
}
if (Want 'ffx_live_attach') {
    Invoke-ClangCL -Name 'ffx_live_attach' -Sources @("$repo\tools\ffx_observer\live_attach.cpp") `
        -LinkLibs @('user32.lib') -OutName 'ffx_live_attach.exe'
}
if (Want 'ffx_depth_plane_probe') {
    Invoke-ClangCL -Name 'ffx_depth_plane_probe' -Sources @("$repo\tools\ffx_observer\depth_plane_probe.cpp") `
        -LinkLibs @('d3d12.lib','dxgi.lib','ole32.lib','user32.lib') -OutName 'ffx_depth_plane_probe.exe'
}
if (Want 'ffx_capture_plane_selftest') {
    Invoke-ClangCL -Name 'ffx_capture_plane_selftest' -Sources @("$repo\tools\ffx_observer\capture_plane_selftest.cpp") `
        -LinkLibs @('d3d12.lib','dxgi.lib','ole32.lib','user32.lib') -OutName 'ffx_capture_plane_selftest.exe'
}
if (Want 'ffx_bootstrap') {
    Invoke-ClangCL -Name 'ffx_observer_fixture' -Sources @("$repo\tests\ffx_observer_selftest\fixture.cpp") `
        -LinkLibs @('/DLL') -OutName 'ffx_observer_fixture.dll'
    Invoke-ClangCL -Name 'ffx_bootstrap_host' -Sources @("$repo\tests\ffx_observer_selftest\bootstrap_host.cpp") `
        -LinkLibs @("$out\ffx_observer_fixture.lib") -OutName 'ffx_bootstrap_host.exe'
    Invoke-ClangCL -Name 'ffx_bootstrap_launcher' -Sources @("$repo\tools\ffx_observer\bootstrap_launcher.cpp") `
        -LinkLibs @() -OutName 'ffx_bootstrap_launcher.exe'
}
if (Want 'ffx_output_boundary_selftest') {
    Invoke-ClangCL -Name 'ffx_output_boundary_selftest' -Sources @("$repo\tools\ffx_observer\output_boundary_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_output_boundary_selftest.exe'
}
if (Want 'ffx_resident_surface_selftest') {
    Invoke-ClangCL -Name 'ffx_resident_surface_selftest' -Sources @("$repo\tools\ffx_observer\resident_surface_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_resident_surface_selftest.exe'
}
if (Want 'resident_present') {
    Invoke-ClangCL -Name 'resident_present' -Sources @("$repo\tools\ffx_observer\resident_present.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','user32.lib','/DLL') -OutName 'resident_present.dll'
}
if (Want 'resident_present_selftest') {
    Invoke-ClangCL -Name 'resident_present_selftest' -Sources @("$repo\tools\ffx_observer\resident_present_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','user32.lib') -OutName 'resident_present_selftest.exe'
}
if (Want 'ffx_resident_session_selftest') {
    Invoke-ClangCL -Name 'ffx_resident_session_selftest' -Sources @("$repo\tools\ffx_observer\resident_session_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_resident_session_selftest.exe'
}
if (Want 'ffx_static_preview_selftest') {
    Invoke-ClangCL -Name 'ffx_static_preview_selftest' -Sources @("$repo\tools\ffx_observer\static_preview_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_static_preview_selftest.exe'
}
if (Want 'ffx_output_filter_selftest') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out 'output_residual_filter.dxil') `
        (Join-Path $repo 'tools\ffx_observer\output_residual_filter.hlsl')
    if ($LASTEXITCODE -ne 0) { throw 'DXC build failed: output_residual_filter' }
    Invoke-ClangCL -Name 'ffx_output_filter_selftest' -Sources @("$repo\tools\ffx_observer\output_filter_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_output_filter_selftest.exe'
}
if (Want 'ffx_output_network_selftest') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out 'output_residual_network.dxil') `
        (Join-Path $repo 'tools\ffx_observer\output_residual_network.hlsl')
    if ($LASTEXITCODE -ne 0) { throw 'DXC build failed: output_residual_network' }
    Invoke-ClangCL -Name 'ffx_output_network_selftest' -Sources @("$repo\tools\ffx_observer\output_filter_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -ExtraDefines @('/DFFX_OUTPUT_NETWORK_SELFTEST') `
        -OutName 'ffx_output_network_selftest.exe'
}
if (Want 'full_graph_slot0_d3d12_selftest') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out 'full_graph_slot0_clear.dxil') `
        (Join-Path $repo 'tools\full_graph_d3d12\slot0_clear.hlsl')
    if ($LASTEXITCODE -ne 0) { throw 'DXC build failed: full_graph_slot0_clear' }
    Invoke-ClangCL -Name 'full_graph_slot0_d3d12_selftest' `
        -Sources @("$repo\tools\full_graph_d3d12\slot0_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'full_graph_slot0_d3d12_selftest.exe'
}
if (Want 'full_graph_resolution_plan') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    foreach ($shader in @('resolution_pack','resolution_identity','resolution_unpack')) {
        & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out "full_graph_$shader.dxil") `
            (Join-Path $repo "tools\full_graph_d3d12\$shader.hlsl")
        if ($LASTEXITCODE -ne 0) { throw "DXC build failed: $shader" }
    }
    Invoke-ClangCL -Name 'full_graph_resolution_plan_selftest' `
        -Sources @("$repo\tools\full_graph_d3d12\resolution_plan_selftest.cpp") `
        -LinkLibs @() -OutName 'full_graph_resolution_plan_selftest.exe'
    Invoke-ClangCL -Name 'full_graph_resolution_gpu_selftest' `
        -Sources @("$repo\tools\full_graph_d3d12\resolution_gpu_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'full_graph_resolution_gpu_selftest.exe'
}
if (Want 'ffx_output_filter_split_warp_selftest') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out 'output_residual_filter.dxil') `
        (Join-Path $repo 'tools\ffx_observer\output_residual_filter.hlsl')
    if ($LASTEXITCODE -ne 0) { throw 'DXC build failed: output_residual_filter' }
    Invoke-ClangCL -Name 'ffx_output_filter_split_warp_selftest' -Sources @("$repo\tools\ffx_observer\output_filter_split_warp_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_output_filter_split_warp_selftest.exe'
}
if (Want 'ffx_filter_session_warp_selftest') {
    Invoke-ClangCL -Name 'ffx_filter_session_fixture' -Sources @("$repo\tests\ffx_filter_session_warp\fixture.cpp") `
        -LinkLibs @('/DLL') -OutName 'ffx_filter_session_fixture.dll'
    Invoke-ClangCL -Name 'ffx_filter_session_warp_selftest' -Sources @("$repo\tests\ffx_filter_session_warp\selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_filter_session_warp_selftest.exe'
}

if (Want 'ffx_dispatch_probe') {
    Invoke-ClangCL -Name 'ffx_dispatch_probe' -Sources @("$repo\tools\ffx_observer\dispatch_probe.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_dispatch_probe.exe'
}
if (Want 'ffx_provider_probe') {
    Invoke-ClangCL -Name 'ffx_provider_probe' -Sources @("$repo\tools\ffx_observer\provider_probe.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib') -OutName 'ffx_provider_probe.exe'
}
if (Want 'ffx_observer_selftest') {
    Invoke-ClangCL -Name 'ffx_observer_fixture' -Sources @("$repo\tests\ffx_observer_selftest\fixture.cpp") `
        -LinkLibs @('/DLL') -OutName 'ffx_observer_fixture.dll'
    Invoke-ClangCL -Name 'ffx_observer_selftest' -Sources @("$repo\tests\ffx_observer_selftest\selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib',"$out\ffx_observer_fixture.lib") `
        -OutName 'ffx_observer_selftest.exe'
}

if (Want 'fp8_e4m3_probe') {
    Invoke-HipCC -Name 'fp8_e4m3_probe' `
        -Sources @("$repo\tools\fp8_e4m3_probe\fp8_e4m3_probe.cpp") `
        -OutName 'fp8_e4m3_probe.exe'
}

if (Want 'fp8_mma_probe') {
    Invoke-HipCC -Name 'fp8_mma_probe' `
        -Sources @("$repo\tools\fp8_mma_probe\fp8_mma_probe.cpp") `
        -OutName 'fp8_mma_probe.exe'
}

if (Want 'remaining_primitives_probe') {
    Invoke-HipCC -Name 'remaining_primitives_probe' `
        -Sources @("$repo\tools\remaining_primitives_probe\remaining_primitives_probe.cpp") `
        -OutName 'remaining_primitives_probe.exe'
}

if (Want 'n0_epilogue_probe') {
    Invoke-HipCC -Name 'n0_epilogue_probe' `
        -Sources @("$repo\tools\n0_epilogue_probe\n0_epilogue_probe.cpp") `
        -OutName 'n0_epilogue_probe.exe'
}

if (Want 'nvidia_fp8_reference') {
    Invoke-ClangCL -Name 'nvidia_fp8_reference' `
        -Sources @("$repo\tools\nvidia_fp8_reference\nvidia_fp8_reference.cpp") `
        -LinkLibs @('kernel32.lib') `
        -OutName 'nvidia_fp8_reference.exe'
}

if (Want 'nvidia_mma_reference') {
    Invoke-ClangCL -Name 'nvidia_mma_reference' `
        -Sources @("$repo\tools\nvidia_mma_reference\nvidia_mma_reference.cpp") `
        -LinkLibs @('kernel32.lib') `
        -OutName 'nvidia_mma_reference.exe'
}

if (Want 'nvidia_remaining_reference') {
    Invoke-ClangCL -Name 'nvidia_remaining_reference' `
        -Sources @("$repo\tools\nvidia_remaining_reference\nvidia_remaining_reference.cpp") `
        -LinkLibs @('kernel32.lib') `
        -OutName 'nvidia_remaining_reference.exe'
}

if (Want 'dlss5_feed_host64') {
    $ngxLibPath = '/LIBPATH:' + (Join-Path $repo 'third_party\nvidia-dlss\lib\Windows_x86_64\x64')
    Invoke-ClangCL -Name 'dlss5_feed_host64' `
        -Sources @("$repo\third_party\DLSS5-Feeder\host\dlss5-feed-host64.cpp") `
        -LinkLibs @('version.lib','kernel32.lib','user32.lib','gdi32.lib','advapi32.lib','ole32.lib',$ngxLibPath,'nvsdk_ngx_d.lib') `
        -ExtraDefines $ngxIncArgs `
        -OutName 'dlss5-feed-host64.exe'
}

if (Want 'ngx_abi_probe') {
    Invoke-ClangCL -Name 'ngx_abi_probe' `
        -Sources @("$repo\tools\ngx_abi_probe\ngx_abi_probe.cpp") `
        -LinkLibs @('user32.lib') `
        -ExtraDefines $ngxIncArgs `
        -OutName 'ngx_abi_probe.exe'
}

if (Want 'module_trace_selftest') {
    Invoke-ClangCL -Name 'module_trace_selftest' `
        -Sources @("$repo\tests\module_trace_selftest\selftest_host.cpp") `
        -OutName 'module_trace_selftest.exe'
    Invoke-ClangCL -Name 'module_trace_selftest_dll_a' `
        -Sources @("$repo\tests\module_trace_selftest\test_dll_a.cpp") `
        -LinkLibs @('/DLL') `
        -OutName 'test_dll_a.dll'
    Invoke-ClangCL -Name 'module_trace_selftest_dll_b' `
        -Sources @("$repo\tests\module_trace_selftest\test_dll_b.cpp") `
        -LinkLibs @('/DLL') `
        -OutName 'test_dll_b.dll'
}

if (Want 'nvapi_trampoline_test') {
    Invoke-ClangCL -Name 'nvapi_trampoline_test' `
        -Sources @("$repo\tests\nvapi_trampoline_test\nvapi_trampoline_test.cpp") `
        -OutName 'nvapi_trampoline_test.exe'
}

if (Want 'texture_interop_test') {
    Invoke-HipCC -Name 'texture_interop_test' `
        -Sources @("$repo\tests\texture_interop_test\texture_interop_test.cpp") `
        -LinkLibs @('dxgi','d3d12','ole32','user32') `
        -OutName 'texture_interop_test.exe' -NeedsWindowsHeaders $true
}

if (Want 'module_trace_d3d12_selftest') {
    Invoke-ClangCL -Name 'module_trace_d3d12_selftest' `
        -Sources @("$repo\tests\module_trace_d3d12_selftest\module_trace_d3d12_selftest.cpp") `
        -LinkLibs @('d3d12.lib','user32.lib') `
        -OutName 'module_trace_d3d12_selftest.exe'
}

if (Want 'amd_graph_replay') {
    Invoke-HipCC -Name 'amd_graph_replay' `
        -Sources @("$repo\tools\amd_graph_replay\amd_graph_replay.cpp") `
        -LinkLibs @('dxgi','ole32','user32') `
        -OutName 'amd_graph_replay.exe' -NeedsWindowsHeaders $true
}

if (Want 'residual_compositor') {
    Invoke-HipCC -Name 'residual_compositor' `
        -Sources @("$repo\tools\residual_compositor\residual_compositor.cpp") `
        -OutName 'residual_compositor.exe'
}

if (Want 'output_head_projection') {
    Invoke-HipCC -Name 'output_head_projection' `
        -Sources @("$repo\tools\output_head_projection\output_head_projection.cpp") `
        -OutName 'output_head_projection.exe'
}

if (Want 'output_head_mma_replay') {
    Invoke-HipCC -Name 'output_head_mma_replay' `
        -Sources @("$repo\tools\output_head_mma_replay\output_head_mma_replay.cpp") `
        -OutName 'output_head_mma_replay.exe'
}

if (Want 'output_head_activation_fusion') {
    Invoke-HipCC -Name 'output_head_activation_fusion' `
        -Sources @("$repo\tools\output_head_activation_fusion\output_head_activation_fusion.cpp") `
        -OutName 'output_head_activation_fusion.exe'
}

if (Want 'output_head_first_projection') {
    Invoke-HipCC -Name 'output_head_first_projection' `
        -Sources @("$repo\tools\output_head_first_projection\output_head_first_projection.cpp") `
        -OutName 'output_head_first_projection.exe'
}

if (Want 'output_head_first128') {
    Invoke-HipCC -Name 'output_head_first128' `
        -Sources @("$repo\tools\output_head_first128\output_head_first128.cpp") `
        -OutName 'output_head_first128.exe'
}

if (Want 'output_head_mma128_175') {
    Invoke-HipCC -Name 'output_head_mma128_175' `
        -Sources @("$repo\tools\output_head_mma128_175\output_head_mma128_175.cpp") `
        -OutName 'output_head_mma128_175.exe'
}

if (Want 'output_head_qk_attention') {
    Invoke-HipCC -Name 'output_head_qk_attention' `
        -Sources @("$repo\tools\output_head_qk_attention\output_head_qk_attention.cpp") `
        -OutName 'output_head_qk_attention.exe'
}

if (Want 'output_head_softmax_v') {
    Invoke-HipCC -Name 'output_head_softmax_v' `
        -Sources @("$repo\tools\output_head_softmax_v\output_head_softmax_v.cpp") `
        -OutName 'output_head_softmax_v.exe'
}

if (Want 'swin1h_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    $sharedStages = @('first128_hidden','first128_project','mma128_175','prepare_qk','prepare_v','qk','softmax_v')
    foreach ($stage in $sharedStages) {
        & $dxc -T cs_6_0 -E main -O3 -D NR_ATTENTION_WEIGHT_SHIFT=-112 `
            -Fo (Join-Path $out "swin1h_$stage.dxil") `
            (Join-Path $repo "tools\output_head_surface_d3d12\output_head_full_${stage}_d3d12.hlsl")
        if ($LASTEXITCODE -ne 0) { throw "DXC failed: swin1h $stage" }
    }
    & $dxc -T cs_6_0 -E main -O3 -D NR_ATTENTION_WEIGHT_SHIFT=-112 `
        -Fo (Join-Path $out 'swin1h_final_projection.dxil') `
        (Join-Path $repo 'tools\output_head_surface_d3d12\output_head_final_projection_d3d12.hlsl')
    if ($LASTEXITCODE -ne 0) { throw 'DXC failed: swin1h final projection' }
    foreach ($stage in @('gather','scatter')) {
        & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out "swin1h_$stage.dxil") `
            (Join-Path $repo "tools\swin1h_d3d12\swin1h_$stage.hlsl")
        if ($LASTEXITCODE -ne 0) { throw "DXC failed: swin1h $stage" }
    }
    Invoke-ClangCL -Name 'swin1h_d3d12' -Sources @("$repo\tools\swin1h_d3d12\swin1h_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') -OutName 'swin1h_d3d12.exe'
}

if (Want 'output_head_final_projection') {
    Invoke-HipCC -Name 'output_head_final_projection' `
        -Sources @("$repo\tools\output_head_final_projection\output_head_final_projection.cpp") `
        -OutName 'output_head_final_projection.exe'
}

if (Want 'output_head_fp16_tail') {
    Invoke-HipCC -Name 'output_head_fp16_tail' `
        -Sources @("$repo\tools\output_head_fp16_tail\output_head_fp16_tail.cpp") `
        -OutName 'output_head_fp16_tail.exe'
}

if (Want 'output_head_tail_isa_probe') {
    Invoke-HipCC -Name 'output_head_tail_isa_probe' `
        -Sources @("$repo\tools\output_head_tail_isa_probe\output_head_tail_isa_probe.cpp") `
        -OutName 'output_head_tail_isa_probe.exe'
}

if (Want 'output_head_surface_store') {
    Invoke-HipCC -Name 'output_head_surface_store' `
        -Sources @("$repo\tools\output_head_surface_store\output_head_surface_store.cpp") `
        -OutName 'output_head_surface_store.exe'
}

if (Want 'output_head_resident') {
    Invoke-HipCC -Name 'output_head_resident' `
        -Sources @("$repo\tools\output_head_resident\output_head_resident.cpp") `
        -OutName 'output_head_resident.exe'
}

if (Want 'output_head_resident_d3d12') {
    Invoke-HipCC -Name 'output_head_resident_d3d12' `
        -Sources @("$repo\tools\output_head_resident_d3d12\output_head_resident_d3d12.cpp") `
        -LinkLibs @('dxgi', 'd3d12', 'ole32', 'user32') `
        -OutName 'output_head_resident_d3d12.exe' `
        -NeedsWindowsHeaders $true
}

if (Want 'output_head_surface_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 `
        -Fo (Join-Path $out 'output_head_surface_d3d12.dxil') `
        (Join-Path $repo 'tools\output_head_surface_d3d12\output_head_surface_d3d12.hlsl')
    if ($LASTEXITCODE -ne 0) { throw "DXC build failed: output_head_surface_d3d12" }
    Invoke-ClangCL -Name 'output_head_surface_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_surface_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_surface_d3d12.exe'
}

if (Want 'output_head_tail_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 `
        -Fo (Join-Path $out 'output_head_tail_d3d12.dxil') `
        (Join-Path $repo 'tools\output_head_surface_d3d12\output_head_tail_d3d12.hlsl')
    if ($LASTEXITCODE -ne 0) { throw "DXC build failed: output_head_tail_d3d12" }
    Invoke-ClangCL -Name 'output_head_tail_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_surface_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_tail_d3d12.exe'
}

if (Want 'output_head_final_projection_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 `
        -Fo (Join-Path $out 'output_head_final_projection_d3d12.dxil') `
        (Join-Path $repo 'tools\output_head_surface_d3d12\output_head_final_projection_d3d12.hlsl')
    if ($LASTEXITCODE -ne 0) { throw "DXC build failed: output_head_final_projection_d3d12" }
    Invoke-ClangCL -Name 'output_head_final_projection_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_surface_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_final_projection_d3d12.exe'
}

if (Want 'output_head_softmax_v_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 `
        -Fo (Join-Path $out 'output_head_softmax_v_d3d12.dxil') `
        (Join-Path $repo 'tools\output_head_surface_d3d12\output_head_softmax_v_d3d12.hlsl')
    if ($LASTEXITCODE -ne 0) { throw "DXC build failed: output_head_softmax_v_d3d12" }
    Invoke-ClangCL -Name 'output_head_softmax_v_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_surface_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_softmax_v_d3d12.exe'
}

if (Want 'output_head_tail_surface_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 `
        -Fo (Join-Path $out 'output_head_tail_d3d12.dxil') `
        (Join-Path $repo 'tools\output_head_surface_d3d12\output_head_tail_d3d12.hlsl')
    if ($LASTEXITCODE -ne 0) { throw "DXC build failed: output_head_tail_d3d12" }
    & $dxc -T cs_6_0 -E main -O3 `
        -Fo (Join-Path $out 'output_head_surface_d3d12.dxil') `
        (Join-Path $repo 'tools\output_head_surface_d3d12\output_head_surface_d3d12.hlsl')
    if ($LASTEXITCODE -ne 0) { throw "DXC build failed: output_head_surface_d3d12" }
    Invoke-ClangCL -Name 'output_head_tail_surface_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_tail_surface_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_tail_surface_d3d12.exe'
}

if (Want 'output_head_final_tail_surface_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    foreach ($shader in @('output_head_final_projection_d3d12','output_head_tail_d3d12','output_head_surface_d3d12')) {
        & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out "$shader.dxil") `
            (Join-Path $repo "tools\output_head_surface_d3d12\$shader.hlsl")
        if ($LASTEXITCODE -ne 0) { throw "DXC build failed: $shader" }
    }
    Invoke-ClangCL -Name 'output_head_final_tail_surface_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_final_tail_surface_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_final_tail_surface_d3d12.exe'
}

if (Want 'output_head_attention_suffix_d3d12') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    foreach ($shader in @('output_head_softmax_v_d3d12','output_head_final_projection_d3d12','output_head_tail_d3d12','output_head_surface_d3d12')) {
        & $dxc -T cs_6_0 -E main -O3 -Fo (Join-Path $out "$shader.dxil") `
            (Join-Path $repo "tools\output_head_surface_d3d12\$shader.hlsl")
        if ($LASTEXITCODE -ne 0) { throw "DXC build failed: $shader" }
    }
    Invoke-ClangCL -Name 'output_head_attention_suffix_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_attention_suffix_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_attention_suffix_d3d12.exe'
}

if ((Want 'output_head_full_d3d12') -or (Want 'output_head_recording_selftest') -or (Want 'nvapi_amd') -or (Want 'output_head_infer_d3d12')) {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    $shaderDir = Join-Path $repo 'tools\output_head_surface_d3d12'
    $shaders = @(
        'output_head_full_activation_d3d12','output_head_full_first128_hidden_d3d12',
        'output_head_full_first128_project_d3d12','output_head_full_mma128_175_d3d12',
        'output_head_full_prepare_qk_d3d12','output_head_full_prepare_v_d3d12',
        'output_head_full_qk_d3d12','output_head_full_softmax_v_d3d12',
        'output_head_final_projection_d3d12','output_head_tail_d3d12','output_head_surface_d3d12'
    )
    foreach ($shader in $shaders) {
        & $dxc -T cs_6_0 -E main -O3 -I $shaderDir -Fo (Join-Path $out "$shader.dxil") `
            (Join-Path $shaderDir "$shader.hlsl")
        if ($LASTEXITCODE -ne 0) { throw "DXC build failed: $shader" }
    }
    Invoke-ClangCL -Name 'output_head_full_d3d12' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_full_d3d12.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_full_d3d12.exe'
    Invoke-ClangCL -Name 'output_head_recording_selftest' `
        -Sources @("$repo\tools\output_head_surface_d3d12\output_head_recording_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'output_head_recording_selftest.exe'
    if (Want 'output_head_infer_d3d12') {
        Invoke-ClangCL -Name 'output_head_infer_d3d12' `
            -Sources @("$repo\tools\output_head_surface_d3d12\output_head_infer_d3d12.cpp") `
            -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
            -OutName 'output_head_infer_d3d12.exe'
    }
}

if (Want 'd3d12_residual_bridge') {
    Invoke-HipCC -Name 'd3d12_residual_bridge' `
        -Sources @("$repo\tools\d3d12_residual_bridge\d3d12_residual_bridge.cpp") `
        -LinkLibs @('dxgi', 'd3d12', 'ole32', 'user32') `
        -OutName 'd3d12_residual_bridge.exe' `
        -NeedsWindowsHeaders $true
}

if (Want 'nvapi_amd') {
    $dxc = Join-Path $envInfo.sdk_root "bin\$($envInfo.sdk_ver)\x64\dxc.exe"
    if (-not (Test-Path -LiteralPath $dxc)) { throw "DXC not found: $dxc" }
    & $dxc -T cs_6_0 -E main -O3 `
        -Fo (Join-Path $out 'cg2r_copy.dxil') `
        (Join-Path $repo 'tools\nvapi_amd\cg2r_copy.hlsl')
    if ($LASTEXITCODE -ne 0) { throw "DXC build failed: cg2r_copy" }
    Invoke-HipCC -Name 'nvapi_amd' `
        -Sources @("$repo\tools\nvapi_amd\nvapi_amd.cpp") `
        -LinkLibs @('d3d12') -ExtraFlags @(
            '-shared',
            '-Xlinker','/export:nvapi_QueryInterface',
            '-Xlinker','/export:NvapiAmd_GetDiagnostics',
            '-Xlinker','/export:NvapiAmd_ResetDiagnostics',
            '-Xlinker','/export:NvapiAmd_RegisterExternalBuffer',
            '-Xlinker','/export:NvapiAmd_UnregisterExternalBuffer',
            '-Xlinker','/export:NvapiAmd_RegisterDescriptorResource',
            '-Xlinker','/export:NvapiAmd_BeginHeadD3D12',
            '-Xlinker','/export:NvapiAmd_ReleaseHeadD3D12',
            '-Xlinker','/export:NvapiAmd_SealHeadD3D12',
            '-Xlinker','/export:NvapiAmd_GetHeadD3D12Diagnostics'
        ) `
        -OutName 'nvapi64_amd.dll' -NeedsWindowsHeaders $true
}

if (Want 'nvapi_amd_selftest') {
    Invoke-ClangCL -Name 'nvapi_amd_selftest' `
        -Sources @("$repo\tests\nvapi_amd_selftest\nvapi_amd_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'nvapi_amd_selftest.exe'
}

if (Want 'nvapi_amd_output_head_selftest') {
    Invoke-ClangCL -Name 'nvapi_amd_output_head_selftest' `
        -Sources @("$repo\tests\nvapi_amd_output_head_selftest\nvapi_amd_output_head_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'nvapi_amd_output_head_selftest.exe'
}

if (Want 'nvapi_amd_output_head_d3d12_selftest') {
    Invoke-ClangCL -Name 'nvapi_amd_output_head_d3d12_selftest' `
        -Sources @("$repo\tests\nvapi_amd_output_head_d3d12_selftest\nvapi_amd_output_head_d3d12_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'nvapi_amd_output_head_d3d12_selftest.exe'
}

if (Want 'nvapi_amd_copy_selftest') {
    Invoke-ClangCL -Name 'nvapi_amd_copy_selftest' `
        -Sources @("$repo\tests\nvapi_amd_copy_selftest\nvapi_amd_copy_selftest.cpp") `
        -LinkLibs @('dxgi.lib','d3d12.lib','ole32.lib','user32.lib') `
        -OutName 'nvapi_amd_copy_selftest.exe'
}

if (Want 'zluda_ptx_probe') {
    Invoke-ClangCL -Name 'zluda_ptx_probe' `
        -Sources @("$repo\tools\zluda_ptx_probe\zluda_ptx_probe.cpp") `
        -LinkLibs @('user32.lib') `
        -OutName 'zluda_ptx_probe.exe'
}

Write-Host "`nAll requested builds finished."
