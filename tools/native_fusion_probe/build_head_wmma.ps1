param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/../../scripts/build_common.ps1"
if(-not [IO.Path]::IsPathFullyQualified($OutputDirectory) -or (Test-Path -LiteralPath $OutputDirectory)){throw 'new absolute output directory required'}
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$sdk=Join-Path $repo '.venv-rocm/Lib/site-packages/_rocm_sdk_core'
$oldHip=$env:HIP_PATH
try {
    $env:HIP_PATH=$sdk
    $info=Resolve-BuildEnv
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $source="$PSScriptRoot/head_wmma.hip"
    $common=@('-x','hip',$source,'--offload-arch=gfx1201',"--rocm-path=$sdk/lib/llvm",'-O2','-std=c++17','-ffp-contract=off','-fno-fast-math',"-I$sdk/include")
    $link=@('--hip-link','-shared','-fuse-ld=lld',"-L$sdk/lib",'-lamdhip64')
    foreach($arg in ((Get-SdkLibArgs $info)+(Get-MsvcLibArgs $info))){$link+=(($arg -replace '/LIBPATH:','-L') -replace '"','')}
    & $info.clangxx @common @link '-o' "$OutputDirectory/head_wmma.dll"
    if($LASTEXITCODE -ne 0){throw 'DLL compile failed'}
    & $info.clangxx @common '--cuda-device-only' '-S' '-o' "$OutputDirectory/head_wmma.s"
    if($LASTEXITCODE -ne 0){throw 'ISA compile failed'}
    $isa=Get-Content "$OutputDirectory/head_wmma.s" -Raw
    if($isa -notmatch 'v_wmma_f32_16x16x16_fp8_fp8' -or $isa -notmatch 'v_wmma_f32_16x16x16_f16'){throw 'missing WMMA ISA'}
    [ordered]@{compiler=$info.clangxx;arguments=$common;link=$link;source_sha256=(Get-FileHash $source).Hash;dll_sha256=(Get-FileHash "$OutputDirectory/head_wmma.dll").Hash;isa_sha256=(Get-FileHash "$OutputDirectory/head_wmma.s").Hash;gpu_executed=$false} | ConvertTo-Json -Depth 5 | Out-File "$OutputDirectory/build.json" -Encoding utf8
    Select-String -Path "$OutputDirectory/head_wmma.s" -Pattern 'NumVgprs|ScratchSize|LDSByteSize|\.amdhsa_group_segment_fixed_size|\.amdhsa_private_segment_fixed_size'
} finally { $env:HIP_PATH=$oldHip }
