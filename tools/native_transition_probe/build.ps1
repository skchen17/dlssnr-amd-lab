param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $repo 'scripts/build_common.ps1')
if(-not [IO.Path]::IsPathFullyQualified($OutputDirectory)){throw 'OutputDirectory must be absolute'}
if(Test-Path -LiteralPath $OutputDirectory){throw 'Refusing to overwrite existing build directory'}
$sdk=Join-Path $repo '.venv-rocm/Lib/site-packages/_rocm_sdk_core'
$oldHip=$env:HIP_PATH
try {
    $env:HIP_PATH=$sdk
    $info=Resolve-BuildEnv
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $source="$PSScriptRoot/encoder_transition.hip"
    $common=@('-x','hip',$source,'--offload-arch=gfx1201',"--rocm-path=$sdk/lib/llvm",'-O2','-std=c++17','-ffp-contract=off','-fno-fast-math',"-I$sdk/include")
    $link=@('--hip-link','-shared','-fuse-ld=lld',"-L$sdk/lib",'-lamdhip64')
    foreach($arg in ((Get-SdkLibArgs $info)+(Get-MsvcLibArgs $info))){$link+=(($arg -replace '/LIBPATH:','-L') -replace '"','')}
    & $info.clangxx @common @link '-o' "$OutputDirectory/native_encoder_transition.dll"
    if($LASTEXITCODE -ne 0){throw "HIP compile failed: $LASTEXITCODE"}
    & $info.clangxx @common '--cuda-device-only' '-S' '-o' "$OutputDirectory/native_encoder_transition.s"
    if($LASTEXITCODE -ne 0){throw "HIP ISA compile failed: $LASTEXITCODE"}
    [ordered]@{compiler=$info.clangxx;arguments=$common;link=$link;source_sha256=(Get-FileHash $source).Hash;
        dll_sha256=(Get-FileHash "$OutputDirectory/native_encoder_transition.dll").Hash;
        isa_sha256=(Get-FileHash "$OutputDirectory/native_encoder_transition.s").Hash;gpu_executed=$false} |
        ConvertTo-Json -Depth 5 | Out-File "$OutputDirectory/build.json" -Encoding utf8
    Select-String -Path "$OutputDirectory/native_encoder_transition.s" -Pattern 'NumVgprs|ScratchSize|LDSByteSize|\.amdhsa_group_segment_fixed_size|\.amdhsa_private_segment_fixed_size'
} finally {$env:HIP_PATH=$oldHip}
