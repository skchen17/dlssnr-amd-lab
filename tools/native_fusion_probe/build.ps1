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
    $argsList=@('-x','hip',"$PSScriptRoot/native_fusion_quantize.hip",'--offload-arch=gfx1201',"--rocm-path=$sdk/lib/llvm",'--hip-link','-shared','-O2','-std=c++17','-ffp-contract=off','-fno-fast-math','-fuse-ld=lld',"-I$sdk/include","-L$sdk/lib",'-lamdhip64','-o',"$OutputDirectory/native_fusion_quantize.dll")
    foreach($arg in ((Get-SdkLibArgs $info)+(Get-MsvcLibArgs $info))){$argsList+=(($arg -replace '/LIBPATH:','-L') -replace '"','')}
    & $info.clangxx @argsList
    if($LASTEXITCODE -ne 0){throw "HIP compile failed: $LASTEXITCODE"}
    Get-FileHash "$OutputDirectory/native_fusion_quantize.dll"
} finally { $env:HIP_PATH=$oldHip }
