param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/../../scripts/build_common.ps1"
if(-not [IO.Path]::IsPathFullyQualified($OutputDirectory)-or(Test-Path -LiteralPath $OutputDirectory)){throw 'new absolute output directory required'}
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$sdk=Join-Path $repo '.venv-rocm/Lib/site-packages/_rocm_sdk_core'
$oldHip=$env:HIP_PATH
try{
  $env:HIP_PATH=$sdk;$info=Resolve-BuildEnv;New-Item -ItemType Directory -Path $OutputDirectory|Out-Null
  $common=@('-x','hip','--offload-arch=gfx1201',"--rocm-path=$sdk/lib/llvm",'-O2','-std=c++17',"-I$sdk/include")
  $link=@('--hip-link','-fuse-ld=lld',"-L$sdk/lib",'-lamdhip64','-lbcrypt')
  foreach($arg in ((Get-SdkLibArgs $info)+(Get-MsvcLibArgs $info))){$link+=(($arg-replace '/LIBPATH:','-L')-replace '"','')}
  $stageSource=Join-Path $PSScriptRoot '..\native_stage_fp8\stage_fp8.hip'
  $vitSource=Join-Path $PSScriptRoot '..\native_vit_fp8\vit_fp8.hip'
  $bottleneckSource=Join-Path $PSScriptRoot '..\native_bottleneck_fp8\bottleneck_fp8.hip'
  $transitionSource=Join-Path $PSScriptRoot '..\native_transition_fp8\transition_fp8.hip'
  $ioSource=Join-Path $PSScriptRoot '..\native_io_fp8\io_fp8.hip'
  & $info.clangxx @common "$PSScriptRoot/nr_plan.cpp" $stageSource $vitSource $bottleneckSource $transitionSource $ioSource @link '-shared' '-o' "$OutputDirectory/nr_plan.dll"
  if($LASTEXITCODE-ne 0){throw 'NRPlan DLL compile failed'}
  & $info.clangxx @common "$PSScriptRoot/nr_plan.cpp" $stageSource $vitSource $bottleneckSource $transitionSource $ioSource "$PSScriptRoot/selftest.hip" @link '-o' "$OutputDirectory/nr_plan_selftest.exe"
  if($LASTEXITCODE-ne 0){throw 'NRPlan selftest compile failed'}
  [ordered]@{source_sha256=@{header=(Get-FileHash "$PSScriptRoot/nr_plan.h").Hash;runtime=(Get-FileHash "$PSScriptRoot/nr_plan.cpp").Hash;stage_fp8=(Get-FileHash $stageSource).Hash;vit_fp8=(Get-FileHash $vitSource).Hash;bottleneck_fp8=(Get-FileHash $bottleneckSource).Hash;transition_fp8=(Get-FileHash $transitionSource).Hash;io_fp8=(Get-FileHash $ioSource).Hash;selftest=(Get-FileHash "$PSScriptRoot/selftest.hip").Hash};binary_sha256=@{dll=(Get-FileHash "$OutputDirectory/nr_plan.dll").Hash;selftest=(Get-FileHash "$OutputDirectory/nr_plan_selftest.exe").Hash};gpu_executed=$false}|ConvertTo-Json -Depth 4|Out-File "$OutputDirectory/build.json" -Encoding utf8
}finally{$env:HIP_PATH=$oldHip}
