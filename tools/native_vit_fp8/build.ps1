param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
. "$PSScriptRoot/../../scripts/build_common.ps1"
if(-not [IO.Path]::IsPathFullyQualified($OutputDirectory)-or(Test-Path -LiteralPath $OutputDirectory)){throw 'new absolute output directory required'}
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$sdk=Join-Path $repo '.venv-rocm/Lib/site-packages/_rocm_sdk_core'
$oldHip=$env:HIP_PATH
try{
  $env:HIP_PATH=$sdk;$info=Resolve-BuildEnv;New-Item -ItemType Directory -Path $OutputDirectory|Out-Null
  $common=@('-x','hip','--offload-arch=gfx1201',"--rocm-path=$sdk/lib/llvm",'-O3','-std=c++17',"-I$sdk/include")
  $link=@('--hip-link','-fuse-ld=lld',"-L$sdk/lib",'-lamdhip64')
  foreach($arg in ((Get-SdkLibArgs $info)+(Get-MsvcLibArgs $info))){$link+=(($arg-replace '/LIBPATH:','-L')-replace '"','')}
  & $info.clangxx @common "$PSScriptRoot/vit_fp8.hip" @link '-shared' '-o' "$OutputDirectory/native_vit_fp8.dll"
  if($LASTEXITCODE-ne 0){throw 'native ViT FP8 compile failed'}
  & $info.clangxx @common "$PSScriptRoot/vit_fp8.hip" '--cuda-device-only' '-S' '-o' "$OutputDirectory/native_vit_fp8.s"
  if($LASTEXITCODE-ne 0){throw 'native ViT FP8 ISA compile failed'}
  $isa=Get-Content "$OutputDirectory/native_vit_fp8.s" -Raw
  $wmma=([regex]::Matches($isa,'v_wmma_f32_16x16x16_fp8_fp8')).Count
  if($wmma-lt 5){throw "expected five ViT FP8 WMMA sites, found $wmma"}
  [ordered]@{source_sha256=(Get-FileHash "$PSScriptRoot/vit_fp8.hip").Hash;binary_sha256=(Get-FileHash "$OutputDirectory/native_vit_fp8.dll").Hash;isa_sha256=(Get-FileHash "$OutputDirectory/native_vit_fp8.s").Hash;target='gfx1201';fp8_wmma_sites=$wmma;gpu_executed=$false;scope='1024-channel resident-FP8 ViT with streamed global attention'}|ConvertTo-Json|Out-File "$OutputDirectory/build.json" -Encoding utf8
}finally{$env:HIP_PATH=$oldHip}
