$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
. (Join-Path $PSScriptRoot 'build_common.ps1')

$repo = Split-Path -Parent $PSScriptRoot
$dll = Join-Path $repo 'build\nvapi64_amd.dll'
$exe = Join-Path $repo 'build\nvapi_amd_selftest.exe'
# The diagnostics ABI evolves with the backend.  Always rebuild both sides so a
# stale self-test executable cannot silently use an older struct size.
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only nvapi_amd,nvapi_amd_selftest
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

$buildEnv = Resolve-BuildEnv
$env:PATH = "$($buildEnv.hip_path)\bin;$($buildEnv.hip_path)\lib\llvm\bin;$env:PATH"
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_nvapi_amd_selftest"
New-Item -ItemType Directory -Path $result | Out-Null
$json = Join-Path $result 'nvapi_amd_selftest.json'
$log = Join-Path $result 'stdout.log'
& $exe --dll $dll --json $json 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
$manifest = [ordered]@{
    schema = 1
    experiment = 'nvapi_amd_selftest'
    classification = 'LAB_TRANSPORT_ONLY'
    counts_as_s6 = $false
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash
    dll_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash
    nvapi_source_commit = 'cd6918f60b3c9a0476fdfe7e89bb32330602049d'
    exit_code = $code
}
$manifest | ConvertTo-Json | Out-File -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Write-Host "NVAPI AMD self-test exit=$code result=$result"
exit $code
