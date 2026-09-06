param([Parameter(Mandatory=$true)][string]$FreshStageDirectory,
      [string]$GameDirectory='C:\DATA\GAME\GODOFWAR',
      [string]$Python='C:\DATA\Tools\ANACONDA\python.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$provider=(Resolve-Path -LiteralPath (Join-Path $GameDirectory 'amd_fidelityfx_dx12.dll')).Path
$audit=Get-Content -LiteralPath (Join-Path $repo 'results\20260905_gowr_target_audit\manifest.json') -Raw | ConvertFrom-Json
$expected=@($audit.inventory | Where-Object name -eq 'amd_fidelityfx_dx12.dll')
if ($expected.Count -ne 1 -or (Get-FileHash -LiteralPath $provider).Hash -ne $expected[0].sha256) {throw 'Provider hash mismatch'}
$signature=Get-AuthenticodeSignature -LiteralPath $provider
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Advanced Micro Devices') {throw 'Provider signature mismatch'}
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_output_boundary_selftest,ffx_dispatch_probe -FreshOutputDirectory $FreshStageDirectory
if ($LASTEXITCODE -ne 0) {throw 'Boundary build failed'}
$stage=(Resolve-Path -LiteralPath $FreshStageDirectory).Path
$probe=Join-Path $stage 'ffx_dispatch_probe.exe'
$primitive=Join-Path $stage 'ffx_output_boundary_selftest.exe'
$result=Join-Path $stage 'controls'
New-Item -ItemType Directory -Path $result | Out-Null
$provenance=@{provider=$provider;provider_sha256=$expected[0].sha256;probe_sha256=(Get-FileHash -LiteralPath $probe).Hash;
  primitive_sha256=(Get-FileHash -LiteralPath $primitive).Hash;game_files_deployed=@();game_launched=$false;live_hook_validated=$false;runs=@()}
try {
    & $primitive (Join-Path $stage 'primitive')
    if ($LASTEXITCODE -ne 0) {throw 'Boundary primitive failed'}
    foreach ($name in @('fsr3','fsr4')) {
        $version=if ($name -eq 'fsr3') {'3.1.0'} else {'4.1.1 *'}
        foreach ($mode in @('baseline','output-boundary','output-roundtrip')) {
            $dest=Join-Path $result $(if ($mode -eq 'baseline') {$name} elseif ($mode -eq 'output-boundary') {$name+'_boundary'} else {$name+'_roundtrip'})
            if ($mode -eq 'baseline') {& $probe $provider $version $dest} else {& $probe $provider $version $dest $mode}
            $provenance.runs+=@{provider=$version;mode=$mode;exit_code=$LASTEXITCODE}
            if ($LASTEXITCODE -ne 0) {throw "Boundary provider control failed: $name/$mode"}
        }
    }
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_output_boundary.py') $result --output (Join-Path $result 'analysis.json')
    if ($LASTEXITCODE -ne 0) {throw 'Boundary artifact verification failed'}
} finally {
    $provenance.provider_sha256_after=(Get-FileHash -LiteralPath $provider).Hash
    $provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stage 'provenance.json') -Encoding utf8
    if ($provenance.provider_sha256_after -ne $provenance.provider_sha256) {throw 'Provider changed'}
}
Write-Host "Boundary controls: $stage"
