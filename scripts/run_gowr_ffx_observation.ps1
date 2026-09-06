param([string]$GameDirectory='C:\DATA\GAME\GODOFWAR', [switch]$DecodeUpscaleMetadata)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$game=(Resolve-Path -LiteralPath $GameDirectory).Path
if (Get-Process -Name GoWR -ErrorAction SilentlyContinue) { throw 'GoWR is already running. Exit normally before this new-process observation; no existing process will be attached or stopped.' }
$audit=Get-Content -LiteralPath (Join-Path $repo 'results\20260905_gowr_target_audit\manifest.json') -Raw | ConvertFrom-Json
$names=@('GoWR.exe','sl.interposer.dll','amd_fidelityfx_dx12.dll','version.dll')
$before=@{}
foreach ($name in $names) {
    $expected=@($audit.inventory | Where-Object name -eq $name)
    $hash=(Get-FileHash -LiteralPath (Join-Path $game $name)).Hash
    if ($expected.Count -ne 1 -or $hash -ne $expected[0].sha256) { throw "Audited image changed: $name" }
    $before[$name]=$hash
}
$signature=Get-AuthenticodeSignature -LiteralPath (Join-Path $game 'amd_fidelityfx_dx12.dll')
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Advanced Micro Devices') { throw 'AMD signature validation failed' }
$result=Join-Path $repo ('results\{0}_gowr_ffx_observation' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$launchArgs=@((Join-Path $game 'GoWR.exe'),(Join-Path $repo 'build\ffx_observer.dll'),$result)
if ($DecodeUpscaleMetadata) { $launchArgs+='decode-upscale' }
& (Join-Path $repo 'build\ffx_bootstrap_launcher.exe') @launchArgs
$exitCode=$LASTEXITCODE
$after=@{}
foreach ($name in $names) { $after[$name]=(Get-FileHash -LiteralPath (Join-Path $game $name)).Hash }
@{game_directory=$game;before=$before;after=$after;launcher_exit_code=$exitCode;
  observer_sha256=(Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_observer.dll')).Hash;
  launcher_sha256=(Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_bootstrap_launcher.exe')).Hash;
  profile=$(if ($DecodeUpscaleMetadata) {'sdk_1_1_3_metadata_only'} else {'headers_only'});game_files_deployed=@();texture_capture_enabled=$false} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $result 'provenance.json') -Encoding utf8
foreach ($name in $names) { if ($before[$name] -ne $after[$name]) { throw "Image changed during bootstrap: $name" } }
if ($exitCode -ne 0) { throw "Game observation bootstrap failed: $result" }
Write-Host "Observation started. Result: $result"
Write-Host 'Enter a playable scene manually. Exit the game normally when finished; no proxy DLL was installed.'
