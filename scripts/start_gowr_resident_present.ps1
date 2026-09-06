param([Parameter(Mandatory=$true)][string]$StageDirectory,
      [Parameter(Mandatory=$true)][string]$WorkerRun)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
$worker=(Resolve-Path -LiteralPath $WorkerRun).Path
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Exit game normally first'}
$validation=Get-Content -LiteralPath (Join-Path $stage 'present_validation.json') -Raw | ConvertFrom-Json
if($validation.status -ne 'PASS'){throw 'Display bridge not validated'}
foreach($entry in $validation.files.PSObject.Properties){if((Get-FileHash -LiteralPath (Join-Path $stage $entry.Name)).Hash -ne $entry.Value){throw 'Validated binary changed'}}
foreach($entry in $validation.evidence.PSObject.Properties){if((Get-FileHash -LiteralPath $entry.Value.path).Hash -ne $entry.Value.sha256){throw 'Validation evidence changed'}}
$ready=Get-Content -LiteralPath (Join-Path $worker 'worker.json') -Raw | ConvertFrom-Json
if($ready.status -ne 'READY' -or $ready.frames.Count -ne 0 -or $ready.runtime.model_sha256 -ne 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5' -or $ready.runtime.slot_count -ne 156 -or $ready.runtime.rtx_intermediate_injection){throw 'Fresh original-weight worker required'}
$game='C:\DATA\GAME\GODOFWAR'
$audit=Get-Content -LiteralPath (Join-Path $repo 'results\20260905_gowr_target_audit\manifest.json') -Raw | ConvertFrom-Json
$before=@{}
foreach($name in @('GoWR.exe','amd_fidelityfx_dx12.dll','sl.interposer.dll','version.dll')){
    $expected=@($audit.inventory | Where-Object name -eq $name)
    $hash=(Get-FileHash -LiteralPath (Join-Path $game $name)).Hash
    if($expected.Count -ne 1 -or $hash -ne $expected[0].sha256){throw "Audited game image changed: $name"}
    $before[$name]=$hash
}
Copy-Item -LiteralPath (Join-Path $worker 'worker_name.txt') -Destination (Join-Path $stage 'worker_name.txt')
$result=Join-Path $repo ('results\{0}_gowr_resident_present' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& (Join-Path $stage 'ffx_bootstrap_launcher.exe') (Join-Path $game 'GoWR.exe') (Join-Path $stage 'resident_present.dll') $result
$code=$LASTEXITCODE
$launch=Get-Content -LiteralPath (Join-Path $result 'bootstrap.json') -Raw | ConvertFrom-Json
if($code -ne 0 -or $launch.status -ne 'BOOTSTRAP_PASS'){throw "Bootstrap failed: $result"}
$process=Get-Process -Id $launch.pid
@{pid=$process.Id;process_start=$process.StartTime.ToString('o');exit_code=$code;before=$before;
  stage=$stage;worker=$worker;validation_sha256=(Get-FileHash -LiteralPath (Join-Path $stage 'present_validation.json')).Hash;
  files=$validation.files;worker_name_sha256=(Get-FileHash -LiteralPath (Join-Path $stage 'worker_name.txt')).Hash;
  display_referred_debug_route=$true;network_enabled=$false;game_files_deployed=@()} |
    ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $result 'provenance.json') -Encoding utf8
Write-Host "Dormant display bridge: $result"

