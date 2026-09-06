param([Parameter(Mandatory=$true)][string]$StageDirectory,
      [Parameter(Mandatory=$true)][string]$WorkerRun,
      [ValidateSet('v1','v2')][string]$ControlVersion='v1')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
$worker=(Resolve-Path -LiteralPath $WorkerRun).Path
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Prepare immutable stage before game launch'}
$gatePath=Join-Path $stage 'resident_gate.json'
if(Test-Path -LiteralPath $gatePath){throw 'Gate already exists'}
$dllHash=(Get-FileHash -LiteralPath (Join-Path $stage 'ffx_capture_session.dll')).Hash
$controls=Get-Content -LiteralPath (Join-Path $stage 'controls\provenance.json') -Raw | ConvertFrom-Json
if($controls.inspector_sha256 -ne $dllHash -or @($controls.runs | Where-Object exit_code -ne 0).Count){throw 'FSR controls not bound/passed'}
$checks=@{}
foreach($pair in @(
    @('resident','results\20260905_resident_graph_v1\validation.json'),
    @('warp',"results\20260905_resident_session_warp_$ControlVersion\summary.json"),
    @('amd',"results\20260905_resident_session_amd_$ControlVersion\summary.json"),
    @('surface','results\20260905_resident_surface_amd_full_v1\summary.json'),
    @('fallback',"results\20260905_resident_failure_controls_$ControlVersion\failure_controls.json"))){
    $path=Join-Path $repo $pair[1];$data=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if($data.status -ne 'PASS'){throw "Gate failed: $($pair[0])"}
    $checks[$pair[0]]=@{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}
}
$failureStage=if($ControlVersion -eq 'v1'){'results\20260905_resident_failure_test_build'}else{'results\20260905_resident_session_stage_v2'}
if((Get-FileHash -LiteralPath (Join-Path $repo "$failureStage\ffx_capture_session.dll")).Hash -ne $dllHash){throw 'Failure test DLL mismatch'}
$ready=Get-Content -LiteralPath (Join-Path $worker 'worker.json') -Raw | ConvertFrom-Json
if($ready.status -ne 'READY' -or $ready.frames.Count -ne 0 -or $ready.runtime.model_sha256 -ne 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5' -or $ready.runtime.slot_count -ne 156){throw 'Worker not fresh/ready with accepted full graph'}
Copy-Item -LiteralPath (Join-Path $worker 'worker_name.txt') -Destination (Join-Path $stage 'worker_name.txt')
$files=@{}
foreach($name in @('ffx_capture_session.dll','ffx_session_control.exe','ffx_live_attach.exe','worker_name.txt')){$files[$name]=(Get-FileHash -LiteralPath (Join-Path $stage $name)).Hash}
@{status='RESIDENT_BRINGUP_GATE_PASS';files=$files;checks=$checks;worker_run=$worker;max_frames=12;independent_tiles=$true;full_frame_attention_verified=$false;game_runtime_verified=$false} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $gatePath -Encoding utf8
Get-Content -LiteralPath $gatePath
