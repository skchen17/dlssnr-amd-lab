param(
    [Parameter(Mandatory=$true)][string]$StageDirectory,
    [string]$GameDirectory='C:\DATA\GAME\GODOFWAR',
    [string]$Python='C:\DATA\Tools\ANACONDA\python.exe'
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
$probe=Join-Path $stage 'ffx_dispatch_probe.exe'
$session=Join-Path $stage 'ffx_capture_session.dll'
$provider=(Resolve-Path -LiteralPath (Join-Path $GameDirectory 'amd_fidelityfx_dx12.dll')).Path
$audit=Get-Content -LiteralPath (Join-Path $repo 'results\20260905_gowr_target_audit\manifest.json') -Raw | ConvertFrom-Json
$expected=@($audit.inventory | Where-Object name -eq 'amd_fidelityfx_dx12.dll')
if ($expected.Count -ne 1 -or (Get-FileHash -LiteralPath $provider).Hash -ne $expected[0].sha256) { throw 'Provider hash mismatch' }
$signature=Get-AuthenticodeSignature -LiteralPath $provider
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Advanced Micro Devices') { throw 'Provider signature mismatch' }
$result=Join-Path $stage 'controls'
if (Test-Path -LiteralPath $result) { throw 'Controls already exist; use a fresh staging directory' }
New-Item -ItemType Directory -Path $result | Out-Null
$provenance=@{session_sha256=(Get-FileHash -LiteralPath $session).Hash;probe_sha256=(Get-FileHash -LiteralPath $probe).Hash;provider_sha256=(Get-FileHash -LiteralPath $provider).Hash;game_launched=$false;runs=@()}
$provenance.inspector_sha256=$provenance.session_sha256
$provenance.command_path_required=$true
$provenance.resource_identity_required=$true
$provenance.boundary_return_required=$true
$provenance.live_output_capture_required=$true
$provenance.live_output_roundtrip_required=$true
$provenance.live_output_patch_required=$true
$provenance.live_output_filter_required=$true
$provenance.live_output_network_required=$true
$provenance.client_sha256=(Get-FileHash -LiteralPath (Join-Path $stage 'ffx_live_attach.exe')).Hash
try {
    foreach ($name in @('fsr3','fsr4')) {
        $version=if ($name -eq 'fsr3') {'3.1.0'} else {'4.1.1 *'}
        foreach ($mode in @('baseline','session-observe','session-window','session','session-boundary-negative','session-output-boundary','session-output-roundtrip','session-output-patch','session-output-filter','session-output-network')) {
            $suffix=switch ($mode) {'baseline' {''} 'session-observe' {'_observe'} 'session-window' {'_window'} 'session' {'_session'} 'session-boundary-negative' {'_boundary_negative'} 'session-output-boundary' {'_boundary_hook'} 'session-output-roundtrip' {'_roundtrip_hook'} 'session-output-patch' {'_patch_hook'} 'session-output-filter' {'_filter_hook'} 'session-output-network' {'_network_hook'}}
            $destination=Join-Path $result ($name+$suffix)
            if ($mode -eq 'baseline') { & $probe $provider $version $destination }
            else { & $probe $provider $version $destination $mode }
            $provenance.runs+=@{provider=$version;mode=$mode;exit_code=$LASTEXITCODE}
            if ($LASTEXITCODE -ne 0) { throw "Native control failed: $name / $mode" }
        }
    }
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_session.py') $result
    if ($LASTEXITCODE -ne 0) { throw 'Capture regression failed' }
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_state_trace.py') $result --paired --output (Join-Path $result 'state_trace_analysis.json')
    if ($LASTEXITCODE -ne 0) { throw 'Trace control failed' }
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_state_trace.py') $result --window-pairs --output (Join-Path $result 'window_analysis.json')
    if ($LASTEXITCODE -ne 0) { throw 'Window control failed' }
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_command_path.py') $result --controls --output (Join-Path $result 'command_path_analysis.json')
    if ($LASTEXITCODE -ne 0) { throw 'Command path control failed' }
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_resource_path.py') $result --controls --output (Join-Path $result 'resource_path_analysis.json')
    if ($LASTEXITCODE -ne 0) { throw 'Resource identity control failed' }
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_boundary_return.py') $result --controls --output (Join-Path $result 'boundary_return_analysis.json')
    if ($LASTEXITCODE -ne 0) { throw 'Boundary callback return control failed' }
} finally {
    $provenance | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $result 'provenance.json') -Encoding utf8
}
Write-Host "Controls: $result"
