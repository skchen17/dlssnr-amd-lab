param([string]$GameDirectory='C:\DATA\GAME\GODOFWAR', [string]$Python='C:\DATA\Tools\ANACONDA\python.exe', [switch]$CaptureValidation, [switch]$LiveInspectionValidation, [switch]$DepthPlaneValidation, [switch]$ContractValidation, [switch]$AlignedDepthControl, [switch]$SessionValidation)
$ErrorActionPreference='Stop'
if ($ContractValidation -and $LiveInspectionValidation) { throw 'Select only one inspector validation mode' }
if ($SessionValidation -and ($ContractValidation -or $LiveInspectionValidation)) { throw 'Select only one session/inspector validation mode' }
if ($AlignedDepthControl -and -not $DepthPlaneValidation) { throw 'AlignedDepthControl requires DepthPlaneValidation' }
$repo=Split-Path -Parent $PSScriptRoot
$dll=(Resolve-Path -LiteralPath (Join-Path $GameDirectory 'amd_fidelityfx_dx12.dll')).Path
$audit=Get-Content -LiteralPath (Join-Path $repo 'results\20260905_gowr_target_audit\manifest.json') -Raw | ConvertFrom-Json
$expected=@($audit.inventory | Where-Object name -eq 'amd_fidelityfx_dx12.dll')
if ($expected.Count -ne 1 -or (Get-FileHash -LiteralPath $dll).Hash -ne $expected[0].sha256) { throw 'Provider differs from audited file; repeat static audit first.' }
$signature=Get-AuthenticodeSignature -LiteralPath $dll
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Advanced Micro Devices') { throw 'Expected valid AMD publisher signature.' }
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_dispatch_probe
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
if ($LiveInspectionValidation) {
    & (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_live_inspector
    if ($LASTEXITCODE -ne 0) { throw 'Inspector build failed' }
}
if ($ContractValidation) {
    & (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_contract_inspector
    if ($LASTEXITCODE -ne 0) { throw 'Contract inspector build failed' }
}
if ($SessionValidation) {
    & (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_capture_session
    if ($LASTEXITCODE -ne 0) { throw 'Session build failed' }
}
$result=Join-Path $repo ('results\{0}_ffx_dispatch' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $result | Out-Null
$runs=@()
$inspectorHash=if ($LiveInspectionValidation) { (Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_live_inspector.dll')).Hash } else { $null }
if ($ContractValidation) { $inspectorHash=(Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_contract_inspector.dll')).Hash }
if ($SessionValidation) { $inspectorHash=(Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_capture_session.dll')).Hash }
try {
    # Separate processes and explicit version overrides, never a silent fallback.
    foreach ($provider in @('3.1.0','4.1.1 *')) {
        $name=if ($provider -eq '3.1.0') {'fsr3'} else {'fsr4'}
        $directory=Join-Path $result $name
        & (Join-Path $repo 'build\ffx_dispatch_probe.exe') $dll $provider $directory
        $runs+=@{provider=$provider;directory=$name;exit_code=$LASTEXITCODE}
        if ($LASTEXITCODE -ne 0) { Write-Warning "$provider failed; inspect $directory" }
        if ($LiveInspectionValidation) {
            $inspectName=$name+'_inspect'
            & (Join-Path $repo 'build\ffx_dispatch_probe.exe') $dll $provider (Join-Path $result $inspectName) inspect
            $runs+=@{provider=$provider;directory=$inspectName;exit_code=$LASTEXITCODE}
            if ($LASTEXITCODE -ne 0) { throw "$provider inspection failed" }
        }
        if ($ContractValidation) {
            $contractName=$name+'_contract'
            & (Join-Path $repo 'build\ffx_dispatch_probe.exe') $dll $provider (Join-Path $result $contractName) contract
            $runs+=@{provider=$provider;directory=$contractName;exit_code=$LASTEXITCODE}
            if ($LASTEXITCODE -ne 0) { throw "$provider contract inspection failed" }
        }
        if ($SessionValidation) {
            $sessionName=$name+'_session'
            & (Join-Path $repo 'build\ffx_dispatch_probe.exe') $dll $provider (Join-Path $result $sessionName) session
            $runs+=@{provider=$provider;directory=$sessionName;exit_code=$LASTEXITCODE}
            if ($LASTEXITCODE -ne 0) { throw "$provider session failed" }
        }
        if ($CaptureValidation) {
            $captureName=$name+'_capture'
            & (Join-Path $repo 'build\ffx_dispatch_probe.exe') $dll $provider (Join-Path $result $captureName) capture
            $runs+=@{provider=$provider;directory=$captureName;exit_code=$LASTEXITCODE}
            if ($LASTEXITCODE -ne 0) { Write-Warning "$provider capture failed" }
        }
        if ($DepthPlaneValidation) {
            $depthModes=if ($AlignedDepthControl) { @('depth-aligned-base','depth-aligned-capture') } else { @('depth-base','depth-capture') }
            foreach ($mode in $depthModes) {
                $depthName=$name+'_'+$mode
                & (Join-Path $repo 'build\ffx_dispatch_probe.exe') $dll $provider (Join-Path $result $depthName) $mode
                $runs+=@{provider=$provider;directory=$depthName;exit_code=$LASTEXITCODE}
                if ($LASTEXITCODE -ne 0) { throw "$provider $mode failed" }
            }
        }
    }
} finally {
    $after=(Get-FileHash -LiteralPath $dll).Hash
    if ($LiveInspectionValidation -and (Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_live_inspector.dll')).Hash -ne $inspectorHash) { throw 'Inspector changed during validation' }
    if ($ContractValidation -and (Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_contract_inspector.dll')).Hash -ne $inspectorHash) { throw 'Contract inspector changed during validation' }
    if ($SessionValidation -and (Get-FileHash -LiteralPath (Join-Path $repo 'build\ffx_capture_session.dll')).Hash -ne $inspectorHash) { throw 'Session changed during validation' }
    @{dll=$dll;sha256_before=$expected[0].sha256;sha256_after=$after;inspector_sha256=$inspectorHash;signature_status=[string]$signature.Status;runs=$runs;game_launched=$false} |
        ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $result 'provenance.json') -Encoding utf8
    if ($after -ne $expected[0].sha256) { throw 'Provider changed during probe' }
}
& $Python (Join-Path $PSScriptRoot 'analyze_ffx_dispatch_probe.py') $result
if ($LASTEXITCODE -ne 0) { throw "Dispatch verification failed: $result" }
Write-Host "Result: $result"
if ($LiveInspectionValidation) {
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_live_inspection.py') $result --pairs
    if ($LASTEXITCODE -ne 0) { throw "Live inspection verification failed: $result" }
}
if ($ContractValidation) {
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_live_inspection.py') $result --pairs --suffix _contract
    if ($LASTEXITCODE -ne 0) { throw "Contract inspection verification failed: $result" }
}
if ($SessionValidation) {
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_session.py') $result
    if ($LASTEXITCODE -ne 0) { throw "Session validation failed: $result" }
}
if ($CaptureValidation) {
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_texture_capture.py') $result
    if ($LASTEXITCODE -ne 0) { throw "Capture verification failed: $result" }
}
if ($DepthPlaneValidation) {
    $depthLayout=if ($AlignedDepthControl) {'aligned'} else {'observed'}
    & $Python (Join-Path $PSScriptRoot 'analyze_ffx_depth_capture.py') $result --layout $depthLayout
    if ($LASTEXITCODE -ne 0) { throw "Depth capture verification failed: $result" }
}
