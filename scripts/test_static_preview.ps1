param([Parameter(Mandatory=$true)][string]$StageDirectory)
$ErrorActionPreference='Stop'
if (Get-Process GoWR -ErrorAction SilentlyContinue) {throw 'Exit game normally before GPU controls'}
$stage=(Resolve-Path -LiteralPath $StageDirectory).Path
$gate=Join-Path $stage 'preview_gate'
if (Test-Path -LiteralPath $gate) {throw 'Preview gate exists; use a fresh stage'}
New-Item -ItemType Directory -Path $gate | Out-Null
$names=@('ffx_capture_session.dll','ffx_static_preview_selftest.exe','ffx_filter_session_fixture.dll','ffx_session_control.exe','preview_input.rgba16f','preview_output.rgba16f')
$hashes=@{};foreach($name in $names){$hashes[$name]=(Get-FileHash -LiteralPath (Join-Path $stage $name)).Hash}
$runs=@{}
foreach($mode in @('warp','amd')) {
    $result=Join-Path $gate $mode
    & (Join-Path $stage 'ffx_static_preview_selftest.exe') $result "--$mode"
    if($LASTEXITCODE -ne 0){throw "Preview $mode execution failed"}
    $report=Get-Content -LiteralPath (Join-Path $result 'summary.json') -Raw | ConvertFrom-Json
    if($report.status -ne 'STATIC_PREVIEW_SESSION_PASS' -or !$report.all_pixels_exact -or !$report.stop_restores_next_frame -or $report.d3d12_errors -ne 0 -or $report.d3d12_warnings -ne 0 -or $report.cases -ne 9){throw "Preview $mode verification failed"}
    if($mode -eq 'amd' -and $report.adapter_vendor -ne 4098){throw 'AMD adapter required'}
    $runs[$mode]=$report
}
foreach($name in $names){if((Get-FileHash -LiteralPath (Join-Path $stage $name)).Hash -ne $hashes[$name]){throw "File changed during controls: $name"}}
@{status='STATIC_PREVIEW_GATE_PASS';files=$hashes;runs=$runs;live_inference=$false} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $gate 'gate.json') -Encoding utf8
Write-Host "Preview gate passed: $gate"
