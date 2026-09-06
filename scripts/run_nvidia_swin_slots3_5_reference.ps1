[CmdletBinding()]
param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$scriptDirectory = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDirectory)) { $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path }
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $scriptDirectory }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }
$probe = Join-Path $payload 'neural_reference_probe.exe'
$cases = @(
    [ordered]@{slot=3;ptx='chained.ptx';function='cc_tinlayout_fused_swin_1h_32_1_chained_fp8';grid_x=41;grid_y=25;releases=1025;extra=$false},
    [ordered]@{slot=4;ptx='chained.ptx';function='cc_tinlayout_fused_swin_1h_32_1_chained_fp8';grid_x=41;grid_y=24;releases=984;extra=$false},
    [ordered]@{slot=5;ptx='ds_wait.ptx';function='cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8';grid_x=40;grid_y=25;releases=0;extra=$true}
)
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_swin_slots3_5_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$records = @()
foreach ($case in $cases) {
    $slot = $case.slot
    $caseDir = Join-Path $result "slot$slot"
    New-Item -ItemType Directory -Path $caseDir | Out-Null
    $output = Join-Path $caseDir 'output.raw'
    $sync = Join-Path $caseDir 'sync.raw'
    $probeJson = Join-Path $caseDir 'probe.json'
    $args = @(
        '--nvcuda',$nvcuda,'--ptx',(Join-Path $payload $case.ptx),'--function',$case.function,
        '--json',$probeJson,'--n1-input',(Join-Path $payload "slot${slot}_input.raw"),
        '--n1-weights',(Join-Path $payload "slot${slot}_weights.raw"),
        '--n1-params',(Join-Path $payload "slot${slot}_params.raw"),
        '--n1-output',$output,'--n1-sync-out',$sync,'--n1-wait-ready',
        '--n1-weight-view-offset','0','--n1-grid-x',[string]$case.grid_x,
        '--n1-grid-y',[string]$case.grid_y,'--n1-expected-releases',[string]$case.releases
    )
    $extraOutput = Join-Path $caseDir 'extra_output.raw'
    if ($case.extra) { $args += @('--n1-extra-output',$extraOutput) }
    & $probe @args *> (Join-Path $caseDir 'stdout.log')
    $exitCode = $LASTEXITCODE
    $probeResult = if (Test-Path -LiteralPath $probeJson) { Get-Content -LiteralPath $probeJson -Raw | ConvertFrom-Json } else { $null }
    $records += [ordered]@{
        slot=$slot;exit_code=$exitCode;pass=[bool]($probeResult -and $probeResult.pass)
        device_name=if($probeResult){$probeResult.device_name}else{$null}
        kernel_launched=[bool]($probeResult -and $probeResult.kernel_launched)
        execution_verified=[bool]($probeResult -and $probeResult.execution_verified)
        observed_releases=if($probeResult){[uint64]$probeResult.n1_sync_zero_words}else{0}
        output_bytes=if(Test-Path -LiteralPath $output){(Get-Item -LiteralPath $output).Length}else{0}
        output_sha256=if(Test-Path -LiteralPath $output){(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash}else{$null}
        extra_output_bytes=if($case.extra -and (Test-Path -LiteralPath $extraOutput)){(Get-Item -LiteralPath $extraOutput).Length}else{0}
        extra_output_sha256=if($case.extra -and (Test-Path -LiteralPath $extraOutput)){(Get-FileHash -LiteralPath $extraOutput -Algorithm SHA256).Hash}else{$null}
        sync_sha256=if(Test-Path -LiteralPath $sync){(Get-FileHash -LiteralPath $sync -Algorithm SHA256).Hash}else{$null}
    }
}
$packageManifest = Get-Content -LiteralPath (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
$hashes = [ordered]@{}
foreach ($name in $packageManifest.payload_sha256.psobject.Properties.Name) {
    $hashes[$name] = (Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash
}
$integrity = $true
foreach ($name in $hashes.Keys) { $integrity = $integrity -and $hashes[$name] -eq $packageManifest.payload_sha256.$name }
$pass = $integrity -and @($records | Where-Object { -not $_.pass }).Count -eq 0
[ordered]@{
    schema=1;experiment='rtx_swin_slots3_5_same_input_reference'
    status=if($pass){'PASS'}else{'FAIL'}
    classification='RTX_ORIGINAL_PTX_NUMERICAL_REFERENCE'
    counts_as_s7=$false;payload_integrity=$integrity;payload_sha256=$hashes;slots=$records
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$archive = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $archive"
exit $(if($pass){0}else{1})
