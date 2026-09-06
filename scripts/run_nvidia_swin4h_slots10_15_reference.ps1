[CmdletBinding()]
param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $PSScriptRoot }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }
$probe = Join-Path $payload 'neural_reference_probe.exe'
$cases = @(
    [ordered]@{slot=10;ptx='inpview.ptx';function='cc_tinlayout_fused_swin_4h_128_4_inpview_tilesync_fp8';grid=@(10,6,1);releases=60;wait=$false;release=$true;extra=$false},
    [ordered]@{slot=11;ptx='chained.ptx';function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';grid=@(11,7,1);releases=77;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=12;ptx='chained.ptx';function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';grid=@(11,6,1);releases=66;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=13;ptx='chained.ptx';function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';grid=@(10,7,1);releases=70;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=14;ptx='chained.ptx';function='cc_tinlayout_fused_swin_4h_128_4_chained_fp8';grid=@(10,6,1);releases=60;wait=$true;release=$true;extra=$false},
    [ordered]@{slot=15;ptx='ds_wait.ptx';function='cc_tinlayout_fused_swin_4h_128_4_ds_wait_fp8';grid=@(11,7,1);releases=0;wait=$true;release=$false;extra=$true}
)
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_swin4h_slots10_15_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$records = @()
foreach ($case in $cases) {
    $slot = $case.slot
    $caseDir = Join-Path $result "slot$slot"
    New-Item -ItemType Directory -Path $caseDir | Out-Null
    $input = Join-Path $payload "slot${slot}_input.raw"
    $weights = Join-Path $payload "slot${slot}_weights.raw"
    $params = Join-Path $payload "slot${slot}_params.raw"
    $output = Join-Path $caseDir 'output.raw'
    $sync = Join-Path $caseDir 'sync.raw'
    $probeJson = Join-Path $caseDir 'probe.json'
    $args = @(
        '--nvcuda',$nvcuda,'--ptx',(Join-Path $payload $case.ptx),'--function',$case.function,
        '--json',$probeJson,'--n1-input',$input,'--n1-weights',$weights,'--n1-params',$params,
        '--n1-output',$output,'--n1-sync-out',$sync,'--n1-weight-view-offset','0',
        '--n1-wait-param-offset','48','--n1-release-param-offset','64','--n1-extra-param-offset','72',
        '--n1-grid-x',[string]$case.grid[0],'--n1-grid-y',[string]$case.grid[1],
        '--n1-block-x','32','--n1-block-y','4','--n1-expected-releases',[string]$case.releases
    )
    if ($case.wait) { $args += '--n1-wait-ready' }
    if (-not $case.release) { $args += '--n1-no-release' }
    $extraOutput = Join-Path $caseDir 'extra_output.raw'
    if ($case.extra) { $args += @('--n1-extra-output',$extraOutput) }
    & $probe @args *> (Join-Path $caseDir 'stdout.log')
    $exitCode = $LASTEXITCODE
    $p = if (Test-Path $probeJson) { Get-Content $probeJson -Raw | ConvertFrom-Json } else { $null }
    $records += [ordered]@{
        slot=$slot;grid=$case.grid;block=@(32,4,1);exit_code=$exitCode
        pass=[bool]($p -and $p.pass);device_name=if($p){$p.device_name}else{$null}
        kernel_launched=[bool]($p -and $p.kernel_launched);execution_verified=[bool]($p -and $p.execution_verified)
        observed_releases=if($p){[uint64]$p.n1_sync_zero_words}else{0}
        input_sha256=(Get-FileHash $input -Algorithm SHA256).Hash
        weights_sha256=(Get-FileHash $weights -Algorithm SHA256).Hash
        params_sha256=(Get-FileHash $params -Algorithm SHA256).Hash
        output_bytes=if(Test-Path $output){(Get-Item $output).Length}else{0}
        output_sha256=if(Test-Path $output){(Get-FileHash $output -Algorithm SHA256).Hash}else{$null}
        extra_output_bytes=if($case.extra -and (Test-Path $extraOutput)){(Get-Item $extraOutput).Length}else{0}
        extra_output_sha256=if($case.extra -and (Test-Path $extraOutput)){(Get-FileHash $extraOutput -Algorithm SHA256).Hash}else{$null}
        sync_sha256=if(Test-Path $sync){(Get-FileHash $sync -Algorithm SHA256).Hash}else{$null}
    }
}
$packageManifest = Get-Content (Join-Path $root 'manifest.json') -Raw | ConvertFrom-Json
$hashes = [ordered]@{}
foreach ($name in $packageManifest.payload_sha256.psobject.Properties.Name) {
    $hashes[$name] = (Get-FileHash (Join-Path $payload $name) -Algorithm SHA256).Hash
}
$integrity = $true
foreach ($name in $hashes.Keys) { $integrity = $integrity -and $hashes[$name] -eq $packageManifest.payload_sha256.$name }
$pass = $integrity -and @($records | Where-Object { -not $_.pass }).Count -eq 0
[ordered]@{schema=1;experiment='rtx_swin4h_slots10_15_same_input_reference';status=if($pass){'PASS'}else{'FAIL'};classification='RTX_ORIGINAL_PTX_NUMERICAL_REFERENCE';counts_as_s7=$false;payload_integrity=$integrity;payload_sha256=$hashes;slots=$records} |
    ConvertTo-Json -Depth 8 | Set-Content (Join-Path $result 'manifest.json') -Encoding utf8
$archive = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content (Join-Path $result 'manifest.json')
Write-Host "Result: $archive"
exit $(if($pass){0}else{1})
