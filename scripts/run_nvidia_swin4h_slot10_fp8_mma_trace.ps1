[CmdletBinding()]
param([string]$PackageRoot)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$scriptDirectory = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptDirectory)) { $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path }
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $scriptDirectory }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$package = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }

$actualHashes = [ordered]@{}
$integrity = $true
foreach ($property in $package.payload_sha256.psobject.Properties) {
    $path = Join-Path $payload $property.Name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing payload: $($property.Name)" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $actualHashes[$property.Name] = $actual
    $integrity = $integrity -and $actual -eq [string]$property.Value
}
if (-not $integrity) { throw 'payload hash verification failed' }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$work = Join-Path $root "_swin4h_slot10_fp8_mma_trace_work_$stamp"
$result = Join-Path $root "_swin4h_slot10_fp8_mma_trace_reference_result_$stamp"
New-Item -ItemType Directory -Path $work,$result | Out-Null
$probe = Join-Path $payload 'neural_reference_probe.exe'

function Invoke-Slot10([string]$PtxName, [string]$Prefix, [bool]$WithTrace) {
    $output = Join-Path $work "$Prefix.output.raw"
    $sync = Join-Path $work "$Prefix.sync.raw"
    $probeJson = Join-Path $work "$Prefix.probe.json"
    $arguments = @(
        '--nvcuda',$nvcuda,'--ptx',(Join-Path $payload $PtxName),
        '--function',[string]$package.function,'--json',$probeJson,
        '--n1-input',(Join-Path $payload 'input.raw'),
        '--n1-weights',(Join-Path $payload 'weights.raw'),
        '--n1-params',(Join-Path $payload 'params.raw'),
        '--n1-output',$output,'--n1-sync-out',$sync,
        '--n1-output-initial',(Join-Path $payload 'output_initial.raw'),
        '--n1-sync-initial',(Join-Path $payload 'sync_initial.raw'),
        '--n1-weight-view-offset','0','--n1-release-param-offset','64',
        '--n1-grid-x','10','--n1-grid-y','6','--n1-grid-z','1',
        '--n1-block-x','32','--n1-block-y','4','--n1-block-z','1',
        '--n1-expected-releases','60'
    )
    $traceFull = Join-Path $work "$Prefix.trace_full.raw"
    if ($WithTrace) {
        $arguments += @('--n1-extra-output',$traceFull,'--n1-extra-param-offset','72')
    }
    & $probe @arguments *> (Join-Path $work "$Prefix.stdout.log")
    $exitCode = $LASTEXITCODE
    $probeData = if (Test-Path -LiteralPath $probeJson) { Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json } else { $null }
    $trace = Join-Path $result "$Prefix.trace.raw"
    $traceNonzero = 0
    $traceTailNonzero = 0
    if ($WithTrace -and (Test-Path -LiteralPath $traceFull)) {
        $all = [IO.File]::ReadAllBytes($traceFull)
        $traceBytes = [int]$package.trace_bytes
        if ($all.Length -lt $traceBytes) { throw "trace is shorter than required: $($all.Length)/$traceBytes" }
        $slice = [byte[]]::new($traceBytes)
        [Array]::Copy($all, 0, $slice, 0, $traceBytes)
        [IO.File]::WriteAllBytes($trace, $slice)
        $traceNonzero = @($slice | Where-Object { $_ -ne 0 }).Count
        if ($all.Length -gt $traceBytes) {
            $traceTailNonzero = @($all[$traceBytes..($all.Length - 1)] | Where-Object { $_ -ne 0 }).Count
        }
    }
    $outputHash = if (Test-Path -LiteralPath $output) { (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash } else { $null }
    $syncHash = if (Test-Path -LiteralPath $sync) { (Get-FileHash -LiteralPath $sync -Algorithm SHA256).Hash } else { $null }
    return [ordered]@{
        prefix=$Prefix;instrumented=$WithTrace;exit_code=$exitCode
        probe_pass=[bool]($probeData -and $probeData.pass)
        execution_verified=[bool]($probeData -and $probeData.execution_verified)
        device_name=if($probeData){$probeData.device_name}else{$null}
        output_sha256=$outputHash
        output_reference_exact=[bool]($outputHash -eq [string]$package.output_reference_sha256)
        sync_sha256=$syncHash
        sync_reference_exact=[bool]($syncHash -eq [string]$package.sync_reference_sha256)
        trace_bytes=if(Test-Path -LiteralPath $trace){(Get-Item -LiteralPath $trace).Length}else{0}
        trace_nonzero_bytes=$traceNonzero
        trace_tail_nonzero_bytes=$traceTailNonzero
        trace_sha256=if(Test-Path -LiteralPath $trace){(Get-FileHash -LiteralPath $trace -Algorithm SHA256).Hash}else{$null}
    }
}

$baseline = Invoke-Slot10 'slot10_original.ptx' 'baseline' $false
$runs = @(
    (Invoke-Slot10 'slot10_fp8_mma_trace.ptx' 'run1' $true),
    (Invoke-Slot10 'slot10_fp8_mma_trace.ptx' 'run2' $true)
)
$repeat = $runs[0].trace_sha256 -and $runs[0].trace_sha256 -eq $runs[1].trace_sha256
$all = @($baseline) + $runs
$pass = $integrity -and $repeat -and @($all | Where-Object {
    $_.exit_code -ne 0 -or -not $_.probe_pass -or -not $_.execution_verified -or
    $_.device_name -notmatch 'NVIDIA' -or -not $_.output_reference_exact -or
    -not $_.sync_reference_exact
}).Count -eq 0 -and @($runs | Where-Object {
    $_.trace_bytes -ne [int]$package.trace_bytes -or $_.trace_nonzero_bytes -le 0 -or
    $_.trace_tail_nonzero_bytes -ne 0
}).Count -eq 0

[ordered]@{
    schema=1;package_revision=[string]$package.package_revision
    experiment='rtx_swin4h_slot10_selected_warp_fp8_mma_trace'
    status=if($pass){'PASS'}else{'FAIL'}
    classification='RTX_SWIN4H_SLOT10_EXACT_INPUT_FP8_MMA_ORACLE'
    counts_as_s7=$false;payload_integrity=$integrity;payload_sha256=$actualHashes
    function=[string]$package.function;target_cta=@($package.target_cta)
    target_warp_y=[int]$package.target_warp_y;mma_count=[int]$package.mma_count
    trace_bytes=[int]$package.trace_bytes
    output_reference_sha256=[string]$package.output_reference_sha256
    sync_reference_sha256=[string]$package.sync_reference_sha256
    trace_repeat_bitwise_exact=[bool]$repeat;baseline=$baseline;runs=$runs
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8

$archive = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $archive"
exit $(if($pass){0}else{1})
