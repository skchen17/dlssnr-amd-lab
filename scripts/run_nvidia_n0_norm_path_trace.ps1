[CmdletBinding()]
param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $PSScriptRoot }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$packageManifest = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }
$required = @('neural_reference_probe.exe','n0_norm_path_trace.ptx','n0_original.ptx','input_rgba16f.raw','weights.raw','params.raw')
$actualHashes = [ordered]@{}
foreach ($name in $required) {
    $path = Join-Path $payload $name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing payload: $name" }
    $actualHashes[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}
$integrity = $true
foreach ($name in $required) { $integrity = $integrity -and $actualHashes[$name] -eq $packageManifest.payload_sha256.$name }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_n0_norm_path_trace_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$probeJson = Join-Path $result 'probe.json'; $scratch = Join-Path $result 'scratch.raw'; $output = Join-Path $result 'output.raw'
& (Join-Path $payload 'neural_reference_probe.exe') --nvcuda $nvcuda `
    --ptx (Join-Path $payload 'n0_norm_path_trace.ptx') `
    --function 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8' --json $probeJson `
    --n0-input (Join-Path $payload 'input_rgba16f.raw') --n0-weights (Join-Path $payload 'weights.raw') `
    --n0-params (Join-Path $payload 'params.raw') --n0-scratch-out $scratch --n0-output $output `
    --n0-grid-x 1 --n0-grid-y 1 *> (Join-Path $result 'stdout.log')
$probeExit = $LASTEXITCODE
$probe = if (Test-Path -LiteralPath $probeJson) { Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json } else { $null }
$baselineJson = Join-Path $result 'baseline_probe.json'; $baselineScratch = Join-Path $result 'baseline_scratch.raw'; $baselineOutput = Join-Path $result 'baseline_output.raw'
& (Join-Path $payload 'neural_reference_probe.exe') --nvcuda $nvcuda `
    --ptx (Join-Path $payload 'n0_original.ptx') `
    --function 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8' --json $baselineJson `
    --n0-input (Join-Path $payload 'input_rgba16f.raw') --n0-weights (Join-Path $payload 'weights.raw') `
    --n0-params (Join-Path $payload 'params.raw') --n0-scratch-out $baselineScratch --n0-output $baselineOutput `
    --n0-grid-x 1 --n0-grid-y 1 *> (Join-Path $result 'baseline_stdout.log')
$baselineExit = $LASTEXITCODE
$baselineProbe = if (Test-Path -LiteralPath $baselineJson) { Get-Content -Raw -LiteralPath $baselineJson | ConvertFrom-Json } else { $null }
$trace = Join-Path $result 'norm_path_trace.raw'; $traceBytes = [int]$packageManifest.trace_bytes; $traceOffset = [int64]$packageManifest.trace_offset
if (Test-Path -LiteralPath $scratch) {
    $stream = [IO.File]::OpenRead($scratch)
    try {
        [void]$stream.Seek($traceOffset, [IO.SeekOrigin]::Begin); $buffer = [byte[]]::new($traceBytes); $read = 0
        while ($read -lt $traceBytes) { $count = $stream.Read($buffer, $read, $traceBytes - $read); if ($count -eq 0) { break }; $read += $count }
        if ($read -ne $traceBytes) { throw "Trace extraction short read: $read/$traceBytes" }
        [IO.File]::WriteAllBytes($trace, $buffer)
    } finally { $stream.Dispose() }
}
$nonzero = if (Test-Path -LiteralPath $trace) { @([IO.File]::ReadAllBytes($trace) | Where-Object { $_ -ne 0 }).Count } else { 0 }
$outputHash = if(Test-Path -LiteralPath $output){(Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash}else{$null}
$baselineHash = if(Test-Path -LiteralPath $baselineOutput){(Get-FileHash -LiteralPath $baselineOutput -Algorithm SHA256).Hash}else{$null}
$preserved = $outputHash -and $outputHash -eq $baselineHash
$pass = $integrity -and $probeExit -eq 0 -and $probe -and $probe.pass -and $probe.execution_verified -and `
    $baselineExit -eq 0 -and $baselineProbe -and $baselineProbe.pass -and $baselineProbe.execution_verified -and `
    (Test-Path -LiteralPath $trace) -and (Get-Item -LiteralPath $trace).Length -eq $traceBytes -and $nonzero -gt 0
[ordered]@{
    schema=1;experiment='rtx_n0_normalization_path_trace';status=if($pass){'PASS'}else{'FAIL'}
    classification='RTX_N0_FIRST_DIVERGENT_ACTIVATION_PATH_ORACLE';counts_as_s7=$false
    payload_integrity=$integrity;payload_sha256=$actualHashes;probe_exit=$probeExit;probe_pass=[bool]($probe -and $probe.pass)
    device_name=if($probe){$probe.device_name}else{$null};baseline_probe_exit=$baselineExit;baseline_probe_pass=[bool]($baselineProbe -and $baselineProbe.pass)
    grid=@(1,1,1);block=@(32,1,1);kernel_launched=[bool]($probe -and $probe.kernel_launched)
    stage_count=[int]$packageManifest.stage_count;stages=$packageManifest.stages;checkpoint_bytes=$traceBytes;checkpoint_nonzero_bytes=$nonzero
    input_variant=$packageManifest.input_variant
    checkpoint_sha256=if(Test-Path -LiteralPath $trace){(Get-FileHash -LiteralPath $trace -Algorithm SHA256).Hash}else{$null}
    output_sha256=$outputHash;baseline_output_sha256=$baselineHash;reference_output_preserved=[bool]$preserved;instrumentation_perturbed=[bool](-not $preserved)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$archive = "$result.zip"; Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json'); Write-Host "Result: $archive"
exit $(if($pass){0}else{1})
