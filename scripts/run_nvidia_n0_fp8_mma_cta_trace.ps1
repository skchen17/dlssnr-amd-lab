[CmdletBinding()]
param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'; $PSNativeCommandUseErrorActionPreference = $false
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $PSScriptRoot }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path; $payload = Join-Path $root 'payload'; $packageManifest = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'; if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }
$required = @('neural_reference_probe.exe','n0_cta1_0_fp8_mma_trace.ptx','n0_original.ptx','input_rgba16f.raw','weights.raw','params.raw'); $actualHashes = [ordered]@{}
foreach ($name in $required) { $path = Join-Path $payload $name; if (-not (Test-Path -LiteralPath $path)) { throw "Missing payload: $name" }; $actualHashes[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
$integrity = $true; foreach ($name in $required) { $integrity = $integrity -and $actualHashes[$name] -eq $packageManifest.payload_sha256.$name }
$targetX = [int]$packageManifest.target_cta[0]; $targetY = [int]$packageManifest.target_cta[1]
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'; $work = Join-Path $root "_n0_cta${targetX}_${targetY}_fp8_mma_trace_work_$stamp"; $result = Join-Path $root "_n0_cta${targetX}_${targetY}_fp8_mma_trace_reference_result_$stamp"
New-Item -ItemType Directory -Path $work,$result | Out-Null
$probeJson = Join-Path $work 'probe.json'; $scratch = Join-Path $work 'scratch.raw'; $output = Join-Path $work 'output.raw'
& (Join-Path $payload 'neural_reference_probe.exe') --nvcuda $nvcuda --ptx (Join-Path $payload 'n0_cta1_0_fp8_mma_trace.ptx') `
    --function 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8' --json $probeJson --n0-input (Join-Path $payload 'input_rgba16f.raw') `
    --n0-weights (Join-Path $payload 'weights.raw') --n0-params (Join-Path $payload 'params.raw') --n0-scratch-out $scratch --n0-output $output `
    --n0-grid-x 80 --n0-grid-y 48 --n0-scratch-extra-bytes ([int]$packageManifest.scratch_extra_bytes) *> (Join-Path $work 'stdout.log')
$probeExit = $LASTEXITCODE; $probe = if (Test-Path -LiteralPath $probeJson) { Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json } else { $null }
$baselineJson = Join-Path $work 'baseline_probe.json'; $baselineScratch = Join-Path $work 'baseline_scratch.raw'; $baselineOutput = Join-Path $work 'baseline_output.raw'
& (Join-Path $payload 'neural_reference_probe.exe') --nvcuda $nvcuda --ptx (Join-Path $payload 'n0_original.ptx') `
    --function 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8' --json $baselineJson --n0-input (Join-Path $payload 'input_rgba16f.raw') `
    --n0-weights (Join-Path $payload 'weights.raw') --n0-params (Join-Path $payload 'params.raw') --n0-scratch-out $baselineScratch --n0-output $baselineOutput `
    --n0-grid-x 80 --n0-grid-y 48 *> (Join-Path $work 'baseline_stdout.log')
$baselineExit = $LASTEXITCODE; $baselineProbe = if (Test-Path -LiteralPath $baselineJson) { Get-Content -Raw -LiteralPath $baselineJson | ConvertFrom-Json } else { $null }
$trace = Join-Path $result 'mma_trace.raw'; $traceBytes = [int]$packageManifest.trace_bytes; $traceOffset = [int64]$packageManifest.trace_offset
if (Test-Path -LiteralPath $scratch) { $stream = [IO.File]::OpenRead($scratch); try { [void]$stream.Seek($traceOffset,[IO.SeekOrigin]::Begin); $buffer=[byte[]]::new($traceBytes);$read=0;while($read-lt$traceBytes){$count=$stream.Read($buffer,$read,$traceBytes-$read);if($count-eq 0){break};$read+=$count};if($read-ne$traceBytes){throw "Trace extraction short read: $read/$traceBytes"};[IO.File]::WriteAllBytes($trace,$buffer)}finally{$stream.Dispose()} }
$nonzero = if(Test-Path -LiteralPath $trace){@([IO.File]::ReadAllBytes($trace)|Where-Object{$_-ne 0}).Count}else{0}; $outputHash=if(Test-Path -LiteralPath $output){(Get-FileHash $output -Algorithm SHA256).Hash}else{$null};$baselineHash=if(Test-Path -LiteralPath $baselineOutput){(Get-FileHash $baselineOutput -Algorithm SHA256).Hash}else{$null};$preserved=$outputHash-and$outputHash-eq$baselineHash
$pass=$integrity-and$probeExit-eq 0-and$probe-and$probe.pass-and$probe.execution_verified-and$baselineExit-eq 0-and$baselineProbe-and$baselineProbe.pass-and$baselineProbe.execution_verified-and(Test-Path $trace)-and(Get-Item $trace).Length-eq$traceBytes-and$nonzero-gt 0
Copy-Item $probeJson,(Join-Path $work 'stdout.log'),$baselineJson,(Join-Path $work 'baseline_stdout.log') -Destination $result
[ordered]@{schema=1;experiment='rtx_n0_selected_cta_fp8_mma_trace';status=if($pass){'PASS'}else{'FAIL'};classification='RTX_N0_FULL_GRID_FIRST_MISMATCH_CTA_ORACLE';counts_as_s7=$false;payload_integrity=$integrity;payload_sha256=$actualHashes;probe_exit=$probeExit;probe_pass=[bool]($probe-and$probe.pass);device_name=if($probe){$probe.device_name}else{$null};baseline_probe_exit=$baselineExit;baseline_probe_pass=[bool]($baselineProbe-and$baselineProbe.pass);grid=@(80,48,1);block=@(32,1,1);target_cta=$packageManifest.target_cta;kernel_launched=[bool]($probe-and$probe.kernel_launched);mma_count=[int]$packageManifest.mma_count;checkpoint_bytes=$traceBytes;checkpoint_nonzero_bytes=$nonzero;checkpoint_sha256=if(Test-Path $trace){(Get-FileHash $trace -Algorithm SHA256).Hash}else{$null};input_variant=$packageManifest.input_variant;scratch_extra_bytes=[int]$packageManifest.scratch_extra_bytes;output_sha256=$outputHash;baseline_output_sha256=$baselineHash;reference_output_preserved=[bool]$preserved;instrumentation_perturbed=[bool](-not$preserved)}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$archive="$result.zip";Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal;Get-Content -Raw (Join-Path $result 'manifest.json');Write-Host "Result: $archive";exit $(if($pass){0}else{1})
