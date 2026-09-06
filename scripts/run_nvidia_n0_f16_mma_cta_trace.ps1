[CmdletBinding()]
param([string]$PackageRoot)
$ErrorActionPreference = 'Stop'; $PSNativeCommandUseErrorActionPreference = $false
if ([string]::IsNullOrWhiteSpace($PackageRoot)) { $PackageRoot = Split-Path -Parent $PSScriptRoot }
$root = (Resolve-Path -LiteralPath $PackageRoot).Path; $payload = Join-Path $root 'payload'; $pm = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'; if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }
$required = @('neural_reference_probe.exe','n0_f16_mma_trace.ptx','n0_original.ptx','input_rgba16f.raw','weights.raw','params.raw'); $actual = [ordered]@{}
foreach ($name in $required) { $path = Join-Path $payload $name; if (-not (Test-Path -LiteralPath $path)) { throw "Missing payload: $name" }; $actual[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
$integrity = $true; foreach ($name in $required) { $integrity = $integrity -and $actual[$name] -eq $pm.payload_sha256.$name }
$targetX = [int]$pm.target_cta[0]; $targetY = [int]$pm.target_cta[1]; $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$work = Join-Path $root "_n0_cta${targetX}_${targetY}_f16_mma_trace_work_$stamp"; $result = Join-Path $root "_n0_cta${targetX}_${targetY}_f16_mma_trace_reference_result_$stamp"
New-Item -ItemType Directory -Path $work,$result | Out-Null
$probeJson = Join-Path $work 'probe.json'; $scratch = Join-Path $work 'scratch.raw'; $output = Join-Path $work 'output.raw'
& (Join-Path $payload 'neural_reference_probe.exe') --nvcuda $nvcuda --ptx (Join-Path $payload 'n0_f16_mma_trace.ptx') `
    --function $pm.function --json $probeJson --n0-input (Join-Path $payload 'input_rgba16f.raw') --n0-weights (Join-Path $payload 'weights.raw') `
    --n0-params (Join-Path $payload 'params.raw') --n0-scratch-out $scratch --n0-output $output --n0-grid-x 80 --n0-grid-y 48 `
    --n0-scratch-extra-bytes ([int]$pm.scratch_extra_bytes) *> (Join-Path $work 'stdout.log')
$probeExit = $LASTEXITCODE; $probe = if (Test-Path $probeJson) { Get-Content -Raw $probeJson | ConvertFrom-Json } else { $null }
$baseJson = Join-Path $work 'baseline_probe.json'; $baseScratch = Join-Path $work 'baseline_scratch.raw'; $baseOutput = Join-Path $work 'baseline_output.raw'
& (Join-Path $payload 'neural_reference_probe.exe') --nvcuda $nvcuda --ptx (Join-Path $payload 'n0_original.ptx') `
    --function $pm.function --json $baseJson --n0-input (Join-Path $payload 'input_rgba16f.raw') --n0-weights (Join-Path $payload 'weights.raw') `
    --n0-params (Join-Path $payload 'params.raw') --n0-scratch-out $baseScratch --n0-output $baseOutput --n0-grid-x 80 --n0-grid-y 48 *> (Join-Path $work 'baseline_stdout.log')
$baseExit = $LASTEXITCODE; $baseProbe = if (Test-Path $baseJson) { Get-Content -Raw $baseJson | ConvertFrom-Json } else { $null }
$trace = Join-Path $result 'mma_trace.raw'; $stream = [IO.File]::OpenRead($scratch)
try { [void]$stream.Seek([int64]$pm.trace_offset,[IO.SeekOrigin]::Begin); $buffer=[byte[]]::new([int]$pm.trace_bytes); $read=0; while($read-lt$buffer.Length){$count=$stream.Read($buffer,$read,$buffer.Length-$read);if($count-eq 0){break};$read+=$count};if($read-ne$buffer.Length){throw "Trace extraction short read: $read/$($buffer.Length)"};[IO.File]::WriteAllBytes($trace,$buffer) } finally { $stream.Dispose() }
$nonzero=@($buffer|Where-Object{$_-ne 0}).Count; $oh=(Get-FileHash $output -Algorithm SHA256).Hash; $bh=(Get-FileHash $baseOutput -Algorithm SHA256).Hash; $preserved=$oh-eq$bh
$pass=$integrity-and$probeExit-eq 0-and$probe-and$probe.pass-and$probe.execution_verified-and$baseExit-eq 0-and$baseProbe-and$baseProbe.pass-and$baseProbe.execution_verified-and$preserved-and$nonzero-gt 0
Copy-Item $probeJson,(Join-Path $work 'stdout.log'),$baseJson,(Join-Path $work 'baseline_stdout.log') -Destination $result
[ordered]@{schema=1;experiment='rtx_n0_selected_cta_f16_mma_trace';status=if($pass){'PASS'}else{'FAIL'};classification='RTX_N0_FIRST_OBSERVABLE_FP8_INPUT_PRODUCER_ORACLE';counts_as_s7=$false;payload_integrity=$integrity;payload_sha256=$actual;probe_exit=$probeExit;probe_pass=[bool]($probe-and$probe.pass);device_name=if($probe){$probe.device_name}else{$null};baseline_probe_exit=$baseExit;baseline_probe_pass=[bool]($baseProbe-and$baseProbe.pass);grid=@(80,48,1);block=@(32,1,1);target_cta=$pm.target_cta;kernel_launched=[bool]($probe-and$probe.kernel_launched);mma_count=[int]$pm.mma_count;checkpoint_bytes=[int]$pm.trace_bytes;checkpoint_nonzero_bytes=$nonzero;checkpoint_sha256=(Get-FileHash $trace -Algorithm SHA256).Hash;input_variant=$pm.input_variant;scratch_extra_bytes=[int]$pm.scratch_extra_bytes;output_sha256=$oh;baseline_output_sha256=$bh;reference_output_preserved=$preserved;instrumentation_perturbed=(-not$preserved)} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$archive="$result.zip";Compress-Archive -LiteralPath $result -DestinationPath $archive -CompressionLevel Optimal;Get-Content -Raw (Join-Path $result 'manifest.json');Write-Host "Result: $archive";exit $(if($pass){0}else{1})
