[CmdletBinding()]
param([string]$TraceRoot = 'results\20260902_160000_n0_cta7_1_residual_bundle')
$ErrorActionPreference='Stop';$repo=Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe;if($LASTEXITCODE-ne 0){throw'probe build failed'}
$traceRoot=if([IO.Path]::IsPathRooted($TraceRoot)){(Resolve-Path -LiteralPath $TraceRoot).Path}else{(Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path}
$box=Get-Content -Raw -LiteralPath (Join-Path $traceRoot 'box_outputs_instrumentation.json')|ConvertFrom-Json
$mma=Get-Content -Raw -LiteralPath (Join-Path $traceRoot 'mma_instrumentation.json')|ConvertFrom-Json
if(($box.target_cta -join ',')-ne($mma.target_cta -join ',')){throw'trace target mismatch'}
$x=[int]$box.target_cta[0];$y=[int]$box.target_cta[1];$stamp=Get-Date -Format yyyyMMdd_HHmmss
$stage=Join-Path $repo "deliverables\n0_cta${x}_${y}_residual_bundle_reference_$stamp";$payload=Join-Path $stage payload;$scripts=Join-Path $stage scripts
New-Item -ItemType Directory -Path $payload,$scripts|Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\zluda_ptx_probe.exe') -Destination (Join-Path $payload 'neural_reference_probe.exe')
foreach($item in @(@('n0_box_outputs_trace_original.ptx','n0_box_outputs_trace.ptx'),@('n0_mma_trace_original.ptx','n0_mma_trace.ptx'))){$text=[IO.File]::ReadAllText((Join-Path $traceRoot $item[0]));$normalized=$text-replace'(?m)^\.version 9\.4$','.version 8.7';if($normalized-eq$text){throw"PTX version not found: $($item[0])"};[IO.File]::WriteAllText((Join-Path $payload $item[1]),$normalized,[Text.UTF8Encoding]::new($false))}
$original=Join-Path $repo 'results\20260831_011219_zluda_ptx_probe\neural_isolated.ptx';$text=[IO.File]::ReadAllText($original);$normalized=$text-replace'(?m)^\.version 9\.4$','.version 8.7';[IO.File]::WriteAllText((Join-Path $payload 'n0_original.ptx'),$normalized,[Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath (Join-Path $repo 'results\20260901_153000_n0_zero_input\input_rgba16f_zero.raw') -Destination (Join-Path $payload 'input_rgba16f.raw')
$base=Join-Path $repo 'deliverables\n0_full_reference_20260831_130219\payload';foreach($name in @('weights.raw','params.raw')){Copy-Item -LiteralPath (Join-Path $base $name) -Destination (Join-Path $payload $name)}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'run_nvidia_n0_cta_residual_bundle.ps1') -Destination $scripts
Copy-Item -LiteralPath (Join-Path $repo 'N0_CTA_RESIDUAL_BUNDLE_REFERENCE_README.md') -Destination (Join-Path $stage 'README.md')
$required=@('neural_reference_probe.exe','n0_box_outputs_trace.ptx','n0_mma_trace.ptx','n0_original.ptx','input_rgba16f.raw','weights.raw','params.raw');$hashes=[ordered]@{};foreach($name in $required){$hashes[$name]=(Get-FileHash -LiteralPath (Join-Path $payload $name) -Algorithm SHA256).Hash}
[ordered]@{schema=1;experiment='rtx_n0_selected_cta_residual_bundle_package';input_variant='zero_rgba16f_full_graph';function='cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8';grid=@(80,48,1);block=@(32,1,1);target_cta=$box.target_cta;box_outputs=[ordered]@{stage_count=[int]$box.stage_count;stages=$box.stages;sample_count=[int]$box.sample_count;trace_offset=[int64]$box.trace_offset;trace_bytes=[int]$box.trace_bytes;scratch_extra_bytes=[int]$box.scratch_extra_bytes};mma=[ordered]@{mma_count=[int]$mma.mma_count;trace_offset=[int64]$mma.trace_offset;trace_bytes=[int]$mma.trace_bytes;scratch_extra_bytes=[int]$mma.scratch_extra_bytes};payload_sha256=$hashes}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $stage manifest.json) -Encoding utf8
$zip="$stage.zip";Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal;Write-Host "Package: $zip";Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
