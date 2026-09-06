[CmdletBinding()]
param([string]$TraceRoot = 'results\20260902_001000_n0_cta8_0_box_muller_trace')
$ErrorActionPreference = 'Stop'; $repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only zluda_ptx_probe; if ($LASTEXITCODE -ne 0) { throw 'probe build failed' }
$traceRoot = if ([IO.Path]::IsPathRooted($TraceRoot)) { (Resolve-Path -LiteralPath $TraceRoot).Path } else { (Resolve-Path -LiteralPath (Join-Path $repo $TraceRoot)).Path }
$inst = Get-Content -Raw -LiteralPath (Join-Path $traceRoot 'instrumentation.json') | ConvertFrom-Json
$x=[int]$inst.target_cta[0];$y=[int]$inst.target_cta[1];$stamp=Get-Date -Format 'yyyyMMdd_HHmmss';$stage=Join-Path $repo "deliverables\n0_cta${x}_${y}_box_muller_trace_reference_$stamp"
$payload=Join-Path $stage 'payload';$scripts=Join-Path $stage 'scripts';New-Item -ItemType Directory -Path $payload,$scripts|Out-Null
Copy-Item (Join-Path $repo 'build\zluda_ptx_probe.exe') (Join-Path $payload 'neural_reference_probe.exe')
$source=Join-Path $traceRoot 'n0_box_trace_original.ptx';$text=[IO.File]::ReadAllText($source);$normalized=$text-replace'(?m)^\.version 9\.4$','.version 8.7';if($normalized-eq$text){throw'PTX version not found'};[IO.File]::WriteAllText((Join-Path $payload 'n0_box_trace.ptx'),$normalized,[Text.UTF8Encoding]::new($false))
$original=Join-Path $repo 'results\20260831_011219_zluda_ptx_probe\neural_isolated.ptx';$text=[IO.File]::ReadAllText($original);$normalized=$text-replace'(?m)^\.version 9\.4$','.version 8.7';[IO.File]::WriteAllText((Join-Path $payload 'n0_original.ptx'),$normalized,[Text.UTF8Encoding]::new($false))
Copy-Item (Join-Path $repo 'results\20260901_153000_n0_zero_input\input_rgba16f_zero.raw') (Join-Path $payload 'input_rgba16f.raw');$base=Join-Path $repo 'deliverables\n0_full_reference_20260831_130219\payload';Copy-Item (Join-Path $base 'weights.raw'),(Join-Path $base 'params.raw') -Destination $payload
Copy-Item (Join-Path $PSScriptRoot 'run_nvidia_n0_box_muller_cta_trace.ps1') $scripts;Copy-Item (Join-Path $repo 'N0_BOX_MULLER_CTA_TRACE_REFERENCE_README.md') (Join-Path $stage 'README.md')
$required=@('neural_reference_probe.exe','n0_box_trace.ptx','n0_original.ptx','input_rgba16f.raw','weights.raw','params.raw');$hashes=[ordered]@{};foreach($n in $required){$hashes[$n]=(Get-FileHash (Join-Path $payload $n) -Algorithm SHA256).Hash}
[ordered]@{schema=1;experiment='rtx_n0_selected_cta_box_muller_trace_package';input_variant='zero_rgba16f_full_graph';function='cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8';grid=@(80,48,1);block=@(32,1,1);target_cta=$inst.target_cta;sample_count=[int]$inst.sample_count;stage_count=[int]$inst.stage_count;stages=$inst.stages;bytes_per_sample=[int]$inst.bytes_per_sample;trace_offset=[int64]$inst.trace_offset;trace_bytes=[int]$inst.trace_bytes;scratch_extra_bytes=[int]$inst.scratch_extra_bytes;payload_sha256=$hashes}|ConvertTo-Json -Depth 8|Set-Content (Join-Path $stage 'manifest.json') -Encoding utf8
$zip="$stage.zip";Compress-Archive -LiteralPath $stage -DestinationPath $zip -CompressionLevel Optimal;Write-Host "Package: $zip";Write-Host "SHA256: $((Get-FileHash $zip -Algorithm SHA256).Hash)"
