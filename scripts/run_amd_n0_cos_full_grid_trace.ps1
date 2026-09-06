[CmdletBinding()]
param([string]$TraceRoot='results\20260904_010000_n0_cos_full_grid_trace')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$root=if([IO.Path]::IsPathRooted($TraceRoot)){(Resolve-Path $TraceRoot).Path}else{(Resolve-Path (Join-Path $repo $TraceRoot)).Path}
$meta=Get-Content -Raw (Join-Path $root 'instrumentation_amd.json')|ConvertFrom-Json
$probe=Join-Path $repo 'build\zluda_ptx_probe.exe'
$nvcuda=Join-Path $repo '.tools\zluda-v7-preview.3\zluda\nvcuda.dll'
$payload=Join-Path $repo 'deliverables\n0_full_reference_20260831_130219\payload'
$args=@('--nvcuda',$nvcuda,'--ptx',(Join-Path $root 'n0_cos_trace_amd.ptx'),'--function','cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8','--json',(Join-Path $root 'amd_probe.json'),'--n0-input',(Join-Path $repo 'results\20260901_153000_n0_zero_input\input_rgba16f_zero.raw'),'--n0-weights',(Join-Path $payload 'weights.raw'),'--n0-params',(Join-Path $payload 'params.raw'),'--n0-scratch-out',(Join-Path $root 'amd_scratch.raw'),'--n0-output',(Join-Path $root 'amd_output.raw'),'--n0-grid-x','80','--n0-grid-y','48','--n0-scratch-extra-bytes',[string]$meta.scratch_extra_bytes)
& $probe @args
if($LASTEXITCODE-ne 0){throw"AMD probe failed: $LASTEXITCODE"}
$scratch=Join-Path $root 'amd_scratch.raw';$trace=Join-Path $root 'amd_cos_trace.raw';$stream=[IO.File]::OpenRead($scratch)
try{[void]$stream.Seek([int64]$meta.trace_offset,[IO.SeekOrigin]::Begin);$buf=[byte[]]::new([int]$meta.trace_bytes);$read=0;while($read-lt$buf.Length){$n=$stream.Read($buf,$read,$buf.Length-$read);if($n-eq 0){break};$read+=$n};if($read-ne$buf.Length){throw"short trace: $read"};[IO.File]::WriteAllBytes($trace,$buf)}finally{$stream.Dispose()}
$p=Get-Content -Raw (Join-Path $root 'amd_probe.json')|ConvertFrom-Json;$expected=(Get-FileHash (Join-Path $repo 'results\20260902_150000_n0_square_pair_fusion_candidate\output.raw')).Hash;$actual=(Get-FileHash (Join-Path $root 'amd_output.raw')).Hash;$pass=$p.pass-and$p.execution_verified-and$actual-eq$expected
[ordered]@{schema=1;experiment='rx9070xt_n0_cos_full_grid_trace';status=if($pass){'PASS'}else{'FAIL'};device_name=$p.device_name;sample_count=[int]$meta.sample_count;trace_bytes=[int]$meta.trace_bytes;trace_sha256=(Get-FileHash $trace).Hash;output_sha256=$actual;baseline_output_sha256=$expected;output_preserved=$actual-eq$expected}|ConvertTo-Json -Depth 6|Set-Content (Join-Path $root 'amd_manifest.json') -Encoding utf8
Get-Content -Raw (Join-Path $root 'amd_manifest.json');exit $(if($pass){0}else{1})
