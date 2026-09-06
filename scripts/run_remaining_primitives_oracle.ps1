[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$ReferenceDirectory)
$ErrorActionPreference='Stop';$PSNativeCommandUseErrorActionPreference=$false
$repo=Split-Path -Parent $PSScriptRoot;$ref=(Resolve-Path -LiteralPath $ReferenceDirectory).Path
$names='f16_a.raw','f16_b.raw','f16_c.raw','f16_d.raw','mov_input.raw','mov_output.raw'
$p=@{};foreach($n in $names){$p[$n]=Join-Path $ref $n;if(-not(Test-Path $p[$n])){throw "missing $($p[$n])"}}
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only remaining_primitives_probe;if($LASTEXITCODE-ne 0){throw 'build failed'}
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss';$result=Join-Path $repo "results\${stamp}_amd_remaining_n0_primitives_oracle";New-Item -ItemType Directory $result|Out-Null
$exe=Join-Path $repo 'build\remaining_primitives_probe.exe';$json=Join-Path $result 'comparison.json'
& $exe $p['f16_a.raw'] $p['f16_b.raw'] $p['f16_c.raw'] $p['f16_d.raw'] $p['mov_input.raw'] $p['mov_output.raw'] $json *> (Join-Path $result 'stdout.log');$exitCode=$LASTEXITCODE
$run=if(Test-Path $json){Get-Content $json -Raw|ConvertFrom-Json}else{$null}
$pass=$exitCode-eq 0-and$run-and$run.status-eq'PASS'-and$run.f16_stable_mismatches-eq 0-and$run.movmatrix_mismatches-eq 0-and$run.negative_verifier_detected
$hashes=[ordered]@{};foreach($n in $names){$hashes[$n]=(Get-FileHash $p[$n] -Algorithm SHA256).Hash}
[ordered]@{schema=1;experiment='amd_remaining_n0_primitives_rtx_oracle';status=if($pass){'PASS'}else{'FAIL'};classification='NUMERICAL_PRIMITIVE';counts_as_s6=$false;exit_code=$exitCode;reference_directory=$ref;reference_sha256=$hashes;executable_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $result 'manifest.json') -Encoding utf8
Get-Content (Join-Path $result 'stdout.log');Get-Content (Join-Path $result 'manifest.json');Write-Host "Result: $result";exit $(if($pass){0}else{1})
