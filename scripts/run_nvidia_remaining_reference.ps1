[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $root "_remaining_reference_result_$stamp"
New-Item -ItemType Directory -Path $result | Out-Null
$exe = Join-Path $root 'nvidia_remaining_reference.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "missing $exe" }
& $exe $result *> (Join-Path $result 'stdout.log')
$exitCode = $LASTEXITCODE
$jsonPath = Join-Path $result 'remaining_reference.json'
$run = if (Test-Path -LiteralPath $jsonPath) {
    Get-Content $jsonPath -Raw | ConvertFrom-Json
} else { $null }
$expected = [ordered]@{
    'f16_a.raw' = 4096; 'f16_b.raw' = 2048; 'f16_c.raw' = 2048
    'f16_d.raw' = 2048; 'mov_input.raw' = 1024; 'mov_output.raw' = 1024
}
$hashes = [ordered]@{}
$filesPass = $true
foreach ($name in $expected.Keys) {
    $path = Join-Path $result $name
    if (-not (Test-Path $path) -or (Get-Item $path).Length -ne $expected[$name]) {
        $filesPass = $false
    } else {
        $hashes[$name] = (Get-FileHash $path -Algorithm SHA256).Hash
    }
}
$pass = $exitCode -eq 0 -and $run -and $run.status -eq 'PASS' `
    -and $run.negative_verifier_detected -and $filesPass
[ordered]@{schema=1;package_revision='v16_rtx_remaining_n0_primitives';status=if($pass){'PASS'}else{'FAIL'};classification='RTX_NUMERICAL_ORACLE';counts_as_s6=$false;exit_code=$exitCode;executable_sha256=(Get-FileHash $exe -Algorithm SHA256).Hash;raw_sha256=$hashes}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $result 'manifest.json') -Encoding utf8
$zip = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $zip -CompressionLevel Optimal -Force
Get-Content (Join-Path $result 'stdout.log')
Get-Content (Join-Path $result 'manifest.json')
Write-Host "Result: $zip"
exit $(if ($pass) { 0 } else { 1 })
