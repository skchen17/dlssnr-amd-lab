param([Parameter(Mandatory)][string]$Executable,[Parameter(Mandatory)][string]$InputFile,[Parameter(Mandatory)][string]$InputSha256,
      [Parameter(Mandatory)][ValidateSet(1920,2560,3840)][int]$Width,[Parameter(Mandatory)][string]$OutputDirectory,
      [ValidateRange(1,12)][int]$Iterations=3,[string]$GameDirectory='C:\DATA\GAME\GODOFWAR')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dll=(Resolve-Path -LiteralPath (Join-Path $GameDirectory 'amd_fidelityfx_dx12.dll')).Path
$audit=Get-Content -LiteralPath (Join-Path $repo 'results/20260905_gowr_target_audit/manifest.json') -Raw | ConvertFrom-Json
$expected=@($audit.inventory | Where-Object name -eq 'amd_fidelityfx_dx12.dll')
if($expected.Count -ne 1 -or (Get-FileHash -LiteralPath $dll).Hash -ne $expected[0].sha256){throw 'Provider static audit mismatch'}
$signature=Get-AuthenticodeSignature -LiteralPath $dll
if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Advanced Micro Devices'){throw 'AMD signature required'}
$inputPath=(Resolve-Path -LiteralPath $InputFile).Path
if((Get-FileHash -LiteralPath $inputPath).Hash -ne $InputSha256){throw 'Input hash mismatch'}
$exe=(Resolve-Path -LiteralPath $Executable).Path
$exeHash=(Get-FileHash -LiteralPath $exe).Hash
$height=@{1920=1080;2560=1440;3840=2160}[$Width]
if((Get-Item -LiteralPath $inputPath).Length -ne [long]$Width*$height*8){throw 'RGBA16F input length mismatch'}
$out=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $out){throw 'New output directory required'}
if(Get-Process GoWR -ErrorAction SilentlyContinue){throw 'Game must be closed before this isolated test'}
$supervision=$out+'.supervision'
if(Test-Path -LiteralPath $supervision){throw 'New supervision directory required'}
New-Item -ItemType Directory -Path $supervision | Out-Null
# Paths cannot contain quotes on Windows; reject control characters rather than
# passing ambiguous Start-Process command-line quoting to the child.
foreach($argument in @($exe,$dll,$inputPath,$out)){
    if($argument -match '["\r\n]'){throw 'Unsafe command-line path'}
}
$childArgs=@(('"'+$dll+'"'),('"'+$inputPath+'"'),[string]$Width,[string]$height,('"'+$out+'"'),[string]$Iterations)
$child=Start-Process -FilePath $exe -ArgumentList $childArgs -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $supervision 'stdout.log') -RedirectStandardError (Join-Path $supervision 'stderr.log')
$timer=[Diagnostics.Stopwatch]::StartNew()
$finished=$false
while($timer.Elapsed.TotalSeconds -lt 90){
    if($child.WaitForExit(1000)){$finished=$true;break}
}
if(-not $finished){
    @{status='TIMEOUT_CHILD_RETAINED_NO_KILL';pid=$child.Id;elapsed_seconds=$timer.Elapsed.TotalSeconds;manual_disposition_required=$true;
      input_sha256=$InputSha256;executable_sha256=$exeHash;provider_sha256=$expected[0].sha256} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $supervision 'supervision.json') -Encoding utf8
    throw "Timeout; PID $($child.Id) remains alive and may retain GPU resources. Stop further GPU work; inspect logs/manual disposition."
}
$child.WaitForExit()
$code=$child.ExitCode
@{status='EXITED';pid=$child.Id;exit_code=$code;elapsed_seconds=$timer.Elapsed.TotalSeconds} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $supervision 'supervision.json') -Encoding utf8
if(Test-Path -LiteralPath $out){
    $record=@{scope='SYNTHETIC_STATIC_ENGINEERING_NOT_GAME_QUALITY';provider_sha256=$expected[0].sha256;signature=[string]$signature.Status;input_sha256=$InputSha256;executable_sha256=$exeHash;exit_code=$code;game_launched=$false}
    if(Test-Path -LiteralPath (Join-Path $out 'output.rgba16f')){$record.output_sha256=(Get-FileHash -LiteralPath (Join-Path $out 'output.rgba16f')).Hash}
    $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'provenance.json') -Encoding utf8
}
if((Get-FileHash -LiteralPath $dll).Hash -ne $expected[0].sha256 -or (Get-FileHash -LiteralPath $inputPath).Hash -ne $InputSha256 -or (Get-FileHash -LiteralPath $exe).Hash -ne $exeHash){throw 'Source changed during test'}
if($code -ne 0){throw "Probe failed: $code; do not retry automatically"}
