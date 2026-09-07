param(
    [Parameter(Mandatory=$true)][string]$OutputName,
    [switch]$Instrument,
    [ValidateRange(1,12)][int]$Samples=1,
    [ValidateRange(0,5)][int]$Warmup=0
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$started=Get-Date
$running=Get-Process | Where-Object {$_.ProcessName -match '^(python|pythonw|GoWR|RadeonGPUProfiler|rocprof)$'}
if($running){throw 'Another possible GPU workload is running; review before measuring'}
$record=Get-Content -Raw results/20260907_71_record_elide_post_publish_gate_v1/process.json | ConvertFrom-Json
$commandArgs=[System.Collections.Generic.List[string]]::new()
foreach($arg in $record.command[1..($record.command.Count-1)]){
    if($arg -ne '--child'){$commandArgs.Add($arg)}
}
$commandArgs[$commandArgs.IndexOf('--dll')+1]=Join-Path $repo 'results/20260907_native_nr_plan_v90_operation_timing/nr_plan.dll'
$out=Join-Path $repo ('results/'+$OutputName)
if(Test-Path -LiteralPath $out){throw 'Output must be new'}
$commandArgs[$commandArgs.IndexOf('--output')+1]=$out
if($Instrument){$commandArgs.Add('--operation-timing')}
$commandArgs.Add('--samples');$commandArgs.Add([string]$Samples)
$commandArgs.Add('--warmup');$commandArgs.Add([string]$Warmup)
& $record.command[0] @commandArgs
$exitCode=$LASTEXITCODE
$events=@(Get-WinEvent -FilterHashtable @{LogName='System';Id=41,6008;StartTime=$started} -ErrorAction SilentlyContinue)
$reports=@(Get-ChildItem -LiteralPath C:\Windows\LiveKernelReports\WATCHDOG -File -ErrorAction SilentlyContinue | Where-Object {$_.LastWriteTime -ge $started})
$wer=@(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1001;StartTime=$started} -ErrorAction SilentlyContinue | Where-Object {$_.Message -match 'LiveKernelEvent|141'})
$health=[ordered]@{started=$started.ToString('o');finished=(Get-Date).ToString('o');exit_code=$exitCode;new_system_events=$events.Count;new_watchdog_reports=$reports.Count;new_wer_events=$wer.Count;automatic_retry=$false;rgp_used=$false}
$health | ConvertTo-Json | Out-File -LiteralPath (Join-Path $out 'health.json') -Encoding utf8
if($exitCode -ne 0 -or $events.Count -or $reports.Count -or $wer.Count){throw 'Measurement failed or new device incident: stop GPU testing; no retry'}
