[CmdletBinding()]
param([Parameter(Mandatory)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputPath) { throw 'Audit output already exists' }
$repo = Split-Path -Parent $PSScriptRoot
$baseline = Get-Content (Join-Path $repo 'results\20260905_rocm_native_baseline\frozen_full.json') -Raw | ConvertFrom-Json
$assets = @($baseline.assets | ForEach-Object {
    $actual = (Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash
    [ordered]@{path=$_.path; unchanged=($actual -eq $_.sha256); sha256=$actual}
})
$gameHashes = @{
    'GoWR.exe'='51112CF1A7683BC14C51D10295C2D468C51B2E19460B1C118030BBDC27BA63FA'
    'amd_fidelityfx_dx12.dll'='77809405A0FF464B63654F1264F0EC0FCF8F243DAC7C15B5F5C032615520D143'
    'sl.interposer.dll'='B683FF87BB3293FE9EF679D88E7F0D0071E5831C3E178DA8FD4784B2E6C523C8'
    'version.dll'='ED56DB15C7DB2B1E73E8DDE71FED35E8BEBD418B6FF615358173B320D14B0D4E'
}
$games = @($gameHashes.GetEnumerator() | ForEach-Object {
    $file = Join-Path 'C:\DATA\GAME\GODOFWAR' $_.Key
    $actual = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
    [ordered]@{name=$_.Key; sha256=$actual; unchanged=($actual -eq $_.Value)}
})
$eventErrors = @()
$events = @(Get-WinEvent -FilterHashtable @{LogName='System'; Id=4101,41,6008; StartTime=(Get-Date).AddHours(-1)} -ErrorAction SilentlyContinue -ErrorVariable eventErrors | Select-Object TimeCreated,Id,ProviderName)
$unexpectedReadErrors = @($eventErrors | Where-Object { $_.FullyQualifiedErrorId -notmatch 'NoMatchingEventsFound' })
$running = @(Get-Process GoWR -ErrorAction SilentlyContinue)
$report = [ordered]@{
    schema=1; timestamp_utc=(Get-Date).ToUniversalTime().ToString('o')
    baseline_unchanged=(@($assets | Where-Object { -not $_.unchanged }).Count -eq 0)
    game_files_unchanged=(@($games | Where-Object { -not $_.unchanged }).Count -eq 0)
    game_running=($running.Count -gt 0)
    last_hour_system_events=$events; event_log_read_ok=($unexpectedReadErrors.Count -eq 0)
    assets=$assets; game_files=$games; training_started=$false; native_graph_complete=$false
}
$parent = Split-Path -Parent ([IO.Path]::GetFullPath($OutputPath))
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding UTF8
[pscustomobject]$report | Select-Object baseline_unchanged,game_files_unchanged,game_running,event_log_read_ok,training_started,native_graph_complete | ConvertTo-Json
if (-not $report.baseline_unchanged -or -not $report.game_files_unchanged -or -not $report.event_log_read_ok -or $events.Count -gt 0) { throw 'Audit requires review; do not continue GPU experiments' }
