$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repo 'results\20260831_012648_rtx5070_feature18_descriptors'
$trace = Join-Path $source 'module_trace.jsonl'
$summary = Join-Path $source 'summary.json'
if (-not (Test-Path -LiteralPath $trace) -or -not (Test-Path -LiteralPath $summary)) {
    throw "v10 descriptor trace result is missing under $source"
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_descriptor_trace_join"
New-Item -ItemType Directory -Path $result | Out-Null
$output = Join-Path $result 'descriptor_trace_join.json'
$analyzer = Join-Path $PSScriptRoot 'analyze_descriptor_trace.py'
& python $analyzer --trace $trace --summary $summary --output $output
$code = $LASTEXITCODE

$archive = 'C:\DATA\Python_File\Qoder_workspace\dlssnr_feature18_result_20260831_012648.zip'
$manifest = [ordered]@{
    schema = 1
    experiment = 'descriptor_trace_join'
    classification = 'RTX_OBSERVATIONAL_ABI_EVIDENCE'
    counts_as_s6 = $false
    exit_code = $code
    returned_archive_sha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    trace_sha256 = (Get-FileHash -LiteralPath $trace -Algorithm SHA256).Hash
    summary_sha256 = (Get-FileHash -LiteralPath $summary -Algorithm SHA256).Hash
    analyzer_sha256 = (Get-FileHash -LiteralPath $analyzer -Algorithm SHA256).Hash
    output_sha256 = if (Test-Path -LiteralPath $output) {
        (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
    } else { $null }
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Write-Host "Result: $result"
exit $code
