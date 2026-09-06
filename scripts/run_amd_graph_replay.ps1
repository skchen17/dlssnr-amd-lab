# Runs the first AMD-side transport/scheduler experiment from the complete RTX
# frame graph. This uses a lab marker kernel and explicitly does not count as S6.
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\amd_graph_replay.exe'
$graph = Join-Path $repo 'results\20260831_002356_rtx5070_feature18_full_frame'
if (-not (Test-Path -LiteralPath $exe)) {
    & (Join-Path $PSScriptRoot 'build_all.ps1') -Only amd_graph_replay
}
foreach ($name in @('module_map.csv','function_map.csv','frame_001_sequence.csv')) {
    if (-not (Test-Path -LiteralPath (Join-Path $graph $name))) {
        throw "Captured graph input missing: $name"
    }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_amd_graph_replay"
New-Item -ItemType Directory -Force -Path $result | Out-Null
$json = Join-Path $result 'amd_graph_replay.json'
$log = Join-Path $result 'stdout.log'
& $exe --graph-dir $graph --json $json 2>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
$inputs = @('module_map.csv','function_map.csv','frame_001_sequence.csv') | ForEach-Object {
    $path = Join-Path $graph $_
    [ordered]@{ file = $_; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash }
}
$manifest = [ordered]@{
    schema = 1
    experiment = 'amd_graph_replay'
    classification = 'LAB_TRANSPORT_ONLY'
    counts_as_s6 = $false
    executable_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash
    source_graph = $graph
    source_inputs = $inputs
    exit_code = $code
    result_json = 'amd_graph_replay.json'
    stdout_log = 'stdout.log'
}
$manifest | ConvertTo-Json -Depth 4 | Out-File -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
Write-Host "AMD graph replay exit=$code json=$json"
exit $code
