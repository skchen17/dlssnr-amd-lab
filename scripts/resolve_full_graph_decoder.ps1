param(
    [string]$LoweringRoot = '',
    [string]$ZludaRoot = ''
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repo = Split-Path -Parent $PSScriptRoot
if (-not $LoweringRoot) {
    $LoweringRoot = Join-Path $repo 'results\20260831_223000_decoder_slots99_154_lowering'
}
if (-not $ZludaRoot) {
    $ZludaRoot = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
}

$loweringRootResolved = (Resolve-Path -LiteralPath $LoweringRoot).Path
$zludaRootResolved = (Resolve-Path -LiteralPath $ZludaRoot).Path
$nvcuda = Join-Path $zludaRootResolved 'nvcuda.dll'
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "ZLUDA nvcuda.dll not found: $nvcuda" }
if (-not (Test-Path -LiteralPath $probe)) { throw "Probe not found: $probe" }

$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zludaRootResolved;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"

$lowering = Get-Content -LiteralPath (Join-Path $loweringRootResolved 'manifest.json') -Raw |
    ConvertFrom-Json
$entries = @()
foreach ($entry in $lowering.entries) {
    $ptx = Join-Path $loweringRootResolved "$($entry.tag)_full.ptx"
    $json = Join-Path $loweringRootResolved "$($entry.tag)_resolve.json"
    $log = Join-Path $loweringRootResolved "$($entry.tag)_resolve.log"
    if (-not (Test-Path -LiteralPath $ptx)) { throw "Final PTX not found: $ptx" }

    & $probe --nvcuda $nvcuda --ptx $ptx --function $entry.function --json $json *> $log
    $exitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $json)) { throw "Probe JSON not written: $json" }
    $observed = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    $pass = ([bool]$observed.pass -and [bool]$observed.module_loaded -and
             [bool]$observed.function_resolved)
    $entries += [ordered]@{
        tag = [string]$entry.tag
        function = [string]$entry.function
        pass = $pass
        exit_code = $exitCode
        module_loaded = [bool]$observed.module_loaded
        function_resolved = [bool]$observed.function_resolved
        ptx_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $ptx).Hash
        result_file = Split-Path -Leaf $json
        log_file = Split-Path -Leaf $log
    }
}

$allPass = @($entries | Where-Object { -not $_.pass }).Count -eq 0
$manifest = [ordered]@{
    schema = 1
    experiment = 'decoder_slots99_154_rx9070xt_function_resolution'
    status = $(if ($allPass) { 'PASS' } else { 'FAIL' })
    device = 'AMD Radeon RX 9070 XT [ZLUDA]'
    entry_count = $entries.Count
    entries = $entries
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath `
    (Join-Path $loweringRootResolved 'resolution_manifest.json') -Encoding utf8
$manifest | ConvertTo-Json -Depth 8 -Compress | Write-Output
if (-not $allPass) { exit 1 }
