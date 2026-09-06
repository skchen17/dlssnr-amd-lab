param(
    [string]$ModuleRoot = '',
    [string]$ZludaRoot = '',
    [string]$Python = 'C:\Users\20426\anaconda3\python.exe'
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

$repo = Split-Path -Parent $PSScriptRoot
if (-not $ModuleRoot) {
    $ModuleRoot = Join-Path $repo 'results\20260831_010100_all_runtime_modules'
}
if (-not $ZludaRoot) {
    $ZludaRoot = Join-Path $repo '.tools\zluda-v7-preview.3\zluda'
}
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }

$moduleRootResolved = (Resolve-Path -LiteralPath $ModuleRoot).Path
$zludaRootResolved = (Resolve-Path -LiteralPath $ZludaRoot).Path
$nvcuda = Join-Path $zludaRootResolved 'nvcuda.dll'
$zludaLauncher = Join-Path $zludaRootResolved 'zluda.exe'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "ZLUDA nvcuda.dll not found: $nvcuda" }
if (-not (Test-Path -LiteralPath $zludaLauncher)) { throw "ZLUDA launcher not found: $zludaLauncher" }

& (Join-Path $repo 'scripts\build_all.ps1') -Only zluda_ptx_probe
if ($LASTEXITCODE -ne 0) { throw "zluda_ptx_probe build failed: $LASTEXITCODE" }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$resultDir = Join-Path $repo "results\${stamp}_zluda_ptx_probe"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

$rocm = 'C:\Users\20426\Documents\ComfyUI\.venv\Lib\site-packages\_rocm_sdk_core'
$env:PATH = "$zludaRootResolved;$rocm\bin;$rocm\lib\llvm\bin;$env:PATH"
$probe = Join-Path $repo 'build\zluda_ptx_probe.exe'
$extractor = Join-Path $repo 'scripts\extract_ptx_entry.py'

$entries = @(
    [ordered]@{
        id = 'clear'
        source = 'module_06_010E2A40.ptx'
        entry = 'cc_cb_clear'
    },
    [ordered]@{
        id = 'copy'
        source = 'module_14_0113D3C0.ptx'
        entry = 'cg2r_copy_kernel'
    },
    [ordered]@{
        id = 'neural'
        source = 'module_00_000DF0E0.ptx'
        entry = 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'
    }
)

foreach ($entry in $entries) {
    $source = Join-Path $moduleRootResolved $entry.source
    $output = Join-Path $resultDir "$($entry.id)_isolated.ptx"
    $metadata = Join-Path $resultDir "$($entry.id)_extraction.json"
    & $Python $extractor --input $source --entry $entry.entry --output $output --metadata $metadata
    if ($LASTEXITCODE -ne 0) { throw "PTX entry extraction failed: $($entry.entry)" }
}

$cases = @(
    [ordered]@{
        id = 'baseline'
        ptx = Join-Path $repo 'tests\zluda_ptx_probe\simple_store.ptx'
        function = 'simple_store'
        expected_pass = $true
        clear_words = 0
    },
    [ordered]@{
        id = 'clear'
        ptx = Join-Path $resultDir 'clear_isolated.ptx'
        function = 'cc_cb_clear'
        expected_pass = $true
        clear_words = 27648
    },
    [ordered]@{
        id = 'copy'
        ptx = Join-Path $resultDir 'copy_isolated.ptx'
        function = 'cg2r_copy_kernel'
        expected_pass = $false
        clear_words = 0
    },
    [ordered]@{
        id = 'neural'
        ptx = Join-Path $resultDir 'neural_isolated.ptx'
        function = 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'
        expected_pass = $false
        clear_words = 0
    }
)

$caseResults = @()
foreach ($case in $cases) {
    $jsonPath = Join-Path $resultDir "$($case.id)_probe.json"
    $logPath = Join-Path $resultDir "$($case.id)_probe.log"
    $args = @('--nvcuda', $nvcuda, '--ptx', $case.ptx, '--function',
              $case.function, '--json', $jsonPath)
    if ($case.clear_words -gt 0) {
        $args += @('--clear-words', [string]$case.clear_words)
    }
    & $probe @args *> $logPath
    $exitCode = $LASTEXITCODE
    $probeJson = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
    $caseResults += [ordered]@{
        id = $case.id
        expected_pass = $case.expected_pass
        observed_pass = [bool]$probeJson.pass
        expectation_met = ([bool]$probeJson.pass -eq [bool]$case.expected_pass)
        exit_code = $exitCode
        module_loaded = [bool]$probeJson.module_loaded
        function_resolved = [bool]$probeJson.function_resolved
        kernel_launched = [bool]$probeJson.kernel_launched
        execution_verified = [bool]$probeJson.execution_verified
        clear_mismatches = [uint64]$probeJson.clear_mismatches
        ptx_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $case.ptx).Hash
        result_file = Split-Path -Leaf $jsonPath
    }
}

# Capture translator parser diagnostics for the two currently unsupported
# function-level modules. The trace wrapper makes the exact instruction blocker
# reproducible instead of treating CUDA_ERROR_NOT_FOUND as a generic failure.
$traceRoot = Join-Path $env:TEMP 'zluda'
foreach ($caseId in @('copy', 'neural')) {
    $case = $cases | Where-Object { $_.id -eq $caseId } | Select-Object -First 1
    $started = Get-Date
    $traceJson = Join-Path $resultDir "$caseId`_trace_probe.json"
    $traceStdout = Join-Path $resultDir "$caseId`_trace_stdout.log"
    & $zludaLauncher --zluda-trace -- $probe --nvcuda $nvcuda --ptx $case.ptx `
        --function $case.function --json $traceJson *> $traceStdout
    $traceExit = $LASTEXITCODE
    $traceDir = Get-ChildItem -LiteralPath $traceRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $started.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($traceDir) {
        $parserLog = Join-Path $traceDir.FullName 'module_0001.log'
        if (Test-Path -LiteralPath $parserLog) {
            Copy-Item -LiteralPath $parserLog -Destination (Join-Path $resultDir "$caseId`_parser.log")
        }
    }
    Set-Content -LiteralPath (Join-Path $resultDir "$caseId`_trace_exit.txt") `
        -Value ([string]$traceExit) -Encoding ascii
}

$expectationsMet = @($caseResults | Where-Object { -not $_.expectation_met }).Count -eq 0
$clearCase = $caseResults | Where-Object { $_.id -eq 'clear' } | Select-Object -First 1
$copyCase = $caseResults | Where-Object { $_.id -eq 'copy' } | Select-Object -First 1
$neuralCase = $caseResults | Where-Object { $_.id -eq 'neural' } | Select-Object -First 1
$manifest = [ordered]@{
    schema_version = 1
    classification = 'PTX_TRANSLATION_PROBE'
    timestamp = $stamp
    device = 'AMD Radeon RX 9070 XT [ZLUDA]'
    overall_pass = $expectationsMet
    counts_as_s6 = $false
    full_translation_pass = ([bool]$copyCase.observed_pass -and [bool]$neuralCase.observed_pass)
    original_nvidia_ptx_executed = [bool]$clearCase.execution_verified
    neural_math_executed = $false
    notes = @(
        'PASS means the compatibility probe behaved as expected, not that DLSS Neural Rendering is complete.',
        'cc_cb_clear is isolated unchanged from the NVIDIA runtime PTX and executed with the exact R21 count of 27648 words.',
        'Copy and neural entries are expected failures until their unsupported PTX instructions are translated or lowered.'
    )
    zluda = [ordered]@{
        release = 'v7-preview.3'
        nvcuda_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $nvcuda).Hash
        root = $zludaRootResolved
    }
    module_root = $moduleRootResolved
    cases = $caseResults
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath `
    (Join-Path $resultDir 'manifest.json') -Encoding utf8

Write-Host "Result: $resultDir"
Write-Host "Expected outcomes met: $expectationsMet"
Write-Host "Original NVIDIA PTX clear executed: $($manifest.original_nvidia_ptx_executed)"
Write-Host "Full copy+neural translation: $($manifest.full_translation_pass)"
if (-not $expectationsMet) { exit 1 }
