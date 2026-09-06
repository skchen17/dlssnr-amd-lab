<#
.SYNOPSIS
  Self-contained Windows NVIDIA reference/diagnostic runner for a packaged lab build.
.DESCRIPTION
  Runs environment capture, instrumentation self-tests, official NGX ABI probing,
  vanilla DLSS/DLAA Stage A, and an observational DLSSNR load run. Proprietary
  runtimes are referenced by path, hashed, and never copied into the result bundle.

  IMPORTANT: the current standalone host does not provide a ReShade add-on host,
  so RenoDX feature 18 is NOT executed here. Stage B is explicitly labeled as an
  observational DLL-load/NGX run rather than a Neural Rendering success test.
#>
[CmdletBinding()]
param(
    [string]$DlssDll = '',
    [string]$DlssNrDll = '',
    [string]$RenoDxAddon = '',
    [int]$Frames = 8,
    [int]$Width = 512,
    [int]$Height = 512
)

$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build'

function Resolve-ExistingFile([string]$Path, [string]$Label, [bool]$Required) {
    if (-not $Path -and $Required) { $Path = Read-Host "Path to $Label" }
    if (-not $Path) {
        if ($Required) { throw "$Label is required; rerun with its full path." }
        return $null
    }
    try { return (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path }
    catch {
        if ($Required) { throw "$Label not found: $Path" }
        Write-Warning "$Label not found; optional stage skipped: $Path"
        return $null
    }
}

function File-Record([string]$Path) {
    if (-not $Path) { return [ordered]@{ present = $false } }
    $item = Get-Item -LiteralPath $Path
    $sig = Get-AuthenticodeSignature -LiteralPath $Path -ErrorAction SilentlyContinue
    return [ordered]@{
        present = $true
        filename = $item.Name
        size = $item.Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        file_version = $item.VersionInfo.FileVersion
        product_version = $item.VersionInfo.ProductVersion
        signature_status = if ($sig) { $sig.Status.ToString() } else { 'Unavailable' }
        signature_subject = if ($sig -and $sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { $null }
    }
}

function Save-Json([object]$Value, [string]$Path) {
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Run-Native([string]$Name, [scriptblock]$Command, [string]$Log) {
    Write-Host "`n=== $Name ==="
    & $Command *>&1 | Tee-Object -FilePath $Log | ForEach-Object { Write-Host $_ }
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    Write-Host "$Name exit=$code"
    return [int]$code
}

if ($Frames -lt 1 -or $Width -lt 8 -or $Height -lt 8) {
    throw 'Frames must be >=1 and Width/Height must be >=8.'
}

$requiredTools = @(
    'env_probe.exe', 'ngx_abi_probe.exe', 'binary_probe.exe', 'nr_host.exe',
    'module_trace.dll', 'module_trace_selftest.exe', 'test_dll_a.dll',
    'test_dll_b.dll', 'nvapi64.dll', 'nvapi_trampoline_test.exe'
)
foreach ($name in $requiredTools) {
    if (-not (Test-Path -LiteralPath (Join-Path $build $name))) {
        throw "Packaged tool missing: build\$name"
    }
}

$DlssDll = Resolve-ExistingFile $DlssDll 'nvngx_dlss.dll' $true
$DlssNrDll = Resolve-ExistingFile $DlssNrDll 'nvngx_dlssnr.dll' $false
$RenoDxAddon = Resolve-ExistingFile $RenoDxAddon 'RenoDX DLSS5 addon' $false

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runsRoot = Join-Path $root 'cloud_runs'
$runDir = Join-Path $runsRoot $stamp
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$summary = [ordered]@{
    schema = 1
    package_revision = 'v3_round3_evaluate_contract'
    timestamp_local = $stamp
    timestamp_utc = (Get-Date).ToUniversalTime().ToString('o')
    requested = [ordered]@{ frames = $Frames; width = $Width; height = $Height }
    stages = [ordered]@{}
    warnings = @(
        'Stage B is observational only: this runner does not host or inject the RenoDX ReShade addon.',
        'No proprietary DLL or addon is copied into the returned ZIP.'
    )
}

$oldDlss = $env:DLSS_DLL_PATH
$oldNr = $env:DLSSNR_DLL_PATH
$oldAddon = $env:RENODX_DLSS5_ADDON_PATH
$oldModuleLog = $env:MODULE_TRACE_LOG
$oldNvapiLog = $env:NVAPI_TRACE_LOG

try {
    # Environment capture.
    $envRaw = & (Join-Path $build 'env_probe.exe') --json 2>&1 | Out-String
    $envRaw | Set-Content -LiteralPath (Join-Path $runDir 'environment_dxgi.json') -Encoding UTF8
    $wmiGpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object {
        [ordered]@{
            name = $_.Name
            pnp_device_id = $_.PNPDeviceID
            driver_version = $_.DriverVersion
            driver_date = $_.DriverDate
            video_processor = $_.VideoProcessor
        }
    })
    $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    $smi = $null
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        $smi = (& nvidia-smi.exe --query-gpu=index,name,uuid,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1 | Out-String).Trim()
        & nvidia-smi.exe -q 2>&1 | Set-Content -LiteralPath (Join-Path $runDir 'nvidia_smi_q.txt') -Encoding UTF8
    }
    Save-Json ([ordered]@{
        computer_name = $env:COMPUTERNAME
        windows_caption = $os.Caption
        windows_version = $os.Version
        windows_build = $os.BuildNumber
        gpus = $wmiGpus
        nvidia_smi_csv = $smi
    }) (Join-Path $runDir 'environment.json')

    $summary.runtimes = [ordered]@{
        dlss = File-Record $DlssDll
        dlssnr = File-Record $DlssNrDll
        renodx_addon = File-Record $RenoDxAddon
        tools = [ordered]@{
            nr_host = File-Record (Join-Path $build 'nr_host.exe')
            module_trace = File-Record (Join-Path $build 'module_trace.dll')
            nvapi_trace = File-Record (Join-Path $build 'nvapi64.dll')
        }
    }
    Save-Json $summary.runtimes (Join-Path $runDir 'runtime_hashes.json')

    # Instrumentation readiness gates.
    $selfDir = Join-Path $runDir 'selftests'
    New-Item -ItemType Directory -Force -Path $selfDir | Out-Null
    $moduleLog = Join-Path $selfDir 'module_trace.jsonl'
    $moduleCode = Run-Native 'module_trace_selftest' {
        & (Join-Path $build 'module_trace_selftest.exe') `
            (Join-Path $build 'module_trace.dll') `
            (Join-Path $build 'test_dll_a.dll') $moduleLog
    } (Join-Path $selfDir 'module_trace_stdout.log')
    $summary.stages.module_trace_selftest = [ordered]@{ exit = $moduleCode; status = if ($moduleCode -eq 0) { 'PASS' } else { 'FAIL' } }

    $nvapiDir = Join-Path $selfDir 'nvapi'
    New-Item -ItemType Directory -Force -Path $nvapiDir | Out-Null
    Push-Location $nvapiDir
    try {
        $nvapiCode = Run-Native 'nvapi_trampoline_test' {
            & (Join-Path $build 'nvapi_trampoline_test.exe') (Join-Path $build 'nvapi64.dll')
        } (Join-Path $selfDir 'nvapi_trampoline_stdout.log')
    } finally { Pop-Location }
    $summary.stages.nvapi_trampoline_selftest = [ordered]@{ exit = $nvapiCode; status = if ($nvapiCode -eq 0) { 'PASS' } else { 'FAIL' } }

    # Official ABI and static payload probes.
    $abiCode = Run-Native 'ngx_abi_probe' {
        & (Join-Path $build 'ngx_abi_probe.exe') --dll $DlssDll --json (Join-Path $runDir 'ngx_abi_test.json')
    } (Join-Path $runDir 'ngx_abi_stdout.log')
    $summary.stages.ngx_abi_probe = [ordered]@{ exit = $abiCode; status = if ($abiCode -eq 0) { 'PASS' } else { 'FAIL' } }

    if ($DlssNrDll) {
        $binaryCode = Run-Native 'binary_probe_dlssnr' {
            & (Join-Path $build 'binary_probe.exe') $DlssNrDll --json (Join-Path $runDir 'binary_manifest_dlssnr.json')
        } (Join-Path $runDir 'binary_probe_stdout.log')
        $summary.stages.binary_probe_dlssnr = [ordered]@{ exit = $binaryCode; status = if ($binaryCode -eq 0) { 'PASS' } else { 'FAIL' } }
    } else {
        $summary.stages.binary_probe_dlssnr = [ordered]@{ status = 'SKIPPED_MISSING_OPTIONAL_DLL' }
    }

    # Stage A: real, public DLSS/DLAA path. Never expose NR/addon to this process.
    $stageA = Join-Path $runDir 'stage_a_dlaa'
    New-Item -ItemType Directory -Force -Path $stageA | Out-Null
    $env:DLSS_DLL_PATH = $DlssDll
    $env:DLSSNR_DLL_PATH = $null
    $env:RENODX_DLSS5_ADDON_PATH = $null
    $startA = Get-Date
    Push-Location $root
    try {
        $stageACode = Run-Native 'stage_a_dlaa' {
            & (Join-Path $build 'nr_host.exe') --frames $Frames --width $Width --height $Height `
                --output (Join-Path $stageA 'output_rgba8.bin') --json (Join-Path $stageA 'host.json')
        } (Join-Path $stageA 'stdout.log')
    } finally { Pop-Location }
    $newInputPackage = Get-ChildItem (Join-Path $root 'results') -Directory -Filter '*_reference_package' -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $startA.AddSeconds(-2) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newInputPackage) { Copy-Item -LiteralPath $newInputPackage.FullName -Destination (Join-Path $stageA 'deterministic_inputs') -Recurse -Force }
    $outputA = Join-Path $stageA 'output_rgba8.bin'
    $summary.stages.stage_a_dlaa = [ordered]@{
        exit = $stageACode
        status = if ($stageACode -eq 0 -and (Test-Path -LiteralPath $outputA)) { 'PASS' } else { 'FAIL' }
        output_sha256 = if (Test-Path -LiteralPath $outputA) { (Get-FileHash -LiteralPath $outputA -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
    }

    # Stage B observation: load DLSSNR beside a real NVIDIA DLAA run and trace modules.
    # RenoDX feature 18 is intentionally NOT claimed: no ReShade add-on host exists here.
    if ($DlssNrDll) {
        $stageB = Join-Path $runDir 'stage_b_observation'
        New-Item -ItemType Directory -Force -Path $stageB | Out-Null
        $env:DLSS_DLL_PATH = $DlssDll
        $env:DLSSNR_DLL_PATH = $DlssNrDll
        $env:RENODX_DLSS5_ADDON_PATH = $RenoDxAddon
        $env:MODULE_TRACE_LOG = Join-Path $stageB 'module_trace.jsonl'
        $env:NVAPI_TRACE_LOG = Join-Path $stageB 'nvapi_trace.jsonl'
        Push-Location $root
        try {
            $stageBCode = Run-Native 'stage_b_observation' {
                & (Join-Path $build 'nr_host.exe') --frames $Frames --width $Width --height $Height --trace `
                    --output (Join-Path $stageB 'output_rgba8.bin') --json (Join-Path $stageB 'host.json')
            } (Join-Path $stageB 'stdout.log')
        } finally { Pop-Location }
        $summary.stages.stage_b_observation = [ordered]@{
            exit = $stageBCode
            status = if ($stageBCode -eq 0) { 'OBSERVATION_COMPLETE' } else { 'OBSERVATION_FAILED' }
            feature_18 = 'NOT_RUN_NO_RESHADE_ADDON_HOST'
        }
    } else {
        $summary.stages.stage_b_observation = [ordered]@{ status = 'SKIPPED_MISSING_DLSSNR_DLL'; feature_18 = 'NOT_RUN' }
    }
} finally {
    $env:DLSS_DLL_PATH = $oldDlss
    $env:DLSSNR_DLL_PATH = $oldNr
    $env:RENODX_DLSS5_ADDON_PATH = $oldAddon
    $env:MODULE_TRACE_LOG = $oldModuleLog
    $env:NVAPI_TRACE_LOG = $oldNvapiLog
}

Save-Json $summary (Join-Path $runDir 'summary.json')
$forbidden = @(Get-ChildItem -LiteralPath $runDir -Recurse -File | Where-Object {
    $_.Name -match '^(nvngx(_dlss|_dlssnr)?\.dll|renodx.*\.(dll|addon64))$'
})
if ($forbidden.Count -gt 0) {
    throw "Safety check failed: proprietary runtime appeared in result directory: $($forbidden.FullName -join ', ')"
}

$returnDir = Join-Path $root 'return_to_lab'
New-Item -ItemType Directory -Force -Path $returnDir | Out-Null
$zip = Join-Path $returnDir "dlssnr_cloud_result_$stamp.zip"
Compress-Archive -Path (Join-Path $runDir '*') -DestinationPath $zip -CompressionLevel Optimal -Force
$zipHash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "`n=== COMPLETE ==="
Write-Host "Return this file: $zip"
Write-Host "SHA256: $zipHash"
Write-Host 'The ZIP contains no nvngx or RenoDX proprietary binary.'
# Stage failures are evidence recorded in summary.json, not runner failures. Once
# the result archive is produced successfully, return success to PowerShell.
exit 0
