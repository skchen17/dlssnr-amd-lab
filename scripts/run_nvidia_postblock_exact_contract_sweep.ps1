[CmdletBinding()]
param([string]$PackageRoot)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
    $PackageRoot = Split-Path -Parent $PSScriptRoot
}
$root = (Resolve-Path -LiteralPath $PackageRoot).Path
$payload = Join-Path $root 'payload'
$package = Get-Content -Raw -LiteralPath (Join-Path $root 'manifest.json') | ConvertFrom-Json
$nvcuda = Join-Path $env:WINDIR 'System32\nvcuda.dll'
if (-not (Test-Path -LiteralPath $nvcuda)) { throw "NVIDIA driver not found: $nvcuda" }

$actual = [ordered]@{}
$integrity = $true
foreach ($property in $package.payload_sha256.PSObject.Properties) {
    $path = Join-Path $payload $property.Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing payload: $($property.Name)"
    }
    $actual[$property.Name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $integrity = $integrity -and $actual[$property.Name] -eq $property.Value
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$work = Join-Path $root "_postblock_exact_contract_work_$stamp"
$result = Join-Path $root "_postblock_exact_contract_reference_result_$stamp"
New-Item -ItemType Directory -Path $work,$result | Out-Null
$runs = @()
try {
    foreach ($stage in $package.stages) {
        foreach ($index in 1..2) {
            $tag = "$($stage.name)_run$index"
            $probeJson = Join-Path $work "$tag.json"
            $output = Join-Path $work "$tag.raw"
            $sync = Join-Path $work "$tag.sync.raw"
            $log = Join-Path $work "$tag.log"
            $arguments = @(
                '--nvcuda',$nvcuda,
                '--ptx',(Join-Path $payload $stage.file),
                '--function',$package.function,
                '--json',$probeJson,
                '--n1-arena',(Join-Path $payload 'activation_arena.raw'),
                '--n1-arena-input-offset','13873152',
                '--n1-arena-output-offset','0',
                '--n1-weights',(Join-Path $payload 'model_arena.raw'),
                '--n1-params',(Join-Path $payload 'params.raw'),
                '--n1-output',$output,
                '--n1-sync-out',$sync,
                '--n1-input-param-offset','0',
                '--n1-output-param-offset','16',
                '--n1-no-weights-param','--n1-no-release','--n1-expected-releases','0',
                '--n1-grid-x','81','--n1-grid-y','49','--n1-grid-z','1',
                '--n1-block-x','32','--n1-block-y','1','--n1-block-z','1',
                '--n1-arena-param-view','0:13873152',
                '--n1-arena-param-view','8:110592',
                '--n1-weight-param-view','24:147429888',
                '--n1-weight-param-view','104:147429376',
                '--n1-rgba16f-surface-param-offset','16',
                '--n1-rgba16f-surface-initial',(Join-Path $payload 'surface_initial.raw'),
                '--n1-rgba16f-zero-texture-param-offset','56'
            )
            & (Join-Path $payload 'neural_reference_probe.exe') @arguments *> $log
            $probeExit = if ($null -eq $LASTEXITCODE) { -1 } else { [int]$LASTEXITCODE }
            $probe = if (Test-Path -LiteralPath $probeJson) {
                Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json
            } else { $null }
            if (-not (Test-Path -LiteralPath $output)) {
                [IO.File]::WriteAllBytes($output,[byte[]]::new(0))
            }
            $outputPath = Join-Path $result "$tag.raw"
            Copy-Item -LiteralPath $output -Destination $outputPath
            Copy-Item -LiteralPath $log -Destination $result
            if ($probe) { Copy-Item -LiteralPath $probeJson -Destination $result }
            $outputBytes = (Get-Item -LiteralPath $outputPath).Length
            $outputHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
            $runs += [ordered]@{
                stage = $stage.name
                run = $index
                probe_exit = $probeExit
                probe_pass = [bool]($probe -and $probe.pass)
                execution_verified = [bool]($probe -and $probe.execution_verified)
                surface_initial_loaded = [bool]($probe -and $probe.n1_rgba16f_surface_initial_loaded)
                device_name = if ($probe) { $probe.device_name } else { $null }
                output_file = "$tag.raw"
                output_bytes = $outputBytes
                output_sha256 = $outputHash
                reference_exact = $outputBytes -eq $package.logical_output_bytes -and
                    $outputHash -eq $package.output_reference_sha256
            }
        }
    }
} finally {
    $resolvedWork = [IO.Path]::GetFullPath($work)
    $resolvedRoot = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    if ($resolvedWork.StartsWith($resolvedRoot,[StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedWork)) {
        for ($attempt=1; $attempt -le 8; $attempt++) {
            try { Remove-Item -LiteralPath $resolvedWork -Recurse -Force -ErrorAction Stop; break }
            catch { Start-Sleep -Milliseconds (250*$attempt) }
        }
    }
}

$stageRecords = @()
foreach ($stage in $package.stages) {
    $stageRuns = @($runs | Where-Object stage -eq $stage.name)
    $valid = $stageRuns.Count -eq 2 -and @($stageRuns | Where-Object {
        $_.probe_exit -ne 0 -or -not $_.probe_pass -or -not $_.execution_verified -or
        -not $_.surface_initial_loaded -or $_.device_name -notmatch 'NVIDIA' -or
        $_.output_bytes -ne $package.logical_output_bytes
    }).Count -eq 0
    $stageRecords += [ordered]@{
        name = $stage.name
        order = [int]$stage.order
        valid = $valid
        repeat_bitwise_exact = $stageRuns.Count -eq 2 -and
            $stageRuns[0].output_sha256 -eq $stageRuns[1].output_sha256
        reference_exact = $stageRuns.Count -eq 2 -and
            $stageRuns[0].reference_exact -and $stageRuns[1].reference_exact
        runs = $stageRuns
    }
}
$pass = $integrity -and @($stageRecords | Where-Object {
    -not $_.valid -or -not $_.repeat_bitwise_exact -or -not $_.reference_exact
}).Count -eq 0
[ordered]@{
    schema = 1
    package_revision = $package.package_revision
    experiment = 'rtx_postblock_exact_frame1_contract_stage_sweep'
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = 'RTX_POSTBLOCK_EXACT_FRAME1_CONTRACT'
    counts_as_s7 = $false
    payload_integrity = $integrity
    payload_sha256 = $actual
    function = $package.function
    grid = $package.grid
    block = $package.block
    surface_initial_sha256 = $package.surface_initial_sha256
    output_reference_sha256 = $package.output_reference_sha256
    stages = $stageRecords
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$zip = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $zip -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
# A failing exactness result is still required diagnostic evidence and must be returned.
exit 0
