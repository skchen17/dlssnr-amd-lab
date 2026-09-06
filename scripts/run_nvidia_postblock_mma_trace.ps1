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
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing payload: $($property.Name)" }
    $actual[$property.Name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $integrity = $integrity -and $actual[$property.Name] -eq $property.Value
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stem = if ($package.result_stem) { [string]$package.result_stem } else { 'postblock_mma_trace' }
$work = Join-Path $root "_${stem}_work_$stamp"
$result = Join-Path $root "_${stem}_reference_result_$stamp"
New-Item -ItemType Directory -Path $work,$result | Out-Null
$runs = @()
try {
    $activation = [IO.File]::ReadAllBytes((Join-Path $payload 'activation_arena.raw'))
    $arena = [byte[]]::new($activation.Length + [int]$package.trace_bytes)
    [Array]::Copy($activation,$arena,$activation.Length)
    $arenaInitial = Join-Path $work 'arena_initial.raw'
    [IO.File]::WriteAllBytes($arenaInitial,$arena)
    foreach ($index in 1..2) {
        $tag = "run$index"
        $probeJson = Join-Path $work "$tag.json"
        $arenaAfter = Join-Path $work "$tag.arena.raw"
        $output = Join-Path $work "$tag.output.raw"
        $log = Join-Path $work "$tag.log"
        & (Join-Path $payload 'neural_reference_probe.exe') `
            --nvcuda $nvcuda --ptx (Join-Path $payload $package.ptx_file) `
            --function $package.function --json $probeJson `
            --n1-arena $arenaInitial --n1-arena-out $arenaAfter `
            --n1-arena-input-offset 13873152 --n1-arena-output-offset 0 `
            --n1-weights (Join-Path $payload 'model_arena.raw') `
            --n1-params (Join-Path $payload 'params_trace.raw') `
            --n1-output $output --n1-sync-out (Join-Path $work "$tag.sync.raw") `
            --n1-input-param-offset 0 --n1-output-param-offset 16 `
            --n1-no-weights-param --n1-no-release --n1-expected-releases 0 `
            --n1-grid-x 81 --n1-grid-y 49 --n1-grid-z 1 `
            --n1-block-x 32 --n1-block-y 1 --n1-block-z 1 `
            --n1-arena-param-view 0:13873152 --n1-arena-param-view 8:110592 `
            --n1-arena-param-view "184:$($activation.Length)" `
            --n1-weight-param-view 24:147429888 --n1-weight-param-view 104:147429376 `
            --n1-rgba16f-surface-param-offset 16 `
            --n1-rgba16f-surface-initial (Join-Path $payload 'surface_initial.raw') `
            --n1-rgba16f-zero-texture-param-offset 56 *> $log
        $probeExit = if ($null -eq $LASTEXITCODE) { -1 } else { [int]$LASTEXITCODE }
        $probe = Get-Content -Raw -LiteralPath $probeJson | ConvertFrom-Json
        $stream = [IO.File]::OpenRead($arenaAfter)
        try {
            [void]$stream.Seek($activation.Length,[IO.SeekOrigin]::Begin)
            $trace = [byte[]]::new([int]$package.trace_bytes)
            $read = 0
            while ($read -lt $trace.Length) {
                $count = $stream.Read($trace,$read,$trace.Length-$read)
                if ($count -eq 0) { break }
                $read += $count
            }
            if ($read -ne $trace.Length) { throw "Trace extraction short read: $read/$($trace.Length)" }
        } finally { $stream.Dispose() }
        $tracePath = Join-Path $result "$tag.trace.raw"
        [IO.File]::WriteAllBytes($tracePath,$trace)
        Copy-Item -LiteralPath $probeJson,$log -Destination $result
        $outputHash = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
        $runs += [ordered]@{
            run = $index
            probe_exit = $probeExit
            probe_pass = [bool]$probe.pass
            execution_verified = [bool]$probe.execution_verified
            device_name = $probe.device_name
            surface_initial_loaded = [bool]$probe.n1_rgba16f_surface_initial_loaded
            trace_file = "$tag.trace.raw"
            trace_bytes = $trace.Length
            trace_nonzero_bytes = @($trace | Where-Object { $_ -ne 0 }).Count
            trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash
            output_bytes = (Get-Item -LiteralPath $output).Length
            output_sha256 = $outputHash
            output_reference_exact = $outputHash -eq $package.output_reference_sha256
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
$repeatExact = $runs.Count -eq 2 -and $runs[0].trace_sha256 -eq $runs[1].trace_sha256
$valid = $runs.Count -eq 2 -and @($runs | Where-Object {
    $_.probe_exit -ne 0 -or -not $_.probe_pass -or -not $_.execution_verified -or
    $_.device_name -notmatch 'NVIDIA' -or -not $_.surface_initial_loaded -or
    $_.trace_bytes -ne $package.trace_bytes -or $_.trace_nonzero_bytes -eq 0 -or
    -not $_.output_reference_exact
}).Count -eq 0
$pass = $integrity -and $valid -and $repeatExact
[ordered]@{
    schema = 1
    package_revision = $package.package_revision
    experiment = [string]$package.result_experiment
    status = if ($pass) { 'PASS' } else { 'FAIL' }
    classification = [string]$package.result_classification
    counts_as_s7 = $false
    payload_integrity = $integrity
    payload_sha256 = $actual
    function = $package.function
    target_cta = $package.target_cta
    trace_bytes = $package.trace_bytes
    output_reference_sha256 = $package.output_reference_sha256
    trace_repeat_bitwise_exact = $repeatExact
    runs = $runs
} | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $result 'manifest.json') -Encoding utf8
$zip = "$result.zip"
Compress-Archive -LiteralPath $result -DestinationPath $zip -CompressionLevel Optimal
Get-Content -Raw -LiteralPath (Join-Path $result 'manifest.json')
Write-Host "Result: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
exit 0
