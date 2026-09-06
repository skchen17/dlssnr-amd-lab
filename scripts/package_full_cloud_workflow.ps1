[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PayloadDir,
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'deliverables' }
$payloadSource = (Resolve-Path -LiteralPath $PayloadDir -ErrorAction Stop).Path
$outputResolved = [IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $outputResolved | Out-Null

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only module_trace,dlss5_feed_host64
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

$inventory = Join-Path $repo 'results\20260831_175000_full_graph_inventory\full_graph_inventory.json'
if (-not (Test-Path -LiteralPath $inventory -PathType Leaf)) {
    throw "full graph inventory missing: $inventory"
}
$inventoryJson = Get-Content -LiteralPath $inventory -Raw | ConvertFrom-Json
if ($inventoryJson.status -ne 'PASS' -or $inventoryJson.slot_count -ne 156 -or
    $inventoryJson.unique_function_count -ne 43) {
    throw 'full graph inventory gate failed'
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$staging = Join-Path $outputResolved "_full_graph_v20_staging_$stamp"
$buildDir = Join-Path $staging 'build'
$packagePayload = Join-Path $staging 'payload'
$scriptDir = Join-Path $staging 'scripts'
$inventoryDir = Join-Path $staging 'inventory'
New-Item -ItemType Directory -Path $staging,$buildDir,$packagePayload,$scriptDir,$inventoryDir | Out-Null

$requiredPayload = @(
    'nvngx_dlss.dll',
    'nvngx_dlssnr.dll',
    'renodx-dlss5-01.addon64',
    'renodx-dlss5-02.addon64',
    'renodx-dlss5-02.5.addon64',
    'ReShade64.dll'
)
foreach ($name in $requiredPayload) {
    $source = Join-Path $payloadSource $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "required user-supplied payload missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $packagePayload $name)
}
foreach ($name in @('dlss5-feed-host64.exe','module_trace.dll')) {
    Copy-Item -LiteralPath (Join-Path $repo "build\$name") -Destination (Join-Path $buildDir $name)
}
foreach ($name in @(
    'run_full_cloud_workflow.ps1',
    'run_feature18_reference.ps1',
    'consolidate_full_graph_capture.ps1'
)) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $scriptDir $name)
}
Copy-Item -LiteralPath $inventory -Destination (Join-Path $inventoryDir 'full_graph_inventory.json')
Copy-Item -LiteralPath (Join-Path $repo 'FULL_GRAPH_PACKAGE_README.md') -Destination (Join-Path $staging 'README.md')

function Get-PackageFileRecord([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    $base = [IO.Path]::GetFullPath($staging).TrimEnd('\') + '\'
    $full = [IO.Path]::GetFullPath($item.FullName)
    if (-not $full.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) {
        throw "package file escapes staging directory: $full"
    }
    $relative = $full.Substring($base.Length).Replace('\','/')
    [ordered]@{
        path = $relative
        size = $item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$records = @(Get-ChildItem -LiteralPath $staging -Recurse -File |
    Sort-Object FullName | ForEach-Object { Get-PackageFileRecord $_.FullName })
$manifest = [ordered]@{
    schema = 1
    package_revision = 'v20_full_graph_one_shot'
    generated_local = $stamp
    scope = 'Private local experiment package assembled from user-supplied files; do not redistribute'
    purpose = 'One-command RTX acquisition of bounded pre/post buffer windows for all 156 DLSSNR graph slots, plus descriptor/texture/final-copy evidence'
    default_variant = '02'
    default_capture_limit_bytes = 8388608
    minimum_free_space_bytes = 32212254720
    static_inventory = [ordered]@{
        status = $inventoryJson.status
        slot_count = $inventoryJson.slot_count
        unique_function_count = $inventoryJson.unique_function_count
        module_count = $inventoryJson.module_count
        sha256 = (Get-FileHash -LiteralPath $inventory -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    files = $records
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath `
    (Join-Path $staging 'PACKAGE_MANIFEST.json') -Encoding utf8

$archive = Join-Path $outputResolved "dlssnr-windows-full-graph-v20-$stamp.zip"
Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Package: $archive"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash)"
Write-Host "Staging: $staging"
