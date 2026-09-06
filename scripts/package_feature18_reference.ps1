[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PayloadDir,
    [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path (Split-Path -Parent $repo) 'deliverables' }
$payloadSource = (Resolve-Path -LiteralPath $PayloadDir -ErrorAction Stop).Path
$outputResolved = [IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $outputResolved | Out-Null

& (Join-Path $PSScriptRoot 'build_all.ps1') -Only module_trace,dlss5_feed_host64
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$staging = Join-Path $outputResolved "_feature18_v26_staging_$stamp"
$buildDir = Join-Path $staging 'build'
$packagePayload = Join-Path $staging 'payload'
$scriptDir = Join-Path $staging 'scripts'
New-Item -ItemType Directory -Path $staging,$buildDir,$packagePayload,$scriptDir | Out-Null

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
foreach ($name in @('dlss5-feed-host64.exe','dlss5-feed-host64.pdb','module_trace.dll','module_trace.pdb')) {
    Copy-Item -LiteralPath (Join-Path $repo "build\$name") -Destination (Join-Path $buildDir $name)
}
Copy-Item -LiteralPath (Join-Path $repo 'FEATURE18_PACKAGE_README.md') -Destination $staging
Copy-Item -LiteralPath (Join-Path $repo 'scripts\run_feature18_reference.ps1') -Destination $scriptDir

function Get-PackageFileRecord([string]$Path) {
    $item = Get-Item -LiteralPath $Path
    # Windows PowerShell 5.1 uses the .NET Framework Path API, which does not
    # expose GetRelativePath. Both paths are already resolved below the staging
    # root, so a validated prefix removal is sufficient and portable.
    $stagingPrefix = [IO.Path]::GetFullPath($staging).TrimEnd('\') + '\'
    $itemPath = [IO.Path]::GetFullPath($item.FullName)
    if (-not $itemPath.StartsWith($stagingPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "package file escaped staging root: $itemPath"
    }
    $relative = $itemPath.Substring($stagingPrefix.Length).Replace('\','/')
    $sig = Get-AuthenticodeSignature -LiteralPath $item.FullName -ErrorAction SilentlyContinue
    [ordered]@{
        path = $relative
        size = $item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        file_version = if ($item.VersionInfo.FileVersion) { $item.VersionInfo.FileVersion } else { $null }
        signature_status = if ($sig) { $sig.Status.ToString() } else { 'Unavailable' }
        signature_subject = if ($sig -and $sig.SignerCertificate) { $sig.SignerCertificate.Subject } else { $null }
    }
}

$records = @(Get-ChildItem -LiteralPath $staging -Recurse -File |
    Sort-Object FullName | ForEach-Object { Get-PackageFileRecord $_.FullName })
$downstreamOffsets = [ordered]@{
    slot3='0x44DA00'; slot4='0x1390400'; slot5='0x7681600'
    slot6='0x86FCE00'; slot7='0x8BC7400'; slot8='0x8C8A600'; slot9='0x8C9F000'
    slot10='0x8CB0200'; slot11='0xA800'; slot12='0x3AC00'; slot13='0x6B000'
    slot14='0x9B400'; slot15='0xCB800'; slot16='0x103C00'; slot17='0x1AC200'
    slot18='0x254800'; slot19='0x2FCE00'; slot20='0x3A5400'; slot21='0x452C00'
    slot22='0x4FB200'; slot23='0x5A3800'
}
$manifest = [ordered]@{
    schema = 1
    package_revision = 'v26_feature18_submission_topology'
    generated_local = $stamp
    scope = 'Private local experiment package assembled from user-supplied files; do not redistribute'
    purpose = 'Record the frame-1 launch/Close/ExecuteCommandLists topology and capture slot155 input/output after queue completion, with intrusive inline snapshots disabled'
    default_variant = '02'
    downstream_weight_offsets = $downstreamOffsets
    files = $records
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $staging 'PACKAGE_MANIFEST.json') -Encoding utf8

$zip = Join-Path $outputResolved "dlssnr-windows-feature18-submission-topology-v26-$stamp.zip"
Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Package: $zip"
Write-Host "SHA256: $((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash)"
Write-Host "Staging: $staging"
