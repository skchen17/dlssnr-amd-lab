param(
    [Parameter(Mandatory=$true)][string]$FreshOutputDirectory,
    [string]$LibAmdRdfRoot = ''
)
$ErrorActionPreference='Stop'
$PSNativeCommandUseErrorActionPreference=$false
. (Join-Path $PSScriptRoot 'build_common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$output=[IO.Path]::GetFullPath($FreshOutputDirectory)
if (Test-Path -LiteralPath $output) { throw 'Fresh output directory already exists; refusing overwrite' }
New-Item -ItemType Directory -Path $output | Out-Null
if (-not $LibAmdRdfRoot) { $LibAmdRdfRoot=Join-Path $repo '.tools\libamdrdf-src' }
$rdf=[IO.Path]::GetFullPath($LibAmdRdfRoot)
foreach ($path in @("$rdf\rdf\inc\amdrdf.h","$rdf\rdf\src\amdrdf.cpp","$rdf\imported\zstd\src\zstd.c")) {
    if (-not (Test-Path -LiteralPath $path)) { throw "missing public libamdrdf source: $path" }
}
$envInfo=Resolve-BuildEnv
$args=@('--driver-mode=cl','/nologo','/EHsc','/std:c++20','/O2','/MD','-fuse-ld=lld-link',
        '/DRDF_BUILD_LIBRARY','/DRDF_BUILD_STATIC=1','/DRDF_CXX_BINDINGS=1','/DRDF_PLATFORM_WINDOWS=1','/D_CRT_SECURE_NO_WARNINGS',
        "/I$rdf\rdf\inc","/I$rdf\imported\zstd\inc",
        "$repo\tools\rgp_rdf_dump\rgp_rdf_dump.cpp","$rdf\rdf\src\amdrdf.cpp","$rdf\imported\zstd\src\zstd.c")
$args+=Get-SdkIncludeArgs $envInfo
$args+="/Fe:$output\rgp_rdf_dump.exe"
$args+='/link';$args+=Get-SdkLibArgs $envInfo;$args+=Get-MsvcLibArgs $envInfo
& $envInfo.clangxx @args
if ($LASTEXITCODE -ne 0) { throw "rgp_rdf_dump build failed: $LASTEXITCODE" }
Get-FileHash -Algorithm SHA256 -LiteralPath "$output\rgp_rdf_dump.exe" | Format-List
