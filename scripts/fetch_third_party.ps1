# Fetch third-party sources via GitHub API (git clone over HTTPS is blocked on this box).
# Records repo commit/date for provenance. No proprietary NVIDIA binaries are fetched:
# NVIDIA/DLSS is the public SDK repo (headers + license); DLSS5-Feeder is MIT.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = 'c:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\third_party'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$provFile = Join-Path $root 'PROVENANCE.md'
$prov = @("# Third-party provenance", "", "Fetched via GitHub REST API (git HTTPS blocked on this machine). Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')", "")

function Fetch-Repo {
    param([string]$Owner, [string]$Repo, [string]$DestName, [string]$Note)
    $meta = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo" -Headers @{ 'User-Agent' = 'dlssnr-amd-lab' } -TimeoutSec 30
    $sha = $meta.default_branch
    $commit = Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo/commits/$sha" -Headers @{ 'User-Agent' = 'dlssnr-amd-lab' } -TimeoutSec 30
    $csha = $commit.sha
    $cdate = $commit.commit.committer.date
    $tar = Join-Path $env:TEMP "$Repo.tar.gz"
    $zip = Join-Path $env:TEMP "$Repo.tar"
    Write-Host "downloading $Owner/$Repo @ $($csha.Substring(0,12)) ..."
    Invoke-WebRequest -Uri "https://api.github.com/repos/$Owner/$Repo/tarball/$sha" -OutFile $tar -Headers @{ 'User-Agent' = 'dlssnr-amd-lab' } -TimeoutSec 600
    $dest = Join-Path $root $DestName
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    # Windows 10+ built-in tar handles gzip
    tar -xzf $tar -C $dest --strip-components=1 2>&1 | Out-String | Write-Host
    if ($LASTEXITCODE -ne 0) { Write-Host "tar failed for $Repo"; exit 1 }
    Remove-Item $tar -Force -ErrorAction SilentlyContinue
    Write-Host ("extracted to {0} ({1} entries)" -f $dest, (Get-ChildItem $dest).Count)
    return @{ sha = $csha; date = $cdate; license = $meta.license.spdx_id; owner = $Owner; repo = $Repo }
}

$dlss = Fetch-Repo -Owner 'NVIDIA' -Repo 'DLSS' -DestName 'nvidia-dlss' -Note 'official public NGX SDK headers'
$prov += "## NVIDIA/DLSS"
$prov += "- repo: https://github.com/NVIDIA/DLSS"
$prov += "- commit: $($dlss.sha) ($($dlss.date))"
$prov += "- license: $($dlss.license)"
$prov += "- purpose: official NGX public headers (nvsdk_ngx*.h) for ABI-correct nr_host"
$prov += ""

$f5 = Fetch-Repo -Owner 'jlrouzies-fr' -Repo 'DLSS5-Feeder' -DestName 'DLSS5-Feeder' -Note 'MIT DLAA contract reference'
$prov += "## jlrouzies-fr/DLSS5-Feeder"
$prov += "- repo: https://github.com/jlrouzies-fr/DLSS5-Feeder"
$prov += "- commit: $($f5.sha) ($($f5.date))"
$prov += "- license: $($f5.license)"
$prov += "- purpose: proven genuine-DLSS/DLAA contract reference (synthetic DLAA -> DLSS5 addon -> feature 18)"
$prov += ""
Set-Content -Path $provFile -Value ($prov -join "`n") -Encoding UTF8
Write-Host "provenance written: $provFile"
