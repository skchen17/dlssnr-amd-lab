# Ledger proprietary files BEFORE first use (spec section 2 requirement).
# Writes docs/PROPRIETARY_FILES.md; never copies the binaries into the repo.
$ErrorActionPreference = 'Continue'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot
$files = @(
    (Join-Path $repo '..\file\nvngx_dlss.dll'),
    (Join-Path $repo '..\file\nvngx_dlssnr.dll')
)
$lines = @()
$lines += '# PROPRIETARY_FILES.md'
$lines += ''
$lines += 'Pre-use ledger per spec section 2 (filename / version / SHA-256 / Authenticode).'
$lines += 'Binaries stay in the user-supplied folder outside the repo; only metadata is recorded.'
$lines += "Recorded: " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
$lines += ''
foreach ($f in $files) {
    $full = (Resolve-Path $f).Path
    $lines += "## " + (Split-Path $full -Leaf)
    $lines += "- Path (user-supplied, outside repo): $full"
    $lines += ("- Size: {0} bytes" -f (Get-Item $full).Length)
    $vi = (Get-Item $full).VersionInfo
    $lines += "- FileVersion: $($vi.FileVersion)"
    $lines += "- ProductVersion: $($vi.ProductVersion)"
    $lines += "- FileDescription: $($vi.FileDescription)"
    $lines += "- CompanyName: $($vi.CompanyName)"
    $sig = Get-AuthenticodeSignature $full
    $lines += "- Authenticode: $($sig.Status) $(if ($sig.SignerCertificate) { '(' + $sig.SignerCertificate.Subject + ')' })"
    Write-Host "hashing $full ..."
    $h = (Get-FileHash $full -Algorithm SHA256).Hash
    $lines += "- SHA-256: $h"
    $lines += ''
}
Set-Content (Join-Path $repo 'docs\PROPRIETARY_FILES.md') ($lines -join "`r`n")
Write-Host 'done'
