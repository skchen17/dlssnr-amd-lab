param([string]$GameDirectory='C:\DATA\GAME\GODOFWAR')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$dll=(Resolve-Path -LiteralPath (Join-Path $GameDirectory 'amd_fidelityfx_dx12.dll')).Path
$audit=Get-Content -LiteralPath (Join-Path $repo 'results\20260905_gowr_target_audit\manifest.json') -Raw | ConvertFrom-Json
$expected=@($audit.inventory | Where-Object name -eq 'amd_fidelityfx_dx12.dll')
if ($expected.Count -ne 1 -or (Get-FileHash -LiteralPath $dll).Hash -ne $expected[0].sha256) { throw 'Provider differs from audited file; repeat static audit first.' }
$signature=Get-AuthenticodeSignature -LiteralPath $dll
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Advanced Micro Devices') { throw 'Expected valid AMD publisher signature.' }
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_provider_probe
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$result=Join-Path $repo ('results\{0}_ffx_provider_query' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $result | Out-Null
& (Join-Path $repo 'build\ffx_provider_probe.exe') $dll (Join-Path $result 'manifest.json')
if ($LASTEXITCODE -ne 0) { throw 'Provider query ABI probe failed' }
if ((Get-FileHash -LiteralPath $dll).Hash -ne $expected[0].sha256) { throw 'Provider changed during probe' }
Write-Host "Result: $result"
