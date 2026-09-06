param([string]$Python='C:\DATA\Tools\ANACONDA\python.exe')
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only ffx_bootstrap
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
$result=Join-Path $repo ('results\{0}_ffx_bootstrap_test' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $result | Out-Null
$launcher=Join-Path $repo 'build\ffx_bootstrap_launcher.exe'
$fixture=Join-Path $repo 'build\ffx_bootstrap_host.exe'
$observer=Join-Path $repo 'build\ffx_observer.dll'
$before=(Get-FileHash -LiteralPath $fixture).Hash
$positive=Join-Path $result 'positive with spaces'
& $launcher $fixture $observer $positive wait-test
if ($LASTEXITCODE -ne 0) { throw "Positive bootstrap test failed: $result" }
$savedHash=(Get-FileHash -LiteralPath (Join-Path $positive 'bootstrap.json')).Hash
& $launcher $fixture $observer $positive wait-test
if ($LASTEXITCODE -eq 0 -or (Get-FileHash -LiteralPath (Join-Path $positive 'bootstrap.json')).Hash -ne $savedHash) { throw 'Existing result overwritten or accepted' }
# Missing-export fixture: only copy our generated synthetic DLL, never game DLLs.
$negativeDll=Join-Path $result 'missing_export_dll'
New-Item -ItemType Directory -Path $negativeDll | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'build\ffx_observer_fixture.dll') -Destination (Join-Path $negativeDll 'ffx_observer.dll')
$negative=Join-Path $result 'negative'
& $launcher $fixture (Join-Path $negativeDll 'ffx_observer.dll') $negative wait-test
if ($LASTEXITCODE -eq 0) { throw 'Missing export unexpectedly accepted' }
$failed=Get-Content -LiteralPath (Join-Path $negative 'bootstrap.json') -Raw | ConvertFrom-Json
if (Get-Process -Id $failed.pid -ErrorAction SilentlyContinue) { throw 'Failed newly created child still running' }
if ((Get-FileHash -LiteralPath $fixture).Hash -ne $before) { throw 'Fixture image changed on disk' }
@{existing_result_preserved=$true;negative_child_exited=$true;fixture_sha256_before=$before;fixture_sha256_after=(Get-FileHash -LiteralPath $fixture).Hash} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $result 'checks.json') -Encoding utf8
& $Python (Join-Path $PSScriptRoot 'analyze_ffx_bootstrap.py') $result
if ($LASTEXITCODE -ne 0) { throw "Bootstrap evidence failed: $result" }
Write-Host "Result: $result"
