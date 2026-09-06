# run_module_trace_selftest.ps1 — Phase E readiness gate for module_trace.
#
# Verifies the full load->resolve->unload chain is logged:
# baseline, hooks_installed, load x3 (mt/dll_a/dll_b), getprocaddr, free.
# Only when this is READY may "no nvapi64/nvcuda load observed" even be
# treated as weak evidence (Round 2 Phase E rule).
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo 'build\module_trace_selftest.exe'
# Rebuild the tracer and its test fixtures together. The tracer's JSON schema and
# escaping behavior are part of this readiness gate, so stale binaries are unsafe.
& (Join-Path $PSScriptRoot 'build_all.ps1') -Only module_trace_selftest,module_trace
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $repo "results\${ts}_module_trace_selftest\module_trace.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

& $exe (Join-Path $repo 'build\module_trace.dll') `
       (Join-Path $repo 'build\test_dll_a.dll') `
       $log
$code = $LASTEXITCODE
Copy-Item $log (Join-Path (Split-Path $log) 'selftest_result.log') -Force -ErrorAction SilentlyContinue
Write-Host "module_trace selftest exit=$code log=$log"
exit $code
