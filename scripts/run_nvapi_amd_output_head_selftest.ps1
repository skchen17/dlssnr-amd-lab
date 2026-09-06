$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $PSScriptRoot

& (Join-Path $PSScriptRoot 'build_all.ps1') `
    -Only module_trace,nvapi_amd,nvapi_amd_output_head_selftest
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$result = Join-Path $repo "results\${stamp}_nvapi_amd_output_surface"
New-Item -ItemType Directory -Force -Path $result | Out-Null
$env:MODULE_TRACE_LOG = Join-Path $result 'module_trace.log'
$env:MODULE_TRACE_AMD_INTEROP = '1'
$runtime = Join-Path $repo 'build\nvapi_output_surface_runtime'
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$testNvapi = Join-Path $runtime 'nvapi64.dll'
$testDxil = Join-Path $runtime 'cg2r_copy.dxil'
# module_trace intentionally wraps only the production drop-in basename.
Copy-Item (Join-Path $repo 'build\nvapi64_amd.dll') $testNvapi -Force
Copy-Item (Join-Path $repo 'build\cg2r_copy.dxil') $testDxil -Force
$arguments = @(
    (Join-Path $repo 'build\module_trace.dll'),
    $testNvapi,
    (Join-Path $repo 'deliverables\postblock_mma_trace_reference_20260904_154238\payload\activation_arena.raw'),
    (Join-Path $repo 'local_models\decoded_310_8\model_arena.raw'),
    (Join-Path $repo 'results\20260831_234000_full_graph_integrated_plan\params\slot154.raw'),
    (Join-Path $repo 'results\20260905_014000_output_head_resident_rx9070xt\output_rgba16f.raw'),
    (Join-Path $result 'output_rgba16f.raw'),
    (Join-Path $result 'manifest.json')
)
& (Join-Path $repo 'build\nvapi_amd_output_head_selftest.exe') @arguments
$code = $LASTEXITCODE
exit $code
