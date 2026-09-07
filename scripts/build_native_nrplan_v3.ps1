param(
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$WeightCache = '',
    [string]$Model = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not [IO.Path]::IsPathFullyQualified($OutputDirectory) -or (Test-Path -LiteralPath $OutputDirectory)) {
    throw 'OutputDirectory must be a new absolute path'
}
if (-not $WeightCache) { $WeightCache = Join-Path $repo 'local_models\gfx1201_stage_cache_v3\manifest.json' }
if (-not $Model) { $Model = Join-Path $repo 'local_models\native_single_color_v1\model.json' }
foreach ($path in @($WeightCache, $Model)) {
    if (-not [IO.Path]::IsPathFullyQualified($path) -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "missing absolute private input: $path"
    }
}
$python = Join-Path $repo '.venv-rocm\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'missing project ROCm Python environment' }

# Compile-only plus CPU topology generation. This script deliberately never
# imports torch.cuda, launches HIP work, or changes the profiler safety lock.
& (Join-Path $repo 'tools\native_nr_plan\build.ps1') -OutputDirectory $OutputDirectory
if ($LASTEXITCODE) { throw 'native NRPlan compile failed' }

$arena = Join-Path $OutputDirectory 'arena_1080p_approx.json'
$wide = Join-Path $OutputDirectory 'swin32_256_topology.json'
$split = Join-Path $OutputDirectory 'split512_topology.json'
$vit = Join-Path $OutputDirectory 'vit_topology.json'
$bottleneck = Join-Path $OutputDirectory 'bottleneck_topology.json'
$transitions = Join-Path $OutputDirectory 'transition_topology.json'
$edges = Join-Path $OutputDirectory 'edge_topology.json'
& $python (Join-Path $repo 'scripts\native_nr_arena.py') --width 1920 --height 1080 --precision-profile approx_fp8 --output $arena
& $python (Join-Path $repo 'scripts\build_native_stage_topology.py') --weights $WeightCache --arena $arena --model $Model --output $wide
& $python (Join-Path $repo 'scripts\build_native_split512_topology.py') --weights $WeightCache --arena $arena --model $Model --output $split
& $python (Join-Path $repo 'scripts\build_native_vit_topology.py') --weights $WeightCache --arena $arena --output $vit
& $python (Join-Path $repo 'scripts\build_native_bottleneck_topology.py') --weights $WeightCache --arena $arena --output $bottleneck
& $python (Join-Path $repo 'scripts\build_native_transition_topology.py') --weights $WeightCache --arena $arena --wide $wide --output $transitions
& $python (Join-Path $repo 'scripts\build_native_edge_topology.py') --weights $WeightCache --model $Model --output $edges
& $python (Join-Path $repo 'scripts\audit_native_topology_coverage.py') --wide $wide --split512 $split --vit $vit --bottleneck $bottleneck --transitions $transitions --edges $edges --output (Join-Path $OutputDirectory 'topology_coverage.json')
if ($LASTEXITCODE) { throw 'native NRPlan CPU topology generation failed' }

[ordered]@{
    schema = 1
    status = 'COMPILE_AND_CPU_TOPOLOGY_ONLY_NOT_RUNTIME_ACCEPTED'
    resolution = @(1920, 1080)
    gpu_executed = $false
    profiler_safety_lock_modified = $false
    output = [IO.Path]::GetFullPath($OutputDirectory)
} | ConvertTo-Json | Out-File (Join-Path $OutputDirectory 'assembly_summary.json') -Encoding utf8
