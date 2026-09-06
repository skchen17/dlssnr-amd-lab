param(
    [string]$GraphDir = '',
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $GraphDir) {
    $GraphDir = Join-Path $repo 'results\20260831_002356_rtx5070_feature18_full_frame'
}
if (-not $OutputDir) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputDir = Join-Path $repo "results\${stamp}_kernel_inventory"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$sequence = @(Import-Csv -LiteralPath (Join-Path $GraphDir 'frame_001_sequence.csv'))
$moduleMap = @(Import-Csv -LiteralPath (Join-Path $GraphDir 'module_map.csv'))

function Get-Class([string]$name) {
    if ($name -eq 'cc_cb_clear' -or $name -eq 'cg2r_copy_kernel') { return 'boundary' }
    if ($name -match 'pre_block|post_block|repack|upsample') { return 'io_transform' }
    return 'neural_core'
}
function Get-Stage([string]$name) {
    if ($name -eq 'cc_cb_clear') { return 0 }
    if ($name -match 'pre_block') { return 1 }
    if ($name -match 'post_block') { return 9 }
    if ($name -match 'swin_1h') { return 2 }
    if ($name -match 'swin_2h') { return 3 }
    if ($name -match 'swin_4h') { return 4 }
    if ($name -match 'swin_8h') { return 5 }
    if ($name -match 'split_swin_16h') { return 6 }
    if ($name -match 'vit_1d') { return 7 }
    if ($name -match '^cc_dec_') { return 8 }
    if ($name -eq 'cg2r_copy_kernel') { return 10 }
    return 99
}

$used = foreach ($group in ($sequence | Group-Object function_name)) {
    $rows = @($group.Group)
    $first = $rows[0]
    $grids = @($rows | ForEach-Object { "$($_.grid_x)x$($_.grid_y)x$($_.grid_z)" } | Sort-Object -Unique)
    $blocks = @($rows | ForEach-Object { "$($_.block_x)x$($_.block_y)x$($_.block_z)" } | Sort-Object -Unique)
    $sizes = @($rows.param_size | ForEach-Object { [int]$_ } | Sort-Object -Unique)
    [pscustomobject][ordered]@{
        stage = Get-Stage $group.Name
        classification = Get-Class $group.Name
        function_name = $group.Name
        module_dll_offset = $first.module_dll_offset
        module_fnv1a64 = $first.module_fnv1a64
        slot_count = $rows.Count
        first_slot = ($rows.slot | ForEach-Object { [int]$_ } | Measure-Object -Minimum).Minimum
        last_slot = ($rows.slot | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum
        parameter_sizes = $sizes -join ';'
        parameter_bytes_per_frame = ($rows.param_size | ForEach-Object { [int]$_ } | Measure-Object -Sum).Sum
        all_parameters_stable_across_5_frames = (@($rows | Where-Object { $_.parameters_stable_across_5_frames -ne 'True' }).Count -eq 0)
        grid_variants = $grids -join ';'
        block_variants = $blocks -join ';'
        dynamic_shared_variants = (@($rows.dynamic_shared | Sort-Object -Unique) -join ';')
    }
}
$used = @($used | Sort-Object stage,first_slot)
$used | Export-Csv -LiteralPath (Join-Path $OutputDir 'used_functions.csv') -NoTypeInformation -Encoding utf8

$modules = foreach ($module in $moduleMap) {
    $slots = @($sequence | Where-Object module_dll_offset -eq $module.dll_offset)
    $functions = @($slots.function_name | Sort-Object -Unique)
    [pscustomobject][ordered]@{
        dll_offset = $module.dll_offset
        fnv1a64 = $module.fnv1a64
        container_size = $module.size
        created_function_count = $module.function_count
        used_function_count = $functions.Count
        used_slot_count = $slots.Count
        parameter_bytes_per_frame = ($slots.param_size | ForEach-Object { [int]$_ } | Measure-Object -Sum).Sum
        first_slot = if ($slots) { ($slots.slot | ForEach-Object { [int]$_ } | Measure-Object -Minimum).Minimum } else { -1 }
        last_slot = if ($slots) { ($slots.slot | ForEach-Object { [int]$_ } | Measure-Object -Maximum).Maximum } else { -1 }
        used_functions = $functions -join ';'
    }
}
$modules | Sort-Object first_slot | Export-Csv -LiteralPath (Join-Path $OutputDir 'module_usage.csv') -NoTypeInformation -Encoding utf8

$classes = @($used | Group-Object classification | ForEach-Object {
    [ordered]@{ name = $_.Name; functions = $_.Count; slots = ($_.Group.slot_count | Measure-Object -Sum).Sum }
})
$summary = [ordered]@{
    schema = 1
    experiment = 'kernel_inventory'
    source_graph = (Resolve-Path $GraphDir).Path
    source_graph_structural_sha256 = 'be12be4011601716284d43b8053f00aa4936c13006288e001917a1119ad52064'
    modules_loaded = $moduleMap.Count
    functions_created = 96
    functions_used = $used.Count
    graph_slots = $sequence.Count
    parameter_bytes_per_frame = ($sequence.param_size | ForEach-Object { [int]$_ } | Measure-Object -Sum).Sum
    stable_parameter_slots = @($sequence | Where-Object parameters_stable_across_5_frames -eq 'True').Count
    variable_parameter_slots = @($sequence | Where-Object parameters_stable_across_5_frames -ne 'True').Count
    classifications = $classes
    first_boundary_target = 'cc_cb_clear'
    first_neural_target = 'cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'
    counts_as_s6 = $false
}
$summary | ConvertTo-Json -Depth 5 | Out-File -LiteralPath (Join-Path $OutputDir 'summary.json') -Encoding utf8
Write-Host "kernel inventory: modules=$($moduleMap.Count) functions=$($used.Count) slots=$($sequence.Count) output=$OutputDir"
