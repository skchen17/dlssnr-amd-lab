from scripts.native_nr_arena import (ALIGNMENT, build_arena_plan,
                                     stage_feature_bytes, wide_window_elements)


def test_1080_arena_is_deterministic_aligned_and_overlap_free():
    first = build_arena_plan(1920, 1080)
    assert first == build_arena_plan(1920, 1080)
    assert first['overlap_free']
    assert first['workspace_bytes'] % ALIGNMENT == 0
    assert all(r['offset'] % r['alignment'] == 0 for r in first['regions'])
    assert len({r['name'] for r in first['regions']}) == len(first['regions'])


def test_odd_size_uses_recovered_128_pixel_padding():
    sizes = stage_feature_bytes(641, 361)
    assert sizes['c32'] == (768 // 2) * (384 // 2) * 32 * 2
    assert sizes['c64'] == (768 // 4) * (384 // 4) * 64 * 2


def test_temporal_bindings_are_reserved_before_activations():
    plan = build_arena_plan(1920, 1080)
    names = [r['name'] for r in plan['regions']]
    assert names[:8] == ['frame.input_rgba16f', 'frame.output_residual_rgba16f',
                         'frame.history_rgba16f', 'frame.next_history_rgba16f',
                         'frame.motion_rg16f', 'frame.depth_r32f',
                         'frame.exposure', 'frame.controls']


def test_approximate_arena_uses_true_one_byte_resident_features():
    strict = build_arena_plan(1920, 1080, 'strict_fp16')
    approx = build_arena_plan(1920, 1080, 'approx_fp8')
    strict_regions = {r['name']: r for r in strict['regions']}
    approx_regions = {r['name']: r for r in approx['regions']}
    assert approx['resident_item_bytes'] == 1
    assert approx_regions['c256.ping']['bytes'] * 2 == strict_regions['c256.ping']['bytes']
    assert approx_regions['operator.fp16_workspace']['bytes'] == strict_regions['operator.fp16_workspace']['bytes']
    wide = max(wide_window_elements(1920, 1080).values())
    assert all(approx_regions[f'operator.fp8_{name}']['bytes'] >= wide
               for name in ('grouped', 'post', 'q', 'k', 'v', 'value'))
    assert approx_regions['operator.qkv_projection_fp16']['bytes'] >= wide * 6
    # C32 whole-window Q/K/V scratch is explicit in the approximate plan, so
    # the complete arena can be slightly larger even though every persistent
    # stage activation itself is half the strict-path size.
    assert approx['workspace_bytes'] < 1_000_000_000
