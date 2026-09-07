import pathlib


ROOT=pathlib.Path(__file__).resolve().parents[1]


def test_wide_layout_export_and_all_channel_families_are_explicit():
    hip=(ROOT/'tools/native_fusion_probe/native_fusion_quantize.hip').read_text(encoding='utf-8')
    wrapper=(ROOT/'scripts/native_c32_layout_fusion.py').read_text(encoding='utf-8')
    assert 'native_fusion_packed_layout' in hip
    assert 'channels!=32&&channels!=64&&channels!=128&&channels!=256&&channels!=512' in hip
    assert 'allowed-{32}' in wrapper
    assert "channels=(32,)" in wrapper


def test_full_benchmark_requires_explicit_wide_layout_selection():
    source=(ROOT/'scripts/benchmark_nr_resolution.py').read_text(encoding='utf-8')
    assert "--resident-layout-channels" in source
    assert "default='32'" in source
    assert "wide resident layouts require --c32-layout-dll" in source
