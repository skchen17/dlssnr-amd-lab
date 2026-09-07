import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_wide_norm_uses_one_cross_family_resident_arena():
    source = (ROOT / 'scripts' / 'native_matrix_fusion.py').read_text(encoding='utf-8')
    assert 'self.wide_norm_arenas={}' in source
    assert 'key=(projection.device,projection.dtype)' in source
    assert 'capacity=max(vectors*2,vectors)' in source
    assert 'value[:vectors].view(windows,heads,64,32)' in source


def test_grown_arena_keeps_graph_referenced_storage_alive():
    source = (ROOT / 'scripts' / 'native_matrix_fusion.py').read_text(encoding='utf-8')
    assert 'self.wide_norm_retired=[]' in source
    assert 'self.wide_norm_retired.append(arena[1])' in source
