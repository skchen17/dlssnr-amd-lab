from scripts.audit_nr_plan_ownership import audit


def test_captured_graph_is_not_deployment_ready():
    result = audit({'owns_workspace': True, 'owns_weights': True,
                    'owns_graph_source': False, 'owns_graph_executable': True,
                    'graph_references_external_allocations': True,
                    'deployment_ready': False, 'arena_region_count': 24,
                    'frame_bindings_abi_version': 2})
    assert not result['deployment_ready']
    assert 'owns_graph_source' in result['failures']
    assert 'graph_references_external_allocations=false' in result['failures']


def test_fully_owned_recorder_graph_passes():
    result = audit({'owns_workspace': True, 'owns_weights': True,
                    'owns_graph_source': True, 'owns_graph_executable': True,
                    'graph_references_external_allocations': False,
                    'deployment_ready': True, 'arena_region_count': 24,
                    'frame_bindings_abi_version': 3})
    assert result['deployment_ready']
    assert not result['failures']


def test_v2_binding_is_a_transitional_runtime_not_deployment():
    result = audit({'owns_workspace': True, 'owns_weights': True,
                    'owns_graph_source': True, 'owns_graph_executable': True,
                    'graph_references_external_allocations': False,
                    'deployment_ready': True, 'arena_region_count': 24,
                    'frame_bindings_abi_version': 2})
    assert not result['deployment_ready']
    assert 'frame_bindings_abi_version>=3' in result['failures']
