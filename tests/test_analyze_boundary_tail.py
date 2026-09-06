from scripts.analyze_boundary_tail import analyze


def test_draws_after_boundary_do_not_prove_shader_reads_or_accept_game():
    result=analyze([{'event':'output_boundary_tail','metadata_only':True,'close_succeeded':True,'draw':1,'draw_indexed':5}])
    assert result['same_list_gpu_commands_after_boundary']
    assert not result['shader_reads_of_target_proven'] and not result['hud_boundary_proven']
    assert not result['safe_command_list_split_implemented'] and not result['game_native_network_accepted']


def test_no_trace_is_not_no_consumer():
    assert analyze([])['status']=='INSUFFICIENT_EVIDENCE'


def test_failed_close_cannot_be_accepted():
    result=analyze([{'event':'output_boundary_tail','metadata_only':True,'close_succeeded':False,'dispatch':1}])
    assert result['status']=='INSUFFICIENT_EVIDENCE'
