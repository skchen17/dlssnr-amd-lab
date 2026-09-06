import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('bootstrap_analysis', Path(__file__).parents[1] / 'scripts/analyze_ffx_bootstrap.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def evidence(tmp_path):
    positive = tmp_path / 'positive with spaces'
    negative = tmp_path / 'negative'
    positive.mkdir()
    negative.mkdir()
    (positive / 'bootstrap.json').write_text(json.dumps(dict(status='BOOTSTRAP_PASS', entry_instruction_restored=True,
        child_exit_code=0, observer_profile=0, game_launched=False, texture_capture_enabled=False)))
    (positive / 'host.json').write_text(json.dumps(dict(status='HOST_PASS', original_calls=5, last_error=0xFACE, concurrent_log_read=True)))
    (negative / 'bootstrap.json').write_text('{"status":"BOOTSTRAP_FAIL"}')
    (tmp_path / 'checks.json').write_text(json.dumps(dict(existing_result_preserved=True, negative_child_exited=True,
        fixture_sha256_before='a'*64, fixture_sha256_after='a'*64)))
    events = [dict(event=n, profile=0, body_decoded=False, return_code=s) for n, s in
              zip(['ffxCreateContext','ffxQuery','ffxConfigure','ffxDispatch','ffxDestroyContext'],[0,31,23,37,29])]
    (positive / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    return tmp_path


def test_complete_evidence(evidence):
    result = module.analyze(evidence)
    assert result['status'] == 'PASS'
    assert result['game_runtime_ready'] is False


@pytest.mark.parametrize('kind', ['missing_event','status','body','image','cleanup','last_error','entry'])
def test_false_positive_rejected(evidence, kind):
    positive = evidence / 'positive with spaces'
    events = [json.loads(line) for line in (positive / 'events.jsonl').read_text().splitlines()]
    if kind in ['missing_event','status','body']:
        if kind == 'missing_event': events.pop()
        elif kind == 'status': events[3]['return_code'] = 0
        else: events[0]['body_decoded'] = True
        (positive / 'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
    else:
        path = evidence / 'checks.json' if kind in ['image','cleanup'] else positive / ('host.json' if kind == 'last_error' else 'bootstrap.json')
        value = json.loads(path.read_text())
        key = dict(image='fixture_sha256_after', cleanup='negative_child_exited', last_error='last_error', entry='entry_instruction_restored')[kind]
        value[key] = False
        path.write_text(json.dumps(value))
    assert module.analyze(evidence)['status'] == 'FAIL'
