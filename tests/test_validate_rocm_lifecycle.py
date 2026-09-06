import json
from pathlib import Path
import sys
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from validate_rocm_lifecycle import acceptance, supervise, summarize_event_times,configure_fault_logging,host_timeout_for
import pytest

PHASES = ['gpu_work_complete', 'resources_released', 'python_atexit']
CHILD = {'checks_pass': True, 'allocated_after_release_bytes': 0}


def test_saved_result_is_not_successful_exit():
    assert not acceptance(None, True, CHILD, PHASES)
    assert not acceptance(1, False, CHILD, PHASES)
    assert not acceptance(0, False, CHILD, PHASES[:-1])
    assert not acceptance(0, False, dict(CHILD, allocated_after_release_bytes=256), PHASES)
    assert not acceptance(0, False, dict(CHILD, checks_pass=False), PHASES)
    assert acceptance(0, False, CHILD, PHASES)


def test_invalid_gpu_timing_cannot_hide_in_positive_total():
    for values in ([47.4, -.3], [float('nan')], [float('inf')], []):
        summary = summarize_event_times(values)
        assert not summary['event_timing_valid'] and summary['gpu_event_sum_ms'] is None
    assert summarize_event_times([1., 2.])['gpu_event_sum_ms'] == 3.


def test_supervisor_requires_child_evidence(tmp_path):
    assert not supervise([sys.executable, '-c', 'pass'], tmp_path, timeout=5)
    report = json.loads((tmp_path / 'manifest.json').read_text())
    assert report['normal_exit'] and not report['pass']
    assert not report['historical_hang_root_cause_resolved']


def test_supervisor_success_is_not_full_graph_claim(tmp_path):
    (tmp_path / 'child.json').write_text(json.dumps(CHILD))
    (tmp_path / 'phases.jsonl').write_text('\n'.join(json.dumps({'phase': p}) for p in PHASES))
    assert supervise([sys.executable, '-c', 'pass'], tmp_path, timeout=5)
    report = json.loads((tmp_path / 'manifest.json').read_text())
    assert not report['native_graph_complete'] and not report['image_quality_verified']


def test_timeout_does_not_kill_or_retry_child(tmp_path, monkeypatch):
    calls = []
    class Process:
        pid = 12345
        def wait(self, timeout):
            raise subprocess.TimeoutExpired('diagnostic', timeout)
        def kill(self):
            raise AssertionError('must not imply GPU cancellation')
    def launch(*args, **kwargs):
        calls.append(args)
        return Process()
    monkeypatch.setattr(subprocess, 'Popen', launch)
    assert not supervise(['diagnostic'], tmp_path, timeout=1)
    report = json.loads((tmp_path/'manifest.json').read_text())
    assert len(calls) == 1 and report['child_left_running_on_timeout']
    assert not report['normal_exit'] and not report['automatic_retry']


def test_monitor_keeps_fatal_trace_without_concurrent_timer_walk():
    calls=[]
    class Handler:
        def enable(self,**kwargs):
            calls.append(kwargs)
        def dump_traceback_later(self,*args,**kwargs):
            raise AssertionError('periodic stack walker is held pending crash review')
    configure_fault_logging('trace',Handler())
    assert calls==[{'file':'trace'}]


def test_longer_host_budget_does_not_enable_arbitrary_gpu_batches():
    assert host_timeout_for('whole_frame640',1)==90
    assert host_timeout_for('whole_frame640',12)==90
    assert host_timeout_for('whole_frame640',100)==360
    with pytest.raises(ValueError):
        host_timeout_for('vit_bottleneck',100)
    with pytest.raises(ValueError):
        host_timeout_for('whole_frame640',1000)
