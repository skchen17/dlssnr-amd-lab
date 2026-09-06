"""Bounded hardware validation of the resident runtime; game must be closed."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

try:
    from .resident_full_graph import ResidentGraph, FRAME_BYTES, replay
except ImportError:
    from resident_full_graph import ResidentGraph, FRAME_BYTES, replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--nvcuda', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--control-reference', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    # Read-only process gate, before any GPU initialization.
    check = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                            'if(Get-Process GoWR -ErrorAction SilentlyContinue){exit 1}'], timeout=15)
    if check.returncode:
        raise RuntimeError('GoWR is running; close normally before independent GPU tests')
    color = args.input.read_bytes()
    reference, control = args.reference.read_bytes(), args.control_reference.read_bytes()
    alternate = np.frombuffer(color, dtype='<f2').reshape(360, 640, 4)[:, ::-1].copy().tobytes()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'runs': [], 'game_runtime_verified': False,
              'dlss5_quality_verified': False}
    path = args.output / 'validation.json'
    graph = None
    try:
        graph = ResidentGraph(args.plan.resolve(), args.nvcuda.resolve(),
                              progress=lambda x: print(json.dumps(x), flush=True))
        report['runtime'] = graph.info
        outputs = []
        sequence = [(color, color, True), (color, color, False),
                    (bytes(FRAME_BYTES), color, False), (color, color, False),
                    (alternate, alternate, False), (alternate, alternate, True)]
        for i, (neural, post, sync) in enumerate(sequence):
            raw, info = graph.infer(neural, post_data=post, diagnostic_sync_each=sync)
            (args.output / f'frame{i + 1}.raw').write_bytes(raw)
            outputs.append(raw)
            report['runs'].append(info)
            replay.write_progress(path, report)
            print(json.dumps(info), flush=True)
        report['checks'] = {
            'real_matches_offline': all(outputs[i] == reference for i in (0, 1, 3)),
            'neural_only_zero_matches_offline': outputs[2] == control,
            'neural_response_present': outputs[1] != outputs[2],
            'changed_live_pre_and_post_changes_output': outputs[3] != outputs[4],
            'alternate_sync_modes_exact': outputs[4] == outputs[5],
            'model_immutable': replay.sha256(replay.d_to_h(graph.api, graph.model.value,
                graph.plan['model_arena']['bytes'], 'verify model')) == graph.info['model_sha256'],
            'all_156_slots_every_frame': all(r['slot_count'] == 156 for r in report['runs']),
        }
        report['status'] = 'PASS' if all(report['checks'].values()) else 'FAIL'
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['failure'] = str(exc)
        raise
    finally:
        replay.write_progress(path, report)
        if graph:
            graph.close()
    print(json.dumps({'status': report['status'], 'checks': report['checks']}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
