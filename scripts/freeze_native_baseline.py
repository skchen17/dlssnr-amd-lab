"""Hash the accepted read-only assets before native reconstruction; no copying."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest().upper()


def freeze(root: Path, output: Path) -> dict:
    root = root.resolve(strict=True)
    if output.exists():
        raise FileExistsError(output)
    selected = [root / 'local_models/decoded_310_8/model_arena.raw',
                root / 'local_models/decoded_310_8/manifest.json',
                root / 'results/20260905_present_game_worker_v2/worker.json',
                root / 'results/20260905_present_game_analysis_v3/summary.json',
                root / 'results/20260905_resident_present_stage_v3/resident_present.dll',
                root / 'scripts/resident_full_graph.py',
                root / 'scripts/resident_frame_worker.py']
    selected.append(root / 'results/20260905_real_color_probe/plan.json')
    for name in ('input1.raw', 'output1.raw', 'input2.raw', 'output2.raw'):
        selected.append(root / 'results/20260905_present_game_worker_v2' / name)
    worker = json.loads(selected[2].read_text(encoding='utf-8-sig'))
    # Resolve the actual accepted plan from the worker provenance, not a guessed variant.
    for value in _strings(worker):
        if value.endswith('.json') and 'plan' in value:
            p = Path(value)
            if not p.is_absolute():
                p = root / p
            if p.is_file():
                selected.append(p.resolve())
    for plan_path in tuple(selected):
        if 'plan' not in plan_path.name or plan_path.suffix != '.json':
            continue
        plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
        for slot in plan.get('slots', []):
            for value in [slot.get('ptx'), slot.get('params', {}).get('path')]:
                if value:
                    p = Path(value)
                    selected.append(p if p.is_absolute() else plan_path.parent / p)
    records = [{'path': str(p.resolve(strict=True)), 'bytes': p.stat().st_size,
                'sha256': digest(p)} for p in dict.fromkeys(selected)]
    report = {'schema': 1, 'status': 'FROZEN_HASH_INVENTORY',
              'native_model_complete': False, 'assets': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return report


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = freeze(args.root, args.output)
    print(json.dumps({'status': report['status'], 'assets': len(report['assets'])}))
