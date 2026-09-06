"""Independent checks of real FFX dispatch artifacts; not an NR quality gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze_provider(directory: Path) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest.get('status') != 'DISPATCH_READBACK_PASS':
        raise ValueError('native dispatch did not pass')
    if manifest.get('debug_errors') != 0 or manifest.get('completed_fences') != 4:
        raise ValueError('GPU completion/debug gate failed')
    if manifest.get('output_poisoned_nan') is not True:
        raise ValueError('output-write sentinel gate missing')
    render=manifest.get('render_size',[320,180]);output=manifest.get('output_size',[640,360])
    if not isinstance(render,list) or not isinstance(output,list) or len(render)!=2 or len(output)!=2 or any(type(v) is not int or not 1<=v<=4096 for v in render+output):
        raise ValueError('invalid declared dimensions')
    rw,rh=render;ow,oh=output
    frames = []
    inputs = []
    hashes = []
    for i in range(4):
        raw = (directory / f'output_{i}.raw').read_bytes()
        if len(raw) != ow * oh * 8:
            raise ValueError('output size mismatch')
        frame = np.frombuffer(raw, '<f2').astype(np.float32).reshape(oh, ow, 4)
        if not np.isfinite(frame).all():
            raise ValueError('nonfinite output')
        if float(frame[..., :3].std()) < 0.02:
            raise ValueError('blank or nearly constant output')
        source = (directory / f'input_{i}.raw').read_bytes()
        if len(source) != rw * rh * 8:
            raise ValueError('input size mismatch')
        inputs.append(source)
        frames.append(frame)
        hashes.append(hashlib.sha256(raw).hexdigest())
    if not (hashes[0] == hashes[1] == hashes[3]) or hashes[0] == hashes[2]:
        raise ValueError('reset repeatability/input sensitivity failed')
    if not (inputs[0] == inputs[1] == inputs[3]) or inputs[0] == inputs[2]:
        raise ValueError('input experiment invalid')
    difference = float(np.abs(frames[0][..., :3] - frames[2][..., :3]).mean())
    if difference < .01:
        raise ValueError('input response too small')
    return dict(status='ARTIFACT_PASS', output_sha256=hashes, input_response_mae=difference,
                rgb_min=float(frames[0][..., :3].min()), rgb_max=float(frames[0][..., :3].max()),
                rgb_std=float(frames[0][..., :3].std()), dlss_nr_verified=False,
                game_runtime_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = {}
    for name in ('fsr3', 'fsr4'):
        try:
            report[name] = analyze_provider(args.directory / name)
        except (ValueError, OSError, KeyError) as exc:
            report[name] = dict(status='ARTIFACT_FAIL', error=str(exc))
    report['status'] = 'PASS' if all(v['status'] == 'ARTIFACT_PASS' for v in report.values()) else 'FAIL'
    (args.directory / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
