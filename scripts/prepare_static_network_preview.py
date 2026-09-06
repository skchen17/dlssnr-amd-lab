"""Package verified offline network pixels for an explicitly labeled game preview."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scripts.analyze_full_graph_color import read_verified, rgba
except ModuleNotFoundError:
    from analyze_full_graph_color import read_verified, rgba


def panel(raw: bytes, label: str) -> bytes:
    frame = rgba(raw)
    if not np.isfinite(frame).all():
        raise ValueError('nonfinite preview')
    result = np.ones((384, 644, 4), dtype='<f2')
    result[..., :3] = [1, 0, 1]  # visible frame around a STATIC image
    result[22:382, 2:642] = frame  # all 640x360 original half pixels, no transform
    bar = Image.new('RGB', (640, 20), (15, 15, 15))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((3, 1), 'STATIC PREVIEW - ' + label + ' | NOT LIVE INFERENCE', fill='white', font=font)
    result[2:22, 2:642, :3] = np.asarray(bar).astype('f4') / 255
    return result.tobytes()


def prepare(execution: Path, destination: Path) -> dict:
    report = json.loads((execution / 'execution.json').read_text(encoding='utf-8-sig'))
    if not report.get('pass') or not report.get('color_input', {}).get('external') or report.get('rtx_intermediate_state_injection'):
        raise ValueError('requires verified external-color execution without injections')
    outputs = []
    for run in report['runs']:
        if [s['slot'] for s in run['slots'] if s.get('executed')] != list(range(156)):
            raise ValueError('incomplete graph')
        outputs.append(read_verified(Path(run['final']['path']), run['final']['sha256']))
    if len(outputs) < 2 or any(x != outputs[0] for x in outputs[1:]):
        raise ValueError('repeated outputs must match')
    color = report['color_input']
    raw = read_verified(Path(color['path']), color['sha256'])
    files = {'preview_input.rgba16f': panel(raw, 'ORIGINAL INPUT'),
             'preview_output.rgba16f': panel(outputs[0], 'NETWORK OUTPUT')}
    if any((destination / name).exists() for name in [*files, 'preview_manifest.json']):
        raise FileExistsError('preview already prepared')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(execution=str(execution.resolve()), source_input_sha256=color['sha256'],
        source_output_sha256=report['runs'][0]['final']['sha256'], model_sha256=report['model_sha256'],
        image_rect=[2, 22, 640, 360], panel_size=[644, 384], source_transform='none',
        static_offline_pixels=True, live_inference=False, same_input_rtx_quality_verified=False,
        files={name: hashlib.sha256(data).hexdigest().upper() for name, data in files.items()})
    for name, data in files.items():
        (destination / name).write_bytes(data)
    (destination / 'preview_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.execution, args.destination), indent=2))
