"""Render an audited RGBA16F FFX output as an 8-bit tone-mapped PNG preview."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def render(run: Path, output: Path, folder='output_boundary', artifact='boundary_output.raw'):
    source = run / folder
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8-sig'))
    if metadata.get('format') != 10 or metadata.get('raw_bytes') != metadata.get('width') * metadata.get('height') * 8:
        raise ValueError('expected audited RGBA16F metadata')
    raw = (source / artifact).read_bytes()
    if len(raw) != metadata['raw_bytes']:
        raise ValueError('raw byte count')
    rgba = np.frombuffer(raw, dtype='<f2').astype(np.float32).reshape(metadata['height'], metadata['width'], 4)
    if not np.isfinite(rgba).all():
        raise ValueError('non-finite output')
    rgb = np.maximum(rgba[:, :, :3], 0.0)
    rgb = rgb / (1.0 + rgb)
    rgb = np.power(rgb, 1.0 / 2.2)
    image = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    if output.exists():
        raise FileExistsError('refusing to overwrite preview')
    image.save(output)
    return dict(width=image.width, height=image.height, tone_map='Reinhard then gamma 1/2.2', source_channels='RGB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--folder', default='output_boundary')
    parser.add_argument('--artifact', default='boundary_output.raw')
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.output, args.folder, args.artifact), indent=2))
