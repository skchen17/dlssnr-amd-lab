"""Prepare a byte-preserving game-frame crop for fixed-geometry color diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(session: Path, output: Path, x: int | None = None, y: int | None = None) -> dict:
    folder = session / 'output_network'
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8-sig'))
    audit = json.loads((session / 'output_network_analysis.json').read_text(encoding='utf-8-sig'))
    source = folder / 'boundary_before.raw'
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    width, height = int(meta['width']), int(meta['height'])
    if (meta.get('format') != 10 or len(raw) != width * height * 8
            or len(raw) != meta.get('raw_bytes') or not meta.get('fence_completed')
            or not meta.get('game_frame_capture_verified')
            or digest.lower() != audit['before_sha256'].lower()):
        raise ValueError('capture metadata/hash/provenance mismatch')
    if width < 640 or height < 360:
        raise ValueError('source too small for a byte-preserving crop')
    x = (width - 640) // 2 if x is None else x
    y = (height - 360) // 2 if y is None else y
    if x < 0 or y < 0 or x + 640 > width or y + 360 > height:
        raise ValueError('crop outside source')
    image = np.frombuffer(raw, dtype='<f2').reshape(height, width, 4)
    crop = image[y:y + 360, x:x + 640].copy()
    if not np.isfinite(crop).all() or float(crop[..., :3].astype('f4').std()) == 0:
        raise ValueError('crop must be finite and nonconstant')
    payload = crop.tobytes()
    report = dict(status='PREPARED_NOT_EXECUTED', source=str(source.resolve()),
                  source_sha256=digest, source_resolution=[width, height],
                  crop_xywh=[x, y, 640, 360], input_sha256=hashlib.sha256(payload).hexdigest().upper(),
                  bytes=len(payload), transform='byte_exact_crop_no_rescale_no_tonemap',
                  rgb_min=float(crop[..., :3].min()), rgb_max=float(crop[..., :3].max()),
                  rgb_above_one=int(np.count_nonzero(crop[..., :3] > 1)),
                  input_semantics='captured_FFX_output_color_only',
                  full_frame=False, temporal_inputs_bound=False, rtx_teacher_available=False,
                  nr_input_color_space_verified=False, dlss5_quality_verified=False)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'input_rgba16f.raw').write_bytes(payload)
    (output / 'input_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.session, args.output, args.x, args.y), indent=2))
