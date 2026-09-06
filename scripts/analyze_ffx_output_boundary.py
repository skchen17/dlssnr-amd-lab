"""Verify output-only boundary-copy controls, never promote them to game/NR proof."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze_ffx_dispatch_probe import analyze_provider


def compare_pair(base, captured, roundtrip=False):
    analyze_provider(base)
    analyze_provider(captured)
    a = json.loads((base / 'manifest.json').read_text(encoding='utf-8-sig'))
    b = json.loads((captured / 'manifest.json').read_text(encoding='utf-8-sig'))
    if roundtrip:
        if (a.get('output_boundary_enabled') is not False or b.get('output_boundary_enabled') is not False or
                a.get('output_roundtrip_enabled') is not False or b.get('output_roundtrip_enabled') is not True):
            raise ValueError('roundtrip on/off mode')
    elif a.get('output_boundary_enabled') is not False or b.get('output_boundary_enabled') is not True:
        raise ValueError('boundary on/off mode')
    for field in ('requested_provider', 'requested_provider_id', 'adapter_vendor', 'adapter_device', 'render_size', 'output_size'):
        if a.get(field) is None or a.get(field) != b.get(field):
            raise ValueError('provider/adapter/shape mismatch')
    if a['adapter_vendor'] != 0x1002 or any(m.get('debug_warnings') != 0 for m in (a, b)):
        raise ValueError('AMD/debug control')
    if roundtrip:
        if b.get('output_roundtrip_completed') != 4 or b.get('output_roundtrip_exact') is not True:
            raise ValueError('roundtrip completion')
    elif b.get('output_boundary_completed') != 4 or b.get('output_boundary_exact') is not True:
        raise ValueError('boundary completion')
    hashes = []
    for i in range(4):
        if (base / f'input_{i}.raw').read_bytes() != (captured / f'input_{i}.raw').read_bytes():
            raise ValueError('on/off input mismatch')
        expected = (base / f'output_{i}.raw').read_bytes()
        observed = (captured / f'boundary_{i}.raw').read_bytes()
        if observed != expected or (captured / f'output_{i}.raw').read_bytes() != expected:
            raise ValueError('on/off boundary/downstream output mismatch')
        hashes.append(hashlib.sha256(observed).hexdigest())
    return dict(status='OUTPUT_ROUNDTRIP_PROVIDER_PASS' if roundtrip else 'OUTPUT_BOUNDARY_PROVIDER_PASS', output_sha256=hashes,
                output_bytes_per_frame=len(expected), on_off_output_equal=True,
                write_back_performed=roundtrip, replacement_pixels_supplied=False,
                game_frame_capture_verified=False, dlss_nr_verified=False)


def analyze(root):
    reports = {p: compare_pair(root / p, root / (p + '_boundary')) for p in ('fsr3', 'fsr4')}
    roundtrips = {p: compare_pair(root / p, root / (p + '_roundtrip'), True) for p in ('fsr3', 'fsr4')}
    return dict(status='OUTPUT_BOUNDARY_CONTROL_PASS', providers=reports, roundtrip_providers=roundtrips,
                live_hook_validated=False, game_frame_capture_verified=False, dlss_nr_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite report')
    report = analyze(args.directory)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))
