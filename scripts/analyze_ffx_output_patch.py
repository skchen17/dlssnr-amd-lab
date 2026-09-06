"""Validate an exact 8x8 supplied-pixel patch, in isolated or live output."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct

from analyze_ffx_state_trace import read_events

MARKERS = ('boundary_output_arm', 'boundary_output_recorded',
           'boundary_output_submitted', 'boundary_output_complete')
PATCH = struct.pack('<4H', 0x3c00, 0x0000, 0x3c00, 0x3c00)


def pixel_proof(before, after, width, height):
    if len(before) != width * height * 8 or len(after) != len(before):
        raise ValueError('raw byte count')
    changed_pixels = changed_components = 0
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 8
            old, new = before[offset:offset + 8], after[offset:offset + 8]
            if x < 8 and y < 8:
                if new != PATCH:
                    raise ValueError('patch pixel mismatch')
            elif new != old:
                raise ValueError('outside patch changed')
            if new != old:
                changed_pixels += 1
                changed_components += sum(a != b for a, b in zip(struct.unpack('<4H', old), struct.unpack('<4H', new)))
    if not changed_pixels or not changed_components:
        raise ValueError('patch caused no change')
    return changed_pixels, changed_components


def analyze(root: Path, live=True):
    root = Path(root); events = read_events(root / 'session.jsonl')
    if any(e.get('event') == 'failure' for e in events):
        raise ValueError('session failure')
    markers = [e for e in events if e.get('event') in MARKERS]
    if tuple(e.get('event') for e in markers) != MARKERS:
        raise ValueError('patch lifecycle order/count')
    returns = [e for e in events if e.get('event') == 'output_boundary_return']
    accepted = [e for e in returns if e.get('batch_supported') is True and e.get('lifecycle_match') is True]
    if len(accepted) != 1:
        raise ValueError('accepted output boundary count')
    selected = [markers[0], accepted[0], *markers[1:]]
    if [events.index(e) for e in selected] != sorted(events.index(e) for e in selected):
        raise ValueError('patch lifecycle order/count')
    if len({e.get('window') for e in selected + returns}) != 1 or selected[0].get('window', 0) < 1:
        raise ValueError('patch lifecycle window')
    for candidate in returns:
        if (type(candidate.get('batch_supported')) is not bool or type(candidate.get('lifecycle_match')) is not bool or
                candidate.get('original_callback_returned') is not True or candidate.get('original_forward_calls') != 1 or
                candidate.get('capture_authorized') is not False):
            raise ValueError('output boundary contract')
    _, recorded, submitted, complete = selected[1:]
    if (recorded.get('isolated_host_only') is live or submitted.get('isolated_host_only') is live or
            any(e.get('write_back_performed') is not True or e.get('replacement_pixels_supplied') is not True
                for e in (markers[0], recorded, submitted, complete)) or
            recorded.get('game_frame_capture_verified') is not False or submitted.get('game_frame_capture_verified') is not False or
            submitted.get('submitted_identity_matches') != 1 or
            not isinstance(submitted.get('identity_aliases'), int) or submitted['identity_aliases'] < 1 or
            not isinstance(submitted.get('closed_identity_aliases'), int) or submitted['closed_identity_aliases'] < 1 or
            type(submitted.get('raw_pointer_match')) is not bool or complete.get('fence_completed') is not True or
            complete.get('game_frame_capture_verified') is not live or complete.get('nr_verified') is not False or
            complete.get('patch_rect') != [0, 0, 8, 8]):
        raise ValueError('game/patch provenance')
    output = root / 'output_patch'; metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8-sig'))
    width, height, fmt = metadata.get('width'), metadata.get('height'), metadata.get('format')
    if (type(width) is not int or type(height) is not int or not 8 <= width <= 8192 or not 8 <= height <= 8192 or fmt != 10 or
            metadata.get('raw_bytes') != width * height * 8 or metadata.get('fence_completed') is not True or
            metadata.get('write_back_performed') is not True or metadata.get('replacement_pixels_supplied') is not True or
            metadata.get('patch_rect') != [0, 0, 8, 8] or metadata.get('game_frame_capture_verified') is not live or
            metadata.get('nr_verified') is not False):
        raise ValueError('patch metadata')
    before = (output / 'boundary_before.raw').read_bytes(); after = (output / 'boundary_output.raw').read_bytes()
    if len(before) != width * height * 8 or len(after) != len(before):
        raise ValueError('raw byte count')
    words = array('H'); words.frombytes(after)
    if any((word & 0x7c00) == 0x7c00 for word in words):
        raise ValueError('non-finite half output')
    changed_pixels, changed_components = pixel_proof(before, after, width, height)
    values = []
    stride = max(1, width * height // 65536)
    for pixel in range(0, width * height, stride): values.extend(struct.unpack_from('<4e', after, pixel * 8))
    mean = math.fsum(values) / len(values); variance = math.fsum((v - mean) ** 2 for v in values) / len(values)
    if max(values) - min(values) <= 1e-5 or variance <= 1e-12:
        raise ValueError('blank/constant output')
    if live:
        provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8-sig'))
        if (provenance.get('exit_code') != 0 or provenance.get('before') != provenance.get('after') or
                provenance.get('game_files_deployed') != [] or provenance.get('observation_deferred') is not True):
            raise ValueError('session provenance')
    return dict(status='GAME_OUTPUT_PATCH_PASS' if live else 'ISOLATED_OUTPUT_PATCH_PASS', width=width, height=height,
                raw_bytes=len(after), before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(),
                supplied_pixels=64, changed_pixels=changed_pixels, changed_components=changed_components,
                patch_rect=[0, 0, 8, 8], outside_patch_exact=True, fence_completed=True,
                game_frame_capture_verified=live, game_writeback_verified=live,
                output_replacement_verified=live, replacement_pixels_supplied=True,
                reconstructed_network_output=False, dlss_nr_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('path', type=Path)
    parser.add_argument('--isolated', action='store_true'); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise SystemExit('Refusing to overwrite report')
    report = analyze(args.path, not args.isolated)
    with args.output.open('x', encoding='utf-8') as handle: json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps(report, indent=2))
