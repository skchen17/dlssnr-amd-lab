"""Validate externally weighted dynamic residual-network output."""
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


def analyze(root: Path, live=True):
    root = Path(root)
    events = read_events(root / 'session.jsonl')
    if any(event.get('event') == 'failure' for event in events):
        raise ValueError('session failure')
    markers = [event for event in events if event.get('event') in MARKERS]
    if tuple(event.get('event') for event in markers) != MARKERS:
        raise ValueError('network lifecycle order/count')
    returns = [event for event in events if event.get('event') == 'output_boundary_return']
    accepted = [event for event in returns if event.get('batch_supported') is True and
                event.get('lifecycle_match') is True]
    if len(accepted) != 1:
        raise ValueError('accepted output boundary count')
    arm, recorded, submitted, complete = markers
    selected = [arm, accepted[0], recorded, submitted, complete]
    if ([events.index(event) for event in selected] != sorted(events.index(event) for event in selected) or
            len({event.get('window') for event in selected + returns}) != 1):
        raise ValueError('network lifecycle window/order')
    if (recorded.get('isolated_host_only') is live or submitted.get('isolated_host_only') is live or
            any(event.get('write_back_performed') is not True or
                event.get('replacement_pixels_supplied') is not True
                for event in (arm, recorded, submitted, complete)) or
            recorded.get('gpu_residual_network_recorded') is not True or
            recorded.get('external_weights') is not True or
            recorded.get('internal_hook_bypass') is not True or
            submitted.get('submitted_identity_matches') != 1 or
            submitted.get('submission_split_performed') is not True or
            type(submitted.get('submission_batch_count')) is not int or
            type(submitted.get('submission_batch_index')) is not int or
            type(submitted.get('submission_suffix_lists')) is not int or
            submitted['submission_batch_count'] < 3 or
            not 0 <= submitted['submission_batch_index'] < submitted['submission_batch_count'] or
            submitted['submission_suffix_lists'] != submitted['submission_batch_count'] - submitted['submission_batch_index'] - 1 or
            submitted['submission_suffix_lists'] < 1 or complete.get('fence_completed') is not True or
            complete.get('filter_kind') != 'dynamic_residual_cnn_3x3_8' or
            complete.get('gpu_residual_filter') is not True or complete.get('external_weights') is not True or
            complete.get('trained_weights') is not False or complete.get('patch_rect') is not None or
            complete.get('game_frame_capture_verified') is not live or complete.get('nr_verified') is not False):
        raise ValueError('network provenance')

    output = root / 'output_network'
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8-sig'))
    width, height = metadata.get('width'), metadata.get('height')
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 8192 or
            not 1 <= height <= 8192 or metadata.get('format') != 10 or
            metadata.get('raw_bytes') != width * height * 8 or metadata.get('fence_completed') is not True or
            metadata.get('write_back_performed') is not True or
            metadata.get('replacement_pixels_supplied') is not True or metadata.get('patch_rect') is not None or
            metadata.get('filter_kind') != 'dynamic_residual_cnn_3x3_8' or
            metadata.get('gpu_residual_filter') is not True or metadata.get('external_weights') is not True or
            metadata.get('trained_weights') is not False or
            metadata.get('game_frame_capture_verified') is not live or metadata.get('nr_verified') is not False):
        raise ValueError('network metadata')
    before = (output / 'boundary_before.raw').read_bytes()
    after = (output / 'boundary_output.raw').read_bytes()
    if len(before) != width * height * 8 or len(after) != len(before):
        raise ValueError('raw byte count')
    words = array('H'); words.frombytes(after)
    if any((word & 0x7c00) == 0x7c00 for word in words):
        raise ValueError('non-finite half output')
    changed_pixels = changed_components = 0
    alpha_exact = True
    rgb_abs = []
    for offset in range(0, len(after), 8):
        old = struct.unpack_from('<4e', before, offset); new = struct.unpack_from('<4e', after, offset)
        changed_pixels += before[offset:offset + 8] != after[offset:offset + 8]
        changed_components += sum(before[offset + 2*c:offset + 2*c + 2] !=
                                  after[offset + 2*c:offset + 2*c + 2] for c in range(4))
        alpha_exact &= before[offset + 6:offset + 8] == after[offset + 6:offset + 8]
        rgb_abs.extend(abs(float(new[c]) - float(old[c])) for c in range(3))
    if not changed_pixels or not changed_components or not alpha_exact or not max(rgb_abs):
        raise ValueError('network did not produce valid residual change')
    if live:
        provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8-sig'))
        if (provenance.get('exit_code') != 0 or provenance.get('before') != provenance.get('after') or
                provenance.get('game_files_deployed') != [] or provenance.get('observation_deferred') is not True):
            raise ValueError('session provenance')
    return dict(status='GAME_GPU_WEIGHTED_NETWORK_PASS' if live else 'ISOLATED_GPU_WEIGHTED_NETWORK_PASS',
                width=width, height=height, raw_bytes=len(after),
                before_sha256=hashlib.sha256(before).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(),
                changed_pixels=changed_pixels, changed_components=changed_components, alpha_exact=True,
                mean_abs_rgb_change=math.fsum(rgb_abs)/len(rgb_abs), max_abs_rgb_change=max(rgb_abs),
                submission_batch_count=submitted['submission_batch_count'],
                submission_batch_index=submitted['submission_batch_index'],
                submission_suffix_lists=submitted['submission_suffix_lists'], internal_hook_bypass=True,
                queue_wait_gate_used=False, game_frame_capture_verified=live, game_writeback_verified=live,
                output_replacement_verified=live, gpu_residual_filter=True, external_weights=True,
                trained_weights=False, reconstructed_network_output=True, dlss_nr_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path); parser.add_argument('--isolated', action='store_true')
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise SystemExit('Refusing to overwrite report')
    report = analyze(args.path, not args.isolated)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps(report, indent=2))
