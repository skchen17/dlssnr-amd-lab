"""Validate one fenced, exact no-op output roundtrip in a live GoWR process."""
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


def analyze(root: Path):
    root = Path(root)
    events = read_events(root / 'session.jsonl')
    if any(e.get('event') == 'failure' for e in events):
        raise ValueError('session failure')
    markers = [e for e in events if e.get('event') in MARKERS]
    if tuple(e.get('event') for e in markers) != MARKERS:
        raise ValueError('roundtrip lifecycle order/count')
    returns = [e for e in events if e.get('event') == 'output_boundary_return']
    accepted = [e for e in returns if e.get('batch_supported') is True and e.get('lifecycle_match') is True]
    if len(accepted) != 1:
        raise ValueError('accepted output boundary count')
    selected = [markers[0], accepted[0], *markers[1:]]
    if [events.index(e) for e in selected] != sorted(events.index(e) for e in selected):
        raise ValueError('roundtrip lifecycle order/count')
    if len({e.get('window') for e in selected + returns}) != 1 or selected[0].get('window', 0) < 1:
        raise ValueError('roundtrip lifecycle window')
    boundary, recorded, submitted, complete = selected[1:]
    for candidate in returns:
        if (type(candidate.get('batch_supported')) is not bool or type(candidate.get('lifecycle_match')) is not bool or
                candidate.get('original_callback_returned') is not True or candidate.get('original_forward_calls') != 1 or
                candidate.get('capture_authorized') is not False):
            raise ValueError('output boundary contract')
    if (recorded.get('isolated_host_only') is not False or submitted.get('isolated_host_only') is not False or
            any(e.get('write_back_performed') is not True for e in (markers[0], recorded, submitted, complete)) or
            recorded.get('game_frame_capture_verified') is not False or
            submitted.get('game_frame_capture_verified') is not False or
            submitted.get('submitted_identity_matches') != 1 or
            not isinstance(submitted.get('identity_aliases'), int) or submitted['identity_aliases'] < 1 or
            not isinstance(submitted.get('closed_identity_aliases'), int) or submitted['closed_identity_aliases'] < 1 or
            type(submitted.get('raw_pointer_match')) is not bool or
            complete.get('game_frame_capture_verified') is not True or
            complete.get('fence_completed') is not True or
            complete.get('replacement_pixels_supplied') is not False or complete.get('nr_verified') is not False):
        raise ValueError('game/writeback provenance')

    output = root / 'output_roundtrip'
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8-sig'))
    width, height, fmt = metadata.get('width'), metadata.get('height'), metadata.get('format')
    if (type(width) is not int or type(height) is not int or not 1 <= width <= 8192 or not 1 <= height <= 8192 or
            fmt != 10 or metadata.get('raw_bytes') != width * height * 8 or
            metadata.get('fence_completed') is not True or metadata.get('game_frame_capture_verified') is not True or
            metadata.get('write_back_performed') is not True or metadata.get('replacement_pixels_supplied') is not False or
            metadata.get('nr_verified') is not False):
        raise ValueError('roundtrip metadata')
    for field in ('width', 'height', 'format', 'raw_bytes'):
        if complete.get(field) != metadata.get(field):
            raise ValueError('event/metadata mismatch')

    raw = (output / 'boundary_output.raw').read_bytes()
    if len(raw) != metadata['raw_bytes']:
        raise ValueError('raw byte count')
    words = array('H'); words.frombytes(raw)
    if any((word & 0x7c00) == 0x7c00 for word in words):
        raise ValueError('non-finite half output')
    pixel_count = width * height
    stride = max(1, pixel_count // 65536)
    samples = []
    channels = [[], [], [], []]
    for pixel in range(0, pixel_count, stride):
        values = struct.unpack_from('<4e', raw, pixel * 8)
        samples.extend(values)
        for channel, value in zip(channels, values):
            channel.append(value)
    minimum, maximum = min(samples), max(samples)
    mean = math.fsum(samples) / len(samples)
    variance = math.fsum((value - mean) ** 2 for value in samples) / len(samples)
    if maximum - minimum <= 1e-5 or variance <= 1e-12:
        raise ValueError('blank/constant output')

    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8-sig'))
    if (provenance.get('exit_code') != 0 or provenance.get('before') != provenance.get('after') or
            provenance.get('game_files_deployed') != [] or provenance.get('observation_deferred') is not True):
        raise ValueError('session provenance')
    channel_stats = []
    for values in channels:
        channel_mean = math.fsum(values) / len(values)
        channel_variance = math.fsum((value - channel_mean) ** 2 for value in values) / len(values)
        channel_stats.append(dict(min=min(values), max=max(values), mean=channel_mean,
                                  stddev=math.sqrt(channel_variance), unique_half_values=len(set(values))))
    scene_content = (all(row['unique_half_values'] >= 32 for row in channel_stats[:3]) and
                     max(row['max'] - row['min'] for row in channel_stats[:3]) >= 0.25 and
                     max(row['stddev'] for row in channel_stats[:3]) >= 0.01)
    return dict(status='GAME_OUTPUT_ROUNDTRIP_PASS', width=width, height=height, format=fmt,
                raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), sample_count=len(samples),
                sampled_min=minimum, sampled_max=maximum, sampled_mean=mean,
                sampled_stddev=math.sqrt(variance), channel_stats=channel_stats,
                scene_content_verified=scene_content, dataset_usable=scene_content, fence_completed=True,
                source='existing game FFX output texture copied through scratch and restored exactly',
                game_frame_capture_verified=True, game_writeback_verified=True,
                write_back_performed=True, replacement_pixels_supplied=False,
                output_replacement_verified=False, dlss_nr_verified=False, teacher_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite report')
    report = analyze(args.path)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))
