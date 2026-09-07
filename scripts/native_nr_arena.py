"""Deterministic fixed-layout workspace ABI for the native NR runtime.

This module deliberately plans memory only.  A captured PyTorch graph is not
declared deployable merely because an equivalent C++ arena has been allocated:
every graph argument must first be rebound to one of these regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ABI_VERSION = 3
ALIGNMENT = 256
KINDS = {'dynamic_resource': 1, 'activation': 2, 'skip': 3, 'workspace': 4}
PRECISION_PROFILES = {'strict_fp16': 2, 'approx_fp8': 1}


def align(value: int, alignment: int = ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError('invalid value/alignment')
    return (value + alignment - 1) & -alignment


def _region(regions, cursor, name, size, kind):
    if size <= 0:
        raise ValueError(f'empty arena region: {name}')
    offset = align(cursor)
    regions.append({'name': name, 'offset': offset, 'bytes': align(size),
                    'alignment': ALIGNMENT, 'kind': KINDS[kind]})
    return offset + align(size)


def stage_feature_elements(width: int, height: int) -> dict[str, int]:
    """Match the recovered network padding/scales used by the graph census."""
    if width <= 0 or height <= 0:
        raise ValueError('positive dimensions required')
    pw, ph = align(width, 128), align(height, 128)
    return {
        'c32': (pw // 2) * (ph // 2) * 32,
        'c64': (pw // 4) * (ph // 4) * 64,
        'c128': (pw // 8) * (ph // 8) * 128,
        'c256': (pw // 16) * (ph // 16) * 256,
        'c512': (pw // 32) * (ph // 32) * 512,
        'vit': align(pw // 64, 4) * align(ph // 64, 4) * 1024,
    }


def wide_window_elements(width: int, height: int) -> dict[str, int]:
    """Largest gathered-window storage for any recovered (-4/0) origin."""
    pw, ph = align(width, 128), align(height, 128)
    result = {}
    for channels in (32, 64, 128, 256):
        feature_width = pw // 2 // (channels // 32)
        feature_height = ph // 2 // (channels // 32)
        windows = ((feature_width + 11) // 8) * ((feature_height + 11) // 8)
        result[f'c{channels}'] = windows * 64 * channels
    return result


def stage_feature_bytes(width: int, height: int,
                        precision_profile: str = 'strict_fp16') -> dict[str, int]:
    if precision_profile not in PRECISION_PROFILES:
        raise ValueError('unknown precision profile')
    item_bytes = PRECISION_PROFILES[precision_profile]
    return {name: count * item_bytes
            for name, count in stage_feature_elements(width, height).items()}


def build_arena_plan(width: int, height: int,
                     precision_profile: str = 'strict_fp16') -> dict:
    pixels = width * height
    sizes = stage_feature_bytes(width, height, precision_profile)
    fp16_sizes = stage_feature_bytes(width, height, 'strict_fp16')
    regions, cursor = [], 0
    # Slots are present now so temporal semantics can be added without changing
    # the frame-binding ABI or all subsequent activation offsets.
    for name, size in (
        ('frame.input_rgba16f', pixels * 8),
        ('frame.output_residual_rgba16f', pixels * 8),
        ('frame.history_rgba16f', pixels * 8),
        ('frame.next_history_rgba16f', pixels * 8),
        ('frame.motion_rg16f', pixels * 4),
        ('frame.depth_r32f', pixels * 4),
        ('frame.exposure', ALIGNMENT),
        ('frame.controls', ALIGNMENT),
    ):
        cursor = _region(regions, cursor, name, size, 'dynamic_resource')
    for stage in ('c32', 'c64', 'c128', 'c256', 'c512', 'vit'):
        cursor = _region(regions, cursor, f'{stage}.ping', sizes[stage], 'activation')
        cursor = _region(regions, cursor, f'{stage}.pong', sizes[stage], 'activation')
        if stage != 'vit':
            cursor = _region(regions, cursor, f'{stage}.skip', sizes[stage], 'skip')
    if precision_profile == 'approx_fp8':
        # C64/C128/C256 stage attention reuses these sequential scratch slots.
        # Their lifetime ends before the next block starts, so four maximum-size
        # buffers cover all wide families without one allocation per block.
        wide_bytes = max(wide_window_elements(width, height).values())
        for name in ('grouped', 'post', 'q', 'k', 'v', 'value'):
            cursor = _region(regions, cursor, f'operator.fp8_{name}', wide_bytes,
                             'workspace')
        # Standard C32-C256 QKV projection is an explicit FP16 diagnostic and
        # correctness boundary. It is shared sequentially by all wide blocks.
        cursor = _region(regions, cursor, 'operator.qkv_projection_fp16',
                         wide_bytes * 3 * 2, 'workspace')
        # The ViT FFN has a semantic 1024->4096 resident E4M3 boundary. Keep
        # one fixed hidden buffer for all eight sequential blocks; it is never
        # converted to a logical FP16 tensor and cannot overlap Q/K/V while the
        # FFN contract is consuming it.
        vit_tokens = stage_feature_elements(width, height)['vit'] // 1024
        cursor = _region(regions, cursor, 'operator.vit_hidden_fp8',
                         vit_tokens * 4096, 'workspace')
    # Approximate resident tensors are one-byte E4M3. Residual/norm/softmax
    # boundaries retain an explicitly shared FP16/FP32 workspace; using FP8 for
    # the resident ABI must not silently change those mathematical boundaries.
    max_fp16_stage = max(fp16_sizes.values())
    # Three simultaneous semantic FP16 window views are required by the
    # standard stage descriptor (seed, post and projected output). Shifted C32
    # windows are slightly larger than its packed image, so stage image bytes
    # are not a safe upper bound.
    max_fp16_windows = max(wide_window_elements(width, height).values()) * 2
    cursor = _region(regions, cursor, 'operator.fp16_workspace',
                     max_fp16_windows * 3, 'workspace')
    cursor = _region(regions, cursor, 'operator.fp32_reduction', max_fp16_stage, 'workspace')
    return {
        'schema': 2,
        'abi_version': ABI_VERSION,
        'precision_profile': precision_profile,
        'resident_item_bytes': PRECISION_PROFILES[precision_profile],
        'resolution': [width, height],
        'alignment': ALIGNMENT,
        'workspace_bytes': align(cursor),
        'stage_feature_bytes': sizes,
        'regions': regions,
        'overlap_free': all(a['offset'] + a['bytes'] <= b['offset']
                            for a, b in zip(regions, regions[1:])),
        'deployment_note': 'plan only; readiness requires every graph pointer to be rebound to this C++ allocation and all temporal contracts to pass',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--precision-profile', choices=sorted(PRECISION_PROFILES),
                        default='strict_fp16')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_arena_plan(args.width, args.height, args.precision_profile)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'workspace_bytes': result['workspace_bytes'],
                      'regions': len(result['regions'])}, indent=2))


if __name__ == '__main__':
    main()
