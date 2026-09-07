"""Deterministic private NR model package v2.

The fixed binary header is intentionally C-readable: the deployment runtime
validates the selected section before allocating/uploading it.  The package
contains weights only, never captured activations or teacher frames.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct


MAGIC = b'NRMPKG2\0'
VERSION = 2
HEADER_BYTES = 512
ALIGNMENT = 256
FLAG_STRICT = 1
FLAG_GFX1201_FP8 = 2
ORIGINAL_MODEL_SHA256 = 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'
ARCHITECTURE = 'single_color_71_v2'
INDEX_MAGIC = b'NRWIDX2\0'
INDEX_RECORD_BYTES = 128


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def _sha(raw: bytes) -> bytes:
    return hashlib.sha256(raw).digest()


def _fixed(text: str, size: int) -> bytes:
    raw = text.encode('utf-8')
    if not raw or len(raw) >= size:
        raise ValueError(f'text must use 1..{size - 1} UTF-8 bytes')
    return raw + bytes(size - len(raw))


def _segments(raw: bytes, profile: int, specs) -> list[tuple]:
    specs = [('all', 0, len(raw))] if specs is None else list(specs)
    result, cursor, names = [], 0, set()
    for spec in specs:
        if not isinstance(spec, (list, tuple)) or len(spec) != 3:
            raise ValueError('weight segment must be (name, offset, bytes)')
        name, offset, size = spec
        encoded = str(name).encode('utf-8')
        if not encoded or len(encoded) >= 64 or name in names or offset != cursor or \
                type(size) is not int or size <= 0 or size > len(raw) - offset:
            raise ValueError('weight segments must uniquely and contiguously cover the section')
        names.add(name); cursor += size
        result.append((profile, str(name), offset, size, _sha(raw[offset:offset + size])))
    if cursor != len(raw):
        raise ValueError('weight segments do not cover the section')
    return result


def _index(records: list[tuple]) -> bytes:
    value = bytearray(16 + len(records) * INDEX_RECORD_BYTES)
    value[:8] = INDEX_MAGIC
    struct.pack_into('<II', value, 8, len(records), INDEX_RECORD_BYTES)
    for number, (profile, name, offset, size, digest) in enumerate(records):
        first = 16 + number * INDEX_RECORD_BYTES
        struct.pack_into('<IIQQ32s64s8s', value, first, profile, 0, offset, size,
                         digest, _fixed(name, 64), bytes(8))
    return bytes(value)


def build(strict_weights: bytes, *, fp8_weights: bytes | None,
          temporal_contract_id: str, target_arch: str = 'gfx1201',
          strict_segments=None, fp8_segments=None) -> bytes:
    if not strict_weights:
        raise ValueError('strict weights required')
    flags = FLAG_STRICT | (FLAG_GFX1201_FP8 if fp8_weights is not None else 0)
    strict_offset = HEADER_BYTES
    fp8_offset = _align(strict_offset + len(strict_weights)) if fp8_weights is not None else 0
    records = _segments(strict_weights, 0, strict_segments)
    if fp8_weights is not None:
        records += _segments(fp8_weights, 1, fp8_segments)
    index = _index(records)
    weights_end = fp8_offset + len(fp8_weights) if fp8_weights is not None else strict_offset + len(strict_weights)
    index_offset = _align(weights_end)
    total = index_offset + len(index)
    package = bytearray(total)
    package[:8] = MAGIC
    struct.pack_into('<IIIIQQQQ', package, 8, VERSION, HEADER_BYTES, flags,
                     2 if fp8_weights is not None else 1, strict_offset,
                     len(strict_weights), fp8_offset,
                     len(fp8_weights) if fp8_weights is not None else 0)
    package[56:88] = bytes.fromhex(ORIGINAL_MODEL_SHA256)
    package[88:120] = _sha(strict_weights)
    package[120:152] = _sha(fp8_weights) if fp8_weights is not None else bytes(32)
    package[152:184] = _fixed(ARCHITECTURE, 32)
    package[184:248] = _fixed(temporal_contract_id, 64)
    package[248:264] = _fixed(target_arch, 16)
    struct.pack_into('<QQ', package, 264, index_offset, len(index))
    package[280:312] = _sha(index)
    package[strict_offset:strict_offset + len(strict_weights)] = strict_weights
    if fp8_weights is not None:
        package[fp8_offset:fp8_offset + len(fp8_weights)] = fp8_weights
    package[index_offset:index_offset + len(index)] = index
    return bytes(package)


def inspect(raw: bytes) -> dict:
    if len(raw) < HEADER_BYTES or raw[:8] != MAGIC:
        raise ValueError('not an NR model package v2')
    version, header_bytes, flags, section_count, strict_offset, strict_bytes, fp8_offset, fp8_bytes = \
        struct.unpack_from('<IIIIQQQQ', raw, 8)
    if version != VERSION or header_bytes != HEADER_BYTES or flags & ~3 or not flags & FLAG_STRICT:
        raise ValueError('unsupported package header')
    expected_sections = 2 if flags & FLAG_GFX1201_FP8 else 1
    if section_count != expected_sections or strict_offset != HEADER_BYTES or not strict_bytes:
        raise ValueError('invalid package section table')
    if strict_offset + strict_bytes > len(raw):
        raise ValueError('truncated strict weights')
    index_offset, index_bytes = struct.unpack_from('<QQ', raw, 264)
    if not index_bytes or index_offset < HEADER_BYTES or index_offset + index_bytes != len(raw):
        raise ValueError('invalid weight index bounds')
    index = raw[index_offset:index_offset + index_bytes]
    if _sha(index) != raw[280:312] or len(index) < 16 or index[:8] != INDEX_MAGIC:
        raise ValueError('weight index hash/magic mismatch')
    record_count, record_bytes = struct.unpack_from('<II', index, 8)
    if record_bytes != INDEX_RECORD_BYTES or len(index) != 16 + record_count * record_bytes:
        raise ValueError('invalid weight index table')
    if flags & FLAG_GFX1201_FP8:
        if fp8_offset != _align(strict_offset + strict_bytes) or not fp8_bytes or fp8_offset + fp8_bytes > index_offset:
            raise ValueError('invalid FP8 section')
    elif fp8_offset or fp8_bytes or strict_offset + strict_bytes > index_offset:
        raise ValueError('unexpected package suffix')
    strict = raw[strict_offset:strict_offset + strict_bytes]
    fp8 = raw[fp8_offset:fp8_offset + fp8_bytes] if fp8_bytes else None
    if _sha(strict) != raw[88:120] or (fp8 is not None and _sha(fp8) != raw[120:152]):
        raise ValueError('weight section hash mismatch')
    sections = {0: strict, **({1: fp8} if fp8 is not None else {})}
    segment_tables = {0: [], 1: []}
    for number in range(record_count):
        first = 16 + number * record_bytes
        profile, reserved, offset, size, digest, name, tail = struct.unpack_from(
            '<IIQQ32s64s8s', index, first)
        name = name.split(b'\0', 1)[0].decode('utf-8')
        if profile not in sections or reserved or tail != bytes(8) or not name or \
                size <= 0 or offset + size > len(sections[profile]) or \
                _sha(sections[profile][offset:offset + size]) != digest:
            raise ValueError('invalid weight segment record')
        segment_tables[profile].append({'name': name, 'offset': offset, 'bytes': size,
                                        'sha256': digest.hex().upper()})
    for profile in sections:
        records = sorted(segment_tables[profile], key=lambda x: x['offset'])
        cursor = 0
        for record in records:
            if record['offset'] != cursor:
                raise ValueError('weight segment coverage gap/overlap')
            cursor += record['bytes']
        if cursor != len(sections[profile]) or len({x['name'] for x in records}) != len(records):
            raise ValueError('weight segment coverage/name mismatch')
    if raw[56:88].hex().upper() != ORIGINAL_MODEL_SHA256:
        raise ValueError('original model provenance mismatch')
    def text(start, end):
        value = raw[start:end].split(b'\0', 1)[0]
        return value.decode('utf-8')
    architecture, temporal, target = text(152, 184), text(184, 248), text(248, 264)
    if architecture != ARCHITECTURE or not temporal or not target:
        raise ValueError('unsupported architecture or empty contract')
    return {
        'schema': 2, 'architecture': architecture,
        'original_model_sha256': ORIGINAL_MODEL_SHA256,
        'temporal_contract_id': temporal, 'target_arch': target,
        'sections': {
            'strict_fp16': {'offset': strict_offset, 'bytes': strict_bytes,
                            'sha256': raw[88:120].hex().upper(),
                            'segments': segment_tables[0]},
            **({'gfx1201_fp8': {'offset': fp8_offset, 'bytes': fp8_bytes,
                                'sha256': raw[120:152].hex().upper(),
                                'segments': segment_tables[1]}} if fp8 is not None else {}),
        },
        'weight_index_sha256': raw[280:312].hex().upper(),
        'private': True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-weights', type=Path, required=True)
    parser.add_argument('--fp8-weights', type=Path)
    parser.add_argument('--temporal-contract-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = build(args.strict_weights.read_bytes(),
                fp8_weights=args.fp8_weights.read_bytes() if args.fp8_weights else None,
                temporal_contract_id=args.temporal_contract_id)
    inspect(raw)
    args.output.write_bytes(raw)
    print(inspect(raw))


if __name__ == '__main__':
    main()
