"""Build a private, CPU-only gfx1201 derived-layout cache for the 71-block model.

The cache is deliberately separate from the immutable source model.  Matrix
weights whose recovered semantics are E4M3 are stored as one-byte E4M3; scale,
bias, conditioning and index tensors retain their semantic dtype.  No GPU is
used and the output is not a deployment-readiness claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SOURCE_MANIFEST_SHA256 = 'AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3'
ORIGINAL_MODEL_SHA256 = 'A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5'


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest().upper()


def encode_tensor(name, tensor):
    import torch
    value = tensor.detach().cpu()
    if value.dtype == torch.float16 and value.ndim >= 2 and not name.endswith('position_bias'):
        # The FP8 hipBLASLt QKV consumer expects B in column-major storage.
        column_major = name.endswith('.qkv')
        storage = value.t().contiguous() if column_major else value.contiguous()
        fp8 = storage.clamp(-448, 448).to(torch.float8_e4m3fn)
        if not torch.equal(fp8.to(torch.float16), storage):
            # A matrix shape alone is not evidence of an E4M3 semantic boundary
            # (the Pre input projection is a known counterexample). Preserve it.
            return value.contiguous().numpy().tobytes(), 'fp16_le', 'row_major', list(value.shape)
        raw = fp8.view(torch.uint8).numpy().tobytes()
        return raw, 'e4m3fn', ('column_major_transposed_storage' if column_major else 'row_major'), list(value.shape)
    if value.dtype == torch.float16:
        return value.contiguous().numpy().tobytes(), 'fp16_le', 'row_major', list(value.shape)
    if value.dtype == torch.float32:
        return value.contiguous().numpy().tobytes(), 'fp32_le', 'row_major', list(value.shape)
    if value.dtype == torch.int64:
        # These are decoded static indices/permutations, not neural weights.
        # Int32 is sufficient for the recovered maximum index and halves cache IO.
        if value.numel() and (int(value.min()) < -(1 << 31) or int(value.max()) >= (1 << 31)):
            raise ValueError(f'{name} does not fit the native int32 index ABI')
        packed = value.to(torch.int32).contiguous()
        return packed.numpy().tobytes(), 'int32_le', 'row_major', list(value.shape)
    raise ValueError(f'unsupported derived tensor dtype: {name}: {value.dtype}')


def with_native_derived_records(state):
    """Append immutable native-only indexes derived exactly from source state."""
    import torch
    entries = list(state.items())
    for name, tensor in list(entries):
        if not name.endswith('.ffn.permutation'):
            continue
        value = tensor.detach().cpu().to(torch.int64)
        if value.ndim != 1 or sorted(value.tolist()) != list(range(value.numel())):
            raise ValueError(f'not a permutation: {name}')
        inverse = torch.empty_like(value)
        inverse[value] = torch.arange(value.numel(), dtype=torch.int64)
        entries.append((name[:-len('permutation')] + 'inverse_permutation', inverse))
    return entries


def build(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from native_model_package import load_package
    from native_whole_frame import SingleColorWholeFrame

    records, origins, settings = load_package(source, SOURCE_MANIFEST_SHA256)
    model = SingleColorWholeFrame(records, origins, color_scale=settings['color_scale'],
                                  conditioning=settings['conditioning']).eval()
    payload = bytearray(); table = []; stored = {}
    for name, tensor in with_native_derived_records(model.state_dict()):
        raw, dtype, layout, logical_shape = encode_tensor(name, tensor)
        digest = _sha(raw)
        identity = (digest, dtype, layout, len(raw))
        if identity in stored:
            offset = stored[identity]
            shared = True
        else:
            offset = (len(payload) + 255) & -256
            payload.extend(bytes(offset - len(payload))); payload.extend(raw)
            stored[identity] = offset; shared = False
        table.append({'name': name, 'offset': offset, 'bytes': len(raw),
                      'sha256': digest, 'dtype': dtype, 'storage_layout': layout,
                      'logical_shape': logical_shape, 'deduplicated_reference': shared})
    output.mkdir(parents=True)
    (output / 'weights_gfx1201.bin').write_bytes(payload)
    manifest = {
        'schema': 1, 'status': 'DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED',
        'target_arch': 'gfx1201', 'architecture': 'single_color_71_v2',
        'source_manifest_sha256': SOURCE_MANIFEST_SHA256,
        'original_model_sha256': ORIGINAL_MODEL_SHA256,
        'source_model_immutable': True, 'temporal_contract_id': 'temporal-unresolved-v0',
        'weights_file': 'weights_gfx1201.bin', 'weights_sha256': _sha(payload),
        'records': table, 'record_count': len(table),
        'unique_storage_records': len(stored), 'bytes': len(payload),
        'matrix_policy': 'recovered E4M3 matrix weights; QKV column-major, other matrices row-major',
        'activation_resident_runtime_required': True,
        'runtime_accepted': False, 'private': True,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def inspect(root: Path):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    payload = (root / manifest['weights_file']).read_bytes()
    if manifest.get('schema') != 1 or manifest.get('target_arch') != 'gfx1201' or \
            manifest.get('original_model_sha256') != ORIGINAL_MODEL_SHA256 or \
            manifest.get('weights_sha256') != _sha(payload):
        raise ValueError('derived cache provenance/payload mismatch')
    for record in manifest.get('records', []):
        first, size = record.get('offset'), record.get('bytes')
        if type(first) is not int or type(size) is not int or first < 0 or size <= 0 or \
                size > len(payload) - first or _sha(payload[first:first + size]) != record.get('sha256'):
            raise ValueError('derived cache record mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('local_models/native_single_color_v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    result = inspect(args.output) if args.inspect else build(args.source, args.output)
    print(json.dumps({key: result[key] for key in
                      ('status', 'record_count', 'unique_storage_records', 'bytes')}, indent=2))


if __name__ == '__main__':
    main()
