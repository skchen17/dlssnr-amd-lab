#!/usr/bin/env python3
"""Decode a locally extracted WEIGHTS_HT resource into the GPU model arena.

The resource format is a size-prefixed sequence of named FP16 tensor records.
The runtime arena is the tensor payloads in resource order, each padded to a
512-byte boundary.  Outputs are private, user-local interoperability artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


ALIGNMENT = 512
DTYPE_FP16 = 1
RECORD_TRAILER_PREFIX = (0, 0, 1, 0)


@dataclass(frozen=True)
class TensorRecord:
    name: str
    resource_offset: int
    data_offset: int
    data_bytes: int
    element_count: int


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_resource(blob: bytes) -> list[TensorRecord]:
    if len(blob) < 8:
        raise ValueError("weight resource is shorter than its size header")
    declared_bytes = struct.unpack_from("<Q", blob, 0)[0]
    if declared_bytes != len(blob):
        raise ValueError(
            f"resource size header is {declared_bytes}, actual file is {len(blob)} bytes"
        )

    records: list[TensorRecord] = []
    names: set[str] = set()
    position = 8
    while position < len(blob):
        record_offset = position
        if position + 8 > len(blob):
            raise ValueError(f"truncated tensor-name length at offset {position}")
        name_bytes = struct.unpack_from("<Q", blob, position)[0]
        position += 8
        if not 1 <= name_bytes <= 1024 or position + name_bytes + 28 > len(blob):
            raise ValueError(f"invalid tensor-name length {name_bytes} at offset {record_offset}")
        try:
            name = blob[position : position + name_bytes].decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(f"tensor name at offset {record_offset} is not ASCII") from exc
        position += name_bytes
        packed_bytes_a, packed_bytes_b, data_bytes, dtype = struct.unpack_from(
            "<QQQI", blob, position
        )
        position += 28
        data_offset = position
        trailer_offset = data_offset + data_bytes
        if trailer_offset + 20 > len(blob):
            raise ValueError(f"tensor {name!r} extends beyond the resource")
        trailer = struct.unpack_from("<IIIII", blob, trailer_offset)

        if name in names:
            raise ValueError(f"duplicate tensor name {name!r}")
        if packed_bytes_a != packed_bytes_b or packed_bytes_a != data_bytes + 40:
            raise ValueError(f"tensor {name!r} has inconsistent record sizes")
        if dtype != DTYPE_FP16:
            raise ValueError(f"tensor {name!r} has unsupported dtype id {dtype}")
        if trailer[:4] != RECORD_TRAILER_PREFIX or trailer[4] * 2 != data_bytes:
            raise ValueError(f"tensor {name!r} has an invalid FP16 trailer")

        records.append(
            TensorRecord(name, record_offset, data_offset, data_bytes, trailer[4])
        )
        names.add(name)
        position = trailer_offset + 20

    if not records:
        raise ValueError("weight resource contains no tensor records")
    return records


def decode_resource(
    resource_path: Path, output_dir: Path, reference_arena: Path | None = None
) -> dict:
    resource_path = resource_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if any(part.casefold() == "deliverables" for part in output_dir.parts):
        raise ValueError("decoded model data cannot be written to deliverables")

    blob = resource_path.read_bytes()
    records = parse_resource(blob)
    output_dir.mkdir(parents=True)
    arena_path = output_dir / "model_arena.raw"
    tensors = []
    arena_offset = 0
    with arena_path.open("wb") as stream:
        for index, record in enumerate(records):
            payload = memoryview(blob)[
                record.data_offset : record.data_offset + record.data_bytes
            ]
            stream.write(payload)
            tensors.append(
                {
                    "index": index,
                    "name": record.name,
                    "resource_offset": record.resource_offset,
                    "arena_offset": arena_offset,
                    "data_bytes": record.data_bytes,
                    "element_count": record.element_count,
                    "dtype": "fp16",
                    "sha256": sha256_bytes(payload),
                }
            )
            padded_end = align_up(arena_offset + record.data_bytes)
            stream.write(b"\0" * (padded_end - arena_offset - record.data_bytes))
            arena_offset = padded_end

    arena_hash = sha256_file(arena_path)
    reference = None
    status = "PASS"
    if reference_arena is not None:
        reference_path = reference_arena.resolve(strict=True)
        reference_hash = sha256_file(reference_path)
        exact = reference_path.stat().st_size == arena_offset and reference_hash == arena_hash
        reference = {
            "filename": reference_path.name,
            "bytes": reference_path.stat().st_size,
            "sha256": reference_hash,
            "exact_match": exact,
        }
        if not exact:
            status = "FAIL"

    source_manifest = resource_path.parent / "manifest.json"
    manifest = {
        "schema": 1,
        "experiment": "local_dlssnr_weight_resource_decode",
        "status": status,
        "redistributable": False,
        "source": {
            "filename": resource_path.name,
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
            "extraction_manifest": source_manifest.name if source_manifest.is_file() else None,
        },
        "format": {
            "tensor_count": len(records),
            "dtype": "fp16",
            "tensor_alignment_bytes": ALIGNMENT,
        },
        "model_arena": {
            "filename": arena_path.name,
            "bytes": arena_offset,
            "sha256": arena_hash,
        },
        "reference": reference,
        "tensors": tensors,
        "notice": "Local interoperability research only. Do not commit or redistribute decoded data.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resource", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference-arena", type=Path)
    args = parser.parse_args()
    try:
        manifest = decode_resource(args.resource, args.output_dir, args.reference_arena)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "tensor_count": manifest["format"]["tensor_count"],
                "model_arena": manifest["model_arena"],
                "reference_exact_match": (
                    manifest["reference"]["exact_match"] if manifest["reference"] else None
                ),
            },
            indent=2,
        )
    )
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
