"""Extract user-owned runtime module containers for local interoperability analysis.

ELF outputs are proprietary derivatives and are ignored by .gitignore. Only the
JSON metadata (offsets, sizes, hashes) is suitable for repository evidence.
Requires the `zstandard` Python package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
from pathlib import Path

import zstandard


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
ELF_MAGIC = b"\x7fELF"
MODULE_MAGIC = 0xBA55ED50


def parse_int(text: str) -> int:
    return int(text, 0)


def elf_extent(data: bytes, offset: int) -> int:
    if data[offset : offset + 4] != ELF_MAGIC or data[offset + 4] != 2:
        raise ValueError("expected ELF64")
    endian = "<" if data[offset + 5] == 1 else ">"
    phoff = struct.unpack_from(endian + "Q", data, offset + 32)[0]
    shoff = struct.unpack_from(endian + "Q", data, offset + 40)[0]
    phentsize, phnum = struct.unpack_from(endian + "HH", data, offset + 54)
    shentsize, shnum = struct.unpack_from(endian + "HH", data, offset + 58)
    end = max(phoff + phentsize * phnum, shoff + shentsize * shnum)
    for index in range(shnum):
        base = offset + shoff + index * shentsize
        section_type = struct.unpack_from(endian + "I", data, base + 4)[0]
        section_offset, section_size = struct.unpack_from(endian + "QQ", data, base + 24)
        if section_type != 8:  # SHT_NOBITS has no file bytes
            end = max(end, section_offset + section_size)
    return int(end)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", required=True, type=Path)
    parser.add_argument("--module-map", type=Path)
    parser.add_argument("--all-containers", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dll = args.dll.read_bytes()
    records: list[dict[str, object]] = []
    if args.all_containers:
        modules = []
        cursor = 0
        magic_bytes = struct.pack("<I", MODULE_MAGIC)
        while True:
            offset = dll.find(magic_bytes, cursor)
            if offset < 0:
                break
            if offset + 16 <= len(dll):
                size = struct.unpack_from("<Q", dll, offset + 8)[0] + 16
                if 16 < size <= len(dll) - offset:
                    modules.append(
                        {"dll_offset": hex(offset), "size": str(size), "fnv1a64": ""}
                    )
            cursor = offset + 4
    elif args.module_map:
        with args.module_map.open("r", encoding="utf-8-sig", newline="") as handle:
            modules = list(csv.DictReader(handle))
    else:
        parser.error("provide --module-map or --all-containers")

    for index, module in enumerate(modules):
        container_offset = parse_int(module["dll_offset"])
        container_size = int(module["size"])
        blob = dll[container_offset : container_offset + container_size]
        magic = struct.unpack_from("<I", blob, 0)[0] if len(blob) >= 4 else 0
        if magic != MODULE_MAGIC:
            raise ValueError(f"module {index}: bad container magic 0x{magic:08x}")
        zstd_offset = blob.find(ZSTD_MAGIC)
        if zstd_offset < 0:
            raise ValueError(f"module {index}: no Zstandard frame")
        decompressed = zstandard.ZstdDecompressor().decompress(
            blob[zstd_offset:], max_output_size=128 * 1024 * 1024
        )
        base_record: dict[str, object] = {
            "index": index,
            "dll_offset": f"0x{container_offset:X}",
            "container_size": container_size,
            "container_sha256": hashlib.sha256(blob).hexdigest(),
            "captured_fnv1a64": module["fnv1a64"],
            "zstd_offset": zstd_offset,
            "decompressed_size": len(decompressed),
            "decompressed_sha256": hashlib.sha256(decompressed).hexdigest(),
        }
        text_head = decompressed[:4096].decode("ascii", errors="ignore")
        version_match = re.search(r"(?m)^\s*\.version\s+([^\s]+)", text_head)
        target_match = re.search(r"(?m)^\s*\.target\s+([^\s,]+)", text_head)
        if version_match and target_match:
            out_name = f"module_{index:02d}_{container_offset:08X}.ptx"
            (args.out / out_name).write_bytes(decompressed)
            text = decompressed.decode("utf-8", errors="replace")
            entry_names = re.findall(
                r"(?m)^\s*(?:\.visible\s+)?\.entry\s+([^\s(]+)", text
            )
            base_record.update(
                {
                    "payload_kind": "PTX",
                    "ptx_version": version_match.group(1),
                    "ptx_target": target_match.group(1),
                    "entry_count": len(entry_names),
                    "entry_names": entry_names,
                    "local_payload": out_name,
                }
            )
        else:
            elf_offset = decompressed.find(ELF_MAGIC)
            if elf_offset < 0:
                raise ValueError(f"module {index}: payload is neither PTX nor ELF")
            extent = elf_extent(decompressed, elf_offset)
            elf = decompressed[elf_offset : elf_offset + extent]
            machine = struct.unpack_from("<H", elf, 18)[0]
            out_name = f"module_{index:02d}_{container_offset:08X}.elf"
            (args.out / out_name).write_bytes(elf)
            base_record.update(
                {
                    "payload_kind": "ELF",
                    "elf_offset_in_decompressed": elf_offset,
                    "elf_size": len(elf),
                    "elf_machine": machine,
                    "elf_sha256": hashlib.sha256(elf).hexdigest(),
                    "local_payload": out_name,
                }
            )
        records.append(base_record)

    manifest = {
        "schema": 1,
        "experiment": "runtime_module_extraction",
        "source_dll_size": len(dll),
        "source_dll_sha256": hashlib.sha256(dll).hexdigest(),
        "module_count": len(records),
        "proprietary_payload_outputs_ignored": True,
        "modules": records,
    }
    (args.out / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"extracted {len(records)} runtime modules to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
