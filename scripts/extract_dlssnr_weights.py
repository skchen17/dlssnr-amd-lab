#!/usr/bin/env python3
"""Extract a named PE resource from a user-owned DLSS NR DLL.

The output is deliberately local-only.  It is not a decoded NVIDIA model and
must never be added to source control or a redistributable package.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


LOAD_LIBRARY_AS_DATAFILE = 0x00000002
LOAD_LIBRARY_AS_IMAGE_RESOURCE = 0x00000020


@dataclass(frozen=True)
class ResourceId:
    value: int
    label: str

    @property
    def text(self) -> str:
        return self.label

    def pointer(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(self.value)


@dataclass(frozen=True)
class ResourceRecord:
    resource_type: ResourceId
    name: ResourceId
    language: int

    def matches(self, token: str) -> bool:
        wanted = token.casefold()
        return wanted in {self.resource_type.text.casefold(), self.name.text.casefold()}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def enumerate_resources(module: int) -> list[ResourceRecord]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    enum_type_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t)
    enum_name_proc = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ssize_t
    )
    enum_lang_proc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ushort,
        ctypes.c_ssize_t,
    )
    kernel32.EnumResourceTypesW.argtypes = [ctypes.c_void_p, enum_type_proc, ctypes.c_ssize_t]
    kernel32.EnumResourceNamesW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        enum_name_proc,
        ctypes.c_ssize_t,
    ]
    kernel32.EnumResourceLanguagesW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        enum_lang_proc,
        ctypes.c_ssize_t,
    ]

    records: list[ResourceRecord] = []
    callbacks: list[object] = []

    def on_type(_module: int, type_ptr: int, _context: int) -> bool:
        type_value = int(type_ptr)
        resource_type = ResourceId(
            type_value, f"#{type_value}" if type_value <= 0xFFFF else ctypes.wstring_at(type_value)
        )

        def on_name(_m: int, _t: int, name_ptr: int, _c: int) -> bool:
            name_value = int(name_ptr)
            name = ResourceId(
                name_value, f"#{name_value}" if name_value <= 0xFFFF else ctypes.wstring_at(name_value)
            )

            def on_language(_lm: int, _lt: int, _ln: int, language: int, _lc: int) -> bool:
                records.append(ResourceRecord(resource_type, name, int(language)))
                return True

            language_callback = enum_lang_proc(on_language)
            callbacks.append(language_callback)
            if not kernel32.EnumResourceLanguagesW(
                ctypes.c_void_p(module),
                resource_type.pointer(),
                name.pointer(),
                language_callback,
                0,
            ):
                error = ctypes.get_last_error()
                if error != 0:
                    raise OSError(error, "EnumResourceLanguagesW failed")
            return True

        name_callback = enum_name_proc(on_name)
        callbacks.append(name_callback)
        if not kernel32.EnumResourceNamesW(
            ctypes.c_void_p(module), resource_type.pointer(), name_callback, 0
        ):
            error = ctypes.get_last_error()
            if error != 0:
                raise OSError(error, "EnumResourceNamesW failed")
        return True

    type_callback = enum_type_proc(on_type)
    callbacks.append(type_callback)
    if not kernel32.EnumResourceTypesW(ctypes.c_void_p(module), type_callback, 0):
        error = ctypes.get_last_error()
        if error != 0:
            raise OSError(error, "EnumResourceTypesW failed")
    return records


def load_resource(module: int, record: ResourceRecord) -> bytes:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FindResourceExW.restype = ctypes.c_void_p
    kernel32.FindResourceExW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ushort,
    ]
    kernel32.SizeofResource.restype = ctypes.c_uint32
    kernel32.SizeofResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.LoadResource.restype = ctypes.c_void_p
    kernel32.LoadResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.LockResource.restype = ctypes.c_void_p
    kernel32.LockResource.argtypes = [ctypes.c_void_p]

    resource = kernel32.FindResourceExW(
        ctypes.c_void_p(module),
        record.resource_type.pointer(),
        record.name.pointer(),
        record.language,
    )
    if not resource:
        raise ctypes.WinError(ctypes.get_last_error())
    size = int(kernel32.SizeofResource(ctypes.c_void_p(module), resource))
    handle = kernel32.LoadResource(ctypes.c_void_p(module), resource)
    address = kernel32.LockResource(handle)
    if not handle or not address or size <= 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return ctypes.string_at(address, size)


def extract(dll_path: Path, output_dir: Path, token: str) -> dict:
    if os.name != "nt":
        raise RuntimeError("PE resource extraction requires Windows")
    dll_path = dll_path.resolve(strict=True)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LoadLibraryExW.restype = ctypes.c_void_p
    kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    module = kernel32.LoadLibraryExW(
        str(dll_path), None, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE
    )
    if not module:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        records = enumerate_resources(int(module))
        matches = [record for record in records if record.matches(token)]
        if not matches:
            available = sorted({f"{r.resource_type.text}/{r.name.text}" for r in records})
            raise LookupError(
                f"resource {token!r} was not found; available resources: {available[:80]}"
            )
        payloads = [load_resource(int(module), record) for record in matches]
    finally:
        kernel32.FreeLibrary(ctypes.c_void_p(module))

    output_dir.mkdir(parents=True)
    entries = []
    for index, (record, payload) in enumerate(zip(matches, payloads)):
        filename = "weights_resource.bin" if len(matches) == 1 else f"weights_resource_{index:02d}.bin"
        path = output_dir / filename
        path.write_bytes(payload)
        entries.append(
            {
                "type": record.resource_type.text,
                "name": record.name.text,
                "language": record.language,
                "file": filename,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "schema": 1,
        "experiment": "local_user_owned_dlssnr_weight_resource_extraction",
        "status": "PASS",
        "redistributable": False,
        "source_filename": dll_path.name,
        "source_bytes": dll_path.stat().st_size,
        "source_sha256": sha256_file(dll_path),
        "resource_token": token,
        "resources": entries,
        "notice": "Local interoperability research only. Do not commit or redistribute extracted data.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dll", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--resource-token", default="WEIGHTS_HT")
    args = parser.parse_args()
    try:
        manifest = extract(args.dll, args.output_dir, args.resource_token)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
