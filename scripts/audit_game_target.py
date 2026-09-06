"""Read-only God of War Ragnarok target inventory; never loads game code.

PE imports establish static candidate interfaces only, not observed runtime use.
Output is restricted to the lab results directory, outside the game directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import pefile


def pe_inventory(path: Path) -> dict:
    data = path.read_bytes()
    pe = pefile.PE(data=data, fast_load=True)
    try:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY[n] for n in (
            "IMAGE_DIRECTORY_ENTRY_IMPORT", "IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT",
            "IMAGE_DIRECTORY_ENTRY_EXPORT", "IMAGE_DIRECTORY_ENTRY_RESOURCE")])
        imports = []
        for field, delayed in (("DIRECTORY_ENTRY_IMPORT", False), ("DIRECTORY_ENTRY_DELAY_IMPORT", True)):
            for descriptor in getattr(pe, field, []):
                imports.append({"dll": descriptor.dll.decode("utf-8", "replace"), "delayed": delayed,
                    "symbols": [i.name.decode("utf-8", "replace") if i.name else f"ordinal:{i.ordinal}"
                                for i in descriptor.imports]})
        versions = {}
        for group in getattr(pe, "FileInfo", []):
            for item in group:
                for table in getattr(item, "StringTable", []):
                    for key, value in table.entries.items():
                        if key in (b"FileVersion", b"ProductVersion", b"ProductName", b"CompanyName"):
                            versions[key.decode()] = value.decode("utf-8", "replace")
        return {"name": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest().upper(),
                "machine": hex(pe.FILE_HEADER.Machine), "version": versions, "imports": imports,
                "exports": [s.name.decode("utf-8", "replace") for s in
                    getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []) if s.name]}
    finally:
        pe.close()


def audit(game: Path, output: Path) -> dict:
    game, output = game.resolve(strict=True), output.resolve()
    lab_results = Path(__file__).resolve().parents[1] / "results"
    if not output.is_relative_to(lab_results) or output.is_relative_to(game):
        raise ValueError("output must be under lab results, never the game directory")
    if not (game / "GoWR.exe").is_file():
        raise ValueError("GoWR.exe is absent")
    names = ["GoWR.exe", "nvngx_dlss.dll", "nvngx_dlssg.dll", "nvngx_dlssnr.dll",
             "sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "sl.dlss_g.dll",
             "amd_fidelityfx_dx12.dll", "version.dll", "dxgi.dll", "d3d12.dll"]
    inventory = [pe_inventory(game / name) for name in names if (game / name).is_file()]
    settings = {}
    settings_path = game / "settings.ini"
    if settings_path.is_file():
        wanted = {"Adapter", "WindowSize", "DisplayMode", "ScalerMethod", "ScalerQuality",
                  "DynamicScalingMode", "FrameGen", "Vsync"}
        for line in settings_path.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in wanted:
                settings[key.strip()] = value.split(";", 1)[0].strip()
    markers = [str(p.relative_to(game)) for p in game.rglob("*dlssnr*") if p.is_file()]
    report = {"experiment": "gowr_read_only_static_target_audit", "game_directory": str(game),
              "game_executable": str(game / "GoWR.exe"), "settings": settings, "inventory": inventory,
              "nr_named_files": markers, "game_launched": False, "game_files_modified": False,
              "runtime_interfaces_observed": False, "frame_capture_ready": False,
              "rtx_nr_reference_ready": False, "game_runtime_ready": False,
              "limitations": ["Imports/exports and filenames are static evidence, not runtime observations.",
                  "No NR-named file does not prove absence of dynamically loaded external code.",
                  "Existing version.dll must not be overwritten.",
                  "SR/FSR output is not a DLSS-NR ground-truth target.",
                  "The native neural graph and temporal/game bindings are still incomplete."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game, args.output)
    print(json.dumps({"settings": result["settings"], "nr_named_files": result["nr_named_files"]}, ensure_ascii=False))
    for item in result["inventory"]:
        print(item["name"], item["version"])
        print("graphics imports:", {d["dll"]: [s for s in d["symbols"] if s.startswith(("sl", "ffx", "D3D12", "CreateDXGI"))]
                                  for d in item["imports"] if any(t in d["dll"].lower() for t in ("d3d", "dxgi", "interposer", "fidelity"))})
        if item["name"] in ("sl.interposer.dll", "amd_fidelityfx_dx12.dll", "version.dll"):
            print("export count:", len(item["exports"]))
