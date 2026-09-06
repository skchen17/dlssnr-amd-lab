"""Validate the observer harness JSON and GPU copy without promoting game readiness."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    def invalid_constant(value):
        raise ValueError(f"nonstandard JSON numeric constant: {value}")
    return [json.loads(line, parse_constant=invalid_constant) for line in path.read_text().splitlines() if line.strip()]


def analyze(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    decoded = read_jsonl(directory / "decoded.jsonl")
    generic = read_jsonl(directory / "header_only.jsonl")
    dispatches = [x for x in decoded if x["event"] == "ffxDispatch"]
    full = [x for x in dispatches if x.get("body_decoded")]
    payload = (directory / "gpu_output.raw").read_bytes()
    expected = bytes((i * 13 + i // 77) % 251 for i in range(640 * 360 * 8))
    checks = {
        "harness_pass": manifest.get("status") == "PASS",
        "event_counts": len(decoded) == 9 and len(generic) == 2 and len(dispatches) == 5,
        "known_upscale_fields": len(full) == 2 and all(x["render_size"] == [640, 360] and
            x["jitter"] == [.25, -.125] and x["motion_scale"] == [640, 360] and x["reset"] == 1 and
            x["resources"]["color"]["format"] == 4 and x["resources"]["output"]["width"] == 640 for x in full),
        "short_body_not_read": any(x.get("body_reason") == "short_dispatch" and not x["body_decoded"] for x in dispatches),
        "unknown_not_cast": any(x.get("body_reason") == "unknown_descriptor_type" for x in dispatches),
        "cyclic_chain_bounded": any(x.get("extension_chain") == "cycle" for x in decoded),
        "nonfinite_json_safe": len(full) == 2 and full[1]["camera_far"] is None,
        "header_only_no_resource_reads": all(x["profile"] == 0 and not x["body_decoded"] and "resources" not in x for x in generic),
        "budget_does_not_stop_backend": manifest.get("dropped_events") == 8,
        "gpu_output_independent_pattern": payload == expected,
        "no_debug_errors": manifest.get("debug_errors") == 0,
        "not_game_claim": manifest.get("game_launched") is False and manifest.get("game_abi_verified") is False,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "gpu_output_sha256": hashlib.sha256(payload).hexdigest().upper(),
            "gpu_frame_capture_ready": False, "game_abi_verified": False,
            "scope": "Pinned-ABI synthetic caller and real AMD GPU copy; metadata-only observer."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = analyze(args.directory)
    (args.directory / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
