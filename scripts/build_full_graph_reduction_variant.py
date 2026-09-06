#!/usr/bin/env python3
"""Create a full-graph plan variant using a selected chained-1h PTX module."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FUNCTION = "cc_tinlayout_fused_swin_1h_32_1_chained_fp8"
DEFAULT_TARGETS = ["%r1855", "%r1871", "%r2235", "%r2251", "%r2283"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build(plan: dict, ptx: Path, targets: list[str] | None = None,
          kind: str = "slot3_selective_f16_reduction_candidate") -> tuple[dict, list[int]]:
    output = json.loads(json.dumps(plan))
    changed = []
    digest = sha256(ptx)
    for record in output["slots"]:
        if record["function"] == FUNCTION and int(record["slot"]) in (3, 4):
            record["ptx"] = str(ptx.resolve())
            record["ptx_sha256"] = digest
            changed.append(int(record["slot"]))
    if changed != [3, 4]:
        raise ValueError(f"expected to replace slots 3 and 4, got {changed}")
    output["variant"] = {
        "kind": kind,
        "changed_slots": changed,
        "ptx": str(ptx.resolve()),
        "ptx_sha256": digest,
        "targets": DEFAULT_TARGETS if targets is None else targets,
        "counts_as_s7": False,
    }
    return output, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", action="append")
    parser.add_argument("--kind", default="slot3_selective_f16_reduction_candidate")
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    ptx_path = args.ptx.resolve(strict=True)
    output_path = args.output.resolve()
    if output_path.exists():
        parser.error(f"output already exists: {output_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    output, changed = build(plan, ptx_path, args.target, args.kind)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_path),
                      "changed_slots": changed, "ptx_sha256": sha256(ptx_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
