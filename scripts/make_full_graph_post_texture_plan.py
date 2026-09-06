#!/usr/bin/env python3
"""Create a full-graph variant that restores the captured post-block texture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def make_plan(base_path: Path, ptx_path: Path, texture_path: Path,
              output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(output_path)
    plan = json.loads(base_path.read_text(encoding="utf-8-sig"))
    texture = texture_path.read_bytes()
    if len(texture) != 640 * 360 * 8:
        raise ValueError("post texture must be a 640x360 RGBA16F image")
    ptx = ptx_path.read_bytes()
    slots = [record for record in plan.get("slots", []) if int(record["slot"]) == 154]
    if len(slots) != 1:
        raise ValueError("expected exactly one slot 154")
    slot = slots[0]
    slot["ptx"] = str(ptx_path.resolve())
    slot["ptx_sha256"] = sha256(ptx)
    slot["special"] = "post_linear_surface_texture"
    plan["experiment"] = "full_graph_integrated_real_post_texture_replay"
    plan["classification"] = "RX9070XT_REAL_POST_TEXTURE_DIAGNOSTIC"
    plan["post_texture"] = {
        "path": os.path.relpath(texture_path.resolve(), output_path.parent.resolve()),
        "bytes": len(texture),
        "sha256": sha256(texture),
        "width": 640,
        "height": 360,
        "format": "RGBA16F",
        "captured_descriptor": "0x0000080200009801",
        "sampler": "point_border_normalized_coordinates",
    }
    limitations = [item for item in plan.get("limitations", [])
                   if "null SRV" not in item]
    limitations.append(
        "The captured merged texture/sampler at slot154 offset 56 is recreated from "
        "the supplied RGBA16F snapshot using point filtering and border addressing."
    )
    plan["limitations"] = limitations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(args.base.resolve(strict=True), args.ptx.resolve(strict=True),
                     args.texture.resolve(strict=True), args.output.resolve())
    slot = next(record for record in plan["slots"] if int(record["slot"]) == 154)
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve()),
                      "post_texture": plan["post_texture"],
                      "slot154_ptx_sha256": slot["ptx_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
