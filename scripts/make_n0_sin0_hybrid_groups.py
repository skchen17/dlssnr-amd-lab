#!/usr/bin/env python3
"""Create grouped 65K-baseline/524K-override sine models for GPU ablation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

try:
    from scripts.make_n0_sqrt0_boundary_safe_model import corrected
except ModuleNotFoundError:
    from make_n0_sqrt0_boundary_safe_model import corrected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--accepted-model", type=Path, required=True)
    parser.add_argument("--override-model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--segment-min", type=int)
    parser.add_argument("--segment-max", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_model.read_text(encoding="utf-8"))
    accepted = json.loads(args.accepted_model.read_text(encoding="utf-8"))
    override = json.loads(args.override_model.read_text(encoding="utf-8"))
    trace = np.fromfile(args.trace, dtype="<u4").reshape(-1, 2)
    phase, amd = trace[:, 0].view("<f4"), trace[:, 1].view("<f4")
    accepted_value, _ = corrected(phase, amd, accepted)
    override_value, override_segment = corrected(phase, amd, override)
    changed = np.unique(override_segment[
        accepted_value.view("<u4") != override_value.view("<u4")
    ])
    if args.segment_min is not None:
        changed = changed[changed >= args.segment_min]
    if args.segment_max is not None:
        changed = changed[changed <= args.segment_max]
    chunks = np.array_split(changed, args.groups)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, chunk in enumerate(chunks):
        model = copy.deepcopy(base)
        model["experiment"] = f"n0_sin0_hybrid_group_{index}_of_{args.groups}"
        model["classification"] = "GPU_ABLATION_524K_OVERRIDE_GROUP_ON_FLATTENED_65K_BASE"
        for segment in chunk:
            model["coefficients_u32"][int(segment)] = override["coefficients_u32"][int(segment)]
        path = args.output / f"group_{index}.json"
        path.write_text(json.dumps(model, separators=(",", ":")) + "\n", encoding="utf-8")
        summary.append({"group": index, "segment_count": len(chunk),
                        "first_segment": int(chunk[0]) if len(chunk) else None,
                        "last_segment": int(chunk[-1]) if len(chunk) else None})
    report = {"schema": 1, "status": "PASS", "changed_segment_count": len(changed),
              "group_count": args.groups, "groups": summary}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
