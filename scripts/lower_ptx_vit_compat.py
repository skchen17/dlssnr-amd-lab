#!/usr/bin/env python3
"""Lower ViT PTX ordering and vector-reduction forms unsupported by ZLUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


VECTOR_RED = re.compile(
    r"(?m)^(?P<i>[ \t]*)red\.global\.v4\.f16x2\.add\.noftz\s+"
    r"\[(?P<addr>%rd[0-9]+)\],\s*\{\s*"
    r"(?P<r0>%r[0-9]+),\s*(?P<r1>%r[0-9]+),\s*"
    r"(?P<r2>%r[0-9]+),\s*(?P<r3>%r[0-9]+)\s*\};[ \t]*$"
)
RELEASE_FENCE = re.compile(r"(?m)^(?P<i>[ \t]*)fence\.release\.gpu;[ \t]*$")


def lower(text: str) -> tuple[str, dict]:
    vector_count = 0
    fence_count = 0

    def replace_vector(match: re.Match[str]) -> str:
        nonlocal vector_count
        vector_index = vector_count
        vector_count += 1
        indent = match.group("i")
        address = match.group("addr")
        blocks = []
        for index in range(4):
            stem = f"__dlssnr_red_{vector_index}_{index}"
            loop = f"$L__DLSSNR_RED_{vector_index}_{index}"
            blocks.append("\n".join((
                indent + "{",
                indent + f".reg .b32 %{stem}_old, %{stem}_assumed, %{stem}_new;",
                indent + f".reg .pred %{stem}_retry;",
                indent + f"ld.global.b32 %{stem}_old, [{address}+{index * 4}];",
                loop + ":",
                indent + f"mov.b32 %{stem}_assumed, %{stem}_old;",
                indent + f"add.f16x2 %{stem}_new, %{stem}_assumed, "
                         f"{match.group(f'r{index}')};",
                indent + f"atom.global.cas.b32 %{stem}_old, "
                         f"[{address}+{index * 4}], %{stem}_assumed, %{stem}_new;",
                indent + f"setp.ne.u32 %{stem}_retry, %{stem}_old, %{stem}_assumed;",
                indent + f"@%{stem}_retry bra {loop};",
                indent + "}",
            )))
        return "\n".join(blocks)

    def replace_fence(match: re.Match[str]) -> str:
        nonlocal fence_count
        fence_count += 1
        return match.group("i") + "membar.gl;"

    lowered = VECTOR_RED.sub(replace_vector, text)
    lowered = RELEASE_FENCE.sub(replace_fence, lowered)
    return lowered, {
        "vector_f16x2_reductions_lowered": vector_count,
        "release_gpu_fences_lowered": fence_count,
        "remaining_vector_f16x2_reductions": len(VECTOR_RED.findall(lowered)),
        "remaining_release_gpu_fences": len(RELEASE_FENCE.findall(lowered)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-vector-red", type=int, required=True)
    parser.add_argument("--expected-release-fence", type=int, required=True)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    lowered, counts = lower(source_bytes.decode("utf-8"))
    lowered_bytes = lowered.encode("utf-8")
    status = "PASS" if counts == {
        "vector_f16x2_reductions_lowered": args.expected_vector_red,
        "release_gpu_fences_lowered": args.expected_release_fence,
        "remaining_vector_f16x2_reductions": 0,
        "remaining_release_gpu_fences": 0,
    } else "FAIL"
    report = {
        "schema": 1,
        "experiment": "ptx_vit_zluda_compat_lowering",
        "status": status,
        "classification": "SEMANTICS_PRESERVING_OR_STRENGTHENING_PTX_REWRITE",
        "counts_as_s7": False,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(lowered_bytes).hexdigest(),
        "expected_vector_f16x2_reductions": args.expected_vector_red,
        "expected_release_gpu_fences": args.expected_release_fence,
        **counts,
        "semantics": (
            "A v4 f16x2 reduction is split into four packed f16x2 atomic-CAS "
            "loops; release.gpu is strengthened to membar.gl. Exact "
            "RTX graph-edge replay remains the normative oracle."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(lowered_bytes)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
