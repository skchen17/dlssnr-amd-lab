#!/usr/bin/env python3
"""Lower m16n8k16 f16 MMA to warp shuffles and scalar FP32 accumulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


REG = r"%[A-Za-z_][A-Za-z0-9_]*"
F16_MMA = re.compile(
    rf"mma\.sync\.aligned\.m16n8k16\.row\.col\.f16\.f16\.f16\.f16\s*"
    rf"\{{\s*(?P<d0>{REG})\s*,\s*(?P<d1>{REG})\s*\}}\s*,\s*"
    rf"\{{\s*(?P<a0>{REG})\s*,\s*(?P<a1>{REG})\s*,\s*(?P<a2>{REG})\s*,\s*(?P<a3>{REG})\s*\}}\s*,\s*"
    rf"\{{\s*(?P<b0>{REG})\s*,\s*(?P<b1>{REG})\s*\}}\s*,\s*"
    rf"\{{\s*(?P<c0>{REG})\s*,\s*(?P<c1>{REG})\s*\}}\s*;",
    re.DOTALL,
)


def cd_coord(lane: int, element: int) -> tuple[int, int]:
    group, thread = lane >> 2, lane & 3
    return group + (8 if element >= 2 else 0), thread * 2 + (element & 1)


def a_source(lane: int, element: int, k: int) -> tuple[int, int]:
    row, _ = cd_coord(lane, element)
    return (row & 7) * 4 + ((k & 7) >> 1), (k & 1) + (2 if row >= 8 else 0) + (4 if k >= 8 else 0)


def b_source(lane: int, element: int, k: int) -> tuple[int, int]:
    _, column = cd_coord(lane, element)
    return column * 4 + ((k & 7) >> 1), (k & 1) + (2 if k >= 8 else 0)


def replacement(match: re.Match[str], index: int) -> str:
    d = [match.group("d0"), match.group("d1")]
    a = [match.group(f"a{i}") for i in range(4)]
    b = [match.group(f"b{i}") for i in range(2)]
    c = [match.group(f"c{i}") for i in range(2)]
    r = lambda name: f"%__f16mma_{index}_r_{name}"
    h = lambda name: f"%__f16mma_{index}_h_{name}"
    p = lambda name: f"%__f16mma_{index}_p_{name}"
    b32 = [r(name) for name in (
        "lane", "group_base", "thread", "thread_x8", "source_a", "source_b",
        "word_a", "word_b", "bits", "value_a", "value_b",
        "acc0", "acc1", "acc2", "acc3"
    )]
    b16 = [h(name) for name in ("a", "b", "c", "out0", "out1", "out2", "out3")]
    lines = [
        f".reg .b32 {', '.join(b32)};",
        f".reg .b16 {', '.join(b16)};",
        f".reg .pred {p('valid_a')}, {p('valid_b')};",
        f"mov.u32 {r('lane')}, %laneid;",
        f"and.b32 {r('group_base')}, {r('lane')}, 28;",
        f"and.b32 {r('thread')}, {r('lane')}, 3;",
        f"shl.b32 {r('thread_x8')}, {r('thread')}, 3;",
    ]
    for element in range(4):
        c_word = c[element // 2]
        if element & 1:
            lines.append(f"shr.u32 {r('bits')}, {c_word}, 16;")
        else:
            lines.append(f"and.b32 {r('bits')}, {c_word}, 65535;")
        lines.extend([
            f"cvt.u16.u32 {h('c')}, {r('bits')};",
            f"cvt.f32.f16 {r(f'acc{element}')}, {h('c')};",
        ])
        for k in range(16):
            a_register = a[(1 if element >= 2 else 0) + (2 if k >= 8 else 0)]
            b_register = b[k >> 3]
            column_component = (element & 1) * 4
            b_lane_tail = column_component + ((k & 7) >> 1)
            lines.extend([
                f"add.u32 {r('source_a')}, {r('group_base')}, {(k & 7) >> 1};",
                f"add.u32 {r('source_b')}, {r('thread_x8')}, {b_lane_tail};",
                f"shfl.sync.idx.b32 {r('word_a')}|{p('valid_a')}, {a_register}, {r('source_a')}, 31, -1;",
                f"shfl.sync.idx.b32 {r('word_b')}|{p('valid_b')}, {b_register}, {r('source_b')}, 31, -1;",
            ])
            if k & 1:
                lines.append(f"shr.u32 {r('bits')}, {r('word_a')}, 16;")
            else:
                lines.append(f"and.b32 {r('bits')}, {r('word_a')}, 65535;")
            lines.extend([
                f"cvt.u16.u32 {h('a')}, {r('bits')};",
                (f"shr.u32 {r('bits')}, {r('word_b')}, 16;"
                 if k & 1 else f"and.b32 {r('bits')}, {r('word_b')}, 65535;"),
                f"cvt.u16.u32 {h('b')}, {r('bits')};",
                f"cvt.f32.f16 {r('value_a')}, {h('a')};",
                f"cvt.f32.f16 {r('value_b')}, {h('b')};",
                f"fma.rn.f32 {r(f'acc{element}')}, {r('value_a')}, {r('value_b')}, {r(f'acc{element}')};",
            ])
        lines.append(f"cvt.rn.f16.f32 {h(f'out{element}')}, {r(f'acc{element}')};")
    lines.extend([
        f"mov.b32 {d[0]}, {{{h('out0')}, {h('out1')}}};",
        f"mov.b32 {d[1]}, {{{h('out2')}, {h('out3')}}};",
    ])
    return "{\n" + "\n".join(lines) + "\n}"


def lower(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        result = replacement(match, count)
        count += 1
        return result

    return F16_MMA.sub(replace, text), count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-count", type=int, default=16)
    args = parser.parse_args()
    source = args.input.read_bytes()
    lowered, count = lower(source.decode("utf-8"))
    output = lowered.encode("utf-8")
    remaining = len(F16_MMA.findall(lowered))
    status = "PASS" if count == args.expected_count and remaining == 0 else "FAIL"
    report = {
        "schema": 1,
        "experiment": "ptx_f16_m16n8k16_scalar_lowering",
        "status": status,
        "classification": "SCOPED_RTX_AMD_ORACLE_VALIDATED_PTX_REWRITE",
        "counts_as_s6": False,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "f16_mma_lowered": count,
        "expected_f16_mma": args.expected_count,
        "remaining_f16_mma": remaining,
        "accumulation": "f32_fma_then_single_f16_round",
        "scope": "functional_range_exact; overflow-order parity not claimed",
        "translator_execution_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
