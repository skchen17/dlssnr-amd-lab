#!/usr/bin/env python3
"""Lower immediate CTA mbarrier arrive/wait pairs to a stronger bar.sync."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


INIT = re.compile(
    r"(?m)^(?P<i>[ \t]*)mbarrier\.init\.shared(?:::[A-Za-z0-9_]+)?\.b64"
    r"\s+\[[^\]\r\n]+\],\s*[^;\r\n]+;[ \t]*$"
)
ARRIVE = re.compile(
    r"(?m)^(?P<i>[ \t]*)mbarrier\.arrive\.shared(?:::[A-Za-z0-9_]+)?\.b64"
    r"\s+(?P<state>%rd[0-9]+),\s*\[[^\]\r\n]+\],\s*(?P<count>%r[0-9]+|1);[ \t]*$"
)
TRY_WAIT = re.compile(
    r"(?m)^(?P<i>[ \t]*)mbarrier\.try_wait\.shared(?:::[A-Za-z0-9_]+)?\.b64"
    r"\s+(?P<pred>[A-Za-z_$][A-Za-z0-9_$]*|%p[0-9]+),\s*"
    r"\[[^\]\r\n]+\],\s*%rd[0-9]+;[ \t]*$"
)
ASYNC_BULK = re.compile(
    r"(?m)^(?P<i>[ \t]*)cp\.async\.bulk\.shared::cta\.global\."
    r"mbarrier::complete_tx::bytes\s+\[(?P<dst>%r[0-9]+)\],\s*"
    r"\[(?P<src>%rd[0-9]+)\],\s*(?P<size>%r[0-9]+),\s*"
    r"\[%r[0-9]+\];[ \t]*$"
)
EXPECT_TX = re.compile(
    r"(?m)^(?P<i>[ \t]*)mbarrier\.expect_tx\.relaxed\.cta\.shared::cta\.b64"
    r"\s+\[%r[0-9]+\],\s*%r[0-9]+;[ \t]*$"
)
ELECT = re.compile(
    r"(?m)^(?P<i>[ \t]*)elect\.sync\s+_\|(?P<pred>[A-Za-z_$][A-Za-z0-9_$]*|%p[0-9]+),"
    r"\s*(?P<mask>%r[0-9]+);[ \t]*$"
)


def lower(text: str) -> tuple[str, dict]:
    init_count = 0
    arrive_count = 0
    wait_count = 0
    bulk_count = 0
    expect_count = 0
    elect_count = 0

    def bulk_replacement(match: re.Match[str]) -> str:
        nonlocal bulk_count
        index = bulk_count
        bulk_count += 1
        indent = match.group("i")
        prefix = f"__dlssnr_bulk_{index}"
        loop = f"$L__DLSSNR_BULK_{index}"
        done = f"$L__DLSSNR_BULK_DONE_{index}"
        return "\n".join((
            indent + "{",
            indent + f".reg .b32 %{prefix}_v0, %{prefix}_v1, %{prefix}_v2, %{prefix}_v3;",
            indent + f".reg .b32 %{prefix}_off, %{prefix}_dst;",
            indent + f".reg .b64 %{prefix}_off64, %{prefix}_src;",
            indent + f".reg .pred %{prefix}_done;",
            indent + f"mov.u32 %{prefix}_off, 0;",
            loop + ":",
            indent + f"setp.ge.u32 %{prefix}_done, %{prefix}_off, {match.group('size')};",
            indent + f"@%{prefix}_done bra {done};",
            indent + f"add.u32 %{prefix}_dst, {match.group('dst')}, %{prefix}_off;",
            indent + f"cvt.u64.u32 %{prefix}_off64, %{prefix}_off;",
            indent + f"add.u64 %{prefix}_src, {match.group('src')}, %{prefix}_off64;",
            indent + f"ld.global.v4.b32 {{%{prefix}_v0, %{prefix}_v1, %{prefix}_v2, %{prefix}_v3}}, [%{prefix}_src];",
            indent + f"st.shared.v4.b32 [%{prefix}_dst], {{%{prefix}_v0, %{prefix}_v1, %{prefix}_v2, %{prefix}_v3}};",
            indent + f"add.u32 %{prefix}_off, %{prefix}_off, 16;",
            indent + f"bra {loop};",
            done + ":",
            indent + "}",
        ))

    def expect_replacement(match: re.Match[str]) -> str:
        nonlocal expect_count
        expect_count += 1
        return match.group("i") + "// dlssnr: synchronous bulk copy needs no transaction expectation"

    def elect_replacement(match: re.Match[str]) -> str:
        nonlocal elect_count
        prefix = text[:match.start()]
        mask = re.escape(match.group("mask"))
        assignments = re.findall(rf"mov\.b32\s+{mask},\s*([^;\r\n]+);", prefix)
        if not assignments or assignments[-1].strip().lower() not in {
                "-1", "0xffffffff", "4294967295"}:
            return match.group(0)
        elect_count += 1
        return (f"{match.group('i')}setp.eq.u32 {match.group('pred')}, "
                "%laneid, 0;")

    def init_replacement(match: re.Match[str]) -> str:
        nonlocal init_count
        init_count += 1
        return match.group("i") + "// dlssnr: mbarrier init removed; pair uses bar.sync 0"

    def arrive_replacement(match: re.Match[str]) -> str:
        nonlocal arrive_count
        arrive_count += 1
        indent = match.group("i")
        return f"{indent}bar.sync 0;\n{indent}mov.b64 {match.group('state')}, 0;"

    def wait_replacement(match: re.Match[str]) -> str:
        nonlocal wait_count
        wait_count += 1
        return f"{match.group('i')}mov.pred {match.group('pred')}, 1;"

    lowered = ELECT.sub(elect_replacement, text)
    lowered = ASYNC_BULK.sub(bulk_replacement, lowered)
    lowered = EXPECT_TX.sub(expect_replacement, lowered)
    lowered = INIT.sub(init_replacement, lowered)
    lowered = ARRIVE.sub(arrive_replacement, lowered)
    lowered = TRY_WAIT.sub(wait_replacement, lowered)
    remaining = len(re.findall(r"\bmbarrier\.", lowered))
    remaining_bulk = len(re.findall(r"\bcp\.async\.bulk\.", lowered))
    remaining_elect = len(re.findall(r"\belect\.sync\b", lowered))
    return lowered, {
        "full_warp_elect_lowered": elect_count,
        "async_bulk_lowered": bulk_count,
        "mbarrier_expect_tx_lowered": expect_count,
        "mbarrier_init_lowered": init_count,
        "mbarrier_arrive_lowered": arrive_count,
        "mbarrier_try_wait_lowered": wait_count,
        "remaining_mbarrier_instructions": remaining,
        "remaining_async_bulk_instructions": remaining_bulk,
        "remaining_elect_instructions": remaining_elect,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-init", type=int, required=True)
    parser.add_argument("--expected-bulk", type=int, required=True)
    parser.add_argument("--expected-elect", type=int, required=True)
    parser.add_argument("--expected-arrive", type=int, required=True)
    parser.add_argument("--expected-try-wait", type=int, required=True)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    lowered, counts = lower(source_bytes.decode("utf-8"))
    lowered_bytes = lowered.encode("utf-8")
    status = "PASS" if counts == {
        "full_warp_elect_lowered": args.expected_elect,
        "async_bulk_lowered": args.expected_bulk,
        "mbarrier_expect_tx_lowered": args.expected_bulk,
        "mbarrier_init_lowered": args.expected_init,
        "mbarrier_arrive_lowered": args.expected_arrive,
        "mbarrier_try_wait_lowered": args.expected_try_wait,
        "remaining_mbarrier_instructions": 0,
        "remaining_async_bulk_instructions": 0,
        "remaining_elect_instructions": 0,
    } else "FAIL"
    report = {
        "schema": 1,
        "experiment": "ptx_immediate_cta_mbarrier_lowering",
        "status": status,
        "classification": "CTA_BARRIER_STRENGTHENING_PENDING_ORACLE_VALIDATION",
        "counts_as_s6": False,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(lowered_bytes).hexdigest(),
        "expected_init": args.expected_init,
        "expected_bulk": args.expected_bulk,
        "expected_elect": args.expected_elect,
        "expected_arrive": args.expected_arrive,
        "expected_try_wait": args.expected_try_wait,
        **counts,
        "semantics": (
            "Each elected-lane asynchronous bulk copy is made synchronous, and each "
            "all-CTA mbarrier arrive is immediately followed by its wait loop; bar.sync 0 "
            "is a stronger synchronization point with no useful overlap removed."
        ),
        "oracle_validation": "required from exact RTX before/after replay",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(lowered_bytes)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
