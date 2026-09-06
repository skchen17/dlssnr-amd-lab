#!/usr/bin/env python3
"""Semantics-preserving PTX rewrites for constructs unsupported by ZLUDA v7."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DISCARD_MOV = re.compile(
    r"mov\.b64\s*\{\s*(?P<low>_|%[A-Za-z0-9_]+)\s*,\s*"
    r"(?P<high>_|%[A-Za-z0-9_]+)\s*\}\s*,\s*"
    r"(?P<src>%[A-Za-z0-9_]+)\s*;"
)

WIDE_STORE = re.compile(
    r"\{\s*\.reg\s+\.b128\s+(?P<tmp>[A-Za-z0-9_%]+)\s*;\s*"
    r"mov\.b128\s+(?P=tmp)\s*,\s*\{\s*"
    r"(?P<r0>%[A-Za-z0-9_]+)\s*,\s*(?P<r1>%[A-Za-z0-9_]+)\s*,\s*"
    r"(?P<r2>%[A-Za-z0-9_]+)\s*,\s*(?P<r3>%[A-Za-z0-9_]+)\s*\}\s*;\s*"
    r"st\.global\.L1::no_allocate\.b128\s*\[\s*(?P<addr>%[A-Za-z0-9_]+)\s*\]\s*,\s*"
    r"(?P=tmp)\s*;\s*\}"
)

RELEASE_CACHE_HINT_STORE = re.compile(
    r"st\.release\.(?P<scope>cta|gpu|sys)\.global\.L1::no_allocate\.(?P<type>[A-Za-z0-9]+)"
)

RELAXED_CACHE_HINT_LOAD = re.compile(
    r"ld\.relaxed\.(?P<scope>cta|gpu|sys)\.global\.L1::no_allocate\.(?P<type>[A-Za-z0-9]+)"
)

SHARED_CTA_SCOPE = re.compile(r"(?P<op>ld|st)\.shared::cta\.")


def lower(text: str) -> tuple[str, dict]:
    discard_count = 0

    def replace_discard(match: re.Match[str]) -> str:
        nonlocal discard_count
        low, high, src = match.group("low", "high", "src")
        if low != "_" and high != "_":
            return match.group(0)
        scratch = f"%__dlssnr_discard_{discard_count}"
        discard_count += 1
        low = scratch if low == "_" else low
        high = scratch if high == "_" else high
        return f"{{ .reg .b32 {scratch}; mov.b64 {{{low}, {high}}}, {src}; }}"

    lowered = DISCARD_MOV.sub(replace_discard, text)
    wide_count = 0

    def replace_wide(match: re.Match[str]) -> str:
        nonlocal wide_count
        wide_count += 1
        addr = match.group("addr")
        registers = [match.group(f"r{i}") for i in range(4)]
        stores = [f"st.global.b32 [{addr}], {registers[0]};"]
        stores.extend(
            f"st.global.b32 [{addr}+{i * 4}], {registers[i]};" for i in range(1, 4)
        )
        return "{ " + " ".join(stores) + " }"

    lowered = WIDE_STORE.sub(replace_wide, lowered)
    release_cache_hint_count = 0

    def replace_release_cache_hint(match: re.Match[str]) -> str:
        nonlocal release_cache_hint_count
        release_cache_hint_count += 1
        return f"st.release.{match.group('scope')}.global.{match.group('type')}"

    lowered = RELEASE_CACHE_HINT_STORE.sub(replace_release_cache_hint, lowered)
    relaxed_cache_hint_count = 0

    def replace_relaxed_cache_hint(match: re.Match[str]) -> str:
        nonlocal relaxed_cache_hint_count
        relaxed_cache_hint_count += 1
        return f"ld.relaxed.{match.group('scope')}.global.{match.group('type')}"

    lowered = RELAXED_CACHE_HINT_LOAD.sub(replace_relaxed_cache_hint, lowered)
    shared_cta_scope_count = 0

    def replace_shared_cta_scope(match: re.Match[str]) -> str:
        nonlocal shared_cta_scope_count
        shared_cta_scope_count += 1
        # The shared state space is CTA-local by definition. Removing this
        # explicit scope qualifier does not change visibility or ordering.
        return f"{match.group('op')}.shared."

    lowered = SHARED_CTA_SCOPE.sub(replace_shared_cta_scope, lowered)
    remaining_discard = sum(
        match.group("low") == "_" or match.group("high") == "_"
        for match in DISCARD_MOV.finditer(lowered)
    )
    report = {
        "discard_mov_b64_lowered": discard_count,
        "wide_mov_store_b128_lowered": wide_count,
        "release_cache_hint_stores_lowered": release_cache_hint_count,
        "relaxed_cache_hint_loads_lowered": relaxed_cache_hint_count,
        "shared_cta_scopes_lowered": shared_cta_scope_count,
        "remaining_discard_operands": remaining_discard,
        "remaining_b128_moves": lowered.count("mov.b128"),
        "remaining_no_allocate_b128_stores": lowered.count("st.global.L1::no_allocate.b128"),
    }
    return lowered, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-discard", type=int, default=4)
    parser.add_argument("--expected-wide-store", type=int, default=4)
    parser.add_argument("--expected-release-store", type=int, default=0)
    parser.add_argument("--expected-relaxed-load", type=int, default=0)
    parser.add_argument("--expected-shared-cta-scope", type=int, default=0)
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    source = source_bytes.decode("utf-8")
    lowered, counts = lower(source)
    lowered_bytes = lowered.encode("utf-8")
    status = (
        "PASS"
        if counts == {
            "discard_mov_b64_lowered": args.expected_discard,
            "wide_mov_store_b128_lowered": args.expected_wide_store,
            "release_cache_hint_stores_lowered": args.expected_release_store,
            "relaxed_cache_hint_loads_lowered": args.expected_relaxed_load,
            "shared_cta_scopes_lowered": args.expected_shared_cta_scope,
            "remaining_discard_operands": 0,
            "remaining_b128_moves": 0,
            "remaining_no_allocate_b128_stores": 0,
        }
        else "FAIL"
    )
    report = {
        "schema": 1,
        "experiment": "zluda_ptx_compat_lowering",
        "status": status,
        "classification": "SEMANTICS_PRESERVING_PTX_REWRITE",
        "counts_as_s6": False,
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(lowered_bytes).hexdigest(),
        "expected_discard_mov_b64": args.expected_discard,
        "expected_wide_mov_store_b128": args.expected_wide_store,
        "expected_release_cache_hint_stores": args.expected_release_store,
        "expected_relaxed_cache_hint_loads": args.expected_relaxed_load,
        "expected_shared_cta_scopes": args.expected_shared_cta_scope,
        **counts,
        "wide_store_note": "b128 non-atomic vector store split into four ordered per-thread b32 stores; L1 no-allocate is a cache hint",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(lowered_bytes)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
