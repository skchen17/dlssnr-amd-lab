#!/usr/bin/env python3
"""Extract one self-contained PTX entry for translator compatibility probes.

The NVIDIA runtime containers group several kernels in one PTX module. A
translator can reject the entire module because an unused entry contains an
unsupported instruction. This tool preserves module directives and globals,
then emits exactly one requested `.entry` body so compatibility is measured at
the function granularity used by LaunchCuKernelChain.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ENTRY_RE_TEMPLATE = r"(?m)^\.visible\s+\.entry\s+{name}\s*\("
ANY_ENTRY_RE = re.compile(r"(?m)^\.visible\s+\.entry\s+")


def find_balanced_body_end(text: str, entry_start: int) -> int:
    opening = text.find("{", entry_start)
    if opening < 0:
        raise ValueError("entry has no opening brace")
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError("entry body has unbalanced braces")


def extract_entry(text: str, name: str) -> tuple[str, dict[str, object]]:
    match = re.search(ENTRY_RE_TEMPLATE.format(name=re.escape(name)), text)
    if not match:
        raise ValueError(f"PTX entry not found: {name}")

    first_entry = ANY_ENTRY_RE.search(text)
    if not first_entry:
        raise ValueError("module contains no PTX entries")

    entry_end = find_balanced_body_end(text, match.start())
    preamble = text[: first_entry.start()].rstrip()
    entry = text[match.start() : entry_end].rstrip()
    output = f"{preamble}\n\n{entry}\n"
    metadata: dict[str, object] = {
        "schema_version": 1,
        "entry": name,
        "source_bytes": len(text.encode("utf-8")),
        "output_bytes": len(output.encode("utf-8")),
        "source_entry_count": len(ANY_ENTRY_RE.findall(text)),
        "output_entry_count": 1,
        "version": (re.search(r"(?m)^\.version\s+(\S+)", text) or [None, None])[1],
        "target": (re.search(r"(?m)^\.target\s+(\S+)", text) or [None, None])[1],
    }
    return output, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    text = raw.decode("utf-8").rstrip("\x00")
    output, metadata = extract_entry(text, args.entry)
    metadata["source"] = str(args.input.resolve())
    metadata["output"] = str(args.output.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
