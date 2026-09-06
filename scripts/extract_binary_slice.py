#!/usr/bin/env python3
"""Extract an exact bounded slice from a binary experiment artifact."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.bytes < 1:
        parser.error("offset must be non-negative and bytes must be positive")
    data = args.input.read_bytes()
    end = args.offset + args.bytes
    if end > len(data):
        parser.error(f"requested end {end} exceeds input size {len(data)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data[args.offset:end])
    print(f"Extracted {args.bytes} bytes at offset {args.offset}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
