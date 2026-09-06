#!/usr/bin/env python3
"""Create an exactly sized zero-filled binary test tensor."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path);parser.add_argument("--bytes",required=True,type=int);args=parser.parse_args()
    if args.bytes <= 0 or args.output.exists():
        parser.error("--bytes must be positive and output must not already exist")
    data=bytes(args.bytes);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(data)
    print(f"{args.output.resolve()} bytes={args.bytes} sha256={hashlib.sha256(data).hexdigest().upper()}");return 0


if __name__ == "__main__":
    raise SystemExit(main())
