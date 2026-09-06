#!/usr/bin/env python3
"""Diagnostic N0 rewrite: bias the two negative Box-Muller lg2 results toward +Inf."""

from __future__ import annotations

import argparse, hashlib, json, re
from pathlib import Path


LG2 = re.compile(r"lg2\.approx\.ftz\.f32\s+(?P<dst>%r(?:218|222))\s*,\s*(?P<src>%r(?:181|205))\s*;")


def lower(text: str, ulps: int) -> tuple[str, int]:
    count = 0
    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return match.group(0) + f"\nsub.u32 {match.group('dst')}, {match.group('dst')}, {ulps};"
    return LG2.sub(replace, text), count


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('report',type=Path);p.add_argument('--ulps',type=int,required=True,choices=range(1,17));a=p.parse_args()
    source=a.input.read_bytes();text,count=lower(source.decode('utf-8'),a.ulps);output=text.encode('utf-8');passed=count==2
    report={'schema':1,'experiment':'n0_lg2_observed_direction_ulp_bias','status':'PASS'if passed else'FAIL','classification':'DIAGNOSTIC_NOT_PROMOTABLE_WITHOUT_GENERALIZATION','ulps_toward_positive_infinity':a.ulps,'sites_rewritten':count,'source_sha256':hashlib.sha256(source).hexdigest().upper(),'output_sha256':hashlib.sha256(output).hexdigest().upper()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(output);a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,separators=(',',':')));return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
