"""Aggregate nine isolated RGP captures for the full-grid Output Head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ORDER=("input_pack","ffn","qkv_project","qkv_norm","qk","softmax","pv","projection","tail")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[]
    for index,name in enumerate(ORDER,1):
        directory=args.root/f"{index:02d}_{name}"
        spm=json.loads((directory/"spm_summary.json").read_text(encoding="utf-8-sig"))
        capture=json.loads((directory/"manifest.json").read_text(encoding="utf-8-sig"))
        target=json.loads((directory/"target"/"target.json").read_text(encoding="utf-8-sig"))
        if (not capture.get("checks_pass") or capture.get("dispatch_count")!=1 or
                capture.get("trigger_mode")!="interactive_after_ready" or capture.get("ready_token_observed") is not True or
                not target.get("checks_pass") or target.get("stage")!=name or target.get("native_dispatches_after_ready")!=1):
            raise ValueError(f"invalid or uncorrelated capture {directory}")
        counters={row["name"]:row for row in spm["counters"]}
        active=float(spm["sampled_active_time_ms"])
        def mean(counter):return float(counters[counter]["active_mean"])
        def total(counter):return float(counters[counter]["active_sum"])
        rows.append({"dispatch":index,"stage":name,"sampled_active_time_ms":active,
            "sampled_inactive_gap_ms":float(spm["sampled_inactive_gap_ms"]),
            "memory_unit_busy_percent":mean("Memory unit busy"),
            "memory_unit_stalled_percent":mean("Memory unit stalled"),
            "write_unit_stalled_percent":mean("Write unit stalled"),
            "l0_hit_percent":mean("L0 cache hit"),"l2_hit_percent":mean("L2 cache hit"),
            "fetch_bytes":total("Fetch size"),"write_bytes":total("Write size"),
            "local_video_memory_bytes":total("Local video memory bytes"),
            "capture_sha256":capture["profile_sha256"]})
    total_active=sum(row["sampled_active_time_ms"] for row in rows)
    for row in rows:row["active_time_fraction"]=row["sampled_active_time_ms"]/total_active if total_active else 0
    report={"schema":1,"checks_pass":len(rows)==len(ORDER),"geometry":[640,384],"cta_count":3969,
        "schedule":"whole_grid","isolated_dispatches":rows,"sum_isolated_active_time_ms":total_active,
        "method":"One bounded official RGP counter capture per known native dispatch. Times are sampled GPU timestamp spans, not host submit/wait times.",
        "limitations":["Each dispatch is captured in a separate process, so sums exclude inter-dispatch overlap and are profiling-perturbed.",
            "Runtime occupancy is not present in the selected SPM set; static resource occupancy is reported by audit_matrix_isa.py."]}
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
    return 0 if report["checks_pass"] else 2


if __name__=="__main__":raise SystemExit(main())
