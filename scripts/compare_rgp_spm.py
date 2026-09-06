"""Compare two RGP SPM summaries without substituting host timing for GPU activity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    data=json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema")!=1 or not data.get("active_sample_count") or not data.get("sampled_active_time_ms"):
        raise ValueError(f"invalid RGP SPM summary: {path}")
    return data


def counters(summary: dict) -> dict[str,dict]:
    # The profile has repeated generic names such as Requests. Only consume the
    # uniquely named derived counters used below.
    result={}
    for row in summary["counters"]:
        name=row["name"]
        if name not in result:result[name]=row
    return result


def summarize(label: str, summary: dict, work_items: int, dispatches: int, peak_dram_gbps: float|None) -> dict:
    if work_items<=0 or dispatches<=0:raise ValueError("positive work and dispatch counts required")
    c=counters(summary);active_ms=float(summary["sampled_active_time_ms"])
    inactive_ms=float(summary["sampled_inactive_gap_ms"]);observed_ms=active_ms+inactive_ms
    def mean(name):return float(c[name]["active_mean"])
    def total(name):return float(c[name]["active_sum"])
    def gbps(name):return total(name)/1e6/active_ms
    local_gbps=gbps("Local video memory bytes")
    memory_busy=mean("Memory unit busy");memory_stalled=mean("Memory unit stalled")
    gap_fraction=inactive_ms/observed_ms if observed_ms else 0.0
    if memory_busy>=80 and memory_stalled>=10:
        classification="memory-pipeline-bound"
    elif gap_fraction>=0.25 and memory_busy<70:
        classification="launch-or-scheduling-bound"
    else:
        classification="mixed-or-undetermined"
    return {
        "label":label,"classification":classification,"work_items":work_items,"dispatches":dispatches,
        "sampled_active_time_ms":active_ms,"sampled_inactive_gap_ms":inactive_ms,
        "inactive_gap_fraction":gap_fraction,"active_segments":summary["active_segments"],
        "active_us_per_dispatch":active_ms*1000.0/dispatches,
        "active_us_per_work_item":active_ms*1000.0/work_items,
        "memory_unit_busy_percent":memory_busy,"memory_unit_stalled_percent":memory_stalled,
        "write_unit_stalled_percent":mean("Write unit stalled"),
        "l0_hit_percent":mean("L0 cache hit"),"l2_hit_percent":mean("L2 cache hit"),
        "fetch_bytes":total("Fetch size"),"write_bytes":total("Write size"),
        "local_video_memory_bytes":total("Local video memory bytes"),
        "fetch_gbps_during_active_samples":gbps("Fetch size"),
        "write_gbps_during_active_samples":gbps("Write size"),
        "local_video_memory_gbps_during_active_samples":local_gbps,
        "peak_dram_gbps":peak_dram_gbps,
        "sampled_peak_dram_fraction":None if peak_dram_gbps is None else local_gbps/peak_dram_gbps,
        "fetch_bytes_per_work_item":total("Fetch size")/work_items,
        "write_bytes_per_work_item":total("Write size")/work_items,
        "local_video_memory_bytes_per_work_item":total("Local video memory bytes")/work_items,
        "lds_bank_conflict_percent":mean("LDS Bank Conflict"),
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,required=True);parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--baseline-label",default="baseline");parser.add_argument("--candidate-label",default="candidate")
    parser.add_argument("--baseline-work-items",type=int,required=True);parser.add_argument("--candidate-work-items",type=int,required=True)
    parser.add_argument("--baseline-dispatches",type=int,required=True);parser.add_argument("--candidate-dispatches",type=int,required=True)
    parser.add_argument("--peak-dram-gbps",type=float);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    baseline_raw,candidate_raw=load(args.baseline),load(args.candidate)
    if baseline_raw["asic"]["name"]!=candidate_raw["asic"]["name"]:
        raise ValueError("profiles are from different devices")
    baseline=summarize(args.baseline_label,baseline_raw,args.baseline_work_items,args.baseline_dispatches,args.peak_dram_gbps)
    candidate=summarize(args.candidate_label,candidate_raw,args.candidate_work_items,args.candidate_dispatches,args.peak_dram_gbps)
    ratios={}
    for key in ("sampled_active_time_ms","active_us_per_work_item","fetch_bytes_per_work_item",
                "write_bytes_per_work_item","local_video_memory_bytes_per_work_item"):
        ratios[f"candidate_to_baseline_{key}"]=candidate[key]/baseline[key] if baseline[key] else None
    report={"schema":1,"checks_pass":all(math.isfinite(x) for x in (
                baseline["sampled_active_time_ms"],candidate["sampled_active_time_ms"])),
            "device":baseline_raw["asic"],"baseline":baseline,"candidate":candidate,"ratios":ratios,
            "method":"Official RGP counter collection decoded from public AMD RDF chunks. Active and inactive times use SPM GPU timestamps; no host submit/wait timing is substituted.",
            "classification_rules":{"memory-pipeline-bound":"memory busy >=80% and stalled >=10%",
                "launch-or-scheduling-bound":"inactive gap fraction >=25% and memory busy <70%"},
            "limitations":["SPM is sampled and profiling perturbs timing.",
                "Fetch/write counters include cache-path traffic; Local video memory bytes is the DRAM traffic measure.",
                "A capture must cover equivalent work before whole-call ratios are interpreted."]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2));return 0 if report["checks_pass"] else 2


if __name__=="__main__":raise SystemExit(main())
