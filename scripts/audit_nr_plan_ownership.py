"""Fail closed when an NRPlan result still relies on external allocations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TRUE = ('owns_workspace', 'owns_weights', 'owns_graph_source',
                 'owns_graph_executable', 'deployment_ready')


def audit(stats: dict) -> dict:
    failures = [name for name in REQUIRED_TRUE if stats.get(name) is not True]
    if stats.get('graph_references_external_allocations') is not False:
        failures.append('graph_references_external_allocations=false')
    if int(stats.get('arena_region_count', 0)) <= 0:
        failures.append('arena_region_count>0')
    if int(stats.get('frame_bindings_abi_version', 0)) < 3:
        failures.append('frame_bindings_abi_version>=3')
    return {'schema': 1, 'deployment_ready': not failures,
            'failures': failures, 'resource_stats': stats}


def load_stats(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if 'cpp_nr_plan_resources' in value:
        return value['cpp_nr_plan_resources']
    if 'probe' in value and 'cpp_nr_plan_resources' in value['probe']:
        return value['probe']['cpp_nr_plan_resources']
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_stats(args.input))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['deployment_ready'] else 2)


if __name__ == '__main__':
    main()
