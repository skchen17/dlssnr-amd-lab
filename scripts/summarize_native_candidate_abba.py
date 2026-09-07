"""Strict A-B-B-A summary for two native full-frame graph configurations."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def p95(values):
    ordered = sorted(values)
    return ordered[math.ceil(.95 * len(ordered)) - 1]


def load(root):
    root = Path(root)
    child = json.loads((root / 'child.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if child.get('checks_pass') is not True or manifest.get('pass') is not True or manifest.get('normal_exit') is not True:
        raise ValueError(f'failed process: {root}')
    if child.get('allocated_after_release_bytes') or child.get('reserved_after_release_bytes'):
        raise ValueError(f'resources not released: {root}')
    if len(child['runs']) != 12:
        raise ValueError(f'exactly 12 runs required: {root}')
    warm = child['runs'][1:]
    hashes = {row['output_sha256'] for row in child['runs']}
    if len(hashes) != 1:
        raise ValueError(f'output changed: {root}')
    host = [row['host_forward_submit_wait_ms'] for row in warm]
    return child, {'directory': str(root), 'median_ms': statistics.median(host),
                   'p95_ms': p95(host), 'warm_samples': len(warm),
                   'kernel_nodes': child.get('gpu_dispatch_count'),
                   'graph_nodes': child.get('gpu_graph_nodes'),
                   'output_sha256': next(iter(hashes)),
                   'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in warm),
                   'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in warm),
                   'device_used_bytes': max(row['device_used_bytes_sample'] for row in warm)}


def summarize(paths):
    loaded = [load(path) for path in paths]
    children, rows = zip(*loaded)
    if len({json.dumps(child['input'], sort_keys=True) for child in children}) != 1:
        raise ValueError('A/B inputs differ')
    if len({row['output_sha256'] for row in rows}) != 1:
        raise ValueError('A/B outputs differ')
    if rows[0]['kernel_nodes'] != rows[3]['kernel_nodes'] or rows[1]['kernel_nodes'] != rows[2]['kernel_nodes']:
        raise ValueError('node count changed within A or B')
    a = statistics.mean((rows[0]['median_ms'], rows[3]['median_ms']))
    b = statistics.mean((rows[1]['median_ms'], rows[2]['median_ms']))
    return {'schema': 1, 'checks_pass': True, 'order': ['A', 'B', 'B', 'A'],
            'rows': list(rows), 'a_mean_of_medians_ms': a, 'b_mean_of_medians_ms': b,
            'a_mean_of_p95_ms': statistics.mean((rows[0]['p95_ms'], rows[3]['p95_ms'])),
            'b_mean_of_p95_ms': statistics.mean((rows[1]['p95_ms'], rows[2]['p95_ms'])),
            'latency_reduction_percent': 100 * (1 - b / a), 'speedup': a / b,
            'a_kernel_nodes': rows[0]['kernel_nodes'], 'b_kernel_nodes': rows[1]['kernel_nodes'],
            'kernel_node_reduction_percent': 100 * (1 - rows[1]['kernel_nodes'] / rows[0]['kernel_nodes']),
            'b_nr_jobs_per_second': 1000 / b, 'b_multiple_of_50ms_target': b / 50,
            'b_multiple_of_20ms_target': b / 20, 'output_sha256': rows[0]['output_sha256'],
            'gpu_profiler_used': False,
            'timing_scope': 'full-frame host submit/wait; finite scan and CPU readback excluded; not kernel-busy sum'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a1', type=Path, required=True)
    parser.add_argument('--b1', type=Path, required=True)
    parser.add_argument('--b2', type=Path, required=True)
    parser.add_argument('--a2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize((args.a1, args.b1, args.b2, args.a2))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
