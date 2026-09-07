"""Summarize diagnostic native graph events; never call these hardware busy time."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import hashlib
import math
import re
from pathlib import Path
import statistics


def dot_names(text):
    """Resolve names by dependency order, never by GraphGetNodes enumeration."""
    labels = dict(re.findall(r'"(graph_\d+_node_\d+)"\[.*?label="(.*?)"\];', text, re.S))
    edges = re.findall(r'"(graph_\d+_node_\d+)"\s*->\s*"(graph_\d+_node_\d+)"', text)
    incoming = {b for a, b in edges}
    roots = set(labels) - incoming
    if len(roots) != 1:
        raise ValueError('DOT must have a single serial root')
    successor = {}
    for a, b in edges:
        if a in successor:
            raise ValueError('DOT branches')
        successor[a] = b
    node = roots.pop()
    names = []
    while node:
        label = labels[node].strip().splitlines()
        names.append(label[1].strip() if len(label) > 1 else label[0])
        if len(names) > len(labels):
            raise ValueError('DOT cycle')
        node = successor.get(node)
    if len(names) != len(labels):
        raise ValueError('DOT disconnected')
    return names


def load_sample(path):
    sample = json.loads(path.read_text())
    dot = path.parent / 'original_graph.dot'
    if dot.exists():
        names = dot_names(dot.read_text())
        if len(names) != len(sample['nodes']):
            raise ValueError('DOT count differs from timed topology')
        for row, name in zip(sample['nodes'], names):
            if row['kind'] == 'kernel':
                row['name'] = name
    return sample


def short_name(name):
    match = re.match(r'_Z(?:N12_GLOBAL__N_1)?(\d+)(.*)', name)
    if not match:
        return name
    function = match.group(2)[:int(match.group(1))]
    template = re.match(r'ILi(\d+)', match.group(2)[int(match.group(1)):])
    return function + (f'<{template.group(1)}>' if template else '')


def time_stats(values):
    ordered = sorted(values)
    return {'median_ms': statistics.median(values),
            'p95_nearest_rank_ms': ordered[math.ceil(.95*len(ordered))-1],
            'min_ms': min(values), 'max_ms': max(values), 'samples': len(values)}


def summarize(samples):
    if not samples:
        raise ValueError('no timing samples')
    first = samples[0]['nodes']
    signatures = [(r['index'], r['scope'], r['kind'], r.get('name')) for r in first]
    grouped = defaultdict(list)
    per_node = []
    for sample in samples:
        if [(r['index'], r['scope'], r['kind'], r.get('name')) for r in sample['nodes']] != signatures:
            raise ValueError('node topology changed across samples')
        totals = defaultdict(float)
        for r in sample['nodes']:
            if not math.isfinite(r['event_ms']) or r['event_ms'] < 0:
                raise ValueError('invalid event duration')
            stage = r['scope'].split('/')[0]
            totals[(stage, r.get('name', r['kind']))] += r['event_ms']
            totals[(stage, 'TOTAL')] += r['event_ms']
        for key, value in totals.items():
            grouped[key].append(value)
    for i, row in enumerate(first):
        times = [s['nodes'][i]['event_ms'] for s in samples]
        per_node.append({**row, 'median_ms': statistics.median(times),
                         'min_ms': min(times), 'max_ms': max(times)})
    aggregate = [{'stage': stage, 'operation': op, 'median_ms': statistics.median(times),
                  'min_ms': min(times), 'max_ms': max(times),
                  'nodes_per_frame': sum(r['scope'].split('/')[0] == stage and
                       (op == 'TOTAL' or r.get('name', r['kind']) == op) for r in first)}
                 for (stage, op), times in grouped.items()]
    return {'measurement': 'HIP event intervals in instrumented serial graph, includes event/dispatch gaps; not hardware busy time',
            'sample_count': len(samples), 'aggregate': aggregate, 'nodes': per_node}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reject-invalid-samples', action='store_true',
                   help='Exclude entire invalid timestamp frames; retain explicit rejection evidence')
    p.add_argument('--baseline', type=Path, nargs='*', default=[])
    a = p.parse_args()
    paths = [path for root in a.input for path in sorted(root.glob('operations_*.json'))]
    samples, rejected, accepted_paths = [], [], []
    for path in paths:
        sample = load_sample(path)
        bad = [r for r in sample['nodes'] if not math.isfinite(r['event_ms']) or r['event_ms'] < 0]
        if bad and a.reject_invalid_samples:
            rejected.append({'source': str(path.resolve()), 'reason': 'invalid HIP event interval; entire frame excluded, no clamping', 'invalid_nodes': bad})
        else:
            samples.append(sample)
            accepted_paths.append(path)
    report = summarize(samples)
    report['rejected_samples'] = rejected
    report['accepted_sources'] = [str(p.resolve()) for p in accepted_paths]
    report['sources'] = [str(p.resolve()) for p in paths]
    comparisons = []
    pooled = defaultdict(list)
    for mode, roots in [('baseline', a.baseline), ('instrumented', a.input)]:
        for root in roots:
            result = json.loads((root / 'child.json').read_text())
            if not result['checks_pass'] or len(set(result['sample_hashes'])) != 1:
                raise ValueError('measurement correctness gate failed')
            pooled[mode].extend(result['event_samples_ms'])
            comparisons.append({'mode': mode, 'source': str(root.resolve()),
                **time_stats(result['event_samples_ms']), 'sha256': result['output_sha256']})
    report['whole_frame_runs'] = comparisons
    report['whole_frame_pooled'] = {mode: time_stats(values) for mode, values in pooled.items()}
    repo = Path(__file__).resolve().parents[1]
    evidence = {Path(__file__).resolve(), repo / 'scripts/validate_native_71_record_nrplan.py',
                repo / 'scripts/measure_native_operations.ps1',
                repo / 'tools/native_nr_plan/operation_timing.inc',
                repo / 'tools/native_nr_plan/nr_plan.cpp', repo / 'tools/native_nr_plan/nr_plan.h'}
    for root in a.input + a.baseline:
        command = json.loads((root / 'process.json').read_text())['command']
        for flag in ('--dll', '--package', '--arena', '--stage-topology', '--split-topology',
                     '--vit-topology', '--bottleneck-topology', '--transition-topology', '--edge-topology'):
            evidence.add(Path(command[command.index(flag)+1]))
    report['evidence_sha256'] = {}
    for path in sorted(evidence):
        with path.open('rb') as stream:
            report['evidence_sha256'][str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Native 71-block operation timing', '', report['measurement'], '',
             f"Samples: {report['sample_count']}", '',
             '| Stage | Operation (fused kernel name) | Nodes/frame | Median ms | Min ms | Max ms |',
             '|---|---|---:|---:|---:|---:|']
    for r in report['aggregate']:
        lines.append(f"| {r['stage']} | {short_name(r['operation'])} | {r['nodes_per_frame']} | {r['median_ms']:.4f} | {r['min_ms']:.4f} | {r['max_ms']:.4f} |")
    lines += ['', '## Every executed node', '', '| Index | Stage/block | Kernel/copy | Median ms |', '|---:|---|---|---:|']
    for r in report['nodes']:
        lines.append(f"| {r['index']} | {r['scope']} | {short_name(r.get('name', r['kind']))} | {r['median_ms']:.5f} |")
    lines += ['', '## Whole-frame controls (not instrumented stage sums)', '',
              '```json', json.dumps(report['whole_frame_pooled'], indent=2), '```',
              '', '## Rejected timestamp frames', '', '```json', json.dumps(rejected, indent=2), '```']
    (a.output / 'operations.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps([r for r in report['aggregate'] if r['operation']=='TOTAL'], indent=2))


if __name__ == '__main__':
    main()
