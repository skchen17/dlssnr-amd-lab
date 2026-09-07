"""Repository-local guard for GPU experiments after a recorded device reset."""
from __future__ import annotations

import json
from pathlib import Path


HALT_FILE=Path(__file__).resolve().parents[1]/'safety'/'GPU_PROFILE_HALT.json'


def require_gpu_tests_enabled(scope='GPU experiment',*,profiler=False):
    if HALT_FILE.is_file():
        state=json.loads(HALT_FILE.read_text(encoding='utf-8'))
        halted=state.get('halted') is True if profiler else state.get('non_profiler_gpu_tests_halted',True) is True
        if halted:
            raise RuntimeError(f'{scope} blocked by {HALT_FILE}; review the recorded LiveKernelEvent 141 incident first')
        # A limited resume authorizes one reviewed gate, not an unbounded chain
        # of experiments.  Once that gate records an explicit negative next-gate
        # decision, every subsequent non-profiler probe must stop until the
        # incident record is deliberately updated after review.  Missing legacy
        # fields retain the old behaviour so historical fixtures remain usable.
        last_gate=state.get('last_minimal_gate')
        if not profiler and isinstance(last_gate,dict) and \
                last_gate.get('authorizes_next_gpu_gate') is False:
            raise RuntimeError(
                f'{scope} blocked by {HALT_FILE}; the last reviewed minimal gate '
                'does not authorize another GPU submission')
