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
