"""Read-only existing-server prerequisite check; never changes drivers or files."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def run(output, host):
    if not host or host.startswith('-') or any(c.isspace() for c in host):
        raise ValueError('explicit SSH host or configured host alias required')
    if output.exists():
        raise FileExistsError(output)
    command = 'nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader; command -v nvcc; command -v wine; command -v python3'
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                           host, command], capture_output=True, text=True, timeout=30)
    report = {'schema': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'host': host, 'command': command, 'exit_code': proc.returncode,
              'stdout': proc.stdout, 'stderr': proc.stderr,
              'status': 'ENVIRONMENT_READ_ONLY_NOT_TEACHER_ACCEPTED',
              'teacher_accepted': False, 'remote_files_created': [],
              'historical_evidence': 'results/20260904_4090d_box_stage_compatibility/manifest.json',
              'next_action': 'Use RTX5070 candidate harness; no driver upgrades or sm120 declaration-only adaptation.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    return proc.returncode


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--host', required=True, help='SSH host/alias; keep private endpoints outside public source')
    a = p.parse_args()
    raise SystemExit(run(a.output, a.host))
