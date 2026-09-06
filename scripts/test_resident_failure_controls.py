"""Hostile local IPC fixtures; isolated WARP session only, never GoWR."""
import argparse
import json
import mmap
import subprocess
import uuid
from pathlib import Path

try:
    from .resident_frame_worker import WinEvents, HEADER, MAGIC, MAPPING_BYTES, HEADER_BYTES, MAX_BYTES
    from .resident_full_graph import replay
except ImportError:
    from resident_frame_worker import WinEvents, HEADER, MAGIC, MAPPING_BYTES, HEADER_BYTES, MAX_BYTES
    from resident_full_graph import replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--present', action='store_true', help='test the separate display-referred bridge')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    exe = args.stage / ('resident_present_selftest.exe' if args.present else 'ffx_resident_session_selftest.exe')
    report = {'status': 'RUNNING', 'cases': [], 'fixture_only': True}
    for mode in ('error_status', 'wrong_sequence', 'nonfinite', 'timeout'):
        events = WinEvents()
        name = 'Local\\DLSSNRLive_' + uuid.uuid4().hex
        request = events.create(name + '_request')
        response = events.create(name + '_response')
        shared = mmap.mmap(-1, MAPPING_BYTES, tagname=name)
        shared[:HEADER.size] = HEADER.pack(MAGIC, 0, 0, 0, 0, 0, 1, 0, 1)
        # Dedicated pre-game test stage only; refuse running game before changing config.
        check = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            'if(Get-Process GoWR -ErrorAction SilentlyContinue){exit 1}'], timeout=15)
        if check.returncode:
            raise RuntimeError('game is running')
        (args.stage / 'worker_name.txt').write_text(name + '\n', encoding='ascii')
        child = subprocess.Popen([str(exe.resolve()), str((args.output / mode).resolve()), '--warp', '--expect-worker-failure'])
        try:
            if events.k.WaitForSingleObject(request, 10000) != 0:
                raise RuntimeError('fixture never requested')
            header = list(HEADER.unpack(shared[:HEADER.size]))
            header[2] = header[1]
            header[6] = 1
            if mode == 'error_status':
                header[6] = 3
            elif mode == 'wrong_sequence':
                header[2] += 1
            elif mode == 'nonfinite':
                shared[HEADER_BYTES + MAX_BYTES:HEADER_BYTES + MAX_BYTES + 2] = b'\x00\x7e'
            if mode != 'timeout':
                shared[:HEADER.size] = HEADER.pack(*header)
                events.k.SetEvent(response)
            code = child.wait(timeout=30)
            result = json.loads((args.output / mode / 'summary.json').read_text())
            status = json.loads((args.output / mode / 'resident_status.json').read_text())
            passed = code == 0 and result['status'] == 'PASS' and result['expected_worker_failure'] and status['failed'] and not status['enabled'] and status['completed_frames'] == 0
            report['cases'].append({'mode': mode, 'pass': passed, 'summary': result, 'session': status})
            replay.write_progress(args.output / 'failure_controls.json', report)
            if not passed:
                raise RuntimeError('fallback control failed')
        finally:
            if child.poll() is None:
                child.kill()  # exact child standalone test, never a game or arbitrary process
                child.wait(timeout=5)
            shared.close()
            events.close()
    report['status'] = 'PASS'
    replay.write_progress(args.output / 'failure_controls.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
