"""Run real input sequences through the Windows NVIDIA component host.

Outputs remain candidates: successful public DLAA and feature18 log messages
alone do not establish correct NR tensor identity, color contract or quality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
import numpy as np
from audit_teacher_sequences import audit, finite_number


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest().upper()


def prepare(manifest, output):
    audit(manifest, require_complete=False, require_teacher=False)
    data = json.loads(manifest.read_text(encoding='utf-8-sig'))
    for sequence in data['sequences']:
        frames = sequence['frames']
        if len(frames) > 32 or not frames[0]['reset']:
            raise ValueError('teacher sequences need <=32 frames and explicit initial reset')
        flags = sequence.get('ngx_create_flags')
        if type(flags) is not int or flags < 0 or flags > 0x7fffffff:
            raise ValueError('capture-derived NGX create flags required')
        first = frames[0]
        for f in frames:
            if (f['width'], f['height'], f['color_mode'], f['color_contract_id']) != (first['width'], first['height'], first['color_mode'], first['color_contract_id']):
                raise ValueError('split sequence at size/color-contract change')
            if f['width'] * f['height'] > 3840 * 2160:
                raise ValueError('teacher harness initially bounded to 4K pixel count')
            if bool(flags & 1) != (f['color_mode'] == 'HDR'):
                raise ValueError('HDR flag disagrees with capture metadata')
            if len(f.get('jitter', [])) != 2 or not all(finite_number(x) for x in f['jitter']):
                raise ValueError('capture-derived jitter required')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source_manifest.json').write_text(json.dumps(data, indent=2))
    prepared = []
    for number, sequence in enumerate(data['sequences']):
        folder = output / f'sequence_{number:03d}'
        folder.mkdir()
        first, frames = sequence['frames'][0], sequence['frames']
        lines = [f"NRTEACHER1 {first['width']} {first['height']} {len(frames)} {sequence['ngx_create_flags']}"]
        for i, frame in enumerate(frames):
            target = folder / str(i)
            target.mkdir()
            for name in ('color', 'depth', 'motion'):
                source = (manifest.parent / frame[name]['path']).resolve(strict=True)
                shutil.copyfile(source, target / f'{name}.raw')
                if sha(target / f'{name}.raw') != frame[name]['sha256'].upper():
                    raise ValueError('input changed during preparation')
            values = [i, int(frame['reset']), frame['pre_exposure'], frame['exposure_scale'], *frame['motion_scale'], *frame['jitter']]
            lines.append(' '.join(map(str, values)))
        (folder / 'sequence.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
        prepared.append((folder, sequence))
    return prepared


def run(manifest, payload, host, output, prepare_only=False):
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError(output)
    records = []
    report = {'schema': 1, 'status': 'NOT_RUN', 'training_allowed': False,
              'teacher_accepted': False, 'sequences': records}
    if not prepare_only:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        if not re.search(r'RTX\s+50\d\d', gpu):
            raise ValueError('this package is gated to RTX 50 series; 4090D requires a separate validated path')
        report['gpu'] = gpu
        source_files = {host: 'dlss5-feed-host64.exe', payload / 'ReShade64.dll': 'dxgi.dll',
                        payload / 'renodx-dlss5-02.addon64': 'renodx-dlss5.addon64',
                        payload / 'nvngx_dlss.dll': 'nvngx_dlss.dll',
                        payload / 'nvngx_dlssnr.dll': 'nvngx_dlssnr.dll'}
        for p in source_files:
            if not p.is_file():
                raise FileNotFoundError(p)
        report['binaries'] = [{'name': name, 'sha256': sha(p)} for p, name in source_files.items()]
    prepared = prepare(manifest, output)
    report['status'] = 'PREPARED_NOT_RUN'
    try:
        if prepare_only:
            return report
        for folder, sequence in prepared:
            work = folder / 'work'
            work.mkdir()
            for source, name in source_files.items():
                shutil.copyfile(source, work / name)
            with (folder / 'stdout.log').open('w', encoding='utf-8') as stream:
                env = {k: v for k, v in os.environ.items() if not k.startswith('MODULE_TRACE_')}
                env['PYTHONNOUSERSITE'] = '1'
                proc = subprocess.run([str((work / 'dlss5-feed-host64.exe').resolve()), '--teacher-sequence', str(folder.resolve())],
                                      cwd=work, stdout=stream, stderr=subprocess.STDOUT,
                                      env=env, timeout=600,
                                      creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            logs = ''
            for name in ('ReShade.log', 'dlss5-feed-host.log', 'ReShade.ini'):
                p = work / name
                if p.is_file():
                    shutil.copyfile(p, folder / name)
                    logs += p.read_text(encoding='utf-8', errors='replace')
            rec = {'sequence_id': sequence['id'], 'exit_code': proc.returncode,
                   'feature18_created_log': bool(re.search(r'feature\s+18.*created', logs, re.I)),
                   'feature18_evaluated_log': bool(re.search(r'feature\s+18.*evaluation succeeded', logs, re.I)),
                   'frames': []}
            records.append(rec)
            if proc.returncode:
                raise RuntimeError('teacher host failed; stop, do not retry automatically')
            for i, frame in enumerate(sequence['frames']):
                directory = folder / str(i)
                uploaded = sha(directory / 'uploaded_color.raw')
                if uploaded != frame['color']['sha256'].upper():
                    raise ValueError('actual uploaded texture differs from source')
                candidate = directory / 'candidate_output.raw'
                values = np.fromfile(candidate, dtype='<f2')
                if values.size != frame['width'] * frame['height'] * 4 or not np.isfinite(values).all():
                    raise ValueError('candidate output missing, malformed or nonfinite')
                rec['frames'].append({'frame_id': frame['frame_id'], 'uploaded_color_sha256': uploaded,
                                      'candidate_output_sha256': sha(candidate)})
            if not rec['feature18_created_log'] or not rec['feature18_evaluated_log']:
                raise RuntimeError('feature18 evidence missing; public DLAA success is insufficient')
        report['status'] = 'CANDIDATES_COLLECTED_NOT_TEACHER_ACCEPTED'
        return report
    except Exception as error:
        report['status'] = 'FAILED_NO_RETRY'
        report['error'] = str(error)
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        # Explicit allowlist: never return user NVIDIA/ReShade binaries or model files.
        archive = output.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in output.rglob('*'):
                if path.is_file() and 'work' not in path.relative_to(output).parts and path.suffix in ('.json', '.txt', '.log', '.raw', '.ini'):
                    z.write(path, path.relative_to(output))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--payload-dir', type=Path, required=True)
    p.add_argument('--host', type=Path, default=Path(__file__).resolve().parents[1] / 'build/dlss5-feed-host64.exe')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.manifest, a.payload_dir, a.host, a.output, a.prepare_only), indent=2))
