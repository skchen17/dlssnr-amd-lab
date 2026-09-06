"""Package only research host + source scripts; no weights, inputs or vendor DLLs."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from freeze_native_baseline import digest


def package(root, build, output):
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists():
        raise FileExistsError(output)
    files = {
        build / 'dlss5-feed-host64.exe': 'build/dlss5-feed-host64.exe',
        root / 'scripts/run_teacher_sequence.py': 'scripts/run_teacher_sequence.py',
        root / 'scripts/audit_teacher_sequences.py': 'scripts/audit_teacher_sequences.py',
        root / 'docs/ROCM_TEACHER_PACKAGE.md': 'README.md',
    }
    for source in files:
        if not source.is_file():
            raise FileNotFoundError(source)
    output.mkdir(parents=True)
    records = []
    for source, relative in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append({'path': relative, 'bytes': target.stat().st_size, 'sha256': digest(target)})
    report = {'schema': 1, 'status': 'BUILT_NOT_RTX_VALIDATED', 'files': records,
              'contains_weights_or_vendor_dlls': False, 'contains_game_sequences': False,
              'requires_real_capture_manifest': True, 'teacher_accepted': False}
    (output / 'package_manifest.json').write_text(json.dumps(report, indent=2))
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for file in output.rglob('*'):
            if file.is_file():
                z.write(file, file.relative_to(output))
    print(json.dumps({'archive': str(archive.resolve()), 'sha256': digest(archive), **report}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    package(Path(__file__).resolve().parents[1], a.build, a.output)
