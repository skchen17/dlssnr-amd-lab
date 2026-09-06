from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.parametrize('kind,success',[('same',True),('missing',False),('changed',False),('coverage',False)])
def test_source_gate_rejects_stale_or_legacy_proof(kind,success):
    pwsh=shutil.which('pwsh')
    if not pwsh:pytest.skip('PowerShell7 required')
    helper=Path(__file__).resolve().parents[1]/'scripts/native_preview_provenance.ps1'
    expected={
        'same':"@([pscustomobject]@{Path='pre.py';Hash='AAA'})",
        'missing':'$null',
        'changed':"@([pscustomobject]@{Path='pre.py';Hash='OLD'})",
        'coverage':"@([pscustomobject]@{Path='another.py';Hash='AAA'})",
    }[kind]
    code=f". '{helper.as_posix()}'; $ErrorActionPreference='Stop'; Assert-NativePreviewSources ({expected}) @([pscustomobject]@{{Path='pre.py';Hash='AAA'}})"
    run=subprocess.run([pwsh,'-NoProfile','-Command',code],capture_output=True,text=True,encoding='utf-8',errors='replace')
    assert (run.returncode==0)==success,run.stderr
