# Output-only boundary prototype — 2026-09-05

Latest: the actual callback route now performs a fenced copy in isolated
FSR3/FSR4 hosts; see `BOUNDARY_CALLBACK.md`. That fresh stage is ready for a
game observation-only attach after normal exit, not for live output copying.
The primitive results and initial limitations below remain historical context.

The output-copy primitive and independent real-provider controls now pass.
This is NOT a game capture pass, live-hook validation, input/temporal dataset,
NR teacher, or working NR game renderer. No changed DLL was loaded into GoWR.

## Why this boundary

The current game session identifies the actual FFX output resource, but its
observed state at the FFX boundary differs from the FFX-declared UAV state.
Rather than manufacture state knowledge, the proposed output-only path waits
for the game's own complete transition AFTER the successful FFX call. It would
record a copy immediately after forwarding that entire barrier callback, then
restore the exact resulting state before the game records its next command.

The primitive accepts only an exact raw output resource, one mip/array/sample,
RGBA16F, a direct list on the same device, and a complete plane-0/all-plane
transition from state 8 or 64 to 192. It rejects split barriers, unrelated
resources, other states/subresources and simultaneous-access textures. It does
not derive state from UAV ordering or reinterpret the original descriptor.

This output-only operation needs resource/list/queue lifetime proof, not an
invented FFX context. The existing multi-input/context collector and its gates
remain unchanged. This does not solve acquisition of matched color/depth/motion
inputs or temporal metadata, and does not make FSR output an NR reference.

## Results

Stage: `results/20260905_135025_947_ffx_output_boundary_controls/`.

- `primitive/summary.json`: four synthetic cases, 640x360 and actual game
  2342x1317, each exercising source state 8 and 64. 53,037,024 total raw bytes
  match the CPU pattern exactly. A second downstream readback also matches,
  checking source preservation and state restoration. Zero D3D12 errors/warnings.
- Thirty-two negative checks reject invalid barriers/byte limits/duplicate
  recording. Eight early polls return without mapping, including deliberately
  blocked GPU-queue cases. Data is unpacked only after a collector-owned fence.
- `controls/analysis.json`: independent baseline vs output-boundary FSR3 3.1.0
  and FSR4 4.1.1 * runs pass. Four frames/provider, 640x360 RGBA16F. All boundary,
  downstream and independent baseline outputs are byte-identical; zero D3D12
  errors/warnings. These are isolated real-provider controls, not game frames.
- `results/20260905_133833_941_gowr_capture_session/output_boundary_candidates.json`:
  seven of eight detailed existing game calls match the structural boundary
  checks. The first call is excluded because list reset was unobserved. All
  reports keep `live_copy_ready`, `capture_authorized` and NR/game proof false.
- Python regression: 426 passed in 63.71 seconds, including 23 new tests for
  boundary association and artifact verification. Earlier fixture failures were
  corrected by supplying mandatory hook/authorization metadata, not weakening
  production validation.

Probe SHA256: `292058EBBFB1C96F684560B711EF74EFDC620322D9DC5B24301E813B0143EB82`.
Primitive SHA256: `D3D8F703041F7519FE92DA1DA952A326F44469FCF368F0CA05911F4BF86926CD`.
Audited provider SHA256 remains
`77809405A0FF464B63654F1264F0EC0FCF8F243DAC7C15B5F5C032615520D143`.

## Remaining live integration requirements

1. Track the successful FFX call's exact output resource/list generation until
   the first relevant game barrier callback; discard on reset, failed close,
   descriptor mutation, repeated FFX call or lifecycle ambiguity.
2. Validate the WHOLE original barrier array, not just one transition log row.
   The existing trace logs before forwarding and has no barrier-callback IDs;
   it cannot exclude a later transition/alias in that same array. Confirm return
   from original forwarding, callback reentry handling and final output state.
3. Associate the exact successfully closed recording with one submission on a
   verified same-device direct queue. Retain source, readback and command-list
   references until the collector-owned fence completes. Do not release jobs
   on a timeout, failed signal or uncertain submission.
4. Validate the real callback path in isolated provider hosts, including
   on/off equality, multiple transitions in one callback, split barriers,
   reset/repeated submission, reentry, queue mismatch and early-poll rejection.
   Only then prepare a fresh staged game session and a separate bounded
   output-only command. The existing session's `arm` must NOT be repurposed.

The current helper's fail-stop destructor is suitable for isolated tests only:
it terminates the test if an unretired recorded job would be released. A live
owner must retain uncertain jobs; do not drop this helper into a game hook as-is.

Current GoWR PID 30652 remains on the older validated resource-identity stage,
stopped after two windows, six remaining. No game UI input, texture copy, game
file replacement or restart was performed during this prototype work.

The existing FSR4 actual-size repeatability failure is NOT waived: the actual-
size test above is synthetic copying, while provider equality used 640x360.
Full-network NR quality, real-game generalization and performance still fail
or remain unverified independently of this integration work.

## Run commands

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
# A new absolute staging path is required. No game attach/replacement occurs.
$stage=Join-Path (Get-Location) ('results\{0}_ffx_output_boundary_controls' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
.\scripts\run_ffx_output_boundary.ps1 -FreshStageDirectory $stage
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q tests/test_ffx_output_boundary.py tests/test_ffx_output_boundary_candidates.py
# Re-analyze existing game evidence; use a new report filename.
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_output_boundary_candidates.py results/20260905_133833_941_gowr_capture_session/session.jsonl --output results/20260905_133833_941_gowr_capture_session/output_boundary_candidates_new.json
```
