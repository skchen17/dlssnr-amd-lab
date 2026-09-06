# Within-call resource identity diagnostics — 2026-09-05

Latest follow-up: `OUTPUT_BOUNDARY.md` describes a validated isolated output-
copy primitive and seven structural candidates from these logs. It has NOT
been integrated into this live session; the PID, DLL and window budget below
remain unchanged. Existing pre-forward logs do not authorize game copying.

This records FFX descriptor pointers and barrier metadata, not texture pixels.
It does not authorize capture, infer GPU completion, or prove NR quality.

## Validated revision

Stage: `results/20260905_133303_873_ffx_resource_identity_stage/`.
Session SHA256: `325123ED040C3E7FF35E485696A1813599DF6CC394611241100A4DB6D41DA5E0`.
Control/attach client SHA256: `147514F6F5C90B5B16132BCD959085D434B5C0FF8F18F8611C7ED2BE21519480`.

All eight independent native FSR3/FSR4 runs pass. Outputs remain byte-identical
with diagnostics on/off; zero D3D12 errors/warnings. Isolated capture still
saves five textures / 2,764,804 exact raw bytes per provider. Session, window,
state-trace, command-path and new resource-path control reports all pass.
Python regression: 403 passed in 46.49 seconds.

Both providers' controls record output-targeted UAV barriers without matching
output state transitions inside the FFX call. UAV ordering and state transitions
must not be conflated. The previous recorder detailed watched transitions only;
its zero-transition observation never established zero matching UAV barriers.

The first four FFX calls in each explicit window include before/after snapshots
of all seven resource roles, declared FFX states, command-list pointer, raw
resource pointers, canonical COM IUnknown identity and resource descriptions.
All legacy barrier types (transition/UAV/aliasing), including unmatched resources,
are detailed during those calls. A separate 512-event budget and explicit limit
event prevent the 1,024-event state trace from hiding these records.

The analyzer recomputes raw and COM matches, validates call/window/generation
association, snapshot ordering, command-path counts and descriptor mutation.
Equal dimensions are never identity proof; equal COM identity is not a license
to propagate GPU storage/state across wrappers, queues or lifetimes. Missing or
capped observations remain explicit. Existing capture state gates are unchanged.

## Current process and commands

After the user's normal exit of PID 24336, new GoWR PID 30652 started at
`2026-09-05T13:38:00.3596469+08:00`. Bootstrap:
`results/20260905_133800_343_gowr_ffx_observation/`; deferred session:
`results/20260905_133833_941_gowr_capture_session/`.
The stage is already loaded: never reattach, rebuild, overwrite or hot-unload it.
Game files were not deployed or changed; their audited hashes and the loaded
stage's on-disk hash were rechecked after observation.

## Playable-scene result

Two explicit windows completed and stopped; 6/8 remain. `command_path_v2.json`
contains 128 paired calls, each with 28 Dispatch callbacks and 26 legacy barriers
from 16 callbacks, no enhanced/indirect/other-interface callbacks, no table
mismatches and all-zero original return statuses. Counts are not GPU completion.

`resource_path_v2.json`: eight complete detailed calls / 224 detail events,
no truncation or incomplete calls. Each records 18 UAV and eight transition
barriers. Exactly two UAV barriers per call match the actual output raw pointer
AND canonical COM identity. There are zero identity-only matches, no matching
input/output transition barriers, no null barrier resources, and no change to
the snapshotted descriptor pointers/states/list. Output is 2342x1317 format 10.
The second warmed-up window reproduces the same identity finding.

This rules out a raw-vs-COM pointer mismatch as the explanation for the sampled
output UAV barriers. It does not rule out unobserved transition paths or prove
all resource lifetimes/queue state. The resource recorder observes metadata only.

`state_v2.json`: 2,048 capped events, 50 complete FFX call windows and 98 complete
submission batches (zero incomplete/invalid). All 50 first observed post-call
output transitions start at 64; no watched transitions occur within those calls.
`prefix_v2.json` associates 49 calls, with last observed output prefix state 64
in 47 and 8 in two. These are partial observations, not a complete state proof.
The session still lacks observed context creation and has zero ready candidates,
failures or input-copy events. No arm was sent. Full NR runtime readiness is false.

`provider_metadata_audit.json` again identifies the private FSR4 `4.1.1 *`
metadata and driver callback route, using 20,901 read-only process bytes. It does
not establish public-query provenance or initialize the session context registry.
Computer-use was limited to selecting, observing and activating the game window;
no settings, save, movement or combat input was sent.

Next: locate the missing state/ownership coverage at the game-side FFX boundary
and establish context lifetime from creation. Use known output identity to keep
the next trace narrowly targeted; do not repeat generic pointer discovery, infer
state from UAV ordering, or relax the capture gate. The current stopped process
can provide more bounded observations; a changed DLL requires normal exit and
a fresh independently validated stage, not hot replacement.

Run from the repository root. Verify PID/start time against provenance before
using live commands after any exit. Report destinations must be new.

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$stage='C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\20260905_133303_873_ffx_resource_identity_stage'
$run='results\20260905_133833_941_gowr_capture_session'
& "$stage\ffx_session_control.exe" 30652 'C:\DATA\GAME\GODOFWAR\GoWR.exe' "$stage\ffx_capture_session.dll" observe
& "$stage\ffx_session_control.exe" 30652 'C:\DATA\GAME\GODOFWAR\GoWR.exe' "$stage\ffx_capture_session.dll" stop
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_resource_path.py "$run\session.jsonl" --output "$run\resource_path_new.json"
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_command_path.py "$run\session.jsonl" --output "$run\command_path_new.json"
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_state_trace.py "$run\session.jsonl" --output "$run\state_new.json"
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_prefix_states.py "$run\session.jsonl" --output "$run\prefix_new.json"
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/audit_ffx_live_entries.py --provenance "$run\provenance.json" --log "$run\session.jsonl" --provider-route --output "$run\provider_route_new.json"
```

Only `observe` and `stop` are used here; do not use `arm`. Independent controls
do not waive FSR4 actual-size repeatability failure, missing live context/state
proof, full-network NR quality failure, temporal or performance requirements.
