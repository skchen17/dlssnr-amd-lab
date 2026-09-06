# Explicit observation windows — 2026-09-05

Latest: resource identity diagnostics in PID 30652 completed and stopped two
windows (6/8 remain). See `RESOURCE_IDENTITY.md` for current stage, evidence and
commands. All PIDs/stages below are historical; do not reuse their live commands.

Previous: a newer command-path revision was loaded in PID 24336, with one
playable-scene window complete and stopped (7/8 remain). It observes internal
compute and legacy barriers and confirms private FSR4 metadata. See
`COMMAND_PATH.md` for the exact NEW loaded bundle and current run commands.
The paths/PID below describe the previous process and must not be reused.

The previous revision was loaded in GoWR PID 23192, started at
`2026-09-05T12:50:20.9814199+08:00`, after the user's normal exit of PID 15544.
The loaded DLL is in the validated stage below, NOT `build/`. Do not overwrite,
unload, or stack a replacement onto that pinned session. Two windows have been
observed and stopped; six remain. No capture arm or GPU texture copy occurred.

Validated bundle:
`results/20260905_124552_868_ffx_observe_window_stage/`.

## Actual playable-scene evidence

Bootstrap: `results/20260905_125020_962_gowr_ffx_observation/`.
Session: `results/20260905_125046_331_gowr_capture_session/`.
All three chained inspection/session attaches passed exact-binary controls;
audited game images remain unchanged. The assistant only inspected the game
window this turn and did not issue movement, combat, menu, save or settings input.
The existing scene was visibly loaded before triggering the first window.

- `window1_analysis.json`: first window, 64 candidates, 1,024 trace events,
  25 complete FFX call windows, 48 complete batches and 1 truncated batch.
- `window2_analysis.json`: cumulative two-window report, 128 candidates,
  2,048 trace events, 50 complete call windows, 96 complete batches, 1 incomplete
  and 0 invalid batches. Both windows explicitly hit the trace cap. Zero ready
  candidates, failure events or recorded input copies. These are partial traces,
  not complete frames or GPU execution/fence proofs.
- `prefix_analysis_v2.json`: 48 uniquely associated calls (24 per window).
  Predecessor command-list transitions are ordered by explicit batch membership,
  not by interleaved CPU log order; the FFX list is cut off at dispatch entry.
  Last observed prefix states: color/reactive 47x64 + 1x8; depth/motion 48x64;
  output 46x64 + 2x8. Declared inputs are 192 and output 8. All 50 complete call
  windows have a first observed post-call output transition starting at 64.
  No watched legacy transitions are observed inside those calls; this is NOT
  proof that the call does no GPU work.
- The only depth chain-gap row is window 1/sample 1/batch 1 (160 -> next-before
  192 at sequence 12, then 16 -> next-before 160 at 13). That first sample lacks
  an observed FFX-list reset. Sample 2 also has last observed output state 8.
  These cannot be explained away as warm-up without more coverage. No chain-
  continuity/capture permission is inferred even for the other rows.
- `live_entry_audit.json`: read-only QUERY_INFORMATION + VM_READ snapshot,
  20,592 bytes read, zero process writes. The first 32 bytes of all four FFX
  exports match the exact on-disk provider image. Two pinned sampled command
  lists share a D3D12Core vtable; Close/Reset/ResourceBarrier point to the current
  stage's session DLL, Dispatch to D3D12Core. No recognized entry jumps in these
  samples. Internal redirects, cached calls, alternate interfaces and enhanced
  barriers are NOT excluded. Raw comparison is not relocation-normalized.

Session context knowledge remains false: creation occurred before session
attach. The outer bootstrap did observe this process's same context pointer
with max dimensions 1552x872 -> 2342x1317; that independent metadata is not
silently imported into the live registry and does not identify the provider.
No next capture is authorized by these reports; NR quality remains failed.
Full regression with the new analyzers: **366 passed in 63.33 seconds**.

Next implementation target is command-path coverage (including alternative
barrier/list paths) with independent no-change controls, then validated context
and state/lifetime tracking before any bounded fenced game texture collection.
No additional observation-only restart is necessary on the current version.

## Behavior

- Config version 2 starts idle and requires `capture_enabled=0`. Context
  creation/destruction metadata is still tracked. Menu dispatches do not consume
  the candidate or state-trace budget while idle.
- `observe` (command 3) starts a new numbered window, never enables capture.
  Each window has a 60-second tracking deadline, 64 configured candidate samples
  and 1,024 trace-event cap. The explicit trace-limit marker remains mandatory.
  At most 8 windows can be started per session; no unlimited log growth.
- `stop` (command 4) disables observation without unhooking or modifying the
  game's commands. Stop before opening another window, or wait for its deadline.
- Both commands reject armed/in-flight/completed capture and failed sessions.
  They cannot cancel GPU work or turn a failed session into a passing one.
- New windows clear local states and reset-observed flags; COM references and
  context epochs remain pinned/bounded. Window tokens reject callbacks crossing
  an old/new-window boundary. Context data from another process is never reused.
- Version 1 behavior and separate explicit `arm`/`poll` remain supported. Do
  **not** use arm as a logging-restart command. Game capture gates are unchanged.

Every log event now includes a window ID. Queue submit-begin snapshots include
a unique batch ID and ordered list identities/generations, matched by later
submit-return and submit-end events. Analysis does not use adjacent log lines
as proof of a shared batch and never promotes CPU order to global GPU state.
Queue waits/signals, alternate command interfaces, and enhanced barriers still
need additional coverage before resolving the game's output-state discrepancy.

## Validation

`controls/session_analysis.json`, `state_trace_analysis.json`, and
`window_analysis.json` all PASS for exact FSR 3.1.0 and 4.1.1* providers.

The window host renders four controlled frames: frames 0 and 2 are idle;
frames 1 and 3 are separately triggered windows. Depth and motion are generated
on a **different command list** from FSR; both are submitted in one ordered
two-list batch. Each provider logs 42 trace events across the two active windows.
All four outputs exactly match the independent single-list baseline, with zero
D3D12 errors/warnings. No capture artifacts or capture lifecycle events exist
in these observation runs. Local depth/motion remain unknown on the FSR list,
and the capture gate correctly remains false rather than copying inferred states.

Busy-window rejection, stop/restart, maximum 8 windows, invalid command rejection,
and attempts to observe/stop an armed capture are exercised. Capture regression
still saves one frame / 5 exact textures per provider with unchanged output.
Full Python regression: **346 passed in 59.98 seconds**.

This validates the recorder and a synthetic two-list control, not a real game
frame, temporal NR quality, FSR4 at the failing game-size baseline, or DLSS5.

## Commands

To implement a future native revision, build in a NEW directory while the
current game remains open (never overwrite its loaded staged bundle):

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$stage=Join-Path (Get-Location) ('results\{0}_ffx_observe_window_stage' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
.\scripts\build_all.ps1 -Only ffx_capture_session,ffx_dispatch_probe,ffx_session_control,ffx_live_attach -FreshOutputDirectory $stage
.\scripts\run_ffx_staged_session.ps1 -StageDirectory $stage
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q
```

After normal game exit, use the existing bootstrap and inspector attach steps
with fresh process provenance. For the final session attach, select the staged
bundle explicitly; no copying into the game directory is necessary:

```powershell
# Set these from the NEW successful bootstrap and contract-inspection results:
# $newGamePid, $newObservationRun, $newContractInspectionRun
$stage='C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\20260905_124552_868_ffx_observe_window_stage'
.\scripts\run_gowr_live_inspection.ps1 -Session -DeferObservation `
  -BinaryDirectory $stage -GameProcessId $newGamePid `
  -ObservationRun $newObservationRun -PriorInspectionRun $newContractInspectionRun `
  -ValidationRun (Join-Path $stage 'controls')
```

Only after that deferred attach succeeds and the desired existing scene loads:

```powershell
& (Join-Path $stage 'ffx_session_control.exe') $newGamePid 'C:\DATA\GAME\GODOFWAR\GoWR.exe' (Join-Path $stage 'ffx_capture_session.dll') observe
# After the bounded trace has been obtained:
& (Join-Path $stage 'ffx_session_control.exe') $newGamePid 'C:\DATA\GAME\GODOFWAR\GoWR.exe' (Join-Path $stage 'ffx_capture_session.dll') stop
```

PID/start time and attach paths are run-specific; do not blindly reuse PID 23192
after an exit. The older DLL formerly in PID 15544 did not implement observe/
stop. A new command client does not upgrade an already-loaded session.

Read-only offline state analysis and live entry audit (each output must be new):

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$run='results\20260905_125046_331_gowr_capture_session'
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_prefix_states.py "$run\session.jsonl" --output "$run\prefix_analysis_new.json"
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/audit_ffx_live_entries.py --provenance "$run\provenance.json" --log "$run\session.jsonl" --output "$run\live_entry_audit_new.json"
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest tests/test_ffx_prefix_states.py tests/test_ffx_live_entries.py tests/test_ffx_state_trace.py -q
```

The live audit rejects wrong process start/executable, changed DLL hashes,
missing pinned session, mismatched session-log path or exhausted read budgets.
It only reads bounded code/vtable evidence and never loads a DLL or arms capture.
