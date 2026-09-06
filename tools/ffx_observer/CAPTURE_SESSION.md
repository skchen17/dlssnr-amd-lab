# Bounded capture session and state diagnosis — 2026-09-05

Latest: resource identity observation in PID 30652 completed two windows and
stopped. Sampled output UAV barriers match raw and COM identity; context/state
proof still blocks collection. See `RESOURCE_IDENTITY.md` for current commands.
No game pixel capture or NR quality pass. PID 24336 below is historical.

Previous: command-path observation was loaded in PID 24336 and completed
one playable-scene window. Compute and legacy-barrier callbacks are present;
watched texture states and context lifetime still block collection. The private
backend metadata matches FSR4 4.1.1 *. See `COMMAND_PATH.md` for current evidence
and commands. No game pixel capture or NR quality pass.

Previous live revision: explicit idle/observe/stop windows and submission batch
IDs are loaded in new PID 23192 after normal exit of PID 15544. Two playable-
scene windows are complete and stopped; no arm/copies occurred. Same-batch
prefix analysis and a bounded read-only entry audit narrow, but do not resolve,
the state discrepancy. See `OBSERVATION_WINDOWS.md` for current evidence,
the exact loaded bundle, operating limits and run commands. Everything below
describes earlier processes; do not reuse their context epochs or loaded paths.

## Latest restart and live trace (12:30)

The user exited the old game normally. Its previous session DLL is preserved at
`results/20260905_121239_377_gowr_capture_session/ffx_capture_session.loaded.dll`.
The rebuilt replacement passes exact-binary FSR3/FSR4 controls in
`results/20260905_122935_190_ffx_dispatch/` before attach. The new game is PID
15544, started at 12:29:51 via
`results/20260905_122951_145_gowr_ffx_observation/`. All three follow-up attach
steps succeeded; audited game files are unchanged and no game DLL was deployed.

**The new state recorder IS now loaded**, SHA256
`AFFB26633B84FE89E1742EE63CC734DD8DB1F90ACFBC19DECD32B4AACE87ECF4`.
The earlier staged/not-loaded statements below describe the previous run.
Do not rebuild or overwrite it while PID 15544 is running.

Evidence: `results/20260905_123013_029_gowr_capture_session/boundary_analysis.json`.
The bounded log has 1,024 trace events, 64 candidates, and 34 complete FFX call
windows. The limit marker is present; this is a partial trace, not a complete
frame or all-queue recording. These samples occurred in the main menu BEFORE
Continue loaded the existing scene. The scene subsequently loaded successfully;
no manual/new-save, combat or movement input was issued by the assistant.

- All 34 windows show depth transitions on other command lists. Motion
  transitions also occur on other lists. Thus absence on the FFX list does not
  mean these resources have no transitions.
- All 34 windows have no observed legacy transition on the watched resources
  inside the FFX call; all have an output transition afterward whose StateBefore
  is 64, not the API-declared UAV value 8. This is NOT proof that FFX does no GPU
  work. Cached calls, other interfaces/resource identities or unobserved barrier
  paths have not been excluded.
- Example sample 5, list generation 2: output 8→64 at sequence 117;
  FFX begin/end at 120/121; output 64→192 at 122; close 136; submission 141
  (index 33 of a batch of 36). These are CPU recording facts, not a complete
  GPU state proof. Do not reinterpret metadata or waive the capture gate.
- This new context predates the session attach and is **unknown**; the prior
  PID's verified epoch must not be reused. Attaching during startup did not
  guarantee interception before context creation.
- No capture arm was sent; zero input-copy events and no game pixel captures.

Boundary-correlation tests cover incomplete calls, wrong list generations and
external transitions that must not be inferred as local state. Full regression:
**332 passed in 64.85 seconds**.

Next implementation needs an observation-only retrigger independent of capture
arming, a scene-selectable bounded trace window, and actual command-interface /
cross-list ordering coverage. Resolve the output boundary before inserting any
copies. Avoid another blind startup trace whose budget is consumed by menus.

Analysis command (use a fresh output filename):

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$boundaryReport=Join-Path (Get-Location) ('results\{0}_boundary_analysis.json' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_state_trace.py results/20260905_123013_029_gowr_capture_session/session.jsonl --output $boundaryReport
```

Current direction: use the game's FSR 3 integration boundary for color, depth,
motion and reactive inputs. FSR output is not an NR teacher or proof of DLSS5.
The menu was observed at **AMD FSR 3.1 / Quality**, frame generation off, on
RX 9070 XT. The menu label does not verify the selected runtime provider ID.

## What has passed

- `results/20260905_121214_603_ffx_dispatch/session_analysis.json`: both exact
  FSR3/FSR4 providers pass automatic command-list submission association,
  owned-fence completion and one-frame capture. Each captures five textures /
  2,764,804 exact raw bytes, preserving all four independent original outputs.
- `results/20260905_122303_633_ffx_state_trace_stage/controls/`: NEW staged
  version passes baseline / observe-only / capture controls for both providers.
  Observe-only saves 68 state events per provider and no capture artifacts;
  all four outputs remain byte-identical. Capture regression also passes.
- Full regression after these changes: **326 passed in 61.23 seconds**.
  Observe-only native controls have zero D3D12 errors and warnings.

These are synthetic independent hosts. The current-game-size FSR4 baseline
failure documented in `DEPTH_CAPTURE.md` remains unresolved and is not waived.

## Live-game result: observation only, do not arm

`results/20260905_121239_377_gowr_capture_session/state_observation_analysis.json`
summarizes 264 candidate samples, zero ready candidates, zero input-copy events.
Same game PID 10968. Four newly observed context creations now supply a verified
epoch; the latest is epoch 4, flags 41, max render 1552x872 and max upscale
2342x1317. Requested provider ID is 0 (no explicit override), therefore selected
FSR backend remains unverified. This fixes the earlier missing-context evidence,
not the resource-state proof.

| Resource | Last state recorded on FFX list | Declared D3D12 state |
|---|---:|---:|
| Color / reactive | 64, non-pixel shader read | 192, pixel + non-pixel read |
| Depth / motion | Unknown | 192 |
| Output | 64 | 8, unordered access |

Unknown does not mean corrupt; transitions may be on other lists or outside the
observed window. A last local state is not a queue-ordered global-state proof.
Do not substitute declared states, merge read states blindly, or enable copies
to test whether the driver happens to tolerate them. No game pixels were copied.

## Staged diagnostic implementation

`capture_session.cpp` now records up to 1,024 diagnostic events: watched resource
identities, exact transition before/after states and subresources/flags,
command-list generations, reset/close, FFX begin/end, and queue submission return.
Candidates include resource identities to correlate roles with transitions.
Exhaustion emits `state_trace_limit`; a truncated trace is not complete evidence.
CPU callback/log order does not prove GPU execution order. Queue waits/signals,
enhanced barriers and cross-queue synchronization are not fully tracked.

Capture gates are unchanged. Context lifetimes, exact local states, reset/close,
resource shape/device identity, 192 MiB / one-frame cap and fence completion
remain mandatory. Ambiguous in-flight failures retain references until process
exit. No hot-unload or forced process termination is supported.

**The new diagnostic binary is NOT loaded in the current game.** The currently
loaded `build/ffx_capture_session.dll` is still the validated earlier version,
SHA256 `2E26E1401900FDEFACD287638424FA5CE62E5114E3A33383AB2E21E004E2F8AE`.
Do not rebuild/overwrite loaded lab DLLs. Restart normally before changing the
loaded version, then validate the exact replacement binaries before attaching.
The old observation's ten-minute tracking window expires; do not use `arm` just
to restart logging, because it also authorizes capture.

## Reproduction commands

These isolated controls can run while the game remains open. They create a
fresh directory and never rebuild the game-loaded copies in `build`.

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$traceStage=Join-Path (Get-Location) ('results\{0}_ffx_state_trace_stage' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
.\scripts\build_all.ps1 -Only ffx_capture_session,ffx_dispatch_probe,ffx_session_control,ffx_live_attach -FreshOutputDirectory $traceStage
.\scripts\run_ffx_staged_session.ps1 -StageDirectory $traceStage
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q
```

Read-only-in-game observation analysis (output report must not already exist):

```powershell
$report=Join-Path (Get-Location) ('results\{0}_state_observation.json' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_state_trace.py results/20260905_121239_377_gowr_capture_session/session.jsonl --output $report
```

Next: normal game exit, load a freshly validated diagnostic session, capture a
bounded barrier/list/queue trace with pixels still disabled, reconcile missing
and conflicting states, then attempt one safely fenced real-game frame.
Native NR full-graph quality, temporal generalization and performance remain
separate failed/unverified gates; `game_runtime_ready` is still false.
