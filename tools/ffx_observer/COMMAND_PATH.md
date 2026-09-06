# FFX command-path observation — 2026-09-05

This is an observation-only diagnosis of the existing FSR integration, not an
NR implementation or a game texture-capture pass. State gates are not relaxed.

Latest: PID 24336 has exited normally. Resource identity diagnostics in new
PID 30652 now confirm output-targeted UAV barriers in two windows; see
`RESOURCE_IDENTITY.md` for current evidence and commands. The PID/stage/commands
below are historical and must not be reused against the new process.

## Previous playable-scene result

Previous process: GoWR PID 24336, started `2026-09-05T13:20:38.8730962+08:00`.
`results/20260905_132114_421_gowr_capture_session/command_path_analysis.json`
has 64 matched FFX calls. Every call records 28 Dispatch callbacks, 16 legacy
barrier callbacks containing 26 barriers, no ExecuteIndirect or enhanced-barrier
callbacks, and no other-interface calls. All original statuses are zero and all
sampled FFX list tables match the installed table. This establishes command
recording observations, not standalone GPU completion or NR output evidence.

`state_analysis.json`: 1,024 capped state events, 25 complete call windows and
48 complete submission batches, zero incomplete/invalid batches. The trace is
still truncated, not full-frame coverage. No watched input/output transitions
are observed inside those 25 calls; all first post-call output transitions start
at 64. `prefix_analysis.json` associates 24 calls; it remains evidence-only.

The apparent contradiction is now narrower: internal compute/barrier calls ARE
observed on the same list pointer, but no recorded state transitions match the
watched game input/output resource pointers. This revision did not record UAV
barrier resource identity; it cannot claim zero matching barriers of every type.
Next diagnose internal resources /
COM identity and the actual output boundary; do not assume enhanced barriers or
no FSR work, and do not infer that zero watched barriers makes copying safe.

Fresh `provider_metadata_audit.json` again finds the driver callback at
`amdxcffx64.dll + 0x199d40` and private provider metadata `4.1.1 *` /
`17700776142811697153`, matching the independent FSR4 control. This supersedes
the older unknown-backend statement at the level of the audited private fields,
not public-query provenance or session context-lifetime verification.

The user exited the old process normally; the assistant activated/observed the
new game window using computer-use, waited for the existing scene, and issued
no save, movement, combat or settings input. Window 1 is stopped; 7/8 remain.
Zero ready candidates, failure events or game input-copy events; no arm sent.
Game files and loaded stage remain unchanged. Full NR readiness is still false.

## Verified internal forwarding snapshot

In former GoWR PID 23192, the exact loaded game FFX DLL has SHA256
`77809405A0FF464B63654F1264F0EC0FCF8F243DAC7C15B5F5C032615520D143`.
The hash is also in `provider_metadata_audit.json`; the audit
validates the pinned hash and exact live machine-code signatures before reading
any private-layout context pointers. No remote functions are executed.

Evidence: `results/20260905_125046_331_gowr_capture_session/provider_metadata_audit.json`.

- The recorded context points to a provider object in the game FFX DLL. Its
  virtual dispatch slot reaches the DLL's external-provider forwarding thunk,
  which points to `amdxcffx64.dll` in the installed AMD DriverStore directory.
- Private metadata getter fields contain ID `17700776142811697153` and name
  `4.1.1 *`, exactly matching the independent FSR4 provider control. Thus the
  prior menu label AMD FSR 3.1 is not sufficient evidence of the actual backend.
- This is a double-read pointer snapshot from exact-image disassembly, not a
  supported public context query, a lifetime guarantee, or complete GPU call
  trace. `selected_public_provider_id_verified` remains false. It does not
  initialize the capture session's context registry.
- Driver callback image SHA256:
  `D8740074A73FD6A693806DE4B530E66CEB9D702646E00ADA0ED5E3B67DA8F41E`.
  Read-only audit: 21,049 process bytes, no writes, no game file replacement.

## New native observation

Validated bundle: `results/20260905_131819_826_ffx_command_path_stage/`.
Session SHA256: `E6F3861748E19B395865D4D89BE6A841F9885770ACD5BBA949DABDA0D5D2A201`.
After the user exited PID 23192 normally, this bundle was loaded in new PID
24336. Bootstrap: `results/20260905_132038_852_gowr_ffx_observation/`;
session: `results/20260905_132114_421_gowr_capture_session/`.
Do not overwrite, rebuild or unload the loaded staged DLL.

The recorder now counts synchronous calls during the original FFX invocation:
`Dispatch`, `ExecuteIndirect`, legacy `ResourceBarrier`, and optional list-7
`Barrier`. The scope excludes our own Before/After collector calls. It records
other interface-pointer calls separately and never equates these counters with
unique GPU dispatches: wrapper reentry may count a command more than once.
Worker threads, cached method pointers and other unpatched vtables can escape
observation; zero counts are not evidence of no GPU work.

Enhanced barriers invalidate tracked local legacy-state knowledge instead of
inventing a legacy state/layout conversion. Prefix analysis also withholds state
when a prefix list has enhanced-state uncertainty. No new state-gate exemption.

## Independent controls

`controls/session_analysis.json`, `state_trace_analysis.json`,
`window_analysis.json`, `command_path_analysis.json` all pass for both providers.

- FSR3: 56 observed Dispatch callbacks across four FFX calls (14 each), 52
  legacy barrier callbacks; FSR4: 216 (54 each), 61 legacy barrier callbacks.
  These are wrapper-level observations, not a performance or kernel-count claim.
- Both window controls add a valid enhanced global ordering barrier before FFX.
  The callback is observed in both active windows and invalidates local legacy
  state. It is correctly excluded from inside-FFX counters. Idle windows do not
  consume trace budgets. Each provider has 44 trace events in this control.
- Four original outputs remain byte-identical to independent baseline outputs;
  all eight native runs have zero D3D12 errors/warnings. Capture regression still
  saves five textures / 2,764,804 exact raw bytes per provider in isolated hosts.
- Python regression: 390 passed in 46.81 seconds.

These small controlled frames do not waive the existing FSR4 repeatability
failure at actual game dimensions, full-network NR quality failure, temporal
validation or performance requirements. No FSR output is an NR teacher.

## Commands

Run from the repository root. All report destinations must be new; tools refuse
overwriting existing reports. Live audit requires matching PID/start time and
the exact session provenance/log pair.

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$stage='C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\20260905_131819_826_ffx_command_path_stage'
$run='results\20260905_132114_421_gowr_capture_session'
# Current PID only; verify it against provenance before using after any exit.
& "$stage\ffx_session_control.exe" 24336 'C:\DATA\GAME\GODOFWAR\GoWR.exe' "$stage\ffx_capture_session.dll" observe
& "$stage\ffx_session_control.exe" 24336 'C:\DATA\GAME\GODOFWAR\GoWR.exe' "$stage\ffx_capture_session.dll" stop
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_command_path.py "$run\session.jsonl" --output "$run\command_path_new.json"
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/audit_ffx_live_entries.py --provenance "$run\provenance.json" --log "$run\session.jsonl" --provider-route --output "$run\provider_route_new.json"
```

Do not use `arm` to restart observation. The new stage is already loaded;
never repeat attach in this process. For future builds use a fresh staging
directory, run `run_ffx_staged_session.ps1`, and attach only after normal game
exit with new provenance. Command-path validation is required by stage
provenance and checked by `run_gowr_live_inspection.ps1` before attachment.
