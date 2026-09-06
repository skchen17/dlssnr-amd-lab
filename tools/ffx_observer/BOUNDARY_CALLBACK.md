# Post-forward boundary callback and isolated hook copy — 2026-09-05

## Update: playable-scene output transport complete

`results/20260905_144814_826_gowr_capture_session/` is the first usable live
scene artifact. It was armed only after the user confirmed gameplay. The raw
2342x1317 RGBA16F output has SHA256
`79DBAF343981F9695365AD690D16E3B1309E76D6F31211C9EC1F6BD20DBD8603`;
fence, canonical identity, generation, Close and exact-once submission checks
pass. `game_output_analysis.json` marks both game frame and scene content true.
The deterministic preview visibly contains the animal-pen scene.

This closes the output readback/transport boundary. It does not close input and
temporal-resource acquisition, reconstructed NR execution, or safe write-back.
No output replacement was performed and the FFX frame is not an NR teacher.

## Update: real game transport succeeds; captured launch frame is unusable

Fresh PID 31864 confirms the canonical-identity fix. The one-shot copy completed
and wrote `results/20260905_143913_151_gowr_capture_session/output_boundary/`:
2342x1317 RGBA16F, 24,675,312 finite bytes, fenced SHA256
`FC17DAE7604FC5286AD2802381F01CC7461C561CA19CC21E2D4F824A73E9E5CA`.
This is the first real-game pass of the output transport path.

It was armed too early during launch. All sampled channels contain only eight
nearby half values (10.7578125..10.8125), so the revised analyzer marks
`scene_content_verified:false` and `dataset_usable:false`. Do not use this frame
for quality comparison, training or NR evidence. The completed session cannot
be re-armed or stacked. Start a fresh process with the same stage, wait for an
explicit playable-scene confirmation, then issue the one-shot commands.

## Update: live wrapper alias found; corrected stage awaits fresh process

The first real `output-arm` in PID 30212 reached a supported second callback and
recorded the copy, but queue submission exposed a different command-list
interface pointer. The previous pre-identity lookup therefore failed before it
could apply the intended canonical COM check. The session fail-stopped, retained
the uncertain references and wrote no output. Do not poll/retry or stack another
session in that process.

Corrected stage: `results/20260905_143600_233_ffx_canonical_output_stage/`.
It first requires exactly one submitted canonical IUnknown match, then verifies
the original recording entry's generation/reset and at least one closed same-
identity/same-generation alias. `raw_pointer_match` is logged but not required.
Both provider control suites, all six reports and 455 Python tests pass. This
stage is not loaded in PID 30212; a normal exit and fresh bootstrap are required.

## Update: bounded game-output command is validated offline

Stage `results/20260905_142713_314_ffx_live_output_stage/` introduces separate
`output-arm` and `output-poll` commands. `output-arm` observes for at most 60
seconds and accepts one output texture only at the already validated complete
post-forward boundary. It never creates the multi-input collector, never changes
the FFX dispatch descriptor and never replaces a game output. Observation stops
as soon as the matching list is submitted. `output-poll` writes only after the
collector-owned fence completes.

Both FSR3/FSR4 isolated controls now use the same attach-compatible version-2
command path. A blocked queue proves early polling returns incomplete; the saved
1,843,200-byte RGBA16F file equals independent baseline exactly and downstream
bytes are unchanged. Isolated metadata correctly says
`game_frame_capture_verified:false`. Six reports and 455 Python tests pass.

The earlier stage was loaded in fresh GoWR PID 31044 for observation only.
Across two playable-scene windows, seven of eight callbacks pass; the lone
startup rejection lacks a witnessed reset, while all four callbacks in the
second window pass. All eight are single whole-resource 64 -> 192 transitions
with no aliases/unknown barriers and one successful original forward. No copy
was armed. The user reported exit, but PID 31044 is still responsive, so the new
stage has deliberately not been stacked into that process. A fresh process is
required for the first real-game output-only test.

After a fresh deferred session is attached from the new stage, use only:

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$stage=(Resolve-Path '.\results\20260905_142713_314_ffx_live_output_stage').Path
$session=(Resolve-Path '<new gowr_capture_session result>').Path
$pid=<new GoWR PID>
& (Join-Path $stage 'ffx_session_control.exe') $pid 'C:\DATA\GAME\GODOFWAR\GoWR.exe' (Join-Path $stage 'ffx_capture_session.dll') output-arm
& (Join-Path $stage 'ffx_session_control.exe') $pid 'C:\DATA\GAME\GODOFWAR\GoWR.exe' (Join-Path $stage 'ffx_capture_session.dll') output-poll
& 'C:\DATA\Tools\ANACONDA\python.exe' .\scripts\analyze_ffx_game_output.py $session --output (Join-Path $session 'game_output_analysis.json')
```

If `output-poll` reports 2, the fence is not complete; poll again. Result 3 is
a fail-stop and must not be retried automatically. Do not use `arm`; it is the
separate multi-input path. The expected live artifact is
`output_boundary/boundary_output.raw` plus `metadata.json`. Passing it proves
only that one existing game FFX output was safely copied, not DLSS-NR execution,
teacher quality or output replacement.

The actual callback copy path now passes in isolated FSR3/FSR4 hosts. A fresh
validated game-diagnostic stage is ready, but is NOT loaded into current GoWR
PID 30652. The user has been asked to exit normally; no forced exit or hot
replacement is permitted. Game texture copying and NR output remain disabled.

## Validated stage

`results/20260905_140838_246_ffx_boundary_hook_stage/`

- Session SHA256: `5EB4374459655A00198E6380112A47C40BDD56E498BFDAAD6FCE6E1A47FE982F`.
- Probe SHA256: `272439D5AD775BADFEC1B01208A52F7D412EDE87AB0C0832F087A4277095A5D0`.
- Attach-client SHA256: `12A0C5D1EB5297FC19AEE97AC137A9DEC4003089E60C36C9D3966C5DDB40549D`.
- Provider hash remains `77809405A0FF464B63654F1264F0EC0FCF8F243DAC7C15B5F5C032615520D143`.

All 12 native processes pass: baseline, observe, window, legacy capture,
boundary-negative and boundary-hook-copy for each provider. Ten normal runs
have zero D3D12 errors/warnings. Each of the two separate negative runs has
exactly two expected inefficient-barrier warnings: repeated transitions and
begin/end split transitions in one callback. Their exact message categories,
counts and zero errors are checked; additional/unexpected warnings fail.
Both negative cases remain rejected even though their final state is 192.

All six control reports pass, including `boundary_return_analysis.json`.
Python regression: 443 passed in 64.26 seconds (17 new callback-analyzer tests).
The earlier `140216` stage is a failed/superseded control, not an attach target;
`140429` validates diagnostics only and is superseded by the stage above.

## What is now verified

`output_boundary_batch.h` examines the whole original ResourceBarrier array.
It requires exactly one transition of the exact output resource, full flags,
plane 0/all subresources, state 8/64 -> 192, and no aliases/unknown barrier types.
Unrelated UAV barriers do not masquerade as output transitions.

The session records `output_boundary_return` only AFTER forwarding the original
callback. The pending target is consumed before forwarding so nested callbacks
cannot select it twice. Output identity, successful original FFX return, unchanged
output pointer/state/list, observed reset, generation, open-list status and one
FFX call per generation are checked. Reset, close and enhanced barriers clear
pending targets; even unsampled later FFX calls invalidate earlier targets.
This observation alone does not authorize a GPU copy or certify completion.

In isolated configuration version 3 only, the first supported returned boundary
also invokes the real output-copy primitive from INSIDE the callback. Its own
barriers reenter the same patched table without a pending target or held session
lock. The resulting list is matched by COM identity, successful close and exact
generation at queue submission. A collector-owned queue fence gates readback.

For each provider, `controls/fsr*_boundary_hook/boundary_output.raw` contains
1,843,200 bytes, exactly equal to the independent baseline first output. All four
ordinary downstream outputs also remain baseline-identical. A deliberately
blocked queue makes early poll return incomplete; one recorded/submitted/complete
event chain is required before accepting the file. This verifies the actual
hook route in an isolated host, not merely a direct call to the copy helper.

Uncertain jobs are held by a process-lifetime raw owner rather than destructed;
only successfully retired jobs are deleted. This narrow test does NOT yet prove
all live reset/repeated-submission/cross-queue/driver-reentry cases. The callback
log's `capture_authorized:false` is a non-promotion statement: the separate test
mode and fence/artifact checks explicitly describe the isolated copy operation.

## Game limits and next step

Live `FfxSession_Attach` rejects version 3. Version 1/2 game sessions only gain
the new post-forward observation; neither the existing `arm` nor any new live
command enables this isolated output-copy mode. Existing input/context capture
gates are unchanged. No game files were deployed, no live stage overwritten,
and PID 30652 remains on the previous resource-identity revision until normal exit.

After exit, use the existing bootstrap -> live inspector -> contract inspector
chain and attach the NEW stage with `-Session -DeferObservation`. The validation
directory must be this stage's `controls`; provenance now requires the boundary
return report in addition to previous exact-hash/on-off controls. Do not reuse
old PID/start-time provenance or stack sessions.

Then observe the playable scene to determine whether the actual game callback
has one supported transition, no aliasing, and matching lifecycle AFTER original
forwarding. Only after reviewing that evidence and completing remaining live
ownership/adversarial checks should a separately bounded game output-copy mode
be considered. No FSR output is an NR teacher; original-network quality,
actual-size FSR4 repeatability and temporal/performance problems remain open.

## Run commands

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
# Build and validate a NEW directory; does not attach or replace game DLLs.
$stage=Join-Path (Get-Location) ('results\{0}_ffx_boundary_hook_stage' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
.\scripts\build_all.ps1 -Only ffx_capture_session,ffx_dispatch_probe,ffx_live_attach,ffx_session_control -FreshOutputDirectory $stage
.\scripts\run_ffx_staged_session.ps1 -StageDirectory $stage
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q tests/test_ffx_boundary_return.py
# Existing validated report can be reproduced to a NEW filename.
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_boundary_return.py results/20260905_140838_246_ffx_boundary_hook_stage/controls --controls --output results/20260905_140838_246_ffx_boundary_hook_stage/controls/boundary_return_recheck.json
```

Do not pass `session-output-boundary` to the game control client. It is an
isolated probe mode only, not a game command. Game observation still uses
`observe` / `stop` after a correctly proven fresh attach.

## Exact roundtrip revision

The later immutable stage
`results/20260905_150435_007_ffx_output_roundtrip_live_stage/` adds a second
bounded command pair, `roundtrip-arm` / `roundtrip-poll`. It uses the same exact
post-forward boundary and canonical submission proof as output capture, but also
copies a private GPU texture back into the game output before restoring state.
The bytes written back are the bytes just copied from that same output; no neural
result or externally supplied replacement is involved.

Both `session-output-roundtrip` isolated provider controls are required by stage
provenance and pass baseline byte equality, early-poll blocking and fence checks
for FSR3 and FSR4. Python regression is 471 passed. The live command must only be
used once in a fresh, provenance-chained game process at a user-confirmed playable
scene, followed by `scripts/analyze_ffx_game_roundtrip.py`. Never stack it on an
already loaded session.

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$stage=Join-Path (Get-Location) ('results\{0}_ffx_output_roundtrip_live_stage' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
.\scripts\build_all.ps1 -Only ffx_capture_session,ffx_session_control,ffx_live_attach,ffx_dispatch_probe -FreshOutputDirectory $stage
.\scripts\run_ffx_staged_session.ps1 -StageDirectory $stage
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q tests/test_ffx_game_roundtrip.py tests/test_ffx_boundary_return.py
# After a separately verified fresh live attachment and user-confirmed scene:
& "$stage\ffx_session_control.exe" <PID> roundtrip-arm
& "$stage\ffx_session_control.exe" <PID> roundtrip-poll
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_game_roundtrip.py <SESSION_RUN> --output <SESSION_RUN>\game_roundtrip_analysis.json
```
