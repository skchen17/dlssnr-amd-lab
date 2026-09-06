# Original-weight resident game bring-up

This branch prioritizes a **low-FPS debugging loop**, before operator speed or
RTX teacher distillation. It does not claim smooth realtime or DLSS5 parity.

Important display finding: the FFX producer-list-end path below passed texture
writeback but NOT final screen acceptance. Two 12-frame real game runs produced
changing original-weight results; screenshots at completed frames 5 and 10 did
not show their obvious tiled artifacts. A consumer can already run inside that
producer list. `analyze_resident_game.py` therefore reports only
`GAME_TEXTURE_INFERENCE_LOOP_PASS`, with `final_presentation_verified: false`.
The older `GAME_ORIGINAL_WEIGHT_TILED_BRINGUP_PASS` report name is superseded.

The separate `resident_present.dll` development path processes the current
swap-chain buffer immediately before Present. It is DISPLAY-REFERRED, after
tonemapping and HUD, not a validated NR/FSR internal color contract. See
`RESIDENT_PRESENT_DEBUG.md`. Do not combine the two injected runtimes.

## Implemented architecture

`resident_full_graph.py` validates all 156 parameter assets, all 51 PTX assets,
and the original 147,719,680-byte model before device creation. Context, modules,
model, two textures, buffers, and relocated launch arguments are persistent.
Both preblock input and postblock base update per request. The activation arena
is GPU-cleared per frame; no captured activation or RTX state is injected.
Default-stream order replaces per-slot synchronization. A completion event
gates readback. A failed/timed-out executor is poisoned, never reused.

`resident_frame_worker.py` exchanges whole current surfaces over randomly named
LOCAL shared memory/events with one client PID and strictly increasing request
and response IDs. It performs independent 640x360 tiles, edge replication, no
resizing, exact full-surface coverage, and original-alpha preservation. Tiles
do **not** prove equivalence to full-frame attention, correct seams or temporal
semantics. The transport accepts dimensions up to 8192 per axis and at most
3840*2160 pixels, subject to a 20-second surface deadline. Unsupported sizes or
expired frames fall back, not silently resize. Final arbitrary-resolution
network/quality support remains a separate release gate.

The game hook submits the original producer prefix, then a private readback
list. It SIGNALS a fence before any bounded CPU wait (2 seconds). After matched
worker completion (22-second IPC ceiling), it uploads the result on a private
list, verifies that texture's readback against the worker bytes, and forwards
the original downstream suffix. Failure also forwards that suffix. There is
no GPU queue Wait on future work, no GPU spin loop and no session mutex held
while creating/recording GPU objects or running inference. The first two frames
are saved for cross-process byte audits. Stop can allow one in-flight request
to finish. Twelve completed frames or 180 seconds ends the test window.

This deliberately stalls the game's submission CPU thread, so the initial
full-surface debug version is extremely slow. It is not an asynchronous stale
overlay. Native buffers with uncertain GPU completion remain retained until
process exit; a CPU deadline cannot cancel hung GPU work.

## Validation and commands

Run from the repository root in PowerShell. Close GoWR normally before the
independent GPU tests. Do not modify or replace a loaded session DLL.

```powershell
$python='C:\DATA\Tools\ANACONDA\python.exe'
& $python scripts/validate_resident_full_graph.py `
  --plan results/20260905_real_color_probe/plan.json `
  --nvcuda .tools/zluda-v7-preview.3/zluda/nvcuda.dll `
  --input results/20260905_real_color_probe/input/input_rgba16f.raw `
  --reference results/20260905_real_color_probe/zero_initial_arena/run1/final_rgba16f.raw `
  --control-reference results/20260905_real_color_probe/zero_preblock_control/execution/run1/final_rgba16f.raw `
  --output results/resident_validation_NEW
```

Verified evidence (not placeholders):

- `results/20260905_resident_graph_v1/validation.json`: six frames, interleaved
  inputs, reference equality, sync/async equality and immutable model readback.
  Unsynchronized normal frame host time 308–311 ms at 640x360.
- `results/20260905_resident_surface_warp_v2/summary.json`: actual worker output
  through WARP producer/readback/writeback/consumer, all bytes exact.
- `results/20260905_resident_surface_amd_full_v1/summary.json`: actual 2342x1317
  source, 16 complete network tiles, consumer output exact, zero D3D12 errors
  or warnings. Worker surface time 5,343.8 ms.
- `results/20260905_resident_session_{warp,amd}_v2/summary.json`: installed hooks,
  two consecutive frames, same-frame output and next-frame stop restoration.
- `results/20260905_resident_failure_controls_v2/failure_controls.json`: error
  status, wrong request identity, NaN and 22-second no-response timeout; each
  passes untouched original frames downstream and disables the session.
- `results/20260905_resident_session_stage_v2/controls/`: FSR3/FSR4 legacy paths.

The tested game stage is `results/20260905_resident_session_stage_v2`. A worker
must be launched and warmed before the game. Each new run needs a fresh worker,
unique output directory and fresh session; never reuse an expired IPC identity.
Example worker command (runs until 12 requests or a bounded idle timeout):

```powershell
& $python scripts/resident_frame_worker.py `
  --plan results/20260905_real_color_probe/plan.json `
  --nvcuda .tools/zluda-v7-preview.3/zluda/nvcuda.dll `
  --output results/resident_worker_NEW --max-frames 12 --idle-seconds 900
```

The worker stays running in that console. Use another PowerShell console for
the game/session control. `prepare_resident_game_stage.ps1` binds a fresh worker
and existing accepted controls to an immutable stage gate, before game launch.
It currently references the explicit accepted evidence above, not arbitrary
unvalidated builds. An existing gate is not overwritten.

Launch via `start_gowr_resident_session.ps1 -StageDirectory <fresh-stage>` after
`prepare_resident_game_stage.ps1 -StageDirectory <fresh-stage> -WorkerRun
<worker> -ControlVersion v2`. This starts the bootstrap and the two required
inspection stages before attaching the dormant session module.
Recreate the FSR context through the game's settings if it predates attachment,
then restore AMD FSR 3.1 Quality, frame generation off. Load an existing scene.

```powershell
./scripts/run_gowr_resident_network.ps1 -StageDirectory <stage> -SessionRun <session> -Action start
./scripts/run_gowr_resident_network.ps1 -StageDirectory <stage> -SessionRun <session> -Action status
./scripts/run_gowr_resident_network.ps1 -StageDirectory <stage> -SessionRun <session> -Action stop
& $python scripts/analyze_resident_game.py --session <session> --worker <worker> --output <NEW-analysis>
./scripts/audit_gowr_resident_network.ps1 -SessionRun <session> -Phase after_run
```

The placeholders identify paths returned for that specific fresh run, not paths
to paste literally. Raw outputs and the shared display-transform PNGs are
diagnostic evidence, not same-input RTX quality targets. Keep full native graph
optimization, real full-frame/temporal semantics and teacher-based tuning after
this bring-up milestone; do not resume per-instruction equality as a prerequisite.
