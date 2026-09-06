# Original-weight same-buffer presentation debug path

## Live result (E-174, 2026-09-05)

`results/20260905_192944_171_gowr_resident_present` (PID 4592) completed 12 distinct
2342x1382 SDR game buffers with 16 full graph tiles each. Every Present returned
S_OK. `results/20260905_present_game_analysis_v3/summary.json` independently
verifies worker identity, 156 slots per tile and exact packed output conversion.
The first two frames changed 3,236,010 and 3,235,959 packed pixels. Whole-surface
network time was 5,081.9–5,201.2 ms, not playable realtime.

`screen_verification.json` binds actual running screenshots at completed frames
3 and 11: the network's brightness seams are visible, then disappear after
automatic stop (`resident_stopped.png`). This is visual verification, not an
assertion that screenshots are byte-identical to GPU buffers. After-run and
after-exit audits show normal response/exit, unchanged game images, no newly
recorded display-reset/unexpected-restart event. `hdr_restoration.json` and
`hdr_restored.png` record restoration to HDR ON / 80 / UI 50 in a separate
non-network game process, followed by normal exit. Commands below reproduce this
bounded debug experiment, not a finished or HDR-capable game mod.

This is a separate, deliberately low-FPS bring-up path, not a DLSS5-quality
release. The fixed 640x360 recovered full graph runs with original weights in
the persistent AMD worker. A whole current display buffer is processed as
independent tiles, without shrinking the frame. This introduces seams, changes
the model context, and does not implement true arbitrary-resolution attention.

Unlike the FFX end-of-producer-list experiment, this path reads the current
swap-chain buffer immediately before Present and writes that same buffer back.
The queue comes from the actual DXGI factory creation call, not a device-only
guess. There is no stale-frame overlay. Private read/write lists restore
PRESENT state and wait only on fences already signaled by this queue. The
observer mutex is released before calling the application's original Present.

The input is SDR display-referred normalized RGB after tonemapping and HUD.
No undocumented inverse color transform is applied. This is NOT the recovered
NR pre-tonemap exposure/temporal contract. HUD may be altered. HDR, partial
Present1 updates, nonblocking/restart flags and unverified queues are rejected.
R8G8B8A8, B8G8R8A8 and SDR R10G10B10A2 conversion preserve original packed alpha.
Changing dimensions stops the test; a fresh process is required to restart it.
ResizeBuffers1 invalidates queue identity and disables this run.

Limits: 12 frames or 180 seconds; 2-second already-signaled GPU fence waits,
22-second IPC timeout; per-worker full-surface budget 20 seconds. CPU timeouts
cannot cancel a hung GPU. Uncertain native resources are retained until process
exit. IPC failure before writeback presents the original buffer; GPU failures
are not claimed to be safely reversible. This is not an unattended burn-in.

## Validated artifacts

- `results/20260905_present_warp_v3/summary.json`
- `results/20260905_present_amd_v3/summary.json`
- `results/20260905_present_failure_v3/failure_controls.json`
- `results/20260905_resident_present_stage_v3/present_validation.json`

The first two use actual original-weight inference, Present and Present1,
current buffer readback, stop-to-baseline, a non-inference TEST present, and
resize after resource release. Zero D3D12 errors and warnings were recorded.
The four failure cases cover error status, wrong request ID, nonfinite output
and no-response timeout. Each preserves the baseline and latches disabled.

API contract references: [Microsoft D3D12 swap chains](https://learn.microsoft.com/en-us/windows/win32/direct3d12/swap-chains)
requires a direct queue at creation and PRESENT state at presentation;
[DXGI color-space definitions](https://learn.microsoft.com/en-us/windows/win32/api/dxgicommon/ne-dxgicommon-dxgi_color_space_type)
identify 0 as SDR sRGB/BT.709 and 12 as ST.2084/BT.2020. These API definitions
do not establish the neural model's input color contract.

## Running a bounded game experiment

From the repository root, first close GoWR normally. Never replace a loaded DLL
or edit a running stage's worker identity. No game-directory DLL is installed.
Start a fresh worker in one PowerShell console:

```powershell
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/resident_frame_worker.py `
  --plan results/20260905_real_color_probe/plan.json `
  --nvcuda .tools/zluda-v7-preview.3/zluda/nvcuda.dll `
  --output results/present_worker_NEW --max-frames 12 --idle-seconds 900
```

After `worker.json` says READY, run in another console:

```powershell
./scripts/start_gowr_resident_present.ps1 `
  -StageDirectory results/20260905_resident_present_stage_v3 `
  -WorkerRun results/present_worker_NEW
```

The script prints the fresh session directory. This game was originally HDR ON
(HDR brightness 80, UI brightness 50), which correctly failed the first gate.
For this SDR-only experiment, use the game's Display/View -> Screen Calibration
to temporarily switch HDR OFF; verify a `color_space_set: 0` event. Do not change
Windows HDR settings. Restore the original game HDR values after the run.
Load an existing game scene.
Substitute that actual session directory for `SESSION` below:

```powershell
./scripts/control_gowr_resident_present.ps1 -SessionRun SESSION -Action start
./scripts/control_gowr_resident_present.ps1 -SessionRun SESSION -Action status
./scripts/control_gowr_resident_present.ps1 -SessionRun SESSION -Action stop
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_resident_present.py `
  --session SESSION --output results/present_analysis_NEW
./scripts/audit_gowr_resident_network.ps1 -SessionRun SESSION -Phase after_run
# After a normal game exit:
./scripts/audit_gowr_resident_network.ps1 -SessionRun SESSION -Phase after_exit
```

The analyzer checks same-buffer identity, accepted Present, worker provenance,
all 156 slots per tile, exact independent packed-pixel conversion, original
alpha, finite changed outputs and distinct real inputs. It does not derive
physical screen visibility from texture logs; save and inspect running/stopped
screen captures separately. It never equates visible changes with better image
quality or matched RTX output.
