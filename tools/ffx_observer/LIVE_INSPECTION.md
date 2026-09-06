# GoWR live inspection — 2026-09-05

Follow-up: [DEPTH_CAPTURE.md](DEPTH_CAPTURE.md) implements the explicit collector
depth-plane/default-dimension policy and validates it independently. Game pixels
are still not captured; current context maximum and live states remain unproven.

This milestone inspects an already lab-observed game; it does not render neural
output or copy game pixels. PID 10968 continued without restart. A separate DLL
atomically chains the existing observer's one main-image ffxDispatch import and
samples 16 live COM descriptors. A temporary device-owned direct queue discovers
one runtime vtable; ExecuteCommandLists calls are correlated through IUnknown
list identities. This is not proof of all queue classes or resource states.

## Results and boundaries

- On/off controls: `results/20260905_112352_927_ffx_dispatch/`. For each explicit
  real AMD provider (`3.1.0`, `4.1.1 *`), four outputs are byte-identical with
  inspection enabled, four calls correlate to the real queue, zero debug errors.
  The tested inspector DLL hash is recorded and checked before game attach.
- Game: `results/20260905_112424_948_gowr_live_inspection/`. Sixteen resource
  samples and zero-status original returns correlate to one direct queue, on
  LUID 94973 (same local adapter as the standalone RX 9070 XT tests). Resource
  devices match the list device; color/motion/reactive/output match the pinned
  SDK description. Exposure/transparency resources are null.
- **Strict game contract fails**, retained as `LIVE_INSPECTION_FAIL`: depth
  is DXGI 20, D32_FLOAT_S8X24_UINT. Game FFX format 28 describes its depth view;
  usage is 4 (depth), versus SDK whole-resource usage 36 (depth + stencil).
  Do not globally waive whole-struct mismatches or reinterpret stencil as depth.
- Current inputs: 1552x872; output: 2342x1317. Earlier metadata includes
  2560x1440 and context maximum 3840x2160 with zero/default dispatch upscale size.
  The cause of the dimension change was not observed. Current context/default
  dimensions must be tracked; neither a fixed size nor active provider is proven.
- Five current candidate staging footprints total 48,833,856 bytes, with depth
  plane 0 only. Earlier larger layouts estimate 129 MB. Use queried footprints;
  logical width * bytes per pixel is not necessarily the GPU row pitch.
- `results/20260905_112656_305_ffx_depth_plane/`: an independent AMD process
  copies depth and stencil planes after patterned clears at 321x181, 1552x872
  and 2560x1440. All six artifacts match independently constructed pixel arrays.
  Plane 0 copies as R32_TYPELESS with four-byte depth floats; plane 1 copies as
  R8_TYPELESS with one-byte stencil. Zero debug errors; three expected performance
  warnings from deliberately changing optimized clear values. This is not proof
  of the game's actual plane states, nor an implemented game capture policy.

Four audited executable/library hashes remain unchanged. No game files were
deployed; version.dll was untouched. Normal game-generated caches/configuration
and automatic saves are outside this hash statement. No movement, combat or save
command was issued during this attach. Full-frame NR quality remains failed.

## Reproduction commands

PowerShell, repository root. Inspector and observer DLLs remain loaded until
normal game exit: **do not rebuild them or repeat attach in the current process**.
No hot-unload is implemented because arbitrary game calls may be in flight.
Logs stop at sample/unmatched budgets; forwarding hooks remain until process exit.

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
# Independent validation; only rebuild after the game releases these DLLs:
.\scripts\run_ffx_dispatch_probe.ps1 -LiveInspectionValidation
.\scripts\build_all.ps1 -Only ffx_live_attach,ffx_depth_plane_probe
# Current attach is already completed; DO NOT rerun these commented commands:
# .\scripts\run_gowr_live_inspection.ps1 -GameProcessId 10968 `
#   -ObservationRun results/20260905_110646_754_gowr_ffx_observation `
#   -ValidationRun results/20260905_112352_927_ffx_dispatch

# Current evidence analysis: exit 1 is the expected strict contract failure.
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_live_inspection.py `
  results/20260905_112424_948_gowr_live_inspection

# Independent depth-copy test, always a fresh result directory:
$depthResult=Join-Path (Get-Location) ('results\{0}_ffx_depth_plane' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& .\build\ffx_depth_plane_probe.exe $depthResult
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_ffx_depth_planes.py $depthResult
```

## Remaining work

1. Completed in the follow-up: explicit depth-plane-only policy, exact remaining
   descriptor checks, stencil preservation and provider-on/off controls. FSR4's
   current-size uninstrumented repeatability failure remains separate/open.
2. Track current context create/destroy and default dimensions; snapshot jitter,
   resets, exposure and camera metadata with each collected frame.
3. Establish per-plane barrier states and list reset/abandonment lifetimes, then
   connect one bounded capture and collector-owned queue completion fence.
   Queue association does not prove GPU completion or safe mapping.
4. Validate raw game artifacts, then obtain same-input NR reference outputs and
   evaluate native full-network quality. FSR output is not an NR teacher.
