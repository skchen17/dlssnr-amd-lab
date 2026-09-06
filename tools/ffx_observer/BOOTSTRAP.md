# External entry-point bootstrap (metadata only)

Current continuation: see [LIVE_INSPECTION.md](LIVE_INSPECTION.md). The original
header-only process exited normally; PID 10968 is the decoded replacement run.
The separate live inspector does not change the launcher's metadata-only scope.

## Verified results

- `results/20260905_105834_540_ffx_bootstrap_test/`: separate-process fixture
  passes all five original calls/statuses/last-error values, header-only logs,
  restored entry instruction, paths with spaces and clean exit. Existing results
  are refused without overwrite. A missing-export DLL fails and only the child
  created by that launcher is terminated. Fixture executable hash is unchanged.
- `results/20260905_105936_344_gowr_ffx_observation/`: actual GoWR process PID
  17484 passed bootstrap and reached the main menu (verified using the game
  window). Observer profile is 0, event limit 256, texture capture is disabled.
  GoWR.exe, version.dll, sl.interposer.dll and amd_fidelityfx_dx12.dll retain
  their audited hashes. No DLL was deployed in the game directory.
- That process later exited normally: its 256 records contain 253 dispatches,
  one create and two queries, all status zero. The logger now uses tested
  `_wfsopen(...,_SH_DENYWR)` reader sharing. Concurrent-read bootstrap regression
  passes in `results/20260905_110638_418_ffx_bootstrap_test/`.
- Replacement `results/20260905_110646_754_gowr_ffx_observation/` uses profile 1
  and a 10,000-event budget, including 9,997 decoded dispatches. Snapshot:
  `results/20260905_110900_gowr_metadata_snapshot/`. Budget exhaustion only stops
  logging, not original forwarding. Actual provider version remains unknown.
- Regression after bootstrap work: 265 Python tests pass (64.76 seconds).

## Commands

Run from PowerShell in the repository. Exit the game normally before rebuilding
the observer (the running process holds the DLL) or starting a new observation.

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\scripts\run_ffx_bootstrap_selftest.ps1
.\scripts\run_gowr_ffx_observation.ps1 -GameDirectory 'C:\DATA\GAME\GODOFWAR'
# Verified descriptor decoder (only when GoWR is not already running):
.\scripts\run_gowr_ffx_observation.ps1 -DecodeUpscaleMetadata
```

No existing GoWR process is attached, stopped or replaced. The wrapper refuses
to start when GoWR is already running or any audited executable/library hash
differs. Provider publisher verification is retained. Normal game startup may
write its own cache/configuration; only the four named executable/library hashes
are certified unchanged, not every writable game-generated file or save.

## Mechanism and limits

The launcher uses `DEBUG_ONLY_THIS_PROCESS`, stops the newly created x64 process
at its PE entry point, restores the temporarily changed instruction/RIP, holds
the primary thread explicitly, and detaches the debugger. It resolves the actual
remote owner-module address of LoadLibraryW (including local export forwarders),
loads the observer by absolute path, and calls the pointer-free
`FfxObserver_Bootstrap` export. Bootstrap attaches five main-image FFX imports
with header-only observation, outside DllMain, then the launcher resumes the
primary thread. Remote memory contains arguments only; no custom executable
stub is allocated. No DLL/file replacement, admin elevation, security setting
change, anti-debug/DRM bypass or child-tree process termination is implemented.

The entry-point gate controls the primary thread; it is **not** proof of universal
quiescence in arbitrary programs with background TLS/DllMain workers. Scope is
the explicit fixture and this audited GoWR installation, not a general-purpose
injector. Packed/modified entry points, protected processes, other architectures,
launchers that relaunch a different process and prior foreign hooks are not
supported. Unexpected failure before readiness terminates only the freshly
created child; existing user's processes are never targets. On success the
game stays open for the user and must be exited normally.

The loaded observer remains metadata-only. Queue interception, live resource/state
validation, collector wiring, temporal history and full neural rendering are not
implemented by this launcher. Do not rename it or the observer to a game proxy.
In particular, leave the existing version.dll untouched.

## Next steps (updated)

1. Completed: header parsing, reader-sharing fix/tests, body decoding, approved
   existing scene, bounded real queue correlation through a separate inspector.
2. Remaining: depth/stencil plane policy, current context/default dimensions,
   per-plane state and abandonment/lifetime handling before fenced game capture.
3. Preserve original FFX computation. Never rebuild a DLL still loaded by GoWR.

Windows semantics checked against Microsoft's
[debugging events documentation](https://learn.microsoft.com/en-us/windows/win32/debug/debugging-events),
[debugger detach API](https://learn.microsoft.com/en-us/windows/win32/api/debugapi/nf-debugapi-debugactiveprocessstop)
and [remote thread API](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createremotethread).
