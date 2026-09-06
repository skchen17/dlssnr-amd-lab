# Bounded FFX metadata observer (harness stage)

Depth-plane collector follow-up: [DEPTH_CAPTURE.md](DEPTH_CAPTURE.md), including
passing on/off controls, the separate FSR4 size-dependent baseline failure and
commands. D32S8 requires an explicit policy; original single-plane behavior is
retained by default. Live context/state/lifetime wiring is still outstanding.

Latest game continuation: [LIVE_INSPECTION.md](LIVE_INSPECTION.md). Actual
queue correlation is established, but the depth/stencil texture contract fails
the old single-plane collector gate. No game pixels or NR output are captured.
Earlier harness-only statements below refer to those isolated milestones.

External metadata-only bootstrap now passes a separate-process fixture and
starts the audited GoWR installation to its main menu. See [BOOTSTRAP.md](BOOTSTRAP.md)
for commands, live-log sharing issue, actual game evidence and limitations.
The game-side event log still awaits normal game exit and parsing; no actual
game texture capture or neural rendering success is claimed.

The observer wraps all five normal named FFX imports in one selected module. It
forwards the original pointers exactly once, preserving return codes and Windows
last-error state. It issues no GPU commands, waits, neural computation or texture
readback. Resource pointers/states are **metadata**, not captured pixels.

Profile 0 (real-game default policy) reads only generic descriptor headers and
bounded extension chains. Profile 1 requires the caller to attest the SDK 1.1.3
body ABI; it records seven resources, dimensions, command list, jitter, motion
scale, exposure, camera parameters, flags and reset. Unknown types are not cast;
unreadable/short buffers and extension cycles are handled without changing the
original call. Extensions stop after eight entries. Nonfinite floats become JSON
null. Event limits are 1-10,000; exhausted budgets stop decoding, not rendering.

Headers: AMD's MIT-licensed
[SDK v1.1.3, commit 54fbaaf](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/54fbaafdc34716811751bea5032700e78f5a0f33/ffx-api/include/ffx_api).
Compile-time checks verify the x64 16-byte header, 48-byte resource, 432-byte
upscale descriptor and selected offsets. This verifies our compiler's layout,
**not** the actual game's complete dispatch ABI. Copyright notices are retained;
see `third_party/PROVENANCE.md` for the one export-macro adaptation.

## Run commands

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\scripts\run_ffx_observer_selftest.ps1
.\scripts\run_ffx_provider_probe.ps1 -GameDirectory 'C:\DATA\GAME\GODOFWAR'
```

The selftest builds a synthetic FFX provider and patches its host's actual import
table. Its original dispatch records a real AMD D3D12 texture copy through the
unchanged descriptor pointer; the observer does not record that copy. All
1,843,200 bytes match independently reconstructed expected data, with zero debug
errors. Context lifecycle, query output mutation, nonzero statuses, incoming/
outgoing last-error, invalid descriptors, bounded logging and restored imports
are tested. Evidence: `results/20260905_054707_ffx_observer_selftest/`.

The provider probe verifies the audited DLL hash and valid AMD publisher signature
before loading only that graphics library in an isolated process. It creates a
D3D12 device and queries versions/jitter, but no FFX context, dispatch or game.
Evidence: `results/20260905_054537_ffx_provider_query/` reports QUERY_ABI_PASS;
available names are `4.1.1 *`, `3.1.0`, `2.3.2`. The 2560-to-3840 jitter phase is
18 and first offset `(0,-0.166667)`. Enumeration does not establish the active
game provider; the asterisk is retained without interpreting its meaning.

## Deployment limits

### Real-provider dispatch/readback probe

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\scripts\run_ffx_dispatch_probe.ps1 -GameDirectory 'C:\DATA\GAME\GODOFWAR'
```

This separate probe verifies the audited hash and AMD signature, then starts one
process per exact provider name (`3.1.0`, `4.1.1 *`). It queries the IDs at runtime
and requests each via the official version-override descriptor; no silent fallback
or game launch occurs. It uses the pinned SDK DX12 backend/resource definitions.
Successful override creation is recorded, not an independent inspection of
provider internals or proof of the provider selected inside the game.

Four A/A/B/A reset frames use 320x180 RGBA16F color, R32F depth, RG16F zero motion,
1x1 R32F exposure and 640x360 RGBA16F UAV output. Output is NaN-poisoned before each
dispatch to detect unwritten channels. All upload, default and readback resources
remain alive until the actual queue fence completes. Copy footprints are obtained
from the device; padded rows are packed/unpacked explicitly. Uncertain GPU
completion is fail-stop at process scope, never a resource-recycling path.

`results/20260905_103732_540_ffx_dispatch/`: both providers pass create, four dispatches,
four fences and destroy; debug errors/warnings are zero. Independent NumPy checks
confirm finite/nonblank output, exact A repeats and B input sensitivity (RGB MAE
0.25511 for 3.1.0, 0.26327 for 4.1.1). Provenance verifies the library is unchanged.
The earlier `103605_505` run lacked output poisoning; the `103732_540` run supersedes
it. Neither run uses game frames, jittered temporal history or the NR model.

The readback implementation here owns the entire test queue; it is not yet an
in-game collector. Live capture still needs resource/state validation, bounded
staging, actual submission/fence ownership, and a recoverable bootstrap. All
frames in this probe reset history, so it makes no temporal-quality claim.

### Remaining game deployment gates

The caller-coordinated texture collector is now validated in the independent
real-provider host (see below), but is deliberately not wired into this DLL's
metadata hooks. A live-game submission/bootstrap contract is still required.

- A scoped external entry-point launcher is available (BOOTSTRAP.md); there is
  no automatic DllMain hook or proxy-DLL game installation.
- Attach/detach require a **caller-established quiescent boundary**. The active
  call counter alone does not make concurrent installation/removal safe.
- One module's ordinary imports only; delay imports, other importers and
  GetProcAddress callers are not wrapped. All five direct exports must resolve
  to one retained backend. Pre-existing hooks/incomplete sets fail attachment.
- The observer is pinned for process lifetime. Never rename it to or overwrite
  an existing `version.dll`, `dxgi.dll` or other installation module.
- The log must be an absolute, new path. Failed/partial installation or foreign
  hook interference is not a supported live-game lifecycle; restart the test
  process before retrying. No game deployment readiness is claimed.

Next gates: actual context/dispatch ABI and selected provider validation;
recoverable external bootstrap; queue/fence-owned raw texture readback with
verified resource states, lifetime, footprints/formats and bounded storage;
then a user-selected playable scene. No real game frame has been collected.
Ordinary FSR output is not an NR teacher target, and the native-network image
gate remains failed/incomplete independently of this observer milestone.

## Bounded texture collector validation

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\scripts\run_ffx_dispatch_probe.ps1 -CaptureValidation
```

The wrapper runs each real provider with capture off and on in separate processes,
verifies the installed library hash/signature, then compares every output byte.
It writes only new result directories in this repository. No game is launched.
Evidence: `results/20260905_104954_587_ffx_dispatch/capture_analysis.json`;
both providers pass, each with 15 textures / 8,294,412 captured raw bytes.

`texture_capture.h` exposes a caller-coordinated `Collector`, not a hook:

1. `Before` validates the typed FFX descriptor/live COM objects, allocates the
   complete capture job before recording, and copies input textures on the same
   direct command list while restoring their declared legacy resource states.
2. The caller invokes the **original** FFX dispatch exactly once.
3. `After` records output readback and restores the original output state.
4. The caller submits that list exactly once, then calls `Submitted` on that same
   direct queue before list reset/reuse. The collector appends its own fence signal.
5. `Poll`, called outside rendering callbacks, writes only fence-complete jobs.

The host validates three A/A/B captures plus a fourth uncaptured A frame per
provider. Color/depth/motion/exposure/output bytes agree with the independently
known producer data and normal output readback. The 1x1 exposure texture exercises
256-byte GPU row alignment versus four-byte raw storage. Collector-on output is
byte-identical to collector-off output for both FSR3 and FSR4.

Tests also cover eight invalid/duplicate descriptors, polling before submission,
a deliberately gated GPU queue (the collector must not map early), a total-byte
budget smaller than one job, and exhaustion of the three-frame budget while the
fourth original dispatch still executes. All jobs retire; D3D12 errors/warnings
are zero. Frame metadata retains context flags/provider ID, dimensions, jitter,
motion-vector scale, exposure, camera/reset/sharpening flags and copy layouts.

Scope is intentionally narrow: single-threaded caller coordination, one direct
queue, at most one pending capture per command list, new frame IDs, single-mip
non-MSAA non-array typed 2D textures. Supported resource states are compute-read,
pixel/compute-read and UAV; COMMON/unknown states and descriptor extension chains
are rejected. Color formats supported by the copier are RGBA16F, R32F, RG16F,
RG32F and R8 UNORM; this provider test exercises the first three only. Optional
reactive/transparency resources may be absent. Defaults cap the whole run at
three frames and 64 MiB of staging allocations (not an unbounded ring).

Live pointers/states, quiescence and the real submission must be attested by the
caller: this component cannot establish them from untrusted metadata alone.
No worker thread, multi-queue ownership, abandoned-list recovery or game bootstrap
is supplied. Destroying an undrained collector deliberately retains its COM
references until process exit to avoid GPU use-after-free; this is a fail-stop
diagnostic policy, not a supported in-game cleanup mechanism. IO failures leave
incomplete evidence and must not be reported as a capture pass.

These are controlled synthetic frames through the **real** installed AMD provider,
not screenshots, not game frames, and not DLSS-NR reference outputs. The existing
metadata-only observer remains unchanged and does not itself capture textures.
