# God of War Ragnarok: selected local game target

User-selected directory: `C:\DATA\GAME\GODOFWAR`.
Executable: `GoWR.exe`. Audit performed without loading game DLLs, starting a
process, changing configuration, installing hooks or touching saves.

## Static evidence (2026-09-05)

Machine-readable hashes, PE versions, imports and exports:
`results/20260905_gowr_target_audit/manifest.json`.

- Executable product version: `GoWR-6349293-Fri Jan 31 16:00:38 2025`.
- The game imports D3D12 creation through `sl.interposer.dll` and imports
  `slSetTag`, `slSetConstants`, `slGetNewFrameToken`, `slEvaluateFeature`.
- It also directly imports `ffxCreateContext`, `ffxDispatch`, `ffxQuery`,
  `ffxConfigure`, `ffxDestroyContext` from `amd_fidelityfx_dx12.dll`.
- Streamline version 2.4.15; DLSS SR 3.7.20; DLSS-G 3.7.10. No NR-named file
  was found under this installation. This does not exclude dynamically loaded
  external components, but there is no evidence of a feature18/NR game path.
- Current settings: RX 9070 XT, fullscreen 3840x2160, FSR3 Quality, dynamic
  scaling off, frame generation off, VSync 1.
- Existing `version.dll` is an installation dependency of unknown provenance
  with many non-version exports. Do not overwrite, delete, rename or repurpose
  it. Nothing in this audit executes or modifies that module.

Imports indicate candidate interfaces, not proof of live invocation or resource
availability. In particular, Streamline tags may not be populated on the FSR path.

## Integration decision

Latest: PID 10968 continues without restart. PID 17484 exited normally and its
log was parsed; reader sharing was fixed/tested before a decoded replacement run.
Live COM inspection correlates 16/16 FFX calls to one direct queue, but correctly
FAILS the old collector contract: actual depth is D32_FLOAT_S8X24_UINT and FFX
declares depth-only usage. Current dimensions also differ from earlier samples.
No game pixels were copied. See `tools/ffx_observer/LIVE_INSPECTION.md` for the
actual evidence, separate passing depth-plane test and remaining state checks.
Four audited image hashes are unchanged and no proxy was deployed. The game
provider version remains unknown. Earlier standalone-only claims below are
historical; the bounded collector itself still has no live-game wiring.

Bounded collection now passes independent real-provider capture-on/off controls
in `results/20260905_104954_587_ffx_dispatch/` (E-145). Each provider captures color,
depth, motion, exposure and output for three synthetic frames without changing
any output byte. The collector owns its staging references and completion fence;
its caller must still identify the actual submission and attest resource states.
This code is not installed in or attached to GoWR. Next game-facing work is the
recoverable bootstrap and queue/list observation layer, then matching actual game
descriptors to the supported capture contract. Do not skip those steps or assume
the standalone caller contract automatically applies to a live game.

2026-09-05 follow-up: `results/20260905_103732_540_ffx_dispatch/` passes real
FFX context/dispatch/readback in an isolated host on the local AMD GPU for the
enumerated `4.1.1 *` and `3.1.0` providers, each explicitly requested by version
override. This establishes a working minimal ABI for this installed library;
it does not establish which provider the game selects, the game's extension
chains, or game command-queue/resource lifetime. User permits FSR4 as an input/
integration route. Retain the game's original upscale operation during initial
capture; do not relabel FSR4 output as DLSS-NR. Commands and test limits are in
`tools/ffx_observer/README.md`.

Update: bounded FFX metadata observation passes the synthetic IAT/D3D12 harness
in `results/20260905_054707_ffx_observer_selftest/`. This is not texture readback
or game deployment. The signed, hash-checked installed AMD provider passes
isolated version/jitter query checks in `results/20260905_054537_ffx_provider_query/`;
available names are `4.1.1 *`, `3.1.0`, `2.3.2`. Actual game-context selection and
full dispatch ABI remain unverified; game policy stays header-only. No FFX
context or dispatch was issued by the query probe. See `tools/ffx_observer/README.md`.

Use this game as a source of representative rendering inputs and ultimately as
the output integration target. Do **not** install the head-only NVAPI backend as
if this were the laboratory feature18 host. The current native graph is incomplete
and the game's current resolution differs from the tested 640x360 output contract.

Game-facing integration unit: connect the bounded collector and metadata observer, starting
with the imported FFX context/dispatch boundary used by the configured FSR path;
observe Streamline tags/constants when present. First validate the matching ABI
against headers and a synthetic host, then collect actual resources, states,
command list/queue ownership, dimensions, jitter and reset/history signals.
Fail closed for unknown descriptor types; do not cast an unverified ABI or assume
the upscaler provides every tensor required by the NR model.

Do not replace FSR computation or alter the rendered output during initial capture.
Retain original module hashes and verify before/after any eventual installation.
Deployment must be recoverable and must not overwrite an existing proxy DLL.

## Data and acceptance boundaries

1. A baseline playable scene must be selected by the user. Capture still camera,
   motion/disocclusion and camera-cut/reset sequences, recording actual frame
   resources rather than screenshots alone. No capture has happened yet.
2. Run the reconstructed model offline on those inputs first. Existing laboratory
   output-head controls and near-black frame references cannot certify game quality.
3. This installation's SR/FSR output is **not an RTX DLSS-NR teacher target**.
   A matching NR reference would require a separately functioning reference
   renderer/host consuming the same captured inputs. Simply running this game's
   ordinary DLSS on an RTX card does not provide that ground truth.
4. Only after full-network quality passes: resident D3D12 scheduling, game-frame
   composition, HUD/color-space checks, temporal validation, then performance.

The game directory selection closes the target-choice question, not O4/O5/O6,
S7/S8, the RTX reference-data requirement, or the full-frame quality failure.

## Repeat the read-only audit

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/audit_game_target.py `
  --game 'C:\DATA\GAME\GODOFWAR' `
  --output results/gowr_target_audit_new_run
```

Use a fresh output directory. The script accepts output only under the lab's
`results` directory, never inside the game installation.
