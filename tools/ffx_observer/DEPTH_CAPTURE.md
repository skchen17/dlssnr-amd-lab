# Explicit depth-plane collection — 2026-09-05

The collector now supports D32_FLOAT_S8X24_UINT under an explicit
`DepthPlanePolicy::D32S8DepthOnly` option. Default policy still rejects it.
Only the depth role and shader-read FFX states are accepted. The one permitted
descriptor normalization is whole-resource depth+stencil usage 36 to the game's
depth-only view usage 4; every other descriptor field is checked exactly.
Both copy and transition target subresource 0. Stencil plane 1 is not touched.
Manifests distinguish source DXGI format 20 from plane-copy R32_TYPELESS 39.

Zero/default upscale dimensions are resolved from the caller's verified context
for validation, while manifests preserve both raw and effective dimensions.
Mixed-zero sizes, exceeded/stale context bounds and incompatible texture extents
are rejected before any GPU copy is recorded. This does NOT supply the missing
live-game context registry or attest real resource states.

## Evidence

- `results/20260905_113938_561_ffx_dispatch/depth_capture_analysis.json`:
  **FSR3 passes, aggregate FAIL retained because FSR4 baseline fails**. FSR3 at
  1552x872 -> 2342x1317 captures three synthetic frames / 15 textures /
  143,046,480 raw bytes. Every captured pixel agrees with its known producer or
  independently read output. All four output buffers are unchanged by capture;
  stencil stays 0xA7. Zero D3D12 errors/warnings. Flags include HDR, inverted depth
  and auto exposure; exposure is null and reactive input is present.
- Fifteen invalid/duplicate dispatch cases are rejected, including seven new
  bad depth/dimension cases. Two further collectors reject missing depth policy
  and stale context defaults. Byte-budget rejection writes nothing; three-frame
  cap and gated-fence early-read tests pass. Staging budget is explicit 192 MiB
  for these large synthetic controls, not a global increase to the 64 MiB default.
- FSR4 uninstrumented baseline at those dimensions has a repeatability failure,
  despite finite output, zero debug errors/warnings and unchanged stencil.
  `results/20260905_114148_633_ffx_depth_fresh_context/` still fails with fresh
  context per frame. Frame 0 vs 1 differences cover 1,336 pixels near the bottom
  right (x2292..2341, y1196..1316), max channel delta 0.04833984375. This is not
  a collector-on regression, not an AMD-vs-RTX comparison, and not yet an
  identified implementation root cause. It must not be silently waived.
- `results/20260905_114847_910_ffx_dispatch/depth_aligned_capture_analysis.json`:
  FSR3 AND FSR4 pass at the separate 1536x864 -> 2304x1296 control layout. Each
  captures 139,345,920 raw bytes, preserves all four outputs and stencil, has
  zero D3D12 errors/warnings and retires all three jobs. Smaller legacy capture
  controls also pass in that run. This supports a size-dependent condition in
  the tested FSR4 path; it does NOT certify FSR4 at the current game dimensions.
- `results/20260905_115055_125_ffx_capture_plane_state/` deliberately leaves
  stencil in DEPTH_WRITE (16) while depth is shader-read (192). Collector depth
  and output bytes match, stencil remains intact, debug errors are zero. No
  provider is invoked in this isolated state-contract test; provider ID 1 is a
  fixture metadata sentinel, not a detected game/FSR version.
- Current game contract resample:
  `results/20260905_114603_696_gowr_live_inspection/`, same PID 10968. Sixteen
  calls correlate to the same direct queue. Actual renderSize is 1552x872,
  upscaleSize remains [0,0], motion scale [-776,436], cameraNear 4000 and
  cameraFar 0.1. Thus the observed render dimensions are not merely an allocation
  estimate. Current context maximum/provider remain unknown. The strict old
  whole-resource inspector gate intentionally still fails depth usage.

The contract resampler is a distinct `ffx_contract_inspector.dll`, chaining the
already loaded lab inspector without rebuilding it. Independent on/off tests
passed in `results/20260905_114528_108_ffx_dispatch/` before attach. No GPU copy
was performed in GoWR; game files and resolution were not changed. All three
diagnostic DLLs remain pinned until normal game exit; do not rebuild them live.

## Commands

From PowerShell in the repository. These commands build/run isolated EXEs only
and do not restart the currently open game or rebuild the loaded observers.

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
# Passing separate layout control plus legacy capture regressions:
.\scripts\run_ffx_dispatch_probe.ps1 -CaptureValidation -DepthPlaneValidation -AlignedDepthControl
# Observed game dimensions: currently EXPECTED TO FAIL at the FSR4 baseline.
.\scripts\run_ffx_dispatch_probe.ps1 -DepthPlaneValidation

.\scripts\build_all.ps1 -Only ffx_capture_plane_selftest
$planeStateResult=Join-Path (Get-Location) ('results\{0}_ffx_capture_plane_state' -f (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& .\build\ffx_capture_plane_selftest.exe $planeStateResult

& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q
```

Do not run `-ContractValidation` or `-LiveInspectionValidation` while the
corresponding DLL is loaded in GoWR. Their sources now include later metadata
fields; rebuilding a held DLL is neither required nor supported.

## Next integration boundary

The remaining capture work is current-context lifecycle/default-size tracking,
per-plane live barrier state and list reset/abandonment handling, then a single
bounded game capture with completion-fence evidence. Do not continue collecting
numeric micro-traces or mistake FSR4's separate baseline edge issue for native
NR network correctness. FSR is an input integration path; its output is not an
NR teacher. Full native neural reconstruction, representative RTX references,
temporal quality and in-game performance remain separate unpassed release gates.
