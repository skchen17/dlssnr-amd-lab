# NR file → real FSR 3.1.0 engineering probe

This is **synthetic static engineering**, not game quality, temporal validation,
RTX comparison, or a production GPU-resident frame path. The file may contain
NR output, but depth=0.5, motion=(0,0), jitter=(0,0), exposure=1 are explicit
synthetic scene definitions. No unknown game buffers are fabricated.

Supported input: tightly packed little-endian RGBA16F, 1920×1080, 2560×1440 or
3840×2160. Output is 3840×2160 RGBA16F. Frame generation is not used.
The installed provider must enumerate exactly `3.1.0`; the version override is
passed to creation. The wrapper verifies the audited DLL SHA256 and valid AMD
Authenticode signature before loading anything. Use the wrapper, not the raw exe.

The provider context persists for one cold warmup and 1–12 measured repeats.
Every dispatch has reset=true. This prevents claiming history validation; it is
not a benchmark of converged real-game history. Input uploads happen once.
Output poisoning/readback happen every repeat but are outside the D3D12 timestamp
pair, which brackets only `ffxDispatch`'s recorded commands. GPU wait is bounded
at 10 seconds. On timeout the child writes `gpu_fence_timeout_resources_retained`
and keeps the process, device and resources alive for manual disposition; it does
not retry, unwind, destroy resources or resume automatically if the fence later
signals. The wrapper polls in one-second intervals for at most 90 seconds, then
records the retained PID in the adjacent `.supervision` directory and returns an
error **without killing the child**. Stop all further GPU work on timeout; inspect
logs and decide recovery manually. Long initial CPU shader compilation also can
exceed the parent deadline; it is not grounds for killing the process or calling
the run a pass. The GPU fence deadline remains 10 seconds.

Local-process DXGI video-memory usage is sampled before/after context/resources
and each dispatch, and rejects >6,000,000,000 bytes or the OS budget. This is a
fail-stop sampled guard, **not a hard allocator cap or proof that unsampled
transient allocations never crossed the threshold**. The FSR provider controls
its own allocation; no custom allocator limit is installed. Run alone, not beside
the NR GPU benchmark or game. The launcher refuses an active GoWR process.
Adapter selection requires AMD VendorId and `RX 9070 XT` in the device name;
there is no fallback to a different AMD GPU or WARP.

## Build (CPU only)

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\tools\nr_fsr_probe\build.ps1 -OutputDirectory "$PWD\build_nr_fsr_probe_20260906_v2"
```

The build directory must not exist. Existing dispatch helper code is included
with its old entry point renamed; no legacy benchmark or source is modified.
This initial executable was compiled successfully, but compilation alone is
not a GPU pass.

## Run (GPU; only after the previous GPU experiment has fully exited)

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$nrInput = "$PWD\results\20260906_nr_resolution1080_v1\output.rgba16f"
$nrHash = (Get-FileHash -LiteralPath $nrInput -Algorithm SHA256).Hash
.\tools\nr_fsr_probe\run.ps1 `
  -Executable "$PWD\build_nr_fsr_probe_20260906_v2\nr_fsr_probe.exe" `
  -InputFile $nrInput -InputSha256 $nrHash -Width 1920 -Iterations 1 `
  -OutputDirectory "$PWD\results\20260906_nr_fsr1080_one_v1"
```

After a successful one-frame check and no device issues, use a **new output
directory** and `-Iterations 3` or `12`. For 1440p use `-Width 2560` and
`results/20260906_nr_resolution1440_v1/output.rgba16f`; for native-AA engineering
use `-Width 3840` and the 2160p input. 4K NR baseline itself does not include FSR.

`report.json` contains timestamp samples (index 0 is warmup), exact repeated
output status, selected provider ID, finite status, D3D12 error count, sampled
process-local memory and explicit scope. `provenance.json` binds DLL/input/exe
SHA256 and output SHA256. `progress.jsonl` flushes each completed frame. Output
NaN poison is checked across all RGBA half words, so partially unwritten output
fails. A failure is not retried automatically. No new game/hook deployment occurs.

Comparison against native4K NR, if generated from the same synthetic fixture,
can diagnose static pixel changes only. It does not prove NR effect retention,
real detail quality, flicker, exposure handling, or motion stability.
