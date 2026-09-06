# D3D12 output-head recorder

`output_head_d3d12_runtime.h` provides `output_head_dx12::Executor` for the
captured 640x360 output-head contract. It accepts caller-owned default-buffer
slices for main/skip/model, RGBA16F base/output textures, and a direct or compute
command list. Eleven DXIL dispatches perform the reconstructed head; texture
copies stage the base and output. Main and skip may share an allocation.

## Run the recording test

From the repository root in PowerShell:

```powershell
.\scripts\run_output_head_recording_selftest.ps1
```

The script builds the shaders/executable and prints its timestamped result
directory. Defaults use the local captured activation/model and the accepted
standalone DXIL output. Override `-ActivationArena`, `-Model`, `-Reference` and
`-OutputDirectory` for other local paths. `Reference` must be the full-head DXIL
zero-base output for the same activation/model, not an arbitrary RTX image.
To regenerate that reference first:

```powershell
.\scripts\run_output_head_full_d3d12.ps1 -OutputDirectory results\head_baseline
.\scripts\run_output_head_recording_selftest.ps1 `
  -Reference results\head_baseline\output_rgba16f.raw
```

The test records A/B/C/A on one list: captured main with zero base; captured main
with a patterned nonzero base; zeroed main with zero base; original inputs again.
It submits that list twice with a completion fence between submissions. It checks
standalone byte parity, changed-input sensitivity, restored-input repeatability,
CPU reconstruction of the surface from the computed residual, finite residuals
and output, four invalid bindings, and D3D12 debug-layer errors when available.
The CPU composition oracle validates transport/scatter/composition, not network
quality for the changed activation. Timestamp measurements include base/output
staging and diagnostic residual copies, exclude startup and final surface
readback, and are not a release-mode performance benchmark.

## Integration contract

- Construct one executor with the device and eleven shader byte arrays. Keep it
  alive until every recorded use has completed on the caller's fence.
- Call `Record(list, inputs)` after producer commands on the same ordered queue.
  Recording performs no file reads, CPU mapping, queue submit, or GPU wait.
- Supply the actual resource states. They are restored on return in the recorded
  command stream. Shared main/skip allocations must declare the same state.
- Keep caller resources alive until GPU completion. Compute PSO, root signature
  and root arguments are overwritten; the caller must rebind subsequent work.
- A single executor can be reused in order on one queue. Do not concurrently
  record or execute its scratch allocations on independent queues.
- Inputs must be default-heap buffers with bounded, 4-byte-aligned offsets.
  Base and output must be distinct, single-mip/slice/sample 640x360 RGBA16F
  textures. Other dimensions, in-place textures, descriptor subviews and
  alternate output/blend contracts are not implemented.

This is a tested recording component, not a game-ready replacement. The NVAPI
slot154 handler now selects this recorder for explicitly registered host sessions
(see `../nvapi_amd/README_OUTPUT_HEAD_D3D12.md`). Other lists keep the laboratory
HIP path. Automatic live-game integration still needs actual state tracking and
queue/fence ownership; the controlled NVAPI test supplies these explicitly.
Upstream stages still need to produce main/skip activations on the ordered path.
The optional residual readback is only for diagnostics and is omitted in normal
recording. No real multi-frame scene or dynamic-resolution claim follows from
the synthetic eight-invocation test.
