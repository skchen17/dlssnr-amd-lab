# RTX Feature-18 reference package — submission topology v26

This package runs the vendored DLSS5-Feeder standalone host for exactly one
synthetic DLAA evaluation on an RTX machine. It is a targeted diagnostic for the
slot154/155 output, not a long stability run.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_feature18_reference.ps1
```

The default RenoDX variant is `02`, validated on the RTX 5070 reference machine.
If Feature 18 is not created, retry once with `-RenoDxVariant 01`. Variant `02.5`
is reserved for RTX 40-series compatibility testing.

Why v26 exists: v23 and v24 inserted texture readbacks inside the neural command
list. Their returned slot154 images covered different partial row regions despite
identical activation, parameters and zero resources, so those images are now
classified as synchronization-perturbed diagnostics rather than complete
oracles.

Like v25, v26 sets `MODULE_TRACE_DISABLE_INLINE_SNAPSHOTS=1`. The tracer records the graph
and resource bindings but inserts no pre-slot154 or pre-slot155 copies. After the
single evaluation has been submitted, `ModuleTrace_DumpCopyResources` appends a
fresh copy operation and fence to the same queue, waits for completion, and saves:

- `copy_input.raw`: slot155's input, i.e. the completed slot154 RGBA16F surface;
- `copy_output.raw`: slot155's completed RGBA16F destination;
- `copy_snapshot.json`: dimensions, resource identities and hashes.

v26 additionally records every relevant D3D12 command-list creation, `Close`,
and `ExecuteCommandLists` call, and assigns a monotonically increasing sequence
number to all 156 frame-1 neural launches. `summary.json` reports whether slots
0-155 share one command list, whether slots 154/155 are on that same list, and
which close/queue submission follows the final launch. This decides whether the
AMD backend can synchronize at an external queue boundary or must execute the
replacement work directly in the host's D3D12 command list.

Both raw files must be 1,843,200 bytes and bitwise identical. A valid result also
requires one public DLAA evaluation, Feature 18 creation/evaluation, strict JSONL,
the D3D12 device/resource hooks, and zero inline-snapshot arm events.

Return `return_to_lab\dlssnr_feature18_result_<timestamp>.zip`. The result contains
only logs, metadata and captured raw textures; the supplied NGX/ReShade/RenoDX
binaries remain in the local experiment package.
