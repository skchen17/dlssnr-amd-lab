# Full-graph D3D12 prefix and resolution contract

This directory contains the native D3D12 migration boundary for the captured
156-slot network. `PrefixSlot0` is the accepted slot-0 synchronization clear.
It is not a complete network.

`ResolutionPlan` tests variable-size transport around an identity placeholder.
It does not prove arbitrary-resolution neural inference. It defines two candidate modes:

- native dynamic geometry with a proposed eight-pixel alignment; and
- a compatibility route that runs the captured 640x384 working graph over
  overlapping 640x360 output tiles.

Compatibility tiles use midpoint ownership. Every destination pixel is written
exactly once, while overlap gives interior tile edges additional scene context.
At most 16 tiles are resident at once; larger frames are submitted in batches,
so the compatibility working-set size does not grow without bound.

Build and run the CPU planner and WARP boundary tests:

```powershell
$out = Join-Path $PWD 'build_resolution_check'
.\scripts\build_all.ps1 -Only full_graph_resolution_plan -FreshOutputDirectory $out
& "$out\full_graph_resolution_plan_selftest.exe" "$out\plan.json"
& "$out\full_graph_resolution_gpu_selftest.exe" `
  "$out\full_graph_resolution_pack.dxil" `
  "$out\full_graph_resolution_identity.dxil" `
  "$out\full_graph_resolution_unpack.dxil" `
  "$out\gpu.json"
```

The identity shader is test-only. Replacing it with the reconstructed 156-slot
network still requires validation of alignment, vertical halo, tile context,
seams and temporal state. In particular, exact identity transport does not
establish that independently processed tiles reproduce full-frame attention.
