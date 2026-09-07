# NRPlan resident FP8 pool bring-up

## Outcome

The C32, C64, C128, and C256 standard Swin block families now have a fixed-sequence
bring-up path that keeps FP8 activations in a compact `NRPlan`-owned allocation.
It no longer relies on the byte-packed subregions of the 550,736,384-byte generic
arena for producer/consumer hand-off.

This is an approximate-performance track. The strict graph path is unchanged and
remains the fallback. The new path has not yet produced a full-frame performance
result and is not enabled for game deployment.

## Root cause isolated

Record 15 (C256), one 8x8 window, established all of the following:

- the fixed command sequence was entered exactly once;
- the plan-owned E4M3 publisher produced a byte-exact post-FFN resident tensor;
- copying the same completed bytes into the generic arena resident subregion did
  not produce a valid consumer-visible tensor;
- pointer audits, the E4M3 LUT, model weights, FP16 FFN output, and resource
  lifecycle were valid;
- QKV project, normalization, and resident publication also require an explicit
  producer/consumer boundary on this runtime.

The selected bring-up design therefore uses a fixed seven-segment resident pool:

1. post-FFN E4M3;
2. Q;
3. K;
4. V;
5. attention value;
6. projected logical resident output;
7. packed next-block output.

The pool is allocated once when standard stage blocks are configured. No
per-frame allocation or CPU neural operation was added. QKV uses a completed
FP16 project+norm boundary followed by explicit resident publication. Output
projection continues to consume the semantic FP16 residual, then publishes its
E4M3 result to the resident pool.

## Minimal GPU gates

All gates used one window and one submit. They released allocator state to zero
and reported no new system GPU/restart event.

| Family | Final max abs | Final NRMSE | FP16 FFN NRMSE | FP16 output NRMSE | Approx gate |
|---|---:|---:|---:|---:|---|
| C64 record 5 | 0.0625 | 0.022728 | 0.002847 | 0.012339 | pass |
| C128 record 9 | 0.0625 | 0.028150 | 0.003266 | 0.016793 | pass |
| C256 record 15 | 0.03125 | 0.031244 | 0.005187 | 0.018440 | pass |

The legacy strict per-block NRMSE limit remains 0.02 and is not weakened. A
separate approximate-performance gate records finite output, NRMSE <= 0.05, and
max absolute error <= 0.0625. Passing it authorizes continued performance
research only; it does not establish RTX quality or game readiness.

## Current implementation state

- `nr_plan.cpp` owns and releases the resident pool.
- C64/C128/C256 share the same fixed-sequence resident ABI.
- Ordered QKV publishers exist for all three channel families.
- Consecutive same-resolution standard blocks can consume the previous packed
  resident pool output without materializing it back into the generic arena.
- The C64 record 5 -> 6 normal/shifted-window chain passed one-submit GPU
  validation: NRMSE 0.047536, max absolute error 0.125, and all two stage/
  boundary counters were observed.
- The middle Encoder transitions now use a plan-owned reusable target plus a
  persistent skip pool instead of publishing their E4M3 target/skip into the
  generic arena. Record 8 C64 -> record 9 C128 passed at NRMSE 0.050560 and
  max error 0.03125; record 14 C128 -> record 15 C256 passed at NRMSE 0.064962
  and max error 0.015625. Each gate observed exactly one resident transition.
- The C32 special one-group FFN now shares the same resident boundary and its
  own ordered QKV publisher. Record 4 C32 -> record 5 C64 passed the direct
  Encoder transition at NRMSE 0.038678 and max error 0.0625.
- A complete C64 -> C128 -> C64 round trip passed with one retained Encoder
  skip, one Encoder transition, one Decoder transition, and three stage
  consumers. Final NRMSE was 0.073520 and max error 0.03125.
- Production configuration still requires all eight transition descriptors.
  The partial transition entry point is explicitly debug-only and accepts only
  one C64/C128 Encoder boundary for bounded bring-up.
- The captured strict graph and RGP counter safety halt are unchanged.

## C512, ViT, and complete trunk update

- C512 now uses an eight-segment compact resident pool. Record 23 passed its
  one-window gate at NRMSE 0.032801 and max error 0.125.
- C256 -> C512 passed at NRMSE 0.062453/max 0.0625. The symmetric
  C256 -> C512 -> C256 round trip passed at NRMSE 0.085806/max 0.1875.
- The central path owns a 6,553,600-byte ViT pool and a 1,105,920-byte C512
  skip. All sixteen C512 blocks, eight ViT blocks, and both bottleneck
  projections executed and released correctly. Its aggregate approximation
  was finite at NRMSE 0.209533, but max error 24.0 rejected the numerical gate;
  this is retained as functional evidence only.
- The entire record 1..69 1080p trunk (Pre and Head excluded) passed one-submit
  functional validation. It executed 60 Swin/C512 blocks, all eight scale
  transitions, eight ViT blocks, and both bottleneck directions. The output was
  finite, had 17,693,453 nonzero bytes, and hashed to
  `B97CDE1719F1FEAEC0167E14490F8DAD412116629EF49B05E5EB72C9FCD9C3A1`.
- Removing the outer host boundaries preserved that hash and reduced the
  diagnostic event span from 181.493 to 163.573 ms (9.87%). This measurement
  contains neither Pre nor Head and is not the final hot-path benchmark.
- The corrected resident HIP Graph gate passed with the same exact output hash.
  The graph is plan-owned, references no external allocations, and contains
  718 kernel plus 5 memcpy nodes. Its one-frame diagnostic event span was
  150.295 ms, 8.12% below the 163.573 ms asynchronous fixed sequence and
  17.19% below the original 181.493 ms boundary-heavy sequence. Allocator state
  returned to zero and no new GPU/restart system event was observed. Pre and
  Head are still excluded, so this remains a trunk bring-up measurement rather
  than the complete network benchmark.

## Next work

1. The reset/single-color native Pre and Head descriptors are now present. The
   complete record 0..70 graph passed one 1080p submission on RX 9070 XT: all
   8,294,400 FP16 output values were finite and nonzero, output SHA-256 was
   `2A756063FB97D2F88BE384A09129F085055EE08C1E9917B1BF8005270DDB56E3`, and
   the measured graph contained 738 kernels plus 5 memcpy nodes. Its diagnostic
   one-frame event span was 186.000 ms. This is a functional approximate reset
   route, not reference parity or the original temporal route.
2. Graph, weights and workspace are plan-owned and no external allocation is
   referenced. `complete_native_topology=true`, while `deployment_ready=false`
   intentionally remains closed because the history/motion/depth/exposure
   contract is not verified. The initial status-reporting bug that omitted this
   condition was corrected and revalidated.
3. The edge scratch lifetime plan was tightened from 16 to 12 full-resolution
   scalar lanes, reducing its planned size from 1,145,077,760 to 858,808,320
   bytes at 1080p without changing a mathematical boundary. It still requires
   a subsequent bounded output-hash gate.
4. Remove the diagnostic generic arena once the complete graph no longer needs
   its fallback/capture regions, then run reproducible 1080p A/B timing.

## Full-QKV and node-budget update

- A bounded `(window, head, 16-token tile)` QKV kernel now performs projection,
  the established FP16 reduction order, and LUT-backed uniform 32-bit E4M3
  publication in one launch. The complete output hash remained exact.
- Removing the second, redundant standard-block post publisher reduced the
  complete graph to 510 kernels plus 5 memcpy nodes, below the 512-kernel
  acceptance limit. The one-frame diagnostic span was 162.726 ms versus
  186.000 ms for the first native 71-record graph.
- A different 510-kernel candidate that quantized FP16 post repeatedly inside
  QKV regressed to 172.099 ms and was rejected. Kernel count alone is not used
  as the performance gate.
- The 510-kernel configuration is still a one-frame candidate. It needs fresh
  process 12-run B-A-B-A before stable performance acceptance and does not
  change the closed temporal/deployment gate.

## Reproduction commands

Run these from the repository root with the ROCm virtual environment. Omit
`--execute` for CPU-only descriptor validation.

```powershell
.venv-rocm\Scripts\python.exe scripts\validate_native_encoder_chain_nrplan.py `
  --dll results\20260907_native_nr_plan_build_v77_c512_c256_resident_roundtrip\nr_plan.dll `
  --package local_models\native_single_color_gfx1201_v3.nrmpkg `
  --arena results\20260907_native_nr_plan_build_v30\arena_1080p_approx.json `
  --topology results\20260907_native_nr_plan_build_v30\swin32_256_topology.json `
  --split-topology results\20260907_native_nr_plan_build_v30\split512_topology.json `
  --transitions results\20260907_native_nr_plan_build_v30\transition_topology.json `
  --channels 256 --roundtrip --output results\c256_c512_c256_gate --execute

.venv-rocm\Scripts\python.exe scripts\validate_native_69_record_resident_nrplan.py `
  --dll results\20260907_native_nr_plan_build_v79_resident_graph_replay\nr_plan.dll `
  --package local_models\native_single_color_gfx1201_v3.nrmpkg `
  --arena results\20260907_native_nr_plan_build_v30\arena_1080p_approx.json `
  --stage-topology results\20260907_native_nr_plan_build_v30\swin32_256_topology.json `
  --split-topology results\20260907_native_nr_plan_build_v30\split512_topology.json `
  --vit-topology results\20260907_native_nr_plan_build_v30\vit_topology.json `
  --bottleneck-topology results\20260907_native_nr_plan_build_v30\bottleneck_topology.json `
  --transition-topology results\20260907_native_nr_plan_build_v30\transition_topology.json `
  --output results\resident_graph_gate --async-boundaries --execute

.venv-rocm\Scripts\python.exe scripts\validate_native_71_record_nrplan.py `
  --dll results\20260907_native_nr_plan_build_v81_edges_contract\nr_plan.dll `
  --package local_models\native_single_color_gfx1201_v3.nrmpkg `
  --arena results\20260907_native_nr_plan_build_v81_edges_contract\arena_1080p_approx.json `
  --stage-topology results\20260907_native_nr_plan_build_v81_edges_contract\swin32_256_topology.json `
  --split-topology results\20260907_native_nr_plan_build_v81_edges_contract\split512_topology.json `
  --vit-topology results\20260907_native_nr_plan_build_v81_edges_contract\vit_topology.json `
  --bottleneck-topology results\20260907_native_nr_plan_build_v81_edges_contract\bottleneck_topology.json `
  --transition-topology results\20260907_native_nr_plan_build_v81_edges_contract\transition_topology.json `
  --edge-topology results\20260907_native_nr_plan_build_v81_edges_contract\edge_topology.json `
  --output results\record0_70_reset_graph_gate --execute
```
