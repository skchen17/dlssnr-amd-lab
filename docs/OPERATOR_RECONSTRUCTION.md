# AMD-native operator reconstruction track

## Objective

Run a complete neural-rendering frame on RDNA4 using a user-local model pack and
AMD-native tensor operators.  The game-facing output is a complete RGBA16F
frame.  The implementation does not load NVIDIA code at run time and does not
require instruction-level, ULP-level, or intermediate bitwise parity.

The prior exact-parity work remains a read-only reference oracle.  It is not the
delivery critical path for this track.

## Proprietary-data boundary

- The repository never contains or distributes NVIDIA binaries or model data.
- A user supplies a legitimately obtained DLL locally.
- `extract_dlssnr_weights.py` extracts the matching PE resource into
  `local_models/`, which is ignored by Git.
- Captured `model_arena.raw` data may be converted into a local operator pack for
  interoperability research, but must never be copied into `deliverables/`.
- Publication or distribution requires a separate license review.  Local
  extraction alone is not a conclusion about license rights.

## Phase-one gates

| Gate | Requirement | State |
|---|---|---|
| O0 | Build a validated local weight-resource artifact | **PASS** — 153 local FP16 records decode byte-exactly to the accepted arena |
| O1 | Collapse the accepted 156-slot graph into stable operator stages | **PASS** — 14 stages, zero unclassified slots |
| O2 | Prove fused residual-to-complete-frame output on gfx1201 | **PASS** — RX 9070 XT, exact identity/control values, 0.031334 ms at 1080p |
| O3 | Replace one full stage with AMD-native tensor operators | **PASS for captured output-head contract** — full-grid DXIL and caller-resource recorder, E-139/E-140 |
| O4 | Execute all stages without PTX/ZLUDA or NVIDIA runtime code | pending |
| O5 | Produce a visually useful captured frame | pending |
| O6 | Connect the complete output to a D3D12 game path | pending |

## Semantic stage graph

The captured graph is treated as evidence for topology and bindings, not as an
execution prescription:

1. input/pre-block and 1h/32 encoder
2. 2h/64 encoder
3. 4h/128 encoder
4. 8h/256 encoder
5. 16h/512 window-transformer bottleneck
6. 1D transformer bottleneck
7. 16h/512 decoder
8. 8h/256, 4h/128, 2h/64 and 1h/32 decoder
9. output head and complete-frame write

Each stage is an AMD fusion boundary.  Multiple captured functions may become a
single HIP kernel or a short rocWMMA/Composable Kernel pipeline.  Acceptance is
based on finite deterministic output, image quality, temporal stability, and
performance rather than equality of NVIDIA intermediate values.

## Local commands

```powershell
python scripts\extract_dlssnr_weights.py `
  C:\path\to\nvngx_dlssnr.dll `
  local_models\resource_310_8

python scripts\decode_dlssnr_weight_resource.py `
  local_models\resource_310_8\weights_resource.bin `
  local_models\decoded_310_8 `
  --reference-arena results\20260831_234000_full_graph_integrated_plan\model_arena.raw

python scripts\build_operator_model_pack.py `
  results\20260831_234000_full_graph_integrated_plan\plan.json `
  local_models\operator_310_8_local_decode `
  --model-arena local_models\decoded_310_8\model_arena.raw

powershell -ExecutionPolicy Bypass -File scripts\build_all.ps1 `
  -Only residual_compositor

.\build\residual_compositor.exe `
  --width 1920 --height 1080 --iterations 500 `
  --json results\operator_residual_compositor.json
```

## Phase-one observed result

The locally extracted `WEIGHTS_HT` resource is a complete model source rather
than a partial cache. It contains 153 named FP16 records. Removing the record
headers and padding every payload to a 512-byte boundary produces a
147,719,680-byte arena with SHA-256
`A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`.
That is byte-for-byte identical to the previously accepted RTX-captured model
arena. RTX hardware is therefore no longer required to obtain the phase-one
model bytes; it remains useful only as a quality oracle while native stages are
being replaced.

The complete-frame epilogue also passes on the local RX 9070 XT. At 1920x1080,
500 launches average 0.031334 ms, the zero-residual path is bitwise identical to
the base image, and the nonzero test has zero component mismatches. This measures
the residual composition cost only, not the neural network inference cost.

## Output-head reconstruction progress

The complete 81x49 output-head grid and final RGBA16F store now pass (E-139/E-140).
The earlier internal reconstruction contracts are retained below:

| Sub-gate | Evidence | State |
|---|---|---|
| O3a | Join slot 154 launch, bindings, dimensions, local tensors and PTX inventory | **PASS** |
| O3b | Recover the packed 512x16 projection weight layout | **PASS** — 4,096 RTX B-operand bytes checked, zero mismatches |
| O3c | Generate pre-MMA A fragments from raw main/skip activations on gfx1201 | **PASS for CTA (70,26)** — 2,048 FP16 and 2,048 E4M3 values, zero RTX mismatches |
| O3d | Execute the captured output-head FP8 MMA arithmetic on gfx1201 | **PASS** — 256 MMAs, 32,768 FP16 outputs, zero RTX mismatches |
| O3e | Generalize activation fusion and the projection chain to all 81x49 CTAs | **PASS through FP8 MMA 175** — raw-input pipeline, 22,528 selected-CTA FP16 checkpoints, zero RTX mismatches |
| O3f | Execute the remaining FP8/FP16 projections, pixel shuffle and RGBA16F write | **PASS for captured contract** — 11 DXIL dispatches and texture recorder |

The recovered activation fusion is:

```text
main_rounded = fp16_rn(main_e4m3 * main_scale_fp16)
fused_fp16   = fp16_rn(fma(skip_e4m3, skip_scale_fp16, main_rounded))
mma_a_e4m3  = e4m3_rn_satfinite(fused_fp16)
```

This fusion order is material: rounding both products before the addition does
not reproduce the RTX operand. The native kernel applies the observed contract
explicitly and therefore does not depend on compiler contraction choices.

The full-grid activation fusion is now deterministic over all 81x49 CTAs and
produces 8,128,512 E4M3 values without NaN codes.  The native path from raw
activations through FP8 MMA 127 validates 16,384 selected-CTA FP16 checkpoints
with zero mismatches, including the observed activation function, residual
seed and the three dependent accumulation steps.  FP8 MMAs 128 through 175 add
another 6,144 zero-mismatch checkpoints. All 256 FP8 MMAs are now generated
from raw inputs and local weights: the remaining path includes Q/K/V
projection, cosine normalization, Q times K, stable softmax, softmax times V,
and the final attention projection. The native route intentionally uses a
standard stable softmax rather than NVIDIA's instruction-level approximation.

Observed RX 9070 XT timings are 0.277135 ms for full-grid activation fusion,
13.023720 ms for the correctness-first MMA 0-127 implementation, 5.534660 ms
for MMA 128-175, 3.596625 ms for Q/K/V plus QK, 3.466570 ms for softmax-times-V,
and 1.843400 ms for the final attention projection. These scalar HIP kernels
favor an auditable numerical
contract over throughput and must be replaced with tiled wave/WMMA kernels
before game integration. The 0.013211 ms all-MMA captured-operand replay remains
a microbenchmark and is not an end-to-end network estimate.

Machine-readable evidence:

- `results/20260904_241000_output_head_contract/output_head_contract.json`
- `results/20260904_241500_output_head_weight_layout/weight_layout.json`
- `results/20260904_243000_output_head_mma_replay_rx9070xt/manifest.json`
- `results/20260904_244000_output_head_activation_fusion_rx9070xt/manifest.json`
- `results/20260905_003000_output_head_mma_dependencies/mma_dependencies.json`
- `results/20260905_004000_output_head_first128_rx9070xt/manifest.json`
- `results/20260905_005000_output_head_mma128_175_rx9070xt/manifest.json`
- `results/20260905_007000_output_head_qk_attention_rx9070xt/manifest.json`
- `results/20260905_008000_output_head_softmax_v_rx9070xt/manifest.json`
- `results/20260905_009000_output_head_final_projection_rx9070xt/manifest.json`
- `results/20260905_010000_output_head_fp16_tail_rx9070xt/manifest.json`
- `results/20260905_011000_output_head_surface_store_rx9070xt/manifest.json`
- `results/20260905_012000_d3d12_residual_bridge_rx9070xt/manifest.json`
- `results/20260905_013000_output_head_pipeline_rx9070xt/manifest.json`

The selected-CTA Q/K/V pack differs from RTX in only four Q and one K E4M3
codes out of 6144; V is bit exact. QK mean/max absolute error is
0.000658/0.042969. Stable-softmax attention is 0.044274/0.398438, and the final
32-channel attention projection is 0.042550/0.703125. All full-grid runs are
deterministic and contain no non-finite values. These are explicit native
tolerances, not a claim that the NVIDIA softmax approximation is bit exact.

The remaining output path is now reconstructed too. Sixteen FP16 MMAs project
the 32-channel attention result to four residual channels. The PTX shuffle
reduces to four 4x4 token tiles arranged as an 8x8 CTA, with grid origin
`(-4,-4)`. The selected RTX store trace proves the final RGB contract is
`clamp(base + 0.25 * residual, 0, 1)` with alpha fixed to one. On RX 9070 XT,
the spatial mapping and 192 RTX formula values are exact; the final RGB error
from native residual math is mean `0.000159`, max `0.000764`. A real D3D12
RGBA16F texture staging roundtrip then composes the locally generated residual
through an imported HIP heap and external fence with zero output mismatches.

Local reproduction commands:

```powershell
python scripts\analyze_output_head_contract.py `
  results\20260831_234000_full_graph_integrated_plan\plan.json `
  local_models\decoded_310_8\manifest.json `
  local_models\decoded_310_8\model_arena.raw `
  results\20260831_230000_decoder_full_graph_exact_state\cases\manifest.json `
  results\20260831_175500_full_graph_ptx_access\cc_tinlayout_fused_post_block_swin_1h_32_fp8.ptx `
  results\output_head_contract.json

python scripts\analyze_output_head_weight_layout.py `
  results\20260904_760000_postblock_exact_mma_trace_cross_vendor\rtx_trace.raw `
  local_models\decoded_310_8\model_arena.raw `
  results\output_head_weight_layout.json

powershell -ExecutionPolicy Bypass -File scripts\build_all.ps1 `
  -Only output_head_activation_fusion,output_head_first128,output_head_mma128_175,output_head_qk_attention,output_head_softmax_v,output_head_final_projection,output_head_fp16_tail,output_head_surface_store,d3d12_residual_bridge

.\build\output_head_activation_fusion.exe `
  deliverables\postblock_mma_trace_reference_20260904_154238\payload\activation_arena.raw `
  local_models\decoded_310_8\model_arena.raw `
  results\20260904_780000_postblock_exact_e4m3_mov_cross_vendor\rtx_trace.raw `
  results\output_head_activation_fusion.json 1000 `
  results\full_grid_a_e4m3.raw results\full_grid_a_fp16.raw

.\build\output_head_first128.exe `
  results\full_grid_a_e4m3.raw results\full_grid_a_fp16.raw `
  local_models\decoded_310_8\model_arena.raw `
  results\20260904_760000_postblock_exact_mma_trace_cross_vendor\rtx_trace.raw `
  results\output_head_first128.raw results\output_head_first128.json 20

.\build\output_head_mma128_175.exe `
  results\output_head_first128.raw `
  local_models\decoded_310_8\model_arena.raw `
  results\20260904_760000_postblock_exact_mma_trace_cross_vendor\rtx_trace.raw `
  results\output_head_mma128_175.raw results\output_head_mma128_175.json 20

.\build\output_head_qk_attention.exe `
  results\output_head_mma128_175.raw local_models\decoded_310_8\model_arena.raw `
  results\20260904_760000_postblock_exact_mma_trace_cross_vendor\rtx_trace.raw `
  results\20260904_780000_postblock_exact_e4m3_mov_cross_vendor\rtx_trace.raw `
  results\q.raw results\k.raw results\v.raw results\qk.raw results\qkv.json 20

.\build\output_head_softmax_v.exe `
  results\qk.raw results\v.raw `
  results\20260904_760000_postblock_exact_mma_trace_cross_vendor\rtx_trace.raw `
  results\20260904_780000_postblock_exact_e4m3_mov_cross_vendor\rtx_trace.raw `
  results\softmax.raw results\attention.raw results\softmax_v.json 20

.\build\output_head_final_projection.exe `
  results\attention.raw results\output_head_first128.raw `
  local_models\decoded_310_8\model_arena.raw `
  results\20260904_760000_postblock_exact_mma_trace_cross_vendor\rtx_trace.raw `
  results\20260904_780000_postblock_exact_e4m3_mov_cross_vendor\rtx_trace.raw `
  results\attention_projected.raw results\attention_projection.json 20

.\build\output_head_fp16_tail.exe `
  results\attention_projected.raw local_models\decoded_310_8\model_arena.raw `
  results\20260904_761000_postblock_exact_mma_trace_cross_vendor_corrected\rtx_trace.raw `
  results\rgba_residual.raw results\fp16_tail.json 20

.\build\output_head_surface_store.exe `
  results\rgba_residual.raw - `
  results\20260904_761000_postblock_exact_mma_trace_cross_vendor_corrected\rtx_trace.raw `
  results\20260904_530000_postblock_store_trace_cross_vendor\rtx_postblock_store_trace.raw `
  results\zero_base_output.raw results\surface_store.json 20

.\build\d3d12_residual_bridge.exe `
  results\rgba_residual.raw results\d3d12_output.raw `
  results\d3d12_bridge.json 100

# Or run all nine native stages in dependency order.
.\scripts\run_output_head_pipeline.ps1 -Iterations 20
```

The preferred validation path is now the complete D3D12 executor:

```powershell
.\scripts\run_output_head_full_d3d12.ps1
```

It consumes the captured activation arena and original model weights, then
records all 11 reconstructed dispatches on one command list. The per-stage HIP
commands above remain diagnostic fallbacks, not the primary development loop.

## Current whole-frame decision (E-142..E-144)

Automatic head state/fence binding is tested in a controlled host; run
`scripts/run_nvapi_amd_output_head_d3d12_selftest.ps1 -Automatic -Batches 250`.
This does not establish game integration or temporal support.

The native chained 1h/32 family covers slots 3/4/151/152 as an experimental
implementation. All four isolated cases pass, but substituting the family into
the complete graph worsens the final-frame gate. It is **not promoted**. See
`tools/swin1h_d3d12/README.md` for runnable commands and evidence.

The key remaining work is complete upstream reconstruction and a representative
real-frame/sequence acceptance set, followed by resident graph/game integration
and optimization. Stop using captured intermediate parity or near-black PSNR as
the definition of completion. Do not request another individual RTX instruction
trace merely because a local family test differs bitwise.

## Earlier recorder integration unit (historical)

The complete output head is available as a D3D12 recorder in
`tools/output_head_surface_d3d12/output_head_d3d12_runtime.h`. Run its eight-call,
two-submission test with:

```powershell
.\scripts\run_output_head_recording_selftest.ps1
```

It consumes caller-owned GPU slices/textures, records all 11 dispatches without
submission or waits, restores resource states and returns an actual RGBA16F
texture. The first/recovered outputs match the standalone DXIL baseline exactly;
changed main/base inputs take effect. This validates scheduling and resources in
a controlled harness, not an intercepted feature18 evaluation.

The slot154 handler now selects this recorder for explicitly registered host
sessions with resource states, descriptor identities and fence/lifetime ownership
(E-141). Its eight-call NVAPI test passes with byte-exact recorder outputs; run
`.\scripts\run_nvapi_amd_output_head_d3d12_selftest.ps1`. Automatic game state and
completion tracking are the remaining binding task. Unregistered lists still use
the previous synchronous HIP laboratory path.
The captured recorder fixes blend to 0.25 and dimensions to 640x360; validate the
launch contract rather than assuming every live frame uses these values. Earlier
encoder/bottleneck/decoder families must then supply the native activations. Full
frame RGB and temporal gates remain the acceptance target; no new per-instruction
RTX experiments are required by the recorder result.
