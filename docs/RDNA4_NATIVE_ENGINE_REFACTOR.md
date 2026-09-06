# RDNA4 native inference-engine refactor

Status: in progress, GPU profiling safety-halted on 2026-09-06 after two
`LiveKernelEvent 141` resets.  The halt and invalid captures are documented in
[GPU_PROFILING_INCIDENT_20260906.md](GPU_PROFILING_INCIDENT_20260906.md).

## Measurement rules

- GPU kernel activity and SPM counters are accepted only from a complete RGP
  manifest plus a valid AMD RDF profile.
- Event spans remain useful for same-process A/B gates, but are not reported as
  kernel-busy sums.
- Every candidate is opt-in.  Bitwise/hash gates run before promotion, and a
  dispatch-count reduction is not a performance pass by itself.
- The two incomplete Pre RGP captures are invalid and excluded from all tables.

## Output Head: first bounded full-grid implementation

The new `head_attention_window_fused` kernel assigns one 128-thread workgroup
to one complete 8x8 window.  Q/K/V and the 64x64 score/probability matrix remain
in 20 KiB bounded LDS; Q storage is reused after QK.  FFN and tail remain
separate, giving four dispatches for the complete Head instead of nine.  The
kernel preserves the recovered FP16 reduction/store boundaries and uses the
original weights without training.

Two independent 640x384, 12-iteration gates were bitwise exact.  Their median
Head event spans changed from 2.6887 to 2.5362 ms (6.0%) and 2.7224 to 2.4788 ms
(9.8%).  Peak PyTorch allocation in the first gate changed from 255.37 to
74.41 MiB.

A completed handshaken RGP A/B capture gives the following sampled GPU data:

| Metric | 9-dispatch global intermediates | 4-dispatch bounded LDS |
|---|---:|---:|
| sampled active time | 3.50248 ms | 2.68660 ms |
| sampled inactive gaps | 1.17696 ms | 0.84232 ms |
| fetch traffic | 289.98 MB | 196.63 MB |
| write traffic | 303.49 MB | 156.43 MB |
| local-video-memory traffic | 6.11 MB | 4.77 MB |
| memory-unit busy | 87.68% | 80.15% |
| memory-unit stalled | 17.17% | 4.06% |
| L0 hit | 37.52% | 51.46% |
| L2 hit | 68.91% | 76.38% |

The baseline is classified as memory-pipeline/cache-traffic bound, not peak-DRAM
bandwidth bound.  The bounded candidate reduces sampled active time to 76.7%,
fetch traffic to 67.8%, and write traffic to 51.5%.  RGP sampling perturbs
timing, so the independent event A/B gates remain the latency cross-check.

The FP16 bounded kernel compiles to 121 VGPR, 52 SGPR, 20 KiB LDS, no scratch,
and 20 WMMA sites.  From the captured gfx1201 limits its static upper-bound
occupancy is 6 waves/SIMD (37.5%), limited by LDS.  This is a static resource
bound, not a runtime occupancy counter.

Evidence:

- `results/20260906_head_bounded_attention_gate640x384_v1`
- `results/20260906_head_bounded_attention_gate640x384_v2`
- `results/20260906_rgp_head_whole_grid_ready_640x384_v2`
- `results/20260906_rgp_head_bounded_ready_640x384_v1`
- `results/20260906_rgp_head_bounded_compare_640x384_v1`
- `results/20260906_head_bounded_attention_build_v1/isa_occupancy_v3.json`

## Pre input/project/pack

`native_fusion_pre_project_pack` now combines reflected input sampling,
conditioning construction, the original FP16 16x32 projection, and the final
packed C32 store in one grid.  The validated positional-noise generator remains
outside this kernel, so special-function semantics were not changed.

With the same supplied noise, original block-0 weight, and input, both 128x128
and 640x384 12-iteration gates are bitwise exact.  Median event spans were
0.9073 to 0.1475 ms (6.15x) and 1.3301 to 0.3262 ms (4.08x).  This is the input
construction/project/pack boundary only, not all of block 0.

Evidence:

- `results/20260906_pre_project_pack_gate128_v2`
- `results/20260906_pre_project_pack_gate640x384_v1`

## Whole-grid scheduling gate

`native_grid_policy.py` adds an explicit per-family scheduling policy.  It can
remove Python `split -> loop -> cat` without changing windows, coordinates or
math.  It is disabled by default and is separately selectable for `pre`, `c32`,
and later channel families.

The first full-network 128x128 A-B-B-A test compared otherwise identical paths:
A used the existing window batching; B enabled the new Pre project/pack kernel
and whole-grid scheduling for Pre/C32.  All outputs matched the established
SHA-256.  Mean-of-process medians was 154.84 ms for A and 155.97 ms for B, a
0.7% regression.  Therefore whole-grid Pre/C32 was not promoted.  The likely
resource/cache cause still requires safe profiler evidence; kernel-count
reduction alone is explicitly insufficient.

Evidence:

- `results/20260906_pre_c32_abba128_a1`
- `results/20260906_pre_c32_abba128_b1`
- `results/20260906_pre_c32_abba128_b2`
- `results/20260906_pre_c32_abba128_a2`

## Profiler coverage and current block

Reliable RGP data currently exists only for the completed Output Head A/B pair.
The attempted module-core profiler target correctly keeps post-ready work to one
prevalidated graph replay, but both initial Pre counter-collection attempts
triggered watchdog resets.  Consequently Pre, C32, C64, C128, C256, C512 and
ViT do **not** yet have accepted RGP busy/gap/VGPR/LDS/occupancy/DRAM/cache
classification under this new measurement protocol.

RGP counter collection remains blocked while `safety/GPU_PROFILE_HALT.json`
has `halted: true`.  After explicit user direction, the safety state permits at
most one minimal non-profiler correctness gate at a time; it still forbids
stress/performance loops and automatic retry.  CPU/static work may continue,
but no candidate can be promoted without its GPU gates.  In particular, true
FP8 resident activations, the remaining bounded channel-family redesign, and
native `NRPlan` graph replay remain pending; the current FP8 WMMA candidates
still receive decoded FP16 tensors and are not a resident FP8 data path.

## Post-halt implementation work

- C32 now has a separate `c32_attention_bounded` policy and exported
  `nr_c32_attention_window_fused` ABI. It reuses the 20 KiB per-window bounded
  attention implementation but cannot inherit Head's correctness result.
- The first isolated one-window C32 GPU gate completed without a device event,
  but was rejected for correctness: 320 half elements differed, max absolute
  error was 0.00048828125, RMSE 0.00013108 and NRMSE 0.00030266. The candidate
  was also slower in that micro-gate (0.1649 ms versus 0.1525 ms), so no
  performance claim is made.
- Static diagnosis found that the first candidate had incorrectly reused the
  Head-specific QK, PV K32-boundary and projection rounding rules. The kernel is
  now parameterized so C32 preserves its established QK inner-half boundary,
  single-K64 PV accumulation and pre-residual projection rounding.
- After explicit user direction, the corrected kernel passed exactly one
  isolated one-window/one-iteration non-RGP gate. It was bitwise identical to
  the seven-launch baseline (zero differing components, identical SHA-256),
  reduced matrix launches from 7 to 2, and released allocator/reserved memory
  to zero. The single sample measured 0.17000 ms versus 0.13218 ms (1.286x),
  which is only a smoke-gate indication and not an accepted performance result.
  No new watchdog, restart or `LiveKernelEvent 141` entry was observed.
- A separately authorized 16-window/one-iteration gate also passed bitwise and
  released all allocator state, but reversed the apparent timing result:
  0.12120 ms baseline versus 0.16963 ms bounded (0.714x). Peak allocated memory
  fell from 1,305,088 to 649,728 bytes and launches remained 7 versus 2. This
  confirms that launch-count reduction alone is not sufficient: the 20 KiB LDS
  kernel's 37.5% static occupancy and per-window workgroup cost dominate once
  more windows are resident. The C32 candidate therefore remains rejected for
  performance redesign despite passing the expanded correctness smoke gate.
- A lower-LDS three-stage experiment then split C32 attention into a 12 KiB
  QKV+norm kernel, an 8 KiB QK+softmax+PV kernel, and the established
  projection. Static occupancy bounds improved to 62.5% and 75%, respectively,
  with zero scratch. Its authorized one-window gate was bitwise exact and
  reduced launches from 7 to 4, but measured 0.32934 ms versus the 0.15870 ms
  baseline. The fused QKV kernel serializes sixteen normalization rows per wave,
  so the candidate is rejected despite its lower LDS footprint.
- The useful 8 KiB attention core has been retained in a new conservative
  `c32_attention_core` strategy. It uses the established parallel QKV projection
  and normalization kernels, then fuses only QK/softmax/PV before the established
  projection. This gives five total launches including FFN instead of seven,
  avoids global score and probability tensors, and is statically/CPU validated;
  it has not yet been submitted to the GPU.
- C64/C128 now have `c64_attention_bounded` and
  `c128_attention_bounded` policies. One workgroup owns one `(window, head)`;
  Q/K/V, scores and probability remain in 20 KiB LDS. Only the final 64x32
  value per head reaches global memory, followed by one separate cross-head
  output projection. This changes six attention dispatches to two while
  avoiding the rejected full-4C-LDS design.
- Both wide FP16 kernels compile with 101 VGPR, 20 KiB LDS, zero scratch and 17
  WMMA sites. Using the captured gfx1201 resource limits, the static upper bound
  is 6 waves/SIMD (37.5%), LDS-limited. The FP8 variants compile with 67 VGPR.
- The build exports all three distinct C32/C64/C128 bounded entry points. CPU
  policy/layout tests pass. None of these facts is a GPU correctness or latency
  pass, so all three candidates remain disabled.

Static evidence:

- `results/20260906_c32_bounded_attention_build_v1`
- `results/20260906_c32_bounded_attention_gate1_v1`
- `results/20260906_c32_bounded_attention_rounding_fix_gate1_v1`
- `results/20260906_c32_bounded_attention_rounding_fix_gate16_v1`
- `results/20260906_c32_staged_attention_build_v1`
- `results/20260906_c32_staged_attention_gate1_v1`
- `results/20260906_bounded_wide_attention_build_v1`
- `results/20260906_bounded_attention_rounding_fix_build_v1`
- `scripts/validate_c32_bounded_attention.py`
- `scripts/validate_wide_bounded_attention.py`
