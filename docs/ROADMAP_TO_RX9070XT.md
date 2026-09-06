# DLSS 5 Neural Rendering on RX 9070 XT — execution roadmap

## Active route and current result (2026-09-05)

The next user-approved milestone is now original-weight ROCm native whole-frame
quality, including SDR AND HDR, with no minimum FPS for this first milestone.
No NVIDIA/PTX/ZLUDA runtime or CPU neural fallback is allowed in the final native
executor. Original weights are frozen first; optional derived fine-tuning is
limited to eight GPU hours and requires structural/teacher gates. See
`ROCM_NATIVE_IMPLEMENTATION.md` for current partial implementation and evidence.
The old "unmodified DLL" completion contract below is historical compatibility
work ONLY; it is explicitly not required by the active native reconstruction.

Immediate user-selected milestone: complete continuous original-weight game
inference/display as a deliberately low-FPS debug version BEFORE optimizing
native families or teacher-based fine-tuning. E-172 proves persistent execution
and E-173 proves the changing game-texture inference/writeback loop; E-173 does
NOT prove the output reaches the screen. E-174's separate pre-Present debug bridge
passes 12 distinct live buffers plus visual running/stop verification
(`RESIDENT_PRESENT_DEBUG.md`), at about 5.2 seconds per frame. It uses display-referred
input and includes HUD, so it does not replace the final NR color/temporal game
integration. Independent 640x360 tiles are a transport bring-up, not genuine
arbitrary-resolution network quality or full-frame attention equivalence.

The user-selected native reconstruction route is specified in
`OPERATOR_RECONSTRUCTION.md`: user-local original weights, AMD-native operators,
complete game-facing frame, no runtime NVIDIA code and no intermediate bitwise
requirement. The legacy unmodified-DLL contract below applies to the older
compatibility track, not as an extra requirement on this reconstruction route.

E-142 closes controlled automatic head binding and 1,000-head-call retirement.
E-143 proves actual AMD upstream reaches the DXIL head, but the image gate fails
(RGB correlation 0.683468, NRMSE 0.812474). E-144 adds one reusable native chained
Swin family across four slots; isolated cases pass, while the integrated candidate
regresses (0.638833/0.854350), so it remains experimental. No game-ready claim.

Remaining release gates: other native upstream families and tensor views;
representative same-input real-frame RTX targets and temporal/reset sequences;
full-frame quality acceptance; one-queue resource-resident scheduling; selected
game input/output binding; 1,000 real frames with failure/lifetime checks; profiling
and optimization. Head-only controls and zero-input images cannot close these.

## Historical compatibility completion contract (not the active native route)

The project is complete only when one reproducible run satisfies all of these:

1. The unmodified, user-owned `nvngx_dlssnr.dll` is loaded and feature 18 is
   created through a documented host/compatibility path.
2. Every required operation in a complete frame is executed by HIP/AMD code on
   DXGI vendor `0x1002`, architecture `gfx1201`; no FSR, XeSS, CPU substitute or
   hard-coded success result is used.
3. The output is finite, non-identity, deterministic for a deterministic input,
   and completes without skipped slots or placeholder kernels.
4. The same input sequence is run on the RTX reference. Intermediate tensors and
   final images meet tolerances established per precision class; the final target
   is PSNR >= 35 dB and SSIM >= 0.98 versus RTX unless repeated-reference variance
   proves a different justified bound.
5. A 1,000-frame stability run has zero API failures, device faults, NaN/Inf
   outputs or resource-lifetime leaks. Performance is measured only after this.

Returning success from NGX/NVAPI, loading a DLL, replaying marker kernels, or
implementing only clear/copy operations never satisfies this contract.

## Dependency chain

| Milestone | Experiment and hard pass criterion | State |
|---|---|---|
| M0 Reference graph | RTX feature 18 creates/evaluates; exact 9-module, 96-function, 156-slot graph captured across five stable frames | **PASS** R-17..R-21 |
| M1 AMD graph envelope | RX 9070 XT replays 156 geometries and GPU-hashes all 11,624 parameter bytes with zero mismatch | **PASS** R-22/E-18 |
| M2 Drop-in NVAPI ABI | A DLL returned by `nvapi_QueryInterface` implements the five lifecycle/launch calls plus both descriptor-object calls; real AMD D3D12 device runs 9/96/156 lifecycle self-test | **PASS transport-only** |
| M3 Resource translation | Build D3D12-VA -> imported HIP-pointer interval map; prove buffer offsets, aliasing and fence ordering in both directions | **IN PROGRESS**; direct address identity disproved |
| M4 Boundary operations | RTX-observe and AMD-match `cc_cb_clear` then `cg2r_copy_kernel` exactly, including address relocation | **PASS**: unchanged PTX clear passes; RTX 640x360 RGBA16F copy is bitwise identity; RX 9070 XT native copy matches the exact RTX input/output bytes |
| M5 Tensor oracle | Targeted RTX package snapshots bounded pre/post ranges for selected slots, with resource identity, dtype, shape, stride and hashes | **IN PROGRESS / SLOT154 ORACLE COMPLETE**: the queue-complete output and output-preserving E4M3/movmatrix/MMA traces are accepted; intrusive packed-F16 traces are explicitly rejected |
| M6 Neural translation/reconstruction | Function-isolated PTX translation or clean-room HIP fallbacks pass operator-level RTX comparisons in dependency order | **OUTPUT HEAD COMPLETE / UPSTREAM IN PROGRESS**: activation fusion through final RGBA16F composition now runs as 11 DXIL dispatches on one D3D12 list using original weights and no HIP. Final surface mean/max error is `3.41e-6 / 8.63e-4`. The active boundary is live slot-154 binding and then the upstream slots 0-153 graph |
| M7 Standalone frame | All 156 slots run real implementations on AMD; complete output meets numerical/image gate | **EXECUTION PASS / IMAGE FAIL**: the current N0, slots2-6 and post-block candidate completes two deterministic 156-slot runs without injection. Fine checkpoints through slot 10 pass; slot 11 is the first failure (`0.134164531`). Non-S7 exact-state controls show slot 11 itself passes and localize the next native target to propagated slot-9/slot-10 state |
| M8 Feature-18 integration | Unmodified DLSSNR runtime reaches the AMD NVAPI backend and completes a frame | pending |
| M9 Product gate | 1,000-frame stability, multi-input validation, then profiling/optimization | pending S9/S10 |

## Kernel translation/reconstruction order

The tested graph uses 43 functions. Work follows data dependencies rather than
module file order:

1. Boundary and address mechanics: `cc_cb_clear`, final `cg2r_copy_kernel`.
2. Input transform: `cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8`.
3. Swin pyramid: 1h/32, 2h/64, 4h/128, 8h/256 families.
4. Split 16h/512 family: QKV, attention/projection, FFN and synchronization.
5. ViT 1D family: QKV, attention, projection, FFN, 2D/1D repacks.
6. Decoder/upsample and output post block.

For every function the acceptance record must contain: parameter schema; every
pointer translated through a registered interval; tensor dtype/shape/stride;
weight provenance; RTX input/output hashes; AMD error metrics; launch geometry;
and a negative test that detects an intentionally changed byte or dimension.

## Resource and synchronization design

R-23 proves that an imported D3D12 heap does not retain address identity:
`D3D12 VA 0x200c40000 != HIP pointer 0x304000000`. The backend therefore needs a
registry entry per shared allocation:

```
{d3d12_va_begin, byte_size, hip_base, resource_kind, format, heap_offset,
 producer_fence, consumer_fence, generation}
```

Every pointer-like kernel field is resolved as
`hip_base + (captured_d3d12_va - d3d12_va_begin)`. Unregistered, out-of-range or
stale-generation addresses are fatal. D3D12/HIP queue transitions use imported
external semaphores; global device synchronization is allowed only in bring-up
tests. Textures use explicit staging until AMD swizzle-aware addressing is proven.

## RTX oracle acquisition

Routine full-frame RTX reruns are finished. New RTX runs are requirement-driven:

- B0: slot-0 clear buffer before/after plus 16-byte parameter decode.
- B1: slot-155 copy source/destination footprints before/after.
- N0: pre-block input/output tensor window, dtype/shape/stride and weight handles.
- N1+: one capture package per newly reconstructed neural family.

Snapshots are bounded and hashed; proprietary binaries and bulk model data are
never placed in the repository. The returned package contains only user-generated
experiment outputs needed for interoperability validation.

## Current blockers and decisions

- All 15 runtime containers carry compressed PTX 9.4/sm_120 as well as CUBIN.
  Function isolation makes the simple clear entry runnable through ZLUDA on AMD.
  The main route is now PTX reuse plus targeted lowering; clean-room HIP remains
  the fallback for instructions ZLUDA cannot translate.
- The copy kernel's CUDA surface-store gap has a native D3D12 lowering. The RTX
  footprint is bitwise identity and the RX 9070 XT lowering reproduces that exact
  640x360 RGBA16F input/output pair, so the M4 copy content gate is closed.
- For isolated N0, tuple and cache-policy diagnostics are now eliminated by the
  compatibility lowering. Its remaining 728 diagnostics are exactly the four
  numerically oracle-covered families: FP8 conversion, FP8 MMA, f16 MMA and
  movmatrix. Integrating those implementations and recovering captured tensor
  dataflow are now the critical path.
- The N0 tiled 2x2 downsample epilogue is now an actual RX 9070 XT HIP stage with
  RTX raw comparison. The next boundary moves upstream: recover the pre-epilogue
  32-channel FP16 fragments from the learned weight/MMA body, then replace the
  captured scratch input with locally generated N0 activations.
- All isolated N0 PTX instruction families now lower, and the full `80x48x32`
  entry runs with a valid texture object, captured weight window and relocated
  264-byte ABI on RX 9070 XT. Dense scratch/output readback closes the execution
  gate. The immediate gate is now the prepared RTX 50 same-input run of the
  original SM120 PTX, followed by tensor error localization and correction. The
  55.8 MB scalar form remains a correctness baseline; FP8 MMA must later be
  replaced with an efficient AMD matrix path.
- The pre-MMA checkpoint is bitwise exact, while the original post-f16 checkpoint
  differed in every FP16 value. This exposed and corrected the scalar lowering's
  A-fragment source mapping. The repaired post-f16 export is bitwise exact and the
  complete corrected N0 now passes the full scratch/output numerical gates on RX
  9070 XT. N0 becomes the correctness template for the next dependent neural
  family; efficient AMD matrix lowering is deferred until graph correctness.
- The first downstream Swin entry introduces no new matrix/conversion semantics.
  Existing oracle-backed lowerings plus removal of a non-semantic cache hint make
  it compile, resolve and execute on RX 9070 XT. Its output and 960 CTA release
  writes are deterministic. The immediate external gate is the prepared RTX
  slot-2 same-input package; a parity pass will allow slots 3-5 to reuse the same
  module resources and advance the complete 1h/32 family.
- NVIDIA `nvdisasm` 13.x supports SM120 and JSON output, but it is not installed
  locally and its package carries NVIDIA's EULA. Installation requires the user's
  license acceptance; work that does not need it continues meanwhile.
- The public NGX host rejects AMD before feature creation. Initial development
  uses a standalone compatibility harness; integration with the unmodified
  runtime occurs only after real kernels and resource translation are validated.
- Marker, identity, clear and copy backends remain labeled `LAB_TRANSPORT_ONLY`.
- The clean-room output head is no longer a marker/identity path. All 256 FP8
  MMAs, 16 FP16 tail MMAs, spatial scatter, residual composition and D3D12
  texture staging execute on RX 9070 XT in one 9/9 PASS run. This closes the
  standalone slot-154 operator gate but not M7: its input is still a captured
  activation tensor rather than one produced by an entirely native preceding
  decoder.
- The output head no longer requires nine processes or disk intermediates. The
  resident executable matches the modular residual bitwise, and its D3D12
  variant imports the base/output staging allocation and shared fence in that
  same process with zero texture-component mismatches. Its current 20.24 ms
  measurement is an unoptimized correctness baseline. The next boundary is the
  actual feature-18 evaluate resource registry: replace file-loaded activation
  and harness-owned textures with registered live resources, then optimize
  matrix kernels only after the live correctness gate passes.
- Slot154 offset 56 resolves to a real 640x360 RGBA16F SRV using a point/border
  sampler, but v21 proves that resource is genuinely all zero immediately before
  launch. The original RTX output is nevertheless byte-exact. The former claim
  that nonzero texture pixels were missing is rejected; the next oracle run
  locates the first output-changing transform across seven cumulative lowering
  stages while preserving native CUDA surface and texture semantics.
- The slot-154 compatibility entry now preserves the real independent-surface
  token, automatically joins D3D12 descriptors to resources, and writes the
  accepted output into the real RGBA16F UAV. The same opt-in tracker also makes
  and imports the two proven committed buffer sizes without self-test-side
  registration. Remaining integration work is no longer descriptor discovery;
  it is execution ordering and data production. A HIP call made during command
  list recording cannot observe earlier, not-yet-submitted D3D12 commands.
  Either the neural graph must be lowered to D3D12 compute on that list, or an
  explicit split/fence scheduling contract must be established before a game
  run can be safe. Slots 0-153 must then produce the activation in that ordered
  path.
- The decision trace is now implemented and locally regression-tested. Feature-
  18 package v26 records all frame-1 launches plus subsequent list-close and
  queue-submit boundaries in one sequence domain. The next external action is
  one RTX 5070 run; only its returned topology determines which scheduling
  branch is valid.
- The v26 RTX return selects the scheduling branch: slots 0-155 all occupy one
  command list and are submitted only after slot155. Therefore an independent
  HIP queue/fence cannot consume slot153 output at slot154. Implement the
  resident output head as D3D12 compute recorded on the supplied list, validate
  its existing `A9CD8774...` standalone oracle, and then migrate upstream
  families into the same ordered path. An external split remains only a
  bring-up fallback if interception is moved above command-list recording.
- The first migration slice is complete: the surface scatter/compositor is now
  native DXIL and byte-exact on RX 9070 XT. The second slice is also complete:
  the FP16 tail and surface execute together on one D3D12 list. Its final image
  error is bounded at `3.05e-5`; the residual's non-bitwise `v_dot2` difference
  is accepted under the numerical gate. The third slice ports final projection
  to DXIL with zero mismatches across 16,257,024 bytes and retains that exactness
  when chained with the two later stages. Softmax/V is also native DXIL and the
  four-stage attention-to-surface suffix passes without a HIP dependency. Port
  backward from the Q/K/V boundary in dependency order: Q/K/V preparation and
  QK scoring -> first projection/activation. This work is now complete as a
  single 11-dispatch output-head executor; do not return to per-layer RTX
  experiments unless a live-frame checkpoint fails. Each stage must use
  explicit FP16 RN-even where DXIL conversion differs and pass its existing
  intermediate oracle before being wired into live slot154.
