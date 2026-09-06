# STATUS.md

2026-09-06 NATIVE_OPT3: 4K FULL-FRAME OFFLINE 12 REPEATS PASS WITHIN SAMPLED 6 GB ENVELOPE.
Opt-in window_batch768/query_chunk1024/key_chunk128/compact layout, frozen weights.
3840x2160 synthetic gradient/checker (NOT game/teacher), padded3840x2176:
paired opt2 hot3051.77ms -> opt3 eleven-hot median2515.27ms (2512.01..2525.26ms).
All14 outputs exact; allocated peak4,018,241,536/reserved4,687,134,720 bytes;
device-wide stage samples max5,122,818,048 bytes, not continuous total-process peak.
Opt3 runtime and offline allocator cap5,000,000,000 bytes; non-tensor external
allocations not covered. Offline completed-stage samples exceeding6GB stop.
2342x1382 twelve-repeat hot median1134.79ms, same historical corrected output;
641x361 exact; resident640 twelve alternating inputs pass, hot interval188.20ms.
No game launch/deployment or training; default remainsbaseline12. NOT4K60 or quality.
Larger1536 batch rejected for speed; 1.106GB output-head geometry cache implemented
and tested exact but excluded from named profiles for insufficient whole-frame gain.
CPU815 passed; final audit baseline/game unchanged, checked events empty, zero
releases and normal process exits. Details/commands: `docs/NATIVE_OPT3.md`.

Historical optimization checkpoint:

2026-09-06 NATIVE_OPT2:2342x1382 STAGED HOT MEDIAN1.293s; LIVE TENSOR PEAK1.874GB.
384-window scheduling + global query_chunk128/key_chunk128 + exact compact layout.
No weight/formula/context changes; original FP8 boundaries and input projection fix kept.
128/640/641x361/2342 offline exact A/B passed;640 and2342 each12 candidate static repeats.
2342 paired96 baseline3.573s -> candidate11-hot median1.293s (1.2805..1.3138s).
Tensor peak3,266,070,528 ->1,874,142,720 bytes; NOT total game/process VRAM.
Named opt-in schedule connected to resident executor and metadata-only GPU worker;
128 one,640 one,640 twelve alternating-input D3D12/ROCm TWO-PROCESS tests passed.
640 worker hot interval median199.38ms, first2239.86ms; distinct stable outputs,
first-frame historical baseline exact, every D3D12 consumer output exact, normal
exits/zero releases. Producer never initialized HIP. NOT game FPS or4K60.
Frozen-head-weight cache also tested exact but small benefit; excluded fromnative_opt2.
Game default remainsbaseline12; no game deployment/start, HDR/HUD/quality revalidation
or training. All proof and commands: `docs/NATIVE_OPT2.md`.
Final CPU regression:812 passed in53.86s. Final audit confirms frozen baseline
and game hashes unchanged, game closed, checked recent system events empty;
see results/20260906_native_opt2_final_audit.json. No TDR changes.

Historical optimization checkpoint:

2026-09-06 OFFLINE NATIVE BATCH OPTIMIZATION:2342x1382 HOT STAGED23.18s ->3.49s.
Read original fused-kernel inventory and measured stages; all-window96 schedule is
opt-in at model API/probe, not yet deployed to game (resident game default remains12).
128/640/2342 offline A/B: four outputs per case bitwise exact against corrected
same-input candidate.640 hot1.811s ->0.414s;2342 hot23.176s ->3.491s (6.64x).
One hot sample per schedule, staged host wait/finite timing; NOT game FPS or4K60.
2342 peakallocated3,266,070,528/reserved3,948,937,216 bytes, zero after release.
During validation OLD unbounded input projection failed same-input repetition at
2342 before any schedule change. Function boundaries isolated16->32 FP32 GEMM;
same-input/weight isolated GPU result failed independent1024-row CPU oracle.
Replaced only pointwise projection scheduling with65536-row FP32 batches, original
FP16 store/weights/formula preserved.128/640/2342 row oracles and subsequent full
chain repeats pass.640 retains historical accepted output;2342 corrected output
changes vs old faulty projection, NOT a proof of NVIDIA parity or artifact removal.
Exact underlying BLAS/runtime defect remains unproven. Failed reports preserved.
No game launch/training/TDR edits. Old ReShade proofs without matching complete
native-source hashes now rejected BEFORE game deployment; fresh proof required.
Evidence, failure review, commands: `docs/NATIVE_BATCH_OPTIMIZATION.md`.
Final CPU786 passed/54.41s; Python/PowerShell syntax and diff checks passed.
Audit `results/20260906_native_batch_final_audit.json`:219 frozen assets/4 game files
unchanged, game closed, no specified4101/41/6008 events in last hour.

Historical game checkpoint (before the input-projection fix and batch optimization):

2026-09-06 REAL GOWR SDR SINGLE-FRAME ROCM WRITEBACK OBSERVED; NOT REALTIME/QUALITY PASS.
ReShade v6 preview processed actual gameplay at2342x1382 R10G10B10A2_UNORM/SDR.
Session `results/20260906_gowr_native_reshade_one_v2`: sequence1 written at present24463,
then matching ReShade-present event and saved display PNG; original game rendering resumed.
Native71-block single-color candidate uses frozen original weights and GPU-only pixel transport.
Worker frame26,080.84ms, peakallocated3,240,176,640/reserved4,370,464,768 bytes;
normal closed acknowledgement, released allocated/reserved0. Not pure GPU event time/game FPS.
Screenshot shows visible dark rectangular artifacts: quality FAIL, cause NOT yet established.
No same-frame RTX comparison or temporal/HDR/HUD/4K60 acceptance. Do not expand game loop yet.
Independent same-size10-bit network display comparison passed all pixels, D3D12 errors0;
separate quick10-bit transport comparison failed full-frame due to ReShade startup overlay,
retained as NOT_ACCEPTED. Prior8-bit proof alone was not used to authorize10-bit game writeback.
Details, screenshot, restoration status and next gates: `docs/NATIVE_RESHADE_PREVIEW.md`.
HDR restored (user confirmation + SetColorSpace1(12) log), game exited normally;
v1/v2 additive test files recovered, originals unchanged. Final audit219 assets/4 game files
unchanged, no specified4101/41/6008 events in last hour, no game/worker remaining.
Audit `results/20260906_reshade_game_one_final_audit.json`; CPU778 passed/74.64s.

Historical checkpoint (superseded by the single-frame game evidence above):

2026-09-06 RESHADE NATIVE GPU PREVIEW PASSED IN STANDALONE; GAME HDR GATE ACTIVE.
User-approved ReShade-first preview route implemented, no speed/weight optimization.
Native whole single-color model + cross-process GPU shared buffers + GPU format
conversion + same effect target writeback passed640x360 x12,1280x720 x1 and
2342x1382 x1. All audited display buffers exactly match encoded network output;
normal worker teardown/original path restoration; D3D12 errors0.
2342x1382 measured25.4s cold frame, peakallocated3.24GB; emphatically NOT realtime.
Actual GoWR PID19280 has loaded ReShade/addon, but reports HDR10/format24;
native network NOT ARMED/NOT WRITTEN in game. Requested temporary SDR scene.
Game originals preserved; reversible additive deployment recorded.
Details, commands and limitations: `docs/NATIVE_RESHADE_PREVIEW.md`.
CPU regression776 passed/72.98s.

2026-09-06 CROSS-PROCESS GPU LOOP PASSED; REAL GAME IN-LIST CONSUMER WORK REMAINS.
User priority: game integration BEFORE speed optimization; no speed/weight changes this turn.
Pure D3D12 producer (HIP not initialized) and separate original-weight ROCm worker
passed640x360 twelve alternating frames; metadata-only IPC, GPU shared heaps/fences,
exact per-frame consumer output, both normal exits and zero released allocations.
Resolved two pre-submit identity rejections: Windows venv launcher/worker PID and
two same-name DXGI adapters. OS child verification and exact LUID binding retained.
Game observation only: scene window2 has2 Draw +33 DrawIndexed AFTER the FFX output
boundary but BEFORE Close in each of4 observed lists. Shader target reads/HUD identity
not yet resolved; list-end insertion is not justified. Native game writeback NOT enabled.
Need state-preserving in-list split, actual consumer dependency, then validated bridge
attachment/size/color checks. Did NOT reuse failed list-end or old pre-Present tile path.
Observer stopped; game left on original output after detected manual window input.
Final regression771 passed/70.17s; diff whitespace check passed. No worker remains.
Final audit: `results/20260906_native_game_process_final_audit.json`;
219 frozen assets/4 game hashes unchanged; no specified system event in last hour.
Details/evidence/commands: `docs/NATIVE_GAME_PROCESS_BRIDGE.md`.

2026-09-06 NATIVE STANDALONE D3D12 TEXTURE LOOP PASSED; NOT GAME INTEGRATION.
Private non-pickle model package and resident whole-frame ROCm executor added.
D3D12 texture -> shared GPU buffer -> native71-block model -> shared GPU buffer ->
D3D12 output texture passed at128x128 and640x360; exact same-candidate output.
641x129 pitched transport also passed. Same resources processed12 alternating inputs
with correct per-frame consumer bytes, distinct/stable outputs and normal teardown.
640x360 warm host network submit/wait median1706.24ms; NOT game FPS or4K60.
No CPU neural fallback; CPU upload/readback are OFFLINE FIXTURE endpoints only.
This is a SAME-PROCESS standalone harness, NOT a cross-process game bridge.
Still missing actual pre-HUD consumer hook, formal color/HDR, larger size/performance
validation and game stability. No game launch, weight training, TDR changes or deletes.
Evidence/scope/commands: `docs/NATIVE_GPU_TEXTURE_BRIDGE.md`.
Gate: `results/20260906_native_bridge_assessment.json`.
Audit: `results/20260906_native_bridge_final_audit.json` (219 baseline/4 game hashes unchanged).
Final regression750 passed/51.96s; new Python scripts compile; diff whitespace check passed.

2026-09-06 MONITOR REVIEW COMPLETED; 100 STATIC FULL-CHAIN GPU REPEATS PASSED.
Supersedes the GPU-review hold below for reviewed, bounded offline development only.
After removing periodic concurrent Python stack walking, staged128 single,640 single,
640x12 and640x100 tests all exited normally.100 identical input/seed outputs match;
all71 blocks execute on gfx1201 without RTX intermediate inputs or CPU neural fallback.
100-run process225.718s; warm host staged median2126.73ms (NOT optimized game FPS).
Frame-end allocation spread56832 bytes; allocated/reserved counters zero after release.
Source snapshot unchanged during100-run probe. Root cause NOT proven; historical
access violation and C256 shutdown hang remain recorded, no automatic failure retry.
No gameplay, temporal/HDR acceptance or4K60. No new training/game start/TDR changes.
Assessment: `results/20260906_rocm_whole640_hundred_assessment.json`.
Review and commands: `docs/ROCM_TRACEBACK_REVIEW.md`.
Regression720 passed/54.37s. Final audit219 baseline assets/4 game hashes unchanged:
`results/20260906_monitorv2_hundred_final_audit.json`; no validator process remained.

Historical checkpoint (superseded by reviewed tests above):

2026-09-06 WHOLE SINGLE-COLOR CHAIN CONNECTED; GPU REPEATS HELD AFTER CRASH.
Supersedes older statements that encoder/bottleneck/decoder require captured skips.
All71 blocks0..70 now execute from actual RGBA16F through original-weight tensor
preprocessing, encoder, ViT, decoder and legacy SDR diagnostic head. ALL skips are
generated internally, no RTX activations/PTX/DLL/ZLUDA or CPU neural fallback in
the GPU forward pass.144 active parameter records,8 ABI placeholders,1 inactive
temporal blend scalar remain explicitly distinct. NOT full temporal/HDR contract.
CPU128x128 synthetic,640x360 GAME CROP and641x361 synthetic completed, all finite.
RX9070XT single runs128 and640 passed with normal exit/zero released allocations.
128x128 repeated12 identical-input outputs match;640 single diagnostic~3986ms,
peak tensor allocation596218368 bytes. NOT real-time or4K60. Color/HDR NOT accepted.
640 repeated12 FAILED at30s with0xC0000005, python312!PyCode_Addr2Line during timed
traceback collection.11 completed-frame phase markers are NOT repeat hash evidence.
No further GPU test was run. Removed periodic concurrent stack walking as a monitor
mitigation, kept fatal trace/PID/phases/host timeout, added per-frame hash journal;
GPU mitigation not retested, root cause unresolved. Review this hold before any GPU
continuation; no auto retry/TDR edits. Game stayed closed; no training.
Current details/commands: `docs/NATIVE_SINGLE_COLOR_CHAIN.md`.
Coverage: `results/20260906_native_whole_frame_reconstruction_v2/model_description.json`.
Safety/evidence review: `results/20260906_native_whole_frame_review_v2.json`.
Regression709 passed/50.45s; syntax checks passed.219 baseline assets/4 game hashes
unchanged in `results/20260906_whole_frame_final_audit.json`; no training/game start.

2026-09-06 SCALE/DECODER CONTINUATION (supersedes corresponding missing edges below):
Fixed pair-interleaved outview storage; captured CPU C128/C256 downsample boundary
NRMSE now4.13%/6.36%. Implemented all four original decoder upsample/skip-fusion
families and22 consecutive blocks48..69. Actual-feature CPU and bounded RX9070XT
ROCm execution completed, without interior RTX substitution or CPU neural fallback.
Entry and four encoder skips remain REAL CAPTURED EXTERNAL INPUTS: NOT a full RGB
network, not native encoder execution. Final GPU feature NRMSE6.55% vsRTX and4.69%
vs same CPU candidate; finite, NOT byte-identical or image-quality accepted.
Single diagnostic stage/wait sum1886.51ms, peak tensor allocation156802560 bytes;
NOT optimized full-frame timing or game FPS. C256/C32 upsample gradient probes and
the22-block GPU process exited normally, allocator/reserved bytes zero after release.
No training, game start or original-weight changes.4K60 remains NOT achieved.
Details/commands: `docs/NATIVE_REALTIME_MILESTONE.md` (scale continuation section).
Evidence: `results/20260906_rocm_decoder_pyramid_one/manifest.json`;
record coverage: `results/20260906_native_scale_reconstruction/model_description.json`.
Regression699 passed/50.91s; syntax checks passed. Final audit:
`results/20260906_native_scale_final_audit.json` confirms219 baseline assets and4 game
program hashes unchanged, closed game, no queried last-hour device/power events.
No remaining native/ROCm validation process was found.

2026-09-06 USER TARGET UPDATE: first realtime game milestone is3840x2160/60 FPS;
NVIDIA-quality matching deferred, but native whole-frame, SDR/HDR, live pre-HUD
output, GPU-only image path and stability remain required. NOT achieved.
Implemented explicit native_fp16 whole-GEMM policy and shape-specific ROCm graph
replay alongside untouched default recovered_k32 arithmetic. Original weights
remain immutable. Eight-block96-token ViT test:12 identical-input outputs stable;
warm host medians59.48ms(block waits),18.45ms(frame wait),6.55ms(graph replay).
These are SMALL SUBGRAPH timings, NOT4K image or game FPS. Graph/eager FP16 outputs
match; RTX feature NRMSE48.28% remains diagnostic. No training or game start.
C128 downsample and C512 native_fp16 forward/selected-gradient probes also passed.
Regression685 passed/42.64s. Final audit confirms immutable baseline/game hashes,
closed game and no queried last-hour device/power events; no GPU probe left running.
Game-only assessment correctly rejects this subgraph evidence:
`results/20260906_realtime_gate_current.json`. Plan/commands and remaining full-graph
and pre-HUD interop gaps: `docs/NATIVE_REALTIME_MILESTONE.md`.

2026-09-06 latest GPU continuation (supersedes earlier CPU-only evidence below):
Original block40 split512, block31 ViT1024 and block14 downsample128 candidates
passed bounded ROCm forward + input/selected-weight backward on RX9070XT.
All eight original ViT blocks31..38 then ran consecutively on GPU at96tokens;
no internal RTX substitution, byte-identical to the same CPU candidate chain.
RTX feature NRMSE remains47.77%; this is NOT NVIDIA image-quality acceptance.
Host submission/per-block-wait time~531ms, peak tensor allocation296718848 bytes.
GPU event timing produced negative durations: INVALID, do not use original event
sum for throughput/FPS. See the immutable-report correction in
`results/20260906_rocm_gpu_execution_review.json`.
Independent supervisor verified normal exit and zero allocator bytes after explicit
BLAS workspace release. The previous C256 shutdown-hang root cause is still unknown;
these probes do not establish long-run stability or permit automatic hang retries.
No game run, fine-tuning, CPU neural fallback or original-weight modification.
Regression:666 tests passed in43.57s;38 relevant CPU tests also passed in the
independent ROCm environment. Final audit confirms219 baseline assets and4 game
program hashes unchanged, game closed and no queried last-hour reset/power events.
Updated per-record evidence: `results/20260906_native_tensor_reconstruction/model_description.json`.
Continue completing pre/input/noise, unresolved pool/outview layouts, decoder
upsample/fusion and dynamic whole-frame graph BEFORE optional fine-tuning.

2026-09-05 latest continuation: COMPLETE-TENSOR-GRAPH-FIRST remains the priority.
Added original four-record split-Swin512, eight-block ViT1024 bottleneck,
streaming global attention, learned downsample candidates and bottleneck boundary
projections. New modules were exercised ONLY as CPU references while the prior
ROCm teardown safety hold remains unresolved. No new GPU run or training occurred.
The eight original ViT blocks ran consecutively without RTX internal substitution;
NRMSE 47.77% is recorded for later full-graph structural/quality review, not used
to stop reconstruction. Downsample target layouts and decoder spatial fusion are
still unresolved. The 153-record description explicitly distinguishes implemented
tensors, CPU-only evidence, unused ABI placeholders and missing boundaries:
`results/20260905_native_tensor_reconstruction/model_description.json`.
Next build work: original pre/input/noise path, four upsample-fusion families,
pool/outview adapters and whole-frame shape/resource wiring. No fine-tuning yet.

2026-09-05 ROCm native route implementation is underway (partial, NOT complete).
Latest continuation: grouped FFN + multi-head attention + packed-window adapters
now run original C64/C128/C256 Swin weights natively. Eighteen encoder/decoder
isolated RTX comparisons meet the existing operator tolerance (NRMSE 4.18-5.61%).
A six-block GPU-resident C256 chain reaches 10.89% NRMSE and FAILS that gate;
isolated passes do not authorize promotion. Its numerical report was saved but
the Python process failed to exit normally and was explicitly terminated after
inspection. No further GPU experiment was run; runtime teardown remains open.
See ROCM_NATIVE_IMPLEMENTATION.md for new evidence and the safety hold.
The isolated Python 3.12/ROCm environment passes gfx1201 tensor and backward probes.
Original-weight Swin 1h/32 has four isolated RTX controls; the output head has a
ROCm port with matching activation fusion. Neither replaces the full translated
network yet. Full-frame SDR/HDR quality, dynamic graph, teacher sequences and
game deployment remain open. See `ROCM_NATIVE_IMPLEMENTATION.md` for evidence,
commands, the unvalidated RTX5070 candidate package, and remaining gates.
No game run or optimizer update was performed for this implementation batch.

Latest visible dynamic bring-up (2026-09-05, E-174): original-weight inference
now reaches the actual GoWR display through a separate pre-Present debug path.
PID 4592 processed 12 distinct current 2342x1382 SDR buffers, 16 full 156-slot
network tiles per frame, on RX 9070 XT. Present accepted every processed frame.
Independent packed-pixel conversion and post-write readback are exact; original
alpha is preserved. Screen captures after completed frames 3 and 11 visibly
show the network's tile seams; automatic stop removes them and restores normal
rendering. Analysis: `results/20260905_present_game_analysis_v3/summary.json`;
screen/exit evidence: `results/20260905_192944_171_gowr_resident_present/`.
Commands: `docs/RESIDENT_PRESENT_DEBUG.md`.

This closes ONLY the bounded low-FPS dynamic original-weight/display experiment.
Whole-surface network inference is 5,081.9–5,201.2 ms/frame, NOT playable realtime.
It processes SDR display-referred pixels after tonemapping/HUD, not the final
NR color/temporal game contract. Independent tiles have visible brightness seams.
No same-input RTX quality, arbitrary-size attention, HDR support, smooth FPS or
long-run stability acceptance is claimed. The game was normally exited after
automatic 12-frame stop; no new System 4101/41/6008 events or game-image changes
were found. `hdr_restoration.json` and `hdr_restored.png` record restoration of
game HDR ON, HDR brightness 80, UI brightness 50, through a separate process
without the resident network. No Windows display setting was changed.

Next: profile dominant graph families with this executable evidence; prioritize
native/tensor-efficient replacements and genuine full-frame/context handling.
Then bind pre-HUD/color/temporal semantics and same-input teacher targets. Do not
resume per-instruction bitwise equality as a prerequisite, or present this debug
bridge as a finished DLSS5 game mod.

## Earlier resident experiments

2026-09-05 resident bring-up (E-172/E-173): the original-weight 156-slot graph
is now persistent on RX 9070 XT. Six alternating-input controls preserve E-170
reference bytes, sync/async equality and immutable weights. Normal 640x360
inference is about 308–311 host ms; no per-slot synchronization or disk output.

The live FFX producer-list-end loop completed 12 distinct 2342x1317 input
frames in EACH of two runs, with 16 complete tiles/frame and exact worker-to-
game texture readback. Whole-surface inference is roughly 5.1–5.3 seconds/frame.
However final screen acceptance FAILED: the obvious tile artifacts in raw
network output were absent from screenshots at completed frame 5 and frame 10.
End-of-producer-list replacement can be too late for in-list consumers. These
results are texture-loop evidence, not successful visible original-network
integration. See `results/20260905_resident_game_presentcheck_analysis/summary.json`
and `docs/RESIDENT_GAME_BRINGUP.md`. Both runs stopped after 12 frames, normal
game exit succeeded, game image hashes stayed unchanged and no new System
4101/41/6008 event was found.

Follow-up development: a separate current-backbuffer / pre-Present bridge,
described in `RESIDENT_PRESENT_DEBUG.md`. Its WARP/AMD current-buffer, Present,
Present1, stop, resize and four IPC-failure controls pass. Its first live game
activation was safely blocked at zero frames by an unsupported color space;
the cause was game HDR ON (PQ/BT.2020). E-174 above subsequently passes the SDR
bounded visual bring-up after temporarily switching game HDR OFF.

The user-selected immediate priority is continuous original-weight game
bring-up first, accepting a low-FPS debug version; native family optimization
and RTX teacher tuning follow. This does NOT waive full-frame, arbitrary-size,
color, temporal, stability or performance gates for the final project release.

## Prior milestones

Latest game preview milestone (2026-09-05, E-171): PID 9524 displays the E-170
original-weight network's PRECOMPUTED 640x360 output inside the live GoWR FFX
output surface. A magenta frame and explicit STATIC / NOT LIVE INFERENCE label
surround the image. Switching to the matching original-input static image and
stopping the overlay were both visually verified. The first window recorded
6,682 region copies before manual stop.
An additional 60-second window recorded 5,853 copies then expired automatically;
`automatic_timeout_audit.json` and `preview_timeout.png` verify overlay removal.
The game remains responsive; audited game binaries are unchanged and no new
System 4101/41/6008 events were found.
Session: `results/20260905_175035_151_gowr_capture_session/`;
commands: `docs/STATIC_GAME_PREVIEW.md`.

This is static-result display transport, NOT per-frame original-network
inference. The image does not track the camera or current frame. It neither
closes realtime performance nor DLSS5 image/temporal quality or arbitrary-size
network gates. The existing FSR3 Quality setting and disabled frame generation
were restored after a temporary provider switch to observe a fresh FSR context.

Latest full-network milestone (2026-09-05, E-170): the original-weight,
156-slot translated graph now processes a byte-preserving 640x360 crop from
the real GoWR FFX output on RX 9070 XT. Two repeats are exact. Replacing the
entire captured initial activation arena with zero leaves both the complete
29,773,824-byte pre-head arena and final output byte-identical. Keeping the
postblock base fixed while zeroing only the neural input changes the pre-head
arena and 656,711 final RGB components. This is evidence of actual image-driven
inference, not merely base-image composition or captured-activation replay.
See `results/20260905_real_color_probe/controls_summary.json` and
`docs/REAL_COLOR_REPLAY.md` for commands and scope.

This is NOT a DLSS5-quality or game-runtime pass. The current frame has no
matched RTX output, no verified NR color-space/exposure contract, and no motion
or temporal state binding. The recovered output clamps HDR input to [0,1].
Launch-plus-per-slot-sync time was 406.8-428.2 ms for the first two real-color
runs (not a resident GPU benchmark). Native D3D12 head output on the same
AMD-produced activation differs from the translated head: RGB MAE 0.000443146,
maximum 0.0883789; it is not accepted as an equivalent replacement. It also
took 36.6-38.9 GPU ms, versus about 8 ms launch+sync for translated slot 154;
these timing scopes differ but give no evidence of a native speedup yet.

Next priority: establish same-input teacher/color/temporal semantics and
resident GPU timing before broad native migration. Optimize measured operator
families, not each slot independently. The arbitrary-resolution pack/identity/
unpack tests below prove transport only; alignment, halos, overlap, neural tile
seams, dynamic shapes and multi-frame quality remain unverified. No new game
injection or game launch was performed in this experiment after normal exit.

Latest live milestone (2026-09-05, PID 21584): an externally weighted,
dynamic-resolution residual CNN now executes inside a user-confirmed playable
GoWR scene. Session `results/20260905_163007_931_gowr_capture_session/`
processes one 2342x1317 RGBA16F frame at the proven FFX output boundary. The
original submission contains 36 command lists; the observer inserts one private
network list after index 34 and preserves the final downstream consumer list.
Before/after SHA256 is
`AA8E0F16E765564D5EBB2489653E0EDB7CA8C68379BEC1D308E3D631B059ED86`
and `E834265FAD80C10C651A1703B6048B55787EEE05F5F1D4E14656C29F03DEF29F`.
3,058,969 pixels / 8,730,452 components change; alpha is bitwise exact, all
values are finite, the game remains responsive, and no new GPU-reset or
unexpected-restart event is present. Report status:
`GAME_GPU_WEIGHTED_NETWORK_PASS`.

The runtime is a dynamic 3x3 RGB -> 8 ReLU feature -> RGB residual CNN. Its
256-float constant buffer is parsed from an independent sparse weight file, so
future weights no longer require shader or hook changes. The current calibration
weights exactly reproduce the E-166 safe five-tap residual output on WARP, RX
9070 XT, FSR3 and FSR4 controls. They are deliberately labeled
`external_weights=true`, `reconstructed_network_output=true`,
`trained_weights=false`, and `dlss_nr_verified=false`. Transport and weighted
inference are complete; the remaining quality gate is acquiring paired teacher
data and training/evaluating scene-responsive weights, then validating temporal
stability and performance across real gameplay.

Full-network reconstruction resumed after E-168. The current external game
network accepts only 256 floats and cannot consume the original 147,719,680-byte
model arena. The fixed reference graph contains 156 slots and a 29,773,824-byte
activation arena. Its first slot is now implemented as a native D3D12 dispatch:
`results/20260905_185000_full_graph_slot0_d3d12/warp_result/` clears exactly the
captured 110,592-byte synchronization region and preserves all remaining arena
bytes, with zero WARP debug errors/warnings. This is the first prefix migration
step, not a complete-network or game-runtime pass. At that milestone GPU validation
was deferred pending normal game exit; slot 1 and the other
upstream families remain unported to the resident D3D12 graph.

Arbitrary output resolution has an experimental transport planner, not a
verified full-network contract. `tools/full_graph_d3d12/resolution_plan.h` provides an
8-pixel-aligned native-dynamic candidate plus a bounded tiled candidate for
operator families that still require the captured 640x384 working shape. The
fallback uses overlapping 640x360 output tiles, deterministic midpoint crops,
and at most 16 resident tiles per batch. CPU validation covers 524 plans and
26,672 tiles through 7680x4320. The WARP D3D12 pack -> identity-network ->
unpack test is half-bit exact at 320x180, 641x361, 1920x1080, the live-game
2342x1317 size, and 3840x2160; the 4K case spans 49 tiles / 4 batches. It reports
zero component mismatches and zero D3D12 errors/warnings in
`results/20260905_190000_arbitrary_resolution_plan/`. The identity stage is a
boundary test only: slots 1-154 still have to replace it before this becomes a
complete DLSS network.

Latest live milestone (2026-09-05, PID 9372): dynamic-resolution GPU output
replacement now passes in a user-confirmed playable GoWR scene. Session
`results/20260905_160743_488_gowr_capture_session/` inserts one private compute
list after batch index 33 of 36 and before the two original downstream consumer
lists. The 2342x1317 RGBA16F before/after artifacts are each 24,675,312 bytes;
SHA256 changes from
`C2C59E56E3F9DE4488D827685D058564E66F89944948520EB8D8A085C0B8858C`
to `0BB5E0F6C55A76CFF6BB0A0811815C8210828A233DC495A66AF95170ABCA9B10`.
3,060,032 pixels / 8,727,182 components change, alpha remains bitwise exact,
all half values are finite, the game remains responsive and no new display-reset
event is present. Report: `output_filter_analysis.json`, status
`GAME_GPU_RESIDUAL_FILTER_PASS`.

The filter is an intentionally simple dynamic five-tap residual plumbing proof,
not a trained network: `trained_weights=false`,
`reconstructed_network_output=false`, and `dlss_nr_verified=false`. It closes
the real-game dynamic compute/list-insertion/write-back transport gate. The next
quality gate is a scene-responsive reconstructed/trained network whose inputs
are produced for the live frame; the existing captured 640x360 output head
cannot be relabeled as such because its upstream activations are not live.

Two earlier experimental runs at 15:44 and 15:48 caused system hangs followed
by Kernel-Power 41 restarts. Their logs identify the deterministic cause as
`resource deadlock would occur`: a private command list was constructed while
the global ResourceBarrier hook held the session mutex, and its Close callback
re-entered that mutex. The corrected implementation constructs/records outside
the lock and uses an explicit internal-command hook-bypass scope. The unsafe
unsignaled queue-wait fault injection was removed from this path. Before live
use, WARP primitive and full-session tests passed with two suffix lists and zero
D3D12 warnings, single-frame FSR3/FSR4 hardware controls passed, all 18 staged
runs passed, and the full Python regression reached 485 passed.

Latest live result (2026-09-05, playable scene): one real GoWR FFX output frame
passes the complete output-only route in PID 2920. Result:
`results/20260905_144814_826_gowr_capture_session/`. The 2342x1317 RGBA16F
artifact is 24,675,312 bytes, finite, SHA256
`79DBAF343981F9695365AD690D16E3B1309E76D6F31211C9EC1F6BD20DBD8603`.
The canonical list identity is submitted exactly once, the recorded generation
and observed Close match, and the collector-owned fence completes. No failure.

Content validation passes: sampled RGB contains 7,165/7,391/7,161 distinct half
values, range 0..5.78125, and meaningful per-channel variance. The tone-mapped
preview visibly contains the playable animal-pen scene with Atreus, Angrboda,
animals, fence and vegetation. `scene_content_verified=true` and
`dataset_usable=true`. All audited game files remain unchanged. This establishes
the real game output transport boundary only: `output_replacement_performed=false`,
`dlss_nr_verified=false`, `teacher_verified=false`.

Latest live write-back result (2026-09-05, PID 23500): the same proven playable-
scene boundary completes an exact output -> private texture -> output GPU
roundtrip. `results/20260905_151638_018_gowr_capture_session/` contains a
2342x1317 RGBA16F artifact (24,675,312 finite bytes), SHA256
`2AD69C0E4F8DC93C11E4A0E75FBE110FE320A64B37DF80F3575B20D8C5E632E7`.
The canonical list is submitted once, the owned fence completes, sampled RGB is
nonconstant, and all audited game files remain exact. Report status is
`GAME_OUTPUT_ROUNDTRIP_PASS`; write-back and game-frame verification are true.
This writes the original bytes back and therefore remains false for replacement
pixels and DLSS-NR.

The next supplied-pixel stage is independently validated at
`results/20260905_152548_455_ffx_supplied_patch_live_stage/`. Its FSR3 and FSR4
controls replace exactly the requested 8x8 corner, preserve every pixel outside
that rectangle byte-for-byte, block early polling and retain baseline equality
before the patch. All 478 Python tests pass. The stage is not loaded in PID
23500; normal exit is required before the one-frame live patch test.

Latest live replacement result (2026-09-05, PID 29836): the supplied-pixel
stage passes in a user-confirmed playable scene. Session
`results/20260905_152923_020_gowr_capture_session/` records exact before/after
2342x1317 RGBA16F outputs. Exactly the requested 64 pixels change, all
3,084,350 pixels outside the 8x8 rectangle remain byte-identical, and the
downstream-visible output equals the post-patch readback. Fence, canonical-list
identity and game-file checks pass. This closes externally supplied output
replacement transport. The supplied content is a diagnostic constant, not a
reconstructed network result; network execution and temporal validation remain
open.

Latest live result (2026-09-05, PID 31864): canonical COM submission handling
passes in the real game. One 2342x1317 RGBA16F FFX output was recorded, matched
through a different raw interface pointer, submitted exactly once and read only
after its fence. Artifact: 24,675,312 bytes, SHA256
`FC17DAE7604FC5286AD2802381F01CC7461C561CA19CC21E2D4F824A73E9E5CA` in
`results/20260905_143913_151_gowr_capture_session/output_boundary/`.

Transport status is `GAME_OUTPUT_CAPTURE_PASS`, but this launch-time frame is
near-uniform: each sampled channel has only eight values between 10.7578125 and
10.8125. Therefore `scene_content_verified=false` and `dataset_usable=false`.
It proves the game integration/fence path, not a usable gameplay image, NR output
or teacher. All four audited game files still match. Full regression is now 456
passed. Next: fresh process, wait for the user to confirm a playable scene, then
use the same one-shot stage; do not re-arm/stack the completed current session.

Latest live result (2026-09-05, PID 30212): the first bounded game output-copy
attempt correctly fail-stopped before writing an artifact. The game D3D12 layer
exposed different interface pointers at boundary recording and queue submission;
the old code required the recording pointer's map entry to be closed before it
compared canonical COM identity. Session:
`results/20260905_143334_141_gowr_capture_session/`. The game remained responsive,
no output file was accepted, and the uncertain job is retained until exit.

The fix is staged at `results/20260905_143600_233_ffx_canonical_output_stage/`.
It associates the submitted list by canonical IUnknown identity first, then
requires exactly one submitted identity, unchanged recorded generation, and at
least one closed same-identity/same-generation interface. Raw pointer equality
is logged, not required. Both provider control suites and six reports pass;
full regression is 455 passed in 61.66s. This stage is NOT loaded in PID 30212.
Normal game exit is required before a fresh bootstrap; never retry/stack the
failed session. Real game-frame and NR proof remain false.

Latest implementation milestone (2026-09-05, 14:27 stage): a distinct live
`output-arm` / `output-poll` command now copies exactly one existing FFX output
texture. It does not arm the multi-input collector and cannot replace output.
The same version-2 command route passes FSR3/FSR4 native controls, including
byte-identical on/off output, a deliberately blocked early poll, exact
record/submit/fence lifecycle, and false game provenance in the isolated host.
All six control reports pass; full Python regression is 455 passed in 62.73s.
Stage: `results/20260905_142713_314_ffx_live_output_stage/`.

The preceding diagnostic stage was loaded in GoWR PID 31044 and observed two
playable-scene windows: eight boundary returns, seven supported and one startup
sample rejected solely because reset was unobserved. The second window is 4/4
supported. Every callback has exactly one complete output transition, no alias/
unknown barrier, and returns from the original callback. The session is stopped.
The user reported exit, but PID 31044 remains responsive with its game window,
so the new DLL has not been hot-loaded or stacked. A genuine normal process exit
and fresh bootstrap are still required before the one-frame game test. No game
frame or NR result is claimed yet. Current commands/details:
`tools/ffx_observer/BOUNDARY_CALLBACK.md`.

Latest implementation milestone (2026-09-05, 14:08 stage): whole-barrier
post-forward callback diagnostics and the actual isolated hook-copy path pass
for FSR3/FSR4. Each provider's hook writes one 1,843,200-byte output identical to
baseline; downstream outputs remain unchanged and a gated queue blocks early
polling. Ten normal native runs have zero D3D12 errors/warnings. Two separate
negative runs each deliberately produce exactly two checked inefficiency warnings
and reject multi/split output transitions. Six control reports pass; 443 Python
tests pass. Stage: `results/20260905_140838_246_ffx_boundary_hook_stage/`.

NOT yet loaded into GoWR PID 30652: normal exit requested before a fresh
diagnostic attach. Live Attach rejects the isolated copy configuration version
3, so no game copy command is enabled. Game frame/NR proof remain false. Next:
verify the actual game's entire callback AFTER forwarding and its lifecycle,
then finish ownership/adversarial checks before a bounded output-only copy.
Current commands/details: `tools/ffx_observer/BOUNDARY_CALLBACK.md`.

Latest implementation milestone (2026-09-05): an output-only copy primitive at
the explicit POST-FFX game transition passes isolated validation. Actual-size
synthetic tests (2342x1317 plus 640x360) preserve 53,037,024 raw bytes and the
downstream result exactly, with zero D3D12 errors/warnings. Independent real
FSR3/FSR4 boundary-copy vs baseline controls pass for four 640x360 frames each.
Seven of eight detailed existing game samples match the candidate structure;
the first lacks reset evidence. This is NOT live-hook/capture/NR validation.

The next integration gap is precise: forward and validate the WHOLE original
barrier callback, then prove list-generation/submission/fence ownership. Current
logs are pre-forward and cannot exclude a later barrier in the same callback.
No modified DLL was loaded, no game texture copied, and existing session gates
were not relaxed. PID 30652 stays stopped with six windows left. Full regression:
426 passed. See `tools/ffx_observer/OUTPUT_BOUNDARY.md` for results and commands.

Latest live milestone (2026-09-05, process started 13:38:00): resource identity
diagnostics in GoWR PID 30652 completed two playable-scene windows and stopped
(6/8 remain). `results/20260905_133833_941_gowr_capture_session/` contains 128
paired FFX calls and eight complete detailed resource snapshots. Each detailed
call has two UAV barriers matching the exact game output pointer AND COM identity,
zero identity-only matches and unchanged descriptor fields. No watched state
transitions occur inside those calls. This corrects the earlier overbroad
zero-matching-barriers wording; it does not certify safe texture copying.

The 2,048-event capped state trace has 50 complete calls / 98 complete batches;
post-call output transitions still start at 64, not declared UAV state 8. Context
creation is still unobserved. Zero ready/failure/copy events, no arm, unchanged
game files. Private FSR4 4.1.1 * route reproduced. Both independent providers'
five control reports pass; 403 Python tests pass in 46.49 seconds. Full-network
NR quality and game_runtime_ready remain false. Next target: missing state/
ownership coverage at the known game output boundary and context lifetime.
Exact stage, limits, results and run commands: `tools/ffx_observer/RESOURCE_IDENTITY.md`.

Previous live milestone (2026-09-05, process started 13:20:38): command-path
diagnostics are loaded in NEW GoWR PID 24336 from the independently validated
`results/20260905_131819_826_ffx_command_path_stage/` (not build/).
`results/20260905_132114_421_gowr_capture_session/` records 64 playable-scene FFX
calls, each with 28 Dispatch callbacks and 16 legacy-barrier callbacks, no
observed enhanced/indirect callbacks or other-interface-pointer calls. All statuses
zero. The capped state trace contains 25 complete calls and 48 complete batches;
no inside-call state transitions match watched game texture pointers. UAV
resource identities were not recorded by that revision. This
narrows the next work to resource identity/output-boundary coverage, not an
assumption that FSR does no work. No GPU completion/capture inference from counts.

Read-only exact-image private-context audit, repeated across both processes,
finds `4.1.1 *` / ID 17700776142811697153 and dispatch callback in the AMD driver
`amdxcffx64.dll`. These match the independent FSR4 control despite the earlier
FSR3.1 menu label. Public-query provenance and the live session's context registry
remain unverified; existing FSR4 actual-size repeatability failure is NOT waived.
One observation window stopped, seven remain. Zero ready/failure/copy events,
no arm, no game-file replacement. 390 Python tests pass in 46.81 seconds; both
providers' independent on/off, window, enhanced-hook and capture controls pass.
Full NR quality/game_runtime_ready remain false. See
`tools/ffx_observer/COMMAND_PATH.md` for results, exact hashes and commands.

Previous live milestone (2026-09-05, process started 12:50:20): after the user's
normal exit, deferred observation is loaded in NEW GoWR PID 23192 from
`results/20260905_124552_868_ffx_observe_window_stage/ffx_capture_session.dll`.
Two explicitly triggered playable-scene windows succeeded without another
restart. `results/20260905_125046_331_gowr_capture_session/` contains 128
candidates, 2,048 capped trace events, 50 complete FFX call windows and 96
complete submission batches (1 incomplete, 0 invalid). Zero ready candidates,
failures or input copies; no arm was sent. Observation is stopped (2/8 used).

Same-batch prefix analysis associates 48 calls. Most last observed states are
64, versus declared inputs 192/output 8; all 50 observed first post-call output
transitions start at 64. The first warm-up sample has no observed FFX-list reset
and contains depth transition-chain gaps; it is not a state proof. A read-only
entry audit finds exact disk/live equality for the first 32 bytes of four FFX
exports and confirms session barrier hooks on two sampled command lists. This
does not exclude internal/alternate command paths. Native binaries and game
files were not replaced. Context registry is still unknown in this session;
the bootstrap did observe this process's create metadata, but that observation
does not initialize the session's registry or identify the backend provider.
Next: resolve command-interface/state coverage, then bounded fenced input
capture. Do not waive state gates or conflate this with NR quality. No game
pixels, no output replacement; `game_runtime_ready=false`.
See `tools/ffx_observer/OBSERVATION_WINDOWS.md` for current evidence and commands.
Full regression after the new analysis/audit tools: 366 passed in 63.33 seconds.

Previous staged milestone (2026-09-05 12:46): deferred observation windows
and explicit submission batch IDs pass both providers' independent controls in
`results/20260905_124552_868_ffx_observe_window_stage/controls/`. Idle frames do
not record; two triggered windows associate depth/motion producer and FSR lists
in two-list batches, preserving all baseline bytes with zero D3D12 messages.
Capture gates stay conservative; armed captures reject observe/stop commands.
346 regression tests passed. At that milestone it was NOT loaded in PID 15544;
normal exit is needed before using it. Once loaded, up to 8 bounded observation
windows can be started without reloading the DLL. No new game pixels/NR pass.
See `tools/ffx_observer/OBSERVATION_WINDOWS.md` for commands and limits.

Latest restart (2026-09-05 12:30): user exited PID 10968; new game PID 15544
launched through the observer. The new state recorder is now loaded after exact
binary controls (`results/20260905_122935_190_ffx_dispatch/`). Live evidence
`results/20260905_123013_029_gowr_capture_session/boundary_analysis.json` contains
34 complete menu-stage call windows within a capped 1,024-event trace. Depth/
motion transitions are present on other lists; output's first observed post-call
transition starts from state 64, not declared UAV 8. Underlying backend/barrier
path remains unresolved; no state-gate waiver, no arm, no game pixel copies.
The existing save has loaded successfully. This process's context predates
attach and remains unknown; never reuse the previous PID's context epoch.
Next: scene-selectable observation-only retrigger plus command-interface and
cross-list ordering coverage, not another blind startup trace. See latest
section of `tools/ffx_observer/CAPTURE_SESSION.md`. Native NR readiness is false.
Regression after boundary-correlation analysis: 332 tests passed in 64.85 seconds.

Prior FSR3 game-input milestone (2026-09-05): game menu confirmed AMD FSR 3.1 /
Quality, frame generation off. Current capture-session observation now has a
verified context epoch (max render 1552x872, max upscale 2342x1317), but all
264 sampled candidates fail the resource-state gate; zero game copies.
Color/reactive/output last local states differ from declared states; depth and
motion states are unknown. Backend provider selection remains unverified.
Automatic session capture passes independent FSR3/FSR4 controls. A NEW bounded
state-transition recorder is staged separately and passes observe-only on/off
and capture-regression controls; it is not loaded in the game. Normal exit is
needed before replacing loaded lab DLLs. See
`tools/ffx_observer/CAPTURE_SESSION.md` for commands/evidence. No NR quality pass.
Full regression: 326 tests pass in 61.23 seconds. Audited game images and the
currently loaded session DLL's on-disk hash remain unchanged.

The following entries describe prior milestones; the latest context information
above supersedes their missing-context statements, not their quality failures.

Depth-plane collector adaptation (2026-09-05): explicit plane-0-only policy,
precise depth-view metadata checks and context-default dimension validation are
implemented. FSR3 at actual game sizes captures three synthetic frames with
143,046,480 exact raw bytes and unchanged original outputs/stencil. FSR3/FSR4
both pass a separate aligned-layout on/off test. An isolated distinct-depth/
stencil-state test passes with zero D3D12 errors. FSR4 at the current game sizes
still fails its uninstrumented repeatability gate near the bottom-right edge;
fresh contexts do not eliminate it. No game settings were changed or failure
waived. See `tools/ffx_observer/DEPTH_CAPTURE.md` for evidence and commands.
Current game resample confirms renderSize 1552x872 and raw upscaleSize [0,0].
Live context maximum and barrier/lifetime tracking still precede game capture.
No real game pixels have been copied; full NR quality/game_runtime_ready remain false.

Live game inspection (2026-09-05): same running GoWR PID 10968, no restart or
game-file deployment. `results/20260905_112424_948_gowr_live_inspection/` correlates
16/16 live FFX calls with one direct queue on adapter LUID 94973. Original return
statuses are all zero. The strict descriptor gate correctly FAILS: actual depth
is D32_FLOAT_S8X24_UINT, while the game declares depth-only FFX usage 4; the
pinned helper derives depth+stencil usage 36. No game GPU copies occurred.
Current sizes are 1552x872 inputs and 2342x1317 output, unlike earlier metadata.

`results/20260905_112352_927_ffx_dispatch/` proves inspector on/off output equality
and 4/4 queue matches for each real FSR3/FSR4 provider in isolated hosts.
`results/20260905_112656_305_ffx_depth_plane/` independently verifies patterned
depth AND stencil plane copies at three sizes, zero bad pixels and zero debug
errors (three intentional clear-value performance warnings). This explains a
capture-contract gap; it does not pass game resource state, pixels or NR quality.
At that milestone next: explicit depth-plane policy, current context/dimension tracking, per-plane
state/lifetime validation, then bounded fenced game collection. Full NR quality
and game_runtime_ready remain false. See `tools/ffx_observer/LIVE_INSPECTION.md`.

The previous log reader-sharing problem is fixed and tested. PID 17484 exited
normally; its header log parsed successfully. The replacement metadata run
`results/20260905_110646_754_gowr_ffx_observation/` has 9,997 decoded dispatches,
all status zero, saved in `results/20260905_110900_gowr_metadata_snapshot/`.
A default existing save was loaded with user approval; no new-game/manual-save
command was issued. Earlier no-game-launch statements below are historical.

Bounded FFX texture collection (2026-09-05): independent real-provider on/off
controls pass in `results/20260905_104954_587_ffx_dispatch/`. FSR3 and FSR4 each
capture three frames / 15 textures / 8,294,412 raw bytes with exact producer and
output agreement; all four original outputs remain byte-identical with capture
enabled. A deliberately gated GPU queue verifies no early mapping. Eight invalid
or duplicate descriptor cases, total-byte cap and frame-budget exhaustion pass;
zero pending jobs and zero D3D12 errors/warnings remain. Context/frame parameters
and padded copy layouts are stored. Regression: 257 tests pass in 45.72 seconds.
This is a caller-coordinated single-queue collector in a synthetic host, not a
game hook. Metadata observer remains unchanged; game bootstrap/queue interception,
game ABI/state validation and a user-selected scene remain unimplemented/unverified.
No game launch, installation mutation, real frame capture or NR quality pass.

FSR integration candidate (2026-09-05): the installed AMD DLL now passes real
context creation with explicit enumerated version overrides, GPU upscale dispatch
and fenced texture readback for both `4.1.1 *` and `3.1.0` on AMD device 0x7550.
Evidence: `results/20260905_103732_540_ffx_dispatch/` (four 320x180-to-640x360 reset
frames per provider). Every output channel starts as NaN and is overwritten with
finite data; A/A/B/A repeatability and input-response checks pass independently.
Both providers have zero D3D12 errors/warnings; the installed DLL hash is unchanged.
This closes the isolated minimal dispatch/readback gate, not game capture,
temporal history, full NR rendering, or the failed neural full-frame quality gate.
FSR is a candidate transport/input boundary, not a substitute NR implementation.
Regression after this probe: 242 Python tests pass in 49.26 seconds.

Game capture preparation (2026-09-05): bounded FFX metadata observation passes
real IAT forwarding and an AMD D3D12 copy harness, with unchanged 1,843,200-byte
output, zero debug errors and invalid-descriptor/budget checks. The installed
signed AMD DLL passes isolated version/jitter queries; available provider names
include `4.1.1 *`, `3.1.0`, `2.3.2`. This does not establish the actual game-context
provider or the game's full dispatch ABI. No game launch/deployment or real frame capture
has happened. See `tools/ffx_observer/README.md` for commands and limitations.
Regression after observer work: 232 Python tests pass (44.86 seconds); the
native observer and independent JSON/GPU artifact checks pass. Installed FFX
DLL remains SHA-256 `77809405A0FF464B63654F1264F0EC0FCF8F243DAC7C15B5F5C032615520D143`.

Selected game target: God of War Ragnarok at `C:\DATA\GAME\GODOFWAR`.
The read-only PE/configuration audit confirms D3D12/Streamline and FFX dispatch
candidate interfaces; current settings use FSR3 Quality at 4K on RX 9070 XT.
No NR-named module was found. No game launch, capture, deployment or settings/save
changes were performed. See `GOWR_TARGET.md`; ordinary game SR output is not an
NR teacher target, and the game choice does not close the neural quality gate.

Current reconstruction result (2026-09-05, E-142..E-144): automatic slot154
state/descriptor binding and actual queue-fence retirement pass 1,000 controlled
head calls, with zero output-byte mismatches, debug errors or remaining sessions.
This is not 1,000 game frames. The native 1h/32 chained family now reuses eight
head math shaders plus gather/scatter and passes four same-input cases at slots
3, 4, 151 and 152 (NRMSE 0.01769-0.02580). It does not read an RTX oracle at runtime.

**The full-frame gate still fails.** Against the canonical queue-complete frame1
reference `8ADB4DE9...E6E8C6`, AMD upstream plus the DXIL head gives RGB correlation
0.683468 / NRMSE 0.812474. Replacing the four chained modules with native DXIL
gives 0.638833 / 0.854350: a regression, retained only as an experimental option.
Neither per-operator passes nor a ~65 dB unit-range PSNR on this near-black frame
justify promotion. Older final metrics using reference `CD556...` are not the
same-frame comparison and must not be mixed into current claims.

O4/O5/O6 and S7/S8 remain open. Other upstream families still use ZLUDA in the
offline harness; native-family handoffs currently use CPU staging, not one GPU
queue. The release-mode scalar DXIL head alone takes about 50 ms. Real-scene
inputs/reference sequences, temporal/dynamic-resolution contracts, remaining
native families, resident integration and performance work are still required.
No new NVIDIA trace package, server experiment or game modification was made.
Final regression: 227 Python tests pass in 43.73 seconds; shared output-head
shaders, module-trace load/unload, D3D12 capture and explicit-host NVAPI recording
tests pass. `git diff --check` reports no whitespace errors.

Historical milestones below are chronological evidence, not current blockers.

Latest NVAPI milestone (2026-09-05, E-141/R-144): explicitly registered host
command lists now select the complete DXIL output head at the real NVAPI
launch-chain entry. Eight calls across two fence-retired sessions match the
standalone recorder byte-for-byte, with zero D3D12 debug errors, 12 invalid-call
rejections and zero remaining sessions. Ordinary activation/model buffers need
no HIP import. Unregistered lists keep the laboratory HIP route; live game
state/fence tracking, upstream producers and history modes remain open.
The original HIP self-test and all 222 Python tests still pass.

Latest native-route milestone (2026-09-05, E-140/R-143): the complete output
head now has a reusable D3D12 recorder accepting caller-owned buffer slices,
base/output textures and command list. Eight invocations across two submissions
pass standalone byte parity, changed-base/activation sensitivity, restored-input
repeatability and CPU surface composition with zero D3D12 debug errors. This is
still a captured-input test, not live NVAPI integration or complete-network
validation. At that milestone the NVAPI slot154 branch still called HIP. Measured DXIL
head time in the second submission is 39.6-44.9 ms with debug layer and diagnostic
copy enabled; release performance is unmeasured. Full-frame S7/S8 remain open.

The earlier instruction-translation route and reference history follow.

Current highest achieved S-level: **S6 PASS (isolated N0)**. The accepted N0
lowering executes its full `80x48x32` grid on RX 9070 XT and differs from the RTX
5070 zero-input oracle in only 29 of 1,966,080 E4M3 bytes. On an independent
nonzero captured RGBA16F input it differs in 46 bytes and repeats bitwise,
providing an initial input-domain generalization result. A single-context
156-slot replay also completes twice without RTX intermediate injection and
propagates the N0 improvement through every recorded boundary. A new
NVIDIA-style square-pair fusion lowering across slots 2-6 moves the first strict
same-capture failure from slot 6 to slot 7: slot 6 now passes at NRMSE
`0.023386232`; slots 7, 8 and 10 now pass at `0.061686839`, `0.086983541` and
`0.096071659`. Slot 11 is the first strict failure at `0.134164531`. Strict S7
remains open, and the
public AMD NGX host still returns `0xBAD00001`.
A queue-complete frame-1 RTX capture establishes the definitive slot154 output
hash `8ADB4DE9...E6E8C6`. Replaying the exact frame-1 activation on RX 9070 XT
with the inferred first-32 packed-F16 fusion contract now produces correlation
`0.999161906` / NRMSE `0.040940560` and passes the post-block numerical gate.
An exact-state store trace reconstructs the AMD output byte-for-byte,
proving that this residual is created before surface conversion/addressing. The
returned selected-CTA MMA trace narrows it further: all 16 FP8 operations whose
complete warp inputs match also produce bitwise-exact outputs, while the other
240 already have divergent input fragments. The subsequent E4M3/movmatrix trace
finds zero matched-input failures in either family; its first difference is a
one-ULP packed-F16 input to E4M3 conversion 4. The returned live-register
snapshot perturbs the NVIDIA output and is rejected as a causal trace, but its
operands plus the accepted original-path `%r944` values uniquely fit the contract
"round the first product to F16, then fuse the second product into the add" for
all 32 lanes. Applying that contract to the first 32 of 64 candidates closes the
exact-input post-block gate. The v21 same-command-list
capture proves slot154 offset 56 is a real 640x360 RGBA16F SRV whose contents are
genuinely all zero immediately before launch. v22 captured the complete frame-1
activation resource, but injecting it on RX 9070 XT left the end-of-run comparison
essentially unchanged (correlation `0.421357178`, NRMSE `1.105763648`). Trace review
then exposed the baseline defect: frame-1 parameters/activation were being compared
with the texture read back after 300 evaluations. Slot154 texture parameters +88/+96
are zero in frame 1 but populated by frame 61, proving frame state changes. v23 now
captures slot154's output immediately before frame-1 slot155, but the returned
image is only a partial spatial wavefront. v24 captures a different, larger
wavefront despite byte-identical inputs. A native RTX replay produces every
component present in either inline capture bit-for-bit, so both inline readbacks
are now treated as synchronization-perturbed diagnostics. v25 disables all
inline copies, runs exactly one evaluation, and reads slot155 input/output only
after queue completion; its accepted result exactly matches native original PTX.
The live-resource slot-154 bridge now also has D3D12 submission-boundary
instrumentation: local validation observes command-queue/list creation, every
list `Close`, and every `ExecuteCommandLists` item without regressing existing
captures. RTX feature-18 package v26 is ready to determine whether all 156
launches share one submitted list. No queue-ordering conclusion is claimed until
that returned trace is analyzed.

The returned v26 trace now resolves that decision. All 156 frame-1 launches use
one command list; slot154/155 are sequences 159/160, followed by close 161 and
submit 162. There is no producer/output-head submission boundary available for
a HIP fence. The live route is therefore same-command-list D3D12 compute (or a
higher-level whole-list rewrite), not synchronous HIP inside the NVAPI wrapper.
The last two stages on that route now pass together. An RX 9070 XT DXIL tail
projection and the byte-exact surface shader execute in one D3D12 command list,
with no HIP/fence boundary. The tail retains a documented `v_dot2` arithmetic
difference in 0.22499% of residual halves; after composition the mean/max image
error is `1.41e-9 / 3.05e-5`, below the practical stage gate. Explicit software
round-to-nearest-even remains required for subnormal surface outputs. The next
same-list migration boundary is final projection; earlier output-head and graph
producers remain before live slot154 can be ordered correctly.

| Level | Meaning | State | Evidence |
|---|---|---|---|
| S0 | environment recorded | **PASS** | results/20260830_170305/environment.json |
| S1 | AMD HIP + RDNA4 compute works | **PASS** (WMMA test UNVERIFIED, S-B4) | results/20260830_170305/hip_probe.json (overall_pass=true, tests A–E PASS; F not counted) |
| S2 | D3D12/HIP interop works | **PASS-A** (buffers + fence, both directions mismatch=0). Texture path: roundtrip byte-exact but layout is NOT identity (swizzled), see Phase G | results/20260830_170305/d3d12_hip_interop.json, results/*_texture_interop/texture_interop.json, docs/INTEROP.md |
| S3A | DLSSNR binary understood (static) | **PASS-CORRECTED** — 15 hybrid containers with compressed PTX 9.4/sm_120 plus CUBIN; 231 PTX entries | docs/BINARY_ANALYSIS.md, results/20260831_010100_all_runtime_modules/extraction_manifest.json |
| S3B | DLSSNR dynamic call chain understood | **PASS-GRAPH** — complete stable 156-slot graph captured across five frames; 9 runtime modules mapped to DLL offsets, 96 functions owned, 43 used by tested path, 153/156 parameter blocks stable | results/20260831_002356_rtx5070_feature18_full_frame/, docs/CALLGRAPH.md |
| S4 | reference NR host works on NVIDIA | **PASS** — RTX 5070: public DLAA 300/300; signed DLSSNR 310.8 initialized; feature 18 created and inline evaluation succeeded under an armed module/export tracer | results/20260830_235021_rtx5070_feature18_trace/ |
| S5 | AMD passes NGX/NVAPI init | **BLOCKED** — the exact public host that completes on RTX returns `0xBAD00001` on AMD; this is a validated NVIDIA-hardware support gate rather than an invalid-ID/host defect | docs/NGX_ABI_AUDIT.md, results/20260830_232657_rtx5060_v3/ |
| S6 | first DLSSNR-originated neural kernel on AMD | **PASS (isolated N0)** — corrected learned N0 runs the complete grid on RX 9070 XT; same-input RTX scratch/output both pass the numerical gate and AMD repeats are bitwise deterministic | results/20260831_140500_amd_n0_full_grid_corrected/comparison.json, E-46 |
| S7 | complete single NR frame on AMD | **IN PROGRESS** — offline 156-slot replay executes twice and is deterministic; exact-input slot154/post-block passes. Hierarchical and joint N0 correction plus square-pair fusion makes slots 6-10 pass; slot 11 is now the earliest failure (`0.134164531`). Live integration remains open | results/20260905_043000_full_graph_fine_n0_jointulp_slots2to6_square_rx9070xt/manifest.json |
| S8 | output matches reference | **IN PROGRESS / FULL-FRAME FAIL** — exact-input post-block passes at correlation 0.999161906 / NRMSE 0.040940560. Current injection-free full graph first fails at slot 11 (`0.134164531`); final RGB correlation is `0.435413076` and NRMSE is `1.111083053` | results/20260905_043000_full_graph_fine_n0_jointulp_slots2to6_square_rx9070xt/manifest.json |
| S9 | stable 1000-frame run | not started | docs/RESULTS.md |
| S10 | performance optimization | not started | docs/RESULTS.md |

## Log

- 2026-09-05 (same-list D3D12 tail + surface passes): the final 32-to-4 FP16
  projection and surface compositor now execute as two ordered DXIL dispatches
  on one RX 9070 XT command list with no HIP runtime dependency. The tail's
  `v_dot2` lowering produces 2,286 residual half differences, but final surface
  mean/max absolute error is only `1.41e-9 / 3.05e-5`, with zero non-finite
  values. This is a numerical PASS, not a byte-exact claim. Next is final
  projection.

- 2026-09-05 (first D3D12-native output-head stage passes): the surface stage
  runs as DXIL on RX 9070 XT, records compute and readback in one D3D12 list,
  and matches the accepted 640x360 RGBA16F oracle byte-for-byte. The initial
  HLSL conversion exposed 1,709 one-ULP subnormal differences; explicit IEEE
  RN-even conversion closes them without exceptions. Next is the FP16 tail.

- 2026-09-05 (RTX v26 selects same-command-list D3D12 compute): the returned
  archive passes all runtime/integrity gates and shows exactly 156 unique slots
  on one list. Slot154/155 are adjacent and the list is closed/submitted only
  afterward. The simple external-fence design is rejected because its producer
  work does not yet exist on a GPU queue at the interception point. S7 remains
  open; the next implementation target is a D3D12-compute output-head prototype
  recorded directly onto the captured list.

- 2026-09-05 (submission-topology tracer and RTX v26 package ready):
  `module_trace` 3.5 now assigns one ordered sequence to neural launches,
  command-list closes and queue submissions. The expanded D3D12 self-test passes
  with one hooked queue, two hooked command lists, two closes and two submitted
  list items while all prior snapshots remain green. The one-run RTX package is
  `deliverables/dlssnr-windows-feature18-submission-topology-v26-20260905_022332.zip`
  (SHA-256 `D4A56F15...FDB6D1A1`). It is an observation package, not S7 evidence.

- 2026-09-05 (exact 4h controls clear slot 11 as an independent defect):
  injecting only the native RTX slot-10 output makes slots 11-15 pass and moves
  the first failure to slot 23; injecting only the exact slot-9 input keeps
  slot 10 onward on RX and makes slots 10-13 pass before slot 14 first fails at
  `0.113606190`. Both are diagnostic, non-S7 runs. A 32-site square-pair rewrite
  of slot 10 is rejected because slot 10 slightly worsens and slot 11 remains
  the first failure (`0.133910018`). The accepted injection-free baseline and
  its slot-11 frontier are unchanged; the next target is propagated state at
  the slot-9-to-slot-10 boundary, not slot 11 itself. The first exact-input
  mismatch is now mapped to CTA `(4,0)`, warp-y `1`; an AMD-output-preserving
  272-MMA RTX trace package is ready at
  `deliverables/swin4h_slot10_fp8_mma_trace_reference_20260904_215525.zip`.

- 2026-09-05 (joint normal0/normal1 correction moves the frontier to slot 11):
  a full-grid search accepts ten phase segments only when paired cos0/sqrt0 ULP
  shifts fix consumed normal0 or normal1 without additions in either. N0 reaches
  29 differing bytes on zero input and 46 on the independent nonzero input,
  improving the preceding 64/68 result by 35/22 bytes with zero additions. The
  deterministic injection-free graph now passes slots 6-10; slot 11 becomes the
  first strict failure at `0.134164531`. Final image parity and live integration
  remain open.

- 2026-09-05 (hierarchical normal2 correction makes slot 7 pass): a GPU-ablation
  search retains normal1 segment 55169 and a trace-domain-safe 524,288-segment
  normal2/cos1 hierarchy. Complete N0 now differs from RTX in 83 bytes on zero
  input and 86 on the independent nonzero input, fixing 33/23 bytes relative to
  the earlier accepted 116/109 result with no additions. The two-run,
  injection-free 156-slot graph is deterministic; slot 6 is `0.034749463`, slot
  7 passes at `0.089691090`, and slot 8 becomes the first failure at
  `0.123923003`. Final image parity and live integration remain open.

- 2026-09-04 (normal1 full-grid oracle accepted; N0 residual reduced again):
  returned RTX 5070 traces pass archive, payload, launch, repeat and untouched-
  output controls. Across 245,760 samples, phase is exact; sine differs in
  206,632 samples, sqrt in 87,577, but consumed `normal1_f16` differs in only
  380. A conservative 65,536-segment sine correction plus sparse sqrt ULP shifts
  reduces the complete N0 result from 154 to 116 differing bytes on zero input
  and from 168 to 109 on the independent nonzero capture, with zero additions in
  both comparisons. The injection-free full graph remains deterministic; slot 6
  improves to `0.039886535` and slot 7 to `0.103740020`, still narrowly outside
  the strict `0.1` gate. S7/S8 remain open.

- 2026-09-04 (square-pair fusion closes isolated slots 2/3 and moves the full
  graph frontier): applying the NVIDIA-like half2 contract to all 32
  add-of-square-pair sites in the relevant slot-2-through-slot-6 functions makes
  the two-run, injection-free RX graph pass slot 6 at NRMSE `0.042475154`; slot 7
  becomes the first failure at `0.110049329`. On the accepted isolated inputs,
  slot 2 now matches all 1,966,080 RTX bytes, while slot 3 matches all 896 traced
  32-bit register words and its complete output bit-for-bit. Diagnostic exact
  slot1/slot2 injection makes slots 2-6 bitwise exact and slots 7/8/9 pass at
  `0.009966183`/`0.031927418`/`0.028862428`, moving the first failure to slot 11.
  These injections are non-S7, but they prove the current slot-7 failure is
  amplified from the remaining 154-byte N0 residual. A full-grid `sin0` /
  `normal1_f16` RTX oracle package is ready at
  `deliverables/n0_normal1_full_grid_traces_reference_20260904_193326.zip`,
  SHA-256 `4545FFB1...9DE71A`; the README contains the exact command.

- 2026-09-04 (slot-6 upstream diagnosis resumed; delayed slot-3 normalization
  oracle ready): same-capture sensitivity proves the slot-6 failure is primarily
  amplified slot-3–5 state rather than a standalone 2h defect. The earlier
  producer-adjacent reduction trace changed RTX output and is inadmissible.
  A replacement snapshots 31 square/reduction/clamp/rsqrt registers only at the
  previously output-preserving E4M3 boundary. Its full-grid RX 9070 XT run loads,
  executes and writes 3,968 trace bytes; its output hash `C05D1463...94D581`
  exactly matches the earlier output-safe AMD path trace. RTX acceptance requires
  two bitwise-repeat traces and exact preservation of `33FE6004...50BD4F9`.
  Package `deliverables/slot3_delayed_norm_snapshot_reference_20260904_185812.zip`,
  SHA-256 `9CC85E2D...5E3BFD`; README contains the command.

- 2026-09-04 (pre-E4 snapshot rejected; packed-F16 fusion contract closes the
  exact-input post-block): archive `B77A5010...60603A` passes device, execution,
  repeat and payload controls, but both RTX outputs are `D82309DE...DE1D` rather
  than `8ADB...E6E8C6`; its RTX/RX traces are equal only on the perturbed path.
  Offline evaluation of the four captured operands against the previously
  accepted original-path `%r944` oracle fits all 32 lanes only when the first
  product is rounded to F16 and the second multiply is fused into the add.
  Applying this rule to the first 32 of 64 candidates on RX 9070 XT gives
  correlation `0.999161906`, NRMSE `0.040940560`, exact fraction `0.961141493`,
  no nonfinite values and exact alpha: the exact-input post-block gate passes.
  A 156-slot integrated run executes twice deterministically; it still fails
  upstream (fine boundary slot 6; first boundary in its coarse set slot 15).

- 2026-09-04 (three-producer-site trace also rejected; pre-E4 snapshot ready):
  archive `2E33A03B...0A3CA` passes integrity and execution controls, but its two
  RTX outputs are `9EC0AACC...42530` instead of `8ADB...E6E8C6`. Its 1,920-byte
  RTX trace again equals RX bit-for-bit, so merely placing stores adjacent to
  `%r816`, `%r817` and `%r944` changes NVIDIA JIT behavior. The replacement does
  not instrument those producer instructions. It snapshots their four inputs,
  three outputs and E4M3 result once at the already output-preserving conversion-4
  boundary. Package `deliverables/postblock_r944_snapshot_reference_20260904_183411.zip`,
  SHA-256 `77443FB2...79A5AD`.

- 2026-09-04 (all-F16x2 trace rejected for perturbation; narrow path package ready):
  archive `C08B3467...A9FCFB` has valid payload integrity, two successful RTX
  executions and bitwise-repeat trace hash `205A2351...66D79C`. However, both
  instrumented outputs hash to `9A3D8A1D...3B834`, not the required queue-complete
  `8ADB4DE9...E6E8C6`; the strict receiver therefore rejects it as a causal
  oracle. Its RTX trace is incidentally byte-identical to the RX trace, confirming
  that recording all 1,522 operations perturbs/regularizes the path. A replacement
  traces only the two multiplies `%r816`/`%r817` and add `%r944` that directly
  feed the first differing E4M3 input. Package:
  `deliverables/postblock_f16x2_path_trace_reference_20260904_181851.zip`, SHA-256
  `60E49965...8FB55`; acceptance still requires exact final output preservation.

- 2026-09-04 (E4M3/movmatrix oracle accepted; packed-F16 trace ready): archive
  `BC0F6D6...1F9967B4` passes strict integrity, NVIDIA-device, repeat and exact
  output-preservation controls. The two RTX traces are bitwise identical. Across
  388 E4M3 conversions there are zero records with equal input and unequal
  output. Across 32 movmatrix operations there are likewise zero unequal outputs
  when the complete 32-lane input warp matches. The first observed upstream
  difference is E4M3 conversion 4, lane 1: packed half input `B32EBD09` on RTX
  versus `B32EBD08` on RX, while that conversion's output is equal. Scalarizing
  every packed-F16 arithmetic instruction on RX leaves the final output unchanged,
  showing that a blanket rewrite is not a correction. The next package traces
  all 1,522 packed-F16 operations at the same CTA:
  `deliverables/postblock_f16x2_trace_reference_20260904_174308.zip`, SHA-256
  `5862355F...1E5F77`; its README contains the command.

- 2026-09-04 (post-block MMA oracle accepted; first mismatch moves pre-FP8):
  archive `8E28085F...F5AC5AE` passes integrity, repeat, device, output-preservation
  and exact-reference controls. Both RTX traces are bitwise identical and the
  instrumented output remains `8ADB4DE9...E6E8C6`. A warp-aware comparison
  corrects the preliminary lane-local interpretation: no FP8 operation with
  complete matching warp inputs has a different output. All 16 fully matched
  FP8 MMAs are bitwise exact; the 240 differing MMAs already receive different
  fragments. The first input difference is MMA 0, A0 high byte at lanes 9 and 29
  (`0x1B` RTX versus `0x1A` AMD). F16 divergence is downstream propagation.
  The next package traces all 388 E4M3 conversions and 32 movmatrix operations:
  `deliverables/postblock_e4m3_mov_trace_reference_20260904_171152.zip`, SHA-256
  `C4890373...BDFF9DE`. All 175 Python tests pass.

- 2026-09-04 (v25 queue-complete oracle accepted; post-block residual isolated):
  archive `72317298...1BE2E` passes every strict control. Slot155 input/output are
  bitwise identical with SHA-256 `8ADB4DE9...E6E8C6`, exactly matching the
  independent original-PTX replay. Against this oracle, RX 9070 XT with exact
  frame-1 activation scores correlation `0.989590977` / NRMSE `0.144267266`;
  alpha is exact. A complete pre-store trace covers all 245,760 padded surface
  coordinates and reconstructs the same AMD visible output `EA6CF1CB...C0EEBA`
  byte-for-byte. Surface addressing, FP32-to-FP16 conversion and writeback are
  excluded; the remaining post-block defect lies in FP8/F16 MMA lowering. The
  next RTX package is `deliverables/postblock_mma_trace_reference_20260904_154238.zip`
  (SHA-256 `13664500...319EDC`). The D3D12 self-test and all 173 Python tests pass.

- 2026-09-04 (exact-contract return accepted; inline oracle invalidated; v25 ready):
  archive `8D789F06...0D953A2` is diagnostically valid. Original, compat, E4M3 and
  movmatrix execute twice on RTX 5070, are mutually bitwise identical, and produce
  SHA-256 `8ADB4DE9...E6E8C6`. Delta analysis against the exact all-zero initial
  surface shows this native output matches **100%** of the components changed by
  v23 (`370,428/370,428`) and v24 (`389,845/389,845`). v23 only contains a
  wavefront beginning at row 200; v24 begins at row 188 with a different boundary,
  although activation, parameters, weights, texture and initial surface are
  identical. Therefore v23/v24 are not complete numerical oracles, and the prior
  AMD final metrics against them are withdrawn. v25 performs one evaluation with
  every inline snapshot disabled, then fences the queue before reading slot155's
  input/output. Package:
  `deliverables/dlssnr-windows-feature18-deferred-frame1-output-v25-20260904_152127.zip`,
  SHA-256 `E97A0D7A...B8AF373`. The tracer self-test and all 170 tests pass.

- 2026-09-04 (v24 accepted; superseded by queue-complete capture analysis):
  archive `DA9292BC...394491` passes strict ingestion. Slot154 changes only
  677,929/1,843,200 bytes (`36.7800%`) of the zero destination surface. The
  initial surface hash `8DF6D450...E88A3` is the SHA-256 of 1,843,200 zero bytes
  and matches the AMD integrated plan. The later exact-contract return proves the
  v24 after image is another incomplete inline wavefront, so its former use as a
  final AMD oracle and the associated `0.338362441` / `1.497234543` conclusion
  are superseded.

- 2026-09-04 (v23 accepted; destination initial-state capture v24 ready): returned
  archive `DAA579C3...EE4977C` passes every strict control. The frame-1 slot154
  output hash is `94A5E9E6...BA5DF3`; it differs from the end-of-run output in
  1,473,125 bytes, proving the earlier oracle was temporally misaligned. However,
  the RX 9070 XT complete-activation replay still fails against the corrected
  frame-1 oracle at correlation `0.338362441` / NRMSE `1.497234543`. The v23 and
  original 184-byte parameter blocks differ only in four rebased GPU pointers;
  all scalar fields match. v24 therefore captures slot154's destination surface
  both before and after the kernel to test the remaining zero-initialization
  assumption. E-116 later withdraws these metrics because v23 is incomplete.
  All 168 tests and the D3D12 self-test pass.

- 2026-09-04 (v22 accepted; temporal-oracle defect isolated; v23 ready): returned
  archive `D76AB579...2EA8FB` passes Feature 18, D3D12, texture, activation and
  final-copy controls. Its frame-1 activation buffer is 27,807,744 bytes with
  SHA-256 `925AE733...6CD33`. Injecting it before slot154 on RX 9070 XT runs all
  156 slots twice and is deterministic, but remains at correlation `0.421357178`
  / NRMSE `1.105763648` versus the end-of-run reference. The launch trace proves
  slot154 parameter offsets +88/+96 are zero in frame 1 and nonzero by frame 61;
  therefore pairing frame-1 graph state with a post-300-evaluation copy was not a
  valid same-input oracle. v23 captures slot154 output on the same command list
  immediately before frame-1 slot155. D3D12 self-test and all 167 tests pass.

- 2026-09-04 (native stage sweep returned; full activation capture v22 ready):
  archive `E811AE23...35BF259` is diagnostically valid. On RTX 5070, original,
  compat, E4M3 and movmatrix all execute twice, repeat bitwise and share output
  hash `8ADB4DE9...E6E8C6`; therefore these three transformations do not create
  the current mismatch. However, original versus live output still fails at RGB
  correlation `0.421336375` / NRMSE `1.105928784`, proving the standalone input
  contract is incomplete. FP8-MMA and later generated forms separately fail JIT
  with `CUDA_ERROR_INVALID_PTX`; they cannot yet be compared numerically. v22 now
  captures the entire live slot154 activation buffer on its pre-launch command
  list. All 165 tests and the D3D12 tracer self-test pass.

- 2026-09-04 (pre-launch zero texture accepted; lowering sweep ready): v21
  completes Feature 18 `300/300`. Its same-command-list frame-1 readback is a
  valid all-zero 640x360 RGBA16F resource. Trace events link resource creation,
  SRV descriptor, merged CUDA texture handle, pre-launch arm and slot-154 bind;
  the copy input/output also remain byte-identical to the historical reference
  hash `CD556E0D...71FD243`. The runner's FAIL only reflects v21's now-disproved
  nonzero expectation. Missing texture pixels are therefore not the cause. A
  single RTX package now runs original, compat, E4M3, movmatrix, FP8 MMA, FP16
  MMA and decoder-compat stages twice with native CUDA surface/texture objects;
  its original stage must reproduce the reference exactly. All 164 tests pass.

- 2026-09-04 (late texture snapshot rejected; pre-launch v21 ready): the v20
  RTX 5070 return passes package integrity and Feature 18 completes `300/300`,
  but all 1,843,200 captured texture bytes are zero. Replaying that file on RX
  9070 XT executes all 156 slots twice and repeats bitwise, yet exactly reproduces
  the old zero-texture hash `8CCF23A5...77D25` and failed RGB result
  (`0.427049` correlation / `1.117199` NRMSE). This proves the end-of-run readback
  was initially attributed to a post-run clear. v21 inserts the copy immediately
  before frame-1 slot 154 on the same command list and originally rejected an
  all-zero return. The newer entry above supersedes that interpretation.
  Its package SHA-256 is `2E0B016B...F4794DC`; all 162 tests pass.

- 2026-09-04 (post-block arithmetic closed; real texture capture required): the
  returned RTX 5070 trace is integrity-valid, deterministic and reconstructs all
  245,760 output coordinates. Against RX 9070 XT, both store sites pass on all
  four components; RGB correlations are `0.999999640` to `0.999999763` and
  NRMSE is `0.000688709` to `0.000848467`. The runner's FAIL is retained as a
  separate control failure: its transformed RTX output is only correlation
  `0.421336` / NRMSE `1.105929` versus the captured original. Descriptor evidence
  shows offset 56 resolves to a real 640x360 RGBA16F SRV with point/border
  sampling, invalidating the zero-texture assumption. A trial using the final
  image as that texture changes the output substantially but still fails, so it
  is not promoted. The v20 Feature-18 package now captures the actual resource;
  its strict return receiver is covered by the current 162-test passing suite.

- 2026-09-04 (post-block ABI correction and RGB-only audit): restoring the exact
  full 29,773,824-byte RTX arena before slot 154 did not change the former biased
  output. Preserving captured parameter offset 56, instead of clearing it, changed
  exact-state output NRMSE from the old aggregate `1.001163550` to `0.001365541`
  and removes the 0.5 RGB bias. The same fix in an injection-free 156-slot run is
  deterministic and produces RGB means close to RTX. However, the old metric was
  dominated by constant alpha=1. After excluding alpha, exact-state slot 154 is
  NRMSE/correlation `1.105764`/`0.421357`, and the native full graph is
  `1.117199`/`0.427049`. S7/S8 therefore remain open; the next numerical target is
  the post-block RGB path, while slot 6 remains the earliest internal failure.
- 2026-09-04 (post-block store trace package ready): both pre-surface-store sites
  are instrumented over the complete 81x49x32 launch. The RX 9070 XT trace runs
  twice with identical SHA-256 `A57E7C4D...D40D87FC`; 122,880 records are active
  at each site. The RTX reference package is
  `deliverables/postblock_store_trace_reference_20260904_115440.zip` (SHA-256
  `05146F6E...FE220757`) and includes its exact PowerShell command in README.
  The safe ZIP receiver validates device/integrity/repeat/output controls and
  automatically reports the first differing site, CTA, lane, coordinate and
  component. The 245,760 active RX records cover every 640x384 surface coordinate
  exactly once and reconstruct the actual RGBA16F output byte-for-byte; its repeat
  self-check is exact and all 158 tests pass.
- 2026-09-04 (distributed slot-3 sensitivity and decoder checkpoints): none of
  16 single-reduction or 9 combination candidates makes slot 6 pass. Every one of
  eight disjoint slot-3 tensor chunks improves slot 6 when replaced from RTX,
  showing distributed sensitivity. Same-capture decoder checkpoints reveal
  non-monotonic error through slots 99-153; late main/scratch and full-arena
  injections isolate a separate slot-154 execution/ABI issue rather than an
  unobserved upstream tensor.
- 2026-09-04 (prior slot-3 selective candidate rescored on the current graph):
  replacing slots 3 and 4 with the five-node FP32-reduction candidate improves
  local slot-3 NRMSE only from `0.005225894` to `0.005146451`, but worsens the
  causal early boundaries: slot 5 `0.021818147 -> 0.022040888`, slot 6
  `0.102493965 -> 0.103664437`, and slots 7/8 also regress. Later slot 98 and
  final NRMSE improve, but the final delta is only `-0.000078227`. The candidate
  remains rejected. Future slot-3/4 search must score slot 5/6 propagation, not
  optimize isolated slot-3 NRMSE alone.
- 2026-09-04 (same-capture early-Swin localization): fine checkpoints were
  extracted directly from the same 4.06 GB RTX full-graph capture, avoiding an
  initially detected cross-capture reference mismatch. Slots 3/4/5 pass at
  NRMSE `0.005225894`/`0.014557257`/`0.021818147`; slot 6 is the first strict
  failure at `0.102493965`, followed by slot 7/8 at `0.255073016`/`0.318781847`.
  Explicit non-S7 RTX-state injections after slots 2/3/4/5 are deterministic.
  Injecting the exact slot-5 state makes slots 6-8 all pass at
  `0.015492492`/`0.049286892`/`0.084323278`, proving the 2h implementation is
  primarily amplifying precision residuals from the slot 3-5/1h chain. The next
  correction target is therefore slot 3-5 propagation, not a 2h semantic rewrite.
- 2026-09-04 (CTA `(8,7)`, nonzero validation and integrated propagation): the
  returned RTX bundle passes integrity and output-preservation controls. Only
  `normal1_f16` differs (2/64 samples, each 1 ULP); MMA 12 already receives the
  changed A fragment, while scalar replay of identical RTX fragments matches RTX
  MMA exactly. The accepted N0 candidate generalizes to a nonzero captured input:
  168 output differences, NRMSE `0.000383619918`, and bitwise-identical repeats.
  Against the square-pair-fusion baseline, cosine corrections fix 94 bytes and add
  none. Two complete 156-slot runs are deterministic; boundary NRMSE improves at
  every coarse recorded stage through slot 98 and slot 9 now passes. At that
  coarse resolution slot 15 was the first failure; the later same-capture fine
  checkpoint study above supersedes the localization with slot 6. Final NRMSE
  improves only slightly to `1.001121844`, so S7 remains open.
- 2026-09-04 (full-grid second Box-Muller correction): the returned RTX oracle
  passes all controls. Only 342/245,760 consumed `normal2_f16` values differ;
  RTX cosine with AMD sqrt theoretically leaves 29. A conservative 32,768-bin
  second-cosine model fixes 48 complete-N0 output bytes with zero additions,
  reducing the accepted residual from 202 to 154 with deterministic repeats.
  The 65,536-bin alternative adds one error and is rejected. The next residual
  is CTA `(8,7)`; S7 and real-frame validation remain open.
- 2026-09-04 (CTA `(20,4)` residual localized to second Box-Muller path):
  `normal0_f16` and `normal1_f16` are exact, while `normal2_f16` differs only at
  sample 2 (`0xBC8A` versus `0xBC8B`) and reaches MMA 0 through its A fragment.
  A single RTX package now captures both `phase1/cos1` and
  `sqrt1/normal2_f16` over the full grid; its two matching RX traces preserve
  the accepted 202-byte output exactly. S7 remains open.
- 2026-09-04 (full-grid normal0 boundary and conservative cosine correction):
  only 356/245,760 consumed FP16 values differ. A 4,096-segment model restricted
  to 88 boundary-safe segments fixes 39 complete-N0 output bytes with no new
  differences, reducing the accepted candidate from 241 to 202; two RX runs are
  byte-identical. The remaining first residual is CTA `(20,4)` and its combined
  reference package is ready. S7 remains open.
- 2026-09-04 (CTA `(35,3)` residual localized): only consumed `normal0_f16`
  sample 54 differs, and MMA 12 first receives it in A. Sqrt is exact at that
  sample; cosine alone reproduces the FP16 boundary change. A third-input causal
  ablation fixes two additional final bytes with zero additions (235 diagnostic,
  241 production). The next package captures the relevant boundary over all
  245,760 samples rather than proceeding CTA-by-CTA; S7 remains open.
- 2026-09-04 (CTA `(35,3)` clean sweep): the returned RTX 5070 package passes
  integrity and output-preservation gates. Rounded sqrt differs in 20/64 samples
  and cosine in 52/64, so the clean stage data is valid but not causal by itself.
  A combined consumed-FP16 plus 256-MMA bundle is ready; matching RX traces are
  output-preserving. The accepted candidate stays at 241 differences and S7 is
  open.
- 2026-09-04 (full-grid cosine oracle): all 245,760 RTX/RX cosine inputs match
  bitwise while 202,961 rounded outputs differ, proving architecture-specific
  approximation behavior. Global 1,024/4,096-segment corrections regress the
  accepted 241-byte residual to 247/248 and are rejected. A diagnostic two-angle
  ablation fixes four old output bytes with zero additions and moves the first
  residual to CTA `(35,3)`; a clean one-run RTX stage package is ready. The
  accepted implementation remains unchanged and S7 remains open.
- 2026-08-30: repo scaffolded; docs/RESEARCH.md written (13 topics verified online).
- 2026-08-30: S0 achieved — environment.json recorded (RX 9070 XT 0x1002/0x7550, HIP 7.2.0, FL 12.2, SM 6.8).
- 2026-08-30: S1 achieved — hip_probe A–E all PASS on gfx1201 (fp32 maxErr 5.96e-08, fp16 GEMM maxErr 3.26e-03, fp8 e4m3/e5m2 exact). WMMA f32 builtin does not compile in this ROCm 7.2 LLVM (S-B4); test F recorded UNVERIFIED, not counted.
- 2026-08-30: S2 achieved (PASS-A) — d3d12_hip_interop: D3D12Heap buffer + D3D12Fence imported into HIP; T3/T4 bidirectional transfers mismatch=0; T5 texture import succeeded (informational at the time). PASS-B (command-list fusion) = no HIP API, recorded as survey (T6).
- 2026-08-30: Phase 3/4 tools delivered — binary_probe self-tested (R-3); module_trace.dll + nvapi64.dll shim smoke-tested (Round 1 tooling, since superseded — see Round 2 below).
- 2026-08-30: Phase 5 delivered — nr_host skeleton builds and runs; without proprietary DLLs it exits 3 BLOCKED_MISSING_PREREQUISITE (R-4), emits deterministic reference package (two runs, identical SHA-256). No fabricated execution.
- 2026-08-30: Phase 6/7 framework delivered — docs/CALLGRAPH.md holds the CASE A/B/C/D decision tree and the GATE-0..4 PTX go/no-go chain; evidence slots PENDING_PREREQUISITE (B-1/B-2).
- 2026-08-30 (evening): B-1/B-2 resolved — user supplied legally owned nvngx_dlss.dll (310.7) and nvngx_dlssnr.dll (310.8); ledgered with SHA-256/Authenticode (docs/PROPRIETARY_FILES.md).
- 2026-08-30 (evening): initial S3A scan found 15 sm_120 CUBIN/ELF representations and no plaintext PTX markers. This historical “pure SASS” conclusion was superseded on 2026-08-31 when the runtime containers were decompressed.
- 2026-08-30 (evening): first REAL load on AMD — both DLLs load cleanly; NGX init returned 0xbad00001 on vendor=0x1002 under a RECONSTRUCTED ABI. Round 2 later showed that run's ABI was invalid (docs/NGX_ABI_AUDIT.md E1/E3), so it was re-executed under the official ABI.
- 2026-08-30 (Round 2): instrument recalibration, Phases A–G:
  - A: official NVIDIA/DLSS headers fetched (commit a291cc7d2cc6); full ABI audit docs/NGX_ABI_AUDIT.md (E1–E7); nr_host split into official ABI + private_ngx_compat.h (feature 18 = PRIVATE ABI only).
  - A milestone: under the OFFICIAL ABI (both snippet-4-arg and loader-5-arg Init paths) the AMD adapter still returns 0xBAD00001 = FAIL_FeatureNotSupported. Round 1's observation reproduces on correct ABI; vendor-gate interpretation strengthened but remains UNCONFIRMED (no positive control without an RTX box).
  - B: tools/ngx_abi_probe — compile-time static_asserts + runtime export-table check; PASS (dll is a snippet surface, E7).
  - C: nr_host rewritten as a genuine NGX host (static-linked nvsdk_ngx_d.lib; Init→GetCapabilityParameters→AllocateParameters→create DLSS/DLAA→command-list evaluate→readback→release). On this box it reaches Init rejection (exit 6) — the rest of the pipeline is exercised on RTX. Persisted official-ABI run: results/20260830_205123_nr_host_r2/ (scripts/run_nr_host_official_abi.ps1).
  - D: Stage A/B separation; scripts/run_nvidia_reference.ps1 one-command pipeline + reference_bundle/ (this box correctly emits BLOCKED_EXTERNAL_HARDWARE bundle).
  - E: module_trace v2 — synchronous ModuleTrace_InitializeAndWait gate, GetProcAddress interception, immediate IAT patching of fresh modules, ldr-notification best-effort; tests/module_trace_selftest READY (full load→resolve→unload chain logged).
  - F: nvapi trampoline ABI self-proof. Round 1's "stack args ride along" was DISPROVED (args 5+ corrupted by the dispatcher frame); trampoline replaced with a pure-jmp passthrough, re-proved byte-exact for 1..10 args (tests/nvapi_trampoline_test).
  - G: tests/texture_interop_test gate — RGBA8/RGBA16F/R32F/RG16F: HIP→D3D12 roundtrip byte-exact for all four formats; D3D12→HIP linear-layout assumption FAILS (AMD texture memory is swizzled). Interop usable only with swizzle-aware addressing.
- 2026-08-30 (Round 2 delivery, Phase H+I): S-level restructure + repo-wide recalibration of over-strong claims (STATUS/CALLGRAPH/RESULTS/EVIDENCE/BLOCKERS/ARCHITECTURE/INTEROP/README); FINAL_REPORT gained the Round 2 chapter answering A–G; 7 milestone commits landed locally (2486b1b..2b4de03, per-command identity, no config changes); push to skchen17/dlssnr-amd-lab DEFERRED at user's choice — git direct/explicit-proxy still blocked (curl 000), WinHTTP Git Data API channel verified reachable and awaits a user-supplied PAT.
- 2026-08-30 (RTX 40 remote control): on a Linux server with 2× RTX 4090 D
  (`sm_89`), an `sm_89` positive-control CUBIN loaded on both GPUs and an `sm_90`
  control was rejected cleanly. Loading either of two structurally valid DLSSNR
  `sm_120` CUBINs crashed driver 550.144.03 in `cuModuleLoad` (exit 139). This
  confirms that this RTX 40/old-driver environment cannot consume the payload,
  but does not advance S4: the host cannot run the Windows D3D12/NGX feature-18
  pipeline and driver age is confounded with architecture. See RESULTS R-13.
- 2026-08-30 (RTX 5060 Windows control): Windows cloud host exposed a real
  Blackwell `sm_120` D3D12 adapter (driver 591.59); tracer self-tests, ABI export
  probe, DLL signatures, and static payload probe all passed. The unchanged host
  still returned `0xBAD00001` at NGX Init on vendor 0x10de, before loading DLSS,
  NVAPI, or CUDA modules. This disproves use of that result as vendor-gate
  evidence and identifies the host contract as the next gate. Comparison with
  DLSS5-Feeder found the lab used different self-assigned App/Project IDs; nr_host
  now uses Feeder's `0x1000000` and `a0f57b54-...` identifiers. See RESULTS R-14.
- 2026-08-30 (RTX 5060 Windows v2 control): corrected Feeder identifiers changed
  NGX Init from `0xBAD00001` to Success. Capabilities, parameters, upload, and
  public DLAA CreateFeature all passed. The trace captured driver `_nvngx.dll`,
  `nvngx_dlss.dll`, system NVAPI, `nvcuda64.dll`, `nvdxgdmal64.dll`, and
  `nvobjectloader64.dll`. All eight Evaluate calls returned `0xBAD00005` because
  the host omitted `InRenderSubrectDimensions`; Round 3 now mirrors Feeder's full
  Evaluate contract. No feature-18/DLSSNR execution is claimed. See RESULTS R-15.
- 2026-08-30 (RTX 5060 Windows Round 3): adding the required render-subrect fields
  fixed Evaluate completely. Stage A and traced Stage B each completed 8/8 frames,
  read back 1,048,576 bytes, and produced byte-identical output SHA-256
  `9B98DFC631755EED553036786D23527D7E810E1DFA204BEB802787AD394C5041`.
  The output differs from the deterministic input in 302,026 bytes, confirming a
  real processing path. This proves the public reference host; feature 18 remains
  the next distinct gate. See RESULTS R-16.
- 2026-08-30 (RTX 5070 Feature-18 reference): ReShade 6.8 registered "DLSS 5
  Neural Rendering" API 18; RenoDX installed D3D12 NGX hooks, initialized the
  signed DLSSNR 310.8 runtime, created feature 18 twice (initial + warm-up rebuild),
  and logged successful inline feature-18 evaluation at counts 1 and 60. The
  public host completed 300/300. S4 is now PASS. See RESULTS R-17.
- 2026-08-31 (AMD graph transport): the RX 9070 XT/gfx1201 replay reconstructed
  9 module and 96 function lifetimes and submitted all 156 captured slots in exact
  order. GPU-side hashing validated 11,624 uploaded parameter bytes with zero
  mismatches. R-22/E-18 is a transport prerequisite only; S6 remains unachieved.
- 2026-08-31 (runtime-container correction): all 15 `0xBA55ED50` containers
  expose Zstd-compressed PTX 9.4/sm_120 (231 entries total). GATE-1 is corrected
  from FAIL to PASS; the payload is hybrid PTX+CUBIN, not pure SASS.
- 2026-08-31 (AMD original-PTX execution): after function isolation, ZLUDA
  v7-preview.3 compiled and launched unchanged `cc_cb_clear` PTX on RX 9070 XT.
  All 27,648 words read back as `0xffffffff` with zero mismatches. Copy remains
  blocked at `sust.p.2d`; neural PTX remains blocked at MMA/FP8 and related
  constructs. This advances the translator route but does not count as S6.
- 2026-08-31 (AMD final-copy boundary): `nvapi_amd` added the two official R610
  descriptor-object APIs and a native D3D12 lowering for `cg2r_copy_kernel`'s
  decoded 72-byte ABI. RX 9070 XT readback matched all 8,192 RGBA32F components
  exactly (`max_abs_error=0`); the clear/VA/lifecycle regression also re-passed.
  Boundary progress does not count as S6; the next gate is an RTX tensor oracle
  followed by the first real neural operator.
- 2026-08-31 (first neural ABI): static analysis joined the recovered module-0
  PTX to R-21 slot 1. Its 264-byte pre-block parameter array has 37 direct loads
  covering 252 bytes with exact offsets and captured values. This narrows the N0
  oracle but does not assign unproven tensor semantics or advance S6.
- 2026-08-31 (RTX descriptor identity): v10 preserved the positive Feature-18
  result and captured 618 merged plus 21 independent descriptor calls. All five
  copy launches resolved both handles (`unresolved=0`); pre-block offsets +0/+8/+16
  map to primary/auxiliary/history textures. Object-to-resource content remains
  pending and S6 is unchanged.
- 2026-08-31 (resource-trace v11): D3D12 resource/view/copy hooks passed a real
  RX 9070 XT self-test with strict JSONL. A verified v11 RTX package now targets
  resource format/dimensions/GPU-VA identity needed before bounded tensor capture.
- 2026-08-31 (RTX resource identity): returned v11 preserved 300/300 and Feature-18
  success. The formal join resolved every selected copy and pre-block resource;
  copy source/destination are distinct 640x360 RGBA16F textures, and all three
  direct pre-block GPU addresses map to exact allocation offsets.
- 2026-08-31 (copy snapshot v12): a real RX 9070 XT self-test passed the complete
  ordered GPU readback and numerical comparison path with 32,768/32,768 identical
  bytes. The verified v12 RTX package is ready for the one remaining copy-content
  oracle run; this is boundary evidence and does not advance S6.
- 2026-08-31 (N0 snapshot v13): static origin analysis classifies pre-block +216
  and +248 as write-only candidates and +224 as read-only. A real RX 9070 XT
  ordered pre/post capture self-test exported all six windows and detected the
  exact injected change. A verified v13 RTX package now combines M4 copy content
  and N0 neural activity in one requirement-driven run; S6 remains unchanged.
- 2026-08-31 (returned v13/M4 closure): RTX 5070 slot-155 source and destination
  are bitwise-identical 640x360 RGBA16F data. RX 9070 XT reproduced the exact
  same input/output pair with zero mismatches, closing M4. The same RTX run proves
  strict N0 scratch/output activity with unchanged weights and exact FP8 logical
  shape/stride candidates; this is observation, not AMD neural execution.
- 2026-08-31 (first AMD neural numeric primitive): a HIP `gfx1201` E4M3
  decode/encode kernel round-tripped all 1,966,080 RTX N0 output bytes and all 254
  finite codes exactly, including a negative mutation check. Arbitrary FP16
  rounding, MMA and the fused N0 operator remain open, so S6 is unchanged.
- 2026-08-31 (v14 exhaustive conversion oracle): a self-contained NVIDIA-driver
  package is ready to run the exact packed FP16-to-E4M3 PTX conversion over all
  65,536 half bit patterns. This one requirement-driven RTX run will close the
  remaining rounding/lane-order ambiguity before MMA reconstruction.
- 2026-08-31 (v14 returned/E4M3 closed): RTX 5070 produced all 65,536 conversion
  results. The first AMD comparison found only NVIDIA's positive-NaN
  canonicalization; after correcting it, RX 9070 XT matched every finite, Inf and
  NaN pattern with zero mismatches. N0 output regression remains exact. The next
  critical primitive is `mma.sync.m16n8k32` FP8 fragment/math behavior.
- 2026-08-31 (v15 MMA oracle ready): a three-file RTX package now targets the
  exact N0 FP8 MMA form with eight deterministic, complete 32-lane A/B/C/D
  register-fragment cases. One returned run will enable an RX 9070 XT software
  MMA comparison without another full Feature-18 capture.
- 2026-08-31 (v15 returned/FP8 MMA closed): RTX 5070 emitted eight complete
  m16n8k32 fragment cases. RX 9070 XT reconstructed the documented fragment
  matrices and matched all 1,024 FP16 results bitwise. The next isolated gaps in
  N0 are m16n8k16 f16 MMA and movmatrix before fused-kernel reconstruction.
- 2026-08-31 (v16 remaining primitives ready): f16 MMA and movmatrix are combined
  into one self-contained RTX numerical package. After its return, work shifts
  from instruction semantics to reproducing N0's captured scratch/weight/output
  dataflow on RX 9070 XT.
- 2026-08-31 (v16 returned): movmatrix is bitwise exact on RX 9070 XT, and f16
  MMA is bitwise exact for all functional-range/subnormal cases. The 337
  overflow-stress differences are retained under PTX's explicitly unspecified
  accumulation-order rule. Instruction micro-oracle acquisition is now sufficient
  to begin fused N0 scratch/weight/output reconstruction without another RTX run.
- 2026-08-31 (N0 semantic coverage audit, corrected): the original log contains
  740 diagnostics, not 736: 728 numerical statements have cross-vendor semantic
  oracles and 12 diagnostics are tuple/cache mechanics. Translator integration of
  the numerical operations remains explicitly at zero.
- 2026-08-31 (N0 tuple/cache lowering): semantics-preserving rewrites remove all
  12 mechanical diagnostics on a real RX 9070 XT ZLUDA module load. Exactly 728
  numerical diagnostics remain; function resolution and launch remain open while
  numerical lowering and captured N0 dataflow reconstruction proceed.
- 2026-08-31 (first AMD N0 dataflow stage): PTX address/shuffle recovery proves
  scratch is 4x4x32 MMA-tiled and output uses two HxWx16 planes. A real RX 9070 XT
  HIP kernel now runs the complete multi-tile downsample epilogue over the RTX
  scratch capture. Its raw bytes exactly match the independent reconstruction and
  reach 92.85% exact-or-adjacent E4M3 agreement with RTX (96.86% outside the
  capture's four leading all-zero tile rows). Full learned N0 remains open.
- 2026-08-31 (isolated N0 fully resolves): staged PTX lowering removes all 740
  original ZLUDA diagnostics. The 55.8 MB lowered entry completes a real module
  compile on RX 9070 XT and `cuModuleGetFunction` succeeds. This moves the blocker
  from instruction translation to valid texture/parameter provisioning and real
  launch/output verification; no launch is claimed yet.
- 2026-08-31 (full N0 runs on RX 9070 XT): valid texture, weights, scratch/output
  allocations and the relocated 264-byte ABI now drive the complete `80x48x32`
  learned N0 entry through ZLUDA. Launch and synchronization succeed; independent
  readback verifies 7,863,132/7,864,320 nonzero scratch bytes and
  1,965,587/1,966,080 nonzero output bytes. A private RTX 50 package is ready to
  run the original SM120 PTX over the identical inputs. S6 remains open until its
  returned tensors establish cross-vendor numerical correctness.
- 2026-08-31 (first RTX N0 reference attempt): the RTX 5070 and all payload hashes
  are correct, but its installed driver rejects the captured PTX 9.4 declaration
  at module load with CUDA code 222; no kernel ran. A replacement package lowers
  only the header declaration to PTX 8.7 and independently proves the remaining
  module text is byte-identical. Same-input tensor comparison remains pending.
- 2026-08-31 (same-input N0 comparison): both RTX 5070 and RX 9070 XT complete the
  full N0, but parity fails decisively: scratch/output correlations are -0.3637
  and -0.5426. RX output is repeatable across three bitwise-identical runs, so the
  failure is semantic rather than random. Direct execution of the generated FP8
  MMA and movmatrix lowering matches RTX bitwise, while f16 behavior retains only
  its documented overflow scope. The next reference captures the 3,072-byte
  checkpoint immediately before the first MMA to test texture/preprocessing.
- 2026-08-31 (pre-MMA checkpoint closed): RTX and RX 9070 XT match all 3,072
  exported bytes immediately before the first f16 MMA, split as 2,048 input A and
  1,024 weight B bytes with zero mismatches. The first-CTA texture/preprocessing,
  shared layout and weight load paths are therefore correct. The next package
  compares 4,096 output bytes immediately after the first 16 f16 MMAs.
- 2026-08-31 (S6 achieved): the post-f16 RTX result exposed an incorrect A-fragment
  lane/register/half mapping in the scalar m16n8k16 lowering. After correction,
  all 4,096 post-MMA bytes match RTX bitwise. The corrected full N0 runs on RX
  9070 XT and passes both same-input tensor gates: scratch/output correlations are
  0.99999852/0.99999880 and exact-or-adjacent E4M3 rates are
  99.9873%/99.9895%. Two AMD runs are bitwise deterministic. This is S6 PASS;
  full graph/frame integration remains S7/S8 work.
- 2026-08-31 (first downstream N1 executes): slot 2's isolated Swin entry reuses
  the N0-validated FP8/E4M3/movmatrix lowerings and adds one semantics-preserving
  release-store cache-hint rewrite. RX 9070 XT completes `40x24x32`, produces
  1,964,665 nonzero output bytes, and performs exactly 960 release publications
  into a sentinel-filled sync region. Output and sync hashes repeat bitwise. An
  RTX same-input package is ready; S7 remains open until its numerical return.
- 2026-08-31 (N1 slot-2 parity closed): the returned original-PTX RTX 5070 run
  matches RX 9070 XT over identical N0 output, weights and ABI. Correlation is
  0.99999888, NRMSE is 0.001495 and 99.9966% of E4M3 codes are exact-or-adjacent;
  the full sync buffer is bitwise exact and AMD repeats bitwise. N0 and its first
  dependent learned kernel are now numerically closed; S7 still requires slots
  3-155 and complete-frame integration.
- 2026-08-31 (slots 3-5 weight package ready): Feature-18 v17 captures three
  additional immutable 64 KiB model-resource views at the exact offsets consumed
  by graph slots 3/4/5. The expanded D3D12 readback self-test and all 46 Python
  tests pass; the verified package SHA-256 is `92563382...C4F7C0`. The next run is
  requirement-driven acquisition for the already translated downstream chain.
- 2026-08-31 (slots 3-5 execute on AMD): returned v17 weights drive the two
  chained Swin launches and ds-wait launch in tensor dependency order on RX 9070
  XT. Release counts are exactly 1,025/984; slot 5 produces its additional
  downsample tensor; two full runs are bitwise deterministic. One verified RTX
  package now carries all three identical-input original-PTX references. S7
  remains open pending numerical return and the rest of the graph.
- 2026-08-31 (slots 3-5 parity closed): the returned RTX 5070 references pass
  payload integrity and all four tensor gates. Main-output correlations are
  0.99999104/0.99999944/0.99999973; slot-5 FP16 downsample correlation is
  0.99999793; synchronization is byte-exact. Work advances to slot 6's `2h/64`
  successor family while S7 remains open.
- 2026-08-31 (slots 3-23 acquisition ready): the v18 capture harness resolves 21
  graph-observed downstream model views and passes its local D3D12 negative-test
  suite. One RTX run can now return every weight window for the 2h/4h/8h Swin
  families. Slot 6's parser and backend-only compatibility gaps are fully
  lowered; a 4.2 MB compact module loads/resolves, and a slot-3 regression is
  bitwise exact. Real slot-6 execution now waits only for the v18 weight capture.
- 2026-08-31 (2h slots 6-8 execute on AMD): returned v18 weights drive the real
  `32x2` slot 6/7/8 launches in dependency order on RX 9070 XT. Release counts
  are 240/273/252 and two runs are bitwise deterministic. Slot 9 correctly stops
  at a 308-byte capture shortfall instead of reading beyond evidence. The v19
  package expands all downstream views up to 1 MiB and is ready for one RTX run.
- 2026-08-31 (2h slots 6-9 AMD family closed): v19 returns a valid 267,776-byte
  slot-9 view. The ds-wait entry produces dense main/downsample FP16 outputs with
  zero allocation tails and repeats bitwise. Slots 6-9 are now deterministic on
  RX 9070 XT; one verified RTX package is ready to close numerical parity.
- 2026-08-31 (2h slots 6-9 parity closed): the returned RTX 5070 archive and all
  16 payload hashes pass. Four main FP16 tensors and slot 9's downsample tensor
  pass correlation/NRMSE/exact-or-adjacent gates, sync storage is byte-exact, and
  AMD repeats bitwise. Slots 1-9 are numerically closed in dependency order; work
  advances to slot 10's 4h/128 family while S7 remains open.
- 2026-08-31 (4h slots 10-15 execute on AMD): three isolated 4h/128 entries pass
  complete lowering, and six dependency-ordered `32x4` launches complete on RX
  9070 XT with exact release counts. Main/extra tails are zero and two runs are
  bitwise deterministic. A 22-payload RTX 5070 package is ready to close parity;
  S7 remains open.
- 2026-08-31 (one-shot RTX acquisition ready): tracer v3.0 generalizes bounded
  pre/post buffer capture to the complete 156-slot frame and content-addresses
  duplicate raw tensors. Static 156/43/15 coverage, D3D12 mutation self-test,
  Windows PowerShell 5.1 package validation, ZIP safety and 55 unit tests pass.
  One cloud run now replaces the remaining per-family reference runs; execution
  evidence is still pending its returned archive.
- 2026-08-31 (one-shot RTX acquisition complete): the valid 3.79 GiB return ZIP
  passes all 1,022 manifest size/SHA-256 records. Feature 18 completes 300/300;
  881 buffer windows cover slots 0-154 and the exact descriptor-object copy
  snapshot covers slot 155, giving strict 156/156 hybrid graph coverage. This
  closes bulk RTX acquisition and moves work to offline lowering/reconstruction
  of slots 10-154 followed by complete-frame AMD integration; S7 remains open.
- 2026-08-31 (4h slots 10-15 parity closed from full-graph evidence): an extended
  RX probe restores each native RTX before-state output/sync allocation instead
  of assuming zero initialization. All six FP16 outputs, the slot-15 E4M3
  downsample, release regions and two AMD repetitions pass. Slots 1-15 are now
  numerically closed; work advances to the contiguous 8h/256 slots 16-23 while
  S7 remains open.
- 2026-08-31 (8h slots 16-23 parity closed from full-graph evidence): all eight
  exact native-state launches execute twice on RX 9070 XT with bitwise sync and
  repeat determinism. FP16 correlations are 0.99987072-0.99999983 and NRMSE is
  0.000583-0.016082; slot 23's E4M3 extra tensor also passes. Slot 17's 94.9683%
  exact-or-adjacent rate is retained as an explicit advisory under PTX's
  unspecified f16 accumulation order/rounding semantics, while all normative
  numerical gates pass. Slots 1-23 are closed; Split 16h slots 24-56 are next
  and S7 remains open.
- 2026-08-31 (Split 16h slots 24-56 parity closed): all 33 exact-state launches
  execute twice on RX 9070 XT with bitwise synchronization and determinism. The
  12 apparent failures were caused by treating asynchronous producer-after
  snapshots as settled data; every affected AMD tensor equals the aliased next
  consumer input on RTX. With downstream-wait oracles, 30/33 main tensors are
  bitwise exact and slots 42/50/54 pass tight FP16 numerical gates; both E4M3
  outputs are bitwise exact. Slots 1-56 are closed, slots 57-104 are next, and
  integrated S7 remains open.
- 2026-08-31 (ViT 1D slots 57-98 dependency edges closed): nine final entries
  pass 54 strict lowering reports and all resolve on RX 9070 XT. The generalized
  arena ABI restores every observed activation/sync pointer; all 42 launches run
  twice with bitwise full-arena determinism. All 58 adjacent-consumer settled
  edges pass, 42 bitwise exactly and 16 QKV E4M3 edges with 100%
  exact-or-adjacent agreement. Twenty-four side accumulators remain diagnostic
  pending a settled decoder consumer. Work advances to decoder slots 99-154;
  full-frame S7 remains open.
- 2026-08-31 (decoder slots 99-154 exact-state study; corrected 2026-09-04): 19 final entries
  pass 171 strict reports and all 56 launches execute twice on RX 9070 XT with
  exact RTX inputs and bitwise AMD repeat determinism. All settled numerical
  edges pass; slot 132's 94.1711% E4M3 code-adjacency remains an explicit
  advisory. The originally reported 0.99999907 correlation and 0.001366 NRMSE
  incorrectly included constant alpha=1. The corrected RGB-only result is
  correlation `0.421357` and NRMSE `1.105764`, so slot 154 parity was not closed.
  Execution and repeat determinism remain valid; the numerical conclusion is
  superseded by E-108/R-110.
- 2026-08-31 (integrated 156-slot execution complete, numerical gate failed):
  two single-context RX 9070 XT runs execute all slots 0-155 with no RTX
  intermediate injection and reproduce the final bytes exactly. Slot 155 is a
  real GPU copy. Error first crosses the declared gate at slot 9 and the final
  RGB level is near 0.501 versus the RTX reference near 0.001, so S7 is
  deliberately not claimed. The execution integration itself is now closed.
- 2026-08-31 (first-divergence diagnostic ready): slot-3 sweeps rule out FP8
  mantissa width, reduction order, packed-FP16 translation, approximate
  reciprocal math and subnormal flushing as improvements. A one-run RTX 50
  package now exports all 256 MMA A/B/C/D fragments for direct comparison with
  the completed RX trace. Work is blocked only on that returned oracle, after
  which the first internal arithmetic mismatch can be corrected and the full
  integrated frame rerun.
- 2026-09-01 (MMA trace analyzer self-validated): the automatic candidate
  evaluator reproduces all 32,768 half-precision outputs of the existing RX
  trace bitwise with the current scalar model. This validates fragment decoding,
  lane mapping and model replay independently of RTX. Forty-three accumulation
  hypotheses are now ranked automatically from one cloud return. The safe result
  receiver verifies the NVIDIA manifest, trace size/nonzero count and SHA-256,
  rejects path traversal and altered payloads, and emits persistent receipt and
  comparison artifacts. All 87 Python tests pass. The RTX 50 trace result remains
  the sole external input required.
- 2026-09-01 (candidate-to-fix bridge verified on RX): 42/43 trace candidates now
  map directly to selectable PTX rewrites; only CPU-only exact summation is not
  executable. Four representative new semantics all load and run with the exact
  slot-3 state on RX 9070 XT. FP16-product and pairwise outputs equal baseline;
  RZ and staged-dot are numerically worse against RTX, so no guess is promoted.
  The returned internal RTX trace remains the selection oracle; 87 tests pass.
- 2026-09-01 (first returned MMA trace exact, divergent CTA localized): the RTX
  5070 and RX 9070 XT CTA `(0,0)` traces are bitwise identical for all A/B/C/D
  fragments and 256 MMA instructions, while instrumented full outputs retain the
  established cross-vendor mismatch. Reverse mapping the first differing output
  byte identifies CTA `(2,0)`, not the boundary CTA `(0,0)`. The generalized
  tracer has already produced a valid real-RX CTA `(2,0)` trace, and a matching
  one-run RTX package is ready with SHA-256
  `8DC7BAF79107F81D56C8D962A8949E98CBC171B2B7CC734C89CED284A164C143`.
  Its result will choose between MMA correction and epilogue instrumentation;
  S7 remains open and all 87 tests pass.
- 2026-09-01 (CTA `(2,0)` divergence moved upstream of MMA): the returned RTX
  trace first differs at MMA 216 in a single A activation code (`0x39` versus
  `0x38`); weights B and accumulator C are exact, and D does not differ until
  MMA 220. The code comes from `%r2106` after two packed-FP16 multiply stages,
  immediately before the previously validated E4M3 conversion. A 28-register
  checkpoint now runs without perturbation on RX 9070 XT, and the matching
  one-run RTX package has SHA-256
  `AAFA3ADB563D845D86FB251C2F9D3BB83B5AD1796F8FB8189CBCDC9FC1CA287A`.
  All 90 tests pass; S7 remains open.
- 2026-09-01 (rsqrt attribution corrected by full trace): the returned 16-block
  oracle proves that all 944 equal-input samples round to identical half outputs
  on RTX and AMD; one-ULP FP32 approximation differences are erased. Every
  effective scale mismatch begins with an already-different half input, so rsqrt
  is not the root cause. Local replay from exact activations places the remaining
  difference in the preceding square/packed-half butterfly reduction. A complete
  34-register reduction package is ready with SHA-256
  `81FE297A717DA2F6AA550C1DB8D0F2BEE1569438FB96287CD838F4E6C92C9F8E`;
  all 96 tests pass and S7 remains open.
- 2026-09-05 (AMD-native output head and D3D12 bridge pass): the clean-room
  slot-154 path now runs activation fusion, all 256 FP8 MMAs, stable attention,
  all 16 FP16 tail MMAs, exact 8x8 spatial scatter, residual composition and
  D3D12 RGBA16F texture staging in one 9/9 PASS pipeline on RX 9070 XT. The
  selected RTX store formula is exact; native final RGB mean/max absolute error
  is `0.000159/0.000764`. The D3D12/HIP roundtrip has zero component mismatches
  and a working imported-fence wait/signal chain. This is a complete native
  output-head operator, but S7/S8 remain open because its activation input is
  still captured rather than produced by the native preceding decoder.
- 2026-09-05 (resident output head crosses the D3D12 boundary): the same
  activation-to-RGBA path now runs in one process with every neural intermediate
  resident on the RX 9070 XT. Its 1,016,064 residual half values are bitwise
  identical to the accepted modular pipeline. A second single-process executable
  imports a shared D3D12 heap and fence, consumes a staged 640x360 RGBA16F base
  texture, and returns a D3D12 texture with zero mismatches across 921,600
  components. No RTX trace is read at runtime. The measured unoptimized output
  head is 20.24 ms; live feature-18 resource binding and upstream native decoder
  activations remain required before S7/S8 or game readiness can be claimed.
- 2026-09-05 (slot-154 NVAPI dispatch lowering passes): `nvapi_amd` now
  recognizes `cc_tinlayout_fused_post_block_swin_1h_32_fp8`, validates its real
  `81x49x1`/`32x1x1` launch and 184-byte parameter ABI, translates main, skip,
  head, blend and linearized output addresses from two registered D3D12 buffers,
  and runs the resident neural output head. The resulting 640x360 RGBA16F bytes
  match the standalone resident baseline exactly. Existing clear and texture-copy
  self-tests still pass. This proves the NVAPI function-dispatch bridge, but the
  original live ABI supplies a surface object (`0x9802`) at parameter offset 16;
  descriptor-resource resolution plus a queue-ordered staging copy remains the
  next live-integration gate.
- 2026-09-05 (real slot-154 surface ABI and automatic resource binding pass):
  offset 16 now remains an independent CUDA surface-object token. With the
  opt-in `MODULE_TRACE_AMD_INTEROP=1` path, `module_trace` promotes only the
  observed 27,807,744-byte activation and 147,719,680-byte model buffers to
  shared committed resources, imports both into `nvapi_amd`, and transfers the
  UAV descriptor-to-resource identity automatically. The RX 9070 XT test records
  one staging-to-`R16G16B16A16_FLOAT` texture copy and matches all 1,843,200
  reference bytes. This closes address and descriptor binding for the output
  head. S7/S8 remain open: HIP currently executes synchronously while the D3D12
  command list is still being recorded, and slots 0-153 have not yet been wired
  as a native producer of the live slot-154 activation.
- 2026-09-05 (same-list suffix extended through final projection): native DXIL
  final projection matches all 16,257,024 oracle bytes on RX 9070 XT. Final
  projection, FP16 tail, and RGBA16F surface composition also execute together
  on one D3D12 command list without HIP; projection remains exact and the final
  surface retains the accepted mean/max error `1.41295863e-9 / 3.05175781e-5`.
  The current backward migration boundary is attention softmax/V. S7/S8 remain
  open because the upstream graph and live slot dispatch are not yet D3D12-
  resident end to end.
- 2026-09-05 (same-list suffix extended through attention): DXIL softmax/V
  differs in only 96/8,128,512 half values from the full HIP attention oracle.
  The attention, final-projection, tail and surface stages run as four dispatches
  on one command list; the final 640x360 RGBA16F surface has mean/max error
  `1.89543546e-9 / 0.000163555145` and no non-finite values. The active migration
  boundary is now Q/K/V preparation plus QK scoring; S7/S8 remain open.
- 2026-09-05 (complete output head moved to one D3D12 list): the new full
  executor consumes the activation arena and original weights, records all 11
  reconstructed output-head dispatches on one command list, and produces the
  final RGBA16F surface without HIP or runtime RTX traces. First128 and
  MMA128-175 are exact; final surface mean/max error is
  `3.4076811e-6 / 0.000862598419`. A repeat run is identical and all 222 tests
  pass. The per-layer output-head migration loop is closed. The remaining S7
  gate is live slot-154 resource/command-list binding, followed by upstream graph
  and real multi-frame game validation.
- 2026-09-05 (exact output write-back route validated offline): the post-FFX
  boundary hook now copies RGBA16F output to a private texture and writes those
  exact bytes back on the same direct command list before restoring the proven
  readable state. FSR3 and FSR4 isolated controls are byte-identical to their
  baselines, block an early fence poll, and emit exactly one canonical-identity
  submission chain with zero D3D12 errors/warnings. Fresh immutable stage
  `results/20260905_150435_007_ffx_output_roundtrip_live_stage/` requires both
  output-copy and output-roundtrip controls before live attachment; all 471
  Python tests pass. This is a no-op transport proof only: no replacement pixels
  or reconstructed-network output have been supplied. A normal game exit and
  fresh process are required for the one-shot live roundtrip.
