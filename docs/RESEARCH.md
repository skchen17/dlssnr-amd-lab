# RESEARCH.md — Fact Verification Log

Query date: **2026-08-30**. All entries record source URL, version/commit where found, and
explicitly distinguish confirmed fact from community claim.

Terminology separation enforced throughout this project:

- **DLSS 5** = officially announced product (fall 2026)
- **Leaked DLSSNR runtime** = pre-release `nvngx_dlssnr.dll` circulating in the community
- **RenoDX addon** = community ReShade addon `renodx-dlss5.addon64` (open source ecosystem, but ships closed addon binary)
- **DLSS Super Resolution** = classic DLSS 2-style upscaler (`nvngx_dlss.dll`)
- **NGX** = NVIDIA NGX loader/feature API (`NVXG_*` entry points)
- **Streamline** = NVIDIA plugin framework (SL)
- **NVAPI** = `nvapi64.dll`, entry via `nvapi_QueryInterface`
- **CUDA Driver API** = `nvcuda.dll` (`cu*` functions)

---

## 1. NVIDIA official DLSS 5 / Neural Rendering

**Status: CONFIRMED announced, not yet generally released.**

- NVIDIA press release: "NVIDIA DLSS 5 Delivers AI-Powered Breakthrough in Visual Fidelity for Games" — arriving **Fall 2026**, introduces a real-time neural rendering model that infuses pixels with photoreal lighting and materials.
  - Source: nvidianews.nvidia.com/news/nvidia-dlss-5-delivers-ai-powered-breakthrough-in-visual-fidelity-for-games (checked 2026-08-30)
- Shown at GTC 2026 (Jensen Huang keynote); COMPUTEX 2026 GeForce announcements mention DLSS Ray Reconstruction as neural rendering for all RTX GPUs.
- NVIDIA driver traces: NVIDIA Inspector found "DLSS NR" profile settings in new drivers (overclock3d.net, videocardz.com, checked 2026-08-30). Feature cannot be forced in games yet via driver profiles.
- The official DLSS SDK repo (github.com/NVIDIA/DLSS) currently documents DLSS Super Resolution / Frame Generation / Ray Reconstruction; Neural Rendering feature id 18 is what the community addon targets.

**Conclusion:** DLSS 5 NR is officially announced; the leaked runtime predates official release and is an early development build.

## 2. RenoDX community DLSS 5 NR implementation

**Status: ACTIVE, community-verified working on RTX 50-series.**

- The NR path today runs through a ReShade addon: `renodx-dlss5.addon64` + `nvngx_dlssnr.dll` placed next to the game exe.
- Distribution channels: RHI installer (github.com/RankFTW/RHI/releases, "downloads and deploys them for you"), RenoDX Discord, Nexus Mods guide "Applying RR and DLSS 5 RenoDX for the games" (nexusmods.com/site/mods/2224).
- clshortfuse/renodx repo is the RenoDX framework; the DLSS5 addon itself is a closed binary distributed with it (community claim, not verified by this project yet — addon content is not publicly source-audited).
- Reddit megathread "Unofficial DLSS 5 Neural Rendering Mods Comparison" (r/nvidia, checked 2026-08-30) tracks multiple mod variants.

## 3. Current `nvngx_dlssnr.dll` version and known invocation

**Status: leaked build identified; exact version number must be read from the user's local file (PE FileVersion) — not yet available.**

- Origin per community analysis (ithome.com/0/995/821.htm, tomshardware): leaked from **NBA 2K27**; assessed as an early development build, likely never compiled for Ada (RTX 40) originally — initially Blackwell-only.
- Invocation known from DLSS5-Feeder docs (github.com/jlrouzies-fr/DLSS5-Feeder, checked 2026-08-30):
  - The NR addon hooks a D3D12 `NGX_D3D12_EVALUATE_DLSS` call and inserts its neural pass;
  - It reports "feature 18 created … inline feature 18 evaluation succeeded" — **feature id 18 = Neural Rendering inline feature**;
  - It requires `nvngx_dlss.dll` (DLSS SR runtime) next to the game.

## 4. RTX 40 / Ada DLSSNR patch progress

**Status: CONFIRMED working via CUDA binary replacement.**

- tomshardware (checked 2026-08-30): "DLSS 5 has already been ported to work on RTX 4000 Series cards — incompatible CUDA instructions get patched to work on previous-gen hardware".
- wccftech: modders enabled NR on RTX 40 through "a simple CUDA binaries swap", ~39% perf reduction in Control.
- windowsforum: unofficial patch runs on RTX 4080 / 4090 demos.
- **Key inference for this project:** the GPU payload inside the DLL is a locatable, replaceable CUDA binary container (fatbin/cubin). Someone extracted the Blackwell-only SASS kernels and replaced/patched them for Ada. This strongly suggests:
  1. kernels ship as SASS (per-arch cubins) — Ada patch had to *rebuild* SASS, implying SASS-to-SASS patching or recompile, not PTX JIT;
  2. whether PTX exists alongside is still UNKNOWN and is exactly our Phase 3 question.

## 5. vosen/ZLUDA

**Status: ACTIVE. Latest release v7-preview.10 (2026-08-26), commit `9c8b43f242985150f86a7f485218b7b82c3e96ca`.**

- Release cadence (checked 2026-08-30): v7-preview.1 (2026-07-01) … v7-preview.10 (2026-08-26). v6 stable = "ZLUDA 5" (Q3 2025).
- Q1&Q2 2026 blog "back to the roots" (vosen.github.io/ZLUDA/blog/zluda-update-q1q2-2026/): Windows loader `zluda.exe` made more robust, auto-handles performance libraries; new workloads PhysX (pre-alpha) and Blender.
- RDNA4/Windows known issues (checked 2026-08-30):
  - GitHub discussion: "ZLUDA support for ROCm HIP SDK 6.4.3 on Windows (gfx1201)" — indicates gfx1201 work is being tracked.
  - Community reports: ZLUDA on RX 9070 XT requires ROCm >= 6.4.x; older ROCm versions fail (r/StableDiffusion).
  - mayocream/koharu issue #754: bundled ZLUDA crashes on startup with ROCm 7.1 (version skew issue).
- ZLUDA FAQ claim "ability to enqueue HIP kernels into Direct3D command list" — NOT verified to be a public API; Phase 2 must treat it as unverified claim.

## 6. OptiScaler / FakeNVAPI

**Status: exists as separate repo `optiscaler/fakenvapi`.**

- GitHub optiscaler/fakenvapi (checked 2026-08-30): "NVAPI library for spoofing an Nvidia GPU". Limited to AMD RDNA1+ cards and Windows; unreliable overlay readings.
- Current scope: vendor spoofing so games call AMD Anti-Lag 2 instead of Reflex etc. It does NOT implement NVAPI D3D12 CUBIN/CUDA extension APIs today — any DLSSNR compatibility layer on this base would be new code.
- OptiScaler main repo (optiscaler/OptiScaler) remains the FSR4/XeSS/DLSS replacement scaler — explicitly off-limits as a "success" substitute.

## 7. NVIDIA/nvapi

- github.com/NVIDIA/nvapi — official headers/sample repo; defines `nvapi_QueryInterface` ordinal-based dispatch and public `NvAPI_D3D12_*` surface. The CUBIN/CuModule-family functions (`NvAPI_D3D12_CreateCubinComputeShader*`, `NvAPI_D3D12_LaunchCuKernelChain*`, etc.) are present in the nvapi headers as known ordinals — the exact set actually used by DLSSNR must be determined by Phase 4 trace, not assumed.

## 8. jlrouzies-fr/DLSS5-Feeder

**Status: ACTIVE, MIT, highly relevant — effectively the reference implementation for our nr_host design.**

- Source: github.com/jlrouzies-fr/DLSS5-Feeder (README read in full, 2026-08-30).
- What it does: builds a synthetic DLAA contract (color + ReShade depth + estimated motion vectors) and runs a genuine `NGX_D3D12_EVALUATE_DLSS` on a private D3D12 device; the NR addon hooks that evaluate and inserts the neural pass ("cannot tell the contract is synthetic").
- Verified working in 5 games (Metro 2033 Redux 64-bit D3D11 4K DLAA+NR; LotR WotN 64-bit D3D12 120fps; Splinter Cell Blacklist 32-bit via cross-process host64; BioShock Remastered; Fable Anniversary via dgVoodoo2).
- Architecture details relevant to us:
  - D3D11→D3D12 transport uses **shared NT handles + shared fence** (same technique our Phase 2 needs, but D3D11↔D3D12, not D3D12↔HIP);
  - 32-bit path spawns `dlss5-feed-host64.exe` and passes handles via named pipe + `DuplicateHandle` — proof that the NR feature can run in a standalone host process with a minimal D3D12 swap chain window;
  - NGX calls wrapped in SEH; faulting command lists discarded; NGX re-init after repeated failures;
  - Requires: ReShade 6.8+, `renodx-dlss5.addon64` + `nvngx_dlssnr.dll`, `nvngx_dlss.dll`; build needs NGX SDK + MSVC v143/v145 + Windows SDK; contains a `spike/` standalone shared-resource proof.
- Config knobs worth reusing: `reset_every` (per-frame NGX Reset diagnostic), `warmup_rebuild` (workaround for addon latching STANDBY on first create), `create_delay` (addon arms NGX hooks asynchronously — calling too early crashes).

## 9. NIGos/dlss5-dx11-bridge

- GitHub NIGos/dlss5-dx11-bridge (checked 2026-08-30): MIT ReShade addon; lets D3D12-only DLSS 5 NR addon run in DX11 games. Its release notes document a failure mode: "A module that exports the NGX entry points but is not the driver's loader is named when the D3D12 session faults" (Prey 2017 + OptiScaler case) — relevant caution for any NGX-entry-point-exporting shim we build.

## 10. NIGos/dlss5-d3d12-fix

- GitHub NIGos/dlss5-d3d12-fix (checked 2026-08-30): "DLSS 5 D3D12 Mip Fix" — ReShade addon making a DLSS 5 NR addon work in a D3D12 game whose DLSS output carries a mip chain. Confirms NR output handling has mip-related edge cases.

## 11. clshortfuse/renodx

- RenoDX framework repo (checked 2026-08-30): active; hosts the ecosystem the DLSS5 addon plugs into. The closed `renodx-dlss5.addon64` binary is what hooks DLSS evaluate; the framework itself is open. Exact license/commit of the DLSS5 addon distribution to be pinned once acquired.

## 12. AMD HIP Windows external resource interoperability

**Status: API EXISTS. D3D12-specific handle type support requires runtime verification (Phase 2).**

- HIP docs "External resource interoperability" (rocm.docs.amd.com/projects/HIP, checked 2026-08-30):
  - `hipImportExternalMemory` / `hipExternalMemoryGetMappedBuffer` / `hipImportExternalSemaphore` / `hipWaitExternalSemaphoresAsync` / `hipSignalExternalSemaphoresAsync` exist.
  - Windows examples use `hipExternalMemoryHandleTypeOpaqueWin32KMT` (from Vulkan). The enum also defines `hipExternalMemoryHandleTypeOpaqueWin32` (NT handle) — the one D3D12 `CreateSharedHandle` produces. Whether the Windows HIP runtime accepts a D3D12 NT handle is UNVERIFIED — this is the core Phase 2 experiment.
  - Official example is Vulkan interop (rocm-examples HIP-Basic/vulkan_interop). No official D3D12 sample found — absence noted, not assumed.
- HIP SDK for Windows 7.2.0 is the current line (rocm release notes, checked 2026-08-30). Known issue: "Windows HIP SDK VS Plugin Failing on 9070 XT" (ROCm/rocm-install-on-windows issue #127) — environment pitfalls exist on exactly our target GPU.

## 13. AMD RDNA4 FP8/WMMA ISA and gfx1200/gfx1201 HIP status

**Status: ISA support CONFIRMED; tooling support has documented gaps.**

- GPUOpen "Using the Matrix Cores of AMD RDNA 4 architecture GPUs" (checked 2026-08-30): WMMA intrinsics how-to for RDNA4 (`v_wmma_*`, including f16/bf16 and **FP8 variants** on gfx1200/gfx1201).
- LLVM AMDGPU backend (llvm.org/docs/AMDGPUUsage.html): gfx12 has FP8 and BF8 instructions + FP8/BF8 conversion instructions.
- Known gap: ROCm/ROCm issue #6025 "ROCm Documentation Gap: WMMA Output Lane Mapping for gfx12" — ISA reference documents `v_wmma_f32_16x16x16_f16` encoding and HIP intrinsic signatures, but output lane mapping documentation is incomplete. Correctness validation in hip_probe must not trust docs blindly; use numerical ground truth.
- ZLUDA discussion confirms gfx1201 + HIP SDK 6.4.3 on Windows is a tracked support target.

---

## Summary table

| # | Topic | Verified fact | Open question |
|---|---|---|---|
| 1 | DLSS 5 official | Announced, Fall 2026 | GA feature id set |
| 2 | RenoDX NR | Working community addon, feature id 18 | addon internals closed |
| 3 | nvngx_dlssnr.dll | Leaked from NBA 2K27, early build | exact version on user machine |
| 4 | RTX40 patch | Works via CUDA binary swap | whether payload has PTX |
| 5 | ZLUDA | v7-preview.10, 9c8b43f, 2026-08-26 | D3D command-list enqueue claim unverified |
| 6 | FakeNVAPI | optiscaler/fakenvapi exists, vendor spoof only | no CUBIN API impl today |
| 7 | NVIDIA/nvapi | Headers public, ordinals known | which set DLSSNR uses |
| 8 | DLSS5-Feeder | MIT, synthetic DLAA contract proven | — |
| 9 | dx11-bridge | MIT, NT handle + fence transport | NGX loader naming pitfall |
| 10 | d3d12-fix | Mip chain edge case confirmed | — |
| 11 | renodx | Framework open, addon closed | license pin pending |
| 12 | HIP ext interop | API exists, Win32 KMT proven | D3D12 NT handle import unverified |
| 13 | RDNA4 FP8/WMMA | ISA + LLVM support confirmed | lane-mapping doc gap (ROCm#6025) |

**Proprietary file prerequisite (user must provide, cannot be downloaded by this project):**

| File | Legal source |
|---|---|
| `nvngx_dlssnr.dll` | from a legally owned DLSS-5-shipping game install (e.g. NBA 2K27) |
| `nvngx_dlss.dll` | same game install, or official NVIDIA DLSS update package |
| `renodx-dlss5.addon64` (optional) | RHI installer / RenoDX Discord (community distribution) |
| NVIDIA NGX SDK (for building nr_host) | developer.nvidia.com account (free registration) |
