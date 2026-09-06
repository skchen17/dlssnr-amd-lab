# Third-party provenance

## AMD FidelityFX API headers (2026-09-05)

- Repository: https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
- Tag `v1.1.3`, commit `54fbaafdc34716811751bea5032700e78f5a0f33`.
- Source: `ffx-api/include/ffx_api/`; local: `third_party/fidelityfx-api-1.1.3/`.
- Files: `ffx_api.h`, `ffx_api_types.h`, `ffx_upscale.h`, `dx12/ffx_api_dx12.h`.
- MIT license; original AMD copyright and permission notices retained.
- Changes: LF normalization and one terminal newline; `FFX_API_ENTRY` is guarded
  by `#ifndef` to let observer consumers suppress DLL export declarations. No
  types, layouts or enum values changed.
- Purpose: compiler-checked observer ABI, not proof of actual game dispatch ABI.
  No executable, proprietary model or game content is included.
- The DX12 header was added from the same pinned commit via GitHub raw HTTPS
  on 2026-09-05 for the independent context/dispatch/readback probe. Its backend
  descriptor and resource-format helper are used without ABI modifications.

Fetched via GitHub REST API (git HTTPS blocked on this machine). Date: 2026-08-30 19:36:47 +08:00

## NVIDIA/DLSS
- repo: https://github.com/NVIDIA/DLSS
- commit: a291cc7d2cc642a51566f3dfd5376f635cd1b284 (2026-06-23T16:00:25Z)
- license: NOASSERTION
- purpose: official NGX public headers (nvsdk_ngx*.h) for ABI-correct nr_host

## jlrouzies-fr/DLSS5-Feeder
- repo: https://github.com/jlrouzies-fr/DLSS5-Feeder
- commit: 80abd23fa41a562960208dd6f4094fa62b707f43 (2026-08-30T10:38:19Z)
- license: NOASSERTION
- purpose: proven genuine-DLSS/DLAA contract reference (synthetic DLAA -> DLSS5 addon -> feature 18)
- Local extension 2026-09-05: `host/teacher_sequence.h` and the optional
  `--teacher-sequence` host mode accept explicit real frame resources/metadata,
  verify uploaded FP16 color and read back output candidates after queue fences.
  Default test/serve arguments remain supported; Evaluate gained defaulted
  exposure/jitter arguments and submission/fence failure checks were hardened.
  This modified research host is not upstream code and is not an accepted NR
  teacher merely because its public DLAA call succeeds. No vendor model/runtime
  files are included in the candidate package.

## NVIDIA/NVAPI
- repo: https://github.com/NVIDIA/nvapi
- commit: cd6918f60b3c9a0476fdfe7e89bb32330602049d (R610-Developer SDK,
  2026-06-24)
- license: MIT for headers/import libraries; see `third_party/nvapi/License.txt`
- purpose: authoritative five-call D3D12 CuModule/CuFunction ABI and interface IDs

