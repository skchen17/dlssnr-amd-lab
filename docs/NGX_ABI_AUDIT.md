# NGX ABI Audit — reconstructed declarations vs. official NVIDIA headers

- Date: 2026-02 (Round 2, Phase A / R1)
- Official source: `third_party/nvidia-dlss/` (NVIDIA/DLSS @ commit `a291cc7d2cc6`,
  fetched via GitHub API; provenance in `third_party/PROVENANCE.md`).
  Headers audited: `include/nvsdk_ngx.h` (706 L), `include/nvsdk_ngx_defs.h` (860 L),
  `include/nvsdk_ngx_params.h` (117 L).
- Reconstructed target: `tools/nr_host/nr_host.cpp` L45-L63 (Round 1 state).
- Verdict: **6 signature/semantic defects found; 5 are ABI-breaking on x64.**
  Every Round 1 dynamic run went through a wrong calling surface. Any result
  produced through the reconstructed ABI (including the `0xBAD00001`
  observation) is **NOT valid evidence about NVIDIA's vendor gate** until
  re-run through the official ABI.

Legend per item: **Previous state / Official signature / Runtime consequence / Fix / Validation.**

---

## E1 — `NVSDK_NGX_D3D12_Init` arity, types and missing trailing argument

**Previous state** (`nr_host.cpp` L50-L52):

```cpp
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_Init)(
    unsigned int appId, const wchar_t* appDataPath, ID3D12Device* device,
    NVSDK_NGX_Parameter** outParams);
// called as: pInit(0x4C424C21u, L".", device.Get(), &params);
```

**Official signature** (`nvsdk_ngx.h` L169, C++/non-snippet build):

```cpp
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Init(
    unsigned long long InApplicationId,
    const wchar_t *InApplicationDataPath,
    ID3D12Device *InDevice,
    const NVSDK_NGX_FeatureCommonInfo *InFeatureInfo = nullptr,
    NVSDK_NGX_Version InSDKVersion = NVSDK_NGX_Version_API); // = 0x0000015
```

**Runtime consequence** — two defects in one call:
1. The 4th reconstructed argument (`NVSDK_NGX_Parameter**`, a *non-const pointer
   to a pointer*) lands in the official `const NVSDK_NGX_FeatureCommonInfo*`
   slot. NGX dereferences it as a struct
   `{NVSDK_NGX_PathListInfo; InternalData*; LoggingInfo}` → reads the caller's
   uninitialized local `NVSDK_NGX_Parameter* params` as `PathListInfo.Path`,
   treats garbage as `Path`/`Length` → wildcard memory reads / crash /
   undefined behavior. Official Init returns no parameter map; the call was
   waiting for an output that never comes.
2. The official 5th argument (`InSDKVersion`, r9d) is never supplied →
   whatever stack garbage sits there is read as the API version.
   `appId` type `unsigned int` vs `unsigned long long` is harmless in the
   x64 calling convention (register zero-extension) but wrong by contract.

**Fix** — call the real 5-argument form; no output parameter. Application ID
remains the synthetic lab value (`0x4C424C21`), explicitly recorded as
*self-assigned, not NVIDIA-issued*; if NGX rejects it, fall back to
`NVSDK_NGX_D3D12_Init_with_ProjectID("…GUID…", NVSDK_NGX_ENGINE_TYPE_CUSTOM,
"1.0", …)` (GUID-like project ID, `nvsdk_ngx.h` L247).

**Validation** — Phase B `tools/ngx_abi_probe` asserts the loaded function
pointer type against the official typedef (`std::is_same_v`); runtime check in
Phase C: Init result decoded via `GetNGXResultAsString`, parameter map obtained
separately (see E2).

---

## E2 — Parameter-map lifecycle bypassed entirely

**Previous state** — `NVSDK_NGX_Parameter` treated as an opaque forward
declaration; the host expected `Init` to *output* a parameter map and never
called any official allocator/deallocator.

**Official interface** (`nvsdk_ngx.h` L380-L451, `nvsdk_ngx_params.h`):

```cpp
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters); // pre-populated, app-owned
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** OutParameters);      // empty, app-owned
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* InParameters);
// NVSDK_NGX_D3D12_GetParameters is deprecated (NGX_ENABLE_DEPRECATED_GET_PARAMETERS)
// and only valid as fallback for drivers <= 445.
```

`NVSDK_NGX_Parameter` is a C++ abstract interface (8 `Set` + 8 `Get` virtuals +
`Reset()`, `nvsdk_ngx_params.h` L53-L74); instances are allocated and destroyed
by NGX only through the functions above. Freeing them with `delete`/`free` is
forbidden.

**Runtime consequence** — with E1 combined, the host had *no* parameter map at
all; any future `Set(...)` call would dereference a never-initialized pointer.
Additionally the official `*_Feature_Create_Params` struct layout
(`nvsdk_ngx_params.h` L27-L43) and the documented order
(Init → AllocateParameters → set Width/Height/… → GetScratchBufferSize →
CreateFeature) could not be honored.

**Fix** — Phase C pipeline: `Init → AllocateParameters` (or
`GetCapabilityParameters` when querying `SuperSampling.Available` etc.) →
`Set` via official parameter-name macros (`nvsdk_ngx_defs.h` L647+) →
`GetScratchBufferSize` → `CreateFeature` → … → `DestroyParameters` before
`Shutdown1`.

**Validation** — probe asserts prototypes; Phase C records a per-step
machine-readable status (`params_allocate: SUCCESS/FAIL…`).

---

## E3 — `NVSDK_NGX_D3D12_ReleaseFeature` fabricated extra first argument

**Previous state** (`nr_host.cpp` L60-L61):

```cpp
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_ReleaseFeature)(
    ID3D12Device* device, NVSDK_NGX_Handle* handle);
```

**Official signature** (`nvsdk_ngx.h` L574):

```cpp
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle *InHandle);
```

**Runtime consequence** — x64 fastcall: reconstructed call puts `device` in
rcx, `handle` in rdx. Official implementation reads rcx as the handle →
treats the *device pointer* as a feature handle → `FAIL_FeatureNotFound` at
best, heap corruption at worst.

**Fix** — single-argument form; device association is implicit from `Init`.

**Validation** — probe `std::is_same_v` on the pointer type; Phase C release
step reports official `NVSDK_NGX_Result`.

---

## E4 — Shutdown export name mismatch (`Shutdown` vs `Shutdown1`)

**Previous state** (`nr_host.cpp` L53):

```cpp
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_D3D12_Shutdown)(ID3D12Device* device);
// resolved by name "NVSDK_NGX_D3D12_Shutdown" (planned)
```

**Official** (`nvsdk_ngx.h` L277-L282): the zero-argument
`NVSDK_NGX_D3D12_Shutdown(void)` is declared only under
`NGX_ENABLE_DEPRECATED_SHUTDOWN` and is deprecated; the supported export is

```cpp
NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Shutdown1(ID3D12Device *InDevice); // nullptr => all instances
```

**Runtime consequence** — the exported symbol on modern feature DLLs is
`NVSDK_NGX_D3D12_Shutdown1`; looking up `NVSDK_NGX_D3D12_Shutdown` may fail
(`GetProcAddress` → NULL → host aborts or silently skips cleanup), or an
unrelated symbol could be hit if names ever collide across DLL generations.

**Fix** — resolve `NVSDK_NGX_D3D12_Shutdown1` and call with the Init device
(or nullptr). Probe enumerates the DLL export table and records which of
`Shutdown` / `Shutdown1` actually exist (export name check, Phase B).

**Validation** — probe export-name report in `results/<ts>/ngx_abi_test.json`.

---

## E5 — Command-list recording / execution semantics missing

**Previous state** — reconstructed `CreateFeature`/`EvaluateFeature` accepted a
command-list pointer but the host never opened/recorded/executed one; the
Round 1 skeleton called neither (no D3D12 command infrastructure existed).

**Official contract** (`nvsdk_ngx.h` L508-L513, L649-L653): the D3D12
command list passed to `CreateFeature`/`EvaluateFeature` **must** be
- open and recording,
- node mask covering the Init device,
- later executed by the app on a **non-copy** command queue
  (`ExecuteCommandLists` + fence wait) — NGX only *records* into it.

**Runtime consequence** — without a recording command list and explicit
`ExecuteCommandLists`, any `EvaluateFeature` success code records nothing on
the GPU; a "PASS" without GPU execution evidence is meaningless. Conversely,
passing a closed/execute-only list is undefined behavior inside NGX.

**Fix** — Phase C: dedicated direct-queue command allocator/list;
`Reset → EvaluateFeature → Close → ExecuteCommandLists → WaitForFence` around
every evaluate; GPU-side evidence via readback hash.

**Validation** — Phase C `evaluate.json` contains `cmdlist_state` (recorded,
executed, fence value) and readback hash; on NVIDIA reference machine the
readback must differ from input.

---

## E6 — Feature-ID domain: public enum vs. private `feature 18`

**Previous state** (`nr_host.cpp` L63):

```cpp
static const unsigned int NGX_FEATURE_ID_NEURAL_RENDERING = 18;  // per DLSS5-Feeder
```

used directly as the `featureId` of `CreateFeature`.

**Official domain** (`nvsdk_ngx_defs.h` L186-L237): `NVSDK_NGX_Feature` is an
enum; public members end at `RayReconstruction = 13`; **14–18 are
`NVSDK_NGX_Feature_Reserved14..18`** — reserved placeholders, not public
features. `CreateFeature` takes `NVSDK_NGX_Feature`, not `unsigned int`.

**Runtime consequence** — passing `18` through the *public* ABI is by
definition "reserved/unknown feature"; observed failures are uninterpretable
without acknowledging this is a **private ABI** observed in DLSS5-Feeder
(`feature 18 created`, MIT, commit `80abd23fa41a`). With the reconstructed
ABI already broken (E1/E3), Round 1 could not distinguish "vendor gate" from
"wrong calling surface".

**Fix** — code layering (this commit): everything official comes from
`third_party/nvidia-dlss/include/`; the value 18 lives **only** in
`tools/nr_host/private_ngx_compat.h` under a `PRIVATE ABI — inferred` banner.
Stage A (Phase D) creates public features (`SuperSampling`=1 for DLSS, DLAA
via `PerfQualityValue=DLAA`) first; feature 18 is exercised only in Stage B
with the DLSS5 addon, and results are labeled `PRIVATE_ABI` in all artifacts.

**Validation** — `grep` gate: no literal feature-ID 18 outside
`private_ngx_compat.h` + its single include site; Stage B artifacts carry the
label.

---

## Reference facts established by this audit

| Fact | Official value | Source |
|---|---|---|
| `NVSDK_NGX_Result_Success` | `0x1` | `nvsdk_ngx_defs.h` L99 |
| `NVSDK_NGX_Result_Fail` base | `0xBAD00000` | L102 |
| **`0xBAD00001`** | `NVSDK_NGX_Result_FAIL_FeatureNotSupported` (`Fail \| 1`) | L106 — **confirmed** |
| Success test | `NVSDK_NGX_SUCCEED(v) == ((v & 0xFFF00000) != 0xBAD00000)` | L183 |
| API version macro | `NVSDK_NGX_VERSION_API_MACRO = 0x0000015` | L56 |
| Calling convention | `NVSDK_CONV = __cdecl` (MSVC); C linkage, undecorated exports | L42-L46 |
| Result enum range | `Success=0x1`, `0xBAD00000`…`0xBAD00012` (18 failure codes) | L96-L178 |
| Feature enum | 0–13 public; 14–18 Reserved; 32764–32766 SDK/Core/Unknown | L186-L237 |
| `PerfQuality_Value_DLAA` | `5` | L259 |
| DLSS feature flags | `IsHDR=1<<0`, `MVLowRes=1<<1`, `MVJittered=1<<2`, `DepthInverted=1<<3`, `AutoExposure=1<<6`, `AlphaUpscaling=1<<7` | L286-L300 |
| Export names (undecorated) | `NVSDK_NGX_D3D12_Init`, `_Init_with_ProjectID`, `_Shutdown1`, `_AllocateParameters`, `_GetCapabilityParameters`, `_DestroyParameters`, `_GetScratchBufferSize`, `_CreateFeature`, `_ReleaseFeature`, `_EvaluateFeature`, `_EvaluateFeature_C`, `_GetFeatureRequirements`, `NVSDK_NGX_UpdateFeature`, `GetNGXResultAsString` | `nvsdk_ngx.h` |
| `GetFeatureRequirements` | callable **before** Init; fills `NVSDK_NGX_FeatureRequirement{FeatureSupported, MinHWArchitecture, MinOSVersion[255]}` | L598, L536-L546 |
| `Init_with_ProjectID` | `(const char* projectId[GUID-like], NVSDK_NGX_EngineType, const char* engineVersion, const wchar_t* dataPath, ID3D12Device*, const FeatureCommonInfo*=nullptr, Version=0x15)` | L247 |

## Provenance of the reference DLL

The official `lib/` tree of NVIDIA/DLSS ships `nvngx_dlss.dll` (Release). Its
SHA-256 `BE6E434A94CA32499515EB62CA0E6C274526055D568D0426E4C652DCDFB6EE6E`
**matches byte-for-byte** the DLL lawfully provided to this lab (310.7.0).
Provenance chain recorded in `third_party/PROVENANCE.md`.

## E7 (runtime-confirmed) — this DLL is an NGX *Snippet*, not the SDK loader

Export-table dump of `rel/nvngx_dlss.dll` (59 exports, script
`scripts/_tmp_dump_exports.ps1`, Round 2) shows:

- present: `NVSDK_NGX_D3D12_Init`, `_Init_Ext`, `_CreateFeature`,
  `_EvaluateFeature`, `_ReleaseFeature`, `_Shutdown`, `_Shutdown1`,
  `_GetFeatureRequirements`, `_GetScratchBufferSize`,
  `_PopulateParameters_Impl`, plus `NGX_SNIPPETS_GetRequiredDriverSupport`,
  `NVSDK_NGX_GetAPIVersion`, `NVSDK_NGX_GetSnippetVersion`,
  `NVSDK_NGX_GetDriverVersionEx`, `NVSDK_NGX_GetGPUArchitecture`,
  DirectSR entry points.
- **absent**: `NVSDK_NGX_D3D12_AllocateParameters`, `_GetCapabilityParameters`,
  `_DestroyParameters`, `NVSDK_NGX_D3D12_EvaluateFeature_C`,
  `NVSDK_NGX_UpdateFeature`, `GetNGXResultAsString`.

Consequences:
1. The DLL corresponds to the header's **`NGX_SNIPPET_BUILD` branch**
   (`nvsdk_ngx.h` L148-L165): Init is `(AppId, DataPath, Device,
   [Version], [const NVSDK_NGX_Parameter* InParameters])` via `_Init_Ext`,
   parameter population goes through `_PopulateParameters_Impl` — the
   official SDK-loader parameter lifecycle (E2) is **not exported** here.
2. `EvaluateFeature` exists only as the C++-mangled `extern "C"` undecorated
   name (4-arg, progress callback type `PFN_NVSDK_NGX_ProgressCallback`);
   `_C` variant is absent. Calling it with a `bool*` cancel slot
   (`_C` semantics) is still byte-compatible since the callback is optional
   and NULL is passed.
3. On the NVIDIA reference machine the *driver-side* NGX core
   (`ngx_core`/`nvngx_loader`) may provide the full loader ABI; this lab must
   therefore test **both** surfaces: the snippet ABI of the shipped DLL
   (now) and the loader ABI when present (probe reports which exports exist).
4. Phase B probe must not hard-fail on missing `AllocateParameters` etc.;
   it records per-export `present/absent` and gates the host on the
   intersection actually available.

Fix status: `nr_host.cpp` now resolves the full official set and reports a
per-export presence vector; the missing entries above are expected for this
snippet build and recorded in `ngx_abi_test.json`.

## Status

- R1 static layer: **DONE** (this document + header layering).
- R1 dynamic validation (probe + official-ABI Init on real DLL): Phase B/C.
- Verdict on Round 1's `0xBAD00001`: **UNCONFIRMED as vendor gate** — produced
  under ABI defects E1/E3; must be re-run through the official ABI before any
  conclusion.

## Post-script: official-ABI re-run on this machine (2026-02)

`nr_host.exe --force-load` with the official `rel/nvngx_dlss.dll` (hash above)
on the AMD RDNA4 adapter (vendor 0x1002, device 0x7550):

```
NGX exports ... init=1 init_ext=1 abi_branch=snippet(NGX_SNIPPET_BUILD) ...
NGX init (OFFICIAL ABI): 0xbad00001
```

The same `0xBAD00001 = FAIL_FeatureNotSupported` observed in Round 1 now
reproduces **under the verified official ABI** (snippet-branch 4-arg Init,
correct calling convention, no garbage FeatureCommonInfo). This materially
strengthens the vendor-gate hypothesis: the rejection is not explained by the
E1/E3 defects. It remains classified **UNCONFIRMED** until reproduced on an
NVIDIA RTX reference machine (expected `NVSDK_NGX_Result_Success` there);
no NVIDIA hardware is available on this box (`BLOCKED_EXTERNAL_HARDWARE`).

## Addendum: RTX 5060 control invalidates the vendor-gate inference

An external Windows RTX 5060 (`sm_120`, vendor 0x10de, driver 591.59) reproduced
the same `0xBAD00001` before any DLSS feature DLL, NVAPI, or CUDA module load
(`results/20260830_230102_rtx5060/`). Therefore the return code cannot be used
as evidence of an AMD vendor gate. The control also exposed a host-contract
difference: nr_host used lab-only App/Project identifiers, while the vendored
known-working DLSS5-Feeder uses AppId `0x1000000` and ProjectID
`a0f57b54-1daf-4934-90ae-c4035c19df04`. nr_host now matches those identifiers.

The corrected v2 RTX rerun (`results/20260830_231421_rtx5060_v2/`) returned
Success from Init and passed through public DLAA CreateFeature. The same corrected
contract still returns `0xBAD00001` on AMD, so the difference is now valid evidence
of an adapter-dependent NVIDIA support gate rather than an identifier or ABI
artifact. v2 then failed Evaluate with `0xBAD00005` because it omitted the required
`InRenderSubrectDimensions`; Round 3 fills those fields and the RTX control then
completed 8/8 evaluations plus readback (`results/20260830_232657_rtx5060_v3/`).
This proves the public ABI/host contract. It still says nothing about private
feature 18 or DLSSNR kernel compatibility.
