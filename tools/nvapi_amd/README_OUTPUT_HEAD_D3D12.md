# NVAPI slot154 D3D12 recording path

The actual NVAPI launch-chain entry now records the complete DXIL output head
for command lists registered with `NvapiAmd_BeginHeadD3D12`. Unregistered lists
keep the laboratory HIP path. Resource states are supplied explicitly.

## Run

From the repository root in PowerShell:

```powershell
.\scripts\run_nvapi_amd_output_head_d3d12_selftest.ps1
```

The script builds the backend, required shaders and test, then prints the result
directory. The default reference is
`results\20260905_042914_output_head_recording_selftest`. To regenerate it:

```powershell
.\scripts\run_output_head_recording_selftest.ps1 -OutputDirectory results\head_recording_reference
.\scripts\run_nvapi_amd_output_head_d3d12_selftest.ps1 `
  -ReferenceDirectory results\head_recording_reference
```

`-SkipBuild` uses existing artifacts. No cloud experiment or game installation
is needed for this local test.

## Host contract

The private ABI is in `nvapi_amd_d3d12_head.h`:

1. Create the usual surface and merged texture/sampler tokens, registering their
   descriptor-to-resource identities using the existing backend API.
2. Begin a session with the direct list/queue, a future completion-fence value,
   and actual activation/model/base/output resource states. This preloads shader
   PSOs and retains the list, queue, fence, resources and scratch allocations.
3. Record input producers and call the actual NVAPI launch-chain entry. Slot154
   resolves addresses and descriptor tokens from its 184-byte parameters, then
   records 11 DXIL dispatches plus texture copies without executing the HIP head,
   submitting work or waiting for the GPU.
4. The recorder restores declared resource states. Rebind subsequent compute
   root signature/arguments and PSO, submit on the registered queue and signal the
   completion fence after that submission.
5. Release after fence completion. Early release and recording into a completed
   session are rejected. CPU-signaling the fence does not establish GPU completion
   and violates the contract.

One session records serially and owns its scratch set. Resources must have their
declared states at every head invocation. Do not reset or reuse the list/fence
generation before retirement. Device-loss teardown is not implemented.

Supported parameters are restricted to the captured model310.8 layout, 640x360
RGBA16F full texture views, accepted original model/blend pack, fixed composition
and no populated history fields. All non-relocatable scalar ABI words are checked.
Model contents remain the host's accepted local pack; they are not read back or
hashed while recording. The blend address/range is checked but the recorder still
uses its reconstructed fixed composition contract. Other blend semantics, view
subresources, dimensions and history modes are not implemented.

Other functions, including the separate slot155 copy, are rejected inside this
head-only session. Do not register an entire 156-slot list until its other
handlers are integrated.

## Automatic opt-in bridge (controlled-host tested)

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\scripts\run_nvapi_amd_output_head_d3d12_selftest.ps1 -Automatic -Batches 250
```

`MODULE_TRACE_AMD_D3D12_HEAD=1` enables descriptor/address resolution and conservative
whole-resource legacy-barrier tracking. NVAPI recording creates a deferred session;
`ExecuteCommandLists` seals it with the actual queue and an owned completion fence.
Completed sessions are collected on subsequent hook calls or by the nonblocking
`ModuleTrace_CollectAmdHeads` export at final drain. The script restores environment
settings after the test. The core recorder itself neither submits nor waits.

Only one pending head list per submission is supported. Unknown/promoted states,
mixed head kernel chains, duplicate head lists and premature reset/reuse fail
closed rather than fall back to synchronous HIP. Enhanced barriers, arbitrary
views/history, abandoned lists, device-loss teardown and module unloading are not
implemented. This narrow opt-in path is not yet suitable for general games.

1,000 head calls / 250 submissions passed in `results/20260905_050734_nvapi_output_head_d3d12/`.
Additional chain/batch rejection controls passed in `results/20260905_051316_nvapi_output_head_d3d12/`.
These are A/B/C/A synthetic tensor controls, not 1,000 real scene frames.

## Test coverage

Two distinct lists/sessions each execute four cases: original main/zero base,
patterned base, zeroed main/zero base, then restored original inputs. Pending
input copies and head computations share each list. All eight textures match
the standalone recorder byte-for-byte. Twelve invalid calls, two premature
releases and duplicate begin/release attempts are rejected. Both sessions retire
with zero active sessions and no D3D12 debug errors.

Activation/model buffers are ordinary D3D12 allocations with no HIP import.
The backend DLL still links HIP and legacy descriptor setup still creates a HIP
staging allocation; this is not a HIP-free DLL claim. No marker substitutes for
head calculation. Inputs remain captured/control tensors, not native decoder
outputs. Full-network quality, live feature18 evaluation, temporal history,
dynamic resolution and game performance remain open.
