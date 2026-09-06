# Native fusion probe (not enabled by default)

First bounded candidate: `quantize_e4` for contiguous FP16 inference tensors.
The existing expression materializes clamp, E4M3 cast and FP16 cast outputs.
This kernel uses integer IEEE bit manipulation for the same rounding and
saturation in **one explicit HIP dispatch**, one input read and one output write.
It is a first helper fusion, **not a replacement for the complete NVIDIA head
kernel**. Whole-network measurements and their limits are recorded in
`../../docs/DUAL_NR_ROUTES_BENCHMARK.md`.

## Build and CPU gate (no GPU execution)

Run from the repository root in PowerShell. The project-local ROCm SDK is used;
no ComfyUI installation is changed. Choose a fresh absolute output directory.

```powershell
./tools/native_fusion_probe/build.ps1 -OutputDirectory "$PWD/results/native_fusion_quantize_build"
./.venv-rocm/Scripts/python.exe tools/native_fusion_probe/native_fusion_cpu_gate.py --dll "$PWD/results/native_fusion_quantize_build/native_fusion_quantize.dll"
```

Build uses `gfx1201`, `-ffp-contract=off`, `-fno-fast-math`. The shared CPU/device
conversion is exhaustively checked against the installed PyTorch operation:
63,490 finite/Inf FP16 patterns require exact output bits including signed zero;
2,046 NaNs must stay NaN (payload/sign not promised, nonfinite network tensors
already fail network acceptance). No rounding tolerance is permitted.

## GPU gate (only when operator testing is safe)

Do not run concurrently with another GPU benchmark or the game. Supervise the
process externally; a CPU timeout cannot cancel GPU work. Stop on any failure,
device reset or timeout, without automatic retry. First run one iteration, then
12 only if it passes:

```powershell
./.venv-rocm/Scripts/python.exe tools/native_fusion_probe/native_fusion_gpu_gate.py --dll "$PWD/results/native_fusion_quantize_build/native_fusion_quantize.dll" --iterations 1
./.venv-rocm/Scripts/python.exe tools/native_fusion_probe/native_fusion_gpu_gate.py --dll "$PWD/results/native_fusion_quantize_build/native_fusion_quantize.dll" --iterations 12
```

These tiny-event times are not network timings or baseline speedup measurements.
The candidate emits one launch by construction; CPU ATen counts elsewhere must
not be described as hardware dispatch counts. Full baseline dispatch count still
requires a working device profiler or launch interception.

## Loading and remaining gates

```python
from native_fusion_quantize import QuantizeE4
quantize = QuantizeE4(absolute_dll_path)
y = quantize(x)  # contiguous HIP FP16, requires_grad=False
```

Caller keeps tensors alive until the current PyTorch HIP stream completes.
No synchronization, allocation of CPU images, implicit contiguous copy, CPU
fallback, autograd fallback or default-model monkeypatch is provided. Frozen
weights remain unchanged. Parameter quantization and FP32 inputs are not silently
redirected to this FP16-only implementation.

Before broader promotion: captured real activation distributions;
then 640/odd-size and full-frame exact output hashes, 1/12 repeated inputs,
bounded memory and no device errors. Keep unfused implementation as explicit
reference. Further activation polynomial fusion must preserve all intermediate
FP16 round-to-nearest-even stores and separate FP32 multiply/add boundaries;
normalization fusion must preserve the exact adjacent-pair reduction tree and
validate the installed PyTorch rsqrt rounding before acceptance.

## Existing output-head code is not a drop-in baseline

`tools/output_head_activation_fusion` already has HIP code, but it references
captured 320-wide offsets, fixed CTA positions, packed source arena addresses and
an explicitly reconstructed FMA contract. Current native PyTorch modules use
different logical tensor interfaces. Importing that old kernel as-is would mix
contracts and fixed-size layouts; it cannot safely be promoted to dynamic
full-frame fusion. The first candidate above deliberately operates at a shared,
fully specified numerical boundary before larger head/layout reconstruction.

## Second candidate: cubic SiLU + E4

`CubicQuantizeE4` and the `native_fusion_cubic_quantize_f16` export combine the
exact existing `quantize_e4(cubic_silu(x))` expression into one dispatch. Each
FP32 multiply and add remains separate (`-ffp-contract=off`), and all three
intermediate FP16 round-to-nearest-even boundaries are retained. This candidate
is intended for the explicit grouped FFN and ViT expression sites, not arbitrary
SiLU or an NVIDIA instruction-exact claim. Explicit grouped FFN, Swin and ViT
sites now support a scoped `native_fusion_policy.fusion_policy(dll)` opt-in.
The default remains the unfused reference. The policy explicitly counts any
contiguous copies; the low-level binding itself does not make implicit copies.

Build to a new directory to preserve the quantization-only DLL evidence:

```powershell
./tools/native_fusion_probe/build.ps1 -OutputDirectory "$PWD/results/native_fusion_cubic_build"
./.venv-rocm/Scripts/python.exe tools/native_fusion_probe/native_fusion_cpu_gate.py --dll "$PWD/results/native_fusion_cubic_build/native_fusion_quantize.dll" --operation cubic_quantize
# GPU execution only when separately supervised and no other GPU work is active:
./.venv-rocm/Scripts/python.exe tools/native_fusion_probe/native_fusion_gpu_gate.py --dll "$PWD/results/native_fusion_cubic_build/native_fusion_quantize.dll" --operation cubic_quantize --iterations 1
./.venv-rocm/Scripts/python.exe tools/native_fusion_probe/native_fusion_gpu_gate.py --dll "$PWD/results/native_fusion_cubic_build/native_fusion_quantize.dll" --operation cubic_quantize --iterations 12
```

```python
from native_fusion_quantize import CubicQuantizeE4
activation_quantize = CubicQuantizeE4(absolute_dll_path)
y = activation_quantize(x)
```

CPU exhaustive compiled-function gate passed all 65,536 input bit patterns:
63,489 non-NaN reference outputs require identical bits; 2,047 reference NaNs
require NaN outputs. The extra NaN comes from negative infinity multiplied by
the reference polynomial's zero, and is not silently saturated to a number.
Both primitives passed supervised GPU exhaustive 1/12-run gates on gfx1201.
Opt-in whole-frame 1080p/1440p/4K inference passed 12 exact repetitions against
the unfused outputs; 640 also matched. This is not an arbitrary-input, odd-size,
real-game temporal or NVIDIA-equivalence acceptance. See the linked report for
fresh-output-directory supervised commands. Game defaults are unchanged.
