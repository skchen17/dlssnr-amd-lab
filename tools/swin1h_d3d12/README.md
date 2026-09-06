# Native chained Swin 1h/32 prototype

One shared operator implementation covers slots 3, 4, 151 and 152 of the captured
graph. It records ten DXIL dispatches: tiled gather, FFN expansion/nonlinearity,
FFN projection/residual, QKV projection, Q/K normalization, V packing, QK scoring,
softmax/V, final projection/residual and E4 scatter. Eight mathematical shaders
are reused from the reconstructed output head; the attention weights begin 112
bytes earlier in this family's 20,672-byte local weight record.

This is an **experimental implementation, not an accepted whole-frame upgrade**.
Four same-input RTX controls pass (NRMSE 1.77%-2.58%), but the integrated variant
regresses the canonical final-frame comparison to correlation 0.638833 / NRMSE
0.854350. It does not replace the default reference pipeline. No new per-instruction
RTX experiment is needed to reproduce these results.

## Run commands

From PowerShell:

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\scripts\build_all.ps1 -Only swin1h_d3d12,output_head_infer_d3d12

# Four isolated encoder/decoder controls, verified source hashes, two GPU repeats.
# Choose a fresh results path for each invocation.
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/run_swin1h_native_validation.py `
  --output results/swin1h_family_new_run

# Full 156-slot diagnostic, with these four operators actually running DXIL.
# No captured intermediate is injected. Other operators still use ZLUDA.
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/run_full_graph_integrated.py `
  --plan results/20260831_234000_full_graph_integrated_plan/plan_fine_n0_jointulp_slots2to6_square_postblock.json `
  --nvcuda .tools/zluda-v7-preview.3/zluda/nvcuda.dll `
  --output results/swin1h_full_graph_new_run `
  --export-pre-head --native-swin build/swin1h_d3d12.exe

# Feed actual AMD upstream into the reconstructed head and check final RGB.
# Exit 1 / IMAGE_GATE_FAIL is the current expected finding, not a script error.
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_reconstructed_head_full_frame.py `
  --upstream results/swin1h_full_graph_new_run `
  --output results/swin1h_full_frame_new_run
```

Omit `--native-swin` to retain the original all-translated upstream baseline.
Neither runner changes that plan or its default implementation choices.

The standalone inference interface consumes no reference output:

```powershell
.\build\swin1h_d3d12.exe input.e4 local_weights.raw 320 192 -4 -4 results/operator_new_run
```

Input storage is 4x4 fragment tiles, 32 E4M3 bytes per pixel; the output has the
same layout. The weight file may include a larger captured window; only its first
20,672 bytes are consumed. Width/height must be positive multiples of eight, at
most 640x384; window origins are independently 0 or -4. These are implementation
bounds, not a claim of validation at all dimensions. The four accepted controls
use 320x192. Different/downsample/outview/upsample families are not implemented.

## Evidence and limits

- `results/20260905_052500_swin1h_native_family/`: four isolated controls PASS.
- `results/20260905_053100_native_swin_full_graph/`: complete repeated execution.
- `results/20260905_053200_native_swin_head_full_frame/`: image gate FAIL.
- Full-graph experimental handoffs currently use CPU readback/upload and a new
  D3D12 process per invocation. Timing includes setup and two diagnostic repeats;
  it is not a game-performance measurement or a resource-resident implementation.
- Synchronization words are released only after the complete native producer
  returns. No placeholder computation replaces a neural operation.
- Local operator references are used only by the coordinator for comparison,
  never as intermediate values inside inference. Whole-network O4/S7/S8 remain
  incomplete, and no game directory has been changed.
