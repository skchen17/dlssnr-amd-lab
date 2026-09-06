# RTX post-block native-resource lowering sweep

This diagnostic runs the exact slot-154 state through seven cumulative PTX
stages on an RTX GPU. The original stage uses a CUDA RGBA16F surface and the
captured all-zero RGBA16F texture semantics (point filtering, border addressing,
normalized coordinates). It must reproduce the captured RTX output exactly.
Later stages cumulatively add compatibility, E4M3, movmatrix, FP8 MMA, FP16 MMA,
and decoder compatibility rewrites. The first output-changing stage identifies
the correction target without another one-hypothesis-at-a-time round trip.

Run from the extracted package directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_nvidia_postblock_native_stage_sweep.ps1
```

Return `_postblock_native_sweep_reference_result_<timestamp>.zip` from the
package root. A FAIL result is still useful: return the ZIP whenever it is
created, because its logs identify whether the original native-resource replay
contract itself failed or a later lowering stage diverged.
