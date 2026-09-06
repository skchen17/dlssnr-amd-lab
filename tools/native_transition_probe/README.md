# Native scale-transition probe

This opt-in experiment replaces encoder and decoder boundary logical-layout
round trips with two authored HIP launches around each existing projection GEMM.
The encoder path is:

1. packed source -> exact FP16 2x2 pool + channel permutation, while producing
   the quantized packed skip;
2. the unchanged `native_fp16` projection;
3. E4M3 quantize + direct next-scale resident packed output.

The decoder path consumes resident low/skip features, forms the permuted matrix
input, retains the existing projection GEMM, then performs 2x expansion,
half-rounded skip fusion and direct packed output. Intermediate family outviews,
`repeat_interleave`, and full logical skip materialization are removed. The last
C32 output still preserves the recovered Head outview ABI.

It does not use RGP or counter collection. It does not fuse the preceding Swin
scatter and it does not claim measured DRAM traffic from allocation accounting.

```powershell
powershell -ExecutionPolicy Bypass -File tools/native_transition_probe/build.ps1 -OutputDirectory C:\temp\native_transition_build
```

Validation and full-frame benchmark commands are documented by
`scripts/validate_encoder_transition.py --help` and
`scripts/benchmark_nr_resolution.py --help`. The transition is disabled unless
both a DLL and the explicit encoder-transition option are supplied.
