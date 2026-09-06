# cubin_compat_probe

Small CUDA Driver API probe used to test whether a GPU/driver can load a given
CUBIN. It deliberately stops at `cuModuleLoad`; it does not launch proprietary
kernels.

Linux build and run:

```bash
g++ -std=c++17 -O2 -I/usr/local/cuda/include cubin_compat_probe.cpp \
    -L/usr/lib/x86_64-linux-gnu -lcuda -o cubin_compat_probe
nvcc -cubin -arch=sm_89 sm89_positive_control.cu -o sm89_positive_control.cubin
./cubin_compat_probe sm89_positive_control.cubin <cubin-under-test>
```

Always run a same-architecture positive control first. A load failure proves
only that the particular GPU/driver cannot accept the file; it does not by
itself identify whether the cause is GPU architecture, driver age, or a malformed
module.
