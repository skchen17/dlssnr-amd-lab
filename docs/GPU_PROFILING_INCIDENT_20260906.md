# GPU profiling safety halt — 2026-09-06

Two module-level Radeon GPU Profiler counter captures were followed by system
hangs/restarts.  Windows Error Reporting recorded `LiveKernelEvent 141` with
watchdog reports named `WATCHDOG-20260906-2218.dmp` and
`WATCHDOG-20260906-2230.dmp`.  System log evidence includes `Kernel-Power 41`,
`EventLog 6008`, and failed dump creation (`volmgr 161`).

The incomplete capture directories are:

- `results/20260906_rgp_pre_core_reference768_v1`
- `results/20260906_rgp_pre_core_reference768_v2`

Neither directory contains a completed capture manifest or a valid `.rgp`
profile.  They are invalid evidence and must not be used for performance or
classification claims.  The target-side JSON from the first attempt only proves
that the pre-capture correctness check completed; it does not validate the RGP
capture.

`scripts/capture_rgp.py` now reads `safety/GPU_PROFILE_HALT.json` and refuses to
start while the halt is active.  Do not bypass the halt, change TDR, request
persistent clock overrides, or automatically retry.  Resumption requires an
explicit review and user direction, followed by one reduced isolated capture.

After CPU-side review and explicit user direction to continue, only minimal
non-RGP correctness gates may resume one at a time. RGP counter collection
remains halted. This does not authorize stress loops, profiler retries, TDR or
clock changes.

Previously completed Head profiles remain separate evidence: their manifests
were complete, their RDF files were valid, and they preceded this incident.
They do not prove that the new module-level capture setup is safe.
