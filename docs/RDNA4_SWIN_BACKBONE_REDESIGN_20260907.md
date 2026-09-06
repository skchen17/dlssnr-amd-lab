# RDNA4 Swin 主干重构结果（2026-09-07）

## 结论

本轮完成了 C32 conservative attention、C64/C128/C256 grouped FFN、C64/C128
两种 attention 粒度、resident FP8 权重基础和原生 `NRPlan` Graph 基础设施。所有通过的
GPU correctness gate 都保持输出逐位一致，RGP counter 安全锁未解除，也没有进行 4K GPU
实验。

真正取得可重复整帧收益的是低 LDS grouped FFN。1080p B-A-B-A 四个热态样本的中位数为：

| 路径 | HIP event span 中位数 | 输出 | allocator 峰值 |
|---|---:|---|---:|
| 当前 reference | 1031.478 ms | SHA256 `892FB11D...01B7` | 1,480,598,016 B |
| C64/C128/C256 grouped FFN | 902.266 ms | 同一 SHA256 | 1,669,704,192 B |

整帧下降 12.527%（1.143×）。这里是整条 stream 的 event span，不是 profiler kernel-busy
sum；由于 RGP 被安全锁停，不能把它表述成纯 kernel activity。

## C32 conservative attention core

保留既有并行 QKV projection 和 normalization，只把 QK、softmax、PV 放进 8 KiB LDS
核心，projection 独立。静态 ISA 为 0 scratch，FP16 版本约 99 VGPR。结果：

- 1 window：0.14530 → 0.17832 ms，0.815×；
- 16 windows：0.18870 → 0.26456 ms，0.713×；
- 完整 C32 FFN+attention 的 authored launch 7 → 5；输出逐位一致。

dispatch 减少但模块变慢，因此候选拒绝，不进入 12 次或整帧门。瓶颈不是单纯的 20 KiB
LDS；该粒度下融合核心自身的标量 softmax/串行工作仍大于省下的全局中间流量。

## C64/C128/C256 grouped FFN

新 kernel 以 `(window, head, 16-token tile)` 为 grid。每个 32-thread workgroup 只执行
当前 head 的 `C → 128 → 32`，LDS 固定为 4096 B，用后立即释放。跨 head 的 `C → C`
mix 保持 library GEMM，随后单独 residual add，因此一个 logical FFN 为三个 dispatch。

| 家族（144 windows，12 次中位数） | reference | grouped | 加速 | grid | LDS | FP16 VGPR | 静态 occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| C64 | 0.85686 ms | 0.34695 ms | 2.47× | 1152 | 4 KiB | 102 | 50% |
| C128 | 1.99754 ms | 1.26972 ms | 1.57× | 2304 | 4 KiB | 100 | 50% |
| C256 | 5.83506 ms | 4.39137 ms | 1.33× | 4608 | 4 KiB | 100 | 50% |

三者的基础、重复、换输入、guard、finite 和资源释放检查均通过且逐位一致。1080p 一帧
实际调用 52 个 grouped custom kernel，对应 156 个 logical dispatch。窗口几何覆盖 C64
624/660/684/721/768、C128 540/558/570/589、C256 135/144/150/160，不是只在小 batch
上成立。

## C64/C128 attention A/B

方案 A 是 `(window,head)`、每组 20 KiB LDS、4 waves；约 101 VGPR，静态 occupancy
37.5%。方案 B 是 `(window,head,16-query tile)`、每组 2 KiB LDS、1 wave；约 121 VGPR，
静态 occupancy 62.5%。两者都保持 QKV/norm 与 projection 独立，只融合
QK+softmax+PV。

- C64/16 windows：A 为 0.55380 → 0.66392 ms；B 为 0.54740 → 1.08650 ms。
- C128/16 windows：A 为 1.66230 → 1.39298 ms；B 为 1.87930 → 1.73542 ms。
- C128/144 windows：A 为 2.43670 → 2.60314 ms，代表规模回退 6.8%。

全部数值逐位一致，但没有方案在代表规模稳定胜出。A 受 LDS/occupancy 限制；B 虽降低
LDS，却让每个 query tile 串行完成 16 行 exp/softmax 并增加工作组开销。按规则两者均
拒绝，也没有把参数直接复制到 C256 attention。

## resident FP8 权重基础

新增 `c{64,128,256}_ffn_grouped_fp8w`：expand/contract 权重初始化时一次性保存为
E4M3 字节，WMMA 直接读取，不再在每次 MMA 前从 FP16 现场转换权重。激活仍在已确认的
边界量化，grouped 输出仍以 E4M3-rounded FP16 交给 library C×C GEMM，所以这不是完整
resident activation。

- ISA：三个家族均为 5 个 FP8 WMMA site、99 VGPR、36 SGPR、4 KiB LDS、0 scratch，
  静态 occupancy 50%。
- C64/144 一次：普通 FP8 0.73564 ms，resident weight 0.73614 ms，无收益。
- C128/144 十二次中位数：1.49509 → 1.45784 ms，快 2.49%，低于 5% 晋级门槛。
- 两组均逐位一致且完整释放资源。

该实现保留为后续 activation resident 和原生 mix 的数据格式基础，不默认启用，也不继续
扩大调参。单纯消除权重转换不足以解决当前主瓶颈。

## 原生 NRPlan 状态

`tools/native_nr_plan` 新增可编译 C++/HIP 基础：

- 初始化时一次性分配 workspace、weight storage、device binding table 和 pinned 参数块；
- recorder 只在初始化 capture 一次，随后 `hipGraphLaunch`；
- 每帧只更新 input/output、尺寸、帧号和资源代次；无每帧 allocation 或 CPU 神经回退；
- 显式 input-ready/output-done event；单 plan 只允许一帧在途，避免异步参数覆盖；
- reset、destroy 和错误码均显式处理。

gfx1201 marker 自测连续重放两帧，并更换 input/output 指针、尺寸、帧号和资源代次，结果
通过。构建同时产出可链接的 `nr_plan.dll`。但它还没有装入全部 71-block 网络
kernel，因此只是 runtime 基础，不是已经完成的整帧 NRPlan，也没有部署到游戏。

## 安全、复现和未完成项

RGP `GPU_PROFILE_HALT` 保持启用；全部数据来自安全 HIP event、静态 ISA 和有界 A/B。
所有 GPU 子进程正常退出，allocator/reserved 归零，实验后未观察到新的 System 41/141/6008。

核心复现命令：

```powershell
$py = "$PWD\.venv-rocm\Scripts\python.exe"
$build = "$PWD\results\swin_backbone_build"
.\tools\native_fusion_probe\build_head_wmma.ps1 -OutputDirectory $build
& $py scripts\audit_matrix_isa.py --assembly "$build\head_wmma.s" `
  --asic-spm-summary results\20260906_rgp_head_dispatch_sweep_ready_640x384_v1\02_ffn\spm_summary.json `
  --output "$build\isa_occupancy.json"
& $py scripts\validate_grouped_wide_ffn.py --dll "$build\head_wmma.dll" `
  --channels 128 --windows 144 --iterations 12 --profile wmma_fp8 `
  --resident-fp8-weights --output results\c128_fp8w_repro
.\tools\native_nr_plan\build.ps1 -OutputDirectory "$PWD\results\nr_plan_repro"
```

下一步不是继续扩大 attention 单 kernel，而是把已经获益的 grouped FFN、resident
transition、Head/Pre 已通过路径接入 `NRPlan` 的固定 command sequence；随后在严格量化
边界内解决 activation FP8 常驻和 C×C mix，最后才重新做 1080p/1440p整帧门。4K、游戏
部署、SDR/HDR 和 4K60 均未在本轮关闭。
