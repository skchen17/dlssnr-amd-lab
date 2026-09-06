# Encoder/Decoder 尺度切换审计与原生化记录（2026-09-07）

## 结论

当前尺度切换的主要浪费不是单独的 `average_pool2x2`，而是尺度边界两侧反复物化整张 feature：

- Encoder：`scatter -> unpack -> pool -> channel permutation -> projection -> quantize -> pack_image -> pack_outview`，下一尺度又 `unpack_outview -> pack_image -> gather`。
- Decoder：`unpack low + unpack skip -> projection -> repeat_interleave -> skip_scale -> logical add -> pack -> gather`。

`downsampled` 已确认不被 `native_whole_frame.py` 正式 hot path 消费。它只用于兼容测试和诊断，因此 resident 路径不再分配第二份 downsample feature；debug/capture outview 也改为显式选择。

本轮实现了默认关闭的 bounded native transition：两个专用 whole-grid HIP launch 包围数值语义不变的现有 FP16 library GEMM。它不是“只优化 pooling”，也没有强行形成一个高 VGPR/LDS 的超大 kernel。

## 实现边界

### Encoder downsample

当前保守原生路径为：

1. `packed Swin output -> HIP pool/permutation + quantized skip`
2. 原有 `native_fp16`、E4M3 输入/权重边界的 projection GEMM
3. `HIP E4M3 quantize + next-scale resident packed write`

HIP pooling 保留参考实现的三次 FP16 加法舍入顺序：先两行分别相加，再相加，最后乘 0.25 并舍入。projection 的权重、channel permutation 和 E4M3 boundary 未改变。

该版本仍保留前一 Swin 的 scatter，也仍物化 projection 的矩阵输入/输出；这是有意控制寄存器压力的第一版 bounded fusion。只有实测表明把 projection 融入同一 kernel 不降低 occupancy，才扩大融合边界。

### Decoder upsample

中间 Decoder family 的末块（55/61/65）现在可以保持 packed resident；下一尺度不再要求 outview。block69 仍生成原 Head ABI 需要的 outview，避免改变 Output Head 语义。block47 的 Split512 输出也可在 resident 模式保持 packed。

当前保守原生路径为：

1. `resident low -> HIP unpack + channel permutation matrix input`
2. 原有 `native_fp16` projection GEMM
3. `HIP 2x expansion + FP16 skip_scale/add + destination resident packed write`

它删除了 `repeat_interleave`、skip 的完整 logical unpack、完整 logical fused tensor和 outview 往返。下一 Swin 的 gather 仍保留；将 gather 进一步并入 whole-grid Swin 属于后续 block-runtime 融合，而不是在 transition 中伪造跨窗口存储语义。

## 静态存储审计

下表是源码中可见 FP16 feature 的“命名张量写入量”，不是实测 DRAM/L2 流量，也不是 allocator 峰值。权重读取、框架内部 workspace 和 cache traffic 均未计入。

### 4K（3840×2160，内部补齐为 3840×2176）Encoder

| Transition | 旧路径写入 | 仅 resident 路由 | bounded native | 相对旧路径 |
|---|---:|---:|---:|---:|
| C32→C64 | 802,160,640 B | 601,620,480 B | 434,503,680 B | -45.8% |
| C64→C128 | 401,080,320 B | 300,810,240 B | 217,251,840 B | -45.8% |
| C128→C256 | 200,540,160 B | 150,405,120 B | 108,625,920 B | -45.8% |
| C256→C512 | 100,270,080 B | 75,202,560 B | 54,312,960 B | -45.8% |

仅 resident 路由相对旧路径减少 25%；加入本轮 bounded native transition 后，再相对 resident 路由减少约 27.8%。
源码可见的 Encoder 命名写入由旧路径 10 份降到 bounded 路径 5 份；Decoder 由 9 份降到 4 份（均包含下一 Swin gather 时的计数口径）。

### 4K Decoder

| Transition | 旧路径写入 | bounded native（含下一次 gather） | 相对旧路径 |
|---|---:|---:|---:|
| C512→C256 | 121,159,680 B | 45,957,120 B | -62.1% |
| C256→C128 | 242,319,360 B | 91,914,240 B | -62.1% |
| C128→C64 | 484,638,720 B | 183,828,480 B | -62.1% |
| C64→C32 | 969,277,440 B | 367,656,960 B | -62.1% |

1080p、1440p 和 4K 的完整逐边界字节记录可由 `audit_scale_transitions.py` 重建；由于各层 feature 面积按固定比例变化，静态写入降幅保持一致，绝对收益随像素数线性扩大。

## Correctness 与性能状态

- C32→C64、8×8 单窗口、RX 9070 XT：skip 与 resident 均逐位一致，差异元素 0，max error、RMSE、NRMSE 全为 0。
- 一次 baseline 和一次 candidate 均正常结束，释放后 PyTorch allocated/reserved 都为 0。
- 此次 baseline event 535.480 ms、candidate 0.01945 ms 被明确判为不可用于性能结论：baseline 包含首次 library/GEMM 初始化，而 candidate 复用了已初始化状态。
- C32/C64/C128/C256 Encoder 与 C256/C128/C64/C32 Decoder 的 8×8 单窗口 GPU gate 已全部逐位一致。
- 按用户后续要求，GPU 性能验收取消 4K，只保留 1080p 和 1440p；4K 静态字节审计仍保留为布局分析资料。
- 候选路径每个 transition 有 2 个 authored HIP dispatch 和 1 次 library projection。实际总 GPU dispatch、kernel busy sum、DRAM/L2/occupancy 仍未知；RGP counter lock 生效期间不以源码节点数替代测量。

三个独立 A-B-B-A 进程合计提供每个实现、每个 transition 12 个有效样本。下表是隔离 transition 的 HIP event stream interval 中位数；不是整帧时间，也不是 kernel busy sum：

| 分辨率/方向 | Reference transition 中位数之和 | Native transition 中位数之和 | 节省 | 比值 |
|---|---:|---:|---:|---:|
| 1080p Encoder | 16.687 ms | 4.968 ms | 11.719 ms | 3.36× |
| 1080p Decoder | 20.220 ms | 6.270 ms | 13.950 ms | 3.22× |
| 1440p Encoder | 22.260 ms | 8.375 ms | 13.885 ms | 2.66× |
| 1440p Decoder | 25.369 ms | 7.258 ms | 18.111 ms | 3.50× |

逐 transition 结果与单阶段最大增量 allocator 峰值如下：

| 模式 | Transition | Reference | Native | 加速 | Ref peak | Native peak |
|---|---|---:|---:|---:|---:|---:|
| 1080p Enc | C32→C64 | 6.40 ms | 2.06 ms | 3.11× | 97,591,296 B | 79,888,384 B |
| 1080p Enc | C64→C128 | 4.35 ms | 1.35 ms | 3.22× | 48,676,864 B | 39,813,120 B |
| 1080p Enc | C128→C256 | 2.87 ms | 0.77 ms | 3.74× | 24,625,152 B | 20,168,704 B |
| 1080p Enc | C256→C512 | 3.07 ms | 0.80 ms | 3.85× | 12,230,656 B | 10,027,008 B |
| 1080p Dec | C512→C256 | 3.38 ms | 0.81 ms | 4.15× | 21,012,480 B | 7,741,440 B |
| 1080p Dec | C256→C128 | 3.99 ms | 0.94 ms | 4.23× | 42,024,960 B | 15,482,880 B |
| 1080p Dec | C128→C64 | 5.70 ms | 1.32 ms | 4.31× | 84,049,920 B | 30,965,760 B |
| 1080p Dec | C64→C32 | 7.15 ms | 3.19 ms | 2.24× | 168,886,272 B | 62,193,664 B |
| 1440p Enc | C32→C64 | 8.69 ms | 4.20 ms | 2.07× | 173,023,232 B | 141,557,760 B |
| 1440p Enc | C64→C128 | 5.70 ms | 2.09 ms | 2.73× | 87,572,480 B | 71,827,456 B |
| 1440p Enc | C128→C256 | 4.29 ms | 1.15 ms | 3.74× | 43,286,528 B | 35,389,440 B |
| 1440p Enc | C256→C512 | 3.58 ms | 0.94 ms | 3.82× | 21,692,416 B | 17,694,720 B |
| 1440p Dec | C512→C256 | 4.71 ms | 1.02 ms | 4.60× | 38,404,096 B | 13,762,560 B |
| 1440p Dec | C256→C128 | 4.82 ms | 1.50 ms | 3.22× | 77,856,768 B | 28,573,696 B |
| 1440p Dec | C128→C64 | 6.54 ms | 2.12 ms | 3.08× | 150,470,656 B | 56,098,816 B |
| 1440p Dec | C64→C32 | 9.29 ms | 2.61 ms | 3.56× | 299,892,736 B | 111,149,056 B |

所有计时样本对应的输出均逐位一致。一个未纳入汇总的 1440p Decoder 进程曾返回 -1.388 ms 的无效 HIP event；其正确性仍通过，但该计时证据被拒绝并保留，benchmark 已增加 `elapsed > 0 && finite` 强制门禁。替代进程在人工检查系统健康后单独运行。

第一次 1080p 整帧 resident reference 在 C512 入口由软件布局枚举检查停止：runtime 状态使用 `resident`，而 Split512 ABI 枚举使用 `packed`。映射已修复并有回归测试，但异常后的 Python/ROCm runtime 未在 180 秒监督期限内正常退出，因此结果被标记为 `STOP_GPU_REQUIRES_REVIEW_HOST_TIMEOUT`，没有自动重跑。系统日志没有新增 41/141/6008，仍不能据此假定已挂起 GPU 工作被取消；non-profiler GPU gate 已重新锁定。

用户随后明确授权结束该实验进程树。复核确认 PID 26376 是 PID 17752 的同命令子进程后，先结束子进程，父启动器随即消失；两者均不存在后才恢复 bounded non-RGP gate。修复后的 1080p/1440p 整帧均正常退出，RGP 锁始终未解除。

完整帧使用独立进程 A-B-B-A 顺序，每个进程 3 帧、丢弃首个冷帧，最终每实现保留 4 个热态 host submit/wait 样本：

| 分辨率 | Resident reference | Native transitions | 节省 | 降幅 | 输出 |
|---|---:|---:|---:|---:|---|
| 1080p | 694.896 ms | 682.944 ms | 11.952 ms | 1.72% | bitwise exact |
| 1440p | 1150.162 ms | 1135.372 ms | 14.790 ms | 1.29% | bitwise exact |

1080p 峰值为 1,480,598,016 B allocated、1,715,470,336 B reserved、2,134,376,448 B 设备用量采样；1440p 分别为 2,314,749,440 B、2,705,326,080 B、3,111,649,280 B。reference/native 的整帧峰值相同，因为固定模型和其他 stage 主导全局峰值，transition 临时量减少没有改变全图最高水位。

结论是：尺度切换模块本身已取得 2.66×–3.50× 的合计局部收益，整帧只改善 1.29%–1.72%。这符合当前瓶颈分析——尺度边界已不再是主要总耗时来源，后续应回到 Swin/Head/Pre 和固定原生执行计划。该候选满足“模块有明确收益、整帧不回退、数值与资源通过”的显式优化配置门槛，但仍保持默认关闭，不直接改变游戏默认路径。

静态 gfx1201 ISA metadata 已随构建生成；这证明当前 bounded kernel 没有 scratch/LDS 压力，但不等于实测 occupancy：

| Kernel | VGPR | SGPR | LDS | Scratch |
|---|---:|---:|---:|---:|
| Encoder pool/permutation/skip | 23 | 24 | 0 | 0 |
| Encoder quantize/pack | 12 | 19 | 0 | 0 |
| Decoder unpack/permutation | 12 | 19 | 0 | 0 |
| Decoder expand/skip/pack | 14 | 20 | 0 | 0 |

## 可复现实验

构建：

```powershell
pwsh -ExecutionPolicy Bypass -File tools/native_transition_probe/build.ps1 `
  -OutputDirectory C:\temp\native_transition_build
```

最小 Encoder/Decoder gate（每次只运行一个）：

```powershell
.venv-rocm\Scripts\python.exe scripts\validate_encoder_transition.py `
  --dll C:\temp\native_transition_build\native_encoder_transition.dll `
  --channels 32 --output results\encoder_transition_c32_gate

.venv-rocm\Scripts\python.exe scripts\validate_decoder_transition.py `
  --dll C:\temp\native_transition_build\native_encoder_transition.dll `
  --channels 32 --output results\decoder_transition_c32_gate
```

通过相应 correctness gate 后，单分辨率 transition A-B-B-A：

```powershell
.venv-rocm\Scripts\python.exe scripts\benchmark_scale_transitions.py `
  --dll C:\temp\native_transition_build\native_encoder_transition.dll `
  --size 2160 --kind encoder --iterations 12 `
  --ack-bounded-performance --output results\transition_4k_encoder_abba
```

完整整帧先生成 resident reference，再启用 native transition 做同输入 hash gate；当前 GPU 验收只运行 `1080` 和 `1440`：

```powershell
.venv-rocm\Scripts\python.exe scripts\benchmark_nr_resolution.py `
  --size 1080 --iterations 12 --resident-transitions `
  --output results\nr1080_resident_reference

.venv-rocm\Scripts\python.exe scripts\benchmark_nr_resolution.py `
  --size 1080 --iterations 12 --resident-transitions `
  --transition-dll C:\temp\native_transition_build\native_encoder_transition.dll `
  --encoder-transition --decoder-transition `
  --reference-output results\nr1080_resident_reference\output.rgba16f `
  --output results\nr1080_native_transitions
```

## 安全与后续门槛

RGP/硬件 counter capture 继续由 `safety/GPU_PROFILE_HALT.json` 禁用。不得自动解除、重试、修改 TDR 或时钟。鉴于两次 LiveKernelEvent 141/黑屏历史，GPU 工作按“一次一个最小 gate”推进：

1. C64、C128、C256 Encoder 单窗口 gate；
2. C256、C128、C64、C32 Decoder 单窗口 gate；
3. 1080p、1440p 单 transition A-B-B-A（已完成）；
4. 完整 1080p resident reference 与 native transition A/B；
5. 完整 1440p resident reference 与 native transition A/B。

任一 non-finite、hash 差异、event 超时、设备重置或黑屏都会停止后续 GPU 实验。只有两组 A-B-B-A 都显示模块至少 5% 收益、整帧不回退超过 2%、correctness 和 5/6 GB 资源门槛通过，候选才可进入显式优化配置；默认游戏路径仍不自动切换。
