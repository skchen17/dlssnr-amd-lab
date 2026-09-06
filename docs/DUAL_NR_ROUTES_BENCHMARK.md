# 双路线首轮基准 — 2026-09-06

状态：离线基准和两项 opt-in HIP 原语完成；完整 NVIDIA 粒度融合、真实游戏 NR→FSR 常驻链路和时序画质尚未完成。游戏默认配置不变。本轮没有部署插件、修改游戏文件或训练权重。

## 1. NVIDIA 执行证据

详见 [原版执行审计](NVIDIA_EXECUTION_GRANULARITY.md)。本地 5070 的 **640×360、variant02、Feature18** 捕获显示：156 次 NVAPI launch、43 种函数，每次一个 kernel，同一 command list。Pre、Output Head、各 C32/C64 block 各一次 dispatch；C512 每块四次、ViT 每块五次。不能把此数量外推成原版 4K 的实测值。

Pre/Head 静态代码各有 256 个 FP8 MMA、16 个 FP16 MMA 指令位置，不等于动态 FLOPs；模块间存在 global store 和完成标志，并非整网中间值都驻留寄存器。布局含 CTA 分块、打包 half/FP8 和共享内存协作；原版模块内融合不等于整帧独立小图拼接。没有原版 GPU 时长或实际 DRAM 流量测量，不能量化 NVIDIA/AMD 硬件带宽差距。

当前 AMD 张量实现把上述模块拆为大量量化、激活、布局变换和矩阵调用。4K ATen 调用从 **177,181 降至 142,135**，但包括 view 等操作，**不是 GPU dispatch 数**。这证明了执行碎片化，并不足以证明每个 ATen 都产生一次 launch。

## 2. 第一批精确融合

实现单 kernel `clamp→E4M3FN→FP16`，以及单 kernel `cubic SiLU→E4M3FN→FP16`。后者保留三次 FP16 舍入、独立 FP32 乘加，禁止 fast-math/隐式 FMA。通过 scoped policy 接入 Swin、grouped FFN、ViT 的明确调用点，不替换默认执行路径，不修改权重。还不是整个 Pre/Head/C32/C64 的单 kernel 重写。

CPU 和 GPU 对全部 65,536 种 FP16 位型验证：非 NaN 输出逐位一致，NaN 验证语义；GPU 1/12 次门槛通过。640 整帧一致，但约 181.5→184.6 ms，没有加速。1080/1440/4K 每种融合前后各 12 次，全帧 hash 与同输入参考一致。

| NR 尺寸 | 主机提交＋等待，前→后 ms | HIP 整帧流间隔，前→后 ms | 主机耗时下降 |
|---|---:|---:|---:|
| 1920×1080 | 689.65→476.99 | 688.01→475.35 | 30.8% |
| 2560×1440 | 1139.91→759.38 | 1139.49→758.03 | 33.4% |
| 3840×2160 | 2402.39→1494.84 | 2401.04→1492.18 | 37.8% |

每组首帧冷启动，其余 11 帧中位数；不包括有限值检查、图像读回。**HIP 流间隔包含 CPU 供给不足产生的空隙，不是纯 kernel 忙碌时间总和。** 当前 Windows PyTorch 无 Kineto GPU profiler；全网 GPU dispatch 总数和 kernel busy sum 未取得，不能将本表冒充这两项验收。

4K 已知自写融合 launch 为 10,797 次量化＋833 次激活量化，共 11,630 次，另有 1,193 次显式 contiguous 调用；不包含库 kernel 和布局 kernel。下一步收益重点是扩大融合范围、减少分块提交和重复布局物化，而非仅替换矩阵精度。

### 分阶段时钟门槛未通过

`nr_stage_clock_baseline_v3` 的分段可加性通过，`nr_stage_clock_fused_v3` 出现 **-0.22617 ms** 的相邻事件间隔，故 `valid=false`。总和虽接近整帧间隔，也不能接受负间隔。旧 v2 仅检查分段非负，门槛不足；本报告撤回其作为逐层 GPU 性能对比的用途，保留文件用于诊断。

另外，Head 的主机提交约 306.9 ms，但事件段仅约 25.0 ms，前方流空隙约 282.5 ms；不能据此说 Head 只占 25 ms。需要可信 launch interception/设备 profiler 后再报告纯 GPU 热点。整帧 12 次计时与主机等待一致，可作吞吐/延迟基准，但不能解释为纯计算时间。

## 3. 低分辨率 NR + 实际 FSR 3.1

使用签名和 SHA 验证的游戏 AMD provider，显式选择 FSR **3.1.0**，D3D12 时间戳包围实际 FSR dispatch，非普通 resize 冒充 FSR。输出 4K；上下文常驻，一次 warmup 后 12 次。输入来自同一合成渐变/棋盘 4K 图，经 area 下采样后 NR。合成 depth=.5、motion=0、jitter=0、exposure=1，**每帧 reset=true**，不是游戏缺失数据的替身。

| 模式 | NR 流间隔 ms | FSR dispatch ms | 分开测量相加 ms，非端到端 | NR reserved GB | 相对当前 4K NR PSNR / SSIM |
|---|---:|---:|---:|---:|---:|
| 4K NR | 1492.18 | — | 1492.18 | 4.767 | 自身参考 |
| 1440p NR→FSR→4K | 758.03 | 0.688 | 758.72 | 2.454 | 28.23 dB / 0.9253 |
| 1080p NR→FSR→4K | 475.35 | 0.548 | 475.90 | 1.573 | 28.55 dB / 0.9312 |

1440p/1080p 像素分别为 4K 的 44.4%/25%，NR 延迟分别为 50.8%/31.9%：明显随像素数下降，但不是完全线性。FSR 本体远小于当前 NR 开销；即使 1080p 路线也远未达到 16.67 ms/帧。

NR allocator live 峰值分别 3.951/2.063/1.338 GB；device-wide 帧后采样分别 5.203/2.860/1.992 GB。FSR 单独进程 local usage 1440p 为 0.453 GB、1080p 为 0.357 GB。**NR 与 FSR 在分开的进程测量，中间通过文件交换；没有测量同时常驻的总显存峰值、GPU 互操作和端到端耗时。** GB 为十进制；采样不是连续硬上限证明。

质量为固定范围 RGB、11×11 uniform SSIM、无逐帧归一化，仅当前候选之间的一张静态合成图差异，不是 RTX 画质。1440p 未必更高分说明不能用单样本排名。不能据此宣布 NR 效果保留、运动稳定、细节或 HDR 合格。真实输入审计暂无完整连续 pre-FSR color/depth/motion/jitter/exposure 集合；见 [FSR 输入约定](NR_FSR_ROUTE.md)。

## 4. 复现与 evidence

从仓库根目录运行 PowerShell；需要本地私有模型和既有合成 fixture，公开 source-only checkout 不附权重、vendor DLL 或结果。所有输出目录必须全新。游戏必须退出，GPU 实验串行；各阶段先 1 次通过再 12 次，失败/设备错误/超时即停止，不自动重试或修改 TDR。

```powershell
$py = "$PWD\.venv-rocm\Scripts\python.exe"
.\tools\native_fusion_probe\build.ps1 -OutputDirectory "$PWD\results\fusion_repro_build"
$dll = "$PWD\results\fusion_repro_build\native_fusion_quantize.dll"
& $py scripts/validate_native_fusion.py --dll $dll --operation quantize --iterations 1 --output results/repro_quant_one
& $py scripts/validate_native_fusion.py --dll $dll --operation cubic --iterations 1 --output results/repro_cubic_one
# 检查上述 child.json checks_pass 和正常退出后，分别改为 iterations 12 和全新 output。
& $py scripts/benchmark_nr_resolution.py --size 1080 --iterations 1 --output results/repro_base1080_one
& $py scripts/benchmark_nr_resolution.py --size 1080 --iterations 1 --fusion-dll $dll --reference-output results/repro_base1080_one/output.rgba16f --output results/repro_fused1080_one
# 两者通过后，分别以 iterations 12、新 output 重跑；1440/2160 同样先 1 再 12。
# --count-operators 是额外不计时 pass；--stage-timestamps 只作时钟诊断，当前不可作 GPU 分层验收。
.\tools\nr_fsr_probe\build.ps1 -OutputDirectory "$PWD\build_nr_fsr_repro"
$nrInput = "$PWD\results\repro_fused1080_one\output.rgba16f"
$nrHash = (Get-FileHash -LiteralPath $nrInput -Algorithm SHA256).Hash
.\tools\nr_fsr_probe\run.ps1 -Executable "$PWD\build_nr_fsr_repro\nr_fsr_probe.exe" -InputFile $nrInput -InputSha256 $nrHash -Width 1920 -Iterations 1 -OutputDirectory "$PWD\results\repro_fsr1080_one"
# 通过后改 Iterations 12、新目录；1440p 使用对应 NR 文件及 Width 2560。
```

正式本轮 evidence：`results/20260906_nr_{resolution,fused}{1080,1440,2160}_twelve_v2/child.json`，FSR `results/20260906_nr_fsr{1080,1440}_twelve_v1/report.json`。Hash 绑定的聚合命令（源路径固定为本轮 evidence）：

```powershell
C:\DATA\Tools\ANACONDA\python.exe scripts/summarize_dual_nr_routes.py --output results/dual_routes_reaggregate.json
C:\DATA\Tools\ANACONDA\python.exe scripts/compare_nr_fsr_static.py --output results/static_quality_reaggregate.json
C:\DATA\Tools\ANACONDA\python.exe -m pytest tests -q
```

本轮 CPU 测试 828 passed。GPU 全部正常退出、allocator 释放为零；最终审计 baseline/game 文件 hash 未变，无本轮检测到的设备重置/异常关机事件，不代表长期稳定性已通过。

## 5. 下一步与未关闭的门槛

1. 获取可信 GPU launch/时间 trace，分开 CPU 提交空隙与设备计算，修复分段计时门槛；不再用 ATen 数代替 dispatch。
2. 优先 Head/Pre 的布局＋量化＋激活与矩阵 epilogue 融合，再 C32/C64；对每次改动保留 unfused、完整图像逐位 gate、1/12 benchmark。完整模块 kernel 尚未实现。
3. 捕获真实 pre-FSR 连续资源与 metadata，再把 NR→FSR 放在同一常驻 GPU 链路中；验证同步、总显存、实际端到端延迟以及遮挡/切镜时序质量。不能用本轮静态零运动数据替代。
4. 目前不提升为游戏默认，也不宣称达到 DLSS5 质量或 4K60。现有数据支持继续融合优化和低分辨率实验，不支持关闭最终里程碑。
