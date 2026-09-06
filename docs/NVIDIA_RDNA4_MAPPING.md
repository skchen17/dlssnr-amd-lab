# NVIDIA NR 执行方式与 RX 9070 XT 映射分析

2026-09-06；只读原模块/既有基准和官方资料。本轮不运行 GPU、不改模型、权重、驱动或游戏。

## 结论

借鉴 NVIDIA 的模块内融合和低精度片上数据流，但重新设计 AMD 的矩阵片段布局及活跃数据寿命。
不是照抄 PTX，也不是将所有计算强塞进一个 kernel。优先 Head、Pre、C32，再推广其他通道家族。
当前没有硬件利用率、DRAM 流量和真实 NVIDIA 4K 同输入时间，不能保证十倍提升或 4K60。

## 1. 原版可以确认的细节

沿用 [原始审计及哈希](NVIDIA_EXECUTION_GRANULARITY.md) 中的原 DLL 解出模块、5070
640×360 variant02 捕获。此次重新运行审计，代表条目结果如下（静态指令站点）：

| 原 slot | 线程块 | FP8 MMA | FP16 MMA | 显式存储线索 |
|---|---|---:|---:|---|
| 1 Pre | 32×1 | 256 | 16 | global 输出、shared 交换 |
| 3 C32 | 32×1 | 256 | 0 | 四处 128-bit global 输出站点 |
| 7 C64 | 32×2 | 304 | 0 | global 输出、shared 交换 |
| 154 Head | 32×1 | 256 | 16 | 两处 surface store 站点 |

FP8 指令为 `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`；
FP16 为 `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`。
Head entry 静态审计没有发现显式 `st.global` / `ld.local` / `st.local`，这加强了
模块内部避免显式全局临时张量的证据，但 PTX 不是最终 SASS，不能据此排除编译器 spilling。

原版一帧 156 次 launch，Pre/Head 各一次，C32/C64 每个 block 一次；
C512、ViT 则分别多 kernel，说明原版也选择融合边界，而非“全网单 kernel”。
同 command-list、release flag 证明调度/依赖设计，不能证明任意跨 kernel 并行。
不要在 AMD 上照搬自旋等待协议：不同调度与资源占用可能导致生产者无法获得执行机会。

## 2. AMD 有硬件能力，但算力数字要用对

RX 9070 XT 官方峰值：FP16 dense matrix 195 TFLOPs、FP8 dense matrix 389 TFLOPs，
显存带宽最高 640 GB/s，16 GB 显存，64 MB Infinity Cache。
这些是峰值，不是本模型实测；779 TFLOPs 是结构化稀疏 FP8，不能用于当前未证明满足稀疏条件的权重。
[AMD 产品规格](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html)

因此仅从已高效 FP16 matrix 换成 FP8 matrix，理论吞吐优势约 2×，不是天然 10×。
更大收益若存在，应来自利用率、融合、布局、减少临时流量的共同改善，不能把各项倍率相乘当预测。
当前 Head 有 FP32 operand GEMM，但在读取实际 ISA 前，不宣称全部退化为标量/SIMT。

## 3. 矩阵指令不是一对一替换

NVIDIA 上述 FP8 MMA 是 M16/N8/K32、FP16 accumulator/output。
gfx12 的 wave32 FP8 builtin 是 M16/N16/K16、FP32 accumulator/output：
`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`。
这不等于两平台数学块数量存在固定速度比；要重新分解 N/K 并映射片段。
[LLVM builtin 定义](https://raw.githubusercontent.com/llvm/llvm-project/main/clang/include/clang/Basic/BuiltinsAMDGPU.td)

FP32 accumulator 最后转换 half，不保证等于 NVIDIA 原 MMA 的累加/舍入，
也不保证与当前 PyTorch 库 GEMM 逐位一致。精度更高不等于输出更一致。
继续保留当前结果为 oracle；新 WMMA 路径先做累加、转换、归约和整帧 gate，失败不升级默认。
若需要容许误差的新精度策略，必须另列候选，不把它伪装成无数值变化优化。

本地 SDK `include/hip/amd_detail/amd_hip_fp8.h` 对 gfx1201 明确选择 OCP FP8，
而 gfx942 分支选择 FNUZ。移植时用 E4M3 不能误用 E4M3_FNUZ；检查饱和、subnormal、NaN、signed zero。
不要直接套用面向 MI300/CDNA 的 MFMA 内核或假设其格式/指令适用于 gfx1201。

## 4. RDNA4 最值得利用的是片段布局

RDNA4 改变了 WMMA 寄存器布局，减少 RDNA3 的输入复制。AMD 官方给出了同 kernel 连续 GEMM，
让前一次结果在寄存器中接入后一次计算的例子。应以片段布局设计推理计算，而不是每个子算子还原
NVIDIA packed tensor 再转标准 tensor。永久权重可以加载时精确重排；源权重不可修改。
[RDNA4 matrix cores](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)

具体可测试：把特征安排为适合后续输入片段的方向，必要时利用转置等式重新排列矩阵计算，
同时正确转换 bias、residual、归一化的坐标，不能只是交换 A/B 指针。
[AMD fused GEMM 方案](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/)

矩阵加载和“已计算的寄存器结果转置”是不同问题。RDNA4 的转置 global load 不能消除所有
attention 片段变换。AMD 提供过 WMMA+单位矩阵的寄存器内转置方案，可作为候选，
但会消耗矩阵执行资源，也要验证特殊值；不默认比 shuffle/LDS 更快。
[寄存器内转置](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-3/)

对于 FP8 数据，可把两个 K16 片段配对加载以使用 128-bit 向量读取，再分别执行矩阵指令。
它是供数优化，不是把矩阵计算量减半。官方 wide-K 示例主要展示整数验证，不能据浮点“结合律”
推导本项目的 FP8/FP16 路径逐位相同；不能移动我们明确保留的舍入边界。
[Wide-K WMMA](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/)

## 5. 为什么不能盲目整块塞入寄存器

以一个 64-token C32 窗口的逻辑张量大小作静态估算：

| 张量示例 | FP16 容量 |
|---|---:|
| 输入 64×32 | 4 KiB |
| FFN 展开 64×128 | 16 KiB |
| QKV 64×96 | 12 KiB |
| 单头 attention scores 64×64 | 8 KiB |

这些不是同时必须驻留的集合。若把完整 64×128 FP32 展开 accumulator 同时分摊给单 wave32，
光此一项就有 256 个 32-bit 元素/线程，还没有输入、权重和其他临时量。
应沿 token/channel 子片段流水执行、及时复用寄存器，必要时使用有界 LDS。
这只是算术资源估算，不是编译后 VGPR 数或占用率。

推荐比较 1/2/4 个 wave32 每工作组，以及“完整块”和“FFN/attention 两段”候选。
LDS 是片上共享存储，和 4–6 GB 显存容量不是一回事。多占显存不会增加寄存器或 LDS；
不能靠堆缓冲解决 spilling。看编译 VGPR、scratch、LDS 和实测占用率决定融合边界。
[AMD occupancy 原理](https://gpuopen.com/learn/occupancy-explained/)

## 6. 显存带宽为什么不能忽略

静态举例：3840×2176×32 的 FP16 张量约 0.535 GB，一次完整写回再读约 1.07 GB。
按 640 GB/s 峰值算，即使只有这一对显存传输也约 1.67 ms；这不是实际测得的 DRAM 流量，
缓存命中会改变它。FFN 展开张量更大，而整个 60FPS 帧只有 16.67 ms，游戏本身还要用 GPU。
因此 4–6 GB 内“装得下”和“跑得快”没有等价关系。允许的额外显存优先用于精确预打包权重、
常驻工作区和有界双缓冲，不用于更多全分辨率中间结果。

## 7. 分阶段建议

| 家族 | AMD 首选研究对象 | 保留的边界/风险 |
|---|---|---|
| Head | 加载+skip → FFN 矩阵融合；QKV/norm；局部 attention；tail/写图 | 保留当前分段舍入；先几段融合，再看完整 Head |
| Pre | 坐标噪声/颜色+16→32 投影；Swin；pool 与两路输出 | 噪声特殊函数误差、边缘、旧无界 GEMM 失败 |
| C32 | native packed load→矩阵链→store，FFN/attention 内不落大 tensor | shifted windows、跨窗口 halo、encoder/decoder 边界 |
| C64/C128 | 更多 wave 协作，明确 LDS 生命周期 | 多头布局、量化、下采样/上采样不同 |
| C256/C512 | 局部融合+高效库 GEMM 对比，不强求单 kernel | 大 accumulator/权重带来的寄存器压力 |
| ViT | 后续优化 QK/softmax/PV 的分块流水 | 不擅自改变全局注意力或其归约数学 |

HIP 优先用于矩阵原型，D3D12 负责已有 GPU pack/unpack 和接入。普通 DXIL shader
不自动保证编译为 WMMA；若选 DXIL 矩阵路径，先验功能、编译 ISA、驱动和同精度性能。
Windows PyTorch 有 gfx1201/FP8 支持，但不代表全套 ROCm 工具均可用；此次本地未找到
rocWMMA 头文件，不能把网站支持表当作已经安装/通过本地验证。
[Windows ROCm 7.2 支持边界](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)

## 8. 计时解释和下一轮证据门槛

[综合结果](RDNA4_MATRIX_FUSION_RESULTS.md) 中保留的早期 graph 1.10–1.33× 只表示所测子图重放收益。
graph 没有减少设备 kernel 数、消除全部设备端 launch 成本或中间张量流量，不能据此宣布
“dispatch 已不重要”。同样，没有 profiler 不能宣称现有路径主要受 DRAM 或矩阵算力限制。

下一轮矩阵原型应记录：正确性 → ISA 中实际 WMMA → VGPR/scratch/LDS → 代表 shape 的
热 GPU/host 时间与实际节点数 → 实际生成的 stage 输入 → 同输入整帧 ABBA。
比较基准包括现有 eager、graph、FP16 fused 和独立标记的 FP8 fused 候选。
优化输出和 RTX 画质是两项不同 gate；保留前者不意味着后者已通过。

离线只读重审命令（仓库根目录）：

```powershell
C:\DATA\Tools\ANACONDA\python.exe scripts/audit_nvidia_execution_granularity.py
```
