# 完整原生 71-block：阶段与逐操作计时重测

## 结论与计量边界

RX 9070 XT / gfx1201，1920×1080 输入，内部 padded 1920×1152，原生近似 FP8
reset/single-color 路径，固定随机种子 17071。不是实际游戏序列，不包括原始时序或 D3D12/HIP 接入。
原图 510 kernel + 5 memcpy，计时图另外插入 516 个 HIP event-record 节点。
捕获图经过单链验证，不更改 kernel、权重、输入、执行依赖或数学关系。

**本轮未插桩整图 median 212.908 ms，P95 260.854 ms（48 帧）。**
计时图整帧 median 213.190 ms，P95 245.343 ms（48 帧）。
此前 162.726 ms 是历史单帧诊断，不能继续作为当前稳定热态性能。
这次并未做算子性能优化，不能把两个时点的差异判定为重构退化。

所有时间均为 HIP event 区间，包含该区间内的调度间隙、event 成本与可能的设备抢占；
**不是硬件 kernel-busy 时间**。RGP counters 没有运行，缺少频率/带宽/占用率动态证据，
不能把本次波动归因于某一个确定原因，也不能通过减去一个固定值“校正”每个操作。
两类整图 pooled median 接近不代表每个短 kernel 均不受插桩影响。

## 方法与复现

先单帧校验，再两组 B-A-B-A；B=原图，A=逐节点 event 计时图。
每进程初始化后 5 次 warmup，再 12 次测量，GPU 进程串行。
输出读回/hash 在整图 event 区间之外；每次测量后检查 hash，因此它不是连续游戏帧间距测试。
各操作先按 stage 聚合同一帧内的调用，再对有效帧取 median；阶段总时间也先逐帧求和再取 median。
**各操作的 median 不保证相加等于阶段 median，各阶段 median 也不保证等于整图 median。**

| 顺序 | 模式/结果目录后缀 | 整帧 median ms |
|---|---|---:|
| 1 | B / b0 | 244.069 |
| 2 | A / a0 | 209.469 |
| 3 | B / b1 | 211.622 |
| 4 | A / a1 | 207.842 |
| 5 | B / b2 | 202.830 |
| 6 | A / a2 | 240.683 |
| 7 | B / b3 | 206.603 |
| 8 | A / a3 | 212.590 |

原始目录统一为 `results/20260907_native_operations_{a0..a3,b0..b3}`。
每个目录保存进程命令、12 帧整图时间、hash 和健康检查；A 目录另含
`operations_00.json` 至 `operations_11.json` 及原图 `original_graph.dot`。
每个节点有 stage/block、原始函数名、grid/block、LDS/register/local 属性和 event 时间。
名称通过原图 DOT 依赖拓扑匹配，不能依赖 GraphGetNodes 返回次序。

从仓库根目录运行（必须先审查并重新授权非 counter 测量，当前测试额度已消费）：

```powershell
# 重建到新目录时，更新 measure_native_operations.ps1 中的 DLL 路径；不覆盖旧构建。
& tools/native_nr_plan/build.ps1 -OutputDirectory C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\YOUR_NEW_TIMING_BUILD

# B：无插桩；A：每个节点插入时间戳。OutputName 每次必须唯一。
& scripts/measure_native_operations.ps1 -OutputName YOUR_NEW_B0 -Samples 12 -Warmup 5
& scripts/measure_native_operations.ps1 -OutputName YOUR_NEW_A0 -Instrument -Samples 12 -Warmup 5

# 仅 CPU 汇总，始终可以运行。输出目录必须是新目录。
.venv-rocm\Scripts\python.exe scripts/report_native_operation_timings.py --input results/20260907_native_operations_a0 results/20260907_native_operations_a1 results/20260907_native_operations_a2 results/20260907_native_operations_a3 --baseline results/20260907_native_operations_b0 results/20260907_native_operations_b1 results/20260907_native_operations_b2 results/20260907_native_operations_b3 --output results/YOUR_NEW_SUMMARY --reject-invalid-samples
```

## 阶段总开销

47 个有效节点计时帧；通道家族包括 Encoder + Decoder 的全部对应 block。

| 阶段 | kernel + copy 数 | median ms | min–max ms |
|---|---:|---:|---:|
| Pre | 6 | 20.566 | 17.311–30.760 |
| C32 | 48 | 36.570 | 31.349–47.095 |
| C64 | 64 | 26.608 | 23.909–38.793 |
| C128 | 96 | 33.405 | 30.662–38.187 |
| C256 | 128 | 40.558 | 37.271–46.227 |
| C512 | 112 | 25.216 | 23.169–28.063 |
| ViT | 40 | 13.068 | 10.946–14.433 |
| 8 个尺度切换 | 8 + 4 | 3.495 | 3.255–6.217 |
| ViT 前后 bottleneck | 2 + 1 | 0.603 | 0.450–0.785 |
| Head | 6 | 15.704 | 13.355–19.580 |

## Pre / Head 操作

每项为实际融合 kernel，内部数学操作不能单独计时；拆开计时将改变执行方式。

| 操作 | Pre ms | Head ms |
|---|---:|---:|
| Pre 输入构建/conditioning + 16→32 投影 + pack；Head 输入/skip 合成 | 4.203 | 2.009 |
| FFN expand/activation/contract/residual 融合 | 9.819 | 7.188 |
| QKV projection + normalization + FP8 publish 融合 | 2.217 | 2.459 |
| QK + softmax + PV 融合 | 1.635 | 2.033 |
| attention output projection + residual | 1.143 | 1.202 |
| Pre pool/skip 输出；Head tail/输出写入 | 1.194 | 0.998 |

## C32 / C64 / C128 / C256 操作

所有对应 block 的同类操作累计；C32 为 8 block，C64 为 8，C128 为 12，C256 为 16。

| 操作 | C32 ms | C64 ms | C128 ms | C256 ms |
|---|---:|---:|---:|---:|
| C32 完整 FFN / 其他 grouped FFN | 21.083 | 14.821 | 20.723 | 26.444 |
| FFN 跨组 mix | 已融合 | 1.294 | 1.792 | 1.997 |
| QKV projection/norm/FP8 publish | 4.836 | 3.094 | 4.233 | 6.041 |
| QK/softmax/PV | 4.085 | 2.237 | 1.780 | 1.353 |
| output projection/residual | 2.469 | 2.166 | 2.435 | 2.403 |
| FP16→E4M3 pack（每帧分别 8/16/24/32 次） | 1.701 | 1.582 | 1.130 | 0.850 |
| scatter/layout 输出 | 2.426 | 1.487 | 1.607 | 1.492 |

## C512 / ViT 操作

| C512 操作（16 次/帧） | ms | ViT 操作（8 次/帧） | ms |
|---|---:|---|---:|
| FFN preprojection | 5.766 | FFN expand | 2.753 |
| grouped FFN | 3.524 | FFN contract/residual | 3.586 |
| FFN output projection | 2.725 | QKV/norm | 2.753 |
| QKV/norm 融合 | 9.375 | global attention | 2.706 |
| QK/softmax/PV | 0.955 | output projection | 1.158 |
| attention projection/residual | 2.139 | — | — |
| scatter | 0.666 | — | — |

## Scale transitions / Bottleneck 操作

| 操作（每项一次/帧） | ms |
|---|---:|
| Encoder C32→C64 pool/projection/量化/layout | 0.598 |
| Encoder C64→C128 | 0.676 |
| Encoder C128→C256 | 0.679 |
| Encoder C256→C512 | 0.729 |
| Encoder 4 次边界 memcpy 合计 | 0.030 |
| Decoder C512→C256 projection/扩展/skip/layout | 0.135 |
| Decoder C256→C128 | 0.121 |
| Decoder C128→C64 | 0.185 |
| Decoder C64→C32 | 0.287 |
| Encoder final→ViT | 0.492 |
| Bottleneck memcpy | 0.015 |
| ViT→Decoder input | 0.094 |

## 正确性、安全与异常

96 个测量帧均有限且输出 SHA256 相同：
`2A756063FB97D2F88BE384A09129F085055EE08C1E9917B1BF8005270DDB56E3`。
它证明与当前原生近似基线一致，不代表 NVIDIA/严格参考一致。
所有成功进程正常释放资源；健康检查未发现新增 Event 41/6008、WER LiveKernelEvent 或 watchdog 文件。
RGP counter halt 未解除，没有修改 TDR，没有游戏运行、1440p/4K 测试或训练。

首次诊断 v89 已导出节点时间，但读取名称时把 graph function pointer 当作 hipKernel_t，
产生 host-thread sticky invalid-handle 错误，故该结果整体不用于性能汇总。
v90 移除此查询，使用 DOT 名称；单帧正确性验证通过后才扩大测量。

正式 a0/operations_02.json 的 index505（C32/block69）出现 **−1.378059983 ms**。
这是无效计时，不能宣称只是亚微秒精度误差；整帧 515 个操作时间从汇总中排除，
不取绝对值、不归零、不仅剔除慢值。原始数据保留，47/48 个计时帧参与操作统计。
该帧整图区间为正且 hash 通过，因此仍保留在单独的整图 48 帧结果中。
完整 frame/operation 源路径、异常节点、各节点 min/max 与构建/输入/权重哈希见 JSON 汇总。

## 可用结论与缺失项

当前主要热点是 C256/C32/C128 的 FFN，而不是仅剩 launch 数的问题；
C512 的 QKV/norm 是该家族最大单项。Pre 与 Head 合计也不应忽略。
这些数据足以确定下一轮调查顺序，但运行间波动不允许声称百分之几的小收益已成立。
无法给出融合内部每次 MMA/activation 的独立毫秒数，也没有真实 DRAM/L2 流量和动态 occupancy。
下一轮若需要更精细的因果归因，应先降低 event 测量扰动、重复验证计时稳定性；不能自动恢复 RGP。

全部 515 个执行节点（逐 block）的明细：
[`operations.md`](../results/20260907_native_operations_report_final/operations.md)。
机器可读数据和来源哈希：
[`summary.json`](../results/20260907_native_operations_report_final/summary.json)。
