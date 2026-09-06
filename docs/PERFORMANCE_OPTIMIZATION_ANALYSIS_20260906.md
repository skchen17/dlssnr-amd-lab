# 截至 2026-09-06 的性能优化分析

本文汇总当前 AMD 原生 NR 路径已经完成的性能工作、有效证据、失败候选和下一阶段
决策。本文讨论的是执行效率，不等同于 RTX/DLSS5 画质验收、游戏稳定接入或 4K60
完成声明。

## 1. 当前基线与最终目标

当前完整网络已经能在 RX 9070 XT 上使用原始权重执行动态尺寸整帧推理，但热路径仍然
混合了 Python/PyTorch 调度、window batch、多个独立 WMMA/点算子 kernel、动态张量和
FP16 中间结果。当前选定配置的 4K 热整帧约为 1.2 秒，距离 4K60 的 16.67 ms 预算仍
有约两个数量级差距。

目标执行结构是：

```text
当前：Python 调度 + window split/loop/cat + 小 kernel + 全局中间 tensor
目标：whole-grid dispatch + bounded fusion + resident layout/FP8 + C++/HIP NRPlan + graph replay
```

所有性能候选遵守以下约束：原始权重、网络拓扑、窗口坐标、颜色与合成约定冻结；严格
轨必须逐位一致；近似 FP8 轨单独标记，不能自动进入游戏；任何 kernel 数下降都必须由
模块和整帧实测确认。

## 2. 已完成优化总览

| 模块/工作 | 优化前 | 当前候选或结果 | 数值 | 决策 |
|---|---|---|---|---|
| Head 早期矩阵链 | 大量分批和中间节点 | graph 节点 112→8 | bitwise exact | 保留 |
| Head bounded attention | 完整 Head 9 dispatch | 4 dispatch；Q/K/V/score 留在20 KiB LDS | bitwise exact | 640×384 已晋级显式候选 |
| Pre project/pack | 输入构建、投影、packing 分离 | 单个 whole-grid kernel | bitwise exact | 边界候选保留，整帧未晋级 |
| Pre/C32 whole-grid | Python window batching | 整批一次提交 | 输出哈希一致 | 128整帧回退0.7%，不晋级 |
| C32 早期矩阵链 | 多批次/多节点 | graph 节点 55→7 | bitwise exact | 早期显式候选 |
| C32 20 KiB 大融合 | 含FFN共7 launch | 2 launch | 修复后bitwise exact | 16窗口慢40%，拒绝 |
| C32 低LDS三段 | 7 launch | 4 launch；12/8 KiB LDS | bitwise exact | QKV norm串行化，拒绝 |
| C32 保守核心融合 | 7 launch | 5 launch；只融合QK/softmax/PV | CPU/ISA通过 | 等待GPU门禁 |
| C64 完整矩阵链 | 65节点 | 7节点 | bitwise exact | 2.38→2.52 ms，拒绝 |
| C128 完整矩阵链 | 65节点 | 7节点 | bitwise exact | 4.44→12.51 ms，拒绝 |
| C256 FFN大融合 | 65节点 | 48节点 | bitwise exact | 13.33→70.37 ms，拒绝 |
| C512 grouped FFN | 88节点 | 61节点 | bitwise exact | 7.99→4.36 ms，保留 |
| 4K C512-only | Head+C32 基线 | 额外启用C512 grouped FFN | 输出哈希一致 | 1237.02→1204.95 ms，−2.59% |

## 3. Output Head

### 3.1 实现

Head 先完成 FFN、QKV、normalization、QK、softmax、PV、projection 和 tail 的原生
WMMA/精确点算子边界。随后增加 `head_attention_window_fused`：一个128线程工作组负责
一个8×8窗口，Q/K/V和64×64 score/probability 保存在20 KiB LDS中，完整 Head 从9次
dispatch降为4次。

### 3.2 正确性和性能

640×384 两组独立12次门禁均逐位一致：

| 组 | 基准 | bounded Head | 收益 |
|---|---:|---:|---:|
| v1 | 2.6887 ms | 2.5362 ms | 6.0% |
| v2 | 2.7224 ms | 2.4788 ms | 9.8% |

首组 PyTorch 峰值分配从255.37 MiB降至74.41 MiB。有效 RGP A/B 进一步得到：

| 指标 | 9-dispatch | 4-dispatch bounded |
|---|---:|---:|
| sampled active time | 3.50248 ms | 2.68660 ms |
| inactive gaps | 1.17696 ms | 0.84232 ms |
| fetch traffic | 289.98 MB | 196.63 MB |
| write traffic | 303.49 MB | 156.43 MB |
| memory-unit busy | 87.68% | 80.15% |
| memory-unit stalled | 17.17% | 4.06% |
| L0 hit | 37.52% | 51.46% |
| L2 hit | 68.91% | 76.38% |

这说明 Head 基准主要受内存流水线和缓存流量影响，而非已经打满峰值 DRAM 带宽。融合
通过减少全局中间值取得了真实收益。FP16 bounded kernel 为121 VGPR、52 SGPR、20 KiB
LDS、0 scratch、20个WMMA站点，静态 occupancy 上限37.5%。Head 工作量足够大，减少
流量的收益能够超过 occupancy 损失。

## 4. Pre-block 与 whole-grid 调度

`native_fusion_pre_project_pack` 将反射采样、条件构建、原始 FP16 16→32 投影和最终
C32 packed 写入合并。位置噪声生成保留在外部，避免同时改变特殊函数语义。

| 尺寸 | 原路径 | fused project/pack | 收益 |
|---|---:|---:|---:|
| 128×128 | 0.9073 ms | 0.1475 ms | 6.15× |
| 640×384 | 1.3301 ms | 0.3262 ms | 4.08× |

这是边界级收益，不是完整 Pre 或完整网络收益。启用 Pre/C32 whole-grid 后，128×128
完整网络 A-B-B-A 输出哈希一致，但进程中位数均值为154.84→155.97 ms，回退0.7%。
结论是 Python batching 确实应移除，但必须同时设计合适的 tile、occupancy、缓存布局和
原生 plan；单纯把 batch 拼成一次大调用不会自动加速。

## 5. C32 融合演进

### 5.1 20 KiB 单核

第一版将 QKV、normalization、QK、softmax、PV 和 projection 全部放入一个窗口 kernel，
含 FFN 的 launch 从7降为2。最初出现320个half差异，根因是误用了 Head 的三个舍入
边界：QK accumulator、PV K32分段和projection residual。按C32语义修复后，1窗口和
16窗口均逐位一致。

16窗口下性能却从0.12120 ms变为0.16963 ms，仅为基准的0.714×；峰值分配从
1,305,088 B降到649,728 B。主要原因是：

- 每窗口仅产生一个工作组，16窗口无法填满64 CU；
- 20 KiB LDS把静态 occupancy 限制在37.5%；
- 多次 `__syncthreads` 使四个 wave 串行跨越所有 attention 阶段；
- 小规模数据能较好命中缓存，节省的显存流量不足以覆盖并行度损失。

### 5.2 12 KiB + 8 KiB 三段版

第二版拆成 QKV+norm、QK+softmax+PV、projection 三段，含 FFN 共4个launch。静态资源：

| kernel | FP16 VGPR/SGPR | LDS | scratch | WMMA | 静态occupancy |
|---|---:|---:|---:|---:|---:|
| QKV+norm | 111/52 | 12 KiB | 0 | 2 | 62.5% |
| attention core | 99/12 | 8 KiB | 0 | 16 | 75.0% |

单窗口逐位一致，但0.32934 ms慢于0.15870 ms基准。原因不是LDS，而是融合后的QKV kernel
让每个wave串行处理16行×3个Q/K/V部分；原normalization可让独立warp并行处理各向量。
该候选已拒绝。

### 5.3 当前保守版

当前 `c32_attention_core` 恢复原并行QKV projection和normalization，只保留8 KiB的
QK→softmax→PV核心，再接原projection：

```text
C32 fused FFN
  → parallel QKV projection
  → parallel normalization
  → 8 KiB fused QK + softmax + PV
  → output projection + residual
```

含FFN共5个launch，仍少于基准7个；score和probability不再落显存，Q/K/V仍保存在GPU
workspace。该方案已通过CPU、ABI和ISA检查，但尚未执行GPU门禁，因此没有性能声明，
也没有成为默认路径。

## 6. C64/C128/C256/C512

C64/C128 的第一轮全矩阵链证明“大LDS、大kernel、统一wave配置”不适合所有通道家族。
C64独立FFN曾有约62%局部收益，但连接attention后完整块从2.38变为2.52 ms；C128从
4.44变为12.51 ms。C256 FFN由于每wave需要32 KiB LDS，768窗口从13.33变为70.37 ms；
4-wave配置会需要128 KiB，入口直接拒绝。

C512 的 grouped fusion 是成功对照：只融合八个独立64→256→64 MLP，512×512跨组投影
继续交给库GEMM。局部hidden用完立即释放，完整块7.99→4.36 ms。它证明 RDNA4 更适合
按head/group做有界融合，而不是把完整4C hidden放进LDS。

目前 C64/C128 已增加“每个(window, head)一个工作组”的20 KiB bounded候选，attention
从6段降到“局部attention + 跨head projection”2段。FP16均为101 VGPR、20 KiB LDS、
0 scratch、17个WMMA站点，静态occupancy 37.5%；FP8版本为67 VGPR。它们仅完成编译、
ABI和CPU策略检查，尚未通过GPU门，不能启用。

## 7. FP8、常驻数据与原生运行时状态

当前 `wmma_fp8` 候选确实生成 gfx1201 FP8 WMMA，但仍在kernel入口附近把已解码FP16
转换/打包为FP8，并不是真正的resident FP8数据路径。已经完成的是权重缓存失效规则、
显式精度策略和FP8/FP16双轨基础设施；尚未完成的是：

- 初始化时永久预打包FP8权重；
- 只在语义允许的位置让activation以FP8格式常驻；
- 跨kernel直接消费FP8，取消FP8→FP16→FP8往返；
- 用原生workspace arena统一管理生命周期。

Python仍是reference oracle、correctness harness和当前部署桥。固定C++/HIP `NRPlan`、统一
workspace arena、参数预绑定和HIP Graph重放尚未完成，因此完整帧仍承担Python对象、动态
allocation和提交间隙。这是后续整帧数量级优化的必要工作，而不是可选收尾。

## 8. 显存与4K结果

C512-only 4K A-B-B-A 的两侧均值为1237.02→1204.95 ms，下降2.59%，输出哈希保持
`46F357EC1DA510A7CB7BC6D141E8A83DBA4DAD24179D72DC18BD150CD573056F`。候选路径：

- allocator live峰值：4,097,502,720 B；
- reserved：4,773,117,952 B；
- 设备用量离散采样峰值：5,202,509,824 B；
- 进程退出后分配和保留均为0。

它满足本轮5 GB allocator/6 GB设备用量门，但离4K60仍极远。当前证据说明局部kernel
优化有效，却不足以弥补完整执行计划、分辨率规模和调度结构的总体开销。

## 9. Profiler证据和安全边界

只有Output Head取得了完整、有效的RGP A/B数据。随后两次Pre模块counter capture与
`LiveKernelEvent 141`、watchdog和系统重启相关；两个不完整目录无manifest和有效RDF，
不能作为性能证据。RGP counter collection继续由`safety/GPU_PROFILE_HALT.json`硬锁定。

用户明确指示后，只允许每次一个最小非profiler正确性门；禁止自动重试、压力循环、修改
TDR或频率。最近的C32门均正常退出、完全释放显存，且没有新增141事件，但这不证明历史
故障根因已经解决。

## 10. 当前瓶颈判断

按现有可靠证据，瓶颈优先级为：

1. Python/window batching、动态allocation和跨kernel提交间隙；
2. Pre、Head、C32等高分辨率阶段的中间张量和布局流量；
3. 不合适的大LDS融合导致occupancy和工作组数量下降；
4. FP8尚未常驻，转换和FP16往返仍在；
5. 缺少固定C++/HIP plan和graph replay；
6. 更深层C64/C128/C256需按家族独立tile/bounded方案，而不能复用统一参数。

ViT目前不是已知最大热点，继续保持基准，除非未来可靠profiler证据改变排序。

## 11. 下一阶段顺序

1. 对当前C32保守核心融合执行1窗口正确性门；通过后才测试16窗口和真实stage输入。
2. 若C32保守版在16窗口仍无收益，停止该边界，转向原生plan而非继续堆叠融合。
3. C64/C128先做单窗口数值门，再根据资源结果决定是否设计8/12 KiB分段版。
4. 建立C++/HIP `NRPlan`：权重预打包、固定workspace、固定tensor/layout计划和graph重放。
5. 在语义确认的位置建立真实resident FP8链，并与严格FP16轨分别验收。
6. RGP保持锁定；恢复任何counter collection前必须单独审查驱动/watchdog故障。

## 12. 证据索引

- 总矩阵结果：`docs/RDNA4_MATRIX_FUSION_RESULTS.md`
- 新执行引擎记录：`docs/RDNA4_NATIVE_ENGINE_REFACTOR.md`
- GPU事故与锁停：`docs/GPU_PROFILING_INCIDENT_20260906.md`
- Head bounded：`results/20260906_head_bounded_attention_gate640x384_v1`、`v2`
- Head RGP：`results/20260906_rgp_head_bounded_compare_640x384_v1`
- Pre project/pack：`results/20260906_pre_project_pack_gate128_v2`、`gate640x384_v1`
- C32 corrected bounded：`results/20260906_c32_bounded_attention_rounding_fix_gate1_v1`、`gate16_v1`
- C32 staged：`results/20260906_c32_staged_attention_build_v1`、`gate1_v1`
- 4K C512 A-B-B-A：`results/20260906_c512_only_abba_a1`、`b1`、`b2`、`a2`

`results/`和`local_models/`均不进入Git：报告保留可复查的目录名和摘要，私有权重、教师
数据、原始帧、编译产物和大型profile不会上传公共仓库。

提交前完整CPU回归为`849 passed in 54.02s`；提交范围敏感信息扫描和`git diff --check`
通过，仅保留Windows工作树既有的LF/CRLF转换提示。
