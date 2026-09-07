# 1080p 原生推理引擎：Graph census、C++ arena 与 resident FP8

更新时间：2026-09-07。

## 结论

后续的 wide-layout 和四家族 stage-specific QKV norm 已把严格 1080p 图继续降到
**4,653 kernel + 59 memcpy、220.061 ms GPU median**。此前最后一次带 marker 的 census
虽然在首次 graph submit 返回 HIP error 1，但失败前已经完整、只读地导出了 4,725 个
kernel 条目和 DOT；因此本轮没有重跑 GPU，而是离线移除 72 个 marker 并恢复了当前图的
逐节点归属。最新分类见下文“1.1 当前 4,653-node census”。本节保留的 16,558/9,790
数据用于展示重构阶梯，不再代表当前最优图。

本轮把“约 16,558 个节点”的描述变成了逐节点可复现 census，并为 `NRPlan` 增加了
自有权重、自有固定 workspace arena、版本化动态资源绑定和 fail-closed 部署门禁。
当前最终已验证组合进一步启用了既有的、逐位一致的 C32 primitive rewrite。1080p、
每进程 12 帧 A-B-B-A 中，C++ graph replay 从 16,558 降至 **9,790 kernel nodes**，
并从 262.680 降至 **241.059 ms median / 242.588 ms P95 / 4.148 jobs/s**，延迟下降
8.23%，每帧输出 SHA-256 均为
`892FB11D82C642996AD321052B2804A72BAC919620DA9F696BC086D74DA401B7`。

这还不是完全独立部署的 C++ runtime。C++ 确实持有 147,683,776 B 权重副本、
362,045,952 B/26-region workspace arena 和 graph executable，但现有 graph 的 kernel
参数仍指向初始化阶段 PyTorch capture pool；因此 `owns_graph_source=false`、
`graph_references_external_allocations=true`、`deployment_ready=false`。报告和 API 都拒绝
把它误报成“完全 C++ owned”。要通过该门禁，必须把剩余 stage 重写成使用 arena offset
和 C++ 权重地址的 recorder graph。

## 1. 完整 1080p Graph census

诊断 graph 在 71 个 block 周围插入 72 个一线程 marker。原始诊断图有 16,630 个 kernel，
去除 marker 后部署图精确为 **16,558 kernel nodes + 120 memcpy nodes**。每个节点均由
marker 归属到唯一 block；Windows ROCm 的 `hipKernelGetName` 无法解析捕获节点，因此函数名
来自官方 `hipGraphDebugDotPrint` DOT，grid/block/VGPR/LDS/local 属性来自 HIP API。

| 模块 | kernel/frame | HIP event stream interval（ms） |
|---|---:|---:|
| Pre | 2,684 | 7.518 |
| C32 | 5,849 | 26.076 |
| C64 | 1,661 | 24.356 |
| C128 | 1,277 | 35.752 |
| C256 | 1,701 | 37.029 |
| C512 | 1,925 | 17.585 |
| ViT | 1,160 | 12.755 |
| transition | 114 | 0.279 |
| Head | 187 | 15.657 |

模块 interval 合计约 176.999 ms；同一无逐 stage drain 的整条 stream interval 为
238.316 ms，剩余约 61.317 ms 是 block 之间的 stream gap/host submission fragmentation。
这些是非 RGP event envelope，不是硬件 kernel-busy 求和。RGP safety halt 仍启用，因而
没有用 event span 冒充 DRAM/L2/occupancy counter。

以下分类是改写前 16,558-node census 的总数：

| 类别 | 节点数 |
|---|---:|
| pointwise | 8,066 |
| quantize | 2,577 |
| cast | 1,415 |
| other | 1,430 |
| layout | 1,143 |
| activation | 543 |
| copy | 541 |
| norm/reduction | 474 |
| attention | 255 |
| GEMM | 114 |

11,979 个节点被识别为 PyTorch/library，2,843 个为项目 HIP，1,736 个为其它库或无法
可靠归类。由此可以确认当前首要问题不是 ViT 或 GEMM 数，而是 C32/Pre 的 pointwise、
quantize、cast 和 layout primitive。

优先级最高的 family 如下。traffic 列是“每次 primitive 按所在 stage 最大 FP16 feature
读写一次”的排序代理，不是硬件实测流量，也不能相加后声称真实 DRAM 字节数。

| 模块 / family | count/frame | traffic proxy | 所在模块 interval | 优先级 |
|---|---:|---:|---:|---|
| C32 / `aten_add` | 1,348 | 95.410 GB | 26.076 ms | P0：并入 producer epilogue/residual |
| C32 / `quantize_e4` | 1,067 | 75.521 GB | 26.076 ms | P0：producer 直接写 E4M3 resident |
| C32 / `aten_mul` | 756 | 53.509 GB | 26.076 ms | P0：与 norm/scale/quantize 有界融合 |
| Pre / `aten_add` | 621 | 43.954 GB | 7.518 ms | P0：整 stage 原生输入构建 |
| C32 / `aten_index` | 500 | 35.389 GB | 26.076 ms | P0：producer-native packed layout |
| Pre / `quantize_e4` | 497 | 35.177 GB | 7.518 ms | P0：写入 C32 resident layout |
| C64 / `aten_add` | 392 | 13.873 GB | 24.356 ms | P1：grouped/attention 边界 epilogue |
| C256 / `aten_add` | 368 | 3.256 GB | 37.029 ms | P1：按实际时间优先于 ViT |

完整逐节点/逐函数结果位于本地私有实验目录
`results/20260907_nr1080_graph_census_v1/graph_census_ranked_v2.json`。源代码使用
`scripts/summarize_graph_census.py` 可复现；结果目录依发布边界不进入 Git。

### C32 primitive rewrite 晋级结果

审计发现先前已经通过逐位一致门的 `c32_ffn,c32_attention` 没有纳入本轮 16,558-node
控制配置。将它们加入相同 71-block graph 后，单帧门和两组 12-repeat B 进程均通过，
输出哈希不变。新 census 的部署 kernel 为 **9,790**：Pre 524、C32 1,241、C64 1,661、
C128 1,277、C256 1,701、C512 1,925、ViT 1,160、transition 114、Head 187。

相对控制图，Pre 从 2,684 降至 524，C32 从 5,849 降至 1,241；全图减少 6,768 个
kernel（40.8745%）。新图分类为 pointwise 5,246、GEMM 819、quantize 885、cast 569、
layout 438、copy 400、activation 261、attention 255、norm 192、other 725。这里是节点
census，不是硬件 DRAM/L2 counter；RGP safety halt 仍未解除。

### 1.1 当前 4,653-node census

当前严格图按 71 block 的模块归属如下。节点/函数数据来自失败提交前已完成的 HIP Graph
只读导出；没有把失败的执行结果计入性能或 correctness。时间列沿用最近一次正常完成的
非 RGP stage-event 包络，仅用于优先级，不是本次 graph 的 kernel-busy 求和。

| 模块 | kernel/frame | 最近 stage interval（ms） |
|---|---:|---:|
| Pre | 471 | 7.518 |
| C32 | 817 | 26.076 |
| C64 | 565 | 24.356 |
| C128 | 305 | 35.752 |
| C256 | 405 | 37.029 |
| C512 | 629 | 17.585 |
| ViT | 1,160 | 12.755 |
| transition | 114 | 0.279 |
| Head | 187 | 15.657 |

类别为 pointwise 1,115、GEMM 819、other 793、quantize 681、layout 377、cast 297、
attention 255、copy 196、activation 64、norm/reduction 56。实现来源中项目 HIP 为 1,934、
PyTorch/library 为 1,963、其它库或无法解析为 756。节点最多的单一 family 已变为 ViT
`aten_add`（256 次），其次是 C512 `quantize_e4`（196 次）、ViT `quantize_e4`（152 次）、
ViT `aten_mul`（144 次）、C64 `quantize_e4`（131 次）。但 ViT 的时间包络仍小于 C64/C128/
C256，故不会仅按节点数提升为 P0。

本轮 structural P0 应是 C64/C128/C256 的完整 stage resident ABI：将 grouped FFN、
projection/residual、QKV 输入和下一 block 布局视作一条固定数据流，而不是继续添加单个
FP8 转换边界。C512 是四记录 split 结构，需要单独把 grouped 64→256→64、512×512
projection/residual 和 attention QKV 设计成 resident 链。完整离线结果位于
`results/20260907_nr1080_allattnnorm_arena_census_summary_v1.json`。

## 2. C++ workspace 和资源所有权

`NRFrameBindings` ABI v2 已预留：input/output、history/next-history、motion、depth、
exposure、controls、reset、flags 和 resource generation。当前单帧网络不会伪造尚未恢复的
temporal 语义，但以后加入这些资源不需要再次破坏 ABI。

`scripts/native_nr_arena.py` 为任意尺寸生成确定性、256 B 对齐、无重叠的固定 arena。
1080p 计划包含 26 个 region、总计 362,045,952 B：动态帧资源、C32/C64/C128/C256/
C512/ViT ping-pong、encoder skip 和有界 operator workspace。`nrPlanConfigureArena` 在
finalize 前验证对齐、范围和重叠。

| 所有权检查 | 当前捕获图 | 原生 recorder selftest |
|---|---:|---:|
| C++ workspace | PASS | PASS |
| C++ weights | PASS | PASS |
| C++ graph executable | PASS | PASS |
| C++ graph source | **FAIL** | PASS |
| graph 不引用外部分配 | **FAIL** | PASS |
| 固定 arena 非空 | PASS | PASS |
| `deployment_ready` | **FAIL** | PASS |

16 元素 recorder selftest 完成两次动态 pointer update，逐元素正确、无每帧 allocation，
所有权门禁通过。完整 71-block 当前只通过“captured graph runtime”门，没有通过部署门。
`scripts/audit_nr_plan_ownership.py` 会以非零退出码拒绝误发布。

## 3. 跨 kernel resident FP8

### 独立语义原型

`tools/native_fp8_chain` 实现了两个等价链：

1. baseline：producer 将 E4M3 舍入值物化为 FP16，consumer 再打包并执行 FP8 WMMA；
2. resident：producer 直接写 1-byte E4M3，consumer 直接把 resident byte 送入 gfx1201
   FP8 WMMA。

64 个矩阵的门中，两条链输出逐位一致，边界从 32,768 B 降到 16,384 B。ISA 中确认存在
两个 `v_wmma_f32_16x16x16_fp8_fp8` site。该微型门的 event 数值落在当前计时器可靠范围
以下，所以只证明 ABI、语义和 50% boundary-byte 缩减，不声称速度收益。

### 实际 C64 Swin 候选（拒绝）

实际 grouped FFN prototype 将 `group expand/activation/contract → E4M3 resident` 和
`FP8 C×C mix + FP16 residual` 变为两个项目 HIP kernel。producer 为 102 VGPR、4 KiB LDS、
0 scratch；consumer 为 60 VGPR、0 LDS/scratch，且确实生成 FP8 WMMA。

- 1 window：最大误差 0.000488、NRMSE 0.000303，显式近似轨门通过；单样本速度不可推广。
- 16 windows：确定性/guard/finite/资源释放均通过，但最大误差 0.001953 超过 0.001 门；
  0.15606 ms 还慢于 reference 0.10968 ms（0.703×）。

因此该候选按规则拒绝，不扩大到 144 windows、不接完整网络、不放宽门槛、不成为默认。
误差来源是 FP8 WMMA 与 rocBLAS 的 reduction order，而非重复 FP16→FP8 转换；残差仍在
FP16 边界。这个结果证明“真正 resident activation”在 ABI 上可行，也证明不能仅因少一个
kernel/少一半边界字节就晋级。

### C64/C128/C256 FFN→QKV resident 链（完成并拒绝）

后续候选已让 grouped-FFN dual epilogue 同时发布 FP16 residual view 与 E4M3 resident
view，QKV 权重初始化时一次性预打包为 column-major E4M3，下一次 hipBLASLt QKV 直接
消费 resident bytes。16-window 模块门分别取得 1.89×/1.73×/1.68×，且各自跨次逐位
稳定、finite、资源完全释放。

完整 1080p、12 次/进程 B-A-B-A 却只有 220.061→219.355 ms（−0.321%），kernel
4,653→4,445；peak allocated 增加 44,626,944 B，输出 NRMSE 0.11402、最大误差
0.32764。该候选因此拒绝并保持默认关闭。这是一个真实跨 kernel FP8 activation 原型，
但不是“完整 activation resident FP8 已落地”；下一实现单位必须扩展为整个 stage。

## 4. 1080p 性能阶梯

历史 A/B 和本轮结果使用相同输入及输出哈希，但早期阶梯每进程只有 3 次；只有最后一行
完成严格的 12 次/进程 A-B-B-A（11 个 warm 样本）。因此跨行用于阶段趋势，不冒充一次
完全同日的六路 12-repeat 实验。

| 阶段 | host median（ms） | kernel nodes | correctness | 说明 |
|---|---:|---:|---|---|
| A Python reference | 564.856 | 未捕获 | bitwise | 原 35,010-node 数学路径 |
| B baseline NRPlan replay | 561.665 | 35,010 | bitwise | 仅移除逐帧 Python submit，收益 0.565% |
| grouped FFN + transitions NRPlan | 458.956 | 30,976 | bitwise | 已验收 bounded 模块 |
| selected primitive/native modules NRPlan | 266.008 | 16,558 | bitwise | 旧 3-repeat A-B-B-A |
| C++ weights + fixed arena + selected graph | 262.792 | 16,558 | bitwise | 12-repeat A-B-B-A；graph 仍引用外部分配 |
| resident FP8 C64 | 未进入整帧 | 预计减少但未采纳 | **REJECT** | 误差超门且模块变慢 |
| 加入 C32 primitive rewrite | **241.059** | **9,790** | bitwise | 同日 12-repeat A-B-B-A；相对16,558-node控制快8.23% |
| wide layout + stage QKV norm | **约 221.93 GPU** | **4,653** | bitwise | 严格图进一步收敛 |
| 当前同 DLL 严格复测 | **220.061 GPU** | **4,653** | bitwise | resident-chain B-A-B-A 的 A 组 |
| C64/C128/C256 resident chain | **219.355 GPU** | **4,445** | approximate, **REJECT** | 仅快0.321%，NRMSE 0.11402 |
| 本轮最终可用配置 | **220.061 GPU** | **4,653** | bitwise | 严格显式配置；仍不是独立部署runtime |

本轮最终配置：P95 242.588 ms，4.148 jobs/s，距离 50 ms 为 4.821×，距离 20 ms 为
12.053×。从初始 35,010 到 9,790，节点减少 72.036%；从 baseline NRPlan 561.665 ms
到当前 241.059 ms，阶段累计下降 57.08%。这说明按完整高频 primitive family 改写的
收益明显高于孤立微算子，但 9,790 个 kernel 仍远非产品级 runtime。

当前 Python-selected 与 C++-arena-selected 的同日 12-repeat 比较为 276.305→262.792 ms
（−4.891%）。这量化了固定 graph submit 的收益，也说明剩余约 260 ms 主要不在 Python
入口，而在 captured tensor graph 本身。

随后保持 C++ arena/NRPlan 不变的 C32 A-B-B-A 为 262.680→241.059 ms（−8.231%），
kernel 16,558→9,790（−40.875%），说明移除大量小 primitive 的确有效，但 kernel 数下降
远大于延迟下降：剩余耗时已经更多转移到 C64/C128/C256/C512 和库矩阵计算，不能再按
“删 kernel 数”等比例外推。

## 5. 下一条真正能收敛的实现顺序

1. C32 primitive rewrite 已晋级；下一轮以新 census 中 C64/C128/C256/C512 的高频 add、
   quantize、layout 和库 GEMM 边界为 rewrite unit，不按旧 C32 排名重复优化。
2. 为每个 Pre/C32 及后续 stage 定义固定 arena 输入/输出 offset，producer epilogue 直接写 consumer
   packed/E4M3 layout；该 stage 完整迁移后从 PyTorch capture graph 删除对应节点。
3. 每迁完一个 stage，用 `deployment_ready` 前置条件检查 kernel 参数不再指向 capture pool；
   只有 71 block 全部完成，才能销毁 Python 模型/allocator 并宣称完全 C++ owned。
4. resident FP8 重新选择调用次数高、两端本来都是项目 kernel 的边界；C64 library-mix
   原型已经证明不适合。首选 C32 producer epilogue→下一 native consumer。
5. ViT 当前为 1,160 节点、12.755 ms stream envelope，不是第一优先级；保持 reference，
   直到 C64/C128/C256/C512 收敛后重新 census。

## 6. 稳定性与复现

RGP counter safety halt 未解除，本轮没有 RGP/counter capture、TDR/时钟修改或自动 GPU
重试。所有完成的 1080p A-B-B-A 进程均正常退出，显存 allocated/reserved 回到 0，检查期
间没有新增 Event 41/141/6008。C64 16-window 是数值/性能拒绝，不是设备故障。

resident-FP8 独立原型的第一次门在任何 kernel launch 之前错误拒绝了合法的 HIP default
stream 句柄 `0`；子解释器随后停在退出清理。等待并确认未进入 `hipLaunchKernelGGL`、没有
系统/GPU 事件后，只向该验证进程发送了一次 Ctrl+C，未结束其它进程。修复是取消对合法
default stream 的非空要求，随后 1/64 matrix 门均正常退出并逐位一致。第一次结果不计入
性能或正确性统计；1-matrix 的低于计时分辨率 event 样本也明确标为无效。

关键命令：

```powershell
# 固定 arena manifest
C:\DATA\Tools\ANACONDA\python.exe scripts\native_nr_arena.py `
  --width 1920 --height 1080 --output results\nr1080_arena_NEW.json

# census 汇总（不使用 RGP）
C:\DATA\Tools\ANACONDA\python.exe scripts\summarize_graph_census.py `
  --raw results\CENSUS\graph_kernel_nodes.json --dot results\CENSUS\graph.dot `
  --width 1920 --height 1080 --stage-times results\TIMES\stage_times.json `
  --output results\CENSUS\graph_census_ranked_NEW.json

# fail-closed 所有权审计；当前 captured graph 应返回非零
C:\DATA\Tools\ANACONDA\python.exe scripts\audit_nr_plan_ownership.py `
  --input results\RUN\child.json --output results\RUN\ownership_audit_NEW.json

# resident FP8 模块门；先 1 window，再经人工复核决定是否扩大
.venv-rocm\Scripts\python.exe scripts\validate_grouped_wide_ffn.py `
  --dll build\HEAD_FP8A\head_wmma.dll --channels 64 --windows 1 `
  --iterations 1 --resident-fp8-activations --output results\FP8A_NEW
```

本轮没有执行 1440p/4K，因为 1080p 仍为 241.1 ms 且完整 C++ 所有权未通过；这符合先把
1080p native runtime 做透的约束。
