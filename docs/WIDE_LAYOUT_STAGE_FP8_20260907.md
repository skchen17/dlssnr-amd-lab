# 宽通道 resident layout、stage kernel 与 FP8 整帧验证

更新时间：2026-09-07。

## 结论

本轮已把 C64/C128/C256/C512 的 packed gather/scatter 从 Python window batching
替换为 whole-grid HIP layout kernel，并为四个通道家族增加了低资源的 stage-specific
QKV norm：QKV/QK/PV/projection 仍调用稳定的 ROCm library GEMM，只把 Q/K 的 32 元素
归约、q-scale 和 Q/K/V E4M3 store 合并为一个 kernel。该 kernel 在 gfx1201 ISA 中为
15 VGPR、0 scratch，不使用大 LDS。

1080p 严格 B-A-B-A（每进程 12 帧，舍弃首帧）结果：

| 路径 | GPU median | host median | kernel | peak allocated | 输出 |
|---|---:|---:|---:|---:|---|
| A：宽通道 resident layout | 230.836 ms | 233.475 ms | 6,557 | 862,901,248 B | exact |
| B：A + C64/128/256/512 stage norm + shared arena | **221.930 ms** | **224.348 ms** | **4,653** | 901,092,352 B | exact |

B 相对 A 的 GPU/host 降幅为 3.858%/3.909%，kernel 减少 29.038%。两组 B 的 GPU
median 为 221.969/221.891 ms，P95 为 222.356/222.506 ms；所有输出 SHA-256 均为
`892FB11D82C642996AD321052B2804A72BAC919620DA9F696BC086D74DA401B7`。

相对上一轮 C32 后的 9,790 kernel / 241.059 ms，本轮累计达到 4,653 kernel /
221.930 ms，即 kernel 再减少 52.47%，GPU span 再降低约 7.94%。相对最初 35,010-node
graph，节点已减少约 86.71%；但这不是 4K60，也不是独立部署完成。

## 1. 宽通道布局

`native_fusion_packed_layout` 现在显式支持 C32/C64/C128/C256/C512；一次 grid launch
覆盖完整 window 集合，并保持原始反射边界、offset 和 packed mapping。C64-C512 共八个
offset/方向用例均通过 gather、scatter、repeat bitwise gate。完整 1080p 图从 9,790
kernel 降到 6,557，输出 hash 不变。

这仍是“native gather/scatter”，不是最终零转换 ABI。下一步 C++ recorder 应让 producer
直接写 consumer resident offset，届时才能删除成对 layout launch，而非只加速它们。

## 2. stage-specific attention norm

16-window 单模块结果如下；数值均逐位一致，释放后 allocated/reserved 为 0。

| 家族 | reference | candidate | speedup |
|---|---:|---:|---:|
| C64 | 1.16840 ms | 0.44848 ms | 2.61× |
| C128 | 1.07676 ms | 0.67094 ms | 1.60× |
| C256 | 1.04408 ms | 0.66100 ms | 1.58× |
| C512 | 1.01290 ms | 0.68668 ms | 1.48× |

全图 kernel 6,557→4,653。减少 kernel 远大于 3.86% 的时间收益，再次证明剩余图已更多
受 GEMM 和内存路径支配，不能按 dispatch 数等比例推测性能。

最初实现为每种 `(channel, window-count)` 保留三份 Q/K/V workspace，peak allocated 达到
1,128,952,832 B。现在所有家族在同一 stream 上复用一个有界 arena；必要扩容时旧地址会
保留，避免已实例化 graph 悬空。最终 peak allocated 降到 901,092,352 B，只比 A 增加
38,191,104 B，reserved/device-used 与 A 相同。

## 3. resident FP8：实现有效，当前整帧候选拒绝

本轮保留两条真实 E4M3 路径：

- producer 直接写 resident E4M3 activation，项目 FP8 WMMA consumer 直接读取；
- producer 直接写 resident E4M3，随后使用 ROCm `_scaled_mm.out`/hipBLASLt 和初始化时
  column-major 预打包的 FP8 weight。

库路径的 C64/C128/C256 16-window 模块门分别约 2.85×/1.63×/1.62×，且逐位一致；但
完整图 A-B-B-A 只有 0.024% median 改善且 P95 变差，所以不晋级默认。项目 consumer
路径的 C128/C256 微基准约 3.22×/1.96×，属于近似轨；C64 已因更慢而拒绝。

新整帧近似门允许显式 `--allow-approximate-output`，但仍强制 finite、同输入跨帧 hash
稳定，并报告 max error/RMSE/NRMSE。C128+C256 resident-FP8 组合跑通后为 221.644 ms、
4,625 kernel，反而略慢于严格 stage-norm 的 220.335 ms 单帧；相对参考 max error
0.06134、RMSE 0.004347、NRMSE 0.006473、不同元素比例 72.76%。因此放宽精度仍没有
带来整帧性能收益，该组合明确拒绝。严格轨默认行为未被修改。

## 4. 当前限制与下一步

- `NRPlan` graph executable 仍引用 PyTorch capture pool，`deployment_ready=false`；当前
  “每帧零 Python operation”仅指 replay hot path，不等于所有 graph 参数归 C++ 所有。
- 剩余 4,653 kernel 仍包含大量 library/pointwise/copy。下一步应从更新 census 中选择
  producer/consumer 成对重写，并让 C++ arena offset 成为唯一内部 ABI。
- resident FP8 下一轮只应选择能让上下游都直接消费 byte layout 的长链；只把单个边界
  改成 FP8 再回 FP16 不会产生数量级收益。蒸馏或微调只能处理允许的数值误差，不能修复
  调度和带宽开销。
- RGP counter safety halt 保持启用；本轮只使用 HIP event 和正常退出的串行门禁，没有
  新增 Event 41/141/6008。

最终尝试用 census 专用 `NRPlan` 获取 4,653 节点的函数细分类时，graph capture 完成但
首次 `nrPlanSubmit` 返回 HIP error 1。该失败没有超时，资源释放为 0 B、进程树消失且没有
新增系统/GPU事件；按安全规则未自动重试。因此 4,653 是普通 NRPlan 已验证 graph stats，
本轮不提供该版本的逐函数/逐类别 census，也不以旧 census 代替新数据。

本地复现摘要：`results/20260907_normarena_baba12_summary.json`。原始权重、输出图、构建
DLL 和完整 result 目录仍按项目政策保持本地私有，不进入公开仓库。

## 5. FFN → attention 跨 kernel resident FP8（已完成 GPU 门，整帧拒绝）

在上述模块内原型之后，新增了真正跨算子边界的
`c64/c128/c256_ffn_attention_fp8a`：grouped-FFN producer 的最终 mix epilogue 同时写出
FP16 residual view 和 E4M3 resident view；QKV 权重在初始化时预打包为 column-major
E4M3，后续 hipBLASLt QKV 直接读取 resident bytes。这样 attention residual 保留网络需要
的FP16值，同时删除QKV入口的独立FP16→FP8转换。dual epilogue的C128/C256 ISA为35
VGPR、0 LDS/scratch、静态occupancy 16；C64为63 VGPR、同样无LDS/scratch，并确认含
RDNA4 FP8 WMMA。

首次 C64/1-window 门在 producer 提交后，被错误的 QKV stride guard 拒绝：`[C,3C]` 的合法
column-major stride 是 `(1,C)`，代码误写成 `(1,3C)`。等待残留进程自然消失、用户明确允许
继续并复核系统事件后，修正版才按 C64→C128→C256 顺序逐个运行，没有自动重试或并发 GPU
实验。

16-window、12 次模块门均正常退出、结果跨次逐位稳定：

| 家族 | reference | resident chain | speedup | NRMSE |
|---|---:|---:|---:|---:|
| C64 | 1.25443 ms | 0.66544 ms | 1.89× | 0.59168 |
| C128 | 1.28832 ms | 0.74612 ms | 1.73× | 0.46264 |
| C256 | 1.36044 ms | 0.81007 ms | 1.68× | 0.48797 |

模块收益没有转化为可用的整帧收益。1080p、每进程 12 帧、舍弃首帧的 B-A-B-A 中，严格
stage-norm 路径为 220.061 ms，三家族 resident-chain 为 219.355 ms，只快 **0.321%**；
host 中位数只改善 0.285%。kernel 4,653→4,445（−4.47%），peak allocated 增加
44,626,944 B，设备用量采样增加 104,923,136 B。候选固定输出 NRMSE 为 0.11402、最大误差
0.32764、74.91% 元素不同。两组候选输出 hash 一致、资源释放为 0 B，也没有新增系统/GPU
事件，但收益不足且误差明显，因此明确拒绝，不晋级默认配置。

该结果证明“producer FP8 输出 → 固定 FP8 buffer → 下一 QKV FP8 GEMM”已经真实实现并运行，
但只跨一个边界仍不足以显著缩短 71-block 图。下一轮不能继续堆相同短链，应以完整 stage
resident ABI 为边界，让 projection、residual、下一 block 输入都保持可直接消费的布局；
C512 因四记录 split 结构不同，需单独设计，不能机械套用 C64/C128/C256 的 dual epilogue。

复现汇总：`results/20260907_fp8chain_abba12_summary.json`。
