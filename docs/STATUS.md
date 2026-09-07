# 当前状态

更新时间：2026-09-07。

截至当前的完整性能优化分析见
[`PERFORMANCE_OPTIMIZATION_ANALYSIS_20260906.md`](PERFORMANCE_OPTIMIZATION_ANALYSIS_20260906.md)。
最新 Swin 主干重构实测见
[`RDNA4_SWIN_BACKBONE_REDESIGN_20260907.md`](RDNA4_SWIN_BACKBONE_REDESIGN_20260907.md)。
完整 71-block `NRPlan` 迁移与 1080p A/B 见
[`NATIVE_NR_PLAN_MIGRATION_20260907.md`](NATIVE_NR_PLAN_MIGRATION_20260907.md)。
最新完整 Graph census、C++ arena 所有权门禁和 resident FP8 实验见
[`NATIVE_ENGINE_CENSUS_FP8_20260907.md`](NATIVE_ENGINE_CENSUS_FP8_20260907.md)。
宽通道 resident layout、stage-specific attention 和整帧 FP8 结果见
[`WIDE_LAYOUT_STAGE_FP8_20260907.md`](WIDE_LAYOUT_STAGE_FP8_20260907.md)。
ABI v3、分段模型包、gfx1201 派生缓存、时序门和 33 ms 统一验收门见
[`NATIVE_TEMPORAL_ENGINE_V3_20260907.md`](NATIVE_TEMPORAL_ENGINE_V3_20260907.md)。
最新 resident-FP8 尺度切换与 C256 最小 GPU 门见
[`NATIVE_SCALE_TRANSITION_FP8_20260907.md`](NATIVE_SCALE_TRANSITION_FP8_20260907.md)。

## 已完成

- 原生运行时接口升级到 ABI v3 并编译通过：固定 shape、显式 current/history/motion/depth/
  exposure/jitter/reset binding、strict/approx precision、GPU event 性能统计，以及导入 D3D12
  fence 的 `wait → graph → signal` 提交。未核验 temporal contract 会 fail closed。
- 模型包 v2 新增逐段 SHA-256/连续覆盖校验；完整 71-block 本地权重已生成 gfx1201 派生
  缓存 v3（1,014 records、666 unique、148,300,672 B），确认 E4M3 的矩阵才转换，Pre projection
  等非 E4M3 内容继续 FP16。该缓存尚未被完整 native graph 消费，不等于性能路径完成。
- ABI v3 arena 的1080p strict workspace为435,196,416 B；当前诊断版approx arena为
  550,736,384 B。approx stage resident为1 byte，并复用grouped/post/Q/K/V/value scratch；
  新增108,539,904 B固定FP16 QKV投影区，用于将矩阵与norm错误独立定界。
- C32–C256 共44个 Swin block core、16个独立C512 split core 与8个ViT core已形成 C++ 自有
  resident-FP8 recorder topology：诊断期C32每块7 dispatch，C64–C256每块8，C512每块7，ViT每块5；
  加上中央C512↔ViT transition和8个Encoder/Decoder尺度切换后，合计覆盖69/71个record、
  理论506 kernel＋5 D2D memcpy；诊断拆分后已只剩6个kernel节点预算，不能直接加入Pre/Head。
  C32首次192 VGPR/scratch候选已拒绝；8-wave重写为50 VGPR、4 KiB LDS、
  0 scratch。C512 grouped为72 VGPR/8 KiB LDS；所有矩阵路径均生成gfx1201 FP8 WMMA且
  0 scratch。当前状态为`PARTIAL_69_OF_71_NATIVE_RECORDS_WITH_SCALE_TRANSITIONS_NOT_RUNTIME_ACCEPTED`，
  完整拓扑与deployment门仍为false。
- ViT为640-token流式全局attention，不物化N×N矩阵；五类kernel均静态生成FP8 WMMA，
  VGPR为62/62/42/88/85、最大LDS 3 KiB、0 scratch。该结论仅来自编译/ISA，不代表数值或速度通过。
- 中央边界已直接实现resident C512 pooling→512×1024→ViT，以及ViT→1024×512→2×expand＋skip；
  其余4个Encoder与4个Decoder尺度边界也已进入recorder。Encoder直接消费上一core的FP16
  window输出并保存resident skip；Decoder直接projection/2×expand/skip-add，均不生成logical
  HWC或repeat_interleave。八个边界目前只有编译/ISA/ABI证据，尚未做GPU数值门。
- 新增原始时序序列 schema v2 与最终33 ms统一门禁；结构有效不会被误报为时序语义通过。
  C256 record15的8×8单窗口分界显示QKV projection FP16为NRMSE 0.015319、max error
  0.011276，首个大误差位于Q/K/V发布边界。wave32、去提前返回和单连续基址实验均未使
  原标量发布可靠；同步后的不同字节poison在提交前已逐段验证，Graph/DOT也确认实际执行
  新kernel，CPU字节直方图排除简单重排。当前因此拒绝scalar FP8＋byte store组合，并已
  FP8x2显式打包、内联software-E4M3、device-symbol LUT、拆分FP16 norm/标量pack，及显式
  `NRPlan`自有64 KiB LUT均已逐一做单窗口门；Q/K/V与最终输出的SHA-256在所有版本中完全
  不变。最新pack节点确认为11 VGPR、0 LDS、b8 store，Graph/DOT依赖顺序也正确，但仍发布
  decoded-E4M3的FP16字节。因此转换算法、arena重叠和隐式symbol重定位均已排除，下一步改为
  审计捕获kernel参数与实际写目标，不再继续换编码公式。所有最小门均正常释放到0 B且无新增
  系统事件；没有游戏、RGP、性能循环或4K，下一次GPU门仍为关闭状态。

- 原始权重驱动的 71-block ROCm 整帧候选已连通，正式神经路径不依赖 NVIDIA DLL、
  PTX、ZLUDA、RTX 中间值或 CPU 神经算子回退。
- 动态尺寸离线执行覆盖 640×360、1920×1080、2560×1440、3840×2160、
  3440×1440、641×361、2342×1382。
- Head、C32、C64/C128、C256 FFN、C512 分组 MLP 已有默认关闭的 gfx1201
  WMMA 候选。经过正确性和性能门，当前仅建议显式启用 Head、C512；C32 新的
  bounded-attention 候选尚未过门，不能启用。
- 当前选择配置在上述尺寸保持参考输出位型一致；4K C512-only A-B-B-A 热整帧
  host submit/wait 均值 1237.02→1204.95 ms（−2.59%）。
- allocator live 峰值约 4.10 GB、reserved 4.77 GB，设备用量离散采样 5.20 GB，
  未越过本轮 5 GB/6 GB 门。
- 独立 D3D12/ROCm GPU 共享缓冲路径和一次实际游戏 SDR 写回已有证据，但后者画质失败，
  当前优化模块没有部署为游戏默认路径。
- Output Head 已新增 bounded-LDS full-grid attention 候选：640×384 两组12次门逐位
  一致并提速6.0%/9.8%，完整 Head dispatch 9→4。握手式 RGP 对照显示 sampled active
  3.5025→2.6866 ms、fetch/write 流量降至67.8%/51.5%。
- Pre 输入构建＋原始FP16 16×32投影＋C32 packing 单 kernel 在128×128和640×384均
  逐位一致，边界耗时分别约6.15×和4.08×加速；仍为opt-in。
- 安全锁停后完成了C32/C64/C128 bounded-attention实现：attention目标由6个全局
  中间阶段降为“每head局部LDS＋单独跨head投影”2阶段，均成功编译且0 scratch。
  C32首次单窗口GPU门出现320个half差异；定位并修复三处舍入边界后，修正版通过一次
  单窗口及一次16窗口门，均逐位一致且launch 7→2。但16窗口下bounded路径
  0.16963 ms，慢于基准0.12120 ms；其显存下降但20 KiB LDS/低occupancy抵消了调度收益。
  C64/C128也尚未执行GPU门，因此三者仍不能默认启用。
- C32低LDS三段候选静态占用改善到62.5%/75%，但单窗口仍因QKV归一化串行化而
  0.32934 ms 对0.15870 ms，已拒绝。当前保留8 KiB QK/softmax/PV核心，另建复用既有
  并行QKV/norm的保守四段候选；CPU/ABI已通过，尚未运行GPU门。
- C32保守核心已经完成GPU门：16窗口从0.18870增至0.26456 ms，尽管launch 7→5且
  逐位一致，仍按规则拒绝。
- C64/C128/C256新 grouped FFN 以`(window,head,16-token tile)`并行，固定4 KiB LDS；
  144窗口12次中位数分别取得2.47×、1.57×、1.33×加速，全部逐位一致。1080p
  B-A-B-A整帧中位数1031.478→902.266 ms（−12.527%），这是当前可保留的显式候选。
- C64/C128 attention的20 KiB per-head和2 KiB query-tile两案均未在代表规模稳定胜出，
  已拒绝且未复制到C256。
- 已建立E4M3 resident-weight路径；C128仅提速2.49%，未晋级。完整71-block现已捕获为
  C++ `NRPlan` HIP Graph并通过128/640/1080逐位一致门。最终选择配置1080p Python/NRPlan
  B-A-B-A为311.863→266.008 ms（−14.703%），kernel节点16,558，设备用量约3.16 GB。
  当前仍由Python捕获器保持权重和graph allocation pool，不等于C++独立资源所有权。
- 已完成1080p完整Graph census：部署图精确为16,558 kernel + 120 memcpy；其中
  PyTorch/library节点11,979个。C32有5,849个节点、Pre有2,684个，pointwise/quantize/
  cast/layout合计13,201个，是下一轮首要替换对象。非RGP stage interval显示C256/C128/
  C32/C64依次为37.029/35.752/26.076/24.356 ms；这不是kernel-busy counter。
- `NRPlan`新增固定arena ABI和资源所有权门禁；旧 ABI v2 仍仅供捕获图回归，新部署门强制
  ABI v3。完整捕获图仍指向PyTorch allocation pool，所以部署门禁正确为FAIL。
- 独立跨kernel resident E4M3原型逐位一致并把边界字节减半；实际C64 grouped FFN候选在
  16窗口下最大误差0.001953且0.15606 ms慢于0.10968 ms reference，已拒绝且未接整帧。
- 新的1080p 12次/进程A-B-B-A全部正常退出且逐帧hash一致：同一selected路径
  Python 276.305 ms，C++ arena/NRPlan 262.792 ms（−4.891%），P95 263.960 ms，
  3.805 jobs/s，距离50/20 ms目标仍有5.256×/13.140×。
- 重新纳入既有逐位一致的C32 primitive rewrite后，完整图16,558→9,790 kernel
  （−40.875%）；同口径两组12次B-A-B-A为262.680→241.059 ms（−8.231%），
  P95 242.588 ms、4.148 jobs/s，输出hash不变，显存释放回0。它已晋级为当前最优
  显式配置，但graph仍引用PyTorch capture pool，`deployment_ready`仍为false。
- C64/C128/C256/C512 packed layout 已改为 whole-grid HIP 路径；stage-specific QKV
  norm 采用15 VGPR、0 scratch的有界kernel，并保留库QKV/QK/PV/projection。最终1080p
  严格B-A-B-A为230.836→221.930 ms GPU median、6,557→4,653 kernel（−29.038%），
  host median 233.475→224.348 ms；输出逐位一致。共享Q/K/V arena将候选peak allocated
  从1.129 GB压回0.901 GB，仅比对照增加约38 MB。
- C64/C128/C256的resident E4M3＋hipBLASLt模块微基准有收益，但整帧仅快0.024%且P95
  变差；C128+C256项目FP8-WMMA整帧近似轨为221.644 ms，反而慢于严格候选单帧
  220.335 ms，且NRMSE 0.006473，故两条FP8组合均未晋级。近似门现在会显式报告误差并
  强制跨帧确定性，严格hash门仍是默认。
- 4,653-node最终图的无RGP细分类census在graph capture后首次submit返回HIP error 1；
  进程未超时、资源释放为0 B、没有新增系统/GPU事件，已按规则停止且不重试。因此当前只
  报告普通NRPlan已验证的节点总数，不伪报新的逐函数分类。
- C64/C128/C256 FFN→attention 跨kernel resident E4M3 ABI已完成GPU门：16-window模块
  分别为1.89×/1.73×/1.68×，结果确定且资源完全释放。但1080p 12次/进程B-A-B-A仅将
  GPU中位数220.061→219.355 ms（−0.321%）、kernel 4,653→4,445，peak allocated增加
  约44.6 MB；输出NRMSE 0.11402、最大误差0.32764。该短链因此拒绝且保持默认关闭，后续
  必须改为完整stage resident ABI，而不是继续增加单个FP8边界。
- 未重跑GPU即从上次失败提交前已完整写出的Graph/DOT恢复了当前4,653-node离线census：
  ViT 1,160、C32 817、C512 629、C64 565、Pre 471、C256 405、C128 305、Head 187、
  transition 114；类别为pointwise 1,115、GEMM 819、quantize 681、layout 377、cast 297。
  这证明下一优化单位应是C64/C128/C256完整stage resident ABI以及单独设计的C512 split
  resident链；ViT虽节点最多，但最近正常stage interval仅约12.8 ms，暂不盲目优先。

## 尚未完成

- 1080p 33 ms：最后可采信完整严格路径仍为220.061 ms、4,653 kernel+59 memcpy；本轮
  原生接口/缓存工作没有新的整帧GPU时间，不能按理论或编译成功推算收益。
- 先完成C256捕获kernel参数/写目标的静态与CPU侧诊断，修复Q/K/V发布边界后，再完成
  C256→C128→C64→C32→C512 resident block 的串行 GPU correctness/performance gate；
  scale transitions已绑定但尚未GPU验收。随后实现
  Pre与Head并验证已编译的ViT，使0 PyTorch hot node、
  ≤512 graph node真正跑通。当前安全记录不授权自动启动下一GPU门。
- RTX 5070 24×32原始时序序列与有界差分结论；在它们完成前 temporal mode保持禁用。

- 4K60：当前同口径约 1.2 秒/帧，距离 16.67 ms 很远。
- RTX/DLSS5 画质对照、SDR/HDR、HUD、时间稳定性和真实游戏长时间稳定性验收。
- Pre/C32/C64/C128/C256/C512/ViT 的完整可靠 RGP 分类。Head 已取得有效RGP证据，
  但随后两次模块级 counter capture 触发 `LiveKernelEvent 141` 和系统重启，RGP采集
  仍被安全锁停；两个不完整目录作废，不能继续自动重试。非profiler测试也只允许逐个
  最小门禁执行。
- Pre 与跨 stage 数据流的进一步融合，以及后续 ViT 性能研究。
- 游戏原生常驻执行器的最终 pre-HUD 接入和安全回退复验。
- 将完整图内仍由PyTorch持有的权重、workspace和ATen/library节点逐模块重绑定到C++
  自有arena/module handle；随后建立真正跨kernel的resident FP8 activation。当前已经
  完成整帧native submit与固定graph，不可宣称独立游戏部署runtime。
- 逐stage用C++ arena offset重写C32/Pre等高频primitive并移除capture-pool指针；只有
  `owns_graph_source=true`、`graph_references_external_allocations=false`且71 block均迁移后，
  才能把“完全C++ owned NRPlan”从未完成项移除。

## 当前实验配置

```text
profile=wmma_fp16
waves=2
modules=head_ffn,head_attention_bounded,head_output,c32_ffn,c32_attention,c64_ffn_grouped,c64_attention_norm,c128_ffn_grouped,c128_attention_norm,c256_ffn_grouped,c256_attention_norm,c512_ffn,c512_attention_norm
```

该配置仍为 opt-in；默认行为未修改。`*_attention_norm`只融合归一化、scale和E4M3 store，
不恢复已拒绝的QK/softmax/PV大融合。FP8保留为独立近似研究轨。

## 关键文档

- [RDNA4 矩阵融合结果](RDNA4_MATRIX_FUSION_RESULTS.md)
- [RDNA4 Swin 主干重构结果](RDNA4_SWIN_BACKBONE_REDESIGN_20260907.md)
- [RDNA4 原生引擎重构进度](RDNA4_NATIVE_ENGINE_REFACTOR.md)
- [GPU profiling 安全事件](GPU_PROFILING_INCIDENT_20260906.md)
- [NVIDIA 与 RDNA4 执行映射](NVIDIA_RDNA4_MAPPING.md)
- [NVIDIA 执行粒度审计](NVIDIA_EXECUTION_GRANULARITY.md)
- [ROCm 原生实现](ROCM_NATIVE_IMPLEMENTATION.md)
- [低分辨率 NR + FSR 路线](DUAL_NR_ROUTES_BENCHMARK.md)
- [游戏接入状态](NATIVE_RESHADE_PREVIEW.md)
- [目标与路线图](ROADMAP_TO_RX9070XT.md)

## 完整性检查

- 本轮 stage/layout/近似轨目标回归通过；排除5个需要Pillow/pefile的图像/PE测试后，
  完整CPU回归为`916 passed in 42.02s`。ROCm venv 的全量收集仍会因未安装这些可选依赖
  停止，GPU/目标门不受影响；没有为此修改隔离环境。
- 公开快照：本轮新增/修改相关测试 `28 passed`；全套 `834 passed, 9 failed`，
  9 项均因有意不发布的 PTX/采集清单夹具缺失而在读取时抛出 `FileNotFoundError`，
  没有代码断言失败。私有夹具不会为追求公开测试全绿而上传。
- 原模型 arena SHA256：
  `A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`。
- 最终选择配置的 4K 输出 SHA256：
  `46F357EC1DA510A7CB7BC6D141E8A83DBA4DAD24179D72DC18BD150CD573056F`。
- 最终审计时游戏未运行，也没有遗留本项目 GPU 验证进程。
