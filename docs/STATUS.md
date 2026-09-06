# 当前状态

更新时间：2026-09-06。

截至当前的完整性能优化分析见
[`PERFORMANCE_OPTIMIZATION_ANALYSIS_20260906.md`](PERFORMANCE_OPTIMIZATION_ANALYSIS_20260906.md)。

## 已完成

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

## 尚未完成

- 4K60：当前同口径约 1.2 秒/帧，距离 16.67 ms 很远。
- RTX/DLSS5 画质对照、SDR/HDR、HUD、时间稳定性和真实游戏长时间稳定性验收。
- Pre/C32/C64/C128/C256/C512/ViT 的完整可靠 RGP 分类。Head 已取得有效RGP证据，
  但随后两次模块级 counter capture 触发 `LiveKernelEvent 141` 和系统重启，RGP采集
  仍被安全锁停；两个不完整目录作废，不能继续自动重试。非profiler测试也只允许逐个
  最小门禁执行。
- Pre 与跨 stage 数据流的进一步融合，以及后续 ViT 性能研究。
- 游戏原生常驻执行器的最终 pre-HUD 接入和安全回退复验。

## 当前实验配置

```text
profile=wmma_fp16
waves=2
modules=head_ffn,head_attention,head_softmax,head_output,c32_ffn,c32_attention,c512_ffn
```

该配置仍为 opt-in；默认行为未修改。Pre、C64、C128、C256 候选因完整模块收益
不稳定或为负而不启用。FP8 保留为独立研究轨。

## 关键文档

- [RDNA4 矩阵融合结果](RDNA4_MATRIX_FUSION_RESULTS.md)
- [RDNA4 原生引擎重构进度](RDNA4_NATIVE_ENGINE_REFACTOR.md)
- [GPU profiling 安全事件](GPU_PROFILING_INCIDENT_20260906.md)
- [NVIDIA 与 RDNA4 执行映射](NVIDIA_RDNA4_MAPPING.md)
- [NVIDIA 执行粒度审计](NVIDIA_EXECUTION_GRANULARITY.md)
- [ROCm 原生实现](ROCM_NATIVE_IMPLEMENTATION.md)
- [低分辨率 NR + FSR 路线](DUAL_NR_ROUTES_BENCHMARK.md)
- [游戏接入状态](NATIVE_RESHADE_PREVIEW.md)
- [目标与路线图](ROADMAP_TO_RX9070XT.md)

## 完整性检查

- 本地完整环境 CPU 回归：`843 passed in 66.22s`。
- 公开快照：本轮新增/修改相关测试 `28 passed`；全套 `834 passed, 9 failed`，
  9 项均因有意不发布的 PTX/采集清单夹具缺失而在读取时抛出 `FileNotFoundError`，
  没有代码断言失败。私有夹具不会为追求公开测试全绿而上传。
- 原模型 arena SHA256：
  `A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`。
- 最终选择配置的 4K 输出 SHA256：
  `46F357EC1DA510A7CB7BC6D141E8A83DBA4DAD24179D72DC18BD150CD573056F`。
- 最终审计时游戏未运行，也没有遗留本项目 GPU 验证进程。
