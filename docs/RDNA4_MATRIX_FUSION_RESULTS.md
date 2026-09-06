# RDNA4 原生矩阵融合与全阶段验证结果

2026-09-06。本文是本轮矩阵融合计划的最终实现记录。它完成了 Head、Pre、C32、
C64/C128、C256/C512 的有界候选和收益判断；ViT 按计划只保留既有基准。
所有路径默认关闭，未部署游戏，也不代表 RTX 画质或 4K60。

## 最终结论

最终建议的显式实验配置为：

```text
profile=wmma_fp16, waves=2
modules=head_ffn,head_attention,head_softmax,head_output,c32_ffn,c32_attention,c512_ffn
```

Pre、C64、C128、C256 均保留 reference。原因不是正确性失败，而是两组稳定收益门或
完整模块性能未通过。`wmma_fp8` 作为独立研究轨保留，不自动部署。

| 家族 | 实现边界 | graph kernel 节点 | 代表规模结果 | 位型误差 | 决策 |
|---|---|---:|---:|---:|---|
| Head | FFN、QKV、norm、QK、softmax、PV、project、tail | 112→8 | 分段 ABBA 均通过；最后 softmax 增量 Head −7.46%/−7.82% | 0 | 保留 |
| Pre | 16→32 input project | 4→1 | 两组 ABBA：一组 +5.07%，一组 −5.05% | exact 标量轨为0；WMMA近似轨 max 2.44e-4 | 拒绝 |
| C32 | FFN + 完整 window attention | 55→7 | block2 ABBA −32.10%/−27.73%，整帧 −9.32%/−7.62% | 0 | 保留 |
| C64 | FFN + 完整 window attention | 65→7 | 753窗完整块 2.38→2.52 ms | 0 | 拒绝 |
| C128 | FFN + 完整 window attention | 65→7 | 753窗完整块 4.44→12.51 ms | 0 | 拒绝 |
| C256 | FFN 单核候选 | 65→48 | 768窗 FFN 13.33→70.37 ms | 0 | 拒绝 |
| C512 | 仅8组 64→256→64 MLP，512投影仍用库GEMM | 88→61 | 完整块 7.99→4.36 ms | 0 | 保留 |

表中 host submit/wait 或 bulk 时间均不是 profiler 的 kernel-busy 总和。C64 的独立 FFN
曾测得约 −62%，但连接 attention 后完整块回退，因而没有用局部微基准掩盖模块回退。
C128/C256 的大 LDS/VGPR 方案同样没有因节点更少而晋级。

## 4K 整帧和显存

C512-only 的 4K A-B-B-A 使用 Head+C32 作为 A，B 只增加 C512 分组 MLP：

| 顺序 | 热11帧中位数 ms | 输出哈希 |
|---|---:|---|
| A1 | 1238.29 | `46F357EC1DA5…` |
| B1 | 1209.78 | `46F357EC1DA5…` |
| B2 | 1200.13 | `46F357EC1DA5…` |
| A2 | 1235.74 | `46F357EC1DA5…` |

两侧均值 1237.02→1204.95 ms，下降 2.59%。B 的 allocator live 峰值
4,097,502,720 B、reserved 4,773,117,952 B；设备用量离散采样最大
5,202,509,824 B，分别低于 5 GB/6 GB 门。所有进程正常退出并释放为0。
这仍远离 16.67 ms 的 4K60 预算，不能称为实时版本。

最终选择配置用最新 DLL 再跑一次 4K 完整网络，输出严格等于既有参考。另在
640×360、1080p、1440p、3440×1440、641×361、2342×1382 上完成完整网络检查；
这些尺寸也都逐位一致。640 使用既有游戏裁剪，其余主要是合成图，不替代 HDR、HUD、
真实时间序列或 RTX 教师验收。

## 数值轨和资源证据

- FP16/FP8、1/2/4 wave 的 Head、C32、C64、C128、C512 候选均做过原权重、换输入、
  重复提交；对应验证器还检查权重缓存失效。C256 因每 wave 32 KiB LDS，只测试合法的
  1/2 wave；4 wave 需 128 KiB，在提交前明确拒绝。
- C64 初版失败来自 LDS 原地复用覆盖同一 head 尚未消费的 hidden 通道，修复后所有门
  逐位一致；失败结果保留，没有自动重试设备故障。
- C32 workspace 复用后做了三个独立12次稳定进程，解决的是 GPU 临时张量生命周期，
  不是放宽数值门。
- 最新 ISA 审计逐个候选函数确认实际包含 gfx1201 FP16/FP8 WMMA。所有候选 scratch=0；
  代表最大资源：C64 165 VGPR/32 KiB LDS，C128 132 VGPR/64 KiB LDS，
  C256 132 VGPR/64 KiB LDS，C512 69 VGPR/32 KiB LDS。
- 纯 GPU activity、DRAM 流量和 kernel busy 总和仍不可用；本文没有用 event span 或
  host 时间冒充这些指标。

关键证据目录：

- 最终构建与 ISA：`results/20260906_c64_c128_attention_wmma_build_v1`
- C64/C128 六配置：`results/20260906_c{64,128}_full_matrix_*_gate1_v1`
- C256：`results/20260906_c256_ffn_accuracy_{gate1,perf768}_v1`
- C512：`results/20260906_c512_group_ffn_accuracy_perf144_v1`、
  `results/20260906_c512_group_ffn_graph144_12_v1`
- C512 4K ABBA：`results/20260906_c512_only_abba_{a1,b1,b2,a2}`
- 最终4K：`results/20260906_matrix_selected_full2160_final1_v1`

## 实现边界

HIP ABI 使用调用方当前 stream、GPU输入和显式输出；不做CPU读回、设备同步或静默回退。
权重预打包缓存按参数对象、地址、版本、设备和dtype失效。入口拒绝训练/梯度、别名、错误
架构、错误dtype、非连续输入和超资源配置。C32/C64/C128 attention 的中间工作区常驻于
策略对象，仍是 Python 部署桥的实现，尚未升级成游戏原生调用方提供的统一 workspace ABI。

没有修改原权重、网络拓扑、窗口语义、全局注意力、颜色或合成约定；没有训练、游戏启动、
TDR 修改或 CPU 神经算子回退。当前结果只关闭本轮“候选实现和有界收益验证”计划，不关闭
原生整帧画质、SDR/HDR、真实游戏上屏、稳定性和 4K60 里程碑。

本地完整环境 CPU 回归 `843 passed in 66.22s`；公开快照本轮相关测试 `28 passed`，
全套 `834 passed, 9 failed`，9 项均为未发布 PTX/采集夹具的 `FileNotFoundError`，没有
代码断言失败。Python 编译检查和 `git diff --check` 通过（仅既有 LF/CRLF 提示）。
原 arena SHA256 仍为
`A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`；最终选择配置的
4K 输出仍为 `46F357EC1DA510A7CB7BC6D141E8A83DBA4DAD24179D72DC18BD150CD573056F`。
审计时游戏未运行，也没有遗留本项目 GPU 验证进程。

## 复现

仓库根目录 PowerShell。必须已有本地私有模型和参考；每次使用新的输出目录。GPU串行，
任一非零退出码立即停止并检查，不自动重试或杀死不确定的 GPU 任务。

```powershell
$py = "$PWD\.venv-rocm\Scripts\python.exe"
$cpu = "C:\DATA\Tools\ANACONDA\python.exe"
$build = "$PWD\results\matrix_repro_build"
.\tools\native_fusion_probe\build_head_wmma.ps1 -OutputDirectory $build
& $cpu scripts\audit_matrix_isa.py --assembly "$build\head_wmma.s" --output "$build\isa_audit.json"

foreach ($family in @('c64','c128')) {
  foreach ($profile in @('wmma_fp16','wmma_fp8')) {
    foreach ($waves in @(1,2,4)) {
      & $py scripts\audit_head_graph.py `
        --dll results\20260906_head_qkv_build_v1\native_fusion_quantize.dll `
        --quant-dll results\20260906_native_fusion_cubic_build_v1\native_fusion_quantize.dll `
        --family $family --batch 1 --iterations 1 `
        --matrix-dll "$build\head_wmma.dll" --matrix-profile $profile --matrix-waves $waves `
        --matrix-modules "${family}_ffn,${family}_attention" `
        --output "results\matrix_repro_${family}_${profile}_${waves}"
      if ($LASTEXITCODE -ne 0) { throw 'STOP GPU: review evidence' }
    }
  }
}

& $py scripts\validate_wide_ffn_wmma.py --dll "$build\head_wmma.dll" `
  --channels 256 --batch 768 --iterations 12 --output results\matrix_repro_c256
if ($LASTEXITCODE -ne 0) { throw 'STOP GPU: review evidence' }
& $py scripts\validate_c512_group_ffn_wmma.py --dll "$build\head_wmma.dll" `
  --batch 144 --iterations 12 --output results\matrix_repro_c512
if ($LASTEXITCODE -ne 0) { throw 'STOP GPU: review evidence' }
```

最终选择配置的4K correctness gate：

```powershell
& $py scripts\benchmark_nr_resolution.py `
  --output results\matrix_repro_4k --size 2160 --iterations 1 `
  --fusion-dll results\20260906_native_fusion_cubic_build_v1\native_fusion_quantize.dll `
  --head-input-dll results\20260906_head_qkv_build_v1\native_fusion_quantize.dll `
  --head-weight-cache --head-epilogue --head-qkv `
  --reference-output results\20260906_nr_fused2160_twelve_v2\output.rgba16f `
  --matrix-dll "$build\head_wmma.dll" --matrix-profile wmma_fp16 --matrix-waves 2 `
  --matrix-modules head_ffn,head_attention,head_softmax,head_output,c32_ffn,c32_attention,c512_ffn
if ($LASTEXITCODE -ne 0) { throw 'STOP GPU: review evidence' }
```
