# 71-block C++/HIP NRPlan 迁移结果

日期：2026-09-07。目标是先固定完整 71-block 拓扑，消除每帧 Python/PyTorch 算子提交，
再迁移已经通过 correctness/performance gate 的模块。所有 GPU 实验保持 RGP 安全锁，
只使用 128、640 和 1080 的有界生命周期测试，没有运行 4K、counter capture 或游戏。

## 本轮达到的边界

完整 71-block 单色整帧网络现已能在初始化时捕获为 HIP Graph，由 C++ `NRPlan` 实例化并
逐帧重放。每帧接口只更新输入/输出指针、尺寸、frame id 和 resource generation，然后执行：

```text
input D2D -> hipGraphLaunch -> output D2D -> optional event
```

`nrPlanSubmit` 内没有 `hipMalloc`、PyTorch 算子或设备同步。动态输入和输出已用两组不同
GPU 指针验证；128、640、1080 的完整网络输出均与 reference 逐位一致。

这是 `CPP_OWNED_CAPTURED_KERNEL_GRAPH`：C++ 拥有 stream、graph executable、提交和围栏，
但初始化时的模型权重与 graph allocation pool 仍由 Python/PyTorch 捕获器保持生命周期。
因此它是完整可运行的 71-block native submit runtime，不是最终的“所有权重和 arena 都由
C++ 独立加载/分配”的部署版本。后者仍需逐模块把 graph 参数重绑定到 `NRPlan` 自有 arena。

## 必要的 capture 修复

- 固定 conditioning 改为模型 resident buffer，取消 forward 内 host-to-device `torch.tensor`。
- layout 常量改为纯设备算术，取消捕获期 host 常量拷贝。
- reference scatter 从动态 boolean `nonzero/indexPut` 改为固定形状 `index_copy_` 加丢弃哨兵行；
  CPU 布局回归和 128/640/1080 整帧哈希均一致。
- PyTorch 只在初始化时管理 capture pool，C++ 直接实例化 raw HIP graph。
- ROCm Windows 在较大长依赖图的默认宿主栈上会发生 stack overflow。实例化现改在一个
  仅初始化使用、预留 64 MiB 栈的线程中完成；不修改 TDR，不改变 GPU 工作负载。
- C++ 查询 graph node type，得到可复现的 kernel/memcpy 节点数，不用 RGP 或 host 时间
  冒充 kernel busy time。

## 1080p B-A-B-A

每个位置是独立进程、3 帧，取后 2 帧中位数；A 为 Python 每帧提交同一数学路径，B 为
`NRPlan` graph replay。四个进程输出均为
`892FB11D82C642996AD321052B2804A72BAC919620DA9F696BC086D74DA401B7`，并正常退出、释放为 0。

| 配置 | Python host ms | NRPlan host ms | NRPlan 自身变化 | NRPlan kernel nodes |
|---|---:|---:|---:|---:|
| reference Swin/ViT + native Head input/compose | 564.856 | 561.665 | -0.565% | 35,010 |
| + resident scale transitions + C64/C128/C256/C512 grouped FFN | 458.635 | 458.956 | +0.070% | 30,976 |
| + accepted Head bounded WMMA + Pre project/pack + quantize/cubic kernels | 311.863 | 266.008 | -14.703% | 16,558 |

仅 graph replay 对碎片化 reference 路径几乎没有收益，证明主问题仍是图内大量小 kernel 和
中间操作；当已通过的 Head/Pre/grouped/transition 模块迁入后，固定重放才额外降低约
14.7% host submit/wait。最终组合相对首行 Python reference 总体下降 52.91%，但 266 ms
仍远离实时目标。

Python 路径的 HIP event 只覆盖设备 stream 区间，不能包含大量 host-side gap/blocking，
所以本轮 runtime A/B 以 host submit/wait 为主口径。NRPlan 的约 257 ms event span 是 graph
所在 stream 的 elapsed time，仍不是 kernel busy sum。RGP 安全锁未解除。

最终 NRPlan 单进程样本的 allocator peak 为 775,547,904 B、reserved 为 2,776,629,248 B，
设备用量离散采样约 3,156,541,440 B；低于 5/6 GB 门槛。相同输入多帧、所有独立进程均
哈希一致。

## 已迁入与未迁入

已装入最终固定图：encoder/decoder native scale transition、C64/C128/C256 low-LDS grouped
FFN、C512 grouped FFN、Head FFN/bounded attention/output、Pre project/pack、剩余 Swin
reference 路径和 ViT reference 路径。C32 conservative core、C64/C128 bounded/query
attention 等已拒绝候选没有重新启用。resident FP8 activation 仍按计划推迟。

仍需完成：让 C++ 加载并预打包全部权重、用显式 offset manifest 分配统一 workspace arena、
将剩余 ATen/library 节点替换为原生模块句柄，并让 graph 节点只引用 C++ 所有资源。完成前
不得把本轮结果描述成独立游戏部署 runtime、RTX 画质、HDR/时序通过或实时版本。

## 复现

先构建：

```powershell
.\tools\native_nr_plan\build.ps1 `
  -OutputDirectory "$PWD\results\nr_plan_build"
```

完整选择配置的 B 路径命令（A 路径移除 `--cpp-nr-plan-dll`）：

```powershell
.venv-rocm\Scripts\python.exe scripts\benchmark_nr_resolution.py `
  --size 1080 --iterations 3 `
  --cpp-nr-plan-dll results\nr_plan_build\nr_plan.dll `
  --fusion-dll results\20260906_native_fusion_cubic_build_v1\native_fusion_quantize.dll `
  --head-input-dll results\20260906_head_pre_c32_build_v1\native_fusion_quantize.dll `
  --pre-features-dll results\20260906_pre_project_pack_build_v1\native_fusion_quantize.dll `
  --pre-project-pack --resident-transitions `
  --transition-dll build\native_transition_20260907_v3\native_encoder_transition.dll `
  --encoder-transition --decoder-transition `
  --matrix-dll results\20260907_grouped_fp8w_build_v1\head_wmma.dll `
  --matrix-profile wmma_fp16 --matrix-waves 2 `
  --matrix-modules head_ffn,head_attention_bounded,head_output,c64_ffn_grouped,c128_ffn_grouped,c256_ffn_grouped,c512_ffn `
  --reference-output results\20260907_nr1080_python_head_reference_v1\output.rgba16f `
  --output results\nrplan_1080_b1
```

四进程完成后用 `scripts/summarize_nr_plan_abba.py` 汇总；输入顺序必须是 A-B-B-A，脚本会
拒绝异常退出、资源未释放、输入不一致或输出哈希不一致的结果。
