# 单颜色输入的原生整帧计算链（2026-09-06）

**计算链已贯通；完整原始时序/颜色契约、稳定性、游戏接入和4K60均未完成。**

最新接入续进：私有模型包及GPU常驻执行器已实现，独立D3D12纹理闭环通过640×360交替12次测试。
仍不是游戏跨进程接入或实际上屏。见 `NATIVE_GPU_TEXTURE_BRIDGE.md`。

最新复验：审查并移除定时堆栈采样后，640×360同输入/同seed连续100次正常退出，
71块完整运行，输出哈希一致，释放后分配/保留为零。仅恢复受控离线开发；
不是动态序列或游戏稳定性验收，原故障根因仍未证实。详见 `ROCM_TRACEBACK_REVIEW.md`。

本轮从真实颜色输入开始，顺序执行block0–70全部71个网络块：
前处理/坐标噪声/原始投影 → 四尺度编码器 → C512编码器 → 池化/投影 → 八块ViT →
解码器输入融合 → C512解码器 → 四尺度解码器 → 输出头/诊断合成。
所有编码器skip由此次计算产生，前向过程没有任何RTX捕获中间值替换。
独立图像小块拼接和内部图像缩放均未使用。

## 已实现的准确范围

- 保留原始权重；原模型153条记录中，144条实质神经参数记录已在本分支使用，
  8条ViT ABI占位记录不是神经权重；1条时序blend标量在无历史分支中不生效。
  不能把“153条均已解码”称为“153条均参与推理”。
- 前处理复现历史调用中的**无历史/运动/深度/掩码纹理分支**。该分支本身规定以当前颜色
  填充历史颜色特征，不是给需要真实运动数据的分支伪造零值。其它分支没有实现，明确不接受。
- 噪声由全图坐标和显式frame seed生成，输入投影仍为原FP16权重；后续采用显式native_fp16候选策略。
  权重未更新，未训练，未做逐指令仿真。
- 原始窗口偏移从哈希校验的ABI离线导出；运行模型只接收权重和数学配置，
  不读PTX、NVIDIA DLL、ZLUDA或采集激活数据。配置导出工具不是正式部署模型包。
- 全图特征按128对齐并在末端裁剪；该候选策略在已测尺寸可运行，不等于已验证所有原尺寸契约。
  默认离线预算最多1,048,576个补齐像素，4K请求明确拒绝，不缩图、不回退硬拼接。
  提高预算需资源/安全审查，不能直接全负载尝试4K。
- 末端使用明确命名的旧SDR诊断合成（base+0.25×residual、clamp、alpha=1）。
  这不是正式HDR/曝光转换，也未通过真实pre-HUD消费者上屏。

实现：`native_preblock.py`、`native_split_image512.py`、`native_transition_projections.py`、
`native_whole_frame.py`，复用既有Swin/ViT/解码器/输出头模块。

## 证据

边界检查（CPU，仅作结构定位，不要求逐位精度）：

- 前处理原始零颜色参考：skip特征NRMSE2.13%，池化outview1.43%。
  `results/20260906_native_preblock_v1/manifest.json`。
- 编码器最后投影残差、池化、瓶颈投影：0.43%、0.95%、1.68%。
  `results/20260906_encoder_pool_v2/manifest.json`。
  v1因为验证脚本将ABI维度字段顺序读反而失败，保留原报告/目录；不代表GPU失败。
- 二维与序列双向重排：逐字节一致。解码器入口投影/融合：0.52%。
  `results/20260906_bottleneck_adapters_v1/manifest.json`。

全链检查：

| 输入 | CPU全链 | RX9070XT单次全链 | 连续同输入12次 |
| --- | --- | --- | --- |
| 128×128合成渐变 | 71块有限值 | 通过，输出与同候选CPU一致 | 通过，12次输出哈希一致 |
| 640×360实际游戏图像裁剪 | 71块有限值 | 通过，CPU对照RGB RMSE0.03128 | 监控修改后12次及100次通过；旧失败另存 |
| 641×361合成渐变 | 71块有限值，输出尺寸准确 | 未测 | 未测 |

640×360输入是历史2342×1317游戏FFX输出的字节精确裁剪，不是本轮游戏全帧采集。
原始颜色语义尚未确认，RGB最大5.605、存在>1的值；因此当前clamp后的图像不能用于HDR画质验收。
详情：`results/20260905_real_color_probe/input/input_manifest.json`。
合成渐变只是测试数据，不是RTX教师数据。

GPU诊断数据（含逐块提交/等待，非优化帧率）：

- 128×128单次约2290.50ms，峰值张量分配404,745,216字节。
- 128×128十二次测试热态约635–651ms，12次同输入输出一致，循环末端分配389,949,440字节保持不变。
- 640×360单次约3986.18ms，峰值张量分配596,218,368字节；输出约5.06%的RGB分量等于1。
- 上述**成功进程**均正常退出，模型/BLAS工作区释放后分配和保留统计归零。
  不用小尺寸/静态输入测试证明游戏时序稳定、HDR质量或4K60。

CPU报告：`results/20260906_single_color_chain128_v1/manifest.json`、
`results/20260906_single_color_gamecrop640_v1/manifest.json`、
`results/20260906_single_color_odd641_v1/manifest.json`。
GPU报告：`results/20260906_rocm_whole_frame128_one/manifest.json`、
`results/20260906_rocm_whole_frame640_one/manifest.json`、
`results/20260906_rocm_whole_frame128_twelve/manifest.json`。

## 历史安全停止点与复验范围

`results/20260906_rocm_whole_frame640_twelve/manifest.json` 为失败证据，**不可用单次通过覆盖**。
子进程PID8576在约30秒返回0xC0000005，无主机超时；阶段日志显示11次block70结束，
第12次前处理期间触发异常。没有最终逐帧输出哈希报告，不能宣称11次输出一致。
日志同时显示30秒定时faulthandler正在遍历堆栈，原生异常栈位于python312.dll的
PyCode_Addr2Line/_Py_DumpTracebackThreads。说明堆栈采集参与了异常现场，**不证明底层根因已确认**。

发现后没有重试GPU或启动游戏。已采取代码级缓解：移除并发定时堆栈遍历，保留致命错误日志、
90秒主机监督、PID与阶段日志；新增每个完成帧的增量哈希记录，避免末尾失败丢失全部摘要。
以上为故障当时的处置。随后按 `ROCM_TRACEBACK_REVIEW.md` 审查并分级复验，
已完成640连续100次；历史C256退出挂起的根因仍未知。
仅恢复受控离线开发，不自动升级负载/启动游戏，新异常立即停止，更不改TDR。

异常后审计：`results/20260906_whole_frame_after_exception_audit.json`，基线及游戏哈希不变、游戏关闭，
系统日志读取成功，近一小时未查到目标设备/电源事件。此有限查询不能证明没有驱动问题。

## 复现命令

项目根目录运行，输出目录必须不存在。原权重及采集数据保持私有。

```powershell
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_bottleneck_adapters.py --output results/bottleneck_adapters_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_encoder_pool.py --output results/encoder_pool_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_preblock.py --output results/preblock_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_whole_frame.py --width 128 --height 128 --output results/single_color128_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_whole_frame.py --width 641 --height 361 --output results/single_color641_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_whole_frame.py --width 640 --height 360 --input results/20260905_real_color_probe/input/input_rgba16f.raw --output results/single_color_gamecrop_NEW
```

以下是GPU命令说明，**仅在异常审查完成后手动逐项执行**，不是当前建议立即运行的批次。
每项检查实际正常退出、有限值及释放统计后才扩大规模；失败不自动重试。

```powershell
powershell -NoProfile -File scripts/audit_rocm_implementation.ps1 -OutputPath results/whole_preflight_NEW.json
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe whole_frame128 --precision native_fp16 --output results/whole128_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe whole_frame640 --precision native_fp16 --output results/whole640_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe whole_frame128 --precision native_fp16 --iterations 12 --output results/whole128_repeat_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe whole_frame640 --precision native_fp16 --iterations 12 --output results/whole640_repeat_NEW
```

GPU验证器引用本节已留存的CPU输入/输出哈希基线；不会自动把NEW目录提升为参考。

## 接下来

1. 已完成定时采样监控审查及100次静态复验；保留异常即停，后续新负载仍需分级验证。
2. 正式化模型包、输入/输出颜色与曝光契约，补齐必要的历史/运动分支，避免HDR截断。
3. 做整链GPU图捕获、窗口批次/矩阵算子融合及工作区常驻优化，测量真实瓶颈。
4. GPU共享缓冲区/D3D12围栏与pre-HUD写入，然后真实游戏尺寸、序列和SDR/HDR验收。

不再把“全部编码器/瓶颈/解码器尚未连接”作为现状；现在是**单颜色分支连通，正式契约和稳定性未通过**。

最终回归709项通过（50.45s），修改脚本语法检查通过。
上述为旧检查点；监控复验新增测试后为720项通过（54.37s），最新审计见
`results/20260906_monitorv2_hundred_final_audit.json`。
最终审计 `results/20260906_whole_frame_final_audit.json`：219份基线、4个游戏文件哈希不变，
游戏关闭，目标系统设备/电源事件查询为空；未修改ComfyUI、训练模型或清理任何历史数据。
证据汇总 `results/20260906_native_whole_frame_review_v2.json` 保留失败批次，不将11个完成阶段
标志提升为成功的12帧测试。覆盖清单见
`results/20260906_native_whole_frame_reconstruction_v2/model_description.json`。
