# 原生游戏实时里程碑（2026-09-06 更新）

用户已选择 **3840×2160、60 FPS**。当前状态：**未完成**。
当前顺序：先游戏接入，再速度优化。跨进程原生GPU闭环已通过12帧；实际游戏边界后仍有
35次绘制留在同一列表内，需资源依赖解析及状态保真的列表内部拆分，尚未启用游戏网络写回。
最新证据与接入任务见 `NATIVE_GAME_PROCESS_BRIDGE.md`。
最新：独立D3D12共享缓冲区↔原权重ROCm整帧网络↔输出纹理已经通过640×360交替12帧测试。
不是游戏跨进程接入，未证明pre-HUD可见上屏；细节与下一步见 `NATIVE_GPU_TEXTURE_BRIDGE.md`。
最新续进已贯通单颜色输入的原生整帧71块计算链，所有编码器skip自行计算；
定时堆栈采集异常已审查并修改监控；640×360固定输入连续100次复验正常退出、输出哈希一致。
热态逐层主机计时中位数约2126.73ms，尚未优化，不是游戏帧率；原故障根因仍未证实。
仅恢复受控离线开发，真实序列、颜色/HDR、pre-HUD上屏及4K60仍未验收。
复验范围及证据见 `ROCM_TRACEBACK_REVIEW.md`。
完整现状以 `NATIVE_SINGLE_COLOR_CHAIN.md` 为准；下方解码器仍依赖捕获skip的记录已成为历史。
本次调整将“匹配NVIDIA图像质量”延后，不再用RTX的PSNR/SSIM或局部NRMSE阻止实时版本开发。
原始权重保持不可修改；算术近似必须另有精度描述，不冒充原指令语义或已达到DLSS5效果。

## 这次调整没有放弃的要求

- 整帧网络输入与输出；尺寸由输入决定。内部合法padding/window调度不等于独立小图拼接。
  不静默缩小网络输入、不跳帧显示上一结果、不用原游戏回退帧冒充网络生效。
- 正式推理不依赖NVIDIA DLL/PTX/ZLUDA/RTX中间值，神经计算不回退CPU。
- D3D12与HIP共享GPU缓冲区、围栏和资源代次管理；正式图像路径不做CPU读回/共享内存像素交换。
- 网络实际在HUD之前输出并上屏；SDR/HDR颜色和曝光可用，无黑帧、网格接缝、明显闪烁或HUD损坏。
- 分级安全验证，最终SDR/HDR各至少1000帧；发生设备故障/挂起停止，不自动重试或改TDR。
- 游戏设置/文件可回退。任意分辨率支持不代表任意分辨率都保证同样帧率。
- 原权重、架构与教师数据依然私有；微调仍另存派生模型，首轮不超过8 GPU小时。
  未自动授权换轻量学生架构或降低网络输入分辨率。

60 FPS总帧预算约16.67ms，需要包括实际游戏渲染和接入成本。子图低于16.67ms不能算整机达标。
验收工具还检查稳定帧时（p95呈现间隔≤16.67ms）；无法达标时如实报告实际FPS和抖动，不四舍五入为完成。

## 已实施的性能路径

`native_execution_policy.py` 增加两个显式、作用域隔离的配置：

- `recovered_k32`：保留现有逐K32产品/FP16部分累加的诊断参考，仍是默认值。
- `native_fp16`：相同原权重和张量关系，使用整矩阵FP16库GEMM，再作FP16残差加法。
  FP8量化边界、原缩放、窗口/全局注意力区分仍保留。大矩阵的累加/舍入行为与旧参考不同，明确计入误差。

当前加速覆盖分组FFN、多头窗口注意力、C512四阶段、ViT与已实现的边界投影。
独立旧C32/输出头路径没有因此自动变成完整加速版。
配置不改写原权重，也没有优化器更新。不能由“FP16整矩阵”推导为“已经完整恢复原网络”。

`native_graph_replay.py` 新增形状固定时的ROCm图捕获/重放：
权重、输入和输出驻留GPU，一次提交重放整个已实现子图。
虽然PyTorch将此接口命名为`torch.cuda.CUDAGraph`，运行前会强制检查HIP；本次实测设备为gfx1201。
准备新尺寸需要新图，不能继续使用旧尺寸资源；提交者线程/流改变时拒绝执行。
该类是实验性张量执行器，不是D3D12游戏共享缓冲区的完整接入。

## 实测证据与边界

对象：原八个ViT块31–38，96tokens，对应此前640×360捕获的瓶颈。
只有入口使用历史捕获特征；八块之间没有RTX结果替换。**不是4K网络输入，也不是游戏序列。**

| 同一FP16候选的执行方式 | 热态主机耗时中位数 | 12次同输入输出 |
| --- | ---: | --- |
| 逐块主机等待 | 59.48ms | 一致 |
| 连续提交、子图末端等待 | 18.45ms | 一致 |
| ROCm图重放、子图末端等待 | 6.55ms | 一致 |

时间包含提交与事件轮询等待，不含权重加载/捕获准备，也不含完整游戏或整帧其它网络阶段。
轮询间隔5ms会影响测量；当前阶段统一采用主机时间，不用先前出现过负数的GPU事件计时推断吞吐量。
图重放与同配置普通执行输出哈希一致；图重放热态最大6.87ms。
相对旧CPU重建候选NRMSE8.59%，相对原RTX特征NRMSE48.28%；后者仍是特征诊断，不是图像质量指标。
大矩阵更改没有消除结构/算子误差；当前只证明原权重GPU执行及提交优化可用。

证据：

- `results/20260906_rocm_fast_vit_chain_twelve/manifest.json`
- `results/20260906_rocm_fast_vit_framewait_twelve/manifest.json`
- `results/20260906_rocm_fast_vit_graph_twelve/manifest.json`
- `results/20260906_rocm_fast_downsample128_one/manifest.json`
- `results/20260906_rocm_fast_split512_one/manifest.json`

C128下采样和C512在新配置下也通过有限前向及输入/选定参数梯度验证。
上述测试均正常退出，模型与BLAS/图工作区释放后的分配/保留统计为零。
这不是长期稳定性验收，也没有解释此前C256链挂起的根因。此次没有启动游戏。

回归：完整CPU测试 **685项通过（42.64s）**，新脚本语法检查通过。
`results/20260906_native_realtime_final_audit.json` 确认219份基线及4个游戏程序哈希不变、
游戏关闭、最近一小时未查到目标设备/电源异常事件；没有遗留GPU验证进程。

## 真实游戏验收

`scripts/assess_native_realtime.py` 单独评估规范化的游戏记录。
完整图、真实4K尺寸、SDR/HDR各1000帧、当前帧网络实际写入、可见输出、资源同步、
图像审查、实际Present帧时缺一不可。缺失字段拒绝验收；不能只提供游戏FPS或一个GPU子图报告。
它不生成采集证据，视觉标志必须由真实截图/序列检查给出，单元测试模拟数据不是游戏结果。

当前输入小子图报告得到的结果为 `results/20260906_realtime_gate_current.json`：**NOT_ACCEPTED**。
这个拒绝是正确行为，当前没有完整游戏证据。

## 距离接入还缺什么

1. 监控异常已审查并完成静态100次复验及独立纹理12次交替输入验证；新负载仍需分级安全检查。
2. 单颜色分支的前处理、编码器、瓶颈空间连接和全部解码器已经串通；
   剩余原始时序输入分支、正式颜色/曝光合成和更大动态尺寸契约需要验证，不能交给微调猜测。
3. 独立同进程共享GPU纹理闭环已通过；仍需游戏跨进程资源/围栏协议与正确的pre-HUD消费者。
   旧CPU像素IPC/pre-Present分块调试版不提升为正式路径。
4. 分级测量实际4K整帧和游戏FPS/显存，完成SDR/HDR稳定性检查。

后续逆向算子和教师微调是可用的质量改进路线，**不是保证恢复到原版效果或保证原模型能4K60**。
若完整图性能证据要求压缩模型、降低内部尺寸或改变架构，必须单独说明取舍，不能在此目标下静默替换。

## 复现命令

在项目根目录，输出目录必须不存在。逐条审查通过后再进行下一规模，禁止GPU挂起后自动重试：

```powershell
powershell -NoProfile -File scripts/audit_rocm_implementation.ps1 -OutputPath results/fast_preflight_NEW.json
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe vit_bottleneck --precision native_fp16 --synchronization frame --execution graph --output results/fast_graph_one_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe vit_bottleneck --precision native_fp16 --synchronization frame --execution graph --iterations 12 --output results/fast_graph_twelve_NEW
.\.venv-rocm\Scripts\python.exe scripts/assess_native_realtime.py --input results/REAL_GAME_EVIDENCE.json --output results/REAL_GAME_ASSESSMENT_NEW.json
```

最后一条需要未来真实游戏记录；当前没有该文件，不能拿子图报告或伪造数据代替。

## 续进：跨尺度布局和22块解码器（2026-09-06）

本节更新前文性能实验之后的实现状态，不覆盖或改写历史报告。
该子图之后的单颜色整链进展及安全停止点见 `NATIVE_SINGLE_COLOR_CHAIN.md`。

- 修复下采样输出的物理通道排列：每组16通道实际将MMA的N0/N8成对交错存储，
  不是连续平面通道。C128/C256捕获边界的特征NRMSE从129.18%/125.20%降至4.13%/6.36%。
  三阶段跨尺度CPU串联slots15–17的最终特征NRMSE为8.02%，无内部RTX替换。
- 四种原权重上采样模块均实现低分辨率投影、2×2展开、实际编码器skip融合及Swin计算。
  CPU同输入局部RTX特征NRMSE分别为C256 5.48%、C128 5.32%、C64 5.02%、C32 2.90%。
- 解码器block48–69共22块连续执行，涵盖四种上采样、普通窗口块和输出布局转换。
  CPU与RX9070XT原生ROCm均跑通，原权重不变，22块内部不读RTX中间值。
  **入口及四个编码器skip仍是实际历史捕获的外部输入**，不是完整RGB网络。
  C32端特征尺寸320×192；本次没有验证4K或任意尺寸的完整图。
- GPU末端相对RTX特征NRMSE6.55%，相对同配置CPU候选4.69%，均输出有限值；
  不是逐字节一致，也不是最终画质指标。保留所有中间层误差，未做微调。
- GPU单次逐层提交/等待累计约1886.51ms，峰值张量分配156,802,560字节。
  这是含冷态调用、窗口分批和逐层等待的诊断路径，尚未图捕获优化，不能换算为游戏FPS。
  它也没有提供4K60可达的证据，后续仍需实测与优化。
- C256/C32上采样小规模GPU前向/选定参数梯度及22块子图均正常退出，释放后
  分配/保留内存为零。不代表历史挂起根因已解决或SDR/HDR长期稳定性通过。

实现与证据：`scripts/native_multiscale.py`、`scripts/native_upsample_swin.py`、
`scripts/native_decoder_pyramid.py`；报告位于：

- `results/20260906_native_downsample_outview_v3/manifest.json`
- `results/20260906_native_scale_chain_v1/manifest.json`
- `results/20260906_native_upsample_cpu_v1/manifest.json`
- `results/20260906_native_decoder_pyramid_cpu_v1/manifest.json`
- `results/20260906_rocm_upsample256_one/manifest.json`
- `results/20260906_rocm_upsample32_one/manifest.json`
- `results/20260906_rocm_decoder_pyramid_one/manifest.json`

复现需要私有原权重、历史真实特征和既有CPU对照报告，缺失时明确失败，不补造数据。
在项目根目录运行，每个GPU结果审查正常退出后才继续；输出目录必须不存在：

```powershell
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_scale_chain.py --output results/scale_chain_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_upsample.py --output results/upsample_cpu_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/validate_native_decoder_pyramid.py --precision native_fp16 --output results/decoder_cpu_NEW
powershell -NoProfile -File scripts/audit_rocm_implementation.ps1 -OutputPath results/decoder_preflight_NEW.json
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe upsample256 --precision native_fp16 --output results/upsample256_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe upsample32 --precision native_fp16 --output results/upsample32_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe decoder_pyramid --precision native_fp16 --output results/decoder_gpu_NEW
```

当前GPU验证器的CPU对照固定为已留存且校验哈希的
`results/20260906_native_decoder_pyramid_cpu_v1`；上述NEW目录是另一次复验，不自动替换基线。

本次完整CPU回归 **699 passed / 50.91s**，新增及修改脚本语法检查通过。
最终审计 `results/20260906_native_scale_final_audit.json`：219份基线、4个游戏程序哈希不变，
游戏关闭，最近一小时未查询到目标系统设备/电源异常事件。进程检查未发现遗留验证器。
覆盖清单 `results/20260906_native_scale_reconstruction/model_description.json` 按原记录哈希
仅提升实际执行过的22份解码器记录，不将捕获skip或同家族其它记录算作原生全图证据。
