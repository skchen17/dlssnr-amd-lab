# 原权重 ROCm 原生路线：实施记录

## 当前结果（2026-09-06）

**PARTIAL_IMPLEMENTATION — 完整原生网络、SDR/HDR 和游戏验收尚未完成。**

最新接入续进见 `NATIVE_GPU_TEXTURE_BRIDGE.md`：原生私有模型包、常驻执行器和独立D3D12纹理
闭环已完成，640×360交替12次输出/资源复用验证通过；游戏跨进程/pre-HUD/HDR仍未完成。

最新：单颜色/无历史纹理分支的71块已在CPU及9070XT原生GPU上从输入贯通到输出，
所有编码器skip本次计算，不再依赖捕获的外部特征。144条活跃参数记录参与该分支，
8条ABI占位及1条未激活时序blend标量不计作已用神经权重。
早期连续测试在30秒堆栈采集时发生0xC0000005，失败报告保留。
审查并移除定时堆栈遍历后，分级完成128单次、640单次/12次/100次GPU复验，均正常退出。
100次同输入/同seed输出哈希一致，释放后分配/保留为零；热态逐层主机计时中位数2126.73ms。
仅恢复受控离线开发；根因未证实，真实序列稳定、HDR及游戏接入均未完成。
复验及自动评估证据：`ROCM_TRACEBACK_REVIEW.md`。
当前实现范围、证据、命令及安全停止点：`NATIVE_SINGLE_COLOR_CHAIN.md`。
下方较早“编码器/瓶颈缺口”和“捕获skip”说明保留为历史，不再代表最新单颜色分支。

2026-09-06 用户更新首个目标为**4K60游戏内稳定输出，暂不要求NVIDIA同等画质**。
已增加原生FP16整矩阵与ROCm图重放的显式实验配置；完整目标尚未实现。
当前计划、性能证据、真实游戏验收和缺口以 `NATIVE_REALTIME_MILESTONE.md` 为准。
下方此前“画质优先、不设帧率门槛”内容保留为历史，不覆盖这次用户选择。

### 最新跨尺度和解码器续进（覆盖下方对应的旧缺口）

原生outview通道交错布局已修复；C128/C256下采样真实捕获CPU边界误差降至4.13%/6.36%。
四种原始权重解码器上采样投影/skip融合已实现，block48–69共22块在CPU和RX9070XT上
连续运行完成；GPU内部不使用RTX替换、没有CPU神经计算回退。
入口和四个编码器skip仍使用历史捕获的真实外部特征，因此不是完整网络或RGB画面输出。
GPU最终特征相对RTX NRMSE6.55%，相对同配置CPU4.69%；不宣称逐位一致或原画质。
单次诊断逐层等待累计约1886.51ms，峰值张量分配156,802,560字节；不是游戏FPS。
上采样C256/C32的小规模梯度探测和22块GPU子图均正常退出，释放后分配/保留统计为零。
原权重未变，未微调、启动游戏、改变TDR或运行4K满负荷测试。

完整命令、证据与剩余图连接：`NATIVE_REALTIME_MILESTONE.md`的“续进”节。
按原记录哈希绑定的更新覆盖清单：
`results/20260906_native_scale_reconstruction/model_description.json`。
下一步仍是补齐前处理、编码器连接、瓶颈两端空间适配及颜色输出；
不要再将四个解码器上采样家族列为完全未实现，也不要因此宣称全模型完成。

### 最新 GPU 续验（2026-09-06；覆盖下方较早的“CPU-only”状态）

新增独立进程监控 `scripts/validate_rocm_lifecycle.py`，把有限数值、GPU工作结束、
资源释放和**实际正常退出**分开验收。输出包含进程PID、阶段日志、调用栈文件和主机超时。
超时不自动杀进程、不重试，更不代表取消已提交的GPU工作。

- 原权重 **block40 C512四阶段、block31 ViT1024、block14 C128下采样候选**
  已在RX9070XT ROCm上完成单窗口/短序列前向、输入与一个选定参数的反向验证。
  本轮小样本输出与同模型CPU参考一致；这不是RTX画质证明，也不是所有参数梯度/全部权重实例验收。
- **八个原始ViT块31–38（96tokens）连续GPU前向已完成**，只在入口使用原捕获特征；
  内部全部使用前一AMD模块的输出。与原CPU八块候选输出逐字节一致。
  与NVIDIA参考的特征NRMSE仍为**47.77%**，不能称为恢复了原画质或可以直接微调掩盖结构错误。
  主机提交+逐块等待约**531ms**；峰值张量分配296,718,848字节、保留316,669,952字节。
  **GPU事件出现负耗时，原报告的事件总耗时无效**；修订后报告会把无效总耗时置空，
  不用该数字计算速度或FPS。这些都是瓶颈子图数据，不是整帧网络性能。
- 五次最终小规模测试均正常退出；释放模型后保留的BLAS工作区通过当前PyTorch诊断API释放后，
  其分配/保留内存统计都归零。第一轮matmul严格零占用检查失败的报告保留，不改写。
  这只解释本次残留工作区，**此前C256链退出挂起的根因仍未定位**，长时稳定性尚未验收。
- 下采样目标布局仍未解决；本轮GPU执行通过不提升其结构/画质状态。没有训练、启动游戏或修改TDR。

证据汇总与事件计时更正：`results/20260906_rocm_gpu_execution_review.json`。
回归测试：CPU环境全套**666 passed /43.57s**；独立ROCm环境的38项相关CPU测试通过。
最终审计 `results/20260906_rocm_gpu_final_audit.json`：219份基线及4个游戏程序哈希不变、
游戏关闭、最近一小时未查到目标系统异常事件；没有残留GPU验证子进程。
按实际记录哈希附加GPU证据的描述：`results/20260906_native_tensor_reconstruction/model_description.json`。
仅将block40的4份记录、八块ViT的32份实质参数记录提升到对应的有限GPU证据级别；
其余60份split512记录仍是CPU证据，不把同家族的一次测试算作全部权重已验收。

复现命令（在项目根目录运行，输出目录必须不存在；每条结果人工审查通过后再运行下一条，不能盲目串联）：

```powershell
powershell -NoProfile -File scripts/audit_rocm_implementation.ps1 -OutputPath results/rocm_preflight_NEW.json
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe matmul --output results/rocm_matmul_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe split512 --output results/rocm_split512_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe vit1024 --output results/rocm_vit1024_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe downsample128 --output results/rocm_downsample128_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe vit_bottleneck --output results/rocm_vit_bottleneck_NEW
```

审查包括游戏关闭、无新设备异常、父报告正常退出、工作区释放完成。
私有原权重、旧参考与CPU对照保留在本地；这些脚本不是可独立分发的教师实验包。
下一开发阶段继续补齐前处理/噪声、跨尺度布局、解码器上采样融合和整帧尺寸连接，
不恢复逐指令精确对齐路线，也不提前开始微调。

### 最新续建：完整张量图优先（2026-09-05）

路线明确为：**先补全原权重张量图 → 检查结构、布局、颜色和完整图像 → 必要时微调**。
局部 NRMSE 是诊断记录，不是暂停剩余模块开发的条件。结构/资源布局错误不能用训练掩盖。
前一轮 ROCm 进程收尾异常仅暂停新增GPU实验；这批只做CPU参考及CPU测试，绝非正式CPU神经算子回退。

本次新增实质计算：

- `native_split_swin512.py`：四条原权重记录组成完整512通道模块。
  前馈为512→512预投影、八组64→256→64前馈，然后前馈投影、16头窗口注意力、最终投影与残差。
  16个原始块的64份记录已全部解包检查；block40/41四阶段连续CPU参考输出有限，NRMSE分别6.45%/9.12%。
  输入和末端投影参数具备有限非零CPU梯度，无优化器更新。
  证据：`results/20260905_native_split512_cpu_v1/manifest.json`。**尚未GPU复测**。
- `native_vit1024.py`：1024→4096→1024前馈、32头QKV、原缩放与归一化、全局注意力、投影和残差。
  按查询/键分段调度，限制score工作区，不物化整幅N×N矩阵，不擅自改成局部注意力。
  保留显式有界指数近似、未归一化FP8权重的P@V及随后归一化；训练导数采用明确标记的STE。
  八个原始块的40份记录已检查，其中8份2字节记录是原attention ABI未读取的占位，不冒充神经权重。
  `results/20260905_native_vit1024_cpu_v1/manifest.json`：两个完整ViT块CPU参考NRMSE17.37%/26.59%。
  `results/20260905_native_vit_bottleneck_cpu_v1/manifest.json`：**八块连续张量路径**（对应slots58–97），
  内部没有RTX中间值替换，末端有限，NRMSE47.77%。这说明张量链已能计算，不代表结构/画质验收通过。
  该次CPU推理15.39秒只供调试参考，不能用于预测ROCm性能。
- `native_sequence_layout.py`：序列packed布局、32-token补齐及有效token读写，支持非整齐token数；
  `native_multiscale.py`：全局4×4单元图像布局、2×2平均池化与原权重通道投影，同时保留skip/下层特征。
  C128/C256 skip对照NRMSE3.53%/7.72%；**目标下采样视图仍严重不匹配，连接布局未解决**。
  `results/20260905_native_downsample_cpu_v1/manifest.json`、`..._v2/manifest.json` 保留两种布局诊断，不删失败证据。
  `captured_outview`目前只是候选，不允许因尺寸正确就直接接入最终运行时。
- `native_transition_projections.py`：原block30.layer4与block39.layer0的512→1024、1024→512通道投影。
  编码器池化有效区、解码器空间上采样/skip融合仍须恢复；缺失融合明确报错，不静默采用插值或全零skip。
- 修正后续ViT采集清单生成器的FFN输出dtype：核函数先转E4M3并打包存储，393216字节是96×4096 FP8，
  不是96×2048 FP16。历史捕获文件/哈希保持原样，新生成清单使用正确标签。

统一记录描述在 `results/20260905_native_tensor_reconstruction/model_description.json`，153份原记录逐份列出来源哈希、
张量实现状态及未恢复连接。记录数量不能换算为项目完成率：预处理、四个上采样融合家族、其他空间适配和最终颜色合成仍未完成。
全部新代码只使用张量运算；本轮CPU参考与真实ROCm验证在报告里明确区分。没有改游戏、上传权重、启动训练或重试GPU挂起。

CPU复现命令（无需启动游戏，也不运行GPU）：

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\.venv-rocm\Scripts\python.exe scripts\validate_native_split512.py --output results\split512_cpu_new
.\.venv-rocm\Scripts\python.exe scripts\validate_native_vit1024.py --output results\vit_cpu_new
.\.venv-rocm\Scripts\python.exe scripts\validate_native_vit1024.py --output results\vit_chain_cpu_new --blocks 31 32 33 34 35 36 37 38 --chain
.\.venv-rocm\Scripts\python.exe scripts\validate_native_downsample.py --output results\downsample_cpu_new
.\.venv-rocm\Scripts\python.exe scripts\describe_native_reconstruction.py --output results\native_description_new\model_description.json
.\.venv-rocm\Scripts\python.exe -m pytest tests\test_native_split_swin512.py tests\test_native_vit1024.py tests\test_native_multiscale.py tests\test_native_transition_projections.py tests\test_describe_native_reconstruction.py -q
```

所有输出目录必须是新路径。数值对照失配保留在报告中，不阻止创建其余张量模块。
下一步优先原输入/噪声预处理、四个上采样融合家族与空间布局连接，再形成完整整帧数学图；暂不微调，也不重回逐指令精确对齐。

回归：完整tests **659项通过**（43.73秒）。`results/20260905_native_tensor_cpu_audit.json` 确认219项基线和4个游戏文件未变，
检查时游戏未运行、无所查系统设备事件。本轮所有新增推理均为明确标注的CPU参考，未消耗GPU微调预算。

### 本次续建：宽通道完整 Swin（2026-09-05）

- 新增 `native_window_attention.py`、`native_grouped_ffn.py`、`native_packed_swin.py`。
  C64/C128/C256 已从原权重计算 FFN、QKV、窗口注意力、投影及残差，并恢复用于这些案例的 packed 窗口适配。
  实际神经计算使用 ROCm 张量算子，不执行 PTX 或 CPU 神经算子。静态 PTX 仅用于恢复数学关系。
- 宽通道 FFN 不能按普通两层 MLP 实现：当前恢复为 `C→4C→分组128→32→C通道混合`。
  扩展区 `4C²` 字节、分组收缩区 `128C` 字节、混合区 `C²` 字节，共 `5C²+128C`，
  对应 C64/C128/C256 的 28672/98304/360448 字节前缀；C32 没有最后混合区。
  权重按实际 tile/head/K32 顺序解包，参数默认冻结，未训练。
- `results/20260905_rocm_multihead_attention_v1/manifest.json`：37 份匹配记录的注意力参数解包有限；
  五种记录类型的小批量 GPU 前向／反向探针通过；四个 C32 原生边界对照各12窗口零差异。
  **37份只指注意力记录检查，不代表153份权重或完整网络已实现。**
- `results/20260905_rocm_wide_swin_v2/manifest.json`：编码器 C64 slots7/8；
  `results/20260905_rocm_wide_swin_v3/manifest.json`：C128 slots11/12、C256 slots17/18；
  `results/20260905_rocm_wide_swin_decoder_v1/manifest.json`：解码器 slots133–138、141–144、147/148。
  共18个**独立模块**均通过旧的局部门槛（correlation≥0.99、NRMSE≤0.10）；NRMSE 4.18%–5.61%，
  最大绝对误差最高3.0，无非有限值。该宽松算子门槛不是PSNR≥35dB/SSIM≥0.98图像门槛。
- `results/20260905_rocm_swin256_chain_v1/manifest.json`：六个解码器模块133→138连续计算，
  只有第一个输入来自RTX捕获，之后使用上一层的GPU输出，不注入RTX中间值。
  NRMSE 4.74%→7.12%→8.45%→9.89%→10.67%→10.89%；后两层失败，不能提升为默认路径。
  这是完整网络的一段，不是完整640×360图像推理。诊断仍逐层读回结果计算指标，正式游戏路径尚未实现。
- 性能记录含每12窗口主机等待和首次算子准备，不能用作游戏FPS或稳态性能承诺。
  串联测量的峰值已分配显存约99MiB，不含完整模型／游戏总显存。

安全与未通过项：串联数值报告写盘后，实验进程没有正常退出并持续占用CPU。
已核实命令行后结束本次实验的Python进程（当时PID22620及其启动器27572），没有结束其他应用。
`results/20260905_rocm_multihead_audit.json` 显示219项基线、4个游戏文件未变，检查时无4101/41/6008事件；
这不足以证明驱动无异常。**本轮停止追加GPU实验，收尾挂起原因尚未定位，不能报告稳定性通过。**
后续代码增加串联FP8转换的有限围栏等待、避免最后一层无用转换、30秒Python堆栈诊断；
这些收尾调整仅做CPU回归，尚未重新GPU验证，不声称修复了挂起。

运行命令（先排查收尾异常、确认GPU运行环境健康，再恢复GPU测试；不要自动重试）：

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\.venv-rocm\Scripts\python.exe -m pytest tests\test_native_grouped_ffn.py tests\test_native_window_attention.py tests\test_native_packed_swin.py tests\test_validate_rocm_wide_swin.py -q
# 独立模块；输出目录必须不存在
.\.venv-rocm\Scripts\python.exe scripts\validate_rocm_wide_swin.py --output results\rocm_wide_new
# 连续子图；不允许不连续层或跨未恢复的分辨率转换
.\.venv-rocm\Scripts\python.exe scripts\validate_rocm_wide_swin.py --output results\rocm_chain_new --slots 133 134 135 136 137 138 --chain
```

下一步：先定位运行时收尾问题；继续恢复预处理、上下采样／输出视图、512通道split-Swin和ViT。
保留串联误差证据，优先补全数学图并检查图像，不能回到逐条指令零差异追踪，也不能在结构和教师数据未就绪时直接训练。

本次CPU回归：完整 `tests` **624项通过**（42.52秒）；独立ROCm环境中的新增模块／选择门槛专项52项通过。
`results/20260905_rocm_swin256_chain_v1/runtime_observation.json` 单独记录非正常收尾，保留原始数值报告不改写为成功。
最终审计 `results/20260905_rocm_multihead_final_audit.json` 确认基线／游戏文件未变、游戏未运行、所查系统事件为空。

### 上一批已验证基础

- 独立 `.venv-rocm`：Python 3.12.11、PyTorch 2.9.1+rocmsdk20260116、ROCm 7.2；未修改 ComfyUI 或系统驱动。依赖和下载哈希在 `requirements-rocm.lock`。
- `results/20260905_rocm_native_baseline/frozen_full.json` 冻结 219 个既有模型／计划／图像／运行资产的哈希，不覆盖旧版本。
- `operator_probe.json`：gfx1201 上 FP16 GEMM、LayerNorm、SDPA、E4M3 转换和反向传播通过。首次调用含编译／初始化，不是网络性能报告。
- `results/20260905_rocm_swin_four_cases/manifest.json`：四个原权重 Swin 1h/32 案例（slots 3/4/151/152）通过已有 RTX 局部容差，NRMSE 1.77%–2.58%。输出与之前 DXIL 原型逐字节一致；**原型的完整网络画质回归仍然存在，不能提升为默认路径**。
- `results/20260905_rocm_head_v1/manifest.json`：融合 → FFN → QKV → 归一化 → 注意力 → 投影 → 残差输出头在 ROCm 上执行。8,128,512 个融合 FP16 值与原生对照完全一致；旧 SDR 合成图像平均/最大误差 3.407e-6/8.626e-4。残差最大误差 0.3555，不能用裁剪后的低误差掩盖；这是移植对照，不是 RTX 整帧画质通过。
- `gradients.json`：Swin 输入／缩放参数和输出投影具有 GPU 上有限非零梯度；无优化器更新，未开始训练，8 GPU 小时微调预算未使用。
- 原始模型 SHA256 始终为 `A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`。

现有张量模块使用初始化时解包的原权重和 ROCm 张量算子；不加载 PTX/ZLUDA。仅覆盖已恢复的模块，**不是完整 Transformer 模型**。
部分投影仍用 FP32 矩阵乘加保留已知分段舍入，因此尚不声称高性能 FP16/FP8 原生矩阵路径已全部使用。

## 复现命令

项目目录：`C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab`。

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
uv venv --python 3.12 .venv-rocm
uv pip sync --python .venv-rocm/Scripts/python.exe --require-hashes requirements-rocm.lock

# 以下输出路径必须不存在；不要覆盖已接受证据。
.\.venv-rocm\Scripts\python.exe scripts\probe_rocm_tensor_ops.py --output results\rocm_probe_new\probe.json
.\.venv-rocm\Scripts\python.exe scripts\validate_rocm_swin.py --output results\rocm_swin_new
.\.venv-rocm\Scripts\python.exe scripts\validate_rocm_head.py --output results\rocm_head_new
.\.venv-rocm\Scripts\python.exe scripts\validate_rocm_gradients.py --output results\rocm_gradients_new\report.json
.\.venv-rocm\Scripts\python.exe -m pytest tests\test_native_swin_torch.py tests\test_native_head_torch.py tests\test_teacher_sequences.py -q
```

每次 GPU 验证前正常退出游戏，不与其他实验并发。探针／模块验证采用小工作量和有限事件等待，超时不自动重试；CPU 截止时间不代表 GPU 工作能安全取消。

## 教师采集

服务器只读检查仍见两张 4090D、驱动 550.144.03、CUDA 12.4；没有检测到 Wine。既有记录证明原 sm120 包在这套环境不兼容，未修改服务器驱动、未上传文件。

`deliverables/rocm_teacher_sequence_20260905_v2.zip` 为已编译但尚未 RTX 验证的候选采集宿主，附运行命令，不含私有权重和组件。v2 增加提交／围栏失败检查；v1 保留为历史构建，不再推荐。
它读取真实 FP16 颜色／深度／运动和显式曝光、抖动、重置值，校验实际上传纹理，保存组件输出候选。详见 [教师包说明](ROCM_TEACHER_PACKAGE.md)。

**当前缺少已验证颜色约定的真实连续帧清单**，因此包内没有假造示例序列，也不能现在宣称可收齐 24×32 个合格教师帧。
成功 DLAA 返回、feature18 日志和有限输出均不足以证明 NR 实际输入/输出身份；这些候选必须经过独立验收，才能用于训练。

`audit_teacher_sequences.py` 检查形状、哈希、有限值、连续帧、场景划分、同输入绑定和显式元数据；结构通过不授予训练资格。

## 下一步及不可跳过的门槛

1. 恢复剩余预处理、多尺度编码器、16h/512、ViT、解码器及张量视图。现有模块不是允许跳过缺失阶段的替代品。
2. 完整640×360数学基线与匹配教师图像通过后，再推广完整网络尺寸；当前动态窗口／输出头覆盖测试不代表动态整网已完成。
3. 正确采集游戏的颜色、深度、运动、曝光和重置元数据，确认 NR 输入/输出资源；形成 SDR/HDR、不同场景的 24×32 教师集，按场景 16/4/4 划分。
4. 原权重完整图像门槛不通过时，先修结构／布局／颜色；之后才允许限额微调，并独立保存派生模型。
5. 复用已验证 D3D12↔HIP 缓冲区和围栏，完成 GPU 常驻整帧执行及 HUD/色调映射前接入；当前 pre-Present 调试代码不修改。
6. SDR/HDR 图像、实际显示、非整齐/超宽尺寸，以及各1000帧稳定性全部通过才关闭里程碑。第一里程碑不设 FPS 硬门槛，但必须报告完整性能数据。

本轮没有启动游戏、改游戏文件、覆盖存档、修改 HDR 设置或进行模型训练。

回归：完整 `tests` 572 项通过；独立 ROCm 环境中新路线专项 39 项通过。
`results/20260905_rocm_native_baseline/final_audit.json` 确认219项基线和4个游戏程序文件未变、游戏未运行、最近一小时未见系统4101/41/6008事件。检查范围不代表任何GPU故障都可被这些日志捕获。
