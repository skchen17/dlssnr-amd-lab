# 原权重 ROCm / ReShade 整帧预览（2026-09-06）

本轮按用户同意的顺序：先做 ReShade 可见预览，再优化网络后端。不是正式 HDR/HUD 前接入，
不是 DLSS5 画质或4K60验收。不使用旧PTX分块路线；不把640×360放大到游戏尺寸。

## 实现

- `tools/native_reshade/addon.cpp`：ReShade6.8/API20的效果链回调，使用框架提供的
  `flush_immediate_command_list()` 提交前置工作；不拆分游戏自己的命令列表。
- `gpu_surface.h`：原始设备/队列上，GPU复制与像素着色器完成打包格式和RGBA16F的双向转换。
  使用像素坐标Load、不缩放；写回保留游戏alpha。显示编码值只是SDR诊断约定，非正式场景线性颜色契约。
- `worker_pipe.h`：仅元数据管道，限制继承句柄，操作系统父子PID核验，精确LUID选择，
  当帧围栏与消费确认。网络输出未完成不会向游戏队列添加等待未知未来值的围栏。
- 原权重完整单颜色候选网络在独立ROCm进程常驻；游戏进程不初始化HIP。
- 每次显式ARM仅允许1或12帧。遇到HDR、尺寸/设备/运行线程变化会拒绝，不使用旧帧填补。
  失败不自动重试，不强制终止GPU工作，不修改TDR。
- `NativePreview.fx`仅为恒等复制，用于触发明确的ReShade效果链边界，本身不是神经网络。

正常帧路径：游戏纹理 → GPU格式转换 → 共享GPU缓冲区 → ROCm网络 → 共享GPU缓冲区 →
GPU格式转换 → 同一效果目标 → ReShade后续显示。CPU图像上传/读回仅存在于独立验证的夹具和显式审计中。
游戏启动器强制 `NR_PREVIEW_AUDIT=0`，不通过CPU搬运游戏像素。

## 独立验证结果

最终验证阶段目录：`results/20260906_reshade_native_stage_v5`。

| 输入尺寸 | 网络帧数 | 结果 | 已审计显示帧 |
|---|---:|---|---|
|640×360|12|正常完成并恢复原路径，D3D12错误0|前2帧，屏幕缓冲区与网络输出转换后全像素一致|
|1280×720|1|正常完成并恢复原路径，D3D12错误0|第1帧，全像素一致|
|2342×1382|1|正常完成并恢复原路径，D3D12错误0|第1帧，全像素一致|

结果目录分别是 `results/20260906_reshade_network640_twelve_v1`、
`results/20260906_reshade_network1280_one_v1`、`results/20260906_reshade_network2342_one_v1`。
查看 `assessment.json`、`addon.jsonl`、`worker_frames.jsonl`、`worker_close.json`、`host.json`。
独立宿主使用R8G8B8A8_UNORM/SDR。并未以此证明游戏的R10G10B10A2/HDR路径正确。

2342×1382单帧处理计时25,404.6ms，峰值allocated3,240,176,640字节、reserved4,370,464,768字节。
1280×720对应8,371.7ms；640×360热身后约1.7秒。
这些是冷/热条件不同的宿主帧计时，覆盖网络提交、完成等待与有限值检查，
**不包含整个游戏帧时间、进程启动、输入转换和最终输出写回**。不能宣称是纯GPU事件时间。
没有性能优化，也不能拿这些结果称“实时”。

## 已审查的前置失败

1. 构建v1/v2：PowerShell include参数拼接及ReShade API接口使用错误，均未执行GPU工作。
2. `reshade_observe640_v1`：没有启用任何effect，插件加载但没有收到finish_effects。
   加入恒等technique后回调生效，未把“宿主退出0”冒充插件链路完成。
3. `reshade_transport640_one_v1`：资源GetDevice返回ReShade代理，与原始设备直接比较不相等。
   在提交输入/网络之前拒绝。核对官方v6.8.0的`d3d12_resource.cpp`后，改为通过同一资源GetDevice
   API对比游戏目标和自有参考纹理的canonical COM身份。没有删除设备身份检查。
   失败宿主退出后工作进程管道EOF退出，没有遗留工作进程或设备重置的新证据。

## 游戏单帧上屏结果（后续 v6）

2026-09-06：`results/20260906_gowr_native_reshade_one_v2`，游戏 PID11884。
用户临时关闭 HDR 后，重新加载 v6；现场确认实际游戏场景，目标2342×1382、
R10G10B10A2_UNORM（24）、SDR（color_space=1），再显式触发一次网络。

- `worker_frames.jsonl`：真实输入经过完整71块单颜色候选网络，输出有限；
  元数据IPC、无CPU神经算子回退。游戏审计读回关闭，所以输入/输出哈希为null，不能伪造同帧比较。
- `addon.jsonl`：sequence1 / present_index24463 的 `frame_written`，随后
  `bounded_session_complete` 与同一索引的 `processed_frame_reached_reshade_present`。
- 显示审计截图：`results/20260906_gowr_native_reshade_one_v2/GoWR 2026-09-06 03-41-29_0 NativeROCm.png`。
  这是最终显示缓冲区的诊断PNG，不是网络输入/输出通过CPU搬运的正式帧路径。
  图中人物与场景可见，但暗部有明显矩形方块伪影；不得称画质通过或无网格。
- 工作进程帧耗时26,080.8387ms；峰值allocated3,240,176,640、reserved4,370,464,768字节。
  这是宿主测得的网络提交/等待/有限值检查，不是纯GPU事件时间，也不是游戏FPS。
- `worker_close.json` 正常关闭，释放后allocated/reserved均为0；后续窗口观察确认游戏原渲染恢复。
  单帧成功不等于连续帧稳定性；尚未验证100/1000帧、HDR、HUD保护或4K60。

在此之前，v6独立宿主补充了同格式、同尺寸验证：
`results/20260906_reshade_network10_2342_one_v1` 全显示像素匹配网络输出的10-bit转换，
D3D12错误0，单帧24,432.7532ms。`results/20260906_reshade_transport10_one_v1`
被ReShade启动提示覆盖上部，完整像素对比为NOT_ACCEPTED，未用下半部一致冒充全帧通过。
此失败没有设备错误；完整网络测试耗时期间启动提示消失，因此通过完整比较。
独立宿主有D3D调试层，不能把它的零错误移作实际游戏的调试层结论。

本次 v6 增加可选 `-Format 24` 夹具、10-bit严格比较、部署格式一致性检查、
运行时重新创建后的新目标元数据，以及首个处理帧的ReShade显示截图。

### 下一步与边界

单帧接入路径已连通，当前主要问题转为画质伪影与约26秒的处理时间。
先在离线真实图像上隔离方块来源（网络窗口/布局/补齐、颜色输出头或接入），
不能仅凭外观认定是分块拼接或浮点误差；当前网络不把独立640×360小图拼成整帧。
需保存获准的同帧输入和对应输出作诊断，不把不同时间的游戏截图当精确A/B。
随后逐阶段测量网络开销，验证每项改动不引入额外画质退化；再按1/12/100帧扩大。
暂不把26秒同步路径改成连续游戏推理，也不以重复显示旧网络帧宣称实时。
原版DLSS5画质、正式颜色/时序契约、HUD保护和HDR仍未完成；本结果不能关闭最终里程碑。

恢复完成：用户确认HDR已恢复，ReShade日志03:43:25记录SetColorSpace1(12)/HDR10。
随后通过正常窗口关闭退出游戏；未强制结束进程。v1/v2本次新增的4个明确文件均已
回收到各自会话的 `recovered_game_files`，没有删除，可恢复；见各自 `restored.json`。
最终审计 `results/20260906_reshade_game_one_final_audit.json`：219项冻结资产和4个原游戏
文件哈希未变，游戏/网络工作进程均已退出，最近一小时指定系统事件4101/41/6008无记录。
这只说明本轮记录范围内未发现这些错误，不是长期无黑屏保证。没有训练或更改TDR。
CPU回归778项通过（74.64秒）；Python编译及git diff --check通过（仅既有换行提示）。
最终结论：单帧实际游戏显示路径已验证；画质/连续稳定性/实时性能均未验收。

## 历史游戏进度（v1，仅观察）

游戏会话：`results/20260906_gowr_native_reshade_one_v1`，PID19280。
已经通过ReShade加载原生插件，进入原有场景。首次观察到2342×1382、格式24、颜色空间HDR10_PQ。
网络未ARM、未启动、未写回。用户已被请求临时切到SDR再回到场景。
**当前不能宣称原生ROCm网络已经在游戏上屏。**

部署只新增`dxgi.dll`、`native_rocm_preview.addon64`、`ReShade.ini`，另有ReShade生成的日志。
原游戏exe/FSR/Streamline/version.dll不覆盖；模型和配置保留在项目私有目录。
回退脚本只移动本次新增的明确文件到结果目录，可恢复，不递归清理游戏目录。
本轮尚未改变游戏图形设置；若后续切换HDR，须恢复原先开启状态。

## 完整命令（PowerShell7，在项目根目录执行）

每个NEW目录必须不存在；独立GPU测试前必须正常退出游戏。每一规模先检查成功证据再扩大。

```powershell
.\scripts\build_native_reshade.ps1 -OutputDirectory results/reshade_build_NEW
.\scripts\run_native_reshade_probe.ps1 -StageDirectory results/reshade_build_NEW -OutputDirectory results/reshade_observe_NEW -Mode observe
.\scripts\run_native_reshade_probe.ps1 -StageDirectory results/reshade_build_NEW -OutputDirectory results/reshade_transport_NEW -Mode transport
.\scripts\run_native_reshade_probe.ps1 -StageDirectory results/reshade_build_NEW -OutputDirectory results/reshade_network_NEW -Mode network -Width 640 -Height 360 -Frames 1
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_native_reshade.py --run results/reshade_network_NEW
# 游戏为10-bit SDR时，取得同尺寸同格式的独立PASS，不能仅复用8-bit结果：
.\scripts\run_native_reshade_probe.ps1 -StageDirectory results/reshade_build_NEW -OutputDirectory results/reshade_network10_NEW -Mode network -Width 2342 -Height 1382 -Frames 1 -Format 24
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_native_reshade.py --run results/reshade_network10_NEW
```

先取得实际游戏尺寸的独立网络PASS，才部署；部署不会自动ARM：

```powershell
.\scripts\start_gowr_native_reshade.ps1 -ValidatedRun results/reshade_network_SAME_SIZE_PASS -OutputDirectory results/gowr_reshade_NEW -Frames 1
# 进入既有场景，确认SDR和实际尺寸后；不通过HDR/尺寸拒绝门槛则不得继续。
.\scripts\arm_gowr_native_reshade.ps1 -SessionDirectory results/gowr_reshade_NEW
# 正常退出游戏以后：
.\scripts\restore_gowr_native_reshade.ps1 -SessionDirectory results/gowr_reshade_NEW
```

SDR预览允许HUD一起处理。完成真实上屏之后，仍需连续帧稳定性、正式颜色/HDR、HUD保护及性能优化。
完整CPU回归776通过（72.98秒），不是游戏质量验收。
