# 原生游戏接入续进：跨进程 GPU 与列表内消费者（2026-09-06）

用户要求：**先完成游戏接入，再做速度优化**。本轮没有改变网络算法、训练权重或做速度优化。
状态：跨进程纹理/网络循环已通过；真实游戏仅进行了命令元数据观察，**原生网络尚未在游戏上屏**。

## 已实现与通过

- `native_gpu_protocol.py`：有界、仅元数据协议，拒绝像素负载、重复JSON字段、错误尺寸/行距、
  旧资源代次、旧帧号、未完成消费就覆盖/释放。
- `native_gpu_worker.py`：独立进程持有原权重ROCm完整单颜色分支，接收共享堆及双向围栏，
  只在GPU上传入、运行、传出。诊断读回为显式测试选项，不经IPC传输图像。
- `native_frame_bridge.cpp`：精确目标PID的句柄复制、工作进程导入、LUID一致性核对、输入/输出
  独立围栏及消费确认。纯D3D12生产端不初始化HIP，模型与HIP由工作进程持有。
- `validate_native_gpu_process.py`：两个真实进程，先单帧、后12次交替输入，核对双端图像哈希、
  正常退出及资源释放。CPU上传/读回只用作独立宿主的输入和消费者验证，不是正式游戏像素路径。

最新版桥DLL：`results/20260906_native_frame_bridge_build_v6/native_frame_bridge.dll`，SHA256：
`AF838F4BD41116A0C89F155CC8609F796AE2FD6011792E13345F1FE7829B3017`。

主要证据：

- `results/20260906_native_process_transport128_v2/manifest.json`
- `results/20260906_native_process_network128_v1/manifest.json`
- `results/20260906_native_process_network640_v1/manifest.json`
- `results/20260906_native_process_pure_d3d128_v2/manifest.json`
- `results/20260906_native_process_pure_d3d640_one/manifest.json`
- `results/20260906_native_process_pure_d3d640_twelve/manifest.json`

最终640×360十二次测试正常退出，总进程时间28.187秒；首帧与此前同候选ROCm结果逐字节相同。
两个交替输入分别对应两个稳定输出，每次纹理消费者读回与工作进程当帧输出一致。
两端分配统计释放后为零，生产端 `producer_hip_initialized=false`。
这不是12个真实连续游戏帧，也不证明跨进程设备丢失、HDR或任意尺寸验收。
当前跨进程描述符仍限制在已审查的小尺寸范围，不会直接接受游戏的2342×1317纹理。

### 两个前置拒绝及处理

1. `20260906_native_process_transport128_v1` 在握手时因PID不匹配退出，尚未复制共享句柄或提交网络。
   纯CPU复现确认Windows虚拟环境启动器PID与实际Python子进程PID不同。
   修正为通过Windows进程表核对真实直接子进程，而不是移除PID验证。失败报告保留。
2. `20260906_native_process_pure_d3d128_v1` 在设备选择时拒绝：DXGI枚举出两个同名RX9070XT，
   LUID分别为95796、153401（本次运行期值，不硬编码）。CIM只有一块对应PCI设备，
   但不能据此把两个DXGI条目当成同一共享资源设备。改为使用已认证工作进程的HIP LUID选择D3D12设备。
   此拒绝发生在创建测试纹理/提交网络之前，没有自动重试未修改的批次。

没有设备重置或GPU挂起的已确认新证据，不将以上CPU身份/选择错误描述为驱动崩溃。

## 实际游戏观察

`capture_session.cpp` 新增输出边界之后、命令列表Close之前的Draw/DrawIndexed/Dispatch/
ExecuteIndirect/CopyTexture/CopyResource计数，保留原调用及参数，不插入GPU计算、等待或像素写入。
重置与观察窗口代次隔离。shader描述符读取没有解析，零copy计数不代表没有shader消费者。

观察器阶段：`results/20260906_boundary_tail_stage_v1`。
FSR3/FSR4各10种已有模式的对照全部退出0；观察开关图像相同，捕获/状态/边界等分析均通过。
独立正例中能记录输出纹理的后续读取复制，证明新增计数实际触发。
旧PowerShell5无法调用已有构建器的IsPathFullyQualified；改在PowerShell7运行，没有修改工具链安装。

实际游戏会话：`results/20260906_025141_372_gowr_capture_session`，PID17088。
通过computer-use检查到游戏进入原有阿特柔斯与安格尔伯达场景，没有新建或覆盖存档、改变图形设置。
观察结果 `results/20260906_game_boundary_tail_scene.json`：

| 观察窗口 | 边界样本 | 每条列表在边界后的绘制 |
| --- | --- | --- |
| 加载界面（window1） | 4 | Draw 1次，DrawIndexed 5次 |
| 实际场景（window2） | 4 | Draw 2次，DrawIndexed 33次 |

这直接证明**输出边界后仍有GPU绘制命令在同一列表内**，因此列表结束后的写回位置没有得到安全性支持。
结合之前静态边界内预览能上屏、列表末尾动态写回未上屏的证据，下一步必须解决列表内部的生产/消费切分。
但这些计数**没有证明**35次绘制中谁读取FFX输出、谁是HUD，不能直接指定某一Draw前为正确位置。

观察已停止（窗口1/2均明确发送stop）。原生网络写回未启用，游戏仍使用原始输出。
退出准备期间窗口工具检测到人工操作，已重新检查窗口并停止发送按键，保留当前游戏会话，未强制结束。

## 本轮最终回归与审计

- CPU回归：771项通过，70.17秒；新增协议状态/身份验证及边界分析测试均在其中。
- `git diff --check` 通过；仅有既存文件的换行转换警告，没有顺手改动它们。
- `results/20260906_native_game_process_final_audit.json`：219项冻结基线和4项游戏文件哈希不变，
  最近一小时指定系统事件为空；事件日志可读。此记录不是没有任何潜在驱动问题的保证。
- 最终进程检查仅发现游戏PID17088，没有遗留Python网络工作进程或独立GPU测试进程。
- 游戏运行中，观察已停止；没有开始训练、改TDR设置、删除实验目录或部署网络写回。
- 审计仍为 `native_graph_complete=false`：候选单颜色链可执行，不等于原网络语义和游戏里程碑验收。

## 下一步接入实现，不先转做速度优化

1. 恢复输出边界之后绘制的描述符/根参数/资源依赖，定位首个实际消费者与HUD合成。
2. 实现并验证保留PSO、根签名/参数、描述符堆、视口/剪裁及其他必要状态的命令列表内部拆分。
   未支持的渲染通道、查询/间接执行或未知状态必须在改写前拒绝；不能裸Close/Reset冒险丢失游戏状态。
3. 在拆分后的生产段完成、消费段尚未提交的间隙，接入已验证的跨进程资源/围栏协议。
   不使用未来围栏阻塞整个队列等候尚未启动的网络，不用上一帧结果冒充当帧。
4. 完成实际尺寸的资源预算审查、颜色/曝光契约和短游戏写回，再验证可见输出及停止恢复。
   这仍不是4K60或DLSS5画质验收。速度优化保持在游戏接入之后。

## 完整运行命令

在项目根目录执行，使用PowerShell7。所有NEW目录/报告必须不存在。
独立GPU测试前正常退出游戏；分级检查通过后才运行下一规模，异常不自动重试。
权重仍使用固定私有模型包，不传服务器或公共包。

```powershell
.\scripts\build_native_frame_bridge.ps1 -OutputDirectory results/process_bridge_build_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --width 128 --dll results/process_bridge_build_NEW/native_frame_bridge.dll --output results/process_transport_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --network --width 128 --dll results/process_bridge_build_NEW/native_frame_bridge.dll --output results/process_network128_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --network --width 640 --dll results/process_bridge_build_NEW/native_frame_bridge.dll --output results/process_network640_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --network --width 640 --iterations 12 --dll results/process_bridge_build_NEW/native_frame_bridge.dll --output results/process_network640_repeat_NEW
```

游戏元数据观察（**不是网络写回启动命令**，不得在已有游戏会话中重新注入）：

```powershell
.\scripts\build_all.ps1 -Only ffx_capture_session,ffx_dispatch_probe,ffx_session_control,ffx_live_attach -FreshOutputDirectory C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\boundary_stage_NEW
.\scripts\run_ffx_staged_session.ps1 -StageDirectory results/boundary_stage_NEW
.\scripts\start_gowr_boundary_observation.ps1 -StageDirectory results/boundary_stage_NEW
```

启动脚本返回会话目录与PID；加载既有场景后再使用准确PID/会话DLL路径：

```powershell
& .\results\boundary_stage_NEW\ffx_session_control.exe <实际PID> C:\DATA\GAME\GODOFWAR\GoWR.exe C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\boundary_stage_NEW\ffx_capture_session.dll observe
& .\results\boundary_stage_NEW\ffx_session_control.exe <实际PID> C:\DATA\GAME\GODOFWAR\GoWR.exe C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab\results\boundary_stage_NEW\ffx_capture_session.dll stop
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/analyze_boundary_tail.py --log <实际会话目录>/session.jsonl --output results/tail_analysis_NEW.json
```

带尖括号的PID/路径必须替换为启动脚本实际返回值；不是可直接粘贴的历史会话身份。
