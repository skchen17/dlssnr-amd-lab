# 原生整帧 GPU 纹理桥：独立验证（2026-09-06）

**独立纹理闭环已通过；尚未接入游戏，不能称为真实上屏或4K60。**

最新跨进程及实际游戏观察续进见 `NATIVE_GAME_PROCESS_BRIDGE.md`：独立两进程12次循环已通过，
但游戏列表内部仍有后续绘制，实际原生写回尚未接通。下文同进程记录保留为历史验证。

暂不以RTX画质对齐为前置条件。本次实际推进的是原权重模型包、常驻执行器及
D3D12纹理→共享线性缓冲区→HIP/PyTorch整帧网络→共享缓冲区→D3D12输出纹理。
没有把旧CPU像素IPC、分块、pre-Present版本提升为正式原生实现。

## 新实现

- `scripts/native_model_package.py`：离线导出私有、非pickle模型包；运行时只读固定包文件，
  校验描述、整体权重及每条记录哈希，不读取旧捕获计划、PTX、DLL或RTX中间值。
  152条记录包含144条活跃神经参数与8条ABI占位，未包括无历史分支不使用的时序blend标量。
  原始arena文件保持不变，无训练。哈希绑定不是数学关系正确性的证明。
- `scripts/native_frame_runtime.py`：初始化、按尺寸准备、GPU帧提交、资源代次重置及释放。
  权重常驻，输出完成及下游消费完成分开处理；拒绝旧帧、错误代次、错误设备、HDR及超预算尺寸。
  未完成或设备状态不明时禁止重用/释放，不把超时理解为GPU取消。
- `tools/native_frame_bridge/native_frame_bridge.cpp`：独立原生测试宿主，D3D12共享堆/缓冲区及
  输入/输出两个共享围栏；使用当前PyTorch已加载的HIP运行库，不另载另一个运行时。
  按LUID核对同一显卡。GPU二维复制处理行距，D3D12虚拟地址与HIP导入指针分别使用。
- `scripts/native_bridge_harness.py` / `validate_native_frame_bridge.py`：受监督的离线验证。
  首尾CPU上传/读回仅用于生成测试纹理、核对字节；网络与共享缓冲区间没有CPU像素搬运或CPU神经回退。
  **当前宿主仍在同一个独立进程中，不是游戏跨进程桥。**
- `scripts/assess_native_bridge.py`：逐帧日志与输出文件哈希验证；只接受独立纹理循环，
  不接受游戏上屏、HDR、动态游戏序列或实时帧率。

模型包位于 `local_models/native_single_color_v1`（已受local_models忽略规则保护，不发布）：

- model.json SHA256：`AAE2A5425199829CFC95773405C94C5A943F8D9AF55423CF129E85792E2A2EE3`
- weights.bin SHA256：`F2997E9ECB162811D11FFA432646C6067BC9930B810F0CD4C01F842845EA9B8D`

## 实测

| 测试 | 结果 |
| --- | --- |
| 128×129共享纹理/水平翻转 | GPU输入及输出消费者字节正确，正常退出 |
| 128×128整帧网络 | 纹理闭环正确，与旧同候选输出完全一致 |
| 640×360真实图像裁剪整帧网络 | 纹理闭环正确，与此前100次静态ROCm输出完全一致 |
| 641×129非整齐行距 | 有效行5128字节、共享行距5376字节，传入/传出正确 |
| 641×129交替输入12次 | 同资源/递增围栏复用正常，两个输入对应两个稳定输出 |
| 640×360整帧网络交替输入12次 | 每帧纹理回写与当帧GPU输出一致，没有上一帧输出冒充 |

最新版桥DLL：`results/20260906_native_frame_bridge_build_v3/native_frame_bridge.dll`，
SHA256 `637AD947BF72592F9D636047F57C363326DE5924D2B61F34D8B0CB5269A4AC9E`。

主要证据：

- `results/20260906_native_bridge_network640_v3_twelve/manifest.json`
- `results/20260906_native_bridge_transport641_v3_twelve/manifest.json`
- `results/20260906_native_bridge_assessment.json`：`NATIVE_STANDALONE_TEXTURE_LOOP_PASS`
- `results/20260906_native_bridge_final_audit.json`：219基线及4个游戏文件不变，游戏未启动，
  查询期间无目标系统设备/电源事件；验证进程均已退出。

640网络12次进程共26.14秒；首次整帧提交/等待3148.93ms，后11次中位数1706.24ms。
这是端到端网络提交及等待的主机时间，不包括游戏渲染，不是4K性能；也不能直接与旧逐层计时求加速比。
热态每帧结束分配393,430,016字节不变，模型/工作区释放后分配及保留为零。
这是PyTorch分配统计，不是整个D3D12/驱动总显存统计。

交替输入为同一张历史游戏裁剪和其水平翻转，seed固定为0；不是12个真实连续游戏帧。
无历史分支、旧诊断颜色合成及分辨率预算保持原限制。未接受任何RTX质量或HDR语义。

### 报告沿革

v1编译因PowerShell参数数组表达式导致链接对象路径错误而失败；未运行GPU。
修正后v2编译及单帧GPU验证通过；v3新增消费完成后复用同资源的入口，逐级验证通过。
失败目录和生成的中间对象保留，没有覆盖旧实验或游戏文件。

v3十二帧报告顶层input_sha256最初记录首帧输入，output_sha256记录末帧输出；
逐帧日志始终一一正确，评估器以日志及实际输出文件为准，不能用这两个顶层字段当作同帧对照。
后续报告代码已修正并增加baseline_input_sha256、baseline_comparison_frame_id；
旧报告没有追改，未为了此元数据修正重跑GPU。

最终完整CPU回归750项通过（51.96秒），新增Python脚本语法检查通过，diff空白检查通过。
这包含资源代次/失败禁止重用、模型包损坏与不支持契约拒绝、纹理证据门槛等测试；
CPU单元测试不替代实际设备故障或游戏验收。

## 距离游戏实际接入的剩余工作

1. 将独立测试宿主拆为游戏侧纹理导出/回写与ROCm工作进程：仅传资源句柄、尺寸、颜色信息、
   代次及围栏值；实现明确的句柄复制、进程生命周期及设备丢失协议。当前同进程成功不等于此项完成。
2. 找到并验证HUD之前实际使用网络结果的消费者位置。旧生产命令列表末尾曾出现写回正确却不上屏，
   不复用该位置宣称成功。需要真实可见标记及截图/序列证据。
3. 恢复正式颜色/曝光合成，尤其当前输入存在>1线性值，旧clamp不可当作HDR。
4. 在资源预算内逐级扩大整帧尺寸与优化性能；默认网络预算仍拒绝4K，不缩图、不硬拼接。
   当前1.7秒量级网络耗时距离4K60目标仍很远，不能将此桥直接标为实时可用。
5. 再进行1/12/100帧游戏测试，最终SDR/HDR各1000帧及恢复审计。

## 命令

项目根目录执行。模型包保持私有，NEW输出目录/文件必须不存在；已有模型包不要重复导出覆盖。
GPU命令逐项检查正常退出、释放统计后再执行下一规模，异常立即停止，不自动重试或修改TDR。

```powershell
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/native_model_package.py --output local_models/native_single_color_NEW
powershell -NoProfile -File scripts/build_native_frame_bridge.ps1 -OutputDirectory results/native_bridge_build_NEW
powershell -NoProfile -File scripts/audit_rocm_implementation.ps1 -OutputPath results/native_bridge_preflight_NEW.json
.\.venv-rocm\Scripts\python.exe scripts/validate_native_frame_bridge.py --mode transport --width 128 --dll results/native_bridge_build_NEW/native_frame_bridge.dll --output results/transport128_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_frame_bridge.py --mode network --width 128 --dll results/native_bridge_build_NEW/native_frame_bridge.dll --output results/network128_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_frame_bridge.py --mode network --width 640 --dll results/native_bridge_build_NEW/native_frame_bridge.dll --output results/network640_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_frame_bridge.py --mode transport --width 641 --iterations 12 --dll results/native_bridge_build_NEW/native_frame_bridge.dll --output results/transport641_repeat_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_frame_bridge.py --mode network --width 640 --iterations 12 --dll results/native_bridge_build_NEW/native_frame_bridge.dll --output results/network640_repeat_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/assess_native_bridge.py --manifest results/network640_repeat_NEW/manifest.json --output results/bridge_assessment_NEW.json
powershell -NoProfile -File scripts/audit_rocm_implementation.ps1 -OutputPath results/native_bridge_audit_NEW.json
```

验证器当前使用已固定哈希的 `local_models/native_single_color_v1`，NEW导出命令仅验证可复现导出，
不会自动将新目录或不同模型设为默认。没有提供游戏启动命令，因为正式游戏接入尚未完成。
