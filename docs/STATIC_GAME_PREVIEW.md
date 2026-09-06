# 游戏内局部静帧预览

这是预计算像素展示，不是游戏内实时执行 DLSS 网络。

中心面板包含完整 640x360 图像、20 像素标签条和 2 像素边框；总占用 644x384 像素。面板以外不写入。图像区域不缩放、不做色调映射，保留原始 RGBA16F 数据；仍经过游戏后续显示处理，所以显示色彩正确性尚未验证。

输入、输出均来自 E-170 同一裁剪。网络输出已由 RX 9070 XT 使用原始模型在独立进程内预计算，游戏中每帧只复制所选静帧。它不会随镜头或场景更新，不能作为当前游戏帧的重建结果。

## 本轮操作命令

PowerShell 工作目录为 `C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab`。

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$previewStage = 'results/20260905_static_preview_stage_v2'
$previewSession = 'results/20260905_175035_151_gowr_capture_session'

# 展示预计算网络输出，标签 NETWORK OUTPUT
.\scripts\run_gowr_static_preview.ps1 -StageDirectory $previewStage -SessionRun $previewSession -Action network

# 同位置切换为原始输入静帧，标签 ORIGINAL INPUT
.\scripts\run_gowr_static_preview.ps1 -StageDirectory $previewStage -SessionRun $previewSession -Action input

# 停止覆盖，后续画面恢复由游戏正常绘制
.\scripts\run_gowr_static_preview.ps1 -StageDirectory $previewStage -SessionRun $previewSession -Action stop

# 查看状态
.\scripts\run_gowr_static_preview.ps1 -StageDirectory $previewStage -SessionRun $previewSession -Action status
```

每次开启或切换后最多 60 秒自动停止写入；最多记录 18,000 次区域复制。已经排队的少量帧可能仍带覆盖。切换时不会改写 GPU 正在读取的上传缓冲；输入和输出使用两个独立只读缓冲。

命令只适用于这次已记录的进程和启动时间。退出游戏后不可复用该会话；需要新启动、重新通过来源验证的会话。没有复制 DLL 到游戏目录，也不修改存档。关闭仅停止之后的写入，不释放 GPU 可能仍引用的上传资源；约 4 MB 上传资源保留到进程退出。

## 独立验证

`results/20260905_static_preview_stage_v2/preview_gate/gate.json` 记录测试前后相同的 DLL、测试程序、控制程序和像素文件哈希。

- WARP 与 AMD 各 9 组：未开启、连续网络输出、原始输入、关闭、关闭后的下一帧、再次开启、再次关闭。
- 全幅逐字节比较通过，包含面板外区域不变、输入/输出切换精确和下一帧恢复。
- 两个后端 D3D12 错误和警告均为 0。
- 同一 DLL 的原有 FSR3/FSR4 回归通过；Python 回归 502 项通过。

新逻辑只在已验证的 FFX 输出完整状态转换后记录 CopyTextureRegion，并恢复 readable 状态。不在游戏线程运行网络、等待 GPU、重置私有分配器或更改根签名。内部 ResourceBarrier 使用已修复的重入绕过规则。

本轮实际游戏中已确认网络静帧显示、原始输入切换和手动关闭恢复。会话目录保存 `preview_network.png`、`preview_input.png`、`preview_stopped.png` 与 `manual_stop_audit.json`。首轮记录 6,682 次区域复制，停止后 `enabled=false`、`failed=false`，进程正常响应；未发现本轮新增显示重置/异常重启事件。此证据仍然不等于实时网络或画质验收。

第二轮验证了 60 秒自动关闭：记录 5,853 次复制后停止，中心面板消失。`preview_timeout.png` 与 `automatic_timeout_audit.json` 保存显示和状态证据。测试结束时预览已关闭，游戏仍停留在当前可玩场景。
