# 低分辨率 NR → FSR 3.1：输入审计与基准契约

2026-09-06。此文是路线 B 的只读资源审计，不是已经获得 FSR 画质结果。
本轮未启动游戏、未运行 GPU、未修改游戏文件或原权重。

## 目前能做什么

- 同一明确标记的合成 4K 夹具派生 2560×1440、1920×1080 输入，测 NR 本身纯 GPU 时间、dispatch 和显存；派生 resize 只是输入生成，不是 FSR。
- 像素比例分别为 4/9 和 1/4，但窗口补齐、通道布局、全局注意力及固定调度成本不必严格按比例降低。报告实际 padded 尺寸、毫秒/百万像素，不将理论比例当实测。
- 已存在真实 AMD FSR 3.1.0 provider 的独立 D3D12 dispatch 工程基础，可以新增文件输入模式测试 NR→FSR 的格式/尺寸链路。
- **目前不能用现有单张 RGB 图给出真实 NR+FSR 时序、细节保留或相对原生 4K NR 的画质结论。**

## 本地接口和资料

`third_party/fidelityfx-api-1.1.3/ffx_upscale.h` 是固定 SDK commit
`54fbaafdc34716811751bea5032700e78f5a0f33` 的 MIT API 头，不是完整 SDK shader 工程。
`third_party/PROVENANCE.md` 记录来源。安装库通过历史独立验证枚举到 `3.1.0`、`4.1.1 *`、`2.3.2`；新测试仍须核对签名、哈希并显式选择 `3.1.0`，不能仅依据游戏菜单的 FSR3 标签推断后端。

本地 API 的 `ffxDispatchDescUpscale` 需要实际 current color、depth、motion、jitter、motion scale、render/upscale size、frame delta、preExposure、camera near/far/FOV、reset；context flags 决定 HDR、反转/无限深度、motion 分辨率及 jitter cancellation。曝光贴图和 reactive/transparency mask 的省略策略必须明确，不以伪造数据代替未知输入。

官方参考：[AMD FSR 3.1 Upscaler 文档](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/)。该公开页面当前介绍 3.1.4，不能证明本机 3.1.0 行为逐项相同；本机 ABI 与 provider 的实测仍是执行依据。

## 已有数据不等于合格序列

CPU 审计 `results/**/manifest.json`：89 个同时具有 collector `game_frame` 和 `resources` 字段的捕获，63 个目录组，**0 个真实游戏帧，0 个可进入序列语义审核的组**。更宽的 `game_frame` 字段统计为 90，额外记录没有资源数组，不能算可回放捕获。

最接近实际画面的资源：

- `results/20260906_single_color_gamecrop640_v1/input.rgba16f`：实际游戏图像裁剪的 640×360 色彩样本，SHA256 `89C150ABCDAD7BF72361AF189A08AA8BA932CD9598E516B2059116DB596C5055`。没有同帧深度、运动、jitter 和连续帧上下文。
- `results/20260906_gowr_native_reshade_one_v2/worker_frames.jsonl` 与同目录 PNG：一次实际 SDR pre-Present 网络写回证据。并非 pre-FSR 抖动输入，不能把已经呈现/抗锯齿的图再次送入 FSR 当作等价渲染输入。
- `results/20260905_104954_587_ffx_dispatch/` 等捕获：真实 AMD provider、**合成** color/depth/motion/exposure，验证上传/dispatch/读回，不是实际场景质量数据。

检查器只审核字段、资源文件长度、紧凑行大小、尺寸和序列元数据。不会把 `game_frame=true` 自述当作真实语义证明，也不检测贴图内容是否伪造；因此 `quality_ready` 固定为 false，后续必须另证资源来源、哈希、格式语义、颜色/HUD 与 NR 改变后的运动对齐。

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/assess_nr_fsr_inputs.py results --min-frames 32
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q tests/test_assess_nr_fsr_inputs.py
```

无候选序列返回码 2，是正确的阻断结果，不是 GPU 失败。6 个 CPU 测试通过。

## 独立工程联通最短路径

已有验证命令（会运行 GPU，不属于本轮只读审计）：

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
.\scripts\run_ffx_dispatch_probe.ps1 -CaptureValidation
```

底层已存在 CLI：`ffx_dispatch_probe <verified AMD DLL> "3.1.0" <new result directory> capture`。
`tools/ffx_observer/dispatch_probe.cpp` 当前没有任意 raw 输入/尺寸参数：Color() 写死合成图，常规 320×180→640×360，depth 模式另用固定尺寸。每帧 reset=true，某些模式每帧重建 context，因此该工具目前不能直接回放 NR 输出或验收连续历史。

可另建严格命名的 synthetic engineering 模式：加载有哈希的 NR RGBA16F 输出、显式尺寸、已知静态平面 depth=0.5、motion=0、jitter=0、reset=true；这些是合成场景定义而不是冒充游戏数据。接真实 3.1.0 provider→4K，只验证格式、完整写入、finite、确定性与纯 GPU 时间，**不能用于细节/拖影/NR 效果保留判断**。不要改现有旧基准定义。

## 最短合法真实采集

1. 新增并独立验证持续 context 的 replay 和有界 collector：第一轮 1 帧，再 12 帧，再 32 帧；预分配预算、完成围栏后落盘。已有 collector 最大帧数32、总字节上限512MiB，完整1440p/4K序列可能超过预算，需要有界分批回收，不能简单增加帧数越过资源上限。
2. 正常启动游戏到既有存档，先 SDR、关闭帧生成/动态分辨率；实际进入 2560×1440→4K 和 1920×1080→4K 两种渲染模式，确认真正 provider 为3.1。采集 FSR 前的原始资源及原始 FSR 输出，保持原渲染。
3. 采集静止、平移、遮挡、人物细纹理和切镜，记录完整 flags/exposure/jitter/reset 及 frame id，HUD 应保持下游合成。至少一个32连续帧段先验证契约，后续按24×32数据集计划扩展。
4. 用户只需提供可用场景；此前已授权正常退出。自动操作无法可靠进入设置/控制场景时再请用户协助；本轮无需用户立即打开游戏，也不启动高负载游戏试验。

## NR 在 FSR 前的语义风险和验收

当前候选 NR 接受单色帧，带固定坐标噪声，最后是 SDR residual clamp；它并未证明保留 jitter 相位、线性预曝光颜色或几何位置。若 NR 改变边缘/细节但沿用原 motion/depth，FSR 历史可能与新增内容错配。不能仅将最终截图缩小后送入 NR 再填入原 motion 当作合法链路。

三模式对照必须固定场景、时间、曝光/色彩转换与输出尺寸：4K NR、1440p NR+真实FSR3.1、1080p NR+真实FSR3.1；另保留无 NR 的同分辨率 FSR 对照以区分 SR 与 NR 损失。逐项报告 NR、GPU打包/同步、FSR、全链纯GPU区间与设备/allocator显存，不把离线 CPU 上传/读回混入纯GPU指标或称作游戏FPS。

质量先只报告相对当前4K候选，不冒充RTX教师：固定显示变换PSNR/SSIM、最差帧、细线/面部ROI、NR相对无NR的改变量；连续帧比较遮挡掩码下的重投影误差、闪烁、亮度跳变、切镜重置。截图静态指标不能替代连续历史。每个尺寸保存输入/权重/source/provider hashes、seed、padding、精度、shader和profile；优化前后同输入 exact gate，若数值策略变更则必须另审明确阈值，不暗中放宽。
