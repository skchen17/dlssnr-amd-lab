# 真实游戏色彩图像：完整网络离线实验（E-170）

本轮证明的是：RX 9070 XT 使用原始权重和重建的 156-slot 转译图，对真实图像产生可重复的网络响应。不是原生 D3D12 全图完成、RTX 同画质通过或游戏实时可用。

## 已保存的结果

根目录：`results/20260905_real_color_probe/`。

| 实验 | 结果 | 范围 |
| --- | --- | --- |
| `upstream_real_post/` | 两次完整图 PASS，输出相同 | 640x360 单帧裁剪，初始激活来自旧捕获 |
| `zero_initial_arena/` | 两次 PASS；与上项中间区、最终输出逐字节相同 | 此输入不依赖旧激活初值；不是时序状态证明 |
| `zero_preblock_control/execution/` | 两次 PASS，输出发生变化 | 与上项同为零初始区，同一合成底图，仅网络入口颜色置零 |
| `native_head/` | 执行 PASS；与转译输出不等价 | 原生输出头接 AMD 生成的中间值；不是完整原生图 |

原始帧来自 `results/20260905_163007_931_gowr_capture_session/output_network/boundary_before.raw`，2342x1317 RGBA16F。裁剪位置 `[851,478,640,360]`；逐字节保留，不缩放、不做色调映射。输入 SHA256：

`89C150ABCDAD7BF72361AF189A08AA8BA932CD9598E516B2059116DB596C5055`

模型 SHA256：`A5513B1845C98A486985ED04F38E66A1854CCE33C2ABA3A505866028BD4EE3E5`。

真实颜色输出 SHA256：`E92278B73ACC06B4C78C5E01370726B5583869542718EEE1EF5EE0178DBB02E6`。

`analysis/input.png` 与 `analysis/output.png` 使用相同 Reinhard + gamma 预览变换；实际推理读写的是 raw。

- RGB 输入最大 5.60546875，有 43,146 个分量大于 1；当前输出头裁剪到 [0,1]，输入色彩/曝光契约未确认。
- 真实输入相对“网络入口置零、保留底图”控制组，最终 RGB MAE 为 0.0498637、最大 0.4873047，656,711 个 RGB 分量改变；29,773,824 字节中间区有 26,735,093 字节改变。
- 原生头对转译头 RGB MAE 0.000443146、最大 0.0883789。仅内部后端比较，不是 RTX 质量误差；不能仅凭低均值放行。
- 首轮两次完整图的逐 slot 启动加同步总和为 428.2 / 406.8 ms。排除模块加载和检查点文件 IO，但不是纯 GPU 事件时间或最终游戏 FPS。
- 原生头两次 GPU 时间为 38.9 / 36.6 ms，目前没有原生改写已提速的证据。

旧计划第一次运行被哈希检查阻止，保存在 `upstream/execution.json`。slot 154 的旧路径内容已变，且将 14 处纹理采样替换为零。本轮从已记录的 `07_surface.ptx` / `07_surface.json` 构建独立计划，保留真实纹理采样并验证 51 个模块哈希；没有覆盖旧计划或删除失败记录。

## 运行指令（当前电脑）

先正常退出游戏；本脚本不会启动、注入或强制结束游戏。需要已有 `.tools/zluda-v7-preview.3/zluda/nvcuda.dll`、原始模型、PTX/参数资产和本机 RX 9070 XT。这些指令引用本仓库本地资产，不是可直接复制到云主机的独立包。

在 PowerShell 执行。每轮使用新目录，不覆盖之前的结果：

```powershell
Set-Location 'C:\DATA\Python_File\Qoder_workspace\dlssnr-amd-lab'
$py = 'C:\DATA\Tools\ANACONDA\python.exe'
$probe = Join-Path $PWD ('results\real_color_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$cuda = '.tools\zluda-v7-preview.3\zluda\nvcuda.dll'

& $py scripts/prepare_full_graph_game_color.py `
  --session results/20260905_163007_931_gowr_capture_session --output "$probe/input"

& $py scripts/make_full_graph_color_plan.py `
  --base results/20260831_234000_full_graph_integrated_plan/plan_fine_n0_jointulp_slots2to6_square_postblock.json `
  --color "$probe/input/input_rgba16f.raw" `
  --post-ptx results/20260904_880000_postblock_mul_add_fusion_first32_candidate/07_surface.ptx `
  --post-audit results/20260904_880000_postblock_mul_add_fusion_first32_candidate/07_surface.json `
  --output "$probe/plan.json"

& $py scripts/run_full_graph_integrated.py --plan "$probe/plan.json" `
  --nvcuda $cuda --output "$probe/upstream_real_post" --export-pre-head

& $py scripts/analyze_full_graph_color.py --execution "$probe/upstream_real_post" `
  --output "$probe/analysis"

& $py scripts/run_full_graph_integrated.py --plan "$probe/plan.json" `
  --nvcuda $cuda --output "$probe/zero_initial_arena" --export-pre-head --activation-init zero

& $py scripts/prepare_full_graph_color_control.py --plan "$probe/plan.json" `
  --output "$probe/zero_preblock_control"

& $py scripts/run_full_graph_integrated.py --plan "$probe/zero_preblock_control/plan.json" `
  --nvcuda $cuda --output "$probe/zero_preblock_control/execution" `
  --export-pre-head --activation-init zero

New-Item -ItemType Directory -Path "$probe/native_head" | Out-Null
& .\build\output_head_infer_d3d12.exe `
  "$probe/upstream_real_post/run1/pre_head_activation.raw" `
  local_models/decoded_310_8/model_arena.raw "$probe/input/input_rgba16f.raw" `
  "$probe/native_head/output.raw" "$probe/native_head/manifest.json"

& $py scripts/analyze_full_graph_color_controls.py `
  --captured "$probe/upstream_real_post" --zero-arena "$probe/zero_initial_arena" `
  --zero-color "$probe/zero_preblock_control/execution" `
  --native "$probe/native_head/output.raw" --output "$probe/controls_summary.json"
```

每条命令失败时停止本轮，保留其结果目录；不要修改哈希绕过验证。默认每次运行重复两遍。不要用旧零输入 RTX 检查点分析器评价这组真实颜色输出；脚本已添加拒绝检查。

## 后续门槛

1. 使用同一真实输入获得可信 RTX 对照，核对 NR 输入色彩/曝光、采样及条件参数；当前 FFX 输出并不自动等价于 NR 原始输入。
2. 加入运动矢量、跨帧状态及镜头切换实验，以序列质量验收，不以单帧“有变化”验收。
3. 测量常驻 GPU 图耗时并优化热点家族。目前重复 Swin-8h 链式家族 12 次调用合计约 74.6 ms，是候选热点；这些仍是启动加同步计时，需 GPU 事件进一步确认。
4. 验证真正动态几何或分块上下文、接缝及状态一致性。现有任意尺寸 identity 拼接通过不代表网络已支持任意分辨率。
5. 完成上述质量、性能和资源生命周期门槛后，再做新游戏进程内的有限帧接入验证。当前不部署完整图到游戏。
