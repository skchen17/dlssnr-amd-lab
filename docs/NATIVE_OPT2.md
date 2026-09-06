# 原生执行优化第二轮（2026-09-06）

结论：原权重与候选网络数学关系不变，新增可选`native_opt2`执行调度。
不是DLSS5画质验收、实时游戏或4K60；本轮没有游戏启动、训练或TDR设置修改。

## 最终配置

- Swin/输入/输出窗口批量384（原游戏执行默认12，上一轮离线96）。
- 全局注意力query_chunk=128，key_chunk仍128；每条查询仍遍历全部键，
  不改成局部注意力，不改变键归约顺序，不物化完整高分辨率N×N矩阵。
- 紧凑特征布局：4×4单元内采用16×C小置换表，加张量重排，替代H×W×C的全局int64索引。
  这只是存储布局，不是把图片分块独立推理；坐标、补齐、裁剪、注意力上下文不变。
- 上一轮的65536行输入投影修正继续保留，权重、FP8边界、累加与FP16存储公式没有新变化。
- 冻结权重转换缓存作为单独实验保留，收益较小，**未启用在native_opt2**。
  仅eval/no-grad/冻结参数可使用缓存；训练、参数替换/版本变化、设备迁移均绕过或失效。
  不缓存学习后的派生权重，也不把缓存加入模型state_dict。

`native_inference_schedule.select_schedule`提供不可变命名配置；
`ResidentNativeFrame(..., optimization_profile='native_opt2')`及GPU工作进程
`--optimization-profile native_opt2`可显式选择。默认仍为`baseline12`，未静默替换游戏执行。
错误配置拒绝，不降级到CPU，也不自动降采样。

## 逐项实验

所有下列目录位于results/，前缀均为20260906_；128/640/2342逐级检查通过后才扩大。
每项A/B先运行两个基线帧，再运行候选帧。同尺寸同输入/seed，最终输出逐字节一致。

| 实验目录后缀 | 2342×1382热态整链 | 说明 |
|---|---:|---|
|opt2_cache96_2342_v1|3540.85→3478.15ms|只缓存输出头权重转换，单热样本收益约2%，不足以认定主要瓶颈|
|opt2_all192_2342_v1|3544.07→2235.17ms|批量96→192|
|opt2_all384_2342_v1|3662.49→1640.35ms|批量96→384|
|opt2_query128_2342_v1|1712.64→1366.98ms|保持批量384，仅查询批次32→128|
|opt2_compact_2342_v1|1308.82→1281.24ms|启用紧凑布局，速度小幅变化，主要收益在活跃张量内存|

以上每次仅一个热态样本，存在时钟/顺序偏差，不将不同冷/热条件相除宣传加速。
计时为逐层主机提交、GPU完成等待及有限值检查之和，不是纯GPU事件时间或游戏FPS。

## 重复与尺寸检查

`opt2_final_641_v1`：641×361合成渐变，补齐后计算并裁回原尺寸，基线/候选四次输出一致。

`opt2_final_640_twelve_v1`：历史实际游戏640×360裁剪；基线96两次后候选12次。
候选12次输出均与基线及历史通过哈希一致，热态11次中位数286.67ms。
这不是12个真实连续游戏帧，不用于时序画质验收。

`opt2_final_2342_twelve_v1`：独立宿主彩条输入，不是RTX教师或游戏画面。
基线96热态3573.48ms；候选12次中热态11次中位数1292.68ms，范围1280.51–1313.78ms，
同批对照约2.76倍。所有输出与修正版基线哈希
`2E6FFA412DBB9AFB4AD3FFE353BC9C8EF4C265C54DF50CE58B103B300E5C82FA`一致。
活跃张量峰值由3,266,070,528降至1,874,142,720字节，约下降43%。
**这不是整个进程或游戏的总显存占用**：A/B进程的缓存分配器仍可能保留较大reserved内存。
各实验结束allocated/reserved均为0，无超时，正常返回0，运行期间源文件快照不变。

## 接入常驻跨进程执行器

`opt2_process128_one_v1`、`opt2_process640_one_v1`后，
`opt2_process640_twelve_v1`通过12帧交替输入验证：

- D3D12生产者不初始化HIP；单独ROCm工作进程执行原权重网络。
- GPU共享缓冲区/围栏传输，管道只传元数据；CPU图像上传和读回只是离线夹具/审计端点。
- 第一帧精确匹配历史640候选输出；两种交替输入得到各自稳定且不同的输出。
- D3D12消费者逐帧精确匹配工作进程输出；工作进程按准备确认回报`native_opt2`，调用方校验。
- 第一次工作进程网络区间2239.86ms；后11帧中位数199.38ms，范围193.84–207.26ms。
  该区间是网络提交/等待/有限值检查，不包含整个D3D12往返、游戏帧或启动时间；
  不能和上面的逐层等待计时直接混为一个口径。
- 双方正常退出，双方释放后allocated/reserved为0。没有扩大为游戏连续写回。

新跨进程证明保存全部native_*.py来源快照（而不是只保存四个入口脚本）。
游戏仍须取得对应代码与配置的ReShade独立证明后再部署；旧证明不自动升级。
真实游戏方块伪影、SDR/HDR/HUD与时序质量尚未复验，不宣称本轮优化修复画质。

## 最终回归与安全审计

完整 CPU 回归：`python -m pytest -q`，812 项通过，耗时 53.86 秒。
`results/20260906_native_opt2_final_audit.json` 确认冻结基线与游戏文件哈希未变，
游戏未运行，最近一小时未发现所检查的显示驱动重置或异常关机事件，事件日志读取成功。
本轮没有训练、修改 TDR 或部署游戏插件；这不替代后续实际游戏稳定性验收。

## 完整运行命令

PowerShell7、项目根目录；NEW目录必须不存在。游戏必须关闭，各GPU命令逐项审查，
不是并行脚本，也不是失败后的自动重试。单次事件等待10秒，主机监督不是GPU取消。

```powershell
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q tests/test_native_weight_cache.py tests/test_native_compact_layout.py tests/test_native_inference_schedule.py
.\scripts\audit_rocm_implementation.ps1 -OutputPath results/opt2_pre_audit_NEW.json
# 以下最终组合分别在128、640、641、2342进行；每一级确认通过后再扩大。
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 128 --reference-batch 96 --batch 384 --middle-batch 384 --query-chunk 128 --compact-layout --output results/opt2_128_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 640 --reference-batch 96 --batch 384 --middle-batch 384 --query-chunk 128 --compact-layout --output results/opt2_640_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 641 --reference-batch 96 --batch 384 --middle-batch 384 --query-chunk 128 --compact-layout --output results/opt2_641_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 2342 --reference-batch 96 --batch 384 --middle-batch 384 --query-chunk 128 --compact-layout --candidate-repeats 12 --output results/opt2_2342_twelve_NEW
# 冻结权重缓存单独测试，不与native_opt2混淆：
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 128 --reference-batch 96 --batch 96 --middle-batch 96 --cache-head-weights --output results/cache_only128_NEW
# 跨进程验证仍按128单帧→640单帧→640的12帧执行：
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --dll results/20260906_reshade_native_stage_v6/native_frame_bridge.dll --network --width 128 --iterations 1 --optimization-profile native_opt2 --output results/opt2_bridge128_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --dll results/20260906_reshade_native_stage_v6/native_frame_bridge.dll --network --width 640 --iterations 1 --optimization-profile native_opt2 --output results/opt2_bridge640_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --dll results/20260906_reshade_native_stage_v6/native_frame_bridge.dll --network --width 640 --iterations 12 --optimization-profile native_opt2 --output results/opt2_bridge640_twelve_NEW
```

下一步：减少剩余小算子的调度与显存往返、评估专用融合内核，再复验新版游戏路径。
当前仍距离4K60很远，不以“能计算”或“静态重复通过”关闭实时里程碑。
