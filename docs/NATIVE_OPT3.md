# Native opt3：在 4–6 GB 预算内继续优化

2026-09-06。仅为原权重单色候选网络的离线及跨进程执行优化；
不表示完整 DLSS5 色彩/时序语义、RTX 画质或游戏实时验收通过。
本轮没有启动游戏、训练、修改 TDR 或部署游戏插件。

## 采用的配置

新增显式 `native_opt3`：窗口批量 768、全局注意力 query_chunk=1024、
key_chunk=128、compact layout。baseline12 与 native_opt2 保留，默认仍为 baseline12。
原权重、FP8 转换边界、归约方向与顺序、原始整帧坐标均不变。
全局注意力仍逐块遍历所有 key，并非局部窗口替代；没有完整 N×N 分数矩阵。
查询工作区上限由默认 1,048,576 个元素显式扩大为 4,194,304 个元素，
并保留默认小工作区策略。查询块扩大不改变 key 的累加顺序。

`ResidentNativeFrame` 对 opt3 设置 **5,000,000,000 字节张量分配器进程级上限**，
不会提高已有的更严格限制。这是隔离工作进程策略，不在 close 时撤销上限。
HIP、BLAS、D3D12 外部资源不受该上限直接管理，不能将其宣称为整个进程的硬上限。
离线探针另在每个已完成阶段采样设备级占用，超过 6,000,000,000 字节停止；
采样仍不是连续的总显存峰值监控，也不能保证所有游戏/其他进程负载下都小于 6 GB。
不缩图、不回退独立图片拼接或 CPU 神经计算；不将主机超时当作 GPU 取消。

## 实测与未采用方案

以下是同进程、同输入 A/B，先两次旧配置，再候选；首个候选单列冷态，
12 次候选中的后 11 次取热态中位数。逐层提交、等待和有限值检查计时，
不等同纯 GPU 时间、常驻执行器区间或游戏 FPS。

| 2342×1382 候选 | 热态耗时 | 分配器 reserved 峰值 | 结论 |
|---|---:|---:|---|
| 384 / query128，旧 opt2 | 配对参考 1.284 秒 | 2.219 GB | 保留 |
| 768 / query128 | 单次热态 1.203 秒 | 2.219 GB | 有改善 |
| 1536 / query256 | 单次热态 1.286 秒 | 2.227 GB | 未采用 |
| 768 / query128 / 输出头坐标缓存 | 1.195 秒 | 3.727 GB | 未采用 |
| 768 / query256 | 1.153 秒 | 2.219 GB | 通过 |
| 768 / query1024，opt3 | **1.135 秒** | **2.219 GB** | 采用为显式实验配置 |

opt3 热态范围 1.1284–1.1707 秒；相对本次配对旧配置耗时下降约 11.6%，
相对上一轮 1.2927 秒中位数下降约 12.2%（跨实验环境波动需区别）。
活跃张量峰值 1,879,385,600 字节；阶段采样设备占用最大 2,623,012,864 字节。
14 次输出完全相同，匹配上一轮修正后候选 SHA256：
`2E6FFA412DBB9AFB4AD3FFE353BC9C8EF4C265C54DF50CE58B103B300E5C82FA`。
该 2342 输入是合成条纹，不是本轮游戏捕获或 NVIDIA 教师。

坐标缓存作为实验实现保留但不加入任何命名配置：仅存输出头两组 int32 索引和 bool 掩码，
2342 常驻缓存 1,105,612,800 字节，单独限制为 1.3 GB。
缓存不包含画面/特征/输出，尺寸、设备、训练模式和 state_dict 加载使缓存失效；
梯度模式绕过缓存，超预算显式拒绝，不静默跳层。
CPU 非零融合输入及变动输入测试避免零权重掩盖索引错误；GPU 12 次输出精确。
它虽然减少局部索引计算，整帧收益不足以选择额外约 1.1 GB 常驻开销。
冻结输出头权重缓存同样没有加入 opt3。

## 4K 整帧容量测试

输入为明确的 CPU 渐变/棋盘合成夹具，3840×2160，内部补齐为 3840×2176。
由 `make_native_opt3_fixture.py` 保存真实输入文件、配方、输入哈希及生成器哈希。
没有使用或伪造教师、运动或曝光数据。源输入 SHA256：
`55A6660C30ADB44E87AFD17FE239DC2F6648EEE4128E7D0C14A11358E984DA19`。

初次 A/B `results/20260906_opt3_4k_ab_v1`：
旧 opt2 热态 3062.93 ms，opt3 冷态 2534.16 ms、热态 2521.20 ms。
四个完整帧逐位一致，输出 SHA256：
`46F357EC1DA510A7CB7BC6D141E8A83DBA4DAD24179D72DC18BD150CD573056F`。
活跃张量峰值 4,018,241,536 字节，reserved 峰值 4,687,134,720 字节，
阶段采样设备占用最大 5,122,818,048 字节。资源释放后 allocated/reserved 都为 0。
这仅证明该输入/尺寸/环境可在预算内完成计算，不代表真实游戏画质、HDR、时序或 4K60。

后续 `results/20260906_opt3_4k_twelve_v1` 的 12 次候选重复全部通过，
后 11 次热态中位数 **2515.27 ms**，范围 2512.01–2525.26 ms，
配对旧 opt2 热态 3051.77 ms，耗时降低约 **17.6%**（速度约 1.21 倍）。
14 个输出 SHA256 完全相同，峰值内存及设备采样上限与初次 A/B 一致，
释放后 allocated/reserved 为零，源文件未变化，无超时，正常退出。

## 常驻执行器与检查

`opt3_process128_one_v1` → `opt3_process640_one_v1` → `opt3_process640_twelve_v1`
均位于 `results/20260906_` 前缀下，跨进程 D3D12↔ROCm GPU 缓冲区及围栏验证通过。
640 的 12 帧交替输入输出稳定且互不相同，第一帧匹配历史候选输出，
D3D12 消费者逐帧匹配网络输出；没有把旧输出冒充新帧。
热态工作进程网络区间中位数 **188.20 ms**，不包含完整桥接往返或游戏帧时间。
工作进程 allocated/reserved 峰值约 0.582/0.640 GB，并记录 5 GB 配置上限。
双方正常退出、资源释放为零、源文件运行期间未变化；生产者没有初始化 HIP。

全量 CPU 回归 **815 项通过，53.25 秒**（4K 夹具生成器及 case 枚举随后补充）。
640/641×361 的 opt3 离线 A/B 通过；640 保持历史输出哈希。
旧游戏部署证明不能自动升级为新代码/配置的证明。本轮不改变游戏默认设置。
最终审计 `results/20260906_native_opt3_final_audit.json`：冻结基线和游戏文件哈希未变，
游戏未运行，所检查的最近一小时显示驱动重置/异常关机事件为空，日志读取成功。
没有遗留本轮 GPU 工作进程。

## 完整复现命令

PowerShell 7，项目根目录；每个 NEW 输出目录必须不存在。游戏须关闭。
GPU 命令逐项审查后执行，不能并行或失败自动重试；权重与输入保持本地私有。

```powershell
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q
.\scripts\audit_rocm_implementation.ps1 -OutputPath results/opt3_pre_NEW.json
# 依次 128 → 640 → 非整齐尺寸 → 2342
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 128 --reference-batch 384 --reference-query-chunk 128 --reference-compact-layout --batch 768 --middle-batch 768 --query-chunk 1024 --compact-layout --output results/opt3_128_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 640 --reference-batch 384 --reference-query-chunk 128 --reference-compact-layout --batch 768 --middle-batch 768 --query-chunk 1024 --compact-layout --output results/opt3_640_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 641 --reference-batch 384 --reference-query-chunk 128 --reference-compact-layout --batch 768 --middle-batch 768 --query-chunk 1024 --compact-layout --output results/opt3_641_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 2342 --reference-batch 384 --reference-query-chunk 128 --reference-compact-layout --batch 768 --middle-batch 768 --query-chunk 1024 --compact-layout --candidate-repeats 12 --output results/opt3_2342_NEW
# 4K 夹具首次生成命令；目录已有时不重建、不覆盖
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/make_native_opt3_fixture.py --output results/20260906_opt3_fixture4k_v1
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 3840 --reference-batch 384 --reference-query-chunk 128 --reference-compact-layout --batch 768 --middle-batch 768 --query-chunk 1024 --compact-layout --output results/opt3_4k_NEW
# 上一项通过后才能扩大 12 次候选
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 3840 --reference-batch 384 --reference-query-chunk 128 --reference-compact-layout --batch 768 --middle-batch 768 --query-chunk 1024 --compact-layout --candidate-repeats 12 --output results/opt3_4k_twelve_NEW
# 常驻 GPU 互操作，仍须逐级验证
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --dll results/20260906_reshade_native_stage_v6/native_frame_bridge.dll --network --width 128 --iterations 1 --optimization-profile native_opt3 --output results/opt3_process128_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --dll results/20260906_reshade_native_stage_v6/native_frame_bridge.dll --network --width 640 --iterations 1 --optimization-profile native_opt3 --output results/opt3_process640_NEW
.\.venv-rocm\Scripts\python.exe scripts/validate_native_gpu_process.py --dll results/20260906_reshade_native_stage_v6/native_frame_bridge.dll --network --width 640 --iterations 12 --optimization-profile native_opt3 --output results/opt3_process640_twelve_NEW
.\scripts\audit_rocm_implementation.ps1 -OutputPath results/opt3_post_NEW.json
```

后续重点应转向融合归一化/量化/布局小算子和减少临时显存往返，
而不是无限扩大批量或以旧帧复用模拟实时。新优化均需同输入完整输出及
正常退出证明，再更新独立游戏证明；4K60、真实上屏、HDR/HUD/时序仍未完成。
