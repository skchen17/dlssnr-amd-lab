# 1080p 原生时序 NR 引擎 v3：第一批实施结果

更新时间：2026-09-07。

## 当前结论

33 ms 里程碑**尚未完成**。当前最后一份可采信的完整 1080p 严格路径仍是
`220.061 ms / 4,653 kernel + 59 memcpy`；本轮没有运行 GPU，因此没有把编译成功或
缓存生成误报成性能收益。

本轮完成的是把下一阶段从 Python capture 原型推进到可编译、可拒绝错误输入的原生
运行时契约：ABI v3、按尺寸固定 arena、分段哈希模型包、gfx1201 派生权重缓存、外部
D3D12 fence 提交接口、原始时序采集结构门和统一最终验收门。C32/C64/C128/C256、独立
C512 split 路径以及 8 个流式全局注意力 ViT core，共 68 个 block core 已加入 C++ 内建
recorder；中央 C512↔ViT bottleneck transition 也已编译接入。尚未执行 GPU 门；Pre、
其余 encoder/decoder 尺度边界和 Head 未迁移前，
`deployment_ready` 必须保持 false。

## 已实现

### ABI v3 与 C++ 所有权

`tools/native_nr_plan/nr_plan.h/.cpp` 新增：

- `STRICT_FP16` / `APPROX_FP8` 精度配置；
- render/output/valid rect、SDR/HDR 格式描述；
- current color、history/next history、motion、depth、exposure、controls、jitter、
  motion scale、frame ID、resource generation 和 reset 的帧绑定；
- `nrPlanLoadModelPackage`、`nrPlanPrepareShape`、`nrPlanSubmitV3`、
  `nrPlanGetPerformanceStats`；
- `nrPlanSubmitExternalV3`：在同一 HIP stream 上等待导入的 D3D12 fence，提交 graph，
  再发出完成 fence，不进行 CPU 等待；纯网络计时不包含 fence wait/signal；
- 只有 ABI v3 recorder、C++ 自有 weight/workspace/source graph/executable、非外部
  allocation、无外部 recorder context 且 `complete_native_topology=true` 时，所有权门才允许
  `deployment_ready=true`。部分 stage 图即使完全由 C++ 持有也不能通过该门。

当前未核验的 temporal contract 会阻止非 reset/带 temporal flags 的提交，避免把零 motion、
当前色复制或自定义滤波器冒充原分支。

最新编译产物：`results/20260907_native_nr_plan_build_v29/`，其中 `build.json` 明确记录
`gpu_executed=false`。DLL 导出包含 ABI v3、模型包、性能统计及外部 fence 接口。

### C32–C512 stage-specific resident-FP8 block

`tools/native_stage_fp8/stage_fp8.hip` 将 C64/C128/C256 block core 固定为六次 whole-grid dispatch：

1. grouped FFN expand/activation/contract；
2. cross-head mix，同时发布 FP16 residual 与 E4M3 resident view；
3. QKV projection + norm，Q/K/V 直接写 E4M3；
4. query-tile QK + softmax + PV；
5. projection + residual，输出 E4M3 logical-window；
6. 按该 block 的真实 origin 直接 scatter/quantize 到 packed-image resident layout。

FFN 从 packed-image resident 按 origin 直接取数，不再假设所有 shifted block 都有相同窗口数；
1080p C64 的四种 origin 分别对应 2160/2196/2220/2257 windows。

C32 使用独立的 8-wave FFN 映射：8 个 wave 并行产生 128 hidden channel，前两个 wave
完成 contract/residual；首次全展开候选为 192 VGPR 且有 scratch，已拒绝。重写后静态资源
为 50 VGPR、4 KiB LDS、0 scratch，每个 C32 block 共 5 dispatch。C512 按其四记录结构
独立实现 preprojection、grouped 64→256→64、FFN residual projection，再接 QKV/norm、
attention、output projection 和 scatter，共 7 dispatch；其中 grouped kernel 为 72 VGPR/
8 KiB LDS，attention 为 90 VGPR/3 KiB LDS，所有矩阵 kernel 均生成 gfx1201 FP8 WMMA，
全部 0 scratch。

C32–C256 的 44 个 block core 为理论 256 个 kernel，C512 的 16 个 core 为理论 112 个；
新增 ViT 每块固定 5 个 kernel，8 块共 40 个。合计覆盖 68/71 个 record 的主 core、理论
408 个 kernel 节点；加上中央 encoder-final/decoder-input 两个 kernel，以及八个尺度切换
kernel 后为理论 418 个 kernel＋5 个 D2D skip memcpy 节点，不引用 PyTorch capture pool。
模型/arena/topology 已逐 offset 与 SHA-256 交叉核对。静态 gfx1201 ISA 审计覆盖30个
stage kernel，其中25个矩阵 kernel 均生成FP8 WMMA且0 scratch，5个scatter为12 VGPR/
0 LDS/scratch；C64–C256 FFN group 为100 VGPR/4 KiB LDS，attention 为90 VGPR/
3 KiB LDS，QKV norm 为39–40 VGPR/3 KiB LDS，mix/project 为37/62–63 VGPR、0 LDS。这些是
静态资源数据，不是运行时 occupancy 或速度；GPU correctness/performance 未过门前状态保持
`PARTIAL_69_OF_71_NATIVE_RECORDS_WITH_SCALE_TRANSITIONS_NOT_RUNTIME_ACCEPTED`。八个尺度切换
已串入 recorder，但尚未通过 GPU correctness；静态 ISA/拓扑覆盖不是 GPU 数值或性能通过。

### ViT 流式全局注意力

`tools/native_vit_fp8/vit_fp8.hip` 将 1024-channel bottleneck 固定为每 block 五段：FFN
expand、FFN contract/residual、QKV/norm、streamed global attention、output
projection/residual。1080p 的 padded bottleneck 为 640 token；attention 以 16-key tile
循环，在 320 B LDS 中复用指数块，不物化 N×N score/probability tensor。初始化时权重已是
gfx1201 E4M3 layout，跨 kernel activation 为一字节 resident E4M3。

独立编译/ISA 审计位于 `results/20260907_native_vit_fp8_build_v1/`：expand/contract/QKV/
attention/project 均生成 FP8 WMMA，VGPR 分别为 62/62/42/88/85，LDS 为 0/0/3072/320/0 B，
全部 0 scratch。这仍是静态证据；global-attention 数值、吞吐和 640-token 长循环尚未过 GPU 门。

`tools/native_bottleneck_fp8/bottleneck_fp8.hip` 取消中央边界的 logical HWC 和
repeat_interleave：block30 后直接按 FP16 pooling 顺序读取 resident C512、保存 skip、做
512→1024 WMMA 并写 padded 32×20 ViT resident；block39 直接做 1024→512 WMMA、2× spatial
expansion、skip-scale add 并写 60×36 packed C512 resident。该实现已随 v29 编译，尚未做 GPU
correctness/performance gate。独立 device-only 编译显示两侧矩阵 kernel 都为 64 VGPR、
0 LDS、0 scratch，并生成 gfx1201 FP8 WMMA；这同样只是静态证据。

### 模型包 v2 与 gfx1201 派生缓存

`scripts/native_model_package_v2.py` 使用固定 C 可读头和 128-byte 权重段表。C++ 加载器会：

1. 检查原模型 SHA-256；
2. 检查 strict / gfx1201 FP8 整段 SHA-256；
3. 逐记录检查 profile、连续覆盖、名称唯一性、范围及 SHA-256；
4. 全部通过后才分配并上传被选择的权重段。

`scripts/build_gfx1201_stage_cache.py` 已在 CPU 上对本地完整 71-block 模型生成独立缓存：

- 1,014 个运行时记录，666 份去重存储；
- 148,300,672 bytes；
- 288 个 row-major E4M3 矩阵记录；
- 70 个 column-major QKV E4M3 记录；
- 非 E4M3 的 Pre input projection、position bias、scale 等保持 FP16；
- 静态索引安全压缩为 int32；
- 额外派生、逐值校验的 FFN inverse permutation，供 native kernel 直接消费；
- 每个尝试编码为 E4M3 的矩阵都先做 FP16 round-trip exact 检查，不能仅按“二维张量”
  猜测精度语义。

缓存位于本地私有 `local_models/gfx1201_stage_cache_v3/`，不进入仓库，也尚未被完整 native
graph 消费，所以状态是 `DERIVED_LAYOUT_CACHE_NOT_RUNTIME_ACCEPTED`。

### 固定 arena 与时序门

`scripts/native_nr_arena.py` 现在按 precision profile 生成 ABI v3 arena：

| 1080p profile | workspace | resident item |
|---|---:|---:|
| strict FP16 | 435,196,416 B | 2 B |
| approximate FP8 | 442,196,480 B | 1 B |

FP8 profile 的 stage ping/pong/skip 使用一字节 resident 表示，并增加可跨所有 C32–C512
block 复用的 grouped/post/Q/K/V/value 六个、按最坏 shifted C32 window 尺寸规划的 scratch；
FP16 residual/norm 和 FP32 reduction
workspace 被显式保留，不能借缩小 arena 偷改数学边界。由于显式保存四份并发 Q/K/V/value、
FFN scratch 及 2,621,440 B ViT hidden resident buffer，完整 approximate arena 略大于 strict
arena；持久 stage tensor 本身仍为一字节，且总量约422 MiB，远低于5 GB allocator门。

`scripts/audit_temporal_teacher_sequences.py` 定义原始时序序列 schema v2，强制真实 GPU
resource identity、current/previous/next history、motion、depth、exposure、jitter、reset、
Pre/Post boundary 和教师输出。结构通过仍只返回
`TEMPORAL_CAPTURE_STRUCTURE_VALID_NOT_SEMANTICS_ACCEPTED`，不会自动启用时序运行。

`scripts/audit_native_engine_v3.py` 统一检查 33 ms、≤512 kernel、0 PyTorch/library hot-path
节点、C++ 所有权、显存、24×32 时序数据、PSNR/SSIM、时序退化、D3D12 链路开销及
SDR/HDR 游戏稳定性，缺一项即失败。

### 其余 Encoder/Decoder 尺度切换

`tools/native_transition_fp8/transition_fp8.hip` 与 `NRApproxScaleTransitionDesc` 已覆盖
4/8/14/22 后的 downsample 和48/56/62/66 前的 upsample。Encoder直接读取上一core的
FP16 logical-window结果保持pooling舍入，再做FP8 WMMA projection并写下一尺度resident；
Decoder用单kernel完成projection、2×expand、skip-scale/add与packed resident写入。静态
ISA为Encoder 52 VGPR、Decoder 60 VGPR，均0 LDS/0 scratch并生成gfx1201 FP8 WMMA。
详见 `NATIVE_SCALE_TRANSITION_FP8_20260907.md`。

## 本轮验证

- NRPlan v29、resident stage、ViT、bottleneck和scale transitions：gfx1201 DLL/ISA/selftest
  编译通过；transition构建本身未执行GPU；
- C256 record15执行了一次8×8单窗口非RGP门：生命周期/释放通过，但NRMSE 3.038113、
  最大误差1.015625，数值拒绝且未重试；
- 当前 transition/ABI/拓扑定向测试：`8 passed`；全量收集因隔离环境缺少 Pillow/pefile
  在 collection 阶段停止，并非本轮断言失败；
- `git diff --check`：无 whitespace error；
- RGP counter 安全锁未解除，未运行 4K，未触发游戏或 GPU 实验。

## 尚未实现的关键路径

1. 68 个 block 的主 core、中央 transition（39）和其余八个尺度切换已静态绑定；Pre（0）
   与 Head（70）仍缺失。当前已验证可运行的完整 graph 仍是旧 PyTorch capture 图。
2. stage 的 gfx1201 cache/arena offset 与 resident-FP8 链已生成，但尚未通过
   1-window→representative-grid→full-stage GPU correctness/performance gate。
3. RTX 5070 的 24×32 原始时序数据和 bounded-difference 语义结论尚未返回，因此运行时
   正确地拒绝启用 temporal branch。
4. 没有新的 1080p GPU A/B、画质、100 帧离线或 SDR/HDR 游戏测试；33 ms 与游戏稳定
   输出均不能宣称完成。

下一步先对C256的FFN、QKV、attention、projection/scatter逐段定位首个差异；当前安全记录
已在一次门后重新置为`authorizes_next_gpu_gate=false`。只有再次完成复核和单次授权才运行
下一门。之后按C256→C128→C64→C32/C512/ViT/bottleneck→scale transitions→Pre→Head推进；
不会重启4K或RGP counter测试。

## 复现命令

```powershell
& .\scripts\build_native_nrplan_v3.ps1 `
  -OutputDirectory 'C:\ABSOLUTE\NEW\nr_plan_build'

& .\.venv-rocm\Scripts\python.exe -m pytest -q `
  tests/test_native_cpp_nr_plan_v3.py `
  tests/test_native_model_package_v2.py `
  tests/test_native_nr_arena.py `
  tests/test_nr_plan_ownership.py `
  tests/test_audit_temporal_teacher_sequences.py `
  tests/test_audit_native_engine_v3.py `
  tests/test_build_gfx1201_stage_cache.py

& .\.venv-rocm\Scripts\python.exe scripts\build_gfx1201_stage_cache.py `
  --source local_models\native_single_color_v1 `
  --output local_models\gfx1201_stage_cache_NEW
```
