# NVIDIA 执行粒度审计（2026-09-06）

## 范围和结论

本报告只读分析本地原始 NVIDIA 模块和已回收 RTX 5070 捕获；没有运行 GPU、游戏或 NVIDIA DLL。
确证范围是 **640×360、variant-02、Feature-18**，不能直接推广为原版 4K/HDR 的同一调度。
原版确实比当前 eager PyTorch 重建版融合得更粗：高分辨率 C32/C64 的一个完整 Swin block 是一次记录的 kernel launch，Pre 和 Head 各一次。
但不是全网一次 dispatch，也不是所有中间值始终留在寄存器：模块间有明确 global 存储和发布旗标。

## 可复现证据与命令

仓库根目录执行（只输出 metadata，不输出权重、参数指针或 PTX 正文）：

```powershell
C:\DATA\Tools\ANACONDA\python.exe scripts\audit_nvidia_execution_granularity.py
C:\DATA\Tools\ANACONDA\python.exe -m pytest -q tests\test_nvidia_execution_granularity.py
```

下面路径相对仓库；数据仍为本地私有，不随公共代码包分发。SHA-256：

| 证据 | SHA-256 |
|---|---|
| `results/20260831_002356_rtx5070_feature18_full_frame/frame_001_sequence.csv` | `a39d3e05ccb7023c47aa591c6bf59a20c14199185f97cbbb72488b6f237e2f35` |
| 同目录 `module_trace.jsonl` | `330151ae84584aabcf7179a89d02091ca6811009364b433613ed7272c7bf3a07` |
| `results/20260831_010100_all_runtime_modules/extraction_manifest.json` | `9935c7c6d3b3b367fa41f777296a6c81e46ebf4f6037dd997a27c72daa32432c` |
| 同目录 `module_00_000DF0E0.ptx`（下称 M0） | `9fa036fc715cb56c9138a1f931db1314fdd444ed30d1c31f66307a6babc7176a` |
| 同目录 `module_01_004A2220.ptx`（下称 M1） | `0ebd93ef1f85ac4b1981a2dc31e49c00273e107ebe397bb13b0a72bc97face99` |

这些是从原 DLL 解出的 PTX，不是 AMD lowered/修补 PTX。原 DLL 来源哈希由 manifest 记录为
`e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`。
脚本逐个核对 CSV 指定模块偏移与原模块 entry，避免仅按函数名猜测调度。
旧 trace 的 baseline 路径事件不是合法 JSON；脚本只严格解析 launch-call 记录，不跳过格式错误的 launch。

## 捕获的实际调用粒度

frame 1：**156 次 launch-chain API 调用、每次恰好 1 个 kernel、43 个不同函数、同一个 D3D12 command-list 指针**。
这不是通过 PTX 文件数或 Python 算子数估算。既有 R-21 报告还记录了五帧一致签名（见 `docs/RESULTS.md` R-21）；本次脚本重新审计 frame 1。

| 原始 slot 范围 | 内容 | 该捕获 launch 数 |
|---|---|---:|
| 0 | clear | 1 |
| 1 | Pre + 初始窗口模块 + 下采样 | 1 |
| 2–5 / 6–9 | Encoder C32 / C64，各 4 个 block | 4 / 4 |
| 10–15 / 16–23 | Encoder C128 / C256 | 6 / 8 |
| 24–55，56 | Encoder C512：8×4，末投影 | 32 + 1 |
| 57，58–97，98 | repack，8×5 ViT，repack | 1 + 40 + 1 |
| 99，100–131 | Decoder 入口，C512：8×4 | 1 + 32 |
| 132–139 / 140–145 | Decoder C256 / C128 | 8 / 6 |
| 146–149 / 150–153 | Decoder C64 / C32 | 4 / 4 |
| 154 / 155 | Head / 最终 copy | 1 / 1 |

例如 slot 1 grid=80×48、block=32×1；slot 3 grid=41×25、block=32×1；
slot 7 grid=21×13、block=32×2；slot 154 grid=81×49、block=32×1。
数千个 CTA 是同一个 launch 的工作网格，不是数千次 dispatch。
`tilesync/chained/wait` 的名称之外，PTX 还存在 release global 旗标和 barrier 指令，支持片区生产/消费依赖的判断。
**同一个 command-list 不证明各 kernel 同时执行、不证明硬件重叠比例，也不等于 API 调用耗时就是 GPU 耗时。**

## 精度、融合和存储证据

以下 MMA 数是 PTX 中静态指令站点数，不是执行次数、FLOPs 或实际 Tensor Core 利用率。
行范围是原模块 entry 范围（结束位置可能包含下一 entry 声明行）。

| 代表路径 | PTX 位置 | FP8 MMA 站点 | FP16 MMA 站点 | 静态 shared 字节 |
|---|---|---:|---:|---:|
| Pre slot 1 | M0:45630–61096 | 256 | 16 | 2048 |
| C32 chained slot 3 | M0:327304–340976 | 256 | 0 | 0 |
| C64 chained slot 7 | M1:92726–105582 | 304 | 0 | 4096 |
| C64 upsample slot 146 | M1:320707–335441 | 276 | 0 | 4096 |
| C32 outview slot 153 | M0:538198–552321 | 256 | 0 | 0 |
| Head slot 154 | M0:78255–94522 | 256 | 16 | 0 |

CSV 中这些 launch 的 dynamic_shared 都是 0。静态 shared 为 0 **不代表没有寄存器溢出或底层 driver 工作区**。
PTX `.reg` 是虚拟寄存器声明，不能当作每线程物理 VGPR 数；需要实际编译代码和 profiler 才能确认占用率/溢出。

- 原版主干明确使用 `mma...f16.e4m3.e4m3.f16`，也有成对 E4M3/FP16 转换。Pre/Head 存在 FP16×FP16→FP16 MMA。不能泛称原版全部 FP8 或 FP32 accumulation。
- 同一 entry 内同时存在矩阵计算、非线性/归一化及量化指令，且真实捕获只有这一次 launch，支持 block 级融合，而非仅靠 `fused` 名字推断。
- C32 chained 尾部有 4 个 `st.global.L1::no_allocate.b128` 站点，随后 `st.release.gpu.global.L1::no_allocate.s32`；C64 也有同类输出发布，以及 shared 交换。**跨 block 的 feature/同步状态确实有 global 边界**。
- `global` 指令不等于实际 DRAM miss：可能命中 L2；此处没有读取 DRAM bytes/cache counters，不能估算外存流量比例。
- Head 有纹理读和 2 个 `sust.p.2d.v4.b32.zero` 站点：原版将窗口计算与输出边界操作融合在 entry 内。它不证明当前调试 residual 合成已等于原版颜色/HDR 约定。
- packed lane/swizzle 与小片区布局可从地址计算、128-bit vector store、`ld.weak.global.ca/cg.v4.u32` 看出；当前已恢复的 `packed_a/packed_b/permute32` 在 `scripts/native_swin_torch.py`。它不是标准连续 NCHW feature。
- 本次没有完成每个中间寄存器的数据流分类；不能声称已证明 FFN/QKV/softmax 每个临时张量绝不写 global/local。

## 与当前 AMD 实现的差距

当前 `native_whole_frame.py` 有 71 个逻辑 stage，**71 不等于 GPU dispatch 数**。
Python 每个 window batch 内继续提交许多独立 torch 运算，量化函数又是 clamp→FP8→原 dtype。
`native_swin_torch.py:146` 起的 FFN、四次分段 contract、QKV、树形 FP16 norm、softmax、两段 attention 累加、project 都显式构造中间张量。
Pre 中颜色/噪声/输入投影，Head 中 gather+skip 融合、Swin、project、合成也分离。
ROCm 能运行这些操作，但不会自动将 eager 跨算子中间结果变成一个原版粒度 kernel。

主要候选差距（需要 profiler 定量，不预先声称百分比）：

1. 高频小算子和按 batch 的 host dispatch，而不是一个 block 对所有窗口发一个网格。
2. FP8 仅模拟量化边界后回到 FP16/FP32，未获得原版 FP8 MMA 的执行密度。
3. 重复 gather/scatter、pack/unpack、dtype 转换和中间写回。
4. 分段 FP32 矩阵累加为了保持候选结果，不能随意换成不同 accumulation 行为。
5. 之前逐 stage 有限值检查/等待混入的计时不能充当纯 GPU 推理基线。

## 首个融合目标与 correctness gate

先做算子级纯 GPU trace、完整 forward GPU event 时间和实际 kernel activity count，再动大结构。
首个低风险目标是 **Head/C32 复用的 cubic-SiLU + 明确 FP16 舍入 + E4M3 量化**，或 Q/K 的固定长度 norm+scale+quantize epilogue：
将多个点操作合并一个 HIP/DXIL dispatch，但保留每个已知 FP16 截断点，禁止默认 fast-math 重关联/意外 FMA。
该步不是“完整 Head 已融合”，收益以实际 trace 为准。

后续按 Head→Pre→高分辨率 C32/C64 发展为 gather/加载→矩阵→融合 epilogue→下一矩阵→输出的 bounded LDS/register pipeline；
先采用多 kernel 有界边界，再决定是否值得整体单 kernel，避免寄存器压力抹掉融合收益。
不可照搬 NVIDIA warp/MMA 到 AMD wave/WMMA 而忽略布局和舍入。

每个优化必须保留：

- 同输入、冻结权重/代码/配置哈希；原 eager 路径独立保留，可显式关闭新 kernel。
- 小算子穷举/边界值（含 E4M3 midpoint、饱和、负零、FP16 极值），再实际 captured/current feature 输入。
- 单模块逐元素误差和整帧输出 hash；首轮要求当前候选逐位相同，失败不提升默认。
- 1 帧→12 帧 gate、奇数尺寸和非零输入；新故障停止，不自动 GPU 重试。
- GPU event 总时间、kernel trace count、活动张量/allocator/设备显存分开记录；排除编译和暖机。

## 尚缺的证据

原版 4K/HDR 的同配置捕获、原生 SASS 性能计数器、每条 kernel 的纯 GPU 时间/DRAM/L2流量/物理寄存器/占用率；
当前 AMD 实际 kernel dispatch trace、完整无逐层同步纯 GPU forward 基线，以及融合前后同输入对照。
本静态审计不填造这些数字，也不将结果解释为 RTX/AMD 硬件极限速度比。
