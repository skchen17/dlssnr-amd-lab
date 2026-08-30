# FINAL_REPORT.md — Round 1 最终交付报告

日期: 2026-08-30 · 运行目录: `results/20260830_170305/` · 机器: RX 9070 XT (gfx1201, 0x1002/0x7550)

本报告对应规范第二十一节。所有结论均可由文内给出的命令、日志路径或可重复实验支持；
凡依赖专有文件 / NVIDIA 硬件的条目一律标注 `MISSING_PREREQUISITE`，不做任何虚构。

---

## 规范第二十一节 14 项清单

### 1. 当前 S-level

**S2 ACHIEVED**（D3D12↔HIP 互操作 PASS-A）。S0/S1/S2 达成；S3+ 全部被硬性前置条件阻塞。
见 `docs/STATUS.md`。

### 2. 实际完成了哪些实验

| # | 实验 | 结果 | 登记 |
|---|---|---|---|
| Phase 0 | 环境全量采集（collect_environment.ps1 / env_probe） | PASS | R-0 |
| Phase 1 | hip_probe RDNA4 计算基线（A–E 全过；F 因编译器缺陷记 UNVERIFIED） | PASS | R-1 |
| Phase 2 | d3d12_hip_interop 共享资源门（PASS-A；PASS-B 调查结论：无公开 API） | PASS | R-2 |
| Phase 3 | binary_probe 工具自测（dxgi.dll / amdhip64_7.dll）；目标 DLL 分析 MISSING_PREREQUISITE | PASS（工具） | R-3 |
| Phase 4 | module_trace.dll + nvapi64.dll 垫片构建与冒烟（smoke_trace.py SELFTEST OK） | PASS（工具） | E-3 |
| Phase 5 | nr_host 骨架构建、BLOCKED 状态机、确定性 reference package | PASS（骨架） | R-4 |
| Phase 6/7 | CASE A–D 决策树 + PTX go/no-go 门链（数据槽全部 PENDING_PREREQUISITE） | 框架交付 | docs/CALLGRAPH.md |

### 3. 每个实验的命令

```
Phase 0:  powershell -ExecutionPolicy Bypass -File scripts\collect_environment.ps1
Phase 1:  scripts\build_all.ps1 -Only hip_probe
          build\hip_probe.exe --json results\20260830_170305\hip_probe.json
Phase 2:  scripts\build_all.ps1 -Only d3d12_hip_interop
          build\d3d12_hip_interop.exe --json results\20260830_170305\d3d12_hip_interop.json
Phase 3:  scripts\build_all.ps1 -Only binary_probe
          build\binary_probe.exe <dll> --json <manifest.json>     # 自测对象: dxgi.dll, amdhip64_7.dll
Phase 4:  scripts\build_all.ps1 -Only nvapi_trace,module_trace
          C:\Users\20426\Documents\ComfyUI\.venv\Scripts\python.exe scripts\smoke_trace.py
Phase 5:  scripts\build_all.ps1 -Only nr_host
          build\nr_host.exe --frames 1 --width 512 --height 512 --trace --json results\20260830_170305\nr_host.json
          （确定性复核: scripts\_determinism_check.ps1）
```

### 4. 每个实验的原始 log 路径

均在 `results/20260830_170305/`：
`environment.json` · `hip_probe.json` · `wmma_diag.log` · `d3d12_hip_interop.json` ·
`interop_build.log` · `selftest_dxgi.json` · `selftest_amdhip.json` ·
`nvapi_trace_selftest.jsonl` · `module_trace_selftest.log` ·
`nr_host_build.log` · `nr_host_run.log` · `nr_host.json` · `nr_host_run2.json`
另：`results/20260830_180553_reference_package/`（color/depth/motion bin + manifest.json）。

### 5. binary analysis 结果

**目标文件 `nvngx_dlssnr.dll`：MISSING_PREREQUISITE（BLOCKERS B-1）。**
分析工具已交付并自测通过：PE/imports/delay-imports、fatbin 容器（magic B1 43 62 46）、
CUBIN ELF（e_machine=190）、PTX 文本标记、熵运行（weights 候选）、SHA-256，
输出机器可读 `binary_manifest.json`（只存 metadata，不存二进制）。
18 个规范问题的覆盖矩阵见 `docs/BINARY_ANALYSIS.md`。拿到 `DLSSNR_DLL_PATH` 后
一条命令即可产出全部静态答案。

### 6. DLSSNR 完整 call graph

**未知（按规范不得预判）。** `docs/CALLGRAPH.md` 已给出：
假设链路（nr_host → NGX → nvngx_dlssnr → NVAPI/CUDA → GPU binary → driver）+
CASE A/B/C/D 判定树（Q1–Q4，证据源排序）+ 替换点表。所有判定槽位
PENDING_PREREQUISITE，等 trace 数据。

### 7. GPU binary 类型

**未知 — MISSING_PREREQUISITE。** 工具侧已能区分 fatbin / CUBIN(SASS) / PTX / 混合，
并可报告 sm target、kernel name、weights 位置。无目标文件，不猜。

### 8. PTX 是否存在

**未知 — 这正是最大的 go/no-go gate（GATE-0..4，见 CALLGRAPH.md §4），
当前状态：PTX path = PENDING_PREREQUISITE。** 门链已完全可执行：
binary_probe 找 fatbin → 查 PTX 标记 → 核对工具链支持 → 单 kernel 编译
→ kernel_lab 数值对比。若无 PTX：记 `PTX path = BLOCKED`，转入 Track B
（重建，明确标注为重新实现而非执行泄露 kernel）。

### 9. AMD D3D12/HIP interop 状态

**PROVEN（S2）。** 证据 `results/20260830_170305/d3d12_hip_interop.json`：
- `hipImportExternalMemory`（D3D12Heap=4 / D3D12Resource=5）导入共享 buffer：OK
- `hipExternalSemaphoreHandleTypeD3D12Fence` 导入共享 fence：OK
- T3 D3D12→HIP、T4 HIP→D3D12 双向传输内容校验 mismatch=0
- T5 纹理导入 OK；T6 调查：无把 HIP kernel 塞进 D3D12 command list 的公开 API
  （ZLUDA FAQ 说法未在本机证实），替换设计按"独立 HIP queue + 外部 fence"建模
完整配方与陷阱（CreateSharedHandle 归属、CommandQueue::Wait、hipEvent 计时异常）见 `docs/INTEROP.md`。

### 10. ZLUDA 能直接复用哪些组件

未实测（无 CUDA 负载可驱动）。基于 RESEARCH.md 核实（2026-08-30）：
- ZLUDA 活跃（"back to the roots"），提供 nvcuda.dll 替换 → 若 trace 判定 CASE B/C
  则为第一复用对象；已知 RDNA4+Windows 兼容问题需逐一复现（S-B2）。
- ZLUDA 的 PTX 前端是 GATE-2/3 的关键组件；本机未装、未编译，属第二轮工作。
- OptiScaler FakeNVAPI 仅在 CASE A 且需骗过 vendor check 之外的真实调用翻译时参考；
  本项目不依赖"骗过检测"作为任何证据。

### 11. 还缺哪些 CUDA/NVAPI semantics

按依赖排序：
1. `nvngx_dlssnr.dll` 本体 → 才能知道调用的是哪套（CASE 判定）。
2. 若 NVAPI Cubin 系：`NvAPI_D3D12_CreateCuModule/EnumFunctionsInModule/CreateCuFunction/
   LaunchCuKernelChain(Ex)` 等 12+ 函数的参数块布局（nvapi_ids.h 已备社区 ID，带 ? 待实证）。
3. 若 CUDA Driver API：cuModuleLoad/cuLaunchKernel 族 → ZLUDA 覆盖度需对实际负载验证。
4. NGX 官方参数块（feature 18 的 Create/Evaluate 参数）→ 当前为重建签名（B-4）。
5. WMMA/MMA 快路径：ROCm 7.2 LLVM 的 `wmma.f32.16x16x16.f16` ISel 缺陷（S-B4），
   备选：f16 累加变体 / inline asm / 新 LLVM。

### 12. 第一个真正 blocker 是什么

**B-1：缺少用户合法提供的 `nvngx_dlssnr.dll`**（连带 `nvngx_dlss.dll` = B-2）。
S3–S10 全部卡在它后面。本机侧（编译链、HIP、D3D12 互操作、trace 垫片、host 骨架、
决策框架）已推进到无专有文件时的最远处，没有任何可再绕过的空间。
次要：B-6（git 身份未设，milestone commits 排队）。

### 13. 当前 "AMD 真 DLSSNR" 可行性判断

**结论：路径技术上成立，关键前提全部就位，但尚未有任何 DLSSNR 计算在 AMD 上执行过。**
- 有利：S1（FP32/FP16/FP8 数值正确）、S2（零拷贝共享 + fence 同步双向验证）两块最难的
  地基已用内容级证据夯实；替换点（CALLGRAPH §3）中互操作链已证明，其余待 trace 定位。
- 风险：(a) payload 若纯 SASS/Blackwell-only 且无 PTX，则转 Track B 重建（成本高，
  且按规范不算"执行泄露 kernel"）；(b) WMMA 编译器缺陷影响快路径但不影响正确性
  （普通 FP16 GEMM 已验证）；(c) ZLUDA 对 RDNA4+Windows 的已知坑。
- 概率性表述刻意避免：在拿到 binary 与 trace 前给百分比是不负责任的。

### 14. 下一轮只做哪些最关键工作

1. **拿到两个专有 DLL 后 30 分钟内完成的三件事**：`binary_probe` 出 manifest（回答 18 问，
   锁定 GATE-0/1）、`module_trace + nvapi_trace` 跑 nr_host 判定 CASE、据此更新 CALLGRAPH。
2. 按 GATE-2/3 结果要么搭 `kernel_lab`（PTX 路线），要么启动 Track B 规划（SASS 路线）。
3. 在任一 NVIDIA 机器上跑 `run_reference.ps1` 产出真 reference（S4）。
4. 解除杂项阻塞：设置 git 身份提交全部 milestones（B-6）；评估 f16-WMMA 变体绕过 S-B4。

---

## A–E 五问（明确回答）

**A. 我们现在是否只是骗过了 GPU detection？**
没有。本项目从头到尾没有做任何 vendor spoofing：所有"成功"条目（S0–S2）都是
在真实的 AMD adapter（0x1002/0x7550）上，由内容级校验（数值误差 / mismatch=0 /
SHA-256 可复现）证明的计算事实。也没有任何"骗过检测"型结果被记为成功。

**B. 是否实际执行了来自 DLSSNR 的计算？**
**没有。** 一台没有 `nvngx_dlssnr.dll` 的机器不可能执行它的计算。nr_host 在缺文件时
显式返回 `BLOCKED_MISSING_PREREQUISITE`（exit 3），拒绝构造任何伪执行（E-4）。
已执行的计算只有我们自己的探针（hip_probe、d3d12_hip_interop kernel），它们证明的是
AMD 侧承接能力，不是 DLSSNR 执行。

**C. 如果执行了，它是否真的运行在 AMD GPU 上？**
不适用（B 的答案为否）。但已证明的 AMD 侧事实都带适配器证据：HIP 报告
gfx1201，D3D12 device 建在 0x1002/0x7550 适配器上，双向共享内存传输内容校验通过
（E-1/E-2）。一旦未来有 DLSSNR-originated kernel 运行，将按同一标准出示
vendor/device/LUID + kernel launch log。

**D. 当前离完整一帧 DLSS Neural Rendering 还差哪一层？**
按 S 等级，从 S2 到 S7 依次差：
1. **S3 二进制/调用链理解** ← 唯一卡点是缺 `nvngx_dlssnr.dll`（B-1/B-2）；
2. S4 NVIDIA 侧真 reference（B-3 无 N 卡，reference package 已备好）；
3. S5 NGX/NVAPI 初始化过检（shim 就绪，等 1）；
4. S6/S7 kernel 翻译层与单帧（依赖 1 判定的 CASE 与 GATE 结果）。
**当前唯一且真正的断点：缺专有文件。** 工具、框架、互操作地基均已到位。

**E. 最关键的下一步是什么？**
**由用户合法提供 `nvngx_dlssnr.dll` 与 `nvngx_dlss.dll`**（设 `DLSSNR_DLL_PATH` /
`DLSS_DLL_PATH`，来源：合法拥有的 DLSS 5 游戏安装目录或官方 DLSS update 包）。
提供后第一步：`build\binary_probe.exe` + `scripts\run_amd.ps1` 各跑一次 ——
前者回答"有没有可用表示（PTX?）"，后者回答"运行时走哪套 API（CASE A–D）"，
两者合起来即解锁 GATE-0..4 与整条决策链。

---

## 附：本轮明确不做 / 未做之事（诚实声明）

- 未从任何镜像下载专有二进制；仓库不含任何专有文件（.gitignore 已排除）。
- 未用 FSR/XeSS/CPU 实现冒充任何结果；无硬编码成功返回值。
- 未把"工具就绪"记为"目标达成"：所有依赖专有文件的实验均记
  MISSING_PREREQUISITE / PENDING_PREREQUISITE。
- WMMA（测试 F）未通过编译（S-B4），如实记为 UNVERIFIED，不计入 S1。
- git milestone commits 因身份未配置（B-6）排队，待用户执行：
  `git config user.name "..."; git config user.email "..."`（仓库内）。

---

# 增补：Round 1.5 —— 专有文件到位后的当日结果（2026-08-30 晚）

用户合法提供了两个 DLL（台账：docs/PROPRIETARY_FILES.md），上文第 5–8、12–14 项与
A–E 的回答相应更新如下（详细证据见 R-5/R-6 与 docs/BINARY_ANALYSIS.md）：

1. **S-level：S3 ACHIEVED（静态）**。S5 首次接触后阻塞于 NGX vendor gate（0xbad00001）。
2. **binary analysis（原第 5 项，已解锁）**：NR payload = `.rsrc` 内 **15 个纯 SASS
   CUBIN 模块（e_machine=190）**，目标 **sm_120（Blackwell）独占**，全文件 **零 PTX**；
   kernel 名与 `.nv.info` 参数 metadata 完整保留（fused Swin-Transformer 骨干，
   FP16/FP8 成对变体）；无 nvcuda/nvapi 静态或延迟导入。18 问全部回答。
3. **PTX 是否存在（原第 8 项）**：**不存在** —— GATE-1 FAIL，按规范第十二节记
   **PTX path = BLOCKED**，不假装 ZLUDA 可执行 SASS，进入 Track B（但因 kernel 名/
   metadata 完整，Track B 起点远好于黑盒）。
4. **call graph（原第 6 项）**：静态链路已定（nr_host→NGX→dlssnr→[模块加载/发射 API]→
   15×CUBIN）；动态发射端证据被 vendor gate 挡住，倾向 CASE A（CUBIN 模块路线，
   与 RTX-40 patch=CUBIN 替换的社区证据一致），待过门后由 trace 实锤。
5. **第一个真正 blocker（原第 12 项）已前移**：S-B5 —— NGX init 在 AMD 设备上返回
   0xbad00001（真实 vendor/硬件能力检查，先于一切 nvapi/cuda 调用）。按规范，
   骗过该检查永远不算成功证据；只允许作为**明确标注的 trace 观察手段**。
6. **首次真实动态接触**：未修改的两个 DLL 在 AMD 机上干净加载、5 个重建 NGX 导出
   全部解析成功、NGX init 真实执行并干净拒绝（未崩溃）—— E-6。

**A–E 更新回答**：
- A：依旧没有任何欺骗。0xbad00001 恰是 runtime 真实识别出非 NVIDIA 硬件的证据。
- B：依旧**没有**执行任何 DLSSNR 神经网络计算（被 vendor gate 合法拒绝）。
- C：不适用。已证明的 AMD 事实仍为 S1/S2 探针（0x1002/0x7550）。
- D：现在差两层——(1) 过 vendor gate（观察用）拿到发射端 API 清单；
  (2) 按 Track B+CASE A 机制提供 15 个 sm_120 SASS 的 AMD 等价 kernel（HSACO）。
- E：**下一步 = 过门观察 + Track B 启动**：以标注清晰的 trace-aid（如 DXGI 适配器
  代理，仅观察）拿到 nvapi_QueryInterface/模块加载全链，确认发射 API；同时按已提取的
  kernel 清单在 HIP 上重建第一个算子（建议从最小的 `post_block` 融合 kernel 起步）。
  仍需用户在任一 RTX 机器上跑 `run_reference.ps1` 产出真 reference（B-3 未解）。
