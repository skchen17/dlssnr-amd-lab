# Resident FP8 尺度切换与最小 GPU 门禁

日期：2026-09-07。

## 结果摘要

- `NRPlan` 已新增四个 Encoder downsample 和四个 Decoder upsample 的固定描述符及
  C++ recorder 插入点。Encoder transition 在 record 4/8/14/22 后执行，Decoder
  transition 在 record 48/56/62/66 前执行。
- Encoder 不再物化 pooled/logical/outview：直接读取前一 Swin core 的 FP16 logical-window
  输出，保持三次 FP16 pooling 舍入，随后用 gfx1201 FP8 WMMA 完成 `C→2C` projection，
  并写入下一 stage 的 packed resident E4M3。当前 stage 的 resident E4M3 通过一次 D2D
  copy 保存为 skip。
- Decoder 不再物化低分辨率 projection 或 `repeat_interleave`：一个 wave 同时计算16个
  low-resolution pixel×16个输出通道，并直接向四个高分辨率位置执行 skip-scale/add和
  packed E4M3 写入。
- 新拓扑仍仅覆盖 record 1–69；Pre（0）和 Head（70）未实现。QKV normalization与publish
  再拆分后静态图预算为506 kernel＋5 memcpy，虽然仍低于512 kernel门，但只剩6个kernel
  节点，显然不能容纳Pre/Head；这是诊断拓扑而非最终性能结构，也不是实测节点数。
- 最新QKV发布候选的编译产物为
  `results/20260907_native_nr_plan_build_v40_explicit_e4_lut/`；其单窗口结果位于
  `results/20260907_c256_nrplan_record15_w1_explicit_e4_lut_gate_v1/`。
  `nrPlanConfigureApproxScaleTransitions` 和两个 transition export 已验证存在。

## 静态 ISA

`transition_isa_audit.json` 覆盖 Encoder/Decoder 的 C32/C64/C128/C256 八个实例：

- 全部矩阵 kernel 均生成 `v_wmma_f32_16x16x16_fp8_fp8`；
- Encoder 为52 VGPR、34 SGPR；
- Decoder 为60 VGPR、22 SGPR；
- 全部0 LDS、0 scratch。

这些仅是 code-object 静态资源，不代表 occupancy、数值正确或吞吐已经通过。

## 本轮最小 GPU 门禁

在 CPU 侧确认无遗留游戏/RGP/Python/HIP进程，且系统日志没有新增41/6008/141/4101、
WHEA或AMD驱动错误后，仅执行了一次 C256 record 15、8×8、one-window、one-submit、
非RGP门禁：

- 生命周期：通过；6 kernel、0 memcpy；进程正常退出；allocated/reserved 均释放到0；
- 设备：RX 9070 XT / gfx1201；
- 数值：拒绝；16,320 bytes不同，最大绝对误差1.015625，NRMSE 3.038113；
- 系统稳定性：没有新增GPU/重启事件；未自动重试；GPU门禁已重新关闭。

随后在同样单次提交上增加只读边界诊断，结果为：FFN post FP16 NRMSE 0.005187；Q/K/V
分别为1.286955/2.357964/2.779706；attention value为3.783763；output projection FP16又降为
0.018440，但最终packed仍为3.038113。由此确认QKV/norm之后出现第一处大误差，并且
projection到packed之间还需要独立检查。CPU穷举已确认scatter索引公式与PyTorch
`scatter_packed`在四种origin下相同；下一次门将读回scatter前E4M3，区分量化放大与实际
kernel写回问题。

进一步将C32–C256的QKV投影与norm拆成两个kernel，新增固定FP16投影arena。C256最小门显示
投影NRMSE 0.015319、最大误差0.011276，而Q/K/V仍分别为1.286955/2.357964/2.779706，
因此首个大分叉已缩小到projection之后。尝试把norm从256线程多逻辑子组改成每个
`(vector,part)`独立32线程workgroup，输出逐项指标完全不变；静态ISA也明确标记
`.amdhsa_wavefront_size32 1`。该假设已拒绝，不再围绕wave64盲调。

后续门从native projection实际输出在CPU上重算Q/K/V，并保存raw、SHA-256和首16字节。
同步的不同字节poison在提交前已逐段读回验证，捕获图为预期7个kernel，DLL code object也
确认为新norm kernel。raw直方图与常见重排不匹配，且偶数字节大量为0/128，证实scalar
FP8＋byte-store发布组合产生了FP16字节形态；去掉提前return仍完全复现，故该候选已拒绝。

后续依次验证了五种发布实现：显式FP8x2/16-bit store、内联software-E4M3、software-E4M3
byte store、65,536项device-symbol LUT，以及拆分FP16 normalization与标量LUT pack。最后又
把LUT改成由`NRPlan`显式分配/持有/释放的64 KiB设备内存，并作为graph kernel参数传入，排除
跨编译单元symbol重定位。每次门的Q/K/V和最终输出SHA-256均完全相同；Q前16字节恰好等于
期望E4M3值解码为FP16后的前8个数（例如`80 bc 00 39 ...`），说明实际发布的是decoded-E4M3
FP16 storage。最新拆分graph为8个串行kernel，pack节点11 VGPR、0 LDS/scratch、单b8 store，
poison在提交前验证且提交后被覆盖；生命周期和释放均通过，没有新增系统事件。

因此当前证据已排除编码公式、store宽度、wave shuffle混合、arena区间重叠、旧DLL、graph
顺序和隐式LUT symbol。剩余最高优先级是读取/校验捕获kernel参数及每个节点的实际写地址，
确认是否存在HIP graph参数绑定或设备代码参数ABI错位；在该诊断完成前不再增加新的转换候选。
只读参数审计ABI已编译为`results/20260907_native_nr_plan_build_v41_argument_audit/`：它通过
`hipGraphKernelNodeGetParams`读取指定节点的pointer/u64参数，并同时返回workspace、weights和
plan-owned LUT地址；接口本身不launch kernel。CPU preflight已通过，尚未授权其下一次GPU门。

该结果说明当前 resident-FP8 C256 core 尚不能晋级代表 grid、完整 stage 或整帧性能测试。
原验证器过去只检查有限值和节点数，会将此类大误差误报为通过；现在已经增加
`NRMSE≤0.02`且`max error≤0.25`的近似单 block 数值门。

## 尚未通过

- 八个新 transition 只完成编译、ABI、offset、静态 ISA 门；尚未执行 GPU correctness。
- C256 core 需要修正Q/K/V发布边界；projection矩阵和E4M3编码算法不再是首要嫌疑。scatter前E4M3已确认
  与最终packed同样产生约3.04 NRMSE，因此不是scatter索引本身造成的放大。
- Pre、Head、原时序输入契约、完整 graph capture、1080p性能和游戏链均未通过。
- RGP counter安全锁保持启用；没有4K、循环压力或游戏测试。

## CPU/编译复现

```powershell
& .\scripts\build_native_nrplan_v3.ps1 `
  -OutputDirectory 'C:\ABSOLUTE\NEW\nr_plan_build'

& .\.venv-rocm\Scripts\python.exe -m pytest -q `
  tests\test_native_transition_topology.py `
  tests\test_native_topology_coverage.py `
  tests\test_native_cpp_nr_plan_v3.py
```

GPU 门禁不由这些命令自动开启；必须先通过 `safety/GPU_PROFILE_HALT.json` 的单次授权。
