# 原生整帧窗口批量优化（2026-09-06）

目的：先消除大量小窗口调用的开销，不改权重、网络公式、全帧坐标或颜色约定。
不是逐指令数值拟合，不训练，也不把独立小图片拼回游戏画面。

## 已有依据

`results/20260831_005343_kernel_inventory/summary.json`记录捕获实例中43个使用函数、
156个调度槽位；原输出头为融合网格内核。当前PyTorch候选按12窗口批次执行，
一个窗口批次又产生多个GPU操作，两种执行组织不能混淆。
`results/20260906_rocm_whole640_monitorv2_one/manifest.json`冷态逐层记录中，
块0约1652ms、块70约583ms，共占3643ms的61%。冷态不能当热态结论。

## 实现与验证范围

`SingleColorWholeFrame.stages/forward`增加可选`boundary_batch`，只传给输入块与输出头。
省略时行为不变，跟随`window_batch`，当前常驻游戏执行器仍默认12。
`scripts/profile_native_boundary_batch.py`复用进程监督、致命错误日志、10秒GPU事件等待、
正常退出和零释放检查。主机监督预算小尺寸90秒、大尺寸四次A/B180秒；不是TDR或GPU取消。
遇到设备/同步/进程异常停止，不自动杀进程或重试。尺寸由已审核输入清单决定。

每一实验按12、12、96、96运行四次；全71块都从同输入/seed计算。
第一次和第三次标为该调度的冷态；热态比较第二次和第四次。
这是每种调度仅一个热样本，存在顺序/时钟偏差，不能作为统计性能验收。
逐层计时包含主机提交、GPU完成等待及有限值检查，不是纯GPU事件时间或游戏FPS。
输出读回仅供离线严格逐字节A/B，不是生产游戏帧路径；任何输出差异立即拒绝该优化验收。
相同输出只证明没有增加相对此候选的误差，不修复其已有方块伪影，也不证明RTX画质。

## 首批结果

- `results/20260906_boundary96_128_v1`：四次输出逐字节一致，正常退出，释放为0。
- `results/20260906_boundary96_640_v1`：历史真实游戏640×360裁剪，四次输出逐字节一致，
  正常退出，释放为0。热态12→96：块0为281.53→39.70ms，块70为552.27→70.94ms，
  全71块1850.69→1120.84ms，约1.65倍。张量峰值约597MB未增加。

原权重与游戏文件未更改，未重新启动游戏。本优化不等于实时版本。

## 大尺寸重复失败与输入投影修正

旧`results/20260906_boundary96_2342_v1`在第二次**原批量12**运行就失败，未执行96优化：
同输入最终9,707,121个分量不同，最大绝对误差0.989807。不得把它当成批量变化的误差。
进程主动返回2以拒绝验收，有python_atexit及零释放记录，没有超时或异常堆栈；
通用监督器因此写STOP_GPU_REQUIRES_REVIEW/normal_exit=false，并非已经证明设备崩溃。

已逐项审查后运行新增定位诊断（不是无修改重试原失败批次）：

1. `results/20260906_repeat2342_first_difference_v1`：输入哈希不变，首个不同阶段为块0。
2. `results/20260906_repeat2342_pre_boundaries_v1`：只运行预处理两次、对函数边界做只读快照。
   single_color_features输出一致，而pack_image的输入，即16→32投影结果，首先不同。
3. `results/20260906_preprojection2342_original_v1`：隔离原FP32三维GEMM；固定输入与权重，
   1024个分布采样行相对独立CPU公式有27,744个不同分量，最大误差2.03125。
   该隔离运行两次GPU结果相同但都不符CPU对照，说明只看重复哈希不足以证明正确。
   TF32关闭。尚未确定是具体BLAS内核、运行库还是其它底层原因，不能定性为AMD硬件缺陷。
4. `preprojection128_rows_v1`、`preprojection640_rows_v1`、`preprojection2342_rows_v1`
   （均带`results/20260906_`前缀）：65536特征行批量的FP32投影均与1024行CPU对照逐字节一致，
   两次完整投影输出一致，显存正常释放。

`native_preblock.project_input_features`据此改为按65536行执行相同16→32公式，最后存FP16；
这是逐像素线性映射的计算批次，不是独立图片窗口，不缩放/裁剪，也不引入CPU神经回退。
CPU单测保留非整齐形状、不同批量、公式一致和梯度验证。

修正后的`results/20260906_repeat2342_pre_rows_v1`预处理各边界/输出两次一致。
`results/20260906_rows_boundary96_128_v1`、`..._640_v1`完整A/B通过；640的输出仍与
此前100次验证哈希`3E514360D21279BE9E0E904EA7A7D70A92363FC45BE1472D9686A4AF36669BF4`一致。

`results/20260906_rows_boundary96_2342_v1`四次整链输出逐字节一致。
同一修正版热态12→仅两端96：22,964.16→13,067.40ms，约1.76倍；
块0为3854.50→499.47ms，块70为7526.61→920.08ms；峰值allocated3,266,070,528字节。
大尺寸修正后基线哈希`2E6FFA412DBB9AFB4AD3FFE353BC9C8EF4C265C54DF50CE58B103B300E5C82FA`
与旧错误投影版本不同，不能宣称整个修正与旧大尺寸结果逐字节一致。
大尺寸用彩条输入；尚未用实际游戏同帧复验方块伪影，不宣称画质已修复。

## 扩大主干窗口批次

新增`--middle-batch 96`，只在A/B后两次生效，前两次保留全部12的修正版基线。
这只是利用模型已有的window_batch接口，不改注意力范围、归约公式或权重。
`results/20260906_rows_all96_128_v1`与`..._640_v1`通过，四次输出一致。
640热态1811.46→414.01ms，约4.38倍；仍不是实时游戏计时。

`results/20260906_rows_all96_2342_v1`同样通过，四次输出与修正版基线逐字节一致；
热态23,176.36→3,490.86ms，约6.64倍。allocated峰值3,266,070,528字节，reserved峰值
3,948,937,216字节；结束释放均为0，源文件快照一致，无超时，进程正常返回0。
当前剩余最大阶段为输出头938.68ms、输入块500.81ms，随后各C32块约120–130ms。
这个速度仍远未达到游戏60FPS；不以离线耗时换算实际游戏帧率。
游戏常驻执行器未切换到96调度；需要新版本独立互操作证明和游戏复验。

## 旧游戏证明不能用于新代码

本轮审查发现旧ReShade独立证明只绑定二进制，而游戏启动时对Python源码重新取哈希，
不能证明源码就是独立测试时的版本。新增`native_preview_provenance.ps1`：
独立宿主测试前保存完整native_*.py与effect源码哈希，结束校验不变；游戏部署前要求
证明具有完整相同源码集合与哈希，否则在写游戏文件前拒绝。
旧证明没有这些字段，保留为历史记录，不自动补造来源；新代码须重新独立测试后才能部署。
本轮未重新运行游戏或修改其设置。

## 本轮最终检查

完整CPU回归786项通过（54.41秒），Python编译、三个PowerShell脚本语法检查及
git diff --check通过（仅既有换行提示）。未把广泛未提交工作区内容覆盖或提交。
`results/20260906_native_batch_final_audit.json`确认219项冻结资产与4个原游戏文件不变，
游戏关闭，最近一小时指定4101/41/6008系统事件无记录。没有训练、TDR更改或自动失败重试。
后续顺序：针对余下输入块/输出头优化，取得新源码版本的独立互操作证明，
再用同一真实游戏输入核查修正后伪影与短序列稳定性；不能跳过这些步骤宣称游戏实时化。

## 完整运行命令

在项目根目录的PowerShell7运行，NEW输出目录必须不存在；先退出游戏并检查系统错误。
每条GPU测试必须人工/代理检查上一项输出成功、正常退出和零释放后再进行下一规模；
这些命令不是自动失败重试脚本。

```powershell
& 'C:\DATA\Tools\ANACONDA\python.exe' -m pytest -q tests/test_native_boundary_batch.py tests/test_native_whole_frame.py
.\scripts\audit_rocm_implementation.ps1 -OutputPath results/batch_pre_audit_NEW.json
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 128 --batch 96 --output results/batch128_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 640 --batch 96 --output results/batch640_NEW
# 2342用已有独立宿主彩条输入，不是实际游戏输入或RTX教师：
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 2342 --batch 96 --output results/batch2342_NEW
# 按128、640、2342逐项检查后验证主干96窗口调度：
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 128 --batch 96 --middle-batch 96 --output results/all96_128_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 640 --batch 96 --middle-batch 96 --output results/all96_640_NEW
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 2342 --batch 96 --middle-batch 96 --output results/all96_2342_NEW
# 以下为定位模式，不作性能报告或优化通过证据：
.\.venv-rocm\Scripts\python.exe scripts/profile_native_boundary_batch.py --case 2342 --diagnose-repeat --pre-only --output results/pre_diagnostic_NEW
.\.venv-rocm\Scripts\python.exe scripts/diagnose_native_projection.py --case 2342 --mode row_chunk_fp32 --output results/projection_diagnostic_NEW
```

查看每个目录的manifest.json、child.json、runs.jsonl及output0..3.rgba16f。
输出与权重都留在项目私有本地目录，不打包上传。
