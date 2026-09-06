# ROCm定时堆栈异常审查（2026-09-06）

本次继续先重新审查`20260906_rocm_whole_frame640_twelve`，不是无修改地重试失败批次。

- 旧进程在30秒定时采集触发时出现0xC0000005；原生栈位于Python的
  PyCode_Addr2Line/_Py_DumpTracebackThreads，错误前记录了11次完整输出阶段。
- 没有已确认的设备重置证据，也没有遗留的验证进程；不据此宣称排除了驱动/内存问题。
- 当前监控已删除并发定时堆栈遍历，保留致命错误日志、进程PID/阶段/主机超时；
  完成帧现在逐次保存哈希，结束时仍要求零残留分配及真实正常退出。
- 相关9项CPU测试通过，确认不会注册定时采集，未取消失败门槛。

据此只恢复分级、有限的离线GPU验证：先128单次，再640单次，逐项检查后才进行640连续12次。
任何新的设备/同步/进程异常立即停止，不自动重试，不启动游戏，不改TDR。
恢复测试是验证已改动监控的实验，不是确认根因已彻底解决。
每份新报告增加监控配置和实现脚本SHA256。历史失败报告不修改、不提升为通过。

## 已完成的监控修改后复验

- `results/20260906_rocm_whole128_monitorv2_one/manifest.json`：单次通过。
- `results/20260906_rocm_whole640_monitorv2_one/manifest.json`：单次通过。
- `results/20260906_rocm_whole640_monitorv2_twelve/manifest.json`：连续12次通过，总进程时间31.25秒，
  跨过此前30秒触发点。输出哈希全部一致，实际正常退出，释放后分配/保留为零。
  每帧结束分配在393,373,696与393,430,528字节之间交替，不是逐帧递增。

这些结果支持缓解有效，但不证明所有根因已排除。初期源码哈希在执行结束时采集；
12次测试期间曾为下一阶段调整监控的主机时间预算（没有更改模型计算），因此该批次的
监控文件哈希不能视为启动时快照。后续改成启动前保存源码快照、结束时核对变更，
不修改旧报告的哈希。神经模型文件未在该测试中修改。

依据以上逐项复验，本轮恢复同尺寸100次离线测试。100次只把**主机进程监督**预算设为360秒，
单次GPU事件等待仍为10秒；未修改TDR，未扩大图片尺寸，未取消任何异常检查。
1000帧及其它内核的100次运行仍不由此开放，失败仍不重试。

新增 `scripts/assess_rocm_repeat.py` 检查完整帧日志、输出文件哈希、稳定输出、源码快照、
全部71阶段、原生GPU/无RTX中间值、进程正常退出与释放状态。通过范围仅限固定输入离线重复测试。

## 100次最终结果

`results/20260906_rocm_whole640_monitorv2_hundred/manifest.json`：通过，进程225.718秒正常退出。
100次输出哈希一致，全部71块由原生ROCm执行；启动源码快照与结束核对无变更。
每帧结束分配的变化范围56,832字节，没有持续增长；释放后分配/保留均为零。
热态逐层主机计时中位数2126.73ms、最大2541.54ms，含诊断逐层提交/等待，不能当成游戏FPS。

CPU只读评估 `results/20260906_rocm_whole640_hundred_assessment.json` 的14项检查均通过，
状态为 `STATIC_GPU_REPEAT_PASS`。输入为历史真实游戏图像的640×360裁剪，100次使用同输入及seed，
不是100个连续游戏帧。该结果不接受时序稳定、HDR、真实上屏、RTX画质或4K60。

本轮未复现原异常，支持此监控缓解有效；不据此证明原故障根因或历史C256退出挂起已经解决。
受控离线开发可以继续，新设备/同步/进程异常仍立即停止，不自动重试。

回归720项通过（54.37秒）。最终审计 `results/20260906_monitorv2_hundred_final_audit.json`
确认219份基线与4个游戏文件哈希不变，游戏关闭、无遗留验证进程；系统日志读取成功，
有限时间范围内未查到目标设备/电源事件。未训练、上传权重、改TDR或删除历史实验。

### 复现与只读评估命令

在项目根目录运行；NEW输出目录/文件必须不存在。GPU命令仅供完成逐级审查后手动运行，
不是异常后的自动重试批次。100次主机监督预算360秒，单次GPU事件等待仍为10秒。

```powershell
.\.venv-rocm\Scripts\python.exe scripts/validate_rocm_lifecycle.py --probe whole_frame640 --precision native_fp16 --iterations 100 --output results/whole640_hundred_NEW
& 'C:\DATA\Tools\ANACONDA\python.exe' scripts/assess_rocm_repeat.py --manifest results/whole640_hundred_NEW/manifest.json --output results/whole640_hundred_assessment_NEW.json
powershell -NoProfile -File scripts/audit_rocm_implementation.ps1 -OutputPath results/whole640_hundred_audit_NEW.json
```
