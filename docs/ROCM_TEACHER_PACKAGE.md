# RTX 5070 整帧序列教师采集候选包

状态：宿主已编译、输入检查和打包有本地测试，**尚未在 RTX 上运行验收**。
本包不包含 NVIDIA DLL、权重或游戏序列。当前项目尚缺带正确颜色／曝光／运动约定的真实连续帧，不能拿单张截图复制 32 次或补零运动替代。

这是通过原组件采集整帧输出的独立宿主，不是逐指令实验。
它经过 DLAA／RenoDX 接入，结果必须另行确认确实对应 NR 输出和匹配输入；日志中 feature 18 成功不等于教师已验收。

## 运行命令

在解压目录打开 PowerShell。要求 Windows、RTX 50 系、兼容组件的 NVIDIA 驱动、Python 3.12。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6

# PayloadDir 是你合法取得的组件目录，必须包含：
# ReShade64.dll、renodx-dlss5-02.addon64、nvngx_dlss.dll、nvngx_dlssnr.dll
# Manifest 是项目实际采集并核验的输入清单，不是截图目录。
.\.venv\Scripts\python.exe scripts\run_teacher_sequence.py `
  --manifest 'C:\NRData\input_manifest.json' `
  --payload-dir 'C:\NRPayload' `
  --output 'C:\NRResults\sequence_run_001'
```

先做纯输入检查时，在最后加 `--prepare-only`，并使用另一个全新输出目录。它不会运行 GPU，也不会生成教师图像。

运行结果返回 `C:\NRResults\sequence_run_001.zip`。失败也会尽可能生成诊断 ZIP；不自动重试。ZIP 不包含运行用的组件。工作目录保留便于排查，不改游戏文件。

## 输入约定

JSON 顶层为 `schema: 1` 和 `sequences` 列表。每段包含：

- `id`、`scene_id`、`split`（train / validation / test）、实际捕获的 `ngx_create_flags`、`frames`。
- 每段 1–32 帧；第一帧必须 `reset: true`。段内尺寸、颜色约定固定，帧号连续。
- 每帧 `frame_id`、`width`、`height`、`valid_rect: [0,0,width,height]`、`reset`。
- `color_mode`（SDR / HDR）、`color_contract_id`、`capture_provenance`、`pre_exposure`、`exposure_scale`、`motion_scale: [x,y]`、`jitter: [x,y]`。
- `color`、`motion`、`depth` 各为 `{path, sha256}`，路径相对清单所在目录且不得逃逸。
- 颜色为紧密排列 RGBA16F，运动为 RG16F，深度为 R32F；均为 little-endian、完整同尺寸图像。不同游戏格式必须在采集端进行已验证转换。

HDR 标记必须与 NGX create flags 的 IsHDR 位一致。这里不自动猜测色域或转换 PQ，也不以默认曝光覆盖捕获值。

## 返回结果的含义

`CANDIDATES_COLLECTED_NOT_TEACHER_ACCEPTED` 仅表示纹理上传逐字节一致、输出有限且组件日志存在。
候选输出在每个序列的数字帧目录下：`uploaded_color.raw`、`candidate_output.raw`。
必须另行核验 NR 实际输入／输出资源身份、颜色约定、模型版本及 4090D/5070 一致性，才能加入训练集。

发生黑屏、设备重置或超时，停止后续 GPU 工作；不要通过反复重跑或修改 TDR 设置掩盖故障。
