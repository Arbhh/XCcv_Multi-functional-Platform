# XCcv 多功能视觉平台 · 常见问题（FAQ）

> 适用版本：V0.8.2（全量开源）。排查顺序：先查本文 → 仍未解决时收集
> `platform_data/logs/<平台>.log` 与 `crash.log` 最后若干行反馈。
> 上手步骤见 [quickstart.md](quickstart.md)；术语与架构见
> [XCcv多功能视觉平台_项目简介.docx](XCcv多功能视觉平台_项目简介.docx)。
> English version: [FAQ_EN.md](FAQ_EN.md)。

**目录**

1. [安装与环境](#1-安装与环境)
2. [权重与示例数据](#2-权重与示例数据)
3. [数据标注](#3-数据标注)
4. [模型训练](#4-模型训练)
5. [测试与项目复盘](#5-测试与项目复盘)
6. [模型工具（剪枝 / TensorRT）](#6-模型工具剪枝--tensorrt)
7. [部署与推理服务](#7-部署与推理服务)
8. [流程图与传统算子](#8-流程图与传统算子)
9. [开源、许可与迁移](#9-开源许可与迁移)

---

## 1. 安装与环境

**Q1.1　需要什么环境？**
Windows 10 / 11（x64）＋ Python 3.11。GPU 可选：有 NVIDIA 显卡时训练与推理显著更快；CPU 也能跑（建议小模型、小尺寸）。依赖占盘约 6~8 GB。

**Q1.2　`pip install -r requirements.txt` 报 torch 版本冲突？**
先单独装 torch（按 CUDA 版本选官方源），再装其余依赖：

```bat
python -m venv venv && venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

CPU 机器把 `cu128` 换成 `cpu` 源即可。ultralytics 对 torch 有上界约束，版本以 requirements 为准，不要随意升 torch 大版本。

**Q1.3　双击 `run_main.bat` 没反应 / 一闪而过？**
启动脚本按以下顺序探测解释器，判据是能否 `import PyQt5.QtWebEngineWidgets`：
`XCCV_PY` → 本目录 `python_path.txt` → PATH 的 `python` → `py -3` → 本目录 `venv` / `.venv`。
推荐在仓库根写一行 `python_path.txt`（完整 python.exe 路径，已 gitignore）。全部不可用时脚本会调用 `tools\env_check.py` 输出每个候选的失败原因。

**Q1.4　报错 `Could not find QtWebEngineProcess` / 页面空白？**
PyQtWebEngine 与 PyQt5 版本必须匹配（requirements 已固定 5.15.7 / 5.15.9）。若手工装过其它版本，先 `pip uninstall PyQt5 PyQtWebEngine` 再按 requirements 重装。企业环境还需确认没有安全软件拦截 `QtWebEngineProcess.exe`。

**Q1.5　可以用中文路径吗？**
可以。数据集、权重、输出目录均可含中文；平台内部读写统一走 Unicode 接口。个别第三方组件（如某些 OpenCV 版本）对中文路径敏感，平台已内置兼容读写。

**Q1.6　启动卡在闪屏超过 30 秒？**
多为杀毒软件首次扫描 QtWebEngine 或解压缓存导致。把仓库目录加入杀软白名单；仍慢时在任务管理器确认 python 进程是否存活，并查看 `platform_data/logs/`。

**Q1.7　GPU 可用但训练很慢 / 掉频？**
先确认 `torch.cuda.is_available()` 为真且设备解析到 GPU；笔记本混合架构可能出现功耗/调度降频，参考主界面「设置」中的设备与推理服务显存面板；必要时关闭系统省电模式。

**Q1.8　能不能不用虚拟环境？**
可以但不推荐。`run_*.bat` 不要求目录内 venv，可用 `XCCV_PY` 指向任意 Python 3.11 环境。混装多版本 torch 容易踩坑，建议一项目一环境。

---

## 2. 权重与示例数据

**Q2.1　仓库里为什么没有 `.pt` / `.pth` 权重？**
本项目**不分发任何模型权重**：权重许可各异（Apache-2.0 / AGPL-3.0 / 各发布方自定义），随仓库分发会产生许可传染风险。平台提供下载/自建脚本，权重由使用者按各发布方条款自行获取。

**Q2.2　AI 智能标注提示缺权重？**
需要 SAM 系列权重（如 `sam2.1_t.pt`、`mobile_sam.pt`），放到仓库根 `weights/sam/`。缺失时智能多边形、画笔、批量预标注按钮给出指引，其余标注功能不受影响。

**Q2.3　CIF / SubspaceAD 提示缺骨干？**
CIF 需要 DINO ViT-B/8 权重（`weights/dino/dino_vitbase8_pretrain.pth`，或环境变量 `XCCV_DINO_CKPT`）；SubspaceAD 优先用 `weights/dinov2/` 本地目录（或 `XCCV_DINOV2_CKPT`），本地缺失时可能经 transformers 从 Hugging Face 下载标准 DINOv2 权重。国内网络可设 `HF_ENDPOINT=https://hf-mirror.com`。

**Q2.4　预训练基准权重（YOLO COCO / ImageNet）怎么一次性备好？**

```bat
python xccv\engine\weights\download_weights.py
```

脚本把 YOLO 与 torchvision 权重下到 `xccv/engine/weights/`（已 gitignore）。离线机器可在有网环境跑同一个脚本后整目录拷贝。

**Q2.5　maskrcnn18 的 slim 初始权重从哪来？**

- 自建（推荐）：`python tools\build_maskrcnn18_init.py`（含六项自检，产出 `weights/maskrcnn18_slim_init.pt`）；
- 转换 mmdet 权重：`python tools\convert_mmdet_r18.py 权重.pth --out weights/maskrcnn18_slim_coco.pt`。

两个产物都是明文 `.pt`，训练面板勾「slim 头 FPN64」+ 预训练时自动注入，自建 init 优先。

**Q2.6　缺陷合成的示例图在哪？**
示例背景与缺陷贴图约 33 MB，独立分发：下载 `XCcv-Platform-testdata-0.8.0.zip` 解压到 `xccv/defectgen/`（得到 `testdata/`）。缺失时合成界面仍可用，自行指定背景与贴图目录。

**Q2.7　权重可以放进 git 吗？**
不建议也不必要。`.gitignore` 已排除 `weights/`、`xccv/engine/weights/`、`*.pt/.pth/.onnx/.engine/.xva/.xcs`；确有团队共享需求请用对象存储或发布页附件。

---

## 3. 数据标注

**Q3.1　支持哪些标注形式？**
矩形、多边形；分类任务可用 rectangle 标注、`label` 写类别（或按目录归档）；OCR 任务用 rectangle 标注、`label` 写文本内容。标注结果自动保存为与图片同名的 labelme `.json`。

**Q3.2　「OK / 良品 / 正常 / 合格 / 负样本 / 背景」目录有什么用？**
这些目录下未标注的图片会被当作纯背景样本参与训练与验证（训练面板「负样本训练」默认开启），模型学到「这些图上什么都不该检出」，直接压低误检率。建议 OK : 缺陷 ≈ 1:1~1:2。

**Q3.3　AI 智能标注的三种模式怎么用？**

- `S` 智能多边形：左键点目标、实时预览轮廓、右键撤销锚点、回车确认；
- `B` 画笔：左键并集塑形、右键差集挖除；
- `L` 单张智能标注：对当前图一次生成；「工具 → AI 批量预标注」可整目录预生成。

**Q3.4　标注格式与 labelme 兼容吗？**
兼容。`shapes/label/points/shape_type/imagePath` 为标准字段，平台额外写入少量 additive 字段（如来源标记），不影响 labelme 打开。

**Q3.5　类别体系怎么维护？**
首次标注会提示创建 `labels.txt`；标注页「类别归并 / 重命名」可统一类别名；导出与训练按 `labels.txt` + json 双口径校验，避免类别不一致。

**Q3.6　标注错了怎么快速改？**
列表选中形状可编辑顶点；`Ctrl+Z` 撤销；「AI 复核当前图（疑似漏标）」会列出检出但未标注的区域，逐条确认即可补标。

---

## 4. 模型训练

**Q4.1　`CUDA out of memory` 怎么办？**
按代价从低到高：① 减小批量（4→2→1）；② 减小训练输入尺寸（增强策略「调整图像大小」）；③ 勾「冻结预训练骨干」与 AMP；④ R-CNN 勾「轻量掩码头」；⑤ P2 小目标头显存约为普通 2 倍，批量从 2 起步。

**Q4.2　损失不动 / 变成 NaN？**
看训练完成页的 **Loss 自动判读**：欠拟合 → 加轮次；过拟合 → 用 best 权重或减轮次；NaN → 降学习率、检查空/损坏标注；需要复现时勾「确定性训练」。

**Q4.3　验证指标很好，产线却漏检/误检多？**
五个自查项：① 产线打光/机构与训练数据是否一致；② 是否过拟合（看判读）；③ 平台与部署是否同口径（用部署平台「一致性自检」）；④ 数据是否不平衡（看 OK 占比）；⑤ **测试集上调出的阈值不代表产线全局**——用新一批产线图复验。

**Q4.4　同一项目两次训练结果不一样？**
深度学习默认非确定（cuDNN 自动调优、并行采样）。勾「确定性训练」后同机同种子同依赖两次指标一致（速度略有损失），并在 `run_manifest.json` 标记 `deterministic=true`。

**Q4.5　选哪个模型最快？**
分类 < 检测 < 分割；同任务下 n/s 系 < R-CNN50。速度优先：`yolo11n`（检测）、`yolo11n-seg`（分割）、`resnet18`（分类）；再导出 TensorRT FP16 可再提速。

**Q4.6　异常检测的三个后端怎么选？**

| 后端 | 训练方式 | 适用 |
|---|---|---|
| PatchCore | 无监督，只用 OK 图 | 零训练基线，数据充足时首选 |
| CIF | 少样本（k-shot）+ DINO 特征 + 模糊超图 | 缺陷样本极少、纹理复杂 |
| SubspaceAD | 少样本 + DINOv2 PCA 子空间 | 少样本且算力有限 |

三者都在训练面板自动隐藏轮次类参数，按阶段显示进度。

**Q4.7　切图训练怎么配？**
训练面板勾「切图训练」（R-CNN 家族），设 tile 边长（32 的倍数最佳）；背景 tile 自动保留 15%、10% 原图整图混入、验证集始终整图评估。**推理侧必须配合分块**（测试平台「4K 大图分块推理」或服务 `tiled=true`），否则口径不一致。

**Q4.8　训练产物是什么？**
`best.pt`（明文权重）+ `metrics.json` + `run_manifest.json`（环境指纹与配置快照）；勾「保存每轮权重」时另有 `epoch_weights/epoch_NNN.pt`，新训练会清空旧轮次。

**Q4.9　迁移学习权重类别不一致？**
平台会校验类别表，不一致直接拒绝并提示「重新生成」。确需跨类别迁移时，先在同类别数据集上生成剪枝/初始权重，或用「训练前剪枝」按当前数据集重建。

---

## 5. 测试与项目复盘

**Q5.1　阈值怎么调？**
标准页滑条实时生效；项目复盘页拖维度阈值（置信度 / 面积 / 短边 / 长边 / 灰度差）只复用已算结果重新分组，零重复推理。

**Q5.2　复盘页的「已评估 / 未评估」是什么口径？**
已评估 = 有 GT 或 OK 标记的图（按正确 / 漏检 / 过检三组）；未评估 = 无标注的生产图（按检出是否过阈分 OK / NG）。两者解耦，曲线随当前范围换口径。

**Q5.3　异常检测的分数为什么不是 0~1？**
异常检测模型的分数语义是 z（`1/conf−1` 变换，无上界），滑条与曲线按动态轴上界渲染；分类是概率、检测是置信度，语义均由模型声明，界面自动适配。

**Q5.4　A/B 比对会重复推理吗？**
不会。相同权重与参数的比对结果会缓存复用，误关窗口重开也不重推理。

**Q5.5　4K 大图怎么做？**
测试平台勾「4K 大图分块推理」：滑窗切分、重叠去重合并；与「切图训练」配套使用时口径一致。

**Q5.6　结果能导出成什么格式？**
统一 JSON（`schema_version=1`）、labelme、COCO 三种；批量运行同时输出渲染图与摘要（每类检出数、耗时、GT 匹配统计）。

---

## 6. 模型工具（剪枝 / TensorRT）

**Q6.1　剪枝后精度掉了怎么恢复？**
剪枝按 L1 范数裁通道并对齐 8 的倍数，分类头/检测头/RPN/工业模块一律忽略。建议剪 20%~40% 后用原数据集微调若干轮；剪枝结果与原模型可在测试平台直接 A/B 对比。

**Q6.2　YOLO 为什么不能剪枝？**
Torch-Pruning 对 ultralytics 图的分组会内存指数爆炸，平台默认**受控停用**并给出原因。专家模式设 `XCCV_YOLO_PRUNE=1` 可强制启用（务必在独立进程执行，内存看门狗会兜底终止）。

**Q6.3　剪枝/加速过程能中途停止吗？**
能。「紧急终止剪枝」立即终止子进程；原子落盘保证不留半截文件，源权重不受影响。

**Q6.4　TensorRT 导出报「未检测到 TensorRT 运行库」？**
与本机 CUDA 大版本匹配地 `pip install tensorrt`；或设置环境变量 `TENSORRT_DIR` 指向含 `nvinfer_10.dll` 的目录。导出需要 NVIDIA GPU。

**Q6.5　引擎在另一台机器加载失败？**
TensorRT 引擎与构建时的显卡算力和 TensorRT 版本绑定。跨显卡会得到「请在本机重新生成」的明确提示，而不是文件损坏误导。请在目标机器上用源权重重新导出。

**Q6.6　导出的 `.engine` 是明文吗？**
是。容器头（XCVE）记录类别、输入尺寸、构建算力与 TensorRT 版本，不加密；加载侧据此在反序列化前预检兼容性。

---

## 7. 部署与推理服务

**Q7.1　服务起不来 / 端口被占用？**
部署平台看 `platform_data/logs/service.log` 尾部；端口占用时改端口（面板可改）或先停旧服务。服务为单 worker（多 worker 会重复占用显存）。

**Q7.2　API Key 在哪、怎么换？**
首次启动自动生成，存于 `platform_data/service/config.json`；部署面板可一键复制 / 重新生成（会等待端口释放后自动重启服务）。请求头 `X-XCcv-Key`。

**Q7.3　能跨机访问吗？**
默认只监听 `127.0.0.1`。切到 `0.0.0.0` 前请注意：需配置模型路径白名单（非回环时强制启用）、保管好 API Key、确认网络隔离策略。

**Q7.4　C# 测试卡片编译失败？**
需要系统自带 .NET Framework 的 `csc.exe`（Windows 10/11 默认具备）。也可手动运行 `xccv\service\csharp\build_demo.bat`，看具体编译错误；Demo 为 WinForms，需桌面会话。

**Q7.5　「一致性自检」不一致怎么办？**
自检用同一张图分别走本地推理与 HTTP 服务，逐框比对 IoU（≥0.98）与分数差（<0.02）。不一致通常意味着两边权重/参数/预处理口径不同：检查加载的模型路径、conf/iou、是否开了分块推理。

**Q7.6　服务会返回权重吗？**
不会。任何端点都不返回权重字节；模型只在服务进程内加载，接口只给统一 JSON 或叠加图。

**Q7.7　怎么注册成 Windows 服务？**
`xccv\service\install_service.bat`（需 NSSM 与管理员权限）提供模板：设置 `NSSM`、`SVC_EXE`、`SVC_ARGS`、`APPDIR` 后执行，随后 `sc query XCcvInferenceService` 验证。

---

## 8. 流程图与传统算子

**Q8.1　算子到底有多少个？**
内置 57 个（预处理 18、定位 7、测量 2、逻辑 5、检出过滤 3、裁图/逐图 6、AI 推理 5、输出 4、外部算子 1），另有 9 个旧链兼容委托节点，画布共 66 个；官方模板 9 个。`python tools\doc_reconcile.py` 可随时对账。

**Q8.2　「传统平台」工作台和流程图有什么区别？**
工作台用于**单算子试跑与组合链验证**（快速看效果、调参数）；流程图用于**产线级编排**（多分支、逐图执行、结果合并、打包分发）。两者共用同一套算子与端口契约。

**Q8.3　自定义算子安全吗？**
自定义算子会在本地用 `exec` 执行你写的脚本，已实施 AST 白名单（仅允许 import numpy/cv2/math/statistics、禁双下划线属性、受限 builtins），但**这是防误用的受控环境，不是对抗恶意代码的沙箱**。不要运行来源不明的流程 JSON 或算子包。

**Q8.4　外部算子（HALCON / VisionMaster）还能用吗？**
V0.8.0 把外部算子改为**插件后端**：仓库内置 OpenCV 展示后端（常见算子名可直接运行），真实厂商 SDK 由使用方按 `ExternalEngine` 接口实现并注册，节点与流程零改动。仓库不再分发任何厂商适配器或 SDK 文件。

**Q8.5　`.xvap` 怎么用？**
「打包流程」把流程 JSON 与所选模型封入单个 `.xvap`（明文容器）；在另一台机器「解包流程」即可得到流程与「流程模型/」目录，模型引用自动改写为本地路径；推理服务也可直接加载 `.xvap` 作为常驻流程。

**Q8.6　流程跑起来某节点失败？**
节点异常被隔离：该节点标红、下游标记 skipped，独立分支继续运行。看工作台/流程图右下角运行日志的节点级报错，逐节点试跑定位参数。

---

## 9. 开源、许可与迁移

**Q9.1　项目是什么许可？能商用吗？**
本工程为 AGPL-3.0；`labelme/` 为 GPLv3 vendored 副本；训练链路依赖 ultralytics（AGPL-3.0）。AGPL 允许商业使用，但**通过网络对外提供服务时须履行源码开放义务**；如需闭源集成，请自行评估 ultralytics 企业许可与替换 PyQt5 为 LGPL 方案的可行性。

**Q9.2　第三方组件与论文引用在哪登记？**
`THIRD_PARTY_NOTICES.md`（组件、许可、用途）与 `CITATIONS.md`（论文出处）；许可全文在 `licenses/`。

**Q9.3　可以提交 PR 吗？**
欢迎。贡献以 DCO 1.1 签署为准（`git commit -s`）；PR 模板要求声明第三方代码来源。第三方代码准入矩阵与污染协议见 `CONTRIBUTING.md`。

**Q9.4　发现安全问题怎么报？**
不要开公开 Issue。走 `SECURITY.md` 的私下渠道（GitHub Security Advisories 或邮件），附复现步骤与环境信息。

**Q9.5　从 0.7.x 升级要注意什么？**

- 权重容器 `.xva/.xcs` 不再支持：用 0.8.0 重新训练或重新导出，加载旧文件会得到明确迁移提示；
- 训练产物、轮次权重、流程包、引擎一律明文（`best.pt`、`epoch_NNN.pt`、`.xvap`、`.engine`）；
- 模型工具不再有「权重加密 / 解密」卡片；
- 旧流程 JSON 可继续加载（端口迁移表保留兼容），外部算子节点需改选插件后端。

**Q9.6　文档在哪、怎么保证不过期？**
`README*` → `docs/quickstart*` → 本 FAQ → `CHANGELOG*`。算子数/模型数/支持版本由 `python tools\doc_reconcile.py` 自动对账，漂移即失败；UI 英文词条由覆盖检查脚本守护（当前 801 站点 / 1773 词条，覆盖 100%）。
