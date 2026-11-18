# Third-Party Notices / 第三方组件声明

XCcv Community 依赖以下开源组件，在此致谢。各组件归属其原作者，本仓库仅按其原许可引用。

| 组件 | 许可 | 用途 | 备注 |
|---|---|---|---|
| [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) | GPL v3（或 Riverbank 商业许可） | 桌面界面框架 | 本项目按 GPL v3 合规使用；若衍生作品需闭源分发，须自行取得商业许可 |
| [PyQtWebEngine](https://www.riverbankcomputing.com/software/pyqtwebengine/) | GPL v3（或商业许可） | 缺陷样本合成内嵌 Web 界面 | 同上 |
| [PyTorch](https://pytorch.org/) / [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | 训练与推理引擎；分类 / R-CNN 家族模型结构 | |
| [ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0**（或 Ultralytics 企业许可） | YOLO11 训练与推理、SAM 预测器 | AGPL 传染性较强：与本作品组合分发须整体遵循 AGPL-3.0；闭源商用请联系 Ultralytics |
| [labelme](https://github.com/wkentaro/labelme) 5.8.1 | **GPL v3** | 数据标注器（vendored 副本） | 见下方 vendored 说明 |
| [OpenCV](https://github.com/opencv/opencv) (opencv-python) | Apache-2.0 | 图像处理 | |
| [NumPy](https://numpy.org/) | BSD-3-Clause | 数值计算 | |
| [Pillow](https://python-pillow.org/) | HPND（类 MIT） | 图像读写 | |
| [matplotlib](https://matplotlib.org/) | PSF/BSD 风格 | 训练曲线渲染 | |
| [pyqtgraph](https://pyqtgraph.org/) | MIT | 训练可视化 |
| [LiteGraph.js](https://github.com/jagenjo/litegraph.js) | MIT | 流程图 Web 画布（vendored，许可见 `xccv/modeltools/web/litegraph.LICENSE`） | |
| [FastAPI](https://fastapi.tiangolo.com/) / [uvicorn](https://www.uvicorn.org/) / pydantic | MIT / BSD-3-Clause | 缺陷样本合成本地服务 | 仅监听本机回环地址 |
| [onnxruntime](https://onnxruntime.ai/) | MIT | EfficientSAM 编解码推理 | |
| [pycocotools](https://github.com/ppwwyyxx/cocoapi) | BSD-2-Clause | COCO 指标评估与导出 | |
| [scikit-image](https://scikit-image.org/) | BSD-3-Clause（含部分 BSD/PSF 组件） | 轮廓简化等图像工具 | |
| [imgviz](https://github.com/wkentaro/imgviz) | MIT | labelme 依赖 | |
| [natsort](https://github.com/sethmlarson/natsort) | MIT | labelme 文件排序 | |
| [PyYAML](https://pyyaml.org/) | MIT | labelme 配置 | |
| [loguru](https://github.com/Delgan/loguru) | MIT | 日志 | |
| [CIF](https://github.com/Sunny5250/CIF) | **MIT** | CIF 少样本异常检测后端（代码移植改写） | Copyright (c) 2025 Sunny5250；许可全文见 `licenses/CIF-LICENSE-MIT.txt`；修改部分由本项目按 AGPL-3.0 授权 |
| [SubspaceAD](https://github.com/CLendering/SubspaceAD) | **Apache-2.0** | SubspaceAD 少样本异常检测后端（代码移植改写） | 许可全文见 `licenses/SubspaceAD-LICENSE-Apache-2.0.txt`；修改文件已标注 |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | DINO ViT 特征加载（CIF 主路径） | 许可全文见 `licenses/Apache-2.0.txt` |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD-3-Clause | KMeans 聚类与随机投影维度估计（CIF） | 许可全文见 `licenses/BSD-3-Clause.txt` |
| [anomalib](https://github.com/openvinotoolkit/anomalib) | Apache-2.0 | 稀疏随机投影（CIF） | 许可全文见 `licenses/Apache-2.0.txt` |
| [transformers](https://github.com/huggingface/transformers) | Apache-2.0 | DINOv2 特征加载（SubspaceAD） | 许可全文见 `licenses/Apache-2.0.txt`；传递依赖保留各自许可 |

> 上表为直接依赖；timm / anomalib / transformers / scikit-learn 的传递依赖
> （如 tokenizers、safetensors、huggingface-hub、scipy、joblib、omegaconf、
> torchmetrics 等）在安装时由 pip 解析，其版权与许可随各自发行包提供。

## vendored labelme 说明

`labelme/` 目录为 [labelme v5.8.1](https://github.com/wkentaro/labelme/tree/v5.8.1)
的源码副本，并包含针对本平台的本地修改（自动保存、类别归类、AI 智能多边形 / 画笔 /
批量预标注接入等）。按 GPLv3 要求，其许可文本随仓库分发于
[licenses/labelme-LICENSE-GPLv3.txt](licenses/labelme-LICENSE-GPLv3.txt)；
逐文件修改清单见 [labelme/MODIFICATIONS.md](labelme/MODIFICATIONS.md)；
修改部分与本仓库其余代码一并按 AGPL-3.0 授权。

## 模型权重说明

本仓库**不分发任何模型权重文件**（预训练权重、SAM 权重、DINO / DINOv2 骨干等）。
运行时按需从各官方渠道下载，或由用户自行放置；权重的许可遵循其各自发布方
（如 ultralytics 权重遵循 AGPL-3.0、Meta SAM 权重遵循 Apache-2.0 等）。请在使用前自行核对。

- **DINO ViT-B/8**（`facebookresearch/dino`）与标准 **DINOv2**（`facebookresearch/dinov2`）
  的代码与权重按 Apache-2.0 发布；XRay-DINO / cell-DINO 等 NC 变体本项目不使用。
- CIF / SubspaceAD 的骨干权重**不随仓库、源码包或模型容器分发**，由用户自备
  （放置路径见 `docs/quickstart.md` 与代码中的环境变量说明）。SubspaceAD 在本地
  无权重时可能经 transformers 从 Hugging Face 下载标准 DINOv2 权重（Apache-2.0），
  仓库本身不携带或再分发这些文件。
