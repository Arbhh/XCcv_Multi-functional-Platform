"""页 0「项目信息」页（InfoPageMixin）：信息/数据可视化/数据网格/
增强策略/训练参数表单的构建与行为，以及数据统计、数据体检、
rotate90 守卫、免训练范式 UI 适配。

0.7.6 P-1 页级拆分：方法体与拆分前 training_window.py 完全一致
（逐行搬运）；组合根 ProjectWindow 经 mixin 继承装配。
"""

import logging
log = logging.getLogger(__name__)

import random
from pathlib import Path
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)
from ...engine.dataset import IMAGE_EXTS
from ...engine.dataset.base import TaskType
from ...engine.dataset.reader import scan_classification_dataset, scan_labelme_dataset
from ...engine.io_utils import imread_unicode
from ...gui_utils import auto_cleanup_thread
from ...i18n import tr
from ..charts import ConfusionMatrixCanvas, DistributionCanvas, SplitDonutCanvas
from ..dataset_stats import collect_stats
from ..dialogs import CollapsibleSection, DevOptionsDialog, HealthReportDialog, ReportRow
from ..project import (Project, create_project, delete_project, list_projects, load_project, save_project)
from ...ui_widgets import Badge, HoverCard, load_prefs, relative_time, save_prefs, toast
from .common import (
    AUGMENT_ROWS, DEFAULT_LR, _PRESET_DEFS, _PRESET_DEFS_FREE, _Spin, _DoubleSpin, _Combo, _PreflightWorker, _cuda_available,
)


# ============================================================
# 页面方法（自 training_window.py 逐行搬运，语义不变）
# ============================================================

class InfoPageMixin:
    # ---- 页 0：项目信息（滚动）----
    def _build_info_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(12)

        # 项目信息
        g1 = QGroupBox("项目信息")
        f1 = QFormLayout(g1)
        self.info_name = QLineEdit()
        self.info_dataset = QLineEdit()
        btn_ds = QPushButton("浏览…")
        btn_ds.clicked.connect(lambda: self._browse_into(
            self.info_dataset, tr("选择数据集文件夹")))
        ds_row = QHBoxLayout()
        ds_row.addWidget(self.info_dataset, stretch=1)
        ds_row.addWidget(btn_ds)
        self.info_model_dir = QLineEdit()
        btn_md = QPushButton("浏览…")
        btn_md.clicked.connect(lambda: self._browse_into(
            self.info_model_dir, tr("选择模型储存文件夹")))
        md_row = QHBoxLayout()
        md_row.addWidget(self.info_model_dir, stretch=1)
        md_row.addWidget(btn_md)
        from ...i18n import tr

        self.info_task = QLabel(tr(self.project.task_type.label))
        self.info_created = QLabel(self.project.created_at)
        btn_save_info = QPushButton("保存修改")
        btn_save_info.clicked.connect(self._save_info)
        f1.addRow("项目名称", self.info_name)
        f1.addRow("数据集路径", ds_row)
        f1.addRow("模型储存路径", md_row)
        f1.addRow("任务类型", self.info_task)
        f1.addRow("创建时间", self.info_created)
        f1.addRow(btn_save_info)
        lay.addWidget(g1)

        # 数据集可视化
        g2 = QGroupBox("数据集可视化")
        v2 = QVBoxLayout(g2)
        head2 = QHBoxLayout()
        self.stats_hint = QLabel("")
        self.stats_hint.setObjectName("hint")
        btn_refresh = QPushButton("刷新统计")
        btn_refresh.clicked.connect(self._refresh_stats)
        btn_refresh.clicked.connect(lambda: self._refresh_augment_preview(new_image=True))
        # 数据体检：仅提示/警告两档，附「去标注平台处理」闭环（）
        self.btn_health = QPushButton("数据体检")
        self.btn_health.setToolTip(
            "检查类别不均衡、样本过少、验证集类别缺失、疑似重复图、尺寸离群等；\n"
            "结果仅分「提示 / 警告」两档，不会阻止训练；阈值可在右上角 ⚙ 开发者选项调整")
        self.btn_health.clicked.connect(self._run_health_report)
        head2.addWidget(self.stats_hint)
        head2.addStretch(1)
        head2.addWidget(self.btn_health)
        head2.addWidget(btn_refresh)
        v2.addLayout(head2)
        self.dist_canvas = DistributionCanvas()
        v2.addWidget(self.dist_canvas)
        # 训练/验证划分环形图（随统计刷新）
        self.split_canvas = SplitDonutCanvas()
        self.split_canvas.setMaximumHeight(240)
        v2.addWidget(self.split_canvas)
        lay.addWidget(g2)

        # 数据网格（缩略图墙；后台扫描+懒加载缩略+LRU）
        g2b = QGroupBox("数据网格（双击图片用系统看图打开）")
        v2b = QVBoxLayout(g2b)
        from ..data_grid import get_data_grid_class

        self.data_grid = get_data_grid_class()()
        v2b.addWidget(self.data_grid)
        lay.addWidget(g2b)

        # 图像增强策略
        g3 = QGroupBox("图像增强策略（勾选后启用；用于数据倍增与随机变化）")
        grid = QGridLayout(g3)
        self.augment_controls: dict[str, tuple[QCheckBox, dict]] = {}
        self.augment_prob_spins: dict[str, _DoubleSpin] = {}  # 每算子启用概率
        for r, (op, label, params, tip) in enumerate(AUGMENT_ROWS):
            check = QCheckBox(label)
            check.setToolTip(tip)
            spins = {}
            for c, (pname, default, vmin, vmax, is_int) in enumerate(params, start=1):
                spin = _Spin() if is_int else _DoubleSpin()
                if is_int:
                    spin.setRange(int(vmin), int(vmax))
                    spin.setValue(int(default))
                else:
                    spin.setDecimals(2)
                    spin.setRange(vmin, vmax)
                    spin.setSingleStep(0.05)
                    spin.setValue(default)
                grid.addWidget(QLabel(pname), r, c * 2 - 1)
                grid.addWidget(spin, r, c * 2)
                spins[pname] = spin
            # 启用概率列：每个算子独立掷硬币（旧项目无 augment_probs 时回填 0.5）
            pc = len(params) * 2 + 1
            grid.addWidget(QLabel("概率"), r, pc)
            prob = _DoubleSpin()
            prob.setDecimals(2)
            prob.setRange(0.05, 1.0)
            prob.setSingleStep(0.05)
            prob.setValue(0.5)
            prob.setToolTip(tr("该算子的启用概率：每次取样独立判定是否应用"))
            grid.addWidget(prob, r, pc + 1)
            prob.valueChanged.connect(self._collect_augment_ops)
            prob.valueChanged.connect(lambda _v: self._refresh_augment_preview())
            self.augment_prob_spins[op] = prob
            check.stateChanged.connect(self._collect_augment_ops)
            check.stateChanged.connect(lambda _s: self._refresh_augment_preview())
            for sp in spins.values():
                # 参数变化即重预览：与 resize 行为一致（同源图重新应用）
                sp.valueChanged.connect(lambda _v: self._refresh_augment_preview())
            grid.addWidget(check, r, 0)
            self.augment_controls[op] = (check, spins)
        resize_check, resize_spins = self.augment_controls["resize"]
        resize_check.stateChanged.connect(self._refresh_input_size_hint)
        # 预览刷新已在上方循环按行接线（V0.7.1 起全算子）；此处只保留
        # 训练输入尺寸提示。注意接线必须用 lambda 隔离：直接连接会把
        # stateChanged/valueChanged 的整数实参漏进
        # _refresh_augment_preview(new_image=…) 形参（非 0 即真），
        # 每次调整都换成新随机图，"同图对比"预览效果失效
        for sp in resize_spins.values():
            sp.valueChanged.connect(self._refresh_input_size_hint)
        # 随机裁剪（rotate90 兜底）：勾选时自动对齐 resize 的目标宽高比
        crop_check, crop_spins = self.augment_controls["random_crop"]

        def _sync_crop_ratio():
            size = self._current_resize()
            if size and size[1]:
                crop_spins["ratio"].setValue(round(size[0] / size[1], 2))

        crop_check.stateChanged.connect(lambda _s: _sync_crop_ratio())
        btn_save_aug = QPushButton("保存增强策略")
        btn_save_aug.clicked.connect(self._save_augment)
        grid.addWidget(btn_save_aug, len(AUGMENT_ROWS), 0)

        # 效果预览：随机抽取数据集图片，按 resize 参数拉伸缩放（自适应显示）
        self.augment_preview = QLabel()
        self.augment_preview.setMinimumHeight(220)
        self.augment_preview.setAlignment(Qt.AlignCenter)
        self.augment_preview.setStyleSheet(
            "border: 1px solid #d7e2f0; border-radius: 8px; background: #f8fbff;")
        self.preview_info = QLabel("")
        self.preview_info.setObjectName("hint")
        btn_rand = QPushButton("换一张")
        btn_rand.clicked.connect(lambda: self._refresh_augment_preview(new_image=True))
        pv_head = QHBoxLayout()
        pv_head.addWidget(QLabel("效果预览"))
        pv_head.addWidget(self.preview_info, stretch=1)
        pv_head.addWidget(btn_rand)
        pv_box = QWidget()
        pv_lay = QVBoxLayout(pv_box)
        pv_lay.setContentsMargins(0, 6, 0, 0)
        pv_lay.setSpacing(6)
        pv_lay.addLayout(pv_head)
        pv_lay.addWidget(self.augment_preview, stretch=1)
        grid.addWidget(pv_box, len(AUGMENT_ROWS) + 1, 0, 1, 12)  # 跨全部列
        lay.addWidget(g3)

        # 训练参数（Quick/专家双轨，D4：默认快速档 6 核心参数；「专家模式」
        # 展开现有完整折叠表单；顶部三张预设卡一键套用并联动智能推荐；
        # 预算实时摘要明确暂缓）
        g4 = QGroupBox("训练参数")
        v4 = QVBoxLayout(g4)
        mode_row = QHBoxLayout()
        self.btn_mode_quick = QPushButton("⚡ 快速配置")
        self.btn_mode_expert = QPushButton("专家模式")
        for b in (self.btn_mode_quick, self.btn_mode_expert):
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
        self.btn_mode_quick.setToolTip(
            tr("只呈现 6 个核心参数，档位化配置，零经验 30 秒完成"))
        self.btn_mode_expert.setToolTip(tr("展开完整参数表单（数据与验证 / 工业与加速）"))
        self.btn_mode_quick.clicked.connect(self._on_mode_quick)
        self.btn_mode_expert.clicked.connect(self._on_mode_expert)
        mode_row.addWidget(self.btn_mode_quick)
        mode_row.addWidget(self.btn_mode_expert)
        mode_row.addStretch(1)
        self.btn_recommend = QPushButton("✨ 智能推荐配置")
        self.btn_recommend.setToolTip(
            tr("按数据集规模与硬件给出保守建议（冻结骨干 / 工业优化 / 增强概率等）；\n"
               "应用后写回本表单并在状态栏说明理由，仍可手动调整"))
        self.btn_recommend.clicked.connect(self._apply_recommendations)
        mode_row.addWidget(self.btn_recommend)
        self.btn_expand_all = QPushButton("全部展开")
        self.btn_expand_all.setToolTip(tr("展开/折叠下方全部分区"))
        self.btn_expand_all.clicked.connect(self._toggle_all_sections)
        mode_row.addWidget(self.btn_expand_all)
        v4.addLayout(mode_row)

        self.par_model = _Combo()
        self.par_model.currentTextChanged.connect(self._on_model_changed)
        self.input_size_hint = QLabel("")
        self.input_size_hint.setObjectName("accentText")
        self.input_size_hint.setToolTip(tr(
            "训练输入尺寸与上方增强策略的 resize 宽×高一致：图像拉伸到该尺寸训练\n"
            "（比例随设定改变，预览所见即训练所见）。\n"
            "未启用 resize 时使用模型默认尺寸，由各模型自行缩放（YOLO letterbox /\n"
            "R-CNN 短边缩放 / 分类拉伸方形）——长宽比极端的图片建议保持 resize 启用"))
        self.par_train_ratio = _DoubleSpin()
        self.par_train_ratio.setDecimals(2)
        self.par_train_ratio.setRange(0.1, 1.0)
        self.par_train_ratio.setSingleStep(0.05)
        self.par_train_ratio.setToolTip(
            "训练集占整体数据的比例。与验证集比例之和可以大于 100%：\n"
            "此时验证集从整体数据抽出后，训练集仍按占比从整体抽取（允许重叠），\n"
            "例如 训练集 100% + 验证集 20%（稀缺缺陷类别场景，验证集为训练集子集）")
        self.par_val_ratio = _DoubleSpin()
        self.par_val_ratio.setDecimals(2)
        self.par_val_ratio.setRange(0.0, 0.9)
        self.par_val_ratio.setSingleStep(0.05)
        self.par_val_ratio.setToolTip("验证集占整体数据的比例（为 0 时验证集复用训练集）")
        # 训练/验证划分环状图实时贴合这两个参数（改值即重绘）
        self.par_train_ratio.valueChanged.connect(self._refresh_split_canvas)
        self.par_val_ratio.valueChanged.connect(self._refresh_split_canvas)
        self.par_multiply = _Spin()
        self.par_multiply.setRange(1, 50)
        self.par_multiply.setToolTip(
            "每轮重复学习次数：每个 epoch 中训练集被学习的遍数；\n"
            "增强策略按随机概率（约 50%）启用，与该值无关——\n"
            "设为 1 且启用增强时，训练中也会随机增强")
        self.par_batch = _Spin()
        self.par_batch.setRange(1, 256)
        self.par_batch.setToolTip(
            "批量大小：检测/分割推荐 2~4（大分辨率下平台会按显存预算自动下调，\n"
            "必要时启用梯度检查点保证不 OOM）；分类推荐 16~32。\n"
            "建议使用默认批量大小以获得最佳性能")
        self.par_epochs = _Spin()
        self.par_epochs.setRange(1, 2000)
        self.par_epochs.setToolTip(
            "训练代数：工业小数据集建议 12~30；\n"
            "需要更多学习次数时优先调大「每轮重复学习次数」而不是代数")
        self.par_lr = _DoubleSpin()
        self.par_lr.setDecimals(6)
        self.par_lr.setRange(0.000001, 1.0)
        self.par_lr.setSingleStep(0.0001)
        self.par_lr.setToolTip(
            "选择模型时自动带入推荐值：YOLO 0.01 / R-CNN 0.005 / 分类 0.001；\n"
            "从已有权重继续训练（迁移学习）时建议调低为推荐值的 1/5~1/10")
        self.par_wd = _DoubleSpin()
        self.par_wd.setDecimals(6)
        self.par_wd.setRange(0.0, 1.0)
        # §四（V0.7.3）：免训练范式自有参数（仅声明 support_k_shot 的模型显示）
        self.par_kshot = _Spin()
        self.par_kshot.setRange(1, 12)
        self.par_kshot.setValue(4)
        self.par_kshot.setToolTip(tr(
            "支持集张数 k-shot：用多少张 OK 图建正常子空间（1~12，上限防止"
            "脱离免训练范畴）；仅 SubspaceAD 使用"))
        self.lbl_kshot = QLabel("支持集张数")
        self.par_augcount = _Spin()
        self.par_augcount.setRange(0, 60)
        self.par_augcount.setValue(30)
        self.par_augcount.setToolTip(tr(
            "增强样本数：每张支持图生成的随机旋转增强张数（0=关闭）；"
            "仅 SubspaceAD 使用"))
        self.lbl_augcount = QLabel("增强样本数")
        self.par_eval_interval = _Spin()
        self.par_eval_interval.setRange(1, 100)
        self.par_eval_interval.setToolTip("每多少轮验证一次（YOLO 家族每轮都会验证）")
        self.par_device = _Combo()
        self.par_device.addItem("自动", "auto")
        self.par_device.addItem("CPU", "cpu")
        if _cuda_available():
            import torch

            self.par_device.addItem(f"GPU ({torch.cuda.get_device_name(0)})", "cuda")
        self.par_device.currentIndexChanged.connect(self._save_params)
        btn_save_par = QPushButton("保存参数")
        btn_save_par.clicked.connect(self._save_params)
        self.btn_start = QPushButton("▶ 开始训练")
        self.btn_start.setObjectName("accent")
        self.btn_start.clicked.connect(self._start_training)
        self.par_industrial = QCheckBox("工业场景优化（注意力 + 改进损失）")
        self.par_industrial.setToolTip(
            "针对工业缺陷检测的模型与损失改进（详见 docs/工业优化说明.md）；\n"
            "关闭即为基线模型，便于消融对照")
        # 训练加速（GPU 工控机上按需开启；CPU 机器 AMP 自动失效）
        self.par_freeze = QCheckBox("训练加速：冻结骨干网络（快速迁移模式）")
        self.par_freeze.setToolTip(
            "冻结预训练骨干，只训练分类头/检测头与工业模块（CBAM/ECA/DCN 仍参与训练）；\n"
            "每轮迭代提速约 40~60%，小数据集精度通常持平；\n"
            "领域差距极大（红外/医学等）时精度可能受限，可关闭回退；\n"
            "未使用预训练权重时该选项自动忽略")
        self.par_rpn = QCheckBox("削减 RPN 候选框（仅 Faster/Mask R-CNN 生效）")
        self.par_rpn.setToolTip(
            "每图候选 2000→256：工业单图目标少时几乎无精度损失，迭代更快、显存更省")
        self.par_mask_light = QCheckBox("轻量掩码头（仅 Mask R-CNN 生效）")
        self.par_mask_light.setToolTip(
            "掩码头 4 层 256 通道 → 2 层 128 通道：掩码分支更快、更省显存；\n"
            "工业缺陷不需要像素级细腻边缘时几乎无精度损失，可随时关闭回退")
        self.par_anchor = QCheckBox("小目标锚框自适应（仅 R-CNN 家族生效）")
        self.par_anchor.setToolTip(
            "锚框金字塔底径=入网短边/80（默认 32 针对大目标设计，\n"
            "工业小缺陷与最小锚 IoU 过低会导致 RPN 正样本稀少）；\n"
            "小目标数据集建议开启；切图训练时按 tile 边长取底径")
        self.par_slim_head = QCheckBox("slim 头 FPN64（仅 maskrcnn18 生效）")
        self.par_slim_head.setToolTip(
            "slim 头：FPN 64 通道+两层 256 FC+4×64 掩码头（约 12.5M 参数），\n"
            "可加载 mmdet 预训练权重；\n"
            "显存占用约为标准头一半，小数据集收敛更快。仅在使用 maskrcnn18 时生效")
        self.par_p2 = QCheckBox("P2 小目标检测头（仅 YOLO 家族生效）")
        self.par_p2.setToolTip(
            "在所选 YOLO 模型（n/s，检测/分割）上增加 stride 4 的第四检测层，\n"
            "面向短边 ≤8px 的微小缺陷；骨干/颈部/原检测分支继承官方预训练，\n"
            "仅新增的 P2 分支从头学习，建议适当增加训练轮次；\n"
            "显存约为普通模型的 2 倍（1920 输入 n-seg 批量 4 约 13 GiB），\n"
            "显存不足请减小批量；训练产物自带结构，测试/部署无需再勾选；\n"
            "迁移学习/从剪枝模型继续训练时沿用起始权重的结构，本开关不生效。\n"
            "R-CNN 与分类模型忽略该选项")
        self.par_p2.toggled.connect(self._refresh_input_size_hint)
        # R-CNN 优化器三件套
        self.par_rcnn_opt = _Combo()
        self.par_rcnn_opt.addItem("AdamW（推荐）", "adamw")
        self.par_rcnn_opt.addItem("SGD+动量（旧行为）", "sgd")
        self.par_rcnn_opt.currentIndexChanged.connect(self._save_params)
        self.par_rcnn_lr = _DoubleSpin()
        self.par_rcnn_lr.setRange(0.0, 1.0)
        self.par_rcnn_lr.setDecimals(6)
        self.par_rcnn_lr.setSingleStep(1e-5)
        self.par_rcnn_lr.setValue(0.0)
        self.par_rcnn_lr.setToolTip("0=自动（AdamW→1e-4；SGD→学习率字段）")
        self.par_rcnn_lr.valueChanged.connect(self._save_params)
        self.par_rcnn_wd = _DoubleSpin()
        self.par_rcnn_wd.setRange(0.0, 0.5)
        self.par_rcnn_wd.setDecimals(4)
        self.par_rcnn_wd.setSingleStep(0.005)
        self.par_rcnn_wd.setValue(0.05)
        self.par_rcnn_wd.valueChanged.connect(self._save_params)

        self.par_amp = QCheckBox("AMP 混合精度（GPU 提速；CPU 自动关闭）")
        self.par_amp.setChecked(True)
        self.par_tf32 = QCheckBox("GPU 提速：TF32 与 cuDNN 自动调优（Ampere+ 显卡；CPU 无影响）")
        self.par_tf32.setChecked(True)
        self.par_tf32.setToolTip(
            "TF32：Ampere 及以上 NVIDIA 显卡对 FP32 计算零成本加速（约 1.5~2 倍）；\n"
            "cuDNN 自动调优：固定输入尺寸训练时自动搜索并缓存最优卷积算法（约 1.1~1.3 倍）；\n"
            "无 GPU 或较老显卡时不生效，可放心保持开启")
        self.par_epoch_w = QCheckBox("保存每轮权重（轮次权重文件夹，新训练覆盖）")
        self.par_epoch_w.setChecked(True)
        self.par_epoch_w.setToolTip(
            "每次训练共用「模型储存路径\\epoch_weights」文件夹，逐轮保存权重：\n"
            "训练开始时清空旧训练的轮次权重（覆盖式，只占最近一次训练的空间）；\n"
            "轮次权重可在测试平台直接加载对比")
        self.par_patience = _Spin()
        self.par_patience.setRange(0, 200)
        self.par_patience.setValue(10)
        self.par_patience.setToolTip("连续 N 轮验证无提升则提前停止训练；0 = 关闭早停")
        self.par_tiling = QCheckBox("切图训练（大图小目标，R-CNN 家族）")
        self.par_tiling.setToolTip(
            "把原图按滑窗切成 tile 训练：小目标占画面比例更大、更易学到；\n"
            "背景 tile 按 15% 保留、10% 原图整图混入、验证集始终整图评估；\n"
            "**推理必须配合分块**：测试平台「4K 大图分块推理」或服务 tiled=true\n"
            "（与两段式缩放互斥使用更清晰：切图训练时不建议再设网络输入尺寸）")
        self.par_neg = QCheckBox("负样本训练（OK 图降误检，检测/分割）")
        self.par_neg.setChecked(True)
        self.par_neg.setToolTip(
            "数据集中「OK / 良品 / 正常 / 合格 / 负样本 / 背景」等文件夹里的\n"
            "未标注图片会作为纯背景样本参与训练与验证（默认开启）：\n"
            "模型学到「这些图上什么都不该检出」，直接压低误检率；\n"
            "训练日志与历史记录会给出负样本张数与 OK 图误检率指标；\n"
            "标注图划分不受影响；负样本图数超过标注图数时会自动抽样到 1:1")
        # 颜色模式（灰度/彩色）——颜色无判别意义时用灰度防干扰
        self.par_gray = QCheckBox("灰度训练（颜色无判别意义时防颜色干扰）")
        self.par_gray.setToolTip(
            "开启后训练读图统一转灰度（复制 3 通道，骨干零改动），\n"
            "推理/部署侧按权重内记录自动同口径转灰；\n"
            "适合颜色不携带判别信息、打光颜色易漂移的工业场景；\n"
            "对检测/分割/分类家族生效（YOLO 推理侧加速导出暂不透传该标记）")
        # 确定性训练 opt-in
        self.par_det = QCheckBox("确定性训练（可复现，默认关）")
        self.par_det.setToolTip(
            "开启后固定全部随机种子并启用确定性算法（CUBLAS_WORKSPACE_CONFIG\n"
            ":4096:8 + torch.use_deterministic_algorithms）：同机同种子同依赖\n"
            "版本两次训练指标一致，run_manifest.json 标记 deterministic=true；\n"
            "代价：训练速度下降、个别算子不支持确定性实现时降级告警（不中断）")
        self.par_tile_size = _Spin()
        self.par_tile_size.setRange(256, 2048)
        self.par_tile_size.setSingleStep(32)
        self.par_tile_size.setValue(640)
        self.par_tile_size.setToolTip("切图 tile 边长（像素，32 的倍数最佳）")
        self.par_train_input = _Spin()
        self.par_train_input.setRange(0, 4096)
        self.par_train_input.setSingleStep(32)
        self.par_train_input.setValue(0)
        self.par_train_input.setToolTip(
            "两段式缩放（可选，默认 0=跟随增强画布=现状）：\n"
            "显式设 32 的倍数且小于画布短边时启用——画布增强保持画布口径，\n"
            "入网前按比例缩到该尺寸（小目标占画面比例更大、显存更省）；\n"
            "仅 R-CNN 家族生效；推理侧请配合测试平台「4K 大图分块推理」\n"
            "或服务 tiled=true（与切图训练配对使用效果最佳）")
        self.par_transfer = QCheckBox("迁移学习（从已有模型权重开始训练）")
        self.par_transfer.setToolTip("勾选后通过文件选择器指定起始权重；"
                                     "类别需与当前数据集一致")
        self.par_transfer.toggled.connect(self._on_transfer_toggled)
        transfer_row = QHBoxLayout()
        self.par_transfer_edit = QLineEdit()
        self.par_transfer_edit.setPlaceholderText("迁移学习起始权重文件路径")
        btn_pick_model = QPushButton("浏览…")
        btn_pick_model.clicked.connect(self._pick_transfer_file)
        transfer_row.addWidget(self.par_transfer_edit, stretch=1)
        transfer_row.addWidget(btn_pick_model)

        # ---- Quick 档控件（与专家表单 par_* 双向同步；保存仍以 par_* 为准）----
        self.q_model = _Combo()
        self.q_epochs = _Spin()
        self.q_epochs.setRange(1, 2000)
        self.q_batch = _Spin()
        self.q_batch.setRange(1, 256)
        self.q_lr = _DoubleSpin()
        self.q_lr.setDecimals(6)
        self.q_lr.setRange(0.000001, 1.0)
        self.q_lr.setSingleStep(0.0001)
        self.q_device = _Combo()   # 条目在 _fill_quick_controls 与 par_device 同步
        self.q_kshot = _Spin()      # 免训练范式行（V0.7.4 一.1）：free 显示
        self.q_kshot.setRange(1, 12)
        self.q_augcount = _Spin()
        self.q_augcount.setRange(0, 60)
        self.q_size = QLabel("")
        self.q_size.setObjectName("accentText")
        for _qw in (self.q_model, self.q_epochs, self.q_batch, self.q_lr,
                    self.q_device, self.q_kshot, self.q_augcount):
            _qw.setMinimumHeight(34)   # Quick 档大控件
        self._rec_badges: dict = {}    # 参数键 → 「已推荐」徽章
        self._preset_values: dict = {}
        self._preset_labels: dict = {}   # 卡 key → (标题, 描述) 标签

        # 分区一：基础配置（默认展开）
        self.sec_basic = CollapsibleSection(tr("基础配置"), expanded=True)
        fb = QFormLayout()
        fb.setContentsMargins(0, 0, 0, 0)
        self._form_basic = fb                     # §四：free 行隐藏用
        fb.addRow("训练模型", self.par_model)
        fb.addRow(self.par_transfer)
        fb.addRow("模型文件", transfer_row)
        fb.addRow("批量大小", self.par_batch)
        fb.addRow("训练轮次", self.par_epochs)
        fb.addRow("学习率", self.par_lr)
        fb.addRow("权重衰减", self.par_wd)
        fb.addRow("训练设备", self.par_device)
        fb.addRow(self.lbl_kshot, self.par_kshot)     # §四：仅 SubspaceAD 显示
        fb.addRow(self.lbl_augcount, self.par_augcount)
        self.sec_basic.add_layout(fb)
        v4.addWidget(self.sec_basic)

        # 分区二：数据与验证（默认折叠）
        self.sec_data = CollapsibleSection("数据与验证", expanded=False)
        fd = QFormLayout()
        fd.setContentsMargins(0, 0, 0, 0)
        self._form_data = fd                      # §四：free 行隐藏用
        fd.addRow("训练输入尺寸", self.input_size_hint)
        fd.addRow("网络输入尺寸（可选）", self.par_train_input)
        fd.addRow(self.par_tiling)
        fd.addRow("切图 tile 边长", self.par_tile_size)
        fd.addRow(self.par_neg)
        fd.addRow("训练集比例", self.par_train_ratio)
        fd.addRow("验证集比例", self.par_val_ratio)
        fd.addRow("数据倍增倍数", self.par_multiply)
        fd.addRow("早停耐心（轮）", self.par_patience)
        fd.addRow("评估间隔（轮）", self.par_eval_interval)
        fd.addRow(self.par_epoch_w)
        self.sec_data.add_layout(fd)
        v4.addWidget(self.sec_data)

        # 分区三：工业与加速优化（默认折叠）
        self.sec_adv = CollapsibleSection("工业与加速优化", expanded=False)
        fa = QFormLayout()
        fa.setContentsMargins(0, 0, 0, 0)
        self._form_adv = fa                       # §四：free 整区隐藏
        fa.addRow(self.par_industrial)
        fa.addRow(self.par_freeze)
        fa.addRow(self.par_rpn)
        fa.addRow(self.par_mask_light)
        fa.addRow(self.par_anchor)
        fa.addRow(self.par_slim_head)
        fa.addRow(self.par_p2)
        fa.addRow(self.par_gray)  # 颜色模式
        fa.addRow(self.par_det)   # 确定性训练
        fa.addRow("R-CNN 优化器", self.par_rcnn_opt)
        fa.addRow("R-CNN 学习率", self.par_rcnn_lr)
        fa.addRow("R-CNN 权重衰减", self.par_rcnn_wd)
        fa.addRow(self.par_amp)
        fa.addRow(self.par_tf32)
        self.sec_adv.add_layout(fa)

        # ---- Quick 面板：预设卡 + 6 参数行化表单 ----
        self.quick_panel = QWidget()
        qp = QVBoxLayout(self.quick_panel)
        qp.setContentsMargins(0, 4, 0, 0)
        qp.setSpacing(12)

        presets = QHBoxLayout()
        presets.setSpacing(10)
        self._preset_cards = {}
        for key, name, desc, values, color in _PRESET_DEFS:
            card = self._make_preset_card(key, name, desc, values, color)
            presets.addWidget(card)
            self._preset_cards[key] = card
        presets.addStretch(1)
        qp.addLayout(presets)

        qgrid = QGridLayout()
        qgrid.setHorizontalSpacing(24)
        qgrid.setVerticalSpacing(10)
        qgrid.addWidget(self._quick_row("训练模型", self.q_model), 0, 0)
        qgrid.addWidget(self._quick_row("训练设备", self.q_device), 0, 1)
        self._quick_epochs_row = self._quick_row(
            "训练轮次", self.q_epochs, "轮", badge_key="epochs")
        qgrid.addWidget(self._quick_epochs_row, 1, 0)
        self._quick_batch_row = self._quick_row(
            "批量大小", self.q_batch, "", badge_key="batch")
        qgrid.addWidget(self._quick_batch_row, 1, 1)
        self._quick_lr_row = self._quick_row(
            "学习率", self.q_lr, "", badge_key="lr")
        qgrid.addWidget(self._quick_lr_row, 2, 0)
        self._sync_quick_size()
        qgrid.addWidget(self._quick_row("输入尺寸", self.q_size, "",  # noqa: E501
                                        badge_key="size"), 2, 1)
        # 免训练范式行（V0.7.4 一.1）：free 模式显示，epoch 模式隐藏
        self._quick_kshot_row = self._quick_row(
            "支持集张数", self.q_kshot, "张", badge_key="k_shot")
        qgrid.addWidget(self._quick_kshot_row, 3, 0)
        self._quick_augcount_row = self._quick_row(
            "增强样本数", self.q_augcount, "", badge_key="aug_count")
        qgrid.addWidget(self._quick_augcount_row, 3, 1)
        self._quick_kshot_row.hide()
        self._quick_augcount_row.hide()
        qp.addLayout(qgrid)

        qtools = QHBoxLayout()
        self.btn_reset_defaults = QPushButton("↺ 恢复默认")
        self.btn_reset_defaults.setToolTip(
            "把 6 个核心参数恢复为打开项目时的默认值")
        self.btn_reset_defaults.clicked.connect(self._reset_quick_defaults)
        qtools.addWidget(self.btn_reset_defaults)
        qtools.addStretch(1)
        qp.addLayout(qtools)
        v4.addWidget(self.quick_panel)

        # ---- 专家区容器（完整折叠表单，专家模式显示）----
        self.expert_host = QWidget()
        ev = QVBoxLayout(self.expert_host)
        ev.setContentsMargins(0, 4, 0, 0)
        ev.setSpacing(4)
        ev.addWidget(self.sec_basic)
        ev.addWidget(self.sec_data)
        ev.addWidget(self.sec_adv)
        v4.addWidget(self.expert_host)

        # 初始模式（ui_prefs 持久化；默认快速档）
        self._set_param_mode(
            load_prefs().get("train_param_mode", "quick") != "expert",
            save=False)

        btns = QHBoxLayout()
        btns.addWidget(btn_save_par)
        btns.addWidget(self.btn_start)
        btns.addStretch(1)
        v4.addLayout(btns)
        lay.addWidget(g4)

        lay.addStretch(1)
        scroll.setWidget(body)
        # 纵向滚动条常显：折叠分区时滚动条出现/消失会改变视口宽度，
        # 造成整页控件回流闪动（"窗口瞬间变形"）
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll
    # ================= 信息 / 参数 =================
    def _browse_into(self, line_edit: QLineEdit, title: str):
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)
    def _fill_info_from_project(self):
        p = self.project
        self.info_name.setText(p.name)
        self.info_dataset.setText(p.dataset_dir)
        self.info_model_dir.setText(p.model_dir)
        dg = getattr(self, "data_grid", None)
        if dg is not None:      # 数据网格随项目刷新
            dg.set_dataset(p.dataset_dir)
        # 快照源字典：循环内 setChecked/setValue 会同步触发 _collect_augment_ops
        # 把 project.augment_ops 替换为当前 UI 状态，若边读边写，未处理的算子
        # 会被当作"不存在"而漏勾选（rotate90 行曾因此丢失）
        saved_ops = dict(p.augment_ops or {})
        saved_probs = dict(getattr(p, "augment_probs", None) or {})
        for op, (check, spins) in self.augment_controls.items():
            params = saved_ops.get(op)
            if params is None:
                check.setChecked(False)
            else:
                check.setChecked(True)
                for pname, spin in spins.items():
                    spin.setValue(params.get(pname, spin.value()))
            prob = self.augment_prob_spins.get(op)
            if prob is not None:  # 旧项目无 augment_probs 字段 → 回填默认 0.5
                prob.setValue(float(saved_probs.get(op, 0.5)))
    def _on_transfer_toggled(self):
        self.par_transfer_edit.setEnabled(self.par_transfer.isChecked())
    def _pick_transfer_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择迁移学习起始权重", "",
            "模型权重 (*.pt *.engine);;所有文件 (*)")
        if path:
            self.par_transfer_edit.setText(path)
            self.par_transfer.setChecked(True)
    def _fill_params_from_project(self):
        p = self.project
        self.par_model.blockSignals(True)
        self.par_model.clear()
        # 仅列出算法引擎的基准模型；迁移学习改为勾选 + 指定模型文件
        from ...engine.models import list_models  # torch 延迟导入

        for m in list_models(p.task):
            from ...i18n import tr

            self.par_model.addItem(
                f"{m['name']}  ({tr(m['desc'])})", m["name"])
        idx = self.par_model.findData(p.model_name)
        if idx < 0 and p.model_name:  # 旧项目记录的是 pruned:/run: 引用 → 回退基准名
            base = (p.model_name.split(":", 1)[-1].split("/", 1)[0])
            idx = self.par_model.findData(base)
        self.par_model.setCurrentIndex(max(idx, 0))
        self.par_model.blockSignals(False)
        self.par_transfer.setChecked(bool(p.transfer_file))
        self.par_transfer_edit.setText(p.transfer_file or "")
        self._on_transfer_toggled()
        self._refresh_input_size_hint()
        self.par_train_ratio.setValue(getattr(p, "train_ratio", 0.8) or 0.8)
        self.par_val_ratio.setValue(p.val_ratio)
        self.par_multiply.setValue(p.multiply_factor)
        self.par_batch.setValue(p.batch_size)
        self.par_epochs.setValue(p.epochs)
        self.par_lr.setValue(p.learning_rate)
        self.par_wd.setValue(p.weight_decay)
        self.par_eval_interval.setValue(p.eval_interval)
        dev_idx = self.par_device.findData(p.device)
        self.par_device.setCurrentIndex(max(dev_idx, 0))
        self.par_industrial.setChecked(getattr(p, "industrial_opt", True))
        self.par_freeze.setChecked(getattr(p, "freeze_backbone", False))
        self.par_rpn.setChecked(getattr(p, "rpn_reduced", False))
        self.par_mask_light.setChecked(getattr(p, "mask_head_light", False))
        self.par_anchor.setChecked(getattr(p, "anchor_small", False))
        self.par_slim_head.setChecked(getattr(p, "slim_head", False))
        self.par_p2.setChecked(getattr(p, "p2_head", False))
        _ri = self.par_rcnn_opt.findData(
            getattr(p, "rcnn_optimizer", "adamw"))
        self.par_rcnn_opt.setCurrentIndex(max(_ri, 0))
        self.par_rcnn_lr.setValue(getattr(p, "rcnn_learning_rate", 0.0))
        self.par_rcnn_wd.setValue(getattr(p, "rcnn_weight_decay", 0.05))
        self.par_amp.setChecked(getattr(p, "use_amp", True))
        self.par_tf32.setChecked(getattr(p, "tf32", True))
        self.par_epoch_w.setChecked(getattr(p, "save_epoch_weights", True))
        self.par_patience.setValue(getattr(p, "patience", 10))
        self.par_kshot.setValue(int(getattr(p, "k_shot", 4) or 4))
        self.par_augcount.setValue(int(getattr(p, "aug_count", 30) or 30))
        self.par_train_input.setValue(getattr(p, "train_input", 0))
        self.par_tiling.setChecked(getattr(p, "tiling", False))
        self.par_tile_size.setValue(getattr(p, "tile_size", 640))
        self.par_neg.setChecked(getattr(p, "use_negatives", True))
        self.par_gray.setChecked(getattr(p, "color_mode", "color") == "gray")
        self.par_det.setChecked(getattr(p, "deterministic", False))
        # Quick 档控件同步填充 + 默认值快照（「恢复默认」与推荐徽章复位用）
        self._fill_quick_controls()
        self._param_defaults = {
            "model": self.par_model.currentData(),
            "epochs": self.par_epochs.value(),
            "batch": self.par_batch.value(),
            "lr": self.par_lr.value(),
            "device": self.par_device.currentData(),
        }
        self._clear_recommended()
        self._apply_train_mode_ui()   # §四：按项目模型形态调整参数区
    def _collect_augment_ops(self):
        """从界面控件收集增强策略与每算子启用概率（勾选的算子才生效）。"""
        ops = {}
        probs = {}
        for op, (check, spins) in self.augment_controls.items():
            if not check.isChecked():
                continue
            ops[op] = {name: spin.value() for name, spin in spins.items()}
            probs[op] = self.augment_prob_spins[op].value()
        self.project.augment_ops = ops
        self.project.augment_probs = probs
        return ops
    def _save_info(self):
        from ...i18n import tr

        p = self.project
        old_name = p.name
        p.name = self.info_name.text().strip()
        p.dataset_dir = self.info_dataset.text().strip()
        p.model_dir = self.info_model_dir.text().strip()
        try:
            save_project(p, old_name=old_name)
        except ValueError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self.setWindowTitle(f"{p.name}  -  {tr('训练平台')}")
        self.det_title.setText(p.name)
        dg = getattr(self, "data_grid", None)
        if dg is not None:      # 数据集路径可能被修改
            dg.set_dataset(p.dataset_dir)
        if self.on_renamed is not None:
            self.on_renamed(old_name, p.name)
        toast(self, tr("项目信息已保存"), "success")
    def _save_params(self):
        if not self.quick_panel.isHidden():
            # Quick 模式下用户可能改过快速档控件：先回写专家表单再保存
            self._experts_from_quick()
        p = self.project
        p.model_name = self.par_model.currentData() or p.model_name
        p.transfer_file = (self.par_transfer_edit.text().strip()
                           if self.par_transfer.isChecked() else "")
        p.val_ratio = self.par_val_ratio.value()
        p.train_ratio = self.par_train_ratio.value()
        p.multiply_factor = self.par_multiply.value()
        p.batch_size = self.par_batch.value()
        p.epochs = self.par_epochs.value()
        p.learning_rate = self.par_lr.value()
        p.weight_decay = self.par_wd.value()
        p.eval_interval = self.par_eval_interval.value()
        p.device = self.par_device.currentData()
        p.industrial_opt = self.par_industrial.isChecked()
        p.freeze_backbone = self.par_freeze.isChecked()
        p.rpn_reduced = self.par_rpn.isChecked()
        p.mask_head_light = self.par_mask_light.isChecked()
        p.anchor_small = self.par_anchor.isChecked()
        p.slim_head = self.par_slim_head.isChecked()
        p.p2_head = self.par_p2.isChecked()
        p.rcnn_optimizer = self.par_rcnn_opt.currentData()
        p.rcnn_learning_rate = self.par_rcnn_lr.value()
        p.rcnn_weight_decay = self.par_rcnn_wd.value()
        p.use_amp = self.par_amp.isChecked()
        p.tf32 = self.par_tf32.isChecked()
        p.save_epoch_weights = self.par_epoch_w.isChecked()
        p.patience = self.par_patience.value()
        p.train_input = self.par_train_input.value()
        p.tiling = self.par_tiling.isChecked()
        p.tile_size = self.par_tile_size.value()
        p.use_negatives = self.par_neg.isChecked()
        p.color_mode = "gray" if self.par_gray.isChecked() else "color"
        p.deterministic = self.par_det.isChecked()
        # §四：免训练范式自有参数（k-shot 上限 12，既定口径）
        p.k_shot = int(self.par_kshot.value())
        p.aug_count = int(self.par_augcount.value())
        if p.train_input > 960:
            # opt-in 大输入提示（显存显著上升）
            self.statusBar().showMessage(
                tr("⚠ 网络输入尺寸 >960：训练显存占用显著上升，")
                + tr("请确认显卡预算（或降低该值 / 开启 AMP 与梯度检查点）"), 15000)
        self._collect_augment_ops()
        p.save()
        toast(self, tr("参数已保存"), "success")
    def _save_augment(self):
        self._collect_augment_ops()
        self.project.save()
        toast(self, tr("增强策略已保存"), "success")
    def _on_model_changed(self):
        """切换模型时带入建议学习率。"""
        name = self.par_model.currentData()
        if name and name in DEFAULT_LR:
            self.par_lr.setValue(DEFAULT_LR[name])
        if hasattr(self, "augment_controls"):
            self._refresh_input_size_hint()   # P2 显存提示只对 YOLO 模型显示
        self._apply_train_mode_ui()
    # ---- §四（V0.7.3）：免训练范式 UI 适配 ----
    def _train_mode_of(self, name) -> str:
        """模型训练形态：free（免训练）/ epoch（默认轮次式）。"""
        try:
            from ...engine.models.registry import MODEL_META

            return str(MODEL_META.get(name or "", {}).get("train_mode")
                       or "epoch")
        except Exception:  # noqa: BLE001  元数据不可得按轮次式
            return "epoch"
    def _support_k_shot(self, name) -> bool:
        try:
            from ...engine.models.registry import MODEL_META

            return bool(MODEL_META.get(name or "", {}).get("support_k_shot",
                                                           False))
        except Exception:  # noqa: BLE001
            return False
    def _set_form_row_visible(self, form, widget, visible: bool):
        """QFormLayout 行整体显隐（控件 + 行标签；直接加 checkBox 行无标签）。"""
        widget.setVisible(visible)
        try:
            lbl = form.labelForField(widget)
        except Exception:  # noqa: BLE001
            lbl = None
        if lbl is not None:
            lbl.setVisible(visible)
    def _apply_train_mode_ui(self):
        """按所选模型的 train_mode 调整参数区（§四：免训练范式
        不再展示轮次类参数，避免"进度按 50 轮爬行/Loss 判读 6 轮"式歧义）。

        free：隐藏 批量/轮次/学习率/权重衰减 与 倍增/早停/评估间隔/逐轮
        权重，整区隐藏 工业与加速优化；k-shot 模型显示 支持集张数/增强
        样本数；Quick 档强制切专家档（快速档 6 参数行对 free 无意义）。
        """
        name = getattr(getattr(self, "par_model", None), "currentData",
                       lambda: None)()
        free = self._train_mode_of(name) == "free"
        self._train_mode = "free" if free else "epoch"
        if not hasattr(self, "_form_basic"):
            return
        for w in (self.par_batch, self.par_epochs, self.par_lr, self.par_wd):
            self._set_form_row_visible(self._form_basic, w, not free)
        for w in (self.par_multiply, self.par_patience, self.par_eval_interval,
                  self.par_epoch_w):
            self._set_form_row_visible(self._form_data, w, not free)
        self.sec_adv.setVisible(not free)
        kshot = free and self._support_k_shot(name)
        for w in (self.lbl_kshot, self.par_kshot, self.lbl_augcount,
                  self.par_augcount):
            w.setVisible(kshot)
        # Quick 档免训练适配（V0.7.4 bug-优化项一.1）：不再强制切专家档——
        # 隐藏轮次/批量/学习率行、显示支持集张数/增强样本数行、预设卡换
        # 免训练语义（k-shot 模型）；非 k-shot 的免训练模型无卡可换（隐藏）
        if getattr(self, "quick_panel", None) is not None:
            has_rows = hasattr(self, "_quick_epochs_row")
            if has_rows:
                for row in (self._quick_epochs_row, self._quick_batch_row,
                            self._quick_lr_row):
                    row.setVisible(not free)
                for row in (self._quick_kshot_row, self._quick_augcount_row):
                    row.setVisible(kshot)
            if hasattr(self, "_preset_cards"):
                show_cards = (not free) or kshot
                for card in self._preset_cards.values():
                    card.setVisible(show_cards)
                if free and kshot:
                    self._apply_preset_cards(_PRESET_DEFS_FREE)
                elif not free:
                    self._apply_preset_cards(_PRESET_DEFS)
    # ---- 训练输入尺寸（与增强 resize 统一） ----
    def _current_resize(self) -> tuple[int, int] | None:
        """当前增强策略里的 resize 宽高（未启用返回 None）。"""
        check, spins = self.augment_controls.get("resize", (None, {}))
        if check is None or not check.isChecked():
            return None
        return int(spins["width"].value()), int(spins["height"].value())
    def _refresh_input_size_hint(self):
        from ...i18n import tr

        size = self._current_resize()
        if size:
            self.input_size_hint.setText(
            f"{size[0]} × {size[1]}　"
            f"{tr('（随增强策略 resize 设置）')}")
        else:
            self.input_size_hint.setText(
                f"{tr('模型默认')}　{tr('（未启用 resize，分类将拉伸为方形）')}")
        name = getattr(getattr(self, "par_model", None), "currentData", lambda: None)() or ""
        p2 = getattr(self, "par_p2", None)
        if (isinstance(name, str) and name.startswith("yolo")
                and p2 is not None and p2.isChecked()):
            tip = self.input_size_hint.text()
            self.input_size_hint.setText(tip + f"　· {tr('P2 小目标头：显存约 ×2，建议减小批量')}")
        if getattr(self, "q_size", None) is not None:
            self._sync_quick_size()   # Quick 档「输入尺寸」行同步
    @staticmethod
    def _divider():
        """浅色分隔线：用于参数表单的逻辑分区。"""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background: #eef2f8; border: none; max-height: 1px;")
        return line
    # ================= 数据集统计 =================
    def _random_dataset_image(self) -> Path | None:
        """从数据集目录随机抽一张图片。"""
        root = Path(self.project.dataset_dir)
        if not root.is_dir():
            return None
        paths = [p for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                 and not p.name.startswith(".")]
        return random.choice(paths) if paths else None
    def _refresh_augment_preview(self, new_image: bool = False):
        """随机抽取数据集图片，按当前全部启用增强算子预览（不再只
        预览 resize；每算子按启用概率掷硬币，与训练侧 maybe_apply_ops 同口径）。

        预览信息展示原图尺寸、resize 产出与本次实际生效的算子，用户可反复
        「换一张」观察各算子的视觉效果。"""
        if new_image or getattr(self, "_preview_source", None) is None:
            path = self._random_dataset_image()
            img = imread_unicode(path) if path else None
            if img is None:
                self._preview_source = None
                self.augment_preview.setPixmap(QPixmap())
                self.augment_preview.setText(tr("数据集中没有可预览的图片"))
                self.preview_info.setText("")
                return
            self._preview_source = img
            self._preview_name = path.name
        src = self._preview_source
        src_h, src_w = src.shape[:2]
        ops = self._collect_augment_ops()      # 同保存口径：勾选的算子才生效
        probs = dict(getattr(self.project, "augment_probs", None) or {})
        from ...engine.dataset.transforms import preview_apply_ops

        out, applied = preview_apply_ops(src.copy(), ops, probs)
        out_h, out_w = out.shape[:2]
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, out_w, out_h, 3 * out_w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg)
        area = self.augment_preview.size()
        if area.width() > 40 and area.height() > 40:  # 自适应：等比缩放到预览区大小
            pm = pm.scaled(area, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.augment_preview.setPixmap(pm)
        resize_on = "resize" in ops
        tag = (f"{tr('拉伸 → ')} {out_w}×{out_h}" if resize_on
               else tr("未启用 resize"))
        others = [a for a in applied if a != "resize"]
        if others:
            tag += f"　{tr('本次生效：')}{'、'.join(others)}"
        elif not resize_on:
            tag = tr("未启用增强，显示原图")
        self.preview_info.setText(
            f"{getattr(self, '_preview_name', '')}　{tr('原始')} "
            f"{src_w}×{src_h}　{tag}")
    def _refresh_stats(self):
        """数据集统计（移后台线程，主线程不再被万图目录卡住）。"""
        if getattr(self, "_stats_thread", None) is not None:
            return  # 统计中，忽略重复触发
        from ...i18n import tr

        self.stats_hint.setText(tr("统计中…"))
        task = self.project.task_type
        dataset_dir = Path(self.project.dataset_dir)

        class _StatsThread(QThread):
            done = pyqtSignal(object)
            failed = pyqtSignal(str)

            def run(self):
                try:
                    self.done.emit(collect_stats(task, dataset_dir))
                except Exception as e:
                    log.debug("数据集统计线程失败（failed 信号上报）: %s", e)
                    self.failed.emit(f"{e}")

        # 无父对象：窗口可能在线程运行中被关闭——parented 运行线程
        # 会触发 Qt "Destroyed while thread is still running" 崩溃；
        # finished→auto_cleanup_thread 已接 deleteLater 兜底
        th = _StatsThread()
        th.done.connect(self._on_stats_done)
        th.failed.connect(self._on_stats_failed)
        auto_cleanup_thread(self, "_stats_thread", th)
        self._stats_thread = th
        th.start()
    def _on_stats_done(self, stats):
        from ...i18n import tr

        n_ok = getattr(stats, "n_ok", 0)
        total = stats.n_images + n_ok
        ok_part = ""
        if n_ok and total:
            pct = n_ok / total
            ok_part = f"，{tr('OK 负样本')} {n_ok} {tr('张')}"
        # 忽略区域计数可见（与数据体检提示同口径）
        n_ign = getattr(stats, "n_ignore", 0)
        if n_ign:
            ok_part += f"，__ignore__ {tr('区域')} {n_ign} {tr('个')}"
        self.stats_hint.setText(
            f"{tr('共')} {total} {tr('张图片')}，{tr('已标注')} "
            f"{stats.n_labeled} {tr('张')}，"
            f"{tr('类别')} {len(stats.class_counts)} {tr('个')}{ok_part}"
        )
        self.dist_canvas.update_stats(stats)
        # 划分环形图（n_ok 含负样本；分类任务 n_ok=0）
        # 缓存计数供划分参数变动时实时重绘
        self._split_counts = (stats.n_labeled, getattr(stats, "n_ok", 0))
        self.split_canvas.update_split(
            stats.n_labeled, getattr(stats, "n_ok", 0),
            float(getattr(self.project, "train_ratio", 0.0) or 0.0),
            float(self.project.val_ratio or 0.0))
    def _refresh_split_canvas(self, *_):
        """训练集/验证集比例变动 → 环状图按新参数即时重绘（V0.7.1）。"""
        counts = getattr(self, "_split_counts", None)
        if not counts or not hasattr(self, "par_train_ratio"):
            return   # 统计未就绪（项目刚打开）：统计完成后会刷一次
        self.split_canvas.update_split(
            counts[0], counts[1],
            float(self.par_train_ratio.value()),
            float(self.par_val_ratio.value()))
    def _on_stats_failed(self, msg):
        from ...i18n import tr

        self.stats_hint.setText("")
        QMessageBox.warning(self, tr("统计失败"), f"{tr('读取数据集失败：')}{msg}")
    # ================= 数据体检 / 预检 / 推荐 / 开发者选项 =================
    def _open_dev_options(self):
        DevOptionsDialog(self).exec_()
    def _run_health_report(self):
        """数据体检报告（异步扫描；仅提示/警告两档 + 去标注平台处理闭环）。"""
        from ...i18n import tr

        worker = getattr(self, "_health_worker", None)
        if worker is not None and worker.isRunning():
            self.statusBar().showMessage(tr("数据体检正在进行中，请稍候…"), 3000)
            return
        self.btn_health.setEnabled(False)
        self.statusBar().showMessage(tr("数据体检中…（大集合的重复图检测可能稍慢）"))
        # 无父对象：以页为父的运行线程会随页销毁被 Qt 删除，
        # 触发 "Destroyed while thread is still running"（auto_cleanup_thread
        # 的模块级保活负责持有至 finished）
        self._health_worker = _PreflightWorker(self.project)
        self._health_worker.done.connect(self._on_health_done)
        self._health_worker.failed.connect(self._on_health_failed)
        auto_cleanup_thread(self, "_health_worker", self._health_worker)  # A-3
        self._health_worker.start()
    def _on_health_done(self, result: dict):
        from ...i18n import tr

        self.btn_health.setEnabled(True)
        self._health_worker = None
        rows = [ReportRow.from_issue(i) for i in result["issues"]]
        dlg = HealthReportDialog(
            self, tr("数据体检报告"), rows, mode="report",
            summary=f"{tr('共')} {result['n_images']} {tr('张样本')}、"
                    f"{len(result['classes'])} {tr('个类别')}。"
                    + tr("结果仅分「提示 / 警告」两档，不会阻止训练；"
                         "阈值可在 ⚙ 开发者选项调整。"),
            on_annotate=self._goto_annotation)
        dlg.exec_()
    def _on_health_failed(self, message: str):
        from ...i18n import tr

        self.btn_health.setEnabled(True)
        self._health_worker = None
        QMessageBox.warning(self, tr("数据体检失败"),
                            f"{tr('读取数据集失败：')}{message}")
    def _goto_annotation(self):
        """体检闭环：拉起标注平台并定位到数据集目录（launcher 参数透传）。"""
        from ... import launcher

        dataset = self.project.dataset_dir or None
        launcher.launch_platform("annotation", dataset)
        self.statusBar().showMessage(tr("已打开标注平台（定位到数据集目录）"), 5000)
    # ================= 训练 =================
    def _check_rotate_guard(self):
        """rotate90 守卫（设计规则）：数据集长宽比不一致时，旋转算子必须配合
         「随机裁剪」且启用概率 100%，否则无法开始训练。

        返回 None=通过；("enable", 提示文本)=需启用裁剪兜底；
        ("prob", 提示文本)=裁剪已启用但概率不足 100%。
        """
        ops = self.project.augment_ops or {}
        if "rotate90" not in ops:
            return None
        aspects = set()
        for s in self._dataset_samples_for_guard():
            if s.width > 0 and s.height > 0:
                aspects.add(round(s.width / s.height, 3))
        return rotate_guard_decision(ops, self.project.augment_probs, aspects)
    def _dataset_samples_for_guard(self):
        """守卫用的样本元信息（宽高来自标注 JSON / 目录扫描，不解码图片）。"""
        from ...engine.dataset.base import TaskType
        from ...engine.dataset.reader import scan_classification_dataset, \
            scan_labelme_dataset

        d = Path(self.project.dataset_dir)
        if not d.is_dir():
            return []
        try:
            if TaskType(self.project.task) in (TaskType.CLASSIFICATION,
                                               TaskType.ANOMALY):
                # 异常检测同分类式目录扫描（OK/NG 纯图）
                samples, _ = scan_classification_dataset(d)
                return samples
            samples, _, _ = scan_labelme_dataset(d)
            return samples
        except Exception:
            return []  # 扫描失败交由训练线程的既有校验兜底
    def _resolve_rotate_guard(self, guard: tuple) -> bool:
        """rotate90 守卫弹窗：True=已按建议写回配置可继续训练；False=用户取消。

        一键兜底：启用随机裁剪并按 resize 目标宽高比设定 ratio、启用概率
        置 100%，写回项目后同步界面控件。
        """
        kind, text = guard
        from ...i18n import tr

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(tr("无法开始训练"))
        box.setText(text)
        btn_fix = box.addButton(
            tr("启用随机裁剪（100%）并开始") if kind == "enable"
            else tr("设为 100% 并开始"),
            QMessageBox.AcceptRole)
        box.addButton(tr("取消"), QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not btn_fix:
            return False
        check, spins = self.augment_controls["random_crop"]
        check.setChecked(True)
        size = self._current_resize()
        ratio = round(size[0] / size[1], 2) if size and size[1] else 1.0
        spins["ratio"].setValue(ratio)
        self.augment_prob_spins["random_crop"].setValue(1.0)
        self._collect_augment_ops()
        self.project.save()
        self.statusBar().showMessage(
            f"{tr('已启用随机裁剪兜底（宽高比')} {ratio:g}，{tr('概率 100%）')}，"
            f"{tr('继续训练')}", 5000)
        return True


# ============================================================
# 守卫决策（模块级纯函数，测试与守卫共用）
# ============================================================
def rotate_guard_decision(ops: dict, probs: dict, aspects: set) -> tuple | None:
    """rotate90 守卫决策（纯函数，便于测试）。

    设计规则：数据集长宽比不一致时，旋转算子必须配合「随机裁剪（对齐目标
    宽高比）」且启用概率 100%。返回 None=通过；("enable", 文本)=需启用兜底；
    ("prob", 文本)=兜底已启用但概率不足。
    """
    if "rotate90" not in ops:
        return None
    if len(aspects) <= 1:
        return None  # 长宽比一致：旋转后仍一致，训练输入形状稳定
    if "random_crop" in ops:
        crop_prob = float((probs or {}).get("random_crop", 0.5))
        if crop_prob >= 0.999:
            return None
        return ("prob",
                tr("「随机裁剪」作为旋转兜底时启用概率必须为 100%，\n"
                   "否则部分样本旋转后长宽比不一致，训练输入形状混乱。"))
    shown = "、".join(f"{a:g}" for a in sorted(aspects)[:5])
    if len(aspects) > 5:
        shown += "…"
    return ("enable",
            tr("已启用「随机旋转 90°/270°」，但数据集图像长宽比不一致")
            + f"（{len(aspects)} {tr('种')}：{shown}）。\n"
            + tr("旋转后宽高互换会让训练输入比例混乱。请启用「随机裁剪」"
                 "作为兜底，或关闭旋转算子。"))

# ---- Cython 编译态槽包装：mixin 方法经继承进入窗口类，编译态同样需要
# 参数裁剪包装（见 xccv/qt_slots.py；幂等）----
from xccv.qt_slots import apply_auto_slot_class  # noqa: E402

apply_auto_slot_class(InfoPageMixin)



