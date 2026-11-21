"""页 1「训练可视化」页（TrainPageMixin）：页面构建（历史列表+进度+
曲线+逐类表+参数回显）、训练启动（预检流程）、停止与输出目录。

0.7.6 P-1 页级拆分：方法体与拆分前 training_window.py 完全一致
（逐行搬运）；组合根 ProjectWindow 经 mixin 继承装配。
"""

import os
import time
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget, QMessageBox, QProgressBar, QPushButton, QScrollArea, QTableWidget, QVBoxLayout, QWidget,
)
from ...gui_utils import auto_cleanup_thread
from ...i18n import tr
from ..charts import ConfusionMatrixCanvas, DistributionCanvas, SplitDonutCanvas
from ..data_health import LEVEL_WARN, run_health_check
from ..dialogs import CollapsibleSection, DevOptionsDialog, HealthReportDialog, ReportRow
from ..dev_options import load_options
from ...ui_style import bind_style, icon_path, token
from ..trainer_thread import TrainThread
from .common import (
    _safe_run_tag, _IntAxis, _PagePlot, _style_plot, _add_legend, _PlotHover, _PreflightWorker,
)


# ============================================================
# 页面方法（自 training_window.py 逐行搬运，语义不变）
# ============================================================

class TrainPageMixin:
    # ---- 页 1：训练可视化（左：历史记录列表；右：所选记录详情）----
    def _build_train_page(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(12)

        # 左：训练记录列表（读取模型储存目录下的历次训练）
        panel = QFrame()
        panel.setObjectName("runPanel")
        panel.setFixedWidth(280)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(10, 10, 10, 10)
        pv.setSpacing(8)
        cap = QLabel("训练记录")
        bind_style(cap, lambda: f"font-weight: bold; color: {token('primary_deep')};"
                                " border: none; background: transparent;")
        pv.addWidget(cap)
        self.btn_compare = QPushButton("对比所选")
        self.btn_compare.setToolTip("按住 Ctrl 选择 2~3 条记录后对比")
        self.btn_compare.clicked.connect(self._open_compare)  # 常启用：点击时校验并提示
        pv.addWidget(self.btn_compare)
        self.run_hint = QLabel("")
        self.run_hint.setObjectName("hint")
        self.run_hint.setWordWrap(True)
        pv.addWidget(self.run_hint)
        self.run_list = QListWidget()
        # ExtendedSelection：单击切换详情、Ctrl+点击多选；
        # 多选上限 3 条，超出的新选择被拒绝并提示（见 _on_run_selection_changed）
        self.run_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._prev_selected: list = []   # 上一次合法选中的路径集合（超限回退用）
        self.run_list.currentItemChanged.connect(self._on_run_selected)
        self.run_list.itemSelectionChanged.connect(self._on_run_selection_changed)
        pv.addWidget(self.run_list, stretch=1)
        # 空状态占位（与列表同槽互斥显示）
        from ...ui_widgets import EmptyState as _EmptyState

        self.run_empty = _EmptyState("🗂", "暂无训练记录",
                                     "完成一次训练后，记录会自动出现在这里")
        self.run_empty.hide()
        pv.addWidget(self.run_empty, stretch=1)
        lay.addWidget(panel)

        # 右：详情（滚动区；画布滚轮已透传，鼠标悬停画布也能直接滚动页面）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        lay2 = QVBoxLayout(body)
        lay2.setContentsMargins(4, 0, 4, 0)
        lay2.setSpacing(12)

        head = QHBoxLayout()
        self.train_status = QLabel("尚未开始训练")
        self.train_status.setStyleSheet("font-weight: bold;")
        self.gpu_mem = QLabel("")
        self.gpu_mem.setObjectName("muted")
        self.btn_stop = QPushButton("■ 停止训练")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_open_dir = QPushButton("打开输出目录")
        self.btn_open_dir.setEnabled(False)
        self.btn_open_dir.clicked.connect(self._open_output_dir)
        head.addWidget(self.train_status)
        head.addWidget(self.gpu_mem)
        head.addStretch(1)
        head.addWidget(self.btn_stop)
        head.addWidget(self.btn_open_dir)
        lay2.addLayout(head)

        # 训练进度：开始时间 / 结束时间 / 训练时长 + 百分比进度条
        g0 = QGroupBox("训练进度")
        gd = QGridLayout(g0)
        gd.setHorizontalSpacing(10)
        gd.setVerticalSpacing(8)
        self.run_start = QLabel("—")
        self.run_end = QLabel("—")
        self.run_duration = QLabel("—")
        for col, text in enumerate(("开始时间", "结束时间", "训练时长")):
            key = QLabel(text)
            key.setObjectName("muted")
            gd.addWidget(key, 0, col * 2)
        for col, val in enumerate((self.run_start, self.run_end, self.run_duration)):
            gd.addWidget(val, 0, col * 2 + 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #d7e2f0; border-radius: 6px;"
            " background: #ffffff; text-align: center; color: #1f2937;"
            " min-height: 20px; font-weight: bold; font-size: 15px; }"
            "QProgressBar::chunk { background-color: #5ec6dd; border-radius: 5px; }")
        gd.addWidget(self.progress_bar, 1, 0, 1, 6)
        lay2.addWidget(g0)

        # 训练曲线：悬停显示最近轮次的数值（三位小数）
        g1 = QGroupBox("训练曲线")
        v1 = QVBoxLayout(g1)
        self.loss_plot = _PagePlot(axisItems={"bottom": _IntAxis(orientation="bottom")})
        _add_legend(self.loss_plot)
        self.loss_plot.setLabel("bottom", tr("轮次"))
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.setMouseEnabled(x=False, y=False)
        self.loss_plot.hideButtons()
        self.metric_plot = _PagePlot(axisItems={"bottom": _IntAxis(orientation="bottom")})
        _add_legend(self.metric_plot)
        self.metric_plot.setLabel("bottom", tr("轮次"))
        self.metric_plot.showGrid(x=True, y=True, alpha=0.3)
        self.metric_plot.setMouseEnabled(x=False, y=False)
        self.metric_plot.hideButtons()
        self.metric_plot.setYRange(0, 1, padding=0)
        _style_plot(self.loss_plot, tr("损失曲线"))
        _style_plot(self.metric_plot, tr("评估指标"))
        self.metric_hint = QLabel(
            "mAP/mAR：R-CNN 族为 IoU 0.10/0.25/0.50 三档平均；"
            "YOLO 族为 AP@50 与召回率；分类为 Top-1")
        self.metric_hint.setObjectName("hint")
        v1.addWidget(self.metric_hint)
        # §四（V0.7.3）：免训练范式 Loss 区口径提示（无损失曲线语义）
        self.loss_free_hint = QLabel(
            tr("免训练范式无损失曲线；下方指标曲线为 AUROC"))
        self.loss_free_hint.setObjectName("hint")
        self.loss_free_hint.setVisible(False)
        v1.addWidget(self.loss_free_hint)
        v1.addWidget(self.loss_plot)
        v1.addWidget(self.metric_plot)
        lay2.addWidget(g1)
        self._loss_hover = _PlotHover(self.loss_plot, self._loss_series)
        self._metric_hover = _PlotHover(self.metric_plot, self._metric_series)

        g2 = QGroupBox("混淆矩阵")
        v2 = QVBoxLayout(g2)
        self.cm_canvas = ConfusionMatrixCanvas()
        v2.addWidget(self.cm_canvas)
        # 逐类精确率/召回率表（对照商用视觉软件的类型详情表）
        self.per_class_table = QTableWidget(0, 9)
        self.per_class_table.setHorizontalHeaderLabels(
            ["类别", "GT 实例", "检出", "TP", "FP", "FN", "精确率", "召回率", "AP@50"])
        # 取消末列拉伸——全屏后还原窗口时 AP@50 列被异常拉宽；
        # 列宽按内容自适应，9 列总宽通常小于视口，窄窗口出比例正常的横向滚动条
        header = self.per_class_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.per_class_table.verticalHeader().setVisible(False)
        self.per_class_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.per_class_table.setMaximumHeight(190)
        # 横向按像素平滑滚动，滑块拖动手感正常（此前逐格跳变难拖）
        self.per_class_table.setHorizontalScrollMode(
            QAbstractItemView.ScrollPerPixel)
        self.per_class_table.setVerticalScrollMode(
            QAbstractItemView.ScrollPerPixel)
        self.per_class_table.setToolTip(tr(
            "口径说明：\n"
            "· TP/FP/FN 来自 IoU≥0.5 同类匹配混淆矩阵（训练收尾统计）；\n"
            "· 精确率=TP/检出，召回率=TP/GT 实例；检出=TP+FP，GT 实例=TP+FN；\n"
            "· AP@50 为 COCO 口径逐类平均（与汇总 mAP@50 同口径）；\n"
            "· 曲线/摘要的 mAP・mAR 为宽松口径（IoU 0.10/0.25/0.50 三档平均），\n"
            "  两套口径并存，调优时先确认目标再对号入座；\n"
            "· \"-\" 表示该类无样本或无有效评估组合。"))
        v2.addWidget(self.per_class_table)
        lay2.addWidget(g2)

        g3 = QGroupBox("本次训练参数")
        v3 = QVBoxLayout(g3)
        self.echo_form = QFormLayout()
        v3.addLayout(self.echo_form)
        self.final_metrics = QLabel("")
        bind_style(self.final_metrics,
                   lambda: f"color: {token('primary_deep')}; font-weight: bold;")
        self.final_metrics.setWordWrap(True)
        v3.addWidget(self.final_metrics)
        lay2.addWidget(g3)

        # 底部：删除本条历史记录（含输出文件夹）
        del_row = QHBoxLayout()
        self.btn_delete_run = QPushButton("删除本条训练记录（含输出文件夹）")
        self.btn_delete_run.setObjectName("danger")
        self.btn_delete_run.setEnabled(False)
        self.btn_delete_run.clicked.connect(self._delete_current_run)
        del_row.addStretch(1)
        del_row.addWidget(self.btn_delete_run)
        lay2.addLayout(del_row)

        lay2.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, stretch=1)
        return page
    def _start_training(self):
        from ...i18n import tr

        self._save_params()
        guard = self._check_rotate_guard()
        if guard and not self._resolve_rotate_guard(guard):
            return
        # 训练前预检（）：异步执行数据体检 + 参数检查。
        # 设计规则：尽量少弹窗——仅存在「警告」级条目时弹一个对话框；
        # 只有「提示」时状态栏一句话直接开始；预检失败不阻塞训练。
        if not load_options().preflight_enabled:
            self._begin_training()
            return
        self.btn_start.setEnabled(False)
        self.statusBar().showMessage(tr("训练前预检中（数据体检 + 参数检查）…"))
        self._preflight_worker = _PreflightWorker(self.project, self)
        self._preflight_worker.done.connect(self._on_preflight_done)
        self._preflight_worker.failed.connect(self._on_preflight_failed)
        auto_cleanup_thread(self, "_preflight_worker", self._preflight_worker)  # A-3
        self._preflight_worker.start()
    def _apply_preflight_value(self, key, value):
        """预检「一键应用」：把建议值写回训练表单（保存发生在开始训练时）。"""
        if key == "patience":
            self.par_patience.setValue(int(value))
        elif key == "learning_rate":
            self.par_lr.setValue(float(value))
        elif key == "batch_size":
            self.par_batch.setValue(int(value))
        elif key == "device":
            idx = self.par_device.findData(value)
            if idx >= 0:
                self.par_device.setCurrentIndex(idx)
        elif key == "resize":
            check, spins = self.augment_controls["resize"]
            check.setChecked(True)
            spins["width"].setValue(int(value[0]))
            spins["height"].setValue(int(value[1]))
    def _on_preflight_done(self, result: dict):
        from ...i18n import tr

        self.btn_start.setEnabled(True)
        self._preflight_worker = None
        rows = [ReportRow.from_issue(i) for i in result["issues"]]
        rows += [ReportRow.from_finding(f) for f in result["findings"]]
        has_warn = any(r.level == LEVEL_WARN for r in rows)
        # 弹窗条件：存在警告级，或（用户在开发者选项开启了"提示级也弹窗"）。
        # 两者都不满足（完全干净）时状态栏一句话放行，不打扰。
        opt = load_options()
        if not has_warn and not (opt.preflight_info_popup and rows):
            self.statusBar().showMessage(tr("预检通过，开始训练…"), 5000)
            self._begin_training()
            return
        dlg = HealthReportDialog(
            self, tr("训练前预检"), rows, mode="preflight",
            summary=f"{tr('数据集共')} {result['n_images']} {tr('张样本')}、"
                    f"{len(result['classes'])} {tr('个类别')}。"
                    + tr("以下为「提示 / 警告」两档建议，不会强制修改；"
                         "折叠区可查看已应用的参数。"),
            on_apply=self._apply_preflight_value)
        if dlg.exec_() != QDialog.Accepted:
            self.statusBar().showMessage(tr("已取消训练"), 3000)
            return
        self._begin_training()
    def _on_preflight_failed(self, message: str):
        from ...i18n import tr

        self.btn_start.setEnabled(True)
        self._preflight_worker = None
        box = QMessageBox(QMessageBox.Warning, tr("预检失败"),
                          f"{tr('训练前预检执行失败：')}{message}\n\n"
                          f"{tr('仍要直接开始训练吗？')}",
                          QMessageBox.Yes | QMessageBox.No, self)
        if box.exec_() == QMessageBox.Yes:
            self._begin_training()
        else:
            self.statusBar().showMessage(tr("已取消训练"), 3000)
    def _begin_training(self):
        """预检（或其跳过）之后真正启动训练线程。"""
        from ...i18n import tr

        self._save_params()
        p = self.project
        # 数据集校验移入训练线程（大集合扫描不阻塞界面，失败弹窗提示）
        if p.model_dir:
            Path(p.model_dir).mkdir(parents=True, exist_ok=True)
        # 模型引用（pruned:/run: 前缀）含冒号等非法字符，目录名需安全化
        run_dir = Path(p.model_dir) / f"{time.strftime('%Y%m%d_%H%M%S')}_{_safe_run_tag(p.model_name)}"

        self._live_history = []
        self._latest_progress = None
        self._history = self._live_history
        self._live_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._live_start_dt = datetime.now()
        self._selected_run_dir = run_dir
        self.loss_plot.clear()
        self.metric_plot.clear()
        self.cm_canvas.ax.clear()
        self.cm_canvas.draw()
        self.final_metrics.setText("")
        self._update_per_class_table(None)  # 新训练开始前清空逐类表
        self._build_echo_form(run_dir=run_dir)
        self.train_status.setText(tr("训练中…"))
        self.btn_start.setEnabled(False)
        # YOLO 训练由引擎控制，不支持中途停止；§四 免训练范式无轮次概念
        # （stop_requested 仅分类/R-CNN 检查），停止键置灰防误导
        self.btn_stop.setEnabled(not p.model_name.startswith("yolo")
                                 and getattr(self, "_train_mode", "epoch")
                                 != "free")
        self.btn_open_dir.setEnabled(True)

        self._run_dir = run_dir
        self.thread = TrainThread(p, run_dir)
        self.thread.epoch_changed.connect(self._on_epoch)
        self.thread.batch_progress.connect(self._on_batch_progress)
        self.thread.train_finished.connect(self._on_train_finished)
        auto_cleanup_thread(self, "thread", self.thread)  # A-3
        self.thread.start()
        self._switch_detail(1)  # 跳转到训练可视化页（刷新历史列表并选中本次训练）
        self._live_timer.start()
        self.statusBar().showMessage(f"训练输出：{run_dir}")
    def _build_echo_form(self, config: dict | None = None, run_dir=None):
        """把训练设置展示在训练可视化页底部；config 为历史记录里的参数快照。"""
        from ...i18n import tr

        p = self.project
        out_dir = str(run_dir or self._run_dir or "")
        if config is None:  # 实时训练：读当前项目参数
            aug = "、".join(self._collect_augment_ops()) or tr("未启用")
            epoch_w = (f"{tr('每轮保存')}（{Path(p.model_dir) / 'epoch_weights'}）"
                       if p.save_epoch_weights else tr("关闭"))
            rows = [
                (tr("项目名称"), p.name), (tr("任务类型"), tr(p.task_type.label)),
                (tr("训练模型"), p.model_name),
                (tr("工业优化"), tr("启用") if p.industrial_opt else tr("关闭")),
                (tr("冻结骨干"), tr("是") if p.freeze_backbone else tr("否")),
                (tr("RPN 候选削减"), tr("是") if p.rpn_reduced else tr("否")),
                (tr("轻量掩码头"), tr("是") if p.mask_head_light else tr("否")),
        (tr("slim 头"), tr("是") if getattr(p, "slim_head", False) else tr("否")),
        (tr("小目标锚框"), tr("是") if getattr(p, "anchor_small", False) else tr("否")),
        (tr("P2 检测头"), tr("是") if getattr(p, "p2_head", False) else tr("否")),
                (tr("AMP 混合精度"), tr("开启") if p.use_amp else tr("关闭")),
                (tr("GPU 提速(TF32)"), tr("开启") if p.tf32 else tr("关闭")),
                (tr("轮次权重"), epoch_w),
                (tr("早停耐心"),
                 f"{p.patience} {tr('轮')}" if p.patience else tr("关闭")),
                (tr("数据集路径"), p.dataset_dir),
                (tr("模型储存路径"), p.model_dir), (tr("输出目录"), out_dir),
                (tr("训练集比例"), f"{getattr(p, 'train_ratio', 0):.0%}"),
                (tr("验证集比例"), f"{p.val_ratio:.0%}"),
                (tr("每轮重复学习"), f"{p.multiply_factor} {tr('遍')}"),
                (tr("负样本训练"),
                 tr("启用") if getattr(p, "use_negatives", True) else tr("关闭")),
                (tr("颜色模式"),
                 tr("灰度") if getattr(p, "color_mode", "color") == "gray" else tr("彩色")),
                (tr("确定性训练"),
                 tr("开启") if getattr(p, "deterministic", False) else tr("关闭")),
                (tr("增强策略"), aug), (tr("批量大小"), str(p.batch_size)),
                (tr("训练轮次"), str(p.epochs)), (tr("学习率"), f"{p.learning_rate:g}"),
                (tr("权重衰减"), f"{p.weight_decay:g}"),
                (tr("评估间隔"), f"{tr('每')} {p.eval_interval} {tr('轮')}"),
                (tr("训练设备"), tr("自动") if p.device == "auto" else p.device.upper()),
            ]
        else:  # 历史记录：读训练时的参数快照
            aug = "、".join((config.get("augment_ops") or {}).keys()) or tr("未启用")
            dev = config.get("device", "auto")
            epoch_w = (f"{tr('每轮保存')}（{Path(p.model_dir) / 'epoch_weights'}）"
                       if config.get("save_epoch_weights", True) else tr("关闭"))
            rows = [
                (tr("项目名称"), p.name), (tr("任务类型"), tr(p.task_type.label)),
                (tr("训练模型"), config.get("model_name") or p.model_name),
                (tr("工业优化"),
                 tr("启用") if config.get("industrial_opt") else tr("关闭")),
                (tr("冻结骨干"), tr("是") if config.get("freeze_backbone") else tr("否")),
                (tr("RPN 候选削减"), tr("是") if config.get("rpn_reduced") else tr("否")),
                (tr("轻量掩码头"), tr("是") if config.get("mask_head_light") else tr("否")),
        (tr("slim 头"), tr("是") if config.get("slim_head") else tr("否")),
        (tr("小目标锚框"), tr("是") if config.get("anchor_small") else tr("否")),
        (tr("P2 检测头"), tr("是") if config.get("p2_head") else tr("否")),
                (tr("AMP 混合精度"),
                 tr("开启") if config.get("use_amp", True) else tr("关闭")),
                (tr("GPU 提速(TF32)"),
                 tr("开启") if config.get("tf32", True) else tr("关闭")),
                (tr("轮次权重"), epoch_w),
                (tr("早停耐心"),
                 f"{config.get('patience', 0)} {tr('轮')}"
                 if config.get("patience") else tr("关闭")),
                (tr("数据集路径"), p.dataset_dir),
                (tr("模型储存路径"), p.model_dir), (tr("输出目录"), out_dir),
                (tr("训练集比例"),
                 f"{config.get('train_ratio', 0) or round(1 - config.get('val_ratio', 0.2), 2):.0%}"),
                (tr("验证集比例"), f"{config.get('val_ratio', 0):.0%}"),
                (tr("每轮重复学习"), f"{config.get('multiply_factor', 1)} {tr('遍')}"),
                (tr("负样本训练"),
                 tr("启用") if config.get("use_negatives", True) else tr("关闭")),
                (tr("颜色模式"),
                 tr("灰度") if config.get("color_mode", "color") == "gray" else tr("彩色")),
                (tr("确定性训练"),
                 tr("开启") if config.get("deterministic", False) else tr("关闭")),
                (tr("增强策略"), aug),
                (tr("批量大小"), str(config.get("batch_size", "—"))),
                (tr("训练轮次"), str(config.get("epochs", "—"))),
                (tr("学习率"), f"{config.get('learning_rate', 0):g}"),
                (tr("权重衰减"), f"{config.get('weight_decay', 0):g}"),
                (tr("评估间隔"), f"{tr('每')} {config.get('eval_interval', 1)} {tr('轮')}"),
                (tr("训练设备"), tr("自动") if dev == "auto" else str(dev).upper()),
            ]
        while self.echo_form.rowCount():
            self.echo_form.removeRow(0)
        for k, v in rows:
            label = QLabel(str(v))
            label.setWordWrap(True)
            self.echo_form.addRow(k, label)
    def _stop_training(self):
        if self.thread is not None:
            self.thread.stop()
            self.train_status.setText(tr("正在停止（当前轮结束后保存最优权重）…"))
            self.btn_stop.setEnabled(False)
    def _open_output_dir(self):
        run = self._selected_run_dir or self._run_dir
        if run is not None and Path(run).exists():
            os.startfile(str(run))


# ---- Cython 编译态槽包装：mixin 方法经继承进入窗口类，编译态同样需要
# 参数裁剪包装（见 xccv/qt_slots.py；幂等）----
from xccv.qt_slots import apply_auto_slot_class  # noqa: E402

apply_auto_slot_class(TrainPageMixin)

