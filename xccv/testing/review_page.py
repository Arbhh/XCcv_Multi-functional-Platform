"""项目复盘页（用户勘误 2026-11-13）：选项目 + 训练记录 → 自动对训练数据集
批量推理一次 → 分组缩略网格 + 维度阈值复用调整。

与「标准」页的分工（界面解耦勘误）：
- 本页不可自由选模型/图片，只按 项目 → 训练记录 定位权重，默认批量运行
  该项目的训练数据集；批量参数（conf/iou/检出流程）在控制行暴露可改；
- 一次加载只推理一次，结果（match_stats）落临时缓存文件；关闭平台或
  重新选择项目/记录时清理；此后拖维度阈值只复用结果重新分组，零重推理；
- V0.7.2 任务③：已评估/未评估解耦——「已评估」显示 正确/漏检/过检 三组
  （过检 > 漏检 > 正确，见 review_logic），「未评估」显示无标注生产图按
  检出过阈与否分 OK/NG 两组；左侧阈值曲线随范围同步换口径；
- 缩略网格样式与训练平台数据网格同源（懒加载 + LRU，双击系统看图打开）。
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path

from PyQt5.QtCore import QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QPixmap
from PyQt5.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QVBoxLayout, QWidget)

from ..app_paths import app_root
from ..engine.dataset.base import IMAGE_EXTS
from .review_logic import classify_images, stats_line

log = logging.getLogger(__name__)

_THUMB = 104
_LRU_CAP = 600

_CACHE_DIR = app_root() / "platform_data" / "review_cache"
_CACHE_FILE = _CACHE_DIR / "复盘缓存.json"


class ReviewThread(QThread):
    """复盘批量线程：加载权重 → 逐图预测（含检出流程与 _dims）→ GT 匹配统计。

    结果与统计随信号回主线程；逐检出 _dims 在线程内计算（前景占比/灰度差
    需要图像）。临时缓存文件也在此落盘（进度收尾阶段）。
    """

    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(list, dict, list, str)   # results, stats, classes, weights
    failed = pyqtSignal(str)

    def __init__(self, weights: Path, dataset_dir: Path, project_dir: Path,
                 conf: float, iou: float, parent=None):
        super().__init__(parent)
        self._weights = Path(weights)
        self._dataset = Path(dataset_dir)
        self._project_dir = Path(project_dir)
        self._conf, self._iou = float(conf), float(iou)

    def run(self):
        try:
            from ..engine.models import load_model
            from ..engine.io_utils import imread_unicode
            from ..engine.metrics.batch_match import collect_match_stats
            from ..engine.flow.legacy_adapter import apply_flow_safe

            model = load_model(self._weights)
            classes = list(model.classes or [])
            task = getattr(model, "task", None)
            task_v = str(getattr(task, "value", task) or "") if task is not None else ""
            # §二：ANOMALY/分类按数据集目录当弱 GT——类别目录名
            # 集合由数据集扫描得出（与训练平台 scan_classification_dataset 同源）
            weak_classes = None
            if task_v in ("anomaly", "classification"):
                try:
                    from ..engine.dataset.reader import scan_classification_dataset

                    _samples, weak_classes = scan_classification_dataset(
                        self._dataset)
                except Exception:  # noqa: BLE001  扫描失败退父目录名启发
                    weak_classes = None
            flow = None
            fp = self._project_dir / "检出流程.json"
            if fp.is_file():
                from ..engine.detection_flow import load_flow

                flow = load_flow(fp)

            files = []
            for p in sorted(self._dataset.rglob("*")):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS \
                        and not any(part.startswith(".")
                                    for part in p.relative_to(self._dataset).parts[:-1]):
                    files.append(p)
            total = len(files)
            if not total:
                self.failed.emit(f"数据集无图片：{self._dataset}")
                return

            # 预热（不计统计，失败不阻塞）
            try:
                wimg = imread_unicode(files[0])
                if wimg is not None:
                    model.predict(wimg, conf=self._conf, iou=self._iou)
            except Exception:  # noqa: BLE001
                log.warning("复盘预热失败（忽略）", exc_info=True)

            import cv2 as _cv2
            import time as _time

            results = []
            costs = []
            for i, p in enumerate(files, start=1):
                t0 = _time.perf_counter()
                img = imread_unicode(p)
                if img is None:
                    continue
                h, w = img.shape[:2]
                pred = model.predict(img, conf=self._conf, iou=self._iou)
                if flow:
                    pred, _info = apply_flow_safe(pred, (w, h), flow,
                                                  classes=classes or None)
                gray = img if img.ndim == 2 else _cv2.cvtColor(
                    img, _cv2.COLOR_BGR2GRAY)
                dets = []
                for j, box in enumerate(pred.boxes):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    label = (classes[pred.labels[j]]
                             if classes and pred.labels[j] < len(classes)
                             else str(pred.labels[j]))
                    det = {"label": label,
                           "score": round(float(pred.scores[j]), 4),
                           "bbox": [round(x1, 1), round(y1, 1),
                                    round(x2, 1), round(y2, 1)]}
                    from ..engine.flow.ops.det_filters import box_image_dims

                    det["_dims"] = box_image_dims(gray, [x1, y1, x2, y2])
                    dets.append(det)
                entry = {"file": p.name, "path": str(p),
                         "width": w, "height": h, "detections": dets}
                # §二：图像级口径逐图附加值（batch_match 分支消费）
                if task_v == "anomaly" and pred.probs:
                    entry["image_score"] = round(float(pred.probs[0]), 4)
                if task_v == "classification" and pred.labels \
                        and not pred.boxes:
                    lid = int(pred.labels[0])
                    entry["classifications"] = [{
                        "label": (classes[lid]
                                  if classes and lid < len(classes)
                                  else str(lid)),
                        "score": (round(float(pred.scores[0]), 4)
                                  if pred.scores else 0.0),
                    }]
                results.append(entry)
                costs.append(round((_time.perf_counter() - t0) * 1000, 1))
                self.progress.emit(i, total, p.name)

            stats = collect_match_stats(
                results, iou_thr=self._iou, task=task,
                score_semantics=getattr(model, "score_semantics", None),
                weak_classes=weak_classes)
            self._save_cache(stats)
            self.finished_ok.emit(results, stats, classes, str(self._weights))
        except Exception as e:  # noqa: BLE001
            log.error("复盘批量失败: %s", e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _save_cache(self, stats: dict) -> None:
        """临时结果文件（勘误：一次加载一份；重新加载/关闭时清理）。"""
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps({
                "weights": str(self._weights),
                "dataset_dir": str(self._dataset),
                "project_dir": str(self._project_dir),
                "conf": self._conf, "iou": self._iou,
                "match_stats": stats,
            }, ensure_ascii=False), encoding="utf-8")
        except OSError:
            log.warning("复盘缓存写入失败（不影响本次会话）", exc_info=True)


class ProjectReviewPage(QWidget):
    """复盘页：控制行 + 维度阈值面板 + 范围切换分组缩略网格 + 统计行。"""

    openReviewRequested = pyqtSignal()
    openCompareRequested = pyqtSignal()
    applyConfRequested = pyqtSignal(float)
    generateFlowRequested = pyqtSignal(dict)
    statusMessage = pyqtSignal(str)
    contextReady = pyqtSignal(dict)   # {results, stats, weights, project_dir, out_dir, classes, conf, iou}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list = []
        self._stats: dict = {}
        self._classes: list = []
        self._weights: str = ""
        self._project_dir: Path | None = None
        self._task_mode = "detection"   # §二：detection/anomaly/classification
        self._thread = None
        self._thumb_worker = None
        self._icons: OrderedDict = OrderedDict()
        self._items_by_path: dict = {}
        self._banner = None

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(12)

        # ---- 左列：控制 + 维度阈值 + 复判入口 ----
        left = QWidget()
        left.setFixedWidth(520)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)

        g1 = QGroupBox("项目与训练记录")
        f1 = QGridLayout(g1)
        f1.setHorizontalSpacing(8)
        f1.setVerticalSpacing(8)
        self.cmb_project = QComboBox()
        self.cmb_project.currentIndexChanged.connect(self._on_project_changed)
        self.cmb_run = QComboBox()
        # 再勘误①（2026-11-13）：复盘数据集默认=项目训练数据集，可浏览
        # 换成任意文件夹（如新生产图目录）；换目录即清缓存清结果
        self.fld_dataset = QLineEdit()
        self.fld_dataset.setReadOnly(True)
        from ..i18n import tr

        self.btn_browse = QPushButton(tr("浏览…"))
        self.btn_browse.setToolTip(
            tr("复盘数据集默认为本项目的训练数据集；点此选择其他文件夹")
            + tr("（如新一批生产图）后点「开始复盘」"))
        self.btn_browse.clicked.connect(self._browse_dataset)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setDecimals(2)
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setToolTip(tr("批量推理的置信度下限（拖线调整的下限）"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setDecimals(2)
        self.spin_iou.setRange(0.05, 0.9)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.20)
        self.btn_start = QPushButton("▶ 开始复盘")
        self.btn_start.clicked.connect(self._start_review)
        f1.addWidget(QLabel("项目"), 0, 0)
        f1.addWidget(self.cmb_project, 0, 1)
        f1.addWidget(QLabel("训练记录"), 1, 0)
        f1.addWidget(self.cmb_run, 1, 1)
        f1.addWidget(QLabel("数据集"), 2, 0)
        ds_row = QHBoxLayout()
        ds_row.setSpacing(6)
        ds_row.addWidget(self.fld_dataset, stretch=1)
        ds_row.addWidget(self.btn_browse)
        ds_widget = QWidget()
        ds_widget.setLayout(ds_row)
        f1.addWidget(ds_widget, 2, 1)
        self.lbl_conf = QLabel("置信度 conf")
        f1.addWidget(self.lbl_conf, 3, 0)
        f1.addWidget(self.spin_conf, 3, 1)
        self.lbl_iou = QLabel("IOU")
        f1.addWidget(self.lbl_iou, 4, 0)
        f1.addWidget(self.spin_iou, 4, 1)
        f1.addWidget(self.btn_start, 5, 0, 1, 2)
        lv.addWidget(g1)

        g2 = QGroupBox(tr("维度阈值"))
        g2.setToolTip(tr("拖动阈值线或键入阈值后自动重新分组，不重新推理"))
        v2 = QVBoxLayout(g2)
        from .threshold_panel import ThresholdPanel

        self.threshold_panel = ThresholdPanel(compact=True)
        self.threshold_panel.applyRequested.connect(self.applyConfRequested)
        self.threshold_panel.generateFlowRequested.connect(
            self.generateFlowRequested)
        self.threshold_panel.thresholdsChanged.connect(self._regroup)
        v2.addWidget(self.threshold_panel)
        lv.addWidget(g2, stretch=1)

        row = QHBoxLayout()
        self.btn_review = QPushButton(tr("复判与回流…"))
        self.btn_review.setEnabled(False)
        self.btn_review.clicked.connect(self.openReviewRequested)
        self.btn_ab = QPushButton(tr("模型 A/B 比对…"))
        self.btn_ab.setEnabled(False)
        self.btn_ab.clicked.connect(self.openCompareRequested)
        row.addWidget(self.btn_review)
        row.addWidget(self.btn_ab)
        lv.addLayout(row)

        self.lbl_hint = QLabel(
            tr("对所选数据集批量推理一次，调阈值即时重新分组"))
        self.lbl_hint.setObjectName("hint")
        self.lbl_hint.setWordWrap(True)
        lv.addWidget(self.lbl_hint)
        root.addWidget(left)

        # ---- 右侧：范围切换 + 分组缩略网格（V0.7.2 任务③解耦） ----
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(8)
        scope_row = QHBoxLayout()
        self._scope = "evaluated"
        self.btn_scope_eval = QPushButton("已评估")
        self.btn_scope_eval.setObjectName("seg")
        self.btn_scope_eval.setCheckable(True)
        self.btn_scope_eval.setChecked(True)
        self.btn_scope_eval.setToolTip(tr("有 GT 标注或 OK 目录的图"))
        self.btn_scope_eval.clicked.connect(
            lambda: self._set_scope("evaluated"))
        self.btn_scope_unjudged = QPushButton("未评估")
        self.btn_scope_unjudged.setObjectName("seg")
        self.btn_scope_unjudged.setCheckable(True)
        self.btn_scope_unjudged.setToolTip(tr("无标注生产图，按检出分 OK/NG"))
        self.btn_scope_unjudged.clicked.connect(
            lambda: self._set_scope("unjudged"))
        scope_row.addWidget(self.btn_scope_eval)
        scope_row.addWidget(self.btn_scope_unjudged)
        scope_row.addStretch(1)
        rv.addLayout(scope_row)
        self.grp_correct = QGroupBox("正确 ✓（0）")
        self.grp_missed = QGroupBox("漏检（0）")
        self.grp_over = QGroupBox("过检（0）")
        self.grp_uok = QGroupBox("OK（0）")
        self.grp_ung = QGroupBox("NG（0）")
        self.grp_ung.setToolTip(tr("有检出通过阈值的未评估图，检出供肉眼确认"))
        self.grids = {}
        for g in (self.grp_correct, self.grp_missed, self.grp_over,
                  self.grp_uok, self.grp_ung):
            gv = QVBoxLayout(g)
            gv.setContentsMargins(8, 4, 8, 6)
            lst = QListWidget()
            lst.setViewMode(QListWidget.IconMode)
            lst.setResizeMode(QListWidget.Adjust)
            lst.setMovement(QListWidget.Static)
            lst.setSpacing(6)
            lst.setIconSize(QSize(_THUMB + 6, _THUMB + 6))
            lst.setUniformItemSizes(True)
            lst.setWordWrap(True)
            lst.itemDoubleClicked.connect(self._open_image)
            gv.addWidget(lst)
            self.grids[g] = lst
            rv.addWidget(g, stretch=1)
        self._set_scope("evaluated", initial=True)
        # 组标题基名（V0.7.2 i18n：动态改题走 tr，不解析旧标题）
        self._group_bases = {
            self.grp_correct: "正确 ✓",
            self.grp_missed: "漏检",
            self.grp_over: "过检",
            self.grp_uok: "OK",
            self.grp_ung: "NG",
        }
        self.lbl_stats = QLabel("尚未复盘")
        self.lbl_stats.setObjectName("hint")
        rv.addWidget(self.lbl_stats)
        root.addWidget(right, stretch=1)

        self._refresh_projects()

    def _set_scope(self, scope: str, initial: bool = False):
        """已评估/未评估解耦切换（V0.7.2 任务③）：分组与曲线同步换口径。"""
        scope = "unjudged" if scope == "unjudged" else "evaluated"
        self._scope = scope
        self.btn_scope_eval.setChecked(scope == "evaluated")
        self.btn_scope_unjudged.setChecked(scope == "unjudged")
        for g in (self.grp_correct, self.grp_missed, self.grp_over):
            g.setVisible(scope == "evaluated")
        for g in (self.grp_uok, self.grp_ung):
            g.setVisible(scope == "unjudged")
        self.threshold_panel.set_scope(scope)
        if not initial:
            self._regroup(self.threshold_panel.enabled_dims())

    def _apply_task_mode(self, mode: str, semantics: dict | None = None):
        """§二：按模型任务口径调整复盘 UI（组名/行可见性/阈值提示）。

        detection=检测/分割（GT 框口径，默认布局不动）；anomaly=图像级
        弱 GT（三组语义不变，阈值面板换图像分数轴）；classification=对/
        错分组（过检组改题"错误"、漏检组隐藏、IOU 行隐藏）。
        """
        mode = mode if mode in ("anomaly", "classification") else "detection"
        self._task_mode = mode
        from ..i18n import tr

        self._group_bases[self.grp_over] = ("错误" if mode == "classification"
                                            else "过检")
        self.grp_missed.setVisible(mode != "classification")
        for w in (self.lbl_iou, self.spin_iou):
            w.setVisible(mode == "detection")
        sem = semantics or {}
        if mode == "anomaly":
            self.lbl_conf.setText(tr("阈值 conf"))
            self.lbl_conf.setToolTip(tr(
                "conf→图像分数 z=1/conf−1（0.25→z=3）；拖维度阈值面板调分"))
        elif mode == "classification":
            self.lbl_conf.setText(tr("置信度 conf"))
            self.lbl_conf.setToolTip(tr(
                "批量推理的预测概率下限（拖线调整的下限）"))
        else:
            self.lbl_conf.setText(tr("置信度 conf"))
            self.lbl_conf.setToolTip(tr("批量推理的置信度下限（拖线调整的下限）"))
        if sem.get("name"):
            self.spin_conf.setToolTip(
                f"{tr(str(sem.get('name')))}：{tr(str(sem.get('higher', '')))}")

    # ---------- 项目 / 记录 ----------
    def _refresh_projects(self):
        from ..training.project import list_projects

        self.cmb_project.blockSignals(True)
        self.cmb_project.clear()
        for name in list_projects():
            self.cmb_project.addItem(name)
        self.cmb_project.blockSignals(False)
        if self.cmb_project.count():
            self._on_project_changed()

    def _on_project_changed(self, *_):
        from ..training.project import load_project
        from ..training.runs import scan_runs

        self._clear_results()
        self._clear_cache()
        self.cmb_run.blockSignals(True)
        self.cmb_run.clear()
        try:
            p = load_project(self.cmb_project.currentText())
            # 项目输出目录（=模型储存目录）：检出流程.json / project.json 所在
            self._project_dir = Path(p.model_dir)
            self.fld_dataset.setText(str(p.dataset_dir))   # 默认训练数据集
        except Exception:  # noqa: BLE001
            self.fld_dataset.setText("")
            self.cmb_run.blockSignals(False)
            return
        recs = scan_runs(p.model_dir)
        for rec in recs:
            self.cmb_run.addItem(
                f"{rec.run_dir.name}（{rec.status_text}）",
                userData=str(rec.run_dir))
        self.cmb_run.blockSignals(False)

    def _browse_dataset(self):
        """再勘误①：复盘数据集可换任意文件夹（如新生产图；无需 GT）。"""
        path = QFileDialog.getExistingDirectory(self, "选择复盘数据集文件夹",
                                                self.fld_dataset.text())
        if not path:
            return
        self.fld_dataset.setText(path)
        self._clear_results()
        self._clear_cache()

    def _clear_cache(self):
        import shutil

        shutil.rmtree(_CACHE_DIR, ignore_errors=True)

    @staticmethod
    def clear_stale_cache():
        """上次会话遗留缓存兜底清理（窗口构造时调用；关闭路径未走
        closeEvent 时，残留文件在下次启动时清掉）。"""
        import shutil

        shutil.rmtree(_CACHE_DIR, ignore_errors=True)

    def _clear_results(self):
        self._results = []
        self._stats = {}
        self._items_by_path.clear()
        for lst in self.grids.values():
            lst.clear()
        for g in self.grids:
            self._set_group_title(g, 0)
        self.lbl_stats.setText("尚未复盘")
        self.btn_review.setEnabled(False)
        self.btn_ab.setEnabled(False)

    def _set_group_title(self, group, n: int):
        """组标题 = tr 基名 + 计数（EN 模式下基名经 i18n 表）。"""
        from ..i18n import tr

        base = self._group_bases.get(group, group.title())
        group.setTitle(f"{tr(base)} ({n})")

    # ---------- 复盘批量 ----------
    def _start_review(self):
        if self._thread is not None and self._thread.isRunning():
            self.statusMessage.emit("复盘正在进行中，请稍候")
            return
        run_dir = self.cmb_run.currentData()
        if not run_dir:
            self.lbl_hint.setText("该项目暂无训练记录；请先在训练平台完成训练")
            return
        from ..training.project import load_project
        from ..i18n import tr

        try:
            p = load_project(self.cmb_project.currentText())
        except Exception as e:  # noqa: BLE001
            self.lbl_hint.setText(f"{tr('项目读取失败：')}{e}")
            return
        weights = Path(run_dir) / "best.pt"
        if not weights.is_file():
            self.lbl_hint.setText(tr("该训练记录没有 best.pt 权重"))
            return
        dataset = Path(self.fld_dataset.text().strip() or p.dataset_dir)
        if not dataset.is_dir():
            self.lbl_hint.setText(f"{tr('数据集目录不存在：')}{dataset}")
            return
        self._project_dir = Path(p.model_dir)
        self._clear_results()
        self._clear_cache()
        self._show_banner(f"{tr('正在复盘')} {self.cmb_run.currentText()} …")
        self.btn_start.setEnabled(False)
        from ..gui_utils import auto_cleanup_thread

        th = ReviewThread(weights, dataset, self._project_dir,
                          float(self.spin_conf.value()),
                          float(self.spin_iou.value()))
        th.progress.connect(self._on_progress)
        th.finished_ok.connect(self._on_review_done)
        th.failed.connect(self._on_review_failed)
        auto_cleanup_thread(self, "_thread", th)
        self._thread = th
        th.start()

    def _show_banner(self, text: str):
        """顶部悬浮胶囊（U1 同款样式）：批量加载中提示。"""
        self._hide_banner()
        banner = QLabel(text, self)
        banner.setStyleSheet(
            "QLabel { background: rgba(58, 46, 34, 224); color: white;"
            " padding: 6px 18px; border-radius: 12px; font-weight: bold;"
            " font-size: 13px; }")
        banner.adjustSize()
        banner.move((max(1, self.width()) - banner.width()) // 2, 12)
        banner.show()
        banner.raise_()
        self._banner = banner

    def _hide_banner(self):
        if self._banner is not None:
            self._banner.hide()
            self._banner.setParent(None)
            self._banner.deleteLater()
            self._banner = None

    def _on_progress(self, done: int, total: int, name: str):
        if self._banner is not None:
            from ..i18n import tr

            self._banner.setText(
                f"{tr('复盘批量推理')} {done} / {total}　{name}")
            self._banner.adjustSize()
            self._banner.move((max(1, self.width()) - self._banner.width()) // 2, 12)

    def _on_review_failed(self, msg: str):
        self._hide_banner()
        self.btn_start.setEnabled(True)
        from ..i18n import tr

        self.lbl_hint.setText(f"{tr('复盘失败：')}{msg}")
        self.statusMessage.emit(tr("复盘失败"))

    def _on_review_done(self, results: list, stats: dict, classes: list,
                        weights: str):
        self._hide_banner()
        self.btn_start.setEnabled(True)
        from ..i18n import tr

        self._results = results
        self._stats = stats
        self._classes = classes
        self._weights = weights
        # §二：按本次模型任务口径调整组名/行可见性（stats.task 为准）
        self._apply_task_mode(str(stats.get("task") or "detection"),
                              stats.get("score_semantics"))
        total_det = sum(len(r["detections"]) for r in results)
        self.statusMessage.emit(
            f"{tr('复盘完成')}：{len(results)} {tr('张图')}，"
            f"{tr('共检出')} {total_det} {tr('个目标')}；"
            f"{tr('调维度阈值不会重新推理')}")
        self.lbl_hint.setText(
            f"{tr('复盘完成')}：{len(results)} {tr('张图')}；"
            f"{tr('调维度阈值不会重新推理')}。")
        # 维度面板注入（score 维初始=批量 conf）；面板会触发一次 thresholdsChanged
        self.threshold_panel.set_stats(stats, float(self.spin_conf.value()))
        self._regroup(self.threshold_panel.enabled_dims())
        self.btn_review.setEnabled(True)
        self.btn_ab.setEnabled(True)
        self.contextReady.emit({
            "results": results, "stats": stats, "classes": classes,
            "weights": weights, "project_dir": str(self._project_dir),
            "out_dir": str(_CACHE_DIR),
            "conf": float(self.spin_conf.value()),
            "iou": float(self.spin_iou.value()),
            "weights_name": Path(weights).name,
        })

    # ---------- 分组与网格 ----------
    def _regroup(self, thresholds: dict):
        if not self._stats or not self._stats.get("per_image"):
            return
        groups = classify_images(self._stats, thresholds)
        self._items_by_path.clear()
        if self._scope == "unjudged":
            # V0.7.2 任务③：未评估按有无过阈检出分 NG/OK
            ok_entries, ng_entries = [], []
            for e in groups["unjudged_list"]:
                (ng_entries if e.get("fp") else ok_entries).append(e)
            mapping = [("uok", self.grp_uok, "#2E9E5B"),
                       ("ung", self.grp_ung, "#D64545")]
            bucket = {"uok": ok_entries, "ung": ng_entries}
        else:
            mapping = [("correct", self.grp_correct, "#2E9E5B"),
                       ("missed", self.grp_missed, "#B45309"),
                       ("over", self.grp_over, "#D64545")]
            bucket = {key: groups[key] for key, _g, _c in mapping}
        for key, group, color in mapping:
            lst = self.grids[group]
            lst.clear()
            entries = bucket[key]
            for e in entries:
                text = e["file"] + (("　" + e["badges"]) if e["badges"] else "")
                tip = e["path"]
                if e["badges"]:
                    tip += f"\n{e['badges']}"
                if e.get("is_ok"):
                    tip += "\nOK 图（目录名规则，与训练平台负样本同源）"
                if e.get("unjudged"):
                    tip += "\n无 GT 标注：仅展示检出供肉眼确认，不参与判定"
                item = QListWidgetItem(self._placeholder(), text)
                item.setData(Qt.UserRole, e["path"])
                item.setToolTip(tip)
                item.setForeground(QColor(color))
                lst.addItem(item)
                self._items_by_path[e["path"]] = item
            self._set_group_title(group, len(entries))
        self.lbl_stats.setText(
            stats_line(groups, self._scope, self._task_mode)
            + "　（双击图片用系统看图打开）")
        self._start_thumbs()

    def _placeholder(self) -> QIcon:
        pm = QPixmap(_THUMB, _THUMB)
        pm.fill(QColor("#EDF1F7"))
        return QIcon(pm)

    def _start_thumbs(self):
        """懒加载缩略：只解码当前可见条目（LRU 缓存命中即跳过）。

        V0.7.1：缓存命中的条目直接回贴图标（重建列表项后不会经过缩略线程，
        不回贴会一直显示占位灰块）。"""
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
            self._thumb_worker = None
        todo = [p for p in self._items_by_path if p not in self._icons]
        self._apply_cached_icons()
        if not todo:
            return
        from ..gui_utils import auto_cleanup_thread

        w = _ThumbWorker(todo)
        w.ready.connect(self._on_thumbs)
        auto_cleanup_thread(self, "_thumb_worker", w)
        self._thumb_worker = w
        w.start()

    def _apply_cached_icons(self):
        for p, icon in self._icons.items():
            item = self._items_by_path.get(p)
            if item is not None:
                item.setIcon(icon)

    def _on_thumbs(self, batch):
        for p, qimg in batch:
            icon = QIcon(QPixmap.fromImage(qimg))
            self._icons[p] = icon
            self._icons.move_to_end(p)
            while len(self._icons) > _LRU_CAP:
                self._icons.popitem(last=False)
            item = self._items_by_path.get(p)
            if item is not None:
                item.setIcon(icon)

    def _open_image(self, item):
        QDesktopServices.openUrl(QUrl.fromLocalFile(item.data(Qt.UserRole)))

    # ---------- 收尾 ----------
    def shutdown(self):
        """窗口关闭：停线程 + 清理临时缓存（勘误要求）。"""
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
            self._thumb_worker = None
        self._thread = None
        self._clear_cache()

    def has_results(self) -> bool:
        return bool(self._results)


class _ThumbWorker(QThread):
    """缩略图工作线程（data_grid 同款纪律：子线程解码 QImage，主线程建 QPixmap）。"""

    ready = pyqtSignal(list)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self._queue = list(paths)
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        from ..training.data_grid import make_thumbnail, ndarray_to_qimage

        batch = []
        while self._queue and not self._stopped:
            p = self._queue.pop(0)
            try:
                arr = make_thumbnail(p, size=_THUMB)
            except Exception:  # noqa: BLE001
                arr = None
            if arr is not None:
                batch.append((p, ndarray_to_qimage(arr)))
            if len(batch) >= 12 or (batch and not self._queue):
                self.ready.emit(batch)
                batch = []


from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())
