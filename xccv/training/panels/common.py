"""训练平台窗口共享件（0.7.6 P-1 页级拆分）：模块级助手/控件类/预检线程/项目卡。

内容与拆分前 training_window.py 完全一致（逐行搬运）：
- 常量：_TASK_MODEL_ICON/_TASK_BADGES、DEFAULT_LR、AUGMENT_ROWS、
  _CURVE_COLORS、_PRESET_DEFS(_FREE)、_ILLEGAL_NAME_CHARS、_TICK_FONT
- 助手：_cuda_available/_safe_run_tag/_apply_theme/_t2/_t3/_project_mtime/
  _status_dot_icon/_make_badge/_style_plot/_add_legend
- 控件类：_IntAxis/_NoWheel/_Spin/_DoubleSpin/_Combo/_PagePlot/_PlotHover/
  _ProjectCard/_PreflightWorker
调用方法：训练平台各面板（panels/*）与组合根 training_window 导入使用；
外部兼容导入经 training_window 再导出。
"""

import re
from pathlib import Path
import pyqtgraph as pg
from PyQt5.QtCore import QPointF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
)
from ...engine.dataset.base import TaskType
from ...engine.dataset.reader import scan_classification_dataset, scan_labelme_dataset
from ...i18n import tr
from ..data_health import LEVEL_WARN, run_health_check
from ..dev_options import load_options
from ..preflight import dataset_facts, param_findings, recommend_settings
from ..project import (Project, create_project, delete_project, list_projects, load_project, save_project)
from ...ui_style import bind_style, icon_path, token
from ...ui_widgets import Badge, HoverCard, load_prefs, relative_time, save_prefs, toast

pg.setConfigOptions(antialias=True, background=token("panel", "#ffffff"),
                    foreground=token("text_2", "#3a4656"))

# 三类模型的专属图标（v0.5.4 任务二；打开项目/项目卡随任务类型切换）
_TASK_MODEL_ICON = {
    TaskType.CLASSIFICATION: "model_classification.png",
    TaskType.DETECTION: "model_detection.png",
    TaskType.SEGMENTATION: "model_segmentation.png",
    TaskType.ANOMALY: "model_classification.png",  # A1：无监督复用分类图标
    TaskType.OCR: "model_classification.png",      # A2：OCR 复用分类图标
}
def _cuda_available() -> bool:
    """torch 延迟导入：本模块在启动闪屏显示之前导入，
    torch/引擎链必须后置到窗口构建期，否则加载中界面被 torch 拖慢 2s+。"""
    import torch

    return torch.cuda.is_available()
# 各模型的建议默认值（选择模型时自动带入，仍可修改）
DEFAULT_LR = {
    "resnet18": 0.001, "resnet50": 0.001, "mobilenet_v3_small": 0.001,
    "yolo11n": 0.01, "yolo11s": 0.01,
    "yolo11n-seg": 0.01, "yolo11s-seg": 0.01,
    "fasterrcnn": 0.005, "maskrcnn": 0.005, "maskrcnn18": 0.005,
}

# 图像增强策略行：(算子名, 中文名, [(参数名, 默认值, 最小, 最大, 是否整数)], 提示)
# UI 文案约定：界面文字简短精确、不用括号补充，详细说明放 tooltip
AUGMENT_ROWS = [
    ("resize", "调整图像大小", [
        ("width", 640, 2, 8192, True), ("height", 640, 2, 8192, True)],
     "拉伸至设定的宽×高；训练输入尺寸与这里统一"),
    ("random_rescale", "随机比例缩放", [
        ("min_scale", 0.8, 0.1, 2.0, False), ("max_scale", 1.2, 0.1, 3.0, False)],
     "在设定比例区间内随机缩放"),
    ("random_flip_h", "随机水平翻转", [],
     "左右镜像，标注同步翻转"),
    ("random_flip_v", "随机垂直翻转", [],
     "上下镜像，标注同步翻转"),
    ("rotate90", "随机旋转 90°/270°", [],
     "工件朝向增强；长宽比不一致时须配随机裁剪兜底，见下方守卫说明"),
    ("random_crop", "随机裁剪", [
        ("ratio", 1.0, 0.1, 10.0, False), ("min_cover", 0.7, 0.1, 1.0, False)],
     "按目标宽高比裁剪；作为旋转算子的兜底"),
    ("random_brightness", "随机调整亮度", [
        ("max_delta", 20, 1, 100, True)],
     "在 ±max_delta 内随机偏移亮度"),
    ("random_contrast", "随机调整对比度", [
        ("low", 0.8, 0.1, 3.0, False), ("high", 1.2, 0.1, 3.0, False)],
     "在设定区间内随机缩放对比度"),
    ("random_hsv_shift", "随机 HSV 扰动", [
        ("h_delta", 6, 1, 90, True), ("s_delta", 0.15, 0.05, 1.0, False),
        ("v_delta", 0.15, 0.05, 1.0, False)],
     "模拟光照与色温变化"),
    ("gaussian_blur", "随机高斯模糊", [
        ("ksize", 3, 1, 15, True), ("sigma", 0.6, 0.1, 5.0, False)],
     "模拟失焦"),
    ("gaussian_noise", "随机高斯噪声", [
        ("sigma", 3, 1, 60, True)],
     "模拟传感器噪声"),
    ("random_erase", "随机矩形擦除", [
        ("max_ratio", 0.1, 0.05, 0.9, False)],
     "模拟遮挡与反光"),
]
_CURVE_COLORS = ["#4c8bf5", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
_TASK_BADGES = {
    TaskType.CLASSIFICATION: ("badge", "图像分类"),
    TaskType.DETECTION: ("badgeDet", "目标检测"),
    TaskType.SEGMENTATION: ("badgeSeg", "实例分割"),
    TaskType.ANOMALY: ("badge", "无监督异常检测"),  # A1（V0.7）
    TaskType.OCR: ("badge", "OCR 字符识别"),        # A2（V0.7）
}
_ILLEGAL_NAME_CHARS = re.compile(r'[\\/:*?"<>|]')
def _safe_run_tag(model_name: str) -> str:
    """把模型名转成合法目录名片段。

    模型引用形如 "pruned:xxx.pt" / "run:记录/best.pt"，其中的冒号、斜杠
    在 Windows 目录名中非法（曾导致训练无法启动）；这里统一替换并去掉
    扩展名后缀。
    """
    tag = _ILLEGAL_NAME_CHARS.sub("_", model_name or "model")
    for suffix in (".pt", ".engine"):
        if tag.lower().endswith(suffix):
            tag = tag[: -len(suffix)]
            break
    return tag.strip("_") or "model"
def _apply_theme(widget):
    """全局样式套用 + 登记主题实时切换（V0.6 F2-5）。"""
    from ...ui_style import apply_app_theme

    apply_app_theme(widget)
# 预设档位：一键套用 + 联动 recommend_settings；
# 参数键映射见 ProjectWindow._apply_param_kv
_PRESET_DEFS = [
    ("fast", "快速验证", "30 轮 · 冻结骨干\n小样本快速出结果",
     {"epochs": 30, "freeze_backbone": True, "patience": 10, "use_amp": True},
     "#b45309"),
    ("balanced", "均衡训练", "100 轮 · 工业优化\n精度与速度兼顾",
     {"epochs": 100, "freeze_backbone": False, "industrial_opt": True,
      "use_amp": True}, "#11809c"),
    ("quality", "高质量", "300 轮 · 全量加速\n追求最佳精度，耗时更长",
     {"epochs": 300, "freeze_backbone": False, "use_amp": True,
      "tf32": True, "patience": 20}, "#1d8a53"),
]
# 免训练范式（free）预设卡（V0.7.4 bug-优化项一.1）：轮次语义对免训练
# 模型是歧义——按支持集张数/增强样本数给档位（k_shot 上限 12 既定约束）
_PRESET_DEFS_FREE = [
    ("fast", "快速建库", "2 张支持集 · 小增强\n最少样本快速出结果",
     {"k_shot": 2, "aug_count": 6}, "#b45309"),
    ("balanced", "均衡建库", "4 张支持集 · 标准增强\n精度与样本量兼顾",
     {"k_shot": 4, "aug_count": 30}, "#11809c"),
    ("quality", "高质量建库", "8 张支持集 · 大增强\n追求最佳精度，耗时更长",
     {"k_shot": 8, "aug_count": 60}, "#1d8a53"),
]
def _t2() -> str:
    """主题感知次要文字色（暗色主题下自动切换）。"""
    from ...ui_style import token

    return token("text_2", "#5f7086")
def _t3() -> str:
    from ...ui_style import token

    return token("text_3", "#93a3b5")
def _project_mtime(name: str) -> float:
    """项目最近更新时间：project.json 与各次训练 metrics.json 取最新
    （与项目列表排序同一口径——卡片排序按更新时间倒序，要求）。"""
    from ..project import PROJECTS_DIR, _project_touch_time

    return _project_touch_time(PROJECTS_DIR / name)
def _status_dot_icon(color: str, d: int = 12) -> QPixmap:
    """训练历史列表条目的状态圆点图标。"""
    pm = QPixmap(d + 2, d + 2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(1, 1, d, d)
    p.end()
    return pm
def _make_badge(task: TaskType) -> QLabel:
    obj, text = _TASK_BADGES[TaskType(task)]
    badge = QLabel(text)
    badge.setObjectName(obj)
    badge.setAlignment(Qt.AlignCenter)
    return badge
class _IntAxis(pg.AxisItem):
    """轮次轴：只显示整数刻度，避免出现 0.5、1.5 之类的小数。"""

    def tickStrings(self, values, scale, spacing):
        return [str(int(v)) if float(v).is_integer() else "" for v in values]


class _NoWheel:
    """滚轮不改变数值（事件透传给父级滚动区），防止浏览页面时误触参数。"""

    def wheelEvent(self, ev):
        ev.ignore()


class _Spin(_NoWheel, QSpinBox):
    pass


class _DoubleSpin(_NoWheel, QDoubleSpinBox):
    """小数参数框：最小位数显示（0.5 而非 0.50）+ 聚焦全选。

    设计原因：默认两位小数补零后，改值必须先删尾零；最小表示 + 全选让
    点击后直接输入即可整体覆盖。
    """

    def textFromValue(self, val: float) -> str:
        return f"{val:g}"

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return super().valueFromText(text)

    def focusInEvent(self, ev):
        super().focusInEvent(ev)
        self.lineEdit().selectAll()


class _Combo(_NoWheel, QComboBox):
    pass


class _PagePlot(pg.PlotWidget):
    """曲线图：滚轮事件透传给父级滚动区（鼠标缩放已禁用，滚轮只滚页面）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_leave = None
        self._xccv_title = ""
        self.refresh_theme()

    def refresh_theme(self):
        """主题切换：底色/坐标轴/标题随 token 重设（pyqtgraph 不吃 QSS）。"""
        bg, fg, title_c = token("panel", "#ffffff"), token("text_2", "#3a4656"), \
            token("text_1", "#1f2937")
        pg.setConfigOptions(background=bg, foreground=fg)   # 之后新建的图表同色
        self.setBackground(bg)
        pi = self.getPlotItem()
        for ax in ("bottom", "left"):
            axis = pi.getAxis(ax)
            axis.setPen(fg)
            axis.setTextPen(fg)
        if self._xccv_title:
            pi.setTitle(self._xccv_title, color=title_c)

    def wheelEvent(self, ev):
        ev.ignore()

    def leaveEvent(self, ev):
        if self._on_leave is not None:
            self._on_leave()
        super().leaveEvent(ev)
_TICK_FONT = QFont("Microsoft YaHei", 11)
def _style_plot(plot: _PagePlot, title: str):
    """统一图表排版：大号标题（HTML 渲染）+ 可读的轴刻度字体。"""
    pi = plot.getPlotItem()
    plot._xccv_title = f'<span style="font-size: 12pt; font-weight: 600;">{title}</span>'
    pi.setTitle(plot._xccv_title, color=token("text_1", "#1f2937"))
    for ax in ("bottom", "left"):
        pi.getAxis(ax).setTickFont(_TICK_FONT)


def _add_legend(plot: _PagePlot):
    """图例字号放大（旧版 pyqtgraph 无 labelTextSize 参数时回退默认）。"""
    try:
        return plot.addLegend(offset=(8, 8), labelTextSize="10pt")
    except TypeError:
        return plot.addLegend(offset=(8, 8))
class _PlotHover:
    """曲线图悬停提示：跟随鼠标显示最近采样点的数值（三位小数）。

    逐批平滑曲线按亚轮次取值，逐轮曲线按整轮取值；标签自动出现在
    鼠标所在半区的另一侧，避免遮挡光标。
    """

    def __init__(self, plot: _PagePlot, get_series):
        self.plot = plot
        self.get_series = get_series
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#94a3b8", width=1, style=Qt.DashLine))
        self.label = pg.TextItem("", anchor=(0, 1), color="#1f2937")
        self.label.setZValue(60)
        self.vline.setZValue(50)
        plot._on_leave = self.hide_tip
        plot.scene().sigMouseMoved.connect(self._on_move)

    def install(self):
        """plot.clear() 会移除图元，每次重绘后重新挂载。"""
        pi = self.plot.getPlotItem()
        pi.addItem(self.vline, ignoreBounds=True)
        pi.addItem(self.label, ignoreBounds=True)
        self.hide_tip()

    def hide_tip(self):
        self.vline.hide()
        self.label.hide()

    @staticmethod
    def _fmt_epoch(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"

    def _on_move(self, pos):
        vb = self.plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):  # 必须用 ViewBox 的矩形判断
            self.hide_tip()
            return
        series = [(k, xs, ys) for k, xs, ys in self.get_series() if xs]
        if not series:
            self.hide_tip()
            return
        view = vb.mapSceneToView(pos)
        first_xs = series[0][1]
        epoch = min(first_xs, key=lambda x: abs(x - view.x()))
        lines = [f"{tr('第')} {self._fmt_epoch(epoch)} {tr('轮')}"]
        for key, kxs, kys in series:
            j = min(range(len(kxs)), key=lambda i: abs(kxs[i] - epoch))
            if abs(kxs[j] - epoch) <= 0.5 + 1e-9:
                v = kys[j]
                if v is not None:
                    lines.append(f"{key} = {v:.3f}")
        x0, x1 = vb.viewRange()[0]
        mid = (x0 + x1) / 2
        # 锚点取文字顶边（y=0）：标签从视图顶部向下展开，留在可见区内；
        # 若锚在底边（y=1）文字会向上展开到 ViewBox 之外被整体裁剪（只剩竖线可见）
        anchor = QPointF(1, 0) if view.x() > mid else QPointF(0, 0)
        self.label.setAnchor(anchor)
        top = vb.viewRange()[1][1]
        self.vline.setPos(float(epoch))
        self.vline.show()
        self.label.setPos(float(epoch), top)
        self.label.setPlainText("\n".join(lines))
        self.label.show()
# ============================================================
# 项目卡片
# ============================================================
class _ProjectCard(HoverCard):
    clicked = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project_name = project.name
        self.setFixedSize(340, 160)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 14, 12)
        lay.setSpacing(7)

        head = QHBoxLayout()
        model_icon = QLabel()
        model_icon.setPixmap(QPixmap(str(icon_path(
            _TASK_MODEL_ICON.get(project.task_type, "training.png")))).scaled(
            34, 34, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        model_icon.setFixedSize(34, 34)
        head.addWidget(model_icon)
        title = QLabel(project.name)
        title.setObjectName("cardTitle")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(_make_badge(project.task_type))
        lay.addLayout(head)

        from ...i18n import tr

        updated = QLabel(f"{tr('更新于')} "
                         f"{relative_time(_project_mtime(project.name))}")
        updated.setObjectName("muted")
        created = QLabel(f"{tr('创建于')} {project.created_at or '—'}")
        created.setObjectName("muted")
        model = QLabel(f"{tr('模型：')}{project.model_name or tr('未选择')}")
        model.setObjectName("muted")
        lay.addWidget(updated)
        lay.addWidget(created)
        lay.addWidget(model)
        lay.addStretch(1)

        foot = QHBoxLayout()
        tip = QLabel("点击卡片打开 →")
        tip.setObjectName("muted")
        foot.addWidget(tip)
        foot.addStretch(1)
        btn_delete = QPushButton("删除")
        btn_delete.setObjectName("linkDanger")
        btn_delete.setCursor(Qt.PointingHandCursor)
        btn_delete.clicked.connect(lambda: self.deleteRequested.emit(self.project_name))
        foot.addWidget(btn_delete)
        lay.addLayout(foot)

    def _emit_clicked(self):
        # Q2：子类把 clicked 重定义为 clicked[str]，经基类钩子带参发射
        self.clicked.emit(self.project_name)
# ============================================================
# 训练前预检后台线程（数据体检 + 参数建议；大数据集扫描不阻塞界面）
# ============================================================
class _PreflightWorker(QThread):
    done = pyqtSignal(object)   # {"issues": [HealthIssue], "findings": [Finding],
    #                             "n_images": int, "classes": [str]}
    failed = pyqtSignal(str)

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self._project = project

    def run(self):
        try:
            p = self._project
            opt = load_options()
            task = TaskType(p.task)
            d = Path(p.dataset_dir)
            if task in (TaskType.CLASSIFICATION, TaskType.ANOMALY):
                # 异常检测数据集=OK/NG 子目录纯图，与体检 is_anomaly 同口径
                samples, classes = scan_classification_dataset(d)
                scan = (samples, classes)
            else:
                samples, classes, _ = scan_labelme_dataset(d)
                scan = (samples, classes, False)
            report = run_health_check(
                task, d, opt, val_ratio=float(p.val_ratio), seed=42,
                train_ratio=getattr(p, "train_ratio", None) or None, scan=scan)
            issues = list(report.issues)
            # 缩放后实例像素尺寸校验（YOLO=letterbox、R-CNN=stretch）
            if task in (TaskType.DETECTION, TaskType.SEGMENTATION):
                from ..data_health import HealthIssue, LEVEL_WARN
                from ..scale_guard import scale_issue, scale_issue_text

                resize = (p.augment_ops or {}).get("resize") or {}
                w = int(resize.get("width", 0) or 0)
                h = int(resize.get("height", 0) or 0)
                if str(p.model_name or "").startswith("yolo"):
                    mode = "letterbox"   # YOLO 忽略 resize 增强只取长边
                    w = h = (max(w, h) or 640)
                else:
                    mode = "stretch"
                stat = scale_issue(samples, mode, w, h)
                txt = scale_issue_text(stat) if stat else None
                if txt:
                    issues.append(HealthIssue(
                        LEVEL_WARN, "small_instance",
                        "缩放后过小实例占比偏高", txt[0], txt[1]))
            findings = param_findings(p, dataset_facts(samples),
                                      _cuda_available())
            self.done.emit({"issues": issues, "findings": findings,
                            "n_images": report.n_images, "classes": report.classes})
        except Exception as e:  # 预检失败不阻塞训练（训练线程另有校验兜底）
            self.failed.emit(f"{type(e).__name__}: {e}")

# ---- Cython 编译态槽包装（拆分前 training_window 模块级调用，随迁到此）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())


