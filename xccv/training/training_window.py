"""训练平台窗口（组合根，0.7.6 P-1 页级拆分）：
- TrainingWindow：项目列表窗口（项目卡片墙：创建 / 打开 / 删除）
- ProjectWindow：项目窗口（组合根：信息页/Quick 面板/训练页/历史视图经
  panels 包 mixin 继承装配），每个项目一个独立窗口，支持多开；
  重复打开同一项目只激活已有窗口
- CreateProjectDialog：创建项目对话框

页面实现：panels/{common, info_page, quick_panel, train_page, history_view}
页级结构快照锁：tests/modules/snapshots/training_pages.json
运行：python -m xccv.training.training_window
"""

if __name__ == "__main__":  # V0.6 T1-3：-m 直启先亮加载中界面，再做下方重型模块级导入
    from xccv.platform_entry import early_splash

    early_splash("training", "训练平台", heavy=True)

import sys
import time
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget,
)
from ..engine.dataset.base import TaskType
from ..i18n import tr
from .project import (Project, create_project, delete_project, list_projects, load_project, save_project)
from ..ui_style import bind_style, icon_path, token
from .trainer_thread import TrainThread
from ..gui_utils import ensure_console_streams
from ..ui_style import APP_QSS, ACCENT  # noqa: F401  兼容保留（拆分前历史导入）
from .panels.common import (  # noqa: F401  兼容再导出（外部/历史脚本按旧路径导入）
    AUGMENT_ROWS, DEFAULT_LR, _CURVE_COLORS, _Combo, _DoubleSpin, _IntAxis,
    _NoWheel, _PagePlot, _PlotHover, _PRESET_DEFS, _PRESET_DEFS_FREE,
    _PreflightWorker, _ProjectCard, _Spin, _TASK_BADGES, _TASK_MODEL_ICON,
    _apply_theme, _cuda_available, _make_badge, _safe_run_tag, _status_dot_icon,
    _style_plot, _t2, _t3,
)
from .panels.history_view import HistoryViewMixin
from .panels.info_page import InfoPageMixin, rotate_guard_decision  # noqa: F401
from .panels.quick_panel import QuickPanelMixin
from .panels.train_page import TrainPageMixin

import logging

log = logging.getLogger(__name__)

ensure_console_streams()  # pythonw 下 stdout/stderr 为 None，先兜底


# ============================================================
# 项目窗口：一个项目一个独立窗口
# ============================================================
class ProjectWindow(InfoPageMixin, TrainPageMixin, QuickPanelMixin, HistoryViewMixin, QMainWindow):
    def __init__(self, project: Project, on_renamed=None, on_closed=None):
        super().__init__()
        self.project = project
        self.on_renamed = on_renamed      # (旧名, 新名)
        self.on_closed = on_closed        # (项目名,)
        self.thread: TrainThread | None = None
        self._history: list[dict] = []        # 当前展示的历史（实时或某条历史记录）
        self._live_history: list[dict] = []   # 正在进行的训练（实时曲线数据）
        self._latest_progress: tuple | None = None  # 最近一次轮内进度 (epoch, 批序, 批总数)
        self._live_start: str = ""
        self._live_start_dt: datetime | None = None
        self._selected_run_dir: Path | None = None
        self._run_dir: Path | None = None
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(1000)
        self._live_timer.timeout.connect(self._tick_live)

        from ..i18n import tr

        self.setWindowTitle(f"{project.name}  -  {tr('训练平台')}")
        self.setWindowIcon(QIcon(str(icon_path(
            _TASK_MODEL_ICON.get(project.task_type, "training.png")))))
        self.resize(1280, 900)
        _apply_theme(self)
        self._build_ui()
        self._fill_info_from_project()
        self._fill_params_from_project()
        QTimer.singleShot(0, self._refresh_stats)  # 窗口先显示，统计后台补齐（避免打开卡顿）
        self._switch_detail(0)
        QTimer.singleShot(0, self._refresh_augment_preview)  # 布局完成后渲染预览
        from ..i18n import localize_widget_tree  # 动态窗口：英文模式替换层

        localize_widget_tree(self)
    # ================= 界面骨架 =================
    def _build_ui(self):
        from ..i18n import tr

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶栏：标题 + 任务徽章 + 分段切换
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        head = QHBoxLayout(header)
        head.setContentsMargins(18, 8, 18, 8)
        self.det_title = QLabel(self.project.name)
        self.det_title.setObjectName("appTitle")
        head.addWidget(self.det_title)
        head.addWidget(_make_badge(self.project.task_type))
        head.addStretch(1)
        self.btn_page_info = QPushButton("项目信息")
        self.btn_page_info.setObjectName("seg")
        self.btn_page_info.setCheckable(True)
        self.btn_page_info.setChecked(True)
        self.btn_page_info.clicked.connect(lambda: self._switch_detail(0))
        self.btn_page_train = QPushButton("训练可视化")
        self.btn_page_train.setObjectName("seg")
        self.btn_page_train.setCheckable(True)
        self.btn_page_train.clicked.connect(lambda: self._switch_detail(1))
        head.addWidget(self.btn_page_info)
        head.addWidget(self.btn_page_train)
        # 齿轮入口：开发者选项（数据体检 / 预检阈值，）。
        # ⚙ 在雅黑字体下渲染过小被裁切（实测），指定符号字体修复
        self.btn_dev = QPushButton("⚙")
        self.btn_dev.setFont(QFont("Segoe UI Symbol", 14))
        self.btn_dev.setFixedSize(48, 36)
        self.btn_dev.setToolTip(tr("开发者选项：数据体检与训练前预检的阈值标准"))
        self.btn_dev.clicked.connect(self._open_dev_options)
        head.addWidget(self.btn_dev)
        root.addWidget(header)

        self.detail_stack = QStackedWidget()
        root.addWidget(self.detail_stack, stretch=1)
        self.detail_stack.addWidget(self._build_info_page())   # 0
        self.detail_stack.addWidget(self._build_train_page())  # 1
        self.setCentralWidget(central)
    def _switch_detail(self, index: int):
        self.detail_stack.setCurrentIndex(index)
        self.btn_page_info.setChecked(index == 0)
        self.btn_page_train.setChecked(index == 1)
        if index == 1:
            self._reload_history()  # 进入可视化页时重扫模型目录，历史记录保持最新
    def i18n_rebuild_blocked(self) -> bool:
        """语言重建忙时钩子——训练/预检等线程存活即禁重建。"""
        for attr in ("thread", "_preflight_worker", "_health_worker",
                     "_stats_thread"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                return True
        return False
    def closeEvent(self, event):
        for attr in ("_preflight_worker", "_health_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                QMessageBox.warning(self, tr("后台任务进行中"),
                                    tr("数据体检 / 训练前预检正在进行，请稍候再关闭窗口"))
                event.ignore()
                return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, tr("训练中"),
                                tr("训练正在进行，无法关闭窗口；请先等待完成或停止训练"))
            event.ignore()
            return
        if self.on_closed is not None:
            self.on_closed(self.project.name)
        dg = getattr(self, "data_grid", None)
        if dg is not None:      # 缩略/扫描线程收尾（P144 纪律）
            dg.shutdown()
        event.accept()


# ============================================================
# 创建项目对话框
# ============================================================
class CreateProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("创建项目")
        self.setMinimumWidth(480)
        _apply_theme(self)

        form = QFormLayout(self)
        form.setContentsMargins(18, 16, 18, 14)
        form.setSpacing(10)
        self.fld_name = QLineEdit()
        self.fld_name.setPlaceholderText("例如：安全帽检测")
        self.fld_dataset = QLineEdit()
        btn_ds = QPushButton("浏览…")
        btn_ds.clicked.connect(self._browse_dataset)
        ds_row = QHBoxLayout()
        ds_row.addWidget(self.fld_dataset, stretch=1)
        ds_row.addWidget(btn_ds)
        self.fld_model_dir = QLineEdit()
        self.fld_model_dir.setPlaceholderText("留空则使用平台默认位置")
        btn_md = QPushButton("浏览…")
        btn_md.clicked.connect(self._browse_model_dir)
        md_row = QHBoxLayout()
        md_row.addWidget(self.fld_model_dir, stretch=1)
        md_row.addWidget(btn_md)
        self.cmb_task = QComboBox()
        from ..i18n import tr

        for t in TaskType:
            self.cmb_task.addItem(tr(t.label), t.value)
        self.cmb_task.currentIndexChanged.connect(self._fill_models)
        self.cmb_model = QComboBox()

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("创建项目")
        self.buttons.button(QDialogButtonBox.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form.addRow("项目名称", self.fld_name)
        form.addRow("数据集路径", ds_row)
        form.addRow("模型储存路径", md_row)
        form.addRow("任务类型", self.cmb_task)
        form.addRow("训练模型", self.cmb_model)
        form.addRow(self.buttons)
        self._fill_models()
        from ..i18n import localize_widget_tree  # 动态对话框：英文模式替换层

        localize_widget_tree(self)

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择数据集文件夹"))
        if path:
            self.fld_dataset.setText(path)

    def _browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择模型储存文件夹"))
        if path:
            self.fld_model_dir.setText(path)

    def _fill_models(self):
        task = self.cmb_task.currentData()
        self.cmb_model.clear()
        from ..engine.models import list_models  # torch 延迟导入

        for m in list_models(task):
            from ..i18n import tr

            self.cmb_model.addItem(f"{m['name']}  ({tr(m['desc'])})", m["name"])

    def create(self) -> Project:
        """执行创建；失败抛 ValueError（由调用方提示）。"""
        project = create_project(
            name=self.fld_name.text(),
            dataset_dir=self.fld_dataset.text(),
            model_dir=self.fld_model_dir.text(),
            task=self.cmb_task.currentData(),
        )
        project.model_name = self.cmb_model.currentData() or ""
        project.learning_rate = DEFAULT_LR.get(project.model_name, 0.001)
        project.save()
        return project


# ============================================================
# 项目列表窗口：项目卡片墙
# ============================================================


class TrainingWindow(QMainWindow):
    def __init__(self, open_project: str | None = None):
        super().__init__()
        self.setWindowTitle(f"{tr('XCcv 训练平台')} - {tr('项目列表')}")
        from PyQt5.QtGui import QIcon
        self.setWindowIcon(QIcon(str(icon_path("training.png"))))
        self.resize(1120, 720)
        _apply_theme(self)
        self._project_windows: dict[str, ProjectWindow] = {}
        self._cards: list[_ProjectCard] = []

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶栏
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(66)
        head = QHBoxLayout(header)
        head.setContentsMargins(22, 10, 22, 10)
        app_title = QLabel(tr("XCcv 训练平台"))
        app_title.setObjectName("appTitle")
        head.addWidget(app_title)
        head.addStretch(1)
        btn_create = QPushButton(tr("＋ 新建项目"))
        btn_create.setObjectName("accent")
        btn_create.clicked.connect(self._create_project)
        head.addWidget(btn_create)
        root.addWidget(header)

        # 内容区：项目卡片墙
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(22, 16, 22, 12)
        body_lay.setSpacing(10)
        self.list_hint = QLabel("")
        self.list_hint.setObjectName("hint")
        body_lay.addWidget(self.list_hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.cards_host = QWidget()
        self.cards_grid = QGridLayout(self.cards_host)
        self.cards_grid.setContentsMargins(0, 0, 0, 0)
        self.cards_grid.setSpacing(14)
        self.cards_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self.cards_host)
        body_lay.addWidget(scroll, stretch=1)
        root.addWidget(body, stretch=1)

        self.setCentralWidget(central)
        self._refresh_project_list()
        if open_project:
            # 主界面「最近项目」直达：argv 透传项目名，布局完成后自动打开
            self._pending_open = open_project
            QTimer.singleShot(0, self._open_pending_project)

    def _open_pending_project(self):
        name = getattr(self, "_pending_open", "")
        if not name:
            return
        self._pending_open = ""
        try:
            self._open_project(name)
        except Exception as e:   # 项目不存在/损坏：回退项目列表 + 人话提示
            QMessageBox.warning(self, tr("打开项目失败"),
                                f"{tr('项目')}「{name}」{tr('无法打开')}：{e}")

    # ---- 卡片墙 ----
    def _refresh_project_list(self):
        self._cards.clear()
        while self.cards_grid.count():
            item = self.cards_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        names = list_projects()
        for i, name in enumerate(names):
            project = load_project(name)
            card = _ProjectCard(project)
            card.clicked.connect(self._open_project)
            card.deleteRequested.connect(self._confirm_delete)
            self.cards_grid.addWidget(card, i // 3, i % 3)
            self._cards.append(card)
        # 空状态引导（EmptyState 组件）：有项目时不占位
        if not names:
            from ..ui_widgets import EmptyState

            self.cards_grid.addWidget(
                EmptyState("🗂", tr("还没有项目"),
                           tr("点击右上角「＋ 新建项目」，30 秒完成第一个训练项目")),
                0, 0)
        self.list_hint.setText("")

    def project_count(self) -> int:
        return len(self._cards)

    # ---- 打开 / 删除 ----
    def _open_project(self, name: str):
        from PyQt5.QtCore import Qt as _Qt

        window = self._project_windows.get(name)
        if window is not None and window.isVisible():
            window.activateWindow()
            window.raise_()
            return
        # 打开项目要加载深度学习组件（渲染卡数秒）——顶部悬浮胶囊提示
        # （U1：对齐标注平台横幅样式，单行/半透明/圆角；颜色与标注平台
        # 的琥珀底作微差取暗暖灰；V0.7）
        from ..i18n import tr

        banner = QLabel(f"{tr('正在打开项目')}「{name}」…", self)
        banner.setObjectName("openProjectBanner")
        banner.setStyleSheet(
            "QLabel#openProjectBanner { background: rgba(58, 46, 34, 224);"
            " color: white; padding: 6px 18px; border-radius: 12px;"
            " font-weight: bold; font-size: 13px; }")
        banner.adjustSize()
        banner.move((self.width() - banner.width()) // 2, 12)
        banner.show()
        banner.raise_()
        # V0.7.1 实测：重开平台后点卡片，界面常无任何等待提示——窗口空闲时
        # 无挂起重绘，processEvents 单轮不一定派发横幅的绘制事件，横幅从未
        # 上屏就进入 torch 重载的长阻塞。repaint() 强制同步绘制一次保底。
        banner.repaint()
        QApplication.processEvents()
        QApplication.processEvents()
        t0 = time.monotonic()
        self.statusBar().showMessage(
            f"{tr('正在打开项目')}「{name}」…", 8000)
        QApplication.setOverrideCursor(_Qt.WaitCursor)
        try:
            window = ProjectWindow(
                load_project(name),
                on_renamed=self._on_project_renamed,
                on_closed=self._on_project_closed,
            )
            window.show()
        finally:
            QApplication.restoreOverrideCursor()
            # 等待横幅固定可见 ≥0.7s：构建很快时也给出可感知的反馈，
            # 避免"一闪而过/根本没出现"的间歇观感
            remain_ms = max(0, int(700 - (time.monotonic() - t0) * 1000))

            def _hide_open_banner():
                banner.hide()
                banner.setParent(None)
                banner.deleteLater()

            QTimer.singleShot(remain_ms, _hide_open_banner)
        self.statusBar().showMessage(f"{tr('项目')}「{name}」{tr('已打开')}", 4000)
        self._project_windows[name] = window

    def on_platform_message(self, args):
        """单实例激活携带业务参数：args[0]=项目名 → 直接打开该项目。

        主界面「最近项目」卡片在训练平台已运行时点下：新进程把项目名经
        单实例管道转发到本钩子（此前只激活窗口不打开项目）。
        """
        name = args[0] if args else ""
        if not name:
            return
        QTimer.singleShot(0, lambda: self._safe_open(name))

    def _safe_open(self, name: str):
        try:
            self._open_project(name)
        except Exception as e:   # 项目不存在/损坏：人话提示
            QMessageBox.warning(self, tr("打开项目失败"),
                                f"{tr('项目')}「{name}」{tr('无法打开')}：{e}")

    def _confirm_delete(self, name: str):
        if QMessageBox.question(self, tr("删除项目"),
                                f"{tr('确定删除项目')}「{name}」{tr('吗？（不影响已保存的模型文件）')}"
                                ) != QMessageBox.Yes:
            return
        window = self._project_windows.pop(name, None)
        if window is not None and not window.close():
            # closeEvent 被拒绝（如训练进行中禁关窗）：取消删除，
            # 避免删掉正在使用的项目目录
            self._project_windows[name] = window
            self.statusBar().showMessage(
                f"{tr('项目')}「{name}」{tr('的窗口正在忙碌（训练中无法关闭），已取消删除')}",
                5000)
            return
        delete_project(name)
        self._refresh_project_list()

    def _on_project_renamed(self, old_name: str, new_name: str):
        window = self._project_windows.pop(old_name, None)
        if window is not None:
            self._project_windows[new_name] = window
        self._refresh_project_list()

    def _on_project_closed(self, name: str):
        self._project_windows.pop(name, None)

    def open_project_windows(self) -> dict:
        """已打开的项目窗口（测试用）。"""
        return dict(self._project_windows)

    def i18n_transfer_children(self, new_host: "TrainingWindow") -> None:
        """语言重建时项目窗跟随换语言（平台宿主重建路径调用）。

        空闲项目窗：原地重建（watcher 已把进程缓存切到新语言，此时构建
        即新语言）→ 转移到新列表窗 → 关旧窗；忙碌项目窗（训练/预检中）
        原样过户到新列表窗（保持旧语言，重开窗口生效）。
        过户/重建一律重绑 on_renamed/on_closed——旧列表窗即将销毁，
        回调不能指向它（否则关闭项目窗时撞已删 C++ 对象）。"""
        for name, old in list(self._project_windows.items()):
            self._project_windows.pop(name, None)
            if not old.isVisible():
                continue  # 已隐藏/关闭：不转移
            busy = False
            try:
                busy = bool(old.i18n_rebuild_blocked())
            except Exception:  # noqa: BLE001（钩子异常按空闲处理）
                log.debug("语言重建忙判定钩子异常（按空闲处理）")
            if busy:
                old.on_renamed = new_host._on_project_renamed
                old.on_closed = new_host._on_project_closed
                new_host._project_windows[name] = old
                try:
                    old.statusBar().showMessage(
                        tr("任务运行中，语言重开窗口后生效"), 10000)
                except Exception:  # noqa: BLE001（无状态栏不提示）
                    log.debug("旧窗状态栏提示失败")
                continue
            try:
                new_pw = ProjectWindow(
                    load_project(name),
                    on_renamed=new_host._on_project_renamed,
                    on_closed=new_host._on_project_closed,
                )
                new_pw.show()
                new_host._project_windows[name] = new_pw
                old.on_closed = None  # 旧窗关闭不再回调（项目窗已在册）
                old.close()
                old.deleteLater()
            except Exception:  # noqa: BLE001（重建失败保旧窗，对齐宿主语义）
                old.on_renamed = new_host._on_project_renamed
                old.on_closed = new_host._on_project_closed
                new_host._project_windows[name] = old

    # ---- 创建 ----
    def _create_project(self):
        dialog = CreateProjectDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            project = dialog.create()
        except ValueError as e:
            QMessageBox.warning(self, tr("无法创建"), str(e))
            return
        self._refresh_project_list()
        self._open_project(project.name)


def main():
    from ..platform_entry import run_platform

    arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else None
    run_platform("training", "训练平台",
                 lambda: TrainingWindow(open_project=arg), heavy=True)


if __name__ == "__main__":
    main()


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

