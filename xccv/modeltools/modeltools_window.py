"""模型工具平台：面向训练产物的后处理工具集（与标注/训练/测试平台同级独立）。

界面为工具卡片墙（参考项目卡片设计，组件独立实现），点击卡片打开独立工具窗口
（与训练平台"每个项目一个窗口"同模式，支持多开；重复打开同一工具只激活已有窗口）：
- 模型剪枝：训练前剪枝（按项目数据集生成剪枝模型供训练平台迁移学习）与
  训练后剪枝（处理某次训练记录的权重，输出 best_pruned.pt）；
- 模型加速（TensorRT）：把权重导出为 TensorRT 引擎（需 GPU + TensorRT 环境）。
运行：python -m xccv.modeltools.modeltools_window
"""

if __name__ == "__main__":  # -m 直启先亮加载中界面，再做下方重型模块级导入
    from xccv.platform_entry import early_splash

    early_splash("tools", "模型工具", heavy=True)

# QtWebEngine 必须在 QApplication 创建前导入/初始化（流程图编辑器的 web 画布）。
# 注意：必须位于模块 docstring 之后、任何 QApplication 创建之前。
from PyQt5.QtCore import Qt as _Qt
from PyQt5.QtWidgets import QApplication as _QApp

if _QApp.instance() is None:
    _QApp.setAttribute(_Qt.AA_ShareOpenGLContexts, True)
try:
    import PyQt5.QtWebEngineWidgets  # noqa: F401,E402
except ImportError:
    # 模块导入晚于 QApplication 创建（仅测试环境会出现；真实启动路径
    # 模块先于 QApplication 导入，必走 try 分支）
    pass

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from ..app_paths import app_root
from ..gui_utils import auto_cleanup_thread, ensure_console_streams
from ..i18n import tr  # 模块级：动态文案站点直接用（方法级导入保留兼容）
from ..training.dataset_stats import collect_stats
from ..training.project import (
    PRUNED_DIR_NAME, Project, list_projects, load_project,
)
from ..training.runs import scan_runs
from ..ui_style import ACCENT, bind_style, icon_path, token  # noqa: F401
from PyQt5.QtGui import QPixmap
from ..ui_style import APP_QSS as _TOOL_QSS, apply_app_theme  # noqa: F401

ensure_console_streams()

log = logging.getLogger(__name__)


def _export_engine(src, dst, imgsz=None, half=True, progress=None):
    """torch 延迟导入：加速导出在工作线程里才引入 torch/引擎链，
    让本模块导入保持轻量、启动闪屏先显示。"""
    from ..engine.model_export import export_engine

    return export_engine(src, dst, imgsz=imgsz, half=half, progress=progress)

# 训练记录里的权重文件（平台产物为明文 best.pt）
_ORIGINAL_WEIGHTS = ("best.pt",)
_PRUNED_WEIGHTS = ("best_pruned.pt",)


class _ToolCard(QFrame):
    """工具卡片：标题 + 描述 + 进入提示（视觉参考项目卡片，组件独立实现）。"""

    clicked = pyqtSignal(str)

    def __init__(self, key: str, title: str, desc: str, color: str,
                 enabled: bool = True, parent=None, icon: str = "tools.png"):
        super().__init__(parent)
        self.tool_key = key
        self.setObjectName("card")
        self.setFixedSize(340, 152)
        if not enabled:
            self.setEnabled(False)  # 即将上线的工具不响应点击
        else:
            self.setCursor(Qt.PointingHandCursor)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 14, 12)
        lay.setSpacing(7)

        head = QHBoxLayout()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(QPixmap(str(icon_path(icon))).scaled(
            44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_lbl.setFixedSize(44, 44)
        head.addWidget(icon_lbl)
        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {color if enabled else '#9aa7b8'}; font-size: 19px; font-weight: bold;"
            "background: transparent; border: none;")
        head.addWidget(title_label)
        head.addStretch(1)
        if not enabled:
            soon = QLabel("即将上线")
            soon.setObjectName("muted")
            head.addWidget(soon)
        lay.addLayout(head)

        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        bind_style(desc_label, lambda: (
            f"color: {token('text_2')}; font-size: 13px; background: transparent; border: none;"))
        lay.addWidget(desc_label, stretch=1)

        foot = QHBoxLayout()
        tip = QLabel("点击打开窗口 →" if enabled else "敬请期待")
        tip.setStyleSheet(
            f"color: {color if enabled else '#9aa7b8'}; font-size: 14px;"
            "background: transparent; border: none;")
        foot.addWidget(tip)
        foot.addStretch(1)
        lay.addLayout(foot)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.clicked.emit(self.tool_key)
        super().mouseReleaseEvent(event)


class WorkerThread(QThread):
    """后台执行工具任务（剪枝/加速导出），避免界面卡顿。"""

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(float, str)   # (frac, text)：frac<0 = 忙碌态

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:
            log.error("工具任务失败: %s: %s", type(e).__name__, e, exc_info=True)
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---- 剪枝进程隔离参数（强制防死机项）----
PRUNE_RAM_LIMIT_PCT = 80     # 系统 RAM 占比 ≥ 此值 → 自动终止剪枝子进程+告警
PRUNE_HARD_TIMEOUT_S = 600   # 硬超时（秒），超时同样自动终止
PRUNE_POLL_MS = 500          # 水位轮询周期


class _PruneProc(QObject):
    """剪枝子进程托管：主窗口进程零剪枝内存。

    QProcess 启动 `python -m xccv.engine.prune_worker`，QTimer 轮询子进程
    RSS 与系统 RAM 水位（psutil）；超水位/硬超时自动终止；「紧急终止」=
    taskkill /T /F 兜底 + QProcess.kill；stdout JSON 行协议解析相位与结果；
    终止/完成后水位曲线（采样表尾段）落日志，状态经 finished(dict) 回 UI。
    """

    finished = pyqtSignal(dict)   # {"status": applied|<其他result.status>|killed|error, ...}
    phase = pyqtSignal(str, str)  # (相位名, 详情)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: QProcess | None = None
        self._timer: QTimer | None = None
        self._samples: list[tuple[float, float, float]] = []  # (s, rss_mb, sys_pct)
        self._peak_mb = 0.0
        self._t0 = 0.0
        self._timeout_s = PRUNE_HARD_TIMEOUT_S
        self._final: dict | None = None
        self._killed = False
        self._kill_reason = ""

    # ---- 生命周期 ----
    def running(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, args: list, timeout_s: float = PRUNE_HARD_TIMEOUT_S):
        self._samples = []
        self._peak_mb = 0.0
        self._final = None
        self._killed = False
        self._kill_reason = ""
        self._timeout_s = float(timeout_s)
        self._t0 = time.monotonic()
        self.proc = QProcess(self)
        self.proc.setProgram(sys.executable)
        self.proc.setArguments(
            ["-m", "xccv.engine.prune_worker", *args])
        self.proc.setWorkingDirectory(str(app_root()))
        self.proc.setProcessChannelMode(QProcess.SeparateChannels)
        self.proc.readyReadStandardOutput.connect(self._drain_stdout)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self.proc.start()
        if not self.proc.waitForStarted(8000):
            self.proc = None
            raise RuntimeError("剪枝子进程启动失败（python -m xccv.engine.prune_worker）")
        self._timer = QTimer(self)
        self._timer.setInterval(PRUNE_POLL_MS)
        self._timer.timeout.connect(self._watchdog)
        self._timer.start()

    def kill(self, reason: str):
        """紧急终止：taskkill /T /F 兜底 + QProcess.kill。"""
        if not self.running():
            return
        self._killed = True
        self._kill_reason = reason
        pid = self.proc.processId() if self.proc else 0
        log.warning("剪枝子进程紧急终止：%s（pid=%s）", reason, pid)
        try:
            if pid > 0:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=10, check=False)
        except Exception:  # noqa: BLE001
            log.warning("taskkill 强制结束失败", exc_info=True)
        try:
            self.proc.kill()
        except RuntimeError:  # noqa: BLE001  进程刚好自行退出
            pass

    # ---- 看门狗 ----
    def _watchdog(self):
        if not self.running():
            return
        elapsed = time.monotonic() - self._t0
        rss_mb = sys_pct = 0.0
        try:
            import psutil

            pid = self.proc.processId()
            if pid > 0:
                rss_mb = psutil.Process(pid).memory_info().rss / 1e6
                sys_pct = psutil.virtual_memory().percent
        except Exception:  # noqa: BLE001  进程刚退出/psutil 异常：本轮跳过
            log.debug("进程资源采样失败（本轮跳过）")
        self._samples.append(
            (round(elapsed, 1), round(rss_mb, 1), round(sys_pct, 1)))
        self._peak_mb = max(self._peak_mb, rss_mb)
        if sys_pct >= PRUNE_RAM_LIMIT_PCT:
            self.kill(f"系统内存水位 {sys_pct:.0f}% ≥ {PRUNE_RAM_LIMIT_PCT}%"
                      f"（子进程 RSS {rss_mb:.0f} MB）")
            return
        if elapsed >= self._timeout_s:
            self.kill(f"硬超时 {self._timeout_s:.0f}s")

    # ---- stdout JSON 行协议 ----
    def _drain_stdout(self):
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        for line in data.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            ev = obj.get("event")
            if ev == "phase":
                self.phase.emit(str(obj.get("name", "")),
                                str(obj.get("detail", "")))
            elif ev == "done":
                result = obj.get("result", {})
                self._final = {"status": result.get("status", "applied"),
                               "result": result}
            elif ev == "error":
                self._final = {"status": "error",
                               "message": str(obj.get("message", ""))}

    def _on_error(self, err):
        if err == QProcess.FailedToStart and not self._killed:
            self._final = {"status": "error", "message": "剪枝子进程启动失败"}

    def _on_finished(self, code, status):
        if self._timer is not None:
            self._timer.stop()
        self._drain_stdout()  # 收尾排水：done/error 常滞留缓冲
        elapsed = time.monotonic() - self._t0
        log.info("剪枝子进程退出: code=%s elapsed=%.1fs peak_rss=%.0fMB "
                 "水位尾段=%s", code, elapsed, self._peak_mb,
                 self._samples[-8:])
        if self._killed:
            out = {"status": "killed", "reason": self._kill_reason,
                   "peak_rss_mb": self._peak_mb,
                   "elapsed_s": round(elapsed, 1)}
        else:
            out = dict(self._final or {})
            out.setdefault("peak_rss_mb", self._peak_mb)
            out.setdefault("elapsed_s", round(elapsed, 1))
            if not out:
                out = {"status": "error", "message": f"子进程退出码 {code}"}
        proc, self.proc = self.proc, None
        if proc is not None:
            proc.deleteLater()
        self.finished.emit(out)


class _ToolSpin(QSpinBox):
    """滚轮不改变数值（防误触）。"""

    def wheelEvent(self, ev):
        ev.ignore()


class _ToolWindow(QMainWindow):
    """工具窗口基类：一个工具一个独立窗口（与训练平台项目窗口同模式）。

    顶栏 + 项目/训练记录选择（各工具基于同一份模型来源）+ 子类工具设置区；
    后台任务线程、任务失败提示与完成后的刷新为公共逻辑。
    """

    tool_key = ""
    tool_title = ""

    def __init__(self, on_closed=None):
        super().__init__()
        self.on_closed = on_closed      # (工具 key,) 窗口关闭时回调
        self.setWindowTitle(f"{self.tool_title}  -  {tr('XCcv 模型工具')}")
        self.setWindowIcon(QIcon(str(icon_path(
            getattr(self, "tool_icon", "tools.png")))))
        self.resize(880, 640)
        apply_app_theme(self)
        self._project: Project | None = None
        self._worker: WorkerThread | None = None
        self._build_ui()
        if not self.file_mode:
            self._refresh_projects()
        from ..i18n import localize_widget_tree  # 动态工具窗口：英文替换层

        localize_widget_tree(self)

    # ================= 界面骨架 =================
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        head = QHBoxLayout(header)
        head.setContentsMargins(22, 10, 22, 10)
        title = QLabel(self.tool_title)
        title.setObjectName("appTitle")
        head.addWidget(title)
        head.addStretch(1)
        root.addWidget(header)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(22, 16, 22, 14)
        lay.setSpacing(12)

        self.file_mode = getattr(self, "file_mode", False)
        if self.file_mode:
            self._build_settings(lay)
            root.addWidget(body, stretch=1)
            self.setCentralWidget(central)
            return
        g_sel = QGroupBox("选择模型")
        f_sel = QFormLayout(g_sel)
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self._on_project_changed)
        self.run_combo = QComboBox()
        self.run_combo.setToolTip("选择要处理的训练记录")
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        f_sel.addRow("项目（来自训练平台）", self.project_combo)
        f_sel.addRow("训练记录", self.run_combo)
        lay.addWidget(g_sel)
        # 模型档案摘要行（类别数 / OK 负样本 / 数据与依赖指纹）
        self.manifest_hint = QLabel("")
        self.manifest_hint.setObjectName("hint")
        self.manifest_hint.setWordWrap(True)
        lay.addWidget(self.manifest_hint)

        self._build_settings(lay)
        root.addWidget(body, stretch=1)
        self.setCentralWidget(central)

    def _build_settings(self, lay: QVBoxLayout):
        """子类实现：在表单下方添加自己的工具设置区。"""
        raise NotImplementedError

    # ================= 项目 / 记录选择 =================
    def _refresh_projects(self):
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for name in list_projects():
            self.project_combo.addItem(name, name)
        self.project_combo.blockSignals(False)
        self._on_project_changed()

    def _on_project_changed(self):
        """项目切换：刷新含权重文件的训练记录列表（含已剪枝标记）。"""
        name = self.project_combo.currentData()
        self._project = load_project(name) if name else None
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        if self._project is not None:
            for rec in scan_runs(self._project.model_dir):
                has_weights = any((rec.run_dir / n).is_file() for n in _ORIGINAL_WEIGHTS)
                if not has_weights:
                    continue
                pruned = any((rec.run_dir / n).is_file() for n in _PRUNED_WEIGHTS)
                tag = "（已剪枝）" if pruned else ""
                self.run_combo.addItem(
                    f"{rec.start_time or rec.run_dir.name}　{rec.status_text}{tag}",
                    str(rec.run_dir))
        self._refresh_sources(self.run_combo.currentData())
        self._refresh_manifest(self.run_combo.currentData())
        self.run_combo.blockSignals(False)

    def _refresh_sources(self, run_dir):
        """子类按需实现：记录切换后刷新自己的源文件/设置显示。"""

    def _on_run_changed(self):
        self._refresh_manifest(self.run_combo.currentData())
        self._refresh_sources(self.run_combo.currentData())

    def _refresh_manifest(self, run_dir):
        """读 metrics.json + run_manifest.json 出档案摘要。

        上线检查单口径：类别数（类别分布自查入口）/OK 负样本占比依据/
        数据指纹（数据集是否换过）/依赖与锁版本（现场复现训练环境用）。
        """
        text = ""
        rd = Path(run_dir) if run_dir else None
        if rd is not None and rd.is_dir():
            try:
                m = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
                neg = m.get("neg_train_images") or 0
                parts = [f"类别 {len(m.get('classes') or [])} 个"]
                fp = m.get("dataset_fingerprint")
                if fp:
                    parts.append(f"数据指纹 {str(fp)[:10]}…")
                if neg:
                    parts.append(f"OK 负样本 {neg} 张")
                try:
                    man = json.loads((rd / "run_manifest.json").read_text(
                        encoding="utf-8"))
                    env = man.get("environment") or {}
                    if env.get("torch"):
                        parts.append(f"torch {env['torch']}")
                    if env.get("gpu"):
                        parts.append(str(env["gpu"]))
                    if man.get("lock_sha1"):
                        parts.append(f"依赖锁 {str(man['lock_sha1'])[:10]}…")
                    if man.get("deterministic"):
                        parts.append("确定性训练")
                    app = man.get("application") or {}
                    if app.get("version"):
                        parts.append(f"app {app['version']}")
                except Exception:
                    pass  # V0.7 之前的训练记录无 manifest：只显示 metrics 侧
                text = "模型档案：　" + "　·　".join(parts)
            except Exception:
                log.debug("模型 manifest 摘要构建失败（按空提示）")
                text = ""
        self.manifest_hint.setText(text)

    # ================= 任务执行（公共） =================
    def _busy(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, tr("请稍候"), tr("当前有工具任务正在执行"))
            return True
        return False

    def _current_run_dir(self) -> Path | None:
        data = self.run_combo.currentData()
        return Path(data) if data else None

    def _start_worker(self, fn, on_done, on_progress=None):
        self._worker = WorkerThread(fn)
        self._worker.done.connect(on_done)
        self._worker.failed.connect(self._on_task_failed)
        if on_progress is not None:
            self._worker.progress.connect(on_progress)
        auto_cleanup_thread(self, "_worker", self._worker)  # A-3
        self._worker.start()

    def _set_running(self, running: bool):
        """子类实现：任务执行中禁用/恢复自己的执行按钮。"""
        raise NotImplementedError

    def _on_task_failed(self, message: str):
        self._worker = None
        self._set_running(False)
        QMessageBox.critical(self, tr("任务失败"), message)

    def _refresh_after_task(self):
        """任务完成后刷新记录列表与源文件下拉（新产出的文件立即可选）。"""
        if getattr(self, "file_mode", False):
            return  # 文件模式：无项目/记录依赖
        current = self.run_combo.currentData()
        self._on_project_changed()
        if current is not None:
            idx = self.run_combo.findData(current)
            if idx >= 0:
                self.run_combo.setCurrentIndex(idx)

    def i18n_rebuild_blocked(self) -> bool:
        """语言重建忙时钩子——工具任务/剪枝子进程存活即禁重建。"""
        if self._worker is not None and self._worker.isRunning():
            return True
        proc = getattr(self, "_prune_proc", None)
        return proc is not None and proc.running()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, tr("任务进行中"),
                                tr("当前有工具任务正在执行，请等待完成后再关闭窗口"))
            event.ignore()
            return
        if self.on_closed is not None:
            self.on_closed(self.tool_key)
        event.accept()


class PruneToolWindow(_ToolWindow):
    tool_icon = "prune.png"
    """模型剪枝窗口：训练后剪枝（处理已有训练记录）+ 训练前剪枝（按项目生成剪枝模型）。"""

    tool_key = "prune"
    tool_title = "模型剪枝"

    def _build_settings(self, lay: QVBoxLayout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        inner = QVBoxLayout(body)
        inner.setContentsMargins(0, 4, 0, 0)
        inner.setSpacing(12)

        # 模式切换
        g_mode = QGroupBox("剪枝模式")
        fm = QFormLayout(g_mode)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("训练前剪枝：按项目数据集生成剪枝模型", "pre")
        self.mode_combo.addItem("训练后剪枝：处理某次训练记录的权重", "post")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        fm.addRow(self.mode_combo)
        inner.addWidget(g_mode)

        # ---- 模式 A：训练前剪枝 ----
        g_pre = QGroupBox("训练前剪枝")
        f_pre = QFormLayout(g_pre)
        self.pre_model_combo = QComboBox()
        self.pre_model_combo.setToolTip(
            "选择要剪枝的模型架构：按数据集类别数与输入尺寸构建预训练模型后剪枝")
        self.pre_ratio_spin = _ToolSpin()
        self.pre_ratio_spin.setRange(5, 90)
        self.pre_ratio_spin.setValue(30)
        self.pre_ratio_spin.setSuffix(" %")
        self.pre_ratio_spin.setToolTip(
            "剪掉的通道比例：30% 起步通常精度损失 <1 个点，训练可恢复；\n"
            "只剪骨干/颈部冗余通道，分类头/检测头/RPN/工业模块不受影响")
        self.pre_size_hint = QLabel("")
        self.pre_size_hint.setObjectName("accentText")
        self.btn_preprune = QPushButton("▶ 生成剪枝模型")
        self.btn_preprune.setObjectName("accent")
        self.btn_preprune.clicked.connect(self._run_preprune)
        f_pre.addRow("模型架构", self.pre_model_combo)
        f_pre.addRow("剪枝比例", self.pre_ratio_spin)
        f_pre.addRow("输入尺寸", self.pre_size_hint)
        f_pre.addRow(self.btn_preprune)
        inner.addWidget(g_pre)
        self.pre_group = g_pre

        pre_hint = QLabel(
            "剪枝模型（.pt）保存到项目「模型储存路径\\pruned」文件夹；\n"
            "训练平台的模型下拉会自动出现该模型，选择后即可开始迁移学习。")
        pre_hint.setObjectName("hint")
        pre_hint.setWordWrap(True)
        inner.addWidget(pre_hint)
        self.pre_hint = pre_hint

        # ---- 模式 B：训练后剪枝 ----
        g_post = QGroupBox("训练后剪枝（处理所选训练记录）")
        f_post = QFormLayout(g_post)
        self.ratio_spin = _ToolSpin()
        self.ratio_spin.setRange(5, 90)
        self.ratio_spin.setValue(30)
        self.ratio_spin.setSuffix(" %")
        self.ratio_spin.setToolTip(
            "剪掉的通道比例：30% 起步通常精度损失 <1 个点，微调可恢复；\n"
            "只剪骨干/颈部冗余通道，分类头/检测头/RPN/工业模块不受影响；\n"
            "YOLO 权重剪枝已停用（torch-pruning 上游对 ultralytics 图内存爆炸）；\n"
            "剪后副本与源文件同格式，测试平台可直接加载对比效果")
        self.btn_prune = QPushButton("▶ 执行剪枝")
        self.btn_prune.setObjectName("accent")
        self.btn_prune.clicked.connect(self._run_prune)
        f_post.addRow("剪枝比例", self.ratio_spin)
        f_post.addRow(self.btn_prune)
        inner.addWidget(g_post)
        self.post_group = g_post

        self.prune_result = QLabel("")
        bind_style(self.prune_result, lambda: f"color: {token('primary_deep')}; font-weight: bold;")
        self.prune_result.setWordWrap(True)
        inner.addWidget(self.prune_result)

        # 紧急终止：执行中常驻显示；剪枝在独立子进程，终止即时生效
        self.btn_kill = QPushButton("■ 紧急终止剪枝")
        self.btn_kill.clicked.connect(self._kill_prune)
        self.btn_kill.setVisible(False)
        inner.addWidget(self.btn_kill)
        self._prune_proc = None  # _PruneProc（QProcess 托管）

        post_hint = QLabel(
            "剪枝结果保存为所选记录文件夹内的 best_pruned.pt（源权重不变）；\n"
            "可在测试平台直接加载该文件，与原模型对比速度与精度。")
        post_hint.setObjectName("hint")
        post_hint.setWordWrap(True)
        inner.addWidget(post_hint)

        inner.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, stretch=1)
        self._on_mode_changed()

    # ---- 模式切换 ----
    def _on_mode_changed(self):
        pre = self.mode_combo.currentData() == "pre"
        self.pre_group.setVisible(pre)
        self.pre_hint.setVisible(pre)
        self.post_group.setVisible(not pre)
        self.run_combo.setEnabled(not pre)  # 训练前剪枝只需要项目
        self._update_pre_size_hint()

    def _update_pre_size_hint(self):
        resize = (getattr(self, "_project", None) and self._project.augment_ops or {}).get("resize")
        if resize and resize.get("width") and resize.get("height"):
            self.pre_size_hint.setText(
                f"{int(resize['width'])} × {int(resize['height'])}（随项目增强策略 resize）")
        else:
            self.pre_size_hint.setText("模型默认")

    def _on_project_changed(self):
        super()._on_project_changed()
        # 刷新训练前剪枝的模型架构列表（按项目任务过滤）
        task = getattr(self, "_project", None) and self._project.task
        if hasattr(self, "pre_model_combo"):
            self.pre_model_combo.blockSignals(True)
            self.pre_model_combo.clear()
            if task:
                from ..engine.models import list_models  # torch 延迟导入

                for m in list_models(task):
                    from ..i18n import tr

                    self.pre_model_combo.addItem(
                        f"{m['name']}  ({tr(m['desc'])})", m["name"])
            self.pre_model_combo.blockSignals(False)
            self._update_pre_size_hint()

    # ---- 执行 ----
    def _set_running(self, running: bool):
        self.btn_prune.setEnabled(not running)
        self.btn_preprune.setEnabled(not running)
        self.btn_kill.setVisible(running)

    def _busy(self) -> bool:
        if self._prune_proc is not None and self._prune_proc.running():
            QMessageBox.information(self, tr("请稍候"),
                                    tr("剪枝子进程正在执行（可点「紧急终止剪枝」取消）"))
            return True
        return super()._busy()

    def closeEvent(self, event):
        if self._prune_proc is not None and self._prune_proc.running():
            ans = QMessageBox.question(
                self, tr("剪枝进行中"),
                tr("剪枝子进程仍在执行。终止剪枝并关闭窗口吗？")
                + "\n" + tr("（终止是安全的：源权重不受影响，半截输出不会落盘）"))
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            self._prune_proc.kill("窗口关闭")
            proc = self._prune_proc.proc
            if proc is not None:
                proc.waitForFinished(5000)
        super().closeEvent(event)

    def _run_preprune(self):
        if self._busy():
            return
        if self._project is None:
            QMessageBox.warning(self, tr("无法执行"), tr("请先选择项目"))
            return
        dataset_dir = Path(self._project.dataset_dir)
        if not dataset_dir.is_dir():
            # 路径预检：数据集被移动/删除时给出明确指引（统计在工作线程做，
            # 避免大集合扫描或异常打断界面——曾导致未捕获异常崩溃）
            QMessageBox.warning(
                self, tr("无法执行"),
                f"{tr('数据集路径不存在：')}\n{dataset_dir}\n"
                f"{tr('请在训练平台的项目信息中检查。')}")
            return
        model_name = self.pre_model_combo.currentData()
        if not model_name:
            QMessageBox.warning(self, tr("无法执行"), tr("请选择模型架构"))
            return
        out_dir = Path(self._project.model_dir) / PRUNED_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        resize = (self._project.augment_ops or {}).get("resize") or {}
        ratio = self.pre_ratio_spin.value() / 100.0
        stem = f"{model_name}_pruned{self.pre_ratio_spin.value()}"
        dst = out_dir / f"{stem}.pt"
        if dst.exists() and QMessageBox.question(
                self, tr("覆盖确认"),
                f"{dst.name} {tr('已存在，重新生成将覆盖它。继续吗？')}") != QMessageBox.Yes:
            return
        self._set_running(True)
        self.prune_result.setText(
            tr("生成中…（{model}，扫描数据集类别后预剪枝，视模型规模约需数秒到数分钟）"
               ).replace("{model}", model_name))
        log.info("训练前剪枝任务: 项目=%s 模型=%s 比例=%s 输出=%s",
                 self._project.name, model_name, ratio, dst)
        args = ["--mode", "pre",
                "--dataset-dir", str(self._project.dataset_dir),
                "--task", getattr(self._project.task, "value", self._project.task),
                "--model-name", model_name,
                "--width", str(int(resize.get("width", 0) or 0)),
                "--height", str(int(resize.get("height", 0) or 0)),
                "--ratio", repr(ratio), "--dst", str(dst)]
        self._start_prune(
            args, f"生成中…（{model_name}，独立子进程执行，视模型规模数秒到数分钟；"
                  "可随时紧急终止）")

    # ---- 剪枝子进程（主窗口零剪枝内存，超水位/超时自动终止）----
    def _start_prune(self, args: list, busy_text: str):
        self._set_running(True)
        self.prune_result.setText(busy_text)
        log.info("启动剪枝子进程: %s", args)
        mgr = _PruneProc(self)
        mgr.phase.connect(self._on_prune_phase)
        mgr.finished.connect(self._on_prune_finished)
        self._prune_proc = mgr
        try:
            mgr.start(args)
        except RuntimeError as e:
            self._prune_proc = None
            self._set_running(False)
            QMessageBox.critical(self, "启动失败", str(e))

    def _kill_prune(self):
        if self._prune_proc is not None and self._prune_proc.running():
            self._prune_proc.kill("手动紧急终止")

    def _on_prune_phase(self, name: str, detail: str):
        text = {"scan": "扫描数据集", "load": "加载权重", "trace": "追踪依赖图（重）",
                "apply": "执行剪枝", "cleanup": "回收内存", "save": "保存结果"
                }.get(name, name)
        suffix = f"（{detail}）" if detail else ""
        self.prune_result.setText(f"{tr('剪枝中…')} {text}{suffix}")

    def _on_prune_finished(self, outcome: dict):
        from ..ui_widgets import toast

        self._prune_proc = None
        self._set_running(False)
        status = outcome.get("status")
        if status == "applied":
            result = outcome.get("result", {})
            if result.get("dst"):
                self.prune_result.setText(
                    f"✅ 剪枝模型已生成：{Path(result['dst']).name}\n"
                    f"{result.get('detail', '')}\n"
                    "在训练平台的模型下拉中选择它即可开始迁移学习。")
            else:
                self.prune_result.setText(
                    f"✅ 剪枝完成：{result.get('detail', '')}\n"
                    f"参数量 {result.get('params_before', 0):,} → "
                    f"{result.get('params_after', 0):,}；输出：best_pruned.pt"
                    f"（子进程峰值 RSS {outcome.get('peak_rss_mb', 0):.0f} MB）")
        elif status == "killed":
            reason = outcome.get("reason", "")
            toast(self, f"剪枝已终止：{reason}", "warn")
            self.prune_result.setText(
                f"🛑 剪枝已终止：{reason}\n"
                f"（子进程峰值 RSS {outcome.get('peak_rss_mb', 0):.0f} MB；"
                "源权重不受影响，半截输出不会落盘）")
        elif status == "error":
            toast(self, "剪枝失败，详见结果区", "error")
            self.prune_result.setText(
                f"❌ 剪枝失败：{outcome.get('message', '')}")
        else:
            self.prune_result.setText(f"⚠️ {tr('未执行剪枝：')}{status}")
        self._refresh_after_task()

    def _run_prune(self):
        if self._busy():
            return
        run_dir = self._current_run_dir()
        if run_dir is None:
            QMessageBox.warning(self, tr("无法执行"),
                                tr("请先选择项目与训练记录（记录文件夹内需有权重文件）"))
            return
        src = next((run_dir / n for n in _ORIGINAL_WEIGHTS if (run_dir / n).is_file()),
                   None)
        if src is None:
            QMessageBox.warning(self, tr("无法执行"),
                                f"{tr('未找到权重文件：')}{run_dir / 'best.pt'}")
            return
        dst = run_dir / "best_pruned.pt"
        if dst.exists() and QMessageBox.question(
                self, tr("覆盖确认"),
                f"{dst.name} {tr('已存在，重新剪枝将覆盖它。继续吗？')}") != QMessageBox.Yes:
            return
        ratio = self.ratio_spin.value() / 100.0
        args = ["--mode", "post", "--src", str(src), "--dst", str(dst),
                "--ratio", repr(ratio)]
        self._start_prune(
            args, tr("剪枝中…（ratio={r}，独立子进程执行；追踪依赖图为重负载阶段，"
                     "可随时紧急终止）").replace("{r}", f"{ratio:g}"))


def _suggest_imgsz(src: Path) -> int | tuple[int, int] | None:
    """从源权重读取训练输入尺寸（模型加速默认输入随模型走）。

    轻量读取（不构建网络）：YOLO .pt 读 train_args.imgsz /
    model.yaml.imgsz；xccv 自有格式读 train_input / min_size /
    image_size。矩形训练尺寸返回 (高, 宽)；失败返回 None（用户手输）。
    """
    try:
        import torch as _t

        ckpt = _t.load(str(src), map_location="cpu")
        if not isinstance(ckpt, dict):
            return None
        ta = ckpt.get("train_args")
        if isinstance(ta, dict):
            v = ta.get("imgsz")
            if isinstance(v, (int, float)) and int(v) >= 32:
                return int(v)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                h, w = int(v[0]), int(v[1])
                if h >= 32 and w >= 32:
                    return h if h == w else (h, w)
        yml = getattr(ckpt.get("model"), "yaml", None)
        if isinstance(yml, dict):
            v = yml.get("imgsz")
            if isinstance(v, (int, float)) and int(v) >= 32:
                return int(v)
        v = ckpt.get("train_input")
        if isinstance(v, (int, float)) and int(v) >= 32:
            return int(v)
        for key in ("min_size", "image_size", "imgsz"):
            v = ckpt.get(key)
            if isinstance(v, (int, float)) and int(v) >= 32:
                return int(v)
            if isinstance(v, (list, tuple)) and v:
                vals = [int(x) for x in v]
                if len(vals) == 2 and vals[0] != vals[1]:
                    return (vals[0], vals[1])
                return int(min(vals))
    except Exception:  # noqa: BLE001  识别失败不阻塞：留给用户手输
        return None
    return None


def _parse_imgsz(text: str) -> int | tuple[int, int] | None:
    """解析输入尺寸文本：'640' 或 '高x宽'（如 586x1920，矩形引擎）。

    支持 ×/X/* 分隔；每侧范围 32~2048；两侧相等折叠为 int；非法返回 None。
    """
    t = (text or "").strip().lower().replace("×", "x").replace("*", "x")
    if not t:
        return None
    parts = [p for p in (s.strip() for s in t.split("x")) if p]
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        return None
    if not vals or any(v < 32 or v > 2048 for v in vals):
        return None
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return vals[0] if vals[0] == vals[1] else (vals[0], vals[1])
    return None


def _imgsz_text(v) -> str:
    """尺寸值 → 输入框文本（int 或 (高, 宽)）。"""
    if isinstance(v, tuple):
        return f"{v[0]}x{v[1]}"
    return str(v)


class AccelerateToolWindow(_ToolWindow):
    tool_icon = "accelerate.png"
    """模型加速窗口：把训练 / 剪枝产物导出为 TensorRT 引擎（.engine）。

    构建需要 NVIDIA GPU + TensorRT 运行库；缺失时给出明确指引。
    引擎是绑定 GPU 与 TensorRT 版本的编译产物（明文 .engine）。
    """

    tool_key = "accel"
    tool_title = "模型加速（TensorRT）"
    file_mode = True  # 不依赖训练平台项目：文件管理器直接选择权重

    def _build_settings(self, lay: QVBoxLayout):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        inner = QVBoxLayout(body)
        inner.setContentsMargins(0, 4, 0, 0)
        inner.setSpacing(12)

        g2 = QGroupBox("TensorRT 引擎导出")
        f2 = QFormLayout(g2)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择要加速的权重文件（.pt）")
        btn_pick = QPushButton("浏览…")
        btn_pick.clicked.connect(self._pick_accel_source)
        accel_src_row = QHBoxLayout()
        accel_src_row.addWidget(self.source_edit, stretch=1)
        accel_src_row.addWidget(btn_pick)
        self.imgsz_edit = QLineEdit("640")
        self.imgsz_edit.setPlaceholderText("640 或 高x宽（如 586x1920）")
        self.imgsz_edit.setToolTip(
            "引擎输入尺寸（YOLO 用；分类/R-CNN 按训练尺寸自动确定）；\n"
            "支持矩形（高x宽），与训练尺寸一致时矩形模型效果最佳")
        self.precision_combo = QComboBox()
        self.precision_combo.addItem("FP16 半精度（推荐：体积小、速度快）", True)
        self.precision_combo.addItem("FP32 全精度（精度优先：体积大）", False)
        self.precision_combo.setToolTip(
            "FP16：引擎体积约减半、推理更快，检出分数可能有小数级漂移；\n"
            "FP32：与 torch 权重推理数值最接近，体积较大")
        self.btn_export = QPushButton("▶ 导出 TensorRT 引擎")
        self.btn_export.setObjectName("accent")
        self.btn_export.clicked.connect(self._run_export)
        f2.addRow("源权重", accel_src_row)
        f2.addRow("输入尺寸", self.imgsz_edit)
        f2.addRow("精度", self.precision_combo)
        f2.addRow(self.btn_export)
        inner.addWidget(g2)

        self.accel_bar = QProgressBar()
        self.accel_bar.setRange(0, 100)
        self.accel_bar.setValue(0)
        self.accel_result = QLabel("")
        bind_style(self.accel_result, lambda: f"color: {token('primary_deep')}; font-weight: bold;")
        self.accel_result.setWordWrap(True)
        inner.addWidget(self.accel_result)
        self.accel_bar.setObjectName("accelBar")
        inner.addWidget(self.accel_bar)

        accel_hint = QLabel(
            "引擎输出为源权重同目录的同名 .engine 文件；\n"
            "构建需要 NVIDIA GPU 与 TensorRT 运行库（建议在目标部署机执行）；\n"
            "引擎与构建时的 GPU 型号、TensorRT 版本绑定，只能在该环境加载；\n"
            "测试平台可直接加载 .engine 进行预测（需同款 GPU + TensorRT 环境）。")
        accel_hint.setObjectName("hint")
        accel_hint.setWordWrap(True)
        inner.addWidget(accel_hint)

        inner.addStretch(1)
        scroll.setWidget(body)
        lay.addWidget(scroll, stretch=1)

    def _pick_accel_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择要加速的权重文件", "",
            "模型权重 (*.pt *.engine *.onnx);;所有文件 (*)")
        if not path:
            return
        self.source_edit.setText(path)
        # 默认输入尺寸随源模型走（读训练输入尺寸预填，可手改）
        self.accel_result.setText("正在读取源权重的训练输入尺寸…")
        self._start_worker(
            lambda: {"imgsz": _suggest_imgsz(Path(path))},
            self._on_imgsz_suggested)

    def _on_imgsz_suggested(self, result: dict):
        self._worker = None
        v = (result or {}).get("imgsz")
        if isinstance(v, tuple):
            self.imgsz_edit.setText(_imgsz_text(v))
            self.accel_result.setText(
                f"已按源权重自动填入输入尺寸 {v[0]}x{v[1]}（矩形，可手动修改）")
            return
        v = int(v or 0)
        if v >= 32:
            self.imgsz_edit.setText(str(v))
            self.accel_result.setText(
                f"已按源权重自动填入输入尺寸 {v}（可手动修改）")
        else:
            self.accel_result.setText("未能从源权重识别输入尺寸，可手动设置")

    def _on_export_progress(self, frac: float, text: str):
        if frac < 0:
            self.accel_bar.setRange(0, 0)   # 构建期无真实百分比：忙碌指示
        else:
            self.accel_bar.setRange(0, 100)
            self.accel_bar.setValue(int(max(0.0, min(frac, 1.0)) * 100))
        self.accel_result.setText(text)

    def _set_running(self, running: bool):
        self.btn_export.setEnabled(not running)

    def _run_export(self):
        if self._busy():
            return
        stext = self.source_edit.text().strip()
        src = Path(stext) if stext else None
        if src is None or not src.is_file():
            QMessageBox.warning(self, tr("无法执行"), tr("请先通过「浏览…」选择权重文件"))
            return
        dst = src.with_suffix(".engine")
        if dst.exists() and QMessageBox.question(
                self, tr("覆盖确认"),
                f"{dst.name} {tr('已存在，重新导出将覆盖它。继续吗？')}") != QMessageBox.Yes:
            return
        imgsz = _parse_imgsz(self.imgsz_edit.text())
        if imgsz is None:
            QMessageBox.warning(
                self, tr("无法执行"),
                tr("输入尺寸格式：640 或 高x宽（如 586x1920），范围 32~2048"))
            return
        half = self.precision_combo.currentData()
        self._set_running(True)
        self.accel_bar.setRange(0, 100)
        self.accel_bar.setValue(2)
        self.accel_result.setText(tr("导出中…（构建引擎约需 1~10 分钟，取决于模型规模）"))
        log.info("TensorRT 导出任务: src=%s dst=%s imgsz=%s half=%s",
                 src, dst, imgsz, half)
        self._start_worker(
            lambda: _export_engine(
                src, dst, imgsz=imgsz, half=half,
                progress=lambda f, t: self._worker.progress.emit(f, t)),
            self._on_export_done, on_progress=self._on_export_progress)

    def _on_export_done(self, result: dict):
        self._worker = None
        self._set_running(False)
        self.accel_bar.setRange(0, 100)
        self.accel_bar.setValue(100)
        self.accel_result.setText(
            f"✅ {tr('引擎导出完成')}：{Path(result.get('dst', '')).name}"
            f"（{tr('耗时')} {result.get('seconds', 0)}s）\n"
            + tr("引擎与构建环境（GPU/TensorRT 版本）绑定，请在同款部署环境使用；")
            + tr("测试平台可直接加载 .engine 预测。"))

    def _on_task_failed(self, message: str):
        super()._on_task_failed(message)
        self.accel_bar.setRange(0, 100)
        self.accel_bar.setValue(0)
        self.accel_result.setText("⚠️ 导出失败（详见平台日志 logs/tools.log）")


class ModelToolsWindow(QMainWindow):
    """工具卡片墙：点击卡片打开独立工具窗口（可多开，重复打开只激活已有窗口）。"""

    _TOOL_FACTORIES = {
        "prune": PruneToolWindow,
        "accel": AccelerateToolWindow,
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("XCcv 模型工具")
        self.setWindowIcon(QIcon(str(icon_path("tools.png"))))
        self.resize(1180, 540)
        apply_app_theme(self)
        self._tool_windows: dict[str, _ToolWindow] = {}
        self._build_ui()
        from ..status_bus import attach_platform_status

        attach_platform_status(self, "模型工具", "modeltools")

    # ================= 界面骨架 =================
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        head = QHBoxLayout(header)
        head.setContentsMargins(22, 10, 22, 10)
        app_title = QLabel("XCcv 模型工具")
        app_title.setObjectName("appTitle")
        head.addWidget(app_title)
        head.addStretch(1)
        root.addWidget(header)

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(22, 16, 22, 14)
        body_lay.setSpacing(10)

        # 工具卡片墙
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        cards_host = QWidget()
        cards_grid = QGridLayout(cards_host)
        cards_grid.setContentsMargins(0, 0, 0, 0)
        cards_grid.setSpacing(14)
        cards_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        tools = [
            ("prune", "模型剪枝", "结构化通道剪枝：压缩参数量、提升推理速度；"
             "剪后副本可在测试平台与原模型直接对比", "#8b5cf6", True,
             "prune.png"),
            ("accel", "模型加速（TensorRT）", "导出 TensorRT 引擎，产线推理提速数倍；"
             "支持分类 / YOLO / Mask R-CNN（特征段加速），单文件 .engine",
             "#8b5cf6", True, "accelerate.png"),
            ("defectgen", "缺陷生成", "DefectAugmentStudio：在背景板上合成"
             "缺陷样本（抠图/融合/批量），扩充训练集", "#0f9488", True,
             "defectgen.png"),
        ]  # 流程图编辑器已独立为主界面「流程图」平台页（xccv/flow_platform.py）
        for i, (key, title, desc, color, enabled, icon) in enumerate(tools):
            card = _ToolCard(key, title, desc, color, enabled, icon=icon)
            card.clicked.connect(self._open_tool)
            row, col = divmod(i, 3)  # 每行固定 3 个，超出换行 → 垂直滚动
            cards_grid.addWidget(card, row, col)
        scroll.setWidget(cards_host)
        body_lay.addWidget(scroll, stretch=1)

        root.addWidget(body, stretch=1)
        self.setCentralWidget(central)

    # ================= 工具窗口 =================
    def _open_tool(self, key: str):
        if key == "defectgen":
            # 缺陷生成=独立平台进程（单实例/AUMID/加载卡走 platform_entry）
            from ..launcher import launch_platform

            launch_platform("defectgen")
            return
        factory = self._TOOL_FACTORIES.get(key)
        if factory is None:
            return  # 即将上线的工具
        window = self._tool_windows.get(key)
        if window is not None and window.isVisible():
            window.activateWindow()
            window.raise_()
            return
        window = factory(on_closed=self._on_tool_closed)
        window.show()
        self._tool_windows[key] = window

    def _on_tool_closed(self, key: str):
        self._tool_windows.pop(key, None)

    def open_tool_windows(self) -> dict:
        """已打开的工具窗口（测试用）。"""
        return dict(self._tool_windows)


def main():
    from ..platform_entry import run_platform

    run_platform("tools", "模型工具", lambda: ModelToolsWindow(), heavy=True)


if __name__ == "__main__":
    main()


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

