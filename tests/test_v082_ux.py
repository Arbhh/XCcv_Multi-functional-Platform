# -*- coding: utf-8 -*-
"""V0.8.2 T5~T8 契约测试：新建项目/工具开窗等待横幅、?帮助按钮摘除、
对比窗口画布尺寸同步（高分屏裁剪回归锁）。"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

_APP = None


def _app():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


class BusyBannerTest(unittest.TestCase):
    """T5：新建项目/打开项目共用悬浮横幅（U1 模式）。"""

    @classmethod
    def setUpClass(cls):
        _app()
        from xccv.training.training_window import TrainingWindow

        cls.tw = TrainingWindow()
        cls.tw.show()                    # 横幅可见性依赖父链
        _APP.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.tw.close()

    def test_banner_shown_and_hidden(self):
        banner, t0 = self.tw._show_busy_banner("测试等待提示…")
        self.assertTrue(banner.isVisible())
        self.assertEqual(banner.objectName(), "openProjectBanner")
        self.tw._finish_busy_banner(banner, t0, min_ms=0)
        _APP.processEvents()
        self.assertFalse(banner.isVisible(), "横幅应在收尾后隐藏")

    def test_create_project_shows_banner_during_dialog_build(self):
        """T5 核心：新建项目对话框构建期间横幅已上屏（卡顿有反馈）。"""
        from PyQt5.QtWidgets import QDialog, QLabel

        from xccv.training import training_window as tw_mod

        seen = {}

        class _StubDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                seen["banner"] = any(
                    w.objectName() == "openProjectBanner" and w.isVisible()
                    for w in parent.findChildren(QLabel))

            def exec_(self):
                return QDialog.Rejected

        with mock.patch.object(tw_mod, "CreateProjectDialog", _StubDialog):
            self.tw._create_project()
        self.assertTrue(seen.get("banner"),
                        "对话框构建期间必须有可见的等待横幅")


class DevOptionsHelpFlagTest(unittest.TestCase):
    """T6：开发者选项对话框摘除标题栏「?」上下文帮助按钮。"""

    def test_context_help_button_removed(self):
        _app()
        from xccv.training.dialogs import DevOptionsDialog

        dlg = DevOptionsDialog()
        self.assertFalse(
            bool(dlg.windowFlags() & QtCore.Qt.WindowContextHelpButtonHint),
            "无 WhatsThis 内容的对话框不得显示「?」按钮（问号光标卡死）")


class AdaptiveFigureTest(unittest.TestCase):
    """T7：对比窗口画布尺寸必须交给 matplotlib 同步（物理像素口径），
    不得再用逻辑像素二次覆写（高分屏裁剪回归锁）。"""

    def test_figure_tracks_widget_after_resize(self):
        _app()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from xccv.training.compare_window import _AdaptiveFigure

        fig = Figure(dpi=100)
        canvas = _AdaptiveFigure(fig, min_height=340)
        canvas.show()                    # resize 事件需要可见态才派发
        _APP.processEvents()
        canvas.resize(800, 500)
        _APP.processEvents()
        dpr = canvas.devicePixelRatioF() or 1.0
        exp_w = canvas.width() * dpr / fig.dpi
        exp_h = canvas.height() * dpr / fig.dpi
        self.assertAlmostEqual(fig.get_figwidth(), exp_w, places=2,
                               msg="figure 宽必须等于控件物理像素/dpi")
        self.assertAlmostEqual(fig.get_figheight(), exp_h, places=2)
        canvas.close()

    def test_confusion_window_canvas_sized_to_widget(self):
        """端到端：对比窗口构建后画布英寸与控件一致（无裁剪态）。"""
        from pathlib import Path
        from types import SimpleNamespace

        _app()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        from xccv.training.compare_window import CompareWindow

        classes = ["凹点", "流纹", "缩水", "气孔"]
        matrix = [[30, 1, 0, 0], [2, 25, 1, 0], [0, 1, 28, 0],
                  [0, 0, 2, 26]]
        records = [SimpleNamespace(
            metrics={"confusion_matrix": matrix, "classes": classes,
                     "final": {"map50": 0.9},
                     "config": {"model_name": "resnet18", "epochs": 10}},
            run_dir=Path("runs") / name, start_time="2026-11-19 10:00",
            model_name="resnet18", epochs_done=10, status="finished",
            duration_sec=3600, status_text="已完成")
            for name in ("run-A", "run-B")]
        project = SimpleNamespace(dataset_dir="")
        win = CompareWindow(project, records)
        win.resize(1200, 860)
        win.show()
        _APP.processEvents()
        canvases = win.findChildren(FigureCanvasQTAgg)
        self.assertGreaterEqual(len(canvases), 2)
        for c in canvases:
            dpr = c.devicePixelRatioF() or 1.0
            self.assertAlmostEqual(
                c.figure.get_figwidth() * c.figure.dpi,
                c.width() * dpr, delta=2,
                msg="画布 figure 宽必须与控件物理像素一致（防高分屏裁剪）")
            self.assertGreaterEqual(c.height(), c.minimumHeight())
        win.close()


class ToolOpenBannerTest(unittest.TestCase):
    """T8：打开剪枝/加速工具窗口期间显示加载横幅。"""

    def test_banner_during_tool_window_build(self):
        from PyQt5.QtWidgets import QDialog, QLabel, QWidget

        from xccv.modeltools import modeltools_window as mtw

        _app()
        win = mtw.ModelToolsWindow()
        win.show()                       # 横幅可见性依赖父链
        _APP.processEvents()
        seen = {}

        class _StubWindow(QWidget):
            def __init__(self, on_closed=None, **kw):
                super().__init__()
                parent = win
                seen["banner"] = any(
                    w.objectName() == "openProjectBanner" and w.isVisible()
                    for w in parent.findChildren(QLabel))

            def show(self):
                pass

        original = dict(mtw.ModelToolsWindow._TOOL_FACTORIES)
        mtw.ModelToolsWindow._TOOL_FACTORIES = dict(
            original, prune=_StubWindow)
        try:
            win._open_tool("prune")
        finally:
            mtw.ModelToolsWindow._TOOL_FACTORIES = original
        self.assertTrue(seen.get("banner"),
                        "工具窗口构建期间必须有可见的加载横幅")
        self.assertIn("prune", win.open_tool_windows())


if __name__ == "__main__":
    unittest.main()
