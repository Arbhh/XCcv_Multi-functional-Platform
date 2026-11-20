"""回归。

P-B5-1 停服宽限期（优雅退出期间 health 仍 200 不误报）+ 灵动岛走面板同链路；
P-B5-2 build_result_json 分类样预测按分类口径导出（flow 任务）；
P-B5-3 _DEMO_ICON 从包内取（安装态 app_root=xccv 外层不再拼接）；
P-B5-4 训练打开项目模态等待弹窗。
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_APP = None


def _app():
    global _APP
    if _APP is None:
        from PyQt5 import QtWidgets

        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


class StopGraceTest(unittest.TestCase):
    """P-B5-1：停止宽限期内 health 200 不判失败，超期才判。"""

    def _panel(self):
        from xccv.deploy_platform import ServicePanel

        p = ServicePanel.__new__(ServicePanel)   # 跳过 UI 构造
        p._service_ok = True
        p._service_starting = False
        p._service_stopping = True
        p._stop_grace = 2
        p._watchdog_left = 0
        p.tips = []
        p.tip_label = type("T", (), {
            "setStyleSheet": lambda self, s: None,
            "setText": lambda self, s: p.tips.append(s),
        })()
        p.gpu_bar = type("G", (), {"setValue": lambda self, v: None,
                                   "setFormat": lambda self, s: None})()
        p.gpu_text = type("GT", (), {"setText": lambda self, s: None})()
        p.api_list = type("L", (), {"clear": lambda self: None})()
        # 实例属性函数不做方法绑定：第一个形参即第一个实参（info）
        p._update_gpu = lambda info: None
        p._refresh_status = lambda: None
        return p

    def test_grace_period_tolerates_running_health(self):
        import xccv.ui_widgets as uw

        p = self._panel()
        p._stop_grace = 1
        with mock.patch.object(uw, "toast"):
            p._apply_poll(True, False, {})
            self.assertTrue(p._service_stopping)      # 宽限期内仍"停止中"
            self.assertFalse(any("停止失败" in t for t in p.tips))
            p._apply_poll(True, False, {})
        self.assertFalse(p._service_stopping)         # 宽限期耗尽 → 判失败
        self.assertTrue(any("停止失败" in t for t in p.tips))

    def test_stopped_within_grace_is_success(self):
        import xccv.ui_widgets as uw

        p = self._panel()
        p._stop_grace = 12
        with mock.patch.object(uw, "toast"):
            p._apply_poll(False, True, {})
        self.assertFalse(p._service_stopping)
        self.assertTrue(any("已停止" in t for t in p.tips))


class IslandStopRoutesThroughPanelTest(unittest.TestCase):
    """P-B5-1：灵动岛停止走面板 _stop_service（同链路反馈）。"""

    def test_island_prefers_panel_hook(self):
        import xccv.island as island_mod

        src = Path(island_mod.__file__).read_text(encoding="utf-8")
        self.assertIn("self._panel = parent", src)
        self.assertIn('hook = getattr(panel, "_stop_service", None)', src)

    def test_panel_assigned_as_reference(self):
        import xccv.deploy_platform as dp

        src = Path(dp.__file__).read_text(encoding="utf-8")
        self.assertIn("ServiceIsland(self._poller, self)", src)


class ClassificationFlowJsonTest(unittest.TestCase):
    """P-B5-2：flow 任务下分类样预测（probs 有值无框）导出 classifications。"""

    def test_classification_like_prediction_in_flow_task(self):
        from xccv.engine.models.base import Prediction
        from xccv.engine.result_json import build_result_json

        pred = Prediction(labels=[0], scores=[0.95],
                          probs=[0.95, 0.03, 0.02])
        payload = build_result_json(
            task="flow", source={"file": "a.png", "path": ""},
            prediction=pred, classes=["橙", "黄", "红"],
            flow_info={"steps": [], "removed": 0})
        self.assertEqual(len(payload["classifications"]), 3)
        self.assertEqual(payload["classifications"][0]["label"], "橙")
        self.assertEqual(payload["detections"], [])

    def test_detection_prediction_unchanged(self):
        from xccv.engine.models.base import Prediction
        from xccv.engine.result_json import build_result_json

        pred = Prediction(labels=[0], scores=[0.9],
                          boxes=[[10.0, 10.0, 50.0, 50.0]])
        payload = build_result_json(
            task="flow", source={"file": "a.png", "path": ""},
            prediction=pred, classes=["a"])
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["classifications"], [])


class DemoIconPathTest(unittest.TestCase):
    """P-B5-3：_DEMO_ICON 必须取自包内 assets（安装态 app_root 外层无 xccv）。"""

    def test_icon_path_is_package_relative(self):
        from xccv.deploy_platform import _DEMO_ICON, _ASSETS_DIR

        self.assertEqual(_DEMO_ICON.parent, _ASSETS_DIR)
        self.assertTrue(_DEMO_ICON.is_file(),
                        f"icon missing on disk: {_DEMO_ICON}")

    def test_no_app_root_join_in_demo_icon(self):
        import xccv.deploy_platform as dp

        src = Path(dp.__file__).read_text(encoding="utf-8")
        self.assertNotIn('_DEMO_ICON = app_root()', src)


class TrainingWaitDialogTest(unittest.TestCase):
    """P-B5-4 → V0.7 U1：打开项目改为顶部悬浮胶囊提示（对齐标注平台横幅）。

    弹窗样式要求：两行模态弹窗不符预期——单行提示、去"首次加载"字样、
    半透明、颜色与标注平台微差（training_window._open_project）。
    """

    def test_wait_dialog_in_open_project(self):
        import xccv.training.training_window as tw

        src = Path(tw.__file__).read_text(encoding="utf-8")
        # V0.7.2 i18n：横幅文案走 tr()——锁 tr 键与插值结构
        self.assertIn("{tr('正在打开项目')}「{name}」…", src)
        self.assertIn("openProjectBanner", src)
        self.assertIn("rgba(", src)                      # 半透明背景
        self.assertIn("border-radius: 12px", src)        # 圆角胶囊
        self.assertIn("QApplication.processEvents()", src)
        self.assertNotIn("首次打开需加载深度学习组件", src)  # 字样去除
        self.assertNotIn("QDialog(self)", src)           # 不再是模态弹窗


if __name__ == "__main__":
    unittest.main()
