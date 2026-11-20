"""V0.6 实测反馈修复：推理 3s→8s（Windows EcoQoS 节流）+ 单图推理留痕。

根因实测（Intel Core 7 250H 混合架构笔记本）：同一 maskrcnn 单图推理
HighQoS 3.4s / 强制 EcoQoS 8.8s。Windows 11 对失去前台（含主线程阻塞被判
"未响应"）的进程自动施加 EcoQoS。修复：各进程入口显式声明 HighQoS
（xccv/win_qos.py），并为每次单图推理写 INFO 留痕（耗时/检出数/阈值）。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


class WinQosTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "仅 Windows")
    def test_opt_out_sets_high_qos(self):
        from xccv.win_qos import opt_out_power_throttling, power_throttling_state

        self.assertTrue(opt_out_power_throttling())
        state = power_throttling_state()
        if state is None:
            self.skipTest("系统不支持查询进程节流状态")
        control, st = state
        self.assertTrue(control & 0x1, "EXECUTION_SPEED 位应被显式接管")
        self.assertEqual(st & 0x1, 0, "StateMask=0 即 HighQoS（不节流）")

    def test_opt_out_idempotent_and_safe(self):
        from xccv.win_qos import opt_out_power_throttling

        first = opt_out_power_throttling()
        self.assertEqual(opt_out_power_throttling(), first)

    def test_run_platform_opts_out(self):
        """所有平台进程入口（run_platform）都声明 HighQoS。"""
        from PyQt5.QtWidgets import QApplication, QWidget

        from xccv import platform_entry

        _ = QApplication.instance() or QApplication([])
        with mock.patch("xccv.win_qos.opt_out_power_throttling",
                        return_value=True) as m, \
                mock.patch.object(sys, "argv", ["x", "--selftest"]), \
                mock.patch("xccv.taskbar.set_process_app_user_model_id"):
            platform_entry.run_platform("unit_qos", "单测", lambda: QWidget(),
                                        enable_logging=False)
        m.assert_called_once()

    def test_main_and_service_entries_opt_out(self):
        """主界面与推理服务入口同样声明（源码锁：两处均为进程入口）。"""
        app_src = (_ROOT / "xccv" / "app.py").read_text(encoding="utf-8")
        svc_src = (_ROOT / "xccv" / "service" / "main.py").read_text(encoding="utf-8")
        self.assertIn("opt_out_power_throttling()", app_src)
        self.assertIn("opt_out_power_throttling()", svc_src)


class InferLogTest(unittest.TestCase):
    """单图推理留痕：耗时/检出数/阈值写入平台日志（推理变慢排障依据）。"""

    def test_infer_current_logs_cost(self):
        from PyQt5.QtWidgets import QApplication

        from xccv.engine.io_utils import imwrite_unicode
        from xccv.engine.models.base import Prediction
        from xccv.testing import testing_window
        from xccv.testing.testing_window import TestingWindow

        _ = QApplication.instance() or QApplication([])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        img_path = Path(tmp.name) / "单图.png"
        imwrite_unicode(str(img_path), np.zeros((30, 40, 3), np.uint8))

        class _FakeModel:
            classes = ["a"]
            task = "detection"

            def predict(self, image, conf=0.25, iou=0.45):
                # boxes 用 ndarray：日志计数不得用真值判空（P150 同类）
                return Prediction(labels=[0, 0], scores=[0.9, 0.8],
                                  boxes=np.array([[1, 1, 5, 5], [6, 6, 9, 9]],
                                                 dtype=float))

        win = TestingWindow()
        try:
            win.model = _FakeModel()
            win.classes = ["a"]
            win.task = "detection"
            with self.assertLogs(testing_window.log.name, level="INFO") as cm:
                win._set_image_list([img_path], root=None)
            joined = "\n".join(cm.output)
            self.assertIn("单图推理", joined)
            self.assertIn("40x30", joined)
            self.assertIn("检出 2", joined)
        finally:
            win.close()
            win.deleteLater()


if __name__ == "__main__":
    unittest.main()
