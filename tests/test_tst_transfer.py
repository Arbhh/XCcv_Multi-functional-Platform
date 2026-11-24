"""测试平台测试：1 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v073_round17
"""


# == 来源模块：test_v073_round17 ==
"""回归锁：遗留待办 §三.1/§三.2。

- §三.1 语言重建覆盖面补齐：TrainingWindow.i18n_transfer_children
  ——宿主列表窗语言重建时项目窗跟随（空闲窗原地重建换语言、忙碌窗原样
  过户、隐藏窗不转移）；过户/重建一律重绑回调到新宿主（旧窗即将销毁）；
- §三.2 已加载模型纳入忙时判定：TestingWindow.i18n_rebuild_blocked
  含已加载模型；i18n_block_reason 区分"已加载模型"与"任务运行中"。
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_labelme_dataset
from xccv.training import project as project_mod
from xccv.training.project import create_project

_APP = QApplication.instance() or QApplication(sys.argv)


def _wait_stats(pw, timeout_s: float = 30.0):
    """项目窗打开后统计已后台化——泵事件直到统计线程收尾。"""
    from PyQt5.QtCore import QEventLoop, QTimer

    deadline = time.monotonic() + timeout_s
    while getattr(pw, "_stats_thread", None) is not None \
            and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec_()
    QApplication.processEvents()


class ProjectWindowTransferTest(unittest.TestCase):
    """§三.1：宿主列表窗语言重建 → 项目窗跟随（平台重建路径调用）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        det, _ = make_labelme_dataset(Path(self._tmp.name) / "det", n_images=4)
        create_project("ui", str(det), "", "detection")

        from xccv.training.training_window import TrainingWindow

        self._hosts = []
        self.win = self._make_host()
        QApplication.processEvents()

    def _make_host(self):
        from xccv.training.training_window import TrainingWindow

        host = TrainingWindow()
        self._hosts.append(host)
        self.addCleanup(host.close)
        return host

    def _open(self, name="ui"):
        self.win._open_project(name)
        QApplication.processEvents()
        pw = self.win.open_project_windows()[name]
        _wait_stats(pw)  # 统计线程存活会让 busy 判定为真/拒绝关窗
        return pw

    def tearDown(self):
        for host in self._hosts:          # 项目窗逐一关闭，避免窗口/线程泄漏
            for pw in list(host.open_project_windows().values()):
                pw.close()
        QApplication.processEvents()

    def test_idle_window_rebuilt_in_new_language(self):
        """空闲项目窗：原地重建换语言并转移到新宿主，旧窗关闭。"""
        pw = self._open()
        from xccv.i18n import set_lang

        new_host = self._make_host()
        try:
            set_lang("en")  # 模拟 watcher 已把进程缓存切到新语言（重建即 EN）
            self.win.i18n_transfer_children(new_host)
            QApplication.processEvents()
            moved = new_host.open_project_windows().get("ui")
            self.assertIsNotNone(moved, "空闲项目窗未转移到新宿主")
            self.assertIsNot(moved, pw, "空闲项目窗应重建而非原样过户")
            self.assertTrue(moved.isVisible())
            self.assertEqual(moved.btn_page_train.text(), "Training Visuals",
                             "重建后的项目窗未使用新语言渲染")
            self.assertFalse(pw.isVisible(), "旧项目窗未关闭")
            self.assertEqual(self.win.open_project_windows(), {},
                             "旧宿主登记未清空")
            # qt_slots 包装令 bound method 每次访问都是新对象：断言宿主本体
            self.assertIs(moved.on_closed.__self__, new_host)
        finally:
            set_lang("zh")  #3371复位偏好与缓存

    def test_busy_window_transferred_as_is(self):
        """忙碌项目窗（训练中）：原样过户到新宿主，不得关闭，回调重绑。"""
        pw = self._open()
        pw.i18n_rebuild_blocked = lambda: True  # 模拟训练/预检进行中
        new_host = self._make_host()
        self.win.i18n_transfer_children(new_host)
        QApplication.processEvents()
        moved = new_host.open_project_windows().get("ui")
        self.assertIs(moved, pw, "忙碌项目窗应原样过户而非重建")
        self.assertTrue(pw.isVisible(), "忙碌项目窗不得被关闭")
        self.assertIs(pw.on_closed.__self__, new_host,
                      "过户未重绑关闭回调（旧宿主销毁后撞已删 C++ 对象）")
        self.assertIs(pw.on_renamed.__self__, new_host)
        self.assertEqual(self.win.open_project_windows(), {},
                         "旧宿主登记未清空")

    def test_hidden_window_not_transferred(self):
        """已关闭/隐藏的项目窗：不转移。"""
        pw = self._open()
        pw.close()
        QApplication.processEvents()
        new_host = self._make_host()
        self.win.i18n_transfer_children(new_host)
        self.assertEqual(new_host.open_project_windows(), {})

    def test_transfer_failure_keeps_old_window(self):
        """重建失败保旧窗（对齐宿主语义）：过户旧窗并重绑回调。"""
        pw = self._open()
        from xccv.training import training_window as tw_mod

        new_host = self._make_host()
        orig = tw_mod.load_project

        def _boom(name, *a, **k):
            raise RuntimeError("boom")

        tw_mod.load_project = _boom  # 重建路径强制失败
        try:
            self.win.i18n_transfer_children(new_host)
        finally:
            tw_mod.load_project = orig
        QApplication.processEvents()
        moved = new_host.open_project_windows().get("ui")
        self.assertIs(moved, pw, "重建失败应过户旧窗")
        self.assertTrue(pw.isVisible())
        self.assertIs(pw.on_closed.__self__, new_host)


class TestingWindowBusyTest(unittest.TestCase):
    """§三.2：已加载模型纳入语言重建忙时判定。"""

    def test_loaded_model_blocks_and_reason(self):
        from xccv.testing.testing_window import TestingWindow

        win = TestingWindow()
        self.addCleanup(win.close)
        QApplication.processEvents()
        self.assertFalse(win.i18n_rebuild_blocked())
        self.assertEqual(win.i18n_block_reason(),
                         "任务运行中，语言重开窗口后生效")
        win.model = object()  # 引擎模型占位：重建即丢已加载权重
        self.assertTrue(win.i18n_rebuild_blocked(),
                        "已加载模型未纳入忙时判定")
        self.assertEqual(win.i18n_block_reason(),
                         "已加载模型，语言重开窗口后生效")
        win.model = None
        self.assertFalse(win.i18n_rebuild_blocked())

if __name__ == '__main__':
    unittest.main()
