"""T5 回归：SAM 后台加载失败必须解锁 _ai_loading（一次失败曾永久吞点击）。

复测轮发现：initializeAiModelAsync 失败只发 aiModelFailed
状态栏消息，canvas._ai_loading 卡 True——后续点击全部被「AI 模型加载中」
吞掉（mousePressEvent 首锚分支守卫），S 键重试也被入口处
`if self._ai_loading: return` 拦截，AI 标注永久失效直到重启。修复：失败
路径把 _ai_loading 复位为 False，点击与重试即刻恢复可用。
"""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from labelme.widgets import canvas as canvas_mod
from labelme.widgets.canvas import Canvas

_APP = None


def _raising_factory(msg):
    class _Boom:
        name = "boom-sam"

        def __init__(self, *a, **k):
            raise RuntimeError(msg)

    return _Boom


class FailRecoveryTest(unittest.TestCase):
    """加载失败 → _ai_loading 复位 → 点击入栈恢复、S 键重试放行。"""

    def _patch_factory(self, cls):
        orig = canvas_mod.osam.apis.get_model_type_by_name
        canvas_mod.osam.apis.get_model_type_by_name = lambda name: cls
        self.addCleanup(lambda: setattr(canvas_mod.osam.apis,
                                        "get_model_type_by_name", orig))

    def _make_canvas(self):
        global _APP
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cv = Canvas()
        cv.resize(50, 50)   # 部件小于图 → offsetToCenter=0，点击坐标不漂移
        cv.pixmap = QtGui.QPixmap(60, 60)
        cv.pixmap.fill(QtGui.QColor(255, 255, 255))
        cv.createMode = "ai_polygon"
        cv.mode = Canvas.CREATE
        return _APP, cv

    def _wait_unlocked(self, app, cv, timeout=5.0):
        deadline = time.monotonic() + timeout
        while cv._ai_loading and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()

    def test_failed_load_unlocks_clicks(self):
        app, cv = self._make_canvas()
        self._patch_factory(_raising_factory("GPU 暂忙（模拟）"))
        failed = []
        cv.aiModelFailed.connect(failed.append)
        cv.initializeAiModelAsync("boom-sam")
        self._wait_unlocked(app, cv)
        self.assertTrue(failed)                       # 失败信号已发
        self.assertFalse(cv._ai_loading)              # 修复点：解锁
        ev = QtGui.QMouseEvent(
            QtGui.QMouseEvent.MouseButtonPress, QtCore.QPointF(10.0, 10.0),
            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.NoModifier)
        cv.mousePressEvent(ev)
        self.assertEqual(len(cv._ai_prompts), 1)      # 点击不再被吞

    def test_retry_allowed_after_failure(self):
        app, cv = self._make_canvas()
        self._patch_factory(_raising_factory("第一次失败（模拟）"))
        cv.initializeAiModelAsync("boom-sam")
        self._wait_unlocked(app, cv)
        self.assertFalse(cv._ai_loading)
        # 第二次（换名）加载不得再被入口 _ai_loading 拦截
        self._patch_factory(_raising_factory("第二次失败（模拟）"))
        statuses = []
        cv.aiStatus.connect(statuses.append)
        cv.initializeAiModelAsync("boom-sam-2")
        self._wait_unlocked(app, cv)
        self.assertTrue(any("后台加载中（" in s for s in statuses),
                        f"重试被拦截，状态={statuses}")
        self.assertFalse(cv._ai_loading)


if __name__ == "__main__":
    unittest.main()
