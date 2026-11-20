"""批注回归：canvas 重点 bug 与交互修复。

- finalise 无 createMode 守卫（SAM 加载后普通多边形闭合必抛 ValueError）；
- 模式切换不清 AI/画笔残留；
- 画笔拖动追加轨迹；
- AI 点击在模型未就绪时安全拒绝；
- 点数上限简化器。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from labelme.widgets import canvas as canvas_mod
from labelme.widgets.canvas import Canvas
from labelme.shape import Shape


_APP = None

def _make_canvas():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP, Canvas()


def _pts_shape(n=3):
    s = Shape(shape_type="points")
    for i in range(n):
        s.addPoint(QtCore.QPointF(10 + 10 * i, 10))
    return s


class _FakeSam:
    name = "fake-sam"


class FinaliseGuardTest(unittest.TestCase):
    """重点 bug：SAM 已加载 + 普通 polygon/rectangle 闭合不得进 SAM、不得炸。"""

    def test_polygon_close_with_sam_loaded(self):
        _, cv = _make_canvas()
        cv._sam = _FakeSam()
        cv.createMode = "polygon"
        cv.mode = Canvas.CREATE
        cv.current = _pts_shape(3)
        cv.finalise()          # 修复前此处抛 ValueError
        self.assertEqual(len(cv.shapes), 1)
        self.assertIsNone(cv.current)

    def test_rectangle_close_with_sam_loaded(self):
        _, cv = _make_canvas()
        cv._sam = _FakeSam()
        cv.createMode = "rectangle"
        cv.mode = Canvas.CREATE
        cv.current = _pts_shape(2)
        cv.finalise()
        self.assertEqual(len(cv.shapes), 1)

    def test_finalise_swallows_sam_failure(self):
        """SAM 精化异常降级为普通闭合（不再中断用户）。"""
        _, cv = _make_canvas()
        cv._sam = _FakeSam()
        cv.createMode = "ai_polygon"
        cv.mode = Canvas.CREATE
        cv.pixmap = QtGui.QPixmap(40, 40)
        cv.pixmap.fill(QtGui.QColor(255, 255, 255))
        cv.current = _pts_shape(3)
        cv.finalise()          # _update_shape_with_sam 内部异常 → 仍正常闭合
        self.assertEqual(len(cv.shapes), 1)


class ModeSwitchCleanupTest(unittest.TestCase):
    def test_switch_clears_ai_and_brush_residue(self):
        _, cv = _make_canvas()
        cv.createMode = "ai_polygon"
        cv.mode = Canvas.CREATE
        cv._ai_prompts = [(QtCore.QPointF(1, 1), 1)]
        cv._ai_preview_shape = Shape(shape_type="polygon")
        cv._ai_last_mask = object()
        cv.current = _pts_shape(1)
        cv.createMode = "rectangle"      # 切换 → 残留必须清空
        self.assertEqual(cv._ai_prompts, [])
        self.assertIsNone(cv._ai_preview_shape)
        self.assertIsNone(cv._ai_last_mask)
        self.assertIsNone(cv.current)
        self.assertEqual(cv._brush_pts, [])


class BrushDragTest(unittest.TestCase):
    def test_move_appends_stroke_points(self):
        _, cv = _make_canvas()
        cv.createMode = "brush"
        cv.mode = Canvas.CREATE
        cv.resize(50, 50)   # 部件小于图 → offsetToCenter=0，点击坐标不漂移
        cv.pixmap = QtGui.QPixmap(60, 60)
        cv.pixmap.fill(QtGui.QColor(255, 255, 255))
        cv.current = _pts_shape(1)
        cv._brush_pts = [QtCore.QPointF(5.0, 5.0)]
        ev = QtGui.QMouseEvent(
            QtGui.QMouseEvent.MouseMove, QtCore.QPointF(25.0, 12.0),
            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.NoModifier)
        cv.mouseMoveEvent(ev)
        self.assertEqual(len(cv._brush_pts), 2)
        self.assertAlmostEqual(cv._brush_pts[-1].x(), 25.0)
        self.assertAlmostEqual(cv._brush_pts[-1].y(), 12.0)


class AiNotReadyTest(unittest.TestCase):
    def test_click_before_model_ready_is_safe(self):
        _, cv = _make_canvas()
        cv.createMode = "ai_polygon"
        cv.mode = Canvas.CREATE
        cv.resize(50, 50)   # 部件小于图 → offsetToCenter=0，点击坐标不漂移
        cv.pixmap = QtGui.QPixmap(60, 60)
        cv.pixmap.fill(QtGui.QColor(255, 255, 255))
        cv._ai_loading = True          # 后台加载中
        statuses = []
        cv.aiStatus.connect(statuses.append)
        ev = QtGui.QMouseEvent(
            QtGui.QMouseEvent.MouseButtonPress, QtCore.QPointF(10.0, 10.0),
            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.NoModifier)
        cv.mousePressEvent(ev)
        self.assertEqual(cv._ai_prompts, [])
        self.assertIsNone(cv.current)
        self.assertTrue(any("加载中" in s for s in statuses))


class SimplifyPolyTest(unittest.TestCase):
    def test_max_points_respected(self):
        import numpy as np

        import math as _m

        poly = np.stack([100 * np.cos(np.linspace(0, 2 * _m.pi, 400)) + 100,
                         100 * np.sin(np.linspace(0, 2 * _m.pi, 400)) + 100], axis=1)
        out = Canvas._simplify_poly(poly, 50, 0.004)
        self.assertLessEqual(len(out), 50)
        self.assertGreaterEqual(len(out), 3)


if __name__ == "__main__":
    unittest.main()


class BrushEraseTest(unittest.TestCase):
    """批注F1/F6：画笔右键擦除=从掩膜抠除。"""

    def _cv_brush(self):
        _, cv = _make_canvas()
        cv.createMode = "brush"
        cv.mode = Canvas.CREATE
        cv.resize(60, 30)
        cv.pixmap = QtGui.QPixmap(100, 40)
        cv.pixmap.fill(QtGui.QColor(255, 255, 255))
        cv.current = _pts_shape(1)
        return cv

    def test_erase_subtracts_region(self):
        cv = self._cv_brush()
        cv._brush_pts = [QtCore.QPointF(10.0 + 10 * i, 20.0)
                         for i in range(6)]        # 10→60 横笔
        cv._brush_erase_pts = [QtCore.QPointF(45.0 + 5 * i, 20.0)
                               for i in range(4)]  # 45→60 擦除
        cv.finalise()
        self.assertEqual(len(cv.shapes), 1, "擦除后仍有形状")
        xs = [pt.x() for pt in cv.shapes[0].points]
        self.assertLess(max(xs), 50.0, "擦除区域应被抠掉")
        self.assertEqual(cv._brush_erase_pts, [])

    def test_erase_all_cancels_shape(self):
        cv = self._cv_brush()
        cv._brush_pts = [QtCore.QPointF(10.0 + 10 * i, 20.0)
                         for i in range(6)]
        cv._brush_erase_pts = [QtCore.QPointF(5.0 + 15 * i, 20.0)
                               for i in range(5)]  # 全覆盖擦除
        cv.finalise()
        self.assertEqual(len(cv.shapes), 0, "全部擦除 → 无形状")
        self.assertIsNone(cv.current)
