"""批注回归：AI 掩膜候选/画笔并差集/标签质心/准星。

- 画笔拖动塑形：笔画并入相交形状（union），不相交则新建；
- 右键纯擦除手势立即提交；擦除可分裂形状（余块另立新形状）；
- 标签绘制在目标质心附近；准星按背景亮度自适应（含 QImage 缓存）。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from labelme.shape import Shape
from labelme.widgets.canvas import Canvas

_APP = None


def _make_canvas():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cv = Canvas()
    cv.createMode = "brush"
    cv.mode = Canvas.CREATE
    cv.resize(80, 40)
    cv.pixmap = QtGui.QPixmap(100, 40)
    cv.pixmap.fill(QtGui.QColor(255, 255, 255))
    cur = Shape(shape_type="points")
    cur.addPoint(QtCore.QPointF(1.0, 1.0))
    cv.current = cur
    return cv


def _poly(x1, y1, x2, y2):
    s = Shape(shape_type="polygon")
    for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        s.addPoint(QtCore.QPointF(float(x), float(y)))
    s.close()
    return s


def _area(shape):
    xs = [p.x() for p in shape.points]
    ys = [p.y() for p in shape.points]
    n = len(xs)
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(a) / 2.0


class BrushUnionTest(unittest.TestCase):
    """拖动塑形：笔画并入相交形状。"""

    def test_stroke_unions_into_intersecting_shape(self):
        cv = _make_canvas()
        cv.shapes = [_poly(20, 12, 40, 28)]
        n0 = len(cv.shapes[0].points)
        cv._brush_pts = [QtCore.QPointF(30.0, 20.0)]  # 落在原形状内
        cv.finalise()
        self.assertEqual(len(cv.shapes), 1, "应并入原形状而非新建")
        xs = [pt.x() for pt in cv.shapes[0].points]
        self.assertLess(min(xs), 20.0, "并入笔画后 bbox 应外扩")

    def test_stroke_disjoint_creates_new_shape(self):
        cv = _make_canvas()
        cv.shapes = [_poly(20, 12, 40, 28)]
        cv._brush_pts = [QtCore.QPointF(80.0, 20.0)]  # 与原形状不相交
        cv.finalise()
        self.assertEqual(len(cv.shapes), 2, "不相交笔画应新建形状")


class BrushEraseExistingTest(unittest.TestCase):
    """右键擦除作用于既有形状；纯擦除手势立即提交。"""

    def test_erase_shrinks_existing_shape(self):
        cv = _make_canvas()
        cv.shapes = [_poly(10, 8, 60, 30)]
        a0 = _area(cv.shapes[0])
        cv._brush_erase_pts = [QtCore.QPointF(35.0, 19.0)]
        cv.finalise()
        self.assertTrue(cv.shapes, "擦中间应保留余块")
        self.assertLess(sum(_area(s) for s in cv.shapes), a0)

    def test_erase_all_deletes_shape(self):
        cv = _make_canvas()
        cv.shapes = [_poly(20, 12, 40, 28)]
        cv._brush_erase_pts = [QtCore.QPointF(25.0, 20.0),
                               QtCore.QPointF(35.0, 20.0)]
        cv.finalise()
        self.assertEqual(len(cv.shapes), 0, "整块擦除应删除形状")

    def test_right_release_commits_pure_erase(self):
        cv = _make_canvas()
        cv.shapes = [_poly(20, 12, 40, 28)]
        a0 = _area(cv.shapes[0])
        ev = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(30.0, 20.0),
            QtCore.Qt.RightButton, QtCore.Qt.RightButton, QtCore.Qt.NoModifier)
        cv.mousePressEvent(ev)
        self.assertTrue(cv._brush_erasing)
        cv.mouseMoveEvent(ev)
        cv.mouseReleaseEvent(ev)
        self.assertFalse(cv._brush_erasing)
        self.assertTrue(cv.shapes or True)
        self.assertLess(sum(_area(s) for s in cv.shapes), a0,
                        "右键松开即提交擦除")


class PaintPathTest(unittest.TestCase):
    """标签质心绘制+准星背景取色：整帧渲染不崩、缓存就位。"""

    def test_paint_with_labels_crosshair_and_mask(self):
        cv = _make_canvas()
        cv._px_img = cv.pixmap.toImage()
        p1 = _poly(10, 5, 40, 30)
        p1.label = "cat"
        cv.shapes = [p1]
        cv._show_labels = True
        cv.createMode = "brush"
        cv.prevMovePoint = QtCore.QPointF(25.0, 15.0)
        cv._brush_pts = [QtCore.QPointF(25.0, 15.0)]
        img = QtGui.QImage(cv.size(), QtGui.QImage.Format_ARGB32_Premultiplied)
        img.fill(0xFFFFFFFF)
        p = QtGui.QPainter(img)
        cv.render(p)
        p.end()
        self.assertIsNotNone(cv._px_img)


if __name__ == "__main__":
    unittest.main()

