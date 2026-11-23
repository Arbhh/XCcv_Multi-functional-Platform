# -*- coding: utf-8 -*-
"""V0.8.2 标注交互行为锁（T1~T3 场景，QTest 真实鼠标事件）。

T1 拖动时机：点击多边形节点（顶点 epsilon 半径内，含多边形外沿）后立即拖动
必须判定为移动节点，不得被判为画布平移——修复前从外沿点顶点会被当作空白区
平移（平移判定用 containsPoint、节点抓取用 epsilon 半径，两套口径不一致）。
其余手势（空白平移 / 中键平移 / Shift 平移 / 顶点按住拖动 / 整形拖动 /
Alt+Shift 删点 / Alt 加点）为现状行为锁，修复不得波及。
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from labelme.app import MainWindow
from labelme.shape import Shape
from labelme.widgets.canvas import Canvas

_APP = None


def _make_scene(img_w=400, img_h=300, show=True):
    """编辑模式画布 + 一个 4 顶点多边形 (60,60)-(140,140)，滚动区可平移。"""
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    area = QtWidgets.QScrollArea()
    area.resize(260, 220)
    cv = Canvas()
    area.setWidget(cv)
    cv.setFixedSize(img_w, img_h)      # scale=1 且 offsetToCenter=0：pos=图像坐标
    pix = QtGui.QPixmap(img_w, img_h)
    pix.fill(QtGui.QColor(255, 255, 255))
    cv.loadPixmap(pix, clear_shapes=False)
    shape = Shape(shape_type="polygon")
    for pt in ((60, 60), (140, 60), (140, 140), (60, 140)):
        shape.addPoint(QtCore.QPointF(pt[0], pt[1]))
    shape.label = "缺陷A"
    cv.loadShapes([shape], replace=True)
    cv.mode = Canvas.EDIT
    # 镜像主程序接线（app.shapeSelectionChanged 把选中集合写回画布）
    cv.selectionChanged.connect(
        lambda shapes: setattr(cv, "selectedShapes", list(shapes)))
    if show:
        area.show()
    return area, cv


def _mouse(cv, etype, pos, button=QtCore.Qt.NoButton,
           buttons=QtCore.Qt.NoButton, mods=QtCore.Qt.NoModifier):
    ev = QtGui.QMouseEvent(etype, QtCore.QPointF(pos[0], pos[1]),
                           QtCore.QPointF(pos[0], pos[1]), button, buttons, mods)
    QtWidgets.QApplication.sendEvent(cv, ev)


def _move_to(cv, pos):
    _mouse(cv, QtCore.QEvent.MouseMove, pos)


def _drag(cv, start, end, button=QtCore.Qt.LeftButton,
          mods=QtCore.Qt.NoModifier, steps=5):
    _mouse(cv, QtCore.QEvent.MouseButtonPress, start, button=button,
           buttons=button, mods=mods)
    for i in range(1, steps + 1):
        x = start[0] + (end[0] - start[0]) * i / steps
        y = start[1] + (end[1] - start[1]) * i / steps
        _mouse(cv, QtCore.QEvent.MouseMove, (x, y), buttons=button, mods=mods)
    _mouse(cv, QtCore.QEvent.MouseButtonRelease, end, button=button, mods=mods)


def _bars(area):
    return (area.horizontalScrollBar().value(),
            area.verticalScrollBar().value())


def _force_paint(cv):
    """确定性强制绘制：直接投递 QPaintEvent（repaint 依赖窗口暴露状态，
    offscreen 夹具下未 processEvents 时会被静默跳过）。"""
    QtWidgets.QApplication.sendEvent(cv, QtGui.QPaintEvent(cv.rect()))


class _CanvasCase(unittest.TestCase):
    """每个用例复位：原形状 / hover 清空 / 选中清空 / 滚动归零。"""

    @classmethod
    def setUpClass(cls):
        cls.area, cls.cv = _make_scene()

    def setUp(self):
        cv = self.cv
        shape = Shape(shape_type="polygon")
        for pt in ((60, 60), (140, 60), (140, 140), (60, 140)):
            shape.addPoint(QtCore.QPointF(pt[0], pt[1]))
        shape.label = "缺陷A"
        cv.loadShapes([shape], replace=True)
        cv.hVertex = None
        cv.hShape = None
        cv.hEdge = None
        cv.prevhVertex = None
        cv.prevhShape = None
        cv.prevhEdge = None
        cv.selectedShapes = []
        cv.hShapeIsSelected = False
        cv.movingShape = False
        self.area.horizontalScrollBar().setValue(0)
        self.area.verticalScrollBar().setValue(0)

    def _v0(self):
        return self.cv.shapes[0].points[0]

    def _v1(self):
        return self.cv.shapes[0].points[1]

    def _other_points(self):
        """除 v1 外的三个顶点（拖动/编辑不得波及）。"""
        pts = self.cv.shapes[0].points
        return [(pts[0].x(), pts[0].y()), (pts[2].x(), pts[2].y()),
                (pts[3].x(), pts[3].y())]

    def _clear_hover(self):
        _move_to(self.cv, (20, 280))   # 左下角空白处——清全部 hover 状态


class PanGesturesTest(_CanvasCase):
    """现状行为锁：既有平移手势全部保持。"""

    def test_left_drag_on_empty_area_pans(self):
        _drag(self.cv, (320, 260), (260, 220))    # 向左上拖 → 滚动条增大
        h, v = _bars(self.area)
        self.assertTrue(h > 0 or v > 0, "空白区左键拖动必须平移画布")

    def test_middle_drag_pans_anywhere(self):
        _drag(self.cv, (240, 200), (180, 150), button=QtCore.Qt.MiddleButton)
        h, v = _bars(self.area)
        self.assertTrue(h > 0 or v > 0, "中键拖动必须平移画布")

    def test_shift_left_drag_on_shape_pans(self):
        _drag(self.cv, (240, 200), (180, 150), mods=QtCore.Qt.ShiftModifier)
        h, v = _bars(self.area)
        self.assertTrue(h > 0 or v > 0, "Shift+左键在形状上必须平移画布")


class VertexMoveTest(_CanvasCase):
    """顶点拖动：hover 旧路径保持 + 快速点击新路径。"""

    def test_hover_inside_then_hold_drag_moves_vertex(self):
        """现状锁：hover 命中顶点 + 从多边形内侧按住拖动 = 移动顶点。"""
        cv = self.cv
        _move_to(cv, (136, 64))                   # 顶点 1 (140,60) 内侧 4px
        self.assertTrue(cv.selectedVertex())
        _drag(cv, (136, 64), (185, 85))
        self.assertGreater(self._v1().x(), 170, "顶点应随按住拖动移动")
        self.assertGreater(self._v1().y(), 75)
        self.assertEqual(_bars(self.area), (0, 0), "顶点拖动不得平移画布")

    def test_hover_outside_press_is_not_pan(self):
        """T1 目标（等价于快速点击外侧）：hover 已命中顶点、按压点在多边形
        外沿——修复前此按压被平移吞掉（拖画布），修复后必须移动顶点。"""
        cv = self.cv
        _move_to(cv, (144, 64))                   # 顶点 1 外侧 4px，hover 命中
        self.assertTrue(cv.selectedVertex())
        _drag(cv, (144, 64), (185, 85))
        self.assertGreater(self._v1().x(), 170, "点击节点后拖动必须移动节点")
        self.assertGreater(self._v1().y(), 75)
        self.assertEqual(_bars(self.area), (0, 0), "节点拖动不得被判为画布平移")

    def test_fast_click_vertex_outside_then_drag(self):
        """T1 目标：点击节点（顶点外沿 epsilon 内）立即拖动=移动节点，不平移。"""
        cv = self.cv
        self._clear_hover()
        self.assertFalse(cv.selectedVertex())
        _drag(cv, (149, 60), (185, 85))           # 顶点 (140,60) 外沿 9px
        self.assertGreater(self._v1().x(), 170, "点击节点后拖动必须移动节点")
        self.assertGreater(self._v1().y(), 75)
        self.assertEqual(_bars(self.area), (0, 0), "节点拖动不得被判为画布平移")
        self.assertEqual(self._other_points(),
                         [(60.0, 60.0), (140.0, 140.0), (60.0, 140.0)],
                         "其余顶点不得被波及")

    def test_fast_click_on_vertex_exact_then_drag(self):
        """T1 目标：正点顶点（多边形边界）快速点击拖动同样移动节点。"""
        cv = self.cv
        self._clear_hover()
        _drag(cv, (140, 60), (175, 90))
        self.assertGreater(self._v1().x(), 165)
        self.assertGreater(self._v1().y(), 80)
        self.assertEqual(_bars(self.area), (0, 0))

    def test_drag_shape_body_moves_whole_shape(self):
        """现状锁：形状内部（远离顶点）拖动=整形移动。"""
        cv = self.cv
        self._clear_hover()
        pts_before = [(p.x(), p.y()) for p in cv.shapes[0].points]
        _drag(cv, (100, 100), (150, 130))
        pts_after = [(p.x(), p.y()) for p in cv.shapes[0].points]
        self.assertEqual([(x + 50, y + 30) for x, y in pts_before], pts_after,
                         "整形拖动=整体位移 (+50,+30)")
        self.assertEqual(_bars(self.area), (0, 0))


class VertexEditTest(_CanvasCase):
    """现状行为锁：加减点路径不被命中口径改动波及。"""

    def test_alt_shift_click_on_vertex_deletes_point(self):
        cv = self.cv
        _move_to(cv, (140, 60))                   # hover 顶点 0
        _mouse(cv, QtCore.QEvent.MouseButtonPress, (140, 60),
               button=QtCore.Qt.LeftButton, buttons=QtCore.Qt.LeftButton,
               mods=QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier)
        self.assertEqual(len(cv.shapes[0].points), 3, "Alt+Shift 点顶点=删点")

    def test_alt_click_on_edge_adds_point(self):
        cv = self.cv
        _move_to(cv, (140, 100))                  # 右边中点——hover 边
        self.assertIsNotNone(cv.hEdge)
        _mouse(cv, QtCore.QEvent.MouseButtonPress, (140, 100),
               button=QtCore.Qt.LeftButton, buttons=QtCore.Qt.LeftButton,
               mods=QtCore.Qt.AltModifier)
        self.assertEqual(len(cv.shapes[0].points), 5, "Alt 点边=加点")


class DeleteContractTest(unittest.TestCase):
    """T2/T3：绘制期间可删除 + 删除确认弹窗可配置并按名称列出目标。"""

    def _mw(self, shapes, selected):
        """轻量 MainWindow 桩（只挂 deleteSelectedShape 所需成员）。"""
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        area, cv = _make_scene(show=False)   # 不显示——删除契约无需真实绘制
        for s in shapes:
            s.label = s.label or ""
        cv.loadShapes(shapes, replace=True)
        cv.selectedShapes = list(selected)
        mw = MainWindow.__new__(MainWindow)
        mw.canvas = cv
        mw._config = {"delete_confirm": False}
        mw.tr = lambda s, *a: s
        mw.remLabels = lambda shapes: shapes
        mw.setDirty = lambda: None
        mw.noShapes = lambda: False
        mw.actions = QtWidgets.QWidget()
        mw.actions.editMode = QtWidgets.QAction(_APP)
        mw.actions.undoLastPoint = QtWidgets.QAction(_APP)
        mw.actions.undo = QtWidgets.QAction(_APP)
        mw.actions.delete = QtWidgets.QAction(_APP)
        mw.actions.onShapesPresent = []
        mw._area = area          # 保住父级引用，防 Qt 提前回收 Canvas
        return mw

    @staticmethod
    def _two_shapes():
        a = Shape(shape_type="polygon")
        for pt in ((10, 10), (30, 10), (30, 30), (10, 30)):
            a.addPoint(QtCore.QPointF(*pt))
        a.label = "缺陷A"
        b = Shape(shape_type="polygon")
        for pt in ((50, 10), (70, 10), (70, 30), (50, 30)):
            b.addPoint(QtCore.QPointF(*pt))
        b.label = "缺陷B"
        return a, b

    def test_delete_enabled_while_drawing_with_selection(self):
        """T2：绘制模式下有选中 → 删除动作可用；无选中 → 禁用。"""
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        a, b = self._two_shapes()
        mw = self._mw([a, b], [a])
        self.assertTrue(mw.actions.delete.isEnabled())
        mw.toggleDrawingSensitive(drawing=True)
        self.assertTrue(mw.actions.delete.isEnabled(),
                        "绘制模式且有多边形选中时删除必须可用")
        mw.canvas.selectedShapes = []
        mw.toggleDrawingSensitive(drawing=True)
        self.assertFalse(mw.actions.delete.isEnabled())
        mw.toggleDrawingSensitive(drawing=False)
        self.assertTrue(mw.actions.delete.isEnabled())
        _APP.processEvents()

    def test_delete_confirm_off_deletes_without_dialog(self):
        """T3：默认（设置关）不弹窗直接删除。"""
        a, b = self._two_shapes()
        mw = self._mw([a, b], [a])
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warn:
            MainWindow.deleteSelectedShape(mw)
        warn.assert_not_called()
        self.assertEqual(len(mw.canvas.shapes), 1)
        self.assertEqual(mw.canvas.shapes[0].label, "缺陷B")

    def test_delete_confirm_on_lists_names_from_one(self):
        """T3：设置开 → 弹窗按名称列出（序号从 1 起）；No 取消不删。"""
        from unittest import mock as _mock

        a, b = self._two_shapes()
        mw = self._mw([a, b], [a, b])
        mw._config["delete_confirm"] = True
        captured = {}
        with _mock.patch.object(
            QtWidgets.QMessageBox, "warning",
            side_effect=lambda *args, **kw: captured.update(msg=args[2])
            or QtWidgets.QMessageBox.No,
        ):
            MainWindow.deleteSelectedShape(mw)
        self.assertIn("1. 缺陷A", captured["msg"], "确认文案按名称列出（从 1 起）")
        self.assertIn("2. 缺陷B", captured["msg"])
        self.assertEqual(len(mw.canvas.shapes), 2, "No 取消不得删除")
        with _mock.patch.object(
            QtWidgets.QMessageBox, "warning",
            side_effect=lambda *args, **kw: captured.update(msg=args[2])
            or QtWidgets.QMessageBox.Yes,
        ):
            MainWindow.deleteSelectedShape(mw)
        self.assertEqual(len(mw.canvas.shapes), 0, "Yes 后删除")

    def test_delete_confirm_unnamed_fallback(self):
        """T3：空标签形状在确认文案中显示“未命名”占位。"""
        a = Shape(shape_type="polygon")
        for pt in ((10, 10), (30, 10), (30, 30), (10, 30)):
            a.addPoint(QtCore.QPointF(*pt))
        a.label = ""
        mw = self._mw([a], [a])
        mw._config["delete_confirm"] = True
        captured = {}
        with mock.patch.object(
            QtWidgets.QMessageBox, "warning",
            side_effect=lambda *args, **kw: captured.update(msg=args[2])
            or QtWidgets.QMessageBox.Yes,
        ):
            MainWindow.deleteSelectedShape(mw)
        self.assertIn("1. Unnamed", captured["msg"])
        self.assertEqual(len(mw.canvas.shapes), 0)


class OverlayCacheTest(_CanvasCase):
    """T4：静态形状层缓存——失效触发 / 拖动排除 / 缩放重建。"""

    def test_built_on_paint_and_bumped_on_mutations(self):
        cv = self.cv
        v0 = cv._overlay_version
        _force_paint(cv)
        self.assertIsNotNone(cv._overlay_cache, "绘制后应建立静态形状层缓存")
        self.assertEqual(cv._overlay_version, v0)          # 纯绘制不失效
        cv.selectedShapes = [cv.shapes[0]]                  # 选中后删除
        cv.deleteSelected()
        self.assertGreater(cv._overlay_version, v0)
        self.assertIsNone(cv._overlay_cache, "形状集变更后缓存应清空")
        _force_paint(cv)
        self.assertIsNotNone(cv._overlay_cache)

    def test_drag_excludes_moving_shape_from_cache_key(self):
        cv = self.cv
        _force_paint(cv)
        key_idle = cv._overlay_key
        cv.hShape = cv.shapes[0]                            # hover 顶点中
        cv._vertex_drag = True                              # 顶点拖动开始
        _force_paint(cv)
        self.assertIn(id(cv.hShape), cv._overlay_key[-1],
                      "拖动中的顶点形状应进入实时排除集")
        _force_paint(cv)
        self.assertEqual(cv._overlay_key, cv._overlay_key)  # 同键不重建
        cv._vertex_drag = False
        _force_paint(cv)
        self.assertEqual(cv._overlay_key, key_idle, "拖动结束后回到静息键")

    def test_zoom_rebuilds_caches_without_crash(self):
        """缩放状态（含留白偏移）下两缓存按键重建、绘制不崩。"""
        cv = self.cv
        cv.setFixedSize(500, 400)                # 视口与图像尺寸不一致→留白
        cv.scale = 2.5
        _force_paint(cv)
        self.assertIsNotNone(cv._image_cache)
        self.assertEqual(cv._image_cache.width(),
                         round(cv.pixmap.width() * 2.5))
        self.assertEqual(cv._image_cache.height(),
                         round(cv.pixmap.height() * 2.5))
        self.assertIsNotNone(cv._overlay_cache)
        cv.scale = 1.0
        cv.setFixedSize(400, 300)
        _force_paint(cv)


if __name__ == "__main__":
    unittest.main()
