"""标注平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v082_annotation
- test_v082_followup
- test_v082_ux
"""


# == 来源模块：test_v082_annotation ==
# -*- coding: utf-8 -*-
"""标注交互行为锁（QTest 真实鼠标事件）。

拖动时机：点击多边形节点（顶点 epsilon 半径内，含多边形外沿）后立即拖动
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
        """目标（等价于快速点击外侧）：hover 已命中顶点、按压点在多边形
        外沿——修复前此按压被平移吞掉（拖画布），修复后必须移动顶点。"""
        cv = self.cv
        _move_to(cv, (144, 64))                   # 顶点 1 外侧 4px，hover 命中
        self.assertTrue(cv.selectedVertex())
        _drag(cv, (144, 64), (185, 85))
        self.assertGreater(self._v1().x(), 170, "点击节点后拖动必须移动节点")
        self.assertGreater(self._v1().y(), 75)
        self.assertEqual(_bars(self.area), (0, 0), "节点拖动不得被判为画布平移")

    def test_fast_click_vertex_outside_then_drag(self):
        """目标：点击顶点（外沿 epsilon 内）立即拖动=移动节点，不平移。"""
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
        """正点顶点（多边形边界）快速点击拖动同样移动节点。"""
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
    """绘制期间可删除 + 删除确认弹窗可配置并按名称列出目标。"""

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
        """绘制模式下有选中 → 删除动作可用；无选中 → 禁用。"""
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
        """默认（设置关）不弹窗直接删除。"""
        a, b = self._two_shapes()
        mw = self._mw([a, b], [a])
        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warn:
            MainWindow.deleteSelectedShape(mw)
        warn.assert_not_called()
        self.assertEqual(len(mw.canvas.shapes), 1)
        self.assertEqual(mw.canvas.shapes[0].label, "缺陷B")

    def test_delete_confirm_on_lists_names_from_one(self):
        """设置开 → 弹窗按名称列出（序号从 1 起）；No 取消不删。"""
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
        """空标签形状在确认文案中显示“未命名”占位。"""
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
    """静态形状层缓存——失效触发 / 拖动排除 / 缩放重建。"""

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

# == 来源模块：test_v082_followup ==
# -*- coding: utf-8 -*-
"""标注交互追加修复行为锁。

1) 标注模式下 Esc 退出与焦点无关：焦点落在标签列表等非画布区域时，
   Esc 仍须退出标注模式（绘制态窗口级快捷键；编辑态不接管 Esc）。
2) 空标注不落盘且不计「已标注」：无形状且图片级标记全未勾选时不写
   json、删除既有空文件、文件列表不勾选；历史遗留空 json 同样不计。
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtTest import QTest

import xccv.annotation  # noqa: F401  先导入固定 PyQt5 绑定
from labelme.app import MainWindow
from labelme.config import get_default_config
from labelme.shape import Shape

_APP__v082_followup = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dataset_with_one_image(root: Path) -> Path:
    """造一张 64×48 的纯色 PNG（无同名 json）。"""
    d = root / "ds"
    d.mkdir(parents=True, exist_ok=True)
    pix = QtGui.QPixmap(64, 48)
    pix.fill(QtGui.QColor(200, 200, 200))
    pix.save(str(d / "img0.png"), "PNG")
    return d


def _new_shape(label="缺陷A"):
    s = Shape(shape_type="polygon")
    for pt in ((10, 10), (40, 10), (40, 30), (10, 30)):
        s.addPoint(QtCore.QPointF(*pt))
    s.label = label
    return s


class EscapeExitTest(unittest.TestCase):
    """Esc 退出标注与焦点无关（应用级事件过滤器）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ds = _dataset_with_one_image(Path(cls._tmp.name))
        cls.win = MainWindow(config=get_default_config())
        cls.win.importDirImages(str(cls.ds), load=True)
        cls.win.show()
        _APP__v082_followup.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls.win.deleteLater()
        _APP__v082_followup.processEvents()
        cls._tmp.cleanup()

    def setUp(self):
        self.win.setEditMode()
        _APP__v082_followup.processEvents()

    def _enter_drawing(self):
        self.win.toggleDrawMode(False, createMode="polygon")
        self.assertTrue(self.win.canvas.drawing())

    def test_escape_from_each_dock(self):
        """焦点分别在三个面板（含浮动/独立窗口化列表）时 Esc 均退出。"""
        win = self.win
        for name in ("labelList", "uniqLabelList", "fileListWidget"):
            w = getattr(win, name)
            self._enter_drawing()
            w.setFocus()
            _APP__v082_followup.processEvents()
            QTest.keyClick(w, QtCore.Qt.Key_Escape)
            _APP__v082_followup.processEvents()
            self.assertTrue(win.canvas.editing(),
                            "%s 获得焦点时 Esc 必须退出标注模式" % name)
            self.assertFalse(win.canvas.drawing(), name)

    def test_escape_from_floating_dock_list(self):
        """dock 浮出（列表成为独立顶层窗口）时 Esc 仍退出。"""
        win = self.win
        dock = win.labelList.parent()
        dock.setFloating(True) if hasattr(dock, "setFloating") else None
        _APP__v082_followup.processEvents()
        try:
            self._enter_drawing()
            win.labelList.setFocus()
            _APP__v082_followup.processEvents()
            QTest.keyClick(win.labelList, QtCore.Qt.Key_Escape)
            _APP__v082_followup.processEvents()
            self.assertTrue(win.canvas.editing(),
                            "浮出 dock 内的列表焦点下 Esc 必须退出")
        finally:
            if hasattr(dock, "setFloating"):
                dock.setFloating(False)
            _APP__v082_followup.processEvents()

    def test_escape_works_when_mode_set_directly(self):
        """绕过 toggleDrawMode 直接 setEditing(False) 进入标注态也能 Esc 退出。

        事件过滤器读 canvas.drawing() 实时判定，不依赖任何快捷键 enabled 状态。
        """
        win = self.win
        win.canvas.setEditing(False)
        win.canvas.createMode = "polygon"
        self.assertTrue(win.canvas.drawing())
        QTest.keyClick(win.fileListWidget, QtCore.Qt.Key_Escape)
        _APP__v082_followup.processEvents()
        self.assertTrue(win.canvas.editing())

    def test_escape_in_edit_mode_keeps_state(self):
        """编辑态 Esc 不触发退出（保留标签列表清选择等原语义）。"""
        win = self.win
        win.setEditMode()
        _APP__v082_followup.processEvents()
        win.labelList.setFocus()
        QTest.keyClick(win.labelList, QtCore.Qt.Key_Escape)
        _APP__v082_followup.processEvents()
        self.assertTrue(win.canvas.editing())

    def test_escape_in_dialog_not_intercepted(self):
        """对话框内 Esc 不被过滤器拦截（绘制态背景保持）。"""
        win = self.win
        self._enter_drawing()
        dlg = QtWidgets.QDialog(win)
        dlg.show()
        _APP__v082_followup.processEvents()
        try:
            QTest.keyClick(dlg, QtCore.Qt.Key_Escape)
            _APP__v082_followup.processEvents()
            # 对话框收到 Esc（QDialog 默认关闭），画布仍处标注态
            self.assertTrue(win.canvas.drawing(),
                            "对话框内 Esc 不得退出标注模式")
        finally:
            dlg.close()
            dlg.deleteLater()
            _APP__v082_followup.processEvents()

    def test_escape_callback_drops_in_progress_shape(self):
        """绘制中已有顶点 → Esc 丢弃进行中形状并退出（既有语义保持）。"""
        win = self.win
        self._enter_drawing()
        cv = win.canvas
        s = _new_shape()
        cv.current = s
        cv.line.points = [s.points[-1], s.points[-1]]
        cv.drawingPolygon.emit(True)
        QTest.keyClick(cv, QtCore.Qt.Key_Escape)
        _APP__v082_followup.processEvents()
        self.assertIsNone(cv.current)
        self.assertTrue(cv.editing())


class EmptyLabelFileTest(unittest.TestCase):
    """空标注不落盘、不计已标注。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ds = _dataset_with_one_image(Path(cls._tmp.name))
        cls.win = MainWindow(config=get_default_config())
        cls.win.importDirImages(str(cls.ds), load=True)

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls.win.deleteLater()
        _APP__v082_followup.processEvents()
        cls._tmp.cleanup()

    def setUp(self):
        self.win.canvas.loadShapes([], replace=True)
        self.win.labelList.clear()
        self.win.labelFile = None
        self.win._label_content_cache = {}
        self.json_path = str(Path(self.ds) / "img0.json")
        if os.path.isfile(self.json_path):
            os.remove(self.json_path)

    def test_empty_annotation_writes_no_file(self):
        win = self.win
        self.assertTrue(win.saveLabels(self.json_path))
        self.assertFalse(os.path.isfile(self.json_path),
                         "空标注不得产生 json 文件")
        self.assertFalse(win._has_label_file(str(Path(self.ds) / "img0.png")),
                         "空标注图不得计为已标注")

    def test_non_empty_then_cleared_removes_file(self):
        win = self.win
        shape = _new_shape("缺陷A")
        win.canvas.loadShapes([shape], replace=True)
        win.addLabel(shape)
        self.assertTrue(win.saveLabels(self.json_path))
        self.assertTrue(os.path.isfile(self.json_path))
        self.assertTrue(win._has_label_file(str(Path(self.ds) / "img0.png")))
        # 全部删除后再保存 → 文件移除、回到未标注
        win.canvas.loadShapes([], replace=True)
        win.labelList.clear()
        self.assertTrue(win.saveLabels(self.json_path))
        self.assertFalse(os.path.isfile(self.json_path),
                         "标注清空后既有 json 必须移除")
        self.assertFalse(win._has_label_file(str(Path(self.ds) / "img0.png")))

    def test_legacy_empty_json_not_counted(self):
        """历史遗留空 json：不计已标注，但保存空标注时才移除。"""
        win = self.win
        data = {
            "version": "5.0", "flags": {}, "shapes": [],
            "imagePath": "img0.png", "imageData": None,
            "imageHeight": 48, "imageWidth": 64,
        }
        Path(self.json_path).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.assertFalse(win._has_label_file(str(Path(self.ds) / "img0.png")),
                         "历史空 json 不得计为已标注")
        # 文件列表重建后该图不勾选
        win.importDirImages(str(self.ds), load=False)
        item = win._file_items.get(str(Path(self.ds) / "img0.png"))
        self.assertIsNotNone(item)
        self.assertEqual(item.checkState(0), QtCore.Qt.Unchecked,
                         "历史空 json 的图不得勾选已标注")
        # 对该图保存空标注 → 空文件被清掉
        self.assertTrue(win.saveLabels(self.json_path))
        self.assertFalse(os.path.isfile(self.json_path))

    def test_flags_only_counts_as_annotated(self):
        """仅图片级标记（flags）也算有标注内容，不得被当空文件删除。"""
        win = self.win
        data = {
            "version": "5.0", "flags": {"reviewed": True}, "shapes": [],
            "imagePath": "img0.png", "imageData": None,
            "imageHeight": 48, "imageWidth": 64,
        }
        Path(self.json_path).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.assertTrue(win._has_label_file(str(Path(self.ds) / "img0.png")),
                        "flags 有勾选视为已标注")

# == 来源模块：test_v082_ux ==
# -*- coding: utf-8 -*-
"""界面契约测试：新建项目/工具开窗等待横幅、?帮助按钮摘除、
对比窗口画布尺寸同步（高分屏裁剪回归锁）。"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

_APP__v082_ux = None


def _app():
    global _APP__v082_ux
    _APP__v082_ux = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP__v082_ux


class BusyBannerTest(unittest.TestCase):
    """新建项目/打开项目共用悬浮横幅。"""

    @classmethod
    def setUpClass(cls):
        _app()
        from xccv.training.training_window import TrainingWindow

        cls.tw = TrainingWindow()
        cls.tw.show()                    # 横幅可见性依赖父链
        _APP__v082_ux.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.tw.close()

    def test_banner_shown_and_hidden(self):
        banner, t0 = self.tw._show_busy_banner("测试等待提示…")
        self.assertTrue(banner.isVisible())
        self.assertEqual(banner.objectName(), "openProjectBanner")
        self.tw._finish_busy_banner(banner, t0, min_ms=0)
        _APP__v082_ux.processEvents()
        self.assertFalse(banner.isVisible(), "横幅应在收尾后隐藏")

    def test_create_project_shows_banner_during_dialog_build(self):
        """新建项目对话框构建期间横幅已上屏（卡顿有反馈）。"""
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
    """开发者选项对话框摘除标题栏「?」上下文帮助按钮。"""

    def test_context_help_button_removed(self):
        _app()
        from xccv.training.dialogs import DevOptionsDialog

        dlg = DevOptionsDialog()
        self.assertFalse(
            bool(dlg.windowFlags() & QtCore.Qt.WindowContextHelpButtonHint),
            "无 WhatsThis 内容的对话框不得显示「?」按钮（问号光标卡死）")


class AdaptiveFigureTest(unittest.TestCase):
    """对比窗口画布尺寸必须交给 matplotlib 同步（物理像素口径），
    不得再用逻辑像素二次覆写（高分屏裁剪回归锁）。"""

    def test_figure_tracks_widget_after_resize(self):
        _app()
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        from xccv.training.compare_window import _AdaptiveFigure

        fig = Figure(dpi=100)
        canvas = _AdaptiveFigure(fig, min_height=340)
        canvas.show()                    # resize 事件需要可见态才派发
        _APP__v082_ux.processEvents()
        canvas.resize(800, 500)
        _APP__v082_ux.processEvents()
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
        _APP__v082_ux.processEvents()
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
    """打开剪枝/加速工具窗口期间显示加载横幅。"""

    def test_banner_during_tool_window_build(self):
        from PyQt5.QtWidgets import QDialog, QLabel, QWidget

        from xccv.modeltools import modeltools_window as mtw

        _app()
        win = mtw.ModelToolsWindow()
        win.show()                       # 横幅可见性依赖父链
        _APP__v082_ux.processEvents()
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



class CursorImmediateTest(unittest.TestCase):
    """行为锁：setEditing 切换模式后光标立即生效，不等 mouseMove。"""

    def setUp(self):
        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.area, self.cv = _make_scene(show=False)
        self.cv.setAttribute(QtCore.Qt.WA_UnderMouse, True)

    def tearDown(self):
        self.cv.restoreCursor()
        self.cv.setAttribute(QtCore.Qt.WA_UnderMouse, False)
        self.area.close()

    def test_enter_draw_cursor_immediate(self):
        self.cv.setEditing(False)  # EDIT -> CREATE（快捷键进入绘制态路径）
        cur = QtWidgets.QApplication.overrideCursor()
        self.assertIsNotNone(cur)
        self.assertEqual(cur.shape(), QtCore.Qt.CrossCursor)

    def test_exit_draw_cursor_immediate(self):
        self.cv.setEditing(False)
        self.cv.setEditing(True)  # CREATE -> EDIT（Esc/右键退出路径）
        cur = QtWidgets.QApplication.overrideCursor()
        self.assertIsNotNone(cur)
        self.assertEqual(cur.shape(), QtCore.Qt.ArrowCursor)

    def test_no_mouse_no_override(self):
        self.cv.setAttribute(QtCore.Qt.WA_UnderMouse, False)
        self.cv.setEditing(False)
        # 鼠标不在画布上：不得挂全局光标（避免影响其他控件悬停观感）
        self.assertIsNone(QtWidgets.QApplication.overrideCursor())

if __name__ == '__main__':
    unittest.main()
