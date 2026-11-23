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

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


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
        _APP.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls.win.deleteLater()
        _APP.processEvents()
        cls._tmp.cleanup()

    def setUp(self):
        self.win.setEditMode()
        _APP.processEvents()

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
            _APP.processEvents()
            QTest.keyClick(w, QtCore.Qt.Key_Escape)
            _APP.processEvents()
            self.assertTrue(win.canvas.editing(),
                            "%s 获得焦点时 Esc 必须退出标注模式" % name)
            self.assertFalse(win.canvas.drawing(), name)

    def test_escape_from_floating_dock_list(self):
        """dock 浮出（列表成为独立顶层窗口）时 Esc 仍退出。"""
        win = self.win
        dock = win.labelList.parent()
        dock.setFloating(True) if hasattr(dock, "setFloating") else None
        _APP.processEvents()
        try:
            self._enter_drawing()
            win.labelList.setFocus()
            _APP.processEvents()
            QTest.keyClick(win.labelList, QtCore.Qt.Key_Escape)
            _APP.processEvents()
            self.assertTrue(win.canvas.editing(),
                            "浮出 dock 内的列表焦点下 Esc 必须退出")
        finally:
            if hasattr(dock, "setFloating"):
                dock.setFloating(False)
            _APP.processEvents()

    def test_escape_works_when_mode_set_directly(self):
        """绕过 toggleDrawMode 直接 setEditing(False) 进入标注态也能 Esc 退出。

        事件过滤器读 canvas.drawing() 实时判定，不依赖任何快捷键 enabled 状态。
        """
        win = self.win
        win.canvas.setEditing(False)
        win.canvas.createMode = "polygon"
        self.assertTrue(win.canvas.drawing())
        QTest.keyClick(win.fileListWidget, QtCore.Qt.Key_Escape)
        _APP.processEvents()
        self.assertTrue(win.canvas.editing())

    def test_escape_in_edit_mode_keeps_state(self):
        """编辑态 Esc 不触发退出（保留标签列表清选择等原语义）。"""
        win = self.win
        win.setEditMode()
        _APP.processEvents()
        win.labelList.setFocus()
        QTest.keyClick(win.labelList, QtCore.Qt.Key_Escape)
        _APP.processEvents()
        self.assertTrue(win.canvas.editing())

    def test_escape_in_dialog_not_intercepted(self):
        """对话框内 Esc 不被过滤器拦截（绘制态背景保持）。"""
        win = self.win
        self._enter_drawing()
        dlg = QtWidgets.QDialog(win)
        dlg.show()
        _APP.processEvents()
        try:
            QTest.keyClick(dlg, QtCore.Qt.Key_Escape)
            _APP.processEvents()
            # 对话框收到 Esc（QDialog 默认关闭），画布仍处标注态
            self.assertTrue(win.canvas.drawing(),
                            "对话框内 Esc 不得退出标注模式")
        finally:
            dlg.close()
            dlg.deleteLater()
            _APP.processEvents()

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
        _APP.processEvents()
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
        _APP.processEvents()
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


if __name__ == "__main__":
    unittest.main()
