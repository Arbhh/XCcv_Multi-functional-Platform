"""标注平台（内置 labelme 修改版）无头冒烟测试。

验证：主窗口构建、打开目录（递归）、新形状标注流程与 classes.txt 同步、
分类归档。不涉及鼠标键盘交互（人工测试）。
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QApplication

import xccv.annotation  # noqa: F401 先导入：固定 PyQt5 绑定并在需要时装 osam 桩
from labelme.app import MainWindow
from labelme.config import get_default_config
from labelme.shape import Shape

from tests.synthetic import make_classification_dataset, make_labelme_dataset

_APP = QApplication.instance() or QApplication(sys.argv)


def _make_window(dataset_dir) -> MainWindow:
    win = MainWindow(config=get_default_config())
    win.importDirImages(str(dataset_dir), load=True)
    return win


def _append_unlabeled_rect(win):
    s = Shape(shape_type="rectangle")
    s.addPoint(QPointF(1.0, 1.0))
    s.addPoint(QPointF(30.0, 30.0))
    s.close()
    win.canvas.loadShapes(list(win.canvas.shapes) + [s], replace=True)
    return s


class LabelmeForkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.det_dir, _ = make_labelme_dataset(cls.root / "det", n_images=4)
        cls.cls_dir, _ = make_classification_dataset(cls.root / "cls", per_class=3)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_open_dir_loads_recursively(self):
        make_labelme_dataset(self.det_dir / "more", n_images=2)
        try:
            win = _make_window(self.det_dir)
            self.assertEqual(len(win.imageList), 6)  # 4 张 + 子目录 2 张
            # 树形：根目录图片为顶层节点，子文件夹为可展开节点，显示原名
            self.assertEqual(win.fileListWidget.topLevelItemCount(), 5)
            folder = win.fileListWidget.topLevelItem(4)
            self.assertEqual(folder.text(0), "more")
            self.assertEqual(folder.childCount(), 2)
            self.assertIsNone(folder.data(0, 0x0100))  # 文件夹节点无图片路径
            self.assertEqual(folder.child(0).text(0), "img00.png")  # 原名，不带路径
            self.assertTrue(win.filename.endswith("img00.png"))
            self.assertEqual(len(win.canvas.shapes), 1)  # 首图自带标注
        finally:
            import shutil
            shutil.rmtree(self.det_dir / "more")

    def test_shortcuts_and_right_click_undo(self):
        win = _make_window(self.det_dir)
        self.assertEqual(win._config["shortcuts"]["create_polygon"], "Q")
        self.assertEqual(win._config["shortcuts"]["create_rectangle"], "W")
        self.assertEqual(win._config["shortcuts"]["create_circle"], "E")
        self.assertTrue(callable(win.canvas.on_right_click))  # 右键撤销回调已注入
        # 模拟右键撤销：先制造一次编辑，再触发回调
        json_path = self.det_dir / "img00.json"
        original = json_path.read_text(encoding="utf-8")
        try:
            _append_unlabeled_rect(win)
            win.canvas.setLastLabel("bird", {})
            win.addLabel(win.canvas.shapes[-1])
            win.setDirty()
            self.assertEqual(len(win.canvas.shapes), 2)
            win.canvas.on_right_click()
            self.assertEqual(len(win.canvas.shapes), 1)  # 上一步编辑已撤销
        finally:
            json_path.write_text(original, encoding="utf-8")
            (self.det_dir / "classes.txt").write_text("cat\ndog\n", encoding="utf-8")

    def test_pan_press_hold_release(self):
        """平移必须按住才拖、松开即停（回归：此前松开不复位导致永远拖动）。"""
        from PyQt5.QtCore import QEvent, QPointF, Qt
        from PyQt5.QtGui import QMouseEvent

        win = _make_window(self.det_dir)
        canvas = win.canvas

        def mouse(t, button, buttons):
            return QMouseEvent(t, QPointF(3, 3), button, buttons, Qt.NoModifier)

        canvas.mousePressEvent(mouse(QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton))
        self.assertTrue(canvas._panning)  # 编辑模式空白处左键 → 进入平移
        canvas.mouseMoveEvent(mouse(QEvent.MouseMove, Qt.NoButton, Qt.LeftButton))
        self.assertTrue(canvas._panning)  # 按住移动 → 保持
        canvas.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton))
        self.assertFalse(canvas._panning)  # 松开 → 结束
        canvas.mouseMoveEvent(mouse(QEvent.MouseMove, Qt.NoButton, Qt.NoButton))
        self.assertFalse(canvas._panning)  # 之后移动视图不再漂移

    def test_right_click_undo_while_drawing(self):
        """绘制中右键撤销上一个点；点删完自动取消本形状。"""
        from PyQt5.QtCore import QEvent, QPointF, Qt
        from PyQt5.QtGui import QMouseEvent

        win = _make_window(self.det_dir)
        canvas = win.canvas
        canvas.createMode = "polygon"
        canvas.setEditing(False)

        def move(x, y):
            canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(x, y),
                                              Qt.NoButton, Qt.NoButton, Qt.NoModifier))

        def press(x, y, button, buttons):
            canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(x, y),
                                               button, buttons, Qt.NoModifier))

        move(50, 50)
        press(50, 50, Qt.LeftButton, Qt.LeftButton)
        self.assertIsNotNone(canvas.current)  # 开始绘制
        move(70, 55)
        press(70, 55, Qt.LeftButton, Qt.LeftButton)
        self.assertEqual(len(canvas.current.points), 2)  # 不同位置不会闭合
        press(70, 55, Qt.RightButton, Qt.RightButton)
        self.assertEqual(len(canvas.current.points), 1)  # 右键撤销上一个点
        press(70, 55, Qt.RightButton, Qt.RightButton)
        self.assertIsNone(canvas.current)  # 点删完 → 取消本形状

    def test_exit_drawing_state(self):
        """无标注点时右键/ESC 必须退出标注状态（回归：此前只能切换模式退出）。"""
        from PyQt5.QtCore import QEvent, QPointF, Qt
        from PyQt5.QtGui import QKeyEvent, QMouseEvent

        win = _make_window(self.det_dir)
        canvas = win.canvas
        canvas.createMode = "polygon"
        canvas.setEditing(False)
        self.assertTrue(canvas.drawing())

        # 1) 无任何点时右键 → 退出标注状态
        canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(50, 50),
                                           Qt.RightButton, Qt.RightButton, Qt.NoModifier))
        self.assertFalse(canvas.drawing())

        # 2) 放一个点后右键 → 删点并仍留在标注状态；再右键一次才退出
        canvas.createMode = "polygon"
        canvas.setEditing(False)
        canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(50, 50),
                                          Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(50, 50),
                                           Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertIsNotNone(canvas.current)
        canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(60, 60),
                                           Qt.RightButton, Qt.RightButton, Qt.NoModifier))
        self.assertIsNone(canvas.current)
        self.assertTrue(canvas.drawing())  # 删光后仍在标注状态
        self.assertTrue(canvas.editing() is False)
        canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(60, 60),
                                           Qt.RightButton, Qt.RightButton, Qt.NoModifier))
        self.assertFalse(canvas.drawing())  # 再点一次 → 退出标注状态

        # 3) 无点时 ESC → 退出标注状态
        canvas.createMode = "polygon"
        canvas.setEditing(False)
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                       Qt.KeyboardModifier.NoModifier))
        self.assertFalse(canvas.drawing())

        # 4) 有点时 ESC → 取消形状并退出
        canvas.createMode = "polygon"
        canvas.setEditing(False)
        canvas.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(50, 50),
                                          Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(50, 50),
                                           Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        canvas.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape,
                                       Qt.KeyboardModifier.NoModifier))
        self.assertIsNone(canvas.current)
        self.assertFalse(canvas.drawing())

    def test_reenter_drawing_mode_after_exit(self):
        """回归：右键退出标注状态后，再次按 Q 必须能重新进入（动作不能保持禁用）。"""
        from PyQt5.QtCore import QEvent, QPointF, Qt
        from PyQt5.QtGui import QMouseEvent

        win = _make_window(self.det_dir)
        win.toggleDrawMode(False, createMode="polygon")  # 按 Q
        self.assertTrue(win.canvas.drawing())
        self.assertFalse(win.actions.createMode.isEnabled())  # 进入后自身动作禁用（原生行为）

        win.canvas.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(50, 50),
                                               Qt.RightButton, Qt.RightButton, Qt.NoModifier))
        self.assertFalse(win.canvas.drawing())  # 右键退出
        self.assertTrue(win.actions.createMode.isEnabled())  # 修复点：动作已恢复可用

        win.toggleDrawMode(False, createMode="polygon")  # 再按 Q
        self.assertTrue(win.canvas.drawing())  # 能重新进入标注模式

    def test_language_setting_and_translator(self):
        """语言设置持久化 + 合并翻译层（官方 qm 与补充映射）。"""
        from labelme.language import MergedZhTranslator, current_language, set_language

        set_language("zh_CN")
        self.assertEqual(current_language(), "zh_CN")
        translator = MergedZhTranslator()
        self.assertEqual(translator.translate("MainWindow", "Open Dir"), "打开目录")
        self.assertEqual(translator.translate("MainWindow", "Save"), "保存")  # 补充层
        self.assertEqual(translator.translate("MainWindow", "Ctrl+Wheel"),
                         "Ctrl+滚轮")  # 官方层译文
        win = _make_window(self.det_dir)  # 默认中文 → 构建时自动安装翻译
        from PyQt5.QtCore import QCoreApplication
        self.assertEqual(QCoreApplication.translate("MainWindow", "Open Dir"), "打开目录")

    def test_tree_selection_and_undo_state(self):
        win = _make_window(self.det_dir)
        self.assertEqual(len(win.canvas.shapes), 1)
        self.assertFalse(win.image.isNull())  # 图片已真实加载（含已有 json 的图片）

    def test_new_shape_autosaves_and_syncs_classes(self):
        """模拟真实流程：画形状 -> 命名 -> setDirty（auto_save 直接落盘）。"""
        win = _make_window(self.det_dir)
        json_path = self.det_dir / "img00.json"
        original_json = json_path.read_text(encoding="utf-8")
        original_classes = (self.det_dir / "classes.txt").read_text(encoding="utf-8")
        try:
            _append_unlabeled_rect(win)
            win.canvas.setLastLabel("bird", {})
            win.addLabel(win.canvas.shapes[-1])
            win.setDirty()

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["shapes"][-1]["label"], "bird")
            self.assertIsNone(data["imageData"])  # store_data=false，json 轻量
            self.assertEqual(data["shapes"][0]["label"], "cat")  # 原标注保留
            classes = (self.det_dir / "classes.txt").read_text(encoding="utf-8").split()
            self.assertIn("bird", classes)
            self.assertIn("cat", classes)
        finally:
            json_path.write_text(original_json, encoding="utf-8")
            (self.det_dir / "classes.txt").write_text(original_classes, encoding="utf-8")

    def test_archive_to_class(self):
        win = _make_window(self.cls_dir)
        target = Path(win.filename)
        win._archive_to_class_dir("dog")
        moved = self.cls_dir / "dog" / target.name
        self.assertTrue(moved.exists())
        self.assertFalse(target.exists())
        self.assertEqual(Path(win.filename), moved)  # 窗口已定位到移动后的文件
        self.assertEqual(len(win.imageList), 6)  # 列表已刷新

    def test_engine_reads_labelme_output(self):
        """引擎 reader 必须能读取本标注平台的产物。"""
        from xccv.engine.dataset.reader import scan_labelme_dataset

        win = _make_window(self.det_dir)
        samples, classes, from_file = scan_labelme_dataset(self.det_dir)
        self.assertEqual(len(samples), 4)
        self.assertEqual(classes, ["cat", "dog"])
        self.assertTrue(from_file)


if __name__ == "__main__":
    unittest.main()

