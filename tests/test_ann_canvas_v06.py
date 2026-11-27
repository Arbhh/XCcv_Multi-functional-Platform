"""标注平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v06_annotation
- test_v06_annotation_b1
- test_v06_annotation_b2
"""


# == 来源模块：test_v06_annotation ==
# -*- coding: utf-8 -*-
"""V0.6 Phase 5 标注平台大改测试：JSON 瘦身 / AI 引擎契约 / 智能多边形与画笔画布 / 批量预标注。

权重相关的引擎实测冒烟标注 `@unittest.skipUnless(weights_ready(), ...)`（有权重才跑）。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.models.base import Prediction

REPO = Path(__file__).resolve().parents[1]


def weights_ready() -> bool:
    from xccv.annotation import ai_assist

    return any(m["available"] and Path(m["weights"]).stat().st_size > 1024 * 1024
               for m in ai_assist.available_models())


class JsonSlimTest(unittest.TestCase):
    """JSON 瘦身——默认不内嵌 imageData；旧 JSON（外链/内嵌）持续可载。"""

    def test_default_config_store_data_off(self):
        from labelme.config import get_default_config

        cfg = get_default_config()
        self.assertFalse(cfg["store_data"], "平台默认必须不内嵌 imageData")

    def test_save_without_imagedata_small_and_loadable(self):
        from labelme.label_file import LabelFile

        with tempfile.TemporaryDirectory() as td:
            img_path = Path(td) / "图.png"
            imwrite_unicode(img_path, np.full((60, 80, 3), 128, np.uint8))
            out = Path(td) / "图.json"
            lf = LabelFile()
            lf.save(str(out), shapes=[
                {"label": "划痕", "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                 "group_id": None, "shape_type": "polygon", "flags": {},
                 "description": ""}],
                imagePath="图.png", imageHeight=60, imageWidth=80,
                imageData=None)   # store_data=False 时 app 传 None
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIsNone(data["imageData"])
            self.assertLess(out.stat().st_size, 20 * 1024, "JSON 应远小于 20KB")
            back = LabelFile(str(out))   # 外链 imagePath 回读
            self.assertEqual(back.shapes[0]["label"], "划痕")
            self.assertEqual(back.imageData is not None, True)

    def test_old_json_with_imagedata_still_loads(self):
        import base64

        from labelme.label_file import LabelFile

        with tempfile.TemporaryDirectory() as td:
            img = np.full((32, 40, 3), 77, np.uint8)
            img_path = Path(td) / "旧.png"
            imwrite_unicode(img_path, img)
            import io as _io

            from PIL import Image

            buf = _io.BytesIO()
            Image.fromarray(img[..., ::-1]).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            data = {"version": "5.3.1", "flags": {},
                    "shapes": [{"label": "x", "points": [[0, 0], [5, 5]],
                                "group_id": None, "shape_type": "rectangle",
                                "flags": {}}],
                    "imagePath": "旧.png", "imageData": b64,
                    "imageHeight": 32, "imageWidth": 40}
            jp = Path(td) / "旧.json"
            jp.write_text(json.dumps(data), encoding="utf-8")
            lf = LabelFile(str(jp))
            self.assertEqual(len(lf.shapes), 1)


class AiAssistContractTest(unittest.TestCase):
    """引擎契约（注册表/可用性指引/请求响应形状；不需要权重）。"""

    def test_available_models_structure(self):
        from xccv.annotation import ai_assist

        models = ai_assist.available_models()
        names = {m["name"] for m in models}
        self.assertIn("Sam2 (balanced)", names)
        self.assertIn("MobileSAM (fast)", names)
        for m in models:
            self.assertIn("available", m)
            if not m["available"]:
                self.assertIn("权重", m["message"])

    def test_missing_weights_get_model_raises_with_hint(self):
        """缺权重档位仍给中文指引：运行时接管下载失败（无网替身）后
        回到原"权重缺失"指引路径，不真实联网。"""
        from unittest import mock as _mock

        from xccv.annotation import ai_assist
        from xccv.engine.weights import ensure_family_weights

        models = ai_assist.available_models()
        missing = [m["name"] for m in models if not m["available"]]
        if not missing:
            self.skipTest("权重已就绪（跳过缺权重指引用例）")

        def _boom(filenames, notify=None, fetcher=None):
            raise OSError("no network")

        with _mock.patch.object(ensure_family_weights,
                                "ensure_sam_missing_files", _boom):
            with self.assertRaises(ValueError) as ctx:
                ai_assist.get_model_type_by_name(missing[0])
        self.assertIn("权重", str(ctx.exception))

    def test_generate_without_embedding_raises(self):
        from xccv.annotation.ai_assist import GenerateRequest, generate

        with self.assertRaises(ValueError):
            generate(GenerateRequest(model="Sam2 (balanced)"))

    @unittest.skipUnless(os.environ.get("XCCV_SAM_READY") == "1"
                         and weights_ready(),
                         "需要 SAM 权重完整落盘（下载完成后设 XCCV_SAM_READY=1）")
    def test_decode_on_real_photo(self):
        """真权重冒烟：真实照片点提示 → 掩码覆盖点击点、占比合理。

        注：不用合成黑底图——SAM2 对无纹理合成图会把整张"照片"当对象（固有
        行为），真实照片（ultralytics 自带 bus.jpg）才是几何正确性的基准。

        起默认开「小目标放大重解」（场景：小目标完全无法标注）——单击
        优先给首锚附近的局部结构；整车用经典模式（zoom 关）或两击（远处
        第二正锚跳过窗内重解）选中。三种语义分别锁定。
        """
        import ultralytics

        import cv2

        from xccv.annotation import ai_assist
        from xccv.engine.io_utils import imread_unicode

        bus = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        img = imread_unicode(str(bus))
        self.assertIsNotNone(img)
        H, W = img.shape[:2]

        def frac(a):
            return float(a.mask.sum()) / (H * W)

        old = ai_assist.get_params()
        self.addCleanup(ai_assist.set_params, zoom=old["zoom"])
        ai_assist.set_params(zoom=False)                 # 经典：单击 = 整车
        ann = ai_assist.decode_points(
            "Sam2 (balanced)", img, [[500.0, 380.0]], [1], multimask=True)
        self.assertIsNotNone(ann)
        self.assertTrue(ann.mask[380, 500], "掩码须覆盖点击点")
        self.assertGreater(frac(ann), 0.03)
        self.assertLess(frac(ann), 0.80)
        self.assertGreater(ann.score, 0.5)

        ai_assist.set_params(zoom=True)                  # 默认：单击 = 局部
        one = ai_assist.decode_points(
            "Sam2 (balanced)", img, [[500.0, 380.0]], [1], multimask=True)
        self.assertTrue(one.mask[380, 500], "掩码须覆盖点击点")
        self.assertLess(frac(one), frac(ann))
        two = ai_assist.decode_points(                   # 两击（远处第二正锚）
            "Sam2 (balanced)", img, [[500.0, 380.0], [140.0, 260.0]], [1, 1],
            multimask=True)
        self.assertTrue(two.mask[380, 500] and two.mask[260, 140])
        self.assertGreater(frac(two), 0.03)
        self.assertLess(frac(two), 0.80)
        _ = cv2  # noqa: F841（保持 import 对齐文档示例）


class CanvasAiBrushTest(unittest.TestCase):
    """画布模式注册与状态机（不依赖权重）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _canvas(self):
        from labelme.widgets.canvas import Canvas

        c = Canvas()
        from PyQt5.QtGui import QPixmap

        c.pixmap = QPixmap(80, 60)
        c.pixmap.fill()
        c.resize(80, 60)   # 与图同尺寸：未显示部件的 offsetToCenter 才是 0（可点准）
        return c

    def test_create_mode_accepts_ai_and_brush(self):
        c = self._canvas()
        c.createMode = "ai_polygon"
        self.assertEqual(c.createMode, "ai_polygon")
        c.createMode = "brush"
        self.assertEqual(c.createMode, "brush")
        with self.assertRaises(ValueError):
            c.createMode = "nope"

    def test_ai_prompt_state_and_undo_without_model(self):
        """无模型（_sam None）时点击不崩：提示点入栈、右键撤销退出。"""
        from PyQt5 import QtCore

        c = self._canvas()
        c.createMode = "ai_polygon"
        c.setEditing(False)   # drawing 状态
        ev_press = self._press(QtCore.Qt.LeftButton, QtCore.QPoint(20, 20))
        c.mousePressEvent(ev_press)
        self.assertEqual(len(c._ai_prompts), 1)
        self.assertIsNotNone(c.current)
        ev_undo = self._press(QtCore.Qt.RightButton, QtCore.QPoint(20, 20))
        c.mousePressEvent(ev_undo)   # 右键撤销最后一个锚点
        self.assertEqual(len(c._ai_prompts), 0)
        self.assertIsNone(c.current)

    def test_brush_width_clamp_and_wheel(self):
        from PyQt5 import QtCore
        from PyQt5.QtCore import Qt, QPoint, QPointF
        from PyQt5.QtGui import QWheelEvent

        c = self._canvas()
        c.createMode = "brush"
        c.setBrushWidth(999)
        self.assertEqual(c._brush_width, 256)
        c.setBrushWidth(0)
        self.assertEqual(c._brush_width, 2)
        ev = QWheelEvent(QPointF(10, 10), QPointF(10, 10),
                         QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                         Qt.ScrollUpdate, False)
        c.drawing = lambda: True
        c.current = object()   # 非 None 即视为绘制中
        seen = []
        c.zoomRequest.connect(lambda d, p: seen.append(d))
        c.wheelEvent(ev)
        self.assertEqual(c._brush_width, 2)   # 滚轮不再调粗细
        self.assertEqual(seen, [120])         # 滚轮恢复图像缩放

    @staticmethod
    def _press(button, pos):
        from PyQt5 import QtCore
        from PyQt5.QtGui import QMouseEvent
        from PyQt5.QtCore import QEvent, QPointF

        return QMouseEvent(QEvent.MouseButtonPress, QPointF(pos), button, button,
                           QtCore.Qt.NoModifier)

    def test_brush_finalise_makes_polygon(self):
        """画笔轨迹 → 松开成 polygon 形状（合成掩膜路径）。"""
        from PyQt5 import QtCore

        c = self._canvas()
        c.createMode = "brush"
        c.setEditing(False)
        c.current = None
        # 模拟一笔：起笔→轨迹→finalise
        c.current = None
        c._brush_pts = [QtCore.QPointF(10, 10), QtCore.QPointF(40, 10),
                        QtCore.QPointF(40, 40)]
        from labelme.shape import Shape

        c.current = Shape(shape_type="points")
        c._brush_finalise()
        self.assertIsNone(c.current)
        self.assertEqual(len(c.shapes), 1)
        shape = c.shapes[0]
        self.assertEqual(shape.shape_type, "polygon")
        self.assertGreaterEqual(len(shape.points), 4)


class AutoAnnotateTest(unittest.TestCase):
    """掩码转多边形 / Prediction→shapes / JSON 可被 labelme 打开。"""

    def test_mask_to_polygons(self):
        from xccv.annotation.auto_annotate import mask_to_polygons

        m = np.zeros((50, 60), bool)
        m[10:30, 20:40] = True
        pts = mask_to_polygons(m)
        self.assertGreaterEqual(len(pts), 3)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.assertAlmostEqual(min(xs), 20.0, delta=1.5)
        self.assertAlmostEqual(max(xs), 39.0, delta=1.5)

    def test_prediction_to_shapes_rectangle_and_polygon(self):
        from xccv.annotation.auto_annotate import prediction_to_shapes

        m = np.zeros((40, 40), bool)
        m[5:20, 5:20] = True
        pred = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[0, 0, 10, 10], [5, 5, 20, 20]],
                          masks=[m, None])
        shapes = prediction_to_shapes(pred, ["划痕", "脏污"])
        self.assertEqual(shapes[0]["shape_type"], "polygon")   # 有掩码 → 多边形
        self.assertEqual(shapes[0]["label"], "划痕")
        self.assertEqual(shapes[1]["shape_type"], "rectangle")  # 无掩码 → 框
        self.assertEqual(shapes[1]["label"], "脏污")

    def test_write_json_loadable_by_labelme(self):
        from xccv.annotation.auto_annotate import write_labelme_json

        with tempfile.TemporaryDirectory() as td:
            img = np.full((30, 50, 3), 90, np.uint8)
            img_path = Path(td) / "样本.png"
            imwrite_unicode(img_path, img)
            jp = Path(td) / "样本.json"
            write_labelme_json(jp, img_path, [
                {"label": "A", "points": [[1, 1], [9, 1], [9, 9]],
                 "group_id": None, "shape_type": "polygon", "flags": {},
                 "description": "", "score": 0.9}])
            data = json.loads(jp.read_text(encoding="utf-8"))
            self.assertIsNone(data["imageData"])
            self.assertEqual(data["imageWidth"], 50)
            from labelme.label_file import LabelFile

            lf = LabelFile(str(jp))
            self.assertEqual(lf.shapes[0]["label"], "A")

    def test_annotate_folder_with_mock_model(self):
        """批量流程（mock 模型）：跳过已有 JSON、逐图产出、摘要计数正确。"""
        from unittest import mock

        from xccv.annotation import auto_annotate

        with tempfile.TemporaryDirectory() as td:
            for i in range(3):
                imwrite_unicode(Path(td) / f"图{i}.png",
                                np.full((32, 32, 3), 60, np.uint8))
            (Path(td) / "图1.json").write_text("{}", encoding="utf-8")

            class _Task:
                value = "segmentation"

            class _FakeModel:
                classes = ["目标"]
                task = _Task()

                def predict(self, img, conf=0.25, iou=0.45):
                    m = np.zeros(img.shape[:2], bool)
                    m[8:24, 8:24] = True
                    return Prediction(labels=[0], scores=[0.9],
                                      boxes=[[8, 8, 24, 24]], masks=[m])

            def _load(path, device="auto"):
                return _FakeModel()

            with mock.patch("xccv.engine.models.registry.load_model",
                            side_effect=_load):
                summary = auto_annotate.annotate_folder(
                    "fake.pt", td, skip_annotated=True)
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(summary["annotated"], 2)
            self.assertEqual(summary["failed"], 0)
            data = json.loads((Path(td) / "图0.json").read_text(encoding="utf-8"))
            self.assertEqual(data["shapes"][0]["label"], "目标")




class T56InteractionTest(unittest.TestCase):
    """交互精选（路径记忆/标签显隐 C/灰度 G/直线 I）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_line_shortcut_registered(self):
        from labelme.config import get_default_config

        self.assertEqual(get_default_config()["shortcuts"]["create_line"], "I")

    def test_last_open_dir_roundtrip(self):
        from xccv.ui_widgets import load_prefs, save_prefs

        save_prefs({"annotation_last_dir": "E:/某目录"})
        self.assertEqual(load_prefs().get("annotation_last_dir"), "E:/某目录")
        save_prefs({"annotation_last_dir": ""})   # 清理（不影响其它键）

    def test_canvas_label_visibility_toggle(self):
        from PyQt5.QtGui import QPixmap

        from labelme.widgets.canvas import Canvas

        c = Canvas()
        c.pixmap = QPixmap(60, 40)
        c.pixmap.fill()
        self.assertTrue(c._show_labels)
        c.setLabelsVisible(False)
        self.assertFalse(c._show_labels)
        c.setLabelsVisible(True)
        self.assertTrue(c._show_labels)

    def test_app_grayscale_and_label_actions(self):
        """MainWindow：灰度开关切换后画布像素变灰；标签显隐动作联动画布。"""
        from PyQt5 import QtCore, QtGui

        from labelme.app import MainWindow
        from labelme.config import get_default_config
        from tests.synthetic import make_labelme_dataset
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            det, _classes = make_labelme_dataset(Path(td) / "det", n_images=1)
            win = MainWindow(config=get_default_config())
            img_path = next(Path(det).glob("*.png")) or next(Path(det).glob("*.jpg"))
            win.loadFile(str(img_path))
            self.addCleanup(win.close)
            pm_before = win.canvas.pixmap.toImage()
            win.actions.grayscaleDisplay.setChecked(True)
            win.toggleGrayscale()
            pm_after = win.canvas.pixmap.toImage()
            c_after = pm_after.pixelColor(1, 1)
            self.assertEqual(c_after.red(), c_after.green())
            self.assertEqual(c_after.green(), c_after.blue())   # 灰度：RGB 相等
            # 关闭恢复彩色
            win.actions.grayscaleDisplay.setChecked(False)
            win.toggleGrayscale()
            c_back = win.canvas.pixmap.toImage().pixelColor(1, 1)
            c_orig = pm_before.pixelColor(1, 1)
            self.assertEqual((c_back.red(), c_back.green(), c_back.blue()),
                             (c_orig.red(), c_orig.green(), c_orig.blue()))
            # 标签显隐
            win.actions.showLabelTexts.setChecked(False)
            win.toggleLabelTexts()
            self.assertFalse(win.canvas._show_labels)
            win.actions.showLabelTexts.setChecked(True)
            win.toggleLabelTexts()
            self.assertTrue(win.canvas._show_labels)
            _ = QtCore  # noqa: F841


class BigImagePerfTest(unittest.TestCase):
    """大图三件套核验（4K 打开 / 缩放 / 建形状计时，断言宽松上限）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_4k_open_zoom_shape(self):
        import tempfile
        import time

        import numpy as np
        from PyQt5 import QtCore

        from labelme.app import MainWindow
        from labelme.config import get_default_config

        with tempfile.TemporaryDirectory() as td:
            img = np.zeros((2160, 3840, 3), np.uint8)
            img[800:1300, 1500:2400] = 255
            from xccv.engine.io_utils import imwrite_unicode

            p = Path(td) / "4k.png"
            imwrite_unicode(p, img)
            win = MainWindow(config=get_default_config())
            self.addCleanup(win.close)
            t0 = time.perf_counter()
            win.loadFile(str(p))
            t_open = time.perf_counter() - t0
            self.assertTrue(win.canvas.pixmap is not None)
            self.assertLess(t_open, 5.0, f"4K 打开耗时 {t_open:.2f}s 超上限")
            # 缩放（paint 一次计时）
            t0 = time.perf_counter()
            win.canvas.scale = 0.5
            win.canvas.repaint()
            t_zoom = time.perf_counter() - t0
            self.assertLess(t_zoom, 1.5, f"4K 重绘 {t_zoom:.2f}s 超上限")
            # 建形状（矩形两击）
            win.canvas.createMode = "rectangle"
            win.canvas.setEditing(False)

            def press(x, y):
                from PyQt5.QtCore import QEvent, QPointF, Qt
                from PyQt5.QtGui import QMouseEvent

                win.canvas.mousePressEvent(QMouseEvent(
                    QEvent.MouseButtonPress, QPointF(float(x), float(y)),
                    Qt.LeftButton, Qt.LeftButton, QtCore.Qt.NoModifier))

            press(1500, 800)
            t_shape = time.perf_counter() - t0
            self.assertIsNotNone(win.canvas.current)   # 开始绘制（不闭合——
            #  第二击会 finalise→标签对话框，offscreen 下模态挂起，不在此测）
            self.assertLess(t_shape, 2.0, f"4K 建形状 {t_shape:.2f}s 超上限")
            # 关窗不触发保存确认（防 offscreen 模态挂起）
            win.canvas.current = None
            win.dirty = False
            # 记录留痕
            print(f"\n[大图三件套] 4K 打开 {t_open:.2f}s / 重绘 {t_zoom:.3f}s"
                  f" / 建形状 {t_shape:.2f}s")

# == 来源模块：test_v06_annotation_b1 ==
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




class BrushEraseTest(unittest.TestCase):
    """画笔右键擦除=从掩膜抠除。"""

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

# == 来源模块：test_v06_annotation_b2 ==
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



def _make_canvas__v06_annotation_b2():
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
        cv = _make_canvas__v06_annotation_b2()
        cv.shapes = [_poly(20, 12, 40, 28)]
        n0 = len(cv.shapes[0].points)
        cv._brush_pts = [QtCore.QPointF(30.0, 20.0)]  # 落在原形状内
        cv.finalise()
        self.assertEqual(len(cv.shapes), 1, "应并入原形状而非新建")
        xs = [pt.x() for pt in cv.shapes[0].points]
        self.assertLess(min(xs), 20.0, "并入笔画后 bbox 应外扩")

    def test_stroke_disjoint_creates_new_shape(self):
        cv = _make_canvas__v06_annotation_b2()
        cv.shapes = [_poly(20, 12, 40, 28)]
        cv._brush_pts = [QtCore.QPointF(80.0, 20.0)]  # 与原形状不相交
        cv.finalise()
        self.assertEqual(len(cv.shapes), 2, "不相交笔画应新建形状")


class BrushEraseExistingTest(unittest.TestCase):
    """右键擦除作用于既有形状；纯擦除手势立即提交。"""

    def test_erase_shrinks_existing_shape(self):
        cv = _make_canvas__v06_annotation_b2()
        cv.shapes = [_poly(10, 8, 60, 30)]
        a0 = _area(cv.shapes[0])
        cv._brush_erase_pts = [QtCore.QPointF(35.0, 19.0)]
        cv.finalise()
        self.assertTrue(cv.shapes, "擦中间应保留余块")
        self.assertLess(sum(_area(s) for s in cv.shapes), a0)

    def test_erase_all_deletes_shape(self):
        cv = _make_canvas__v06_annotation_b2()
        cv.shapes = [_poly(20, 12, 40, 28)]
        cv._brush_erase_pts = [QtCore.QPointF(25.0, 20.0),
                               QtCore.QPointF(35.0, 20.0)]
        cv.finalise()
        self.assertEqual(len(cv.shapes), 0, "整块擦除应删除形状")

    def test_right_release_commits_pure_erase(self):
        cv = _make_canvas__v06_annotation_b2()
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
        cv = _make_canvas__v06_annotation_b2()
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

if __name__ == '__main__':
    unittest.main()
