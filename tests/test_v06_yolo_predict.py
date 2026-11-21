"""YOLO-seg 预测掩码必须还原原图尺寸 + 可视化防御双保险。

背景：ultralytics YOLO-seg 返回的 masks
停在网络输入分辨率（letterbox 后 imgsz，如 1216），未还原原图（如 2144），
`render.draw_predictions` 内 `overlay[mask] = color` 抛 IndexError 整窗崩溃。

修复三层：
1. engine/models/yolo.py `_masks_to_original`：LetterBox 逆变换还原原图；
   返回 **list[(H,W) bool]**（Prediction.masks 契约；返回 3 维 ndarray 会让
   消费方 `if pred.masks:` 真值歧义抛 ValueError——实测反馈"YOLO 无法正常
   推理"；旧测试只断言 ndarray 形状且随机图零检出，把错契约锁死了）；
2. testing/render.py `_align_mask`：形状不符先尝试对齐，仍失败跳过+留痕；
3. testing_window._show_result：可视化异常降级为原图+状态提示，不再未捕获。
"""

import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch

from xccv.engine.models.base import Prediction
from xccv.engine.models.yolo import _masks_to_original
from xccv.testing.render import _align_mask, draw_predictions

_ROOT = Path(__file__).resolve().parents[1]
_SEG_WEIGHTS = _ROOT / "xccv" / "engine" / "weights" / "ultralytics" / "yolo11n-seg.pt"


class _FakeRes:
    """模拟 ultralytics Results：只暴露 masks.data（网络输入分辨率）。"""

    def __init__(self, masks):
        self.masks = mock.Mock()
        self.masks.data = masks


class MasksToOriginalTest(unittest.TestCase):
    def test_same_size_passthrough(self):
        data = (torch.rand(2, 32, 48) > 0.5).float()
        res = _FakeRes(data)
        out = _masks_to_original(res, (32, 48))
        self.assertIsInstance(out, list)  # Prediction.masks 契约
        self.assertEqual(len(out), 2)
        for m in out:
            self.assertEqual(m.shape, (32, 48))
            self.assertEqual(m.dtype, np.dtype(bool))

    def test_letterbox_restore(self):
        """网络输入 64x80，原图 32x64（纵向 letterbox）：输出必须等于原图尺寸。"""
        data = torch.zeros(1, 64, 80)
        data[0, 16:48, 8:72] = 1.0  # 中心区域（去 padding 后恰为原图范围）
        res = _FakeRes(data)
        out = _masks_to_original(res, (32, 64))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (32, 64))
        # 中心区域映射回原图内部：掩码应非空且不超过原图范围
        self.assertTrue(out[0].any())

    def test_output_dtype_bool_2d(self):
        data = torch.rand(3, 20, 20)
        out = _masks_to_original(_FakeRes(data), (40, 40))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)
        for m in out:
            self.assertEqual(m.ndim, 2)
            self.assertEqual(m.dtype, np.dtype(bool))


class AlignMaskDefenseTest(unittest.TestCase):
    """render._align_mask：双保险防御。"""

    def test_matching_shape_passthrough(self):
        mask = np.zeros((10, 12), bool)
        mask[2:5, 3:8] = True
        out = _align_mask(mask, (10, 12))
        self.assertTrue(out.dtype == np.dtype(bool))
        self.assertEqual(int(out.sum()), 15)

    def test_mismatch_aligned_by_resize(self):
        mask = np.zeros((16, 16), np.uint8)
        mask[4:12, 4:12] = 1
        out = _align_mask(mask, (32, 32))
        self.assertEqual(out.shape, (32, 32))
        self.assertTrue(out.any())

    def test_invalid_mask_returns_none(self):
        self.assertIsNone(_align_mask(np.zeros((0,), bool), (8, 8)))
        self.assertIsNone(_align_mask(np.zeros((2, 3, 4)), (8, 8)))


class DrawPredictionsDefenseTest(unittest.TestCase):
    """draw_predictions 对任何形状不符的掩码不得抛未捕获异常（双保险）。"""

    def test_wrong_size_mask_no_crash(self):
        img = np.zeros((64, 64, 3), np.uint8)
        mask = np.zeros((32, 32), bool)   # 与图像不符（模拟未还原的旧链路）
        mask[8:24, 8:24] = True
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[10, 10, 30, 30]],
                          masks=[mask])
        out = draw_predictions(img, pred, ["a"], "segmentation")
        self.assertEqual(out.shape, img.shape)

    def test_garbage_mask_skipped(self):
        img = np.zeros((48, 48, 3), np.uint8)
        pred = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[5, 5, 20, 20], [22, 22, 40, 40]],
                          masks=[np.zeros((0,), bool), np.zeros((2, 3, 4))])
        out = draw_predictions(img, pred, ["a", "b"], "segmentation")
        self.assertEqual(out.shape, img.shape)

    def test_ndarray_masks_no_truth_ambiguity(self):
        """上游误给 3 维 ndarray（N>1）时不得因 `if masks:` 真值歧义抛错。"""
        img = np.zeros((40, 40, 3), np.uint8)
        masks = np.zeros((2, 40, 40), bool)
        masks[0, 2:10, 2:10] = True
        masks[1, 20:30, 20:30] = True
        pred = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[2, 2, 10, 10], [20, 20, 30, 30]], masks=masks)
        out = draw_predictions(img, pred, ["a", "b"], "segmentation")
        self.assertGreater(float(out[22:28, 22:28].mean()), 0)

    def test_legacy_adapter_image_size_ndarray_masks(self):
        from xccv.engine.flow.legacy_adapter import _image_size

        pred = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[0, 0, 4, 4], [1, 1, 5, 5]],
                          masks=np.zeros((2, 30, 50), bool))
        self.assertEqual(_image_size(pred, None), (50, 30))

    def test_correct_mask_still_rendered(self):
        img = np.zeros((64, 64, 3), np.uint8)
        mask = np.zeros((64, 64), bool)
        mask[10:30, 10:30] = True
        pred = Prediction(labels=[1], scores=[0.8], boxes=[[10, 10, 30, 30]],
                          masks=[mask])
        out = draw_predictions(img, pred, ["a", "b"], "segmentation")
        self.assertGreater(float(out[15:25, 15:25].mean()), 0)


@unittest.skipUnless(_SEG_WEIGHTS.is_file(), "缺少 yolo11n-seg 预训练权重")
class RealYoloSegSmokeTest(unittest.TestCase):
    """真实权重 CPU 冒烟：加载→推理→掩码形状==原图→渲染不抛错。"""

    def test_predict_masks_match_original(self):
        from xccv.engine.models.yolo import YoloModel
        from xccv.engine.dataset.base import TaskType

        model = YoloModel("yolo11n", TaskType.SEGMENTATION, 1,
                          device="cpu", use_pretrained=False)
        model.load_weights(_SEG_WEIGHTS)
        rng = np.random.default_rng(3)
        img = rng.integers(0, 255, size=(2144, 1216 // 2, 3), dtype=np.uint8)
        pred = model.predict(img, conf=0.2)
        if pred.masks is not None:  # 随机图可能零检出（真检出见 RealTrainedYoloSegTest）
            for m in pred.masks:
                self.assertEqual(m.shape, img.shape[:2])
        from xccv.testing.render import draw_predictions

        out = draw_predictions(img, pred, ["a"], "segmentation")
        self.assertEqual(out.shape, img.shape)


_TRAINED_SEG = (_ROOT / "traintest" / "seg" / "模型" / "20260916_140728_yolo11n-seg"
                / "best.pt")
_D1 = _ROOT / "traintest" / "seg" / "数据集" / "D1"


@unittest.skipUnless(_TRAINED_SEG.is_file() and _D1.is_dir(),
                     "缺少用户训练的 yolo11n-seg 模型或 D1 数据集")
class RealTrainedYoloSegTest(unittest.TestCase):
    """实测反馈场景复现：自训 yolo11n-seg + D1 真图，**N>0 真检出**
    → masks 为 list 且逐个等于原图尺寸 → 渲染与测试平台结果展示均不报错。"""

    @classmethod
    def setUpClass(cls):
        from xccv.engine.io_utils import imread_unicode
        from xccv.engine.models import load_model

        cls.model = load_model(_TRAINED_SEG)
        img_path = sorted(_D1.rglob("*.jpg"))[0]
        cls.img = imread_unicode(str(img_path))
        cls.pred = cls.model.predict(cls.img, conf=0.25, iou=0.5)

    def test_real_detections_masks_contract(self):
        self.assertGreater(len(self.pred.boxes), 0, "应有真检出（复现前提）")
        self.assertIsInstance(self.pred.masks, list)
        self.assertEqual(len(self.pred.masks), len(self.pred.boxes))
        for m in self.pred.masks:
            self.assertEqual(m.shape, self.img.shape[:2])
            self.assertEqual(m.dtype, np.dtype(bool))

    def test_real_detections_render_and_show(self):
        out = draw_predictions(self.img, self.pred, list(self.model.classes),
                               "segmentation")
        self.assertEqual(out.shape, self.img.shape)
        from PyQt5.QtWidgets import QApplication

        from xccv.testing.testing_window import TestingWindow

        _ = QApplication.instance() or QApplication([])
        win = TestingWindow()
        try:
            win.classes = list(self.model.classes)
            win.task = "segmentation"
            win._show_result("D1.jpg", self.img, self.pred)
            self.assertNotIn("可视化失败", win.result_stats.text())
        finally:
            win.close()
            win.deleteLater()


class ShowResultNarrowingTest(unittest.TestCase):
    """_show_result 渲染异常降级：状态栏提示+不抛未捕获（crash 面收窄）。"""

    def test_render_failure_degrades(self):
        from PyQt5.QtWidgets import QApplication

        from xccv.testing.testing_window import TestingWindow

        _ = QApplication.instance() or QApplication([])
        win = TestingWindow()
        try:
            img = np.zeros((32, 32, 3), np.uint8)
            pred = Prediction(labels=[0], scores=[0.9], boxes=[[1, 1, 8, 8]])
            with mock.patch("xccv.testing.render.draw_predictions",
                            side_effect=RuntimeError("boom")):
                win._show_result("t.png", img, pred)  # 不应抛
            self.assertIn("可视化失败", win.result_stats.text())
        finally:
            win.close()
            win.deleteLater()


if __name__ == "__main__":
    unittest.main()

