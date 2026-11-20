"""图像增强算子单元测试。"""

import random
import unittest

import cv2
import numpy as np

from xccv.engine.dataset.base import Shape
from xccv.engine.dataset.transforms import (
    AUGMENT_OPS, AUGMENT_PROB, apply_ops, maybe_apply_ops, random_brightness,
    random_contrast, random_rescale, resize,
)


def make_shapes():
    return [Shape(label="cat", shape_type="rectangle", points=[[10.0, 20.0], [30.0, 60.0]])]


class TransformTest(unittest.TestCase):
    def test_resize_stretch_scales_points(self):
        img = np.zeros((100, 50, 3), np.uint8)
        out, shapes = resize(img, make_shapes(), 100, 200, keep_ratio=False)
        self.assertEqual(out.shape, (200, 100, 3))
        self.assertEqual(shapes[0].points, [[20.0, 40.0], [60.0, 120.0]])

    def test_resize_keep_ratio_fits_inside_box(self):
        """keep_ratio：等比缩放恰好装进 宽×高 框，标注点同步等比缩放。"""
        img = np.zeros((100, 50, 3), np.uint8)  # h=100, w=50
        out, shapes = resize(img, make_shapes(), 100, 200, keep_ratio=True)
        h, w = out.shape[:2]
        self.assertLessEqual(w, 100)
        self.assertLessEqual(h, 200)
        self.assertAlmostEqual(w / h, 50 / 100, places=1)  # 长宽比保持
        sx, sy = shapes[0].points[0]
        self.assertAlmostEqual(sx / 10.0, w / 50, places=5)
        self.assertAlmostEqual(sy / 20.0, h / 100, places=5)

    def test_resize_keep_ratio_no_upscale_overflow(self):
        img = np.zeros((2144, 2821, 3), np.uint8)
        out, _ = resize(img, [], 1600, 1600, keep_ratio=True)
        h, w = out.shape[:2]
        self.assertLessEqual(max(w, h), 1600)
        self.assertAlmostEqual(w / h, 2821 / 2144, places=2)

    def test_random_flip_h_mirrors_points(self):
        from xccv.engine.dataset.transforms import random_flip_h

        img = np.zeros((40, 60, 3), np.uint8)
        img[:, 50:] = 255  # 右侧亮块：翻转后应到左侧
        shapes = [Shape(label="cat", shape_type="rectangle",
                        points=[[40.0, 10.0], [55.0, 30.0]])]
        out, flipped = random_flip_h(img, shapes, probability=1.0)
        self.assertTrue(np.array_equal(out, cv2.flip(img, 1)))
        self.assertEqual(flipped[0].points, [[20.0, 10.0], [5.0, 30.0]])

    def test_random_flip_v_mirrors_points(self):
        from xccv.engine.dataset.transforms import random_flip_v

        img = np.zeros((40, 60, 3), np.uint8)
        shapes = [Shape(label="cat", shape_type="rectangle",
                        points=[[5.0, 8.0], [20.0, 35.0]])]
        out, flipped = random_flip_v(img, shapes, probability=1.0)
        self.assertTrue(np.array_equal(out, cv2.flip(img, 0)))
        self.assertEqual(flipped[0].points, [[5.0, 32.0], [20.0, 5.0]])

    def test_random_flip_probability_zero_is_noop(self):
        from xccv.engine.dataset.transforms import random_flip_h

        img = np.zeros((8, 8, 3), np.uint8)
        shapes = make_shapes()
        out, sp = random_flip_h(img, shapes, probability=0.0)
        self.assertIs(out, img)
        self.assertIs(sp, shapes)

    def test_registry_has_flip_ops(self):
        self.assertIn("random_flip_h", AUGMENT_OPS)
        self.assertIn("random_flip_v", AUGMENT_OPS)
        # resize 注册表默认不带 keep_ratio：即函数默认 False（拉伸到 宽×高，
        # 训练/验证/预览所见一致）；等比模式仅显式传参时使用
        self.assertNotIn("keep_ratio", AUGMENT_OPS["resize"][1])
        out, _ = AUGMENT_OPS["resize"][0](np.zeros((8, 8, 3), np.uint8), [], 40, 64)
        self.assertEqual(out.shape[:2], (64, 40))  # (高40→宽, 高64)——非方形精确到设定尺寸（拉伸）

    def test_random_rescale_keeps_points_in_bounds(self):
        img = np.zeros((100, 100, 3), np.uint8)
        out, shapes = random_rescale(img, make_shapes(), min_scale=0.5,
                                     max_scale=1.0, rng=random.Random(0))
        h, w = out.shape[:2]
        self.assertLessEqual(w, 100)
        self.assertGreaterEqual(w, 50)
        for x, y in shapes[0].points:
            self.assertTrue(0 <= x <= w and 0 <= y <= h)

    def test_brightness_keeps_range_and_dtype(self):
        img = np.full((10, 10, 3), 128, np.uint8)
        out, _ = random_brightness(img, [], max_delta=40, rng=random.Random(0))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(int(out.min()), 128 - 40)
        self.assertLessEqual(int(out.max()), 128 + 40)

    def test_contrast_keeps_shape(self):
        img = np.full((10, 10, 3), 128, np.uint8)
        out, _ = random_contrast(img, [], low=0.5, high=2.0, rng=random.Random(0))
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_apply_ops_composes(self):
        img = np.zeros((32, 32, 3), np.uint8)
        out, _ = apply_ops(img, [], {"resize": {"width": 16, "height": 16},
                                     "random_brightness": {}}, rng=random.Random(0))
        self.assertEqual(out.shape, (16, 16, 3))

    def test_registry_contains_required_ops(self):
        # 扩展：新增 rotate90/random_crop/HSV/模糊/噪声/擦除
        self.assertTrue(set(AUGMENT_OPS) >=
                        {"resize", "random_rescale", "random_flip_h",
                         "random_flip_v", "random_brightness", "random_contrast"})


class MaybeApplyOpsTest(unittest.TestCase):
    """随机启用增强：命中概率内既有增强也有原图，且与数据倍增无关。"""

    def test_random_enablement(self):
        import random as _random

        img = np.full((64, 64, 3), 100, np.uint8)
        ops = {"random_brightness": {"max_delta": 60}}
        rng = _random.Random(7)
        raw = aug = 0
        for _ in range(400):
            out, _ = maybe_apply_ops(img, [], ops, rng=rng)
            if out is img or np.array_equal(out, img):
                raw += 1
            else:
                aug += 1
        # 统计性验证：两种结果都大量出现，且比例接近 50/50（±15%）
        self.assertGreater(raw, 120)
        self.assertGreater(aug, 120)
        self.assertAlmostEqual(aug / 400, AUGMENT_PROB, delta=0.15)

    def test_no_ops_returns_original(self):
        img = np.zeros((32, 32, 3), np.uint8)
        out, _ = maybe_apply_ops(img, [], {})
        self.assertIs(out, img)


if __name__ == "__main__":
    unittest.main()
