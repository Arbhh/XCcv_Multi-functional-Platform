# -*- coding: utf-8 -*-
"""增强算子扩展 + YOLO 标签同步 + 每算子概率 + 数据集指纹 + rotate90 守卫。

背景：YOLO 在线增强此前把 shapes 传空（yolo_data.load_image），翻转算子
翻转图像但不翻转标注 → 标签错位。现改为 get_image_and_label 内增强与标签
同步重建，本文件用真实 InMemoryYoloDataset 验证镜像一致性。
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.synthetic import make_labelme_dataset
from xccv.engine.dataset import transforms as T
from xccv.engine.dataset.base import Shape, TaskType
from xccv.engine.dataset.prepare import dataset_fingerprint
from xccv.engine.dataset.reader import scan_labelme_dataset
from xccv.engine.models.yolo_data import InMemoryYoloDataset, build_labels_from_shapes


def _rect(x1, y1, x2, y2, label="缺陷"):
    return Shape(label=label, shape_type="rectangle",
                 points=[[float(x1), float(y1)], [float(x2), float(y2)]])


class RotateShapeTest(unittest.TestCase):
    def test_rotate_shapes_clockwise(self):
        shapes = [_rect(10, 5, 40, 20)]
        out = T._rotate_shapes(shapes, 100, 50, clockwise=True)
        self.assertEqual(out[0].points, [[45.0, 10.0], [30.0, 40.0]])

    def test_rotate_shapes_counter(self):
        shapes = [_rect(10, 5, 40, 20)]
        out = T._rotate_shapes(shapes, 100, 50, clockwise=False)
        self.assertEqual(out[0].points, [[5.0, 90.0], [20.0, 60.0]])

    def test_rotate90_four_times_identity(self):
        """同向旋转 4 次或正反成对必须回到原图与原标注（浮点误差内）。"""
        img = np.random.default_rng(0).integers(0, 255, (50, 100, 3), dtype=np.uint8)
        base = [_rect(10, 5, 40, 20), _rect(60, 30, 90, 45, label="划痕")]

        class _Coin:
            def __init__(self, seq):
                self._seq = list(seq)

            def random(self):
                return self._seq.pop(0) if self._seq else 0.1

        for coin in (_Coin([0.1] * 4), _Coin([0.9] * 4), _Coin([0.1, 0.9, 0.1, 0.9])):
            shapes = [_rect(*p[0], *p[1], label=s.label) for s, p in
                      zip(base, [s.points for s in base])]
            im = img.copy()
            for _ in range(4):
                im, shapes = T.rotate90(im, shapes, rng=coin)
            self.assertEqual(im.shape[:2], (50, 100))
            for a, b in zip(shapes, base):
                self.assertEqual(a.label, b.label)
                self.assertTrue(np.allclose(np.sort(a.points, axis=0),
                                            np.sort(b.points, axis=0), atol=1e-6),
                                f"方向序列 {coin._seq} 下旋转 4 次应还原标注")

    def test_rotate90_dims_swap(self):
        img = np.zeros((20, 50, 3), np.uint8)
        out, _ = T.rotate90(img, [], rng=random.Random(0))
        self.assertEqual(out.shape[:2], (50, 20))


class RandomCropTest(unittest.TestCase):
    def test_passthrough_on_matching_aspect(self):
        img = np.zeros((50, 100, 3), np.uint8)
        shapes = [_rect(10, 5, 40, 20)]
        out, sp = T.random_crop(img, shapes, ratio=2.0, rng=random.Random(0))
        self.assertIs(out, img)
        self.assertIs(sp, shapes)

    def test_crop_makes_target_aspect_and_keeps_labels_inside(self):
        img = np.random.default_rng(1).integers(0, 255, (50, 100, 3), dtype=np.uint8)
        shapes = [_rect(10, 5, 40, 20)]
        out, sp = T.random_crop(img, shapes, ratio=1.0, min_cover=0.5,
                                rng=random.Random(3))
        h, w = out.shape[:2]
        self.assertEqual(w, h)  # 裁成方形
        self.assertGreaterEqual(w * h, 0.5 * 100 * 50 * 0.9)
        for x, y in sp[0].points:
            self.assertTrue(0 <= x <= w and 0 <= y <= h)


class PhotometricOpsTest(unittest.TestCase):
    """光度类算子：标注原样返回（同一对象），图像被修改且尺寸不变。"""

    def test_hsv_shift(self):
        img = np.random.default_rng(2).integers(0, 255, (30, 40, 3), dtype=np.uint8)
        shapes = [_rect(1, 1, 10, 10)]
        out, sp = T.random_hsv_shift(img, shapes, rng=random.Random(0))
        self.assertIs(sp, shapes)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertFalse(np.array_equal(out, img))

    def test_gaussian_blur(self):
        img = np.random.default_rng(2).integers(0, 255, (30, 40, 3), dtype=np.uint8)
        out, sp = T.gaussian_blur(img, shapes := [_rect(1, 1, 10, 10)],
                                  ksize=3, rng=random.Random(0))
        self.assertIs(sp, shapes)
        self.assertFalse(np.array_equal(out, img))

    def test_gaussian_noise(self):
        img = np.full((30, 40, 3), 128, np.uint8)
        out, sp = T.gaussian_noise(img, shapes := [_rect(1, 1, 10, 10)],
                                   sigma=25, rng=random.Random(0))
        self.assertIs(sp, shapes)
        self.assertFalse(np.array_equal(out, img))

    def test_random_erase(self):
        img = np.random.default_rng(2).integers(0, 255, (30, 40, 3), dtype=np.uint8)
        shapes = [_rect(1, 1, 10, 10)]
        out, sp = T.random_erase(img, shapes, max_ratio=0.5, rng=random.Random(2))
        self.assertIs(sp, shapes)
        self.assertTrue(np.any(out != img))  # 均值灰填充改变了噪声图


class MaybeApplyProbsTest(unittest.TestCase):
    def test_zero_prob_never_applies(self):
        img = np.random.default_rng(3).integers(0, 255, (30, 40, 3), dtype=np.uint8)
        shapes = [_rect(1, 1, 10, 10)]
        for _ in range(20):
            out, sp = T.maybe_apply_ops(img, shapes, {"random_flip_h": {}},
                                        rng=random.Random(1),
                                        probs={"random_flip_h": 0.0})
            self.assertIs(out, img)
            self.assertIs(sp, shapes)

    def test_full_prob_applies_with_params(self):
        img = np.random.default_rng(3).integers(0, 255, (30, 40, 3), dtype=np.uint8)
        shapes = [_rect(1, 1, 10, 10)]
        out, _ = T.maybe_apply_ops(img, shapes,
                                   {"random_flip_h": {"probability": 1.0}},
                                   rng=random.Random(0),
                                   probs={"random_flip_h": 1.0})
        self.assertFalse(np.array_equal(out, img))  # 翻转必然发生

    def test_rescale_bounds_with_max_scale(self):
        """随机比例缩放含上限：输出尺寸在 [min, max] 比例范围内。"""
        img = np.zeros((50, 100, 3), np.uint8)
        for seed in range(10):
            out, _ = T.random_rescale(img, [], min_scale=0.8, max_scale=1.2,
                                      rng=random.Random(seed))
            h, w = out.shape[:2]
            self.assertTrue(0.8 * 100 - 2 <= w <= 1.2 * 100 + 2, f"w={w}")
            self.assertTrue(0.8 * 50 - 2 <= h <= 1.2 * 50 + 2, f"h={h}")

    def test_flip_without_probability_always_flips_when_chosen(self):
        """翻转去掉 probability 参数后：被选中即翻转（启用由概率列控制）。"""
        img = np.random.default_rng(5).integers(0, 255, (20, 30, 3), dtype=np.uint8)
        out, _ = T.maybe_apply_ops(img, [], {"random_flip_h": {}},
                                   rng=random.Random(0),
                                   probs={"random_flip_h": 1.0})
        self.assertFalse(np.array_equal(out, img))

    def test_registry_has_new_ops(self):
        for name in ("rotate90", "random_crop", "random_hsv_shift",
                     "gaussian_blur", "gaussian_noise", "random_erase"):
            self.assertIn(name, T.AUGMENT_OPS)

    def test_all_ops_chain(self):
        """全部算子按注册表顺序连跑不异常，shapes 始终是 Shape 列表。"""
        img = np.random.default_rng(4).integers(0, 255, (60, 90, 3), dtype=np.uint8)
        shapes = [_rect(5, 5, 30, 25)]
        ops = {name: {} for name in T.AUGMENT_OPS}
        ops["resize"] = {"width": 64, "height": 64}
        out, sp = T.apply_ops(img, shapes, ops, rng=random.Random(5))
        self.assertEqual(out.ndim, 3)
        self.assertTrue(all(isinstance(s, Shape) for s in sp))


class YoloLabelSyncTest(unittest.TestCase):
    """P0：YOLO 在线增强必须与标注同步（翻转后框镜像）。"""

    def _build(self, ops, probs):
        from ultralytics.cfg import get_cfg

        hyp = get_cfg()
        for k in ("mosaic", "mixup", "copy_paste", "cutmix", "degrees", "translate",
                  "scale", "shear", "perspective", "flipud", "fliplr", "bgr",
                  "hsv_h", "hsv_s", "hsv_v", "erasing"):
            if hasattr(hyp, k):
                setattr(hyp, k, 0.0)
        samples, classes = _samples()
        return InMemoryYoloDataset(
            samples, classes, "detection", ops, augment_probs=probs,
            img_path="xccv://in-memory", imgsz=64, batch_size=2, augment=True,
            hyp=hyp, rect=False, cache=None, single_cls=False, stride=32,
            pad=0.0, prefix="", data={"channels": 3}, fraction=1.0)

    def test_flip_label_sync(self):
        """翻转增强命中的取样：图像翻转且 bbox 水平镜像。"""
        ds = self._build({"random_flip_h": {"probability": 1.0}},
                         {"random_flip_h": 1.0})
        item = ds[0]
        base = ds.labels[0]["bboxes"]
        got = np.asarray(item["bboxes"], dtype=np.float32).reshape(-1, 4)
        self.assertEqual(got.shape, base.shape)
        self.assertTrue(np.allclose(got[:, 0], 1.0 - base[:, 0], atol=1e-5),
                        f"翻转后 bbox 中心 x 应镜像: {got[:, 0]} vs {base[:, 0]}")
        self.assertTrue(np.allclose(got[:, 2], base[:, 2], atol=1e-5))  # 宽不变

    def test_no_ops_keeps_labels(self):
        """无增强：框在网络输入上保持原图像素长宽比（训练几何与验证/推理
        一致——非方形图长边缩放 + letterbox 补边，不再拉伸成方形）。"""
        ds = self._build(None, None)
        item = ds[0]
        base = ds.labels[0]["bboxes"]
        got = np.asarray(item["bboxes"], dtype=np.float32).reshape(-1, 4)
        self.assertEqual(got.shape, base.shape)
        H, W = item["img"].shape[-2:]
        h0, w0 = 64, 96   # _samples() 合成图尺寸
        ar_got = (got[:, 2] * W) / (got[:, 3] * H)
        ar_base = (base[:, 2] * w0) / (base[:, 3] * h0)
        self.assertTrue(np.allclose(ar_got, ar_base, rtol=0.05),
                        f"框像素长宽比应保持: {ar_got} vs {ar_base}")


def _samples():
    """合成 labelme 数据集（非方形 96x64）→ 样本清单。"""
    tmp = tempfile.TemporaryDirectory()
    root, classes = make_labelme_dataset(Path(tmp.name) / "ds", n_images=3,
                                         size=(96, 64), shape_type="polygon",
                                         classes=("cat", "dog"))
    samples, classes2, _ = scan_labelme_dataset(root)
    YoloLabelSyncTest._tmp = tmp  # 保活临时目录
    return samples, classes2


class FingerprintTest(unittest.TestCase):
    def test_stable_and_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("1", encoding="utf-8")
            f1 = dataset_fingerprint(root)
            self.assertEqual(f1, dataset_fingerprint(root))
            (root / "b.txt").write_text("2", encoding="utf-8")
            self.assertNotEqual(f1, dataset_fingerprint(root))
            (root / "b.txt").write_text("3", encoding="utf-8")
            self.assertNotEqual(f1, dataset_fingerprint(root))

    def test_hidden_files_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("1", encoding="utf-8")
            f1 = dataset_fingerprint(root)
            (root / ".hidden").write_text("x", encoding="utf-8")
            self.assertEqual(f1, dataset_fingerprint(root))


class GuardDecisionTest(unittest.TestCase):
    def test_no_rotate_passes(self):
        self.assertIsNone(
            __import__("xccv.training.training_window", fromlist=["x"])
            .rotate_guard_decision({}, {}, {1.5}))

    def test_consistent_aspect_passes(self):
        tw = __import__("xccv.training.training_window", fromlist=["x"])
        self.assertIsNone(
            tw.rotate_guard_decision({"rotate90": {}}, {}, {1.5, 1.5}))

    def test_inconsistent_aspect_blocks(self):
        tw = __import__("xccv.training.training_window", fromlist=["x"])
        decision = tw.rotate_guard_decision({"rotate90": {}}, {}, {1.5, 1.0})
        self.assertEqual(decision[0], "enable")

    def test_crop_with_full_prob_passes(self):
        tw = __import__("xccv.training.training_window", fromlist=["x"])
        self.assertIsNone(tw.rotate_guard_decision(
            {"rotate90": {}, "random_crop": {"ratio": 1.0}},
            {"random_crop": 1.0}, {1.5, 1.0}))

    def test_crop_partial_prob_requires_fix(self):
        tw = __import__("xccv.training.training_window", fromlist=["x"])
        decision = tw.rotate_guard_decision(
            {"rotate90": {}, "random_crop": {}}, {"random_crop": 0.5}, {1.5, 1.0})
        self.assertEqual(decision[0], "prob")

    def test_legacy_missing_prob_defaults_to_block(self):
        """旧项目无 augment_probs 字段：默认 0.5 → 必须提示补齐为 100%。"""
        tw = __import__("xccv.training.training_window", fromlist=["x"])
        decision = tw.rotate_guard_decision(
            {"rotate90": {}, "random_crop": {}}, {}, {1.5, 1.0})
        self.assertEqual(decision[0], "prob")


class DoubleSpinTextTest(unittest.TestCase):
    """Q1：小数参数框最小位数显示，改值不再需要删尾零。"""

    def test_minimal_text(self):
        from xccv.training.training_window import _DoubleSpin

        self.assertEqual(_DoubleSpin.textFromValue(object(), 0.5), "0.5")
        self.assertEqual(_DoubleSpin.textFromValue(object(), 1.0), "1")
        self.assertEqual(_DoubleSpin.textFromValue(object(), 0.25), "0.25")
        self.assertEqual(_DoubleSpin.textFromValue(object(), 1e-3), "0.001")

    def test_value_from_text(self):
        from xccv.training.training_window import _DoubleSpin

        self.assertAlmostEqual(_DoubleSpin.valueFromText(object(), "0.3"), 0.3)


class BuildLabelsFromShapesTest(unittest.TestCase):
    def test_matches_sample_based_builder(self):
        tmp = tempfile.TemporaryDirectory()
        root, classes = make_labelme_dataset(Path(tmp.name) / "ds", n_images=2,
                                             size=(80, 60))
        samples, classes, _ = scan_labelme_dataset(root)
        for s in samples:
            rebuilt = build_labels_from_shapes(s.shapes, classes, "detection",
                                               s.width, s.height,
                                               im_file=str(s.image_path))
            direct = __import__(
                "xccv.engine.models.yolo_data", fromlist=["x"]).build_label_dict(
                    s, classes, "detection")
            for key in ("cls", "bboxes", "segments", "normalized", "bbox_format"):
                a, b = rebuilt[key], direct[key]
                if isinstance(a, np.ndarray) and a.size:
                    self.assertTrue(np.allclose(np.asarray(a, np.float32),
                                                np.asarray(b, np.float32)))
                else:
                    self.assertEqual(a, b)


class YoloAugmentTrainE2ETest(unittest.TestCase):
    """真实训练闭环：翻转+旋转+裁剪全开（概率 100%），YOLO 完整训练 1 轮。

    守卫语义：数据集 96x64 长宽比一致，rotate90 直接可用；随机裁剪 ratio=1.5
    对齐目标比例。重建标签必须能被 ultralytics 正常编入批次并完成训练。
    """

    def test_train_one_epoch_with_geometric_augment(self):
        from xccv.engine.models import create_model
        from xccv.engine.models.base import TrainConfig

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        make_labelme_dataset(root / "ds", n_images=4, size=(96, 64),
                             shape_type="polygon")
        model = create_model("yolo11n", TaskType.DETECTION, 2, device="cpu",
                             use_pretrained=False)
        cfg = TrainConfig(
            task=TaskType.DETECTION, dataset_dir=root / "ds",
            output_dir=root / "run", model_name="yolo11n",
            epochs=1, batch_size=2, workers=0, seed=0,
            image_width=64, image_height=64,
            augment_ops={"random_flip_h": {"probability": 1.0},
                         "rotate90": {},
                         "random_crop": {"ratio": 1.5, "min_cover": 0.5}},
            augment_probs={"random_flip_h": 1.0, "rotate90": 1.0,
                           "random_crop": 1.0},
        )
        metrics = model.train(cfg)
        self.assertIn(metrics.get("status"), ("completed", "failed_early_stop"))
        self.assertTrue(metrics.get("history"))  # 至少完成 1 轮验证


if __name__ == "__main__":
    unittest.main()
