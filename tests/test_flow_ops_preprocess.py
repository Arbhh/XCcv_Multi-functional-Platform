"""预处理 11 算子测试（每算子 ≥2 例：正常路径 + 边界路径）。"""

import os
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import xccv.engine.flow.ops.preprocess as pp  # noqa: F401  触发注册
from xccv.engine.flow.ir import Geometry, ImageData
from xccv.engine.flow.node_base import RunContext

CTX = lambda: RunContext()  # noqa: E731


def _color(h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(0, 255, (h, w, 3)), dtype=np.uint8)


def _gray(h=32, w=32, seed=1):
    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(0, 255, (h, w)), dtype=np.uint8)


def _img(mat):
    return ImageData(mat=mat)


class GaussianBlurTest(unittest.TestCase):
    def test_normal_color_preserved(self):
        out = pp.GaussianBlur().run({"image": _img(_color())},
                                    {"ksize": 3, "sigma": 0}, CTX())["image"]
        self.assertEqual(out.mat.shape, (32, 32, 3))
        self.assertEqual(out.mat.dtype, np.uint8)

    def test_even_ksize_auto_odd_with_warning(self):
        ctx = CTX()
        pp.GaussianBlur().run({"image": _img(_gray())},
                              {"ksize": 4, "sigma": 2.0}, ctx)
        self.assertTrue(any("奇数" in w for w in ctx.warnings))


class MedianBlurTest(unittest.TestCase):
    def test_salt_pepper_removed(self):
        mat = np.full((16, 16), 128, np.uint8)
        mat[4, 4] = 255  # 椒盐点
        out = pp.MedianBlur().run({"image": _img(mat)}, {"ksize": 3}, CTX())["image"]
        self.assertLess(int(out.mat[4, 4]), 200)

    def test_even_ksize_promoted(self):
        out = pp.MedianBlur().run({"image": _img(_gray())}, {"ksize": 6}, CTX())["image"]
        self.assertEqual(out.mat.shape, (32, 32))


class BilateralFilterTest(unittest.TestCase):
    def test_normal_color(self):
        out = pp.BilateralFilter().run(
            {"image": _img(_color())},
            {"d": 5, "sigma_color": 75, "sigma_space": 75}, CTX())["image"]
        self.assertEqual(out.mat.shape, (32, 32, 3))

    def test_min_d_boundary(self):
        out = pp.BilateralFilter().run(
            {"image": _img(_gray())},
            {"d": 1, "sigma_color": 1, "sigma_space": 1}, CTX())["image"]
        self.assertEqual(out.mat.dtype, np.uint8)


class ClaheTest(unittest.TestCase):
    def test_low_contrast_enhanced(self):
        flat = np.tile(np.linspace(100, 110, 32, dtype=np.uint8), (32, 1))
        out = pp.Clahe().run({"image": _img(flat)},
                             {"clip_limit": 2.0, "tile_size": 8}, CTX())["image"]
        self.assertGreater(int(np.ptp(out.mat)), int(np.ptp(flat)) + 5)

    def test_color_auto_gray(self):
        out = pp.Clahe().run({"image": _img(_color())},
                             {"clip_limit": 2.0, "tile_size": 4}, CTX())["image"]
        self.assertEqual(out.mat.ndim, 2)  # 彩色输入自动转灰度


class ThresholdTest(unittest.TestCase):
    def test_global_and_invert(self):
        mat = np.full((8, 8), 50, np.uint8)
        node = pp.Threshold()
        low = node.run({"image": _img(mat)},
                       {"mode": "global", "thresh": 100, "invert": False},
                       CTX())["image"].mat
        inv = node.run({"image": _img(mat)},
                       {"mode": "global", "thresh": 100, "invert": True},
                       CTX())["image"].mat
        self.assertTrue(np.all(low == 0))
        self.assertTrue(np.all(inv == 255))

    def test_otsu_ignores_thresh(self):
        """otsu 模式忽略 thresh：两个不同阈值结果一致。"""
        mat = _gray()
        node = pp.Threshold()
        args = {"mode": "otsu", "invert": False}
        a = node.run({"image": _img(mat)}, {**args, "thresh": 10}, CTX())["image"].mat
        b = node.run({"image": _img(mat)}, {**args, "thresh": 240}, CTX())["image"].mat
        self.assertTrue(np.array_equal(a, b))

    def test_adaptive_even_block_auto_odd(self):
        ctx = CTX()
        out = pp.Threshold().run(
            {"image": _img(_gray())},
            {"mode": "adaptive", "adaptive_method": "gaussian",
             "block_size": 10, "invert": False}, ctx)["image"].mat
        self.assertEqual(out.ndim, 2)
        self.assertTrue(any("奇数" in w for w in ctx.warnings))


class MorphologyTest(unittest.TestCase):
    def _speckle(self):
        mat = np.zeros((24, 24), np.uint8)
        mat[6, 6] = 255          # 孤立亮点（开运算应消除）
        mat[10:16, 10:16] = 255  # 大亮块（保留）
        return mat

    def test_open_removes_speck_keeps_block(self):
        out = pp.Morphology().run(
            {"image": _img(self._speckle())},
            {"op": "open", "kernel_w": 3, "kernel_h": 3, "iterations": 1},
            CTX())["image"].mat
        self.assertEqual(int(out[6, 6]), 0)
        self.assertEqual(int(out[12, 12]), 255)

    def test_tophat_extracts_speck_and_even_kernel(self):
        ctx = CTX()
        out = pp.Morphology().run(
            {"image": _img(self._speckle())},
            {"op": "tophat", "kernel_w": 4, "kernel_h": 4, "iterations": 1},
            ctx)["image"].mat
        self.assertGreaterEqual(int(out[6, 6]), 255 - 1)  # 顶帽提出孤立亮点
        self.assertTrue(any("奇数" in w for w in ctx.warnings))


class CannyEdgeTest(unittest.TestCase):
    def test_edges_on_box(self):
        mat = np.zeros((32, 32), np.uint8)
        mat[8:24, 8:24] = 255
        out = pp.CannyEdge().run({"image": _img(mat)},
                                 {"low": 50, "high": 150, "aperture": 3},
                                 CTX())["image"].mat
        self.assertGreaterEqual(int(out.sum() / 255), 16)  # 框边缘被检出
        self.assertEqual(out.max(), 255)

    def test_low_gt_high_swapped_with_warning_and_color_input(self):
        ctx = CTX()
        out = pp.CannyEdge().run({"image": _img(_color())},
                                 {"low": 200, "high": 50, "aperture": 3},
                                 ctx)["image"].mat
        self.assertEqual(out.ndim, 2)  # 彩色自动转灰度
        self.assertTrue(any("交换" in w for w in ctx.warnings))


class SobelEdgeTest(unittest.TestCase):
    def test_vertical_edge_dx(self):
        mat = np.zeros((16, 16), np.uint8)
        mat[:, 8:] = 255  # 垂直台阶
        out = pp.SobelEdge().run({"image": _img(mat)},
                                 {"dx": 1, "dy": 0, "ksize": 3}, CTX())["image"].mat
        self.assertGreater(int(out[:, 7:10].sum()), 0)

    def test_both_zero_auto_corrected_scharr(self):
        ctx = CTX()
        out = pp.SobelEdge().run({"image": _img(_gray())},
                                 {"dx": 0, "dy": 0, "ksize": 1}, ctx)["image"].mat
        self.assertEqual(out.ndim, 2)
        self.assertTrue(any("不能同时为 0" in w for w in ctx.warnings))


class FftFilterTest(unittest.TestCase):
    def _checker(self):
        yy, xx = np.mgrid[0:32, 0:32]
        return np.where(((yy // 2 + xx // 2) % 2 == 0), 220, 30).astype(np.uint8)

    def test_lowpass_smooths_checker(self):
        img = self._checker()
        out = pp.FftFilter().run({"image": _img(img)},
                                 {"mode": "lowpass", "cutoff_ratio": 0.1,
                                  "bandwidth": 0.05}, CTX())["image"].mat
        self.assertLess(float(out.std()), float(img.std()))  # 高频被抑制

    def test_highpass_and_bandstop_run(self):
        img = self._checker()
        node = pp.FftFilter()
        for mode in ("highpass", "bandstop"):
            out = node.run({"image": _img(img)},
                           {"mode": mode, "cutoff_ratio": 0.1,
                            "bandwidth": 0.05}, CTX())["image"].mat
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)  # clip 回 uint8 契约


class AffineTransformTest(unittest.TestCase):
    def test_identity_keeps_size(self):
        mat = _color()
        out = pp.AffineTransform().run(
            {"image": _img(mat)}, {"matrix": "1,0,0,0,1,0", "interp": "linear"},
            CTX())["image"].mat
        self.assertEqual(out.shape, mat.shape)

    def test_translation_moves_content(self):
        mat = np.zeros((16, 16), np.uint8)
        mat[0, 0] = 255
        out = pp.AffineTransform().run(
            {"image": _img(mat)}, {"matrix": "1,0,4,0,1,3", "interp": "nearest"},
            CTX())["image"].mat
        self.assertEqual(int(out[3, 4]), 255)  # 内容平移 (tx=4, ty=3)

    def test_invalid_matrix_caught_by_validate(self):
        errs = pp.AffineTransform().validate({"matrix": "1,0,0", "interp": "linear"})
        self.assertTrue(any("6 个数字" in e for e in errs))


class RoiExtractTest(unittest.TestCase):
    def test_rect_crop(self):
        mat = _color(40, 60)
        out = pp.RoiExtract().run(
            {"image": _img(mat)},
            {"mode": "rect", "x": 0.5, "y": 0.25, "w": 0.5, "h": 0.5,
             "rin": 0.3, "rout": 0.9}, CTX())["image"]
        self.assertEqual(out.mat.shape, (20, 30, 3))
        self.assertEqual(out.meta["roi_rect"], [30, 10, 60, 30])

    def test_rect_degenerate_rejected(self):
        with self.assertRaises(ValueError, msg="不足 1 像素"):
            pp.RoiExtract().run(
                {"image": _img(_color())},
                {"mode": "rect", "x": 0.9, "y": 0.9, "w": 0.02, "h": 0.02,
                 "rin": 0.3, "rout": 0.9}, CTX())

    def test_annulus_keeps_ring_only(self):
        mat = np.full((40, 40), 200, np.uint8)
        out = pp.RoiExtract().run(
            {"image": _img(mat)},
            {"mode": "annulus", "rin": 0.2, "rout": 0.4,
             "x": 0, "y": 0, "w": 1, "h": 1}, CTX())["image"].mat
        self.assertEqual(int(out[0, 0]), 0)          # 角落：dist≈28.3 > 8
        self.assertEqual(int(out[20, 27]), 200)      # dist=7 ∈ [4, 8] 环内

    def test_polygon_mask_from_geometry(self):
        mat = np.full((20, 20), 100, np.uint8)
        geo = Geometry("polygon", [[2, 2], [18, 2], [18, 18], [2, 18]])
        out = pp.RoiExtract().run(
            {"image": _img(mat), "polygon": geo},
            {"mode": "polygon", "x": 0, "y": 0, "w": 1, "h": 1,
             "rin": 0.3, "rout": 0.9}, CTX())["image"].mat
        self.assertEqual(int(out[10, 10]), 100)  # 多边形内保留
        self.assertEqual(int(out[0, 0]), 0)      # 多边形外置黑

    def test_polygon_mode_requires_geometry(self):
        with self.assertRaises(ValueError, msg="几何输入端口"):
            pp.RoiExtract().run(
                {"image": _img(_color())},
                {"mode": "polygon", "x": 0, "y": 0, "w": 1, "h": 1,
                 "rin": 0.3, "rout": 0.9}, CTX())


class EndToEndThroughExecutorTest(unittest.TestCase):
    def test_loader_clahe_threshold_chain(self):
        """真链路：image_loader → clahe → threshold（Executor 拓扑执行）。"""
        import tempfile

        import xccv.engine.flow.ops.source  # noqa: F401  触发注册
        from xccv.engine.flow.executor import Executor

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "图 片.png"  # 中文+空格路径（imread_unicode 契约）
        from xccv.engine.io_utils import imwrite_unicode  # cv2.imwrite 中文路径不可靠

        imwrite_unicode(path, _color(24, 24, seed=3))
        g = {"nodes": [
                {"id": 1, "type": "image_loader",
                 "properties": {"path": str(path), "is_grayscale": True}},
                {"id": 2, "type": "clahe",
                 "properties": {"clip_limit": 2.0, "tile_size": 4}},
                {"id": 3, "type": "threshold",
                 "properties": {"mode": "otsu", "invert": False}},
             ],
             "links": [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0]]}
        res = Executor(g).run()
        self.assertTrue(all(r.ok for r in res.values()))
        self.assertEqual(res[3].outputs["image"].mat.ndim, 2)
        self.assertEqual(res[3].outputs["image"].mat.max(), 255)


if __name__ == "__main__":
    unittest.main()

