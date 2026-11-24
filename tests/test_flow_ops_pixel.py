"""平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_flow_ops_preprocess
- test_v06_ops_p4
"""


# == 来源模块：test_flow_ops_preprocess ==
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

# == 来源模块：test_v06_ops_p4 ==
"""预处理六枚 / 定位与测量四枚 / 过滤两枚。

几何类一律用已知真值的合成图断言（旋转矩形、平行条带、3-4-5 三角形）。
"""

import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.engine.flow import ops
from xccv.engine.flow.executor import Executor
from xccv.engine.flow.ir import Geometry, ImageData
from xccv.engine.flow.node_base import RunContext
from xccv.engine.models.base import Prediction

ops.load_all()
from xccv.engine.flow.ops import ai_bridge as ab  # noqa: E402
from xccv.engine.flow.ops import locate as lc  # noqa: E402
from xccv.engine.flow.ops import measure as ms  # noqa: E402
from xccv.engine.flow.ops import preprocess as pp  # noqa: E402


def _run(cls, img, ctx=None, **params):
    node = cls()
    full = {k: spec.default for k, spec in node.PARAMS.items()}
    full.update(params)
    return node.run({"image": ImageData(mat=img)}, full, ctx or RunContext())


# ---------------------------------------------------------------- 预处理算子
class PreprocessP4Test(unittest.TestCase):
    def test_sharpen_identity_and_edge_boost(self):
        img = np.zeros((40, 40), np.uint8)
        img[:, 20:] = 200
        img = cv2.GaussianBlur(img, (7, 7), 0)
        same = _run(pp.SharpenUnsharp, img, amount=0.0)["image"].mat
        np.testing.assert_array_equal(same, img)
        out = _run(pp.SharpenUnsharp, img, amount=1.5)["image"].mat
        g_in = np.abs(np.diff(img[20].astype(int))).max()
        g_out = np.abs(np.diff(out[20].astype(int))).max()
        self.assertGreater(g_out, g_in)                       # 边缘更陡
        flat = np.full((20, 20), 100, np.uint8)
        flat[10, 10] = 104                                    # 小幅噪声
        kept = _run(pp.SharpenUnsharp, flat, amount=2.0, threshold=10)["image"].mat
        np.testing.assert_array_equal(kept, flat)             # 阈值抑噪

    def test_gamma_values(self):
        img = np.array([[0, 128, 255]], np.uint8)
        self.assertEqual(_run(pp.GammaCorr, img, gamma=1.0)["image"].mat.tolist(), [[0, 128, 255]])
        lo = _run(pp.GammaCorr, img, gamma=0.5)["image"].mat[0, 1]
        hi = _run(pp.GammaCorr, img, gamma=2.0)["image"].mat[0, 1]
        self.assertAlmostEqual(int(lo), round(255 * (128 / 255) ** 0.5), delta=1)
        self.assertAlmostEqual(int(hi), round(255 * (128 / 255) ** 2.0), delta=1)
        color = np.full((4, 4, 3), 64, np.uint8)
        self.assertEqual(_run(pp.GammaCorr, color, gamma=0.5)["image"].mat.shape, (4, 4, 3))

    def test_invert_and_brightness_contrast(self):
        img = np.array([[0, 100, 255]], np.uint8)
        self.assertEqual(_run(pp.Invert, img)["image"].mat.tolist(), [[255, 155, 0]])
        self.assertEqual(_run(pp.BrightnessContrast, img)["image"].mat.tolist(), [[0, 100, 255]])
        bc = _run(pp.BrightnessContrast, img, contrast=2.0, brightness=-50.0)["image"].mat
        self.assertEqual(bc.tolist(), [[0, 150, 255]])        # 饱和截断

    def test_flat_field_removes_gradient(self):
        h, w = 120, 160
        illum = np.tile(np.linspace(0.4, 1.0, w, dtype=np.float32), (h, 1))
        pattern = np.full((h, w), 180.0, np.float32)
        pattern[50:70, 70:90] = 60.0                           # 暗缺陷
        img = np.clip(pattern * illum, 0, 255).astype(np.uint8)
        out = _run(pp.FlatFieldCorr, img, ksize=61)["image"].mat
        bg_in = img[:20].astype(float).std()
        bg_out = out[:20].astype(float).std()
        self.assertLess(bg_out, bg_in * 0.35)                 # 背景明显均匀
        self.assertLess(int(out[60, 80]), int(out[10, 80]) - 50)   # 缺陷仍可分
        sub = _run(pp.FlatFieldCorr, img, mode="blur_subtract", ksize=61)["image"].mat
        self.assertEqual(sub.shape, img.shape)

    def test_flat_field_reference_mode(self):
        from xccv.engine.io_utils import imwrite_unicode

        flat = np.tile(np.linspace(100, 200, 64).astype(np.uint8), (48, 1))
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "平场.png"
            imwrite_unicode(fp, flat)
            out = _run(pp.FlatFieldCorr, flat, mode="reference", flat_path=str(fp))["image"].mat
            self.assertLess(out.astype(float).std(), 2.0)     # 自身除自身 → 近常数
        with self.assertRaises(ValueError):
            _run(pp.FlatFieldCorr, flat, mode="reference", flat_path="")

    def test_bit_plane(self):
        img = np.array([[200, 1]], np.uint8)                   # 200=11001000b
        self.assertEqual(_run(pp.BitPlane, img, plane=7)["image"].mat.tolist(), [[255, 0]])
        self.assertEqual(_run(pp.BitPlane, img, plane=0)["image"].mat.tolist(), [[0, 255]])
        self.assertEqual(_run(pp.BitPlane, img, plane=6, keep_above=True)["image"].mat.tolist(),
                         [[192, 0]])
        color = np.full((3, 3, 3), 255, np.uint8)
        self.assertEqual(_run(pp.BitPlane, color, plane=3)["image"].mat.ndim, 2)


# ---------------------------------------------------------------- 定位与测量算子
def _rot_rect_img(cx=100, cy=80, w=120, h=40, ang=30.0):
    img = np.zeros((160, 200), np.uint8)
    pts = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img


class FitRectangleTest(unittest.TestCase):
    def test_recovers_center_size_angle(self):
        out = _run(lc.FitRectangle, _rot_rect_img())
        cx, cy, w, h, ang = out["result"].data
        self.assertEqual(out["result"].kind, "rotated_rect")
        self.assertAlmostEqual(cx, 100, delta=1.5)
        self.assertAlmostEqual(cy, 80, delta=1.5)
        self.assertAlmostEqual(w, 120, delta=3)
        self.assertAlmostEqual(h, 40, delta=3)
        self.assertAlmostEqual(ang, 30.0, delta=1.5)
        self.assertAlmostEqual(out["angle"].value, ang, delta=0.01)

    def test_roi_offset_and_none_mode(self):
        out = _run(lc.FitRectangle, _rot_rect_img(), threshold_mode="none", roi="20,20,190,150")
        self.assertAlmostEqual(out["result"].data[0], 100, delta=1.5)   # 坐标回到全图

    def test_no_contour_raises(self):
        with self.assertRaises(ValueError):
            _run(lc.FitRectangle, np.zeros((50, 50), np.uint8), threshold_mode="none")


class FindLinePairTest(unittest.TestCase):
    def _band(self):
        img = np.full((120, 160), 30, np.uint8)
        img[40:80, :] = 200                                    # 亮条带：上边 y=40、下边 y=80
        return cv2.GaussianBlur(img, (3, 3), 0)

    def test_band_width(self):
        out = _run(lc.FindLinePair, self._band(), roi1="10,20,150,60", roi2="10,60,150,100",
                   polarity1="bright", polarity2="dark")
        self.assertAlmostEqual(out["distance"].value, 40.0, delta=1.0)
        # result=间距测量（与 distance 一致）；line1/angle_diff 端口已并入
        self.assertAlmostEqual(out["result"].value, out["distance"].value, places=3)
        self.assertEqual(out["image"].mat.shape[:2], (120, 160))   # 图像直通

    def test_validate_requires_rois(self):
        node = lc.FindLinePair()
        params = {k: s.default for k, s in node.PARAMS.items()}
        errs = node.validate(params)
        self.assertEqual(len([e for e in errs if "测量区" in e]), 2)

    def test_region_error_names_roi(self):
        flat = np.full((60, 60), 90, np.uint8)
        with self.assertRaises(ValueError) as cm:
            _run(lc.FindLinePair, flat, roi1="0,0,60,30", roi2="0,30,60,60")
        self.assertIn("测量区1", str(cm.exception))


class MeasureOpsTest(unittest.TestCase):
    P = Geometry(kind="point", data=np.array([[0.0, 0.0]]))
    Q = Geometry(kind="point", data=np.array([[3.0, 4.0]]))
    H1 = Geometry(kind="line", data=np.array([[0.0, 0.0], [100.0, 0.0]]))
    H2 = Geometry(kind="line", data=np.array([[0.0, 10.0], [100.0, 10.0]]))

    def _dist(self, a, b=None, **kw):
        node = ms.MeasureDistance()
        params = {k: s.default for k, s in node.PARAMS.items()}
        params.update(kw)
        inputs = {"result": a} if b is None else {"result": a, "result2": b}
        return node.run(inputs, params, RunContext())

    def _ang(self, a, b=None, **kw):
        node = ms.MeasureAngle()
        params = {k: s.default for k, s in node.PARAMS.items()}
        params.update(kw)
        inputs = {"result": a} if b is None else {"result": a, "result2": b}
        return node.run(inputs, params, RunContext())

    def test_distances(self):
        self.assertAlmostEqual(self._dist(self.P, self.Q)["distance"].value, 5.0)
        self.assertAlmostEqual(self._dist(self.Q, self.H1)["distance"].value, 4.0)   # 点-线垂距
        self.assertAlmostEqual(self._dist(self.H1, self.H2)["distance"].value, 10.0)  # 线-线
        self.assertAlmostEqual(self._dist(self.H1)["distance"].value, 100.0)          # 线长
        circ = Geometry(kind="circle", data=np.array([5.0, 5.0, 7.5]))
        self.assertAlmostEqual(self._dist(circ)["distance"].value, 15.0)              # 直径
        with self.assertRaises(ValueError):
            self._dist(self.P)

    def test_scale_and_tolerance(self):
        r = self._dist(self.P, self.Q, scale=0.1, tol_min=0.4, tol_max=0.6)
        self.assertEqual(r["result"].unit, "mm")
        self.assertAlmostEqual(r["result"].value, 0.5)
        self.assertTrue(r["result"].passed)
        r2 = self._dist(self.P, self.Q, tol_max=4.0)
        self.assertFalse(r2["result"].passed)
        self.assertIsNone(self._dist(self.P, self.Q)["result"].passed)              # 0/0 不判定

    def test_angles(self):
        diag = Geometry(kind="line", data=np.array([[0.0, 0.0], [10.0, 10.0]]))
        self.assertAlmostEqual(self._ang(diag)["angle"].value, 45.0)
        vert = Geometry(kind="line", data=np.array([[0.0, 0.0], [0.0, 10.0]]))
        self.assertAlmostEqual(self._ang(self.H1, vert)["angle"].value, 90.0)
        self.assertAlmostEqual(self._ang(self.H1, diag, mode="signed")["angle"].value, 45.0)
        rect = Geometry(kind="rotated_rect", data=np.array([10.0, 10.0, 8.0, 4.0, -12.5]))
        self.assertAlmostEqual(self._ang(rect)["angle"].value, -12.5)
        with self.assertRaises(ValueError):
            self._ang(self.P)

    def test_executor_chain_rect_to_angle(self):
        """端口连线：读图→拟合矩形→角度测量（geometry→geometry）全链。"""
        from xccv.engine.io_utils import imwrite_unicode

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "矩形.png"
            imwrite_unicode(p, _rot_rect_img(ang=20.0))
            graph = {"nodes": [{"id": 1, "type": "image_loader", "properties": {"path": str(p)}},
                               {"id": 2, "type": "fit_rectangle", "properties": {}},
                               {"id": 3, "type": "measure_angle", "properties": {}}],
                     "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
            res = Executor(graph, RunContext()).run()
        self.assertTrue(all(r.ok for r in res.values()), {k: r.error for k, r in res.items()})
        self.assertAlmostEqual(res[3].outputs["angle"].value, 20.0, delta=1.5)


# ---------------------------------------------------------------- 过滤算子
def _det():
    masks = [np.full((10, 10), i, np.uint8) for i in range(4)]
    return Prediction(labels=[0, 0, 1, 0], scores=[0.9, 0.8, 0.85, 0.3],
                      boxes=[[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [50, 50, 90, 90]],
                      masks=masks)


class FilterP4Test(unittest.TestCase):
    def _ov(self, **kw):
        node = ab.FilterOverlap()
        params = {k: s.default for k, s in node.PARAMS.items()}
        params.update(kw)
        return node.run({"result": _det()}, params, RunContext())

    def test_overlap_same_class(self):
        r = self._ov(iou=0.5)
        self.assertEqual(r["result"].scores, [0.9, 0.85, 0.3])    # 同类 0.8 被去重
        self.assertEqual(r["removed"].scores, [0.8])
        self.assertEqual(int(r["removed"].masks[0][0, 0]), 1)        # 掩码随检出

    def test_overlap_class_agnostic_and_area(self):
        r = self._ov(iou=0.5, class_agnostic=True)
        self.assertEqual(r["result"].scores, [0.9, 0.3])
        r2 = self._ov(iou=0.5, keep="area")
        self.assertEqual(len(r2["result"].boxes) + len(r2["removed"].boxes), 4)

    def test_topk(self):
        node = ab.FilterTopK()
        base = {k: s.default for k, s in node.PARAMS.items()}
        r = node.run({"result": _det()}, dict(base, k=2), RunContext())
        self.assertEqual(r["result"].scores, [0.9, 0.85])
        self.assertEqual(r["removed"].scores, [0.8, 0.3])
        pc = node.run({"result": _det()}, dict(base, k=1, per_class=True), RunContext())
        self.assertEqual(sorted(pc["result"].labels), [0, 1])
        ar = node.run({"result": _det()}, dict(base, k=1, by="area"), RunContext())
        self.assertEqual(ar["result"].boxes, [[50.0, 50.0, 90.0, 90.0]])

    def test_empty_detection(self):
        empty = Prediction(labels=[], scores=[], boxes=[])
        r = ab.FilterTopK().run({"result": empty}, {"k": 3, "by": "score", "per_class": False},
                                RunContext())
        self.assertEqual(r["result"].boxes, [])


class OperatorCountTest(unittest.TestCase):
    # 内置算子清单（对账基准；其它测试模块会向全局注册表追加 ex_* 等测试桩，
    # 故只断言"全集包含内置清单"，不锁全局长度）
    BUILTINS = {
        # V0.5.6（37）: 图像源 2 / 预处理 11 / 定位 5 / 逻辑 4 / AI 6 / 检出过滤 8 / 输出 1
        "image_loader", "detection_source",
        "gaussian_blur", "median_blur", "bilateral_filter", "clahe", "threshold",
        "morphology", "canny_edge", "sobel_edge", "fft_filter", "affine_transform", "roi_extract",
        "blob_analysis", "find_line", "find_circle", "find_corner", "template_match",
        "compare_value", "math_op", "condition_branch", "result_merge",
        "model_detection", "model_instance_seg", "model_classification", "roi_batch_classify",
        "detection_to_measure", "geometry_to_affine",
        "filter_confidence", "filter_class", "filter_bbox_size", "filter_bbox_area",
        "filter_mask_area", "filter_count_check", "filter_region_block", "filter_region_keep",
        "result_output",
        # V0.6 P3（+6）
        "crop_by_detection", "grid_tile", "relabel_map", "image_sink", "csv_sink", "bool_op",

        "sharpen_unsharp", "gamma_corr", "invert", "brightness_contrast", "flat_field_corr",
        "bit_plane", "fit_rectangle", "find_line_pair", "measure_distance", "measure_angle",
        "filter_overlap", "filter_topk",
        # （+8，）
        "crop_by_detection_all", "grid_tile_all", "model_on_list",
        "op_on_list", "merge_results", "mask_to_box", "external_op",
        # （+1，输出结果节点——不落盘出口）
        "result_view",
        # V0.7 O2 扩展（+3，追加多维阈值——短长边/前景占比/灰度差；
        # 面积过滤沿用 legacy 薄委托 filter_bbox_area，不在本清单重复）
        "filter_bbox_side", "filter_foreground", "filter_gray_diff",
    }

    def test_p4_operator_inventory(self):
        """算子总数对账：
        …63+3 + 输出结果 1 = 67 内置（不含 custom/）。"""
        from xccv.engine.flow import legacy_adapter  # noqa: F401
        from xccv.engine.flow.registry import list_nodes

        names = {n["name"] for n in list_nodes()}
        self.assertLessEqual(self.BUILTINS, names)                # 内置一枚不少
        builtins_in_registry = {n for n in names
                                if not n.startswith("custom/")
                                and not n.startswith(("ex_", "rg_"))}   # 排除测试桩
        self.assertEqual(len(builtins_in_registry), len(self.BUILTINS),
                         sorted(builtins_in_registry ^ self.BUILTINS))

if __name__ == '__main__':
    unittest.main()
