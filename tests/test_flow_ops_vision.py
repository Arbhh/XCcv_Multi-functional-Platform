"""平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_flow_ops_locate
- test_flow_ops_logic
"""


# == 来源模块：test_flow_ops_locate ==
"""定位算子测试（每算子 ≥2 例 + 真值容限断言）。

容限（合成图上）：圆心误差 ≤0.5px、半径 ≤1%、线端点 ≤1px；
template_match 金字塔降级路径在 2400×1500 大图上验证。
"""

import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import xccv.engine.flow.ops.locate  # noqa: F401  触发注册
from xccv.engine.flow.ir import Geometry
from xccv.engine.flow.node_base import RunContext
from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.models.base import Prediction

CTX = RunContext


def node(name: str):
    from xccv.engine.flow.registry import get_node_class

    return get_node_class(name)()


def img(mat):
    from xccv.engine.flow.ir import ImageData

    return ImageData(mat=mat)


class BlobAnalysisTest(unittest.TestCase):
    def _scene(self):
        """黑底 + 两个亮斑：圆(r=15@40,40) 与 方块(20×20@90,60)。"""
        m = np.zeros((120, 140), np.uint8)
        cv2.circle(m, (40, 40), 15, 255, -1)
        cv2.rectangle(m, (90, 60), (110, 80), 255, -1)
        return m

    def test_outputs_prediction_with_boxes(self):
        r = node("blob_analysis").run({"image": img(self._scene())},
                                      {"threshold_mode": "otsu", "thresh": 128,
                                       "invert": False, "min_area": 10,
                                       "max_area": 0, "circularity": 0.0}, CTX())
        pred = r["result"]
        self.assertIsInstance(pred, Prediction)
        self.assertEqual(len(pred.boxes), 2)
        self.assertTrue(all(len(b) == 4 and b[2] > b[0] for b in pred.boxes))

    def test_circularity_filter(self):
        """圆度过滤：细长条（圆度≈0.2）被滤掉，圆保留。"""
        m = np.zeros((100, 160), np.uint8)
        cv2.circle(m, (30, 50), 12, 255, -1)
        cv2.rectangle(m, (90, 46), (150, 54), 255, -1)  # 60×8 细条
        kw = {"threshold_mode": "otsu", "thresh": 128, "invert": False,
              "min_area": 10, "max_area": 0, "circularity": 0.0}
        n_all = len(node("blob_analysis").run({"image": img(m)}, kw, CTX())
                    ["result"].boxes)
        kept = node("blob_analysis").run({"image": img(m)},
                                         dict(kw, circularity=0.8), CTX())["result"]
        self.assertEqual(n_all, 2)
        self.assertEqual(len(kept.boxes), 1)
        x1, y1, x2, y2 = kept.boxes[0]
        self.assertLess(abs((x1 + x2) / 2 - 30), 2.0)   # 留下的是圆

    def test_invert_finds_dark_blobs(self):
        m = 255 - self._scene()  # 亮底暗斑
        kw = {"threshold_mode": "otsu", "thresh": 128, "min_area": 10,
              "max_area": 0, "circularity": 0.0}
        # 不反转：白色背景整体成为前景（无法得到两个暗斑）
        self.assertNotEqual(len(node("blob_analysis").run(
            {"image": img(m)}, kw, CTX())["result"].boxes), 2)
        r = node("blob_analysis").run({"image": img(m)}, dict(kw, invert=True), CTX())
        self.assertEqual(len(r["result"].boxes), 2)


class FindLineTest(unittest.TestCase):
    def test_edge_horizontal_subpixel(self):
        """水平台阶边缘 y=60：竖直卡尺 → 端点误差 ≤1px。"""
        m = np.zeros((100, 120), np.uint8)
        m[60:, :] = 200
        m = cv2.GaussianBlur(m, (3, 3), 0)  # 轻微平滑模拟真实成像
        r = node("find_line").run(
            {"image": img(m)},
            {"target": "edge", "scan": "vertical", "roi": "",
             "num_calipers": 24, "polarity": "bright", "min_contrast": 20},
            CTX())
        line = r["result"]
        self.assertEqual(line.kind, "line")
        pts = line.data
        self.assertTrue(np.all(np.abs(pts[:, 1] - 59.5) <= 1.0),
                        f"端点 y 偏差超限: {pts[:, 1]}")
        self.assertLess(pts[0, 0], pts[1, 0])  # 端点沿 x 展开

    def test_line_mode_bright_on_dark(self):
        """亮线中心 y=50（target=line）：中心误差 ≤1px。"""
        m = np.zeros((100, 100), np.uint8)
        cv2.line(m, (5, 50), (95, 50), 255, 1, cv2.LINE_AA)
        r = node("find_line").run(
            {"image": img(m)},
            {"target": "line", "scan": "vertical", "roi": "",
             "num_calipers": 20, "polarity": "bright", "min_contrast": 20},
            CTX())
        self.assertTrue(np.all(np.abs(r["result"].data[:, 1] - 50.0) <= 1.0),
                        f"线中心偏差超限: {r['result'].data[:, 1]}")

    def test_flat_image_raises(self):
        m = np.full((60, 60), 88, np.uint8)
        with self.assertRaises(ValueError):
            node("find_line").run(
                {"image": img(m)},
                {"target": "edge", "scan": "vertical", "roi": "",
                 "num_calipers": 8, "polarity": "any", "min_contrast": 20}, CTX())

    def test_residual_output(self):
        m = np.zeros((80, 80), np.uint8)
        m[40:, :] = 180
        r = node("find_line").run(
            {"image": img(m)},
            {"target": "edge", "scan": "vertical", "roi": "",
             "num_calipers": 12, "polarity": "bright", "min_contrast": 20}, CTX())
        self.assertLessEqual(r["residual"].value, 1.0)


class FindCircleTest(unittest.TestCase):
    @staticmethod
    def _aa_circle(h, w, cx, cy, r):
        """解析式抗锯齿圆：覆盖度中点严格在 r（cv2.circle 的 AA 栅格化
        有 ~+0.5px 覆盖偏置，不能作亚像素真值）。"""
        yy, xx = np.mgrid[:h, :w]
        dist = np.hypot(xx - cx, yy - cy)
        return (np.clip(r + 0.5 - dist, 0.0, 1.0) * 255).astype(np.uint8)

    def test_center_and_radius_tolerance(self):
        """圆心 (90,80) r=45：圆心误差 ≤0.5px、半径误差 ≤1%（0.45px）。"""
        m = self._aa_circle(170, 190, 90.0, 80.0, 45.0)
        r = node("find_circle").run(
            {"image": img(m)},
            {"roi": "", "num_rays": 36, "polarity": "bright",
             "min_contrast": 20}, CTX())
        c = r["result"].data
        self.assertEqual(r["result"].kind, "circle")
        self.assertLessEqual(abs(c[0] - 90.0), 0.5, f"圆心 x 误差 {abs(c[0]-90):.3f}")
        self.assertLessEqual(abs(c[1] - 80.0), 0.5, f"圆心 y 误差 {abs(c[1]-80):.3f}")
        self.assertLessEqual(abs(c[2] - 45.0) / 45.0, 0.01,
                             f"半径误差 {abs(c[2]-45):.3f}px")

    def test_roi_excludes_neighbor(self):
        """邻圆干扰：ROI 限定后拟合目标圆。"""
        m = np.zeros((180, 180), np.uint8)
        cv2.circle(m, (60, 90), 30, 255, -1, cv2.LINE_AA)   # 目标
        cv2.circle(m, (150, 40), 30, 255, -1, cv2.LINE_AA)  # 干扰
        r = node("find_circle").run(
            {"image": img(m)},
            {"roi": "10,50,110,140", "num_rays": 36, "polarity": "bright",
             "min_contrast": 20}, CTX())
        c = r["result"].data
        self.assertLess(abs(c[0] - 60.0), 1.5)
        self.assertLess(abs(c[1] - 90.0), 1.5)

    def test_too_small_roi_raises(self):
        m = np.full((40, 40), 10, np.uint8)
        with self.assertRaises(ValueError):
            node("find_circle").run(
                {"image": img(m)},
                {"roi": "19,19,21,21", "num_rays": 12, "polarity": "any",
                 "min_contrast": 20}, CTX())


class FindCornerTest(unittest.TestCase):
    def test_harris_finds_rect_corners(self):
        """两个亮矩形的四类角点：每个期望角点 2px 内必有检出。"""
        m = np.zeros((120, 140), np.uint8)
        cv2.rectangle(m, (30, 30), (60, 55), 255, -1)
        cv2.rectangle(m, (85, 65), (115, 95), 255, -1)
        r = node("find_corner").run(
            {"image": img(m)},
            {"mode": "harris", "roi": "", "max_corners": 10,
             "quality": 0.1, "min_dist": 8}, CTX())
        pts = r["result"].data
        self.assertGreaterEqual(len(pts), 4)
        expected = [(30, 30), (60, 55), (85, 65), (115, 95)]
        for ex, ey in expected:
            d = np.hypot(pts[:, 0] - ex, pts[:, 1] - ey).min()
            self.assertLessEqual(d, 2.0, f"角点({ex},{ey}) 最近检出距离 {d:.2f}")

    def test_caliper_cross_single_point(self):
        """L 形亮区角点 (60,50)：单点、误差 ≤1.5px。"""
        m = np.zeros((100, 100), np.uint8)
        m[:50, :60] = 255
        m = cv2.GaussianBlur(m, (3, 3), 0)
        r = node("find_corner").run(
            {"image": img(m)},
            {"mode": "caliper_cross", "roi": "", "max_corners": 1,
             "quality": 0.1, "min_dist": 10}, CTX())
        pts = r["result"].data
        self.assertEqual(len(pts), 1)
        self.assertLess(abs(pts[0, 0] - 59.5), 1.5, f"x={pts[0, 0]}")
        self.assertLess(abs(pts[0, 1] - 49.5), 1.5, f"y={pts[0, 1]}")


class TemplateMatchTest(unittest.TestCase):
    TMP_DIR = tempfile.TemporaryDirectory()

    def _make_template(self):
        rng = np.random.default_rng(7)
        t = rng.integers(60, 200, size=(40, 56), dtype=np.uint8)
        return t

    def _scene_with(self, t, positions, size=(300, 280), rotated_angle=0.0):
        """把模板贴进噪声场景；rotated_angle 时先旋转再贴（黑底场景）。"""
        rng = np.random.default_rng(3)
        scene = rng.integers(0, 40, size=(size[0], size[1]), dtype=np.uint8)
        tw, thh = t.shape[1], t.shape[0]
        for (cx, cy) in positions:
            piece = t
            offx, offy = tw / 2.0, thh / 2.0
            if rotated_angle:
                diag = int(math.ceil(math.hypot(thh, tw)))
                m = cv2.getRotationMatrix2D((tw / 2.0, thh / 2.0), rotated_angle, 1.0)
                m[0, 2] += diag / 2.0 - tw / 2.0
                m[1, 2] += diag / 2.0 - thh / 2.0
                piece = cv2.warpAffine(t, m, (diag, diag))
                offx = offy = diag / 2.0
            x0, y0 = int(cx - offx), int(cy - offy)
            scene[y0:y0 + piece.shape[0], x0:x0 + piece.shape[1]] = piece
        return scene

    def _save_tmpl(self, t) -> str:
        p = Path(self.TMP_DIR.name) / "tmpl_line.png"
        imwrite_unicode(p, t)
        return str(p)

    def test_single_match_angle0(self):
        t = self._make_template()
        scene = self._scene_with(t, [(150, 140)])
        r = node("template_match").run(
            {"image": img(scene)},
            {"template_path": self._save_tmpl(t), "roi": "",
             "angle_range": 10.0, "angle_step": 5.0, "max_matches": 1,
             "threshold": 0.8}, CTX())
        rect = r["result"].data
        self.assertEqual(r["result"].kind, "rotated_rect")
        self.assertLess(abs(rect[0] - 150.0), 2.0, f"cx={rect[0]}")
        self.assertLess(abs(rect[1] - 140.0), 2.0, f"cy={rect[1]}")
        self.assertEqual(rect[2], 56)
        self.assertEqual(rect[3], 40)
        self.assertLess(abs(rect[4]), 3.0)
        self.assertGreaterEqual(r["score"].value, 0.8)

    def test_rotated_match(self):
        """模板在场景中被旋转 10°：角度/位置容差内找回。"""
        t = self._make_template()
        scene = self._scene_with(t, [(140, 150)], rotated_angle=10.0)
        r = node("template_match").run(
            {"image": img(scene)},
            {"template_path": self._save_tmpl(t), "roi": "",
             "angle_range": 15.0, "angle_step": 5.0, "max_matches": 1,
             "threshold": 0.6}, CTX())
        rect = r["result"].data
        self.assertLess(abs(rect[0] - 140.0), 3.0, f"cx={rect[0]}")
        self.assertLess(abs(rect[1] - 150.0), 3.0, f"cy={rect[1]}")
        self.assertLess(abs(rect[4] - 10.0), 4.0, f"angle={rect[4]}")

    def test_multiple_matches(self):
        t = self._make_template()
        scene = self._scene_with(t, [(80, 80), (200, 190)])
        r = node("template_match").run(
            {"image": img(scene)},
            {"template_path": self._save_tmpl(t), "roi": "",
             "angle_range": 0.0, "angle_step": 0.0, "max_matches": 2,
             "threshold": 0.8}, CTX())
        rect = r["result"].data   # 结果端口=最优匹配旋转矩形（多目标取最优）
        d = min(np.hypot(rect[0] - ex, rect[1] - ey) for ex, ey in [(80, 80), (200, 190)])
        self.assertLessEqual(d, 2.0, f"最优匹配偏差 {d:.2f}")
        self.assertGreaterEqual(r["score"].value, 0.8)

    def test_below_threshold_raises(self):
        rng = np.random.default_rng(11)
        scene = rng.integers(0, 255, size=(200, 200), dtype=np.uint8)
        t = self._make_template()
        with self.assertRaises(ValueError):
            node("template_match").run(
                {"image": img(scene)},
                {"template_path": self._save_tmpl(t), "roi": "",
                 "angle_range": 0.0, "angle_step": 0.0, "max_matches": 1,
                 "threshold": 0.99}, CTX())

    def test_large_image_degradation(self):
        """大图降级路径：2400×1500 场景，金字塔粗匹配 + 局部精匹配。"""
        t = self._make_template()
        rng = np.random.default_rng(5)
        scene = rng.integers(0, 40, size=(1500, 2400), dtype=np.uint8)
        scene[700:740, 1200:1256] = t
        r = node("template_match").run(
            {"image": img(scene)},
            {"template_path": self._save_tmpl(t), "roi": "",
             "angle_range": 0.0, "angle_step": 0.0, "max_matches": 1,
             "threshold": 0.8}, CTX())
        rect = r["result"].data
        self.assertLess(abs(rect[0] - 1228.0), 2.0, f"cx={rect[0]}")
        self.assertLess(abs(rect[1] - 720.0), 2.0, f"cy={rect[1]}")

    def test_record_saves_template(self):
        t = self._make_template()
        scene = self._scene_with(t, [(100, 100)])
        out = Path(self.TMP_DIR.name) / "流程模板"
        p = node("template_match").record(scene, (72, 80, 128, 120), out)
        self.assertTrue(Path(p).is_file())
        from xccv.engine.io_utils import imread_unicode

        back = imread_unicode(p)
        if back.ndim == 3:  # imread_unicode 固定 IMREAD_COLOR 读回
            back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        self.assertEqual(back.shape, t.shape)
        self.assertLess(float(np.abs(back.astype(int) - t.astype(int)).mean()), 1.0)

    def test_missing_template_raises(self):
        with self.assertRaises(ValueError):
            node("template_match").run(
                {"image": img(np.zeros((50, 50), np.uint8))},
                {"template_path": "", "roi": "", "angle_range": 0.0,
                 "angle_step": 0.0, "max_matches": 1, "threshold": 0.8}, CTX())


class RegistrySmokeTest(unittest.TestCase):
    def test_locate_nodes_registered_with_geometry_contracts(self):
        """定位分类 5 算子齐备；几何输出形状符合 IR 契约（注册即校验端口类型）。"""
        from xccv.engine.flow.registry import get_node_class, list_nodes

        names = {n["name"] for n in list_nodes(category="定位")}
        # 原 5 算子齐备（V0.6 P4 起本组另有 fit_rectangle/find_line_pair，故断言子集）
        self.assertLessEqual({"blob_analysis", "find_line", "find_circle",
                              "find_corner", "template_match"}, names)
        line = Geometry(kind="line", data=np.zeros((2, 2)))
        circle = Geometry(kind="circle", data=np.zeros(3))
        rect = Geometry(kind="rotated_rect", data=np.zeros(5))
        for g, shape in ((line, (2, 2)), (circle, (3,)), (rect, (5,))):
            self.assertEqual(np.asarray(g.data).shape, shape)

# == 来源模块：test_flow_ops_logic ==
"""逻辑 4 算子测试（每算子 ≥2 例）。"""

import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import xccv.engine.flow.ops.logic  # noqa: F401  触发注册
from xccv.engine.flow.ir import ImageData, MeasureResult, NumberVal
from xccv.engine.flow.node_base import RunContext

CTX__flow_ops_logic = lambda: RunContext()  # noqa: E731


class CompareValueTest(unittest.TestCase):
    def test_gt_pass_and_fail(self):
        node = pp_compare()
        self.assertTrue(node.run({"number": NumberVal(5.0)},
                                 {"op": "gt", "threshold": 1.0, "hysteresis": 0},
                                 CTX__flow_ops_logic())["result"])
        self.assertFalse(node.run({"number": NumberVal(-1.0)},
                                  {"op": "gt", "threshold": 0.0, "hysteresis": 0},
                                  CTX__flow_ops_logic())["result"])

    def test_nan_fails_with_warning(self):
        """除零链路的 NaN：恒判 False 并在人话警告中说明（防静默通过）。"""
        ctx = CTX__flow_ops_logic()
        passed = pp_compare().run({"number": NumberVal(float("nan"))},
                                  {"op": "ge", "threshold": 0.0, "hysteresis": 0},
                                  ctx)["result"]
        self.assertFalse(passed)
        self.assertTrue(any("NaN" in w for w in ctx.warnings))

    def test_hysteresis_holds_state(self):
        """回差：触发需超阈+回差，保持只需不跌破阈-回差（会话状态）。"""
        ctx = CTX__flow_ops_logic()
        node = pp_compare()
        kw = {"op": "gt", "threshold": 10.0, "hysteresis": 2.0}
        self.assertFalse(node.run({"number": NumberVal(11.0)}, kw, ctx)["result"])
        self.assertTrue(node.run({"number": NumberVal(13.0)}, kw, ctx)["result"])
        # 已触发态下 9.0：未跌破 10-2=8 → 保持通过（无回差时 9<10 会翻为 False）
        self.assertTrue(node.run({"number": NumberVal(9.0)}, kw, ctx)["result"])


def pp_compare():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("compare_value")()


class MathOpTest(unittest.TestCase):
    def test_add_and_sub(self):
        node = pp_math()
        out = node.run({"a": NumberVal(7, "mm"), "b": NumberVal(2, "mm")},
                       {"op": "add"}, CTX__flow_ops_logic())["number"]
        self.assertEqual(out.value, 9.0)
        self.assertEqual(out.unit, "mm")
        out2 = node.run({"a": NumberVal(7), "b": NumberVal(2)},
                        {"op": "sub"}, CTX__flow_ops_logic())["number"]
        self.assertEqual(out2.value, 5.0)

    def test_div_by_zero_outputs_nan_with_warning(self):
        """除零不静默：输出 NaN + ctx 警告（下游 compare 恒 False）。"""
        ctx = CTX__flow_ops_logic()
        out = pp_math().run({"a": NumberVal(3), "b": NumberVal(0)},
                            {"op": "div"}, ctx)["number"]
        self.assertTrue(math.isnan(out.value))
        self.assertTrue(any("除零" in w for w in ctx.warnings))

    def test_abs_without_b(self):
        """abs 只需 a（b 端口可选）：缺 b 不报错。"""
        out = pp_math().run({"a": NumberVal(-4)}, {"op": "abs"}, CTX__flow_ops_logic())["number"]
        self.assertEqual(out.value, 4.0)


def pp_math():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("math_op")()


class ConditionBranchTest(unittest.TestCase):
    def _imgs(self):
        a = ImageData(mat=np.zeros((4, 4), np.uint8))
        b = ImageData(mat=np.full((4, 4), 9, np.uint8))
        return a, b

    def test_selects_true_input(self):
        a, b = self._imgs()
        out = pp_branch().run({"cond": True, "true_input": a, "false_input": b},
                              {}, CTX__flow_ops_logic())["image"]
        self.assertIs(out, a)  # 零拷贝直通

    def test_selects_false_input(self):
        a, b = self._imgs()
        out = pp_branch().run({"cond": False, "true_input": a, "false_input": b},
                              {}, CTX__flow_ops_logic())["image"]
        self.assertIs(out, b)


def pp_branch():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("condition_branch")()


class ResultMergeTest(unittest.TestCase):
    def _m(self, value=None, values=None, passed=None):
        return MeasureResult(value=value, values=values, passed=passed)

    def test_allpass_and_anypass(self):
        node = pp_merge()
        ok, ng = self._m(value=1.0, passed=True), self._m(value=0.5, passed=False)
        r1 = node.run({"m1": ok, "m2": ok}, {"op": "allpass"}, CTX__flow_ops_logic())
        self.assertTrue(r1["result"])
        r2 = node.run({"m1": ok, "m2": ng}, {"op": "allpass"}, CTX__flow_ops_logic())
        self.assertFalse(r2["result"])
        r3 = node.run({"m1": ok, "m2": ng}, {"op": "anypass"}, CTX__flow_ops_logic())
        self.assertTrue(r3["result"])

    def test_count_ng(self):
        node = pp_merge()
        r = node.run({"m1": self._m(passed=True),
                      "m2": self._m(passed=False),
                      "m3": self._m(passed=False)}, {"op": "count_ng"}, CTX__flow_ops_logic())
        self.assertEqual(r["number"].value, 2.0)
        self.assertFalse(r["result"])

    def test_mean_over_multi_point_measures(self):
        """多点测量（values 序列）参与聚合。"""
        node = pp_merge()
        r = node.run({"m1": self._m(values=[2.0, 4.0]),
                      "m2": self._m(value=6.0)}, {"op": "mean"}, CTX__flow_ops_logic())
        self.assertEqual(r["number"].value, 4.0)

    def test_empty_inputs_nan_with_warning(self):
        ctx = CTX__flow_ops_logic()
        r = node_run_empty_merge(ctx)
        self.assertTrue(math.isnan(r["number"].value))
        self.assertTrue(any("未收集" in w for w in ctx.warnings))


def node_run_empty_merge(ctx):
    return pp_merge().run({"m1": None}, {"op": "mean"}, ctx)


def pp_merge():
    from xccv.engine.flow.registry import get_node_class

    return get_node_class("result_merge")()

if __name__ == '__main__':
    unittest.main()
