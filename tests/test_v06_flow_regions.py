"""检出过滤区域解析器（矩形 / 圆 / 多边形，旧串零改动兼容）。

- 旧串回归：随机矩形（归一化/像素/坐标颠倒/中文逗号/坏条目混杂）与 V0.5.6
  旧实现逐点判定一致；
- 新形状：circle（归一化半径按短边、像素）与 polygon 命中/未命中/边界；
- 混合区域串、apply_flow 的屏蔽/保留区域端到端；
- 打包严格校验：旧文案逐字不变 + 新形状错误文案；
- 过滤节点 validate（流程图打包口径）接受新形状。
"""

import random
import unittest

import numpy as np

from xccv.engine.flow import regions as rg


def _legacy_parse(value, w, h):
    """V0.5.6 detection_flow._parse_regions 原样（参考实现）。"""
    out = []
    if isinstance(value, str):
        entries = [s for s in value.split(";") if s.strip()]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        return out
    for e in entries:
        try:
            if isinstance(e, str):
                nums = [float(v) for v in e.replace("，", ",").split(",") if v.strip()]
            elif isinstance(e, (list, tuple)):
                nums = [float(v) for v in e]
            else:
                continue
            if len(nums) != 4:
                continue
            x1, y1, x2, y2 = nums
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0 + 1e-9:
                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            out.append([x1, y1, x2, y2])
        except (TypeError, ValueError):
            continue
    return out


def _legacy_in_any(cx, cy, regions, w, h):
    for x1, y1, x2, y2 in _legacy_parse(regions, w, h):
        if x1 * w <= cx <= x2 * w and y1 * h <= cy <= y2 * h:
            return True
    return False


class LegacyRectCompatTest(unittest.TestCase):
    def test_random_legacy_strings_match_old_impl(self):
        rnd = random.Random(20261102)
        w, h = 640, 480
        for _ in range(300):
            parts = []
            for _k in range(rnd.randint(1, 3)):
                if rnd.random() < 0.5:  # 归一化
                    vals = [round(rnd.random(), 3) for _ in range(4)]
                else:  # 像素（含颠倒坐标）
                    vals = [rnd.randint(0, w), rnd.randint(0, h),
                            rnd.randint(0, w), rnd.randint(0, h)]
                sep = "，" if rnd.random() < 0.2 else ","
                parts.append(sep.join(str(v) for v in vals))
            if rnd.random() < 0.2:
                parts.append("a,b,c,d")      # 坏条目：两边都应跳过
            if rnd.random() < 0.2:
                parts.append("1,2,3")        # 坏条目
            value = ";".join(parts)
            for _p in range(10):
                cx, cy = rnd.uniform(0, w), rnd.uniform(0, h)
                self.assertEqual(rg.any_contains(value, cx, cy, w, h),
                                 _legacy_in_any(cx, cy, value, w, h), value)

    def test_list_form_and_boundaries_match(self):
        w, h = 100, 50
        value = [[0.1, 0.2, 0.5, 0.6], [80, 10, 60, 40]]   # 第二条像素+颠倒
        for cx, cy in [(10, 10), (50, 30), (60, 10), (80, 40), (70, 25), (0, 0), (99, 49)]:
            self.assertEqual(rg.any_contains(value, cx, cy, w, h),
                             _legacy_in_any(cx, cy, value, w, h), (cx, cy))

    def test_rect_prefix_equals_bare(self):
        w, h = 200, 100
        for cx, cy in [(20, 20), (100, 50), (150, 90)]:
            self.assertEqual(rg.any_contains("rect:0.1,0.1,0.6,0.6", cx, cy, w, h),
                             rg.any_contains("0.1,0.1,0.6,0.6", cx, cy, w, h))


class NewShapesTest(unittest.TestCase):
    def test_circle_normalized_radius_by_short_side(self):
        w, h = 200, 100       # 短边 100 → r=0.2 即 20px
        region = "circle:0.5,0.5,0.2"
        self.assertTrue(rg.any_contains(region, 100, 50, w, h))      # 圆心
        self.assertTrue(rg.any_contains(region, 120, 50, w, h))      # 边界（距 20）
        self.assertFalse(rg.any_contains(region, 121, 50, w, h))     # 刚出界
        self.assertFalse(rg.any_contains(region, 100, 75, w, h))

    def test_circle_pixel(self):
        region = "circle:300,200,50"
        self.assertTrue(rg.any_contains(region, 330, 230, 640, 480))  # 距 42.4
        self.assertFalse(rg.any_contains(region, 340, 240, 640, 480))  # 距 56.6

    def test_polygon_triangle_hit_miss_boundary(self):
        region = "polygon:10,10,110,10,10,110"  # 直角三角形（像素）
        w, h = 200, 200
        self.assertTrue(rg.any_contains(region, 30, 30, w, h))
        self.assertTrue(rg.any_contains(region, 60, 10, w, h))       # 边上
        self.assertFalse(rg.any_contains(region, 90, 90, w, h))      # 斜边外侧
        self.assertFalse(rg.any_contains(region, 5, 5, w, h))

    def test_polygon_normalized(self):
        region = "polygon:0,0,1,0,1,0.5"
        self.assertTrue(rg.any_contains(region, 90, 10, 100, 100))
        self.assertFalse(rg.any_contains(region, 10, 90, 100, 100))

    def test_mixed_entries_union(self):
        value = "0,0,0.1,0.1;circle:0.9,0.9,0.05;polygon:40,40,60,40,50,60"
        w, h = 100, 100
        self.assertTrue(rg.any_contains(value, 5, 5, w, h))
        self.assertTrue(rg.any_contains(value, 90, 90, w, h))
        self.assertTrue(rg.any_contains(value, 50, 45, w, h))
        self.assertFalse(rg.any_contains(value, 30, 70, w, h))

    def test_bad_new_shape_entries_skipped_at_runtime(self):
        value = "circle:1,2;polygon:1,1,2,2;ellipse:1,2,3;0,0,0.5,0.5"
        self.assertEqual(len(rg.parse_regions(value)), 1)
        self.assertTrue(rg.any_contains(value, 10, 10, 100, 100))


class ApplyFlowRegionShapesTest(unittest.TestCase):
    def _pred(self):
        from xccv.engine.models.base import Prediction

        # 三个检出：中心 (20,20)、(100,50)、(180,80)
        return Prediction(labels=[0, 0, 0], scores=[0.9, 0.8, 0.7],
                          boxes=[[10, 10, 30, 30], [90, 40, 110, 60], [170, 70, 190, 90]])

    def test_region_block_circle(self):
        from xccv.engine.detection_flow import apply_flow

        flow = {"nodes": [{"op": "region_block", "params": {"regions": "circle:100,50,15"}}]}
        out, info = apply_flow(self._pred(), (200, 100), flow)
        self.assertEqual(len(out.boxes), 2)
        self.assertEqual(info["removed"], 1)

    def test_region_keep_polygon(self):
        from xccv.engine.detection_flow import apply_flow

        flow = {"nodes": [{"op": "region_keep",
                           "params": {"regions": "polygon:0,0,120,0,120,100,0,100"}}]}
        out, _info = apply_flow(self._pred(), (200, 100), flow)
        self.assertEqual(len(out.boxes), 2)   # 第三个中心 x=180 在多边形外

    def test_legacy_rect_chain_unchanged(self):
        from xccv.engine.detection_flow import apply_flow

        flow = {"nodes": [{"op": "region_block", "params": {"regions": "0,0,0.25,0.5"}}]}
        out, _info = apply_flow(self._pred(), (200, 100), flow)
        self.assertEqual([b[0] for b in out.boxes], [90.0, 170.0])


class PackValidationTest(unittest.TestCase):
    def test_legacy_messages_verbatim(self):
        from xccv.engine.detection_flow import region_param_problems

        self.assertEqual(region_param_problems("1,2,3"),
                         ["第 1 条区域不是 4 个数字（应为 x1,y1,x2,y2）"])
        self.assertEqual(region_param_problems("5,5,5,5"), ["第 1 条区域宽或高为零/为负"])
        self.assertEqual(region_param_problems("a,b,c,d"), ["第 1 条区域不是有效的数字"])
        self.assertEqual(region_param_problems(12, "节点X："), ["节点X：区域参数类型无效"])
        self.assertEqual(region_param_problems(None), [])
        self.assertEqual(region_param_problems(0), [])

    def test_new_shape_problems(self):
        probs = rg.region_problems(
            "circle:1,2;circle:0.5,0.5,0;polygon:0,0,1,1;polygon:0,0,1,1,2,2;ellipse:1,2,3")
        self.assertEqual(len(probs), 5)
        self.assertIn("圆应为 3 个数字", probs[0])
        self.assertIn("半径", probs[1])
        self.assertIn("至少 3 个顶点", probs[2])
        self.assertIn("面积为零", probs[3])
        self.assertIn("未知区域形状", probs[4])

    def test_valid_new_shapes_pass(self):
        self.assertEqual(rg.region_problems(
            "rect:0.1,0.1,0.5,0.5;circle:0.5,0.5,0.2;polygon:0,0,10,0,5,8"), [])

    def test_filter_node_validate_accepts_new_shapes(self):
        from xccv.engine.flow.legacy_adapter import FilterRegionBlock

        node = FilterRegionBlock()
        self.assertEqual(node.validate({"regions": "circle:0.5,0.5,0.1"}), [])
        self.assertTrue(node.validate({"regions": "circle:0.5,0.5"}))


class RegionGeometryTest(unittest.TestCase):
    def test_to_pixels_shapes(self):
        r = rg.parse_regions("circle:0.5,0.5,0.25")[0]
        self.assertEqual(r.to_pixels(400, 200), (200.0, 100.0, 50.0))
        poly = rg.parse_regions("polygon:0,0,0.5,0,0.5,0.5")[0].to_pixels(100, 100)
        np.testing.assert_allclose(poly, [[0, 0], [50, 0], [50, 50]])


if __name__ == "__main__":
    unittest.main()

