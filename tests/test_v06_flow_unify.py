"""回归：流程图端口彻底统一。

- image_list 端口类型取消：PORT_TYPES 只余 image/result/number，全库节点
  无 image_list 端口（多张集合走 image 端口数据载体 ImageList）；
- 并行输入自动映射（executor._map_over）：多张集合过任意非 HANDLES_LIST
  节点逐张执行；标量输入广播；输出重组 ImageList/ResultList；
  图像项 meta 继承输入项（roi_rect 沿链传递）；结果集合解包 ResultItem.value；
  张数不一致报人话错误；
- 主结果端口标签统一「结果」（定位/测量/AI/过滤族），特殊项数值端口保留专名；
- 结果合并生成即裁剪（前端 onAdded，schema 侧断言 HANDLES_LIST 与迁移）；
- 旧图迁移 schema 3：list_pick 节点连同连线删除。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from xccv.engine.flow.executor import Executor
from xccv.engine.flow.ir import (PORT_TYPES, Geometry, ImageData, ImageList,
                                 NumberVal, ResultItem, ResultList)
from xccv.engine.flow.node_base import RunContext, VisionNode
from xccv.engine.flow.port_migrate import SCHEMA, migrate_ports
from xccv.engine.flow.registry import get_node_class, list_nodes
from xccv.modeltools.flow_session import ensure_ops_registered

ensure_ops_registered()


def _lib():
    return {info["name"]: info for info in list_nodes()}


class PortUnifyTest(unittest.TestCase):
    """端口类型与标签统一（批注二.1/二.2/二.6）。"""

    def test_port_types_three(self):
        self.assertEqual(set(PORT_TYPES), {"image", "result", "number"})

    def test_no_image_list_ports(self):
        for info in list_nodes():
            for p in info["inputs"] + info["outputs"]:
                self.assertNotEqual(p["type"], "image_list",
                                    (info["name"], p["name"]))

    def test_primary_result_labels(self):
        """主结果输出统一「结果」；特殊项数值端口保留专名。"""
        lib = _lib()
        for name in ("find_line", "find_circle", "find_corner", "template_match",
                     "fit_rectangle", "find_line_pair", "blob_analysis",
                     "measure_distance", "measure_angle", "detection_to_measure",
                     "model_detection", "merge_results",
                     "model_on_list", "op_on_list", "mask_to_box",
                     "filter_overlap", "filter_topk", "relabel_map"):
            outs = lib[name]["outputs"]
            r = next(p for p in outs if p["name"] == "result")
            self.assertEqual(r["label"], "结果", name)
        # 特殊项（+特殊项口径）：数值端口保留专名
        lib2 = {info["name"]: info for info in list_nodes()}
        self.assertEqual(next(p for p in lib2["find_line"]["outputs"]
                              if p["name"] == "residual")["label"], "拟合残差")
        self.assertEqual(next(p for p in lib2["template_match"]["outputs"]
                              if p["name"] == "score")["label"], "匹配得分")

    def test_all_ports_chinese_labels(self):
        """全部端口有中文 label（画布不再出现 image/number 英文裸名，批注二.2/二.6）。"""
        import re

        for info in list_nodes():
            if info["name"].startswith("custom/"):
                continue   # 自定义算子端口名用户自定，豁免
            for p in info["inputs"] + info["outputs"]:
                self.assertTrue(re.match(r"^[\u4e00-\u9fff]", p["label"]),
                                (info["name"], p["name"], p["label"]))

    def test_native_list_nodes_flagged(self):
        for name in ("merge_results", "model_on_list", "op_on_list"):
            self.assertTrue(get_node_class(name).HANDLES_LIST, name)
        for name in ("gaussian_blur", "find_circle", "compare_value", "math_op",
                     "crop_by_detection_all"):
            self.assertFalse(get_node_class(name).HANDLES_LIST, name)

    def test_list_pick_unregistered(self):
        self.assertIsNone(get_node_class("list_pick"))


def _tiles():
    """三张 50x40 亮圆裁图（局部坐标同 (25,20)），roi 依次右移 100px。"""
    import cv2

    items = []
    for k in range(3):
        m = np.zeros((40, 50), np.uint8)
        cv2.circle(m, (25, 20), 12, 230, -1)
        items.append(ImageData(mat=cv2.cvtColor(m, cv2.COLOR_GRAY2BGR),
                               meta={"roi_rect": [k * 100, 0, k * 100 + 50, 40],
                                     "source_size": [300, 40]}))
    return ImageList(items=items, meta={"source_size": [300, 40]})


class AutoMapTest(unittest.TestCase):
    """并行输入自动映射（批注二.4 的执行器语义）。"""

    def _run_graph(self, nodes, links, external=None):
        res = Executor({"nodes": nodes, "links": links}, RunContext(),
                       external=external or {}).run()
        errs = {k: r.error for k, r in res.items() if not r.ok}
        self.assertFalse(errs, errs)
        return res

    def test_blur_over_tiles_meta_inherited(self):
        g = {"nodes": [{"id": 1, "type": "gaussian_blur", "properties": {}}],
             "links": []}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): _tiles()})
        out = res[1].outputs["image"]
        self.assertIsInstance(out, ImageList)
        self.assertEqual(len(out), 3)
        self.assertEqual(out.items[1].meta["roi_rect"], [100, 0, 150, 40])
        self.assertEqual(out.items[1].meta["source_size"], [300, 40])

    def test_locate_over_tiles_then_merge_offsets(self):
        """多张过定位节点：结果集合带 roi，合并回原图坐标逐张偏移。"""
        g = {"nodes": [
            {"id": 1, "type": "gaussian_blur", "properties": {}},
            {"id": 2, "type": "find_circle", "properties": {"polarity": "bright"}},
            {"id": 3, "type": "merge_results", "properties": {"nms_iou": 0.0}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): _tiles()})
        rl = res[2].outputs["result"]
        self.assertIsInstance(rl, ResultList)
        self.assertEqual([it.roi_rect for it in rl.items],
                         [[0, 0, 50, 40], [100, 0, 150, 40], [200, 0, 250, 40]])
        boxes = res[3].outputs["result"].boxes
        self.assertEqual(len(boxes), 3)
        for k, b in enumerate(boxes):
            self.assertAlmostEqual((b[0] + b[2]) / 2, 100 * k + 25, delta=1.5)
            self.assertAlmostEqual((b[1] + b[3]) / 2, 20, delta=1.5)

    def test_result_list_unwrapped_to_values(self):
        """结果集合逐张映射解包 ResultItem.value（节点拿原值而非壳）。"""
        rl = ResultList(items=[
            ResultItem("number", NumberVal(value=float(v)), meta={"index": k})
            for k, v in enumerate((2.0, 8.0))])
        g = {"nodes": [{"id": 1, "type": "compare_value",
                        "properties": {"op": "gt", "threshold": 5.0}}],
             "links": []}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): rl})
        out = res[1].outputs["result"]
        self.assertIsInstance(out, ResultList)
        self.assertEqual([it.value for it in out.items], [False, True])
        self.assertEqual([it.kind for it in out.items], ["bool", "bool"])

    def test_math_over_result_list(self):
        rl = ResultList(items=[
            ResultItem("number", NumberVal(value=10.0), meta={"index": 0}),
            ResultItem("number", NumberVal(value=30.0), meta={"index": 1})])
        g = {"nodes": [{"id": 1, "type": "math_op", "properties": {"op": "div"}}],
             "links": []}
        res = Executor(g, RunContext(),
                       external={(1, 0): rl, (1, 1): NumberVal(value=2.0)}).run()
        out = res[1].outputs["number"]
        self.assertIsInstance(out, ResultList)
        self.assertEqual([it.value.value for it in out.items], [5.0, 15.0])

    def test_length_mismatch_raises(self):
        a = _tiles()
        b = ImageList(items=a.items[:2])
        g = {"nodes": [{"id": 1, "type": "condition_branch", "properties": {}}],
             "links": []}
        ex = Executor(g, RunContext(),
                      external={(1, 0): True, (1, 1): a, (1, 2): b})
        res = ex.run()
        self.assertFalse(res[1].ok)
        self.assertIn("张数不一致", res[1].error)

    def test_single_image_untouched(self):
        """单张（非集合）输入不走映射，行为与旧版一致。"""
        img = ImageData(mat=np.full((40, 50, 3), 90, np.uint8))
        g = {"nodes": [{"id": 1, "type": "gaussian_blur", "properties": {}}],
             "links": []}
        res = self._run_graph(g["nodes"], g["links"], external={(1, 0): img})
        out = res[1].outputs["image"]
        self.assertIsInstance(out, ImageData)
        self.assertEqual(out.mat.shape, (40, 50, 3))

    def test_model_on_list_accepts_single(self):
        """原生集合节点收到单张 ImageData 也兼容（包成 1 张集合）。"""
        from xccv.engine.flow.ops.listops import _as_list

        img = ImageData(mat=np.zeros((4, 4, 3), np.uint8))
        lst = _as_list(img)
        self.assertEqual(len(lst), 1)
        self.assertIs(lst.items[0], img)


class MigrateSchema3Test(unittest.TestCase):
    """旧图迁移 schema 3：list_pick 连点带线删除（批注二.4）。"""

    def test_list_pick_dropped(self):
        flow = {"schema": 2, "last_node_id": 3, "last_link_id": 2,
                "nodes": [
                    {"id": 1, "type": "xccv2/grid_tile_all", "properties": {}},
                    {"id": 2, "type": "xccv2/list_pick", "properties": {"index": 0}},
                    {"id": 3, "type": "xccv2/gaussian_blur", "properties": {}}],
                "links": [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0]]}
        out = migrate_ports(flow)
        self.assertEqual(SCHEMA, 3)
        self.assertEqual(out["schema"], 3)
        self.assertEqual({n["type"] for n in out["nodes"]},
                         {"xccv2/grid_tile_all", "xccv2/gaussian_blur"})
        self.assertEqual(out["links"], [])
        # 原图不被修改（浅拷贝纪律）
        self.assertEqual(len(flow["nodes"]), 3)
        self.assertEqual(len(flow["links"]), 2)

    def test_schema3_untouched(self):
        flow = {"schema": 3, "nodes": [], "links": []}
        self.assertIs(migrate_ports(flow), flow)


if __name__ == "__main__":
    unittest.main()
