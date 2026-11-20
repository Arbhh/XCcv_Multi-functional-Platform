"""V0.6 F2-2 回归：多图批处理 + 统一结果格式（7.3 清单与同日建议）。

- 列表生产：按检出全部裁图 / 网格全部分块（roi_rect、越界跳过、空列表、落盘、
  外扩像素、内置区域保留/屏蔽筛选）；单张裁图同样支持外扩像素与区域筛选；
- list_pick 已删除（并行输入下不再需要）；
- 结果合并（result_list → 原图坐标统一检出）：AI 框/掩码回映逐位断言、NMS 去重、
  按类别开关、分类→框、**传统几何（圆/旋转矩形/多边形/点）→ 外接框+掩码**、测量/数值
  记入 extra、空列表、无原图时用 source_size；
- 逐图执行传统算子 op_on_list（真 find_circle）：结果经合并回到原图坐标；参数错误中文报错；
- 端到端：载图 → 全部裁图 → 逐图推理（假模型）→ 合并：模型只加载 1 次、单张失败隔离、
  快照拼图、工作台预览；画布式链"检测 → 区域保留 → 全部裁图"可用；
- 工作台组合链：区域过滤 → 全部裁图（图像自动接原图）→ 逐图执行 → 合并 可串；
- image_list 端口取消——多张集合直接过任意节点（并行输入
  自动映射：逐张执行、输出重组集合、roi_rect 沿链传递）。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.engine.flow.ir import (
    Geometry, ImageData, ImageList, MeasureResult, NumberVal, ResultItem, ResultList,
)
from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.registry import get_node_class
from xccv.engine.models.base import Prediction
from xccv.modeltools.flow_session import ensure_ops_registered

ensure_ops_registered()   # 与产品同一路径：含区域/类别过滤等旧链薄委托节点

_APP = None


def _qapp():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


def _run(name, inputs, **params):
    cls = get_node_class(name)
    node = cls()
    full = {k: s.default for k, s in cls.PARAMS.items()}
    full.update(params)
    errs = node.validate(full)
    assert not errs, errs
    ctx = RunContext()
    return node.run(inputs, full, ctx), ctx


def _img(w=200, h=160):
    m = np.zeros((h, w, 3), np.uint8)
    m[:, :, 1] = np.arange(w, dtype=np.uint8)[None, :]   # 横向渐变：裁图位置可校验
    return ImageData(mat=m)


def _rl(items, **meta):
    return ResultList(items=items, meta=meta)


class ListProducersTest(unittest.TestCase):
    def test_crop_all_rois_and_pixels(self):
        img = _img()
        det = Prediction(labels=[0, 2], scores=[0.9, 0.7],
                         boxes=[[10, 20, 50, 60], [100, 30, 150, 90]])
        out, _ = _run("crop_by_detection_all", {"image": img, "result": det})
        lst = out["images"]
        self.assertIsInstance(lst, ImageList)
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst.items[0].meta["roi_rect"], [10, 20, 50, 60])
        self.assertEqual(lst.items[1].meta["roi_rect"], [100, 30, 150, 90])
        self.assertEqual(lst.items[1].meta["label_id"], 2)
        self.assertAlmostEqual(lst.items[0].meta["score"], 0.9)
        self.assertEqual(lst.items[0].mat.shape, (40, 40, 3))
        self.assertTrue(np.array_equal(lst.items[1].mat, img.mat[30:90, 100:150]))
        self.assertEqual(lst.meta["source_size"], [200, 160])

    def test_crop_all_limit_skip_and_empty(self):
        img = _img()
        det = Prediction(labels=[0, 0, 0], scores=[0.9, 0.8, 0.7],
                         boxes=[[10, 10, 30, 30], [500, 500, 600, 600], [40, 40, 60, 60]])
        out, ctx = _run("crop_by_detection_all", {"image": img, "result": det})
        self.assertEqual(len(out["images"]), 2)                 # 图外框跳过
        self.assertTrue(any("图外" in w for w in ctx.warnings))
        out, _ = _run("crop_by_detection_all", {"image": img, "result": det},
                      max_items=1)
        self.assertEqual(len(out["images"]), 1)
        out, ctx = _run("crop_by_detection_all", {"image": img, "result": Prediction()})
        self.assertEqual(len(out["images"]), 0)
        self.assertTrue(any("空列表" in w for w in ctx.warnings))

    def test_margin_px_and_pct_stack(self):
        img = _img()
        det = Prediction(labels=[0], scores=[0.9], boxes=[[50, 50, 70, 70]])
        out, _ = _run("crop_by_detection_all", {"image": img, "result": det}, margin_px=5)
        self.assertEqual(out["images"].items[0].meta["roi_rect"], [45, 45, 75, 75])
        out, _ = _run("crop_by_detection_all", {"image": img, "result": det},
                      margin_px=5, margin_pct=50.0)   # 20 → 20×2 + 10 = 50
        self.assertEqual(out["images"].items[0].meta["roi_rect"], [35, 35, 85, 85])
        out, _ = _run("crop_by_detection", {"image": img, "result": det}, margin_px=5)
        self.assertEqual(out["image"].meta["roi_rect"], [45, 45, 75, 75])

    def test_region_filter_keep_block(self):
        """裁图节点不再内置区域筛选——「保留/屏蔽区域」过滤后接裁图
        （保留原用例的 keep/block 两种口径，经过滤节点验证）。"""
        img = _img()
        det = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                         boxes=[[10, 10, 30, 30], [150, 100, 170, 120]])
        rf = get_node_class("filter_region_keep")
        full = {k: sp.default for k, sp in rf.PARAMS.items()}
        full.update(regions="0,0,100,80")
        ctx = RunContext()
        r = rf().run({"result": det, "image": img}, full, ctx)
        self.assertEqual(len(r["result"].boxes), 1)
        self.assertEqual(len(r["removed"].boxes), 1)
        self.assertTrue(any("移除" in w or "保留" in w for w in ctx.warnings))
        rb = get_node_class("filter_region_block")
        fullb = {k: sp.default for k, sp in rb.PARAMS.items()}
        fullb.update(regions="circle:160,110,15")
        r2 = rb().run({"result": det, "image": img}, fullb, RunContext())
        self.assertEqual(len(r2["result"].boxes), 1)
        # 过滤结果 → 全部裁图（det_index 与原检出对齐）
        out, _ = _run("crop_by_detection_all", {"image": img, "result": r["result"]})
        self.assertEqual([it.meta["det_index"] for it in out["images"].items], [0])

    def test_region_params_validated(self):
        """区域串语法校验在过滤节点上（打包口径严格）。"""
        from xccv.engine.flow.regions import region_problems

        errs = region_problems("circle:1,2", context="保留区域")
        self.assertTrue(errs and "保留区域" in errs[0], errs)

    def test_crop_all_save_dir_chinese(self):
        img = _img()
        det = Prediction(labels=[0], scores=[0.9], boxes=[[10, 20, 50, 60]])
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "裁图输出"
            out, _ = _run("crop_by_detection_all", {"image": img, "result": det},
                          save_dir=str(out_dir))
            self.assertTrue(Path(out["images"].items[0].meta["file"]).is_file())

    def test_grid_tile_all(self):
        img = _img()
        out, _ = _run("grid_tile_all", {"image": img}, rows=2, cols=2)
        lst = out["images"]
        self.assertEqual(len(lst), 4)
        self.assertEqual([it.meta["roi_rect"] for it in lst.items],
                         [[0, 0, 100, 80], [100, 0, 200, 80], [0, 80, 100, 160],
                          [100, 80, 200, 160]])
        out, _ = _run("grid_tile_all", {"image": img}, rows=2, cols=2, overlap_px=10)
        self.assertEqual(out["images"].items[0].meta["roi_rect"], [0, 0, 110, 90])
        self.assertEqual(out["images"].items[3].meta["roi_rect"], [90, 70, 200, 160])

    def test_list_pick_removed(self):
        """（定稿）取列表第k张节点已删除。"""
        self.assertIsNone(get_node_class("list_pick"))


class MergeResultsTest(unittest.TestCase):
    def test_ai_boxes_and_masks_mapped_back(self):
        m = np.zeros((40, 40), bool)
        m[2:12, 1:11] = True
        rl = _rl([ResultItem("detection", Prediction(labels=[1], scores=[0.8],
                                                     boxes=[[1, 2, 11, 12]], masks=[m]),
                             [10, 20, 50, 60]),
                  ResultItem("detection", Prediction(labels=[0], scores=[0.6],
                                                     boxes=[[5, 5, 25, 15]], masks=[None]),
                             [100, 30, 150, 90])])
        out, _ = _run("merge_results", {"image": _img(), "result": rl}, nms_iou=0.0)
        p = out["result"]
        self.assertEqual(p.boxes, [[11.0, 22.0, 21.0, 32.0], [105.0, 35.0, 125.0, 45.0]])
        self.assertEqual(p.labels, [1, 0])
        ref = np.zeros((160, 200), bool)
        ref[22:32, 11:21] = True
        self.assertTrue(np.array_equal(p.masks[0], ref))
        self.assertFalse(p.masks[1].any())          # 缺掩码项补空掩码（长度对齐）

    def test_nms_dedupes_overlap(self):
        rl = _rl([ResultItem("detection", Prediction(labels=[0], scores=[0.9],
                                                     boxes=[[90, 10, 110, 30]]), [0, 0, 110, 90]),
                  ResultItem("detection", Prediction(labels=[0], scores=[0.7],
                                                     boxes=[[0, 10, 21, 30]]), [90, 0, 200, 90])],
                 source_size=[200, 160])
        out, ctx = _run("merge_results", {"result": rl}, nms_iou=0.5)
        self.assertEqual(len(out["result"].boxes), 1)
        self.assertAlmostEqual(out["result"].scores[0], 0.9)
        self.assertTrue(any("去重" in w for w in ctx.warnings))
        out, _ = _run("merge_results", {"result": rl}, nms_iou=0.0)
        self.assertEqual(len(out["result"].boxes), 2)

    def test_class_aware_switch(self):
        rl = _rl([ResultItem("detection", Prediction(labels=[0, 1], scores=[0.9, 0.8],
                                                     boxes=[[0, 0, 20, 20], [1, 1, 20, 20]]))])
        out, _ = _run("merge_results", {"result": rl}, nms_iou=0.5, class_aware=True)
        self.assertEqual(len(out["result"].boxes), 2)
        out, _ = _run("merge_results", {"result": rl}, nms_iou=0.5, class_aware=False)
        self.assertEqual(len(out["result"].boxes), 1)

    def test_classification_items_become_boxes(self):
        rl = _rl([ResultItem("classification", Prediction(labels=[1], scores=[0.7],
                                                          probs=[0.3, 0.7]), [10, 20, 50, 60]),
                  ResultItem("classification", Prediction(labels=[0], scores=[0.9],
                                                          probs=[0.9, 0.1]), [100, 30, 150, 90])])
        out, _ = _run("merge_results", {"result": rl}, nms_iou=0.0)
        p = out["result"]
        self.assertEqual(p.boxes, [[100.0, 30.0, 150.0, 90.0], [10.0, 20.0, 50.0, 60.0]])
        self.assertEqual(p.labels, [0, 1])          # 按得分降序

    def test_traditional_geometry_unified(self):
        """传统几何（统一度量衡）：平移到原图 → 外接框 + 掩码，精确几何存 extra。"""
        rl = _rl([ResultItem("geometry", Geometry(kind="circle", data=np.array([20.0, 20.0, 10.0])),
                             [100, 50, 140, 90]),
                  ResultItem("geometry", Geometry(kind="point", data=np.array([[5.0, 6.0]])),
                             [10, 10, 30, 30]),
                  ResultItem("measure", MeasureResult(value=12.5, unit="px"),
                             [10, 10, 30, 30]),
                  ResultItem("number", NumberVal(value=3.0), [0, 0, 5, 5])])
        out, _ = _run("merge_results", {"image": _img(), "result": rl}, nms_iou=0.0,
                      geometry_label=7)
        p = out["result"]
        self.assertEqual(p.boxes[0], [110.0, 60.0, 130.0, 80.0])        # 圆 → 外接框
        self.assertEqual(p.boxes[1], [14.0, 15.0, 16.0, 17.0])         # 点 → ±1 小框
        self.assertEqual(p.labels, [7, 7])
        self.assertTrue(p.masks[0][70, 120])                            # 圆心在掩码内
        self.assertFalse(p.masks[0][0, 0])
        self.assertFalse(p.masks[1].any())                              # 点无掩码
        recs = p.extra["merged_items"]
        kinds = sorted(r["kind"] for r in recs)
        self.assertEqual(kinds, ["geometry", "geometry", "measure", "number"])
        circle = next(r for r in recs if r["kind"] == "geometry" and r["value"]["kind"] == "circle")
        self.assertEqual(circle["value"]["data"][:2], [120.0, 70.0])   # 原图坐标精确几何

    def test_empty_list(self):
        out, _ = _run("merge_results", {"result": ResultList()})
        self.assertEqual(out["result"].boxes, [])


class OpOnListTest(unittest.TestCase):
    def _disc_crops(self):
        """两张裁图，各含一个亮圆（中心 30,30 半径 15）；roi 偏移不同。"""
        items = []
        for roi in ([0, 0, 60, 60], [200, 100, 260, 160]):
            m = np.full((60, 60), 20, np.uint8)
            cv2.circle(m, (30, 30), 15, 230, -1, cv2.LINE_AA)
            items.append(ImageData(mat=m, meta={"roi_rect": roi}))
        return ImageList(items=items, meta={"source_size": [400, 300]})

    def test_find_circle_per_crop_merged_to_original(self):
        out, _ = _run("op_on_list", {"images": self._disc_crops()}, op="find_circle",
                      op_params='{"polarity": "bright"}')
        rl = out["result"]
        self.assertEqual([it.kind for it in rl.items], ["geometry", "geometry"])
        self.assertEqual(rl.items[1].roi_rect, [200, 100, 260, 160])
        merged, _ = _run("merge_results", {"result": rl}, nms_iou=0.0)
        boxes = merged["result"].boxes
        self.assertEqual(len(boxes), 2)
        centers = sorted(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes)
        self.assertAlmostEqual(centers[0][0], 30, delta=1.5)
        self.assertAlmostEqual(centers[1][0], 230, delta=1.5)
        self.assertAlmostEqual(centers[1][1], 130, delta=1.5)

    def test_op_on_list_errors(self):
        cls = get_node_class("op_on_list")
        lst = self._disc_crops()
        cases = [({"op": "no_such_op"}, "未找到算子"),
                 ({"op": "filter_class"}, "暂不支持"),
                 ({"op": "find_circle", "op_params": "{bad"}, "JSON"),
                 ({"op": "find_circle", "op_params": '{"num_rays": 1}'}, "参数无效"),
                 ({"op": "find_circle", "output": "nope"}, "没有输出端口")]
        for p, msg in cases:
            full = {k: s.default for k, s in cls.PARAMS.items()}
            full.update(p)
            with self.assertRaises(ValueError) as ctx:
                cls().run({"images": lst}, full, RunContext())
            self.assertIn(msg, str(ctx.exception), p)


class _FakeModel:
    classes = ["part"]

    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on
        self.task = mock.Mock(value="segmentation")

    def predict(self, img, conf=0.25, iou=0.45):
        self.calls += 1
        if self.fail_on is not None and self.calls - 1 == self.fail_on:
            raise RuntimeError("boom")
        h, w = img.shape[:2]
        m = np.zeros((h, w), bool)
        m[h // 4:h // 2, w // 4:w // 2] = True
        return Prediction(labels=[0], scores=[0.8],
                          boxes=[[w / 4, h / 4, w / 2, h / 2]], masks=[m])


def _chain_graph(img_path: str) -> dict:
    return {"nodes": [
        {"id": 1, "type": "image_loader", "properties": {"path": img_path}},
        {"id": 2, "type": "crop_by_detection_all", "properties": {}},
        {"id": 3, "type": "model_on_list", "properties": {"model_path": "fake.pt"}},
        {"id": 4, "type": "merge_results", "properties": {"nms_iou": 0.0}}],
        "links": [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0], [3, 3, 0, 4, 1], [4, 1, 0, 4, 0]]}


class EndToEndListFlowTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.img_path = str(Path(self._tmp.name) / "scene.png")
        cv2.imwrite(self.img_path, np.full((160, 200, 3), 120, np.uint8))
        self.det = Prediction(labels=[0, 0], scores=[0.9, 0.8],
                              boxes=[[10, 20, 50, 60], [100, 30, 140, 70]])

    def _executor(self, ctx, graph=None, external=None):
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.snapshot import SnapshotStore

        ctx.snapshot_store = SnapshotStore()
        return Executor(graph or _chain_graph(self.img_path), ctx,
                        external=external or {(2, 1): self.det})

    def test_chain_merge_model_loaded_once(self):
        fake = _FakeModel()
        loads = []

        def _load(path, device="auto"):
            loads.append(path)
            return fake

        ctx = RunContext()
        with mock.patch("xccv.engine.models.registry.load_model", side_effect=_load):
            ex = self._executor(ctx)
            res = ex.run()
            self.assertTrue(all(r.ok for r in res.values()),
                            {k: r.error for k, r in res.items()})
            ex2 = self._executor(ctx)       # 同会话上下文重跑：模型缓存命中
            ex2.run()
        self.assertEqual(len(loads), 1)
        self.assertEqual(fake.calls, 4)     # 2 张 × 2 次运行
        merged = res[4].outputs["result"]
        self.assertEqual(merged.boxes, [[20.0, 30.0, 30.0, 40.0], [110.0, 40.0, 120.0, 50.0]])
        self.assertEqual(merged.masks[0].shape, (160, 200))
        self.assertTrue(merged.masks[0][30:40, 20:30].all())
        self.assertEqual(int(merged.masks[0].sum()), 100)
        snaps = ctx.snapshot_store
        self.assertTrue(any("拼图" in k for k in snaps.get(2)["images"]))
        self.assertTrue(any("叠加拼图" in k for k in snaps.get(3)["images"]))
        self.assertTrue(snaps.get(4)["images"])   # 有图像直通输出即存快照
        from xccv.workbench import build_preview

        img2, note2 = build_preview(ex2, 2)
        self.assertIsNotNone(img2)
        self.assertIn("图像列表", note2)
        img3, note3 = build_preview(ex2, 3)
        self.assertIsNotNone(img3)
        self.assertIn("逐图结果 2 条", note3)

    def test_single_item_failure_isolated(self):
        fake = _FakeModel(fail_on=0)
        ctx = RunContext()
        with mock.patch("xccv.engine.models.registry.load_model", return_value=fake):
            res = self._executor(ctx).run()
        self.assertTrue(res[3].ok)
        rl = res[3].outputs["result"]
        self.assertEqual([len(it.value.boxes) for it in rl.items], [0, 1])
        self.assertTrue(any("推理失败" in w for w in ctx.warnings))
        self.assertEqual(len(res[4].outputs["result"].boxes), 1)

    def test_canvas_chain_region_filter_then_crop(self):
        """支持先按区域保留/屏蔽筛选检出再裁图：画布上把「保留区域」
        接在检出与裁图之间即可（过滤节点输出仍是检出）。"""
        g = {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": self.img_path}},
            {"id": 2, "type": "filter_region_keep", "properties": {"regions": "0,0,80,160"}},
            {"id": 3, "type": "crop_by_detection_all", "properties": {}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1], [3, 1, 0, 3, 0]]}
        ctx = RunContext()
        res = self._executor(ctx, graph=g, external={(2, 1): self.det}).run()
        self.assertTrue(all(r.ok for r in res.values()),
                        {k: r.error for k, r in res.items()})
        lst = res[3].outputs["images"]
        self.assertEqual([it.meta["roi_rect"] for it in lst.items], [[10, 20, 50, 60]])

    def test_bundle_flows_through_any_node(self):
        """image_list 端口取消——多张集合直接过任意节点
        （并行输入自动映射：逐张执行 gaussian_blur，输出重组集合，
        roi_rect 沿链传递到结果合并）。"""
        from xccv.engine.flow.executor import Executor

        g = {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": self.img_path}},
            {"id": 2, "type": "grid_tile_all", "properties": {"rows": 2, "cols": 2}},
            {"id": 3, "type": "gaussian_blur", "properties": {}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0]]}
        res = Executor(g, RunContext()).run()
        self.assertTrue(all(r.ok for r in res.values()),
                        {k: r.error for k, r in res.items()})
        out = res[3].outputs["image"]
        self.assertIsInstance(out, ImageList)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(it.meta.get("roi_rect") for it in out.items))


class CropUpstreamGeneralityTest(unittest.TestCase):
    """裁图族的上游不锁死为目标检测模型——任何输出检出的节点都可接入
    （过滤、斑点分析、实例分割经掩码转外接框、结果合并…）。"""

    def _seg_pred(self):
        m1 = np.zeros((160, 200), bool)
        m1[30:50, 40:70] = True          # 紧致框应为 [40,30,70,50]
        m2 = np.zeros((160, 200), bool)
        m2[100:104, 150:152] = True      # 8 像素小目标
        return Prediction(labels=[0, 1], scores=[0.9, 0.6],
                          boxes=[[30, 20, 90, 70], [140, 90, 160, 110]], masks=[m1, m2])

    def test_mask_to_box_tight_boxes(self):
        out, _ = _run("mask_to_box", {"result": self._seg_pred()})
        p = out["result"]
        self.assertEqual(p.boxes, [[40.0, 30.0, 70.0, 50.0], [150.0, 100.0, 152.0, 104.0]])
        self.assertEqual(len(p.masks), 2)
        out, ctx = _run("mask_to_box", {"result": self._seg_pred()}, min_area=10)
        self.assertEqual(len(out["result"].boxes), 1)
        self.assertTrue(any("丢弃" in w for w in ctx.warnings))
        nomask = Prediction(labels=[0], scores=[0.5], boxes=[[1, 2, 3, 4]])
        out, _ = _run("mask_to_box", {"result": nomask})
        self.assertEqual(out["result"].boxes, [[1.0, 2.0, 3.0, 4.0]])
        out, _ = _run("mask_to_box", {"result": nomask}, no_mask="drop")
        self.assertEqual(out["result"].boxes, [])

    def _exec(self, graph, external):
        from xccv.engine.flow.executor import Executor

        res = Executor(graph, RunContext(), external=external).run()
        self.assertTrue(all(r.ok for r in res.values()), {k: r.error for k, r in res.items()})
        return res

    def test_seg_masks_to_box_then_crop(self):
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "s.png")
            cv2.imwrite(p, np.full((160, 200, 3), 90, np.uint8))
            g = {"nodes": [
                {"id": 1, "type": "image_loader", "properties": {"path": p}},
                {"id": 2, "type": "mask_to_box", "properties": {}},
                {"id": 3, "type": "crop_by_detection_all", "properties": {}}],
                "links": [[1, 2, 1, 3, 1], [2, 1, 0, 3, 0]]}
            res = self._exec(g, {(2, 1): self._seg_pred()})
            rois = [it.meta["roi_rect"] for it in res[3].outputs["images"].items]
            self.assertEqual(rois[0], [40, 30, 70, 50])   # 掩码紧致框（非模型松框）

    def test_blob_analysis_as_crop_source(self):
        with tempfile.TemporaryDirectory() as td:
            img = np.zeros((160, 200), np.uint8)
            cv2.circle(img, (50, 50), 12, 255, -1)
            cv2.circle(img, (150, 110), 15, 255, -1)
            p = str(Path(td) / "blobs.png")
            cv2.imwrite(p, img)
            g = {"nodes": [
                {"id": 1, "type": "image_loader", "properties": {"path": p}},
                {"id": 2, "type": "blob_analysis", "properties": {}},
                {"id": 3, "type": "crop_by_detection_all", "properties": {}}],
                "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1], [3, 1, 0, 3, 0]]}
            res = self._exec(g, {})
            self.assertEqual(len(res[3].outputs["images"]), 2)


class WorkbenchChainTest(unittest.TestCase):
    """工作台组合链：区域过滤 → 全部裁图（图像自动接原图）→ 逐图执行 → 合并。"""

    def test_build_and_run_list_chain(self):
        _qapp()
        from xccv.modeltools import flow_session as fs
        from xccv.workbench import OperatorWorkbench

        fs._RUN_LOG_DIR = Path(tempfile.gettempdir()) / "xccv_test_flow_logs"
        w = OperatorWorkbench()
        self.addCleanup(w.close)
        lib = w._lib
        w._chain_types = [lib["xccv2/blob_analysis"], lib["xccv2/filter_region_keep"],
                          lib["xccv2/crop_by_detection_all"], lib["xccv2/op_on_list"],
                          lib["xccv2/merge_results"]]
        graph, err = w._build_chain_graph("x.png")
        self.assertIsNone(err, err)
        links = {(l[3], l[4]): (l[1], l[2]) for l in graph["links"]}
        # 图像输出沿链直通（过滤节点带图像输出）；合并的图像
        # 输入连不到上游图像时兜底接原图（Consumption 语义见 _build_chain_graph）
        self.assertEqual(links[(4, 0)], (3, 0))   # 裁图的图像输入 ← 区域保留的图像直通
        self.assertEqual(links[(4, 1)], (3, 1))   # 裁图的检出输入 ← 区域保留
        self.assertEqual(links[(5, 0)], (4, 0))   # 逐图执行 ← 图像列表
        self.assertEqual(links[(6, 0)][0], 1)     # 合并的图像（可选）← 原图
        self.assertEqual(links[(6, 1)], (5, 0))   # 合并 ← 结果列表
        w._chain_types = [lib["xccv2/gaussian_blur"], lib["xccv2/filter_confidence"]]
        graph, err = w._build_chain_graph("x.png")
        self.assertIsNone(graph)
        self.assertIn("无法连接", err)


if __name__ == "__main__":
    unittest.main()
