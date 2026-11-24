"""平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_flow_legacy
- test_flow_and_export
"""


# == 来源模块：test_flow_legacy ==
"""测试：legacy 适配（旧 8 节点薄委托）、三格式统一读取、
新旧引擎等价性、Q2 分类走流程修复、节点级日志捕捉。"""

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import xccv.engine.flow.ops.preprocess  # noqa: F401
import xccv.engine.flow.ops.source  # noqa: F401
from xccv.engine import detection_flow as df
from xccv.engine.flow.executor import Executor, format_session_logs
from xccv.engine.flow.ir import ImageData
from xccv.engine.flow.legacy_adapter import (
    apply_flow_safe, chain_to_xcv_graph, sniff_format,
)
from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.registry import get_node_class, list_nodes
from xccv.engine.models.base import Prediction


def _pred() -> Prediction:
    """合成检出：3 个框（cat 0.9 / dog 0.6 / cat 0.2）+ 掩码。"""
    m1 = np.zeros((100, 120), bool)
    m1[10:30, 20:60] = True
    m2 = np.zeros((100, 120), bool)
    m2[40:70, 70:100] = True
    m3 = np.zeros((100, 120), bool)
    m3[80:95, 5:15] = True
    return Prediction(labels=[0, 1, 0], scores=[0.9, 0.6, 0.2],
                      boxes=[[20, 10, 60, 30], [70, 40, 100, 70], [5, 80, 15, 95]],
                      masks=[m1, m2, m3])


CHAIN = {"nodes": [
    {"op": "confidence", "params": {"min": 0.3}},
    {"op": "class_filter", "params": {"keep": "cat"}},
    {"op": "bbox_area", "params": {"min": 100, "max": 0}},
    {"op": "region_keep", "params": {"regions": "0,0,80,50"}},
    {"op": "count_check", "params": {"min": 1, "max": 2}},
]}
CLASSES = ["cat", "dog"]


class SniffFormatTest(unittest.TestCase):
    def test_three_formats(self):
        self.assertEqual(sniff_format({"nodes": [{"op": "confidence", "params": {}}]}),
                         "chain")
        self.assertEqual(sniff_format({"nodes": [
            {"id": 1, "type": "model/yolo11n", "properties": {}},
            {"id": 2, "type": "post/confidence", "properties": {}}]}), "legacy_graph")
        self.assertEqual(sniff_format({"nodes": [
            {"id": 1, "type": "gaussian_blur", "properties": {}}]}), "xcv_graph")
        self.assertEqual(sniff_format({"nodes": [
            {"id": 1, "type": "filter_confidence", "properties": {}}]}), "xcv_graph")
        self.assertEqual(sniff_format({}), "chain")

    def test_load_flow_rejects_xcv_graph(self):
        """新 IR 图格式不是过滤链：统一读入口返回 None（绝不误跑）。"""
        import xccv.engine.flow.ops.preprocess  # noqa: F401  确保 gaussian_blur 已注册

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "flow.json"
            p.write_text(json.dumps({"nodes": [
                {"id": 1, "type": "gaussian_blur", "properties": {"ksize": 3}}],
                "links": []}, ensure_ascii=False), encoding="utf-8")
            self.assertIsNone(df.load_flow(p))

    def test_load_flow_still_reads_legacy_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "flow.json"
            p.write_text(json.dumps({"nodes": [
                {"id": 1, "type": "model/yolo11n", "properties": {"model_file": ""}},
                {"id": 2, "type": "post/confidence", "properties": {"min": 0.5}}],
                "links": [[1, 1, 0, 2, 0]]}, ensure_ascii=False), encoding="utf-8")
            flow = df.load_flow(p)
            self.assertEqual([n["op"] for n in flow["nodes"]], ["confidence"])


class EquivalenceTest(unittest.TestCase):
    """硬断言：同一过滤链在新旧两套引擎下输出 Prediction 完全一致。"""

    def test_same_chain_same_prediction(self):
        ref, ref_info = df.apply_flow(_pred(), (120, 100), dict(CHAIN), classes=CLASSES)

        graph = chain_to_xcv_graph(CHAIN)
        ctx = RunContext()
        ctx.set("model_classes", CLASSES)
        ctx.set("inject_detection", _pred())   # 检出经源节点注入
        ex = Executor(graph, ctx)
        results = ex.run()
        nid = max(results)
        got = results[nid].outputs["result"]
        info = self._merge_info(ex)

        self.assertEqual(got.labels, ref.labels)
        self.assertEqual(got.scores, ref.scores)
        self.assertEqual(got.boxes, ref.boxes)
        self.assertEqual(info.get("count_check"), ref_info.get("count_check"))

    @staticmethod
    def _merge_info(ex):
        """各单步 info 的汇总口径（新引擎逐步委托，无统一 info 对象）。"""
        return {"count_check": "OK" if ex.results[max(ex.results)].outputs
                .get("passed", True) else "NG"}

    def test_equivalence_without_masks_and_classes(self):
        """无掩码、无类别表（纯序号匹配）时也逐位一致。"""
        p = Prediction(labels=[0, 1], scores=[0.9, 0.6],
                       boxes=[[10, 10, 40, 40], [50, 50, 90, 95]])
        chain = {"nodes": [{"op": "confidence", "params": {"min": 0.7}},
                           {"op": "bbox_size", "params": {"w_min": 20, "w_max": 0,
                                                          "h_min": 0, "h_max": 0}}]}
        ref, _ = df.apply_flow(p, (100, 100), dict(chain))
        graph = chain_to_xcv_graph(chain)
        ctx = RunContext()
        ctx.set("inject_detection", p)
        ex = Executor(graph, ctx)
        res = ex.run()
        got = res[max(res)].outputs["result"]
        self.assertEqual(got.labels, ref.labels)
        self.assertEqual(got.boxes, ref.boxes)


class FilterNodeTest(unittest.TestCase):
    def test_region_block_pixel_string_with_image_port(self):
        p = _pred()
        node = get_node_class("filter_region_block")()
        r = node.run({"result": p,
                      "image": ImageData(mat=np.zeros((100, 120), np.uint8))},
                     {"regions": "60,0,120,100"}, RunContext())
        self.assertEqual(len(r["result"].boxes), 2)  # 中心在屏蔽区外的留下

    def test_count_check_verdict_and_passthrough(self):
        p = _pred()
        node = get_node_class("filter_count_check")()
        r = node.run({"result": p}, {"min": 1, "max": 0}, RunContext())
        self.assertTrue(r["result"].extra["verdict"])  # 3 个检出，下限 1 → OK（verdict 随结果扩展字段）
        # 数量判定不过滤：内容不变（委托 apply_flow 会重建对象，断言内容）
        self.assertEqual(r["result"].boxes, p.boxes)
        self.assertEqual(r["result"].labels, p.labels)
        r2 = node.run({"result": p}, {"min": 5, "max": 0}, RunContext())
        self.assertFalse(r2["result"].extra["verdict"])

    def test_class_filter_uses_ctx_classes(self):
        ctx = RunContext()
        ctx.set("model_classes", CLASSES)
        r = get_node_class("filter_class").run(
            get_node_class("filter_class")(),
            {"result": _pred()}, {"keep": "cat"}, ctx)
        self.assertEqual(r["result"].labels, [0, 0])

    def test_registry_group(self):
        names = {n["name"] for n in list_nodes(category="检出过滤")}
        legacy = {"filter_confidence", "filter_class", "filter_bbox_size", "filter_bbox_area",
                  "filter_mask_area", "filter_count_check", "filter_region_block",
                  "filter_region_keep"}
        self.assertLessEqual(legacy, names)   # 旧 8 节点齐全（V0.6 起本组另有新算子）


class ApplyFlowSafeTest(unittest.TestCase):
    def test_classification_passthrough_keeps_probs(self):
        """Q2：分类 Prediction（无框有 probs）原样直通，probs 不丢。"""
        pred = Prediction(labels=[1], scores=[0.93], probs=[0.05, 0.93, 0.02])
        out, info = apply_flow_safe(pred, (48, 48), CHAIN, classes=CLASSES)
        self.assertIs(out, pred)
        self.assertTrue(info["classification"])

    def test_detection_still_delegates(self):
        pred = _pred()
        out, info = apply_flow_safe(pred, (120, 100),
                                    {"nodes": [CHAIN["nodes"][0]]}, classes=CLASSES)
        self.assertIsNot(out, pred)
        self.assertEqual(len(out.boxes), 2)
        self.assertNotIn("classification", info)


class LoggingCaptureTest(unittest.TestCase):
    """日志捕捉：节点级结构化日志 + 平台日志系统（可精确定位出错节点）。"""

    def _graph_with_failing_node(self):
        return {"nodes": [
            {"id": 1, "type": "image_loader",
             "properties": {"path": "Z:/不存在的图.png"}},
            {"id": 2, "type": "gaussian_blur", "properties": {}}],
            "links": [[1, 1, 0, 2, 0]]}

    def test_failed_node_logged_with_id_and_name(self):
        ex = Executor(self._graph_with_failing_node())
        ex.run()
        errs = [e for e in ex.session_logs if e["level"] == "ERROR"]
        self.assertTrue(errs)
        self.assertEqual(errs[0]["node"], "image_loader")
        self.assertEqual(errs[0]["node_id"], 1)
        self.assertIn("无法读取图片", errs[0]["message"])
        skipped = [e for e in ex.session_logs if e["level"] == "WARN"]
        self.assertTrue(any(e["node"] == "gaussian_blur" for e in skipped))

    def test_success_log_has_duration(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.png"
            cv2.imwrite(str(p), np.zeros((8, 8), np.uint8))
            g = {"nodes": [{"id": 1, "type": "image_loader",
                            "properties": {"path": str(p)}}],
                 "links": []}
            ex = Executor(g)
            ex.run()
            ok = [e for e in ex.session_logs if e["level"] == "INFO"]
            self.assertTrue(any("ms" in e["message"] for e in ok))

    def test_format_session_logs(self):
        ex = Executor(self._graph_with_failing_node())
        ex.run()
        text = format_session_logs(ex.session_logs)
        self.assertIn("[ERROR] image_loader(#1)", text)
        self.assertIn("[WARN] gaussian_blur(#2)", text)

    def test_logs_go_to_platform_logger(self):
        """节点失败同步进 logging（平台 RotatingFileHandler 捕获）。"""
        with self.assertLogs("xccv.engine.flow", level="ERROR") as cm:
            Executor(self._graph_with_failing_node()).run()
        self.assertTrue(any("image_loader" in m for m in cm.output))

    def test_legacy_chain_logs_steps(self):
        with self.assertLogs("xccv.engine.flow.legacy", level="DEBUG") as cm:
            df.apply_flow(_pred(), (120, 100),
                          {"nodes": [{"op": "confidence", "params": {"min": 0.3}}]},
                          classes=CLASSES)
        joined = "\n".join(cm.output)
        self.assertIn("检出过滤链完成", joined)

    def test_legacy_chain_error_names_operator(self):
        """传统算子错误日志必须带 op 名（精确定位到算子）。"""
        bad = {"nodes": [{"op": "confidence", "params": {"min": "abc"}}]}
        with self.assertLogs("xccv.engine.flow.legacy", level="ERROR") as cm:
            with self.assertRaises(Exception):
                df.apply_flow(_pred(), (120, 100), bad)
        self.assertTrue(any("confidence" in m for m in cm.output))

# == 来源模块：test_flow_and_export ==
# -*- coding: utf-8 -*-
"""检出过滤流程 / 引擎容器 / maskrcnn18 / DCN 导出等新增能力测试。"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch

from xccv.engine import detection_flow as df
from xccv.engine.dataset.base import TaskType
from xccv.engine.models import create_model
from xccv.engine.models.base import Prediction


def _pred__flow_and_export():
    n = 6
    boxes = [[10 * i, 10, 10 * i + 30, 40] for i in range(n)]
    scores = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]
    labels = [0, 0, 1, 1, 0, 1]
    masks = None
    return Prediction(labels=labels, scores=scores, boxes=boxes, masks=masks)


class FlowFilterTest(unittest.TestCase):
    def test_confidence(self):
        flow = {"nodes": [{"op": "confidence", "params": {"min": 0.35}}]}
        out, info = df.apply_flow(_pred__flow_and_export(), (320, 60), flow)
        self.assertEqual(len(out.boxes), 3)
        self.assertEqual(info["removed"], 3)
        self.assertTrue(all(s >= 0.35 for s in out.scores))

    def test_bbox_area(self):
        flow = {"nodes": [{"op": "bbox_area", "params": {"min": 500, "max": 0}}]}
        # 每个框 30×30=900，min=500 全保留
        out, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow)
        self.assertEqual(len(out.boxes), 6)

    def test_mask_area(self):
        pred = _pred__flow_and_export()
        pred.masks = [np.zeros((60, 320), bool) for _ in range(6)]
        for i, m in enumerate(pred.masks):
            m[10:20, 10 * i:10 * i + (4 if i < 3 else 40)] = True  # 前3个 40px，后3个 400px
        flow = {"nodes": [{"op": "mask_area", "params": {"min": 100, "max": 0}}]}
        out, _ = df.apply_flow(pred, (320, 60), flow)
        self.assertEqual(len(out.boxes), 3)

    def test_region_block_and_keep(self):
        # 框中心 x = 15,25,35,...,65；区域 x∈[0,32] 覆盖前两个中心
        flow_block = {"nodes": [{"op": "region_block",
                                 "params": {"regions": [[0.0, 0.0, 0.1, 1.0]]}}]}
        out, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow_block)
        self.assertEqual(len(out.boxes), 4)
        flow_keep = {"nodes": [{"op": "region_keep",
                                "params": {"regions": [[0.0, 0.0, 0.1, 1.0]]}}]}
        out2, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow_keep)
        self.assertEqual(len(out2.boxes), 2)

    def test_count_check(self):
        flow = {"nodes": [{"op": "count_check", "params": {"min": 3, "max": 4}}]}
        _, info = df.apply_flow(_pred__flow_and_export(), (320, 60), flow)
        self.assertEqual(info["count_check"], "NG")
        flow2 = {"nodes": [{"op": "confidence", "params": {"min": 0.4}},
                           {"op": "count_check", "params": {"min": 3, "max": 4}}]}
        _, info2 = df.apply_flow(_pred__flow_and_export(), (320, 60), flow2)
        self.assertEqual(info2["count_check"], "OK")

    def test_class_filter_with_names(self):
        flow = {"nodes": [{"op": "class_filter", "params": {"keep": "流纹"}}]}
        out, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow, classes=["划伤", "流纹"])
        self.assertEqual(len(out.boxes), 3)
        self.assertTrue(all(l == 1 for l in out.labels))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "检出流程.json"
            flow = {"nodes": [{"op": "mask_area", "params": {"min": 150, "max": 0}},
                              {"op": "count_check", "params": {"min": 1, "max": 0}}]}
            df.save_flow(p, flow)
            loaded = df.load_flow(p)
            self.assertEqual([n["op"] for n in loaded["nodes"]],
                             ["mask_area", "count_check"])
            self.assertEqual(loaded["nodes"][0]["params"]["min"], 150)

    def test_normalize_drops_unknown(self):
        flow = {"nodes": [{"op": "nope", "params": {}},
                          {"op": "confidence", "params": {}}]}
        out = df.normalize_flow(flow)
        self.assertEqual([n["op"] for n in out["nodes"]], ["confidence"])
        self.assertEqual(out["nodes"][0]["params"]["min"], 0.25)


class TrtMaskPasteTest(unittest.TestCase):
    """TRT 特征段引擎的掩码贴回语义（面积虚高 bug 的回归防护）。

    RoI 头输出 (N,1,28,28) 概率图，必须按检出框贴回原图（torchvision
    paste_masks_in_image 同语义）；旧实现把 28×28 概率图直接拉伸到整张
    原图，小缺陷掩码面积虚高几个数量级。
    """

    def test_mask_confined_to_box(self):
        from xccv.engine.trt_runtime import paste_frame_masks

        prob = np.zeros((28, 28), np.float32)
        prob[8:20, 8:12] = 1.0                    # 框内一个竖条（约 4×12 格）
        boxes = [[100.0, 100.0, 140.0, 180.0]]    # 引擎帧 640×640 内的框
        masks = paste_frame_masks([prob], boxes, (640, 640), (640, 640))
        m = masks[0]
        self.assertEqual(m.shape, (640, 640))
        ys, xs = np.where(m)
        # 掩码严格落在框内（1:1 时框坐标不变）
        self.assertGreaterEqual(xs.min(), 100)
        self.assertLess(xs.max(), 140)
        self.assertGreaterEqual(ys.min(), 100)
        self.assertLess(ys.max(), 180)
        # 面积有界：约为框内比例区域，绝不允许接近整图（旧 bug 时 >10 万 px）
        self.assertLess(int(m.sum()), 0.5 * 40 * 80)
        self.assertGreater(int(m.sum()), 0)

    def test_mask_scales_with_frame_to_orig(self):
        from xccv.engine.trt_runtime import paste_frame_masks

        prob = np.full((28, 28), 1.0, np.float32)  # 全 1 → 整框
        # 引擎帧 640×640 的框 → 原图 1280×960（宽 ×2、高 ×1.5）
        masks = paste_frame_masks([prob], [[320.0, 320.0, 480.0, 480.0]],
                                  (640, 640), (960, 1280))
        m = masks[0]
        self.assertEqual(m.shape, (960, 1280))
        ys, xs = np.where(m)
        self.assertGreaterEqual(xs.min(), 640)     # 320*2
        self.assertLessEqual(xs.max(), 960)        # 480*2
        self.assertGreaterEqual(ys.min(), 480)     # 320*1.5
        self.assertLessEqual(ys.max(), 720)        # 480*1.5

    def test_empty_input(self):
        from xccv.engine.trt_runtime import paste_frame_masks

        self.assertEqual(paste_frame_masks([], [], (64, 64), (100, 80)), [])


class FlowGraphFileTest(unittest.TestCase):
    """流程图编辑器落盘 JSON 在测试平台的加载兼容（后处理失效 bug 回归防护）。"""

    def _editor_graph(self):
        # 与流程图编辑器 serialize() 同构：节点带 type/properties（非 op/params）
        return {"last_node_id": 4, "last_link_id": 3, "nodes": [
            {"id": 1, "type": "xccv/input_image", "pos": [0, 0], "properties": {}},
            {"id": 2, "type": "xccv/model_instance_seg", "pos": [1, 0],
             "properties": {"model_file": "best.pt", "threshold": 0.25}},
            {"id": 3, "type": "xccv/post_confidence", "pos": [2, 0],
             "properties": {"min": 0.5}},
            {"id": 4, "type": "xccv/post_class_filter", "pos": [3, 0],
             "properties": {"keep": "划伤"}},
            {"id": 5, "type": "xccv/output", "pos": [4, 0], "properties": {}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0],
                      [3, 3, 0, 4, 0], [4, 4, 0, 5, 0]]}

    def test_load_flow_accepts_editor_graph_format(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "检出流程.json"
            p.write_text(json.dumps(self._editor_graph(), ensure_ascii=False),
                         encoding="utf-8")
            loaded = df.load_flow(p)
        self.assertIsNotNone(loaded, "图格式流程文件不得被解析成空链")
        self.assertEqual([n["op"] for n in loaded["nodes"]],
                         ["confidence", "class_filter"])
        self.assertEqual(loaded["nodes"][0]["params"]["min"], 0.5)

    def test_loaded_editor_graph_actually_filters(self):
        # 端到端：编辑器图格式 → load_flow → apply_flow 真实过滤
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "检出流程.json"
            p.write_text(json.dumps(self._editor_graph(), ensure_ascii=False),
                         encoding="utf-8")
            flow = df.load_flow(p)
        out, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow, classes=["划伤", "流纹"])
        # 置信度 ≥0.5 剩 3 个（label 0,0,1），类别过滤"划伤"再剩 2 个
        self.assertEqual(len(out.boxes), 2)
        self.assertEqual(out.labels, [0, 0])

    def test_region_params_as_string(self):
        # 编辑器 region 参数为文本输入："x1,y1,x2,y2;x1,y1,x2,y2"（归一化坐标）
        flow = {"nodes": [{"op": "region_block",
                           "params": {"regions": "0,0,0.1,1"}}]}
        out, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow)
        self.assertEqual(len(out.boxes), 4)
        flow_keep = {"nodes": [{"op": "region_keep",
                                "params": {"regions": "0,0,0.1,1"}}]}
        out2, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow_keep)
        self.assertEqual(len(out2.boxes), 2)

    def test_region_malformed_tolerated(self):
        # 损坏的区域串不得让整条流程崩溃（忽略坏条目）
        flow = {"nodes": [{"op": "region_block",
                           "params": {"regions": "垃圾;0,0,0.1,1"}}]}
        out, _ = df.apply_flow(_pred__flow_and_export(), (320, 60), flow)
        self.assertEqual(len(out.boxes), 4)


class XcsHeadsV2Test(unittest.TestCase):
    """引擎头容器（xcve-heads-v2；兼容旧 xcs-heads-v2 命名）。

    旧容器把整网 state_dict 封进 extra 且漏带 mask_head_light 开关：
    轻量掩码头模型（arch=maskrcnn，2 层 128 通道）被按默认头
    （4 层 256 通道）重建，测试平台加载 .engine 必报 shape mismatch。
    """

    HEAD_CONV = "roi_heads.mask_head.0.0.weight"  # light=128 通道 / 默认=256 通道

    def _pack(self, light: bool, subset: bool, with_flag: bool = True):
        import io

        from xccv.engine.models.rcnn import RCNNModel

        m = RCNNModel("maskrcnn", TaskType.SEGMENTATION, 2, device="cpu",
                      use_pretrained=False, industrial=False, mask_head_light=light)
        m._min_size = 96
        state = m.net.state_dict()
        if subset:
            state = {k: v for k, v in state.items()
                     if k.startswith(("rpn.", "roi_heads."))}
        payload = {"format": "xcs-heads-v2" if subset else None,
                   "state_dict": state, "industrial_opt": False,
                   "mask_head_light": light}
        if not with_flag:
            payload.pop("mask_head_light")
        buf = io.BytesIO()
        torch.save(payload, buf)
        return buf.getvalue(), state

    def _meta(self):
        return {"kind": "rcnn", "arch": "maskrcnn", "task": "segmentation",
                "classes": ["a", "b"], "min_size": 96, "max_size": 160,
                "feat_keys": ["0", "1", "2", "3", "pool"]}

    def test_v2_subset_rebuilds_light_mask_head(self):
        from xccv.engine.trt_runtime import TrtRcnnModel

        blob, state = self._pack(light=True, subset=True)
        tm = TrtRcnnModel(self._meta(), device="cpu")
        tm._load_heads(blob)  # 不抛错 = 结构对齐
        self.assertIsInstance(tm._heads.backbone, torch.nn.Identity)  # 骨干置空省显存
        # 轻量掩码头：第一层卷积 128 通道（默认头为 256）
        self.assertEqual(tm._heads.state_dict()[self.HEAD_CONV].shape[0], 128)
        # 头权重逐元素等于容器内打包的源权重
        self.assertTrue(torch.equal(tm._heads.state_dict()[self.HEAD_CONV],
                                    state[self.HEAD_CONV]))

    def test_v2_missing_flag_raises_with_guidance(self):
        """缺结构开关的旧容器应报出"重新加速"指引，而非裸 shape mismatch。"""
        from xccv.engine.trt_runtime import TrtRcnnModel

        blob, _ = self._pack(light=True, subset=True, with_flag=False)
        tm = TrtRcnnModel(self._meta(), device="cpu")
        with self.assertRaises(RuntimeError) as ctx:
            tm._load_heads(blob)
        self.assertIn("重新", str(ctx.exception))

    def test_legacy_full_state_dict_still_loads(self):
        """旧容器（整网 state_dict、无 format 键）按原逻辑加载，向后兼容。"""
        from xccv.engine.trt_runtime import TrtRcnnModel

        blob, _ = self._pack(light=False, subset=False)
        tm = TrtRcnnModel(self._meta(), device="cpu")
        tm._load_heads(blob)
        self.assertEqual(
            tm._heads.state_dict()[self.HEAD_CONV].shape[0], 256)  # 非 light 默认头


class EngineBlobTest(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        from xccv.engine.model_export import pack_engine_blob, unpack_engine_blob

        meta = {"kind": "classifier", "classes": ["a", "b"]}
        engine = b"\x00\x01" * 100
        extra = b"head-weights-bytes"
        payload = pack_engine_blob(meta, engine, extra)
        m2, e2, x2 = unpack_engine_blob(payload)
        self.assertEqual(m2, meta)
        self.assertEqual(e2, engine)
        self.assertEqual(x2, extra)

    def test_plain_payload_passthrough(self):
        from xccv.engine.model_export import unpack_engine_blob

        meta, engine, extra = unpack_engine_blob(b"\x12\x34plain-engine")
        self.assertIsNone(meta)
        self.assertEqual(engine, b"\x12\x34plain-engine")
        self.assertEqual(extra, b"")


class MaskRcnn18Test(unittest.TestCase):
    def test_build_and_forward(self):
        m = create_model("maskrcnn18", TaskType.SEGMENTATION, 2,
                         device="cpu", use_pretrained=False)
        m.net.transform.min_size = (96,)
        m.net.transform.max_size = 128
        m.net.eval()
        with torch.no_grad():
            out = m.net([torch.rand(3, 128, 160)])
        self.assertIn("boxes", out[0])
        # 参数量显著小于 R50 骨干
        m50 = create_model("maskrcnn", TaskType.SEGMENTATION, 2,
                           device="cpu", use_pretrained=False)
        n18 = sum(p.numel() for p in m.net.backbone.body.parameters())
        n50 = sum(p.numel() for p in m50.net.backbone.body.parameters())
        self.assertLess(n18, n50 * 0.6)


class DcnGridSampleTest(unittest.TestCase):
    def test_equivalence_random_offsets(self):
        from xccv.engine.models.modules import DCNv2, set_dcn_export_mode

        torch.manual_seed(0)
        conv = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=True)
        dcn = DCNv2(conv)
        x = torch.randn(2, 8, 32, 32)
        offset = torch.randn(2, 18, 16, 16) * 3.0
        ref = dcn.deform(x, offset)
        set_dcn_export_mode(True)
        try:
            gs = dcn._forward_gridsample(x, offset)
        finally:
            set_dcn_export_mode(False)
        self.assertLess(float((ref - gs).abs().max()), 1e-3)

if __name__ == '__main__':
    unittest.main()
