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


if __name__ == "__main__":
    unittest.main()
