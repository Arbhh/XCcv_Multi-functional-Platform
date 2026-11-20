"""测试：执行器快照自动捕获、FlowSession（运行到此/重跑/快照）、
节点库 desc 下发、Q4 全平台按钮固定宽度源码审计。"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import xccv.engine.flow.legacy_adapter  # noqa: F401  过滤节点注册
import xccv.engine.flow.ops.ai_bridge  # noqa: F401  触发注册
import xccv.engine.flow.ops.locate  # noqa: F401
import xccv.engine.flow.ops.logic  # noqa: F401
import xccv.engine.flow.ops.preprocess  # noqa: F401  触发注册
import xccv.engine.flow.ops.source  # noqa: F401
from xccv.engine.flow.executor import Executor
from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.registry import list_nodes
from xccv.engine.flow.snapshot import SnapshotStore

# FlowSession 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
from xccv.modeltools import flow_session as _fs

import tempfile as _tempfile

_fs._RUN_LOG_DIR = Path(_tempfile.gettempdir()) / "xccv_test_flow_logs"


def _make_image(tmp: Path) -> str:
    p = tmp / "snap_in.png"
    cv2.imwrite(str(p), np.full((24, 24), 120, np.uint8))
    return str(p)


class ExecutorSnapshotTest(unittest.TestCase):
    def test_images_captured_per_node(self):
        """成功节点自动捕获快照：image 输出降采样副本入库。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore()
            ctx = RunContext(snapshot_store=store)
            g = {"nodes": [
                {"id": 1, "type": "image_loader", "properties": {"path": _make_image(Path(tmp))}},
                {"id": 2, "type": "gaussian_blur", "properties": {}}],
                "links": [[1, 1, 0, 2, 0]]}
            Executor(g, ctx).run()
            self.assertEqual(store.node_ids(), [1, 2])
            snap = store.get(2)
            self.assertIn("image", snap["images"])
            self.assertEqual(snap["images"]["image"].shape[:2], (24, 24))

    def test_snapshot_is_copy_not_reference(self):
        """快照契约：存副本——后续修改运行数据不影响已存快照。"""
        store = SnapshotStore()
        mat = np.zeros((4, 4), np.uint8)
        store.save(1, {"image": mat})
        mat[:] = 255
        self.assertEqual(int(store.get(1)["images"]["image"].max()), 0)


class FlowSessionTest(unittest.TestCase):
    def _graph(self, path: str):
        return {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": path}},
            {"id": 2, "type": "gaussian_blur", "properties": {"ksize": 3}}],
            "links": [[1, 1, 0, 2, 0]]}

    def test_run_and_rerun_with_logs_and_snapshot(self):
        from xccv.modeltools.flow_session import FlowSession

        with tempfile.TemporaryDirectory() as tmp:
            s = FlowSession()
            summary = s.run(self._graph(_make_image(Path(tmp))), to_node=2)
            self.assertEqual(summary["errors"], {})
            self.assertEqual(summary["ok_count"], 2)
            self.assertIn("gaussian_blur(#2)", summary["logs"])
            self.assertEqual(summary["snapshot_ids"], [1, 2])
            payload = s.snapshot_payload(2)
            self.assertIn("image", payload)
            self.assertTrue(payload["image"].startswith("data:image/png;base64,"))
            # 单节点重跑（图未变 → Executor 复用）
            again = s.rerun(2)
            self.assertEqual(again["errors"], {})

    def test_run_error_isolated_in_summary(self):
        from xccv.modeltools.flow_session import FlowSession

        g = {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": "Z:/缺失.png"}},
            {"id": 2, "type": "gaussian_blur", "properties": {}}],
            "links": [[1, 1, 0, 2, 0]]}
        s = FlowSession()
        summary = s.run(g)
        self.assertIn("1", summary["errors"])
        self.assertIn("无法读取图片", summary["errors"]["1"])
        self.assertEqual(summary["snapshot_ids"], [])

    def test_rerun_without_run_raises(self):
        from xccv.modeltools.flow_session import FlowSession

        with self.assertRaises(RuntimeError):
            FlowSession().rerun(1)


class NodeLibraryPayloadTest(unittest.TestCase):
    def test_list_nodes_has_desc_and_groups(self):
        import xccv.engine.flow.ops.ai_bridge  # noqa: F401
        import xccv.engine.flow.legacy_adapter  # noqa: F401

        nodes = list_nodes()
        by_name = {n["name"]: n for n in nodes}
        self.assertEqual(by_name["gaussian_blur"]["category"], "预处理")
        self.assertTrue(by_name["gaussian_blur"]["desc"])
        self.assertEqual(by_name["blob_analysis"]["category"], "定位")
        self.assertEqual(by_name["model_detection"]["category"], "AI")
        self.assertEqual(by_name["filter_confidence"]["category"], "检出过滤")
        # schema 自洽：params 带中文 label 与 tooltip
        spec = by_name["find_circle"]["params"]["num_rays"]
        self.assertEqual(spec["label"], "射线数")
        self.assertTrue(spec["tooltip"])


class ButtonWidthAuditTest(unittest.TestCase):
    """Q4/Q6-1 回归锁：全平台禁止对文字按钮 setFixedWidth（字号放大后
    固定宽度必裁切；面板类容器宽度不受限）。"""

    def test_no_fixed_width_on_buttons(self):
        import re

        root = Path(__file__).resolve().parents[1] / "xccv"
        pat = re.compile(r"(\w+)\.setFixedWidth\(\s*\d+\s*\)")
        bad = []
        for py in root.rglob("*.py"):
            for m in pat.finditer(py.read_text(encoding="utf-8")):
                var = m.group(1).lower()
                if any(tag in var for tag in ("btn", "button")):
                    bad.append(f"{py.name}: {m.group(0)}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
