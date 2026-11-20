"""v0.4.4 修复测试（V0.4.3 回传批注）：输出目录选文件夹、蒙版收集全图输出、
批量设置弹窗参数生效、检出名称可设置、下拉首项自动打开、测试平台传统流程模式。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xccv.engine.io_utils import imwrite_unicode  # cv2.imwrite 中文路径不可靠

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.modeltools.flow_session import (
    FlowSession, node_library_payload,
)
from xccv.modeltools import flow_session as _fs
from xccv.engine.flow.ir import Geometry, ImageData, MeasureResult

node_library_payload()

_fs._RUN_LOG_DIR = Path(tempfile.gettempdir()) / "xccv_test_v044_logs"

_ROOT = Path(__file__).resolve().parents[1]


class FolderParamTest(unittest.TestCase):
    """C1/Q2：输出目录为 folder 型参数（浏览选目录而非文件）。"""

    def test_result_output_out_dir_is_folder(self):
        from xccv.engine.flow.registry import list_nodes

        spec = next(n for n in list_nodes()
                    if n["name"] == "result_output")["params"]["out_dir"]
        self.assertTrue(spec["folder"])

    def test_other_path_params_not_folder(self):
        from xccv.engine.flow.registry import list_nodes

        spec = next(n for n in list_nodes()
                    if n["name"] == "image_loader")["params"]["path"]
        self.assertFalse(spec["folder"])


class OverlayAllNodesTest(unittest.TestCase):
    """C2：末节点为「保存结果」时，中游检出/几何仍叠加到可视化。"""

    def test_overlay_collects_all_ok_outputs(self):
        import xccv.engine.flow.ops.output  # noqa: F401
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext
        from xccv.engine.flow.snapshot import overlay_result

        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((50, 60, 3), np.uint8)
            cv2.circle(img, (30, 25), 12, 255, -1)
            p = Path(tmp) / "in.png"
            cv2.imwrite(str(p), img)
            g = {"nodes": [
                {"id": 1, "type": "image_loader", "properties": {"path": str(p)}},
                {"id": 2, "type": "blob_analysis",
                 "properties": {"threshold_mode": "otsu", "thresh": 128,
                                "invert": False, "min_area": 10,
                                "max_area": 0, "circularity": 0.0,
                                "label_name": "目标"}},
                {"id": 3, "type": "result_output", "properties": {}}],
                "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
            ctx = RunContext()
            ctx.set("flow_dir", tmp)
            ex = Executor(g, ctx)
            results = ex.run()
            # 复刻 FlowPage._run_flow_xcv 的收集逻辑（用同一次运行的 results）
            all_outs, base = {}, None
            for nid in sorted(results):
                r = results[nid]
                if not r.ok:
                    continue
                for v in r.outputs.values():
                    if isinstance(v, ImageData):
                        base = v.mat
                    else:
                        all_outs.setdefault(id(v), v)
            vis = overlay_result(img, list(all_outs.values()))
            # blob 的检出框（橙色蒙版）画上去了——框内像素被染色
            x1, y1, x2, y2 = (int(round(v)) for v in
                              results[2].outputs["result"].boxes[0])
            patch = vis[y1 + 2:y2 - 2, x1 + 2:x2 - 2]
            self.assertGreater(int(patch[..., 2].mean()), 10)  # 橙色 R 通道抬升


class LabelNameTest(unittest.TestCase):
    """Q1：检出名称可设置 → 结果 JSON 的 label 使用该名称。"""

    def test_blob_sets_flow_class_names(self):
        import xccv.engine.flow.ops.output  # noqa: F401
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext

        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((50, 60, 3), np.uint8)
            cv2.circle(img, (30, 25), 12, 255, -1)
            p = Path(tmp) / "in.png"
            cv2.imwrite(str(p), img)
            g = {"nodes": [
                {"id": 1, "type": "image_loader", "properties": {"path": str(p)}},
                {"id": 2, "type": "blob_analysis",
                 "properties": {"threshold_mode": "otsu", "thresh": 128,
                                "invert": False, "min_area": 10,
                                "max_area": 0, "circularity": 0.0,
                                "label_name": "缺陷A"}},
                {"id": 3, "type": "result_output", "properties": {}}],
                "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
            ctx = RunContext()
            ctx.set("flow_dir", tmp)
            ctx.set("source_name", "in.png")
            Executor(g, ctx).run()
            self.assertEqual(ctx.get("flow_class_names"), ["缺陷A"])
            data = json.loads((Path(tmp) / "流程结果" / "in.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(data["detections"][0]["label"], "缺陷A")


class BatchSettingsRespectTest(unittest.TestCase):
    """C5：批量弹窗的 save_json/save_images 设置在传统流程批量中生效。"""

    def _flow(self):
        return {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {}},
            {"id": 2, "type": "xccv2/blob_analysis",
             "properties": {"threshold_mode": "otsu", "thresh": 128,
                            "invert": False, "min_area": 10,
                            "max_area": 0, "circularity": 0.0}}],
            "links": [[1, 1, 0, 2, 0]]}

    def _run_batch(self, out_dir: Path, root: Path, save_json: bool,
                   save_images: bool):
        from xccv.flow_platform import _XcvBatchThread

        th = _XcvBatchThread(self._flow(), ["图A.png"], root, out_dir,
                             flow_name="f.json", save_json=save_json,
                             save_images=save_images)
        emitted = []
        th.finished_ok.connect(emitted.append)
        th.failed.connect(lambda m: emitted.append(f"FAIL:{m}"))
        th.run()
        return emitted

    def test_save_json_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "imgs"
            root.mkdir()
            img = np.zeros((50, 60, 3), np.uint8)
            cv2.circle(img, (30, 25), 10, 255, -1)
            imwrite_unicode(root / "图A.png", img)
            out = Path(tmp) / "out"
            self._run_batch(out, root, save_json=False, save_images=True)
            self.assertFalse((out / "检出结果" / "图A.json").exists())
            self.assertTrue((out / "可视化" / "图A.jpg").exists())

    def test_save_images_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "imgs"
            root.mkdir()
            img = np.zeros((50, 60, 3), np.uint8)
            cv2.circle(img, (30, 25), 10, 255, -1)
            imwrite_unicode(root / "图A.png", img)
            out = Path(tmp) / "out"
            self._run_batch(out, root, save_json=True, save_images=False)
            self.assertTrue((out / "检出结果" / "图A.json").exists())
            self.assertFalse((out / "可视化" / "图A.jpg").exists())


class ComboFirstItemTest(unittest.TestCase):
    """Q4：下拉刷新后首项自动激活（_on_flow_selected(0) 被调用）。"""

    def test_refresh_activates_first_item(self):
        from PyQt5.QtWidgets import QApplication, QComboBox

        app = QApplication.instance() or QApplication([])
        from xccv.flow_platform import FlowPage

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.json").write_text('{"nodes": [], "links": []}',
                                              encoding="utf-8")
            page = mock.MagicMock()
            page.model_dir_edit.text.return_value = tmp   # 供 _refresh 读目录
            combo = QComboBox()
            page.flow_combo = combo
            calls = []
            page._on_flow_selected = lambda idx: calls.append(idx)
            with mock.patch("xccv.flow_platform.QFileSystemWatcher"), \
                    mock.patch("xccv.flow_platform.QTimer"):
                FlowPage._refresh_flow_combo(page)
            self.assertEqual(combo.count(), 1)
            self.assertEqual(calls, [0])   # 首项已自动激活


class TestingPlatformFlowModeTest(unittest.TestCase):
    """Q3：测试平台传统流程模式（子进程：testing_window 为重模块，坑 44）。"""

    def test_load_and_infer_flow(self):
        from xccv.engine import flow_package as fp

        with tempfile.TemporaryDirectory() as tmp:
            # 场景图 + 传统流程包（读入图片→blob→保存结果）
            img = np.zeros((50, 60, 3), np.uint8)
            cv2.circle(img, (30, 25), 12, 255, -1)
            img_path = Path(tmp) / "工件.png"
            imwrite_unicode(img_path, img)   # cv2.imwrite 中文路径在 cp1252 下必败
            flow = {"nodes": [
                {"id": 1, "type": "xccv2/image_loader", "properties": {}},
                {"id": 2, "type": "xccv2/blob_analysis",
                 "properties": {"threshold_mode": "otsu", "thresh": 128,
                                "invert": False, "min_area": 10,
                                "max_area": 0, "circularity": 0.0,
                                "label_name": "缺陷A"}}],
                "links": [[1, 1, 0, 2, 0]]}
            xvap = Path(tmp) / "纯传统.xvap"
            fp.save_xvap(xvap, flow, {})
            code = (
                "import os, sys\n"
                "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
                f"sys.path.insert(0, r'{_ROOT}')\n"
                "from PyQt5.QtWidgets import QApplication\n"
                "app = QApplication([])\n"
                "from xccv.testing.testing_window import TestingWindow\n"
                "w = TestingWindow()\n"
                f"flow = w._peek_xcv_flow(r'{xvap}')\n"
                "if flow is None:\n"
                "    import traceback\n"
                "    from xccv.engine import flow_package as fp\n"
                "    from xccv.engine.flow.legacy_adapter import sniff_format\n"
                "    try:\n"
                f"        loaded, _m = fp.load_xvap(r'{xvap}')\n"
                "        print('SNIFF:', sniff_format(loaded))\n"
                "    except Exception:\n"
                "        traceback.print_exc()\n"
                "assert flow is not None, 'peek failed'\n"
                f"w._load_xcv_flow(r'{xvap}', flow)\n"
                "assert w._xcv_flow is not None\n"
                "from pathlib import Path as P\n"
                "w._image_paths = [P(r'" + str(img_path) + "')]\n"
                "class _Item:\n"
                "    def data(self, *a): return r'" + str(img_path) + "'\n"
                "w._current_image_path = lambda: P(r'" + str(img_path) + "')\n"
                "w._infer_xcv_flow()\n"
                "stats = w.result_stats.text()\n"
                "print('STATS:', stats.replace(chr(10), ' | '))\n"
                "print('LISTN:', w.result_list.count())\n"
                "print('FIRST:', w.result_list.item(0).text() if w.result_list.count() else '')\n"
            )
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, encoding="utf-8", timeout=300, env=env)
            self.assertIn("STATS:", r.stdout, r.stdout + r.stderr[-600:])
            self.assertIn("节点成功", r.stdout)
            self.assertIn("缺陷A", r.stdout)   # 检出名称生效 + 结果列表有条目


if __name__ == "__main__":
    unittest.main()
