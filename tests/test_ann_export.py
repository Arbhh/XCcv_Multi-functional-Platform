"""标注平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v06_batch_export
- test_v044_fixes
"""


# == 来源模块：test_v06_batch_export ==
# -*- coding: utf-8 -*-
"""测试平台增强——批量导出格式/统计摘要/4K 分块推理。

- to_labelme 端到端：产物可被 labelme LabelFile.load 加载，形状/标签一致；
- to_coco：categories 完整（classes 全量含零检出类）、bbox 转 [x,y,w,h]、
  id 唯一；
- batch_summary：每类检出数/平均置信度/耗时 p50；
- tiled_predict：桩模型模拟"小目标整图漏检、分块检出"，验证切片/坐标还原/
  同类 NMS 去重；小图回退整图。
"""
import cv2
import numpy as np

os_patched = True
import os  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xccv.engine import result_json  # noqa: E402
from xccv.engine.dataset.base import TaskType  # noqa: E402
from xccv.engine.models.base import Prediction  # noqa: E402
from xccv.engine.tiled_infer import _nms_merge, tiled_predict  # noqa: E402


def _payload(file: str, w: int, h: int, dets: list) -> dict:
    """手工构造统一检出 JSON（结构与 build_result_json 产物一致）。"""
    return {"schema_version": 1, "task": "detection",
            "source": {"file": file, "path": "", "width": w, "height": h},
            "model": {"file": "m.pt", "name": ""},
            "inference": {"conf": 0.25, "iou": 0.45},
            "detections": dets, "classifications": [], "verdict": None,
            "error": None}


class LabelmeRoundtripTest(unittest.TestCase):
    """to_labelme 产物可被 labelme 标注平台加载（端到端）。"""

    def test_labelme_file_loads(self):
        from labelme.label_file import LabelFile

        payload = _payload("a.png", 200, 100, [
            {"label": "cat", "label_id": 0, "score": 0.87,
             "bbox": [10.0, 10.0, 60.0, 50.0], "bbox_area": 2000},
            {"label": "dog", "label_id": 1, "score": 0.5,
             "bbox": [100.0, 20.0, 180.0, 80.0], "bbox_area": 4800},
        ])
        lab = result_json.to_labelme(payload)
        self.assertEqual(lab["version"], result_json.LABELME_VERSION)
        self.assertEqual(lab["imageWidth"], 200)
        self.assertEqual(lab["imageHeight"], 100)
        self.assertEqual(len(lab["shapes"]), 2)
        self.assertEqual(lab["shapes"][0]["shape_type"], "rectangle")
        self.assertEqual(lab["shapes"][0]["flags"]["score"], 0.87)
        with tempfile.TemporaryDirectory() as td:
            img = np.zeros((100, 200, 3), np.uint8)
            cv2.imwrite(str(Path(td) / "a.png"), img)
            out = Path(td) / "a.json"
            result_json.save_result_json(out, lab)
            lf = LabelFile(str(out))
            self.assertEqual(len(lf.shapes), 2)
            labels = {s["label"] for s in lf.shapes}
            self.assertEqual(labels, {"cat", "dog"})

    def test_classification_maps_top1_flag(self):
        payload = _payload("c.png", 32, 32, [])
        payload["task"] = "classification"
        payload["classifications"] = [{"label": "ok", "label_id": 0,
                                       "score": 0.9}]
        lab = result_json.to_labelme(payload)
        self.assertEqual(lab["flags"], {"top1": "ok"})
        self.assertEqual(lab["shapes"], [])


class CocoExportTest(unittest.TestCase):
    """categories 完整 + bbox 转 [x,y,w,h] + id 唯一。"""

    def test_coco_merge(self):
        payloads = [
            _payload("a.png", 640, 480, [
                {"label": "cat", "label_id": 0, "score": 0.9,
                 "bbox": [10, 10, 50, 60], "bbox_area": 2000}]),
            _payload("b.png", 640, 480, [
                {"label": "dog", "label_id": 1, "score": 0.8,
                 "bbox": [100, 20, 200, 120], "bbox_area": 10000}]),
        ]
        coco = result_json.to_coco(payloads, classes=["cat", "dog", "bird"])
        cats = {c["name"]: c["id"] for c in coco["categories"]}
        # 零检出类别 bird 也在 categories（完整性验收点）
        self.assertEqual(cats, {"cat": 1, "dog": 2, "bird": 3})
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual([im["file_name"] for im in coco["images"]],
                         ["a.png", "b.png"])
        anns = coco["annotations"]
        self.assertEqual(len(anns), 2)
        self.assertEqual([a["id"] for a in anns], [1, 2])
        a0 = anns[0]
        self.assertEqual(a0["category_id"], 1)
        self.assertEqual(a0["bbox"], [10.0, 10.0, 40.0, 50.0])  # xywh
        self.assertEqual(a0["area"], 2000)
        self.assertEqual(a0["iscrowd"], 0)

    def test_coco_without_classes_uses_union(self):
        payloads = [_payload("a.png", 10, 10, [
            {"label": "x", "label_id": 5, "score": 0.5,
             "bbox": [0, 0, 5, 5], "bbox_area": 25}])]
        coco = result_json.to_coco(payloads)
        self.assertEqual([c["name"] for c in coco["categories"]], ["x"])


class BatchSummaryTest(unittest.TestCase):
    """每类检出数/平均置信度/耗时 p50。"""

    def test_summary(self):
        results = [
            _payload("a.png", 10, 10, [
                {"label": "cat", "label_id": 0, "score": 0.8,
                 "bbox": [0, 0, 1, 1], "bbox_area": 1},
                {"label": "cat", "label_id": 0, "score": 0.6,
                 "bbox": [2, 2, 3, 3], "bbox_area": 1},
                {"label": "dog", "label_id": 1, "score": 0.9,
                 "bbox": [4, 4, 5, 5], "bbox_area": 1}]),
            _payload("b.png", 10, 10, []),
        ]
        s = result_json.batch_summary(results, costs_ms=[100, 200, 300])
        self.assertEqual(s["images"], 2)
        self.assertEqual(s["detections"], 3)
        self.assertEqual(s["per_class"]["cat"]["count"], 2)
        self.assertEqual(s["per_class"]["cat"]["avg_score"], 0.7)
        self.assertEqual(s["per_class"]["dog"]["avg_score"], 0.9)
        self.assertEqual(s["latency_ms"]["p50_ms"], 200)
        self.assertEqual(s["latency_ms"]["max_ms"], 300)

    def test_summary_no_costs(self):
        s = result_json.batch_summary([], costs_ms=None)
        self.assertEqual(s["images"], 0)
        self.assertIsNone(s["latency_ms"])


class _BrightSquareModel:
    """桩模型：检出输入图中的高亮方块——仅当面积 ≥ 输入面积 0.3% 才报出
    （模拟真实模型"小目标在整图口径下低于可检下限"）。"""

    AREA_FRAC = 0.003

    def predict(self, image, conf=0.25, iou=0.45):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 200).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        labels, scores, boxes = [], [], []
        area_in = image.shape[0] * image.shape[1]
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.AREA_FRAC * area_in:
                continue
            labels.append(0)
            scores.append(0.99)
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
        return Prediction(labels=labels, scores=scores, boxes=boxes)


class TiledInferTest(unittest.TestCase):
    """整图漏检 vs 分块检出；坐标还原；跨块重复去重；小图回退。"""

    def _synth(self, w: int, h: int, squares: list) -> np.ndarray:
        img = np.zeros((h, w, 3), np.uint8)
        for (x, y, size) in squares:
            img[y:y + size, x:x + size] = 255
        return img

    def test_full_miss_vs_tiled_hit(self):
        squares = [(200, 300, 40), (1200, 900, 40)]
        img = self._synth(1600, 1200, squares)
        model = _BrightSquareModel()
        full = model.predict(img)
        self.assertEqual(len(full.boxes), 0, "整图口径下 40px 方块应漏检")
        pred = tiled_predict(model, img, tile=400, overlap=0.15)
        self.assertEqual(len(pred.boxes), 2, "分块后两处方块都应检出")
        got = sorted(tuple(b) for b in pred.boxes)
        for (x, y, size), box in zip(sorted(squares), got):
            self.assertAlmostEqual(box[0], x, delta=2)
            self.assertAlmostEqual(box[1], y, delta=2)
            self.assertAlmostEqual(box[2], x + size, delta=2)
            self.assertAlmostEqual(box[3], y + size, delta=2)

    def test_cross_tile_duplicate_deduped(self):
        # 方块落在两块相邻窗口的重叠区正中：两块各检出一次 → NMS 去重为 1
        img = self._synth(800, 400, [(355, 180, 40)])  # x∈[355,395] ⊂ 重叠区
        model = _BrightSquareModel()
        pred = tiled_predict(model, img, tile=400, overlap=0.15,
                             merge_iou=0.4)
        self.assertEqual(len(pred.boxes), 1, f"重叠区目标应去重: {pred.boxes}")

    def test_small_image_falls_through(self):
        img = self._synth(200, 200, [(50, 50, 40)])
        model = _BrightSquareModel()
        pred = tiled_predict(model, img, tile=400)
        # ≤tile 直接走整图 predict（40²=1600px=4% ≥0.3% 检出）
        self.assertEqual(len(pred.boxes), 1)

    def test_nms_merge_basics(self):
        keep = _nms_merge(
            [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
            [0, 0, 0], [0.9, 0.8, 0.7], 0.5)
        self.assertEqual(len(keep), 2)  # 同类重叠去重 + 独立目标保留
        keep = _nms_merge(
            [[0, 0, 10, 10], [1, 1, 11, 11]], [0, 1], [0.9, 0.8], 0.5)
        self.assertEqual(len(keep), 2)  # 跨类别不互斥
        self.assertEqual(_nms_merge([], [], [], 0.5), [])


class DialogFormatTest(unittest.TestCase):
    """批量设置对话框格式下拉（默认 xcv，可选 labelme/coco）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_format_roundtrip(self):
        from xccv.testing.batch_dialog import BatchSettingsDialog

        dlg = BatchSettingsDialog()
        self.addCleanup(dlg.close)
        with tempfile.TemporaryDirectory() as td:
            dlg.ed_folder.setText(td)
            self.assertEqual(dlg.settings().json_format, "xcv")
            dlg.fmt_combo.setCurrentIndex(1)
            self.assertEqual(dlg.settings().json_format, "labelme")
            dlg.fmt_combo.setCurrentIndex(2)
            self.assertEqual(dlg.settings().json_format, "coco")


class BatchThreadFormatTest(unittest.TestCase):
    """_BatchThread 分格式落盘（labelme 可加载；coco 汇总+摘要）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def _make_thread(self, td: Path, fmt: str):
        from xccv.testing.testing_window import _BatchThread

        model = _BrightSquareModel()
        img = np.zeros((300, 300, 3), np.uint8)
        img[100:200, 100:200] = 255  # 100²=10000px=11% ≥0.3% 检出
        p = td / "a.png"
        cv2.imwrite(str(p), img)
        return _BatchThread(model, [p], 0.25, 0.45, None, ["sq"],
                            "detection", out_dir=td / "out", save_json=True,
                            save_images=False, root=td,
                            json_format=fmt), p

    def _run_thread(self, th):
        th.run()  # 直接同步执行 run()（不经事件循环）

    def test_labelme_output(self):
        with tempfile.TemporaryDirectory() as td:
            th, p = self._make_thread(Path(td), "labelme")
            self._run_thread(th)
            out = Path(td) / "out" / "检出结果" / "a.json"
            self.assertTrue(out.is_file())
            from labelme.label_file import LabelFile

            lf = LabelFile(str(out))
            self.assertEqual(len(lf.shapes), 1)
            self.assertEqual(lf.shapes[0]["label"], "sq")
            del lf  # 释放句柄（Windows 下临时目录清理需无占用）

    def test_coco_output_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            th, _p = self._make_thread(Path(td), "coco")
            self._run_thread(th)
            coco_p = Path(td) / "out" / "检出结果" / "results_coco.json"
            self.assertTrue(coco_p.is_file())
            import json

            coco = json.loads(coco_p.read_text(encoding="utf-8"))
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual([c["name"] for c in coco["categories"]], ["sq"])
            names = {f.name for f in (Path(td) / "out" / "检出结果").glob("*.json")}
            self.assertEqual(names, {"results_coco.json", "摘要.json"})
            # 摘要落盘
            summary_p = Path(td) / "out" / "检出结果" / "摘要.json"
            self.assertTrue(summary_p.is_file())
            summary = json.loads(summary_p.read_text(encoding="utf-8"))
            self.assertEqual(summary["detections"], 1)
            self.assertIn("latency_ms", summary)

    def test_xcv_output_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            th, _p = self._make_thread(Path(td), "xcv")
            self._run_thread(th)
            import json

            out = Path(td) / "out" / "检出结果" / "a.json"
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["detections"]), 1)

# == 来源模块：test_v044_fixes ==
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
    """Q3：测试平台传统流程模式（子进程：testing_window 为重模块）。"""

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

if __name__ == '__main__':
    unittest.main()
