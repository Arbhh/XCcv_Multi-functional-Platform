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


if __name__ == "__main__":
    unittest.main()


