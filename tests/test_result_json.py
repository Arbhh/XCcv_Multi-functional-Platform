"""统一检出结果 JSON 契约测试。"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xccv.engine.models.base import Prediction
from xccv.engine.result_json import (
    SCHEMA_VERSION, build_result_json, classification_topk, save_result_json,
)


def _det_pred():
    return Prediction(labels=[0, 1], scores=[0.91234, 0.5],
                      boxes=[[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 8.0, 8.0]])


def _source():
    return {"file": "img00.png", "path": "D:/ds/img00.png",
            "width": 640, "height": 480}


class DetectionSchemaTest(unittest.TestCase):
    def test_detection_payload(self):
        payload = build_result_json(
            task="detection", source=_source(),
            model={"file": "best.pt", "name": "yolo11n"},
            inference={"conf": 0.25, "iou": 0.45},
            prediction=_det_pred(), classes=["cat", "dog"])
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["task"], "detection")
        self.assertEqual(payload["source"]["file"], "img00.png")
        self.assertEqual(payload["model"]["name"], "yolo11n")
        d0 = payload["detections"][0]
        self.assertEqual(d0["label"], "cat")
        self.assertEqual(d0["label_id"], 0)
        self.assertEqual(d0["score"], 0.9123)
        self.assertEqual(d0["bbox"], [10.0, 20.0, 50.0, 60.0])
        self.assertEqual(d0["bbox_area"], 1600)
        self.assertNotIn("mask_area", d0)  # 检测无掩码字段
        self.assertEqual(payload["classifications"], [])  # 数组恒在

    def test_segmentation_mask_area(self):
        pred = Prediction(labels=[0], scores=[0.8],
                          boxes=[[0.0, 0.0, 10.0, 10.0]],
                          masks=[np.ones((10, 10), bool)])
        payload = build_result_json(task="segmentation", source=_source(),
                                    prediction=pred, classes=["scrach"])
        d0 = payload["detections"][0]
        self.assertEqual(d0["mask_area"], 100)
        self.assertEqual(payload["task"], "segmentation")

    def test_classification_topk_and_empty_detections(self):
        """Q2：分类任务走 classifications（Top-K），detections 恒空数组。"""
        pred = Prediction(labels=[1], scores=[0.97],
                          probs=[0.02, 0.97, 0.01])
        payload = build_result_json(task="classification", source=_source(),
                                    prediction=pred, classes=["OK", "NG", "?"])
        self.assertEqual(payload["detections"], [])
        top = payload["classifications"]
        self.assertEqual([c["label"] for c in top], ["NG", "OK", "?"])
        self.assertEqual(top[0]["score"], 0.97)
        # Top-K 截断
        pred5 = Prediction(labels=[0], scores=[0.5],
                           probs=[0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
        top5 = classification_topk(pred5, [f"c{i}" for i in range(7)], k=5)
        self.assertEqual(len(top5), 5)


class ContractTest(unittest.TestCase):
    def test_ascii_keys_and_json_roundtrip(self):
        """C# 友好：键全 ASCII、纯 JSON 基元、可无损 roundtrip。"""
        payload = build_result_json(
            task="segmentation", source=_source(),
            model={"file": "best.pt"}, inference={"conf": 0.3},
            prediction=Prediction(labels=[0], scores=[0.7],
                                  boxes=[[1, 1, 2, 2]],
                                  masks=[np.zeros((4, 4), bool)]),
            classes=["缺陷"], flow_info={"removed": 2, "count_check": "NG"})
        text = json.dumps(payload, ensure_ascii=False)
        back = json.loads(text)
        self.assertEqual(back["inference"]["flow_removed"], 2)
        self.assertEqual(back["inference"]["count_check"], "NG")
        self.assertTrue(back["inference"]["flow_applied"])
        for key in payload:
            self.assertTrue(key.isascii(), key)
        for det in payload["detections"]:
            for key in det:
                self.assertTrue(key.isascii(), key)

    def test_error_entry_and_null_prediction(self):
        payload = build_result_json(task="detection", source=_source(),
                                    prediction=None,
                                    error="RuntimeError: 无法读取")
        self.assertEqual(payload["detections"], [])
        self.assertIn("无法读取", payload["error"])

    def test_new_task_field_extensible(self):
        """OCR 等新任务：task 自由扩展且不破坏既有字段（向后兼容口径）。"""
        payload = build_result_json(task="ocr", source=_source(), prediction=None)
        self.assertEqual(payload["task"], "ocr")
        self.assertEqual(payload["detections"], [])

    def test_save_utf8(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "img00.json"
        payload = build_result_json(task="detection", source=_source(),
                                    prediction=_det_pred(), classes=["夹渣"])
        save_result_json(out, payload)
        text = out.read_text(encoding="utf-8")
        self.assertIn("夹渣", text)  # ensure_ascii=False：中文直接可读
        back = json.loads(text)
        self.assertEqual(back["detections"][0]["label"], "夹渣")


if __name__ == "__main__":
    unittest.main()


