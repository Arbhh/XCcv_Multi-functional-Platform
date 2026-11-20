"""JSON 契约锁定测试（批注一.2 收官）。

单一出口 xccv/engine/result_json.py：服务 /infer、/flow/run、测试平台批量、
流程图页批量、C# SDK 全部消费这两份 schema。本文件锁定跨 builder 契约：

- 任务无关基键集合：build_flow_result_json ⊇ build_result_json；
  流程/传统仅允许扩展 geometries/measurements（只增不改）；
- verdict 键恒存在（无判定 = null，落盘读回不丢）；
- detections / classifications 数组恒存在（无数据为空数组）；
- SCHEMA_VERSION 锁定为 1（改版本必须显式更新本文件 + docs/result_schema.md）。
"""

import json
import tempfile
import unittest
from pathlib import Path

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xccv.engine.models.base import Prediction
from xccv.engine.result_json import (
    SCHEMA_VERSION, build_result_json, build_flow_result_json, save_result_json,
)


def _source():
    return {"file": "img00.png", "path": "D:/ds/img00.png",
            "width": 640, "height": 480}


AI_BASE_KEYS = {"schema_version", "task", "source", "model", "inference",
                "detections", "classifications", "verdict", "error"}
FLOW_EXT_KEYS = {"geometries", "measurements"}
SOURCE_KEYS = {"file", "path", "width", "height"}
MODEL_KEYS = {"file", "name"}
INF_COMMON_KEYS = {"flow_applied", "timestamp"}
INF_AI_KEYS = INF_COMMON_KEYS | {"conf", "iou", "flow_removed", "count_check"}
INF_FLOW_KEYS = INF_COMMON_KEYS | {"flow_name"}


class BuilderKeyContractTest(unittest.TestCase):
    def test_ai_base_keys(self):
        payload = build_result_json(task="detection", source=_source())
        self.assertEqual(set(payload), AI_BASE_KEYS)
        self.assertEqual(payload["task"], "detection")

    def test_flow_keys_are_superset_of_ai(self):
        payload = build_flow_result_json(source=_source())
        self.assertEqual(set(payload), AI_BASE_KEYS | FLOW_EXT_KEYS)
        self.assertEqual(payload["task"], "flow")

    def test_source_model_inference_shape(self):
        ai = build_result_json(task="detection", source=_source(),
                               model={"file": "best.pt", "name": "yolo11n"})
        fl = build_flow_result_json(source=_source(), flow_name="模板A")
        for p in (ai, fl):
            self.assertEqual(set(p["source"]), SOURCE_KEYS)
            self.assertEqual(set(p["model"]), MODEL_KEYS)
        self.assertEqual(ai["model"], {"file": "best.pt", "name": "yolo11n"})
        self.assertEqual(fl["model"], {"file": "", "name": ""})  # 无模型也保留结构
        self.assertEqual(set(ai["inference"]), INF_AI_KEYS)
        self.assertEqual(set(fl["inference"]), INF_FLOW_KEYS)
        self.assertEqual(set(ai["inference"]) & set(fl["inference"]),
                         INF_COMMON_KEYS)
        self.assertIs(fl["inference"]["flow_applied"], True)
        self.assertIs(ai["inference"]["flow_applied"], False)  # 无 flow_info

    def test_schema_version_locked(self):
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(build_result_json(
            task="detection", source=_source())["schema_version"], 1)
        self.assertEqual(build_flow_result_json(
            source=_source())["schema_version"], 1)

    def test_arrays_always_present(self):
        ai = build_result_json(task="detection", source=_source())
        fl = build_flow_result_json(source=_source())
        for p in (ai, fl):
            self.assertEqual(p["detections"], [])
            self.assertEqual(p["classifications"], [])
        self.assertEqual(fl["geometries"], [])
        self.assertEqual(fl["measurements"], [])


class VerdictAlwaysPresentTest(unittest.TestCase):
    def test_ai_verdict_null_without_count_check(self):
        payload = build_result_json(task="detection", source=_source())
        self.assertIn("verdict", payload)
        self.assertIsNone(payload["verdict"])

    def test_ai_verdict_from_count_check(self):
        payload = build_result_json(task="detection", source=_source(),
                                    flow_info={"removed": 1, "count_check": "NG"})
        self.assertEqual(payload["verdict"],
                         {"passed": False, "source": "count_check",
                          "detail": "数量判定 NG"})

    def test_flow_verdict_null_and_explicit(self):
        self.assertIsNone(build_flow_result_json(source=_source())["verdict"])
        v = {"passed": True, "source": "count_check", "detail": "x"}
        self.assertEqual(
            build_flow_result_json(source=_source(), verdict=v)["verdict"], v)

    def test_verdict_survives_json_roundtrip(self):
        """消费端视角：verdict=null/对象 落盘读回键与值不丢。"""
        ai = build_result_json(task="detection", source=_source(),
                               flow_info={"count_check": "OK"})
        fl = build_flow_result_json(
            source=_source(),
            verdict={"passed": False, "source": "compare_value", "detail": "d"})
        empty = build_flow_result_json(source=_source())
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.json"
            p2 = Path(td) / "b.json"
            p3 = Path(td) / "c.json"
            save_result_json(p1, ai)
            save_result_json(p2, fl)
            save_result_json(p3, empty)
            r1 = json.loads(p1.read_text(encoding="utf-8"))
            r2 = json.loads(p2.read_text(encoding="utf-8"))
            r3 = json.loads(p3.read_text(encoding="utf-8"))
        self.assertEqual(r1["verdict"]["passed"], True)
        self.assertEqual(r2["verdict"]["passed"], False)
        self.assertIn("verdict", r3)
        self.assertIsNone(r3["verdict"])


class FlowExtensionTest(unittest.TestCase):
    def test_detections_shape_matches_ai(self):
        pred = Prediction(labels=[0], scores=[0.9123],
                          boxes=[[10.0, 20.0, 50.0, 60.0]])
        ai = build_result_json(task="detection", source=_source(),
                               prediction=pred, classes=["划痕"])
        fl = build_flow_result_json(source=_source(),
                                    detections=ai["detections"])
        self.assertEqual(fl["detections"], ai["detections"])
        self.assertEqual(fl["detections"][0]["label"], "划痕")
        self.assertNotIn("mask_area", fl["detections"][0])

    def test_geometries_strip_noise(self):
        class _G:
            def to_json(self):
                return {"kind": "circle", "score": float("nan"),
                        "subpixel": False,
                        "data": {"cx": 1.0, "cy": 2.0, "r": 3.0}}

        payload = build_flow_result_json(source=_source(), geometries=[_G()])
        self.assertEqual(payload["geometries"],
                         [{"kind": "circle",
                           "data": {"cx": 1.0, "cy": 2.0, "r": 3.0}}])

    def test_measurements_default_unit_px_dropped(self):
        class _M:
            def to_json(self):
                return {"value": 12.5, "unit": "px"}

        payload = build_flow_result_json(source=_source(), measurements=[_M()])
        self.assertEqual(payload["measurements"], [{"value": 12.5}])

    def test_empty_geometry_shells_dropped(self):
        class _G:
            def to_json(self):
                return {"kind": "circle", "data": {}}

        payload = build_flow_result_json(source=_source(), geometries=[_G()])
        self.assertEqual(payload["geometries"], [])


if __name__ == "__main__":
    unittest.main()
