"""P-07 回归：矩形/分割模型 .engine 引擎链路（容器头 task/classes/imgsz 回填）。

背景：ultralytics 直接加载 .engine 丢 task（分割引擎按检测解析→大量异常框），
且导出请求尺寸被 stride 取整后未记录。修复：容器头记录引擎真实输入形状，
加载侧 apply_engine_meta 回填 task/classes/imgsz，predict 按引擎形状 letterbox。
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fake_yolo_model(task="detect", names=None):
    """ ultralytics YOLO 加载结果的最小替身（overrides 在 predict 时生效）。"""
    return types.SimpleNamespace(task=task,
                                 overrides={},
                                 names=names or {0: "all"},
                                 model="engine-metadata-str")


def _fake_result():
    return types.SimpleNamespace(boxes=None, masks=None)


class ApplyEngineMetaTest(unittest.TestCase):
    """apply_engine_meta：容器头 → YoloModel 状态回填。"""

    def _make_model(self):
        from xccv.engine.dataset.base import TaskType
        from xccv.engine.models.yolo import YoloModel

        m = YoloModel.__new__(YoloModel)  # 跳过权重构建，仅测元数据回填
        m.device = "cpu"
        m.model = _fake_yolo_model()
        m.num_classes = 1
        m.classes = ["all"]
        m.task = TaskType.DETECTION
        return m

    def test_segment_meta_backfills_task_and_names(self):
        from xccv.engine.dataset.base import TaskType

        m = self._make_model()
        m.apply_engine_meta({"kind": "yolo", "task": "segmentation",
                             "classes": ["黑点", "异色"], "imgsz": [608, 1952]})
        self.assertIs(m.task, TaskType.SEGMENTATION)
        self.assertEqual(m.model.task, "segment")
        self.assertEqual(m.model.overrides["task"], "segment")
        self.assertEqual(m.num_classes, 2)
        self.assertEqual(m.classes, ["黑点", "异色"])
        self.assertEqual(m.model.names, {0: "黑点", 1: "异色"})
        self.assertEqual(m._engine_imgsz, [608, 1952])
        self.assertEqual(m.model.overrides["imgsz"], [608, 1952])

    def test_predict_passes_engine_imgsz(self):
        from xccv.engine.dataset.base import TaskType
        from xccv.engine.models.yolo import YoloModel

        m = self._make_model()
        calls = {}

        def fake_predict(img, **kw):
            calls.update(kw)
            return [_fake_result()]

        m.model.predict = fake_predict
        m.apply_engine_meta({"task": "segmentation", "classes": ["a"],
                             "imgsz": [640, 640]})
        import numpy as np

        m.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(calls.get("imgsz"), [640, 640])
        # 未回填引擎几何的 .pt 直载路径：不传 imgsz（走 ultralytics 默认）
        m2 = self._make_model()
        m2._engine_imgsz = None
        calls2 = {}

        def fake_predict2(img, **kw):
            calls2.update(kw)
            return [_fake_result()]

        m2.model.predict = fake_predict2
        m2.predict(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertNotIn("imgsz", calls2)

    def test_empty_meta_noop(self):
        m = self._make_model()
        before = (m.task, m.num_classes, dict(m.model.overrides))
        m.apply_engine_meta(None)
        m.apply_engine_meta({})
        self.assertEqual((m.task, m.num_classes, dict(m.model.overrides)), before)


class ResolveYoloImgszTest(unittest.TestCase):
    """_resolve_yolo_imgsz：显式参数优先，缺省回源训练尺寸（矩形保留）。"""

    def test_explicit_passthrough(self):
        from xccv.engine.model_export import _resolve_yolo_imgsz

        y = types.SimpleNamespace(overrides={"imgsz": [586, 1920]})
        self.assertEqual(_resolve_yolo_imgsz(640, y), 640)
        self.assertEqual(_resolve_yolo_imgsz((480, 800), y), (480, 800))

    def test_fallback_to_overrides(self):
        from xccv.engine.model_export import _resolve_yolo_imgsz

        self.assertEqual(_resolve_yolo_imgsz(None, types.SimpleNamespace(
            overrides={"imgsz": [586, 1920]})), [586, 1920])
        self.assertEqual(_resolve_yolo_imgsz(None, types.SimpleNamespace(
            overrides={"imgsz": 736})), 736)
        self.assertEqual(_resolve_yolo_imgsz(None, types.SimpleNamespace(
            overrides={})), 640)


class ExportMetaImgszTest(unittest.TestCase):
    """_export_yolo_engine：引擎真实输入形状写入 meta（mock ultralytics）。"""

    def test_meta_records_resolved_imgsz(self):
        import xccv.engine.model_export as me

        exported = {}

        class FakeYOLO:
            def __init__(self, path):
                self.overrides = {"imgsz": [586, 1920]}

            def export(self, **kw):
                exported.update(kw)
                import os

                fd, p = tempfile.mkstemp(suffix=".engine")
                os.close(fd)  # 立即关句柄：Windows 下未关闭会阻塞后续 unlink
                out = Path(p)
                out.write_bytes(b"fake-engine")
                return str(out)

        orig = None
        import ultralytics

        orig = ultralytics.YOLO
        ultralytics.YOLO = FakeYOLO
        try:
            meta = {"kind": "yolo", "task": "segmentation", "classes": ["a"]}
            me._export_yolo_engine(Path("m.pt"), Path("out.engine"), None,
                                   "cuda", half=False, meta=meta)
        finally:
            ultralytics.YOLO = orig
        self.assertEqual(exported["imgsz"], [586, 1920])
        self.assertEqual(meta["imgsz"], [586, 1920])  # 反序列化失败时退回请求尺寸
        self.assertEqual(meta["task"], "segmentation")


class SuggestParseImgszTest(unittest.TestCase):
    """_suggest_imgsz 矩形返回 (高, 宽)；_parse_imgsz 双格式解析。"""

    def test_suggest_rect_from_train_args(self):
        from xccv.modeltools.modeltools_window import _suggest_imgsz

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.pt"
            torch.save({"train_args": {"imgsz": [586, 1920]}}, p)
            self.assertEqual(_suggest_imgsz(p), (586, 1920))
            p2 = Path(d) / "s.pt"
            torch.save({"train_args": {"imgsz": [640, 640]}}, p2)
            self.assertEqual(_suggest_imgsz(p2), 640)
            p3 = Path(d) / "old.pt"
            torch.save({"train_args": {"imgsz": 512}}, p3)
            self.assertEqual(_suggest_imgsz(p3), 512)

    def test_parse_imgsz(self):
        from xccv.modeltools.modeltools_window import _parse_imgsz

        self.assertEqual(_parse_imgsz("640"), 640)
        self.assertEqual(_parse_imgsz(" 586x1920 "), (586, 1920))
        self.assertEqual(_parse_imgsz("608×1952"), (608, 1952))
        self.assertEqual(_parse_imgsz("640*640"), 640)
        self.assertIsNone(_parse_imgsz("abc"))
        self.assertIsNone(_parse_imgsz("31"))
        self.assertIsNone(_parse_imgsz("999999"))
        self.assertIsNone(_parse_imgsz(""))
        self.assertIsNone(_parse_imgsz("1x2x3"))


if __name__ == "__main__":
    unittest.main()
