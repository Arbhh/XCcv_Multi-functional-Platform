# -*- coding: utf-8 -*-
"""复测回归：.engine/.engine 加载崩溃。

YOLO() 加载 TensorRT 引擎后 model.model 是引擎元数据字符串（str）而非
nn.Module，load_weights 无保护地 .to(device) 崩溃
AttributeError: 'str' object has no attribute 'to'。
修复=仅对 nn.Module 做 device 迁移。
回归用例：mock YOLO 返回 model.model=str，load_weights 不再崩溃且
num_classes/classes 正常来自 model.names。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xccv.engine.dataset.base import TaskType  # noqa: E402


class EngineLoadWeightsTest(unittest.TestCase):
    """P-06：.engine 元数据 str 不再做 .to(device)。"""

    def _load_with(self, model_attr):
        from xccv.engine.models.yolo import YoloModel

        m = YoloModel.__new__(YoloModel)
        m.device = "cpu"
        m.num_classes = 0
        m.classes = []
        m.task = TaskType.DETECTION

        fake_names = {0: "黑点", 1: "异色"}
        fake = type("FakeYOLO", (), {})()
        fake.model = model_attr
        fake.names = fake_names
        fake.task = "segment"
        with patch("xccv.engine.models.yolo.YOLO", return_value=fake):
            m.load_weights(Path("dummy.pt"))
        return m

    def test_engine_string_model_no_crash(self):
        m = self._load_with("TRT engine metadata string")
        self.assertEqual(m.num_classes, 2)
        self.assertEqual(m.classes, ["黑点", "异色"])
        self.assertEqual(m.task, TaskType.SEGMENTATION)

    def test_pt_module_still_moved(self):
        import torch

        moved = []
        net = torch.nn.Linear(2, 2)
        orig_to = net.to

        def spy(*a, **k):
            moved.append(a)
            return orig_to(*a, **k)

        net.to = spy
        m = self._load_with(net)
        self.assertEqual(len(moved), 1)  # nn.Module 仍执行 device 迁移
        self.assertEqual(m.num_classes, 2)


if __name__ == "__main__":
    unittest.main()
