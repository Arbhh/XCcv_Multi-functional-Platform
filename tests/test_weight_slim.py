# -*- coding: utf-8 -*-
"""训练产物瘦身测试：YOLO ckpt 剥离 optimizer 等推理无关数据。"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from torch import nn

from xccv.engine.models.yolo import _lean_yolo_ckpt


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


class LeanYoloCkptTest(unittest.TestCase):
    def test_weights_in_ema_key_moved_to_model(self):
        """训练期 best.pt：权重挂在 ema、model=None → 归一到 model。"""
        net = _TinyNet()
        ckpt = {"epoch": 3, "best_fitness": 0.5, "model": None, "ema": net,
                "optimizer": {"state": {"x": torch.zeros(1000)}},
                "scaler": {"scale": 1.0}, "updates": 5,
                "train_args": {"task": "detect"}, "train_metrics": {},
                "train_results": {}, "date": "d", "version": "8.4.7"}
        lean = _lean_yolo_ckpt(ckpt)
        self.assertIs(lean["model"], net)
        for absent in ("optimizer", "scaler", "ema", "updates",
                       "train_metrics", "train_results"):
            self.assertNotIn(absent, lean)
        self.assertEqual(lean["epoch"], 3)
        self.assertEqual(lean["train_args"], {"task": "detect"})

    def test_weights_in_model_key_kept(self):
        """strip 后的 ckpt：权重在 model → 原样保留。"""
        net = _TinyNet()
        lean = _lean_yolo_ckpt({"model": net, "ema": None, "optimizer": None,
                                "epoch": 9})
        self.assertIs(lean["model"], net)

    def test_no_weights_raises(self):
        with self.assertRaises(RuntimeError):
            _lean_yolo_ckpt({"model": None, "ema": None})


if __name__ == "__main__":
    unittest.main()

