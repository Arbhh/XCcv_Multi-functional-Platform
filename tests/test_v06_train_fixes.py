# -*- coding: utf-8 -*-
"""V0.6 P9：训练平台可选项——T9-1 freeze bug / T9-2 两段式缩放 / T9-3 切图训练。

- T9-1：maskrcnn18 骨干冻结标志必须真正生效（requires_grad 断言，CPU 可验）；
- T9-2：_apply_train_input 启用/回退边界（0=现状默认、32 倍数、小于画布短边）；
- T9-3：tiling 纯逻辑（网格覆盖/中心归属去重/可见面积过滤/背景保留/整图混入）
  + RcnnDataset 切图条目 CPU tiny 冒烟 + 开关关闭=现状（len 不变）。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

from xccv.engine.dataset.base import TaskType  # noqa: E402
from xccv.engine.dataset.tiling import (  # noqa: E402
    TileEntry, _starts, crop_target_boxes, plan_tile_entries,
)


class FreezeBackboneTest(unittest.TestCase):
    """T9-1：maskrcnn18 冻结标志（0 or 3 假值覆盖 bug 的回归）。"""

    def test_maskrcnn18_freeze_all(self):
        import torch

        from xccv.engine.models.rcnn import RCNNModel

        net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2,
                        device="cpu")._build("maskrcnn18", 2, 320,
                                             use_pretrained=True,
                                             freeze_backbone=True)
        body = [p for p in net.backbone.body.parameters()]
        self.assertTrue(body, "body 参数缺失")
        self.assertFalse(any(p.requires_grad for p in body),
                         "冻结标志下 body 必须全部 requires_grad=False")
        # FPN/RPN/头保持训练
        self.assertTrue(any(p.requires_grad for p in net.backbone.fpn.parameters()))
        self.assertTrue(any(p.requires_grad for p in net.rpn.parameters()))

    def test_maskrcnn18_default_trainable(self):
        from xccv.engine.models.rcnn import RCNNModel

        net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2,
                        device="cpu")._build("maskrcnn18", 2, 320,
                                             use_pretrained=False)
        body = list(net.backbone.body.parameters())
        self.assertTrue(any(p.requires_grad for p in body),
                       "默认（未冻结）body 应有可训练参数")


class ApplyTrainInputTest(unittest.TestCase):
    """T9-2：两段式缩放启用/回退边界。"""

    class _T:
        def __init__(self):
            self.min_size = (640,)
            self.max_size = (1333,)

    class _Net:
        def __init__(self):
            self.transform = ApplyTrainInputTest._T()

    def test_enable(self):
        from xccv.engine.models.rcnn import _apply_train_input

        net = self._Net()
        out = _apply_train_input(net, 512, 640, "maskrcnn18")
        self.assertEqual(out, 512)
        self.assertEqual(net.transform.min_size, (512,))
        self.assertEqual(net.transform.max_size, (round(512 * 4 / 3),))

    def test_zero_is_noop_default(self):
        from xccv.engine.models.rcnn import _apply_train_input

        net = self._Net()
        self.assertEqual(_apply_train_input(net, 0, 640, "maskrcnn18"), 0)
        self.assertEqual(net.transform.min_size, (640,), "默认路径零改动")

    def test_invalid_falls_back(self):
        from xccv.engine.models.rcnn import _apply_train_input

        net = self._Net()
        self.assertEqual(_apply_train_input(net, 500, 640, "maskrcnn18"), 0)  # 非 32 倍数
        self.assertEqual(_apply_train_input(net, 640, 640, "maskrcnn18"), 0)  # >= 画布短边
        self.assertEqual(_apply_train_input(net, 512, 640, "resnet18"), 0)  # 非 R-CNN 家族


class _FakeShape:
    def __init__(self, label, pts):
        self.label = label
        self.points = pts


class _FakeSample:
    def __init__(self, w, h, shapes):
        self.image_path = f"fake://{len(self.shapes_all)}.png"
        self.width, self.height = w, h
        self.shapes = shapes
        self.shapes_all = shapes


class TilingLogicTest(unittest.TestCase):
    """T9-3 纯逻辑：网格/归属去重/可见面积/背景保留/整图混入。"""

    def test_starts_coverage(self):
        s = _starts(1600, 400, 0.25)  # 步长 300
        self.assertEqual(s[0], 0)
        self.assertEqual(s[-1], 1200)  # 末边覆盖
        self.assertTrue(all(b > a for a, b in zip(s, s[1:])))
        self.assertEqual(_starts(300, 400, 0.2), [0])  # 整图小于 tile

    def test_center_assignment_dedup(self):
        # 框 300..480 中心 390：落在两窗口重叠区（[0,400) 与 [100,500) 都含）
        boxes = [[300.0, 10.0, 480.0, 110.0]]
        got0 = crop_target_boxes(boxes, 0, 0, 400, 1600, 200, 0.3)
        got1 = crop_target_boxes(boxes, 100, 0, 400, 1600, 200, 0.3)
        # 各自窗口都能看到完整框；但归属规则由 plan_tile_entries 的首个窗口
        # 承接（下一条断言验证全局唯一）
        self.assertEqual(len(got0), 1)
        self.assertEqual(len(got1), 1)

        class _S:
            width, height = 1600, 200

            class sh:  # noqa: E306
                label = "x"
                points = [[300.0, 10.0], [480.0, 110.0]]

            shapes = [sh]

        entries = plan_tile_entries([_S()], tile=400, overlap=0.25,
                                    bg_tile_keep=0.0, mix_full_view=0.0)
        owners = [e for e in entries if 0 in e.box_ids]
        # 中心 390 落在 x0=0/100/300 三个窗口内（重叠区），但全局归属唯一
        self.assertEqual(len(owners), 1,
                         f"重叠区去重=只归首个含中心窗口: "
                         f"{[(e.x, e.box_ids) for e in entries]}")
        self.assertEqual(owners[0].x, 0)

    def test_visibility_filter(self):
        # 框 250..450 中心 350∈[0,400)：窗口内可见 150/200=0.75
        boxes = [[250.0, 0.0, 450.0, 100.0]]
        got = crop_target_boxes(boxes, 0, 0, 400, 1600, 200, 0.9)
        self.assertEqual(got, [], "可见面积 0.75 < 0.9 应丢弃")
        got = crop_target_boxes(boxes, 0, 0, 400, 1600, 200, 0.3)
        self.assertEqual(len(got), 1)

    def test_bg_keep_and_mix_full(self):
        class _S:
            width, height = 1600, 1200
            shapes = [_FakeShape("x", [[10.0, 10.0], [30.0, 30.0]])]

        e0 = plan_tile_entries([_S()], tile=400, overlap=0.0, bg_tile_keep=0.0,
                               mix_full_view=0.0, seed=42)
        self.assertTrue(all(x is not None for x in
                            [en.x for en in e0 if True]), "bg_keep=0 无背景 tile")
        full0 = [en for en in e0 if en.x is None]
        self.assertEqual(len(full0), 0, "mix=0 无整图条目")
        e1 = plan_tile_entries([_S()], tile=400, overlap=0.0, bg_tile_keep=1.0,
                               mix_full_view=1.0, seed=42)
        full1 = [en for en in e1 if en.x is None]
        self.assertEqual(len(full1), 1, "mix=1 必有整图条目")
        self.assertGreater(len(e1), len(e0))

    def test_seeded_deterministic(self):
        class _S:
            width, height = 1000, 800
            shapes = [_FakeShape("x", [[10.0, 10.0], [30.0, 30.0]])]

        a = plan_tile_entries([_S()], seed=7)
        b = plan_tile_entries([_S()], seed=7)
        self.assertEqual([(en.sample_idx, en.x, en.y) for en in a],
                         [(en.sample_idx, en.x, en.y) for en in b])


class DatasetTiledTest(unittest.TestCase):
    """T9-3 RcnnDataset：开关关闭=现状；切图条目 tiny 冒烟（CPU）。"""

    def _samples(self, root: Path):
        import cv2

        img = np.zeros((600, 800, 3), np.uint8)
        cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
        p = root / "a.png"
        cv2.imwrite(str(p), img)

        class _Sh:
            label = "sq"
            points = [[100.0, 100.0], [200.0, 200.0]]

            def bbox(self):
                return (100.0, 100.0, 200.0, 200.0)

        class _S:
            pass

        s = _S()
        s.image_path = str(p)
        s.width, s.height = 800, 600
        s.shapes = [_Sh()]
        return [s]

    def test_off_is_status_quo(self):
        from xccv.engine.models.rcnn import RcnnDataset

        with tempfile.TemporaryDirectory() as td:
            samples = self._samples(Path(td))
            ds = RcnnDataset(samples, ["sq"], TaskType.DETECTION, train=True)
            self.assertEqual(len(ds), 1)

    def test_tiled_entries_smoke(self):
        import torch

        from xccv.engine.models.rcnn import RcnnDataset

        with tempfile.TemporaryDirectory() as td:
            samples = self._samples(Path(td))
            ds = RcnnDataset(samples, ["sq"], TaskType.DETECTION, train=True,
                             tiling={"tile": 300, "overlap": 0.0,
                                     "bg_tile_keep": 0.0,
                                     "mix_full_view": 0.0,
                                     "min_visibility": 0.3})
            self.assertGreaterEqual(len(ds), 1)
            img, target = ds[0]
            self.assertIsInstance(img, torch.Tensor)
            self.assertIn(img.shape[1], (300, 800))  # tile 或整图混入
            self.assertEqual(target["boxes"].reshape(-1, 4).shape[1], 4)


if __name__ == "__main__":
    unittest.main()


class SlimHeadTest(unittest.TestCase):
    """T9-5：maskrcnn18 slim 头（FPN64+两层256FC+4×64 掩码）。"""

    def test_structure_and_param_budget(self):
        from xccv.engine.models.rcnn import RCNNModel

        net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2, device="cpu",
                        use_pretrained=False)._build(
            "maskrcnn18", 2, 320, use_pretrained=False, slim_head=True)
        self.assertEqual(net.backbone.out_channels, 64)
        self.assertEqual(net.roi_heads.box_head.fc6.in_features, 64 * 7 * 7)
        self.assertEqual(net.roi_heads.box_head.fc6.out_features, 256)
        n = sum(p.numel() for p in net.parameters())
        self.assertTrue(11.2e6 < n < 13.8e6,
                        f"slim 参数量超出 12~13M±10%：{n/1e6:.2f}M")
        # 默认（非 slim）结构不受影响：FPN 仍是 256、参数量 ~30M
        net0 = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2, device="cpu",
                         use_pretrained=False)._build(
            "maskrcnn18", 2, 320, use_pretrained=False)
        self.assertEqual(net0.backbone.out_channels, 256)
        n0 = sum(p.numel() for p in net0.parameters())
        self.assertGreater(n0, 25e6)

    def test_freeze_backbone_under_slim(self):
        from xccv.engine.models.rcnn import RCNNModel

        net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2, device="cpu",
                        use_pretrained=True)._build(
            "maskrcnn18", 2, 320, use_pretrained=True,
            freeze_backbone=True, slim_head=True)
        body = list(net.backbone.body.parameters())
        self.assertTrue(body)
        self.assertFalse(any(p.requires_grad for p in body),
                         "slim 分支冻结语义须与 T9-1 一致")
        self.assertTrue(any(p.requires_grad
                            for p in net.backbone.fpn.parameters()))

    def test_meta_roundtrip_rebuilds_slim(self):
        import tempfile
        from pathlib import Path

        import torch

        from xccv.engine.models.rcnn import RCNNModel

        net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2, device="cpu",
                        use_pretrained=False)._build(
            "maskrcnn18", 2, 320, use_pretrained=False, slim_head=True)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best.pt"
            torch.save({"format": "xccv-rcnn", "arch": "maskrcnn18",
                        "task": "segmentation", "classes": ["a", "b"],
                        "min_size": 320, "slim_head": True,
                        "state_dict": net.state_dict()}, p)
            m = RCNNModel("fasterrcnn", TaskType.DETECTION, 1, device="cpu",
                          use_pretrained=False)
            m.load_weights(p)
            self.assertEqual(m.net.backbone.out_channels, 64,
                             "load_weights 须按 meta slim_head 重建结构")

    def test_converter_mapping_roundtrip(self):
        """tools/convert_mmdet_r18.py 映射表：slim 键逆映射→正向映射=恒等。"""
        import re

        from xccv.engine.models.rcnn import RCNNModel
        from tools.convert_mmdet_r18 import map_key, to_mmdet_for_test

        net = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2, device="cpu",
                        use_pretrained=False)._build(
            "maskrcnn18", 2, 320, use_pretrained=False, slim_head=True)
        model_sd = net.state_dict()
        fake = {to_mmdet_for_test(k): v for k, v in model_sd.items()}
        self.assertTrue(all(map_key(k) for k in fake),
                        "逆映射键必须都有正向规则")
        rt = {map_key(k): v for k, v in fake.items()}
        self.assertEqual(set(rt), set(model_sd))
        for k, v in rt.items():
            self.assertEqual(v.shape, model_sd[k].shape, k)
        net.load_state_dict(rt, strict=True)
