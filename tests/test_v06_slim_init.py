"""自建初始权重：构建器/注入优先级测试。

- tools/build_maskrcnn18_init.py：初始化配方落地 + 确定性 + 产物落盘；
- rcnn._load_slim_pretrained：**自建 init 优先**，转换产物 slim_coco
  仅最后兜底；类别头按形状
  匹配跳过（COCO 类数与项目类数不一致时新头随机初始化）。
全部 imagenet=False（离线、免下载），骨干随机但配方域可断言。
"""

import tempfile
import unittest
from pathlib import Path

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch

from xccv.engine.dataset.base import TaskType
from xccv.engine.models.rcnn import RCNNModel, _load_slim_pretrained
from build_maskrcnn18_init import _apply_recipe, build_init_net, save_artifact


class RecipeTest(unittest.TestCase):
    """配方落地与确定性（离线路径，骨干随机）。"""

    def test_recipe_lands_and_is_deterministic(self):
        import hashlib

        net, notes = build_init_net(num_classes=2, imagenet=False)
        self.assertEqual(notes["backbone"], "random")
        mp = net.roi_heads.mask_predictor
        self.assertLess(float(mp.mask_fcn_logits.weight.std()), 0.002)
        self.assertLess(
            float(net.roi_heads.box_predictor.cls_score.weight.std()), 0.012)
        self.assertEqual(
            float(net.roi_heads.box_predictor.cls_score.bias.abs().max()), 0.0)
        self.assertEqual(net.roi_heads.box_head.fc6.in_features, 64 * 7 * 7)
        n = sum(p.numel() for p in net.parameters())
        self.assertTrue(11.2e6 < n < 13.8e6, f"{n/1e6:.2f}M")
        net2, _ = build_init_net(num_classes=2, imagenet=False)
        keys = sorted(k for k in net.state_dict()
                      if not k.startswith("backbone.body."))

        def _h(sd):
            hh = hashlib.sha256()
            for k in keys:
                hh.update(k.encode())
                hh.update(sd[k].detach().cpu().contiguous()
                          .numpy().tobytes())
            return hh.hexdigest()

        self.assertEqual(_h(net.state_dict()), _h(net2.state_dict()))

    def test_build_uses_platform_class_semantics(self):
        """类别数口径：_build 内部 +1 背景；cls out=80+1（COCO 口径产物）。"""
        net, _ = build_init_net(num_classes=2, imagenet=False)
        self.assertEqual(
            net.roi_heads.box_predictor.cls_score.out_features, 3)


class ArtifactRoundtripTest(unittest.TestCase):
    def test_save_then_load_rebuilds_slim(self):
        net, _ = build_init_net(num_classes=2, imagenet=False)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "maskrcnn18_slim_init.pt"
            save_artifact(net, out)
            self.assertTrue(out.is_file())
            ckpt = torch.load(out, map_location="cpu", weights_only=False)
            m = RCNNModel("fasterrcnn", TaskType.DETECTION, 1, device="cpu",
                          use_pretrained=False)
            m.load_weights(ckpt)
            self.assertEqual(m.net.backbone.out_channels, 64)
            self.assertEqual(
                m.net.roi_heads.box_predictor.cls_score.out_features, 3)


class InjectionPriorityTest(unittest.TestCase):
    """_load_slim_pretrained：自建 init 优先，coco 转换产物仅最后兜底。"""

    def _make_net(self, nc):
        return RCNNModel("maskrcnn18", TaskType.SEGMENTATION, nc,
                         device="cpu", use_pretrained=False)._build(
            "maskrcnn18", nc, 320, use_pretrained=False, slim_head=True)

    def test_init_priority_then_coco_fallback(self):
        """源权重=自建 init 优先；coco 转换产物仅最后兜底。"""
        from unittest import mock

        from build_maskrcnn18_init import build_init_net as _b

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "weights").mkdir()
            net_init, _ = _b(num_classes=3, imagenet=False)
            save_artifact(net_init,
                          root / "weights" / "maskrcnn18_slim_init.pt")
            net_coco, _ = _b(num_classes=3, imagenet=False)
            sd = net_coco.state_dict()
            sd["backbone.body.conv1.weight"] = torch.full_like(
                sd["backbone.body.conv1.weight"], 1.5)
            tmp_pt = root / "coco.pt"
            torch.save({"format": "xccv-rcnn", "arch": "maskrcnn18",
                        "state_dict": sd}, tmp_pt)
            import shutil

            shutil.copy(tmp_pt, root / "weights" / "maskrcnn18_slim_coco.pt")
            tmp_pt.unlink()

            net = self._make_net(3)
            with mock.patch("xccv.engine.offline.WEIGHTS_DIR",
                            root / "weights"), \
                    mock.patch("xccv.app_paths.app_root", return_value=root):
                self.assertTrue(_load_slim_pretrained(net, 3))
            # 优先级：吃 init（非 coco 标记值）
            self.assertNotEqual(
                float(net.backbone.body.conv1.weight.flatten()[0]), 1.5)
            self.assertTrue(torch.equal(
                net.backbone.body.conv1.weight,
                net_init.backbone.body.conv1.weight))
            # 移除 init → 兜底 coco（标记值 1.5）
            (root / "weights" / "maskrcnn18_slim_init.pt").unlink()
            net2 = self._make_net(3)
            with mock.patch("xccv.engine.offline.WEIGHTS_DIR",
                            root / "weights"), \
                    mock.patch("xccv.app_paths.app_root", return_value=root):
                self.assertTrue(_load_slim_pretrained(net2, 3))
            self.assertEqual(
                float(net2.backbone.body.conv1.weight.flatten()[0]), 1.5)

    def test_class_heads_skipped_on_mismatch(self):
        """COCO 80 类产物注入 3 类项目：骨干注入、类别头保持随机（形状跳过）。"""
        from unittest import mock

        from build_maskrcnn18_init import build_init_net as _b

        net_init, _ = _b(num_classes=80, imagenet=False)
        net = self._make_net(3)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "weights").mkdir()
            save_artifact(net_init,
                          root / "weights" / "maskrcnn18_slim_init.pt")
            before = net.roi_heads.box_predictor.cls_score.weight.clone()
            with mock.patch("xccv.engine.offline.WEIGHTS_DIR",
                            root / "weights"), \
                    mock.patch("xccv.app_paths.app_root", return_value=root):
                self.assertTrue(_load_slim_pretrained(net, 3))
            self.assertTrue(torch.equal(
                net.backbone.body.conv1.weight,
                net_init.backbone.body.conv1.weight), "骨干应被注入")
            self.assertTrue(torch.equal(
                net.roi_heads.box_predictor.cls_score.weight, before),
                "类别数不匹配的头必须保持原值")


if __name__ == "__main__":
    unittest.main()

