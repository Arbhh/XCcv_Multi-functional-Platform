"""模块行为锁：engine/models（模型注册元数据 + 权重容器契约）。"""

import tempfile
import unittest
from pathlib import Path


class RegistryMetaTest(unittest.TestCase):
    """注册表元数据契约（UI 免训练适配的数据源，additive 键）。"""

    def test_list_models_meta_contract(self):
        from xccv.engine.models.registry import list_models

        allm = list_models()
        self.assertGreaterEqual(len(allm), 10, "内置模型数不应缩水")
        by = {m["name"]: m for m in allm}
        for m in allm:
            self.assertIn(m["train_mode"], ("epoch", "free"))
            self.assertIsInstance(m["support_k_shot"], bool)
        for free in ("cif", "patchcore"):
            self.assertEqual(by[free]["train_mode"], "free",
                             "免训练模型元数据丢失（UI 适配会回退轮次语义）")
        self.assertTrue(by["subspacead"]["support_k_shot"],
                        "k-shot 标志丢失（快速配置换装会失效）")
        self.assertFalse(by["cif"]["support_k_shot"])
        det_names = {m["name"] for m in list_models("detection")}
        self.assertIn("yolo11n", det_names)

    def test_create_model_validates_combo(self):
        from xccv.engine.models.registry import create_model

        with self.assertRaises(ValueError):
            create_model("cif", "detection", 2, device="cpu")
        with self.assertRaises(ValueError):
            create_model("no_such_model", "anomaly", 2, device="cpu")

    def test_create_model_returns_base(self):
        from xccv.engine.models.base import BaseModel
        from xccv.engine.models.registry import create_model

        m = create_model("cif", "anomaly", 2, device="cpu",
                         use_pretrained=False)
        self.assertIsInstance(m, BaseModel)


class WeightsContainerTest(unittest.TestCase):
    """权重容器契约：明文 .pt 直接加载；旧版加密容器给迁移提示。"""

    def test_plaintext_weights_load_via_registry(self):
        import torch

        from xccv.engine.dataset.base import TaskType
        from xccv.engine.models import create_model
        from xccv.engine.models.registry import load_model

        m = create_model("resnet18", TaskType.CLASSIFICATION, 2,
                         device="cpu", use_pretrained=False)
        ckpt = {"format": "xccv-classifier", "arch": "resnet18",
                "task": "classification", "classes": ["a", "b"],
                "image_size": 64, "state_dict": m.net.state_dict()}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "best.pt"
            torch.save(ckpt, p)
            m2 = load_model(p)
            self.assertEqual(m2.classes, ["a", "b"])

    def test_legacy_container_raises_clear_error(self):
        """旧版加密容器（XCVA 魔数）→ 明确迁移提示，不抛晦涩解析错误。"""
        from xccv.engine.models.registry import load_model

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "legacy.pt"
            p.write_bytes(b"XCVA" + b"\x00" * 64)
            with self.assertRaises(ValueError) as ctx:
                load_model(p)
            self.assertIn("明文", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

