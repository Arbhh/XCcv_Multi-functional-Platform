"""训练加速：冻结骨干（先冻结后插入，工业模块保持可训练）、RPN 削减、
AMP/早停参数贯通、UI 参数回环。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_classification_dataset, make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.engine.io_utils import read_json
from xccv.engine.models import TrainConfig, create_model
from xccv.engine.models.modules import CBAM, DCNv2
from xccv.engine.models.rcnn import RCNNModel
from xccv.training import project as project_mod
from xccv.training.project import create_project, load_project

_APP = QApplication.instance() or QApplication(sys.argv)


def _cls_cfg(dataset_dir: Path, out: Path, **kw) -> TrainConfig:
    base = dict(task=TaskType.CLASSIFICATION, dataset_dir=dataset_dir, output_dir=out,
                model_name="resnet18", epochs=1, batch_size=2, image_size=32,
                workers=0, use_pretrained=True, val_ratio=0.25)
    base.update(kw)
    return TrainConfig(**base)


class ClassifierFreezeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cls_dir, _ = make_classification_dataset(Path(self._tmp.name) / "cls", per_class=4)

    def test_freeze_keeps_head_and_cbam_trainable(self):
        model = create_model("resnet18", TaskType.CLASSIFICATION, 2,
                             use_pretrained=True, )
        cfg = _cls_cfg(self.cls_dir, Path(self._tmp.name) / "run",
                       industrial_opt=True, freeze_backbone=True)
        metrics = model.train(cfg)
        total = sum(p.numel() for p in model.net.parameters())
        trainable = metrics["trainable_params"]
        self.assertLess(trainable, total)            # 骨干被冻结
        self.assertGreater(trainable, 0)
        self.assertTrue(metrics["freeze_backbone"])
        # 工业模块（CBAM）在冻结之后插入 → 可训练
        cbam_params = [p for m in model.net.modules() if isinstance(m, CBAM)
                       for p in m.parameters()]
        self.assertTrue(cbam_params and all(p.requires_grad for p in cbam_params))
        # 原始骨干（layer1）冻结
        self.assertFalse(model.net.layer1[0].conv1.weight.requires_grad)
        # 分类头可训练
        self.assertTrue(model.net.fc.weight.requires_grad)

    def test_no_freeze_baseline(self):
        model = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=True)
        cfg = _cls_cfg(self.cls_dir, Path(self._tmp.name) / "run_base",
                       industrial_opt=False, freeze_backbone=False, use_amp=False)
        metrics = model.train(cfg)
        total = sum(p.numel() for p in model.net.parameters())
        self.assertEqual(metrics["trainable_params"], total)
        self.assertFalse(metrics["freeze_backbone"])
        rec = read_json(Path(self._tmp.name) / "run_base" / "metrics.json")
        self.assertFalse(rec["config"]["freeze_backbone"])
        self.assertFalse(rec["config"]["use_amp"])   # 参数快照贯通


class RcnnBuildFlagsTest(unittest.TestCase):
    def test_freeze_and_rpn_kwargs(self):
        model = RCNNModel("fasterrcnn", TaskType.DETECTION, 2, device="cpu",
                          use_pretrained=True, industrial=True)
        net = model._build("fasterrcnn", 2, 64, True, True,
                           freeze_backbone=True, rpn_reduced=True)
        # 原始骨干冻结（构造期冻结，DCN/CBAM 之后插入仍可训练）
        self.assertFalse(net.backbone.body.inner.layer1[0].conv1.weight.requires_grad)
        self.assertFalse(net.backbone.body.inner.layer2[0].conv1.weight.requires_grad)  # 全冻结（默认行为 layer2 可训）
        dcn = [m for m in net.modules() if isinstance(m, DCNv2)]
        self.assertTrue(dcn and all(p.requires_grad for m in dcn for p in m.parameters()))
        cbam = [m for m in net.backbone.body.cbams.modules() if isinstance(m, CBAM)]
        self.assertTrue(cbam and all(p.requires_grad for m in cbam for p in m.parameters()))
        # FPN / RPN / 头部可训练
        self.assertTrue(next(net.backbone.fpn.parameters()).requires_grad)
        self.assertTrue(next(net.roi_heads.box_head.parameters()).requires_grad)
        # RPN 候选削减生效（train 256 / test 100）
        self.assertEqual(net.rpn._post_nms_top_n, {"training": 256, "testing": 100})

    def test_default_unfrozen(self):
        model = RCNNModel("fasterrcnn", TaskType.DETECTION, 2, device="cpu",
                          use_pretrained=True, industrial=False)
        net = model.net
        # torchvision 默认冻结 stem+layer1（trainable=3），layer2 起可训练；
        # 我们不传参数时保持该默认行为不变
        self.assertTrue(net.backbone.body.layer2[0].conv1.weight.requires_grad)
        self.assertEqual(net.rpn._post_nms_top_n, {"training": 2000, "testing": 1000})

    def test_custom_max_size(self):
        """非方形输入映射：短边=宽高较小值，长边=较大值（保持长宽比）。"""
        model = RCNNModel("fasterrcnn", TaskType.DETECTION, 2, device="cpu",
                          use_pretrained=False, industrial=False)
        net = model._build("fasterrcnn", 2, 96, False, False, max_size=320)
        self.assertEqual(net.transform.min_size, (96,))
        self.assertEqual(net.transform.max_size, 320)


class YoloFreezeTest(unittest.TestCase):
    def test_freeze_reduces_trainable_params(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        det, _ = make_labelme_dataset(Path(tmp.name) / "det", n_images=6)
        model = create_model("yolo11n", TaskType.DETECTION, 2,
                             device="cpu", use_pretrained=True)
        from ultralytics import YOLO
        from xccv.engine.offline import WEIGHTS_DIR
        total = sum(p.numel() for p in YOLO(str(WEIGHTS_DIR / "ultralytics" / "yolo11n.pt")).model.parameters())
        metrics = model.train(TrainConfig(
            task=TaskType.DETECTION, dataset_dir=det, output_dir=Path(tmp.name) / "run",
            model_name="yolo11n", epochs=1, batch_size=2, image_size=64,
            workers=0, use_pretrained=True, val_ratio=0.25,
            freeze_backbone=True, use_amp=True, patience=3))
        self.assertTrue(metrics["freeze_backbone"])
        self.assertGreater(metrics["trainable_params"], 0)
        self.assertLess(metrics["trainable_params"], total)  # 骨干被冻结


class AccelUiRoundTripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        det, _ = make_labelme_dataset(Path(self._tmp.name) / "det", n_images=4)
        create_project("accel", str(det), "", "detection")

        from xccv.training.training_window import ProjectWindow

        self.pw = ProjectWindow(load_project("accel"))
        self.addCleanup(self.pw.close)

    def test_save_and_reload(self):
        self.pw.par_freeze.setChecked(True)
        self.pw.par_rpn.setChecked(True)
        self.pw.par_amp.setChecked(False)
        self.pw.par_tf32.setChecked(False)
        self.pw.par_patience.setValue(3)
        self.pw._save_params()
        p = load_project("accel")
        self.assertTrue(p.freeze_backbone)
        self.assertTrue(p.rpn_reduced)
        self.assertFalse(p.use_amp)
        self.assertFalse(p.tf32)
        self.assertEqual(p.patience, 3)
        # 重新打开窗口回填正确
        from xccv.training.training_window import ProjectWindow

        pw2 = ProjectWindow(load_project("accel"))
        self.addCleanup(pw2.close)
        self.assertTrue(pw2.par_freeze.isChecked())
        self.assertFalse(pw2.par_amp.isChecked())
        self.assertFalse(pw2.par_tf32.isChecked())
        self.assertEqual(pw2.par_patience.value(), 3)
        # 参数回显包含新字段
        self.pw._build_echo_form()
        labels = [self.pw.echo_form.itemAt(i, 0).widget().text()
                  for i in range(self.pw.echo_form.rowCount())]
        for key in ("冻结骨干", "RPN 候选削减", "AMP 混合精度", "GPU 提速(TF32)", "早停耐心"):
            self.assertIn(key, labels)


class NonSquareInputTest(unittest.TestCase):
    """非方形训练输入（与增强 resize 统一的宽×高）。"""

    def test_classifier_wh_input(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cls_dir, _ = make_classification_dataset(Path(tmp.name) / "cls", per_class=4)
        model = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
        metrics = model.train(TrainConfig(
            task=TaskType.CLASSIFICATION, dataset_dir=cls_dir,
            output_dir=Path(tmp.name) / "run", model_name="resnet18",
            epochs=1, batch_size=2, image_width=64, image_height=32,
            workers=0, use_pretrained=False, val_ratio=0.25))
        self.assertEqual(model._input_size, (32, 64))  # (高, 宽)
        self.assertTrue(metrics["status"] == "completed" or metrics.get("final_val_top1") is not None)
        # 权重保存（明文 best.pt）回读非方形尺寸
        import torch as _torch

        ckpt = _torch.load(Path(tmp.name) / "run" / "best.pt",
                           map_location="cpu", weights_only=False)
        self.assertEqual(list(ckpt["image_size"]), [32, 64])
        model.load_weights(ckpt)
        self.assertEqual(model._input_size, (32, 64))
        pred = model.predict(__import__("numpy").zeros((48, 96, 3), "uint8"))
        self.assertEqual(len(pred.probs), 2)


class MaskHeadLightTest(unittest.TestCase):
    """轻量掩码头：Mask R-CNN 专属，4 层 256 → 2 层 128。"""

    def test_light_head_structure(self):
        model = RCNNModel("maskrcnn", TaskType.SEGMENTATION, 2, device="cpu",
                          use_pretrained=True, industrial=True)
        net = model._build("maskrcnn", 2, 64, True, True, mask_head_light=True)
        # 分类卷积：输入 128 通道、输出 3 类（2 类 + 背景）
        self.assertEqual(net.roi_heads.mask_predictor.mask_fcn_logits.weight.shape,
                         (3, 128, 1, 1))
        # 头部只剩 2 层卷积，通道 128
        convs = [m for m in net.roi_heads.mask_head.modules()
                 if isinstance(m, torch.nn.Conv2d)]
        self.assertEqual(len(convs), 2)
        self.assertTrue(all(c.out_channels == 128 for c in convs))

    def test_default_head_unchanged(self):
        model = RCNNModel("maskrcnn", TaskType.SEGMENTATION, 2, device="cpu",
                          use_pretrained=True, industrial=True)
        self.assertEqual(model.net.roi_heads.mask_predictor.mask_fcn_logits.weight.shape,
                         (3, 256, 1, 1))

    def test_light_head_training_end_to_end(self):
        """轻量掩码头 + 工业 Dice 损失 + 1 轮端到端训练。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        det, _ = make_labelme_dataset(Path(tmp.name) / "det", n_images=6)
        model = create_model("maskrcnn", TaskType.SEGMENTATION, 2,
                             device="cpu", use_pretrained=False)
        metrics = model.train(TrainConfig(
            task=TaskType.SEGMENTATION, dataset_dir=det,
            output_dir=Path(tmp.name) / "run", model_name="maskrcnn",
            epochs=1, batch_size=2, image_size=64, workers=0,
            use_pretrained=False, val_ratio=0.25, mask_head_light=True))
        self.assertTrue(metrics["mask_head_light"])
        self.assertTrue(any("mask" in h for h in metrics["history"][0]))


if __name__ == "__main__":
    unittest.main()

