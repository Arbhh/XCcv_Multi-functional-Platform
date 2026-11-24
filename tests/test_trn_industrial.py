"""训练平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_industrial
- test_integration_models
"""


# == 来源模块：test_industrial ==
"""算法引擎工业优化测试：注意力模块、改进损失、工业训练端到端。"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PIL import Image

from tests.synthetic import make_classification_dataset, make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.engine.models import TrainConfig, create_model
from xccv.engine.models.modules import CBAM, DCNv2, ECA, FocalLoss, dice_bce_mask_loss


class ModulesTest(unittest.TestCase):
    def test_cbam_shape(self):
        m = CBAM(64)
        x = torch.randn(2, 64, 16, 16)
        self.assertEqual(m(x).shape, x.shape)

    def test_eca_shape(self):
        m = ECA(64)
        x = torch.randn(2, 64, 8, 8)
        self.assertEqual(m(x).shape, x.shape)

    def test_dcn_zero_offset_equals_conv(self):
        """offset 卷积零初始化 → DCNv2 初始等价于普通卷积。"""
        conv = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False)
        dcn = DCNv2(conv)
        x = torch.randn(1, 8, 32, 32)
        self.assertTrue(torch.allclose(dcn(x), conv(x), atol=1e-5))

    def test_focal_loss_value(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        loss = FocalLoss(gamma=2.0)(logits, targets)
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertFalse(torch.isnan(loss))


class DeepestStagesTest(unittest.TestCase):
    """CBAM/DCN 插入的 stage 通道按骨干实测（回归：硬编码 R50 通道表）。

    此前 _BodyWithCBAM 固定按 {"layer3": 1024, "layer4": 2048} 构造，
    maskrcnn18（ResNet18 的 layer3/layer4 = 256/512）开工业优化即
    mat1/mat2 形状崩溃。
    """

    def test_stage_channels_by_backbone(self):
        from torchvision.models.resnet import resnet18, resnet50

        from xccv.engine.models.rcnn import _deepest_stages

        self.assertEqual(_deepest_stages(resnet18()),
                         {"layer3": 256, "layer4": 512})
        self.assertEqual(_deepest_stages(resnet50()),
                         {"layer3": 1024, "layer4": 2048})

    def test_maskrcnn18_body_cbam_channels_match(self):
        from xccv.engine.models.rcnn import RCNNModel

        m = RCNNModel("maskrcnn18", TaskType.SEGMENTATION, 2, device="cpu",
                      use_pretrained=False, industrial=True)
        cbams = m.net.backbone.body.cbams
        self.assertEqual(set(cbams), {"layer3", "layer4"})
        x = torch.randn(1, 256, 8, 8)
        self.assertEqual(cbams["layer3"](x).shape, x.shape)  # 通道匹配可前向


class IndustrialTrainingTest(unittest.TestCase):
    """工业优化开关端到端：训练可跑、产出指标、权重可加载。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        out = Path(cls._tmp.name)
        cls.cls_dir, _ = make_classification_dataset(out / "cls", per_class=4)
        cls.det_dir, _ = make_labelme_dataset(out / "det", n_images=6)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _cfg(self, task, dataset, out, model, size, industrial):
        return TrainConfig(task=task, dataset_dir=dataset, output_dir=out,
                           model_name=model, epochs=1, batch_size=2, val_ratio=0.25,
                           image_size=size, workers=0, use_pretrained=False,
                           industrial_opt=industrial, patience=1)

    def test_classifier_industrial(self):
        m = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
        metrics = m.train(self._cfg(TaskType.CLASSIFICATION, self.cls_dir,
                                    Path(self._tmp.name) / "cls", "resnet18", 32, True))
        self.assertEqual(metrics["industrial_opt"], True)
        self.assertIn("label_smoothing", metrics["loss_type"])
        self.assertTrue((Path(self._tmp.name) / "cls" / "best.pt").exists())

    def test_yolo_industrial_loss_patch(self):
        m = create_model("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
        metrics = m.train(self._cfg(TaskType.DETECTION, self.det_dir,
                                    Path(self._tmp.name) / "yolo", "yolo11n", 64, True))
        self.assertEqual(metrics["industrial_opt"], True)
        self.assertIn("applied", metrics.get("loss_patch", ""))

    def test_fasterrcnn_industrial_component_losses(self):
        m = create_model("fasterrcnn", TaskType.DETECTION, 2, use_pretrained=False)
        metrics = m.train(self._cfg(TaskType.DETECTION, self.det_dir,
                                    Path(self._tmp.name) / "rcnn", "fasterrcnn", 96, True))
        self.assertEqual(metrics["industrial_opt"], True)
        self.assertTrue(any("classifier" in h for h in metrics["history"][0]))  # 组件损失上报
        self.assertTrue(any("rpn" in h for h in metrics["history"][0]))

    def test_maskrcnn_industrial_dice(self):
        seg_dir, _ = make_labelme_dataset(Path(self._tmp.name) / "seg", n_images=6,
                                          shape_type="polygon")
        m = create_model("maskrcnn", TaskType.SEGMENTATION, 2, use_pretrained=False)
        metrics = m.train(self._cfg(TaskType.SEGMENTATION, seg_dir,
                                    Path(self._tmp.name) / "mrcnn", "maskrcnn", 96, True))
        self.assertTrue(any("mask" in h for h in metrics["history"][0]))  # Dice 掩码损失上报

    def test_maskrcnn18_industrial(self):
        """回归：maskrcnn18 工业优化端到端（CBAM 通道按骨干实测）；
        对外指标为 bbox 口径，mask 口径仅内部记录。"""
        seg_dir, _ = make_labelme_dataset(Path(self._tmp.name) / "seg18", n_images=6,
                                          shape_type="polygon")
        m = create_model("maskrcnn18", TaskType.SEGMENTATION, 2, use_pretrained=False)
        metrics = m.train(self._cfg(TaskType.SEGMENTATION, seg_dir,
                                    Path(self._tmp.name) / "mrcnn18", "maskrcnn18", 96, True))
        self.assertEqual(metrics["industrial_opt"], True)
        h = metrics["history"][0]
        self.assertIn("map50", h)        # 展示口径 = bbox
        self.assertIn("mask_map50", h)   # mask 口径仅内部参数
        self.assertIn("map_loose", h)    # 宽松口径（IoU {0.1,0.25,0.5} 三档平均）
        self.assertIn("mar_loose", h)
        for k in ("map_loose", "map10", "map25", "mar_loose"):
            self.assertTrue(0.0 <= metrics["final"][k] <= 1.0)

    def test_mobilenet_industrial_eca(self):
        """回归：MobileNetV3 工业优化走 SE→ECA 路径（此前函数未定义导致 NameError）。"""
        m = create_model("mobilenet_v3_small", TaskType.CLASSIFICATION, 2, use_pretrained=False)
        metrics = m.train(self._cfg(TaskType.CLASSIFICATION, self.cls_dir,
                                    Path(self._tmp.name) / "mb", "mobilenet_v3_small", 32, True))
        self.assertEqual(metrics["industrial_opt"], True)
        self.assertIn("loss_type", metrics)

    def test_baseline_default_off_path(self):
        """关闭工业优化 → 基线对照路径仍可用。"""
        m = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
        metrics = m.train(self._cfg(TaskType.CLASSIFICATION, self.cls_dir,
                                    Path(self._tmp.name) / "base", "resnet18", 32, False))
        self.assertEqual(metrics["industrial_opt"], False)


class DiceLossTest(unittest.TestCase):
    def test_dice_bce_perfect_prediction(self):
        target = torch.zeros((1, 28, 28))
        target[0, 0:10, 0:10] = 1
        logits = torch.full((1, 28, 28), -10.0)
        logits[0, 0:10, 0:10] = 10.0  # sigmoid≈1 且与目标一致
        loss = dice_bce_mask_loss(logits, target)
        self.assertLess(float(loss), 0.3)

    def test_dice_bce_empty(self):
        logits = torch.zeros((1, 28, 28), requires_grad=True)
        target = torch.zeros((1, 28, 28))
        loss = dice_bce_mask_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))


class DCNv2AmpTest(unittest.TestCase):
    """DCNv2 在 AMP autocast 下强制 fp32（乱检/碎 mask 的根因修复）。"""

    def test_forward_under_autocast_is_fp32_and_matches(self):
        from xccv.engine.models.modules import DCNv2

        conv = torch.nn.Conv2d(8, 16, 3, padding=1)
        dcn = DCNv2(conv)
        x = torch.randn(2, 8, 24, 24)
        with torch.no_grad():
            ref = dcn(x)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = dcn(x)
        self.assertEqual(out.dtype, torch.float32)  # autocast 下仍输出 fp32
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-2)

    def test_rcnn_example_masks_shape(self):
        """_rcnn_ex 的 masks 必须是 (N, H, W)：maskrcnn 训练模式前向不再崩溃
        （此前 (H, W) 形状导致 torchvision roi_align IndexError）。"""
        from xccv.engine.models.pruning import _RcnnAdapter, _rcnn_ex
        from xccv.engine.models.rcnn import RCNNModel

        model = RCNNModel("maskrcnn", TaskType.SEGMENTATION, 2, device="cpu",
                          use_pretrained=False, industrial=True)
        adapter = _RcnnAdapter(model.net)
        total = float(adapter(_rcnn_ex()))  # 适配器返回损失和标量
        self.assertTrue(math.isfinite(total))

    def test_yolo_base_name(self):
        """分割任务自动补 -seg 且不重复叠加（修复 yolo11n-seg-seg.pt）。"""
        from xccv.engine.models.pruning import _yolo_base_name

        self.assertEqual(_yolo_base_name("yolo11n", TaskType.DETECTION), "yolo11n")
        self.assertEqual(_yolo_base_name("yolo11n", TaskType.SEGMENTATION), "yolo11n-seg")
        self.assertEqual(_yolo_base_name("yolo11n-seg", TaskType.SEGMENTATION), "yolo11n-seg")
        self.assertEqual(_yolo_base_name("yolo11s-seg", TaskType.SEGMENTATION), "yolo11s-seg")


class TrainStepGuardTest(unittest.TestCase):
    """非有限损失防护：NaN 步跳过（不更新参数），恢复后继续训练。"""

    def _make(self):
        from xccv.engine.models.rcnn import _rcnn_train_step

        net = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        return _rcnn_train_step, net, opt, scaler

    def test_nan_step_skipped_then_recovers(self):
        step, net, opt, scaler = self._make()

        class _FakeNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = net
                self.calls = 0

            def forward(self, imgs, tgts):
                self.calls += 1
                if self.calls == 1:
                    return {"loss_cls": torch.tensor(float("nan"))}
                return {"loss_cls": self.inner(imgs[0]).sum()}

        fake = _FakeNet()
        imgs, tgts = [torch.randn(3, 4)], [{}]
        loss, comp = step(fake, imgs, tgts, opt, scaler, use_amp=False)
        self.assertIsNone(loss)  # NaN 步被跳过
        self.assertIn("loss_cls", comp)
        self.assertFalse(torch.isfinite(torch.tensor(comp["loss_cls"])))
        w_before = net.weight.detach().clone()
        loss, _ = step(fake, imgs, tgts, opt, scaler, use_amp=False)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(w_before, net.weight.detach()))  # 参数已更新

    def test_finite_step_updates_params(self):
        step, net, opt, scaler = self._make()

        class _FakeNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = net

            def forward(self, imgs, tgts):
                return {"loss_cls": self.inner(imgs[0]).sum()}

        fake = _FakeNet()
        imgs, tgts = [torch.randn(3, 4)], [{}]
        w_before = net.weight.detach().clone()
        loss, comp = step(fake, imgs, tgts, opt, scaler, use_amp=False)
        self.assertIsNotNone(loss)
        self.assertIn("loss_cls", comp)
        self.assertFalse(torch.equal(w_before, net.weight.detach()))

    def test_nan_abort_threshold(self):
        from xccv.engine.models.rcnn import _nan_abort

        self.assertFalse(_nan_abort(0, 42))
        self.assertFalse(_nan_abort(4, 42))
        self.assertTrue(_nan_abort(13, 42))   # ≥30% 且 ≥5
        self.assertTrue(_nan_abort(5, 10))


class CbamIdentityInitTest(unittest.TestCase):
    """CBAM 恒等门控：初始输出严格等于输入，gate 打开后恢复注意力混合。"""

    def test_identity_at_init(self):
        m = CBAM(16)
        m.eval()
        x = torch.randn(2, 16, 8, 8)
        self.assertTrue(torch.equal(m(x), x))

    def test_gate_opens(self):
        m = CBAM(16)
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            m.gate.fill_(1.0)
        expected = x + m.spatial_attn(m.channel_attn(x))
        self.assertTrue(torch.allclose(m(x), expected))


class VramBatchCapTest(unittest.TestCase):
    """显存预算 batch 下调：只在空闲显存不足时生效。"""

    def setUp(self):
        self._orig = torch.cuda.mem_get_info

    def tearDown(self):
        torch.cuda.mem_get_info = self._orig

    def _fake_mem(self, free_gb):
        torch.cuda.mem_get_info = lambda: (free_gb << 30, 16 << 30)

    def test_caps_big_images_on_small_free_vram(self):
        from xccv.engine.models.rcnn import _vram_batch_cap

        self._fake_mem(4)  # 空闲 4GB：1600px AMP 单图约 5.6GB → 只能 1
        self.assertEqual(_vram_batch_cap(8, 1600, True), 1)

    def test_keeps_small_request_when_vram_enough(self):
        from xccv.engine.models.rcnn import _vram_batch_cap

        self._fake_mem(16)
        self.assertEqual(_vram_batch_cap(2, 640, True), 2)

    def test_fallback_when_query_fails(self):
        from xccv.engine.models.rcnn import _vram_batch_cap

        def boom():
            raise RuntimeError("no cuda")

        torch.cuda.mem_get_info = boom
        self.assertEqual(_vram_batch_cap(3, 640, False), 3)


class CpuMaskPostprocessTest(unittest.TestCase):
    """验证/推理期 CPU 分块贴 mask：缩放结果与原生实现一致，实例补丁可恢复。"""

    def _make_net(self):
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        class _Net:
            def __init__(self):
                self.transform = GeneralizedRCNNTransform(
                    64, 128, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return _Net()

    def test_boxes_and_masks_rescaled_to_original(self):
        from xccv.engine.models.rcnn import (_install_cpu_mask_postprocess,
                                             _restore_mask_postprocess)

        net = self._make_net()
        _install_cpu_mask_postprocess(net)
        try:
            im_s = torch.Size([100, 200])  # 模型输入尺度（高, 宽）
            o_s = torch.Size([200, 400])   # 原图尺度
            pred = [{
                "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
                "masks": torch.rand(1, 1, 100, 200),
            }]
            out = net.transform.postprocess(pred, [im_s], [o_s])[0]
            self.assertTrue(torch.allclose(
                out["boxes"], torch.tensor([[20.0, 20.0, 120.0, 120.0]])))
            self.assertEqual(out["masks"].shape, (1, 1, 200, 400))
            self.assertEqual(out["masks"].device.type, "cpu")
        finally:
            _restore_mask_postprocess(net)
        self.assertNotIn("postprocess", net.transform.__dict__)

    def test_empty_detections(self):
        from xccv.engine.models.rcnn import (_install_cpu_mask_postprocess,
                                             _restore_mask_postprocess)

        net = self._make_net()
        _install_cpu_mask_postprocess(net)
        try:
            pred = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
                     "scores": torch.zeros(0), "masks": torch.zeros(0, 1, 100, 200)}]
            out = net.transform.postprocess(
                pred, [torch.Size([100, 200])], [torch.Size([200, 400])])[0]
            self.assertEqual(out["masks"].shape, (0, 1, 200, 400))
        finally:
            _restore_mask_postprocess(net)


class OkImageBackgroundTest(unittest.TestCase):
    """OK 图（空标注）作为纯背景负样本参与 R-CNN 训练。"""

    def test_empty_target_kept_and_forwardable(self):
        import numpy as np

        from xccv.engine.dataset.base import Sample, Shape
        from xccv.engine.models.rcnn import RcnnDataset, _collate
        from torch.utils.data import DataLoader

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ok_path = Path(tmp.name) / "ok.png"
        obj_path = Path(tmp.name) / "obj.png"
        for p in (ok_path, obj_path):
            Image.new("RGB", (128, 128), (128, 128, 128)).save(p)

        samples = [
            Sample(image_path=ok_path, width=128, height=128, shapes=[]),  # OK 图
            Sample(image_path=obj_path, width=128, height=128,
                   shapes=[Shape(label="defect", shape_type="rectangle",
                                 points=[[20.0, 20.0], [60.0, 60.0]])]),
        ]
        ds = RcnnDataset(samples, ["defect"], TaskType.SEGMENTATION, train=True)
        self.assertEqual(len(ds), 2)  # 空标注图不再被丢弃
        loader = DataLoader(ds, batch_size=2, collate_fn=_collate)
        imgs, targets = next(iter(loader))

        model = create_model("maskrcnn", TaskType.SEGMENTATION, 1,
                             device="cpu", use_pretrained=False)
        model.net.transform.min_size = (96,)
        model.net.transform.max_size = 128
        model.net.train()
        imgs = [im.to(model.device) for im in imgs]
        tgts = [{k: v.to(model.device) for k, v in t.items()} for t in targets]
        self.assertEqual(int(tgts[0]["boxes"].shape[0]), 0)  # OK 图目标为空
        self.assertEqual(tuple(tgts[0]["masks"].shape), (0, 128, 128))
        with torch.no_grad():
            loss_dict = model.net(imgs, tgts)
        loss = sum(loss_dict.values())
        self.assertTrue(torch.isfinite(loss))

    def test_transform_size_attributes(self):
        model = create_model("maskrcnn", TaskType.SEGMENTATION, 1,
                             device="cpu", use_pretrained=False)
        # 改训练/评估输入尺寸的直接入口是 GeneralizedRCNNTransform 的属性
        model.net.transform.min_size = (96,)
        model.net.transform.max_size = 128
        self.assertEqual(model.net.transform.min_size, (96,))
        self.assertEqual(model.net.transform.max_size, 128)


class GradCheckpointTest(unittest.TestCase):
    """梯度检查点：前向等价、反向可用、restore 摘除实例补丁。"""

    def _tiny_net(self):
        model = create_model("fasterrcnn", TaskType.DETECTION, 1,
                             device="cpu", use_pretrained=False)
        model.net.transform.min_size = (96,)
        model.net.transform.max_size = 128
        return model

    def _batch(self, model):
        import numpy as np

        from xccv.engine.dataset.base import Sample, Shape

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = Path(tmp.name) / "a.png"
        Image.new("RGB", (128, 128), (90, 90, 90)).save(p)
        samples = [Sample(image_path=p, width=128, height=128,
                          shapes=[Shape(label="defect", shape_type="rectangle",
                                        points=[[20.0, 20.0], [60.0, 60.0]])])]
        from xccv.engine.dataset.base import Sample, Shape
        from xccv.engine.models.rcnn import RcnnDataset, _collate
        from torch.utils.data import DataLoader

        ds = RcnnDataset(samples, ["defect"], TaskType.DETECTION, train=True)
        imgs, tgts = next(iter(DataLoader(ds, batch_size=1, collate_fn=_collate)))
        imgs = [im.to(model.device) for im in imgs]
        tgts = [{k: v.to(model.device) for k, v in t.items()} for t in tgts]
        return imgs, tgts

    def test_checkpoint_train_step_and_restore(self):
        model = self._tiny_net()
        imgs, tgts = self._batch(model)
        from xccv.engine.models.rcnn import (_apply_grad_checkpoint,
                                             _restore_grad_checkpoint)

        model.net.train()
        orig_body_forward = model.net.backbone.body.forward
        patched = _apply_grad_checkpoint(model.net)
        self.assertIsNot(model.net.backbone.body.forward, orig_body_forward)
        loss_dict = model.net(imgs, tgts)
        loss = sum(loss_dict.values())
        loss.backward()
        grads = [p.grad for p in model.net.backbone.body.parameters()
                 if p.grad is not None and p.requires_grad]
        self.assertTrue(len(grads) > 0)  # 检查点路径下梯度仍可回传
        _restore_grad_checkpoint(patched)
        self.assertNotIn("forward", model.net.backbone.body.__dict__)

    def test_eval_falls_back_to_original_forward(self):
        model = self._tiny_net()
        imgs, tgts = self._batch(model)
        from xccv.engine.models.rcnn import (_apply_grad_checkpoint,
                                             _restore_grad_checkpoint)

        model.net.eval()
        orig_body_forward = model.net.backbone.body.forward
        patched = _apply_grad_checkpoint(model.net)
        try:
            with torch.no_grad():
                outs = model.net(imgs)
            self.assertIn("boxes", outs[0])  # 评估走原路径且输出正常
        finally:
            _restore_grad_checkpoint(patched)
        self.assertNotIn("forward", model.net.backbone.body.__dict__)

# == 来源模块：test_integration_models ==
"""模型集成测试：小合成数据集上完成 训练->指标->预测->权重加载 全流程。

CPU 上运行较慢（每个模型约 1-3 分钟），与快速单元测试分开执行。
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.base import TaskType
from xccv.engine.models.base import TrainConfig, TrainReporter
from xccv.engine.models.registry import create_model, load_model
from tests.synthetic import make_classification_dataset, make_labelme_dataset


class Recorder(TrainReporter):
    """记录回调，供断言训练过程中上报了指标。"""

    def __init__(self):
        self.epochs = []
        self.finished = None

    def on_epoch(self, epoch, logs):
        self.epochs.append((epoch, logs))

    def on_finish(self, success, message=""):
        self.finished = (success, message)


class ModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.det_dir, _ = make_labelme_dataset(cls.root / "det", n_images=8)
        cls.seg_dir, _ = make_labelme_dataset(cls.root / "seg", n_images=6, shape_type="polygon")
        cls.cls_dir, _ = make_classification_dataset(cls.root / "cls", per_class=6)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _cfg(self, task, dataset, out, image_size=0):
        return TrainConfig(task=task, dataset_dir=dataset, output_dir=out,
                           model_name="test", epochs=1, batch_size=2, val_ratio=0.25,
                           eval_interval=1, use_pretrained=False, workers=0,
                           image_size=image_size)

    def test_classifier_train_predict_load(self):
        cfg = self._cfg(TaskType.CLASSIFICATION, self.cls_dir,
                        self.root / "out_cls", image_size=32)
        model = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
        rec = Recorder()
        metrics = model.train(cfg, rec)
        self.assertTrue(rec.finished[0])
        self.assertIn("val_top1", rec.epochs[-1][1])
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)

        pred = model.predict(np.zeros((40, 40, 3), np.uint8))
        self.assertEqual(len(pred.probs), 2)
        self.assertEqual(len(pred.labels), 1)

        # 训练产物 best.pt 直接加载并恢复类别
        m2 = load_model(cfg.output_dir / "best.pt")
        self.assertEqual(m2.classes, ["cat", "dog"])
        m2.predict(np.zeros((40, 40, 3), np.uint8))

    def test_yolo_detection_train_predict_load(self):
        cfg = self._cfg(TaskType.DETECTION, self.det_dir,
                        self.root / "out_yolo", image_size=64)
        model = create_model("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
        rec = Recorder()
        metrics = model.train(cfg, rec)
        self.assertTrue(rec.finished[0])
        self.assertIn("train_loss", rec.epochs[-1][1])
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 3)  # 2 类 + 背景

        pred = model.predict(np.zeros((96, 96, 3), np.uint8), conf=0.25)
        self.assertIsInstance(pred.boxes, list)
        self.assertEqual(len(pred.boxes), len(pred.scores))

        m2 = load_model(cfg.output_dir / "best.pt")
        self.assertEqual(m2.num_classes, 2)
        m2.predict(np.zeros((96, 96, 3), np.uint8))

    def test_yolo_segmentation_train_predict(self):
        cfg = self._cfg(TaskType.SEGMENTATION, self.seg_dir,
                        self.root / "out_yoloseg", image_size=64)
        model = create_model("yolo11n-seg", TaskType.SEGMENTATION, 2, use_pretrained=False)
        rec = Recorder()
        metrics = model.train(cfg, rec)
        self.assertTrue(rec.finished[0])
        pred = model.predict(np.zeros((96, 96, 3), np.uint8))
        self.assertTrue(pred.masks is None or len(pred.masks) == len(pred.boxes))

    def test_fasterrcnn_train_predict_load(self):
        cfg = self._cfg(TaskType.DETECTION, self.det_dir,
                        self.root / "out_rcnn", image_size=96)
        model = create_model("fasterrcnn", TaskType.DETECTION, 2, use_pretrained=False)
        rec = Recorder()
        metrics = model.train(cfg, rec)
        self.assertTrue(rec.finished[0])
        self.assertIn("map50", rec.epochs[-1][1])
        self.assertIn("confusion_matrix", metrics)

        pred = model.predict(np.zeros((96, 96, 3), np.uint8), conf=0.5)
        self.assertIsInstance(pred.boxes, list)

        m2 = load_model(cfg.output_dir / "best.pt")
        self.assertEqual(m2.classes, ["cat", "dog"])
        m2.predict(np.zeros((96, 96, 3), np.uint8))

    def test_maskrcnn_train_predict(self):
        cfg = self._cfg(TaskType.SEGMENTATION, self.seg_dir,
                        self.root / "out_mrcnn", image_size=96)
        model = create_model("maskrcnn", TaskType.SEGMENTATION, 2, use_pretrained=False)
        rec = Recorder()
        metrics = model.train(cfg, rec)
        self.assertTrue(rec.finished[0])
        pred = model.predict(np.zeros((96, 96, 3), np.uint8), conf=0.5)
        self.assertTrue(pred.masks is None or len(pred.masks) == len(pred.boxes))

if __name__ == '__main__':
    unittest.main()
