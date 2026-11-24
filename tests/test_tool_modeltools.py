"""平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_modeltools
- test_acceleration
"""


# == 来源模块：test_modeltools ==
"""模型工具平台：三种权重格式的剪枝、剪后可加载推理、工具卡片墙与独立工具窗口
（剪枝窗口 / 加速导出窗口）。"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMessageBox

from tests.synthetic import make_classification_dataset, make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.engine.models import TrainConfig, create_model, load_model
from xccv.engine.models.pruning import prune_checkpoint
from xccv.training import project as project_mod
from xccv.training.project import create_project, load_project

_APP = QApplication.instance() or QApplication(sys.argv)


HEAVY = os.environ.get("XCCV_HEAVY_TESTS")


def _train_classifier(out: Path, dataset: Path) -> Path:
    model = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
    model.train(TrainConfig(
        task=TaskType.CLASSIFICATION, dataset_dir=dataset, output_dir=out,
        model_name="resnet18", epochs=1, batch_size=2, image_size=32,
        workers=0, use_pretrained=False, val_ratio=0.25))
    return out / "best.pt"


def _train_yolo(out: Path, dataset: Path) -> Path:
    model = create_model("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
    model.train(TrainConfig(
        task=TaskType.DETECTION, dataset_dir=dataset, output_dir=out,
        model_name="yolo11n", epochs=1, batch_size=2, image_size=64,
        workers=0, use_pretrained=False, val_ratio=0.25))
    return out / "best.pt"


def _train_fasterrcnn(out: Path, dataset: Path) -> Path:
    model = create_model("fasterrcnn", TaskType.DETECTION, 2, use_pretrained=False)
    model.train(TrainConfig(
        task=TaskType.DETECTION, dataset_dir=dataset, output_dir=out,
        model_name="fasterrcnn", epochs=1, batch_size=2, image_size=64,
        workers=0, use_pretrained=False, val_ratio=0.25))
    return out / "best.pt"


class PruneCheckpointTest(unittest.TestCase):
    """分类剪枝（轻负载，本机可测）：参数量下降、剪后可被测试平台直接加载推理。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cls_dir, _ = make_classification_dataset(self.root / "cls", per_class=4)
        self.det_dir, _ = make_labelme_dataset(self.root / "det", n_images=6)

    def test_prune_classifier_ckpt(self):
        src = _train_classifier(self.root / "run_cls", self.cls_dir)
        before = sum(p.numel() for p in load_model(src).net.parameters())
        result = prune_checkpoint(src, self.root / "cls_pruned.pt", 0.3)
        self.assertEqual(result["status"], "applied", result)
        self.assertLess(result["params_after"], result["params_before"])
        self.assertLess(result["params_after"], before)
        # 测试平台加载剪后模型并推理
        model = load_model(self.root / "cls_pruned.pt")
        pred = model.predict(np.zeros((32, 32, 3), np.uint8))
        self.assertEqual(len(pred.probs), 2)

    def test_invalid_ratio_reported(self):
        src = _train_classifier(self.root / "run_cls2", self.cls_dir)
        result = prune_checkpoint(src, self.root / "noop.pt", 0.0)
        self.assertEqual(result["status"], "off")


@unittest.skipUnless(
    HEAVY,
    "重负载剪枝验证（R-CNN/YOLO 追踪开销大）默认迁移至工控机执行；"
    "XCCV_HEAVY_TESTS=1 可在本机强制运行")
class HeavyPruneTest(unittest.TestCase):
    """R-CNN / YOLO 剪枝（重负载）：在工控机上验证。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.det_dir, _ = make_labelme_dataset(self.root / "det", n_images=6)

    def test_prune_fasterrcnn_ckpt(self):
        src = _train_fasterrcnn(self.root / "run_rcnn", self.det_dir)
        result = prune_checkpoint(src, self.root / "rcnn_pruned.pt", 0.2)
        self.assertEqual(result["status"], "applied", result)
        self.assertLess(result["params_after"], result["params_before"])
        model = load_model(self.root / "rcnn_pruned.pt")
        pred = model.predict(np.zeros((64, 64, 3), np.uint8), conf=0.5)
        self.assertIsInstance(pred.boxes, list)

    def test_prune_yolo_ckpt(self):
        src = _train_yolo(self.root / "run_yolo", self.det_dir)
        result = prune_checkpoint(src, self.root / "yolo_pruned.pt", 0.2)
        if os.environ.get("XCCV_YOLO_PRUNE") != "1":
            # 受控降级：TP 上游对 ultralytics 图内存爆炸，默认快速失败
            self.assertIn("yolo-unsupported", result["status"], result)
            self.assertFalse((self.root / "yolo_pruned.pt").exists())
            return
        self.assertEqual(result["status"], "applied", result)
        self.assertLess(result["params_after"], result["params_before"])
        model = load_model(self.root / "yolo_pruned.pt")
        pred = model.predict(np.zeros((64, 64, 3), np.uint8), conf=0.1)
        self.assertIsInstance(pred.boxes, list)

    def test_pretrain_prune_yolo(self):
        """训练前剪枝 YOLO（实验性）：applied 且剪后参数量下降。"""
        from xccv.engine.models.pruning import pretrain_prune

        result = pretrain_prune("yolo11n", "detection", 2, 0.1,
                                classes=["cat", "dog"])
        if os.environ.get("XCCV_YOLO_PRUNE") != "1":
            self.assertIn("yolo-unsupported", result["status"], result)
            return
        self.assertEqual(result["status"], "applied", result)
        dst = self.root / "yolo11n_pruned10.pt"
        torch.save(result["ckpt"], dst)
        model = load_model(dst)
        pred = model.predict(np.zeros((64, 64, 3), np.uint8), conf=0.1)
        self.assertIsInstance(pred.boxes, list)


class AccelerateExportTest(unittest.TestCase):
    """TensorRT 引擎导出。"""

    def test_export_requires_tensorrt(self):
        """无 TensorRT 环境时给出明确指引（在工控机安装后可构建）。"""
        from xccv.engine import model_export

        if model_export.tensorrt_available():
            self.skipTest("本机已安装 TensorRT：完整构建见 AccelerateBuildOnGpuTest")
        with self.assertRaises(RuntimeError) as ctx:
            model_export.export_engine(Path("nonexistent.pt"), Path("m.engine"))
        self.assertIn("TensorRT", str(ctx.exception))

    def test_engine_container_roundtrip(self):
        """XCVE 引擎容器：pack/unpack 元数据头往返无损。"""
        from xccv.engine import model_export

        data = b"TRT-serialized-engine" * 256
        blob = model_export.pack_engine_blob({"kind": "yolo", "classes": []}, data)
        meta, engine, _extra = model_export.unpack_engine_blob(blob)
        self.assertEqual(meta["kind"], "yolo")
        self.assertEqual(engine, data)


def _gpu_trt_ready() -> bool:
    try:
        import torch

        from xccv.engine import model_export

        return bool(torch.cuda.is_available() and model_export.tensorrt_available())
    except Exception:
        return False


@unittest.skipUnless(_gpu_trt_ready(), "需要 NVIDIA GPU + TensorRT 环境")
class AccelerateBuildOnGpuTest(unittest.TestCase):
    """完整构建（具备 GPU + TensorRT 环境即执行）：
    小分类模型 → TensorRT FP32 引擎 .engine → 元数据记录本机算力 → 测试平台
    统一入口加载 → 推理结果与源权重一致（类别数/概率和）。"""

    def test_classifier_build_load_predict(self):
        from xccv.engine import model_export
        from xccv.engine.trt_runtime import local_compute_capability

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cls_dir, _ = make_classification_dataset(root / "cls", per_class=4)
            src = _train_classifier(root / "run", cls_dir)
            info = model_export.export_engine(src, root / "m.engine", device="cuda",
                                              half=False)
            dst = Path(info["dst"])
            self.assertTrue(dst.is_file())
            meta, engine, _extra = model_export.unpack_engine_blob(dst.read_bytes())
            self.assertEqual(meta["kind"], "classifier")
            self.assertEqual(tuple(meta["cc_build"]), local_compute_capability())
            self.assertTrue(meta.get("trt_build"))
            self.assertGreater(len(engine), 1000)
            img = np.random.default_rng(1).integers(0, 255, (32, 32, 3), dtype=np.uint8)
            ref = load_model(src, device="cuda").predict(img)
            got = load_model(dst).predict(img)
            self.assertEqual(len(got.probs), 2)
            self.assertAlmostEqual(sum(got.probs), 1.0, places=4)
            self.assertEqual(got.labels, ref.labels)   # FP32 引擎：top1 与源权重一致


def _wait_worker(window, timeout: int = 30):
    """等待工具窗口的后台任务结束并处理事件。"""
    deadline = time.time() + timeout
    while window._worker is not None and time.time() < deadline:
        window._worker.wait(50)
        QApplication.processEvents()


class PrePruneTest(unittest.TestCase):
    """训练前剪枝：按数据集预剪枝生成剪枝模型，训练平台读取并迁移学习。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.cls_dir, _ = make_classification_dataset(self.root / "cls", per_class=4)

    def test_pretrain_prune_classifier(self):
        """分类预剪枝：applied、可加载、参数量下降。"""
        from xccv.engine.models.pruning import pretrain_prune

        result = pretrain_prune("resnet18", "classification", 2, 0.3,
                                classes=["cat", "dog"])
        self.assertEqual(result["status"], "applied", result)
        dst = self.root / "resnet18_pruned30.pt"
        torch.save(result["ckpt"], dst)
        self.assertTrue(dst.is_file())
        m = load_model(dst)
        unpruned = sum(p.numel() for p in create_model(
            "resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False).net.parameters())
        n = sum(p.numel() for p in m.net.parameters())
        self.assertLess(n, unpruned)
        self.assertEqual(len(m.predict(np.zeros((32, 32, 3), np.uint8)).probs), 2)

    def test_train_from_pruned_model(self):
        """训练平台读取项目文件夹下的剪枝模型进行迁移学习（端到端）。"""
        from xccv.engine.models.pruning import pretrain_prune
        from xccv.training.trainer_thread import TrainThread

        project_mod.PROJECTS_DIR = self.root / "projects"
        project = create_project("pp", str(self.cls_dir),
                                 str(self.root / "models"), "classification")
        project.epochs = 1
        project.batch_size = 2
        project.workers = 0
        project.augment_ops = {"resize": {"width": 32, "height": 32}}
        result = pretrain_prune("resnet18", "classification", 2, 0.2,
                                classes=["cat", "dog"])
        pruned_dir = Path(project.model_dir) / "pruned"
        pruned_dir.mkdir(parents=True)
        dst = pruned_dir / "resnet18_pruned20.pt"
        torch.save(result["ckpt"], dst)
        project.model_name = "pruned:resnet18_pruned20.pt"

        run_dir = Path(project.model_dir) / "run1"
        thread = TrainThread(project, run_dir)
        thread.run()  # 同步执行
        self.assertTrue((run_dir / "best.pt").is_file())
        # 剪后模型训练产出的权重同样可加载
        m = load_model(run_dir / "best.pt")
        self.assertEqual(len(m.predict(np.zeros((32, 32, 3), np.uint8)).probs), 2)

    def test_build_yolo_net_uses_dataset_class_names(self):
        """YOLO 预剪枝网络必须带数据集类别名（否则训练平台类别校验误报不一致）。"""
        from xccv.engine.models.pruning import _build_yolo_net

        net = _build_yolo_net("yolo11n", 2, TaskType.DETECTION, ["cat", "dog"])
        self.assertEqual(net.names, {0: "cat", 1: "dog"})
        self.assertEqual(net.yaml["nc"], 2)

    def test_train_from_yolo_pruned_checkpoint(self):
        """YOLO 剪枝模型恢复训练全链路（剪后网络随 ckpt 挂载）。"""
        from xccv.engine.models.pruning import _build_yolo_net
        from xccv.training.trainer_thread import TrainThread

        project_mod.PROJECTS_DIR = self.root / "projects3"
        det, _ = make_labelme_dataset(self.root / "det3", n_images=6)
        project = create_project("py", str(det), str(self.root / "models3"), "detection")
        project.epochs = 1
        project.batch_size = 2
        project.workers = 0
        # 模拟训练前剪枝产物（跳过耗时的 TP 追踪，ckpt 结构与 pretrain_prune 一致）
        net = _build_yolo_net("yolo11n", 2, TaskType.DETECTION, ["cat", "dog"])
        ckpt = {"model": net, "train_args": {"task": "detect"},
                "xccv_classes": ["cat", "dog"],
                "xccv_pruned_pretrain": {"ratio": 0.1, "detail": "test"}}
        pruned_dir = Path(project.model_dir) / "pruned"
        pruned_dir.mkdir(parents=True)
        dst = pruned_dir / "yolo11n_pruned10.pt"
        torch.save(ckpt, dst)
        project.model_name = "pruned:yolo11n_pruned10.pt"

        run_dir = Path(project.model_dir) / "run_yolo_resume"
        thread = TrainThread(project, run_dir)
        thread.run()  # 同步执行
        self.assertTrue((run_dir / "best.pt").is_file())
        m = load_model(run_dir / "best.pt")
        pred = m.predict(np.zeros((64, 64, 3), np.uint8), conf=0.1)
        self.assertIsInstance(pred.boxes, list)

    def test_train_from_trained_best(self):
        """从训练记录的 best.pt（run: 前缀）继续训练（权重直读）。"""
        from xccv.training.trainer_thread import TrainThread

        project_mod.PROJECTS_DIR = self.root / "projects4"
        project = create_project("pt", str(self.cls_dir),
                                 str(self.root / "models4"), "classification")
        project.model_name = "resnet18"  # 首轮用标准架构训练产出 best.pt
        project.epochs = 1
        project.batch_size = 2
        project.workers = 0
        project.augment_ops = {"resize": {"width": 32, "height": 32}}
        # 第一轮训练产出 best.pt
        first = TrainThread(project, Path(project.model_dir) / "20260906_100000_resnet18")
        first.run()
        best = Path(project.model_dir) / "20260906_100000_resnet18" / "best.pt"
        self.assertTrue(best.is_file())
        # 模型下拉引用该记录权重继续训练
        project.model_name = "run:20260906_100000_resnet18/best.pt"
        second = TrainThread(project, Path(project.model_dir) / "20260906_110000_resnet18")
        second.run()
        self.assertTrue((Path(project.model_dir) / "20260906_110000_resnet18" / "best.pt").is_file())

    def test_train_from_trained_best_class_mismatch(self):
        """训练记录权重与当前数据集类别不一致时给出明确指引。"""
        from xccv.engine.models.pruning import pretrain_prune
        from xccv.training.trainer_thread import TrainThread

        project_mod.PROJECTS_DIR = self.root / "projects2"
        project = create_project("pm", str(self.cls_dir),
                                 str(self.root / "models2"), "classification")
        project.epochs = 1
        project.batch_size = 2
        project.workers = 0
        result = pretrain_prune("resnet18", "classification", 3, 0.2,
                                classes=["a", "b", "c"])
        pruned_dir = Path(project.model_dir) / "pruned"
        pruned_dir.mkdir(parents=True)
        dst = pruned_dir / "resnet18_pruned20.pt"
        torch.save(result["ckpt"], dst)
        project.model_name = "pruned:resnet18_pruned20.pt"

        thread = TrainThread(project, Path(project.model_dir) / "run2")
        ok = {}

        def _on_done(ok_flag, msg, metrics):
            ok["ok"], ok["msg"] = ok_flag, msg

        thread.train_finished.connect(_on_done)
        thread.run()
        self.assertFalse(ok["ok"])
        self.assertIn("重新生成", ok["msg"])


class ModelToolsWindowTest(unittest.TestCase):
    """工具卡片墙：点击卡片打开独立工具窗口（与项目窗口同模式，可多开）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        det, _ = make_labelme_dataset(Path(self._tmp.name) / "det", n_images=4)
        p = create_project("mt", str(det), str(Path(self._tmp.name) / "models"), "detection")
        run = Path(p.model_dir) / "20260904_100000_yolo11n"
        run.mkdir(parents=True)
        # 占位权重（窗口逻辑只查存在性）
        (run / "best.pt").write_bytes(b"weights" * 1000)

        from xccv.modeltools.modeltools_window import ModelToolsWindow

        self.win = ModelToolsWindow()
        self.addCleanup(self._close_windows)

    def _close_windows(self):
        for w in list(self.win.open_tool_windows().values()):
            w.close()
        self.win.close()

    def test_cards_open_independent_windows(self):
        """每个工具一个独立窗口；重复打开同一工具只激活已有窗口。"""
        self.win._open_tool("prune")
        self.win._open_tool("accel")
        windows = self.win.open_tool_windows()
        self.assertEqual(set(windows), {"prune", "accel"})
        from xccv.modeltools.modeltools_window import (
            AccelerateToolWindow, PruneToolWindow,
        )
        self.assertIsInstance(windows["prune"], PruneToolWindow)
        self.assertIsInstance(windows["accel"], AccelerateToolWindow)
        prune_first = windows["prune"]
        self.win._open_tool("prune")  # 重复打开 → 激活已有窗口
        self.assertIs(self.win.open_tool_windows()["prune"], prune_first)
        # 工具窗口是独立顶层窗口（与训练平台项目窗口一致）
        self.assertTrue(prune_first.isWindow())

    def test_accel_window_opens(self):
        """模型加速窗口：文件模式——不依赖项目，通过文件选择器指定权重。"""
        self.win._open_tool("accel")
        windows = self.win.open_tool_windows()
        self.assertIn("accel", windows)
        w = windows["accel"]
        self.assertFalse(hasattr(w, "project_combo"))  # 不再依赖训练平台项目
        src = Path(self._tmp.name) / "accel_src.pt"
        src.write_bytes(bytes([161]))
        w.source_edit.setText(str(src))
        self.assertEqual(w.source_edit.text(), str(src))

    def test_prune_window_selection(self):
        """剪枝窗口：项目/记录选择与源权重识别（best.pt）。"""
        self.win._open_tool("prune")
        w = self.win.open_tool_windows()["prune"]
        names = [w.project_combo.itemText(i) for i in range(w.project_combo.count())]
        self.assertIn("mt", names)
        w.mode_combo.setCurrentIndex(1)  # 切到"训练后剪枝"模式
        w.project_combo.setCurrentIndex(0)  # 触发 _on_project_changed
        self.assertEqual(w.run_combo.count(), 1)
        self.assertIn("已中断", w.run_combo.itemText(0))  # 占位权重无 metrics.json

    def test_prune_window_pretrain_mode(self):
        """训练前剪枝模式：默认选中；按项目任务列出模型架构；记录选择被禁用。"""
        self.win._open_tool("prune")
        w = self.win.open_tool_windows()["prune"]
        self.assertEqual(w.mode_combo.currentData(), "pre")  # 默认训练前剪枝
        w.project_combo.setCurrentIndex(0)
        names = [w.pre_model_combo.itemText(i) for i in range(w.pre_model_combo.count())]
        self.assertTrue(any("yolo11n" in n for n in names))  # 检测项目 → YOLO 架构
        self.assertFalse(w.run_combo.isEnabled())  # 训练前剪枝不需要训练记录
        self.assertTrue(w.pre_group.isVisibleTo(w))
        w.mode_combo.setCurrentIndex(1)
        self.assertFalse(w.pre_group.isVisibleTo(w))
        self.assertTrue(w.run_combo.isEnabled())

# == 来源模块：test_acceleration ==
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


# == 百宝箱工作台 ==

class ToolboxRegistryTest(unittest.TestCase):
    """注册表：五工具登记、搜索（中英关键词）、重复注册幂等。"""

    def test_five_tools_registered(self):
        from xccv.modeltools.toolbox import all_tools
        keys = [s.key for s in all_tools()]
        self.assertEqual(keys, ["label_replace", "label_delete", "dataset_move",
                                "tiling_export", "video_frames"])

    def test_search_by_zh_and_en_keywords(self):
        from xccv.modeltools.toolbox import search
        self.assertEqual([s.key for s in search("裁图")], ["tiling_export"])
        self.assertEqual([s.key for s in search("tile")], ["tiling_export"])
        self.assertEqual(len(search("")), 5)
        self.assertEqual(len(search("不存在的关键词xyz")), 0)

    def test_register_is_idempotent(self):
        from xccv.modeltools.toolbox import all_tools, registry
        spec = all_tools()[0]
        registry.register(spec)
        self.assertEqual(len(all_tools()), 5)


class ToolboxWindowTest(unittest.TestCase):
    """百宝箱窗口：卡片墙由注册表驱动、搜索过滤、点卡开子窗口。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    def test_cards_follow_registry_and_search(self):
        from xccv.modeltools.toolbox import ToolboxWindow
        w = ToolboxWindow()
        try:
            self.assertEqual(w._cards_grid.count(), 5)
            w.search_edit.setText("裁图")
            w._rebuild_cards()
            self.assertEqual(w._cards_grid.count(), 1)
            w.search_edit.setText("存在不存在xyz")
            w._rebuild_cards()
            self.assertTrue(w.empty_hint.isVisibleTo(w))
            self.assertEqual(w._cards_grid.count(), 0)
            w.search_edit.setText("")
            w._rebuild_cards()
            self.assertEqual(w._cards_grid.count(), 5)
        finally:
            w.close()

    def test_open_tool_window(self):
        from xccv.modeltools.toolbox import ToolboxWindow
        w = ToolboxWindow()
        try:
            w._open_tool("label_replace")
            self.assertIn("label_replace", w.open_tool_windows())
            win = w.open_tool_windows()["label_replace"]
            self.assertTrue(win.isVisible())
            w._open_tool("label_replace")   # 重复打开只激活
            self.assertEqual(len(w.open_tool_windows()), 1)
            win.close()
        finally:
            w.close()


class ModelToolsToolboxCardTest(unittest.TestCase):
    """模型工具卡片墙新增【百宝箱】卡片；点卡打开百宝箱工作台（惰性导入）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_card_wall_contains_toolbox(self):
        from PyQt5.QtWidgets import QLabel
        from xccv.modeltools.modeltools_window import ModelToolsWindow
        w = ModelToolsWindow()
        try:
            texts = [l.text() for l in w.findChildren(QLabel)]
            self.assertTrue(any(t == "百宝箱" for t in texts))
        finally:
            w.close()

    def test_open_toolbox_from_card_wall(self):
        from xccv.modeltools.modeltools_window import ModelToolsWindow
        w = ModelToolsWindow()
        try:
            w._open_tool("toolbox")
            self.assertIn("toolbox", w.open_tool_windows())
            box = w.open_tool_windows()["toolbox"]
            self.assertTrue(box.isVisible())
            box.close()
        finally:
            w.close()

class ToolboxPresentationTest(unittest.TestCase):
    """百宝箱呈现：英文卡片、悬停说明、专用图标。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_every_tool_has_english_title_desc_tooltip(self):
        import re
        from xccv.i18n_en import TABLE
        from xccv.modeltools.toolbox import all_tools
        cjk = re.compile(r"[\u4e00-\u9fff]")
        specs = all_tools()
        self.assertEqual(len(specs), 5)
        for spec in specs:
            for text in (spec.title, spec.desc, spec.tooltip):
                self.assertIn(text, TABLE, spec.key)
                self.assertFalse(cjk.search(TABLE[text]), spec.key)
            self.assertEqual(spec.icon, "toolbox_tool.png", spec.key)

    def test_dedicated_icons_exist(self):
        from xccv.ui_style import icon_path
        for name in ("toolbox.png", "toolbox_tool.png"):
            self.assertTrue(icon_path(name).is_file(), name)

    def test_cards_carry_tooltips(self):
        from xccv.modeltools.toolbox import ToolboxWindow
        w = ToolboxWindow()
        try:
            self.assertEqual(w._cards_grid.count(), 5)
            for i in range(w._cards_grid.count()):
                card = w._cards_grid.itemAt(i).widget()
                self.assertTrue(card.toolTip())
        finally:
            w.close()

    def test_english_mode_renders_english_cards(self):
        import re
        import xccv.i18n as i18n
        from PyQt5.QtWidgets import QLabel
        from xccv.modeltools.toolbox import ToolboxWindow
        cjk = re.compile(r"[\u4e00-\u9fff]")
        old = i18n._lang_cache
        i18n._lang_cache = "en"
        try:
            w = ToolboxWindow()
            try:
                texts = [lbl.text() for lbl in w.findChildren(QLabel)]
                tips = [w._cards_grid.itemAt(i).widget().toolTip()
                        for i in range(w._cards_grid.count())]
                self.assertTrue(texts)
                self.assertFalse([t for t in texts if cjk.search(t)])
                self.assertFalse([t for t in tips if cjk.search(t)])
            finally:
                w.close()
        finally:
            i18n._lang_cache = old


if __name__ == '__main__':
    unittest.main()
