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


if __name__ == "__main__":
    unittest.main()
