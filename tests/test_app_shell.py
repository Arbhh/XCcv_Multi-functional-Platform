"""集成测试：主界面整合行为 + 标注产物→训练→测试 跨模块全链路。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_labelme_dataset
from xccv.app import _PLATFORMS, HomeWindow
from xccv.engine.dataset.base import TaskType
from xccv.engine.io_utils import imread_unicode
from xccv.engine.models import TrainConfig, create_model, load_model
from xccv.testing.render import draw_predictions

_APP = QApplication.instance() or QApplication(sys.argv)


class HomeWindowTest(unittest.TestCase):
    """主界面改为子进程拉起平台：验证启动调用与失败提示，不真正弹窗。"""

    def setUp(self):
        self.win = HomeWindow()
        self.addCleanup(self.win.close)

    def test_has_four_platform_cards(self):
        self.assertEqual(len(_PLATFORMS), 4)
        self.assertEqual([p[0] for p in _PLATFORMS],
                         ["annotation", "training", "testing", "tools"])

    def test_open_platform_launches_independent_process(self):
        with patch("xccv.launcher.launch_platform") as mock_launch:
            self.win._open_platform("annotation")
            self.win._open_platform("training")
            self.win._open_platform("testing")
            self.win._open_platform("tools")
            self.assertEqual(
                [c.args[0] for c in mock_launch.call_args_list],
                ["annotation", "training", "testing", "tools"])

    def test_launch_failure_shows_message(self):
        """启动失败要弹窗提示，而不是点击无反应。"""
        with patch("xccv.launcher.launch_platform", side_effect=OSError("boom")), \
             patch("xccv.app.QMessageBox.critical") as mock_box:
            self.win._open_platform("training")
            self.assertTrue(mock_box.called)


class PipelineTest(unittest.TestCase):
    """跨模块全链路：标注产物（labelme json 数据集）→ 训练 → 测试推理可视化。"""

    def test_annotate_train_infer(self):
        with tempfile.TemporaryDirectory() as tmp:
            det, _ = make_labelme_dataset(Path(tmp) / "det", n_images=6)
            model = create_model("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
            metrics = model.train(TrainConfig(
                task=TaskType.DETECTION, dataset_dir=det,
                output_dir=Path(tmp) / "run", model_name="yolo11n",
                epochs=1, batch_size=2, val_ratio=0.25,
                image_size=64, workers=0, use_pretrained=False))
            self.assertIn("map50", metrics["history"][0])  # 键名标准化后可进入白名单绘制

            from xccv.testing.testing_window import latest_weights
            ckpt = latest_weights(Path(tmp))
            self.assertIsNotNone(ckpt)

            infer_model = load_model(ckpt)
            self.assertEqual(infer_model.task.value, "detection")
            img = imread_unicode(next(det.glob("*.png")))
            self.assertIsNotNone(img)
            pred = infer_model.predict(img, conf=0.1, iou=0.45)
            rendered = draw_predictions(img, pred, infer_model.classes, "detection")
            self.assertEqual(rendered.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
