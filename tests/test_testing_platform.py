"""测试平台测试：渲染、权重定位、模型自动识别、端到端推理、窗口冒烟。"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_classification_dataset, make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.engine.io_utils import imread_unicode
from xccv.engine.models import TrainConfig, create_model, load_model
from xccv.engine.models.base import Prediction
from xccv.testing.render import draw_predictions
from xccv.testing.testing_window import TestingWindow, latest_weights

_APP = QApplication.instance() or QApplication(sys.argv)


def _load_weights_sync(win, path):
    """加载模型（平台内为后台线程）并等待完成。

    等待条件是完成回调把 _load_thread 置空，而不是 isRunning()：
    QThread 刚 start() 后、OS 线程被调度前存在 isRunning()==False 的窗口
    （低核数 CI runner 上可复现），以它作循环入口会整体跳过等待，
    跨线程的 done/finished 队列信号将不被投递。
    """
    win._load_weights(path)
    deadline = time.monotonic() + 600
    while win._load_thread is not None and time.monotonic() < deadline:
        QApplication.processEvents()          # 投递 done/failed/finished 队列信号
        if win._load_thread is not None:
            win._load_thread.wait(50)
    QApplication.processEvents()



def _quick_cls_checkpoint(out_dir: Path) -> Path:
    """1 轮小分类模型，产出明文权重 best.pt。"""
    cls_dir, _ = make_classification_dataset(out_dir / "cls", per_class=4)
    model = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
    cfg = TrainConfig(task=TaskType.CLASSIFICATION, dataset_dir=cls_dir, output_dir=out_dir,
                      model_name="resnet18", epochs=1, batch_size=2, val_ratio=0.25,
                      image_size=32, workers=0, use_pretrained=False)
    model.train(cfg)
    return out_dir / "best.pt"


def _quick_yolo_checkpoint(dataset_dir: Path, out_dir: Path) -> Path:
    model = create_model("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
    cfg = TrainConfig(task=TaskType.DETECTION, dataset_dir=dataset_dir, output_dir=out_dir,
                      model_name="yolo11n", epochs=1, batch_size=2, val_ratio=0.25,
                      image_size=64, workers=0, use_pretrained=False)
    model.train(cfg)
    return out_dir / "best.pt"


def _quick_rcnn_checkpoint(dataset_dir: Path, out_dir: Path) -> Path:
    model = create_model("fasterrcnn", TaskType.DETECTION, 2, use_pretrained=False)
    cfg = TrainConfig(task=TaskType.DETECTION, dataset_dir=dataset_dir, output_dir=out_dir,
                      model_name="fasterrcnn", epochs=1, batch_size=2, val_ratio=0.25,
                      image_size=96, workers=0, use_pretrained=False)
    model.train(cfg)
    return out_dir / "best.pt"


class RenderTest(unittest.TestCase):
    def test_detection_render_changes_pixels(self):
        img = np.full((100, 100, 3), 200, np.uint8)
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[10, 10, 50, 50]])
        out = draw_predictions(img, pred, ["猫"], "detection")
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))

    def test_mask_render(self):
        img = np.zeros((64, 64, 3), np.uint8)
        mask = np.zeros((64, 64), bool)
        mask[10:30, 10:30] = True
        pred = Prediction(labels=[1], scores=[0.8], boxes=[[10, 10, 30, 30]], masks=[mask])
        out = draw_predictions(img, pred, ["a", "b"], "segmentation")
        self.assertGreater(float(out[15:25, 15:25].mean()), 0)  # 掩码区域有颜色叠加

    def test_classification_render(self):
        img = np.zeros((64, 64, 3), np.uint8)
        pred = Prediction(labels=[0], scores=[0.77], probs=[0.77, 0.23])
        out = draw_predictions(img, pred, ["橘色"], "classification")
        self.assertEqual(out.shape, img.shape)


class LatestWeightsTest(unittest.TestCase):
    def test_picks_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run1").mkdir()
            (root / "run2").mkdir()
            (root / "run1" / "best.pt").write_bytes(b"x")
            time.sleep(0.05)
            (root / "run2" / "best.pt").write_bytes(b"yy")
            self.assertEqual(latest_weights(root), root / "run2" / "best.pt")
            empty = root / "empty"
            empty.mkdir()
            self.assertIsNone(latest_weights(empty))

    def test_finds_only_run(self):
        """只有一个训练记录时直接定位其 best.pt。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run1").mkdir()
            (root / "run1" / "best.pt").write_bytes(b"x")
            self.assertEqual(latest_weights(root), root / "run1" / "best.pt")


class ModelLoadInferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        out = Path(cls._tmp.name)
        cls.cls_ckpt = _quick_cls_checkpoint(out / "cls_run")
        det_dir, _ = make_labelme_dataset(out / "det", n_images=6)
        cls.det_dir = det_dir
        cls.yolo_ckpt = _quick_yolo_checkpoint(det_dir, out / "yolo_run")
        cls.rcnn_ckpt = _quick_rcnn_checkpoint(det_dir, out / "rcnn_run")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_load_model_auto_detect(self):
        m1 = load_model(self.cls_ckpt)
        self.assertEqual(m1.task.value, "classification")
        self.assertEqual(m1.classes, ["cat", "dog"])
        m2 = load_model(self.yolo_ckpt)
        self.assertEqual(m2.task.value, "detection")
        self.assertEqual(m2.num_classes, 2)

    def test_predict_probs_and_boxes(self):
        img = imread_unicode(next(self.det_dir.glob("*.png")))
        self.assertIsNotNone(img)
        m1 = load_model(self.cls_ckpt)
        pred = m1.predict(img)
        self.assertEqual(len(pred.probs), 2)
        self.assertAlmostEqual(sum(pred.probs), 1.0, places=5)

        m2 = load_model(self.yolo_ckpt)
        pred2 = m2.predict(img, conf=0.1, iou=0.45)
        self.assertIsInstance(pred2.boxes, list)
        self.assertEqual(len(pred2.boxes), len(pred2.scores))

    def test_rcnn_confidence_filtering(self):
        """置信度阈值必须真正过滤（回归：roi_heads 属性名曾写错导致失效）。"""
        m = load_model(self.rcnn_ckpt)
        img = imread_unicode(next(self.det_dir.glob("*.png")))
        pred_low = m.predict(img, conf=0.01, iou=0.5)
        scores_low = list(pred_low.scores)
        self.assertTrue(all(s >= 0.01 for s in scores_low))
        if scores_low:
            high = min(max(scores_low) + 0.05, 0.99)
            pred_high = m.predict(img, conf=high, iou=0.5)
            self.assertTrue(all(s >= high for s in pred_high.scores),
                            f"存在低于阈值 {high} 的分数：{pred_high.scores}")
            self.assertLessEqual(len(pred_high.boxes), len(scores_low))

    def test_file_tree_groups_subfolders(self):
        """文件列表树形化：子文件夹为可展开节点，图片显示原名。"""
        make_labelme_dataset(self.det_dir / "tree_sub", n_images=2)
        try:
            win = TestingWindow()
            _load_weights_sync(win, self.yolo_ckpt)
            win._set_image_list(sorted(self.det_dir.rglob("*.png")), root=self.det_dir)
            tree = win.image_tree
            self.assertEqual(tree.topLevelItemCount(), 6 + 1)  # 6 张根图片 + 1 个文件夹节点
            folder = tree.topLevelItem(6)
            self.assertEqual(folder.text(0), "tree_sub")
            self.assertEqual(folder.childCount(), 2)
            self.assertIsNone(folder.data(0, 0x0100))  # 文件夹节点无路径
            self.assertEqual(folder.child(0).text(0), "img00.png")  # 原名
        finally:
            import shutil
            shutil.rmtree(self.det_dir / "tree_sub")

    def test_class_filter(self):
        """类别筛选：只保留指定类别的预测结果。"""
        win = TestingWindow()
        _load_weights_sync(win, self.yolo_ckpt)
        fake = Prediction(labels=[0, 1], scores=[0.9, 0.8],
                          boxes=[[0, 0, 10, 10], [20, 20, 30, 30]])
        self.assertEqual(len(win._filter_prediction(fake, None).labels), 2)  # 全部
        filtered = win._filter_prediction(fake, 1)
        self.assertEqual(filtered.labels, [1])
        self.assertEqual(filtered.boxes, [[20, 20, 30, 30]])

    def test_area_display(self):
        """结果显示面积：有掩码显示掩码像素面积，无掩码显示外接框面积。"""
        win = TestingWindow()
        _load_weights_sync(win, self.yolo_ckpt)
        mask = np.zeros((40, 40), bool)
        mask[0:10, 0:10] = True  # 100 像素
        fake = Prediction(labels=[0], scores=[0.9], boxes=[[0, 0, 10, 10]], masks=[mask])
        win.class_filter.setCurrentIndex(0)  # 全部
        win.show_area.setChecked(True)
        win._fill_result_list(fake)
        text = win.result_list.item(0).text()
        self.assertIn("面积 100 px", text)  # 掩码像素面积
        fake_nomask = Prediction(labels=[0], scores=[0.9], boxes=[[0, 0, 10, 10]])
        win._fill_result_list(fake_nomask)
        self.assertIn("面积 100 px²", win.result_list.item(0).text())  # 外接框面积
        win.show_area.setChecked(False)
        win._fill_result_list(fake_nomask)
        self.assertNotIn("面积", win.result_list.item(0).text())

    def test_ad_navigation(self):
        """A/D：在展平图片顺序中切换，边界停止，文件夹节点跳过。"""
        make_labelme_dataset(self.det_dir / "nav_sub", n_images=1)
        try:
            win = TestingWindow()
            _load_weights_sync(win, self.yolo_ckpt)
            win._set_image_list(sorted(self.det_dir.rglob("*.png")), root=self.det_dir)
            self.assertEqual(win._current_image_path().name, "img00.png")
            win._navigate(1)  # D
            self.assertEqual(win._current_image_path().name, "img01.png")
            win._navigate(-1)  # A
            self.assertEqual(win._current_image_path().name, "img00.png")
            win._navigate(-1)  # 边界 → 停留
            self.assertEqual(win._current_image_path().name, "img00.png")
        finally:
            import shutil
            shutil.rmtree(self.det_dir / "nav_sub")

    def test_window_smoke(self):
        win = TestingWindow()
        _load_weights_sync(win, self.yolo_ckpt)
        self.assertEqual(win.task, "detection")
        self.assertTrue(win.iou_slider.isEnabled())  # 检测任务 IOU 可调
        win._set_image_list(sorted(self.det_dir.glob("*.png")))
        self.assertIn("检出目标", win.result_stats.text())
        # 分类模型加载后：滑条应禁用
        win2 = TestingWindow()
        _load_weights_sync(win2, self.cls_ckpt)
        self.assertEqual(win2.task, "classification")
        self.assertFalse(win2.conf_slider.isEnabled())

    def test_fmt_cost(self):
        """耗时文案：<1s 毫秒，≥1s 两位小数秒。"""
        self.assertEqual(TestingWindow._fmt_cost(0.0342), "34 ms")
        self.assertEqual(TestingWindow._fmt_cost(0.5), "500 ms")
        self.assertEqual(TestingWindow._fmt_cost(1.2561), "1.26 s")

    def test_single_image_cost_display(self):
        """单图推理完成后状态摘要须包含「推理耗时」。"""
        win = TestingWindow()
        _load_weights_sync(win, self.yolo_ckpt)
        win._set_image_list(sorted(self.det_dir.glob("*.png")))  # 选中即自动推理
        text = win.result_stats.text()
        self.assertIn("推理耗时", text)
        self.assertIsNotNone(win._last_infer_cost)
        self.assertGreaterEqual(win._last_infer_cost, 0.0)
        # 重绘路径（显示开关/筛选切换）沿用最近一次耗时，不重新推理
        prev = win._last_infer_cost
        win._redraw_last()
        self.assertIn("推理耗时", win.result_stats.text())
        self.assertEqual(win._last_infer_cost, prev)
        # 分类模型同样显示耗时
        win2 = TestingWindow()
        _load_weights_sync(win2, self.cls_ckpt)
        win2._set_image_list(sorted(self.det_dir.glob("*.png")))
        self.assertIn("推理耗时", win2.result_stats.text())
        self.assertIn("识别为", win2.result_stats.text())


if __name__ == "__main__":
    unittest.main()

