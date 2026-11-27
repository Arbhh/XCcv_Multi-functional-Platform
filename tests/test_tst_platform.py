"""测试平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_testing_platform
- test_metrics
- test_result_json
"""


# == 来源模块：test_testing_platform ==
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

# == 来源模块：test_metrics ==
"""指标计算单元测试（人工核算过的样例）。"""

import unittest

import numpy as np

from xccv.engine.metrics.classification import confusion_matrix, topk_accuracy
from xccv.engine.metrics.detection import detection_confusion_matrix, iou_matrix


class ClassificationMetricsTest(unittest.TestCase):
    def test_topk(self):
        probs = np.array([[0.1, 0.8, 0.1], [0.5, 0.4, 0.1], [0.2, 0.2, 0.6]])
        targets = np.array([1, 0, 1])
        self.assertAlmostEqual(topk_accuracy(probs, targets, 1), 2 / 3)
        self.assertAlmostEqual(topk_accuracy(probs, targets, 2), 1.0)

    def test_confusion_matrix(self):
        cm = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], 3)
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[0, 1], 1)
        self.assertEqual(cm[1, 1], 1)
        self.assertEqual(cm[2, 2], 1)
        self.assertEqual(int(cm.sum()), 4)


class DetectionMetricsTest(unittest.TestCase):
    def test_iou_matrix(self):
        a = np.array([[0.0, 0.0, 2.0, 2.0]])
        b = np.array([[1.0, 0.0, 3.0, 2.0]])
        self.assertAlmostEqual(iou_matrix(a, b)[0, 0], 2.0 / 6.0)

    def test_iou_empty(self):
        self.assertEqual(iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape, (0, 3))

    def test_confusion_matching(self):
        gt = [{"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "labels": [0, 1]}]
        pred = [{"boxes": [[0, 0, 10, 10], [50, 50, 60, 60]],
                 "labels": [0, 0], "scores": [0.9, 0.8]}]
        m = detection_confusion_matrix(gt, pred, n_classes=2, iou_thr=0.5)
        self.assertEqual(m[0, 0], 1)   # 类0 命中
        self.assertEqual(m[1, 2], 1)   # 类1 漏检 -> 背景列
        self.assertEqual(m[2, 0], 1)   # 误检为类0 -> 背景行
        self.assertEqual(int(m.sum()), 3)
        self.assertEqual(m.shape, (3, 3))

# == 来源模块：test_result_json ==
"""统一检出结果 JSON 契约测试。"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xccv.engine.models.base import Prediction
from xccv.engine.result_json import (
    SCHEMA_VERSION, build_result_json, classification_topk, save_result_json,
)


def _det_pred():
    return Prediction(labels=[0, 1], scores=[0.91234, 0.5],
                      boxes=[[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 8.0, 8.0]])


def _source():
    return {"file": "img00.png", "path": "D:/ds/img00.png",
            "width": 640, "height": 480}


class DetectionSchemaTest(unittest.TestCase):
    def test_detection_payload(self):
        payload = build_result_json(
            task="detection", source=_source(),
            model={"file": "best.pt", "name": "yolo11n"},
            inference={"conf": 0.25, "iou": 0.45},
            prediction=_det_pred(), classes=["cat", "dog"])
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["task"], "detection")
        self.assertEqual(payload["source"]["file"], "img00.png")
        self.assertEqual(payload["model"]["name"], "yolo11n")
        d0 = payload["detections"][0]
        self.assertEqual(d0["label"], "cat")
        self.assertEqual(d0["label_id"], 0)
        self.assertEqual(d0["score"], 0.9123)
        self.assertEqual(d0["bbox"], [10.0, 20.0, 50.0, 60.0])
        self.assertEqual(d0["bbox_area"], 1600)
        self.assertNotIn("mask_area", d0)  # 检测无掩码字段
        self.assertEqual(payload["classifications"], [])  # 数组恒在

    def test_segmentation_mask_area(self):
        pred = Prediction(labels=[0], scores=[0.8],
                          boxes=[[0.0, 0.0, 10.0, 10.0]],
                          masks=[np.ones((10, 10), bool)])
        payload = build_result_json(task="segmentation", source=_source(),
                                    prediction=pred, classes=["scrach"])
        d0 = payload["detections"][0]
        self.assertEqual(d0["mask_area"], 100)
        self.assertEqual(payload["task"], "segmentation")

    def test_classification_topk_and_empty_detections(self):
        """Q2：分类任务走 classifications（Top-K），detections 恒空数组。"""
        pred = Prediction(labels=[1], scores=[0.97],
                          probs=[0.02, 0.97, 0.01])
        payload = build_result_json(task="classification", source=_source(),
                                    prediction=pred, classes=["OK", "NG", "?"])
        self.assertEqual(payload["detections"], [])
        top = payload["classifications"]
        self.assertEqual([c["label"] for c in top], ["NG", "OK", "?"])
        self.assertEqual(top[0]["score"], 0.97)
        # Top-K 截断
        pred5 = Prediction(labels=[0], scores=[0.5],
                           probs=[0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
        top5 = classification_topk(pred5, [f"c{i}" for i in range(7)], k=5)
        self.assertEqual(len(top5), 5)


class ContractTest(unittest.TestCase):
    def test_ascii_keys_and_json_roundtrip(self):
        """C# 友好：键全 ASCII、纯 JSON 基元、可无损 roundtrip。"""
        payload = build_result_json(
            task="segmentation", source=_source(),
            model={"file": "best.pt"}, inference={"conf": 0.3},
            prediction=Prediction(labels=[0], scores=[0.7],
                                  boxes=[[1, 1, 2, 2]],
                                  masks=[np.zeros((4, 4), bool)]),
            classes=["缺陷"], flow_info={"removed": 2, "count_check": "NG"})
        text = json.dumps(payload, ensure_ascii=False)
        back = json.loads(text)
        self.assertEqual(back["inference"]["flow_removed"], 2)
        self.assertEqual(back["inference"]["count_check"], "NG")
        self.assertTrue(back["inference"]["flow_applied"])
        for key in payload:
            self.assertTrue(key.isascii(), key)
        for det in payload["detections"]:
            for key in det:
                self.assertTrue(key.isascii(), key)

    def test_error_entry_and_null_prediction(self):
        payload = build_result_json(task="detection", source=_source(),
                                    prediction=None,
                                    error="RuntimeError: 无法读取")
        self.assertEqual(payload["detections"], [])
        self.assertIn("无法读取", payload["error"])

    def test_new_task_field_extensible(self):
        """OCR 等新任务：task 自由扩展且不破坏既有字段（向后兼容口径）。"""
        payload = build_result_json(task="ocr", source=_source(), prediction=None)
        self.assertEqual(payload["task"], "ocr")
        self.assertEqual(payload["detections"], [])

    def test_save_utf8(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "img00.json"
        payload = build_result_json(task="detection", source=_source(),
                                    prediction=_det_pred(), classes=["夹渣"])
        save_result_json(out, payload)
        text = out.read_text(encoding="utf-8")
        self.assertIn("夹渣", text)  # ensure_ascii=False：中文直接可读
        back = json.loads(text)
        self.assertEqual(back["detections"][0]["label"], "夹渣")

# == 分数语义适配：标准推理/阈值部分复用复盘口径 ==
"""异常检测（score_semantics ≠ 置信度）在标准页的显示/文案/JSON 适配。"""

import unittest
from types import SimpleNamespace

from xccv.testing.review_logic import (conf_to_score, image_score_of,
                                       is_conf_like, score_semantics_of)


class ScoreSemanticsHelpersTest(unittest.TestCase):
    """共享判定/换算逻辑（review_logic，标准页与复盘页同一实现）。"""

    def test_is_conf_like(self):
        self.assertTrue(is_conf_like({}))
        self.assertTrue(is_conf_like({"key": "conf"}))
        self.assertFalse(is_conf_like({"key": "z"}))

    def test_conf_to_score(self):
        sem_z = {"key": "z"}
        self.assertEqual(conf_to_score({}, 0.25), 0.25)
        self.assertAlmostEqual(conf_to_score(sem_z, 0.25), 3.0)
        self.assertAlmostEqual(conf_to_score(sem_z, 0.0), 999.0)   # 钳 1e-3
        self.assertAlmostEqual(conf_to_score(sem_z, 1.0), 0.0)

    def test_score_semantics_of_sources(self):
        self.assertEqual(score_semantics_of(None), {})
        self.assertEqual(score_semantics_of({"score_semantics": {"key": "z"}}),
                         {"key": "z"})
        m = SimpleNamespace(score_semantics={"key": "z", "name": "图像分数"})
        self.assertEqual(score_semantics_of(m)["name"], "图像分数")

    def test_image_score_of_prediction_and_dict(self):
        pred = Prediction(labels=[0], scores=[2.0, 5.5],
                          boxes=[[0, 0, 1, 1]], probs=[4.5])
        self.assertEqual(image_score_of(pred), 4.5)
        self.assertEqual(image_score_of(Prediction(scores=[2.0, 5.5])), 5.5)
        self.assertEqual(image_score_of({"image_score": 3.2}), 3.2)
        self.assertEqual(image_score_of({"detections": [{"score": 1.5}]}), 1.5)
        self.assertIsNone(image_score_of(Prediction()))
        self.assertIsNone(image_score_of(None))


class AnomalyStandardDisplayTest(unittest.TestCase):
    """标准页异常检测显示：图像分数行 + 语义标注（与复盘横幅同口径）。"""

    def _make_win(self):
        win = TestingWindow()
        win.task = "anomaly"
        win.model = SimpleNamespace(
            score_semantics={"key": "z", "name": "图像分数",
                             "range": [0.0, None],
                             "higher": "分数越高越异常（OK 分布 z 标定）"})
        win.classes = ["异常"]
        return win

    def test_show_result_shows_image_score(self):
        win = self._make_win()
        img = np.full((64, 64, 3), 200, np.uint8)
        pred = Prediction(labels=[0], scores=[3.2, 5.1],
                          boxes=[[10, 10, 30, 30], [40, 40, 60, 60]],
                          probs=[4.5])
        win._show_result("a.png", img, pred)
        text = win.result_stats.text()
        self.assertIn("图像分数", text)
        self.assertIn("4.50", text)
        self.assertIn("检出目标：2 个", text)
        win._fill_result_list(pred)
        self.assertIn("图像分数", win.result_list.item(0).text())
        self.assertIn("4.50", win.result_list.item(0).text())
        self.assertIn("3.20", win.result_list.item(1).text())   # 框级异常分数

    def test_show_result_without_image_score(self):
        win = self._make_win()
        img = np.full((64, 64, 3), 200, np.uint8)
        # 全空预测（无 probs 也无检出）才走「无图像分数」；
        # 有检出无 probs 时退检出最大分（与复盘口径一致）
        win._show_result("a.png", img, Prediction())
        self.assertIn("无图像分数", win.result_stats.text())


class AppliedThresholdLabelTest(unittest.TestCase):
    """应用阈值文案随分数语义切换（z 语义写入的是分数阈值而非 conf）。"""

    def test_label_from_stats_semantics(self):
        win = TestingWindow()
        win._last_match_stats = {"score_semantics":
                                 {"key": "z", "name": "图像分数"}}
        self.assertIn("图像分数", win._applied_threshold_label())
        win._last_match_stats = {}
        win.model = None
        self.assertEqual(win._applied_threshold_label(), "conf")

    def test_threshold_flow_task_source(self):
        """流程图模板 task 取 self.task（回归锁：曾误引不存在的 _task）。"""
        src = Path("xccv/testing/testing_window.py").read_text(
            encoding="utf-8")
        self.assertIn('getattr(self, "task", "")', src)
        self.assertNotIn('getattr(self, "_task", "")', src)


class AnomalyImageScoreSchemaTest(unittest.TestCase):
    """异常检测 JSON 留痕：image_score additive 键（图像级分数）。"""

    def test_image_score_from_probs(self):
        payload = build_result_json(
            task="anomaly", source=_source(),
            prediction=Prediction(labels=[0], scores=[3.2],
                                  boxes=[[0, 0, 4, 4]], probs=[4.5]),
            classes=["异常"],
            score_semantics={"key": "z", "name": "图像分数"})
        self.assertEqual(payload["image_score"], 4.5)
        self.assertEqual(payload["detections"][0]["score"], 3.2)

    def test_image_score_fallback_max_det(self):
        payload = build_result_json(
            task="anomaly", source=_source(),
            prediction=Prediction(labels=[0], scores=[1.5, 2.5],
                                  boxes=[[0, 0, 4, 4], [5, 5, 9, 9]]))
        self.assertEqual(payload["image_score"], 2.5)

    def test_no_image_score_for_detection(self):
        payload = build_result_json(task="detection", source=_source(),
                                    prediction=_det_pred())
        self.assertNotIn("image_score", payload)

if __name__ == '__main__':
    unittest.main()
