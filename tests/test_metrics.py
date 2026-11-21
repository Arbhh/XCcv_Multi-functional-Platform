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


if __name__ == "__main__":
    unittest.main()

