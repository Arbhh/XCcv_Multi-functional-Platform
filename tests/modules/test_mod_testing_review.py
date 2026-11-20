"""模块行为锁：测试平台复盘（分组口径 / 阈值面板 / 结果渲染窗）。

行为级口径：复盘分组纯逻辑（阈值复用零重推理）、阈值面板钳制与
防抖广播、双击渲染窗绿/红框（像素断言）。
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication(sys.argv)

_DET_OK = [0.9, 1, 100.0, 10.0, 10.0, 0.5, 0.1]     # 命中 GT 的高分检出
_DET_FP = [0.6, 0, 50.0, 5.0, 5.0, 0.3, 0.2]        # 未命中的过阈检出
_DET_LOW = [0.10, 0, 10.0, 3.0, 3.0, 0.2, 0.0]      # 低分（score 阈下）


def _stats():
    """最小 match_stats：过检/漏检/正确/未评估 四类各一图。"""
    return {
        "task": "detection",
        "per_image": [
            {"file": "a.png", "path": "a.png", "gt": 2, "gt_available": True,
             "dets": [list(_DET_OK), list(_DET_FP)]},          # → 过检
            {"file": "b.png", "path": "b.png", "gt": 1, "gt_available": True,
             "dets": [list(_DET_LOW)]},                        # → 漏检
            {"file": "c.png", "path": "c.png", "gt": 0, "gt_available": True,
             "dets": []},                                      # OK 图零检出 → 正确
            {"file": "d.png", "path": "d.png", "gt": None,
             "gt_available": False, "dets": [list(_DET_FP)]},  # 无 GT → 未评估
        ],
    }


class ReviewGroupingContractTest(unittest.TestCase):
    """复盘分组纯逻辑（阈值复用现算，零重推理）。"""

    def test_grouping_four_ways(self):
        from xccv.testing.review_logic import classify_images

        g = classify_images(_stats(), {"score": 0.25})
        self.assertEqual([e["file"] for e in g["over"]], ["a.png"])
        self.assertEqual([e["file"] for e in g["missed"]], ["b.png"])
        self.assertEqual([e["file"] for e in g["correct"]], ["c.png"])
        self.assertEqual(g["unjudged"], 1)
        self.assertEqual(g["unjudged_ng"], 1)   # 未评估但有过阈检出
        self.assertEqual(g["total"], 4)
        self.assertIn("过1", g["over"][0]["badges"])

    def test_score_threshold_filters(self):
        from xccv.testing.review_logic import classify_images

        # score 阈抬到 0.65：a 图 0.6 检出被滤（gt=2 只命中 1 → 落漏检组），
        # b 图不变（本就阈下）→ 过检清零
        g = classify_images(_stats(), {"score": 0.65})
        self.assertEqual([e["file"] for e in g["correct"]], ["c.png"])
        self.assertEqual(len(g["over"]), 0)
        self.assertEqual([e["file"] for e in g["missed"]], ["a.png", "b.png"])

    def test_sweep_matches_grouping(self):
        from xccv.testing.review_logic import classify_images, sweep_group_counts

        xs = [i / 20 for i in range(21)]   # 0.00~1.00 网格 21 点
        sweep = sweep_group_counts(_stats(), "score", xs)
        self.assertEqual(len(sweep["correct"]), 21)
        k = xs.index(0.25)
        g = classify_images(_stats(), {"score": 0.25})
        self.assertEqual(
            (sweep["correct"][k], sweep["missed"][k], sweep["over"][k]),
            (len(g["correct"]), len(g["missed"]), len(g["over"])),
            "曲线读数与分组口径不一致（图口径行为锁）")


class ThresholdPanelModuleTest(unittest.TestCase):
    """阈值面板：原生钳制 + 拖线防抖 + 程序化立即广播。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _panel(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        stats = {"gt_available": True, "gt_total": 1, "n_gt_images": 1,
                 "n_ok_images": 0, "n_unmatched": 0, "green": [0.9],
                 "red_defect": [0.4], "red_ok": [],
                 "dims": {"area": {"green": [10000.0],
                                   "red_defect": [400.0], "red_ok": []}},
                 "per_image": []}
        p.set_stats(stats, 0.25)
        self.addCleanup(p.deleteLater)
        return p

    def test_bounds_clamped(self):
        p = self._panel()
        idx = p.cmb_dim.findData("area")
        p.cmb_dim.setCurrentIndex(idx)
        self.assertEqual(tuple(p._line.maxRange), (0.0, p._x_max))
        p._line.setValue(p._x_max + 500.0)
        self.assertLessEqual(float(p._line.value()), p._x_max,
                             "线可滑出轴（钳制失效）")

    def test_drag_debounce_then_immediate(self):
        from PyQt5.QtTest import QTest

        p = self._panel()
        got = []
        p.thresholdsChanged.connect(lambda d: got.append(dict(d)))
        p._line.moving = True                    # 模拟真实拖线中
        p._line.setValue(0.4)
        p._on_line_moved()
        QTest.qWait(30)
        self.assertEqual(got, [], "拖线中不应逐像素广播")
        QTest.qWait(200)                         # 150ms 防抖定时器到点
        self.assertEqual(len(got), 1, "拖线停止后防抖应落一次广播")
        p._line.moving = False
        p._line.setValue(0.7)                    # 程序化设置 → 立即广播
        self.assertEqual(got[-1], {"score": 0.7})


class ResultViewModuleTest(unittest.TestCase):
    """双击渲染窗：阈值绿/红框 + anomaly 横幅（像素级行为）。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        import numpy as np

        cls.np = np
        import tempfile

        cls._dir = tempfile.mkdtemp()
        import cv2

        cv2.imwrite(str(Path(cls._dir) / "a.png"),
                    np.full((120, 160, 3), 235, np.uint8))

    def _entry(self, task="detection"):
        e = {"file": "a.png", "path": str(Path(self._dir) / "a.png"),
             "width": 160, "height": 120,
             "detections": [
                 {"label": "黑点", "score": 0.9, "bbox": [10, 10, 60, 60],
                  "_dims": {"area": 2500, "short": 50, "long": 50}},
                 {"label": "异色", "score": 0.1, "bbox": [80, 40, 130, 90],
                  "_dims": {"area": 2500, "short": 50, "long": 50}}],
             "image_score": 3.2, "classifications": []}
        if task == "anomaly":
            e["classifications"] = [{"label": "ng", "score": 0.77}]
        return e

    def _edge_pixels(self, img):
        out = {"green": 0, "red": 0}
        for x0, y0, x1, y1 in ((10, 10, 60, 60), (80, 40, 130, 90)):
            for x in range(x0, x1 + 1):
                for y in (y0, y1):
                    c = img.pixelColor(min(x, img.width() - 1),
                                       min(y, img.height() - 1))
                    if c.green() > 130 and c.red() < 110:
                        out["green"] += 1
                    if c.red() > 140 and c.green() < 120:
                        out["red"] += 1
            for y in range(y0, y1 + 1):
                for x in (x0, x1):
                    c = img.pixelColor(min(x, img.width() - 1),
                                       min(y, img.height() - 1))
                    if c.green() > 130 and c.red() < 110:
                        out["green"] += 1
                    if c.red() > 140 and c.green() < 120:
                        out["red"] += 1
        return out

    def test_boxes_and_banner(self):
        from xccv.testing.result_view import render_result_pixmap

        pm = render_result_pixmap(self._entry(), {"score": 0.25}, "detection")
        pix = self._edge_pixels(pm.toImage())
        self.assertGreater(pix["green"], 0, "过阈检出未渲染绿框")
        self.assertGreater(pix["red"], 0, "被过滤检出未渲染红框")

        pm2 = render_result_pixmap(self._entry("anomaly"), {"score": 1.0},
                                   "anomaly")
        img = pm2.toImage()
        dark = sum(1 for x in range(0, img.width(), 4)
                   for y in range(0, min(40, img.height()))
                   if img.pixelColor(x, y).lightness() < 90)
        self.assertGreater(dark, 10, "anomaly 结果无横幅显示")


if __name__ == "__main__":
    unittest.main()
