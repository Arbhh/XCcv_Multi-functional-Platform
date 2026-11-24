"""测试平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v071_fixes
- test_v072_fixes
"""


# == 来源模块：test_v071_fixes ==
"""V0.7.1 修复回归锁。

测试平台：
- 阈值面板：固定坐标轴（score 0~1、几何维 0~max 外扩）、阈值数字输入框、
  前景占比维度移除、平滑散点线（正常/过检/漏检三线）、漏检 GT 几何维度；
- 全平台 IOU 默认 0.2（UI 控件/服务配置/生成模板/流程节点默认）；
训练平台：
- 项目卡片等待横幅（repaint 保底 + 最小可见时长，源码锁）；
- 划分环状图贴合 train_ratio/val_ratio 并实时重绘（split_counts 同口径）；
- 数据网格筛选后缓存图标回贴（空白渲染修复）；
- 逐类指标表取消末列拉伸（AP@50 全屏还原修复）；
- 增强预览全算子生效（preview_apply_ops）。
"""

import json
import random
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.dataset.converters import split_counts
from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.metrics.batch_match import collect_match_stats


def _write_png(path: Path, w: int = 40, h: int = 30):
    rng = np.random.default_rng(w * h + 7)
    imwrite_unicode(str(path),
                    rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _write_json(path: Path, shapes: list):
    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {}, "shapes": shapes,
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": 30, "imageWidth": 40,
    }), encoding="utf-8")


def _rect(label: str, box) -> dict:
    return {"label": label, "points": [[box[0], box[1]], [box[2], box[3]]],
            "shape_type": "rectangle"}


# ================= 测试平台：阈值面板（任务 1/2/3） =================

def _panel_stats():
    """V0.7.2 扫描曲线口径：per_image 带逐检出记录（与 dims 汇总一致）。

    语义：img1 两命中（0.9/0.8）；img2 一未命中检出（0.4，GT 1 未命中）；
    img3 OK 图一误检（0.6）。gt_total=3、n_unmatched=1。"""
    return {"gt_available": True, "gt_total": 3, "n_gt_images": 2,
            "n_ok_images": 1, "n_unmatched": 1,
            "green": [0.9, 0.8], "red_defect": [0.4], "red_ok": [0.6],
            "dims": {"area": {"green": [9000.0, 400.0], "red_defect": [900.0],
                              "red_ok": [2500.0]},
                     "short": {"green": [95.0, 20.0], "red_defect": [30.0],
                               "red_ok": [50.0]},
                     "gray_diff": {"green": [88.0], "red_defect": [3.0],
                                   "red_ok": []}},
            "dims_missed": {"area": [1200.0], "short": [34.0], "long": [35.0]},
            "per_image": [
                {"path": "img1.png", "gt": 2, "gt_available": True,
                 "missed": 0, "fp": 0, "fp_ok": 0, "matched_all": True,
                 "dets": [[0.9, True, 9000.0, 95.0, 130.0, None, 88.0],
                          [0.8, True, 400.0, 20.0, 20.0, None, None]]},
                {"path": "img2.png", "gt": 1, "gt_available": True,
                 "missed": 1, "fp": 1, "fp_ok": 0, "matched_all": False,
                 "dets": [[0.4, False, 900.0, 30.0, 30.0, None, 3.0]]},
                {"path": "img3.png", "gt": 0, "gt_available": True,
                 "missed": 0, "fp": 1, "fp_ok": 1, "matched_all": False,
                 "dets": [[0.6, False, 2500.0, 50.0, 50.0, None, None]]},
            ]}


class AxisPlanTest(unittest.TestCase):
    """固定坐标轴规划（既定口径）：score 0~1；几何维 max 外扩取整。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_score_axis_fixed_0_1(self):
        from xccv.testing.threshold_panel import _axis_ticks, ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        x0, x1 = p.plot.getViewBox().viewRange()[0]
        self.assertEqual((x0, x1), (0.0, 1.0))
        ticks = _axis_ticks(1.0, 2)[0]
        self.assertEqual(ticks[0], (0.0, "0.00"))
        self.assertEqual(ticks[-1], (1.0, "1.00"))

    def test_area_axis_padded_nice(self):
        from xccv.testing.threshold_panel import _nice_ceil, ThresholdPanel

        self.assertEqual(_nice_ceil(550.0), 1000.0)
        self.assertEqual(_nice_ceil(0.03), 0.05)
        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        idx = p.cmb_dim.findData("area")
        p.cmb_dim.setCurrentIndex(idx)
        x_max, bins = p._plan_axis("area")
        # 最大 9000 ×1.1=9900 → 箱宽 500 → 20 箱 → 轴上限 10000（箱宽整数倍）
        self.assertEqual(x_max, 10000.0)
        self.assertEqual(bins[0], 0.0)
        self.assertEqual(bins[-1], x_max)
        self.assertAlmostEqual(float(bins[1] - bins[0]), 500.0)
        self.assertEqual(len(bins), 21)
        x0, x1 = p.plot.getViewBox().viewRange()[0]
        self.assertEqual((x0, x1), (0.0, 10000.0))

    def test_axis_without_data_no_negative(self):
        """未加载数据时坐标轴不出现负数（conf 出现 -0.2）。"""
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats({"gt_available": False}, 0.25)
        x0, x1 = p.plot.getViewBox().viewRange()[0]
        self.assertGreaterEqual(x0, 0.0)
        self.assertEqual((x0, x1), (0.0, 1.0))

    def test_y_axis_starts_at_zero(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        y0, _y1 = p.plot.getViewBox().viewRange()[1]
        self.assertEqual(y0, 0.0)


class SpinInputTest(unittest.TestCase):
    """阈值数字输入框（V0.7.1 追加）：键入与拖线同效，且相互同步。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_spin_input_enables_dim(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        got = {}
        p.thresholdsChanged.connect(lambda d: (got.clear(), got.update(d)))
        p.cmb_dim.setCurrentIndex(p.cmb_dim.findData("area"))
        p.spin_thr.setValue(2000.0)
        self.assertEqual(p._dim_state["area"]["thr"], 2000.0)
        self.assertTrue(p._dim_state["area"]["enabled"])
        self.assertEqual(got, {"score": 0.25, "area": 2000.0})
        self.assertAlmostEqual(float(p._line.value()), 2000.0)

    def test_input_clamped_to_axis(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        p.cmb_dim.setCurrentIndex(p.cmb_dim.findData("area"))
        p.spin_thr.setValue(999999.0)
        self.assertLessEqual(p._dim_state["area"]["thr"],
                             p._x_max + 1e-9)
        self.assertAlmostEqual(p._dim_state["area"]["thr"], p._x_max)

    def test_line_drag_updates_spin(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        p.cmb_dim.setCurrentIndex(p.cmb_dim.findData("short"))
        p._line.setValue(40.0)
        self.assertAlmostEqual(float(p.spin_thr.value()), 40.0)

    def test_score_never_negative(self):
        """拖线拖出固定轴（负值）时钳回 0（V0.7.1 既有实测问题）。"""
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        p._line.setValue(-0.5)
        self.assertEqual(p._dim_state["score"]["thr"], 0.0)


class SmoothLineTest(unittest.TestCase):
    """V0.7.2 扫描曲线（重设计）：三条曲线=阈值处 正常/过检/漏检 总数。

    score 维也有漏检线（gt_total−正常）；平滑散点直方图已被扫描曲线取代。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_sweep_counts_semantics(self):
        """扫描曲线数值口径：x=阈值处的 正常/过检/漏检 总数（联合过滤）。"""
        from xccv.engine.metrics.batch_match import sweep_counts

        stats = _panel_stats()
        xs = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]
        sw = sweep_counts(stats, "score", xs)
        self.assertEqual(sw["normal"], [2, 2, 2, 1, 0, 0])
        self.assertEqual(sw["over"], [2, 1, 0, 0, 0, 0])
        self.assertEqual(sw["missed"], [1, 1, 1, 2, 3, 3])
        # 一对一匹配不变量：normal + missed = gt_total
        for n, m in zip(sw["normal"], sw["missed"]):
            self.assertEqual(n + m, 3)

    def test_sweep_joint_filter(self):
        """联合过滤口径：area 阈值参与时 score 扫描只数过面积阈的检出。"""
        from xccv.engine.metrics.batch_match import sweep_metrics

        m = sweep_metrics(_panel_stats(), "score", 0.0,
                          fixed={"area": 500.0})
        # 过面积阈：0.9（area9000 命中）、0.4（area900 未命中）、0.6（OK 误检）；
        # 0.8（area400）被面积阈值筛掉不参与
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 2)

    def test_sweep_group_counts_semantics(self):
        """图口径扫描：曲线读数=该阈值下三组的**图数**，
        与复盘分组完全同口径（过检>漏检>正确）。"""
        from xccv.testing.review_logic import sweep_group_counts

        stats = _panel_stats()
        sw = sweep_group_counts(stats, "score", [0.0, 1.0])
        # x=0：img1 全命中无 FP→正确；img2 有 FP→过检；img3 OK 误检→过检
        self.assertEqual(sw["correct"], [1, 1])
        self.assertEqual(sw["over"], [2, 0])
        self.assertEqual(sw["missed"], [0, 2])
        # 恒等式：正确+过检+漏检 = 已评估图数（3）
        for c, o, m in zip(sw["correct"], sw["over"], sw["missed"]):
            self.assertEqual(c + o + m, 3)

    def test_three_series_on_area(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        p.cmb_dim.setCurrentIndex(p.cmb_dim.findData("area"))
        p._redraw()
        names = {d.name() for d in p.plot.getPlotItem().listDataItems()
                 if d.name()}
        self.assertEqual(names, {"正确", "过检", "漏检"})

    def test_score_has_missed_line_now(self):
        """V0.7.2 重设计：score 维同样有漏检曲线（旧版仅几何维度可画）。"""
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        p.set_stats(_panel_stats(), 0.25)
        p._redraw()
        names = {d.name() for d in p.plot.getPlotItem().listDataItems()
                 if d.name()}
        self.assertEqual(names, {"正确", "过检", "漏检"})

    def test_fg_ratio_removed_from_panel(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        stats = _panel_stats()
        stats["dims"]["fg_ratio"] = {"green": [0.8], "red_defect": [0.01],
                                     "red_ok": []}
        p = ThresholdPanel()
        p.set_stats(stats, 0.25)
        keys = [p.cmb_dim.itemData(i) for i in range(p.cmb_dim.count())]
        self.assertNotIn("fg_ratio", keys)


class MissedDimsTest(unittest.TestCase):
    """batch_match.dims_missed：漏检 GT 框的几何维度（任务 3 数据源）。"""

    def test_missed_gt_dims_collected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [
                _rect("黑点", (0, 0, 100, 60)),      # 命中
                _rect("划痕", (120, 10, 180, 50)),   # 漏检（宽60 高40）
            ])
            results = [{"path": str(d / "a.png"), "detections": [
                {"label": "黑点", "score": 0.9, "bbox": [0.0, 0.0, 100.0, 60.0]},
            ]}]
            stats = collect_match_stats(results, iou_thr=0.5)
            self.assertEqual(stats["dims_missed"]["area"], [2400.0])
            self.assertEqual(stats["dims_missed"]["short"], [40.0])
            self.assertEqual(stats["dims_missed"]["long"], [60.0])

    def test_hit_gt_not_in_missed(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (0, 0, 100, 60))])
            results = [{"path": str(d / "a.png"), "detections": [
                {"label": "黑点", "score": 0.9, "bbox": [0.0, 0.0, 100.0, 60.0]},
            ]}]
            stats = collect_match_stats(results, iou_thr=0.5)
            self.assertEqual(stats["dims_missed"]["area"], [])
            self.assertEqual(stats["n_unmatched"], 0)


# ================= 测试平台：IOU 默认 0.2（任务 4） =================

class IouDefaultTest(unittest.TestCase):
    """全平台 IOU 默认 0.2；引擎默认值除外。"""

    def test_ui_and_service_defaults(self):
        from xccv.deploy_consistency import run_consistency_check
        from xccv.service.config import ServiceConfig

        self.assertEqual(ServiceConfig().infer_iou, 0.20)
        import inspect

        sig = inspect.signature(run_consistency_check)
        self.assertEqual(sig.parameters["iou"].default, 0.20)

    def test_source_locks(self):
        cases = [
            ("xccv/testing/testing_window.py", '"IOU 阈值", 10, 90, 20'),
            ("xccv/testing/review_page.py", "setValue(0.20)"),
            ("xccv/annotation/auto_annotate_window.py", "setValue(0.20)"),
            ("xccv/engine/flow/threshold_flow.py", '"iou": 0.20'),
            ("xccv/engine/flow/ops/ai_bridge.py",
             'ParamSpec("iou", "IOU", "float", 0.20'),
            ("xccv/engine/flow/ops/listops.py",
             'ParamSpec("iou", "IOU", "float", 0.20'),
            ("xccv/engine/trt_runtime.py", "iou: float = 0.20"),
            ("xccv/engine/tiled_infer.py", "iou: float = 0.20"),
        ]
        for f, needle in cases:
            src = Path(f).read_text(encoding="utf-8")
            self.assertIn(needle, src, f"{f} 缺少 IOU 0.2 默认：{needle}")


# ================= 训练平台（任务 1/2/3/4/5） =================

class SplitCountsTest(unittest.TestCase):
    """split_counts 与 split_samples 同口径（环状图推算）。"""

    def test_mutually_exclusive(self):
        self.assertEqual(split_counts(100, 0.2, 0.8), (80, 20))
        self.assertEqual(split_counts(67, 0.2, 0.8), (54, 13))
        self.assertEqual(split_counts(10, 0.5, None), (5, 5))

    def test_overlap(self):
        self.assertEqual(split_counts(100, 0.2, 1.0), (100, 20))
        self.assertEqual(split_counts(100, 0.3, 0.9), (90, 30))

    def test_val_zero_reuses_train(self):
        self.assertEqual(split_counts(100, 0.0, 0.8), (80, 80))

    def test_empty(self):
        self.assertEqual(split_counts(0, 0.2, 0.8), (0, 0))

    def test_donut_canvas_signature(self):
        from PyQt5.QtWidgets import QApplication

        _app = QApplication.instance() or QApplication([])
        from xccv.training.charts import SplitDonutCanvas

        c = SplitDonutCanvas()
        c.update_split(80, 20, 0.8, 0.2)
        c.update_split(80, 0, 0.8, 0.2)
        c.update_split(0, 0, 0.8, 0.2)
        c.update_split(80, 20, 1.0, 0.2)
        self.assertTrue(True)


class TrainingWindowLockTest(unittest.TestCase):
    """训练平台修复源码锁：横幅保底绘制 + 表格列宽 + 实时环状图。"""

    def test_banner_repaint_and_min_visible(self):
        src = Path("xccv/training/training_window.py").read_text(
            encoding="utf-8")
        self.assertIn("banner.repaint()", src)          # 同步绘制保底
        self.assertIn("remain_ms", src)                  # 最小可见时长
        # 横幅逻辑抽为 _show_busy_banner/_finish_busy_banner（新建项目/打开项目共用）
        # （新建项目/打开项目共用），最小可见 700ms 默认进助手签名
        self.assertIn("_show_busy_banner(", src)
        self.assertIn("_finish_busy_banner(banner, t0)", src)
        self.assertIn("min_ms: int = 700", src)

    def test_per_class_table_no_stretch(self):
        # 0.7.6 P-1 拆分：逐类表构建在 panels/train_page.py（锁随迁）
        src = Path("xccv/training/panels/train_page.py").read_text(
            encoding="utf-8")
        self.assertIn("setStretchLastSection(False)", src)
        self.assertIn("QHeaderView.ResizeToContents", src)
        self.assertIn("ScrollPerPixel", src)

    def test_split_donut_live_rebind(self):
        # 0.7.6 P-1 拆分：划分环状图实时重绘在 panels/info_page.py（锁随迁）
        src = Path("xccv/training/panels/info_page.py").read_text(
            encoding="utf-8")
        self.assertIn("par_train_ratio.valueChanged.connect("
                      "self._refresh_split_canvas)", src)
        self.assertIn("par_val_ratio.valueChanged.connect("
                      "self._refresh_split_canvas)", src)
        self.assertIn("def _refresh_split_canvas", src)

    def test_review_page_left_wider(self):
        src = Path("xccv/testing/review_page.py").read_text(encoding="utf-8")
        self.assertIn("setFixedWidth(520)", src)
        self.assertNotIn("拖线后自动重新分组，不重新推理）", src)  # 标题精简


class DataGridReapplyTest(unittest.TestCase):
    """数据网格筛选/切标签后缓存图标回贴（任务 3 空白渲染修复）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory()
        td = Path(cls._tmp.name)
        _write_png(td / "a.png")
        _write_json(td / "a.json", [_rect("黑点", (1, 1, 20, 20))])
        _write_png(td / "b.png")
        cls._td = td

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _pump(self, widget, seconds=5.0):
        end = time.time() + seconds
        while time.time() < end:
            self.app.processEvents()
            if widget._scan_worker is None and widget._thumb_worker is None:
                break
            time.sleep(0.02)
        self.app.processEvents()

    def test_filter_reapplies_cached_icons(self):
        from xccv.training.data_grid import get_data_grid_class

        w = get_data_grid_class()()
        w.set_dataset(self._td)
        self._pump(w)
        self.assertEqual(len(w._cache), 2)
        # 筛选重建条目后：不再等缩略线程，缓存图标必须已回贴
        w.cb_ok.setChecked(False)     # 触发一次 _apply_filter（无实际过滤）
        w.cmb_label.setCurrentIndex(0)
        self._pump(w, 0.5)
        for i in range(w.list.count()):
            item = w.list.item(i)
            pm = item.icon().pixmap(112, 112)
            # 非纯占位灰（#EDF1F7）：真实缩略图存在彩色像素
            img = pm.toImage()
            colors = {img.pixelColor(x, y).name()
                      for x in (0, 50, 100) for y in (0, 50, 100)}
            self.assertTrue(any(c not in ("#edf1f7",) for c in colors),
                            f"{item.text()} 仍是占位灰块")
        w.shutdown()


class PerClassTableOffscreenTest(unittest.TestCase):
    """逐类指标表：末列不拉伸（全屏还原 AP@50 修复）+ 像素滚动。"""

    def test_header_modes(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        _app = QApplication.instance() or QApplication([])
        # 0.7.6 P-1 拆分：锁随迁到逐类表所在面板
        src = Path("xccv/training/panels/train_page.py").read_text(
            encoding="utf-8")
        self.assertNotIn("setStretchLastSection(True)", src)
        # offscreen 不构建 ProjectWindow（matplotlib 重）；以源码锁为准，
        # 行为断言在实测验收（清单 §九）执行


class PreviewOpsTest(unittest.TestCase):
    """增强预览全算子生效（任务 5）。"""

    def test_applied_names_and_flip(self):
        from xccv.engine.dataset.transforms import preview_apply_ops

        img = np.zeros((30, 40, 3), dtype=np.uint8)
        img[:, :20] = 255
        rng = random.Random(7)
        out, applied = preview_apply_ops(
            img, {"random_flip_h": {}}, {"random_flip_h": 1.0}, rng=rng)
        self.assertEqual(applied, ["random_flip_h"])
        self.assertTrue((out[:, :20] == 0).all())

    def test_prob_zero_skips(self):
        from xccv.engine.dataset.transforms import preview_apply_ops

        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        out, applied = preview_apply_ops(
            img, {"gaussian_noise": {"sigma": 99}},
            {"gaussian_noise": 0.0}, rng=random.Random(1))
        self.assertEqual(applied, [])
        self.assertTrue((out == 100).all())

    def test_resize_always_applied(self):
        """resize 是画布锚点：概率 0 也无条件应用（预览恒展示拉伸）。"""
        from xccv.engine.dataset.transforms import preview_apply_ops

        img = np.full((30, 40, 3), 128, dtype=np.uint8)
        out, applied = preview_apply_ops(
            img, {"resize": {"width": 20, "height": 10}},
            {"resize": 0.0}, rng=random.Random(1))
        self.assertEqual(applied, ["resize"])
        self.assertEqual(out.shape, (10, 20, 3))

    def test_resize_first_then_others(self):
        from xccv.engine.dataset.transforms import preview_apply_ops

        img = np.full((30, 40, 3), 128, dtype=np.uint8)
        rng = random.Random(3)
        out, applied = preview_apply_ops(
            img, {"resize": {"width": 20, "height": 10},
                  "random_brightness": {"max_delta": 5}},
            {"resize": 1.0, "random_brightness": 1.0}, rng=rng)
        self.assertEqual(applied, ["resize", "random_brightness"])
        self.assertEqual(out.shape, (10, 20, 3))

# == 来源模块：test_v072_fixes ==
"""V0.7.2 回归锁。

测试平台：
- ① x 轴刻度挤字根因（bottom.setWidth 钳死轴宽）——渲染级回归；
- ② 阈值扫描曲线：sweep_counts/sweep_unjudged/sweep_metrics 数值口径；
- ③ 已评估/未评估解耦：范围切换按钮 + OK/NG 分组 + 面板曲线口径联动；
- ④ A/B 比对缓存复用 + 总体优劣比例条；
平台总体：
- ① i18n：tr 回退/查表、localize_widget_tree 树遍历替换、语言持久化；
- ② 文案精简源码锁（部署页两句啰嗦说明已删）。
"""

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _review_stats():
    """带 per_image 逐检出的合成 match_stats（与 test_v071_fixes 同口径）。"""
    # _panel_stats：本文件 v071 段已定义（合并迁移）

    return _panel_stats()


class TickRenderTest(unittest.TestCase):
    """底部轴刻度文字必须随轴宽摊开（渲染级，插桩 drawPicture）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_bottom_axis_labels_spread(self):
        from pyqtgraph.graphicsItems.AxisItem import AxisItem
        from PyQt5.QtWidgets import QWidget, QHBoxLayout

        from xccv.testing.threshold_panel import ThresholdPanel

        panel = ThresholdPanel(compact=True)
        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(panel)
        host.resize(520, 500)
        panel.set_stats(_review_stats(), 0.25)
        host.show()
        self.app.processEvents()

        captured = []
        orig = AxisItem.drawPicture

        def _spy(self, p, axisSpec, tickSpecs, textSpecs):
            if self.orientation == "bottom":
                captured.extend(round(r.x(), 1) for r, _f, _s in textSpecs)
            return orig(self, p, axisSpec, tickSpecs, textSpecs)

        AxisItem.drawPicture = _spy
        try:
            panel._apply_fixed_axes("score")   # setTicks → picture 失效
            self.app.processEvents()
            panel.plot.grab()                  # 强制一次绘制
            self.app.processEvents()
        finally:
            AxisItem.drawPicture = orig
        axis_w = panel.plot.getAxis("bottom").width()
        # 回归根因断言：底部轴绝不被 setWidth 钳死成窄条
        self.assertGreater(axis_w, 300, f"bottom axis width={axis_w}")
        self.assertGreaterEqual(len(captured), 3, captured)
        self.assertGreater(max(captured) - min(captured), 200,
                           f"刻度文字未摊开：{captured}")

    def test_source_no_bottom_setwidth(self):
        src = Path("xccv/testing/threshold_panel.py").read_text(
            encoding="utf-8")
        self.assertNotIn('getAxis("bottom").setWidth', src)
        self.assertIn('getAxis("left").setWidth(42)', src)


class SweepTest(unittest.TestCase):
    """扫描曲线数据层（已评估三类 + 未评估 OK/NG）。"""

    def test_unjudged_ok_ng_semantics(self):
        from xccv.engine.metrics.batch_match import collect_match_stats, \
            sweep_unjudged

        results = [
            {"path": "p0.png", "detections": [
                {"label": "黑点", "score": 0.7, "bbox": [0, 0, 10, 10]}]},
            {"path": "p1.png", "detections": []},
            {"path": "p2.png", "detections": [
                {"label": "黑点", "score": 0.3, "bbox": [0, 0, 10, 10]}]},
        ]
        stats = collect_match_stats(results, iou_thr=0.5)
        self.assertFalse(stats["gt_available"])
        # 未评估图检出照记（OK/NG 曲线与分组的数据源）
        by = {Path(e["path"]).name: e for e in stats["per_image"]}
        self.assertEqual(len(by["p0.png"]["dets"]), 1)
        sw = sweep_unjudged(stats, "score", [0.0, 0.5, 0.9])
        self.assertEqual(sw["ng"], [2, 1, 0])
        self.assertEqual(sw["ok"], [1, 2, 3])
        self.assertEqual(sw["unjudged_total"], 3)

    def test_sweep_metrics_joint(self):
        from xccv.engine.metrics.batch_match import sweep_metrics

        m = sweep_metrics(_review_stats(), "score", 0.0, fixed={"area": 500.0})
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 2)
        self.assertEqual(m["ok_fp_per_img"], 1.0)
        self.assertEqual(m["unjudged_total"], 0)


class ScopePanelTest(unittest.TestCase):
    """面板随评估范围切换曲线口径与数字行。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_scope_switches_series(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel(compact=True)
        p.set_stats(_review_stats(), 0.25)
        p.set_scope("unjudged")
        p._redraw()
        # 合成 stats 无未评估图 → 不画线，数字行换 NG/OK
        names = {d.name() for d in p.plot.getPlotItem().listDataItems()
                 if d.name()}
        self.assertEqual(names, set())
        self.assertEqual(p.lbl_p_name.text(), "NG")
        self.assertEqual(p.lbl_r_name.text(), "OK")
        p.set_scope("evaluated")
        p._redraw()
        names = {d.name() for d in p.plot.getPlotItem().listDataItems()
                 if d.name()}
        self.assertEqual(names, {"正确", "过检", "漏检"})
        self.assertEqual(p.lbl_p_name.text(), "精确率")

    def test_unjudged_series_drawn(self):
        from xccv.engine.metrics.batch_match import collect_match_stats
        from xccv.testing.threshold_panel import ThresholdPanel

        stats = collect_match_stats([
            {"path": "q0.png", "detections": [
                {"label": "黑点", "score": 0.7, "bbox": [0, 0, 10, 10]}]},
            {"path": "q1.png", "detections": []},
        ], iou_thr=0.5)
        p = ThresholdPanel(compact=True)
        p.set_stats(stats, 0.25)
        p.set_scope("unjudged")
        p._redraw()
        names = {d.name() for d in p.plot.getPlotItem().listDataItems()
                 if d.name()}
        self.assertEqual(names, {"OK", "NG"})


class CompareCacheTest(unittest.TestCase):
    """A/B 比对缓存复用 + 总体优劣比例条。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_cache_restore_without_rerun(self):
        from xccv import i18n as _i18n
        from xccv.testing.model_compare_dialog import ModelCompareDialog

        _i18n.set_lang("zh")   # 断言依赖中文文案（用例末复位）
        try:
            records = [{"file": "a.png", "path": "x/a.png", "gt": 1,
                        "gt_available": True,
                        "a": {"n": 1, "hit": 1}, "b": {"n": 1, "hit": 1},
                        "verdict": "一致"}]
            payload = {"schema": 1, "model_a": "A.pt", "model_b": "B.pt",
                       "conf": 0.25, "iou": 0.2,
                       "summary": {"counts": {"一致": 1, "A优": 0, "B优": 0,
                                              "未评估": 0}},
                       "records": records}
            cache = {}
            d = ModelCompareDialog([{"path": "x/a.png"}], None, 0.25, 0.2,
                                   None, [], model_a_name="A.pt",
                                   cache=cache)
            cache[d._cache_key("B.pt")] = {"records": records,
                                            "payload": payload}
            self.assertTrue(d._try_restore("B.pt"))
            self.assertEqual(d.table.rowCount(), 1)
            self.assertIn("已复用缓存", d.lbl_summary.text())
            self.assertIsNotNone(d.payload())
            self.assertFalse(d._try_restore("C.pt"))   # 未命中不恢复
        finally:
            _i18n.set_lang("zh")

    def test_summary_bar_paint(self):
        from xccv.testing.model_compare_dialog import _SummaryBar

        b = _SummaryBar()
        b.set_counts({"A优": 3, "一致": 5, "B优": 2})
        b.resize(200, 20)
        self.assertFalse(b.grab().isNull())
        b.set_counts({})
        self.assertFalse(b.grab().isNull())

    def test_source_locks(self):
        src = Path("xccv/testing/testing_window.py").read_text(
            encoding="utf-8")
        self.assertIn("_ab_cache", src)
        src2 = Path("xccv/testing/model_compare_dialog.py").read_text(
            encoding="utf-8")
        self.assertIn("_try_restore", src2)
        self.assertIn("class _SummaryBar", src2)


class I18nTest(unittest.TestCase):
    """tr 替换层 + 树遍历 + 语言持久化（测试后还原偏好）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        from xccv import i18n

        cls.i18n = i18n
        cls._orig_lang = i18n.lang()

    @classmethod
    def tearDownClass(cls):
        cls.i18n.set_lang(cls._orig_lang)   # 还原用户语言偏好

    def test_tr_fallback_and_table(self):
        i18n = self.i18n
        try:
            i18n.set_lang("zh")
            self.assertEqual(i18n.tr("项目"), "项目")     # 中文模式原样
            i18n.set_lang("en")
            self.assertEqual(i18n.tr("项目"), "Project")
            self.assertEqual(i18n.tr("不存在的键xyz"), "不存在的键xyz")
        finally:
            i18n.set_lang("zh")   # 泄漏防护：后续模块测试假定中文环境

    def test_localize_widget_tree(self):
        from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QWidget

        i18n = self.i18n
        root = QWidget()
        lbl = QLabel("项目", root)
        grp = QGroupBox("正确 ✓（0）", root)
        btn = QPushButton("标准", root)
        try:
            i18n.set_lang("en")
            n = i18n.localize_widget_tree(root)
            self.assertGreaterEqual(n, 3)
            self.assertEqual(lbl.text(), "Project")
            self.assertEqual(grp.title(), "Correct ✓ (0)")
            self.assertEqual(btn.text(), "Standard")
            self.assertEqual(i18n.localize_widget_tree(root), 0)  # 中文跳过
        finally:
            i18n.set_lang("zh")

    def test_language_watcher_rebuilds_without_restart(self):
        """免重启生效：偏好文件变化 → watcher 触发 rebuild 回调。

        watcher 按生产约定不析构（Win32 监视线程销毁竞态）——
        挂到类属性保活到进程结束。"""
        import time

        from xccv import i18n

        calls = []
        orig = i18n.lang()
        w = i18n.install_language_watcher(lambda: calls.append(i18n.lang()))
        self.assertIsNotNone(w)
        type(self)._keep_alive = [w]   # 持有到进程结束（不析构）
        try:
            i18n.set_lang("en")   # 写文件 → 文件变化 → debounce 触发
            end = time.time() + 4
            while time.time() < end and not calls:
                self.app.processEvents()
                time.sleep(0.05)
            self.assertEqual(calls, ["en"])
        finally:
            i18n.set_lang(orig)
            i18n.set_lang("zh")

    def test_lang_switch_in_settings(self):
        """语言切换入口在主界面设置对话框。"""
        from xccv.i18n import restart_current_platform  # noqa: F401（可导入）

        src = Path("xccv/app.py").read_text(encoding="utf-8")
        self.assertIn("界面语言", src)
        self.assertIn("apply_language_choice", src)
        i18n_src = Path("xccv/i18n.py").read_text(encoding="utf-8")
        self.assertNotIn("make_lang_button", i18n_src)
        self.assertNotIn('"--"', i18n_src)   # 重启 stager 参数不放 "--"
        self.assertNotIn("make_lang_button",
                         Path("xccv/testing/testing_window.py").read_text(
                             encoding="utf-8"))
        self.assertNotIn("make_lang_button",
                         Path("xccv/training/training_window.py").read_text(
                             encoding="utf-8"))

    def test_axis_stable_across_scope_switch(self):
        """实测修复修复：已评估/未评估切换不重排坐标轴（轴只随数据集变化），
        阈值线与输入框不再被切出视图。"""
        from xccv.testing.threshold_panel import ThresholdPanel

        stats = _review_stats()
        # 补未评估图（检出面积远小于已评估），制造"轴会重排"的条件
        stats["per_image"] = stats["per_image"] + [
            {"path": "u0.png", "gt": None, "gt_available": False,
             "dets": [[0.7, False, 2000.0, 40.0, 60.0, None, None]]}]
        p = ThresholdPanel(compact=True)
        p.set_stats(stats, 0.25)
        p.cmb_dim.setCurrentIndex(p.cmb_dim.findData("area"))
        p._line.setValue(5000.0)   # 轴内取值（>x_max 会被 V0.7.1 钳位逻辑拉回）
        self.app.processEvents()
        x_before = p._x_max
        p.set_scope("unjudged")
        self.app.processEvents()
        self.assertEqual(p._x_max, x_before)          # 轴不随范围变化
        self.assertAlmostEqual(float(p._line.value()), 5000.0)
        self.assertEqual(p.spin_thr.maximum(), x_before)
        self.assertEqual(p.lbl_p_name.text(), "NG")   # 数字行口径已切换
        p.set_scope("evaluated")
        self.app.processEvents()
        self.assertEqual(p._x_max, x_before)

    def test_lang_persisted(self):
        i18n = self.i18n
        i18n.set_lang("en")
        data = json.loads(Path("platform_data/ui_lang.json").read_text(
            encoding="utf-8"))
        self.assertEqual(data["lang"], "en")
        i18n.set_lang("zh")
        data = json.loads(Path("platform_data/ui_lang.json").read_text(
            encoding="utf-8"))
        self.assertEqual(data["lang"], "zh")

    def test_localize_on_real_testing_window(self):
        """集成：英文模式下对真实测试平台窗口跑树遍历，常驻文案必须翻译。"""
        from xccv.testing.testing_window import TestingWindow

        self.i18n.set_lang("en")
        w = None
        try:
            # TestingWindow 在 __init__ 末尾自 localize——
            # 这里直接验收常驻文案的翻译结果（不再数替换次数）
            w = TestingWindow()
            self.assertEqual(w.windowTitle(), "XCcv Testing Platform")
            self.assertEqual(w.btn_mode_std.text(), "Standard")
            self.assertEqual(w.btn_mode_review.text(), "Project Review")
            rp = w.review_page
            self.assertEqual(rp.btn_scope_eval.text(), "Evaluated")
            self.assertEqual(rp.btn_scope_unjudged.text(), "Unevaluated")
            self.assertEqual(rp.grp_correct.title(), "Correct ✓ (0)")
            self.assertEqual(rp.grp_uok.title(), "OK (0)")
            self.assertEqual(rp.btn_ab.text(), "Model A/B Compare…")
            self.assertEqual(rp.threshold_panel.btn_apply.text(),
                             "Apply to Detection Flow")
            self.assertEqual(rp.threshold_panel.lbl_p_name.text(),
                             "Precision")
        finally:
            if w is not None:
                w.close()
            self.i18n.set_lang("zh")


class TextTrimLockTest(unittest.TestCase):
    """啰嗦文案精简源码锁（详细说明归操作手册）。"""

    def test_deploy_page_sub_and_hint_trimmed(self):
        src = Path("xccv/deploy_platform.py").read_text(encoding="utf-8")
        self.assertNotIn("三个独立模块", src)
        self.assertNotIn("各模块窗口独立运行", src)
        self.assertIn("推理服务 / 模型加载 / C# 测试，均以独立进程打开", src)
        self.assertIn("推理服务窗口关闭只最小化到托盘", src)

    def test_flow_hints_trimmed(self):
        src = Path("xccv/flow_platform.py").read_text(encoding="utf-8")
        self.assertNotIn("流程以 .json 保存于磁盘", src)
        # 旧版独立编辑器弃用归档，提示语修剪断言改挂 flow_shared
        self.assertFalse(
            Path("xccv/modeltools/flowgraph_main_window.py").exists(),
            "旧版流程图编辑器应已归档（_archive/）")
        src2 = Path("xccv/modeltools/flow_shared.py").read_text(
            encoding="utf-8")
        self.assertNotIn("流程以 .json 保存于磁盘", src2)


class TrImportLockTest(unittest.TestCase):
    """P248 回归锁：复判与回流 / A/B 比对对话框在 zh 模式构造即崩
    （tr 漏导入 NameError 导致的崩溃）。
    构造即触发 _reload_table→_refresh_stats 与 lbl_a 文案两条崩溃路径。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_batch_review_dialog_constructs_zh(self):
        import tempfile

        from xccv.i18n import lang, set_lang
        from xccv.testing.batch_review_dialog import BatchReviewDialog

        prev = lang()
        try:
            set_lang("zh")
            stats = _review_stats()
            review_path = Path(tempfile.gettempdir()) / "p248_review.json"
            dlg = BatchReviewDialog(
                stats.get("per_image") or [], stats, review_path,
                dataset_dir=None, parent=None)
            # 统计行已渲染（崩溃点 _refresh_stats 已执行）
            self.assertIn("：", dlg.lbl_stats.text())
            dlg.close()
            dlg.deleteLater()
        finally:
            set_lang(prev)

    def test_model_compare_dialog_constructs_zh(self):
        from xccv.i18n import lang, set_lang
        from xccv.testing.model_compare_dialog import ModelCompareDialog

        prev = lang()
        try:
            set_lang("zh")
            results_a = [{"path": "a.jpg", "detections": []}]
            dlg = ModelCompareDialog(results_a, out_dir=None, conf=0.25,
                                     iou=0.2, flow=None, classes=[],
                                     model_a_name="", parent=None)
            self.assertTrue(dlg.lbl_a.text())
            dlg.close()
            dlg.deleteLater()
        finally:
            set_lang(prev)

    def test_no_tr_use_without_import(self):
        """静态兜底：runs/audit_tr_import.py 全库扫描 tr 漏导入（仅引擎
        converters 的局部变量 tr 为已知误报）。"""
        import subprocess
        import sys

        script = Path("runs/audit_tr_import.py")
        if not script.is_file():
            self.skipTest("audit script not present")
        r = subprocess.run([sys.executable, str(script)], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        text = Path("runs/tr_import_audit.txt").read_text(encoding="utf-8")
        offenders = [ln for ln in text.splitlines() if ln.strip()
                     and "converters.py" not in ln]
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(offenders, [],
                         "tr 漏导入雷点：%s" % "; ".join(offenders[:5]))

if __name__ == '__main__':
    unittest.main()
