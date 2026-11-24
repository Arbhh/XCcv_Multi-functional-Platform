"""测试平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v07_review_split
- test_v07_o1_per_class
- test_v07_o21_ab
"""


# == 来源模块：test_v07_review_split ==
"""V0.7 重构：测试平台 标准/项目复盘 拆分回归锁。

- review_logic：三分组（过检>漏检>正确）、阈值复用现算、OK 图口径、未评估；
- batch_match：per_image 增逐检出 dets 记录（列序与 review_logic 对齐）；
- 阈值面板：thresholdsChanged 信号（拖线/开关 → enabled_dims）；
- 体检：__ignore__ 不再被当类别统计（"样本过少：__ignore__=1" 瑕疵修复）；
- 测试窗口：双页结构存在、底部页签存在、标准页不再含复判/AB/阈值面板。
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.io_utils import imwrite_unicode
from xccv.engine.metrics.batch_match import collect_match_stats
from xccv.testing.review_logic import (DET_FG, DET_MATCHED, DET_SCORE,
                                       classify_images, det_passes, stats_line)


def _write_png(path: Path):
    imwrite_unicode(str(path), np.full((8, 8, 3), 200, dtype=np.uint8))


def _write_json(path: Path, shapes: list):
    path.write_text(json.dumps({
        "version": "5.8.1", "flags": {}, "shapes": shapes,
        "imagePath": path.with_suffix(".png").name, "imageData": None,
        "imageHeight": 200, "imageWidth": 200,
    }), encoding="utf-8")


def _rect(label: str, box) -> dict:
    return {"label": label, "points": [[box[0], box[1]], [box[2], box[3]]],
            "shape_type": "rectangle"}


def _det(label, score, box, dims=None):
    d = {"label": label, "score": score, "bbox": [float(v) for v in box]}
    if dims:
        d["_dims"] = dims
    return d


class DetPassesTest(unittest.TestCase):

    def test_all_dims_must_pass(self):
        rec = [0.8, True, 5000.0, 50.0, 100.0, 0.6, 120.0]
        self.assertTrue(det_passes(rec, {"score": 0.5, "area": 1000}))
        self.assertFalse(det_passes(rec, {"area": 6000}))
        self.assertFalse(det_passes(rec, {"score": 0.9}))

    def test_none_value_fails(self):
        rec = [0.8, True, 5000.0, 50.0, 100.0, None, None]
        self.assertFalse(det_passes(rec, {"fg_ratio": 0.05}))  # 缺图维度不过
        self.assertTrue(det_passes(rec, {}))


class ClassifyTest(unittest.TestCase):

    def _stats(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (0, 0, 100, 100))])
            _write_png(d / "b.png")
            _write_json(d / "b.json", [_rect("黑点", (0, 0, 100, 100))])
            _write_png(d / "ok.png")
            _write_json(d / "ok.json", [])
            _write_png(d / "nogt.png")
            results = [
                {"path": str(d / "a.png"), "detections": [
                    _det("黑点", 0.9, (0, 0, 100, 100), {"fg_ratio": 0.8})]},
                {"path": str(d / "b.png"), "detections": [
                    _det("黑点", 0.9, (0, 0, 100, 100)),
                    _det("黑点", 0.4, (150, 150, 180, 180), {"fg_ratio": 0.01})]},
                {"path": str(d / "ok.png"), "detections": [
                    _det("黑点", 0.6, (10, 10, 50, 50))]},
                {"path": str(d / "nogt.png"), "detections": []},
            ]
            return collect_match_stats(results, iou_thr=0.5)

    def test_groups_and_rethreshold_without_rerun(self):
        stats = self._stats()
        # conf=0.25：a 正确；b 过检（多一个 0.4 FP）；ok 过检（OK 误检）；nogt 未评估
        g = classify_images(stats, {"score": 0.25})
        self.assertEqual([e["file"] for e in g["correct"]], ["a.png"])
        self.assertEqual([e["file"] for e in g["over"]], ["b.png", "ok.png"])
        self.assertEqual(len(g["unjudged_list"]), 1)
        self.assertEqual(g["unjudged"], 1)
        self.assertEqual(g["total"], 4)
        # 未评估图必须带条目进组（生产图肉眼可确认）
        self.assertEqual(g["unjudged_list"][0]["file"], "nogt.png")
        # conf=0.5：b 的低分 FP 被滤掉 → b 转正确；ok 仍过检（0.6≥0.5）
        g = classify_images(stats, {"score": 0.5})
        self.assertEqual([e["file"] for e in g["correct"]], ["a.png", "b.png"])
        self.assertEqual([e["file"] for e in g["over"]], ["ok.png"])
        # 前景占比维度：把 b 的主检出也滤掉 → b 变漏检
        g = classify_images(stats, {"score": 0.25, "fg_ratio": 0.5})
        self.assertEqual([e["file"] for e in g["missed"]], ["b.png"])

    def test_ok_dir_same_as_training_rule(self):
        """OK 图判定与训练平台目录名规则（NEGATIVE_DIR_NAMES）同源——
        OK 目录内无 json 的图按 OK 评估；非 OK 目录无 json 图 = 未评估。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            okd = d / "良品"
            okd.mkdir()
            _write_png(okd / "k.png")            # OK 目录无 json → OK 图
            _write_png(d / "prod.png")           # 无 json 生产图 → 未评估
            results = [{"path": str(okd / "k.png"), "detections": []},
                       {"path": str(d / "prod.png"), "detections": []}]
            stats = collect_match_stats(results, iou_thr=0.5)
            by = {Path(e["path"]).name: e for e in stats["per_image"]}
            self.assertTrue(by["k.png"]["gt_available"])       # OK 图参与评估
            self.assertTrue(by["k.png"]["ok_dir"])
            self.assertFalse(by["prod.png"]["gt_available"])   # 生产图未评估
            g = classify_images(stats, {})
            self.assertEqual(len(g["correct"]), 1)             # k.png 零检出=正确
            self.assertEqual(len(g["unjudged_list"]), 1)       # prod.png 可见

    def test_prod_only_folder_still_groups(self):
        """纯生产图（无任何 GT json）也必须产出完整 per_image，
        复盘「未评估」组要能展示每张图（修复 gt_available 短路）。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "p0.png")
            _write_png(d / "p1.png")
            results = [{"path": str(d / "p0.png"), "detections": [
                            _det("黑点", 0.5, (10, 10, 60, 60))]},
                       {"path": str(d / "p1.png"), "detections": []}]
            stats = collect_match_stats(results, iou_thr=0.5)
            self.assertFalse(stats["gt_available"])
            self.assertEqual(len(stats["per_image"]), 2)     # 不再被短路吞掉
            g = classify_images(stats, {})
            self.assertEqual(len(g["unjudged_list"]), 2)
            self.assertEqual(g["unjudged"], 2)
            self.assertEqual(g["total"], 2)

    def test_stats_line(self):
        g = {"correct": [1, 2], "missed": [1], "over": [], "unjudged": 3}
        line = stats_line(g)
        self.assertIn("正确 2", line)
        self.assertIn("漏检 1", line)
        self.assertIn("未评估 3", line)

    def test_stats_line_unjudged_scope(self):
        """未评估范围统计行给 OK/NG 口径。"""
        g = {"correct": [], "missed": [], "over": [],
             "unjudged_ok": 5, "unjudged_ng": 3, "unjudged": 8}
        line = stats_line(g, "unjudged")
        self.assertIn("OK 5", line)
        self.assertIn("NG 3", line)
        self.assertNotIn("正确", line)

    def test_unjudged_ok_ng_counts(self):
        """未评估按有无过阈检出分 OK/NG 计数（与切换后分组同口径）。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "p0.png")
            _write_png(d / "p1.png")
            results = [{"path": str(d / "p0.png"), "detections": [
                            _det("黑点", 0.5, (10, 10, 60, 60))]},
                       {"path": str(d / "p1.png"), "detections": []}]
            stats = collect_match_stats(results, iou_thr=0.5)
            g = classify_images(stats, {"score": 0.25})
            self.assertEqual(g["unjudged_ok"], 1)
            self.assertEqual(g["unjudged_ng"], 1)


class DetsRecordTest(unittest.TestCase):

    def test_per_image_dets_columns(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (0, 0, 100, 100))])
            results = [{"path": str(d / "a.png"), "detections": [
                _det("黑点", 0.9, (10, 10, 110, 110), {"fg_ratio": 0.7,
                                                       "gray_diff": 88.0})]}]
            stats = collect_match_stats(results, iou_thr=0.5)
            rec = stats["per_image"][0]["dets"][0]
            self.assertEqual(rec[DET_SCORE], 0.9)
            self.assertTrue(rec[DET_MATCHED])
            self.assertEqual(rec[DET_FG], 0.7)

    def test_ok_image_dets_all_unmatched(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "OK"
            d.mkdir()
            _write_png(d / "k.png")
            _write_json(d / "k.json", [])
            results = [{"path": str(d / "k.png"), "detections": [
                _det("黑点", 0.7, (5, 5, 45, 45))]}]
            stats = collect_match_stats(results, iou_thr=0.5)
            rec = stats["per_image"][0]["dets"][0]
            self.assertFalse(rec[DET_MATCHED])
            self.assertTrue(stats["per_image"][0]["gt_available"])


class PanelSignalTest(unittest.TestCase):
    """thresholdsChanged：拖线即启用并广播 enabled_dims。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_thresholds_changed_signal(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        stats = {"gt_available": True, "gt_total": 1, "n_gt_images": 1,
                 "n_ok_images": 0, "n_unmatched": 0, "green": [0.9],
                 "red_defect": [0.4], "red_ok": [],
                 "dims": {"area": {"green": [10000.0], "red_defect": [400.0],
                                   "red_ok": []}},
                 "per_image": []}
        p.set_stats(stats, 0.25)
        got = {}
        p.thresholdsChanged.connect(
            lambda d: (got.clear(), got.update(d)))
        idx = p.cmb_dim.findData("area")
        p.cmb_dim.setCurrentIndex(idx)
        p._line.setValue(500.0)   # 拖线 → 启用 area
        self.assertEqual(got, {"score": 0.25, "area": 500.0})
        p.cb_enable.setChecked(False)   # 取消参与 → 只剩 score
        self.assertEqual(got, {"score": 0.25})


class DataHealthIgnoreClassTest(unittest.TestCase):

    def test_ignore_not_counted_as_class(self):
        from xccv.training.data_health import run_health_check

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (10, 10, 60, 60)),
                                       _rect("__ignore__", (100, 100, 180, 180))])
            rep = run_health_check("detection", d)
            texts = [i.detail for i in rep.issues]
            self.assertFalse(any("__ignore__=" in t for t in texts),
                             texts)


class TestingWindowStructureTest(unittest.TestCase):
    """双页结构：底部页签 + 标准页不含本轮新增控件 + 复盘页核心件齐备。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        cls.win = None

    @classmethod
    def tearDownClass(cls):
        if cls.win is not None:
            cls.win.close()

    def test_two_pages_and_mode_buttons(self):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from xccv.testing.testing_window import TestingWindow

        w = TestingWindow()
        type(self).win = w
        # 底部页签（状态栏）+ 双页 stack
        self.assertTrue(w.btn_mode_std.isChecked())
        self.assertFalse(w.btn_mode_review.isChecked())
        self.assertEqual(w._detail_stack.count(), 2)
        self.assertIs(w._detail_stack.currentWidget(),
                      w._detail_stack.widget(0))
        w._switch_mode(1)
        self.assertIs(w._detail_stack.currentWidget(), w.review_page)
        self.assertTrue(w.btn_mode_review.isChecked())
        # 标准页已移除本轮新增控件
        self.assertFalse(hasattr(w, "btn_review"))
        self.assertFalse(hasattr(w, "btn_ab"))
        self.assertFalse(hasattr(w, "btn_dist"))
        self.assertFalse(hasattr(w, "threshold_panel"))
        # 复盘页核心件
        self.assertTrue(hasattr(w.review_page, "threshold_panel"))
        self.assertTrue(hasattr(w.review_page, "cmb_run"))
# 覆盖：数据集可浏览 + 五分组（已评估三组 + OK/NG 两组）
        self.assertTrue(hasattr(w.review_page, "btn_browse"))
        self.assertTrue(hasattr(w.review_page, "fld_dataset"))
        self.assertEqual(len(w.review_page.grids), 5)
        self.assertTrue(hasattr(w.review_page, "btn_scope_eval"))
        self.assertTrue(hasattr(w.review_page, "btn_scope_unjudged"))
        self.assertTrue(w.review_page.btn_scope_eval.isChecked())
        self.assertTrue(w.review_page.grp_uok.isHidden())
        w.review_page._set_scope("unjudged")
        self.assertTrue(w.review_page.grp_correct.isHidden())
        self.assertFalse(w.review_page.grp_uok.isHidden())
        self.assertTrue(w.review_page.grp_ung.isVisible() or
                        not w.review_page.grp_ung.isHidden())
        self.assertTrue(w.review_page.threshold_panel._scope == "unjudged")
        w.review_page._set_scope("evaluated")
        self.assertFalse(w.review_page.grp_correct.isHidden())
        # 面板为 compact 上下布局（直方图不被表单挤成窄条）
        from xccv.testing.threshold_panel import ThresholdPanel
        self.assertIsInstance(w.review_page.threshold_panel, ThresholdPanel)
        # 关闭/启动兜底清缓存——close 后 review_cache 目录必须消失
        cache = Path("platform_data/review_cache")
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "复盘缓存.json").write_text("{}", encoding="utf-8")
        w.close()
        self.assertFalse((cache / "复盘缓存.json").is_file(),
                         "closeEvent 未清理复盘缓存")
        w._switch_mode(0)

# == 来源模块：test_v07_o1_per_class ==
"""V0.7 O1：逐类精确率/召回率表回归锁。

聚合逻辑（engine.metrics.per_class）+ 两家族评估器的逐类 AP@50 提取 +
训练窗口三处调用点接线（源码锁）。不跑真训练——AP 提取用假数组/桩。
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from xccv.engine.metrics.per_class import (
    coco_per_class_ap50,
    per_class_rows,
    ultralytics_per_class_ap50,
)


def _cm_a_b() -> list:
    """三类手工矩阵（[gt, pred] 约定，末行/末列=背景）：
    A: TP8/FN2/FP3；B: TP5/FN0/FP1；C 全零。"""
    return [[8, 0, 0, 2],
            [0, 5, 0, 0],
            [0, 0, 0, 0],
            [3, 1, 0, 0]]


class PerClassRowsTest(unittest.TestCase):

    def test_aggregation_matches_manual_counts(self):
        rows = per_class_rows(_cm_a_b(), ["A", "B", "C"])
        a, b, c = rows
        self.assertEqual((a["tp"], a["fp"], a["fn"]), (8, 3, 2))
        self.assertEqual((a["gt"], a["pred"]), (10, 11))
        self.assertAlmostEqual(a["precision"], 8 / 11, places=4)
        self.assertAlmostEqual(a["recall"], 0.8, places=4)
        self.assertEqual((b["tp"], b["fp"], b["fn"]), (5, 1, 0))
        self.assertAlmostEqual(b["recall"], 1.0, places=4)
        self.assertEqual((c["gt"], c["pred"], c["tp"]), (0, 0, 0))

    def test_undefined_ratios_are_none(self):
        rows = per_class_rows(_cm_a_b(), ["A", "B", "C"])
        self.assertIsNone(rows[2]["precision"])   # 零检出分母
        self.assertIsNone(rows[2]["recall"])      # 零实例分母
        self.assertIsNotNone(rows[0]["precision"])

    def test_ap50_column_alignment_and_none(self):
        rows = per_class_rows(_cm_a_b(), ["A", "B", "C"], ap50_by_class=[0.9, 0.5])
        self.assertAlmostEqual(rows[0]["ap50"], 0.9, places=4)
        self.assertAlmostEqual(rows[1]["ap50"], 0.5, places=4)
        self.assertIsNone(rows[2]["ap50"])        # 缺项 → None（UI 显示 -）
        rows2 = per_class_rows(_cm_a_b(), ["A", "B", "C"])
        self.assertTrue(all(r["ap50"] is None for r in rows2))

    def test_shape_mismatch_raises_valueerror(self):
        with self.assertRaises(ValueError):
            per_class_rows([[1]], ["A", "B"])


class CocoPerClassAp50Test(unittest.TestCase):

    def test_valid_entries_averaged_at_iou_050(self):
        # [T=3 IoU(0.1,0.25,0.5), R=5, K=2 类, A=1, M=1]
        prec = np.full((3, 5, 2, 1, 1), -1.0)
        prec[2, :, 0, 0, 0] = [0.8, 0.7, -1.0, 0.9, 0.6]  # 类 0：哨兵剔除后均值 0.75
        self.assertAlmostEqual(coco_per_class_ap50(prec)[0], 0.75, places=4)

    def test_class_without_valid_entries_is_zero(self):
        prec = np.full((3, 5, 2, 1, 1), -1.0)
        out = coco_per_class_ap50(prec)
        self.assertEqual(out[1], 0.0)

    def test_shape_guard(self):
        with self.assertRaises(ValueError):
            coco_per_class_ap50(np.zeros((2, 5, 2, 1, 1)))


class UltralyticsPerClassAp50Test(unittest.TestCase):

    def test_missing_classes_filled_zero(self):
        ns = SimpleNamespace(ap50=[0.9, 0.5], ap_class_index=[0, 2])
        out = ultralytics_per_class_ap50(ns)
        self.assertEqual(out, [0.9, 0.0, 0.5])    # 类 1 未出现在验证集 → 0.0

    def test_absent_or_empty_metric_returns_none(self):
        self.assertIsNone(ultralytics_per_class_ap50(SimpleNamespace()))
        self.assertIsNone(ultralytics_per_class_ap50(
            SimpleNamespace(ap50=[], ap_class_index=[])))

    def test_broken_metric_returns_none(self):
        class Boom:
            @property
            def ap50(self):
                raise RuntimeError("boom")
        self.assertIsNone(ultralytics_per_class_ap50(Boom()))


class TrainingWindowWiringTest(unittest.TestCase):
    """UI 接线源码锁：表格构建 + 三处调用点（开始清空/历史详情/训练完成）。"""

    def test_call_sites_present(self):
        # 0.7.6 P-1 拆分：逐类表构建在 panels/train_page.py，刷新逻辑与
        # 导入在 panels/history_view.py——锁改为跨两文件计数（口径不变）
        train_src = Path("xccv/training/panels/train_page.py").read_text(
            encoding="utf-8")
        history_src = Path("xccv/training/panels/history_view.py").read_text(
            encoding="utf-8")
        self.assertIn("self.per_class_table = QTableWidget(0, 9)", train_src)
        self.assertEqual(
            history_src.count("self._update_per_class_table(")
            + train_src.count("self._update_per_class_table("), 5,
            "5 处调用（开始/进行中/历史详情×2/完成×2）；定义在 history_view")
        self.assertIn(
            "from ...engine.metrics.per_class import per_class_rows",
            history_src)

    def test_engine_per_class_key_emitted(self):
        # R-CNN/YOLO 评估器的 additive 键（源码锁，不跑真训练）
        rcnn_src = Path("xccv/engine/models/rcnn.py").read_text(encoding="utf-8")
        self.assertIn('out["per_class_ap50"] = coco_per_class_ap50(prec)', rcnn_src)
        yolo_src = Path("xccv/engine/models/yolo.py").read_text(encoding="utf-8")
        self.assertIn('out["per_class_ap50"] = per_ap', yolo_src)

# == 来源模块：test_v07_o21_ab ==
"""V0.7 O21：模型 A/B 批量比对回归锁（纯逻辑 + 对话框冒烟）。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xccv.engine.io_utils import imwrite_unicode
from xccv.testing.model_compare import (build_compare_payload,
                                        compare_batch, compare_image,
                                        compare_summary, score_dets)








def _det__v07_o21_ab(label, score, box):
    return {"label": label, "score": score, "bbox": [float(v) for v in box]}


class ScoreTest(unittest.TestCase):

    def test_score_dets(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        dets = [_det__v07_o21_ab("黑点", 0.9, (0, 0, 100, 100)), _det__v07_o21_ab("黑点", 0.5, (150, 150, 180, 180))]
        s = score_dets(dets, gt_b, gt_l)
        self.assertEqual((s["n"], s["hit"], s["fp"]), (2, 1, 1))


class CompareImageTest(unittest.TestCase):

    def test_a_better_by_hit(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        rec = compare_image([_det__v07_o21_ab("黑点", 0.9, (0, 0, 100, 100))],
                            [], gt_b, gt_l)
        self.assertEqual(rec["verdict"], "A优")

    def test_b_better_by_fp(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        rec = compare_image(
            [_det__v07_o21_ab("黑点", 0.9, (0, 0, 100, 100)), _det__v07_o21_ab("黑点", 0.4, (150, 150, 180, 180))],
            [_det__v07_o21_ab("黑点", 0.9, (0, 0, 100, 100))], gt_b, gt_l)
        self.assertEqual(rec["verdict"], "B优")

    def test_same_is_tie(self):
        gt_b = [[0, 0, 100, 100]]
        gt_l = ["黑点"]
        d = [_det__v07_o21_ab("黑点", 0.9, (0, 0, 100, 100))]
        self.assertEqual(compare_image(d, list(d), gt_b, gt_l)["verdict"], "一致")

    def test_ok_image_compares_fp(self):
        # OK 图（GT 空）：B 有误检 A 无 → A 优
        rec = compare_image([], [_det__v07_o21_ab("黑点", 0.6, (10, 10, 50, 50))], [], [])
        self.assertEqual(rec["verdict"], "A优")
        rec = compare_image([_det__v07_o21_ab("黑点", 0.6, (10, 10, 50, 50))],
                            [_det__v07_o21_ab("黑点", 0.6, (10, 10, 50, 50))], [], [])
        self.assertEqual(rec["verdict"], "一致")

    def test_no_gt_is_unjudged(self):
        rec = compare_image([_det__v07_o21_ab("黑点", 0.9, (0, 0, 10, 10))], [], [], [],
                            gt_available=False)
        self.assertEqual(rec["verdict"], "未评估")


class CompareBatchTest(unittest.TestCase):

    def test_batch_records_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_png(d / "a.png")
            _write_json(d / "a.json", [_rect("黑点", (0, 0, 100, 100))])
            _write_png(d / "ok.png")
            _write_json(d / "ok.json", [])
            _write_png(d / "nogt.png")
            results_a = [
                {"file": "a.png", "path": str(d / "a.png"),
                 "detections": [_det__v07_o21_ab("黑点", 0.9, (0, 0, 100, 100))]},
                {"file": "ok.png", "path": str(d / "ok.png"),
                 "detections": [_det__v07_o21_ab("黑点", 0.7, (5, 5, 40, 40))]},   # OK 误检
                {"file": "nogt.png", "path": str(d / "nogt.png"),
                 "detections": []},
            ]
            preds_b = {str(d / "a.png"): [],                    # A 命中 B 全漏
                       str(d / "ok.png"): []}                   # B 无误检 → B 优
            records = compare_batch(results_a, preds_b, 0.5)
            by = {r["file"]: r for r in records}
            self.assertEqual(by["a.png"]["verdict"], "A优")
            self.assertEqual(by["ok.png"]["verdict"], "B优")
            self.assertEqual(by["nogt.png"]["verdict"], "未评估")
            s = compare_summary(records)
            self.assertEqual(s["counts"], {"一致": 0, "A优": 1, "B优": 1,
                                           "未评估": 1})

    def test_payload_shape(self):
        records = [{"gt": 1, "a": {"n": 1, "hit": 1, "fp": 0},
                    "b": {"n": 0, "hit": 0, "fp": 0}, "verdict": "A优",
                    "gt_available": True, "path": "x", "file": "x.png"}]
        payload = build_compare_payload("a.pt", "b.pt", 0.25, 0.5, records)
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(payload["model_a"], "a.pt")
        self.assertIn("records", payload)
        with tempfile.TemporaryDirectory() as td:
            from xccv.testing.model_compare import save_compare_json

            p = Path(td) / "比对.json"
            save_compare_json(p, payload)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["summary"]["counts"]["A优"], 1)


class DialogSmokeTest(unittest.TestCase):
    """offscreen 对话框冒烟：填充表格与统计，不真跑模型。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_fill_and_payload(self):
        from xccv.testing.model_compare_dialog import ModelCompareDialog

        results_a = [{"file": "a.png", "path": "Z:/no/a.png",
                      "detections": [_det__v07_o21_ab("黑点", 0.9, (0, 0, 100, 100))]}]
        dlg = ModelCompareDialog(results_a, None, 0.25, 0.5, None, ["黑点"],
                                 model_a_name="a.pt")
        records = [{"gt": 1, "a": {"n": 1, "hit": 1, "fp": 0},
                    "b": {"n": 0, "hit": 0, "fp": 0}, "verdict": "A优",
                    "gt_available": True, "path": "Z:/no/a.png", "file": "a.png"}]
        dlg._fill_table(records)
        self.assertEqual(dlg.table.rowCount(), 1)
        self.assertEqual(dlg.table.item(0, 6).text(), "A优")
        self.assertIsNone(dlg.payload())

if __name__ == '__main__':
    unittest.main()
