"""回归锁：bug-优化项 一.1/一.2/二.1/二.2/二.3。

- 一.1 免训练范式快速模式适配：free 模型在快速配置下不再出现轮次语义
  （轮次/批量/学习率行隐藏；k-shot 模型换支持集张数/增强样本数行与
  免训练预设卡；非 k-shot 的 free 模型隐藏预设卡）；
- 一.2 i18n 债务：数据体检/预检/Loss 判读/智能推荐理由 tr() 化（EN 表
  键与代码模板逐字一致——以 tr 输出断言非中文）；
- 二.1 阈值线交互：原生 bounds 钳制（线不滑出轴）+ 拖线防抖（程序化
  setValue 立即广播语义不变）；
- 二.2 双击渲染：结果窗按阈值绿/红框 + 无框任务横幅（离屏像素断言）；
- 二.3 流程图跳转：FlowGraphMainWindow(open_file=...) 启动即载入。
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication(sys.argv)


class FreeQuickModeTest(unittest.TestCase):
    """一.1：免训练模型在快速配置下的语义适配。"""

    @classmethod
    def setUpClass(cls):
        from tests.synthetic import make_classification_dataset
        from xccv.training import project as project_mod

        cls._tmp = tempfile.TemporaryDirectory()
        project_mod.PROJECTS_DIR = Path(cls._tmp.name) / "projects"
        cls_dir, _ = make_classification_dataset(
            Path(cls._tmp.name) / "cls", per_class=4)
        from xccv.training.project import create_project

        create_project("free_t", str(cls_dir), "", "anomaly")
        det, _ = __import__("tests.synthetic", fromlist=["make_labelme_dataset"]) \
            .make_labelme_dataset(Path(cls._tmp.name) / "det", n_images=4)
        create_project("epoch_t", str(det), "", "detection")
        from xccv.training.training_window import ProjectWindow

        cls.ProjectWindow = ProjectWindow

    @classmethod
    def tearDownClass(cls):
        from xccv.training import project as project_mod

        project_mod.PROJECTS_DIR = Path.home()  # 占位（下个测试会重定向）
        cls._tmp.cleanup()

    def _window(self, name, model):
        from xccv.training.project import load_project

        pw = self.ProjectWindow(load_project(name))
        pw.par_model.blockSignals(True)
        idx = pw.par_model.findData(model)
        if idx >= 0:
            pw.par_model.setCurrentIndex(idx)
        pw.par_model.blockSignals(False)
        pw._on_model_changed()
        self.addCleanup(pw.close)
        return pw

    def test_free_kshot_quick_mode(self):
        pw = self._window("free_t", "subspacead")
        self.assertEqual(pw._train_mode, "free")
        for row in (pw._quick_epochs_row, pw._quick_batch_row,
                    pw._quick_lr_row):
            self.assertTrue(row.isHidden(), "free 下轮次/批量/学习率行应隐藏")
        self.assertFalse(pw._quick_kshot_row.isHidden())
        self.assertFalse(pw._preset_cards["fast"].isHidden())
        self.assertIn("快速建库", pw._preset_labels["fast"][0].text())
        self.assertEqual(pw._preset_values["fast"],
                         {"k_shot": 2, "aug_count": 6})
        pw._apply_preset("quality")
        self.assertEqual(pw.par_kshot.value(), 8)
        self.assertEqual(pw.par_augcount.value(), 60)
        self.assertEqual(pw.q_kshot.value(), 8, "Quick 控件未随预设同步")

    def test_free_non_kshot_hides_cards(self):
        pw = self._window("free_t", "cif")
        self.assertEqual(pw._train_mode, "free")
        self.assertTrue(pw._quick_kshot_row.isHidden())
        self.assertTrue(pw._quick_augcount_row.isHidden())
        for card in pw._preset_cards.values():
            self.assertTrue(card.isHidden(),
                            "非 k-shot 的免训练模型无预设可换，应隐藏卡片")

    def test_epoch_mode_restores(self):
        pw = self._window("epoch_t", "yolo11n")
        self.assertEqual(pw._train_mode, "epoch")
        self.assertFalse(pw._quick_epochs_row.isHidden())
        self.assertTrue(pw._quick_kshot_row.isHidden())
        self.assertFalse(pw._preset_cards["fast"].isHidden())
        self.assertEqual(pw._preset_values["fast"]["epochs"], 30,
                         "预设卡应恢复轮次语义")


class I18nHealthTextTest(unittest.TestCase):
    """一.2：数据体检/Loss 判读文本在 EN 模式下输出英文。"""

    def setUp(self):
        from xccv.i18n import set_lang

        set_lang("en")
        self.addCleanup(set_lang, "zh")   # P241 复位

    def test_health_findings_english(self):
        from tests.synthetic import make_labelme_dataset
        from xccv.training.data_health import run_health_check

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_labelme_dataset(d, n_images=6)
            rep = run_health_check("detection", str(d))
            joined = " ".join(i.title + i.detail + i.suggestion
                              for i in rep.issues)
            joined = joined.replace("OK/良品/正常", "")  # 文件夹名专有名词
            self.assertNotRegex(joined, r"[\u4e00-\u9fff]",
                                "EN 模式下体检文本仍含中文（一.2 债务回潮）")

    def test_recommend_reasons_english(self):
        from xccv.training.preflight import recommend_settings

        class _P:
            task = "detection"
            model_name = "yolo11n"
            epochs = 10
            batch_size = 8
            freeze_backbone = False
            industrial_opt = True
            use_amp = True
            device = "auto"
            val_ratio = 0.2
            patience = 10
            multiply_factor = 1
            p2_head = False
            tiling = False
            augment_ops = {"resize": {"width": 640, "height": 640}}

        from xccv.training.preflight import DatasetFacts

        facts = DatasetFacts(class_counts={"a": 30})
        recs = recommend_settings(_P(), facts, cuda_available=False)
        self.assertTrue(recs)
        for _k, _v, reason in recs:
            self.assertNotRegex(reason, r"[\u4e00-\u9fff]",
                                "EN 模式下推荐理由仍含中文")

    def test_loss_diagnosis_english(self):
        from xccv.training.loss_diagnosis import diagnose

        hist = [{"epoch": i, "train_loss": 1.0 - i * 0.01,
                 "map_loose": 0.9 - i * 0.05} for i in range(1, 9)]
        out = diagnose(hist)
        self.assertEqual(out["kind"], "overfit")
        self.assertNotRegex(out["text"], r"[\u4e00-\u9fff]")


class ThresholdLineBoundsTest(unittest.TestCase):
    """二.1：阈值线原生钳制 + 拖线防抖。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_line_bounds_follow_axis(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        stats = {"gt_available": True, "gt_total": 1, "n_gt_images": 1,
                 "n_ok_images": 0, "n_unmatched": 0, "green": [0.9],
                 "red_defect": [0.4], "red_ok": [],
                 "dims": {"area": {"green": [10000.0], "red_defect": [400.0],
                                   "red_ok": []}},
                 "per_image": []}
        p.set_stats(stats, 0.25)
        idx = p.cmb_dim.findData("area")
        p.cmb_dim.setCurrentIndex(idx)
        lo, hi = p._line.maxRange
        self.assertEqual((lo, hi), (0.0, p._x_max),
                         "阈值线拖动范围未随固定轴更新（线可滑出轴）")
        p._line.setValue(p._x_max + 500.0)   # 原生钳制应把它收回轴内
        self.assertLessEqual(float(p._line.value()), p._x_max)

    def test_programmatic_set_emits_immediately(self):
        from xccv.testing.threshold_panel import ThresholdPanel

        p = ThresholdPanel()
        got = []
        p.thresholdsChanged.connect(lambda d: got.append(dict(d)))
        p._line.setValue(0.8)
        self.assertEqual(got and got[-1], {"score": 0.8},
                         "程序化 setValue 丢失同步广播（回归锁破坏）")


class ResultViewRenderTest(unittest.TestCase):
    """二.2：双击渲染窗（绿/红框 + 无框任务横幅）。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        import numpy as np

        cls.np = np

    def _entry(self, task="detection"):
        img = self.np.full((120, 160, 3), 235, self.np.uint8)
        out = Path(self._tmpdir())
        import cv2

        cv2.imwrite(str(out / "a.png"), img)
        return {"file": "a.png", "path": str(out / "a.png"),
                "width": 160, "height": 120,
                "detections": [
                    {"label": "黑点", "score": 0.9,
                     "bbox": [10, 10, 60, 60],
                     "_dims": {"area": 2500, "short": 50, "long": 50}},
                    {"label": "异色", "score": 0.1,
                     "bbox": [80, 40, 130, 90],
                     "_dims": {"area": 2500, "short": 50, "long": 50}}],
                "image_score": 3.2,
                "classifications": [{"label": "ng", "score": 0.77}]}

    def _tmpdir(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)   # 临时目录留给系统清理
        return d

    def test_boxes_green_red_by_threshold(self):
        from xccv.testing.result_view import render_result_pixmap

        pm = render_result_pixmap(self._entry(), {"score": 0.25}, "detection")
        self.assertIsNotNone(pm)
        img = pm.toImage()
        # 沿两个框的四条边采样：绿框 (10,10)-(60,60)、红框 (80,40)-(130,90)
        green = red = 0
        for x0, y0, x1, y1 in ((10, 10, 60, 60), (80, 40, 130, 90)):
            for x in range(x0, x1 + 1):
                for y in (y0, y1):
                    c = img.pixelColor(min(x, img.width() - 1),
                                       min(y, img.height() - 1))
                    if c.green() > 130 and c.red() < 110:
                        green += 1
                    if c.red() > 140 and c.green() < 120:
                        red += 1
            for y in range(y0, y1 + 1):
                for x in (x0, x1):
                    c = img.pixelColor(min(x, img.width() - 1),
                                       min(y, img.height() - 1))
                    if c.green() > 130 and c.red() < 110:
                        green += 1
                    if c.red() > 140 and c.green() < 120:
                        red += 1
        self.assertGreater(green, 0, "过阈检出未渲染绿框")
        self.assertGreater(red, 0, "被过滤检出未渲染红框")

    def test_anomaly_banner_not_blank(self):
        from xccv.testing.result_view import render_result_pixmap

        pm = render_result_pixmap(self._entry(), {"score": 1.0}, "anomaly")
        self.assertIsNotNone(pm)
        img = pm.toImage()
        dark = sum(1 for x in range(0, img.width(), 4)
                   for y in range(0, min(40, img.height()))
                   if img.pixelColor(x, y).lightness() < 90)
        self.assertGreater(dark, 10, "异常检测无横幅显示（二.2 不能无显示）")

    def test_dialog_builds(self):
        from xccv.i18n import set_lang
        from xccv.testing.result_view import ResultViewDialog

        set_lang("en")
        try:
            dlg = ResultViewDialog(None, self._entry(), {"score": 0.25},
                                   task="detection")
            self.assertIn("Detection Result", dlg.windowTitle())
            dlg.close()
        finally:
            set_lang("zh")


class FlowGraphOpenFileTest(unittest.TestCase):
    """二.3：流程图跳转管道（V0.7.5 重定向到主界面流程图页新版链路）。

    旧版独立编辑器已弃用归档；跳转 = launch_platform("main", 模板路径)
    → home_jump.handle_args → FlowPage.open_flow_file。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_open_file_via_home_jump(self):
        import types
        from unittest import mock

        from PyQt5.QtWidgets import QMainWindow

        from xccv import home_jump
        from xccv.flow_platform import FlowPage

        f = Path(tempfile.mkdtemp()) / "t.json"
        f.write_text(json.dumps({"nodes": [], "links": []}),
                     encoding="utf-8")
        win = QMainWindow()
        page = FlowPage()
        win.setCentralWidget(page)
        win.show()
        self.addCleanup(win.close)
        fake = types.SimpleNamespace(
            FlowGraphWebWindow=__import__("unittest").mock.MagicMock())
        with mock.patch.dict(sys.modules,
                             {"xccv.modeltools.flowgraph_web_window": fake}):
            self.assertEqual(home_jump.handle_args(win, [str(f)]), str(f))
        self.assertEqual(page.current_flow_path, f)
        if getattr(win, "nav_list", None) is not None:
            self.assertEqual(win.nav_list.currentRow(), 2)  # 流程图页已定位


if __name__ == "__main__":
    unittest.main()

