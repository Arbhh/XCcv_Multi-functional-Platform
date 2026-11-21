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
    from tests.test_v071_fixes import _panel_stats

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


if __name__ == "__main__":
    unittest.main()


