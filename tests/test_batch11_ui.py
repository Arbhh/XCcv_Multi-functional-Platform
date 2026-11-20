"""UI 测试：齿轮开发者选项、体检/预检对话框、预检接线、launcher 透传、字号下限。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.training import dev_options as dev_mod
from xccv.training import project as project_mod
from xccv.training.data_health import HealthIssue, LEVEL_INFO, LEVEL_WARN
from xccv.training.dev_options import DevOptions
from xccv.training.preflight import Finding
from xccv.training.project import create_project, load_project

_APP = QApplication.instance() or QApplication(sys.argv)


class _TempEnv(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        p = mock.patch.object(dev_mod, "options_path",
                              return_value=Path(self._tmp.name) / "dev_options.json")
        p.start()
        self.addCleanup(p.stop)
        det = make_labelme_dataset(Path(self._tmp.name) / "det", n_images=4)[0]
        create_project("ui", str(det), "", "detection")


class CollapsibleSectionTest(_TempEnv):
    def test_toggle(self):
        from xccv.training.dialogs import CollapsibleSection

        sec = CollapsibleSection("测试分区", expanded=True)
        self.assertFalse(sec.content.isHidden())  # 初始展开
        sec.set_expanded(False)                   # 折叠
        self.assertTrue(sec.content.isHidden())
        sec.toggle()                              # 再展开
        self.assertFalse(sec.content.isHidden())
        self.assertIn("▾", sec.btn.text())


class HealthReportDialogTest(_TempEnv):
    def _rows(self, with_finding=True):
        rows = [HealthIssue(LEVEL_WARN, "imbalance", "类别不均衡", "3:1",
                            "建议开启工业优化")]
        if with_finding:
            rows.append(Finding(LEVEL_INFO, "val_zero", "验证集比例为 0"))
            rows.append(Finding(LEVEL_WARN, "patience", "早停不会触发",
                                field="patience", value=5, apply_text="设为 5 轮"))
        return rows

    def test_report_mode_builds(self):
        from xccv.training.dialogs import HealthReportDialog

        dlg = HealthReportDialog(None, "数据体检报告", self._rows(False),
                                 mode="report", summary="测试")
        self.assertEqual(dlg.mode, "report")
        dlg.close()

    def test_preflight_apply_records(self):
        from xccv.training.dialogs import HealthReportDialog

        applied = []
        dlg = HealthReportDialog(None, "训练前预检", self._rows(True),
                                 mode="preflight",
                                 on_apply=lambda k, v: applied.append((k, v)))
        self.assertTrue(dlg._btn_apply_all.isVisibleTo(dlg))
        dlg._apply_all()
        self.assertEqual(applied, [("patience", 5)])
        self.assertEqual(dlg.applied, [("patience", 5)])
        self.assertIn("早停耐心 → 5", dlg.applied_label.text())
        dlg.close()

    def test_no_warn_info_only_preflight(self):
        """仅提示级：对话框无「应用全部」也仍可开始（弹窗逻辑在窗口层判断）。"""
        from xccv.training.dialogs import HealthReportDialog

        rows = [HealthIssue(LEVEL_INFO, "val_zero", "验证集比例为 0")]
        dlg = HealthReportDialog(None, "训练前预检", rows, mode="preflight")
        self.assertFalse(dlg._btn_apply_all.isVisibleTo(dlg))
        dlg.close()


class DevOptionsDialogTest(_TempEnv):
    def test_save_roundtrip(self):
        from xccv.training.dialogs import DevOptionsDialog

        dlg = DevOptionsDialog()
        dlg.sp_imbalance.setValue(4.0)  # 对话框单位是百分比
        dlg.sp_cls.setValue(15)
        dlg.ck_preflight.setChecked(False)
        dlg.ck_info_popup.setChecked(False)
        dlg._save()   # 不走模态 exec_
        opt = dev_mod.load_options()
        self.assertEqual(opt.imbalance_share, 0.04)
        self.assertEqual(opt.min_cls_samples, 15)
        self.assertFalse(opt.preflight_enabled)
        self.assertFalse(opt.preflight_info_popup)

    def test_reset_defaults(self):
        from xccv.training.dialogs import DevOptionsDialog

        dlg = DevOptionsDialog()
        dlg.sp_imbalance.setValue(20.0)
        dlg._reset_defaults()
        self.assertEqual(dlg.sp_imbalance.value(), 5.0)


class PreflightWiringTest(_TempEnv):
    """预检接线：仅提示级不弹窗直接开始；警告级弹窗（stub）确认后开始。"""

    def _window(self):
        from xccv.training.training_window import ProjectWindow

        pw = ProjectWindow(load_project("ui"))
        self.addCleanup(pw.close)
        return pw

    def test_apply_preflight_value_mapping(self):
        pw = self._window()
        pw._apply_preflight_value("patience", 7)
        pw._apply_preflight_value("learning_rate", 0.01)
        pw._apply_preflight_value("batch_size", 8)
        self.assertEqual(pw.par_patience.value(), 7)
        self.assertEqual(pw.par_lr.value(), 0.01)
        self.assertEqual(pw.par_batch.value(), 8)

    def test_apply_resize_value(self):
        pw = self._window()
        pw.augment_controls["resize"][0].setChecked(False)
        pw._apply_preflight_value("resize", (512, 384))
        check, spins = pw.augment_controls["resize"]
        self.assertTrue(check.isChecked())
        self.assertEqual(pw._current_resize(), (512, 384))

    def test_info_only_preflight_starts_directly(self):
        """开发者选项关闭"提示级弹窗"时：仅提示不弹窗直接开始。"""
        # 0.7.6 P-1 拆分：预检对话框装配随 _on_preflight_done 迁入
        # panels/train_page.py，patch 目标随迁
        import xccv.training.training_window as tw
        import xccv.training.panels.train_page as tp

        pw = self._window()
        result = {"issues": [HealthIssue(LEVEL_INFO, "val_zero", "验证集比例为 0")],
                  "findings": [], "n_images": 4, "classes": ["cat", "dog"]}
        with mock.patch.object(tw.ProjectWindow, "_begin_training") as begin, \
                mock.patch.object(tp, "HealthReportDialog") as dlg_cls, \
                mock.patch.object(tp, "load_options",
                                  return_value=DevOptions(preflight_info_popup=False)):
            pw._on_preflight_done(result)
            begin.assert_called_once()
            dlg_cls.assert_not_called()

    def test_info_only_preflight_pops_when_enabled(self):
        """默认开启"提示级弹窗"：仅提示也弹预检窗。"""
        import xccv.training.training_window as tw
        import xccv.training.panels.train_page as tp

        pw = self._window()
        result = {"issues": [HealthIssue(LEVEL_INFO, "val_zero", "验证集比例为 0")],
                  "findings": [], "n_images": 4, "classes": ["cat", "dog"]}
        with mock.patch.object(tw.ProjectWindow, "_begin_training") as begin, \
                mock.patch.object(tp, "HealthReportDialog") as dlg_cls:
            dlg_cls.return_value.exec_.return_value = tw.QDialog.Accepted
            pw._on_preflight_done(result)
            dlg_cls.assert_called_once()
            begin.assert_called_once()

    def test_warn_preflight_shows_dialog_then_starts(self):
        import xccv.training.training_window as tw
        import xccv.training.panels.train_page as tp

        pw = self._window()
        result = {"issues": [HealthIssue(LEVEL_WARN, "imbalance", "类别不均衡")],
                  "findings": [Finding(LEVEL_WARN, "patience", "早停不触发",
                                       field="patience", value=5)],
                  "n_images": 4, "classes": ["cat", "dog"]}
        with mock.patch.object(tw.ProjectWindow, "_begin_training") as begin, \
                mock.patch.object(tp, "HealthReportDialog") as dlg_cls:
            dlg_cls.return_value.exec_.return_value = tw.QDialog.Accepted
            pw._on_preflight_done(result)
            dlg_cls.assert_called_once()
            begin.assert_called_once()


    def test_preflight_worker_runs_end_to_end(self):
        """后台线程同步执行：体检 + 参数建议一起产出（不弹窗）。"""
        from xccv.training.training_window import _PreflightWorker

        pw = self._window()
        results = []
        worker = _PreflightWorker(pw.project)
        worker.done.connect(results.append)
        worker.run()  # 不 start()：在调用线程内同步执行
        self.assertEqual(len(results), 1)
        payload = results[0]
        self.assertEqual(payload["n_images"], 4)
        self.assertEqual(payload["classes"], ["cat", "dog"])
        self.assertIsInstance(payload["issues"], list)
        self.assertIsInstance(payload["findings"], list)


class CompareEntryTest(_TempEnv):
    def _window_with_runs(self):
        from xccv.training.runs import RunRecord
        from xccv.training.training_window import ProjectWindow

        pw = ProjectWindow(load_project("ui"))
        self.addCleanup(pw.close)
        recs = []
        for i in range(3):
            run = Path(self._tmp.name) / f"run{i}"
            run.mkdir()
            rec = RunRecord(run_dir=run, status="completed",
                            start_time=f"2026-09-1{i} 10:00:00",
                            metrics={"config": {"model_name": "yolo11n"},
                                     "history": [{"epoch": 1, "train_loss": 2.0}]})
            recs.append(rec)
        return pw, recs

    def test_selection_capped_at_three(self):
        """多选上限 3 条：第 4 个选择被拒绝并回退。"""
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QListWidgetItem

        pw, recs = self._window_with_runs()
        self.assertTrue(pw.btn_compare.isEnabled())  # 按钮常启用，点击时校验
        for rec in recs:
            item = QListWidgetItem(str(rec.run_dir))
            item.setData(Qt.UserRole, str(rec.run_dir))   # 回退按键（与真实列表一致）
            item.setData(Qt.UserRole + 1, rec)
            pw.run_list.addItem(item)
        item4 = QListWidgetItem("extra")
        item4.setData(Qt.UserRole, "extra")
        item4.setData(Qt.UserRole + 1, recs[0])
        pw.run_list.addItem(item4)
        for it in tuple(pw.run_list.item(i) for i in range(4)):
            it.setSelected(True)
        self.assertEqual(len(pw.run_list.selectedItems()), 3)  # 第 4 个被拒
        self.assertFalse(item4.isSelected())

    def test_open_compare_with_stub(self):
        import xccv.training.compare_window as cw
        import xccv.training.training_window as tw
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QListWidgetItem

        pw, recs = self._window_with_runs()
        for rec in recs[:2]:
            item = QListWidgetItem(str(rec.run_dir))
            item.setData(Qt.UserRole + 1, rec)
            pw.run_list.addItem(item)
            item.setSelected(True)  # 先入列再选中（脱离列表的条目选中无效）
        with mock.patch.object(cw, "CompareWindow") as cmp_cls:
            pw._open_compare()
            cmp_cls.assert_called_once()
            self.assertIsNotNone(pw._compare_window)


class LauncherArgTest(unittest.TestCase):
    def test_launch_with_arg_appends(self):
        import xccv.launcher as launcher

        captured = {}
        with mock.patch.object(launcher, "is_installed", return_value=False), \
                mock.patch.object(launcher.subprocess, "Popen",
                                  side_effect=lambda cmd, **kw:
                                  captured.update(cmd=cmd)):
            launcher.launch_platform("annotation", "D:/datasets/demo")
        self.assertEqual(captured["cmd"][-1], "D:/datasets/demo")
        self.assertIn("-m", captured["cmd"])

    def test_launch_without_arg(self):
        import xccv.launcher as launcher

        captured = {}
        with mock.patch.object(launcher, "is_installed", return_value=False), \
                mock.patch.object(launcher.subprocess, "Popen",
                                  side_effect=lambda cmd, **kw:
                                  captured.update(cmd=cmd)):
            launcher.launch_platform("annotation")
        self.assertEqual(len(captured["cmd"]), 3)  # pythonw -m module


class FontSizeAuditTest(unittest.TestCase):
    def test_no_font_below_14px(self):
        """全局字号审查（界面可读性约束）：QSS 中不得低于 14px。"""
        import re

        from xccv.ui_style import APP_QSS

        sizes = [int(m) for m in re.findall(r"font-size:\s*(\d+)px", APP_QSS)]
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), 15, f"过小字号出现：{sizes}")


if __name__ == "__main__":
    unittest.main()
