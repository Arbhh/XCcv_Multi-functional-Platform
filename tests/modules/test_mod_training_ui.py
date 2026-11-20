"""模块行为锁：训练平台 UI（Quick/专家双轨 / 预设 / 数据体检对话框）。

行为级口径：双轨参数双向同步、预设换装（epoch/free 两态）、
体检报告对话框装配。工程夹具重定向到临时目录。
"""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication(sys.argv)


class TrainingUiBehaviorTest(unittest.TestCase):
    """ProjectWindow 双轨与预设（模块级冒烟，不锁历史实现细节）。"""

    @classmethod
    def setUpClass(cls):
        from tests.synthetic import make_classification_dataset, make_labelme_dataset
        from xccv.training import project as project_mod

        cls._tmp = tempfile.TemporaryDirectory()
        project_mod.PROJECTS_DIR = Path(cls._tmp.name) / "projects"
        cls_dir, _ = make_classification_dataset(
            Path(cls._tmp.name) / "cls", per_class=4)
        det_dir, _ = make_labelme_dataset(Path(cls._tmp.name) / "det",
                                          n_images=4)
        from xccv.training.project import create_project

        create_project("mod_cls", str(cls_dir), "", "anomaly")
        create_project("mod_det", str(det_dir), "", "detection")
        from xccv.training.training_window import ProjectWindow

        cls.ProjectWindow = ProjectWindow

    @classmethod
    def tearDownClass(cls):
        from xccv.training import project as project_mod

        project_mod.PROJECTS_DIR = Path.home()  # 占位（P241 家族约定）
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

    def test_quick_expert_bidirectional_sync(self):
        pw = self._window("mod_det", "yolo11n")
        self.assertEqual(pw._train_mode, "epoch")
        pw.q_epochs.setValue(50)
        pw._experts_from_quick()
        self.assertEqual(pw.par_epochs.value(), 50, "Quick→专家同步失效")
        pw.par_epochs.setValue(33)
        pw._quick_from_experts()
        self.assertEqual(pw.q_epochs.value(), 33, "专家→Quick 同步失效")
        pw._reset_quick_defaults()
        self.assertGreater(pw.q_epochs.value(), 0)

    def test_preset_apply_epoch_semantics(self):
        pw = self._window("mod_det", "yolo11n")
        pw._apply_preset("fast")
        self.assertEqual(pw.par_epochs.value(), 30, "快速预设轮次语义丢失")
        self.assertEqual(pw.q_epochs.value(), 30, "预设未同步 Quick 控件")
        self.assertFalse(pw._rec_badges["epochs"].isHidden(),
                         "预设推荐徽章未点亮（窗口未显示用 isHidden 判定）")

    def test_free_kshot_preset_semantics(self):
        pw = self._window("mod_cls", "subspacead")
        self.assertEqual(pw._train_mode, "free")
        self.assertTrue(pw._quick_epochs_row.isHidden())
        self.assertFalse(pw._quick_kshot_row.isHidden())
        pw._apply_preset("balanced")
        self.assertEqual(pw.par_kshot.value(), 4)
        self.assertEqual(pw.q_kshot.value(), 4, "Quick k-shot 未随预设同步")
        self.assertEqual(pw.par_augcount.value(), 30)


class HealthDialogAssemblyTest(unittest.TestCase):
    """体检报告 → 对话框装配（rows_from_issues 契约）。"""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_dialog_builds_from_issues(self):
        from tests.synthetic import make_labelme_dataset
        from xccv.training.data_health import run_health_check
        from xccv.training.dialogs import HealthReportDialog, rows_from_issues

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            make_labelme_dataset(d, n_images=6)
            rep = run_health_check("detection", str(d))
            self.assertTrue(rep.issues, "合成小数据集体检应至少产出提示项")
            for issue in rep.issues:
                self.assertTrue(issue.title)
                self.assertIsInstance(issue.level, str)
            rows = rows_from_issues(rep.issues)
            self.assertTrue(rows)
            dlg = HealthReportDialog(None, "数据体检", rows, mode="report")
            dlg.close()
            dlg.deleteLater()


class TrainingPageSnapshotTest(unittest.TestCase):
    """P-1 拆分前置：信息页/训练页结构快照（页级组合根契约）。

    ProjectWindow.training_window 拆分为 panels/* 时，以本快照核对
    页面组合未漂移。有意变更页面结构时：跑 runs/gen_page_snapshot19.py
    重新生成快照并在交接卡登记（差异属于有意的 UI 变更）。
    """

    @classmethod
    def setUpClass(cls):
        from xccv.i18n import set_lang

        set_lang("zh")   # 快照口径锁定 zh4654用后复位）
        cls._lang_reset = True
        from tests.synthetic import make_classification_dataset, make_labelme_dataset
        from xccv.training import project as project_mod

        cls._tmp = tempfile.TemporaryDirectory()
        project_mod.PROJECTS_DIR = Path(cls._tmp.name) / "projects"
        cls_dir, _ = make_classification_dataset(
            Path(cls._tmp.name) / "cls", per_class=4)
        det_dir, _ = make_labelme_dataset(Path(cls._tmp.name) / "det",
                                          n_images=4)
        from xccv.training.project import create_project, load_project
        from xccv.training.training_window import ProjectWindow

        create_project("snap_cls", str(cls_dir), "", "anomaly")
        create_project("snap_det", str(det_dir), "", "detection")
        cls.ProjectWindow = ProjectWindow
        

    @classmethod
    def tearDownClass(cls):
        from xccv.i18n import set_lang
        from xccv.training import project as project_mod

        project_mod.PROJECTS_DIR = Path.home()
        cls._tmp.cleanup()
        if getattr(cls, "_lang_reset", False):
            set_lang("zh")

    def _page_sig(self, page):
        from PyQt5.QtWidgets import (QCheckBox, QGroupBox, QLabel,
                                     QPushButton, QRadioButton, QToolButton,
                                     QWidget)

        out = []
        def rec(w):
            name = type(w).__name__
            obj = w.objectName() or ""
            text = ""
            if isinstance(w, (QLabel, QPushButton, QCheckBox, QRadioButton,
                              QToolButton)):
                text = w.text()
            elif isinstance(w, QGroupBox):
                text = w.title()
            elif type(w).__name__ == "CollapsibleSection":
                text = str(getattr(w, "title", ""))
            # 动态内容归一（快照口径：结构锁不含运行时刻/随机文件名/尺寸）
            if re.search(r"\d{4}-\d{2}-\d{2}", text) or \
                    re.search(r"\d{1,2}:\d{2}(:\d{2})?", text):
                text = re.sub(r"\d{4}-\d{2}-\d{2}[ T]?(\d{1,2}:)?\d{1,2}:\d{2}",
                              "<ts>", text)
                text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "<ts>", text)
            text = re.sub(r"img\d+\.(png|jpg|jpeg|bmp)", "img<N>.<ext>", text)
            text = re.sub(r"\d+×\d+", "<WxH>", text)
            text = re.sub(r"\d+\s*/\s*\d+", "<N>/<M>", text)
            out.append(name + (f"[{obj}]" if obj else "")
                       + (f"“{text}”" if text else ""))
            for c in w.children():
                if isinstance(c, QWidget) and type(c).__name__ != "QMenu":
                    rec(c)
        rec(page)
        return out

    def test_pages_match_snapshot(self):
        import json

        from PyQt5.QtWidgets import QStackedWidget

        from xccv.training.project import load_project
        pw = self.ProjectWindow(load_project("snap_det"))
        self.addCleanup(pw.close)
        stack = pw.findChild(QStackedWidget, "detail_stack") or getattr(
            pw, "detail_stack", None)
        self.assertIsInstance(stack, QStackedWidget)
        self.assertEqual(stack.count(), 2, "信息/训练双页结构变更")
        snap_path = Path(__file__).parent / "snapshots" / "training_pages.json"
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        for idx, key in ((0, "info"), (1, "train")):
            got = self._page_sig(stack.widget(idx))
            self.assertEqual(
                got, snap[key],
                f"{key} 页结构与快照不一致（P-1 拆分漂移或有意变更——"
                f"跑 runs/gen_page_snapshot19.py 重生成并登记）")

    def test_quick_panel_present(self):
        from xccv.training.project import load_project
        pw = self.ProjectWindow(load_project("snap_det"))
        self.addCleanup(pw.close)
        self.assertTrue(hasattr(pw, "quick_panel"), "Quick 面板丢失")
        self.assertTrue(hasattr(pw, "sec_basic"), "专家分栏丢失")


if __name__ == "__main__":
    unittest.main()
