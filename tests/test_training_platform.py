"""训练平台测试：项目存取、数据集统计、训练线程端到端（真实训练 1 轮）、窗口构建。"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop
from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_classification_dataset, make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.training import project as project_mod
from xccv.training.dataset_stats import collect_stats
from xccv.training.project import (
    create_project, delete_project, list_projects, load_project, save_project,
)
from xccv.training.trainer_thread import TrainThread

_APP = QApplication.instance() or QApplication(sys.argv)


class ProjectStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"  # 重定向，避免污染真实数据

    def _mk_det(self) -> str:
        d = Path(self._tmp.name) / "det"
        if not d.exists():
            make_labelme_dataset(d, n_images=4)
        return str(d)

    def test_create_load_rename_delete(self):
        project = create_project("demo", self._mk_det(), "", "detection")
        self.assertIn("demo", list_projects())
        loaded = load_project("demo")
        self.assertEqual(loaded.dataset_dir, project.dataset_dir)

        loaded.name = "demo2"
        save_project(loaded, old_name="demo")
        self.assertIn("demo2", list_projects())
        self.assertNotIn("demo", list_projects())

        delete_project("demo2")
        self.assertEqual(list_projects(), [])

    def test_duplicate_and_invalid(self):
        create_project("demo", self._mk_det(), "", "detection")
        with self.assertRaises(ValueError):
            create_project("demo", self._mk_det(), "", "detection")
        with self.assertRaises(ValueError):
            create_project("bad/name", self._mk_det(), "", "detection")
        with self.assertRaises(ValueError):
            create_project("ok", "Z:/not_exist_dir", "", "detection")

    def test_new_project_resize_default_on(self):
        """新建项目：增强策略 resize 默认开启（同时作为训练输入尺寸）。"""
        p = create_project("r1", self._mk_det(), "", "detection")
        self.assertEqual(p.augment_ops.get("resize"), {"width": 640, "height": 640})

    def test_migrate_legacy_image_size(self):
        """旧版 project.json 的方形 image_size 迁移为 resize 宽高。"""
        import json as _json
        d = project_mod.PROJECTS_DIR / "legacy"
        d.mkdir(parents=True)
        (d / "project.json").write_text(_json.dumps({
            "name": "legacy", "dataset_dir": self._mk_det(),
            "model_dir": str(Path(self._tmp.name) / "m"), "task": "detection",
            "image_size": 320, "augment_ops": {},
        }, ensure_ascii=False), encoding="utf-8")
        p = load_project("legacy")
        self.assertEqual(p.augment_ops.get("resize"), {"width": 320, "height": 320})


class DatasetStatsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_detection_stats(self):
        det, _ = make_labelme_dataset(self.root / "det", n_images=6)
        stats = collect_stats(TaskType.DETECTION, det)
        self.assertEqual(stats.n_images, 6)
        self.assertEqual(stats.classes, ["cat", "dog"])
        self.assertEqual(sum(stats.class_counts.values()), 6)
        self.assertEqual(len(stats.widths), 6)

    def test_classification_stats(self):
        cls, _ = make_classification_dataset(self.root / "cls", per_class=5)
        stats = collect_stats(TaskType.CLASSIFICATION, cls)
        self.assertEqual(stats.n_images, 10)
        self.assertEqual(stats.class_counts, {"cat": 5, "dog": 5})


class TrainThreadTest(unittest.TestCase):
    """训练线程端到端：真实训练 1 轮分类模型，产出指标与权重。"""

    def test_classification_training(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cls_dir, _ = make_classification_dataset(Path(tmp.name) / "cls", per_class=4)

        project_mod.PROJECTS_DIR = Path(tmp.name) / "projects"
        project = create_project("t", str(cls_dir),
                                 str(Path(tmp.name) / "models"), "classification")
        project.model_name = "resnet18"
        project.epochs = 1
        project.batch_size = 2
        project.augment_ops = {"resize": {"width": 32, "height": 32}}  # 训练输入与 resize 统一
        project.workers = 0
        project.use_pretrained = False
        project.val_ratio = 0.25

        epochs = []
        loop = QEventLoop()
        result = {}

        thread = TrainThread(project, Path(tmp.name) / "run1")
        thread.epoch_changed.connect(lambda e, logs: epochs.append((e, logs)))
        thread.train_finished.connect(
            lambda ok, msg, metrics: (result.update(ok=ok, msg=msg, metrics=metrics),
                                      loop.quit()))
        thread.start()
        loop.exec_()
        thread.wait(120000)

        self.assertTrue(result["ok"], result.get("msg"))
        self.assertGreaterEqual(len(epochs), 1)
        self.assertIn("train_loss", epochs[-1][1])
        self.assertIn("final_val_top1", result["metrics"])
        # 训练产物明文 best.pt 落盘
        self.assertTrue((Path(tmp.name) / "run1" / "best.pt").exists())


class SafeRunTagTest(unittest.TestCase):
    """训练输出目录名安全化：模型引用（pruned:/run: 前缀）含冒号等非法字符。"""

    def test_pruned_reference(self):
        from xccv.training.training_window import _safe_run_tag

        self.assertEqual(_safe_run_tag("pruned:resnet18_pruned30.pt"),
                         "pruned_resnet18_pruned30")

    def test_run_reference(self):
        from xccv.training.training_window import _safe_run_tag

        self.assertEqual(_safe_run_tag("run:20260906_100000_resnet18/best.pt"),
                         "run_20260906_100000_resnet18_best")

    def test_plain_name_unchanged(self):
        from xccv.training.training_window import _safe_run_tag

        self.assertEqual(_safe_run_tag("yolo11n"), "yolo11n")
        self.assertEqual(_safe_run_tag(""), "model")


class TrainingWindowTest(unittest.TestCase):
    """项目列表窗口 + 独立项目窗口（多开）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        self.det, _ = make_labelme_dataset(Path(self._tmp.name) / "det", n_images=4)
        create_project("ui", str(self.det), "", "detection")

        from xccv.training.training_window import TrainingWindow

        self.win = TrainingWindow()
        self.addCleanup(self.win.close)

    def _open(self, name: str):
        self.win._open_project(name)
        return self.win.open_project_windows()[name]

    def test_project_cards(self):
        self.assertEqual(self.win.project_count(), 1)
        self.assertEqual(self.win._cards[0].project_name, "ui")

    def _wait_stats(self, pw, timeout_s: float = 30.0):
        """统计已后台化——泵事件直到统计线程收尾。"""
        from PyQt5.QtCore import QEventLoop, QTimer

        deadline = time.monotonic() + timeout_s
        while getattr(pw, "_stats_thread", None) is not None \
                and time.monotonic() < deadline:
            loop = QEventLoop()
            QTimer.singleShot(10, loop.quit)
            loop.exec_()
        QApplication.processEvents()

    def test_open_project_window_and_fields(self):
        from xccv.training.training_window import ProjectWindow

        pw = self._open("ui")
        self.assertIsInstance(pw, ProjectWindow)
        self.assertTrue(pw.isVisible())
        self.assertEqual(pw.info_name.text(), "ui")
        self.assertEqual(pw.par_model.currentData(), "yolo11n")
        QApplication.processEvents()  # 数据集统计已延迟（窗口先显示，统计随后补齐）
        self._wait_stats(pw)  # 统计后台线程收尾后断言
        self.assertTrue(pw.stats_hint.text().startswith("共 4 张"))

    def test_multi_open_and_singleton(self):
        create_project("ui2", str(self.det), "", "detection")
        self.win._refresh_project_list()
        self.assertEqual(self.win.project_count(), 2)
        self._open("ui")
        self._open("ui2")
        self.assertEqual(len(self.win.open_project_windows()), 2)  # 多开
        self._open("ui")
        self.assertEqual(len(self.win.open_project_windows()), 2)  # 同项目只激活已有窗口

    def test_rename_updates_window_registry(self):
        pw = self._open("ui")
        pw.info_name.setText("renamed")
        pw._save_info()
        self.assertIn("renamed", self.win.open_project_windows())
        self.assertNotIn("ui", self.win.open_project_windows())
        self.assertIn("renamed", list_projects())
        self.assertEqual(pw.windowTitle().split("  -")[0].strip(), "renamed")

    def test_augment_save_via_window(self):
        pw = self._open("ui")
        pw.augment_controls["random_brightness"][0].setChecked(True)
        ops = pw._collect_augment_ops()
        self.assertIn("random_brightness", ops)
        pw._save_augment()
        self.assertIn("random_brightness", load_project("ui").augment_ops)

    def test_spin_no_wheel(self):
        """参数框滚轮不应改变数值（防误触）。"""
        from PyQt5.QtCore import QPoint, QEvent, QPointF, Qt
        from PyQt5.QtGui import QWheelEvent

        pw = self._open("ui")
        before = pw.par_epochs.value()
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                         Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        pw.par_epochs.wheelEvent(ev)
        self.assertEqual(pw.par_epochs.value(), before)

    def test_bubble_chart_renders(self):
        from xccv.training.charts import DistributionCanvas

        canvas = DistributionCanvas()
        stats = collect_stats(TaskType.DETECTION, self.det)
        canvas.update_stats(stats)  # 不抛异常即通过
        self.assertEqual(len(stats.widths), 4)


if __name__ == "__main__":
    unittest.main()


