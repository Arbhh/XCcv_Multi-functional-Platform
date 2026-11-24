"""训练平台测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_training_platform
- test_training_history
- test_qt_slots
"""


# == 来源模块：test_training_platform ==
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


def _wait_live_threads(timeout_s: float = 30.0) -> None:
    """等待后台线程收尾（自动登记表见 xccv.gui_utils.auto_cleanup_thread）。

    项目页会起统计/预检线程；若在临时目录清理前仍未结束，Windows 上会因
    文件句柄占用报 WinError 32（测试偶发）。本清理先于临时目录删除执行。
    """
    from xccv import gui_utils

    app = QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while gui_utils._LIVE_THREADS and time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        time.sleep(0.02)

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
        self.addCleanup(_wait_live_threads)
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

# == 来源模块：test_training_history ==
"""训练历史记录：metrics.json 留档、历史扫描、训练可视化页历史列表。"""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

from tests.synthetic import make_classification_dataset, make_labelme_dataset
from xccv.engine.dataset.base import TaskType
from xccv.engine.io_utils import read_json
from xccv.engine.models import TrainConfig, create_model
from xccv.engine.models.yolo import YoloModel
from xccv.training import project as project_mod
from xccv.training.project import create_project
from xccv.training.runs import RunRecord, format_duration, load_run, scan_runs
from xccv.training.training_window import _IntAxis, _PagePlot, _PlotHover



def _write_metrics(run_dir: Path, **overrides):
    run_dir.mkdir(parents=True, exist_ok=True)
    m = {
        "status": "completed",
        "start_time": "2026-09-04 10:00:00",
        "end_time": "2026-09-04 10:05:00",
        "duration_sec": 300,
        "epochs_planned": 10,
        "config": {"model_name": "yolo11n", "epochs": 10, "batch_size": 8},
        "history": [{"epoch": i, "train_loss": 1.0 / i, "val_top1": 0.5 + 0.01 * i}
                    for i in range(1, 11)],
        "classes": ["cat", "dog"],
    }
    m.update(overrides)
    (run_dir / "metrics.json").write_text(
        __import__("json").dumps(m, ensure_ascii=False), encoding="utf-8")
    return m


class RunScanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_empty_dir(self):
        self.assertEqual(scan_runs(self.root), [])

    def test_completed_record(self):
        run = self.root / "20260904_100000_yolo11n"
        run.mkdir()
        _write_metrics(run)
        runs = scan_runs(self.root)
        self.assertEqual(len(runs), 1)
        rec = runs[0]
        self.assertEqual(rec.status, "completed")
        self.assertEqual(rec.status_text, "已完成")
        self.assertEqual(rec.start_time, "2026-09-04 10:00:00")
        self.assertEqual(rec.duration_sec, 300)
        self.assertEqual(rec.epochs_done, 10)
        self.assertEqual(rec.progress, 1.0)
        self.assertEqual(rec.model_name, "yolo11n")

    def test_interrupted_record(self):
        run = self.root / "20260904_100001_resnet18"
        run.mkdir()
        (run / "best.pt").write_bytes(b"x")  # 只有权重没有指标 → 中断
        rec = scan_runs(self.root)[0]
        self.assertEqual(rec.status, "interrupted")
        self.assertEqual(rec.status_text, "已中断")

    def test_failed_record_and_corrupt_json(self):
        failed = self.root / "20260904_100002_m1"
        failed.mkdir()
        (failed / "data").mkdir()  # 有残留无产出 → 失败
        corrupt = self.root / "20260904_100003_m2"
        corrupt.mkdir()
        (corrupt / "metrics.json").write_text("{broken", encoding="utf-8")
        statuses = {r.run_dir.name: r.status for r in scan_runs(self.root)}
        self.assertEqual(statuses["20260904_100002_m1"], "failed")
        self.assertEqual(statuses["20260904_100003_m2"], "failed")

    def test_empty_subdir_skipped_and_order_desc(self):
        (self.root / "20260904_100000_a").mkdir()
        _write_metrics(self.root / "20260904_100001_b", status="interrupted")
        (self.root / "20260904_100004_c").mkdir()  # 空目录不进历史
        runs = scan_runs(self.root)
        self.assertEqual([r.run_dir.name for r in runs], ["20260904_100001_b"])

    def test_reserved_dirs_not_records(self):
        """回归：保留子目录不是训练记录（轮次权重/流程图模板）。

        flow/ 内的模板文件名同样带时间戳前缀（<时间>_<模型名>.json），
        修复前被 load_run 判为「有残留无产出 → 失败」混进历史列表
        （训练可视化/复盘页/模型工具箱三处同源）。"""
        (self.root / "epoch_weights").mkdir()
        (self.root / "epoch_weights" / "e5.pt").write_bytes(b"x")
        (self.root / "flow").mkdir()
        (self.root / "flow" / "20260904_100000_yolo11n.json").write_text(
            "{}", encoding="utf-8")
        _write_metrics(self.root / "20260904_100000_yolo11n")
        runs = scan_runs(self.root)
        self.assertEqual([r.run_dir.name for r in runs],
                         ["20260904_100000_yolo11n"])
        self.assertEqual(runs[0].status, "completed")

    def test_load_run_none_for_empty(self):
        d = self.root / "empty_run"
        d.mkdir()
        self.assertIsNone(load_run(d))
        self.assertEqual(format_duration(None), "—")
        self.assertEqual(format_duration(3725), "1时02分05秒")
        self.assertEqual(RunRecord(run_dir=d).status_text, "未知")


class EngineRunMetaTest(unittest.TestCase):
    """引擎留档：训练完成后 metrics.json 含时间/状态/参数快照。"""

    def test_classification_train_writes_meta(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cls_dir, _ = make_classification_dataset(Path(tmp.name) / "cls", per_class=4)
        out = Path(tmp.name) / "run"
        model = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
        metrics = model.train(TrainConfig(
            task=TaskType.CLASSIFICATION, dataset_dir=cls_dir, output_dir=out,
            model_name="resnet18", epochs=1, batch_size=2, image_size=32,
            workers=0, use_pretrained=False, val_ratio=0.25))
        rec = read_json(out / "metrics.json")
        self.assertEqual(rec["status"], "completed")
        self.assertRegex(rec["start_time"], r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertRegex(rec["end_time"], r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        self.assertIsInstance(rec["duration_sec"], int)
        self.assertEqual(rec["epochs_planned"], 1)
        self.assertEqual(rec["config"]["model_name"], "resnet18")
        self.assertTrue(rec["config"]["tf32"])  # GPU 提速参数贯通快照
        self.assertEqual(metrics["status"], "completed")

    def test_batch_progress_reported(self):
        """轮内进度上报：每个次，序号在 1..批总数 之间。"""
        from xccv.engine.models import TrainReporter

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cls_dir, _ = make_classification_dataset(Path(tmp.name) / "cls", per_class=4)
        calls = []

        class R(TrainReporter):
            def on_train_progress(self, epoch, batch_index, batch_total):
                calls.append((epoch, batch_index, batch_total))

        model = create_model("resnet18", TaskType.CLASSIFICATION, 2, use_pretrained=False)
        model.train(TrainConfig(
            task=TaskType.CLASSIFICATION, dataset_dir=cls_dir,
            output_dir=Path(tmp.name) / "run", model_name="resnet18",
            epochs=2, batch_size=2, image_size=32, workers=0,
            use_pretrained=False, val_ratio=0.25), R())
        self.assertGreaterEqual(len(calls), 2)  # 每轮 ≥1 个批次
        for epoch, bi, nb in calls:
            self.assertIn(epoch, (1, 2))
            self.assertGreaterEqual(nb, 1)
            self.assertTrue(1 <= bi <= nb)


class TrainPageHistoryTest(unittest.TestCase):
    """训练可视化页：左侧历史列表 + 右侧详情渲染。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(_wait_live_threads)
        project_mod.PROJECTS_DIR = Path(self._tmp.name) / "projects"
        self.det, _ = make_labelme_dataset(Path(self._tmp.name) / "det", n_images=4)
        model_dir = Path(self._tmp.name) / "models"
        create_project("hist", str(self.det), str(model_dir), "detection")
        # 预置一条历史记录
        run = model_dir / "20260904_100000_yolo11n"
        run.mkdir(parents=True)
        _write_metrics(run)
        self.run_dir = run

        from xccv.training.training_window import ProjectWindow

        self.pw = ProjectWindow(load_project_fn("hist"))
        self.addCleanup(self.pw.close)

    def test_history_list_and_detail(self):
        self.pw._switch_detail(1)  # 进入可视化页 → 自动扫描历史
        self.assertEqual(self.pw.run_list.count(), 1)
        item = self.pw.run_list.item(0)
        rec = item.data(Qt.UserRole + 1)
        self.assertEqual(rec.status, "completed")
        self.assertIn("已完成", item.text())
        self.assertIn("2026-09-04 10:00:00", item.text())
        # 详情区渲染
        self.assertEqual(len(self.pw._history), 10)
        self.assertEqual(self.pw.progress_bar.value(), 100)
        self.assertEqual(self.pw.run_start.text(), "2026-09-04 10:00:00")
        self.assertEqual(self.pw.run_end.text(), "2026-09-04 10:05:00")
        self.assertIn("5分00秒", self.pw.run_duration.text())
        self.assertGreater(self.pw.echo_form.rowCount(), 0)
        self.assertIn("yolo11n", self.pw.echo_form.itemAt(2, 1).widget().text())

    def test_progress_partial(self):
        _write_metrics(self.run_dir, status="interrupted", epochs_planned=10,
                       history=[{"epoch": 3, "train_loss": 0.5}])
        self.pw._switch_detail(1)
        self.assertEqual(self.pw.progress_bar.value(), 30)
        self.assertIn("第 3 / 10 轮", self.pw.progress_bar.format())

    def test_loss_series_labels(self):
        """损失曲线图例：训练损失 + 验证损失（逐轮）。"""
        self.pw._switch_detail(1)
        self.pw._history = [{"epoch": 1, "train_loss": 1.0, "val_loss": 0.9}]
        keys = [k for k, _, _ in self.pw._loss_series()]
        self.assertEqual(keys, ["训练损失", "验证损失"])

    def test_loss_plot_yaxis_zero_based(self):
        """损失曲线 Y 轴固定从 0 起、上限随最大损失自适应（pyqtgraph 默认
        自适应会把下限抬到最小损失附近，零点参照消失）。"""
        self.pw._switch_detail(1)
        self.pw._history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4},
                            {"epoch": 2, "train_loss": 1.2, "val_loss": 0.8}]
        self.pw._update_plots()
        y0, y1 = self.pw.loss_plot.getPlotItem().vb.viewRange()[1]
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(y1, 1.2 * 1.08)
        m0, m1 = self.pw.metric_plot.getPlotItem().vb.viewRange()[1]
        self.assertAlmostEqual(m0, 0.0)  # 指标图仍锁 0~1
        self.assertAlmostEqual(m1, 1.0)

    def test_metric_series_labels(self):
        """指标曲线固定两条：mAP 与 mAR；其余 mAP 变体不绘制。

        - R-CNN 族：宽松口径 map_loose/mar_loose；
        - YOLO 族：map50 与 recall 充当；
        - 分类：仅 Top-1。
        """
        self.pw._switch_detail(1)
        self.pw._history = [{"epoch": 1, "map_loose": 0.9, "mar_loose": 0.7,
                             "map50": 0.8, "map50_95": 0.5}]
        self.assertEqual([k for k, _, _ in self.pw._metric_series()], ["mAP", "mAR"])
        self.pw._history = [{"epoch": 1, "map50": 0.8, "recall": 0.6}]  # YOLO 记录
        self.assertEqual([k for k, _, _ in self.pw._metric_series()], ["mAP", "mAR"])
        self.pw._history = [{"epoch": 1, "val_top1": 0.95}]  # 分类记录
        self.assertEqual([k for k, _, _ in self.pw._metric_series()], ["Top-1"])

    def test_confusion_matrix_canvas_no_accumulation(self):
        """来回切换训练记录：colorbar 不得逐次累积（整图重建）。

        旧实现仅 ax.clear()，colorbar 是独立 axes 清不掉，多次切换后
        混淆矩阵旁排出一排颜色条。
        """
        from xccv.training.charts import ConfusionMatrixCanvas

        canvas = ConfusionMatrixCanvas()
        for _ in range(4):
            canvas.update_matrix([[1, 2], [3, 4]], ["a", "b"])
        self.assertEqual(len(canvas.fig.axes), 2)  # 主轴 + 1 个 colorbar

    def test_finished_run_stays_in_history(self):
        """回归：训练结束后记录不得消失（跳过条件必须限定在训练进行中）。"""
        self.pw._switch_detail(1)
        self.pw.thread = None
        self.pw._run_dir = self.run_dir  # 模拟刚完成的训练
        self.pw._reload_history(select_dir=self.run_dir)
        self.assertEqual(self.pw.run_list.count(), 1)      # 记录仍在列表
        self.assertEqual(self.pw.run_list.currentItem().data(Qt.UserRole),
                         str(Path(self.run_dir).resolve()))  # 且保持选中（project 存的是 resolve 后路径）
        self.assertEqual(len(self.pw._history), 10)        # 详情仍可渲染

    def test_overall_progress_uses_batch_fraction(self):
        """总体进度：结合轮内批次比例（(已完成轮 + 批次比例) / 总轮次）。"""
        self.pw._switch_detail(1)
        self.pw.project.epochs = 4
        self.pw._live_history = [{"epoch": 1, "train_loss": 1.0}]
        self.pw._latest_progress = (2, 4, 10)  # 第 2 轮跑到 4/10 批

        class _FakeThread:
            def isRunning(self):
                return False

        self.pw.thread = _FakeThread()         # 非 None 即视为训练中
        self.pw._update_overall_progress()
        # ((2-1) + 4/10) / 4 轮 = 35%
        self.assertEqual(self.pw.progress_bar.value(), 35)
        self.assertIn("第 2 / 4 轮", self.pw.progress_bar.format())

    def test_delete_run_record(self):
        """删除历史记录：确认后连同文件夹一起删除。"""
        self.pw._switch_detail(1)
        self.assertTrue(self.run_dir.exists())
        self.assertTrue(self.pw.btn_delete_run.isEnabled())
        with patch("xccv.training.training_window.QMessageBox.question",
                   return_value=QMessageBox.Yes):
            self.pw._delete_current_run()
        self.assertFalse(self.run_dir.exists())            # 文件夹一并删除
        self.assertEqual(self.pw.run_list.count(), 0)      # 列表刷新
        self.assertFalse(self.pw.btn_delete_run.isEnabled())

    def test_delete_run_cancelled(self):
        self.pw._switch_detail(1)
        with patch("xccv.training.training_window.QMessageBox.question",
                   return_value=QMessageBox.No):
            self.pw._delete_current_run()
        self.assertTrue(self.run_dir.exists())             # 取消则保留
        self.assertEqual(self.pw.run_list.count(), 1)

    def test_augment_preview(self):
        """增强预览：随机抽图按 resize 拉伸，自适应显示，改尺寸不崩溃。"""
        self.pw._switch_detail(0)
        self.pw.show()
        _APP.processEvents()
        self.pw._refresh_augment_preview(new_image=True)
        self.assertFalse(self.pw.augment_preview.pixmap().isNull())
        self.assertIn("原始", self.pw.preview_info.text())
        # 修改 resize 宽高 → 预览同图拉伸到新尺寸（96×96 → 120×80，比例随设定改变）
        self.pw.augment_controls["resize"][1]["width"].setValue(120)
        self.pw.augment_controls["resize"][1]["height"].setValue(80)
        self.assertIn("拉伸 →  120×80", self.pw.preview_info.text())
        # 训练输入尺寸提示同步
        self.assertIn("120 × 80", self.pw.input_size_hint.text())
        # 关闭 resize → 显示原图 + 提示走模型默认
        self.pw.augment_controls["resize"][0].setChecked(False)
        self.assertIn("模型默认", self.pw.input_size_hint.text())

    def test_hover_installed_and_wheel_passthrough(self):
        self.pw._switch_detail(1)
        self.pw._update_plots()  # 重绘后悬停组件自动重挂
        self.assertFalse(self.pw._loss_hover.vline.isVisible())
        # 曲线图滚轮透传：不改变视图，事件被忽略（交给滚动区滚页面）
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                         Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        self.pw.metric_plot.wheelEvent(ev)
        self.assertFalse(ev.isAccepted())


class HoverTest(unittest.TestCase):
    """悬停提示：吸附最近采样点并显示三位小数数值。"""

    def test_hover_shows_value_near_mouse(self):
        plot = _PagePlot(axisItems={"bottom": _IntAxis(orientation="bottom")})
        plot.resize(400, 300)
        plot.show()
        _APP.processEvents()
        series = [("训练损失(平滑)", [0.5, 1.0, 1.5], [1.0, 0.6, 0.3])]
        hover = _PlotHover(plot, lambda: series)
        hover.install()
        vb = plot.getPlotItem().vb
        rect = vb.sceneBoundingRect()
        self.assertFalse(rect.isEmpty())  # 回归：悬停失效的根因是用错矩形对象
        pos = vb.mapViewToScene(QPointF(1.0, 0.5))  # 数据坐标 (1.0, 0.5) 对应的鼠标位置
        hover._on_move(pos)
        self.assertTrue(hover.label.isVisible())
        text = hover.label.textItem.toPlainText()
        self.assertIn("第 1 轮", text)
        self.assertIn("训练损失(平滑) = 0.600", text)
        # 锚点必须在文字顶边（y=0）：标签从视图顶部向下展开；
        # 若锚在底边（y=1），文字向上展开到 ViewBox 之外被裁剪（表现为只剩竖线）
        self.assertEqual(hover.label.anchor.y(), 0)
        # 离开绘图区 → 隐藏
        hover._on_move(rect.topLeft() - QPointF(50, 50))
        self.assertFalse(hover.label.isVisible())


class YoloEpochMetricsTest(unittest.TestCase):
    """YOLO 每轮指标键名标准化：分割/检测都产出白名单键（map50/map50_95）。"""

    @staticmethod
    def _emit(model, metrics, epoch=0):
        class T:
            pass

        t = T()
        t.metrics = metrics
        t.epoch = epoch
        model._on_epoch_end(t)

    SEG_METRICS = {
        "train/box_loss": 0.5, "train/seg_loss": 0.4,
        "metrics/precision(B)": 0.8, "metrics/recall(B)": 0.7,
        "metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.4,
        "metrics/precision(M)": 0.9, "metrics/recall(M)": 0.8,
        "metrics/mAP50(M)": 0.7, "metrics/mAP50-95(M)": 0.5,
    }

    def test_seg_key_mapping(self):
        from xccv.engine.models import TrainReporter

        m = YoloModel("yolo11n", TaskType.SEGMENTATION, 2, use_pretrained=False)
        m._reporter = TrainReporter()
        m._history = []
        self._emit(m, dict(self.SEG_METRICS))
        h = m._history[-1]
        self.assertEqual(h["map50"], 0.6)        # 对外可视化恒为 bbox 口径
        self.assertEqual(h["map50_95"], 0.4)
        self.assertEqual(h["mask_map50"], 0.7)   # 掩码指标加前缀，仅作内部参数
        self.assertEqual(h["mask_map50_95"], 0.5)
        self.assertEqual(h["train_loss"], 0.9)

    def test_det_key_mapping(self):
        from xccv.engine.models import TrainReporter

        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
        m._reporter = TrainReporter()
        m._history = []
        self._emit(m, {"metrics/mAP50(B)": 0.6, "metrics/mAP50-95(B)": 0.4})
        h = m._history[-1]
        self.assertEqual(h["map50"], 0.6)
        self.assertEqual(h["map50_95"], 0.4)


def load_project_fn(name):
    from xccv.training.project import load_project

    return load_project(name)

# == 来源模块：test_qt_slots ==
"""槽自动包装测试：Cython 编译态信号槽参数裁剪修复（xccv/qt_slots.py）。

编译态下 PyQt5 无法裁剪 clicked(bool) 等信号的多余参数，所有无参槽方法
报 TypeError。apply_auto_slot_module 用纯 Python 包装函数替换编译方法。
本测试在 dev 态（源码函数）下验证包装行为的等价性；编译态由打包冒烟的
"槽包装检查"项兜底（断言类方法为纯 Python 函数）。
"""

import os
import sys
import types
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QPushButton

from xccv.qt_slots import apply_auto_slot, apply_auto_slot_module


class _Dummy(QObject):
    clicked_log = []

    text_changed = pyqtSignal(str)

    def ping(self):
        self.clicked_log.append("ping")

    def with_arg(self, value):
        self.clicked_log.append(f"arg={value}")

    @pyqtSlot()
    def explicit(self):
        self.clicked_log.append("explicit")


class QtSlotsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        _Dummy.clicked_log = []
        apply_auto_slot(_Dummy)

    def test_zero_arg_method_wrapped_as_plain_function(self):
        """零参方法被替换为纯 Python 函数（编译态修复的判定条件）。"""
        self.assertIsInstance(_Dummy.ping, types.FunctionType)
        self.assertIsNotNone(getattr(_Dummy.ping, "__wrapped__", None))

    def test_click_executes_compiled_style_slot(self):
        """连接 clicked 并点击：包装后正常执行、无 TypeError。"""
        d = _Dummy()
        b = QPushButton()
        b.clicked.connect(d.ping)
        b.click()
        self.assertEqual(_Dummy.clicked_log, ["ping"])

    def test_extra_signal_args_trimmed(self):
        """带多余参数的信号（textChanged(str) → 无参槽）：包装层完成裁剪。"""
        d = _Dummy()
        d.text_changed.connect(d.ping)
        d.text_changed.emit("hello")
        self.assertEqual(_Dummy.clicked_log, ["ping"])

    def test_args_passthrough_for_typed_slots(self):
        """带参槽：参数透传不受包装影响。"""
        d = _Dummy()
        d.text_changed.connect(d.with_arg)
        d.text_changed.emit("abc")
        self.assertEqual(_Dummy.clicked_log, ["arg=abc"])

    def test_explicit_pyqt_slot_untouched(self):
        """已显式 @pyqtSlot 的方法不被重复包装。"""
        self.assertIsNone(getattr(_Dummy.explicit, "__wrapped__", None))
        d = _Dummy()
        b = QPushButton()
        b.clicked.connect(d.explicit)
        b.click()
        self.assertEqual(_Dummy.clicked_log, ["explicit"])

    def test_apply_module_scan_and_idempotent(self):
        """模块扫描：只处理本模块定义的 QObject 子类；重复应用幂等。"""
        import importlib

        mod = importlib.import_module("tests._slot_sample")
        apply_auto_slot_module(vars(mod))
        first = mod.Sample.do_it
        apply_auto_slot_module(vars(mod))
        self.assertIs(mod.Sample.do_it, first)
        self.assertIsInstance(first, types.FunctionType)
        d = mod.Sample()
        b = QPushButton()
        b.clicked.connect(d.do_it)
        b.click()
        self.assertEqual(mod.SAMPLE_LOG, ["done"])

    def test_non_qobject_ignored(self):
        """非 QObject 类不处理。"""

        class Plain:
            def method(self):
                return 1

        apply_auto_slot(Plain)
        self.assertIsNone(getattr(Plain.method, "__wrapped__", None))
        self.assertEqual(Plain().method(), 1)

class PlotLegendLifecycleTest(unittest.TestCase):
    """图例不锚定 ViewBox：避免窗口销毁时 geometryChanged 访问已删除对象。"""

    def test_legend_has_no_viewbox_anchor(self):
        from xccv.training.panels.common import _PagePlot, _add_legend
        plot = _PagePlot()
        try:
            legend = _add_legend(plot)
            self.assertIsNone(legend.opts["offset"])
            # GraphicsWidgetAnchor.__parent 为 None 表示未建立 geometryChanged 绑定
            self.assertIsNone(
                getattr(legend, "_GraphicsWidgetAnchor__parent", None))
            plot.resize(420, 300)
            plot.show()
            QApplication.processEvents()
            plot.place_legend()
            self.assertAlmostEqual(legend.pos().x(), 8.0, delta=2.0)
            self.assertAlmostEqual(legend.pos().y(), 8.0, delta=2.0)
        finally:
            plot.close()
            plot.deleteLater()


class PhaseStatusMessageTest(unittest.TestCase):
    """下载/阶段消息独立展示：不覆盖进度条文本（防轮次刷新与下载进度闪烁）。"""

    @staticmethod
    def _dummy():
        from PyQt5.QtWidgets import QLabel, QProgressBar
        from xccv.training.panels.history_view import HistoryViewMixin

        class _Dummy(HistoryViewMixin):
            pass

        d = _Dummy()
        d.phase_status = QLabel("")
        d.phase_status.setVisible(False)
        d.progress_bar = QProgressBar()
        d.progress_bar.setValue(37)
        d.progress_bar.setFormat("37% 第 1 / 5 轮")
        return d

    def test_status_message_does_not_touch_progress_bar(self):
        from xccv.training.panels.history_view import HistoryViewMixin
        d = self._dummy()
        HistoryViewMixin._on_status_message(
            d, "正在下载预训练权重 yolo11n.pt（下载源 1/4）…", 0.42)
        self.assertIn("42%", d.phase_status.text())
        self.assertFalse(d.phase_status.isHidden())
        self.assertEqual(d.progress_bar.value(), 37)
        self.assertEqual(d.progress_bar.format(), "37% 第 1 / 5 轮")

    def test_unknown_progress_and_clear(self):
        from xccv.training.panels.history_view import HistoryViewMixin
        d = self._dummy()
        HistoryViewMixin._on_status_message(d, "正在切换下载源…", -1.0)
        self.assertEqual(d.phase_status.text(), "正在切换下载源…")
        HistoryViewMixin._clear_phase_status(d)
        self.assertEqual(d.phase_status.text(), "")
        self.assertTrue(d.phase_status.isHidden())


if __name__ == '__main__':
    unittest.main()
