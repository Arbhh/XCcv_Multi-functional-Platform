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

_APP = QApplication.instance() or QApplication(sys.argv)


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


if __name__ == "__main__":
    unittest.main()

