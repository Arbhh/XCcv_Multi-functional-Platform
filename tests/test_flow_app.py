"""平台测试：6 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_flow_platform
- test_v06_operator_editor
- test_batch19_ui
- test_batch12
- test_engine_purity
- test_v043_fixes
"""


# == 来源模块：test_flow_platform ==
# -*- coding: utf-8 -*-
"""YOLO 损失增益归一化 + 流程图平台页（FlowPage / 批量线程）。

背景：ultralytics tloss 分量含 box/cls/dfl 增益，小目标数据集 box_loss
按类别得分质量归一化、分母极小导致损失破百（实测 160+）；
现按增益归一化回原始损失均值口径。
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

os_off = None

import numpy as np

from xccv.engine import flow_package as fp
from xccv.engine.dataset.base import TaskType
from xccv.engine.models.base import Prediction


def _wait_scan(page, timeout_s: float = 30.0):
    """目录扫描已后台化——泵事件直到扫描线程收尾。"""
    from PyQt5.QtCore import QEventLoop, QTimer

    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout_s
    while page._scan_thread is not None and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec_()
    QApplication.processEvents()


def _fake_trainer(tloss, gains=(7.5, 0.5, 1.5)):
    """模拟 ultralytics trainer：tloss/loss_names/args/label_loss_items。"""
    from types import SimpleNamespace

    args = SimpleNamespace(box=gains[0], cls=gains[1], dfl=gains[2])
    names = ["box_loss", "cls_loss", "dfl_loss"]

    def label_loss_items(loss_items=None, prefix="train"):
        keys = [f"{prefix}/{n}" for n in names]
        return dict(zip(keys, [round(float(x), 5) for x in loss_items]))

    return SimpleNamespace(metrics={}, tloss=tloss, loss_names=names,
                           args=args, label_loss_items=label_loss_items,
                           epoch=0)


class YoloLossScaleTest(unittest.TestCase):
    def test_loss_normalized_by_gains(self):
        """tloss 分量除回增益：127.5/7.5 + 0.033/0.5 + 1.717/1.5 ≈ 18.2。"""
        from xccv.engine.models.yolo import YoloModel

        m = object.__new__(YoloModel)  # 绕开重型初始化，仅测上报逻辑
        m.task = TaskType.DETECTION
        m._history = []
        m.epoch_store = None

        sent = []

        class _Rep:
            def on_epoch(self, epoch, logs):
                sent.append(logs)

        m._reporter = _Rep()
        trainer = _fake_trainer([127.5, 0.0334, 1.717])
        m._on_epoch_end(trainer)
        self.assertAlmostEqual(sent[0]["train_loss"], 18.206, delta=0.05)

    def test_loss_fallback_without_tloss(self):
        from xccv.engine.models.yolo import YoloModel

        m = object.__new__(YoloModel)
        m.task = TaskType.DETECTION
        m._history = []
        m.epoch_store = None

        class _Rep:
            def on_epoch(self, epoch, logs):
                pass

        m._reporter = _Rep()
        trainer = _fake_trainer(None)
        trainer.tloss = None
        m._on_epoch_end(trainer)
        self.assertEqual(m._history[0]["train_loss"], 0.0)


class FlowPageTest(unittest.TestCase):
    """流程图平台页：数据树 / 流程选择 / .xvap 物化（离屏可跑，无 QtWebEngine）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        # 数据集：根下 1 张 + 子目录 1 张
        import cv2

        ds = root / "ds"
        (ds / "子目录").mkdir(parents=True)
        for p in (ds / "a.png", ds / "子目录" / "b.png"):
            cv2.imwrite(str(p), np.full((40, 60, 3), 128, np.uint8))
        # 流程：json + xvap（内嵌假模型文件）
        flow = {"last_node_id": 1, "last_link_id": 0, "nodes": [
            {"id": 1, "type": "xccv/model_detection",
             "properties": {"model_file": "m.pt"}}], "links": []}
        (root / "flows").mkdir()
        (root / "flows" / "f.json").write_text(
            json.dumps(flow, ensure_ascii=False), encoding="utf-8")
        fp.save_xvap(root / "flows" / "f.xvap", flow,
                     {"m.pt": b"\xa1" * 8})
        self.root = root

        from xccv.flow_platform import FlowPage

        self.page = FlowPage()
        self.addCleanup(self.page.deleteLater)
        self.page.data_edit.setText(str(ds))
        self.page._load_data_root()
        _wait_scan(self.page)
        self.page.model_dir_edit.setText(str(root / "flows"))
        self.page._refresh_flow_combo()

    def test_tree_and_count(self):
        self.assertEqual(self.page.count_label.text(), "图片数量: 2/2")
        img = self.page._selected_image()
        self.assertIsNone(img)  # 未点击前无选中

    def test_flow_combo_lists_json_and_xvap(self):
        names = [self.page.flow_combo.itemText(i)
                 for i in range(self.page.flow_combo.count())]
        self.assertEqual(names, ["f.json", "f.xvap"])

    def test_select_xvap_materializes_models(self):
        idx = self.page.flow_combo.findText("f.xvap")
        self.page.flow_combo.setCurrentIndex(idx)  # 触发 _on_flow_selected
        self.assertIsNotNone(self.page.current_flow_path)
        self.assertEqual(self.page.current_flow_path.suffix, ".json")
        materialized = self.page.current_flow_path.parent / "流程模型" / "m.pt"
        self.assertTrue(materialized.is_file())

    def test_filter_tree(self):
        self.page.search_edit.blockSignals(True)
        self.page.search_edit.setText("a.png")
        self.page.search_edit.blockSignals(False)
        self.page._filter_tree("a.png")
        self.assertIn("1/2", self.page.count_label.text())


class FlowBatchThreadTest(unittest.TestCase):
    """批量线程：假模型驱动全链路（JSON + 可视化导出）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_batch_exports_json_and_vis(self):
        import cv2

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        ds = root / "ds"
        ds.mkdir()
        for name in ("x.png", "y.png"):
            cv2.imwrite(str(ds / name), np.full((48, 64, 3), 200, np.uint8))

        class _FakeModel:
            task = TaskType.DETECTION
            classes = ["缺陷"]

            def predict(self, img):
                return Prediction(labels=[0], scores=[0.9],
                                  boxes=[[2.0, 2.0, 30.0, 20.0]])

        import xccv.engine.models.registry as registry
        import xccv.flow_platform as fpl

        orig = registry.load_model
        registry.load_model = lambda path: _FakeModel()
        try:
            flow = {"last_node_id": 1, "last_link_id": 0, "nodes": [], "links": []}
            out_dir = root / "out"
            th = fpl._FlowBatchThread(flow, "fake.pt",
                                      [Path("x.png"), Path("y.png")], ds, out_dir)
            th.run()  # 直接同步执行（不开线程）
        finally:
            registry.load_model = orig
        # 流程图平台页批量迁移为统一检出结果 schema（每图一个 json）
        summary = json.loads((out_dir / "批量摘要.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["total"], 2)
        self.assertEqual(len(summary["results"]), 2)
        payload = json.loads((out_dir / "检出结果" / "x.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["source"]["file"], "x.png")
        self.assertEqual(payload["detections"][0]["label"], "缺陷")
        self.assertEqual(payload["detections"][0]["bbox"], [2.0, 2.0, 30.0, 20.0])
        self.assertTrue((out_dir / "可视化" / "x.jpg").is_file())

# == 来源模块：test_v06_operator_editor ==
"""QtTest E2E：工作台「编辑算子」新建→Ctrl+S 保存→列表出现→试跑→改代码 Ctrl+S 覆盖→再试跑。

自定义算子目录用临时目录（替换 custom_ops.operators_dir），不触碰真实 platform_data。
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from xccv.engine.flow import custom_ops as co

_INVERT = ("def run(inputs, params, ctx):\n"
           "    img = inputs['image']\n"
           "    return {'image': ImageData(mat=255 - img.mat, meta=dict(img.meta))}\n")
_IDENTITY = ("def run(inputs, params, ctx):\n"
             "    return {'image': inputs['image']}\n")


def _ctrl_s(widget):
    """真实用户按 Ctrl+S 时编辑器是活动窗口（QShortcut 为窗口级快捷键）。"""
    QApplication.setActiveWindow(widget.window())
    QApplication.processEvents()
    QTest.keyClick(widget, Qt.Key_S, Qt.ControlModifier)
    QApplication.processEvents()


def _pump_until(cond, timeout=20.0):
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


class OperatorEditorE2ETest(unittest.TestCase):
    def setUp(self):
        from xccv.engine.io_utils import imwrite_unicode
        from xccv.workbench import OperatorWorkbench

        self.app = QApplication.instance() or QApplication([])
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "operators"
        real = co.operators_dir
        patcher = mock.patch.object(co, "operators_dir",
                                    side_effect=lambda root=None: real(root or self.root))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.img_path = Path(self._tmp.name) / "试跑图.png"
        rng = np.random.default_rng(1)
        self.mat = rng.integers(0, 255, size=(32, 40, 3), dtype=np.uint8)
        imwrite_unicode(self.img_path, self.mat)
        self.wb = OperatorWorkbench()
        self.wb.show()
        self.wb.run_image_edit.setText(str(self.img_path))

    def tearDown(self):
        if self.wb._editor is not None:
            self.wb._editor.close()
        _pump_until(lambda: self.wb._run_thread is None, 10)
        self.wb.close()
        self.wb.deleteLater()
        self.app.processEvents()
        co.load_custom_ops(self.root / "__empty__")   # 注销本用例注册的 custom/ 节点
        self._tmp.cleanup()

    def _run_and_get(self):
        QTest.mouseClick(self.wb.btn_run, Qt.LeftButton)
        self.assertTrue(_pump_until(lambda: self.wb._run_thread is None and
                                    self.wb._session.executor is not None and
                                    2 in self.wb._session.executor.results))
        res = self.wb._session.executor.results[2]
        self.assertTrue(res.ok, res.error)
        return res.outputs["image"].mat

    def test_new_save_list_run_overwrite(self):
        self.wb.op_list.setCurrentRow(-1)
        QTest.mouseClick(self.wb.btn_edit_op, Qt.LeftButton)
        ed = self.wb._editor
        self.assertIsNotNone(ed)
        ed.id_edit.setText("e2e_invert")
        ed.title_edit.setText("E2E 反色")
        ed.impl_edit.setPlainText(_INVERT)
        _ctrl_s(ed.impl_edit)                                      # Ctrl+S 保存（焦点在代码区）
        self.assertTrue((self.root / "e2e_invert.op.json").is_file())
        self.assertIn("已保存并注册", ed.status.text())
        cur = self.wb.op_list.currentItem()
        self.assertIsNotNone(cur)
        self.assertEqual(cur.data(Qt.UserRole), "xccv2/custom/e2e_invert")   # 列表出现且选中
        out1 = self._run_and_get()
        np.testing.assert_array_equal(out1, 255 - self.mat)           # 试跑：反色

        ed.impl_edit.setPlainText(_IDENTITY)                          # 改实现 → Ctrl+S 覆盖
        _ctrl_s(ed)
        saved = json.loads((self.root / "e2e_invert.op.json").read_text(encoding="utf-8"))
        self.assertIn("inputs['image']}", saved["script"]["code"])
        out2 = self._run_and_get()
        np.testing.assert_array_equal(out2, self.mat)                 # 新实现生效（无旧缓存）

    def test_invalid_definition_not_saved(self):
        QTest.mouseClick(self.wb.btn_edit_op, Qt.LeftButton)
        ed = self.wb._editor
        ed.id_edit.setText("Bad-Id")
        _ctrl_s(ed)
        self.assertIn("未保存", ed.status.text())
        self.assertFalse(self.root.exists() and any(self.root.glob("*.op.json")))

    def test_reload_and_import_buttons(self):
        d = co.new_op_template("composite")
        d["id"] = "dropped_in"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "dropped_in.op.json").write_text(json.dumps(d, ensure_ascii=False),
                                                       encoding="utf-8")
        QTest.mouseClick(self.wb.btn_reload_op, Qt.LeftButton)        # 刷新：热重载
        self.assertIn("xccv2/custom/dropped_in", self.wb._lib)
        pkg = co.export_xcvop("dropped_in", Path(self._tmp.name) / "包.xcvop")
        (self.root / "dropped_in.op.json").unlink()
        with mock.patch("xccv.workbench.QFileDialog.getOpenFileName",
                        return_value=(str(pkg), "")):
            QTest.mouseClick(self.wb.btn_import_op, Qt.LeftButton)    # 导入 .xcvop
        self.assertTrue((self.root / "dropped_in.op.json").is_file())
        self.assertEqual(self.wb.op_list.currentItem().data(Qt.UserRole),
                         "xccv2/custom/dropped_in")

# == 来源模块：test_batch19_ui ==
"""测试：执行器快照自动捕获、FlowSession（运行到此/重跑/快照）、
节点库 desc 下发、Q4 全平台按钮固定宽度源码审计。"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import xccv.engine.flow.legacy_adapter  # noqa: F401  过滤节点注册
import xccv.engine.flow.ops.ai_bridge  # noqa: F401  触发注册
import xccv.engine.flow.ops.locate  # noqa: F401
import xccv.engine.flow.ops.logic  # noqa: F401
import xccv.engine.flow.ops.preprocess  # noqa: F401  触发注册
import xccv.engine.flow.ops.source  # noqa: F401
from xccv.engine.flow.executor import Executor
from xccv.engine.flow.node_base import RunContext
from xccv.engine.flow.registry import list_nodes
from xccv.engine.flow.snapshot import SnapshotStore

# FlowSession 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
from xccv.modeltools import flow_session as _fs

import tempfile as _tempfile

_fs._RUN_LOG_DIR = Path(_tempfile.gettempdir()) / "xccv_test_flow_logs"


def _make_image(tmp: Path) -> str:
    p = tmp / "snap_in.png"
    cv2.imwrite(str(p), np.full((24, 24), 120, np.uint8))
    return str(p)


class ExecutorSnapshotTest(unittest.TestCase):
    def test_images_captured_per_node(self):
        """成功节点自动捕获快照：image 输出降采样副本入库。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore()
            ctx = RunContext(snapshot_store=store)
            g = {"nodes": [
                {"id": 1, "type": "image_loader", "properties": {"path": _make_image(Path(tmp))}},
                {"id": 2, "type": "gaussian_blur", "properties": {}}],
                "links": [[1, 1, 0, 2, 0]]}
            Executor(g, ctx).run()
            self.assertEqual(store.node_ids(), [1, 2])
            snap = store.get(2)
            self.assertIn("image", snap["images"])
            self.assertEqual(snap["images"]["image"].shape[:2], (24, 24))

    def test_snapshot_is_copy_not_reference(self):
        """快照契约：存副本——后续修改运行数据不影响已存快照。"""
        store = SnapshotStore()
        mat = np.zeros((4, 4), np.uint8)
        store.save(1, {"image": mat})
        mat[:] = 255
        self.assertEqual(int(store.get(1)["images"]["image"].max()), 0)


class FlowSessionTest(unittest.TestCase):
    def _graph(self, path: str):
        return {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": path}},
            {"id": 2, "type": "gaussian_blur", "properties": {"ksize": 3}}],
            "links": [[1, 1, 0, 2, 0]]}

    def test_run_and_rerun_with_logs_and_snapshot(self):
        from xccv.modeltools.flow_session import FlowSession

        with tempfile.TemporaryDirectory() as tmp:
            s = FlowSession()
            summary = s.run(self._graph(_make_image(Path(tmp))), to_node=2)
            self.assertEqual(summary["errors"], {})
            self.assertEqual(summary["ok_count"], 2)
            self.assertIn("gaussian_blur(#2)", summary["logs"])
            self.assertEqual(summary["snapshot_ids"], [1, 2])
            payload = s.snapshot_payload(2)
            self.assertIn("image", payload)
            self.assertTrue(payload["image"].startswith("data:image/png;base64,"))
            # 单节点重跑（图未变 → Executor 复用）
            again = s.rerun(2)
            self.assertEqual(again["errors"], {})

    def test_run_error_isolated_in_summary(self):
        from xccv.modeltools.flow_session import FlowSession

        g = {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": "Z:/缺失.png"}},
            {"id": 2, "type": "gaussian_blur", "properties": {}}],
            "links": [[1, 1, 0, 2, 0]]}
        s = FlowSession()
        summary = s.run(g)
        self.assertIn("1", summary["errors"])
        self.assertIn("无法读取图片", summary["errors"]["1"])
        self.assertEqual(summary["snapshot_ids"], [])

    def test_rerun_without_run_raises(self):
        from xccv.modeltools.flow_session import FlowSession

        with self.assertRaises(RuntimeError):
            FlowSession().rerun(1)


class NodeLibraryPayloadTest(unittest.TestCase):
    def test_list_nodes_has_desc_and_groups(self):
        import xccv.engine.flow.ops.ai_bridge  # noqa: F401
        import xccv.engine.flow.legacy_adapter  # noqa: F401

        nodes = list_nodes()
        by_name = {n["name"]: n for n in nodes}
        self.assertEqual(by_name["gaussian_blur"]["category"], "预处理")
        self.assertTrue(by_name["gaussian_blur"]["desc"])
        self.assertEqual(by_name["blob_analysis"]["category"], "定位")
        self.assertEqual(by_name["model_detection"]["category"], "AI")
        self.assertEqual(by_name["filter_confidence"]["category"], "检出过滤")
        # schema 自洽：params 带中文 label 与 tooltip
        spec = by_name["find_circle"]["params"]["num_rays"]
        self.assertEqual(spec["label"], "射线数")
        self.assertTrue(spec["tooltip"])


class ButtonWidthAuditTest(unittest.TestCase):
    """Q4/Q6-1 回归锁：全平台禁止对文字按钮 setFixedWidth（字号放大后
    固定宽度必裁切；面板类容器宽度不受限）。"""

    def test_no_fixed_width_on_buttons(self):
        import re

        root = Path(__file__).resolve().parents[1] / "xccv"
        pat = re.compile(r"(\w+)\.setFixedWidth\(\s*\d+\s*\)")
        bad = []
        for py in root.rglob("*.py"):
            for m in pat.finditer(py.read_text(encoding="utf-8")):
                var = m.group(1).lower()
                if any(tag in var for tag in ("btn", "button")):
                    bad.append(f"{py.name}: {m.group(0)}")
        self.assertEqual(bad, [])

# == 来源模块：test_batch12 ==
"""测试：实验对比视图、.xvap 打包前校验、流程-模型匹配、批量运行设置与渲染图。"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop
from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_labelme_dataset
from xccv.engine import detection_flow as df
from xccv.engine.dataset.base import TaskType
from xccv.engine.models.base import Prediction
from xccv.training.runs import RunRecord

_APP = QApplication.instance() or QApplication(sys.argv)


def _record(i: int, tmp: Path, **metrics) -> RunRecord:
    run = tmp / f"2026091{i}_100000_yolo11n"
    run.mkdir(exist_ok=True)
    base = {"status": "completed",
            "start_time": f"2026-09-1{i} 10:00:00", "end_time": "",
            "duration_sec": 60 * i, "epochs_planned": 10,
            "config": {"model_name": "yolo11n", "epochs": 10,
                       "batch_size": 4, "learning_rate": 0.01,
                       "industrial_opt": True, "augment_ops": {"resize": {}},
                       "augment_probs": {"resize": 1.0}, "device": "auto",
                       "val_ratio": 0.2, "patience": 10},
            "history": [{"epoch": 1, "train_loss": 2.0 - i, "val_loss": 1.5,
                         "map50": 0.5 + 0.1 * i, "recall": 0.4}],
            "final": {"map50": 0.6 + 0.1 * i},
            "classes": ["cat", "dog"],
            "confusion_matrix": [[5, 1], [0, 6]]}
    base.update(metrics)
    return RunRecord(run_dir=run, status="completed", duration_sec=60 * i,
                     epochs_done=1, epochs_planned=10, metrics=base)


class CompareHelpersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.rec0 = _record(0, self.root)
        self.rec1 = _record(1, self.root,
                            config={"model_name": "yolo11s", "epochs": 20,
                                    "batch_size": 4, "learning_rate": 0.01,
                                    "industrial_opt": True,
                                    "augment_ops": {}, "augment_probs": {},
                                    "device": "auto", "val_ratio": 0.2,
                                    "patience": 10})

    def test_param_rows_diff_highlight(self):
        from xccv.training.compare_window import param_rows

        rows = param_rows([self.rec0, self.rec1])
        by_label = {label: (texts, diff) for label, texts, diff in rows}
        self.assertTrue(by_label["训练模型"][1])
        self.assertEqual(by_label["训练模型"][0], ["yolo11n", "yolo11s"])
        self.assertFalse(by_label["批量大小"][1])  # 两条相同
        self.assertEqual(len(rows), 20)             # 全部参数键都有行（增强两行合一）
        aug = by_label["增强算子（概率）"]           # 算子与概率合并行
        self.assertEqual(aug[0], ["resize(1)", "未启用"])  # :g 最小位数显示
        self.assertTrue(aug[1])

    def test_metric_rows(self):
        from xccv.training.compare_window import metric_rows

        rows = metric_rows([self.rec0, self.rec1])
        by_label = {label: (texts, diff) for label, texts, diff in rows}
        self.assertEqual(by_label["mAP@50"][0], ["0.600", "0.700"])
        self.assertTrue(by_label["mAP@50"][1])
        self.assertEqual(by_label["训练时长"][0], ["0分00秒", "1分00秒"])

    def test_history_series(self):
        from xccv.training.compare_window import history_series

        loss, metric = history_series(self.rec0.metrics["history"])
        self.assertEqual([k for k, _, _ in loss], ["训练损失", "验证损失"])
        self.assertEqual([k for k, _, _ in metric], ["mAP", "mAR"])  # map50/recall 充当
        # 分类口径
        _, metric_cls = history_series([{"epoch": 1, "val_top1": 0.9}])
        self.assertEqual([k for k, _, _ in metric_cls], ["Top-1"])

    def test_fingerprint_warnings(self):
        from xccv.training.compare_window import fingerprint_warnings

        # v2 口径（路径+大小）：一致无提示；不一致逐条「已修改」
        self.rec0.metrics["dataset_fingerprint"] = "v2:aaa"
        self.rec1.metrics["dataset_fingerprint"] = "v2:aaa"
        self.assertEqual(
            fingerprint_warnings([self.rec0, self.rec1], "v2:aaa"), [])
        warns = fingerprint_warnings([self.rec0, self.rec1], "v2:bbb")
        self.assertEqual(len(warns), 2)  # 与当前数据不一致 → 每条记录各提示一次
        self.assertIn("已修改", warns[0])
        self.rec1.metrics["dataset_fingerprint"] = "v2:ccc"
        warns = fingerprint_warnings([self.rec0, self.rec1], "v2:ccc")
        self.assertTrue(any("不同版本的数据集" in w for w in warns))  # 互不一致

    def test_fingerprint_legacy_neutral(self):
        """旧口径（含 mtime 裸哈希）不判「已修改」，按「口径变更」提示。"""
        from xccv.training.compare_window import fingerprint_warnings

        self.rec0.metrics["dataset_fingerprint"] = "aaa"   # 旧口径裸哈希
        self.rec1.metrics["dataset_fingerprint"] = "v2:bbb"
        warns = fingerprint_warnings([self.rec0, self.rec1], "v2:bbb")
        self.assertTrue(any("口径" in w and "已修改" not in w for w in warns))
        self.assertFalse(any("已修改" in w for w in warns),
                         "旧口径不得判为「数据已修改」")

    def test_compare_window_builds(self):
        from xccv.training.compare_window import CompareWindow

        win = CompareWindow(type("P", (), {"dataset_dir": self._tmp.name})(),
                            [self.rec0, self.rec1])
        self.addCleanup(win.close)
        self.assertEqual(win.windowTitle().count("2"), 1)


class FlowPackValidateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.fake_xva = self.root / "best.pt"
        self.fake_xva.write_bytes(b"PK\x03\x04" + b"payload")

    def _graph(self, model_file=None, regions="", extra_unlinked=False,
               count_bad=False):
        nodes = [
            {"id": 1, "type": "xccv/model",
             "title": "模型", "properties": {"model_file": model_file},
             "pos": [0, 0]},
            {"id": 2, "type": "post/region_block", "title": "屏蔽区域",
             "properties": {"regions": regions}, "pos": [200, 0]},
            {"id": 3, "type": "post/confidence", "title": "置信度",
             "properties": {"min": 0.3}, "pos": [400, 0]},
        ]
        links = [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0]]
        if extra_unlinked:
            nodes.append({"id": 4, "type": "post/bbox_area", "title": "孤立节点",
                          "properties": {"min": 0, "max": 0}, "pos": [600, 0]})
        if count_bad:
            nodes.append({"id": 5, "type": "post/count_check", "title": "数量判定",
                          "properties": {"min": 5, "max": 2}, "pos": [800, 0]})
            links.append([3, 3, 0, 5, 0])
        return {"last_node_id": 5, "last_link_id": 5, "nodes": nodes,
                "links": links}

    def test_valid_flow_passes(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        self.assertEqual(validate_flow_for_pack(
            self._graph(model_file=str(self.fake_xva),
                        regions="10,10,100,100")), [])

    def test_missing_model_file(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        problems = validate_flow_for_pack(
            self._graph(model_file=str(self.root / "nope.pt")))
        self.assertTrue(any("不存在" in p for p in problems))

    def test_bad_model_header(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        bad = self.root / "bad.pt"
        bad.write_bytes(b"NOTMAGICxxxxxxxx")
        problems = validate_flow_for_pack(
            self._graph(model_file=str(bad)))
        self.assertTrue(any("不是有效的 torch 权重" in p for p in problems))

    def test_legacy_container_header(self):
        """旧版加密容器（XCVA 头）给出迁移提示而非笼统格式错误。"""
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        legacy = self.root / "legacy.pt"
        legacy.write_bytes(b"XCVA" + b"\x01" + b"payload")
        problems = validate_flow_for_pack(
            self._graph(model_file=str(legacy)))
        self.assertTrue(any("旧版加密容器" in p for p in problems))

    def test_no_model_selected(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        problems = validate_flow_for_pack(self._graph(model_file=None))
        self.assertTrue(any("尚未选择权重" in p for p in problems))

    def test_bad_regions_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        problems = validate_flow_for_pack(
            self._graph(model_file=str(self.fake_xva),
                        regions="10,10,100,100;abc,2,3,4;50,50,49,60"))
        self.assertTrue(any("不是有效的数字" in p for p in problems))
        self.assertTrue(any("宽或高为零" in p for p in problems))

    def test_unlinked_post_node(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        problems = validate_flow_for_pack(
            self._graph(model_file=str(self.fake_xva), extra_unlinked=True))
        self.assertTrue(any("未与模型节点连通" in p for p in problems))

    def test_count_check_min_gt_max(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        problems = validate_flow_for_pack(
            self._graph(model_file=str(self.fake_xva), count_bad=True))
        self.assertTrue(any("下限 5 大于上限 2" in p for p in problems))

    def test_empty_flow(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        self.assertEqual(validate_flow_for_pack({"nodes": []}),
                         ["流程为空：请先添加模型节点与后处理节点"])

    def test_chain_flow_region_check(self):
        """链式旧格式：不做模型节点检查，但区域参数仍要校验。"""
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        chain = {"nodes": [{"op": "region_block",
                            "params": {"regions": "1,2,3"}}]}
        problems = validate_flow_for_pack(chain)
        self.assertTrue(any("不是 4 个数字" in p for p in problems))
        self.assertFalse(any("模型节点" in p for p in problems))


class FlowClassMismatchTest(unittest.TestCase):
    def test_mismatch_detected(self):
        flow = {"nodes": [
            {"op": "class_filter", "params": {"keep": "scratch, dent"}},
            {"op": "confidence", "params": {"min": 0.3}},
            {"op": "class_filter", "params": {"keep": "rust"}},
        ]}
        self.assertEqual(df.flow_class_mismatch(flow, ["scratch", "blister"]),
                         ["dent", "rust"])

    def test_no_mismatch(self):
        flow = {"nodes": [{"op": "class_filter", "params": {"keep": "cat,dog"}}]}
        self.assertEqual(df.flow_class_mismatch(flow, ["cat", "dog"]), [])
        # 模型类别为空（无法判断）不误报
        self.assertEqual(df.flow_class_mismatch(flow, []), [])
        # 无 class_filter 节点
        self.assertEqual(df.flow_class_mismatch(
            {"nodes": [{"op": "confidence", "params": {}}]}, ["cat"]), [])


class RegionParamProblemsTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(df.region_param_problems("0.1,0.1,0.6,0.7"), [])
        self.assertEqual(df.region_param_problems([[10, 20, 110, 220]]), [])
        self.assertEqual(df.region_param_problems(""), [])
        self.assertEqual(df.region_param_problems(None), [])

    def test_invalid(self):
        self.assertEqual(len(df.region_param_problems("1,2,3")), 1)
        self.assertEqual(len(df.region_param_problems("a,b,c,d")), 1)
        self.assertEqual(len(df.region_param_problems("5,5,5,5")), 1)  # 零宽高
        self.assertEqual(len(df.region_param_problems([[1, 2, 3, 4, 5]])), 1)


class BatchSettingsDialogTest(unittest.TestCase):
    def _dlg(self):
        from xccv.testing.batch_dialog import BatchSettingsDialog

        return BatchSettingsDialog()

    def test_defaults_and_settings(self):
        dlg = self._dlg()
        self.assertTrue(dlg.ck_json.isChecked())
        self.assertTrue(dlg.ck_images.isChecked())
        self.assertEqual(dlg.sp_quality.value(), 95)
        dlg.ed_folder.setText("D:/imgs")
        dlg.ed_out.setText("D:/out")
        s = dlg.settings()
        self.assertEqual((s.folder, s.out_dir), ("D:/imgs", "D:/out"))
        self.assertTrue(s.save_images)

    def test_reject_invalid_folder(self):
        import xccv.testing.batch_dialog as bd

        dlg = self._dlg()
        dlg.ed_folder.setText("Z:/no_such_dir")
        with mock.patch.object(bd, "show_warning") as warn:
            dlg._accept()
            warn.assert_called_once()
        self.assertEqual(dlg.result(), 0)

    def test_reject_no_output_selected(self):
        import xccv.testing.batch_dialog as bd

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dlg = self._dlg()
        dlg.ed_folder.setText(tmp.name)
        dlg.ck_json.setChecked(False)
        dlg.ck_images.setChecked(False)
        with mock.patch.object(bd, "show_warning") as warn:
            dlg._accept()
            warn.assert_called_once()


class BatchThreadImageSaveTest(unittest.TestCase):
    """批量线程按设置保存渲染图（与 JSON 一一对应）。"""

    def test_images_saved_to_vis_dir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        make_labelme_dataset(root / "imgs", n_images=2)
        out_dir = root / "out"

        class _FakeModel:
            classes = ["cat", "dog"]

            def predict(self, img, conf=0.25, iou=0.45):
                h, w = img.shape[:2]
                return Prediction(labels=[0], scores=[0.9],
                                  boxes=[[10, 10, w // 2, h // 2]])

        files = sorted((root / "imgs").glob("*.png"))
        thread = _make_thread(_FakeModel(), files, out_dir)
        loop = QEventLoop()
        thread.finished_ok.connect(lambda _r: loop.quit())
        thread.start()
        loop.exec_()
        thread.wait(30000)
        vis = out_dir / "可视化"
        self.assertTrue(vis.is_dir())
        saved = sorted(p.name for p in vis.glob("*.jpg"))
        self.assertEqual(saved, ["img00.jpg", "img01.jpg"])
        # 每张图片一个同名 .json（无汇总文件）
        # 统一检出结果 schema 契约
        import json as _json

        self.assertFalse((out_dir / "检出结果.json").exists())
        jdir = out_dir / "检出结果"
        jsaved = sorted(p.name for p in jdir.glob("*.json"))
        # 起新增批量统计摘要（摘要.json）随逐图 json 一同落盘
        self.assertEqual(jsaved, ["img00.json", "img01.json", "摘要.json"])
        payload = _json.loads((jdir / "img00.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["task"], "detection")
        self.assertEqual(payload["source"]["file"], "img00.png")
        self.assertEqual(payload["model"]["file"], "best.pt")
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["detections"][0]["label"], "cat")
        self.assertEqual(payload["classifications"], [])


def _make_thread(model, files, out_dir, root=None):
    from xccv.testing.testing_window import _BatchThread

    return _BatchThread(model, files, 0.25, 0.45, None, ["cat", "dog"],
                        "detection", out_dir=out_dir, save_json=True,
                        save_images=True, jpg_quality=95,
                        root=root or Path(files[0]).parent, model_text="best.pt")

# == 来源模块：test_engine_purity ==
"""engine 包纯净性门禁。

`xccv/engine` 是流程引擎/模型训练推理的纯逻辑层——部署服务、无头场景
（fastapi 服务、剪枝子进程、批处理探针）都依赖它可在无 Qt 环境导入。
本测试把"0 处 PyQt/Qt 导入"从靠自觉变成回归门禁：

1. 源码扫描：engine 全部 .py 不得出现 PyQt5/PySide6/QtWidgets/QtCore/QtGui
   的 import 语句（白名单机制留给未来有理由的例外，新增先在此登记）；
2. 子进程导入复核：核心引擎子模块真实 import 后 sys.modules 不得含
   PyQt5（源码扫描防不了"间接拉起"——上游依赖链带 Qt 的回归）。

纯文本+子进程实现，不建 QApplication，offscreen/编译态均可跑。
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent.parent / "xccv" / "engine"

# 未来确有理由的例外在此登记相对 POSIX 路径（如 engine 可视化辅助模块）；
# 当前为空 = engine 全树必须零 Qt。
_PURITY_WHITELIST: set[str] = set()

# 命中即违规的导入语句（行首缩进允许：函数内延迟导入同样违规）
_BAD_IMPORT = re.compile(
    r"^\s*(?:from\s+(?:PyQt5|PySide6|QtWidgets|QtCore|QtGui)(?:\s|\.)"
    r"|import\s+(?:PyQt5|PySide6|QtWidgets|QtCore|QtGui)\b)")

# 子进程导入复核的核心子模块（覆盖 流程引擎/指标/检出流程/统一JSON/数据集）
_CORE_MODULES = [
    "xccv.engine.result_json",
    "xccv.engine.detection_flow",
    "xccv.engine.metrics.detection",
    "xccv.engine.flow.ir",
    "xccv.engine.flow.executor",
    "xccv.engine.dataset.prepare",
]


class EnginePurityTest(unittest.TestCase):
    """xccv/engine 全树禁止 Qt 导入（部署服务/无头场景复用的根基）。"""

    def test_no_qt_imports_in_engine_tree(self):
        offenders = []
        n_py = 0
        for path in sorted(_ENGINE_DIR.rglob("*.py")):
            n_py += 1
            rel = path.relative_to(_ENGINE_DIR).as_posix()
            if rel in _PURITY_WHITELIST:
                continue
            # utf-8-sig：容忍 BOM（Windows 工具写 UTF8 带 BOM 会 defeating
            # 行首锚定的正则——本测试自身就中招过，）
            text = path.read_text(encoding="utf-8-sig")
            for i, line in enumerate(text.splitlines(), start=1):
                if _BAD_IMPORT.match(line):
                    offenders.append(f"{rel}:{i}: {line.strip()}")
        self.assertGreater(n_py, 50, "engine 树异常缩小，扫描可能失效")
        self.assertEqual(offenders, [],
                         "xccv/engine 出现 Qt 导入（部署服务/无头场景会炸）：\n"
                         + "\n".join(offenders))

    def test_core_modules_import_without_qt(self):
        code = (
            "import sys;"
            + ";".join(f"import {m}" for m in _CORE_MODULES)
            + ";bad=sorted(m for m in sys.modules if m.split('.')[0] in "
              "('PyQt5','PySide6'));"
              "print('BAD:'+','.join(bad) if bad else 'CLEAN')"
        )
        proc = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"引擎子模块导入失败：{proc.stderr[-2000:]}")
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout else ""
        self.assertTrue(tail.startswith("BAD:") or tail == "CLEAN",
                        f"探针输出异常：{proc.stdout[-500:]!r}")
        self.assertEqual(tail, "CLEAN",
                         f"引擎子模块间接拉起 Qt：{tail}")

# == 来源模块：test_v043_fixes ==
"""v0.4.3 修复测试：undefined 竞态防线（桥接顺序）、路径显示文件名、
fitView、统一流程结果 JSON（剔除无效信息）、保存结果节点、
纯传统流程批量、测试平台明确提示、流程下拉实时刷新。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.engine.io_utils import imwrite_unicode  # cv2.imwrite 中文路径不可靠
from xccv.modeltools.flow_session import (
    FlowSession, _normalize_types, node_library_payload,
)
from xccv.modeltools import flow_session as _fs
from xccv.engine.flow.ir import Geometry, ImageData, MeasureResult

node_library_payload()  # 注册全部算子（含 output.result_output）

import tempfile as _tempfile

_fs._RUN_LOG_DIR = Path(_tempfile.gettempdir()) / "xccv_test_v043_logs"

# 桥接类守卫导入：全量套件中其他模块已创建 QApplication 时，
# flowgraph_web_window（含 QtWebEngineWidgets 导入）会 ImportError——
# 此时跳过桥接顺序用例（windows E2E 探针兜底）；独立运行本模块时可用
try:
    from xccv.modeltools.flowgraph_web_window import _Bridge as _RealBridge
    _BRIDGE_OK = True
except Exception:
    _RealBridge = None
    _BRIDGE_OK = False


def _app():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class NodeLibraryOrderTest(unittest.TestCase):
    """A1 修复回归：js_ready 握手必须先发节点库再冲刷缓冲事件。

    用 _Bridge + mock window 验证（offscreen 下构造 QtWebEngine 真窗口会
    原生崩溃）。
    """

    def _bridge(self):
        from unittest import mock as _mock

        if not _BRIDGE_OK:
            self.skipTest("全量套件中 QApplication 先于 QtWebEngine 存在，"
                          "桥接顺序用例由 windows E2E 探针兜底")
        _app()   # QObject 必须在 QApplication 之后创建
        win = _mock.MagicMock()
        win._pending_js = []
        return _RealBridge(win), win

    def test_js_ready_sends_library_before_pending(self):
        bridge, win = self._bridge()
        win._pending_js = ['{"event": "loaded"}']
        arrived = []
        bridge.eventArrived.connect(arrived.append)   # 缓冲事件走下行信号
        bridge.do_js_ready({})
        calls = [c.args[0].get("event") for c in win.send_json.call_args_list]
        self.assertEqual(calls[0], "node_library")   # 库先于缓冲的 loaded 下发
        self.assertEqual(json.loads(arrived[0]).get("event"), "loaded")

    def test_get_node_library_action(self):
        bridge, _win = self._bridge()
        bridge.do_get_node_library({})
        payload = _win.send_json.call_args[0][0]
        self.assertEqual(payload["event"], "node_library")
        self.assertGreaterEqual(len(payload["nodes"]), 30)


class FlowResultJsonTest(unittest.TestCase):
    """Q5/Q1：传统算法统一结果 JSON——与 AI 检出同契约、剔除无效信息。"""

    def test_schema_and_invalid_stripped(self):
        from xccv.engine.result_json import build_flow_result_json

        circle = Geometry(kind="circle", data=np.array([90.0, 80.0, 45.0]),
                          subpixel=True)
        meas = MeasureResult(values=[1.5, 2.0], passed=True, tolerance=None)
        payload = build_flow_result_json(
            source={"file": "img.png", "path": "E:/x/img.png",
                    "width": 100, "height": 90},
            detections=[{"label": "0", "label_id": 0, "score": 1.0,
                         "bbox": [10.0, 10.0, 30.0, 30.0], "bbox_area": 400}],
            geometries=[circle], measurements=[meas], flow_name="f.json")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["task"], "flow")
        self.assertEqual(payload["model"], {"file": "", "name": ""})  # 结构一致
        self.assertEqual(payload["detections"][0]["bbox_area"], 400)
        self.assertEqual(payload["geometries"][0]["kind"], "circle")
        self.assertIs(payload["geometries"][0]["subpixel"], True)  # True 保留
        self.assertEqual(payload["measurements"][0]["values"], [1.5, 2.0])
        self.assertNotIn("value", payload["measurements"][0])   # None 剔除
        self.assertNotIn("unit", payload["measurements"][0])    # 默认 px 剔除
        self.assertNotIn("tolerance", payload["measurements"][0])

    def test_nan_and_none_stripped(self):
        from xccv.engine.result_json import build_flow_result_json

        m = MeasureResult(value=float("nan"), values=None)
        payload = build_flow_result_json(source={}, measurements=[m])
        self.assertEqual(payload["measurements"], [])  # 无实质数值的条目整条剔除
        self.assertIsNone(payload["source"]["width"])  # 源结构保留（消费端零分支）

    def test_subpixel_false_stripped(self):
        from xccv.engine.result_json import build_flow_result_json

        g = Geometry(kind="point", data=np.array([[1.0, 2.0]]), subpixel=False)
        payload = build_flow_result_json(source={}, geometries=[g])
        self.assertNotIn("subpixel", payload["geometries"][0])  # False 剔除


class ResultOutputNodeTest(unittest.TestCase):
    """保存结果节点：统一 schema 落盘 + 会话定位。"""

    def test_saves_unified_json(self):
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext
        from xccv.engine.models.base import Prediction

        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((40, 50, 3), np.uint8)
            img[10:30, 20:45] = 255
            p = Path(tmp) / "工件A.png"
            imwrite_unicode(p, img)
            g = {"nodes": [
                {"id": 1, "type": "image_loader", "properties": {"path": str(p)}},
                {"id": 2, "type": "blob_analysis",
                 "properties": {"threshold_mode": "otsu", "thresh": 128,
                                "invert": False, "min_area": 10,
                                "max_area": 0, "circularity": 0.0}},
                {"id": 3, "type": "result_output", "properties": {}}],
                "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
            ctx = RunContext()
            ctx.set("flow_dir", tmp)
            ctx.set("source_name", "工件A.png")
            Executor(g, ctx).run()
            out = Path(tmp) / "流程结果" / "工件A.json"
            self.assertTrue(out.is_file())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["task"], "flow")
            self.assertEqual(data["source"]["file"], "工件A.png")
            self.assertGreaterEqual(len(data["detections"]), 1)  # blob → detections
            self.assertEqual(data["geometries"], [])  # centers 未接入 → 空数组恒在
            # 保存行为进了节点日志（结果区可见）
            self.assertTrue(any("结果已保存" in e["message"] for e in ctx.logs))

    def test_empty_input_warns(self):
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext

        g = {"nodes": [{"id": 1, "type": "result_output", "properties": {}}],
             "links": []}
        ctx = RunContext()
        ctx.set("flow_dir", tempfile.gettempdir())
        Executor(g, ctx).run()
        self.assertTrue(any("未连接任何输入" in w for w in ctx.warnings))


class XcvBatchThreadTest(unittest.TestCase):
    """Q5：纯传统流程批量运行 → 统一 JSON + 蒙版可视化 + 摘要。"""

    def test_batch_outputs_unified_json(self):
        from xccv.flow_platform import _XcvBatchThread
        from PyQt5.QtCore import QEventLoop

        _app()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "imgs"
            root.mkdir()
            for i, name in enumerate(("图A.png", "图B.png")):
                img = np.zeros((50, 60, 3), np.uint8)
                cv2.circle(img, (30, 25), 10 + i * 3, 255, -1)
                imwrite_unicode(root / name, img)
            flow = {"nodes": [
                {"id": 1, "type": "xccv2/image_loader", "properties": {}},
                {"id": 2, "type": "xccv2/blob_analysis",
                 "properties": {"threshold_mode": "otsu", "thresh": 128,
                                "invert": False, "min_area": 10,
                                "max_area": 0, "circularity": 0.0}},
                {"id": 3, "type": "xccv2/result_output", "properties": {}}],
                "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
            out_dir = Path(tmp) / "批量结果"
            th = _XcvBatchThread(flow, ["图A.png", "图B.png"], root, out_dir,
                                 flow_name="检测.json")
            # 同步调 run()：批量逻辑是测试目标，线程机制不在覆盖范围
            # （QThread+QEventLoop 的信号时序在测试进程内偶发不可达 同族）
            emitted = []
            th.finished_ok.connect(emitted.append)
            th.failed.connect(lambda m: emitted.append(f"FAIL:{m}"))
            th.run()   # 同步执行（等价 worker 全部逻辑）
            self.assertTrue(emitted and str(emitted[0]).endswith(str(out_dir)),
                            emitted)
            # 统一 JSON：每图一份、schema 一致
            ja = json.loads((out_dir / "检出结果" / "图A.json").read_text("utf-8"))
            self.assertEqual(ja["schema_version"], 1)
            self.assertEqual(ja["task"], "flow")
            self.assertEqual(ja["source"]["file"], "图A.png")
            self.assertGreaterEqual(len(ja["detections"]), 1)
            # 蒙版可视化
            self.assertTrue((out_dir / "可视化" / "图A.jpg").is_file())
            # 摘要
            summary = json.loads((out_dir / "批量摘要.json").read_text("utf-8"))
            self.assertEqual(summary["total"], 2)


class TestingPlatformHintTest(unittest.TestCase):
    """Q6：测试平台加载纯传统流程包给明确人话指引。

    testing_window 为重模块（torch + Qt 线程类），与本进程既有 QApplication
    组合会触发 sip 原生崩溃——用子进程验证。
    """

    def test_pure_xcv_xvap_rejected_with_hint(self):
        import subprocess
        import sys as _sys

        from xccv.engine import flow_package as fp

        with tempfile.TemporaryDirectory() as tmp:
            flow = {"nodes": [{"id": 1, "type": "xccv2/image_loader",
                               "properties": {}}], "links": []}
            p = Path(tmp) / "pure.xvap"
            fp.save_xvap(p, flow, {})
            code = (
                "import os, sys\n"
                "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
                f"sys.path.insert(0, r'{Path(__file__).resolve().parents[1]}')\n"
                "from xccv.testing.testing_window import TestingWindow\n"
                f"try:\n"
                f"    TestingWindow._materialize_xvap(None, r'{p}')\n"
                "    print('NO-ERROR')\n"
                "except ValueError as e:\n"
                "    print('MSG:', e)\n"
            )
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                               text=True, encoding="utf-8", timeout=180, env=env)
            self.assertIn("MSG:", r.stdout, r.stdout + r.stderr[-500:])
            self.assertIn("流程图」页", r.stdout)


class WatcherTest(unittest.TestCase):
    """Q7：流程下拉目录监视（画布保存新文件后实时刷新）。

    flow_platform 与 QtWebEngine/重模块同进程时 sip 托管顺序敏感，
    用子进程验证真实 FlowPage 行为。
    """

    def test_watcher_refreshes_combo(self):
        import subprocess
        import sys as _sys

        with tempfile.TemporaryDirectory() as tmp:
            code = (
                "import os, sys\n"
                "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
                f"sys.path.insert(0, r'{Path(__file__).resolve().parents[1]}')\n"
                "from PyQt5.QtWidgets import QApplication\n"
                "app = QApplication([])\n"
                "from xccv.flow_platform import FlowPage\n"
                f"page = FlowPage()\n"
                f"page.model_dir_edit.setText(r'{tmp}')\n"
                "page._refresh_flow_combo()\n"
                "assert page.flow_combo.count() == 0, 'empty dir'\n"
                "import json\n"
                f"open(r'{Path(tmp) / chr(26032)}', 'w')\n"   # 占位，下一行真正写流程
                "pass\n"
            )
            # 上面的占位仅触发目录变化；写入一个合法流程文件
            code = code.replace(
                f"open(r'{Path(tmp) / chr(26032)}', 'w')\npass\n",
                f"open(r'{(Path(tmp) / '新流程.json').as_posix()}', 'w', "
                "encoding='utf-8').write('{\"nodes\": [], \"links\": []}')\n"
                "import time\n"
                "deadline = time.time() + 5\n"
                "while time.time() < deadline and page.flow_combo.count() == 0:\n"
                "    app.processEvents(); time.sleep(0.05)\n"
                "print('COUNT:', page.flow_combo.count())\n"
                "print('NAME:', page.flow_combo.itemText(0))\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                               text=True, encoding="utf-8", timeout=180, env=env)
            self.assertIn("COUNT: 1", r.stdout, r.stdout + r.stderr[-500:])
            self.assertIn("NAME: 新流程.json", r.stdout)


_WINDOW_HOLDER = []

if __name__ == '__main__':
    unittest.main()
