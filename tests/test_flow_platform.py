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
    """T2-3：目录扫描已后台化——泵事件直到扫描线程收尾。"""
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


if __name__ == "__main__":
    unittest.main()
