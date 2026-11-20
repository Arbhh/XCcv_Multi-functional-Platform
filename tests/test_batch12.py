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

        self.rec0.metrics["dataset_fingerprint"] = "aaa"
        self.rec1.metrics["dataset_fingerprint"] = "aaa"
        self.assertEqual(fingerprint_warnings([self.rec0, self.rec1], "aaa"), [])
        warns = fingerprint_warnings([self.rec0, self.rec1], "bbb")
        self.assertEqual(len(warns), 2)  # 与当前数据不一致 → 每条记录各提示一次
        self.assertIn("已修改", warns[0])
        self.rec1.metrics["dataset_fingerprint"] = "ccc"
        warns = fingerprint_warnings([self.rec0, self.rec1], "ccc")
        self.assertTrue(any("不同版本的数据集" in w for w in warns))  # 互不一致

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
        # 统一检出结果 schema（Q1/Q2 契约）
        import json as _json

        self.assertFalse((out_dir / "检出结果.json").exists())
        jdir = out_dir / "检出结果"
        jsaved = sorted(p.name for p in jdir.glob("*.json"))
        # P8/T8-2 起新增批量统计摘要（摘要.json）随逐图 json 一同落盘
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


if __name__ == "__main__":
    unittest.main()
