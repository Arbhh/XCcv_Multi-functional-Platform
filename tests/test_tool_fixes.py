"""平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v050_fixes
- test_v06_p_fixes
"""


# == 来源模块：test_v050_fixes ==
"""v0.5.0 修复测试（本轮全量代码审查发现）：

- P86 测试平台 _peek_xcv_flow 读取传统流程 .json 不再 NameError（json 未导入
  被 except 吞掉，流程被误当权重加载）；
- P87 .xvap 打包/解包同时识别 model_file（旧链）与 model_path（新引擎 AI 节点），
  打包校验与权重收集口径一致，AI 流程不再打包出不含权重的空包；
- P88 工作台组合链连接上游实际匹配类型的输出槽（多输出节点非首位输出）；
- P89 执行器缓存键纳入 ctx 修订号（注入检出变化后不再命中过期缓存）；
- P90 detection_to_measure 空检出 + 容差 → 判不通过；
- P91 分类 Prediction 经「保存结果」节点导出 classifications；
- P92 RCNN fit 工业损失补丁以 try/finally 兜底恢复；
- P93 项目窗口关闭被拒（训练中）时取消删除项目目录。
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.modeltools.flow_session import FlowSession, node_library_payload
from xccv.modeltools import flow_session as _fs

node_library_payload()   # 注册全部算子（全部用例的前置，幂等）

# FlowSession 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
_fs._RUN_LOG_DIR = Path(tempfile.gettempdir()) / "xccv_test_v050_logs"

_ROOT = Path(__file__).resolve().parents[1]
_APP = None


def _app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


class TestingPlatformFlowJsonTest(unittest.TestCase):
    """传统流程 .json 能被嗅探（此前 NameError 被吞 → 误导性加载失败）。"""

    def test_peek_xcv_flow_json_returns_flow(self):
        from xccv.testing.testing_window import TestingWindow

        flow = {"nodes": [{"id": 1, "type": "xccv2/image_loader",
                           "properties": {}}],
                "links": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "流程.json"
            p.write_text(json.dumps(flow), encoding="utf-8")
            out = TestingWindow._peek_xcv_flow(mock.MagicMock(), str(p))
        self.assertIsNotNone(out)
        self.assertEqual(out["nodes"][0]["type"], "xccv2/image_loader")

    def test_peek_non_flow_json_returns_none(self):
        """链式检出流程 JSON：给提示并返回 None（不抛异常）。"""
        from xccv.testing.testing_window import TestingWindow

        chain = {"nodes": [{"op": "confidence", "params": {"min": 0.5}}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "检出流程.json"
            p.write_text(json.dumps(chain), encoding="utf-8")
            receiver = mock.MagicMock()
            with mock.patch("xccv.testing.testing_window.QMessageBox.information") \
                    as box:
                out = TestingWindow._peek_xcv_flow(receiver, str(p))
        self.assertIsNone(out)
        self.assertTrue(box.called)


class XvapModelRefTest(unittest.TestCase):
    """模型引用键 model_file / model_path 双兼容（预留 MODEL_PARAM_KEYS）。"""

    @staticmethod
    def _ai_flow(weight: Path) -> dict:
        return {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {}},
            {"id": 2, "type": "xccv2/model_detection",
             "properties": {"model_path": str(weight), "conf": 0.25,
                            "iou": 0.45, "device": "auto"}}],
            "links": [[1, 1, 0, 2, 0]]}

    def test_collect_model_files_supports_model_path(self):
        from xccv.engine import flow_package as fp

        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "best.pt"
            w.write_bytes(b"PK\x03\x04fake-payload")
            models = fp.collect_model_files(self._ai_flow(w))
        self.assertEqual(models, {"best.pt": b"PK\x03\x04fake-payload"})

    def test_pack_roundtrip_rewrites_model_path(self):
        from xccv.engine import flow_package as fp

        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "best.pt"
            w.write_bytes(b"PK\x03\x04fake-payload")
            flow = self._ai_flow(w)
            pkg = Path(tmp) / "flow.xvap"
            fp.save_xvap(pkg, flow, fp.collect_model_files(flow))
            out_flow, out_dir = fp.unpack_to_dir(pkg)
            self.assertTrue((out_dir / "best.pt").is_file())
            node = next(n for n in out_flow["nodes"]
                        if n["type"] == "xccv2/model_detection")
            self.assertEqual(node["properties"]["model_path"],
                             str(out_dir / "best.pt"))

    def test_ai_flow_pack_validate_and_collect_agree(self):
        """打包校验通过的 AI 流程必须同时收集到权重（口径配对）。"""
        from xccv.engine import flow_package as fp
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp) / "best.pt"
            w.write_bytes(b"PK\x03\x04fake-payload")
            flow = self._ai_flow(w)
            self.assertEqual(validate_flow_for_pack(flow), [])
            self.assertTrue(fp.collect_model_files(flow))


class WorkbenchChainPortTest(unittest.TestCase):
    """组合链连上游实际匹配类型的输出槽。"""

    def test_chain_uses_matched_out_slot(self):
        _app()
        from PyQt5.QtCore import Qt

        from xccv.workbench import OperatorWorkbench

        w = OperatorWorkbench()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.png"
            img = np.zeros((40, 50, 3), np.uint8)
            cv2.circle(img, (25, 20), 8, 255, -1)
            cv2.imwrite(str(p), img)
            w.run_image_edit.setText(str(p))

            def add(name):
                for i in range(w.op_list.count()):
                    if w.op_list.item(i).data(Qt.UserRole) == f"xccv2/{name}":
                        w.op_list.setCurrentRow(i)
                        w._chain_add()
                        return True
                return False

            self.assertTrue(add("blob_analysis"))
            self.assertTrue(add("detection_to_measure"))
            self.assertTrue(add("compare_value"))
            graph, error = w._build_chain_graph(str(p))
            self.assertIsNone(error)
            # 末段连线：detection_to_measure(节点3) 的数值输出（2 号槽 count，
            # 端口顺序图像在上：image/result/count）→ compare_value 的 number 输入
            last = graph["links"][-1]
            self.assertEqual((last[1], last[2], last[3]), (3, 2, 4))
            w._run_chain()
            # 组合链已后台化——泵事件直到线程收尾再断言
            from PyQt5.QtCore import QEventLoop, QTimer

            from PyQt5.QtWidgets import QApplication

            deadline = time.monotonic() + 30
            while getattr(w, "_run_thread", None) is not None \
                    and time.monotonic() < deadline:
                loop = QEventLoop()
                QTimer.singleShot(20, loop.quit)
                loop.exec_()
            QApplication.processEvents()
            self.assertIsNone(w._run_thread)
            self.assertIn("4/4 节点成功", w.chain_out.toPlainText())


class ExecutorCacheRevisionTest(unittest.TestCase):
    """缓存键纳入 ctx 修订号，注入状态变化后不返回过期结果。"""

    def test_detection_source_refresh_after_inject(self):
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext
        from xccv.engine.models.base import Prediction

        g = {"nodes": [
            {"id": 1, "type": "detection_source", "properties": {}},
            {"id": 2, "type": "filter_confidence", "properties": {"min": 0.5}}],
            "links": [[1, 1, 0, 2, 1]]}   # result 输入槽 1
        ctx = RunContext()
        ex = Executor(g, ctx)
        ctx.set("inject_detection",
                Prediction(labels=[0], scores=[0.9], boxes=[[0, 0, 10, 10]]))
        ex.run()
        self.assertAlmostEqual(ex.results[2].outputs["result"].scores[0], 0.9)
        ctx.set("inject_detection",
                Prediction(labels=[0], scores=[0.2], boxes=[[0, 0, 10, 10]]))
        ex.run()
        # 断言检出源节点（1）的输出刷新；0.2 低于 min=0.5 会在节点 2 被过滤为空
        self.assertAlmostEqual(ex.results[1].outputs["result"].scores[0], 0.2)
        self.assertEqual(ex.results[2].outputs["result"].boxes, [])

    def test_revision_and_cache_clear_interface(self):
        from xccv.engine.flow.node_base import RunContext

        ctx = RunContext()
        r0 = ctx.revision
        ctx.set("k", 1)
        self.assertEqual(ctx.revision, r0 + 1)
        ctx.cache_set("ck", object())
        ctx.cache_clear()
        self.assertIsNone(ctx.cache_get("ck"))


class DetectionMeasureEmptyTest(unittest.TestCase):
    """空检出不允许"全称量化恒真"判通过。"""

    def test_empty_detection_fails_with_tolerance(self):
        from xccv.engine.flow.node_base import RunContext
        from xccv.engine.flow.registry import get_node_class
        from xccv.engine.models.base import Prediction

        ctx = RunContext()
        node = get_node_class("detection_to_measure")()
        out = node.run({"result": Prediction()},
                       {"metric": "area", "tol_min": 10.0, "tol_max": 100.0}, ctx)
        self.assertIs(out["result"].passed, False)
        self.assertTrue(any("未检出" in w for w in ctx.warnings))

    def test_empty_detection_no_tolerance_keeps_none(self):
        from xccv.engine.flow.node_base import RunContext
        from xccv.engine.flow.registry import get_node_class
        from xccv.engine.models.base import Prediction

        node = get_node_class("detection_to_measure")()
        out = node.run({"result": Prediction()},
                       {"metric": "area", "tol_min": 0.0, "tol_max": 0.0},
                       RunContext())
        self.assertIs(out["result"].passed, None)


class FlowClassificationsExportTest(unittest.TestCase):
    """分类 Prediction（probs 无框）经流程出口导出 classifications。"""

    def test_result_output_exports_classifications(self):
        import xccv.engine.flow.ops.output  # noqa: F401
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext
        from xccv.engine.models.base import Prediction

        with tempfile.TemporaryDirectory() as tmp:
            ctx = RunContext()
            ctx.set("flow_dir", tmp)
            ctx.set("source_name", "图C.png")
            ctx.set("model_classes", ["OK", "NG"])
            g = {"nodes": [
                {"id": 1, "type": "detection_source", "properties": {}},
                {"id": 2, "type": "result_output", "properties": {}}],
                "links": [[1, 1, 0, 2, 1]]}   # result 输入槽 1
            ctx.set("inject_detection", Prediction(probs=[0.1, 0.9]))
            Executor(g, ctx).run()
            payload = json.loads(
                (Path(tmp) / "流程结果" / "图C.json").read_text("utf-8"))
        self.assertEqual(payload["classifications"][0]["label"], "NG")
        self.assertAlmostEqual(payload["classifications"][0]["score"], 0.9, places=3)
        self.assertEqual(payload["detections"], [])

    def test_detection_rows_include_mask_area(self):
        from xccv.engine.models.base import Prediction
        from xccv.engine.result_json import detection_rows

        m = np.zeros((8, 8), bool)
        m[2:6, 2:6] = True
        rows = detection_rows(Prediction(labels=[0], scores=[0.5],
                                         boxes=[[0, 0, 4, 4]], masks=[m]), ["甲"])
        self.assertEqual(rows[0]["label"], "甲")
        self.assertEqual(rows[0]["mask_area"], 16)


class RcnnPatchRestoreTest(unittest.TestCase):
    """fit 的工业损失补丁必须 finally 兜底恢复（异常路径防全局污染）。"""

    def test_fit_has_finally_restore(self):
        import ast

        tree = ast.parse(
            (_ROOT / "xccv" / "engine" / "models" / "rcnn.py").read_text("utf-8"))
        fit = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "fit")

        def _calls(node):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    yield sub.func.id

        guarded = any(
            isinstance(n, ast.Try) and n.finalbody
            and "_restore_rcnn_loss_patches" in list(_calls(n.finalbody[0]))
            for n in ast.walk(fit))
        self.assertTrue(guarded, "fit 内缺少 finally 恢复损失补丁的兜底")

    def test_restore_idempotent(self):
        import torchvision.models.detection.roi_heads as rh

        from xccv.engine.models import rcnn

        orig = rh.fastrcnn_loss
        rcnn._apply_rcnn_loss_patches(with_dice=False)
        self.assertIsNot(rh.fastrcnn_loss, orig)
        rcnn._restore_rcnn_loss_patches()
        rcnn._restore_rcnn_loss_patches()   # 幂等（重复恢复不报错）
        self.assertIs(rh.fastrcnn_loss, orig)


class ProjectDeleteGuardTest(unittest.TestCase):
    """项目窗口关闭被拒（训练中）时取消删除项目目录。"""

    def test_delete_cancelled_when_close_rejected(self):
        from PyQt5.QtWidgets import QMessageBox

        from xccv.training.training_window import TrainingWindow

        win = mock.MagicMock()
        win.close.return_value = False
        host = mock.MagicMock()
        host._project_windows = {"甲项目": win}
        with mock.patch("xccv.training.training_window.QMessageBox.question",
                        return_value=QMessageBox.Yes), \
             mock.patch("xccv.training.training_window.delete_project") as dp:
            TrainingWindow._confirm_delete(host, "甲项目")
        dp.assert_not_called()
        self.assertIn("甲项目", host._project_windows)

    def test_delete_proceeds_when_close_ok(self):
        from PyQt5.QtWidgets import QMessageBox

        from xccv.training.training_window import TrainingWindow

        win = mock.MagicMock()
        win.close.return_value = True
        host = mock.MagicMock()
        host._project_windows = {"甲项目": win}
        with mock.patch("xccv.training.training_window.QMessageBox.question",
                        return_value=QMessageBox.Yes), \
             mock.patch("xccv.training.training_window.delete_project") as dp:
            TrainingWindow._confirm_delete(host, "甲项目")
        dp.assert_called_once_with("甲项目")

# == 来源模块：test_v06_p_fixes ==
# -*- coding: utf-8 -*-
"""V0.6.0 安装实测缺陷修复回归。

- P-01 剪枝 worker done 行崩溃：pretrain_prune 返回的 result 含 "ckpt"
  （整个网络对象），worker `_emit(event="done")` 的 json.dumps 崩溃 →
  子进程 exit 1、UI 报失败（而产物其实已落盘）。修复=emit 前剥离
  ckpt + `_emit` 对不可序列化对象降级兜底。
  回归：pre 模式端到端（真实子进程，分类小数据集）必须 returncode=0、
  done.result 不含 ckpt、dst 落盘——修复前该用例必现复现缺陷。
- P-02 加速导出签名：modeltools `_export_engine` wrapper 缺 progress
  形参而调用处已传 → 必现 TypeError。回归=签名含 progress 且透传
  到引擎层 export_engine。
"""
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

os_guard = __import__("os")
os_guard.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]

from tests.synthetic import make_classification_dataset  # noqa: E402


class EmitFallbackTest(unittest.TestCase):
    """P-01a：_emit 遇不可序列化对象降级而不是崩溃。"""

    def test_emit_tensor_result_degrades(self):
        import torch

        from xccv.engine.prune_worker import _emit

        buf = io.StringIO()
        with redirect_stdout(buf):
            _emit(event="done", result={"status": "applied",
                                        "ckpt": torch.zeros(2)})
        line = buf.getvalue().strip()
        ev = json.loads(line)  # 不抛 = 协议行可用
        self.assertEqual(ev["event"], "done")
        self.assertTrue(ev.get("stripped"))  # 降级标记

    def test_emit_plain_result_untouched(self):
        from xccv.engine.prune_worker import _emit

        buf = io.StringIO()
        with redirect_stdout(buf):
            _emit(event="done", result={"status": "applied", "params": 123})
        ev = json.loads(buf.getvalue().strip())
        self.assertNotIn("stripped", ev)
        self.assertEqual(ev["result"]["params"], 123)


class PreModeEndToEndTest(unittest.TestCase):
    """P-01b：pre 模式端到端（真实子进程）——done 行与产物落盘。"""

    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp(prefix="pfix_pre_"))
        cls._root = root
        cls_dir, _ = make_classification_dataset(root / "cls", per_class=4)
        cls._dataset = cls_dir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls._root), True)

    def test_pre_prune_done_line_and_artifact(self):
        dst = self._root / "out" / "resnet18_pruned.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, "-m", "xccv.engine.prune_worker",
             "--mode", "pre", "--dataset-dir", str(self._dataset),
             "--task", "classification", "--model-name", "resnet18",
             "--width", "32", "--height", "32", "--ratio", "0.3",
             "--dst", str(dst)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300)
        self.assertEqual(r.returncode, 0,
                         "stdout=%s stderr=%s" % (r.stdout[-1500:], r.stderr[-1500:]))
        evs = [json.loads(l) for l in r.stdout.splitlines()
               if l.strip().startswith("{")]
        dones = [e for e in evs if e["event"] == "done"]
        self.assertEqual(len(dones), 1)
        res = dones[0]["result"]
        self.assertEqual(res["status"], "applied")
        self.assertNotIn("ckpt", res)  # P-01：网络对象不得走 stdout
        self.assertTrue(dst.is_file())  # 产物落盘


class ExportEngineWrapperTest(unittest.TestCase):
    """P-02：加速导出 wrapper 接受并透传 progress。"""

    def test_signature_accepts_progress(self):
        import inspect

        from xccv.modeltools.modeltools_window import _export_engine

        self.assertIn("progress", inspect.signature(_export_engine).parameters)

    def test_progress_forwarded_to_engine(self):
        import xccv.modeltools.modeltools_window as mw

        calls = {}
        with patch("xccv.engine.model_export.export_engine",
                   side_effect=lambda *a, **k: calls.update(k) or {}):
            cb = lambda f, t: None  # noqa: E731
            mw._export_engine("s.pt", "d.engine", imgsz=640, half=True,
                              progress=cb)
        self.assertIs(calls.get("progress"), cb)
        self.assertEqual(calls.get("imgsz"), 640)
        self.assertEqual(calls.get("half"), True)

if __name__ == '__main__':
    unittest.main()
