"""测试：5 官方模板可加载可执行、传统平台算子工作台。"""

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

import cv2
import numpy as np

from xccv.modeltools.flow_session import (
    FlowSession, _normalize_types, list_templates, node_library_payload,
)

node_library_payload()  # 注册全部算子（模板图与工作台都依赖注册表）

# FlowSession 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
import tempfile as _tempfile

from xccv.modeltools import flow_session as _fs
from pathlib import Path as _Path

_fs._RUN_LOG_DIR = _Path(_tempfile.gettempdir()) / "xccv_test_flow_logs"

_APP = None


def _app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


def _scene(tmp: Path) -> str:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 35, size=(160, 200), dtype=np.uint8)
    yy, xx = np.mgrid[:160, :200]
    disc = (np.clip(25.5 - np.hypot(xx - 140.0, yy - 50.0), 0, 1) * 255)
    img = np.maximum(img, disc.astype(np.uint8))
    cv2.line(img, (5, 120), (195, 100), 230, 2, cv2.LINE_AA)
    p = tmp / "scene.png"
    cv2.imwrite(str(p), img)
    return str(p)


class TemplatesTest(unittest.TestCase):
    def test_five_templates_with_meta(self):
        ts = list_templates()
        ids = {t["template"] for t in ts}
        self.assertLessEqual({"tmpl_ai_filter", "tmpl_blob_prefilter", "tmpl_circle_locate",
                              "tmpl_line_level", "tmpl_template_match",
                              "tmpl_cascade_det_seg", "tmpl_crop_export"}, ids)  # V0.6 +2
        titles = {t["title"] for t in ts}
        self.assertIn("AI 检测 + 过滤 + 数量判定", titles)
        for t in ts:
            self.assertTrue(t["description"])
            self.assertTrue(t["template"])

    def test_light_templates_run_end_to_end(self):
        """全部无 AI 模板在合成图上全链跑通并出快照（AI 模板由冒烟实测覆盖）。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scene = _scene(tmp)
            for t in list_templates():
                if any(n["type"].split("/")[-1].startswith("model_")
                       for n in t["flow"]["nodes"]):
                    continue   # 含 AI 模型节点的模板由冒烟实测覆盖（无模型文件可跑）
                flow = _normalize_types(t["flow"])
                for n in flow["nodes"]:
                    if n["type"].endswith("image_loader"):
                        n["properties"]["path"] = scene
                    # 落盘类算子指向临时目录（默认会写到 cwd 的「流程结果/」）
                    if n["type"].endswith(("image_sink", "result_output")):
                        n["properties"]["out_dir"] = str(tmp / "out")
                    if n["type"].endswith("csv_sink"):
                        n["properties"]["out_path"] = str(tmp / "out" / "r.csv")
                    if n["type"].endswith(("crop_by_detection", "grid_tile")):
                        n["properties"]["save_dir"] = str(tmp / "crops")
                    if n["type"].endswith("template_match"):
                        import xccv.engine.flow.ops.locate  # noqa: F401
                        from xccv.engine.flow.registry import get_node_class

                        piece = cv2.imread(scene, cv2.IMREAD_GRAYSCALE)[40:80, 20:76]
                        p = get_node_class("template_match")().record(
                            np.ascontiguousarray(piece),
                            (0, 0, piece.shape[1], piece.shape[0]), tmp / "tmpl")
                        n["properties"]["template_path"] = p
                        n["properties"]["threshold"] = 0.6
                        n["properties"]["angle_range"] = 0.0
                s = FlowSession()
                summary = s.run(flow)
                self.assertEqual(summary["errors"], {},
                                 f"{t['title']}: {summary['errors']}")
                self.assertGreater(len(s.ctx.snapshot_store.node_ids()), 0,
                                   f"{t['title']} 未产出快照")

    def test_ai_template_structure_loads(self):
        """AI 模板图结构与注册表自洽。"""
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext

        t = next(t for t in list_templates()
                 if t["template"] == "tmpl_ai_filter")
        flow = _normalize_types(t["flow"])
        Executor(flow, RunContext()).validate()  # 不抛即结构/类型/无环


class WorkbenchTest(unittest.TestCase):
    def setUp(self):
        _app()
        from PyQt5.QtCore import Qt

        self._user_role = Qt.UserRole
        from xccv.workbench import OperatorWorkbench

        self.w = OperatorWorkbench()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _select_op(self, name: str):
        for i in range(self.w.op_list.count()):
            if self.w.op_list.item(i).data(self._user_role) == f"xccv2/{name}":
                self.w.op_list.setCurrentRow(i)
                return True
        return False

    def _wait_run_done(self, timeout_s: float = 30.0):
        """T2-2：试跑已后台化——泵事件直到工作线程收尾（结果区出现汇总）。"""
        from PyQt5.QtCore import QEventLoop, QTimer

        deadline = time.monotonic() + timeout_s
        while self.w._run_thread is not None and time.monotonic() < deadline:
            loop = QEventLoop()
            QTimer.singleShot(20, loop.quit)
            loop.exec_()
        _app().processEvents()
        self.assertIsNone(self.w._run_thread, "试跑工作线程未在超时内结束")

    def test_library_and_detail(self):
        _app()
        self.assertGreaterEqual(self.w.op_list.count(), 25)
        self.assertTrue(self._select_op("gaussian_blur"))
        self.assertIn("高斯模糊", self.w.detail_box.toPlainText())
        self.assertIn("核数", self.w.detail_box.toPlainText())

    def test_param_form_roundtrip(self):
        self.assertTrue(self._select_op("threshold"))
        form = self.w._param_edits
        self.assertIn("mode", form)
        spec, edit = form["mode"]
        edit.setCurrentText("otsu")
        params = self.w._collect_params(
            self.w._lib["xccv2/threshold"])
        self.assertEqual(params["mode"], "otsu")

    def test_single_run_with_logs(self):
        """单算子试跑：真实执行 + 结果区含节点级日志（日志捕捉可见）。"""
        p = Path(self._tmp.name) / "in.png"
        cv2.imwrite(str(p), np.full((20, 20), 90, np.uint8))
        self.w.run_image_edit.setText(str(p))
        self.assertTrue(self._select_op("gaussian_blur"))
        self.w._run_single()
        self._wait_run_done()
        text = self.w.single_out.toPlainText()
        self.assertIn("2/2 节点成功", text)
        self.assertIn("gaussian_blur(#2)", text)

    def test_chain_run_blob_to_count_check(self):
        """组合链：blob_analysis → filter_count_check（检测端口自动匹配）。"""
        rng = np.random.default_rng(1)
        img = np.zeros((80, 100), np.uint8)
        cv2.circle(img, (40, 40), 12, 255, -1)
        p = Path(self._tmp.name) / "chain.png"
        cv2.imwrite(str(p), img)
        self.w.run_image_edit.setText(str(p))
        self.assertTrue(self._select_op("blob_analysis"))
        self.w._chain_add()
        self.assertTrue(self._select_op("filter_count_check"))
        self.w._chain_add()
        self.w._run_chain()
        self._wait_run_done()
        text = self.w.chain_out.toPlainText()
        self.assertIn("3/3 节点成功", text)
        self.assertIn("[count]", text)   # 数量判定：count 数值输出
        self.assertIn("[result] 检出 1 个目标", text)

    def test_chain_type_mismatch_rejected(self):
        """类型不匹配的组合在搭链时报错（不允许生成跑不过的图）。"""
        p = Path(self._tmp.name) / "x.png"
        cv2.imwrite(str(p), np.zeros((10, 10), np.uint8))
        self.w.run_image_edit.setText(str(p))
        self.assertTrue(self._select_op("gaussian_blur"))
        self.w._chain_add()
        self.assertTrue(self._select_op("filter_confidence"))
        self.w._chain_add()   # filter_confidence 需要 detection，blur 只有 image
        graph, error = self.w._build_chain_graph(str(p))
        self.assertIsNone(graph)
        self.assertIn("无法连接", error)

    def test_single_run_requires_image(self):
        self.assertTrue(self._select_op("gaussian_blur"))
        self.w.run_image_edit.setText("")
        from unittest import mock

        with mock.patch("xccv.workbench.QMessageBox.information") as m:
            self.w._run_single()
            self.assertTrue(m.called)


if __name__ == "__main__":
    unittest.main()
