# -*- coding: utf-8 -*-
"""主界面流程图页（FlowPage，新版流程图界面）行为锁：载入/新建/打包/解包/画布。

背景（V0.7.5 规范化）：旧版独立编辑器 FlowGraphMainWindow 弃用归档，
本文件从旧窗口行为锁整体重定向到新版 FlowPage；跳转链路（复盘页 →
主界面流程图页）以 open_flow_file 公共 API + home_jump 接收为口径。
"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication as _QApp  # noqa: E402
from PyQt5.QtWidgets import QMainWindow  # noqa: E402

from xccv.engine import flow_package as fp  # noqa: E402
from xccv.flow_platform import FlowPage  # noqa: E402

_APP = _QApp.instance() or _QApp([])
_FP = "xccv.flow_platform"
_FS = "xccv.modeltools.flow_shared"


def _graph(model_file: str = "") -> dict:
    return {"last_node_id": 2, "last_link_id": 1, "nodes": [
        {"id": 1, "type": "xccv/model_detection",
         "properties": {"model_file": model_file, "threshold": 0.25}},
        {"id": 2, "type": "xccv/output", "properties": {}}],
        "links": [[1, 1, 0, 2, 0]]}


class FlowPageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.td = Path(self._tmp.name)
        self.win = QMainWindow()          # FlowPage 宿主（状态栏语义同主界面）
        self.page = FlowPage()
        self.win.setCentralWidget(self.page)
        self.win.show()

    def tearDown(self):
        self.win.close()
        self._tmp.cleanup()

    def test_open_flow_file_loads_and_selects(self):
        """复盘页跳转入口：载入 json → 下拉选中 → current_flow_path 落位。"""
        j = self.td / "g.json"
        j.write_text(json.dumps(_graph(), ensure_ascii=False), encoding="utf-8")
        self.assertTrue(self.page.open_flow_file(str(j)))
        self.assertEqual(self.page.current_flow_path, j)
        idx = self.page.flow_combo.currentIndex()
        self.assertEqual(self.page.flow_combo.itemData(idx), str(j))
        self.assertEqual(self.page.model_dir_edit.text(), str(self.td))

    def test_open_flow_file_missing_rejected(self):
        self.assertFalse(self.page.open_flow_file(str(self.td / "no.json")))
        self.assertIsNone(self.page.current_flow_path)

    def test_open_xvap_materializes_models_and_json(self):
        (self.td / "best.pt").write_bytes(b"\xa1" * 32)
        xvap = self.td / "q.xvap"
        fp.save_xvap(xvap, _graph("best.pt"), {"best.pt": b"\xa1" * 32})
        self.assertTrue(self.page.open_flow_file(str(xvap)))
        json_path = xvap.with_suffix(".json")
        self.assertTrue(json_path.is_file())
        self.assertTrue((self.td / "流程模型" / "best.pt").is_file())
        self.assertEqual(self.page.current_flow_path, json_path)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        mf = loaded["nodes"][0]["properties"]["model_file"]
        self.assertTrue(Path(mf).is_absolute() and Path(mf).is_file())

    def test_create_flow_writes_skeleton_then_edits(self):
        target = self.td / "f.json"
        with mock.patch(f"{_FP}.QFileDialog.getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), ""))), \
             mock.patch.object(FlowPage, "_edit_flow") as ed:
            self.page._create_flow()
        self.assertTrue(target.is_file())
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["nodes"], [])
        self.assertEqual(self.page.current_flow_path, target)
        ed.assert_called_once()  # 创建后直接进入画布

    def test_pack_flow_produces_xvap(self):
        model = self.td / "best.pt"
        model.write_bytes(b"PK\x03\x04" + b"\xa1" * 32)  # 打包校验要求 torch 权重头
        j = self.td / "g.json"
        j.write_text(json.dumps(_graph(str(model)), ensure_ascii=False),
                     encoding="utf-8")
        self.page.open_flow_file(str(j))
        dst = self.td / "p.xvap"
        with mock.patch(f"{_FP}.QFileDialog.getSaveFileName",
                        staticmethod(lambda *a, **k: (str(dst), ""))), \
             mock.patch(f"{_FP}.QMessageBox.information"):
            self.page._pack_flow()
        self.assertTrue(dst.is_file())
        flow2, models = fp.load_xvap(dst)
        self.assertEqual(len(flow2["nodes"]), 2)
        self.assertIn("best.pt", models)

    def test_pack_blocked_on_bad_model_header(self):
        """打包前校验（）：模型头不可读 → 列出问题、不产出坏包。"""
        model = self.td / "broken.pt"
        model.write_bytes(b"\xa1" * 32)  # 头信息不可读
        j = self.td / "g.json"
        j.write_text(json.dumps(_graph(str(model)), ensure_ascii=False),
                     encoding="utf-8")
        self.page.open_flow_file(str(j))
        dst = self.td / "x.xvap"
        with mock.patch(f"{_FP}.QFileDialog.getSaveFileName",
                        staticmethod(lambda *a, **k: (str(dst), ""))), \
             mock.patch(f"{_FS}._show_pack_problems") as show:
            self.page._pack_flow()
        show.assert_called_once()
        self.assertFalse(dst.exists())  # 坏包不得产出

    def test_unpack_flow_restores_json_and_models(self):
        (self.td / "best.pt").write_bytes(b"\xa1" * 16)
        xvap = self.td / "u.xvap"
        fp.save_xvap(xvap, _graph("best.pt"), {"best.pt": b"\xa1" * 16})
        with mock.patch(f"{_FP}.QFileDialog.getOpenFileName",
                        staticmethod(lambda *a, **k: (str(xvap), ""))):
            self.page._unpack_flow()
        self.assertTrue((self.td / "u.json").is_file())
        self.assertTrue((self.td / "流程模型" / "best.pt").is_file())

    def _patch_canvas(self):
        """把 web 画布模块替换为桩：测试进程不得导入真实 QtWebEngine
        （导入后再退出会段错误，见 test_flowgraph 的环境说明）。"""
        fake = types.SimpleNamespace(FlowGraphWebWindow=mock.MagicMock())
        return mock.patch.dict(sys.modules,
                               {"xccv.modeltools.flowgraph_web_window": fake}), fake

    def test_edit_flow_pushes_payload_to_canvas(self):
        j = self.td / "g.json"
        j.write_text(json.dumps(_graph(), ensure_ascii=False), encoding="utf-8")
        self.page.open_flow_file(str(j))
        patcher, fake = self._patch_canvas()
        with patcher:
            self.page._edit_flow()
        instance = fake.FlowGraphWebWindow.return_value
        self.assertIs(self.page._canvas, instance)
        instance.load_flow_payload.assert_called_once()
        args = instance.load_flow_payload.call_args.args
        self.assertEqual(args[0]["nodes"][0]["type"], "xccv/model_detection")
        self.assertEqual(args[1], "g.json")
        self.assertEqual(instance.current_path, j)

    def test_edit_without_flow_prompts(self):
        patcher, fake = self._patch_canvas()
        with patcher:
            self.page._edit_flow()
        fake.FlowGraphWebWindow.assert_not_called()  # 未打开流程时不得创建画布


class JumpHostWindowTest(unittest.TestCase):
    """独立流程图窗口（flowmain）跳转接收：argv/单实例 → 载入。"""

    def test_wrap_window_platform_message_loads(self):
        j = Path(tempfile.mkdtemp()) / "m.json"
        j.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
        import xccv.flow_platform as fp_mod

        # 画布桩：测试进程不得导入真实 QtWebEngine（退出段错误，见 test_flowgraph）
        fake = types.SimpleNamespace(FlowGraphWebWindow=mock.MagicMock())
        with mock.patch.dict(sys.modules,
                             {"xccv.modeltools.flowgraph_web_window": fake}):
            win = fp_mod._wrap_window()
            self.addCleanup(win.close)
            win.on_platform_message([str(j)])
            self.assertEqual(win._page.current_flow_path, j)
            self.assertIsNotNone(win._page._canvas)   # 跳转语义=直接看到流程


class HomeJumpContractTest(unittest.TestCase):
    """home_jump（独立接收模块）：flow 参数识别 + nav 定位 + 载入委托。"""

    def test_handle_args_opens_flow_page(self):
        from xccv import home_jump

        j = Path(tempfile.mkdtemp()) / "t.json"
        j.write_text(json.dumps({"nodes": [], "links": []}), encoding="utf-8")
        win = QMainWindow()
        page = FlowPage()
        win.setCentralWidget(page)
        win.show()
        self.addCleanup(win.close)
        fake = types.SimpleNamespace(FlowGraphWebWindow=mock.MagicMock())
        with mock.patch.dict(sys.modules,
                             {"xccv.modeltools.flowgraph_web_window": fake}):
            consumed = home_jump.handle_args(win, [str(j), "--x"])
        self.assertEqual(consumed, str(j))
        if getattr(win, "nav_list", None) is not None:
            self.assertEqual(win.nav_list.currentRow(), 2)   # 流程图页
        self.assertEqual(page.current_flow_path, j)

    def test_forward_args_filters_flags(self):
        from xccv import home_jump

        self.assertEqual(home_jump.forward_args(["a.json", "--selftest"]),
                         ["a.json"])
        self.assertEqual(home_jump.forward_args([]), [])


if __name__ == "__main__":
    unittest.main()

