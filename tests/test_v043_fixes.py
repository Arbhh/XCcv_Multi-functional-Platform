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

from xccv.modeltools.flow_session import (
    FlowSession, _normalize_types, node_library_payload,
)
from xccv.modeltools import flow_session as _fs
from xccv.engine.flow.ir import Geometry, ImageData, MeasureResult

node_library_payload()  # 注册全部算子（含 output.result_output）

import tempfile as _tempfile

_fs._RUN_LOG_DIR = Path(_tempfile.gettempdir()) / "xccv_test_v043_logs"

# 桥接类守卫导入（坑 40 变体）：全量套件中其他模块已创建 QApplication 时，
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
    原生崩溃——坑 44）。
    """

    def _bridge(self):
        from unittest import mock as _mock

        if not _BRIDGE_OK:
            self.skipTest("全量套件中 QApplication 先于 QtWebEngine 存在，"
                          "桥接顺序用例由 windows E2E 探针兜底（坑 40）")
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
            cv2.imwrite(str(p), img)
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
                cv2.imwrite(str(root / name), img)
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
            # （QThread+QEventLoop 的信号时序在测试进程内偶发不可达，坑 44 同族）
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
    组合会触发 sip 原生崩溃（坑 44）——用子进程验证。
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

    flow_platform 与 QtWebEngine/重模块同进程时 sip 托管顺序敏感（坑 44），
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


if __name__ == "__main__":
    unittest.main()
