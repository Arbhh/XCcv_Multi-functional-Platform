"""平台测试：2 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_flowgraph
- test_flowgraph_main
"""


# == 来源模块：test_flowgraph ==
# -*- coding: utf-8 -*-
"""流程图（Web 版）测试：图转链、打包、桥接消息、窗口构建。

注意：QtWebEngine 必须在任何 QApplication 创建前导入，故本模块在顶部导入
xccv.modeltools.flowgraph_web_window。
"""

import json
from unittest import mock
import os
import sys
import tempfile
import unittest
from pathlib import Path

if sys.platform == "win32":
    os.environ["QT_QPA_PLATFORM"] = "windows"  # QtWebEngine 不支持 offscreen
else:
    # Linux 无显示器环境（CI）无法初始化 QtWebEngine：Web 窗口用例交由
    # Windows CI / 桌面环境覆盖，其余控件用例照常运行
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox --disable-gpu")

from PyQt5.QtWidgets import QApplication as _QApp  # noqa: E402
from xccv.engine import flow_package as fp  # noqa: E402
try:
    from xccv.modeltools.flowgraph_web_window import FlowGraphWebWindow  # noqa: E402
    # offscreen/minimal 平台下 QtWebEngine 无法初始化（进程内即崩），等效不可用
    _WEB_OK = os.environ.get("QT_QPA_PLATFORM") not in ("offscreen", "minimal")
except Exception:
    FlowGraphWebWindow = None
    _WEB_OK = False
from xccv.engine.detection_flow import graph_to_chain  # noqa: E402

_APP = _QApp.instance() or _QApp([])

# FlowSession 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
import tempfile as _tmpmod  # noqa: E402

from xccv.modeltools import flow_session as _fs  # noqa: E402

_fs._RUN_LOG_DIR = Path(_tmpmod.gettempdir()) / "xccv_test_flow_logs"


def _lg_graph(model_file=""):
    return {"last_node_id": 4, "last_link_id": 3, "nodes": [
        {"id": 1, "type": "xccv/input_image", "pos": [0, 0], "properties": {"preview": False}},
        {"id": 2, "type": "xccv/model_instance_seg", "pos": [1, 0],
         "properties": {"model_file": model_file, "threshold": 0.3, "preview": False}},
        {"id": 3, "type": "xccv/post_mask_area", "pos": [2, 0],
         "properties": {"min": 150, "max": 0, "preview": True}},
        {"id": 4, "type": "xccv/output", "pos": [3, 0], "properties": {}}],
        "links": [[1, 1, 0, 2, 0], [2, 2, 0, 3, 0], [3, 3, 0, 4, 0]]}


class GraphToChainTest(unittest.TestCase):
    def test_chain_reads_properties(self):
        chain = graph_to_chain(_lg_graph())
        self.assertEqual([n["op"] for n in chain["nodes"]], ["mask_area"])
        self.assertEqual(chain["nodes"][0]["params"]["min"], 150)

    def test_legacy_params_fallback(self):
        g = {"nodes": [{"id": 1, "type": "xccv/post_confidence", "params": {"min": 0.4}}],
             "links": []}
        chain = graph_to_chain(g)
        self.assertEqual(chain["nodes"][0]["params"]["min"], 0.4)

    def test_graph_to_chain_litegraph_six_element_links(self):
        """LiteGraph serialize 的 6 元链接 [id,src,slot,dst,slot,type] 必须可解析。

        此前按 5 元解包导致运行预览直接 ValueError。
        """
        g = {"last_node_id": 2, "last_link_id": 1, "nodes": [
            {"id": 1, "type": "xccv/model_detection",
             "properties": {"model_file": "m.pt"}},
            {"id": 2, "type": "xccv/post_confidence",
             "properties": {"min": 0.5, "preview": False}}],
            "links": [[1, 1, 0, 2, 0, "image"]]}
        chain = graph_to_chain(g)
        self.assertEqual([n["op"] for n in chain["nodes"]], ["confidence"])
        self.assertEqual(chain["nodes"][0]["params"]["min"], 0.5)


class FlowPackageTest(unittest.TestCase):
    def test_pack_from_graph(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "best.pt"
            model.write_bytes(b"\xa1" * 32)
            g = _lg_graph(model_file=str(model))
            models = fp.collect_model_files(g)
            self.assertEqual(list(models), ["best.pt"])
            p = Path(td) / "f.xvap"
            fp.save_xvap(p, g, models)
            flow2, m2 = fp.load_xvap(p)
            self.assertEqual(len(flow2["nodes"]), 4)
            self.assertEqual(m2["best.pt"], b"\xa1" * 32)

    def test_collect_missing_file_raises(self):
        g = _lg_graph(model_file="E:/not/exists/model.pt")
        with self.assertRaises(FileNotFoundError):
            fp.collect_model_files(g)


class BridgeTest(unittest.TestCase):
    """桥接消息：保存 JSON / 打包 .xvap / 模型文件选择回填。"""

    def _window(self):
        if not _WEB_OK:
            self.skipTest("offscreen 环境无法初始化 QtWebEngine（真实平台可用）")
        app = _APP
        w = FlowGraphWebWindow()
        w.show()
        app.processEvents()
        return w, app

    def test_webchannel_backend_registered(self):
        """路线 B 桥接：页面必须挂 QWebChannel 并注册 backend 桥对象。

        注意：断言只允许用键名等纯 Python 值——把 PyQt 包装对象放进失败
        消息（如 assertIs 的 two-way repr）会在格式化时触发原生崩溃。
        """
        if not _WEB_OK:
            self.skipTest("offscreen 环境无法初始化 QtWebEngine（真实平台可用）")
        w = FlowGraphWebWindow()
        try:
            ch = w.page.webChannel()
            self.assertIsNotNone(ch)
            self.assertEqual(list(ch.registeredObjects().keys()), ["backend"])
        finally:
            w.close()

    def test_save_json_message(self):
        w, app = self._window()
        with tempfile.TemporaryDirectory() as td:
            w.current_path = Path(td) / "flow.json"
            w.bridge.message("XCCV:" + json.dumps({"action": "save_json",
                                                   "flow": _lg_graph()}))
            self.assertTrue(w.current_path.is_file())
            loaded = json.loads(w.current_path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded["nodes"]), 4)
        w.close()

    def test_save_xvap_message(self):
        w, app = self._window()
        import xccv.engine.flow_package as fpmod

        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "best.pt"
            model.write_bytes(b"PK\x03\x04" + b"\xa1" * 8)  # 打包校验要求 torch 权重头
            g = _lg_graph(model_file=str(model))
            import PyQt5.QtWidgets as qw

            saved = {}
            orig_save = fpmod.save_xvap
            orig_dlg = qw.QFileDialog.getSaveFileName
            qw.QFileDialog.getSaveFileName = staticmethod(
                lambda *a, **k: (str(Path(td) / "flow.xvap"), ""))
            fake_count = {}

            def fake_save(path, flow, models):
                saved["path"] = Path(path)
                fake_count["n"] = len(models)
                orig_save(path, flow, models)

            fpmod.save_xvap = fake_save
            try:
                w.bridge.message("XCCV:" + json.dumps({"action": "save_xvap",
                                                       "flow": g}))
                app.processEvents()
            finally:
                fpmod.save_xvap = orig_save
                qw.QFileDialog.getSaveFileName = orig_dlg
            self.assertTrue(saved["path"].is_file())
            self.assertEqual(fake_count.get("n"), 1)
        w.close()

    def test_save_xvap_blocked_on_bad_model_header(self):
        """画布「打包 .xvap」与 Qt 侧走同一套打包前校验（回归防护：
        此前画布通道绕过校验，假权重/非法区域/未连通节点均未拦截）。"""
        w, app = self._window()
        import xccv.modeltools.flow_shared as fgm   # 校验助手迁至 flow_shared

        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "best.pt"
            model.write_bytes(b"\xa1" * 8)  # 头信息不可读
            g = _lg_graph(model_file=str(model))
            import PyQt5.QtWidgets as qw

            orig_dlg = qw.QFileDialog.getSaveFileName
            qw.QFileDialog.getSaveFileName = staticmethod(
                lambda *a, **k: (str(Path(td) / "flow.xvap"), ""))
            called = {}
            try:
                with mock.patch.object(
                        fgm, "_show_pack_problems",
                        side_effect=lambda p, ps: called.setdefault("ps", ps)):
                    w.bridge.message("XCCV:" + json.dumps(
                        {"action": "save_xvap", "flow": g}))
                    app.processEvents()
            finally:
                qw.QFileDialog.getSaveFileName = orig_dlg
            self.assertIn("torch 权重", " ".join(called.get("ps", [])))  # 校验清单已弹
            self.assertFalse((Path(td) / "flow.xvap").exists())    # 坏包不产出
        w.close()

    def test_pick_model_message(self):
        w, app = self._window()
        import PyQt5.QtWidgets as qw

        orig = qw.QFileDialog.getOpenFileName
        qw.QFileDialog.getOpenFileName = staticmethod(
            lambda *a, **k: ("E:/models/best.pt", ""))
        try:
            w.bridge.message('XCCV:{"action": "pick_model", "node_id": 2}')
            app.processEvents()
        finally:
            qw.QFileDialog.getOpenFileName = orig
        w.close()


class NewEngineBridgeTest(unittest.TestCase):
    """节点库下发 / 运行到节点 / 单节点重跑 / 快照（新引擎桥接）。"""

    def _window(self):
        if not _WEB_OK:
            self.skipTest("offscreen 环境无法初始化 QtWebEngine（真实平台可用）")
        app = _APP
        w = FlowGraphWebWindow()
        w.show()
        app.processEvents()
        return w, app

    @staticmethod
    def _xcv_graph(img_path: str):
        return {"last_node_id": 2, "last_link_id": 1, "nodes": [
            {"id": 1, "type": "xccv2/image_loader", "pos": [0, 0],
             "properties": {"path": img_path, "is_grayscale": False},
             "inputs": [], "outputs": [{"name": "image", "type": "image", "links": [1]}]},
            {"id": 2, "type": "xccv2/gaussian_blur", "pos": [1, 0],
             "properties": {"ksize": 3, "sigma": 0.0},
             "inputs": [{"name": "image", "type": "image", "link": 1}],
             "outputs": [{"name": "image", "type": "image", "links": []}]}],
            "links": [[1, 1, 0, 2, 0]]}

    def test_node_library_payload(self):
        from xccv.modeltools.flow_session import node_library_payload

        payload = node_library_payload()
        self.assertEqual(payload["event"], "node_library")
        self.assertIn("image", payload["port_types"])
        names = {n["name"] for n in payload["nodes"]}
        self.assertGreaterEqual(len(names), 25)  # 源2+预处理11+定位5+逻辑4+AI4+过滤8+检出源
        self.assertIn("template_match", names)
        self.assertIn("model_detection", names)

    def test_run_to_node_and_rerun_and_snapshot(self):
        w, app = self._window()
        sent = []
        orig = w.send_json
        w.send_json = lambda payload: sent.append(payload)
        try:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "in.png"
                import cv2
                import numpy as np
                cv2.imwrite(str(p), np.zeros((16, 16, 3), np.uint8))
                w.bridge.do_run_to_node({"flow": self._xcv_graph(str(p)), "node_id": 2})
                app.processEvents()
                run = next(x for x in sent if x.get("event") == "run_result")
                self.assertEqual(run["errors"], {})
                self.assertEqual(run["ok_count"], 2)
                self.assertIn("gaussian_blur(#2)", run["logs"])
                w.bridge.do_get_snapshot({"node_id": 2})
                app.processEvents()
                snap = next(x for x in sent if x.get("event") == "snapshot")
                self.assertIn("image", snap["images"])
                self.assertTrue(snap["images"]["image"].startswith("data:image/png;base64,"))
                w.bridge.do_rerun_node({"node_id": 2})
                app.processEvents()
                rerun = [x for x in sent if x.get("event") == "run_result"][-1]
                self.assertEqual(rerun["errors"], {})
        finally:
            w.send_json = orig
            w.close()

    def test_run_to_node_error_reported(self):
        w, app = self._window()
        sent = []
        orig = w.send_json
        w.send_json = lambda payload: sent.append(payload)
        try:
            w.bridge.do_run_to_node(
                {"flow": self._xcv_graph("Z:/缺失图片.png"), "node_id": 2})
            app.processEvents()
            run = next(x for x in sent if x.get("event") == "run_result")
            self.assertIn("1", run["errors"])
            self.assertIn("无法读取图片", run["errors"]["1"])
        finally:
            w.send_json = orig
            w.close()

    def test_pick_param_file_message(self):
        w, app = self._window()
        import PyQt5.QtWidgets as qw

        sent = []
        orig = w.send_json
        orig_dlg = qw.QFileDialog.getOpenFileName
        w.send_json = lambda payload: sent.append(payload)
        qw.QFileDialog.getOpenFileName = staticmethod(
            lambda *a, **k: ("E:/imgs/sample.png", ""))
        try:
            w.bridge.message('XCCV:{"action": "pick_param_file", "node_id": 1, "key": "path"}')
            app.processEvents()
        finally:
            qw.QFileDialog.getOpenFileName = orig_dlg
            w.send_json = orig
            w.close()
        ev = next(x for x in sent if x.get("event") == "param_file")
        self.assertEqual(ev["key"], "path")
        self.assertEqual(ev["path"], "E:/imgs/sample.png")

    def test_new_from_template(self):
        """从模板创建：弹选择框 → 模板图下发画布（节点齐全）。"""
        w, app = self._window()
        import PyQt5.QtWidgets as qw

        sent = []
        orig = w.send_json
        orig_dlg = qw.QInputDialog.getItem
        w.send_json = lambda payload: sent.append(payload)
        qw.QInputDialog.getItem = staticmethod(
            lambda *a, **k: ("缺陷斑点预筛 + 数量判定", True))
        try:
            w.bridge.message('XCCV:{"action": "new_from_template"}')
            app.processEvents()
        finally:
            qw.QInputDialog.getItem = orig_dlg
            w.send_json = orig
            w.close()
        loaded = next(x for x in sent if x.get("event") == "loaded")
        self.assertIn("斑点", loaded["name"])
        self.assertGreaterEqual(len(loaded["flow"]["nodes"]), 4)


class FlowEditorIdentityTest(unittest.TestCase):
    """流程图编辑器独立图标与平台归属（脱离模型工具分类）。"""

    def test_assets_exist(self):
        from xccv.ui_style import icon_path

        self.assertTrue(icon_path("flow.png").is_file())
        self.assertTrue(icon_path("flow.ico").is_file())

    def test_window_title_and_icon_detached_from_modeltools(self):
        """窗口标题不再带「XCcv 模型工具」，窗口图标为 flow.png。
        旧版独立编辑器弃用归档 → 断言挂独立流程图宿主窗。"""
        from PyQt5.QtGui import QIcon

        import xccv.flow_platform as fp_mod
        from xccv.ui_style import icon_path

        w = fp_mod._wrap_window()
        self.addCleanup(w.close)
        self.assertIn("XCcv 流程图", w.windowTitle())
        self.assertNotIn("模型工具", w.windowTitle())
        self.assertEqual(
            w.windowIcon().pixmap(64, 64).toImage().pixelColor(32, 32).name(),
            QIcon(str(icon_path("flow.png"))).pixmap(64, 64)
            .toImage().pixelColor(32, 32).name())

    def test_platform_host_window_uses_flow_icon(self):
        """流程图平台独立入口宿主窗口同样用 flow 图标。"""
        import xccv.flow_platform as fp_mod

        win = fp_mod._wrap_window()
        self.addCleanup(win.close)
        self.assertIn("流程图", win.windowTitle())


class RegionPickerJsTest(unittest.TestCase):
    """区域绘制器（矩形/圆/多边形）在真实画布页里的手势与写回。

    纯 JS 往返（解析/序列化/坏条目保留）由 runs/probe_f21_regions_v06.py 覆盖；
    此处覆盖指针手势：真实 PointerEvent 拖出三种形状 → node.properties.regions
    写回 → Python 侧 regions.parse_regions / 节点校验接受。
    """

    IMG_NATURAL = (200, 150)   # dataUrl 底图自然尺寸（w, h）

    @classmethod
    def _png_data_url(cls):
        import base64

        import cv2
        import numpy as np

        img = np.zeros((cls.IMG_NATURAL[1], cls.IMG_NATURAL[0], 3), np.uint8)
        ok, buf = cv2.imencode(".png", img)
        assert ok
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    def _window(self):
        if not _WEB_OK:
            self.skipTest("offscreen 环境无法初始化 QtWebEngine（真实平台可用）")
        app = _APP
        w = FlowGraphWebWindow()
        w.show()
        app.processEvents()
        self.addCleanup(w.close)
        return w, app

    def _js(self, w, code, wait_ms=1000):
        import time as _t

        from PyQt5.QtCore import QEventLoop, QTimer

        box = {}
        loop = QEventLoop()

        def _cb(v):
            box["v"] = v
            loop.quit()
        w.page.runJavaScript(code, _cb)
        deadline = _t.monotonic() + max(wait_ms, 3000) / 1000.0
        while not box and _t.monotonic() < deadline:
            loop.exec_()
        return box.get("v")

    def _wait(self, ms):
        from PyQt5.QtCore import QEventLoop, QTimer

        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec_()

    def test_draw_three_shapes_and_writeback(self):
        w, app = self._window()
        self._wait(4500)   # 页面加载 + 节点库握手（懒加载）
        data_url = self._png_data_url()
        setup = self._js(w, f"""
(() => {{
  graph.clear();
  const node = LiteGraph.createNode("xccv2/filter_region_block");
  graph.add(node);
  showInspector(node);
  rpEvent({{node_id: node.id, path: "probe.png", data_url: "{data_url}"}});
  return true;
}})()
""")
        self.assertTrue(setup)
        # 等底图解码就绪（rpMount/showInspector 均为微任务异步补挂）
        ready = None
        for _ in range(40):
            self._wait(150)
            ready = self._js(w, """
(() => { const i = document.querySelector("#ins-body .rp-box img.rp-img");
         return !!(i && i.naturalWidth); })()
""", wait_ms=300)
            if ready:
                break
        self.assertTrue(ready, "绘制底图未在超时内就绪")

        def client_expr(nx, ny):
            """自然像素 → client 坐标的 JS 表达式（用页面内 rpScale 换算）。"""
            return ("(() => { const i = document.querySelector("
                    '"#ins-body .rp-box img.rp-img"); const s = rpScale(i);'
                    f" const r = i.getBoundingClientRect();"
                    f" return [r.left + {nx} * s.sx, r.top + {ny} * s.sy]; }})()")

        def pointer(layer_sel, nx, ny, extra=""):
            return f"""
(() => {{
  const layer = document.querySelector("{layer_sel}");
  const [cx, cy] = {client_expr(nx, ny)};
  layer.dispatchEvent(new PointerEvent("pointerdown",
      {{bubbles: true, clientX: cx, clientY: cy, button: 0}}{extra}));
  return true;
}})()
"""

        def pointer_up(nx, ny):
            return f"""
(() => {{
  const layer = document.querySelector("#ins-body .rp-box .rp-imgbox");
  const [cx, cy] = {client_expr(nx, ny)};
  layer.dispatchEvent(new PointerEvent("pointerup",
      {{bubbles: true, clientX: cx, clientY: cy, button: 0}}));
  return true;
}})()
"""

        # --- 矩形：矩形工具 + 拖拽 ---
        self._js(w, """
(() => { document.querySelector(
    '#ins-body .rp-toolbar button[data-tool="rect"]').click(); return true; })()
""")
        self._js(w, pointer("#ins-body .rp-box .rp-imgbox", 20, 15))
        self._js(w, pointer_up(120, 90))
        # --- 圆：圆工具 + 拖拽（圆心→半径） ---
        self._js(w, """
(() => { document.querySelector(
    '#ins-body .rp-toolbar button[data-tool="circle"]').click(); return true; })()
""")
        self._js(w, pointer("#ins-body .rp-box .rp-imgbox", 150, 40))
        self._js(w, pointer_up(180, 70))
        # --- 多边形：多边形工具 + 3 次单击 + 双击闭合 ---
        self._js(w, """
(() => { document.querySelector(
    '#ins-body .rp-toolbar button[data-tool="polygon"]').click(); return true; })()
""")
        for nx, ny in ((30, 120), (90, 140), (60, 100)):
            self._js(w, pointer("#ins-body .rp-box .rp-imgbox", nx, ny))
        self._js(w, """
(() => { document.querySelector("#ins-body .rp-box .rp-imgbox")
         .dispatchEvent(new MouseEvent("dblclick", {bubbles: true})); return true; })()
""")
        regions = self._js(w, """
(() => { const n = graph._nodes[0]; return n.properties.regions; })()
""")
        self.assertIsInstance(regions, str, "regions 未写回")
        parts = regions.split(";")
        self.assertEqual(len(parts), 3, f"应有 3 条区域：{regions}")
        rect_s, circle_s, poly_s = parts
        # 矩形不带前缀（旧串兼容），端点即拖拽两点（容差 1px）
        nums = [float(v) for v in rect_s.split(",")]
        self.assertEqual(len(nums), 4)
        for got, want in zip(nums, (20, 15, 120, 90)):
            self.assertAlmostEqual(got, want, delta=1.5)
        # 圆带前缀：圆心≈(150,40)，半径≈拖拽距离
        self.assertTrue(circle_s.startswith("circle:"), circle_s)
        cnums = [float(v) for v in circle_s.split(":")[1].split(",")]
        self.assertAlmostEqual(cnums[0], 150, delta=1.5)
        self.assertAlmostEqual(cnums[1], 40, delta=1.5)
        self.assertAlmostEqual(cnums[2], (30 ** 2 + 30 ** 2) ** 0.5, delta=3.0)  # 拖拽欧氏距离
        # 多边形带前缀：3 个顶点
        self.assertTrue(poly_s.startswith("polygon:"), poly_s)
        pnums = [float(v) for v in poly_s.split(":")[1].split(",")]
        self.assertEqual(len(pnums), 6)
        # --- Python 侧：引擎解析与节点校验接受 ---
        from xccv.engine.flow.regions import any_contains, parse_regions, region_problems

        parsed = parse_regions(regions)
        self.assertEqual(len(parsed), 3, f"引擎解析 3 条：{regions}")
        self.assertEqual({r.shape for r in parsed}, {"rect", "circle", "polygon"})
        self.assertEqual(region_problems(regions), [], "打包校验应无问题")
        # 命中判定抽查：圆心在圆内、矩形含其中心
        self.assertTrue(any_contains([parsed[1]], cnums[0], cnums[1], 200, 150))
        self.assertTrue(any_contains([parsed[0]], 70, 52, 200, 150))
        # --- 放大编辑：弹层打开时渲染进弹层，关闭后回属性面板 ---
        opened = self._js(w, "(() => { rpOpenBig(); return true; })()")
        self.assertTrue(opened)
        self._wait(400)   # 弹层 <img> 解码 + rpRender 异步补绘
        big = self._js(w, """
(() => {
  const modal = document.getElementById("rp-modal");
  const mSvg = modal.querySelector(".rp-box svg.rp-svg");
  const mShapes = mSvg ? mSvg.querySelectorAll(".rp-shape").length : 0;
  return {open: modal.style.display === "flex", shapes: mShapes};
})()
""")
        self.assertTrue((big or {}).get("open"), "放大编辑弹层未打开")
        self.assertEqual((big or {}).get("shapes"), 3, "弹层未绘制 3 条区域")
        closed = self._js(w, "(() => { rpCloseBig(); return true; })()")
        self.assertTrue(closed)
        self._wait(400)   # 关闭后属性面板重建（微任务）+ 底图重解码
        restored = self._js(w, """
(() => {
  const modal = document.getElementById("rp-modal");
  const iSvg = document.querySelector("#ins-body .rp-box svg.rp-svg");
  return {open: modal.style.display === "flex",
          shapes: iSvg ? iSvg.querySelectorAll(".rp-shape").length : 0};
})()
""")
        self.assertFalse((restored or {}).get("open"), "弹层未关闭")
        self.assertEqual((restored or {}).get("shapes"), 3, "关闭后属性面板未恢复绘制")

# == 来源模块：test_flowgraph_main ==
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

if __name__ == '__main__':
    unittest.main()
