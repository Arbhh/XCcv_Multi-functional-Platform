# -*- coding: utf-8 -*-
"""V0.6 P6/T6-4：缺陷生成（DefectAugmentStudio 移植）测试。

三层结构：
- 后端契约（FastAPI 经真实 uvicorn port=0 起服）——任何平台都跑；
- 入口契约（launcher 映射 / 模型工具第四卡 / 闪屏图标 / 窗口模块纪律文本）
  ——offscreen 可跑（全程不导入 QtWebEngine）；
- Qt 壳生命周期（构建→动态端口→loadFinished→关窗停服→端口释放、双实例
  端口互异）——仅 QT_QPA_PLATFORM=windows 时跑294offscreen 无 GL
  上下文，QtWebEngine 不能 offscreen 自动化）。
"""
import contextlib
import json
import os
import socket
import threading
import time
import unittest
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "xccv" / "defectgen" / "web"

import uvicorn  # noqa: E402

from xccv.defectgen.backend.app import create_app  # noqa: E402


_APP_HOLD = None  # QApplication 必须持引用，否则被 GC（P188 同族）


def _ensure_app():
    """创建/复用 QApplication。AA_ShareOpenGLContexts 仅在非 offscreen 平台
    设置：offscreen 插件无 GL 上下文，带该属性构造 QApplication 会直接
    fail-fast（实测 0xC0000409，P188）；offscreen 下也不会导入 QtWebEngine，
    不需要该属性。"""
    global _APP_HOLD
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            from PyQt5.QtCore import Qt

            QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
        app = QApplication([])
        _APP_HOLD = app
    return app


@contextlib.contextmanager
def _serve():
    """真实 uvicorn 起服（port=0 动态端口）；退出即 should_exit 停服。"""
    server = uvicorn.Server(uvicorn.Config(
        create_app(), host="127.0.0.1", port=0, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True, name="t-defectgen-uvicorn")
    th.start()
    port = None
    deadline = time.time() + 20
    while time.time() < deadline and port is None:
        try:
            if server.servers:
                port = int(server.servers[0].sockets[0].getsockname()[1])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.05)
    if port is None:
        server.should_exit = True
        th.join(timeout=5)
        raise AssertionError("uvicorn 动态端口未就绪")
    try:
        yield port
    finally:
        server.should_exit = True
        th.join(timeout=10)


def _get(port, path, timeout=8):
    # 返回 HTTPMessage 本体（.get 大小写不敏感）；dict() 会固化 wire 上的
    # 小写键 "content-type"，导致 MIME 断言误判
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=timeout) as r:
        return r.status, r.headers, r.read()


class DefectgenBackendContractTest(unittest.TestCase):
    """后端契约：health / 首页 / 静态 MIME（P187 锁）/ 路由清单。"""

    def test_health_and_index(self):
        with _serve() as port:
            st, _, body = _get(port, "/api/health")
            self.assertEqual(st, 200)
            self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
            st, _, body = _get(port, "/")
            self.assertEqual(st, 200)
            self.assertIn(b"<html", body[:400].lower())

    def test_static_mime_lock_p187(self):
        """Windows 注册表把 .js 记成 text/plain 会让 ES module 白屏——锁定修复。"""
        js = sorted(p.name for p in (WEB_DIR / "js").glob("*.js"))
        self.assertGreaterEqual(len(js), 10)
        with _serve() as port:
            for name in js:
                st, headers, _ = _get(port, f"/js/{name}")
                self.assertEqual(st, 200, name)
                self.assertIn("javascript",
                              headers.get("Content-Type", "").lower(), name)
            for name in sorted(p.name for p in (WEB_DIR / "css").glob("*.css")):
                _, headers, _ = _get(port, f"/css/{name}")
                self.assertIn("css",
                              headers.get("Content-Type", "").lower(), name)

    def test_api_routes_registered(self):
        paths = {getattr(r, "path", "") for r in create_app().routes}
        for p in ("/api/health", "/api/extract", "/api/sam_status",
                  "/api/poisson_preview", "/api/paint_defect",
                  "/api/placement_map", "/api/composite",
                  "/api/batch/start", "/api/batch/status/{jid}",
                  "/api/batch/cancel/{jid}"):
            self.assertIn(p, paths, p)

    def test_sam_status_contract(self):
        with _serve() as port:
            st, _, body = _get(port, "/api/sam_status")
            self.assertEqual(st, 200)
            self.assertTrue(json.loads(body.decode("utf-8"))["grabcut"])


class DefectgenEntryContractTest(unittest.TestCase):
    """入口三件套 + 窗口模块纪律文本（不导入 QtWebEngine，offscreen 安全）。"""

    def test_launcher_mapping(self):
        from xccv.launcher import PLATFORMS

        exe, module = PLATFORMS["defectgen"]
        self.assertEqual(exe, "XCcv缺陷生成.exe")
        self.assertEqual(module, "xccv.defectgen.defectgen_window")

    def test_modeltools_card_exists(self):
        _ensure_app()
        from xccv.modeltools.modeltools_window import ModelToolsWindow, _ToolCard

        win = ModelToolsWindow()
        self.addCleanup(win.close)
        cards = [c for c in win.findChildren(_ToolCard)
                 if getattr(c, "tool_key", None) == "defectgen"]
        self.assertEqual(len(cards), 1)
        self.assertTrue(cards[0].isEnabled())

    def test_splash_icon_mapping(self):
        from xccv.platform_entry import _SPLASH_ICONS

        self.assertEqual(_SPLASH_ICONS["defectgen"], "defectgen.png")
        assets = ROOT / "xccv" / "assets"
        for f in ("defectgen.png", "defectgen.ico"):
            self.assertTrue((assets / f).is_file(), f)

    def test_window_module_discipline_text(self):
        """不导入模块，锁定 T6-2 纪律点（动态端口/停服/页面销毁/GL 共享）。"""
        src = (ROOT / "xccv" / "defectgen" / "defectgen_window.py"
               ).read_text(encoding="utf-8")
        for token in ("AA_ShareOpenGLContexts", "port=0", "should_exit",
                      "deleteLater", "QTWEBENGINE_CHROMIUM_FLAGS",
                      "actual_port"):
            self.assertIn(token, src, token)


def _shell_ok():
    """壳测试运行前提（装饰期判断，P190）：
    - 平台非 offscreen6052offscreen 无 GL 上下文）；
    - QApplication 尚未创建6102必须先导入窗口模块再建 app——gate 单进程
      discovery 里 runner/兄弟测试早已建 app 且无 AA 标志，此时导入
      QtWebEngine 必炸；兄弟模块 test_flowgraph.py 还会在 import 期把
      QT_QPA_PLATFORM 硬改成 "windows"，env 判断在装饰期不可靠）。"""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() is None


@unittest.skipUnless(_shell_ok(),
                     "QtWebEngine 壳需独立进程/独立平台（当前进程"
                     "已有 QApplication 或平台为 offscreen，均不能安全导入）")
class DefectgenWindowShellTest(unittest.TestCase):
    """T6-4 验收：Qt 壳完整生命周期（真实窗口，windows 平台专项）。"""

    @classmethod
    def setUpClass(cls):
        cls.app = _ensure_app()
        cls.app.setQuitOnLastWindowClosed(False)
        from xccv.defectgen.defectgen_window import DefectGenWindow

        cls.DGW = DefectGenWindow

    @staticmethod
    def _close_all(*wins):
        for w in wins:
            w.close()

    def _wait_port_loaded(self, win, timeout=30.0):
        state = {"loaded": False, "failed": False}

        def _on_load(ok):
            state["loaded" if ok else "failed"] = True

        win.page.loadFinished.connect(_on_load)
        port = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.app.processEvents()
            if port is None:
                port = win._server.actual_port()
            elif state["loaded"] or state["failed"]:
                break
            time.sleep(0.03)
        self.assertIsInstance(port, int, "动态端口未就绪")
        self.assertTrue(state["loaded"], "页面 loadFinished 失败")
        return port

    def _assert_port_closed(self, port, timeout=5.0):
        deadline = time.time() + timeout
        rc = 0
        while time.time() < deadline:
            s = socket.socket()
            s.settimeout(1.0)
            rc = s.connect_ex(("127.0.0.1", port))
            s.close()
            if rc != 0:
                return
            time.sleep(0.1)
        self.fail(f"端口 {port} 未释放（connect_ex={rc}）")

    def test_lifecycle_and_stop_on_close(self):
        win = self.DGW()
        win.show()
        self.addCleanup(self._close_all, win)
        port = self._wait_port_loaded(win)
        self.assertTrue(
            win.page.url().toString().startswith(f"http://127.0.0.1:{port}"))
        st, _, body = _get(port, "/api/health")
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
        win.close()
        self.app.processEvents()
        self._assert_port_closed(port)

    def test_two_instances_distinct_ports(self):
        """与流程图编辑器等同时打开无冲突的根保证：动态端口互异、互不影响。"""
        w1 = self.DGW()
        w1.show()
        w2 = self.DGW()
        w2.show()
        self.addCleanup(self._close_all, w1, w2)
        p1 = self._wait_port_loaded(w1)
        p2 = self._wait_port_loaded(w2)
        self.assertNotEqual(p1, p2)
        st, _, body = _get(p1, "/api/health")
        self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
        st, _, body = _get(p2, "/api/health")
        self.assertTrue(json.loads(body.decode("utf-8"))["ok"])
        w1.close()
        self.app.processEvents()
        self._assert_port_closed(p1)
        st, _, body = _get(p2, "/api/health")
        self.assertEqual(st, 200, "实例 1 关闭不得影响实例 2")
        w2.close()
        self.app.processEvents()
        self._assert_port_closed(p2)


if __name__ == "__main__":
    unittest.main()
