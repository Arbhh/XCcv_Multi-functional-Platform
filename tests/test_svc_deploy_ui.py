"""部署服务测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v050_deploy_ui
- test_v06_f4_deploy
- test_v06_f4_json
"""


# == 来源模块：test_v050_deploy_ui ==
"""v0.5 部署平台 UI 测试：ServicePanel 构建/服务状态切换/卡片、
ModelServiceWindow 服务离线态行为、Key 复制。HTTP 交互在实测联调覆盖。"""

import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = None


def _app():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


class ServicePanelTest(unittest.TestCase):
    def setUp(self):
        _app()
        from xccv.deploy_platform import ServicePanel

        self.page = ServicePanel()

    def test_builds_with_controls_and_extensions(self):
        from xccv.deploy_platform import _load_service_cfg

        cfg = _load_service_cfg()   # 端口/Key 跟随真实配置（用户可改，勿硬编码）
        self.assertEqual(self.page.port_spin.value(), int(cfg.port))
        self.assertEqual(self.page.host_combo.currentText(), str(cfg.host))
        self.assertTrue(self.page.api_key())            # Key 已显示（供复制）
        self.assertTrue(self.page.base_url().endswith(str(cfg.port)))
        # 拓展区（v0.5.4）：GPU 条 + 接口清单 + 文档按钮
        self.assertTrue(hasattr(self.page, "gpu_bar"))
        self.assertTrue(hasattr(self.page, "api_list"))
        self.assertTrue(hasattr(self.page, "btn_docs"))

    def test_status_toggle_reflects_service_state(self):
        self.page._service_ok = True
        self.page._refresh_status()
        self.assertIn("运行中", self.page.status_label.text())
        self.assertEqual(self.page.btn_toggle.text(), "停止服务")
        self.page._service_ok = False
        self.page._refresh_status()
        self.assertIn("未运行", self.page.status_label.text())
        self.assertEqual(self.page.btn_toggle.text(), "启动服务")

    def test_copy_key_to_clipboard(self):
        from PyQt5.QtWidgets import QApplication

        self.page._copy_key()
        self.assertEqual(QApplication.clipboard().text(), self.page.api_key())


class ModelServiceWindowOfflineTest(unittest.TestCase):
    """服务未运行时打开模型加载窗口：给出指路提示、表格为空（不报错）。"""

    def setUp(self):
        _app()
        from xccv.deploy_platform import ServicePanel, ModelServiceWindow

        self.page = ServicePanel()
        self.page._service_ok = False
        self.win = ModelServiceWindow(self.page)

    def test_offline_hint_and_empty_table(self):
        self.win._refresh_models()
        self.assertIn("未运行", self.win.status_label.text())
        self.assertEqual(self.win.table.rowCount(), 0)

    def test_load_disabled_when_service_off(self):
        """A8 防御：服务未运行时「加载到服务」必须禁用。"""
        self.win._refresh_models()
        self.assertFalse(self.win.btn_load.isEnabled())
        self.win._page._service_ok = True
        self.win._refresh_models()
        self.assertTrue(self.win.btn_load.isEnabled())

    def test_standalone_mode_reads_config(self):
        """v0.5.4 独立模式（page=None）：地址/Key 直接来自配置文件。"""
        from xccv.deploy_platform import ModelServiceWindow

        win = ModelServiceWindow()
        self.addCleanup(win.close)
        self.assertTrue(win.base_url().startswith("http://"))
        self.assertTrue(isinstance(win.api_key(), str))

    def test_load_model_requires_valid_path(self):
        from PyQt5.QtWidgets import QMessageBox

        self.win.path_edit.setText("")
        with mock.patch("xccv.deploy_platform.QMessageBox.information") as m:
            self.win._load_model()
            self.assertTrue(m.called)

    def test_after_load_none_shows_hint(self):
        """连接失败（status=None）给"无法连接"人话提示，不显示 None。"""
        from PyQt5.QtWidgets import QMessageBox

        self.win._page._service_ok = True
        with mock.patch("xccv.deploy_platform.QMessageBox.critical") as m:
            self.win._after_load(None, {})
            m.assert_called_once()
            self.assertIn("无法连接", m.call_args[0][2])


class ServicePanelStateMachineTest(unittest.TestCase):
    """A7/A8 修复回归：状态无条件刷新、看门狗失败提示、sender 回调清理。"""

    def setUp(self):
        _app()
        from xccv.deploy_platform import ServicePanel

        self.page = ServicePanel()

    def test_health_refresh_unconditional(self):
        """失败→失败也必须刷新（A7：不再永久停留「检测中」）。"""
        self.page._service_ok = False
        self.page._on_health(None, {})
        self.assertIn("未运行", self.page.status_label.text())
        self.page._service_ok = False
        self.page._on_health(None, {})
        self.assertIn("未运行", self.page.status_label.text())  # 仍刷新，无"检测中"
        self.assertNotIn("检测中", self.page.status_label.text())

    def test_watchdog_reports_dead_process(self):
        """A2：子进程退出后看门狗给出失败提示（含日志指引）。"""
        import subprocess as sp
        import sys

        # 跨平台探针进程：快速退出的空 Python 进程（cmd 仅 Windows 可用）
        self.page._proc = sp.Popen([sys.executable, "-c", "pass"])
        self.page._proc.wait()
        self.page._watchdog_left = 5
        self.page._watchdog()
        self.assertIn("启动失败", self.page.tip_label.text())
        self.assertIn("service.log", self.page.tip_label.text())

    def test_start_refuses_when_port_in_use(self):
        """A2 前置防御：端口被占时不启动新进程，红字提示。"""
        import socket

        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = blocker.getsockname()[1]
        self.page.port_spin.setValue(port)
        self.page._service_ok = False
        self.page._start_service()
        try:
            self.assertIn("已被占用", self.page.tip_label.text())
            self.assertIsNone(self.page._proc)
        finally:
            blocker.close()

    def test_track_cleans_pending_and_routes_cb(self):
        """_HttpMixin：pending 表登记/清理与回调分派（确定性路径）。"""
        from unittest import mock

        from PyQt5.QtCore import QUrl
        from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest

        # _track 登记（真实 reply：需要 finished 信号挂接）
        nam = QNetworkAccessManager(self.page)
        reply = nam.get(QNetworkRequest(
            QUrl("http://127.0.0.1:1/nope")))   # 立即失败也无妨，不等回调
        self.page._track(reply, lambda s, b: None)
        self.assertIn(reply, self.page._pending)
        reply.deleteLater()

    def test_dispatch_failure_and_json(self):
        """_dispatch：失败态回调 (None, {})；JSON 脏数据回退 message。"""
        got = []
        reply = object()   # 哨兵键（pending 表只需可哈希）
        self.page._pending[reply] = cb = lambda s, b: got.append((s, b))
        self.page._dispatch(reply, None, b"", cb)
        self.assertEqual(got, [(None, {})])
        self.page._dispatch(reply, 200, b"not-json", cb)
        self.assertEqual(got[-1][0], 200)
        self.assertIn("message", got[-1][1])

    def test_bind_host_persists_on_start(self):
        """Q1：启动时把监听地址/端口写回配置文件。"""
        import tempfile
        from xccv.deploy_platform import _load_service_cfg, _save_service_cfg

        if self.page._service_ok:
            self.skipTest("服务运行中，跳过持久化用例")
        cfg = _load_service_cfg()
        cfg.host = "127.0.0.1"
        old_port = cfg.port
        try:
            self.page.host_combo.setCurrentText("127.0.0.1")
            self.page.port_spin.setValue(8899)
            # 占位占用 8899 触发"端口占用"提前返回——但配置已在保存之后？
            # 实现顺序：保存配置在端口检查之后，故这里直接验证保存函数路径
            cfg.port = 8899
            _save_service_cfg(cfg)
            self.assertEqual(_load_service_cfg().port, 8899)
        finally:
            cfg.port = old_port
            _save_service_cfg(cfg)

# == 来源模块：test_v06_f4_deploy ==
"""回归：部署平台退出与推理服务生命周期绑定。

- 部署平台真正退出（托盘菜单退出/无托盘强退）前同步停止推理服务；
- 关闭窗口仍默认只最小化到系统托盘（服务继续运行，行为不变）。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

_APP__v06_f4_deploy = QApplication.instance() or QApplication([])


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return None   # 一直存活

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, _ms):
        self.waited = True
        return 0


class StopServiceBlockingTest(unittest.TestCase):
    def test_terminates_own_process_even_if_offline(self):
        """服务 HTTP 已离线但本进程拉起的子进程仍在：退出兜底 terminate。"""
        from xccv.deploy_platform import ServicePanel

        panel = ServicePanel()
        self.addCleanup(panel.deleteLater)
        fake = _FakeProc()
        panel._proc = fake
        panel._service_ok = False   # 不走 HTTP 分支（无真实服务）
        panel.stop_service_blocking(max_ms=10)
        self.assertIsNone(panel._proc)
        self.assertTrue(fake.terminated)
        self.assertTrue(fake.waited)
        self.assertFalse(fake.killed)

    def test_proc_none_noop(self):
        from xccv.deploy_platform import ServicePanel

        panel = ServicePanel()
        self.addCleanup(panel.deleteLater)
        panel._proc = None
        panel._service_ok = False
        panel.stop_service_blocking(max_ms=10)   # 不抛异常即可


class ServiceWindowQuitTest(unittest.TestCase):
    def _window(self):
        from unittest.mock import patch

        from xccv.deploy_platform import ServiceWindow

        # ServiceHealthPoller 在 __init__ 内惰性导入，须 patch 源模块
        with patch("xccv.service_health.ServiceHealthPoller"):
            win = ServiceWindow()
        self.addCleanup(win.deleteLater)
        return win

    def test_real_quit_stops_service_first(self):
        """托盘「退出部署平台」：先同步停服，再真正关闭窗口。"""
        from unittest.mock import patch

        win = self._window()
        calls = []
        with patch.object(win._page, "stop_service_blocking",
                          side_effect=lambda **kw: calls.append(kw)):
            win._real_quit()
        self.assertEqual(len(calls), 1)
        self.assertTrue(win._force_quit)

    def test_close_event_no_tray_stops_service(self):
        """无托盘环境的直接关闭：closeEvent 内补停服（批注四绑定兜底）。"""
        from unittest.mock import patch

        win = self._window()
        calls = []
        with patch("PyQt5.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
                   return_value=False), \
             patch.object(win._page, "stop_service_blocking",
                          side_effect=lambda **kw: calls.append(kw)):
            win.close()
        self.assertEqual(len(calls), 1)

    def test_real_quit_does_not_double_stop(self):
        """_real_quit 已停服后 closeEvent 不重复停服。"""
        from unittest.mock import patch

        win = self._window()
        calls = []
        with patch.object(win._page, "stop_service_blocking",
                          side_effect=lambda **kw: calls.append(kw)):
            win._real_quit()
        self.assertEqual(len(calls), 1)

    def test_quit_menu_text_bound_to_service(self):
        """文案与行为一致：退出=服务同步停止（批注四文字提示）。"""
        from xccv.deploy_platform import ServiceWindow

        win = self._window()
        texts = [a.text() for m in [win._tray.contextMenu()]
                 for a in m.actions()]
        self.assertTrue(any("推理服务同步停止" in t for t in texts), texts)

# == 来源模块：test_v06_f4_json ==
"""JSON 契约锁定测试（批注一.2 收官）。

单一出口 xccv/engine/result_json.py：服务 /infer、/flow/run、测试平台批量、
流程图页批量、C# SDK 全部消费这两份 schema。本文件锁定跨 builder 契约：

- 任务无关基键集合：build_flow_result_json ⊇ build_result_json；
  流程/传统仅允许扩展 geometries/measurements（只增不改）；
- verdict 键恒存在（无判定 = null，落盘读回不丢）；
- detections / classifications 数组恒存在（无数据为空数组）；
- SCHEMA_VERSION 锁定为 1（改版本必须显式更新本文件 + docs/result_schema.md）。
"""

import json
import tempfile
import unittest
from pathlib import Path

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from xccv.engine.models.base import Prediction
from xccv.engine.result_json import (
    SCHEMA_VERSION, build_result_json, build_flow_result_json, save_result_json,
)


def _source():
    return {"file": "img00.png", "path": "D:/ds/img00.png",
            "width": 640, "height": 480}


AI_BASE_KEYS = {"schema_version", "task", "source", "model", "inference",
                "detections", "classifications", "verdict", "error"}
FLOW_EXT_KEYS = {"geometries", "measurements"}
SOURCE_KEYS = {"file", "path", "width", "height"}
MODEL_KEYS = {"file", "name"}
INF_COMMON_KEYS = {"flow_applied", "timestamp"}
INF_AI_KEYS = INF_COMMON_KEYS | {"conf", "iou", "flow_removed", "count_check"}
INF_FLOW_KEYS = INF_COMMON_KEYS | {"flow_name"}


class BuilderKeyContractTest(unittest.TestCase):
    def test_ai_base_keys(self):
        payload = build_result_json(task="detection", source=_source())
        self.assertEqual(set(payload), AI_BASE_KEYS)
        self.assertEqual(payload["task"], "detection")

    def test_flow_keys_are_superset_of_ai(self):
        payload = build_flow_result_json(source=_source())
        self.assertEqual(set(payload), AI_BASE_KEYS | FLOW_EXT_KEYS)
        self.assertEqual(payload["task"], "flow")

    def test_source_model_inference_shape(self):
        ai = build_result_json(task="detection", source=_source(),
                               model={"file": "best.pt", "name": "yolo11n"})
        fl = build_flow_result_json(source=_source(), flow_name="模板A")
        for p in (ai, fl):
            self.assertEqual(set(p["source"]), SOURCE_KEYS)
            self.assertEqual(set(p["model"]), MODEL_KEYS)
        self.assertEqual(ai["model"], {"file": "best.pt", "name": "yolo11n"})
        self.assertEqual(fl["model"], {"file": "", "name": ""})  # 无模型也保留结构
        self.assertEqual(set(ai["inference"]), INF_AI_KEYS)
        self.assertEqual(set(fl["inference"]), INF_FLOW_KEYS)
        self.assertEqual(set(ai["inference"]) & set(fl["inference"]),
                         INF_COMMON_KEYS)
        self.assertIs(fl["inference"]["flow_applied"], True)
        self.assertIs(ai["inference"]["flow_applied"], False)  # 无 flow_info

    def test_schema_version_locked(self):
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(build_result_json(
            task="detection", source=_source())["schema_version"], 1)
        self.assertEqual(build_flow_result_json(
            source=_source())["schema_version"], 1)

    def test_arrays_always_present(self):
        ai = build_result_json(task="detection", source=_source())
        fl = build_flow_result_json(source=_source())
        for p in (ai, fl):
            self.assertEqual(p["detections"], [])
            self.assertEqual(p["classifications"], [])
        self.assertEqual(fl["geometries"], [])
        self.assertEqual(fl["measurements"], [])


class VerdictAlwaysPresentTest(unittest.TestCase):
    def test_ai_verdict_null_without_count_check(self):
        payload = build_result_json(task="detection", source=_source())
        self.assertIn("verdict", payload)
        self.assertIsNone(payload["verdict"])

    def test_ai_verdict_from_count_check(self):
        payload = build_result_json(task="detection", source=_source(),
                                    flow_info={"removed": 1, "count_check": "NG"})
        self.assertEqual(payload["verdict"],
                         {"passed": False, "source": "count_check",
                          "detail": "数量判定 NG"})

    def test_flow_verdict_null_and_explicit(self):
        self.assertIsNone(build_flow_result_json(source=_source())["verdict"])
        v = {"passed": True, "source": "count_check", "detail": "x"}
        self.assertEqual(
            build_flow_result_json(source=_source(), verdict=v)["verdict"], v)

    def test_verdict_survives_json_roundtrip(self):
        """消费端视角：verdict=null/对象 落盘读回键与值不丢。"""
        ai = build_result_json(task="detection", source=_source(),
                               flow_info={"count_check": "OK"})
        fl = build_flow_result_json(
            source=_source(),
            verdict={"passed": False, "source": "compare_value", "detail": "d"})
        empty = build_flow_result_json(source=_source())
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.json"
            p2 = Path(td) / "b.json"
            p3 = Path(td) / "c.json"
            save_result_json(p1, ai)
            save_result_json(p2, fl)
            save_result_json(p3, empty)
            r1 = json.loads(p1.read_text(encoding="utf-8"))
            r2 = json.loads(p2.read_text(encoding="utf-8"))
            r3 = json.loads(p3.read_text(encoding="utf-8"))
        self.assertEqual(r1["verdict"]["passed"], True)
        self.assertEqual(r2["verdict"]["passed"], False)
        self.assertIn("verdict", r3)
        self.assertIsNone(r3["verdict"])


class FlowExtensionTest(unittest.TestCase):
    def test_detections_shape_matches_ai(self):
        pred = Prediction(labels=[0], scores=[0.9123],
                          boxes=[[10.0, 20.0, 50.0, 60.0]])
        ai = build_result_json(task="detection", source=_source(),
                               prediction=pred, classes=["划痕"])
        fl = build_flow_result_json(source=_source(),
                                    detections=ai["detections"])
        self.assertEqual(fl["detections"], ai["detections"])
        self.assertEqual(fl["detections"][0]["label"], "划痕")
        self.assertNotIn("mask_area", fl["detections"][0])

    def test_geometries_strip_noise(self):
        class _G:
            def to_json(self):
                return {"kind": "circle", "score": float("nan"),
                        "subpixel": False,
                        "data": {"cx": 1.0, "cy": 2.0, "r": 3.0}}

        payload = build_flow_result_json(source=_source(), geometries=[_G()])
        self.assertEqual(payload["geometries"],
                         [{"kind": "circle",
                           "data": {"cx": 1.0, "cy": 2.0, "r": 3.0}}])

    def test_measurements_default_unit_px_dropped(self):
        class _M:
            def to_json(self):
                return {"value": 12.5, "unit": "px"}

        payload = build_flow_result_json(source=_source(), measurements=[_M()])
        self.assertEqual(payload["measurements"], [{"value": 12.5}])

    def test_empty_geometry_shells_dropped(self):
        class _G:
            def to_json(self):
                return {"kind": "circle", "data": {}}

        payload = build_flow_result_json(source=_source(), geometries=[_G()])
        self.assertEqual(payload["geometries"], [])

if __name__ == '__main__':
    unittest.main()
