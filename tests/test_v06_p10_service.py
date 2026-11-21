"""部署性能阶段回归：计时头/尺寸策略/双模式 Key/预载/WS。

全部离线 CPU 跑（假模型直插 manager 池），不依赖真实权重与 GPU。
"""

import inspect
import os
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from fastapi.testclient import TestClient

from xccv.service.api import deps
from xccv.service.api.routes_flow import flow_run
from xccv.service.config import ServiceConfig
from xccv.service.core.model_manager import ModelEntry
from xccv.service.main import create_app

_KEY = "p10testkey"
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "testclient"}


class _FakeModel:
    task = "detection"

    def __init__(self, train_input=0):
        self.train_input = train_input
        self.last_image_hw = None

    def predict(self, image, conf=0.25, iou=0.45):
        self.last_image_hw = image.shape[:2]
        h, w = image.shape[:2]
        from xccv.engine.models.base import Prediction

        return Prediction(labels=[1], scores=[0.9],
                          boxes=[[w * 0.25, h * 0.25, w * 0.5, h * 0.5]],
                          masks=None)


def _png_bytes(w=64, h=64):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _register(app, model=None, model_id="t-1"):
    m = model or _FakeModel()
    app.state.manager._entries[model_id] = ModelEntry(
        model_id=model_id, alias="t", path="fake.pt", model=m,
        task="detection", classes=["a"], device="cpu",
        loaded_at=time.time(), train_input=int(getattr(m, "train_input", 0) or 0))
    return m


class TimingHeaderTest(unittest.TestCase):
    """X-XCcv-Timing 计时头三态。"""

    def _client(self, cfg):
        app = create_app(cfg)
        _register(app)
        return app, TestClient(app)

    def test_header_present_by_default(self):
        app, c = self._client(ServiceConfig(api_key=_KEY))
        with mock.patch.object(deps, "_LOOPBACK_HOSTS", _LOOPBACK):
            r = c.post("/v1/infer/image?api_key=" + _KEY,
                       files={"image": ("a.png", _png_bytes(), "image/png")},
                       data={"model_id": "t-1"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("X-XCcv-Timing", r.headers)
        self.assertIn("infer=", r.headers["X-XCcv-Timing"])
        self.assertIn("total=", r.headers["X-XCcv-Timing"])

    def test_header_absent_when_disabled(self):
        app, c = self._client(ServiceConfig(api_key=_KEY, timing_enabled=False))
        with mock.patch.object(deps, "_LOOPBACK_HOSTS", _LOOPBACK):
            r = c.post("/v1/infer/image?api_key=" + _KEY,
                       files={"image": ("a.png", _png_bytes(), "image/png")},
                       data={"model_id": "t-1"})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("X-XCcv-Timing", r.headers)


class QueryKeyTest(unittest.TestCase):
    """?api_key= 查询串仅回环放行。"""

    def setUp(self):
        self.app = create_app(ServiceConfig(api_key=_KEY))
        self.c = TestClient(self.app)

    def test_query_key_loopback_allowed(self):
        with mock.patch.object(deps, "_LOOPBACK_HOSTS", _LOOPBACK):
            r = self.c.get("/v1/config?api_key=" + _KEY)
        self.assertEqual(r.status_code, 200)

    def test_query_key_wrong_rejected(self):
        with mock.patch.object(deps, "_LOOPBACK_HOSTS", _LOOPBACK):
            r = self.c.get("/v1/config?api_key=wrong")
        self.assertEqual(r.status_code, 401)

    def test_query_key_non_loopback_rejected(self):
        r = self.c.get("/v1/config?api_key=" + _KEY)   # testclient 不在回环集合
        self.assertEqual(r.status_code, 401)

    def test_header_key_always_works(self):
        r = self.c.get("/v1/config", headers={"X-XCcv-Key": _KEY})
        self.assertEqual(r.status_code, 200)


class FlowRunSyncTest(unittest.TestCase):
    """/v1/flow/run 必须是同步 def（推理在线程池，不阻事件循环）。"""

    def test_flow_run_not_coroutine(self):
        self.assertFalse(inspect.iscoroutinefunction(flow_run))


class MaxSideTest(unittest.TestCase):
    """降采样推理+坐标还原。"""

    def test_downscale_and_restore(self):
        app = create_app(ServiceConfig(api_key=_KEY))
        m = _register(app, _FakeModel())
        big = np.zeros((480, 640, 3), np.uint8)
        pred, entry = app.state.manager.infer("t-1", big, 0.25, 0.45,
                                              max_side=320)
        self.assertEqual(max(m.last_image_hw), 320, "应降采样到 320")
        x1, _, x2, _ = pred.boxes[0]
        self.assertAlmostEqual(x1, 640 * 0.25, places=0)
        self.assertAlmostEqual(x2, 640 * 0.5, places=0)

    def test_meta_train_input_automatic(self):
        app = create_app(ServiceConfig(api_key=_KEY))
        m = _register(app, _FakeModel(train_input=160))
        big = np.zeros((480, 640, 3), np.uint8)
        app.state.manager.infer("t-1", big, 0.25, 0.45)
        self.assertEqual(max(m.last_image_hw), 160, "meta.train_input 生效")

    def test_small_image_no_scaling(self):
        app = create_app(ServiceConfig(api_key=_KEY))
        m = _register(app, _FakeModel(train_input=1024))
        small = np.zeros((64, 64, 3), np.uint8)
        app.state.manager.infer("t-1", small, 0.25, 0.45)
        self.assertEqual(m.last_image_hw, (64, 64), "小图不缩放")


class PreloadTest(unittest.TestCase):
    """min_loaded_models 预载（失败不阻断服务）。"""

    def test_preload_failure_tolerated(self):
        cfg = ServiceConfig(api_key=_KEY,
                            min_loaded_models=["no_such_model.pt"])
        app = create_app(cfg)
        with TestClient(app) as c:   # with 触发 lifespan（含预载）
            r = c.get("/v1/health")
        self.assertEqual(r.status_code, 200)

    def test_config_defaults(self):
        cfg = ServiceConfig()
        self.assertTrue(cfg.timing_enabled)
        self.assertEqual(cfg.max_infer_side, 0)
        self.assertEqual(cfg.min_loaded_models, [])
        self.assertFalse(cfg.access_log)
        self.assertEqual(cfg.log_level, "info")


class WsInferTest(unittest.TestCase):
    """WebSocket 二进制通道往返与 HTTP 同结果。"""

    def test_ws_roundtrip_matches_http(self):
        app = create_app(ServiceConfig(api_key=_KEY))
        m = _register(app)
        c = TestClient(app)
        png = _png_bytes()
        with mock.patch.object(deps, "_LOOPBACK_HOSTS", _LOOPBACK):
            with c.websocket_connect("/v1/ws/infer") as ws:
                ws.send_text('{"api_key": "%s", "model_id": "t-1"}' % _KEY)
                ack = ws.receive_json()
                self.assertTrue(ack.get("ok"))
                ws.send_bytes(png)
                msg = ws.receive_json()
        self.assertIsNone(msg.get("error"))
        r = c.post("/v1/infer/image", files={"image": ("a.png", png,
                                                       "image/png")},
                   data={"model_id": "t-1"},
                   headers={"X-XCcv-Key": _KEY})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(msg.get("detections", [])),
                         len(r.json().get("detections", [])))

    def test_ws_bad_key_closed(self):
        app = create_app(ServiceConfig(api_key=_KEY))
        c = TestClient(app)
        with c.websocket_connect("/v1/ws/infer") as ws:
            ws.send_text('{"api_key": "bad"}')
            with self.assertRaises(Exception):
                ws.receive_json()


if __name__ == "__main__":
    unittest.main()

