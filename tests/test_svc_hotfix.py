"""部署服务测试：1 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v081_hotfix
"""


# == 来源模块：test_v081_hotfix ==
# -*- coding: utf-8 -*-
"""V0.8.1 止血批契约测试：鉴权 / 配置字段白名单 / 上传限额 / conf-iou 回显 /
params_mb / 降采样 OCR texts / 模型池并发 / 评估 NMS 口径 / defectgen 目录白名单。

口径：不启动真实服务器（TestClient 进程内）、不依赖 GPU 与真实权重、
不读写真实 platform_data 配置（defectgen 注册表重定向临时目录）。
"""

import threading
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _cfg(**over):
    from xccv.service.config import ServiceConfig

    base = dict(host="127.0.0.1", port=8601, api_key="v081-key",
                model_dirs=[], allow_any_path=False, max_loaded_models=2,
                queue_size=2)
    base.update(over)
    return ServiceConfig(**base)


def _client(cfg=None):
    from fastapi.testclient import TestClient

    from xccv.service.main import create_app

    app = create_app(cfg or _cfg())
    app.state.server = None      # 测试态无真实 uvicorn server 可停
    return TestClient(app)


def _key(client):
    return client.get("/v1/models/status", headers={"X-XCcv-Key": "v081-key"})


class ShutdownAuthTest(unittest.TestCase):
    """/v1/admin/shutdown 必须挂 Key 依赖；confirm 语义不变。"""

    @classmethod
    def setUpClass(cls):
        cls.client = _client()

    def test_no_key_rejected(self):
        r = self.client.post("/v1/admin/shutdown", json={"confirm": True})
        self.assertEqual(r.status_code, 401, "停服端点无 Key 必须拒绝")

    def test_wrong_key_rejected(self):
        r = self.client.post("/v1/admin/shutdown", json={"confirm": True},
                             headers={"X-XCcv-Key": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_confirm_false_is_400(self):
        r = self.client.post("/v1/admin/shutdown", json={"confirm": False},
                             headers={"X-XCcv-Key": "v081-key"})
        self.assertEqual(r.status_code, 400)

    def test_confirm_with_key_ok(self):
        r = self.client.post("/v1/admin/shutdown", json={"confirm": True},
                             headers={"X-XCcv-Key": "v081-key"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json().get("ok"))


class ConfigWhitelistTest(unittest.TestCase):
    """/v1/config 字段白名单：凭据与文件系统布局不下发。"""

    @classmethod
    def setUpClass(cls):
        cls.client = _client(_cfg(model_dirs=[r"E:\models"], allow_any_path=True))

    def test_sensitive_fields_absent(self):
        r = self.client.get("/v1/config", headers={"X-XCcv-Key": "v081-key"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for leak in ("api_key", "model_dirs", "allow_any_path"):
            self.assertNotIn(leak, body, f"{leak} 不应出现在 /v1/config")
        for keep in ("host", "port", "max_loaded_models", "queue_size",
                     "infer_conf", "infer_iou", "max_batch", "max_pixels",
                     "max_upload_mb"):
            self.assertIn(keep, body)

    def test_still_requires_key(self):
        r = self.client.get("/v1/config")
        self.assertEqual(r.status_code, 401)


class UploadLimitTest(unittest.TestCase):
    """批量张数 / 单图像素 / 请求体体积三重护栏。"""

    def test_body_over_limit_helper(self):
        from xccv.service.main import body_over_limit

        self.assertFalse(body_over_limit(None, 100.0))
        self.assertFalse(body_over_limit("", 100.0))
        self.assertFalse(body_over_limit("abc", 100.0))
        self.assertFalse(body_over_limit(str(100 * 1024 * 1024), 100.0))
        self.assertTrue(body_over_limit(str(100 * 1024 * 1024 + 1), 100.0))

    def test_decode_image_pixel_cap(self):
        import cv2
        import numpy as np
        from fastapi import HTTPException

        from xccv.service.api.routes_infer import decode_image

        ok, buf = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))
        assert ok
        data = buf.tobytes()
        img = decode_image(data, max_pixels=1_000_000)
        self.assertEqual(img.shape[:2], (20, 20))
        with self.assertRaises(HTTPException) as ctx:
            decode_image(data, max_pixels=100)
        self.assertEqual(ctx.exception.status_code, 413)

    def test_batch_count_cap(self):
        import cv2
        import numpy as np

        client = _client(_cfg(max_batch=2))
        ok, buf = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
        assert ok
        data = buf.tobytes()
        files = [("images", (f"a{i}.png", data, "image/png")) for i in range(3)]
        r = client.post("/v1/infer/batch", files=files,
                        data={"model_id": "m"},
                        headers={"X-XCcv-Key": "v081-key"})
        self.assertEqual(r.status_code, 400, "超过 max_batch 必须 400")
        self.assertIn("max_batch", r.json()["message"])

    def test_single_image_pixel_413(self):
        import cv2
        import numpy as np

        client = _client(_cfg(max_pixels=100))
        ok, buf = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))
        assert ok
        r = client.post("/v1/infer/image",
                        files={"image": ("a.png", buf.tobytes(), "image/png")},
                        data={"model_id": "m"},
                        headers={"X-XCcv-Key": "v081-key"})
        self.assertEqual(r.status_code, 413)


class ConfIouEchoTest(unittest.TestCase):
    """结果 JSON 的 conf/iou 回显实际生效值（未传参=默认值补齐）。"""

    @classmethod
    def setUpClass(cls):
        cls.client = _client(_cfg(infer_conf=0.25, infer_iou=0.20))

    @staticmethod
    def _install_stub(app):
        entry = SimpleNamespace(model_id="m", alias="stub", path="stub.pt",
                                task="detection", classes=["A"],
                                model=SimpleNamespace(score_semantics={}),
                                train_input=0)
        pred = SimpleNamespace(labels=[], scores=[], boxes=[],
                               masks=None, probs=None, texts=None)
        mgr = SimpleNamespace(infer=lambda mid, img, conf, iou, max_side=0:
                              (pred, entry))
        app.state.manager = mgr
        return mgr

    def _post(self, **data):
        import cv2
        import numpy as np

        ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
        assert ok
        return self.client.post(
            "/v1/infer/image",
            files={"image": ("a.png", buf.tobytes(), "image/png")},
            data={"model_id": "m", **data},
            headers={"X-XCcv-Key": "v081-key"})

    def test_defaults_are_echoed(self):
        app = self.client.app
        self._install_stub(app)
        r = self._post()
        self.assertEqual(r.status_code, 200)
        inf = r.json()["inference"]
        self.assertAlmostEqual(inf["conf"], 0.25)
        self.assertAlmostEqual(inf["iou"], 0.20)

    def test_explicit_params_are_echoed(self):
        self._install_stub(self.client.app)
        r = self._post(conf=0.5, iou=0.7)
        self.assertEqual(r.status_code, 200)
        inf = r.json()["inference"]
        self.assertAlmostEqual(inf["conf"], 0.5)
        self.assertAlmostEqual(inf["iou"], 0.7)


class ParamsMbTest(unittest.TestCase):
    """_estimate_params_mb：包装类解包取真实参数量，state_dict 兜底。"""

    def test_unwrap_chain(self):
        import torch

        from xccv.service.core.model_manager import _estimate_params_mb

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(1024, 512)

        class Mid:                       # 无 parameters 的中间包装（net 属性）
            def __init__(self, core):
                self.net = core

        class Wrap:                      # 平台包装（model 属性，parameters 抛错）
            def __init__(self, mid):
                self.model = mid

            def parameters(self):
                raise AttributeError("wrapped")

        n = (1024 * 512 + 512) * 4 / 1e6
        self.assertAlmostEqual(_estimate_params_mb(Wrap(Mid(Core()))),
                               round(n, 1), places=1)

    def test_state_dict_fallback(self):
        import torch

        from xccv.service.core.model_manager import _estimate_params_mb

        class OnlySD:
            def parameters(self):
                raise AttributeError("no")

            def state_dict(self):
                return {"w": torch.zeros(1_000_000)}

        self.assertAlmostEqual(_estimate_params_mb(OnlySD()), 4.0, places=1)

    def test_failure_returns_zero(self):
        from xccv.service.core.model_manager import _estimate_params_mb

        class Nada:
            pass

        self.assertEqual(_estimate_params_mb(Nada()), 0.0)


class RestoreTextsTest(unittest.TestCase):
    """降采样推理结果还原必须透传 OCR texts。"""

    def test_scaled_path_keeps_texts(self):
        import numpy as np

        from xccv.service.core.model_manager import _restore_prediction

        orig = np.zeros((20, 10, 3), np.uint8)
        scaled = np.zeros((10, 5, 3), np.uint8)
        pred = SimpleNamespace(labels=[0], scores=[0.9],
                               boxes=[[1.0, 1.0, 3.0, 3.0]], masks=None,
                               probs=None, texts=["hello"])
        out = _restore_prediction(pred, orig, scaled)
        self.assertEqual(out.texts, ["hello"], "texts 不能在还原时丢失")
        self.assertEqual(out.boxes, [[2.0, 2.0, 6.0, 6.0]], "框按 2x 线性还原")

    def test_same_size_passthrough(self):
        import numpy as np

        from xccv.service.core.model_manager import _restore_prediction

        img = np.zeros((6, 6, 3), np.uint8)
        pred = SimpleNamespace(labels=[], scores=[], boxes=[], masks=None,
                               probs=None, texts=["x"])
        self.assertIs(_restore_prediction(pred, img, img), pred)


class _StubModel:
    task = None
    classes = None
    train_input = 0

    def __init__(self, predict_s: float = 0.0):
        self._predict_s = predict_s
        self.predict_calls = 0
        self._call_lock = threading.Lock()

    def predict(self, img, conf=0.25, iou=0.45):
        with self._call_lock:
            self.predict_calls += 1
        if self._predict_s:
            time.sleep(self._predict_s)
        return SimpleNamespace(labels=[], scores=[], boxes=[], masks=None,
                               probs=None, texts=None)


class PoolConcurrencyTest(unittest.TestCase):
    """模型池并发三修：per-path 单飞 / 卸载与推理互斥 / last_used 锁内更新。"""

    def test_same_path_single_flight(self):
        from xccv.service.core.model_manager import ModelManager

        calls = []
        lock = threading.Lock()

        def fake_load(path, device="auto"):
            with lock:
                calls.append(str(path))
            time.sleep(0.2)
            return _StubModel()

        mgr = ModelManager(max_loaded=4)
        with mock.patch("xccv.engine.models.registry.load_model", fake_load):
            results = []
            errors = []

            def worker():
                try:
                    results.append(mgr.load(Path("m.pt"), device="cpu"))
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            t1 = threading.Thread(target=worker)
            t1.start()
            time.sleep(0.05)         # 让 t1 进入真实加载
            t2 = threading.Thread(target=worker)
            t2.start()
            t1.join(5)
            t2.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1, "同路径并发加载必须单飞（只 load 一次）")
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], results[1], "并发方应复用同一句柄")

    def test_unload_waits_for_inflight_infer(self):
        import numpy as np

        from xccv.service.core.model_manager import ModelManager

        stub = _StubModel(predict_s=0.25)
        mgr = ModelManager(max_loaded=4)
        with mock.patch("xccv.engine.models.registry.load_model",
                        lambda path, device="auto": stub):
            entry = mgr.load(Path("m.pt"), device="cpu")
        infer_err = []

        def infer_worker():
            try:
                mgr.infer(entry.model_id, np.zeros((8, 8, 3), np.uint8),
                          0.25, 0.2)
            except Exception as e:  # noqa: BLE001
                infer_err.append(e)

        t1 = threading.Thread(target=infer_worker)
        t1.start()
        time.sleep(0.05)             # 推理已持有 entry.lock
        done = mgr.unload(entry.model_id)
        t1.join(5)
        self.assertEqual(infer_err, [], "卸载不得打断 in-flight 推理")
        self.assertTrue(done)
        self.assertEqual(mgr.list(), [], "推理结束后模型应已卸载")


class EvalNmsGuardTest(unittest.TestCase):
    """评估 NMS 口径：覆写 roi_heads.nms_thresh（非 box_nms_thresh）且进出恢复。"""

    def test_guard_overwrites_and_restores(self):
        from xccv.engine.models.rcnn import eval_nms_guard

        net = SimpleNamespace(roi_heads=SimpleNamespace(nms_thresh=0.7))
        with eval_nms_guard(net, 0.2):
            self.assertAlmostEqual(net.roi_heads.nms_thresh, 0.2)
        self.assertAlmostEqual(net.roi_heads.nms_thresh, 0.7, "退出必须恢复原值")
        self.assertFalse(hasattr(net.roi_heads, "box_nms_thresh"),
                         "不得再写入无效属性 box_nms_thresh")

    def test_guard_restores_when_body_raises(self):
        from xccv.engine.models.rcnn import eval_nms_guard

        net = SimpleNamespace(roi_heads=SimpleNamespace(nms_thresh=0.45))
        with self.assertRaises(RuntimeError):
            with eval_nms_guard(net, 0.2):
                raise RuntimeError("boom")
        self.assertAlmostEqual(net.roi_heads.nms_thresh, 0.45)


class DefectgenWhitelistTest(unittest.TestCase):
    """defectgen 批量作业目录白名单 + 作业表治理。"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        import xccv.defectgen.backend.app as dg

        cls.dg = dg
        cls.tmp = Path(tempfile.mkdtemp(prefix="dg_allow_"))
        dg._REG_OVERRIDE = str(cls.tmp / "allowed_dirs.json")
        cls.client = TestClient(dg.app)

    @classmethod
    def tearDownClass(cls):
        cls.dg._REG_OVERRIDE = None

    def _batch_payload(self, output_dir, background=None):
        return {"assets": {}, "instances": [],
                "background": background or {"mode": "files",
                                             "files": [{"name": "a.png",
                                                        "b64": "AAAA"}]},
                "output_dir": output_dir, "format": "png"}

    def _outside_dir(self) -> str:
        """平台无关的「存在且在白名单外」目录。

        白名单默认根是仓库内 platform_data；系统临时目录根在其外且恒存在。
        不用 E:\\ 这类平台字面路径——POSIX 上它不是绝对路径，会被相对解析
        进白名单根，语义漂移（ubuntu CI 实测）。
        """
        return tempfile.gettempdir()

    def test_output_outside_whitelist_rejected(self):
        r = self.client.post("/api/batch/start",
                             json=self._batch_payload(self._outside_dir()))
        self.assertEqual(r.status_code, 400)
        self.assertIn("授权白名单", r.json()["detail"])

    def test_background_dir_outside_whitelist_rejected(self):
        r = self.client.post("/api/batch/start", json=self._batch_payload(
            "output", background={"mode": "dir", "dir": self._outside_dir()}))
        self.assertEqual(r.status_code, 400)
        self.assertIn("授权白名单", r.json()["detail"])

    def test_allow_dir_then_start_ok(self):
        allowed = self.tmp / "work"
        allowed.mkdir()
        r = self.client.post("/api/allow_dir", json={"dir": str(allowed)})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])
        r2 = self.client.post("/api/batch/start",
                              json=self._batch_payload(str(allowed / "out")))
        self.assertEqual(r2.status_code, 200)
        self.assertIn("job_id", r2.json())

    def test_allow_nonexistent_dir_rejected(self):
        r = self.client.post("/api/allow_dir",
                             json={"dir": str(self.tmp / "no_such_dir")})
        self.assertEqual(r.status_code, 400)

    def test_platform_data_default_allowed(self):
        import cv2
        import numpy as np

        # platform_data 默认根无需登记即可作输出目录（相对路径语义不变）
        ok, buf = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
        assert ok
        r = self.client.post("/api/batch/start", json=self._batch_payload(
            "t081_probe_out",
            background={"mode": "files",
                        "files": [{"name": "a.png",
                                   "b64": "iVBORw0KGgoAAAANSUhEUg=="}]}))
        self.assertEqual(r.status_code, 200)

    def test_running_cap(self):
        with mock.patch.object(self.dg, "MAX_RUNNING_JOBS", 0):
            r = self.client.post("/api/batch/start",
                                 json=self._batch_payload("output"))
        self.assertEqual(r.status_code, 429)

    def test_ttl_purge(self):
        jid = "t" + "0" * 11
        with self.dg._JOBS_LOCK:
            self.dg._JOBS[jid] = {"total": 0, "done_count": 0, "results": [],
                                  "done": True, "cancel": False,
                                  "cancelled": False, "error": None,
                                  "summary": None,
                                  "t0": time.time() - 3 * 3600}
            self.dg._purge_jobs_locked()
            self.assertNotIn(jid, self.dg._JOBS, "超 TTL 已完成作业应被清理")

if __name__ == '__main__':
    unittest.main()
