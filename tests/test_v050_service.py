"""v0.5 推理服务测试：健康检查 / Key 认证 / 模型加载推理（真实小模型权重）/
统一 JSON 契约（schema_version=1 + verdict）/ 流程运行 / 429 准入 / 白名单 / LRU。

模型夹具：create_model 随机初始化（不训练）→ 明文 .pt 落盘，
走 registry.load_model 完整加载链路（与生产同语义）。
"""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from xccv.service.config import ServiceConfig
from xccv.service.main import create_app

_KEY = "test-key-050"


def _cfg(tmp: Path, **kw) -> ServiceConfig:
    """测试专用配置：注入临时白名单目录，绝不读/写真实 platform_data 配置。"""
    return ServiceConfig(host="127.0.0.1", port=8600, api_key=_KEY,
                         model_dirs=[str(tmp)], allow_any_path=False,
                         max_loaded_models=kw.pop("max_loaded_models", 3),
                         queue_size=kw.pop("queue_size", 4), **kw)


def _make_pt(dir_: Path, classes: list) -> Path:
    """随机初始化 resnet18 → 明文 .pt 权重夹具。"""
    import torch

    from xccv.engine.dataset.base import TaskType
    from xccv.engine.models.registry import create_model

    m = create_model("resnet18", TaskType.CLASSIFICATION, len(classes),
                     device="cpu", use_pretrained=False)
    ckpt = {"format": "xccv-classifier", "arch": "resnet18",
            "task": "classification", "classes": classes,
            "industrial_opt": False, "image_size": 160,
            "state_dict": {k: v.detach().cpu() for k, v in m.net.state_dict().items()}}
    p = dir_ / f"cls{len(classes)}.pt"
    torch.save(ckpt, p)
    return p


def _client(cfg: ServiceConfig):
    from fastapi.testclient import TestClient

    return TestClient(create_app(cfg)), cfg


_H = {"X-XCcv-Key": _KEY}


class ServiceHealthAuthTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client, _ = _client(_cfg(Path(self._tmp.name)))

    def test_health_open_without_key(self):
        r = self.client.get("/v1/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertEqual(r.json()["api_version"], 1)

    def test_business_requires_key(self):
        r = self.client.get("/v1/models/status")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["code"], "error_401")
        r = self.client.get("/v1/models/status", headers={"X-XCcv-Key": "wrong"})
        self.assertEqual(r.status_code, 401)
        r = self.client.get("/v1/models/status", headers=_H)
        self.assertEqual(r.status_code, 200)

    def test_error_envelope_shape(self):
        r = self.client.post("/v1/models/unload", headers=_H,
                             json={"model_id": "nope-1"})
        self.assertEqual(r.status_code, 404)
        body = r.json()
        self.assertEqual(set(body), {"code", "message", "detail"})


class ServiceInferContractTest(unittest.TestCase):
    """真实小模型：加载 → 推理 → 统一 JSON 契约（verdict 键恒存在）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.pt = _make_pt(cls.root, ["OK", "NG"])
        cls.client, _ = _client(_cfg(cls.root))
        r = cls.client.post("/v1/models/load", headers=_H,
                            json={"path": str(cls.pt), "device": "cpu",
                                  "alias": "演示模型", "warmup": True})
        assert r.status_code == 200, r.text
        cls.model_id = r.json()["model_id"]
        cls.task = r.json()["task"]
        # 测试图：纯色 64x64（分类输出任意，契约才是断言点）
        cls.png = cv2.imencode(".png", np.full((64, 64, 3), 128, np.uint8))[1].tobytes()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_load_response_shape(self):
        r = self.client.get("/v1/models/status", headers=_H)
        models = r.json()["models"]
        self.assertEqual(len(models), 1)
        m = models[0]
        self.assertEqual(m["task"], "classification")
        self.assertEqual(m["alias"], "演示模型")
        self.assertTrue(m["model_id"])

    def test_infer_unified_json(self):
        r = self.client.post("/v1/infer/image", headers=_H,
                             files={"image": ("测试图.png", self.png, "image/png")},
                             data={"model_id": self.model_id, "conf": "0.1"})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        # 统一契约：schema/键集/verdict 恒存在
        self.assertEqual(j["schema_version"], 1)
        self.assertEqual(j["task"], "classification")
        self.assertIn("verdict", j)
        self.assertIsNone(j["verdict"])          # 无判定流程 → null
        self.assertEqual(j["source"]["file"], "测试图.png")
        self.assertEqual(j["model"]["name"], "演示模型")
        self.assertIn("detections", j)
        self.assertIn("classifications", j)      # 分类 Top-K 有输出
        self.assertGreaterEqual(len(j["classifications"]), 1)
        c = j["classifications"][0]
        self.assertEqual(set(c), {"label", "label_id", "score"})
        self.assertIn(c["label"], ("OK", "NG"))

    def test_infer_wrong_model_404(self):
        r = self.client.post("/v1/infer/image", headers=_H,
                             files={"image": ("a.png", self.png, "image/png")},
                             data={"model_id": "ghost-9"})
        self.assertEqual(r.status_code, 404)

    def test_bad_image_400(self):
        r = self.client.post("/v1/infer/image", headers=_H,
                             files={"image": ("a.png", b"not-an-image", "image/png")},
                             data={"model_id": self.model_id})
        self.assertEqual(r.status_code, 400)

    def test_render_overlay_png(self):
        r = self.client.post("/v1/render/overlay", headers=_H,
                             files={"image": ("a.png", self.png, "image/png")},
                             data={"model_id": self.model_id})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")
        self.assertEqual(r.content[:8], b"\x89PNG\r\n\x1a\n")

    def test_unload_then_404(self):
        r = self.client.post("/v1/models/unload", headers=_H,
                             json={"model_id": self.model_id})
        self.assertTrue(r.json()["ok"])
        r = self.client.post("/v1/infer/image", headers=_H,
                             files={"image": ("a.png", self.png, "image/png")},
                             data={"model_id": self.model_id})
        self.assertEqual(r.status_code, 404)


class ServiceFlowRunTest(unittest.TestCase):
    """/v1/flow/run：上传流程+图片跑全流程（blob→数量判定 verdict=passed）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        img = np.zeros((60, 80, 3), np.uint8)
        cv2.circle(img, (40, 30), 12, 255, -1)
        from xccv.engine.io_utils import imwrite_unicode; imwrite_unicode(cls.root / "样本.png", img)  # P147
        cls.img_bytes = (cls.root / "样本.png").read_bytes()
        cls.flow = {"schema": 3, "nodes": [   # 程序内新建图必须 stamp 当前代数
            {"id": 1, "type": "xccv2/image_loader", "properties": {}},
            {"id": 2, "type": "xccv2/blob_analysis",
             "properties": {"threshold_mode": "otsu", "thresh": 128,
                            "invert": False, "min_area": 10, "max_area": 0,
                            "circularity": 0.0, "label_name": "缺陷"}},
            {"id": 3, "type": "xccv2/filter_count_check",
             "properties": {"min": 1, "max": 0}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
        cls.client, _ = _client(_cfg(cls.root))
        (cls.root / "流程.json").write_text(
            json.dumps(cls.flow, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_flow_run_upload_with_verdict(self):
        r = self.client.post(
            "/v1/flow/run", headers=_H,
            files={"flow": ("流程.json",
                            json.dumps(self.flow, ensure_ascii=False).encode(),
                            "application/json"),
                   "image": ("样本.png", self.img_bytes, "image/png")})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["schema_version"], 1)
        self.assertEqual(j["task"], "flow")
        self.assertGreaterEqual(len(j["detections"]), 1)
        self.assertEqual(j["detections"][0]["label"], "缺陷")
        # verdict：filter_count_check(min=1, 实际 1) → 判定通过
        self.assertIsNotNone(j["verdict"])
        self.assertTrue(j["verdict"]["passed"])
        self.assertEqual(j["verdict"]["source"], "count_check")

    def test_flow_run_file_path_mode(self):
        r = self.client.post("/v1/flow/run/file", headers=_H,
                             data={"flow_path": str(self.root / "流程.json"),
                                   "image_path": str(self.root / "样本.png")})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["task"], "flow")

    def test_flow_missing_loader_400(self):
        bad = {"nodes": [{"id": 1, "type": "xccv2/blob_analysis",
                          "properties": {}}], "links": []}
        r = self.client.post(
            "/v1/flow/run", headers=_H,
            files={"flow": ("bad.json", json.dumps(bad).encode(),
                            "application/json"),
                   "image": ("a.png", self.img_bytes, "image/png")})
        self.assertEqual(r.status_code, 400)

    def test_flow_run_render_png(self):
        r = self.client.post(
            "/v1/flow/run", headers=_H,
            files={"flow": ("流程.json",
                            json.dumps(self.flow, ensure_ascii=False).encode(),
                            "application/json"),
                   "image": ("样本.png", self.img_bytes, "image/png")},
            data={"render": "true"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")

    def test_path_whitelist_403(self):
        r = self.client.post("/v1/flow/run/file", headers=_H,
                             data={"flow_path": r"E:\其他目录\流程.json",
                                   "image_path": str(self.root / "样本.png")})
        self.assertEqual(r.status_code, 403)


class Gate429Test(unittest.TestCase):
    """推理准入闸：队列满直接 429（不阻塞连接）。"""

    def test_infer_429_when_gate_full(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        client, _test_cfg = _client(_cfg(Path(self._tmp.name)))
        # 占满闸门（模拟正在执行的 GPU 推理）
        gate = threading.BoundedSemaphore(1)
        gate.acquire()
        client.app.state.gate = gate
        r = client.post("/v1/infer/image", headers=_H,
                        files={"image": ("a.png", b"x", "image/png")},
                        data={"model_id": "m-1"})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["code"], "error_429")


class LruEvictionTest(unittest.TestCase):
    """模型池 LRU：超过驻留上限自动卸载最久未用。"""

    def test_evicts_oldest(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        a, b, c = (_make_pt(root, n) for n in (["A"], ["B", "C"], ["D", "E", "F"]))
        client, _ = _client(_cfg(root, max_loaded_models=2))
        ids = [client.post("/v1/models/load", headers=_H,
                           json={"path": str(p), "device": "cpu"}).json()["model_id"]
               for p in (a, b, c)]
        status = client.get("/v1/models/status", headers=_H).json()["models"]
        live = {m["model_id"] for m in status}
        self.assertEqual(len(live), 2)
        self.assertNotIn(ids[0], live)      # 最先加载的被 LRU 淘汰
        self.assertIn(ids[2], live)


class DocsAndAuthSchemeTest(unittest.TestCase):
    """ToA3//docs 离线自托管 + API Key 进 OpenAPI（Swagger Authorize）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client, _ = _client(_cfg(Path(self._tmp.name)))

    def test_docs_served_with_local_assets(self):
        r = self.client.get("/docs")
        self.assertEqual(r.status_code, 200)
        self.assertIn("/docs-static/swagger-ui-bundle.js", r.text)
        a = self.client.get("/docs-static/swagger-ui-bundle.js")
        self.assertEqual(a.status_code, 200)
        self.assertGreater(len(a.content), 100000)
        c = self.client.get("/docs-static/swagger-ui.css")
        self.assertEqual(c.status_code, 200)

    def test_api_key_in_openapi_security_schemes(self):
        r = self.client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        spec = r.json()
        self.assertIn("securitySchemes", spec["components"])
        schemes = spec["components"]["securitySchemes"]
        self.assertTrue(any("X-XCcv-Key" in str(s) for s in schemes.values()))


class FlowManagerApiTest(unittest.TestCase):
    """流程包常驻 API（load → run by id → status → unload）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        img = np.zeros((60, 80, 3), np.uint8)
        cv2.circle(img, (40, 30), 12, 255, -1)
        cls.img_path = cls.root / "样本.png"
        from xccv.engine.io_utils import imwrite_unicode  # P147

        imwrite_unicode(cls.img_path, img)
        cls.flow = {"schema": 3, "nodes": [   # 程序内新建图必须 stamp 当前代数
            {"id": 1, "type": "xccv2/image_loader", "properties": {}},
            {"id": 2, "type": "xccv2/blob_analysis",
             "properties": {"threshold_mode": "otsu", "thresh": 128,
                            "invert": False, "min_area": 10, "max_area": 0,
                            "circularity": 0.0, "label_name": "缺陷"}},
            {"id": 3, "type": "xccv2/filter_count_check",
             "properties": {"min": 1, "max": 0}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
        (cls.root / "检测流程.json").write_text(
            json.dumps(cls.flow, ensure_ascii=False), encoding="utf-8")
        cls.client, _ = _client(_cfg(cls.root))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_flow_lifecycle(self):
        # load
        r = self.client.post("/v1/flows/load", headers=_H,
                             json={"path": str(self.root / "检测流程.json"),
                                   "alias": "斑点检测"})
        self.assertEqual(r.status_code, 200, r.text)
        fid = r.json()["flow_id"]
        self.assertTrue(fid.startswith("flow-"))
        # run by id（multipart 上传图）
        r = self.client.post("/v1/flows/run", headers=_H,
                             data={"flow_id": fid},
                             files={"image": ("样本.png", self.img_path.read_bytes())})
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(j["task"], "flow")
        self.assertGreaterEqual(len(j["detections"]), 1)
        self.assertTrue(j["verdict"]["passed"])
        # run by id（本地图路径 + render PNG）
        r = self.client.post("/v1/flows/run", headers=_H,
                             data={"flow_id": fid,
                                   "image_path": str(self.img_path),
                                   "render": "true"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")
        # status
        r = self.client.get("/v1/flows/status", headers=_H)
        self.assertEqual(len(r.json()["flows"]), 1)
        # 幂等：同路径再 load 不新增
        self.client.post("/v1/flows/load", headers=_H,
                         json={"path": str(self.root / "检测流程.json")})
        r = self.client.get("/v1/flows/status", headers=_H)
        self.assertEqual(len(r.json()["flows"]), 1)
        # unload
        r = self.client.post("/v1/flows/unload", headers=_H,
                             json={"model_id": fid})
        self.assertTrue(r.json()["ok"])
        r = self.client.post("/v1/flows/run", headers=_H,
                             data={"flow_id": fid},
                             files={"image": ("样本.png", self.img_path.read_bytes())})
        self.assertEqual(r.status_code, 404)

    def test_flow_load_missing_404_and_bad_suffix_400(self):
        r = self.client.post("/v1/flows/load", headers=_H,
                             json={"path": str(self.root / "无.json")})
        self.assertEqual(r.status_code, 404)
        bad = self.root / "x.txt"
        bad.write_text("x", encoding="utf-8")
        r = self.client.post("/v1/flows/load", headers=_H,
                             json={"path": str(bad)})
        self.assertEqual(r.status_code, 400)

    def test_flow_run_without_image_400(self):
        r = self.client.post("/v1/flows/load", headers=_H,
                             json={"path": str(self.root / "检测流程.json")})
        fid = r.json()["flow_id"]
        r = self.client.post("/v1/flows/run", headers=_H,
                             data={"flow_id": fid})
        self.assertEqual(r.status_code, 400)


class VerdictHelperTest(unittest.TestCase):
    """verdict 工具：count_check 信息与节点结果两条路径。"""

    def test_from_count_check(self):
        from xccv.engine.result_json import verdict_from_count_check

        self.assertIsNone(verdict_from_count_check({}))
        self.assertIsNone(verdict_from_count_check({"count_check": None}))
        v = verdict_from_count_check({"count_check": "NG"})
        self.assertEqual(v, {"passed": False, "source": "count_check",
                             "detail": "数量判定 NG"})

    def test_auto_fill_in_build_result_json(self):
        from xccv.engine.result_json import build_result_json

        j = build_result_json(task="detection",
                              source={"file": "a.png", "path": ""},
                              flow_info={"removed": 1, "count_check": "OK"})
        self.assertTrue(j["verdict"]["passed"])
        j2 = build_result_json(task="detection", source={"file": "a.png", "path": ""})
        self.assertIsNone(j2["verdict"])

    def test_from_node_results_and_gate(self):
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext
        from xccv.engine.result_json import verdict_from_node_results

        g = {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {}},
            {"id": 2, "type": "blob_analysis",
             "properties": {"threshold_mode": "otsu", "thresh": 128,
                            "invert": False, "min_area": 10, "max_area": 0,
                            "circularity": 0.0, "label_name": "缺陷"}},
            {"id": 3, "type": "filter_count_check",
             "properties": {"min": 1, "max": 0}}],
            "links": [[1, 1, 0, 2, 0], [2, 2, 1, 3, 1]]}   # 图像在上，result 槽 1
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.png"
            cv2.imwrite(str(p), np.full((30, 30), 60, np.uint8))
            g["nodes"][0]["properties"] = {"path": str(p)}
            ctx = RunContext()
            results = Executor(g, ctx).run()
        v = verdict_from_node_results(results)
        self.assertIsNotNone(v)
        self.assertTrue(v["passed"])
        # 无判定节点 → None
        self.assertIsNone(verdict_from_node_results({}))


if __name__ == "__main__":
    unittest.main()
