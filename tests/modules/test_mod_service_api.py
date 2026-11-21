"""模块行为锁：推理服务 HTTP 契约（健康检查 / Key 认证 / 配置注入）。

不启动真实服务器（TestClient 进程内），配置注入临时目录、
绝不读写真实 platform_data 配置（同 tests/test_v050_service 口径）。
"""

import tempfile
import unittest
from pathlib import Path


def _cfg():
    from xccv.service.config import ServiceConfig

    return ServiceConfig(host="127.0.0.1", port=8601,
                         api_key="mod-lock-key", model_dirs=[],
                         allow_any_path=False, max_loaded_models=2,
                         queue_size=2)


class ServiceContractTest(unittest.TestCase):
    """HTTP 契约冒烟：/v1/health 无需 Key，业务端点需 Key。"""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from xccv.service.main import create_app

        cls.client = TestClient(create_app(_cfg()))

    def test_health_open_and_contract(self):
        r = self.client.get("/v1/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("status", body)

    def test_models_status_requires_key(self):
        r = self.client.get("/v1/models/status")
        self.assertEqual(r.status_code, 401, "业务端点无 Key 必须拒绝")
        r2 = self.client.get("/v1/models/status",
                             headers={"X-XCcv-Key": "mod-lock-key"})
        self.assertEqual(r2.status_code, 200)
        self.assertIn("models", r2.json())

    def test_bad_key_rejected(self):
        r = self.client.get("/v1/models/status",
                            headers={"X-XCcv-Key": "wrong"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()

