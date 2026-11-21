"""路由依赖：Key 认证、配置/模型池注入、推理准入闸。"""

import threading

from fastapi import Depends, HTTPException, Query, Request
from fastapi.security import APIKeyHeader

from ..config import _LOOPBACK_HOSTS
from ..security import verify_key

# APIKeyHeader：进入 OpenAPI schema → /docs 出现「Authorize」按钮，
# 浏览器里即可带 Key 试调（此前 Key 只能靠手工构造请求头）
_api_key_header = APIKeyHeader(name="X-XCcv-Key", auto_error=False)


def get_config(request: Request):
    return request.app.state.config


def get_manager(request: Request):
    return request.app.state.manager


def make_gate(queue_size: int) -> threading.BoundedSemaphore:
    """推理并发准入闸：非阻塞获取，满即 429（不排队阻塞连接）。"""
    return threading.BoundedSemaphore(max(1, int(queue_size)))


async def require_key(
    request: Request,
    x_xccv_key: str | None = Depends(_api_key_header),
    api_key_q: str | None = Query(default=None, alias="api_key"),
) -> None:
    """/v1/health 以外的全部端点挂本依赖（路由级 dependencies）。

    ?api_key= 查询串仅回环放行（浏览器直开 /docs 试调用）；
    跨机部署一律走请求头（查询串易进日志/代理缓存）。
    """
    cfg = request.app.state.config
    key = x_xccv_key
    if not key and api_key_q:
        client_host = request.client.host if request.client else ""
        if client_host in _LOOPBACK_HOSTS:
            key = api_key_q
    if not verify_key(key, cfg.api_key):
        raise HTTPException(status_code=401, detail="API Key 无效或缺失"
                            "（请求头 X-XCcv-Key；可在 /docs 右上角 Authorize 填入）")


def acquire_gate(request: Request) -> None:
    """推理类端点入口调用；队列满直接 429（GPU 串行不积压）。"""
    gate = request.app.state.gate
    if not gate.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="推理队列已满，请稍后重试")


def release_gate(request: Request) -> None:
    try:
        request.app.state.gate.release()
    except ValueError:
        pass

