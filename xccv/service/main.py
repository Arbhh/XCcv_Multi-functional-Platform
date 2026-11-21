"""推理服务入口：uvicorn 启动（无头，禁 PyQt）。

开发态：`python -m xccv.service.main --port 8600`
打包态：XCcv推理服务.exe（NSSM 注册为 Windows 服务）。
"""

import argparse
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .. import __version__
from ..logsetup import setup_logging
from .api.deps import make_gate
from .api.schemas import error_envelope
from .config import load_config
from .core.model_manager import ModelManager

log = logging.getLogger(__name__)


def body_over_limit(content_length: str | None, limit_mb: float) -> bool:
    """Content-Length 声明体积是否超限（无声明/非法值不拦，由像素上限兜底）。"""
    if not content_length or not content_length.isdigit():
        return False
    return int(content_length) > int(max(1.0, float(limit_mb)) * 1024 * 1024)


def create_app(config=None) -> FastAPI:
    """应用组装（config 可注入便于测试；生产走 load_config()）。"""
    cfg = config or load_config()
    if not cfg.api_key:
        from .config import generate_api_key, save_config

        cfg.api_key = generate_api_key()
        save_config(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("===== XCcv 推理服务启动 v%s（%s:%s，白名单=%s）=====",
                 __version__, cfg.host, cfg.port,
                 "任意本地路径（回环）" if cfg.allow_any_path else cfg.model_dirs)
        if cfg.min_loaded_models:
            app.state.manager.preload(cfg.min_loaded_models, warmup=False)
            log.info("已按配置预载模型 %d 个", len(cfg.min_loaded_models))
        yield
        log.info("===== XCcv 推理服务停止 =====")

    app = FastAPI(
        title="XCcv Inference Service",
        version=__version__,
        lifespan=lifespan,
        docs_url=None, redoc_url=None,   # 自托管静态资源（离线环境 CDN 不可达）
        description="XCcv 模型推理 HTTP 接口（响应=统一检出 JSON schema_version=1，"
                    "字段说明见 docs/result_schema.md）",
    )

    # ---- 请求体体积护栏：multipart 先整包落盘再进端点，超大请求会耗尽
    #      内存/磁盘（Content-Length 声明即拒；分块上传无声明时由像素上限兜底） ----
    @app.middleware("http")
    async def _limit_request_body(request: Request, call_next):
        limit_mb = float(getattr(request.app.state.config, "max_upload_mb", 100.0))
        if body_over_limit(request.headers.get("content-length"), limit_mb):
            return JSONResponse(status_code=413, content=error_envelope(
                "error_413",
                f"请求体超过上限 {limit_mb:g} MB（max_upload_mb），"
                "请压缩图片或分批调用"))
        return await call_next(request)

    # ---- /docs 自托管（A3：默认 swagger 资源走 CDN，离线客户环境空白） ----
    from pathlib import Path as _Path

    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.staticfiles import StaticFiles

    _static_dir = _Path(__file__).resolve().parent / "api" / "static"
    if _static_dir.is_dir():
        app.mount("/docs-static",
                  StaticFiles(directory=str(_static_dir)), name="docs-static")

    @app.get("/docs", include_in_schema=False)
    def swagger_docs():
        if not (_static_dir / "swagger-ui-bundle.js").is_file():
            from fastapi.responses import HTMLResponse

            return HTMLResponse(
                "<h3>接口文档静态资源缺失</h3><p>离线文档未随包："
                f"缺少 {_static_dir / 'swagger-ui-bundle.js'}。"
                "可直接阅读 <a href='/openapi.json'>/openapi.json</a> 契约原文，"
                "或使用部署平台随带的 C# 测试程序。</p>", status_code=200)
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=app.title + " - Docs",
            oauth2_redirect_url=None,
            swagger_js_url="/docs-static/swagger-ui-bundle.js",
            swagger_css_url="/docs-static/swagger-ui.css",
        )

    @app.get("/docs-lite", include_in_schema=False)
    def docs_lite():
        """自包含接口一览页（服务端渲染，无 JS 依赖——浏览器必定非空）。"""
        import html as _html

        spec = app.openapi()
        rows = []
        for path, methods in sorted((spec.get("paths") or {}).items()):
            for verb, op in sorted(methods.items()):
                summary = _html.escape(str((op or {}).get("summary", "")))
                tag = _html.escape(
                    "、".join(str(t) for t in (op or {}).get("tags", [])))
                rows.append(
                    f"<tr><td><b>{verb.upper()}</b></td>"
                    f"<td><code>{_html.escape(path)}</code></td>"
                    f"<td>{summary}</td><td>{tag}</td></tr>")
        style = ("body{font-family:'Microsoft YaHei',sans-serif;margin:24px;"
                 "color:#1f2937}table{border-collapse:collapse;width:100%}"
                 "td,th{border:1px solid #d7e3ef;padding:7px 10px;"
                 "font-size:14px;text-align:left}th{background:#eaf3f8}"
                 "code{background:#f1f5f9;padding:2px 6px;border-radius:4px}"
                 "h1{font-size:22px}p{color:#5f7086;font-size:14px}")
        head = _html.escape(str(app.title))
        ver = _html.escape(str(app.version if hasattr(app, "version") else ""))
        return HTMLResponse(
            f"<html><head><meta charset='utf-8'><title>{head} 接口一览</title>"
            f"<style>{style}</style></head><body>"
            f"<h1>{head} · 接口一览</h1>"
            f"<p>鉴权：请求头 <code>X-XCcv-Key</code>（同机浏览器可用 "
            f"<code>?api_key=…</code>） · "
            f"完整调试版：<a href='/docs'>/docs</a> · "
            f"契约原文：<a href='/openapi.json'>/openapi.json</a>"
            f"{(' · 版本 ' + ver) if ver else ''}</p>"
            f"<table><tr><th>方法</th><th>路径</th><th>说明</th>"
            f"<th>分组</th></tr>{''.join(rows)}</table></body></html>")
    app.state.config = cfg
    app.state.manager = ModelManager(max_loaded=cfg.max_loaded_models)
    app.state.gate = make_gate(cfg.queue_size)
    app.state.started = time.time()
    from .core.flow_manager import FlowManager

    app.state.flow_manager = FlowManager(max_loaded=cfg.max_loaded_models)

    # ---- health（唯一免 Key 端点：供服务监控/NSSM 探活） ----
    from fastapi import Depends

    from .api.deps import get_manager, require_key

    @app.get("/v1/health", tags=["health"])
    def health(request: Request):
        import torch

        gpu = {}
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                gpu = {"free_mb": round(free / 2**20, 1),
                       "total_mb": round(total / 2**20, 1)}
        except Exception:
            log.debug("健康检查显存探测失败")
        return {"status": "ok", "version": __version__,
                "uptime_s": round(time.time() - request.app.state.started, 1),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "gpu": gpu, "api_version": 1,
                "loaded_models": len(get_manager(request).list())}

    @app.get("/v1/config", tags=["health"], dependencies=[Depends(require_key)])
    def get_runtime_config(request: Request):
        """运行时配置（字段白名单：凭据/文件系统布局不下发——
        api_key、model_dirs、allow_any_path 均不返回）。"""
        cfg_ = request.app.state.config
        return {"host": cfg_.host, "port": cfg_.port,
                "max_loaded_models": cfg_.max_loaded_models,
                "queue_size": cfg_.queue_size,
                "infer_conf": cfg_.infer_conf, "infer_iou": cfg_.infer_iou,
                "max_batch": cfg_.max_batch,
                "max_pixels": cfg_.max_pixels,
                "max_upload_mb": cfg_.max_upload_mb}

    # ---- 业务路由 ----
    from .api.routes_flow import router as flow_router
    from .api.routes_flows import router as flows_router
    from .api.routes_infer import router as infer_router
    from .api.routes_models import router as models_router
    from .api.routes_ws import router as ws_router

    app.include_router(models_router)
    app.include_router(infer_router)
    app.include_router(flow_router)
    app.include_router(flows_router)
    app.include_router(ws_router)

    # ---- 管理端点（停服务；部署平台「停止服务」按钮使用） ----
    from pydantic import BaseModel as _BM

    class _ShutdownReq(_BM):
        confirm: bool = False

    @app.post("/v1/admin/shutdown", tags=["health"],
              dependencies=[Depends(require_key)])
    async def admin_shutdown(request: Request, req: _ShutdownReq):
        if not req.confirm:
            raise HTTPException(status_code=400, detail="confirm 必须为 true")
        server = getattr(request.app.state, "server", None)

        import threading

        if server is not None:
            threading.Timer(0.2, setattr,
                            args=(server, "should_exit", True)).start()
        return {"ok": True}

    # ---- 统一错误信封（SDK 映射 XccvException） ----
    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content=error_envelope(
            f"error_{exc.status_code}", str(exc.detail),
            str(getattr(exc, "headers", "") or "")))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.error("服务内部错误: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content=error_envelope(
            "internal_error", f"{type(exc).__name__}: {exc}"))
    return app


def main(argv=None) -> None:
    # 必须最先兜底 stdio：从 GUI 主界面 Popen（无重定向）拉起时继承无效
    # 句柄，任何库写 stderr 即异常退出
    from ..gui_utils import ensure_console_streams

    ensure_console_streams()
    setup_logging("service")
    from ..win_qos import opt_out_power_throttling

    # 推理服务是纯后台进程（无前台窗口）——最易被系统 EcoQoS 节流（实测推理 2.6× 变慢）
    log.info("进程 QoS：%s", "HighQoS（已退出系统 EcoQoS 节流）"
             if opt_out_power_throttling() else "系统默认（未能设置）")
    # 未捕获异常必须留痕（pythonw stderr=devnull，此前进程"无痕退出"无法排障）
    try:
        from ..gui_utils import install_excepthook

        install_excepthook("XCcv 推理服务")
    except Exception:
        log.warning("崩溃钩子安装失败", exc_info=True)
    parser = argparse.ArgumentParser(description="XCcv 推理服务")
    parser.add_argument("--host", default=None, help="默认读配置（127.0.0.1）")
    parser.add_argument("--port", type=int, default=None, help="默认读配置（8600）")
    args = parser.parse_args(argv)

    import uvicorn

    cfg = load_config()
    host = args.host or cfg.host
    port = args.port or cfg.port
    app = create_app(cfg)
    # 单进程单 worker：多 worker 会各自初始化 CUDA context（显存翻倍/冲突）；
    # 自持 Server 实例以支持 /v1/admin/shutdown 优雅退出
    try:
        import httptools  # noqa: F401

        http_impl = "httptools"
    except Exception:
        log.debug("HTTP 实现探测失败（回退 auto）")
        http_impl = "auto"
    server = uvicorn.Server(uvicorn.Config(
        app, host=host, port=port, workers=1, http=http_impl,
        access_log=cfg.access_log, log_level=cfg.log_level))
    app.state.server = server
    # uvicorn.Config 构造时 dictConfig 会禁用既有 logger（注意：配置必须在
    # 其之后，否则被 disable_existing_loggers 覆盖、错误依旧不落盘）——
    # 显式把 uvicorn 错误接入平台日志文件（bind 失败等必可见）
    root = logging.getLogger()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.INFO)
        lg.disabled = False
    server.run()


if __name__ == "__main__":
    main()


