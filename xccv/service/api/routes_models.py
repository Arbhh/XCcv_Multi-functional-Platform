"""模型管理端点：加载 / 卸载 / 状态。"""

import time

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.model_manager import ModelNotFound
from ..security import PathNotAllowed, ensure_path_allowed
from .deps import get_config, get_manager, require_key
from .schemas import LoadModelRequest, UnloadModelRequest

router = APIRouter(prefix="/v1/models", tags=["models"],
                   dependencies=[Depends(require_key)])


@router.post("/load")
def load_model(req: LoadModelRequest, request: Request):
    """加载 .pt/.engine（白名单校验；同路径幂等）。

    响应即 /v1/models/status 的单个条目（model_id 供推理端点寻址）。
    """
    cfg = get_config(request)
    try:
        ensure_path_allowed(req.path, cfg)
    except PathNotAllowed as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    from pathlib import Path

    p = Path(req.path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"模型文件不存在: {req.path}")
    manager = get_manager(request)
    try:
        entry = manager.load(p, device=req.device, alias=req.alias,
                             warmup=req.warmup)
    except ModelNotFound:
        raise
    except Exception as e:
        # 再抛为 HTTP 500：detail 承载异常上下文——P-2 登记
        raise HTTPException(status_code=500,
                            detail=f"模型加载失败: {type(e).__name__}: {e}") from e
    items = [m for m in manager.list() if m["model_id"] == entry.model_id]
    return items[0] if items else {"model_id": entry.model_id}


@router.post("/unload")
def unload_model(req: UnloadModelRequest, request: Request):
    ok = get_manager(request).unload(req.model_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"模型未加载: {req.model_id}")
    return {"ok": True}


@router.get("/status")
def model_status(request: Request):
    return {"models": get_manager(request).list(),
            "uptime_s": round(time.time() - request.app.state.started, 1)}
