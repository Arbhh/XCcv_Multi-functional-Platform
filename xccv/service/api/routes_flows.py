"""已加载流程端点（B2：流程包常驻为可寻址 API，加载一次多次推理）。

与按次 /v1/flow/run 的关系：按次适合无状态调用；常驻适合上位机启动时
加载、产线连续推理（流程内 AI 模型仅首次加载，后续复用会话缓存）。
"""

import logging
log = logging.getLogger(__name__)

import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request
from fastapi import UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ...engine.flow.snapshot import overlay_result
from ..core.flow_manager import FlowNotFound
from .deps import acquire_gate, get_config, release_gate, require_key
from .routes_flow import run_flow_payload

router = APIRouter(prefix="/v1/flows", tags=["flows"],
                   dependencies=[Depends(require_key)])


def get_flow_manager(request: Request):
    return request.app.state.flow_manager


class LoadFlowBody(BaseModel):
    """POST /v1/flows/load 请求体。"""

    path: str = Field(..., description="流程文件路径（.xvap/.json），白名单校验")
    alias: str = Field("", description="业务别名（默认取文件名）")


class UnloadFlowBody(BaseModel):
    model_id: str = Field(..., description="flow_id")


@router.post("/load")
def load_flow(req: LoadFlowBody, request: Request):
    cfg = get_config(request)
    from ..security import PathNotAllowed, ensure_path_allowed

    try:
        ensure_path_allowed(req.path, cfg)
    except PathNotAllowed as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    p = Path(req.path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"流程文件不存在: {req.path}")
    try:
        entry = get_flow_manager(request).load(p, req.alias, cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # 再抛为 HTTP 500：detail 承载异常上下文——P-2 登记
        raise HTTPException(status_code=500,
                            detail=f"流程加载失败: {type(e).__name__}: {e}") from e
    items = [f for f in get_flow_manager(request).list()
             if f["flow_id"] == entry.flow_id]
    return items[0] if items else {"flow_id": entry.flow_id}


@router.post("/run")
def run_loaded_flow(request: Request, flow_id: str = Form(...),
                    image: UploadFile | None = File(default=None),
                    image_path: str | None = Form(default=None),
                    render: bool = Form(default=False)):
    """运行已加载流程：multipart 图片或本地图路径 → 统一流程 JSON（含 verdict）。"""
    cfg = get_config(request)
    acquire_gate(request)
    tmp = Path(tempfile.mkdtemp(prefix="xccv_frun_"))
    try:
        if image is not None and image.filename:
            src_img = tmp / ("img" + (Path(image.filename).suffix or ".png"))
            src_img.write_bytes(image.file.read())
        elif image_path:
            from ..security import PathNotAllowed, ensure_path_allowed

            try:
                ensure_path_allowed(image_path, cfg)
            except PathNotAllowed as e:
                raise HTTPException(status_code=403, detail=str(e)) from e
            src_img = Path(image_path)
            if not src_img.is_file():
                raise HTTPException(status_code=404,
                                    detail=f"图片不存在: {image_path}")
        else:
            raise HTTPException(status_code=400,
                                detail="缺少图片：请上传 image 或给 image_path")
        try:
            payload, session, base = get_flow_manager(request).run(
                flow_id, src_img, src_img.name)
        except FlowNotFound as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if render:
            try:
                rendered = overlay_result(base, session.executor.results)
            except Exception:
                log.debug("流程渲染失败（回退底图响应）")
                rendered = base
            ok, buf = cv2.imencode(".png", rendered)
            if not ok:
                raise HTTPException(status_code=500, detail="PNG 编码失败")
            return Response(content=buf.tobytes(), media_type="image/png")
        return payload
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        release_gate(request)


@router.post("/unload")
def unload_flow(req: UnloadFlowBody, request: Request):
    ok = get_flow_manager(request).unload(req.model_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"流程未加载: {req.model_id}")
    return {"ok": True}


@router.get("/status")
def flow_status(request: Request):
    return {"flows": get_flow_manager(request).list()}
