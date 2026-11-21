"""推理端点：单帧 / 批量 / 叠加图渲染。

图像输入一律 multipart 原始字节（拒绝 base64 JSON——性能契约）。
响应体 = result_json 统一 JSON（schema_version=1）。
"""

import contextlib
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ...engine.result_json import build_result_json
from ..core.model_manager import ModelNotFound
from ..core.timing import StageTimer
from .deps import (acquire_gate, get_config, get_manager, release_gate,
                   require_key)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["inference"],
                   dependencies=[Depends(require_key)])


def decode_image(data: bytes, name: str = "image",
                 max_pixels: int = 0) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400,
                            detail=f"无法解码图片: {name}（支持 jpg/png/bmp）")
    if max_pixels and img.shape[0] * img.shape[1] > int(max_pixels):
        raise HTTPException(
            status_code=413,
            detail=f"图片像素超过上限 {int(max_pixels)}（w*h）：{name} 为 "
                   f"{img.shape[1]}x{img.shape[0]}，请降采样后再调用")
    return img


def decode_upload_name(name: str | None) -> str:
    """还原上传文件名：.NET HttpClient 对非 ASCII 文件名做 RFC2047 编码
    （=?utf-8?B?...?=），解码回可读名（source.file 契约字段保持人话）。"""
    if not name:
        return "image"
    if "=?" not in name:
        return name
    from email.header import decode_header

    try:
        out = ""
        for data, enc in decode_header(name):
            out += data.decode(enc or "utf-8", "ignore") \
                if isinstance(data, bytes) else data
        return out or name
    except Exception:
        log.debug("上传文件名解析失败（保留原始名）")
        return name


def _source_of(filename: str, img: np.ndarray) -> dict:
    return {"file": filename or "image", "path": "",
            "width": int(img.shape[1]), "height": int(img.shape[0])}


def _attach_timing(resp, timer: StageTimer | None, request: Request) -> None:
    cfg = request.app.state.config
    if cfg.timing_enabled and timer is not None:
        resp.headers["X-XCcv-Timing"] = timer.header()


def _stage(timer, name):
    return timer.stage(name) if timer is not None else contextlib.nullcontext()


def _infer_one(request: Request, image: np.ndarray, filename: str,
               model_id: str, conf: float | None, iou: float | None,
               tiled: bool = False, timer: StageTimer | None = None):
    cfg = get_config(request)
    manager = get_manager(request)
    _conf = float(conf) if conf is not None else cfg.infer_conf
    _iou = float(iou) if iou is not None else cfg.infer_iou
    try:
        with _stage(timer, "infer"):
            if tiled:
                # 大图分块推理（切块原生口径推理+同类 NMS 合并去重）
                from ...engine.tiled_infer import tiled_predict

                entry = manager.get(model_id)
                with entry.lock:
                    pred = tiled_predict(entry.model, image, conf=_conf,
                                         iou=_iou)
                    entry.last_used = time.time()
            else:
                pred, entry = manager.infer(model_id, image, conf=_conf,
                                            iou=_iou,
                                            max_side=cfg.max_infer_side)
    except ModelNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    with _stage(timer, "post"):
        payload = build_result_json(
            task=entry.task, source=_source_of(filename, image),
            model={"file": Path(entry.path).name, "name": entry.alias},
            inference={"conf": _conf, "iou": _iou},   # 回显实际生效值（含默认值补齐）
            prediction=pred, classes=entry.classes,
            score_semantics=getattr(entry.model, "score_semantics", None))
    return payload, entry, pred


@router.post("/infer/image")
def infer_image(request: Request, image: UploadFile = File(...),
                model_id: str = Form(...),
                conf: float | None = Form(default=None),
                iou: float | None = Form(default=None),
                tiled: bool = Form(default=False)):
    """单帧推理 → 统一检出 JSON（tiled=true 走 4K 分块推理）。"""
    acquire_gate(request)
    timer = StageTimer()
    try:
        name = decode_upload_name(image.filename)
        with _stage(timer, "decode"):
            img = decode_image(image.file.read(), name,
                               max_pixels=get_config(request).max_pixels)
        payload, _entry, _pred = _infer_one(request, img, name, model_id,
                                            conf, iou, tiled=tiled, timer=timer)
        resp = JSONResponse(content=payload)
        _attach_timing(resp, timer, request)
        return resp
    finally:
        release_gate(request)


@router.post("/infer/batch")
def infer_batch(request: Request, images: list[UploadFile] = File(...),
                model_id: str = Form(...),
                conf: float | None = Form(default=None),
                iou: float | None = Form(default=None),
                tiled: bool = Form(default=False)):
    """批量推理（服务端串行逐帧；响应为数组，顺序与上传一致）。"""
    if not images:
        raise HTTPException(status_code=400, detail="images 为空")
    cfg = get_config(request)
    if len(images) > int(cfg.max_batch):
        raise HTTPException(
            status_code=400,
            detail=f"单请求批量张数超过上限 {int(cfg.max_batch)}"
                   f"（max_batch），当前 {len(images)}；请分批调用")
    acquire_gate(request)
    timer = StageTimer()
    try:
        out = []
        for f in images:
            try:
                fname = decode_upload_name(f.filename)
                img = decode_image(f.file.read(), fname,
                                   max_pixels=cfg.max_pixels)
                payload, _e, _p = _infer_one(request, img, fname,
                                             model_id, conf, iou,
                                             tiled=tiled, timer=timer)
                out.append(payload)
            except HTTPException as e:
                # 单张失败不中断批量（与平台批量线程同语义）
                out.append(build_result_json(
                    task="unknown", source={"file": f.filename or "", "path": "",
                                            "width": None, "height": None},
                    error=f"{e.status_code}: {e.detail}"))
        resp = JSONResponse(content=out)
        _attach_timing(resp, timer, request)
        return resp
    finally:
        release_gate(request)


@router.post("/render/overlay")
def render_overlay(request: Request, image: UploadFile = File(...),
                   model_id: str = Form(...),
                   conf: float | None = Form(default=None),
                   iou: float | None = Form(default=None),
                   show_labels: bool = Form(default=True),
                   show_masks: bool = Form(default=True)):
    """推理 + 结果叠加 → PNG 字节（image/png，供上位机直接显示）。"""
    acquire_gate(request)
    timer = StageTimer()
    try:
        from ...testing.render import draw_predictions

        name = decode_upload_name(image.filename)
        with _stage(timer, "decode"):
            img = decode_image(image.file.read(), name,
                               max_pixels=get_config(request).max_pixels)
        payload, entry, pred = _infer_one(request, img, name, model_id, conf,
                                          iou, timer=timer)
        with _stage(timer, "render"):
            try:
                rendered = draw_predictions(img, pred, entry.classes,
                                            entry.task, show_labels=show_labels,
                                            show_masks=show_masks)
            except Exception:
                log.warning("叠加图绘制失败，返回原图", exc_info=True)
                rendered = img
            ok, buf = cv2.imencode(".png", rendered)
        if not ok:
            raise HTTPException(status_code=500, detail="PNG 编码失败")
        from fastapi import Response

        resp = Response(content=buf.tobytes(), media_type="image/png")
        _attach_timing(resp, timer, request)
        return resp
    finally:
        release_gate(request)

