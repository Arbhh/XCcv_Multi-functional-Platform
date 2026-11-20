"""流程推理端点：一次调用跑完整流程（传统算子 + AI），返回统一流程 JSON。

输入二选一：
- multipart 上传 flow 文件（.json 流程图 / .xvap 流程包）+ image 字节；
- flow_path / image_path（服务端本地路径，白名单校验）——适合 XCcv 与
  服务同机的部署形态。

.xvap 在服务进程内解包到临时目录（用后即删），模型经 registry.load_model
正常加载——客户端全程接触不到权重文件。
"""

import contextlib
import json
import logging
import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ...engine import flow_package as fp
from ..core.timing import StageTimer
from ...engine.flow.snapshot import overlay_result
from ...engine.io_utils import imread_unicode
from ...engine.result_json import (build_flow_result_json,
                                   classification_topk, detection_rows,
                                   verdict_from_node_results)
from ..security import PathNotAllowed, ensure_path_allowed
from .deps import acquire_gate, get_config, release_gate, require_key
from .routes_infer import decode_image

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/flow", tags=["flow"],
                   dependencies=[Depends(require_key)])


def _stage(timer, name):
    return timer.stage(name) if timer is not None else contextlib.nullcontext()


def _attach_timing(resp, timer, request: Request) -> None:
    if get_config(request).timing_enabled and timer is not None:
        resp.headers["X-XCcv-Timing"] = timer.header()


def _unpack_xvap_bytes(data: bytes, model_dir: Path) -> dict:
    """内存解包 .xvap（模型字节落临时目录+引用改写绝对路径）。"""
    flow, models = fp.unpack_xvap(data)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in models.items():
        (model_dir / name).write_bytes(blob)
    for n in flow.get("nodes", []):
        props = n.setdefault("properties", {})
        for key, name in fp._model_refs(n):
            props[key] = str(model_dir / Path(name).name)
    return flow


def _collect_outputs(ex_results) -> tuple[object | None, object | None, list, list]:
    """按拓扑序收集全图 ok 节点输出（与 _XcvBatchThread / result_output 同口径）。

    返回 (detection, classification, geometries, measurements)；检出与分类
    按内容区分（有框=检测、有概率=分类，Prediction 两属性恒存在）。
    """
    from ...engine.flow.ir import ImageData as _Img

    det, cls_src, geo_list, meas_list = None, None, [], []
    for nid in sorted(ex_results):
        r = ex_results[nid]
        if not getattr(r, "ok", False):
            continue
        for v in (r.outputs or {}).values():
            if isinstance(v, _Img):
                continue
            if hasattr(v, "kind"):
                geo_list.append(v)
            elif getattr(v, "boxes", None):
                det = det or v
            elif getattr(v, "probs", None):
                cls_src = cls_src or v
            elif hasattr(v, "to_json"):
                meas_list.append(v)
    return det, cls_src, geo_list, meas_list


def run_flow_payload(flow: dict, flow_name: str, image_path: Path,
                     source_name: str, session=None):
    """在流程副本上跑单图 → (payload, session, base_image)。

    session 传入常驻会话时模型随会话复用（流程池，仅首次加载）；None 新建。
    """
    from ...modeltools.flow_session import FlowSession

    norm = json.loads(json.dumps(flow))  # 深拷贝（改 path 不污染调用方）
    has_loader = False
    for n in norm.get("nodes", []):
        if str(n.get("type", "")).endswith("image_loader"):
            n.setdefault("properties", {})["path"] = str(image_path)
            has_loader = True
    if not has_loader:
        raise HTTPException(status_code=400, detail="流程缺少「读入图片」节点")

    if session is None:
        session = FlowSession()
    summary = session.run(norm, label="服务调用", source_name=source_name,
                          flow_name=flow_name)
    if summary.get("errors"):
        raise HTTPException(
            status_code=422,
            detail="流程执行失败：" + "；".join(
                f"节点#{k}: {v}" for k, v in summary["errors"].items()))
    ex_results = session.executor.results
    det, cls_src, geo_list, meas_list = _collect_outputs(ex_results)
    classes = (session.ctx.get("model_classes")
               or session.ctx.get("flow_class_names") or [])
    det_list = detection_rows(det, classes) if det is not None else []
    cls_list = classification_topk(cls_src, classes) if cls_src is not None else []
    img = imread_unicode(str(image_path))
    payload = build_flow_result_json(
        source={"file": source_name, "path": str(image_path),
                "width": img.shape[1] if img is not None else None,
                "height": img.shape[0] if img is not None else None},
        detections=det_list, classifications=cls_list,
        geometries=geo_list, measurements=meas_list, flow_name=flow_name,
        verdict=verdict_from_node_results(ex_results))
    # 底图：拓扑序最后的图像输出，否则原图
    from ...engine.flow.ir import ImageData as _Img

    base = img
    for nid in sorted(ex_results):
        r = ex_results[nid]
        if not getattr(r, "ok", False):
            continue
        for v in (r.outputs or {}).values():
            if isinstance(v, _Img) and v.mat.size:
                base = v.mat
    return payload, session, base


def _flow_response(request: Request, flow_bytes: bytes, flow_name: str,
                   image_bytes: bytes, image_name: str,
                   flow_path: str | None, image_path_s: str | None,
                   render: bool):
    cfg = get_config(request)

    def _allowed(p: str) -> Path:
        try:
            return ensure_path_allowed(p, cfg)
        except PathNotAllowed as e:
            raise HTTPException(status_code=403, detail=str(e)) from e

    acquire_gate(request)
    timer = StageTimer() if cfg.timing_enabled else None
    tmp = Path(tempfile.mkdtemp(prefix="xccv_flow_"))
    try:
        flow = None
        src_img = None
        with _stage(timer, "unpack"):
            # 流程来源：上传字节（内存直解，流程文件不落盘）或 本地白名单路径
            if flow_bytes:
                if Path(flow_name).suffix.lower() == ".xvap":
                    flow = _unpack_xvap_bytes(flow_bytes, tmp / "流程模型")
                    flow_name = flow_name or "flow"
                elif Path(flow_name).suffix.lower() == ".json":
                    try:
                        flow = json.loads(flow_bytes.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError) as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"流程 JSON 解析失败: {e}") from e
                    flow_name = flow_name or "flow"
                else:
                    raise HTTPException(status_code=400,
                                        detail="流程文件仅支持 .json / .xvap")
            elif flow_path:
                _allowed(flow_path)
                src_flow = Path(flow_path)
                if not src_flow.is_file():
                    raise HTTPException(status_code=404,
                                        detail=f"流程文件不存在: {flow_path}")
                if src_flow.suffix.lower() == ".xvap":
                    flow = _unpack_xvap_bytes(src_flow.read_bytes(),
                                              tmp / "流程模型")
                elif src_flow.suffix.lower() == ".json":
                    try:
                        flow = json.loads(src_flow.read_text(encoding="utf-8"))
                    except ValueError as e:
                        raise HTTPException(status_code=400,
                                            detail=f"流程 JSON 解析失败: {e}") from e
                else:
                    raise HTTPException(status_code=400,
                                        detail="流程文件仅支持 .json / .xvap")
                flow_name = flow_name or src_flow.stem
            else:
                raise HTTPException(status_code=400,
                                    detail="缺少流程：请上传 flow 文件或给 flow_path")
            # 图片来源：上传字节 或 本地路径（写临时目录——引擎按路径读图）
            if image_bytes:
                src_img = tmp / ("img" + (Path(image_name).suffix or ".png"))
                src_img.write_bytes(image_bytes)
            elif image_path_s:
                _allowed(image_path_s)
                src_img = Path(image_path_s)
                if not src_img.is_file():
                    raise HTTPException(status_code=404,
                                        detail=f"图片不存在: {image_path_s}")
            else:
                raise HTTPException(status_code=400,
                                    detail="缺少图片：请上传 image 或给 image_path")

        with _stage(timer, "infer"):
            payload, _session, base = run_flow_payload(
                flow, flow_name, src_img, src_img.name)
        if render:
            with _stage(timer, "render"):
                try:
                    rendered = overlay_result(base, _session.executor.results)
                except Exception:
                    log.warning("流程叠加图绘制失败，返回原图", exc_info=True)
                    rendered = base
                ok, buf = cv2.imencode(".png", rendered)
            if not ok:
                raise HTTPException(status_code=500, detail="PNG 编码失败")
            resp = Response(content=buf.tobytes(), media_type="image/png")
            _attach_timing(resp, timer, request)
            return resp
        resp = JSONResponse(content=payload)
        _attach_timing(resp, timer, request)
        return resp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        release_gate(request)


@router.post("/run")
def flow_run(request: Request,
             flow: UploadFile | None = File(default=None),
             image: UploadFile | None = File(default=None),
             flow_path: str | None = Form(default=None),
             image_path: str | None = Form(default=None),
             render: bool = Form(default=False)):
    """流程包推理：multipart（flow+image 字节）或 path 形式 → 统一流程 JSON。

    同步 def：真推理在线程池执行，不占用事件循环（T10-4；此前 async 把
    并发请求的推理段串行化在事件循环上）。
    """
    flow_bytes = flow.file.read() if flow is not None else b""
    image_bytes = image.file.read() if image is not None else b""
    return _flow_response(request, flow_bytes, flow.filename or "" if flow else "",
                          image_bytes, image.filename or "" if image else "",
                          flow_path, image_path, render)


@router.post("/run/file")
def flow_run_file(request: Request, flow_path: str = Form(...),
                  image_path: str = Form(...), render: bool = Form(default=False)):
    """同机部署便捷形态：传服务端本地路径（白名单校验）。"""
    return _flow_response(request, b"", "", b"", "", flow_path, image_path, render)
