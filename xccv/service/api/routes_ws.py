"""T10-10 WebSocket 二进制推理通道：/v1/ws/infer。

协议（与 HTTP /v1/infer/image 同口径：同 manager/gate/result_json）：
- 连接后**首帧 JSON 文本头**：{"api_key": "...", "model_id": "…",
  "conf": 0.25, "iou": 0.20}（conf/iou 缺省走服务配置）；
- 鉴权失败 → close(4401)；头部就绪后回一帧 {"ok": true, "model_id": ...}；
- 此后**每帧二进制 = 一张图片**（jpg/png 原始字节）→ 回一帧 JSON 文本
  （统一检出 JSON schema_version=1）；队列满回 {"code":429,...}；
- 客户端断开即结束（无粘性状态，重连重发头部即可）。

C# SDK：XccvClient.InferImageWsAsync（ClientWebSocket，C#5 语法）；
gRPC 是否开工留决策点（方案 §九.2，默认仅 WebSocket）。
"""

import json
import logging

from fastapi import APIRouter, WebSocket
from starlette.concurrency import run_in_threadpool

from ..security import verify_key
from .deps import get_config, get_manager
from .routes_infer import _infer_one, decode_image

log = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])


@router.websocket("/v1/ws/infer")
async def ws_infer(ws: WebSocket):
    await ws.accept()
    cfg = get_config(ws)
    try:
        header = json.loads(await ws.receive_text())
    except Exception:
        log.debug("WebSocket 握手异常（已按 1008 关闭）")
        await ws.close(code=1008)
        return
    if not verify_key(str(header.get("api_key") or ""), cfg.api_key):
        await ws.close(code=4401)
        return
    model_id = str(header.get("model_id") or "")
    conf = header.get("conf")
    iou = header.get("iou")
    await ws.send_text(json.dumps({"ok": True, "model_id": model_id},
                                  ensure_ascii=False))
    try:
        while True:
            data = await ws.receive_bytes()
            gate = ws.app.state.gate
            if not gate.acquire(blocking=False):
                await ws.send_text(json.dumps(
                    {"code": 429, "message": "推理队列已满，请稍后重试"}))
                continue
            try:
                img = decode_image(data, "ws")
                # 同步推理在线程池执行——不阻塞事件循环（否则拖慢全部连接）
                payload, _entry, _pred = await run_in_threadpool(
                    _infer_one, ws, img, "ws", model_id, conf, iou)
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception as e:  # noqa: BLE001  单帧失败不断连接
                log.warning("WS 推理失败: %s", e)
                await ws.send_text(json.dumps(
                    {"error": f"{type(e).__name__}: {e}"}))
            finally:
                gate.release()
    except Exception:
        pass  # 客户端断开（WebSocketDisconnect 等）——正常结束
