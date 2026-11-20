"""推理服务健康轮询唯一源（v0.5 UI 升级期1 新增）。

顶栏服务状态点、部署平台页、桌面灵动岛共用同一个 poller 实例：
轮询只发一份请求，状态经 state_changed 信号广播（避免此前每处各建
QTimer+NAM 的双份请求）。

三态：
- running  /v1/health 200；
- starting 部署页 note_starting() 后的一段窗口期内连不上（服务进程
  正在加载 torch 等重组件，约 5~10 秒）；
- stopped  其余连不上。

HTTP 回调沿用 _HttpMixin 的 sender() 模式（禁 lambda 持 reply）。
本模块保持轻量：只依赖 service.config（纯 stdlib），禁 torch。
"""

import json
import logging
import time

from PyQt5.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest

log = logging.getLogger(__name__)

_STARTING_WINDOW_S = 45.0   # note_starting 后允许「启动中」显示的最长时长


class ServiceHealthPoller(QObject):
    """共享健康轮询：start() 后每 interval 轮询 /v1/health 并广播三态。"""

    state_changed = pyqtSignal(str, dict)   # ("stopped"|"starting"|"running", info)

    def __init__(self, parent=None, interval_ms: int = 1500):
        super().__init__(parent)
        self._nam = QNetworkAccessManager(self)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._poll)
        self._state = "stopped"
        self._starting_since = 0.0

    # ---- 生命周期 ----
    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
            QTimer.singleShot(0, self._poll)   # 启动立即探一次

    def stop(self) -> None:
        self._timer.stop()

    def state(self) -> str:
        return self._state

    def note_starting(self) -> None:
        """部署页拉起服务进程时调用：立刻进入「启动中」态并广播。"""
        self._starting_since = time.time()
        self._publish("starting", {})

    # ---- 轮询 ----
    def base_url(self) -> str:
        from .service.config import load_config

        cfg = load_config()
        host = cfg.host if cfg.host not in ("0.0.0.0", "::") else "127.0.0.1"
        return f"http://{host}:{cfg.port}"

    def _poll(self) -> None:
        try:
            url = QUrl(self.base_url() + "/v1/health")
        except Exception:
            log.warning("健康轮询读取服务配置失败", exc_info=True)
            return
        reply = self._nam.get(QNetworkRequest(url))
        # sender() 模式：连 self 方法，回调内经 self.sender() 取 reply
        reply.finished.connect(self._on_finished)

    def _on_finished(self):
        reply = self.sender()
        if reply is None:
            return
        try:
            import sip

            if sip.isdeleted(reply):   # C++ 侧已回收（超时/错误竞态）
                return
            status = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
            raw = bytes(reply.readAll())
        finally:
            reply.deleteLater()
        if status == 200:
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except ValueError:
                body = {}
            self._starting_since = 0.0
            self._publish("running", body if isinstance(body, dict) else {})
        else:
            self._publish(self._classify_failure(), {})

    def _classify_failure(self) -> str:
        if self._starting_since and \
                (time.time() - self._starting_since) < _STARTING_WINDOW_S:
            return "starting"
        return "stopped"

    def _publish(self, state: str, body: dict) -> None:
        self._state = state
        gpu = body.get("gpu") or {}
        info = {
            "url": self.base_url(),
            "models": int(body.get("loaded_models", 0) or 0),
            "device": str(body.get("device", "") or ""),
            "gpu_free_mb": float(gpu.get("free_mb") or 0),
            "gpu_total_mb": float(gpu.get("total_mb") or 0),
        }
        self.state_changed.emit(state, info)
