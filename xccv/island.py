"""桌面灵动岛：推理服务的独立悬浮状态窗（v0.5 UI 升级期2，D3 首选形态）。

形态：屏幕顶部居中的深色胶囊（宽随文本自适应、高 40，圆角 999，白字），
置顶、不进任务栏（Tool）、可拖动、位置持久化（ui_prefs.json 的 island_pos）。

三态（订阅共享 ServiceHealthPoller，不额外发请求）：
- 停止  ：灰点「推理服务未运行」
- 启动中：黄点呼吸「推理服务启动中…」
- 运行中：绿点「推理服务 · N 个模型」

点击胶囊向下弹出快捷面板：已加载模型摘要（/v1/models/status，带
API Key），以及 打开部署平台 / 复制 API Key / 停止服务 三个动作。
HTTP 回调用 sender() 模式（禁 lambda 持 reply）。

V0.7.2 i18n：文案走 tr()；胶囊与面板宽度按 QFontMetrics 自适应
（英文文本比中文长，固定宽会截断——实测）。

本模块保持轻量：禁 torch；不依赖 deploy_platform（后者反向引用本模块）。
"""

import json
import logging

from PyQt5.QtCore import (
    QEasingCurve, QRectF, Qt, QUrl, QVariantAnimation, pyqtSignal,
)
from PyQt5.QtGui import QColor, QFontMetrics, QPainter
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from .i18n import tr
from .ui_style import C
from .ui_widgets import StatusDot, load_prefs, save_prefs

log = logging.getLogger(__name__)

_CAPSULE_W_MIN, _CAPSULE_H = 360, 40
_MARGIN = 14                 # 胶囊相对窗口的左右边距
_ARROW_ZONE = 34             # 胶囊右侧箭头区
_PANEL_W_MIN = 344
_DRAG_THRESHOLD = 6          # 像素；超过视为拖动而非点击
_DOT_COLOR = {"running": C.success, "starting": C.warn, "stopped": C.text_3}


class ServiceIsland(QWidget):
    """独立桌面悬浮窗：胶囊 + 可展开快捷面板。"""

    open_deploy_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, poller, parent=None):
        super().__init__(parent)
        self._poller = poller
        # 停止服务走面板同链路（_stop_service 有完整反馈与 terminate 兜底）；
        # parent 即创建方 ServicePanel（deploy_platform._sync_island）
        self._panel = parent
        self._state = "stopped"
        self._models_n = 0
        self._press_pos = None
        self._moved = False
        self._expanded = False
        self._opening = False
        self._cap_w = _CAPSULE_W_MIN
        self._panel_w = _PANEL_W_MIN
        self._cur_h = _CAPSULE_H

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # ---- 胶囊内容（子控件直接定位；子控件默认忽略鼠标事件，
        #      未接受事件上抛到本窗口统一处理拖动/点击）----
        self.dot = StatusDot(11, _DOT_COLOR["stopped"], self)
        self.dot.move(_MARGIN + 22, (_CAPSULE_H - 11) // 2)
        self.cap_label = QLabel(tr("推理服务未运行"), self)
        self.cap_label.setStyleSheet(
            "color:#ffffff; font-size:15px; background:transparent; border:none;")
        self.cap_label.move(_MARGIN + 44, 0)
        self.cap_label.setFixedHeight(_CAPSULE_H)
        self.cap_arrow = QLabel("▾", self)
        self.cap_arrow.setStyleSheet(
            "color:rgba(255,255,255,0.75); font-size:15px;"
            "background:transparent; border:none;")
        self.cap_arrow.setAlignment(Qt.AlignCenter)

        # ---- 快捷面板（向下弹出）----
        self.panel = QFrame(self)
        self.panel.setObjectName("islandPanel")
        self.panel.setStyleSheet(
            "QFrame#islandPanel { background:#ffffff;"
            f" border:1px solid {C.border}; border-radius: 12px; }}")
        panel_lay = QVBoxLayout(self.panel)
        panel_lay.setContentsMargins(16, 12, 16, 12)
        panel_lay.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.panel_title = QLabel(tr("推理服务未运行"))
        self.panel_title.setStyleSheet(
            f"font-size:16px; font-weight:bold; color:{C.text_1};"
            "background:transparent; border:none;")
        self.url_label = QLabel("")
        self.url_label.setStyleSheet(
            f"font-size:15px; color:{C.text_3};"
            "background:transparent; border:none;")
        head.addWidget(self.panel_title)
        head.addStretch(1)
        head.addWidget(self.url_label)
        panel_lay.addLayout(head)

        self.models_label = QLabel(tr("已加载模型：—"))
        self.models_label.setWordWrap(True)
        self.models_label.setStyleSheet(
            f"font-size:15px; color:{C.text_2};"
            "background:transparent; border:none;")
        panel_lay.addWidget(self.models_label)

        btns = QHBoxLayout()
        btns.setSpacing(8)
        self.btn_open = QPushButton(tr("部署平台"))
        self.btn_open.setToolTip(tr("打开部署平台窗口"))
        self.btn_open.clicked.connect(self._emit_open_deploy)
        self.btn_key = QPushButton(tr("复制 Key"))
        self.btn_key.clicked.connect(self._copy_key)
        self.btn_stop = QPushButton(tr("停止服务"))
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self._stop_service)
        btns.addWidget(self.btn_open)
        btns.addWidget(self.btn_key)
        btns.addStretch(1)
        btns.addWidget(self.btn_stop)
        panel_lay.addLayout(btns)

        self._apply_widths()
        self.panel.setVisible(False)

        # ---- 展开动画（面板高度 0↔自然高；OutCubic 平滑过渡）----
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_expand_step)
        self._anim.finished.connect(self._on_expand_done)

        # ---- HTTP（models 摘要与停止服务）----
        self._pending = {}
        self._nam = QNetworkAccessManager(self)

        poller.state_changed.connect(self._on_poll_state)
        self._on_poll_state(poller.state(),
                            {"url": "", "models": 0, "device": ""})
        self._restore_pos()

    # ---- 宽度自适应（i18n：英文文本更长，固定宽会截断）----
    def _apply_widths(self, cap_text: str | None = None) -> None:
        """按胶囊文本与面板按钮实测宽度重设窗口/面板尺寸。"""
        text = cap_text if cap_text is not None else self.cap_label.text()
        fm_cap = QFontMetrics(self.cap_label.font())
        need_cap = (_MARGIN + 44 + fm_cap.horizontalAdvance(text)
                    + _ARROW_ZONE + 12 + _MARGIN)
        self._cap_w = max(_CAPSULE_W_MIN, need_cap)
        self.cap_label.resize(self._cap_w - 44 - _ARROW_ZONE, _CAPSULE_H)
        self.cap_arrow.setGeometry(_MARGIN + self._cap_w - _ARROW_ZONE, 0,
                                   22, _CAPSULE_H)

        fm_btn = QFontMetrics(self.btn_stop.font())
        btn_total = (fm_btn.horizontalAdvance(self.btn_open.text())
                     + fm_btn.horizontalAdvance(self.btn_key.text())
                     + fm_btn.horizontalAdvance(self.btn_stop.text())
                     + 2 * 8 + 32 + 16)
        fm_title = QFontMetrics(self.panel_title.font())
        need_panel = max(_PANEL_W_MIN, btn_total,
                         fm_title.horizontalAdvance(self.panel_title.text())
                         + 44)
        self._panel_w = need_panel
        self.panel.setFixedWidth(self._panel_w)
        self.panel.move((self._cap_w + 2 * _MARGIN - self._panel_w) // 2,
                        _CAPSULE_H + 6)
        self.setFixedSize(self._cap_w + 2 * _MARGIN, self._cur_h)

    # ---- 状态 ----
    def _on_poll_state(self, state: str, info: dict) -> None:
        self._state = state
        self._models_n = int(info.get("models", 0) or 0)
        url = str(info.get("url", "") or "")
        self.dot.set_color(_DOT_COLOR.get(state, C.text_3))
        if state == "running":
            if self._models_n <= 0:
                self.cap_label.setText(tr("推理服务运行中"))
            else:
                self.cap_label.setText(
                    f"{tr('推理服务')} · {self._models_n} {tr('个模型')}")
            self.dot.stop_breath()
        elif state == "starting":
            self.cap_label.setText(tr("推理服务启动中…"))
            self.dot.start_breath()
        else:
            self.cap_label.setText(tr("推理服务未运行"))
            self.dot.stop_breath()
        self._apply_widths(self.cap_label.text())
        # 面板头部同步
        if state in ("running", "starting"):
            self.panel_title.setText(
                tr("推理服务运行中") if state == "running"
                else tr("推理服务启动中…"))
            self.url_label.setText(url)
        else:
            self.panel_title.setText(tr("推理服务未运行"))
            self.url_label.setText("")
            self.models_label.setText(tr("已加载模型：—"))
        self._apply_widths(self.cap_label.text())

    def state(self) -> str:
        return self._state

    # ---- 展开 / 收起 ----
    def _panel_natural_height(self) -> int:
        # 150 = 标题行 + 两行模型摘要 + 按钮行 + 间距边距（实测余量）
        return 150

    def toggle_expand(self) -> None:
        """兼容入口：未展开则展开，已展开则收起。"""
        if self._expanded:
            self._animate_expand(False)
        else:
            self._expand_open()

    def _expand_open(self) -> None:
        self._expanded = True
        self.panel.setVisible(True)
        self.cap_arrow.setText("▴")
        self._refresh_details()
        self._animate_expand(True)

    def _animate_expand(self, opening: bool) -> None:
        self._anim.stop()
        self._opening = opening
        start = self.panel.height() if opening else self._panel_natural_height()
        end = self._panel_natural_height() if opening else 0
        self._anim.setStartValue(float(start))
        self._anim.setEndValue(float(end))
        self._anim.start()

    def _on_expand_step(self, v):
        h = int(float(v))
        self._cur_h = _CAPSULE_H + 6 + max(1, h) + 8
        self.panel.setFixedHeight(max(1, h))
        self.setFixedSize(self._cap_w + 2 * _MARGIN, self._cur_h)

    def _on_expand_done(self):
        if not self._opening:
            self.panel.setVisible(False)
            self._expanded = False
            self.cap_arrow.setText("▾")
            self._cur_h = _CAPSULE_H
            self.setFixedSize(self._cap_w + 2 * _MARGIN, _CAPSULE_H)

    def is_expanded(self) -> bool:
        return self._expanded

    # ---- 面板数据 / 动作 ----
    def _cfg(self):
        from .service.config import load_config

        return load_config()

    def _request(self, path: str, method: bytes, payload=None) -> None:
        req = QNetworkRequest(QUrl(self._poller.base_url() + path))
        req.setRawHeader(b"X-XCcv-Key", self._cfg().api_key.encode("utf-8"))
        req.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        reply = self._nam.sendCustomRequest(req, method, data)
        self._pending[reply] = None
        reply.finished.connect(self._on_finished)

    def _on_finished(self):
        reply = self.sender()
        if reply is None:
            return
        try:
            import sip

            if sip.isdeleted(reply):
                self._pending.pop(reply, None)
                return
            status = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
            raw = bytes(reply.readAll())
        finally:
            reply.deleteLater()
        self._pending.pop(reply, None)
        if status != 200 or not raw:
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            return
        self._fill_details(body if isinstance(body, dict) else {})

    def _refresh_details(self) -> None:
        if self._state != "running":
            return
        # 节流：2s 内重复悬停不重复发请求
        import time as _time

        now = _time.time()
        if now - getattr(self, "_last_details_at", 0.0) < 2.0:
            return
        self._last_details_at = now
        self._request("/v1/models/status", b"GET")

    def _fill_details(self, body: dict) -> None:
        models = body.get("models")
        if isinstance(models, list):
            names = []
            for m in models[:4]:
                alias = str(m.get("alias") or m.get("model_id") or "?")
                task = str(m.get("task") or "")
                names.append(f"{alias}（{task}）" if task else alias)
            if len(models) > len(names):
                names.append(f"+{len(models) - len(names)}")
            self.models_label.setText(
                tr("已加载模型：") + ("、".join(names) if names
                                     else tr("（无）")))

    def _emit_open_deploy(self) -> None:
        self.open_deploy_requested.emit()

    def _copy_key(self) -> None:
        QApplication.clipboard().setText(self._cfg().api_key)

    def _stop_service(self) -> None:
        # 此前裸发 shutdown HTTP（失败静默、无 UI 反馈），打包态
        # "点击后无反应"。改走面板 _stop_service 同链路——「停止中」状态、
        # 宽限期轮询、进程 terminate 兜底全部生效；panel 缺席才裸发。
        panel = getattr(self, "_panel", None)
        hook = getattr(panel, "_stop_service", None) if panel is not None else None
        if hook is not None:
            hook()
            self.cap_label.setText(tr("推理服务停止中…"))
            self._apply_widths(self.cap_label.text())
            return
        self._request("/v1/admin/shutdown", b"POST", {"confirm": True})
        self.stop_requested.emit()

    # ---- 悬停自动展开 / 移走自动收起（验收 2-2 交互修正）----
    def enterEvent(self, event):
        self._expand_open()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._press_pos is None:   # 拖动中鼠标可能瞬间越界：不收起
            self._animate_expand(False)
        super().leaveEvent(event)

    # ---- 拖动 + 位置持久化 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_pos = event.globalPos()
            self._moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is not None:
            delta = event.globalPos() - self._press_pos
            if self._moved or abs(delta.x()) > _DRAG_THRESHOLD \
                    or abs(delta.y()) > _DRAG_THRESHOLD:
                self.move(self.pos() + delta)
                self._press_pos = event.globalPos()
                self._moved = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._moved:
            self._save_pos()   # 点击不再切换展开（hover 驱动），仅拖动后存位
        self._press_pos = None
        self._moved = False
        super().mouseReleaseEvent(event)

    def _restore_pos(self) -> None:
        pos = load_prefs().get("island_pos")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            try:
                self.move(int(pos[0]), int(pos[1]))
                return
            except (TypeError, ValueError):
                pass
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2, geo.top() + 8)

    def _save_pos(self) -> None:
        prefs = load_prefs()
        prefs["island_pos"] = [self.x(), self.y()]
        save_prefs(prefs)

    # ---- 自绘胶囊圆角底 ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # 深色胶囊（白字）：#10333f 深青，与主色同族
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#10333f"))
        p.drawRoundedRect(QRectF(_MARGIN, 0, self._cap_w, self._cur_h),
                          _CAPSULE_H / 2.0, _CAPSULE_H / 2.0)


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402

apply_auto_slot_module(globals())
