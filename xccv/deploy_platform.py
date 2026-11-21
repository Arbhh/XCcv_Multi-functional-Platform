"""部署平台页：推理服务控制 + 「模型加载（API 配置）」+「C# 测试程序」。

主界面第 4 导航页（与 AI平台/传统平台/流程图同级）。本页保持轻量——
不 import torch：模型加载/推理全部经 HTTP 交给推理服务进程
（xccv.service，QNetworkAccessManager 异步访问，不阻塞界面）。

卡片逻辑与 AI 平台页一致：模型加载卡片打开 API 配置窗口（进程内、可
多开复用）；C# 测试程序卡片确保服务在跑后编译/启动 WinForms Demo
（参考随包 C# Demo 形态，直接查看 C# 调用后的
JSON 原文）。
"""

if __name__ == "__main__":  # -m 直启先亮加载中界面（argv 分发须与 _MODES 一致）
    import sys as _sys

    from xccv.platform_entry import early_splash

    _mode = next((a for a in _sys.argv[1:] if a in ("serve", "models", "csharp")), "serve")
    early_splash(*{"serve": ("deploy", "XCcv 推理服务"),
                   "models": ("model_load", "XCcv 模型加载"),
                   "csharp": ("csharp_test", "XCcv C# 测试")}[_mode])

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QApplication, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from . import launcher  # noqa: E402  部署页卡片拉起独立进程
from .app_paths import app_root
from .gui_utils import auto_cleanup_thread, ensure_console_streams
from .i18n import tr  # noqa: E402  V0.7.2 i18n（动态状态文案）
from .win_chrome import ChromeMainWindow


def _tok(name: str) -> str:
    """按当前主题取色（状态提示在设置时取色，暗色下用亮色变体，F2-5）。"""
    from .ui_style import token

    return token(name)

ensure_console_streams()
log = logging.getLogger(__name__)

_DEMO_DIR = app_root() / "platform_data" / "csharp_demo"
_DEMO_EXE = _DEMO_DIR / "XCcvCSharpDemo.exe"
# 编译器 win32icon 源从包内取（xccv/assets 随包分发）——此前 app_root()/
# "xccv"/assets 在安装态（app_root=安装根，xccv 在 app\Lib\site-packages）
# 不存在，csc 报错 → C# 测试无法启动（P-B5-3）；开发态两路径等价
_DEMO_ICON = Path(__file__).resolve().parent / "assets" / "csharp_demo.ico"
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
# .NET Framework 自带编译器（Win10/11 标配，4.x 基线）
_CSC_CANDIDATES = (
    r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
    r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe",
)
_CREATE_NO_WINDOW = 0x08000000  # Windows CREATE_NO_WINDOW


class _HttpMixin:
    """QNetworkAccessManager 回调的安全封装（A8 闪退根治）。

    禁止 `reply.finished.connect(lambda: cb(reply))` 这类持引用写法——
    连接失败/超时时 C++ 侧会立即回收 reply，lambda 访问悬空包装即
    access violation 原生崩溃。改为 sender() 模式：
    连接 self 方法，回调内经 self.sender() 取 reply（Qt 保证 slot 执行
    期间有效），cb 用 {reply: cb} 表传递，处理后 deleteLater。
    """

    def _init_http(self) -> None:
        self._pending: dict = {}

    def _track(self, reply, cb) -> None:
        self._pending[reply] = cb
        reply.finished.connect(self._on_finished)

    def _on_finished(self):
        reply = self.sender()
        if reply is None:
            return
        cb = self._pending.pop(reply, None)
        try:
            import sip

            deleted = sip.isdeleted(reply)
        except Exception:
            log.debug("网络回复回收竞态（跳过手动删除，交由 Qt 回收）")
            deleted = False
        if deleted:      # C++ 侧已回收（超时/错误竞态）：只清表，不访问
            return
        status = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
        raw = bytes(reply.readAll())
        reply.deleteLater()
        if cb is None:
            return
        self._dispatch(reply, status, raw, cb)

    def _dispatch(self, reply, status, raw, cb) -> None:
        """响应分派（与网络栈解耦，可确定性测试）。"""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            body = {"message": raw[:300].decode("utf-8", "ignore")}
        try:
            cb(status, body)
        except Exception:
            log.error("服务响应处理失败", exc_info=True)


def _load_service_cfg():
    from .service.config import load_config

    return load_config()


def _save_service_cfg(cfg) -> None:
    from .service.config import save_config

    save_config(cfg)


def _service_exe_frozen() -> Path | None:
    """打包态的独立服务 exe（在安装根；开发态返回 None）。"""
    exe = Path(sys.prefix).resolve().parent / "XCcv推理服务.exe"
    if not exe.is_file():  # 旧布局兜底：exe 与解释器同目录
        exe = Path(sys.executable).parent / "XCcv推理服务.exe"
    return exe if exe.is_file() else None


def _csc_path() -> Path | None:
    for p in _CSC_CANDIDATES:
        if Path(p).is_file():
            return Path(p)
    return None


def _service_log_tail(lines: int = 3) -> str:
    """读 service.log 尾部（启动失败原因提示用；读不到给指引即可）。"""
    p = app_root() / "platform_data" / "logs" / "service.log"
    try:
        rows = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join("　" + r for r in rows[-lines:])
    except OSError:
        return "　（无法读取 platform_data/logs/service.log）"


class _DemoBuildThread(QThread):
    """csc 编译 C# Demo（产物落 platform_data/csharp_demo/，源码随包）。"""

    finished_build = pyqtSignal(bool, str)

    def run(self):
        try:
            csc = _csc_path()
            if csc is None:
                self.finished_build.emit(
                    False, "未找到 .NET Framework 编译器（csc.exe）。"
                            "请在安装 .NET Framework 4.x 的机器上重试，"
                            "或手动执行 xccv/service/csharp/build_demo.bat")
                return
            src_dir = Path(__file__).resolve().parent / "service" / "csharp"
            _DEMO_DIR.mkdir(parents=True, exist_ok=True)
            cmd = [str(csc), "/nologo", "/target:winexe",
                   f"/out:{_DEMO_EXE}",
                   # 注意：list 传参不经 shell，内嵌引号会变成字面量
                   # （csc 把整串当源文件名 → CS2001，）
                   f"/win32icon:{_DEMO_ICON}",
                   "/r:System.Web.Extensions.dll", "/r:System.Net.Http.dll",
                   "/r:System.dll", "/r:System.Core.dll",
                   "/r:System.Drawing.dll", "/r:System.Windows.Forms.dll",
                   str(src_dir / "XCcv.Vision.Sdk.cs"),
                   str(src_dir / "Demo" / "XCcvDemo.cs")]
            # csc 输出跟随控制台代码页（GBK）：显式按 GBK 解码 + 替换坏字节
            #（UTF-8 模式进程曾解码崩溃）；stdout/stderr 兜底空串防 None 相加
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="gbk", errors="replace",
                                  cwd=str(_DEMO_DIR))
            if proc.returncode != 0:
                self.finished_build.emit(
                    False, (proc.stdout or "") + (proc.stderr or ""))
            else:
                try:
                    # 运行期窗体图标从 exe 同目录直读（绕开系统图标缓存）
                    shutil.copyfile(_DEMO_ICON, _DEMO_DIR / "XCcvDemo.ico")
                except OSError:
                    pass   # 图标复制失败只影响观感，不阻断
                self.finished_build.emit(True, str(_DEMO_EXE))
        except Exception as e:
            log.error("C# Demo 编译失败", exc_info=True)
            self.finished_build.emit(False, f"{type(e).__name__}: {e}")


def _demo_exe_stale() -> bool:
    """exe 缺失，或 C# 源码比 exe 新（改源码后未重编译 → 启动旧界面）时 True。

    此前只判断 exe 是否存在：源码更新后旧 exe 仍被直接启动，用户拿到
    旧版界面（双语化后必现）。
    """
    if not _DEMO_EXE.is_file():
        return True
    src_dir = Path(__file__).resolve().parent / "service" / "csharp"
    try:
        newest = max((src_dir / "Demo" / "XCcvDemo.cs").stat().st_mtime,
                     (src_dir / "XCcv.Vision.Sdk.cs").stat().st_mtime)
    except OSError:
        return False
    return _DEMO_EXE.stat().st_mtime < newest


def _demo_lang_args() -> list[str]:
    """C# Demo 界面语言跟随平台设置（ui_lang.json）：中文界面打开即中文 Demo。"""
    from .i18n import lang

    return ["--lang", lang()]


class ModelServiceWindow(ChromeMainWindow, _HttpMixin):
    """「模型加载」大平台窗口（v0.5.4 起可独立进程启动：
    `pythonw -m xccv.deploy_platform models`）。

    page=None 时按 platform_data/service/config.json 自持地址与 Key
    （独立模式）；传入 page（ServicePanel）时沿用其地址/Key（同窗代理，
    旧签名兼容）。
    """

    def __init__(self, page=None, on_closed=None):
        super().__init__(icon_file="model_load.png")
        self._page = page
        self.on_closed = on_closed
        self._init_http()
        self.setWindowTitle(tr("XCcv 模型加载"))
        self.resize(980, 640)
        from .ui_style import apply_app_theme

        apply_app_theme(self)
        self._nam = QNetworkAccessManager(self)
        central = QWidget()
        self.set_chrome_central(central)
        self._build_ui_into(central)
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(2000)
        self._status_timer.timeout.connect(self._refresh_models)
        self._status_timer.start()
        self._refresh_models()
        from .i18n import localize_widget_tree  # 英文替换层

        localize_widget_tree(self)

    # ---- 独立模式 / 代理模式 ----
    def base_url(self) -> str:
        if self._page is not None:
            return self._page.base_url()
        from .service.config import load_config

        cfg = load_config()
        host = cfg.host if cfg.host not in ("0.0.0.0", "::") else "127.0.0.1"
        return f"http://{host}:{cfg.port}"

    def api_key(self) -> str:
        if self._page is not None:
            return self._page.api_key()
        from .service.config import load_config

        return load_config().api_key

    def service_state(self) -> tuple[bool, str]:
        if self._page is not None:
            return self._page.service_state()
        # 独立模式：主动探测服务端口（此前写死 True，
        # 服务没开也显示「推理服务运行中/加载中」，误导用户）
        return self._probe_service(), ""

    def _probe_service(self) -> bool:
        """端口连通即认为服务在运行（服务监听后 /v1/* 由请求结果说话）。"""
        import socket
        from urllib.parse import urlparse

        try:
            u = urlparse(self.base_url())
            host = u.hostname or "127.0.0.1"
            port = u.port or 80
        except Exception:  # noqa: BLE001
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.6)
            return s.connect_ex((host, port)) == 0

    # ---- UI ----
    def _build_ui_into(self, central):
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(12)

        self.status_label = QLabel()
        root.addWidget(self.status_label)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, stretch=1)

        # 左：加载新模型表单
        box = QGroupBox("加载新模型 / 流程包（模型加载输入）")
        form = QVBoxLayout(box)
        form.setSpacing(8)

        row1 = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "模型（.pt/.engine）或流程包（.xvap/.json）")
        btn_pick = QPushButton("浏览")
        btn_pick.clicked.connect(self._pick_model)
        row1.addWidget(self.path_edit, stretch=1)
        row1.addWidget(btn_pick)
        form.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("API 别名"))
        self.alias_edit = QLineEdit()
        self.alias_edit.setPlaceholderText("留空默认取文件名（供 C# 侧辨识）")
        row2.addWidget(self.alias_edit, stretch=1)
        row2.addWidget(QLabel("设备（仅模型）"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "gpu", "cpu"])
        row2.addWidget(self.device_combo)
        form.addLayout(row2)

        row3 = QHBoxLayout()
        self.warmup_check = QCheckBox("加载后预热一次（首次推理更快）")
        self.warmup_check.setChecked(True)
        row3.addWidget(self.warmup_check)
        row3.addStretch(1)
        self.btn_load = QPushButton("加载到服务")
        self.btn_load.setObjectName("btnAccent")
        self.btn_load.clicked.connect(self._load_model)
        row3.addWidget(self.btn_load)
        form.addLayout(row3)

        hint = QLabel(tr("说明：模型在服务进程内加载，接口不返回权重文件；"
                         "每个已加载模型/流程是一个可并行的 API（按 id 寻址）。"
                         "流程包内嵌的 AI 模型在首次推理时自动加载并常驻复用。"
                         "输出为统一检出 JSON（schema_version=1）。"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748b; font-size:13px; border:none;")
        form.addWidget(hint)
        body.addWidget(box, stretch=2)

        # 右：已加载模型/流程表
        box2 = QGroupBox("已加载 API（模型 + 流程，多 API 并行）")
        v2 = QVBoxLayout(box2)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["id", "别名", "类型", "设备/任务", ""])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(self.table.SelectRows)
        v2.addWidget(self.table, stretch=1)
        row4 = QHBoxLayout()
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._refresh_models)
        row4.addWidget(btn_refresh)
        row4.addStretch(1)
        v2.addLayout(row4)
        body.addWidget(box2, stretch=3)

    # ---- HTTP ----
    def _url(self, path: str) -> QUrl:
        return QUrl(self.base_url() + path)

    def _request(self, path: str) -> QNetworkRequest:
        req = QNetworkRequest(self._url(path))
        req.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        req.setRawHeader(b"X-XCcv-Key", self.api_key().encode("utf-8"))
        return req

    def _send(self, req: QNetworkRequest, payload, cb) -> None:
        """异步请求（sender 模式回调，见 _HttpMixin——禁止 lambda 持 reply）。"""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        method = b"GET" if payload is None else b"POST"
        reply = self._nam.sendCustomRequest(req, method, data)
        self._track(reply, cb)

    # ---- 动作 ----
    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型 / 流程包", "",
            "模型与流程 (*.pt *.engine *.xvap *.json);;"
            "流程包 (*.xvap *.json);;所有文件 (*)")
        if path:
            self.path_edit.setText(path)
            if not self.alias_edit.text().strip():
                self.alias_edit.setText(Path(path).stem)

    def _load_model(self):
        path = self.path_edit.text().strip()
        if not path or not Path(path).is_file():
            QMessageBox.information(self, tr("加载"), tr("请先选择有效的模型/流程文件"))
            return
        suffix = Path(path).suffix.lower()
        alias = self.alias_edit.text().strip()
        if suffix in (".xvap", ".json"):
            # 流程包：常驻为流程 API（内嵌模型首次推理时自动加载并常驻）
            payload = {"path": path, "alias": alias}
            self._send(self._request("/v1/flows/load"), payload,
                       self._after_load)
            return
        payload = {"path": path, "device": self.device_combo.currentText(),
                   "alias": alias,
                   "warmup": self.warmup_check.isChecked()}
        self._send(self._request("/v1/models/load"), payload,
                   self._after_load)

    def _after_load(self, status, body):
        if status is None:
            # 连接失败（服务未运行/中途退出）：人话提示，不显示 None
            QMessageBox.critical(
                self, tr("加载失败"),
                tr("无法连接推理服务（服务可能未运行或刚退出）。")
                + "\n" + tr("请回到主界面「部署平台」页确认服务状态后重试。"))
            self._refresh_models()
            return
        if status != 200:
            QMessageBox.critical(self, tr("加载失败"),
                                 f"{tr('服务返回')} {status}: {body.get('message', '')}")
            return
        self.path_edit.clear()
        self.alias_edit.clear()
        self._refresh_models()
        mid = body.get("model_id", "")
        from .ui_widgets import toast

        toast(self, f"{tr('已加载')} {mid}（task={body.get('task', '')}）",
              "success")

    def _refresh_models(self):
        running, _info = self.service_state()
        self.btn_load.setEnabled(running)   # 未运行禁用加载（A8 防御）
        if running:
            self.status_label.setText(
                f"{tr('● 推理服务运行中：')}{self.base_url()}"
                f"{tr('（加载/推理在服务进程内完成，权重文件不出进程）')}")
            self.status_label.setStyleSheet("color:#166534; border:none;")
            self._send(self._request("/v1/models/status"), None, self._fill_models)
        else:
            self.status_label.setText(
                tr("○ 推理服务未运行——请先在主界面「部署平台」页点击「启动服务」"))
            self.status_label.setStyleSheet("color:#b45309; border:none;")
            self.table.setRowCount(0)

    def _fill_models(self, status, body):
        """模型表填充第一段：拉取后缓存，再拉流程表合并渲染。"""
        self._last_models = body.get("models", []) if isinstance(body, dict) else []
        self._send(self._request("/v1/flows/status"), None, self._fill_all)

    def _fill_all(self, status, body):
        flows = body.get("flows", []) if isinstance(body, dict) else []
        models = getattr(self, "_last_models", [])
        self.table.setRowCount(0)
        for m in models:
            self._append_row(str(m.get("model_id", "")),
                             str(m.get("alias", "")), "模型",
                             str(m.get("task", "")), is_flow=False)
        for f in flows:
            self._append_row(str(f.get("flow_id", "")),
                             str(f.get("alias", "")), "流程", "-", is_flow=True)
        if not models and not flows:
            self.statusBar().showMessage(tr("服务上暂无已加载模型/流程"), 4000)

    def _append_row(self, item_id: str, alias: str, kind: str,
                    detail: str, is_flow: bool):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(item_id))
        self.table.setItem(row, 1, QTableWidgetItem(alias))
        self.table.setItem(row, 2, QTableWidgetItem(kind))
        self.table.setItem(row, 3, QTableWidgetItem(detail))
        btn = QPushButton("卸载")
        btn.clicked.connect(lambda _=False, x=item_id, fl=is_flow:
                            self._unload(x, fl))
        self.table.setCellWidget(row, 4, btn)

    def _unload(self, item_id: str, is_flow: bool):
        path = "/v1/flows/unload" if is_flow else "/v1/models/unload"
        self._send(self._request(path), {"model_id": item_id},
                   lambda s, b: self._refresh_models())

    def closeEvent(self, event):
        self._status_timer.stop()
        if self.on_closed is not None:
            self.on_closed()
        super().closeEvent(event)


class ServicePanel(QWidget, _HttpMixin):
    """推理服务控制面板（嵌入 ServiceWindow；v0.5.2 前身为主界面内嵌页
    DeployPage，v0.5.4 部署页改卡片化后本类专职服务控制）。"""

    _WATCHDOG_TICKS = 20     # 启动看门狗：1s×20，超时仍不健康按失败提示
    open_deploy_requested = pyqtSignal()   # 灵动岛「打开部署平台」转发给主界面

    def __init__(self, poller=None):
        super().__init__()
        self._proc: subprocess.Popen | None = None
        self._service_ok = False
        self._service_starting = False
        self._service_stopping = False   # Q1：停止中独立状态（不再是瞬间跳未运行）
        self._model_window: ModelServiceWindow | None = None
        self._build_thread: _DemoBuildThread | None = None
        self._island = None
        self._init_http()
        self._cfg = _load_service_cfg()
        self._nam = QNetworkAccessManager(self)
        self._watchdog_left = 0
        self._wait_restarts = 0
        self._poller = poller
        self._build_ui()
        if poller is not None:
            # 主界面注入共享健康轮询：单一请求源，本页只订阅信号
            # （灵动岛 / 顶栏状态胶囊共用同一实例，避免双份请求）
            poller.state_changed.connect(self._on_poll_state)
            self._on_poll_state(poller.state(),
                                {"url": "", "models": 0, "device": ""})
        else:
            # 独立构造（测试/独立运行）：沿用本页自轮询
            self._health_timer = QTimer(self)
            self._health_timer.setInterval(1500)
            self._health_timer.timeout.connect(self._poll_health)
            self._health_timer.start()
            self._poll_health()
        if self._poller is not None:
            # 灵动岛（D3 首选形态：独立桌面悬浮窗）；开关状态取 ui_prefs
            from .ui_widgets import load_prefs

            self.island_check.setChecked(
                bool(load_prefs().get("island_enabled", True)))
            self.island_check.toggled.connect(self._sync_island)
            self._sync_island()

    # ---- 灵动岛 ----
    def _sync_island(self) -> None:
        if self._poller is None:
            return   # 独立构造（测试）：无共享轮询，不起悬浮窗
        from .ui_widgets import load_prefs, save_prefs

        enabled = self.island_check.isChecked()
        prefs = load_prefs()
        prefs["island_enabled"] = enabled
        save_prefs(prefs)
        if enabled and self._island is None:
            from .island import ServiceIsland

            self._island = ServiceIsland(self._poller, self)
            self._island.open_deploy_requested.connect(self._forward_open_deploy)
            self._island.show()
        elif not enabled and self._island is not None:
            island, self._island = self._island, None
            island.close()
            island.deleteLater()

    def _forward_open_deploy(self) -> None:
        self.open_deploy_requested.emit()

    # ---- 拓展区（v0.5.4）：GPU 显存 / 接口清单 / 在线文档 ----
    _GPU_INFO = {}

    def _open_docs(self) -> None:
        """浏览器打开 /docs 在线接口文档（服务运行时）。"""
        import os as _os

        if not self._service_ok:
            from .ui_widgets import toast

            toast(self, tr("服务未运行：请先启动推理服务"), "warn")
            return
        url = f"{self.base_url()}/docs-lite"   # 自包含页（无 JS 依赖，必非空白）
        try:
            _os.startfile(url)   # noqa: S606 Windows 默认浏览器
        except OSError:
            QMessageBox.information(self, tr("接口文档"), url)

    def _refresh_api_list(self) -> None:
        """拉取 /openapi.json，解析接口清单（路径 + 方法）。"""
        if not self._service_ok:
            self.api_list.clear()
            return
        req = QNetworkRequest(QUrl(self.base_url() + "/openapi.json"))
        req.setRawHeader(b"X-XCcv-Key", self.api_key().encode("utf-8"))
        reply = self._nam.get(req)
        self._track(reply, self._on_openapi)

    def _on_openapi(self, status, body) -> None:
        self.api_list.clear()
        if status != 200 or not isinstance(body, dict):
            item = QListWidgetItem("（无法读取接口清单）")
            self.api_list.addItem(item)
            return
        for path, methods in sorted((body.get("paths") or {}).items()):
            verbs = "/".join(sorted(m.upper() for m in methods
                                    if isinstance(methods, dict)))
            QListWidgetItem(f"{verbs:6s} {path}", self.api_list)

    def _update_gpu(self, info: dict) -> None:
        free = info.get("gpu_free_mb")
        total = info.get("gpu_total_mb")
        if not free or not total:
            self.gpu_bar.setValue(0)
            self.gpu_bar.setFormat("无 GPU 信息")
            self.gpu_text.setText("—")
            return
        used = max(0.0, float(total) - float(free))
        pct = int(round(used / float(total) * 100))
        self.gpu_bar.setValue(pct)
        self.gpu_bar.setFormat(f"{pct}%")
        self.gpu_text.setText(f"{used:.0f} / {total:.0f} MB")

    # ---- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(44, 26, 44, 18)
        root.setSpacing(16)

        title = QLabel("推理服务")
        title.setObjectName("pageTitle")
        sub = QLabel("把训练好的模型/流程以 HTTP API 提供给上位机调用")
        sub.setObjectName("pageSub")
        root.addWidget(title)
        root.addWidget(sub)

        # 服务控制区
        box = QGroupBox("推理服务")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)
        row_status = QHBoxLayout()
        self.status_label = QLabel("○ 未运行（正在检测…）")
        self.status_label.setStyleSheet("border:none;")
        row_status.addWidget(self.status_label, stretch=1)
        self.btn_toggle = QPushButton("启动服务")
        self.btn_toggle.setObjectName("btnAccent")
        self.btn_toggle.clicked.connect(self._toggle_service)
        row_status.addWidget(self.btn_toggle)
        lay.addLayout(row_status)
        self.tip_label = QLabel("")
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet("color:#2563eb; font-size:13px; border:none;")
        lay.addWidget(self.tip_label)
        # GPU 状态条（nvidia-smi 10s 轮询，服务运行时启动；
        # 无 GPU/无 nvidia-smi 自动隐藏——零噪音）
        self.gpu_label = QLabel("")
        self.gpu_label.setStyleSheet("border:none; font-size:12px;")
        self.gpu_label.setVisible(False)
        lay.addWidget(self.gpu_label)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(10000)
        self._gpu_timer.timeout.connect(self._refresh_gpu_line)

        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("监听地址"))
        self.host_combo = QComboBox()
        self.host_combo.setEditable(True)
        self.host_combo.addItems(["127.0.0.1", "0.0.0.0"])
        self.host_combo.setCurrentText(str(self._cfg.host))
        self.host_combo.setMinimumWidth(150)
        row_cfg.addWidget(self.host_combo)
        row_cfg.addWidget(QLabel("端口"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(int(self._cfg.port))
        row_cfg.addWidget(self.port_spin)
        row_cfg.addWidget(QLabel("API Key"))
        self.key_edit = QLineEdit(self._cfg.api_key)
        self.key_edit.setReadOnly(True)
        row_cfg.addWidget(self.key_edit, stretch=1)
        btn_copy = QPushButton("复制")
        btn_copy.clicked.connect(self._copy_key)
        row_cfg.addWidget(btn_copy)
        btn_regen = QPushButton("重新生成")
        btn_regen.clicked.connect(self._regen_key)
        row_cfg.addWidget(btn_regen)
        lay.addLayout(row_cfg)

        row_island = QHBoxLayout()
        self.island_check = QCheckBox(
            "桌面悬浮状态球（灵动岛）——服务状态常驻屏幕顶部，可拖动、"
            "点击展开快捷操作")
        row_island.addWidget(self.island_check)
        row_island.addStretch(1)
        lay.addLayout(row_island)

        # 平台-部署推理一致性自检（一次性工具）
        row_chk = QHBoxLayout()
        self.btn_consistency = QPushButton("一致性自检（本地 vs 服务同图对比）")
        self.btn_consistency.setToolTip(
            "选同一张图：本地加载该权重推理一次、经本服务 HTTP 推理一次，\n"
            "逐框比对（IoU≥0.98、同类别、分数差<0.02）。架构上两边走同一个\n"
            "predict 链路，结果应一致；不一致的明细用于定位加载/环境问题")
        self.btn_consistency.clicked.connect(self._run_consistency_check)
        row_chk.addWidget(self.btn_consistency)
        row_chk.addStretch(1)
        lay.addLayout(row_chk)
        self.consistency_result = QLabel("")
        self.consistency_result.setWordWrap(True)
        self.consistency_result.setStyleSheet("border:none; font-size:12px;")
        lay.addWidget(self.consistency_result)

        hint = QLabel("接口文档 /docs　·　鉴权：请求头 X-XCcv-Key")
        hint.setObjectName("muted")
        lay.addWidget(hint)
        root.addWidget(box)

        # ---- 拓展区：GPU 显存 + 接口清单 + 文档入口（v0.5.4，页面不空泛）----
        ext = QGroupBox("运行洞察")
        ext_lay = QVBoxLayout(ext)
        ext_lay.setSpacing(8)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(QLabel("GPU 显存"))
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setRange(0, 100)
        self.gpu_bar.setValue(0)
        self.gpu_bar.setFormat("服务运行后显示")
        gpu_row.addWidget(self.gpu_bar, stretch=1)
        self.gpu_text = QLabel("—")
        self.gpu_text.setObjectName("muted")
        gpu_row.addWidget(self.gpu_text)
        ext_lay.addLayout(gpu_row)

        api_row = QHBoxLayout()
        api_row.addWidget(QLabel("接口清单"))
        self.api_list = QListWidget()
        self.api_list.setObjectName("apiList")
        self.api_list.setMaximumHeight(132)
        api_row.addWidget(self.api_list, stretch=1)
        api_btns = QVBoxLayout()
        self.btn_docs = QPushButton("接口文档")
        self.btn_docs.setToolTip(tr("在浏览器打开 /docs 在线文档（服务运行时可用）"))
        self.btn_docs.clicked.connect(self._open_docs)
        self.btn_apis = QPushButton("刷新接口")
        self.btn_apis.setToolTip(tr("拉取 /openapi.json 解析接口清单"))
        self.btn_apis.clicked.connect(self._refresh_api_list)
        api_btns.addWidget(self.btn_docs)
        api_btns.addWidget(self.btn_apis)
        api_btns.addStretch(1)
        api_row.addLayout(api_btns)
        ext_lay.addLayout(api_row)
        root.addWidget(ext)
        root.addStretch(1)

    def _make_card(self, title: str, grad_from: str, grad_to: str,
                   desc: str, on_click) -> QFrame:
        from .ui_widgets import HoverCard, gradient_tile

        card = HoverCard()
        card.setFixedSize(360, 178)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 16, 18, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(12)
        icon = QLabel()
        icon.setPixmap(gradient_tile("tools.png", grad_from, grad_to, 44, 12))
        icon.setFixedSize(44, 44)
        head.addWidget(icon)
        t = QLabel(title)
        t.setStyleSheet(f"color:{grad_to}; font-size:19px; font-weight:bold;"
                        "background:transparent; border:none;")
        head.addWidget(t, stretch=1)
        lay.addLayout(head)
        d = QLabel(desc)
        d.setObjectName("muted")
        d.setWordWrap(True)
        lay.addWidget(d, stretch=1)
        tip = QLabel("打开 →")
        tip.setAlignment(Qt.AlignRight)
        tip.setStyleSheet(f"color:{grad_to}; font-size:15px; font-weight:bold;"
                          "background:transparent; border:none;")
        lay.addWidget(tip)
        card.clicked.connect(on_click)
        return card

    # ---- 服务控制 ----
    def base_url(self) -> str:
        """本机访问地址（0.0.0.0 绑定时本机仍走 127.0.0.1）。"""
        host = "127.0.0.1" if self._bind_host() in ("0.0.0.0", "::") \
            else self._bind_host()
        return f"http://{host}:{self.port_spin.value()}"

    def _bind_host(self) -> str:
        return self.host_combo.currentText().strip() or "127.0.0.1"

    def api_key(self) -> str:
        return self.key_edit.text().strip()

    def service_state(self) -> tuple[bool, str]:
        return self._service_ok, ""

    def _poll_health(self):
        req = QNetworkRequest(QUrl(self.base_url() + "/v1/health"))
        reply = self._nam.get(req)
        self._track(reply, self._on_health)

    def _on_health(self, status, body):
        # 无条件刷新（A7：此前只在状态翻转时刷新，失败→失败会永久停在
        # 「检测中」）；status 为 None = 连接失败
        self._apply_poll(status == 200, status is None)

    def _on_poll_state(self, state: str, info: dict):
        """共享轮询（ServiceHealthPoller）的三态回调：running/starting/stopped。"""
        self._apply_poll(state == "running", state == "stopped", info)

    def _apply_poll(self, running: bool, stopped: bool, info: dict | None = None):
        """轮询结果 → 状态机（running/starting/stopping/stopped + Q1 提示闭环）。"""
        info = info or {}
        self._service_ok = running
        self._service_starting = (not running and not stopped
                                  and not self._service_stopping)
        if running and self._watchdog_left:
            self._watchdog_left = 0
        if running:
            self._update_gpu(info)
        if self._service_stopping:
            if running:
                # shutdown 后服务优雅退出需要时间（uvicorn 收尾在途请求，
                # 数百毫秒到数秒），期间 health 仍 200——宽限期内不算失败
                #（此前立即误报"停止失败"，随后服务又正常退出）
                if getattr(self, "_stop_grace", 0) > 0:
                    self._stop_grace -= 1
                else:
                    self._service_stopping = False
                    self.tip_label.setStyleSheet(
                        f"color:{_tok('danger_deep')}; font-size:13px; border:none;")
                    self.tip_label.setText(
                        "停止失败：服务仍在响应（端口可能被其他进程占用），"
                        "请到任务管理器结束对应 python 进程后重试")
            elif stopped:
                self._service_stopping = False
                self.tip_label.setStyleSheet(
                    f"color:{_tok('success_deep')}; font-size:13px; border:none;")
                self.tip_label.setText(tr("服务已停止"))
                from .ui_widgets import toast

                toast(self, tr("推理服务已停止"), "success")
        # 拓展区随状态更新（GPU 条 / 接口清单可用性）
        if running:
            pass   # GPU 条已在上方 _update_gpu 更新；清单按需刷新
        else:
            self.gpu_bar.setValue(0)
            self.gpu_bar.setFormat("服务运行后显示")
            self.gpu_text.setText("—")
            self.api_list.clear()
        self._refresh_status()

    def _refresh_status(self):
        from .i18n import tr

        if self._service_stopping:
            # Q1：点击停止后保持「正在停止」显示，由轮询确认后再切换
            self.status_label.setText(tr("● 正在停止服务…"))
            self.status_label.setStyleSheet("color:#b45309; border:none;")
            self.btn_toggle.setText(tr("停止服务"))
            self.btn_toggle.setEnabled(False)
        elif self._service_ok:
            extra = ""
            if self._bind_host() in ("0.0.0.0", "::"):
                extra = (f"（{tr('对外监听：其他机器用')} <{tr('本机IP')}>:"
                         f"{self.port_spin.value()} {tr('访问')}）")
            self.status_label.setText(
                f"{tr('● 运行中：')}{self.base_url()}{extra}")
            self.status_label.setStyleSheet("color:#166534; border:none;")
            self.btn_toggle.setText(tr("停止服务"))
            self.btn_toggle.setEnabled(True)
        elif self._service_starting:
            self.status_label.setText(tr("● 启动中：正在加载深度学习组件…"))
            self.status_label.setStyleSheet("color:#b45309; border:none;")
            self.btn_toggle.setText(tr("启动服务"))
            self.btn_toggle.setEnabled(False)   # 防止启动期重复拉起进程
        else:
            self.status_label.setText(tr("○ 未运行（启动后供 C# 上位机调用）"))
            self.status_label.setStyleSheet("color:#b45309; border:none;")
            self.btn_toggle.setText(tr("启动服务"))
            self.btn_toggle.setEnabled(True)

    def _toggle_service(self):
        if self._service_ok:
            self._stop_service()
        else:
            self._start_service()

    def _port_in_use(self) -> bool:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            return s.connect_ex(("127.0.0.1", self.port_spin.value())) == 0

    def _start_service(self):
        from .i18n import tr

        self.tip_label.setStyleSheet("color:#2563eb; font-size:13px; border:none;")
        if self._service_ok:
            self.tip_label.setText(tr("服务已在运行，无需重复启动"))
            return
        self._service_stopping = False   # 停止过程中允许重新启动
        # 启动前持久化地址/端口（监听地址可设；跨机监听时服务端安全
        # 收口由 ServiceConfig.__post_init__ 强制启用路径白名单）
        host = self._bind_host()
        if host not in ("127.0.0.1", "localhost", "::1"):
            ret = QMessageBox.warning(
                self, tr("对外监听确认"),
                f"{tr('监听地址')} {host} "
                f"{tr('允许其他机器访问推理服务：')}\n\n"
                f"· {tr('跨机部署时模型路径白名单将被强制启用')}"
                "（platform_data/service/config.json 的 model_dirs）\n"
                f"· {tr('当前接口为 HTTP 明文，生产环境建议置于内网/反向代理之下')}\n\n"
                f"{tr('确认以此地址启动吗？')}",
                QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                return
        if self._port_in_use():
            self.tip_label.setStyleSheet(f"color:{_tok('danger_deep')}; font-size:13px; border:none;")
            self.tip_label.setText(
                f"{tr('端口')} {self.port_spin.value()} "
                f"{tr('已被占用：可能是旧的推理服务残留。请更换端口，或在任务管理器结束旧 python 进程后重试')}")
            return
        self._cfg = _load_service_cfg()
        self._cfg.host = host
        self._cfg.port = self.port_spin.value()
        _save_service_cfg(self._cfg)

        exe = _service_exe_frozen()
        if exe is not None:
            cmd = [str(exe), "--host", host, "--port", str(self._cfg.port)]
        else:
            # 编译安装态 xccv 是 .pyd，不能用 -m（"No code object"秒退，
            # P-09 根因）；importlib 直调 main()（argv 透传 argparse）
            code = ("import importlib; "
                    "importlib.import_module('xccv.service.main').main()")
            cmd = [sys.executable, "-c", code,
                   "--host", host, "--port", str(self._cfg.port)]
        try:
            flags = _CREATE_NO_WINDOW if os.name == "nt" else 0
            # 显式重定向 stdio 到 service.log：主界面是 GUI 进程（无控制台），
            # 子进程继承到的 stdio 句柄无效，写即异常退出（A2 根因）——
            # 文件句柄同样解决，且启动早期（导入期）崩溃也一定留痕
            #（此前 DEVNULL 吞掉全部输出，服务秒死无任何日志）
            log_dir = app_root() / "platform_data" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            logf = open(log_dir / "service.log", "a", encoding="utf-8",
                        errors="replace")
            self._proc = subprocess.Popen(
                cmd, creationflags=flags, cwd=str(app_root()),
                stdin=subprocess.DEVNULL, stdout=logf,
                stderr=subprocess.STDOUT)
        except Exception as e:
            log.warning("推理服务启动失败（已弹窗提示）: %s", e, exc_info=True)
            from .i18n import tr

            QMessageBox.critical(self, tr("启动服务失败"),
                                 f"{type(e).__name__}: {e}")
            return
        self.tip_label.setText(tr("服务进程已启动，正在加载深度学习组件…"))
        if self._poller is not None:
            self._poller.note_starting()   # 共享轮询立即切「启动中」态
        # GPU 状态条随服务运行启动（10s 低频轮询）
        self._refresh_gpu_line()
        self._gpu_timer.start()
        # 启动看门狗（A2：进程退出/端口占用时给出可见失败原因，不再"无反应"）
        self._watchdog_left = self._WATCHDOG_TICKS
        self._watchdog()

    def _refresh_gpu_line(self):
        """单次 nvidia-smi 查询刷新 GPU 状态行（不可用即隐藏并停表）。"""
        from .gpu_monitor import format_gpu_line, query_gpu

        info = query_gpu()
        if info is None:
            self.gpu_label.setVisible(False)
            self._gpu_timer.stop()
            return
        self.gpu_label.setText(format_gpu_line(info))
        self.gpu_label.setVisible(True)

    def _watchdog(self):
        if not self._watchdog_left:
            return
        self._watchdog_left -= 1
        if self._service_ok:
            self.tip_label.setText("")
            return
        proc = self._proc
        if proc is not None and proc.poll() is not None:
            # 子进程已退出：失败原因在 service.log（uvicorn 错误已接入落盘）
            self._watchdog_left = 0
            self.tip_label.setStyleSheet(
                f"color:{_tok('danger_deep')}; font-size:13px; border:none;")
            tail = _service_log_tail()
            self.tip_label.setText(
                f"{tr('服务进程启动失败，已退出（退出码')} {proc.poll()}）。"
                f"{tr('原因（service.log 尾部）：')}\n" + tail)
            return
        if self._watchdog_left <= 0:
            self.tip_label.setStyleSheet(
                f"color:{_tok('danger_deep')}; font-size:13px; border:none;")
            self.tip_label.setText(
                tr("等待健康检查超时：服务进程仍在运行但未就绪（首次加载较慢可"
                   "稍候；持续不就绪请查看 platform_data/logs/service.log）"))
            return
        QTimer.singleShot(1000, self._watchdog)

    def _stop_service(self):
        # Q1：进入「停止中」状态显示；进程退出由健康轮询确认后闭环提示
        self._service_stopping = True
        self._gpu_timer.stop()          # 服务停止即停 GPU 轮询
        self.gpu_label.setVisible(False)
        self._stop_grace = 12   # 停止宽限期（轮询 tick 数，约 12s）：优雅退出期间 health 仍 200 不算失败
        self._refresh_status()
        self.tip_label.setStyleSheet("color:#2563eb; font-size:13px; border:none;")
        self.tip_label.setText(tr("正在停止服务…"))
        try:
            req = QNetworkRequest(QUrl(self.base_url() + "/v1/admin/shutdown"))
            req.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
            req.setRawHeader(b"X-XCcv-Key", self.api_key().encode("utf-8"))
            data = json.dumps({"confirm": True}).encode("utf-8")
            reply = self._nam.sendCustomRequest(req, b"POST", data)
            self._track(reply, None)
        except Exception:
            log.warning("优雅停止请求失败，将直接结束进程", exc_info=True)
        proc, self._proc = self._proc, None
        if proc is not None:
            QTimer.singleShot(2500, lambda: self._terminate_if_alive(proc))

    @staticmethod
    def _terminate_if_alive(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.terminate()

    def stop_service_blocking(self, max_ms: int = 4000) -> None:
        """部署平台真正退出时的同步停服。

        与 _stop_service 的异步闭环不同：退出前没有事件循环可等，这里在
        本线程内做短事件循环等待 shutdown 应答（最多 max_ms），再对本进程
        拉起的子进程 terminate 兜底（外部启动的服务只靠 HTTP 优雅停止）。
        """
        from PyQt5.QtCore import QEventLoop

        if self._service_ok:
            try:
                req = QNetworkRequest(QUrl(self.base_url() + "/v1/admin/shutdown"))
                req.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
                req.setRawHeader(b"X-XCcv-Key", self.api_key().encode("utf-8"))
                data = json.dumps({"confirm": True}).encode("utf-8")
                loop = QEventLoop()
                reply = self._nam.sendCustomRequest(req, b"POST", data)
                reply.finished.connect(loop.quit)
                QTimer.singleShot(max_ms, loop.quit)
                loop.exec_()
            except Exception:
                log.warning("退出前停服请求失败（将按进程兜底处理）", exc_info=True)
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(2000)
            except Exception:
                log.debug("服务进程收尾 kill 失败（进程可能已退出）")
                proc.kill()

    def _copy_key(self):
        QApplication.clipboard().setText(self.api_key())
        from .ui_widgets import toast

        toast(self, "API Key 已复制到剪贴板", "success")

    # ---- 一致性自检（本地 vs 服务同图对比） ----
    def _run_consistency_check(self):
        if not self._service_ok:
            QMessageBox.information(self, tr("一致性自检"), tr("请先启动推理服务"))
            return
        model_path, _ = QFileDialog.getOpenFileName(
            self, "选择参与自检的模型权重（服务能加载的同一份）", "",
            "权重 (*.pt *.engine)")
        if not model_path:
            return
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择测试图片（本地与服务用同一张）", "",
            "图片 (*.png *.jpg *.jpeg *.bmp)")
        if not image_path:
            return
        self.btn_consistency.setEnabled(False)
        self.consistency_result.setText(tr("自检中：本地与服务各推理一次…"))
        from .deploy_consistency import ConsistencyThread

        self._cc_thread = ConsistencyThread(
            model_path, image_path, self.base_url(), self.api_key(),
            conf=float(self._cfg.infer_conf), iou=float(self._cfg.infer_iou),
            parent=self)
        self._cc_thread.done.connect(self._on_consistency_done)
        self._cc_thread.failed.connect(self._on_consistency_failed)
        auto_cleanup_thread(self, "_cc_thread", self._cc_thread)  # A-3
        self._cc_thread.start()

    def _on_consistency_done(self, out: dict):
        self.btn_consistency.setEnabled(True)
        if out.get("consistent"):
            self.consistency_result.setText(
                f"{tr('✔ 一致：本地与服务均')} "
                f"{out.get('local_detections', 0)} "
                f"{tr('个检出（IoU≥0.98、同类别、分数差<0.02）——加载/环境无异常')}")
            self.consistency_result.setStyleSheet("border:none; font-size:12px; color:#166534;")
        else:
            self.consistency_result.setText(
                tr("✘ 不一致（明细）：\n")
                + "\n".join(out.get("details") or [])[:600])
            self.consistency_result.setStyleSheet(
                f"border:none; font-size:12px; color:{_tok('danger_deep')};")

    def _on_consistency_failed(self, message: str):
        self.btn_consistency.setEnabled(True)
        self.consistency_result.setText(f"{tr('自检失败：')}{message}")
        self.consistency_result.setStyleSheet(
            f"border:none; font-size:12px; color:{_tok('danger_deep')};")

    def _regen_key(self):
        from .service.config import generate_api_key

        if QMessageBox.question(
                self, tr("重新生成 API Key"),
                tr("旧的 Key 将立即失效，已接入的 C# 端需要更新。")
                + tr("服务运行中会自动重启生效。继续吗？")) != QMessageBox.Yes:
            return
        self._cfg = _load_service_cfg()
        self._cfg.port = self.port_spin.value()
        self._cfg.api_key = generate_api_key()
        _save_service_cfg(self._cfg)
        self.key_edit.setText(self._cfg.api_key)
        if self._service_ok:
            # 旧实现固定延时 3s 后启动——旧进程未退完（健康轮询又把
            # 状态翻回"运行中"）导致"未运行+已在运行"矛盾、新服务起不来。
            # 改为轮询等待端口真正释放后再启动（最多 30s）。
            self._stop_service()
            self._wait_restarts = 60
            self.tip_label.setText(tr("正在重启服务（等待旧进程退出）…"))
            self._restart_when_stopped()

    def _restart_when_stopped(self):
        # 以端口释放为准（轮询间隙旧服务的最后一次健康回包会短暂把
        # _service_ok 翻回 True，不影响本判定）
        if not self._port_in_use():
            self.tip_label.setText(tr("旧服务已退出，正在启动新服务…"))
            self._start_service()
            return
        self._wait_restarts -= 1
        if self._wait_restarts <= 0:
            self.tip_label.setStyleSheet(f"color:{_tok('danger_deep')}; font-size:13px; border:none;")
            self.tip_label.setText(
            tr("重启超时：旧服务迟迟未退出，请手动停止后再启动"))
            return
        QTimer.singleShot(500, self._restart_when_stopped)

    # ---- 卡片动作 ----
    def _open_model_window(self):
        if self._model_window is not None and self._model_window.isVisible():
            self._model_window.activateWindow()
            self._model_window.raise_()
            return
        self._model_window = ModelServiceWindow(self, on_closed=None)
        self._model_window.show()

    def _launch_demo(self):
        if not self._service_ok:
            QMessageBox.information(
                self, tr("C# 测试程序"), tr("推理服务未运行，请先点击「启动服务」"))
            return
        if not _demo_exe_stale():
            self._run_demo()
            return
        if _csc_path() is None:
            QMessageBox.warning(
                self, tr("C# 测试程序"),
                tr("本机缺少 .NET Framework 编译器（csc.exe），无法自动编译。")
                + "\n" + tr("请在装有 .NET Framework 4.x 的目标机器上打开本功能，")
                + tr("或手动运行 xccv\\service\\csharp\\build_demo.bat"))
            return
        self.btn_toggle.setEnabled(False)
        self._build_thread = _DemoBuildThread()
        self._build_thread.finished_build.connect(self._after_demo_build)
        auto_cleanup_thread(self, "_build_thread", self._build_thread)  # A-3
        self._build_thread.start()

    def _after_demo_build(self, ok: bool, msg: str):
        self.btn_toggle.setEnabled(True)
        self._build_thread = None
        if not ok:
            QMessageBox.critical(self, tr("C# 测试程序编译失败"), msg[:2000])
            return
        self._run_demo()

    def _run_demo(self):
        subprocess.Popen([str(_DEMO_EXE), "--url", self.base_url(),
                          "--key", self.api_key(), *_demo_lang_args()],
                         cwd=str(_DEMO_DIR), close_fds=True)

    def closeEvent(self, event):
        timer = getattr(self, "_health_timer", None)
        if timer is not None:
            timer.stop()   # 共享轮询模式下本页无自有定时器
        if self._island is not None:
            self._island.close()
        super().closeEvent(event)


class ServiceWindow(ChromeMainWindow):
    """推理服务窗口（独立进程；v0.5.4 起由主界面「部署平台」页卡片拉起，
    也可 `pythonw -m xccv.deploy_platform serve` 独立启动）。

    - 窗口内容 = ServicePanel（服务控制 + GPU/接口拓展 + 灵动岛开关）；
    - 关闭窗口 = 最小化到系统托盘（推理服务继续运行，灵动岛不消失）；
    - 托盘菜单：显示窗口 / 灵动岛开关 / 停止推理服务 / 真正退出。
    """

    def __init__(self, poller=None):
        super().__init__(icon_file="deploy.png")
        self._force_quit = False
        self.setWindowTitle(tr("XCcv 推理服务"))
        self.resize(1080, 720)
        from .ui_style import apply_app_theme

        apply_app_theme(self)

        from .service_health import ServiceHealthPoller

        self._poller = poller or ServiceHealthPoller(self)
        self._page = ServicePanel(poller=self._poller)
        # 灵动岛「部署平台」按钮 → 激活本窗口（原主界面跳转语义）
        self._page.open_deploy_requested.connect(self._restore_from_tray)
        # 无边框基座：内容挂在顶栏之下
        self.set_chrome_central(self._page)
        self._build_tray()
        self._poller.start()

    # ---- 托盘 ----
    def _build_tray(self):
        from PyQt5.QtWidgets import QAction, QMenu, QSystemTrayIcon

        icon = QIcon(str(_ASSETS_DIR / "deploy.png"))
        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip(tr("XCcv 部署平台（推理服务继续运行）"))
        menu = QMenu()
        act_show = QAction("显示部署平台", menu)
        act_show.triggered.connect(self._restore_from_tray)
        self._act_island = QAction("隐藏灵动岛", menu)
        self._act_island.triggered.connect(self._toggle_island)
        act_stop = QAction("停止推理服务", menu)
        act_stop.triggered.connect(self._page._stop_service)
        act_quit = QAction("退出部署平台（推理服务同步停止）", menu)
        act_quit.triggered.connect(self._real_quit)
        for a in (act_show, self._act_island):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(act_stop)
        menu.addAction(act_quit)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()
        # 灵动岛开关（页面 checkbox）与托盘菜单文案同步
        self._page.island_check.toggled.connect(self._sync_island_action)

    def _on_tray_activated(self, reason):
        # 托盘单击/双击恢复窗口（QSystemTrayIcon 此前未在
        # 本方法作用域导入 → 每次点击托盘图标 NameError）
        from PyQt5.QtWidgets import QSystemTrayIcon

        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self._restore_from_tray()

    def _restore_from_tray(self):
        self.setWindowState((self.windowState() & ~Qt.WindowMinimized)
                            | Qt.WindowActive)
        self.show()
        self.raise_()
        self.activateWindow()

    def _toggle_island(self):
        self._page.island_check.setChecked(
            not self._page.island_check.isChecked())

    def _sync_island_action(self, checked: bool):
        self._act_island.setText(
            tr("隐藏灵动岛") if checked else tr("显示灵动岛"))

    def _real_quit(self):
        """真正退出：先同步停止推理服务（批注四：部署平台结束=服务同步结束）。"""
        try:
            self._page.stop_service_blocking()
        except Exception:
            log.warning("部署服务停止请求失败（强退继续）", exc_info=True)
        self._force_quit = True
        self.close()

    def closeEvent(self, event):
        from PyQt5.QtWidgets import QApplication, QSystemTrayIcon

        tray_ok = self._tray is not None and QSystemTrayIcon.isSystemTrayAvailable()
        if self._force_quit or not tray_ok:
            # 批注四：真正结束部署平台前同步停服；_real_quit 已停过则不重复
            if not self._force_quit:   # 无托盘环境的直接关闭：此处补停服
                try:
                    self._page.stop_service_blocking()
                except Exception:
                    log.warning("部署服务停止请求失败（应用退出继续）", exc_info=True)
            event.accept()
            QApplication.instance().quit()
            return
        # 默认行为：关闭 = 缩到托盘（推理服务与灵动岛继续运行）
        event.ignore()
        self.hide()
        self._tray.showMessage(
            "XCcv 部署平台",
            "已最小化到系统托盘，推理服务继续运行。\n双击托盘图标恢复窗口。",
            QSystemTrayIcon.Information, 3000)


class DeployPage(QWidget):
    """主界面「部署平台」页（v0.5.4 恢复内嵌，卡片化）。

    三张平台卡片（与 AI 平台一致）：推理服务 / 模型加载 / C# 测试；
    每张卡片以**独立进程**打开对应窗口（launcher + 单实例守卫）。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        from .ui_style import apply_app_theme

        apply_app_theme(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(44, 26, 44, 18)
        lay.setSpacing(16)

        title = QLabel("部署平台")
        title.setObjectName("pageTitle")
        sub = QLabel("推理服务 / 模型加载 / C# 测试，均以独立进程打开")
        sub.setObjectName("pageSub")
        lay.addWidget(title)
        lay.addWidget(sub)

        grid = QGridLayout()
        grid.setSpacing(18)
        grid.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        from .i18n import tr as _tr

        cards = [
            ("deploy", _tr("推理服务"), _tr("启动/停止推理服务，GPU 与接口一览；"
             "桌面灵动岛常驻屏幕顶部"), "deploy.png",
             ("#5ec6dd", "#1f8fb5"), None),
            ("model_load", _tr("模型加载"), _tr("把 .pt/.engine 模型或 .xvap/.json "
             "流程包加载为常驻 API，多 API 并行"), "model_load.png",
             ("#b795ff", "#7243e2"), "models"),
            ("csharp_test", _tr("C# 测试"), _tr("运行 C# Demo（WinForms）：加载模型"
             "→推理→直接查看接口返回的 JSON 原文"), "csharp_demo.png",
             ("#5edcae", "#17a768"), "csharp"),
        ]
        from .ui_widgets import PlatformCard

        self._cards = []
        for i, (key, name, desc, icon, grad, arg) in enumerate(cards):
            card = PlatformCard(key, name, desc, icon, grad)
            card.clicked.connect(lambda k=key, a=arg: self._open_module(k, a))
            grid.addWidget(card, 0, i)
            self._cards.append(card)
        lay.addLayout(grid, stretch=1)

        hint = QLabel("推理服务窗口关闭只最小化到托盘；退出部署平台时同步停止。")
        hint.setObjectName("muted")
        lay.addWidget(hint)

    def set_search_query(self, text: str) -> None:
        """主界面全局搜索：按名称/描述过滤三张模块卡。"""
        q = (text or "").strip().lower()
        for card in getattr(self, "_cards", []):
            card.setVisible(not q or q in card.card_title.lower()
                            or q in card.card_desc.lower())

    def _open_module(self, key: str, arg: str | None) -> None:
        if key == "csharp_test":
            # v0.5.5 依验收批注：C# 卡恢复此前效果——服务运行中直接进入
            # 测试界面（编译/启动 Demo），未运行则弹窗提示，不打开窗口
            self._launch_csharp_flow()
            return
        try:
            launcher.launch_platform(key, arg or None)
        except Exception as e:
            log.warning("模块窗口启动失败（已弹窗提示）: %s", e, exc_info=True)
            QMessageBox.critical(self, tr("无法打开模块"),
                                 f"「{key}」{tr('启动失败')}："
                                 f"{type(e).__name__}: {e}")

    def _launch_csharp_flow(self) -> None:
        import socket

        from .service.config import load_config

        cfg = load_config()
        with socket.socket() as sk:
            sk.settimeout(0.6)
            running = sk.connect_ex(("127.0.0.1", int(cfg.port))) == 0
        if not running:
            QMessageBox.warning(
                self, tr("C# 测试"),
                tr("推理服务未运行。\n请先在「部署平台 → 推理服务」启动服务后重试。"))
            return
        if not _demo_exe_stale():
            self._run_demo_standalone()
            return
        from .ui_widgets import toast

        toast(self, tr("正在编译 C# Demo…（首次约数秒）"), "info")
        self._csharp_build_thread = _DemoBuildThread()
        self._csharp_build_thread.finished_build.connect(self._after_build)
        auto_cleanup_thread(self, "_csharp_build_thread",
                            self._csharp_build_thread)  # A-3
        self._csharp_build_thread.start()

    def _after_build(self, ok: bool, msg: str):
        self._csharp_build_thread = None
        if not ok:
            QMessageBox.critical(self, tr("C# 测试程序编译失败"), msg[:2000])
            return
        self._run_demo_standalone()

    def _run_demo_standalone(self):
        from .service.config import load_config

        cfg = load_config()
        subprocess.Popen([str(_DEMO_EXE), "--url",
                          f"http://127.0.0.1:{cfg.port}",
                          "--key", cfg.api_key, *_demo_lang_args()],
                         cwd=str(_DEMO_DIR), close_fds=True)


class CSharpWindow(ChromeMainWindow, _HttpMixin):
    """「C# 测试」大平台窗口（独立进程：pythonw -m xccv.deploy_platform
    csharp）：确保推理服务在跑后编译/启动 C# Demo。"""

    def __init__(self):
        super().__init__(icon_file="csharp_demo.png")
        self._build_thread: _DemoBuildThread | None = None
        self.setWindowTitle(tr("XCcv C# 测试"))
        self.resize(760, 520)
        from .ui_style import apply_app_theme

        apply_app_theme(self)
        self._init_http()

        from .service_health import ServiceHealthPoller

        self._poller = ServiceHealthPoller(self)
        self._poller.state_changed.connect(self._on_state)
        self._poller.start()

        central = QWidget()
        self.set_chrome_central(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 22, 28, 18)
        root.setSpacing(14)

        title = QLabel("C# 测试程序")
        title.setObjectName("pageTitle")
        sub = QLabel("编译并运行 C# Demo：加载模型 → 推理 → 查看 JSON 返回")
        sub.setObjectName("pageSub")
        root.addWidget(title)
        root.addWidget(sub)

        box = QGroupBox("运行")
        lay = QVBoxLayout(box)
        lay.setSpacing(10)
        self.status_label = QLabel("○ 正在检测推理服务…")
        self.status_label.setStyleSheet("border:none;")
        lay.addWidget(self.status_label)
        self.btn_demo = QPushButton("启动 C# Demo")
        self.btn_demo.setObjectName("btnAccent")
        self.btn_demo.setEnabled(False)
        self.btn_demo.clicked.connect(self._launch_demo)
        lay.addWidget(self.btn_demo)
        tip = QLabel("前提：推理服务已启动。产物输出 platform_data/csharp_demo。")
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        root.addWidget(box)
        root.addStretch(1)


    # ---- 服务状态 / 动作 ----
    def base_url(self) -> str:
        from .service.config import load_config

        cfg = load_config()
        host = cfg.host if cfg.host not in ("0.0.0.0", "::") else "127.0.0.1"
        return f"http://{host}:{cfg.port}"

    def api_key(self) -> str:
        from .service.config import load_config

        return load_config().api_key

    def _on_state(self, state: str, info: dict):
        running = state == "running"
        if running:
            self.status_label.setText(f"● 推理服务运行中：{info.get('url', '')}")
            self.status_label.setStyleSheet("color:#166534; border:none;")
        elif state == "starting":
            self.status_label.setText("● 推理服务启动中…")
            self.status_label.setStyleSheet("color:#b45309; border:none;")
        else:
            self.status_label.setText(
                "○ 推理服务未运行——请先在「部署平台 → 推理服务」启动")
            self.status_label.setStyleSheet("color:#b45309; border:none;")
        self.btn_demo.setEnabled(running)

    def _launch_demo(self):
        if not _demo_exe_stale():
            self._run_demo()
            return
        from .ui_widgets import toast

        self.btn_demo.setEnabled(False)
        self._build_thread = _DemoBuildThread()
        self._build_thread.finished_build.connect(self._after_build)
        self._build_thread.start()
        toast(self, tr("正在编译 C# Demo…"), "info")

    def _after_build(self, ok: bool, msg: str):
        self.btn_demo.setEnabled(self._poller.state() == "running")
        self._build_thread = None
        if not ok:
            QMessageBox.critical(self, tr("C# 测试程序编译失败"), msg[:2000])
            return
        self._run_demo()

    def _run_demo(self):
        subprocess.Popen([str(_DEMO_EXE), "--url", self.base_url(),
                          "--key", self.api_key(), *_demo_lang_args()],
                         cwd=str(_DEMO_DIR), close_fds=True)


_MODES = {
    # argv[1] -> (单实例 key, 窗口标题, 窗口工厂)
    "serve": ("deploy", "XCcv 推理服务", ServiceWindow),
    "models": ("model_load", "XCcv 模型加载", ModelServiceWindow),
    "csharp": ("csharp_test", "XCcv C# 测试", CSharpWindow),
}


def main():
    """独立入口（不依赖主界面）：
    pythonw -m xccv.deploy_platform [serve|models|csharp]"""
    from .platform_entry import run_platform

    mode = "serve"
    for a in sys.argv[1:]:
        if a in _MODES:
            mode = a
            break
    key, title, window_cls = _MODES[mode]

    def build():
        from .gui_utils import apply_app_font

        apply_app_font(QApplication.instance())
        return window_cls()

    run_platform(key, title, build, heavy=False)


if __name__ == "__main__":
    main()


# ---- Cython 编译态槽包装（见 xccv/qt_slots.py；新增 UI 模块必须加） ----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())


