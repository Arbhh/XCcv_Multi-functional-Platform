"""平台单实例守卫。

每个平台是独立进程，重复启动（主界面再点一次 / 用户直接双击 exe）时
应激活已有窗口而不是开第二个进程。实现：QLocalServer 具名管道——
先探测服务是否存在，存在则发激活消息并让新进程退出；
否则本进程注册服务，收到激活消息时激活自己的窗口。

激活消息可携带业务参数（如「最近项目」卡片透传的项目名）。
旧实例收到后先激活窗口，再调用 window.on_platform_message(args) 钩子
（窗口按需实现，缺省只激活）——修复"训练平台已开时点项目卡片只唤起
平台不打开项目"（argv 在单实例拦截处丢失）。
"""

import logging
log = logging.getLogger(__name__)
import json

from PyQt5.QtCore import Qt
from PyQt5.QtNetwork import QLocalServer, QLocalSocket


def already_running(key: str, payload: list | None = None) -> bool:
    """是否已有同平台实例在运行（轻量探测，重型导入前调用）。

    payload：可选业务参数（如项目名列表）；连接成功后随激活消息发给
    已有实例。写完必须主动泵事件：服务端（同进程 register 的 server）
    的 newConnection 分发需要事件循环，而本函数返回后局部 socket 即被
    GC 断开——不泵则数据随管道丢弃（探针实证，激活消息从未送达）。
    """
    sock = QLocalSocket()
    sock.connectToServer(f"xccv-{key}")
    connected = sock.waitForConnected(300)
    if connected:
        if payload:
            msg = json.dumps({"args": [str(a) for a in payload]},
                             ensure_ascii=False).encode("utf-8")
        else:
            msg = b"activate"   # 旧协议（纯激活），接收侧兼容
        sock.write(msg)
        sock.waitForBytesWritten(300)
        try:
            from PyQt5.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                for _ in range(12):
                    app.processEvents()
                    if sock.state() != QLocalSocket.ConnectedState:
                        break   # 服务端已读完并断开：消息送达
        except Exception:  # noqa: BLE001  泵送失败不阻塞启动
            log.debug("单实例激活消息泵送失败")
    return connected


def register(key: str, window) -> None:
    """注册单实例服务：后续重复启动会激活 window（并可携带业务参数）。"""
    name = f"xccv-{key}"
    QLocalServer.removeServer(name)  # 清理异常退出残留（Windows 上为无害空操作）
    server = QLocalServer()
    if not server.listen(name):
        return  # 注册失败不阻断正常使用

    def _activate():
        conn = server.nextPendingConnection()
        if conn is None:
            return
        # 客户端可能不回执即关闭：循环 readAll 等数据到达（最多 ~0.5s）；
        # 旧协议（b"activate"）解析失败按纯激活处理
        raw = b""
        for _ in range(10):
            raw += bytes(conn.readAll())
            if raw:
                break
            conn.waitForReadyRead(50)
        args = None
        try:
            parsed = json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("args"), list):
                args = [str(a) for a in parsed["args"] if str(a)]
        except Exception:  # noqa: BLE001  旧协议/坏包：仅激活
            args = None
        conn.disconnectFromServer()
        window.setWindowState((window.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        window.show()
        window.raise_()
        window.activateWindow()
        if args:
            hook = getattr(window, "on_platform_message", None)
            if hook is not None:
                try:
                    hook(args)
                except Exception:  # noqa: BLE001  业务侧失败不影响激活
                    log.warning("单实例激活回调失败", exc_info=True)

    server.newConnection.connect(_activate)
    window._xccv_single_instance_server = server  # 保活


# ---- Cython 编译态槽包装（信号槽参数裁剪修复，见 xccv/qt_slots.py）----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())

