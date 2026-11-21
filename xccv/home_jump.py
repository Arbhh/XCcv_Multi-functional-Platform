"""主界面跳转接收（V0.7.5，独立文件——纪律：新逻辑不堆进巨型 UI 文件）。

职责：其他平台进程的「前往流程图界面」跳转落点（新版流程图界面 =
主界面左侧导航「流程图」页 FlowPage + HTML 画布窗）：

- 启动参数：argv[1] 携带流程文件路径（.json/.xvap）→ 主界面启动后
  直接定位流程图页并载入（独立编辑器旧版弃用后的替代链路）；
- 已在运行：单实例激活转发 args → HomeWindow.on_platform_message →
  定位 + 载入（激活本身由 single_instance.register 完成）。

app.py 只做两处薄接线：FlowPage 引用保存 + wire(window) 调用。
"""

import sys
from pathlib import Path

FLOW_SUFFIXES = (".json", ".xvap")
_FLOW_NAV_ROW = 2   # app._NAV_ITEMS / stack 第 2 页 = 流程图


def is_flow_arg(a: str) -> bool:
    p = Path(str(a))
    return p.suffix.lower() in FLOW_SUFFIXES and p.is_file()


def open_flow_in_home(window, path: str) -> bool:
    """定位主界面流程图页并载入流程（返回是否成功）。"""
    page = None
    if hasattr(window, "stack"):
        page = window.stack.widget(_FLOW_NAV_ROW)
    if page is None:
        from .flow_platform import FlowPage

        page = window.findChild(FlowPage)   # 非标准宿主兜底（测试/变体布局）
    if page is None or not hasattr(page, "open_flow_file"):
        return False
    if getattr(window, "nav_list", None) is not None:
        window.nav_list.setCurrentRow(_FLOW_NAV_ROW)   # 切到流程图页
    ok = page.open_flow_file(path, open_canvas=True)
    if ok:
        window.raise_()
        window.activateWindow()
    return ok


def handle_args(window, args) -> str | None:
    """处理激活/启动携带的业务参数：流程文件 → 流程图页载入。

    返回被消费的路径（无匹配返回 None；其余参数留待后续扩展）。"""
    for a in args or []:
        if is_flow_arg(str(a)):
            if open_flow_in_home(window, str(a)):
                return str(a)
    return None


def forward_args(argv=None) -> list:
    """提取启动参数里的业务参数（单实例转发 payload）。"""
    argv = sys.argv[1:] if argv is None else list(argv)
    return [a for a in argv if not str(a).startswith("-")]


def wire(window, argv=None) -> None:
    """app.main() 薄接线：装 on_platform_message 钩子 + 处理启动参数。

    主界面单实例：本函数不负责 already_running（重型导入前在 main()
    早期调用）；register 在窗口就绪后由本函数完成。"""
    from .single_instance import register

    def _on_platform_message(args):
        handle_args(window, args)

    window.on_platform_message = _on_platform_message
    register("main", window)
    pending = forward_args(argv)
    if pending:
        handle_args(window, pending)

