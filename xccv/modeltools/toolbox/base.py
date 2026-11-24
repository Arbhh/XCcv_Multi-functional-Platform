# -*- coding: utf-8 -*-
"""百宝箱数据工具窗口基类：目录型工具（无项目/训练记录依赖）。

复用模型工具 _ToolWindow 的忙碌态/后台任务/关闭确认骨架（file_mode 跳过
项目选择器），窗口标题归属"XCcv 百宝箱"。
"""

from ..modeltools_window import _ToolWindow
from ...i18n import tr


class DataToolWindow(_ToolWindow):
    """目录型数据工具窗口基类（一个子功能一个窗口）。"""

    tool_key = ""
    tool_title = ""
    tool_icon = "toolbox_tool.png"   # 百宝箱子功能共用图标
    file_mode = True  # _ToolWindow：跳过项目/记录选择，直接进工具设置区

    def __init__(self, on_closed=None):
        super().__init__(on_closed=on_closed)
        self.setWindowTitle(f"{self.tool_title}  -  {tr('XCcv 百宝箱')}")
