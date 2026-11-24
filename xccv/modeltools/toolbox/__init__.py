# -*- coding: utf-8 -*-
"""百宝箱子包：数据工具集工作台（注册表驱动的卡片墙 + 子功能独立窗口）。

导入本包即完成子功能注册（各 feature 模块自带 register 调用）。
"""

from .registry import ToolSpec, all_tools, get, register, search
from .toolbox_window import ToolboxWindow

# 子功能登记（导入即上架；新增工具在同层加模块并在此导入一行）
from . import label_replace_window  # noqa: F401  替换标签
from . import label_delete_window  # noqa: F401  删除标签
from . import dataset_move_window  # noqa: F401  迁移标注数据
from . import tiling_export_window  # noqa: F401  滑窗裁图
from . import video_frames_window  # noqa: F401  视频切分图片

__all__ = [
    "ToolSpec", "ToolboxWindow", "all_tools", "get", "register", "search",
]
