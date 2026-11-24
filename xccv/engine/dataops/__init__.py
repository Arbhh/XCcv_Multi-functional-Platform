# -*- coding: utf-8 -*-
"""数据集纯逻辑工具层（无 Qt 依赖，可被工具窗口与服务端复用）。

子模块：
- labelops：标注替换 / 删除 / 数据对迁移（labelme JSON 语义）；
- tiling_export：滑窗物理裁图导出（默认口径对齐 engine.dataset.tiling 训练切图）；
- video_frames：视频抽帧导出（OpenCV）。
"""

from .labelops import (
    SamplePair,
    apply_delete,
    apply_move,
    apply_replace,
    iter_labels,
    plan_delete,
    plan_move,
    plan_replace,
    scan_pairs,
)
from .tiling_export import TileOutput, export_tiles, plan_tiles
from .video_frames import FramePlan, FrameResult, export_frames, plan_estimate

__all__ = [
    "SamplePair",
    "iter_labels",
    "scan_pairs",
    "plan_replace",
    "apply_replace",
    "plan_delete",
    "apply_delete",
    "plan_move",
    "apply_move",
    "TileOutput",
    "plan_tiles",
    "export_tiles",
    "FramePlan",
    "FrameResult",
    "plan_estimate",
    "export_frames",
]
