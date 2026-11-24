# -*- coding: utf-8 -*-
"""滑窗裁图导出：数据目录 → tile 图片 + 同步切分标注（窗口坐标）。

默认参数对齐切图训练现行口径（tile=640 / overlap=0.2 / bg_tile_keep=0.15 /
min_visibility=0.3）；归属与去重语义复用 engine.dataset.tiling。大图逐张
读取裁剪（内存安全），后台执行带进度。
"""

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout,
)

from ...engine.dataops.tiling_export import (
    BG_KEEP_DEFAULT, MIN_VIS_DEFAULT, OVERLAP_DEFAULT, TILE_DEFAULT,
    export_tiles, plan_tiles,
)
from ...engine.dataops.labelops import scan_pairs
from ...i18n import tr
from .base import DataToolWindow
from .common import (
    DirRow, NoWheelDoubleSpin, NoWheelSpin, PreviewTable, confirm_execute,
    rel_display,
)
from .registry import ToolSpec, register

log = logging.getLogger(__name__)


class TilingExportWindow(DataToolWindow):
    """滑窗裁图工具窗口。"""

    tool_key = "tiling_export"

    def __init__(self, on_closed=None):
        self._plan = []
        super().__init__(on_closed=on_closed)

    @property
    def tool_title(self):
        return tr("滑窗裁图")

    def _build_settings(self, lay: QVBoxLayout):
        opts = QGroupBox(tr("裁图参数（默认对齐切图训练）"))
        form = QFormLayout(opts)
        self.src_row = DirRow(tr("数据目录"))
        form.addRow(self.src_row)
        self.dst_row = DirRow(tr("输出目录"))
        form.addRow(self.dst_row)

        self.tile_spin = NoWheelSpin()
        self.tile_spin.setRange(128, 4096)
        self.tile_spin.setValue(TILE_DEFAULT)
        self.tile_spin
        form.addRow(tr("tile 边长（像素）"), self.tile_spin)

        self.overlap_spin = NoWheelDoubleSpin()
        self.overlap_spin.setRange(0.0, 0.9)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setValue(OVERLAP_DEFAULT)
        self.overlap_spin
        form.addRow(tr("相邻重叠率"), self.overlap_spin)

        self.bg_spin = NoWheelDoubleSpin()
        self.bg_spin.setRange(0.0, 1.0)
        self.bg_spin.setSingleStep(0.05)
        self.bg_spin.setValue(BG_KEEP_DEFAULT)
        self.bg_spin
        form.addRow(tr("背景 tile 保留率"), self.bg_spin)

        self.vis_spin = NoWheelDoubleSpin()
        self.vis_spin.setRange(0.0, 1.0)
        self.vis_spin.setSingleStep(0.05)
        self.vis_spin.setValue(MIN_VIS_DEFAULT)
        self.vis_spin
        form.addRow(tr("目标最小可见率"), self.vis_spin)

        self.full_box = QCheckBox(tr("同时导出整图副本"))
        form.addRow(self.full_box)
        lay.addWidget(opts)

        self.preview_btn = QPushButton(tr("生成计划预览"))
        self.preview_btn.clicked.connect(self._plan_start)
        lay.addWidget(self.preview_btn)

        self.hint = QLabel(tr("预览确认后开始导出；自动包含子目录"))
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        self.table = PreviewTable([tr("源图"), "tile", tr("目标数"), tr("背景保留")])
        lay.addWidget(self.table, stretch=1)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.bar.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.bar)

        self.exec_btn = QPushButton(tr("开始导出"))
        self.exec_btn.setEnabled(False)
        self.exec_btn.clicked.connect(self._export)
        lay.addWidget(self.exec_btn)

    def _plan_start(self):
        src = self.src_row.path()
        if src is None or not src.exists():
            QMessageBox.information(self, tr("请检查"), tr("请先选择存在的数据目录"))
            return
        if self._busy():
            return

        def _work():
            pairs = scan_pairs(src)
            plan = plan_tiles(
                pairs, tile=self.tile_spin.value(),
                overlap=self.overlap_spin.value(),
                bg_tile_keep=self.bg_spin.value(),
                min_visibility=self.vis_spin.value(),
                export_full=self.full_box.isChecked(),
                root=src)
            rows = []
            for item in plan[:300]:
                rows.append((rel_display(item.source, src),
                             f"x{item.x} y{item.y} {item.w}×{item.h}",
                             len(item.shapes), tr("是") if item.bg_keep else ""))
            return {"plan": plan, "rows": rows, "with_obj":
                    sum(1 for i in plan if i.has_objects)}

        self._set_running(True)
        self._start_worker(_work, self._on_plan_done)

    def _on_plan_done(self, result: dict):
        self._set_running(False)
        self._plan = result["plan"]
        self.table.set_rows(result["rows"])
        self.hint.setText(tr("计划 {n} 个 tile（含目标 {o} 个）；确认后开始导出").format(
            n=len(self._plan), o=result["with_obj"]))
        self.exec_btn.setEnabled(bool(self._plan))

    def _export(self):
        dst = self.dst_row.path()
        if dst is None:
            QMessageBox.information(self, tr("请检查"), tr("请填写输出目录"))
            return
        if not confirm_execute(self, tr("将导出 {n} 个 tile 到输出目录。继续？").format(n=len(self._plan))):
            return
        plan = list(self._plan)
        self.bar.setVisible(True)
        self.bar.setRange(0, len(plan))

        def _progress(done, total, name):
            self.bar.setValue(done)

        def _work():
            return export_tiles(plan, dst, progress=None)

        self._set_running(True)
        self._start_worker(_work, self._on_export_done)

    def _on_export_done(self, result: dict):
        self._set_running(False)
        self.bar.setVisible(False)
        self._plan = []
        self.exec_btn.setEnabled(False)
        self.hint.setText(tr("导出完成：{i} 张 tile 图 / {j} 个标注 JSON").format(
            i=result["images"], j=result["jsons"]))
        self.table.set_rows([])

    def _set_running(self, running: bool):
        self.preview_btn.setEnabled(not running)
        if not running:
            self.exec_btn.setEnabled(bool(self._plan))


register(ToolSpec(
    key="tiling_export",
    title="滑窗裁图",
    desc="大图滑窗切块，标注同步切分导出",
    color="#8b5cf6", icon="toolbox_tool.png",
    keywords="tile tiling 裁图 切图 滑窗 大图 导出",
    tooltip="默认参数对齐切图训练：tile 640、重叠 0.2、背景保留 0.15、最小可见率 0.3。可选择同时导出整图副本。",
    factory=TilingExportWindow,
))
