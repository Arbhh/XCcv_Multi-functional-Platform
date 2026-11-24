# -*- coding: utf-8 -*-
"""删除指定标签：删形状 / 整图过滤两种模式；预览 + 确认后执行。

备份目录作为选项暴露（默认关闭）；shapes 模式删空后按平台
语义删除空 json（与标注平台"空标注不落盘"一致）。
"""

import logging

from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QMessageBox,
    QPushButton, QRadioButton, QVBoxLayout,
)

from ...engine.dataops.labelops import apply_delete, plan_delete, scan_pairs
from ...i18n import tr
from .base import DataToolWindow
from .common import DirRow, PreviewTable, confirm_execute, rel_display
from .registry import ToolSpec, register

log = logging.getLogger(__name__)


class LabelDeleteWindow(DataToolWindow):
    """标签删除工具窗口。"""

    tool_key = "label_delete"

    def __init__(self, on_closed=None):
        self._plan = []
        super().__init__(on_closed=on_closed)

    @property
    def tool_title(self):
        return tr("删除指定标签")

    def _build_settings(self, lay: QVBoxLayout):
        opts = QGroupBox(tr("删除规则"))
        form = QFormLayout(opts)
        self.src_row = DirRow(tr("数据集目录"))
        form.addRow(self.src_row)
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText(tr("标签列表，逗号/分号分隔，如：划痕,脏污"))
        form.addRow(tr("目标标签"), self.labels_edit)

        mode_lay = QFormLayout()
        self.mode_shapes = QRadioButton(tr("只删匹配形状（保留图片）"))
        self.mode_images = QRadioButton(tr("整图过滤（删除图片与标注）"))
        self.mode_shapes.setChecked(True)
        mode_lay.addRow(self.mode_shapes)
        mode_lay.addRow(self.mode_images)
        form.addRow(tr("模式"), mode_lay)

        self.backup_box = QCheckBox(tr("删除前备份到指定目录（默认关闭）"))
        form.addRow(self.backup_box)
        self.backup_row = DirRow(tr("备份目录"))
        self.backup_row.setVisible(False)
        form.addRow(self.backup_row)
        self.backup_box.toggled.connect(self.backup_row.setVisible)
        lay.addWidget(opts)

        self.preview_btn = QPushButton(tr("扫描预览"))
        self.preview_btn.clicked.connect(self._scan)
        lay.addWidget(self.preview_btn)

        self.hint = QLabel(tr("先扫描预览，确认后执行；自动包含子目录"))
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        self.table = PreviewTable([tr("文件"), tr("模式"), tr("命中"), tr("标签")])
        lay.addWidget(self.table, stretch=1)

        self.exec_btn = QPushButton(tr("执行删除"))
        self.exec_btn.setEnabled(False)
        self.exec_btn.clicked.connect(self._execute)
        lay.addWidget(self.exec_btn)

    def _mode(self) -> str:
        return "images" if self.mode_images.isChecked() else "shapes"

    def _scan(self):
        src = self.src_row.path()
        labels = [s.strip() for s in
                  self.labels_edit.text().replace("；", ",").replace(";", ",").split(",")]
        labels = [s for s in labels if s]
        if src is None or not src.exists():
            QMessageBox.information(self, tr("请检查"), tr("请先选择存在的数据集目录"))
            return
        if not labels:
            QMessageBox.information(self, tr("请检查"), tr("请填写目标标签"))
            return
        if self._busy():
            return
        mode = self._mode()

        def _work():
            pairs = scan_pairs(src)
            plan = plan_delete(pairs, labels=labels, mode=mode)
            rows = []
            for e in plan:
                hit = (f"{len(e.matches)} {tr('处形状')}" if mode == "shapes"
                       else tr("整图"))
                rows.append((rel_display(e.json_path, src), e.mode, hit,
                             ",".join(e.labels_seen)[:60]))
            return {"plan": plan, "rows": rows}

        self._set_running(True)
        self._start_worker(_work, self._on_scan_done)

    def _on_scan_done(self, result: dict):
        self._set_running(False)
        self._plan = result["plan"]
        self.table.set_rows(result["rows"])
        mode = self._mode()
        if mode == "shapes":
            total = sum(len(e.matches) for e in self._plan)
            self.hint.setText(tr("命中 {n} 个文件的 {s} 处形状；确认后执行").format(
                n=len(self._plan), s=total))
        else:
            self.hint.setText(tr("命中 {n} 个文件（整图删除图片与标注）；确认后执行").format(
                n=len(self._plan)))
        self.exec_btn.setEnabled(bool(self._plan))

    def _execute(self):
        if not self._plan:
            return
        mode = self._mode()
        backup_dir = self.backup_row.path() if self.backup_box.isChecked() else None
        if mode == "shapes":
            text = tr("将从 {n} 个文件删除 {s} 处形状。继续？").format(
                n=len(self._plan), s=sum(len(e.matches) for e in self._plan))
        else:
            text = tr("将整图删除 {n} 个文件（图片+标注）。继续？").format(n=len(self._plan))
        if not confirm_execute(self, text):
            return
        plan = list(self._plan)

        def _work():
            return apply_delete(plan, backup_dir=backup_dir)

        self._set_running(True)
        self._start_worker(_work, self._on_exec_done)

    def _on_exec_done(self, result: dict):
        self._set_running(False)
        self._plan = []
        self.exec_btn.setEnabled(False)
        self.hint.setText(tr("删除完成：{s} 处形状 / {j} 个 json / {i} 张图片").format(
            s=result["shapes"], j=result["jsons"], i=result["images"]))
        self.table.set_rows([])

    def _set_running(self, running: bool):
        self.preview_btn.setEnabled(not running)
        if not running:
            self.exec_btn.setEnabled(bool(self._plan))


register(ToolSpec(
    key="label_delete",
    title="删除指定标签",
    desc="按标签删除形状或整图过滤，执行前可备份",
    color="#ef4444", icon="toolbox_tool.png",
    keywords="delete label 删除 标签 过滤 整图",
    tooltip="只删匹配形状保留图片，或整图过滤删除图片与标注。删空的 json 按平台语义移除；备份目录默认关闭。",
    factory=LabelDeleteWindow,
))
