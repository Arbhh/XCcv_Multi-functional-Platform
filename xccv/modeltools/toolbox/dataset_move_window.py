# -*- coding: utf-8 -*-
"""移动标注文件：按标签/关键字筛选，图片+JSON 成对迁移。

支持移动未标注图片（无 json 的纯图片同样按规则纳入）；目标
目录校验 + 同名冲突跳过；执行后给移动报告。
"""

import logging

from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout,
)

from ...engine.dataops.labelops import apply_move, plan_move, scan_pairs
from ...i18n import tr
from .base import DataToolWindow
from .common import DirRow, PreviewTable, confirm_execute, rel_display
from .registry import ToolSpec, register

log = logging.getLogger(__name__)


class DatasetMoveWindow(DataToolWindow):
    """数据迁移工具窗口。"""

    tool_key = "dataset_move"

    def __init__(self, on_closed=None):
        self._plan = []
        super().__init__(on_closed=on_closed)

    @property
    def tool_title(self):
        return tr("移动标注数据")

    def _build_settings(self, lay: QVBoxLayout):
        opts = QGroupBox(tr("筛选与目标"))
        form = QFormLayout(opts)
        self.src_row = DirRow(tr("源目录"))
        form.addRow(self.src_row)
        self.dst_row = DirRow(tr("目标目录"))
        form.addRow(self.dst_row)
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText(tr("可选，逗号分隔，如：划痕,脏污"))
        form.addRow(tr("标签筛选"), self.labels_edit)
        self.kw_edit = QLineEdit()
        self.kw_edit.setPlaceholderText(tr("可选，标签含关键字即命中"))
        form.addRow(tr("关键字"), self.kw_edit)
        self.unlabeled_box = QCheckBox(tr("同时移动未标注图片（无 json 纯图片）"))
        self.unlabeled_box.setChecked(True)
        form.addRow(self.unlabeled_box)
        lay.addWidget(opts)

        self.preview_btn = QPushButton(tr("扫描预览"))
        self.preview_btn.clicked.connect(self._scan)
        lay.addWidget(self.preview_btn)

        self.hint = QLabel(tr("先扫描预览，确认后执行；同名文件跳过；自动包含子目录"))
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        self.table = PreviewTable([tr("图片"), "JSON", tr("类型"), tr("标签")])
        lay.addWidget(self.table, stretch=1)

        self.exec_btn = QPushButton(tr("执行移动"))
        self.exec_btn.setEnabled(False)
        self.exec_btn.clicked.connect(self._execute)
        lay.addWidget(self.exec_btn)

    def _scan(self):
        src = self.src_row.path()
        dst = self.dst_row.path()
        if src is None or not src.exists():
            QMessageBox.information(self, tr("请检查"), tr("请先选择存在的源目录"))
            return
        if dst is None:
            QMessageBox.information(self, tr("请检查"), tr("请填写目标目录"))
            return
        if src.resolve() == dst.resolve():
            QMessageBox.information(self, tr("请检查"), tr("目标目录不能与源目录相同"))
            return
        labels = [s.strip() for s in
                  self.labels_edit.text().replace("；", ",").replace(";", ",").split(",")]
        labels = [s for s in labels if s]
        if not labels and not self.kw_edit.text().strip() and not self.unlabeled_box.isChecked():
            QMessageBox.information(self, tr("请检查"),
                                    tr("未指定标签/关键字时仅按\"移动未标注图片\"纳入"))
        if self._busy():
            return

        def _work():
            pairs = scan_pairs(src)
            plan = plan_move(pairs, labels=labels,
                             keyword=self.kw_edit.text(),
                             include_unlabeled=self.unlabeled_box.isChecked())
            rows = []
            for e in plan:
                kind = tr("未标注") if e.unlabeled else tr("已标注")
                rows.append((rel_display(e.image_path, src) if e.image_path else "",
                             rel_display(e.json_path, src) if e.json_path else "",
                             kind, ",".join(e.labels)[:48]))
            return {"plan": plan, "rows": rows}

        self._set_running(True)
        self._start_worker(_work, self._on_scan_done)

    def _on_scan_done(self, result: dict):
        self._set_running(False)
        self._plan = result["plan"]
        self.table.set_rows(result["rows"])
        n_unl = sum(1 for e in self._plan if e.unlabeled)
        self.hint.setText(tr("命中 {n} 对（其中未标注 {u} 对）；确认后执行").format(
            n=len(self._plan), u=n_unl))
        self.exec_btn.setEnabled(bool(self._plan))

    def _execute(self):
        if not self._plan:
            return
        dst = self.dst_row.path()
        if not confirm_execute(self, tr("将移动 {n} 对文件到目标目录。继续？").format(n=len(self._plan))):
            return
        plan = list(self._plan)

        def _work():
            return apply_move(plan, dst)

        self._set_running(True)
        self._start_worker(_work, self._on_exec_done)

    def _on_exec_done(self, result: dict):
        self._set_running(False)
        self._plan = []
        self.exec_btn.setEnabled(False)
        self.hint.setText(tr("移动完成：{p} 对 / {f} 个文件；同名跳过 {s} 对").format(
            p=result["pairs"], f=result["files"], s=result["skipped"]))
        self.table.set_rows([])

    def _set_running(self, running: bool):
        self.preview_btn.setEnabled(not running)
        if not running:
            self.exec_btn.setEnabled(bool(self._plan))


register(ToolSpec(
    key="dataset_move",
    title="移动标注数据",
    desc="按标签或关键字成对迁移图片与标注",
    color="#f59e0b", icon="toolbox_tool.png",
    keywords="move dataset 迁移 移动 标注 未标注 数据",
    tooltip="图片与同名 JSON 成对移动，可同时移动未标注图片。目标目录同名文件跳过并计数。",
    factory=DatasetMoveWindow,
))
