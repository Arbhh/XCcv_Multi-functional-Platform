# -*- coding: utf-8 -*-
"""一键替换标签：选目录 + 旧标签→新标签（支持正则整词匹配）。

流程 = 扫描预览（后台）→ 确认 → 执行（后台）；改 JSON 前不做备份（预览
确认即终局，与标注平台改写语义一致）。同步替换 shapes[].label 与图片级/
形状级 flags 键名。
"""

import logging

from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout,
)

from ...engine.dataops.labelops import apply_replace, plan_replace, scan_pairs
from ...i18n import tr
from .base import DataToolWindow
from .common import DirRow, PreviewTable, confirm_execute, rel_display
from .registry import ToolSpec, register

log = logging.getLogger(__name__)


class LabelReplaceWindow(DataToolWindow):
    """标签替换工具窗口。"""

    tool_key = "label_replace"

    def __init__(self, on_closed=None):
        self._plan = []
        super().__init__(on_closed=on_closed)

    @property
    def tool_title(self):
        return tr("一键替换标签")

    # ================= 界面 =================
    def _build_settings(self, lay: QVBoxLayout):
        opts = QGroupBox(tr("替换规则"))
        form = QFormLayout(opts)
        self.src_row = DirRow(tr("数据集目录"))
        form.addRow(self.src_row)
        self.old_edit = QLineEdit()
        self.old_edit.setPlaceholderText(tr("旧标签（正则模式为整词匹配式）"))
        form.addRow(tr("旧标签"), self.old_edit)
        self.new_edit = QLineEdit()
        self.new_edit.setPlaceholderText(tr("新标签"))
        form.addRow(tr("新标签"), self.new_edit)
        self.regex_box = QCheckBox(tr("正则模式（旧标签按整词正则匹配）"))
        form.addRow(self.regex_box)
        lay.addWidget(opts)

        self.preview_btn = QPushButton(tr("扫描预览"))
        self.preview_btn.clicked.connect(self._scan)
        lay.addWidget(self.preview_btn)

        self.hint = QLabel(tr("先扫描预览，确认后执行；不做备份；自动包含子目录"))
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        self.table = PreviewTable([tr("文件"), tr("形状序号"), tr("旧标签"),
                                   tr("新标签"), tr("flags 键")])
        lay.addWidget(self.table, stretch=1)

        self.exec_btn = QPushButton(tr("执行替换"))
        self.exec_btn.setEnabled(False)
        self.exec_btn.clicked.connect(self._execute)
        lay.addWidget(self.exec_btn)

    # ================= 逻辑 =================
    def _scan(self):
        src = self.src_row.path()
        old = self.old_edit.text().strip()
        new = self.new_edit.text().strip()
        if src is None or not src.exists():
            QMessageBox.information(self, tr("请检查"), tr("请先选择存在的数据集目录"))
            return
        if not old or not new:
            QMessageBox.information(self, tr("请检查"), tr("请填写旧标签与新标签"))
            return
        if self._busy():
            return
        regex = self.regex_box.isChecked()

        def _work():
            pairs = scan_pairs(src)
            plan = plan_replace(pairs, old=old, new=new, regex=regex)
            rows = []
            for e in plan:
                shown = rel_display(e.json_path, src)
                for idx, lab, new_lab in e.hits:
                    rows.append((shown, idx, lab, new_lab, ""))
                for scope, key, _new_key in e.flag_key_hits:
                    rows.append((shown, "", "", key,
                                 "image" if scope == "image" else "shape"))
            return {
                "plan": plan, "rows": rows, "files": len(plan),
                "shapes": sum(len(e.hits) for e in plan),
                "flags": sum(len(e.flag_key_hits) for e in plan),
            }

        self._set_running(True)
        self._start_worker(_work, self._on_scan_done)

    def _on_scan_done(self, result: dict):
        self._set_running(False)
        self._plan = result["plan"]
        self.table.set_rows(result["rows"])
        self.hint.setText(tr("命中 {n} 个文件 / {s} 处形状标签 / {k} 处 flags 键；确认后执行").format(
            n=result["files"], s=result["shapes"], k=result["flags"]))
        self.exec_btn.setEnabled(bool(self._plan))

    def _execute(self):
        if not self._plan:
            return
        total = sum(len(e.hits) for e in self._plan)
        if not confirm_execute(self, tr("将替换 {n} 处标签（不做备份）。继续？").format(n=total)):
            return
        plan = list(self._plan)

        def _work():
            return apply_replace(plan)

        self._set_running(True)
        self._start_worker(_work, self._on_exec_done)

    def _on_exec_done(self, result: dict):
        self._set_running(False)
        self._plan = []
        self.exec_btn.setEnabled(False)
        self.hint.setText(tr("替换完成：{f} 个文件 / {s} 处形状 / {k} 处 flags 键").format(
            f=result["files"], s=result["shapes"], k=result["flag_keys"]))
        self.table.set_rows([])

    def _set_running(self, running: bool):
        self.preview_btn.setEnabled(not running)
        if not running:
            self.exec_btn.setEnabled(bool(self._plan))


register(ToolSpec(
    key="label_replace",
    title="一键替换标签",
    desc="按标签名批量替换数据集标注，先预览再执行",
    color="#0ea5e9", icon="toolbox_tool.png",
    keywords="replace label 替换 标签 批量 正则",
    tooltip="支持正则整词匹配，同步改写形状标签与 flags 键名。执行前不备份，建议先复制数据集副本。",
    factory=LabelReplaceWindow,
))
