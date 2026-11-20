"""自定义算子编辑器（V0.6 T4-5）：信息 / 参数 / 端口 / 实现 四个页签。

**Ctrl+S 保存**：校验 → 写 platform_data/operators/<id>.op.json → 热重载注册
（custom/<id>）→ toast 提示 → saved 信号通知工作台刷新列表并选中，可立即试跑。
非模态：编辑期间工作台照常可用（边改边试跑）。
"""

import json

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QShortcut, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .engine.flow import custom_ops as co
from .engine.flow.ir import PORT_TYPES
from .engine.flow.node_base import PARAM_TYPES
from .i18n import localize_widget_tree, tr
from .ui_widgets import toast

_PARAM_COLS = ("键", "标签", "类型", "默认值", "最小", "最大", "选项")
_HINTS = {
    "script": "脚本：定义 def run(inputs, params, ctx) 并返回 {输出端口名: 数据}。可直接使用 "
              "np / cv2 / math / ImageData / Geometry / MeasureResult / NumberVal；import 仅限 "
              "numpy、cv2、math、statistics；禁用 eval/exec/open 等。",
    "composite": "复合：填子流程 JSON——{\"graph\": 流程图, \"input_map\": {输入名: [[节点id, 输入槽]]}, "
                 "\"output_map\": {输出名: [节点id, \"输出端口名\"]}}；节点属性里写 \"${参数键}\" 注入参数值。",
}


def _combo(items, current) -> QComboBox:
    w = QComboBox()
    w.addItems(list(items))
    w.setCurrentText(str(current))
    return w


def _parse_value(text: str, typ: str):
    t = (text or "").strip()
    if typ == "int":
        return int(float(t or 0))
    if typ == "float":
        return float(t or 0)
    if typ == "bool":
        return t.lower() in ("1", "true", "yes", "是", "开")
    return t


class OperatorEditorDialog(QDialog):
    """自定义算子编辑对话框（非模态）。"""

    saved = pyqtSignal(str)   # 注册名 custom/<id>

    def __init__(self, parent=None, op_def: dict | None = None, root=None):
        super().__init__(parent)
        self.setWindowTitle(tr("编辑自定义算子"))
        self.resize(860, 620)
        self._root = root
        self._build()
        self.load_def(op_def or co.new_op_template("script"))
        self._save_shortcut = QShortcut(QKeySequence.Save, self)
        self._save_shortcut.activated.connect(self.save)
        localize_widget_tree(self)

    # ---------- UI ----------
    def _build(self):
        v = QVBoxLayout(self)
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)

        info = QWidget()
        f = QFormLayout(info)
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("小写字母开头，仅小写字母/数字/下划线，如 my_blur")
        self.title_edit = QLineEdit()
        self.desc_edit = QPlainTextEdit()
        self.desc_edit.setMaximumHeight(100)
        self.kind_combo = _combo(("script", "composite"), "script")
        self.kind_combo.currentTextChanged.connect(self._on_kind_changed)
        f.addRow("算子 id", self.id_edit)
        f.addRow("显示名", self.title_edit)
        f.addRow("功能说明", self.desc_edit)
        f.addRow("实现方式", self.kind_combo)
        self.tabs.addTab(info, "信息")

        pw = QWidget()
        pv = QVBoxLayout(pw)
        self.param_table = QTableWidget(0, len(_PARAM_COLS))
        self.param_table.setHorizontalHeaderLabels(_PARAM_COLS)
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.param_table.horizontalHeaderItem(6).setToolTip("仅 enum 类型：选项以逗号分隔")
        pv.addWidget(self.param_table, 1)
        pv.addLayout(self._row_buttons(self._add_param_row, self.param_table))
        self.tabs.addTab(pw, "参数")

        tw = QWidget()
        tv = QVBoxLayout(tw)
        tv.addWidget(QLabel("输入端口"))
        self.in_table = QTableWidget(0, 3)
        self.in_table.setHorizontalHeaderLabels(("名称", "类型", "必需"))
        self.in_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tv.addWidget(self.in_table, 1)
        tv.addLayout(self._row_buttons(lambda: self._add_port_row(self.in_table), self.in_table))
        tv.addWidget(QLabel("输出端口"))
        self.out_table = QTableWidget(0, 2)
        self.out_table.setHorizontalHeaderLabels(("名称", "类型"))
        self.out_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tv.addWidget(self.out_table, 1)
        tv.addLayout(self._row_buttons(lambda: self._add_port_row(self.out_table), self.out_table))
        self.tabs.addTab(tw, "端口")

        iw = QWidget()
        iv = QVBoxLayout(iw)
        self.impl_hint = QLabel()
        self.impl_hint.setWordWrap(True)
        iv.addWidget(self.impl_hint)
        self.impl_edit = QPlainTextEdit()
        self.impl_edit.setFont(QFont("Consolas", 10))
        iv.addWidget(self.impl_edit, 1)
        self.tabs.addTab(iw, "实现")

        self.status = QLabel("Ctrl+S 保存：校验通过即注册，可在工作台左侧列表直接试跑")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        bar = QHBoxLayout()
        bar.addStretch(1)
        self.btn_save = QPushButton("保存（Ctrl+S）")
        self.btn_save.clicked.connect(self.save)
        self.btn_export = QPushButton("导出 .xcvop…")
        self.btn_export.clicked.connect(self._export)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        for b in (self.btn_save, self.btn_export, btn_close):
            bar.addWidget(b)
        v.addLayout(bar)

    @staticmethod
    def _row_buttons(add_fn, table) -> QHBoxLayout:
        row = QHBoxLayout()
        b_add = QPushButton("添加")
        b_add.clicked.connect(add_fn)
        b_del = QPushButton("删除所选")
        b_del.clicked.connect(lambda: table.removeRow(table.currentRow())
                              if table.currentRow() >= 0 else None)
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addStretch(1)
        return row

    def _add_param_row(self, key="", spec=None):
        spec = spec or {"label": "", "type": "int", "default": 0}
        r = self.param_table.rowCount()
        self.param_table.insertRow(r)
        rng = spec.get("range") or ["", ""]
        cells = [key, spec.get("label", ""), None, spec.get("default", ""),
                 rng[0], rng[1], ",".join(str(o) for o in (spec.get("options") or []))]
        for c, val in enumerate(cells):
            if c == 2:
                self.param_table.setCellWidget(r, c, _combo(PARAM_TYPES, spec.get("type", "int")))
            else:
                self.param_table.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))

    def _add_port_row(self, table, port=None):
        port = port or {"name": "", "type": "image", "required": True}
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, QTableWidgetItem(str(port.get("name", ""))))
        table.setCellWidget(r, 1, _combo(PORT_TYPES.keys(), port.get("type", "image")))
        if table is self.in_table:
            chk = QCheckBox()
            chk.setChecked(bool(port.get("required", True)))
            table.setCellWidget(r, 2, chk)

    def _on_kind_changed(self, kind: str):
        self.impl_hint.setText(_HINTS.get(kind, ""))
        cur = self.impl_edit.toPlainText().strip()
        other = "composite" if kind == "script" else "script"
        if not cur or cur == self._impl_text(co.new_op_template(other)).strip():
            self.impl_edit.setPlainText(self._impl_text(co.new_op_template(kind)))

    @staticmethod
    def _impl_text(d: dict) -> str:
        if d.get("kind") == "script":
            return str((d.get("script") or {}).get("code") or "")
        return json.dumps(d.get("composite") or {}, ensure_ascii=False, indent=1)

    # ---------- 数据 ----------
    def load_def(self, d: dict) -> None:
        self.id_edit.setText(str(d.get("id", "")))
        self.title_edit.setText(str(d.get("title", "")))
        self.desc_edit.setPlainText(str(d.get("description", "")))
        self.param_table.setRowCount(0)
        for key, spec in (d.get("params") or {}).items():
            self._add_param_row(key, spec)
        for table, ports in ((self.in_table, d.get("inputs")), (self.out_table, d.get("outputs"))):
            table.setRowCount(0)
            for p in ports or []:
                self._add_port_row(table, p)
        self.kind_combo.blockSignals(True)
        self.kind_combo.setCurrentText(str(d.get("kind", "script")))
        self.kind_combo.blockSignals(False)
        self.impl_hint.setText(_HINTS.get(d.get("kind", "script"), ""))
        self.impl_edit.setPlainText(self._impl_text(d))

    def _cell(self, table, r, c) -> str:
        item = table.item(r, c)
        return item.text().strip() if item is not None else ""

    def to_def(self) -> dict:
        """表单 → op.json 定义；实现页签 JSON 解析失败抛 ValueError（中文）。"""
        params = {}
        for r in range(self.param_table.rowCount()):
            key = self._cell(self.param_table, r, 0)
            if not key:
                continue
            typ = self.param_table.cellWidget(r, 2).currentText()
            spec = {"label": self._cell(self.param_table, r, 1) or key, "type": typ,
                    "default": _parse_value(self._cell(self.param_table, r, 3), typ)}
            lo, hi = self._cell(self.param_table, r, 4), self._cell(self.param_table, r, 5)
            if typ in ("int", "float") and lo and hi:
                spec["range"] = [_parse_value(lo, typ), _parse_value(hi, typ)]
            opts = self._cell(self.param_table, r, 6)
            if typ == "enum":
                spec["options"] = [o.strip() for o in opts.replace("，", ",").split(",") if o.strip()]
            params[key] = spec
        inputs = [{"name": self._cell(self.in_table, r, 0),
                   "type": self.in_table.cellWidget(r, 1).currentText(),
                   "required": self.in_table.cellWidget(r, 2).isChecked()}
                  for r in range(self.in_table.rowCount()) if self._cell(self.in_table, r, 0)]
        outputs = [{"name": self._cell(self.out_table, r, 0),
                    "type": self.out_table.cellWidget(r, 1).currentText()}
                   for r in range(self.out_table.rowCount()) if self._cell(self.out_table, r, 0)]
        kind = self.kind_combo.currentText()
        d = {"schema": co.SCHEMA_VERSION, "id": self.id_edit.text().strip(),
             "title": self.title_edit.text().strip(),
             "description": self.desc_edit.toPlainText().strip(), "kind": kind,
             "inputs": inputs, "outputs": outputs, "params": params}
        text = self.impl_edit.toPlainText()
        if kind == "script":
            d["script"] = {"code": text}
        else:
            try:
                d["composite"] = json.loads(text or "{}")
            except json.JSONDecodeError as e:
                raise ValueError(f"实现页签 JSON 格式错误：第 {e.lineno} 行 {e.msg}") from None
        return d

    def save(self) -> bool:
        try:
            d = self.to_def()
            co.save_op(d, self._root, overwrite=True)
        except Exception as e:  # noqa: BLE001  校验失败/IO 失败都给人话
            self.status.setText(f"{tr('未保存：')}{e}")
            self.status.setStyleSheet("color:#b91c1c;")
            toast(self, "保存失败，详见对话框底部提示", "error")
            return False
        name = co.PREFIX + d["id"]
        self.status.setText(f"{tr('已保存并注册：')}{name}"
                            f"{tr('（可在工作台左侧算子列表直接试跑）')}")
        self.status.setStyleSheet("color:#047857;")
        toast(self, f"已保存：{d['title']}", "success")
        self.saved.emit(name)
        return True

    def _export(self):
        op_id = self.id_edit.text().strip()
        if not op_id or op_id not in co.list_custom_ops(self._root):
            self.status.setText("请先保存算子再导出")
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("导出自定义算子"), f"{op_id}.xcvop",
                                              tr("XCcv 自定义算子 (*.xcvop)"))
        if path:
            out = co.export_xcvop(op_id, path, self._root)
            toast(self, f"{tr('已导出：')}{out.name}", "success")


# ---- Cython 编译态槽包装（见 xccv/qt_slots.py；新增 UI 模块必须加） ----
from xccv.qt_slots import apply_auto_slot_module  # noqa: E402
apply_auto_slot_module(globals())
