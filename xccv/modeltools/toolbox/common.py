# -*- coding: utf-8 -*-
"""百宝箱子功能窗口公共控件：目录/文件选择行、预览表格、执行确认。"""

from pathlib import Path

from PyQt5.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QWidget,
)

from ...i18n import tr


def rel_display(path, root) -> str:
    """预览列显示：相对数据集目录的路径（含子目录），无法相对化时退回文件名。"""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return Path(path).name


class NoWheelSpin(QSpinBox):
    """滚轮不改值（防误滚，与模型工具 _ToolSpin 同语义）。"""

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpin(QDoubleSpinBox):
    """滚轮不改值（浮点版）。"""

    def wheelEvent(self, event):
        event.ignore()


class DirRow(QWidget):
    """"标签 + 路径框 + 浏览"一行（目录选择）。"""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        btn = QPushButton(tr("浏览") + "…")
        btn.clicked.connect(self._pick)
        lay.addWidget(QLabel(label))
        lay.addWidget(self.edit, stretch=1)
        lay.addWidget(btn)

    def _pick(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择目录"),
                                                self.edit.text() or "")
        if path:
            self.edit.setText(path)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None


class FileRow(QWidget):
    """"标签 + 路径框 + 浏览"一行（文件选择）。"""

    def __init__(self, label: str, name_filter: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        btn = QPushButton(tr("浏览") + "…")
        self._filter = name_filter
        btn.clicked.connect(self._pick)
        lay.addWidget(QLabel(label))
        lay.addWidget(self.edit, stretch=1)
        lay.addWidget(btn)

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("选择文件"),
                                              self.edit.text() or "",
                                              self._filter or "")
        if path:
            self.edit.setText(path)

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None


class PreviewTable(QTableWidget):
    """只读预览表格：一次填充表头与行数据。"""

    def __init__(self, headers: list, parent=None):
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)

    def set_rows(self, rows: list):
        self.setRowCount(0)
        for row in rows:
            r = self.rowCount()
            self.insertRow(r)
            for c, value in enumerate(row):
                self.setItem(r, c, QTableWidgetItem(str(value)))
        self.resizeColumnsToContents()


def confirm_execute(parent, text: str) -> bool:
    """执行前确认（预览确认红线：一律先问后动）。"""
    answer = QMessageBox.question(
        parent, tr("确认执行"), text,
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return answer == QMessageBox.Yes
