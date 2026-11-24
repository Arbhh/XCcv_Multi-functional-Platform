# -*- coding: utf-8 -*-
"""百宝箱工作台：子功能项目卡片墙 + 顶部搜索框（注册表驱动）。

嵌套落位：主界面 → 模型工具 → 【百宝箱】卡片 → 本窗口 → 子功能独立窗口。
扩展端口：新子功能在 feature 模块里 register(ToolSpec(...)) 即上架，
宿主（本窗口与模型工具卡片墙）零改动。
"""

import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QScrollArea, QVBoxLayout, QWidget,
)

from ...i18n import tr
from ...ui_style import apply_app_theme, icon_path
from ..modeltools_window import _QApp, _ToolCard
from .registry import search


class ToolboxWindow(QMainWindow):
    """百宝箱主窗口：搜索框过滤注册表，卡片点击打开子功能独立窗口。"""

    def __init__(self, on_closed=None):
        super().__init__()
        self.on_closed = on_closed
        self.setWindowTitle(tr("百宝箱") + "  -  " + tr("XCcv 模型工具"))
        self.setWindowIcon(QIcon(str(icon_path("toolbox.png"))))
        self.resize(1080, 600)
        apply_app_theme(self)
        self._tool_windows = {}
        self._build_ui()
        from ...i18n import localize_widget_tree

        localize_widget_tree(self)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(64)
        head = QHBoxLayout(header)
        head.setContentsMargins(22, 10, 22, 10)
        title = QLabel(tr("百宝箱"))
        title.setObjectName("appTitle")
        head.addWidget(title)
        head.addStretch(1)
        root.addWidget(header)

        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(22, 16, 22, 14)
        body_lay.setSpacing(10)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(tr("搜索工具（名称/关键词）…"))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._rebuild_cards)
        body_lay.addWidget(self.search_edit)

        self.empty_hint = QLabel(tr("没有匹配的工具"))
        self.empty_hint.setObjectName("hint")
        self.empty_hint.setAlignment(Qt.AlignCenter)
        self.empty_hint.setVisible(False)
        body_lay.addWidget(self.empty_hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._cards_host = QWidget()
        self._cards_grid = QGridLayout(self._cards_host)
        self._cards_grid.setContentsMargins(0, 0, 0, 0)
        self._cards_grid.setSpacing(14)
        self._cards_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self._cards_host)
        body_lay.addWidget(scroll, stretch=1)

        root.addWidget(body, stretch=1)
        self.setCentralWidget(central)
        self._rebuild_cards()

    def _rebuild_cards(self):
        """按注册表 + 搜索词重建卡片墙。"""
        while self._cards_grid.count():
            item = self._cards_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        specs = search(self.search_edit.text())
        self.empty_hint.setVisible(not specs)
        for i, spec in enumerate(specs):
            card = _ToolCard(spec.key, tr(spec.title), tr(spec.desc), spec.color,
                             True, icon=spec.icon, tooltip=tr(spec.tooltip))
            card.clicked.connect(self._open_tool)
            row, col = divmod(i, 3)
            self._cards_grid.addWidget(card, row, col)

    def _open_tool(self, key: str):
        from .registry import get

        spec = get(key)
        if spec is None or spec.factory is None:
            return
        window = self._tool_windows.get(key)
        if window is not None and window.isVisible():
            window.activateWindow()
            window.raise_()
            return
        banner = QLabel(tr("正在加载工具组件") + "…", self)
        banner.setObjectName("openProjectBanner")
        banner.setStyleSheet(
            "QLabel#openProjectBanner { background: rgba(46, 38, 58, 224);"
            " color: white; padding: 6px 18px; border-radius: 12px;"
            " font-weight: bold; font-size: 13px; }")
        banner.adjustSize()
        banner.move((self.width() - banner.width()) // 2, 12)
        banner.show()
        banner.raise_()
        banner.repaint()
        _QApp.processEvents()
        _QApp.processEvents()
        t0 = time.monotonic()
        _QApp.setOverrideCursor(Qt.WaitCursor)
        try:
            window = spec.factory(on_closed=self._on_tool_closed)
            window.show()
        finally:
            _QApp.restoreOverrideCursor()
            remain_ms = max(0, int(700 - (time.monotonic() - t0) * 1000))

            def _hide():
                try:
                    banner.hide()
                    banner.setParent(None)
                    banner.deleteLater()
                except RuntimeError:
                    pass  # 快速关闭时横幅随窗口销毁

            QTimer.singleShot(remain_ms, _hide)
        self._tool_windows[key] = window

    def _on_tool_closed(self, key: str):
        self._tool_windows.pop(key, None)

    def open_tool_windows(self) -> dict:
        """已打开的子功能窗口（测试用）。"""
        return dict(self._tool_windows)

    def closeEvent(self, event):
        running = [k for k, w in self._tool_windows.items() if w.isVisible()]
        if running:
            event.accept()  # 子功能窗口独立存活，不影响关闭工作台
            return
        if self.on_closed is not None:
            self.on_closed("toolbox")
        event.accept()


def main():
    from ...platform_entry import run_platform

    run_platform("toolbox", "百宝箱", lambda: ToolboxWindow(), heavy=True)


if __name__ == "__main__":
    main()
