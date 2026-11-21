"""平台多语言：中文为源语言，英文走替换层。

设计（全平台中文串 3 万+，逐处包裹不现实）：
- tr(s)：英语言模式查 EN_TABLE，命中返回英文，未命中回退原文（增量覆盖，
  不阻断功能）；
- localize_widget_tree(root)：窗口构建完成后一次性遍历控件树替换静态
  文本（windowTitle/tooltip/标签/按钮/复选/组标题/页签/下拉项/表头/
  占位符/菜单）——统一接在 platform_entry.run_platform 的 build() 之后，
  动态创建的对话框在自己 __init__ 末尾各接一行；
- 动态文本（状态栏消息、分组计数等）由调用点显式 tr()；
- 语言持久化 platform_data/ui_lang.json（各平台进程共享）；
- 切换 = 保存 + 延时 1.6s 拉起新进程再关旧窗（错峰规避 single_instance
  守卫；一键切换，重启后生效）。

标注平台（labelme 子目录，上游代码为主）不接入；流程算子名/参数名是 IR 数据键，
不参与翻译，仅翻译其展示层文案。
"""

from __future__ import annotations
import logging
log = logging.getLogger(__name__)

import json
import os
import subprocess
import sys
from pathlib import Path

from .app_paths import app_root

_LANG_FILE = app_root() / "platform_data" / "ui_lang.json"
_lang_cache: str | None = None


def lang() -> str:
    """当前界面语言（"zh"/"en"；进程内缓存，文件为准）。

    环境变量 XCCV_UI_LANG 显式指定时优先（测试/CI 固定口径，避免读入
    用户偏好使断言随外部状态漂移）；未设置时照常读偏好文件。
    """
    global _lang_cache
    if _lang_cache is None:
        env_v = os.environ.get("XCCV_UI_LANG")
        if env_v in ("zh", "en"):
            _lang_cache = env_v
            return _lang_cache
        try:
            v = json.loads(_LANG_FILE.read_text(encoding="utf-8")).get("lang")
        except Exception:  # noqa: BLE001（缺文件/坏 JSON → 默认中文）
            v = None
        _lang_cache = v if v in ("zh", "en") else "zh"
    return _lang_cache


def set_lang(v: str) -> None:
    """保存语言偏好（各平台进程共享；下次启动生效）。"""
    global _lang_cache
    _lang_cache = "en" if v == "en" else "zh"
    try:
        _LANG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LANG_FILE.write_text(json.dumps({"lang": _lang_cache}),
                              encoding="utf-8")
    except OSError:
        pass  # 偏好写失败不阻断切换（本次会话仍生效）


def tr(s: str) -> str:
    """字符串翻译入口：英文模式查表，未命中/中文模式回退原文。"""
    if lang() != "en" or not s:
        return s
    from .i18n_en import TABLE

    return TABLE.get(s, s)


def localize_widget_tree(root) -> int:
    """一次性遍历控件树替换静态文本（仅英文模式生效）。返回替换条数。"""
    if lang() != "en":
        return 0
    from PyQt5.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QLabel,
                                 QLineEdit, QMenu, QPushButton, QRadioButton,
                                 QTableWidget, QTabWidget, QToolButton,
                                 QTreeWidget, QWidget)

    from .i18n_en import TABLE

    n = 0

    def _sub(t: str) -> str:
        nonlocal n
        en = TABLE.get(t)
        if en is not None:
            n += 1
            return en
        return t

    for w in [root] + root.findChildren(QWidget):
        try:
            tt = w.toolTip()
            if tt:
                w.setToolTip(_sub(tt))
            wt = w.windowTitle()
            if wt:
                w.setWindowTitle(_sub(wt))
        except Exception:  # noqa: BLE001（个别控件属性异常不阻断整体）
            log.debug("控件语言适配属性异常（跳过该控件）")
        if isinstance(w, (QLabel, QPushButton, QCheckBox, QRadioButton,
                          QToolButton)):
            t = w.text()
            if t:
                w.setText(_sub(t))
        if isinstance(w, QGroupBox):
            w.setTitle(_sub(w.title()))
        if isinstance(w, QComboBox):
            for i in range(w.count()):
                w.setItemText(i, _sub(w.itemText(i)))
        if isinstance(w, QTabWidget):
            for i in range(w.count()):
                w.setTabText(i, _sub(w.tabText(i)))
        if isinstance(w, QTreeWidget):
            hdr = w.headerItem()
            if hdr is not None:
                for c in range(w.columnCount()):
                    hdr.setText(c, _sub(hdr.text(c)))
        if isinstance(w, QTableWidget):
            for c in range(w.columnCount()):
                it = w.horizontalHeaderItem(c)
                if it is not None:
                    it.setText(_sub(it.text()))
        if isinstance(w, QLineEdit):
            pt = w.placeholderText()
            if pt:
                w.setPlaceholderText(_sub(pt))
        if isinstance(w, QMenu):
            w.setTitle(_sub(w.title()))
        try:
            for act in w.actions():
                if act.text():
                    act.setText(_sub(act.text()))
        except Exception:  # noqa: BLE001
            log.debug("菜单语言适配异常（跳过）")
    return n


def restart_current_platform(delay_s: float = 1.6) -> bool:
    """延时拉起当前平台新进程（错峰退出旧实例，规避单实例守卫）。

    开发态重跑 `pythonw -m <入口模块>`；安装态重跑同目录独立 exe。
    stager 代码经 sys.argv[1:] 取目标命令——目标命令前**不要放双减号
    分隔参数**：python -c 的 sys.argv 不会剥掉它，新进程会启动失败
    。
    返回是否已安排重启（False 时调用方不得关闭窗口，应提示手动重启）。"""
    stage = ("import time, subprocess, sys; time.sleep(%f); "
             "subprocess.Popen(sys.argv[1:], close_fds=True)" % delay_s)
    try:
        if getattr(sys, "frozen", False):
            py = Path(sys.base_prefix) / "pythonw.exe"
            if not py.is_file():
                return False  # 无随包解释器的冻结形态：仅保存，提示手动重启
            subprocess.Popen([str(py), "-c", stage, sys.executable],
                             close_fds=True)
            return True
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        mod = getattr(spec, "name", None) if spec is not None else None
        if not mod:
            return False
        from .launcher import _pythonw

        subprocess.Popen(
            [_pythonw(), "-c", stage, _pythonw(), "-m", mod],
            cwd=str(app_root()), close_fds=True)
        return True
    except Exception:  # noqa: BLE001
        return False


class _LangWatcher:
    """语言偏好监听器：偏好文件变化 → 400ms 防抖 → 调 rebuild 重建界面。

    QFileSystemWatcher/QTimer 全部挂自身且**应用退出前不析构**——刚处理完
    文件变化就被销毁会撞 Win32 监视线程的销毁竞态（0xC0000005 实测）；
    宿主窗口重建时经 set_rebuild 转移回调、把引用挂到新窗。"""

    def __init__(self, rebuild=None, parent=None):
        from PyQt5.QtCore import QFileSystemWatcher, QTimer

        self._rebuild = rebuild
        self._lang = lang()
        if not _LANG_FILE.exists():
            try:
                _LANG_FILE.parent.mkdir(parents=True, exist_ok=True)
                _LANG_FILE.write_text(json.dumps({"lang": self._lang}),
                                      encoding="utf-8")
            except OSError:
                pass  # 文件建不出来：本次监听可能失效（重开生效兜底）
        self._w = QFileSystemWatcher(parent)
        if _LANG_FILE.exists():
            self._w.addPath(str(_LANG_FILE))
        self._debounce = QTimer(self._w)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(400)
        self._w.fileChanged.connect(self._on_file_changed)
        self._debounce.timeout.connect(self._fire)

    def set_rebuild(self, rebuild):
        """宿主窗口重建后转移回调与引用（watcher 自身永不析构）。"""
        self._rebuild = rebuild

    def _on_file_changed(self, path):
        # write_text 替换文件会让 watcher 丢路径——重新挂上
        if str(_LANG_FILE) not in self._w.files():
            self._w.addPath(str(_LANG_FILE))
        self._debounce.start()

    def _fire(self):
        # 偏好文件是跨进程共享状态，lang() 走
        # 进程内缓存——非"保存"进程的缓存恒为启动值，外部改动后
        # new_lang == self._lang 恒成立 → 平台窗口永远不重建（实测复现：
        # 主界面保存 EN 后主界面自己重建、训练/测试平台窗口纹丝不动）。
        # 此处必须绕过缓存直读文件，并同步进程缓存。
        global _lang_cache
        try:
            new_lang = json.loads(_LANG_FILE.read_text(encoding="utf-8")).get("lang")
        except Exception:  # noqa: BLE001（坏文件/缺失：放弃本次，防抖再等）
            return
        if new_lang not in ("zh", "en"):
            return
        if new_lang == self._lang or self._rebuild is None:
            return
        self._lang = new_lang
        _lang_cache = new_lang
        self._rebuild()


def install_language_watcher(rebuild, parent=None):
    """监听语言偏好文件，变化即调 rebuild() 重建界面（免重启生效）。

    rebuild：无参回调——销毁旧窗口并按当前语言重建（由宿主提供：先建新窗
    再关旧窗，避免 quitOnLastWindowClosed 误退）。本进程保存偏好后同样
    经此回调生效（与主题切换的文件监听同款体验）。
    返回监听器（宿主窗口重建时经 set_rebuild 转移回调并保留引用；
    返回 None=文件不可监听，重开生效）。"""
    try:
        return _LangWatcher(rebuild, parent=parent)
    except Exception:  # noqa: BLE001（无 Qt 环境/监听失败 → 重开生效）
        return None



