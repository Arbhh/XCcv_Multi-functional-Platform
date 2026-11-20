"""V0.6 F2-5 回归：浅色/暗色主题**实时切换**（第 2 点需求：此前需重启软件才生效）。

- set_theme：别名常量/APP_QSS 重算；apply_app_theme 登记的窗口重设样式（登记后
  追加的局部规则保留）；bind_style 内联样式重算；refresh_theme() 部件被调用；
- 暗色可读性：QPlainTextEdit/QTextEdit 有主题规则（此前暗色下白底浅字）；
- 主界面：apply_theme_choice 写偏好 + 本进程即时变色（侧栏/窗口键/导航图标）；
- 跨进程：子进程装 install_theme_watcher，父进程改偏好文件 → 子进程收到切换；
- XCCV_THEME 环境变量钉死主题时不跟随文件。

本模块所有用例结束必须恢复浅色（全量门禁同进程：test_ui_upgrade 断言
APP_QSS == build_qss("light")）。
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel, QPlainTextEdit, QWidget

_APP = QApplication.instance() or QApplication([])
_ROOT = Path(__file__).resolve().parents[1]

DARK_SURFACE = "#1c2634"
LIGHT_SURFACE = "#f4f7fb"


def _restore_light():
    import xccv.ui_style as us

    us.set_theme("light")


class SetThemeTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(_restore_light)

    def test_globals_and_registered_widgets_follow(self):
        import xccv.ui_style as us

        w = QWidget()
        us.apply_app_theme(w)
        w.setStyleSheet(w.styleSheet() + "\nQWidget#keepMe { background: transparent; }")
        lbl = QLabel("x")
        us.bind_style(lbl, lambda: f"color:{us.token('text_1')};")
        calls = []

        class _Custom(QWidget):
            def refresh_theme(self):
                calls.append(1)

        custom = _Custom()
        self.assertIn(LIGHT_SURFACE, w.styleSheet())

        self.assertTrue(us.set_theme("dark"))
        self.assertEqual(us.ACTIVE_THEME, "dark")
        self.assertEqual(us.BG, DARK_SURFACE)
        self.assertIn(DARK_SURFACE, us.APP_QSS)
        self.assertIn(DARK_SURFACE, w.styleSheet())
        self.assertNotIn(LIGHT_SURFACE, w.styleSheet())
        self.assertIn("QWidget#keepMe", w.styleSheet())          # 局部追加规则保留
        self.assertIn("#e7eef7", lbl.styleSheet())                # 暗色 text_1
        self.assertEqual(calls, [1])                              # refresh_theme 被调
        self.assertFalse(us.set_theme("dark"))                    # 同主题不重复广播

        self.assertTrue(us.set_theme("light"))
        self.assertIn(LIGHT_SURFACE, w.styleSheet())
        self.assertEqual(us.APP_QSS, us.build_qss("light"))
        del custom

    def test_invalid_theme_rejected(self):
        import xccv.ui_style as us

        with self.assertRaises(ValueError):
            us.set_theme("blue")

    def test_dead_widgets_skipped(self):
        """登记过的部件已销毁：切换不抛错（RuntimeError 被吞并出集）。"""
        import sip

        import xccv.ui_style as us

        w = QWidget()
        us.apply_app_theme(w)
        sip.delete(w)
        us.set_theme("dark")   # 不应抛


class PlainTextReadabilityTest(unittest.TestCase):
    """F2-4 暗色截图实测：日志/说明框暗色下白底浅字——补 QPlainTextEdit 规则。"""

    def test_plaintext_rule_both_themes(self):
        import xccv.ui_style as us

        for theme, panel in (("light", "#ffffff"), ("dark", "#232f3e")):
            qss = us.build_qss(theme)
            self.assertIn("QPlainTextEdit, QTextEdit", qss)
            block = qss.split("QPlainTextEdit, QTextEdit", 1)[1].split("}", 1)[0]
            self.assertIn(panel, block, theme)

    def test_dark_plaintext_background_is_dark(self):
        import xccv.ui_style as us

        self.addCleanup(_restore_light)
        host = QWidget()
        us.apply_app_theme(host)
        edit = QPlainTextEdit(host)
        edit.setPlainText("日志")
        host.resize(200, 120)
        us.set_theme("dark")
        host.show()
        _APP.processEvents()
        img = edit.viewport().grab().toImage()
        c = img.pixelColor(img.width() - 5, img.height() - 5)
        self.assertLess((c.red() + c.green() + c.blue()) / 3, 90, c.name())  # 暗底
        host.close()


class HomeWindowLiveThemeTest(unittest.TestCase):
    """主界面设置：保存即生效（侧栏/窗口键/导航图标/主题偏好落盘）。"""

    def test_apply_theme_choice_live(self):
        import xccv.ui_style as us
        import xccv.ui_widgets as uw
        from xccv.app import HomeWindow

        self.addCleanup(_restore_light)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        prefs = Path(tmp.name) / "ui_prefs.json"
        with mock.patch.object(uw, "_prefs_path", return_value=prefs):
            win = HomeWindow()
            self.addCleanup(win.close)
            side_before = [w for w in win.findChildren(QWidget)
                           if "#f7fafd" in w.styleSheet()]
            self.assertTrue(side_before, "浅色侧栏底色应可见")
            icon_before = win.nav_list.item(0).icon().pixmap(45, 45).toImage()
            win.apply_theme_choice("dark")
            self.assertEqual(json.loads(prefs.read_text("utf-8"))["theme"], "dark")
            self.assertEqual(us.ACTIVE_THEME, "dark")
            sheets = " ".join(w.styleSheet() for w in win.findChildren(QWidget))
            self.assertIn("#18212d", sheets)            # 暗色侧栏底
            self.assertIn(DARK_SURFACE, win.centralWidget().styleSheet())  # 圆角壳
            self.assertEqual(win._btn_close._c_line.name(), "#a6b6ca")      # 窗口键重取色
            icon_after = win.nav_list.item(0).icon().pixmap(45, 45).toImage()
            self.assertNotEqual(icon_before, icon_after)  # 导航矢量图标按新色重绘
            win.apply_theme_choice("light")
            self.assertEqual(us.ACTIVE_THEME, "light")


_CHILD = r'''
import json, os, sys, time
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.pop("XCCV_THEME", None)
sys.path.insert(0, sys.argv[2])
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
import xccv.ui_style as us
prefs = Path(sys.argv[1])
us._prefs_path = lambda: prefs
app = QApplication([])
print("START", us.ACTIVE_THEME, flush=True)
assert us.install_theme_watcher()
us.theme_bus().theme_changed.connect(lambda n: (print("CHANGED", n, flush=True), app.quit()))
print("READY", flush=True)
QTimer.singleShot(15000, app.quit)
app.exec_()
print("END", us.ACTIVE_THEME, flush=True)
'''


class CrossProcessWatcherTest(unittest.TestCase):
    """跨进程：子进程监听偏好文件，父进程（模拟主界面设置）改主题 → 子进程切换。"""

    def test_child_process_follows_prefs_file(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        prefs = Path(tmp.name) / "platform_data" / "ui_prefs.json"
        prefs.parent.mkdir(parents=True)
        prefs.write_text(json.dumps({"theme": "light"}), encoding="utf-8")
        script = Path(tmp.name) / "child.py"
        script.write_text(_CHILD, encoding="utf-8")
        env = dict(os.environ)
        env.pop("XCCV_THEME", None)
        proc = subprocess.Popen([sys.executable, str(script), str(prefs), str(_ROOT)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", env=env)
        lines = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.strip())
            if line.startswith("READY"):
                time.sleep(0.3)
                prefs.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
            if line.startswith("END"):
                break
        proc.wait(timeout=30)
        out = "\n".join(lines)
        self.assertIn("START light", out)
        self.assertIn("CHANGED dark", out, out)
        self.assertIn("END dark", out, out)

    def test_env_pin_ignores_prefs(self):
        import xccv.ui_style as us

        self.addCleanup(_restore_light)
        with mock.patch.dict(os.environ, {"XCCV_THEME": "light"}), \
                mock.patch.object(us, "_prefs_theme", return_value="dark"), \
                mock.patch.object(us, "_watch_paths"):
            us._on_prefs_changed("x")
        self.assertEqual(us.ACTIVE_THEME, "light")
        with mock.patch.dict(os.environ, {}, clear=False), \
                mock.patch.object(us, "_prefs_theme", return_value="dark"), \
                mock.patch.object(us, "_watch_paths"):
            os.environ.pop("XCCV_THEME", None)
            us._on_prefs_changed("x")
        self.assertEqual(us.ACTIVE_THEME, "dark")


if __name__ == "__main__":
    unittest.main()
