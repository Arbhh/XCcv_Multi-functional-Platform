"""回归：部署平台退出与推理服务生命周期绑定。

- 部署平台真正退出（托盘菜单退出/无托盘强退）前同步停止推理服务；
- 关闭窗口仍默认只最小化到系统托盘（服务继续运行，行为不变）。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])


class _FakeProc:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return None   # 一直存活

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, _ms):
        self.waited = True
        return 0


class StopServiceBlockingTest(unittest.TestCase):
    def test_terminates_own_process_even_if_offline(self):
        """服务 HTTP 已离线但本进程拉起的子进程仍在：退出兜底 terminate。"""
        from xccv.deploy_platform import ServicePanel

        panel = ServicePanel()
        self.addCleanup(panel.deleteLater)
        fake = _FakeProc()
        panel._proc = fake
        panel._service_ok = False   # 不走 HTTP 分支（无真实服务）
        panel.stop_service_blocking(max_ms=10)
        self.assertIsNone(panel._proc)
        self.assertTrue(fake.terminated)
        self.assertTrue(fake.waited)
        self.assertFalse(fake.killed)

    def test_proc_none_noop(self):
        from xccv.deploy_platform import ServicePanel

        panel = ServicePanel()
        self.addCleanup(panel.deleteLater)
        panel._proc = None
        panel._service_ok = False
        panel.stop_service_blocking(max_ms=10)   # 不抛异常即可


class ServiceWindowQuitTest(unittest.TestCase):
    def _window(self):
        from unittest.mock import patch

        from xccv.deploy_platform import ServiceWindow

        # ServiceHealthPoller 在 __init__ 内惰性导入，须 patch 源模块
        with patch("xccv.service_health.ServiceHealthPoller"):
            win = ServiceWindow()
        self.addCleanup(win.deleteLater)
        return win

    def test_real_quit_stops_service_first(self):
        """托盘「退出部署平台」：先同步停服，再真正关闭窗口。"""
        from unittest.mock import patch

        win = self._window()
        calls = []
        with patch.object(win._page, "stop_service_blocking",
                          side_effect=lambda **kw: calls.append(kw)):
            win._real_quit()
        self.assertEqual(len(calls), 1)
        self.assertTrue(win._force_quit)

    def test_close_event_no_tray_stops_service(self):
        """无托盘环境的直接关闭：closeEvent 内补停服（批注四绑定兜底）。"""
        from unittest.mock import patch

        win = self._window()
        calls = []
        with patch("PyQt5.QtWidgets.QSystemTrayIcon.isSystemTrayAvailable",
                   return_value=False), \
             patch.object(win._page, "stop_service_blocking",
                          side_effect=lambda **kw: calls.append(kw)):
            win.close()
        self.assertEqual(len(calls), 1)

    def test_real_quit_does_not_double_stop(self):
        """_real_quit 已停服后 closeEvent 不重复停服。"""
        from unittest.mock import patch

        win = self._window()
        calls = []
        with patch.object(win._page, "stop_service_blocking",
                          side_effect=lambda **kw: calls.append(kw)):
            win._real_quit()
        self.assertEqual(len(calls), 1)

    def test_quit_menu_text_bound_to_service(self):
        """文案与行为一致：退出=服务同步停止（批注四文字提示）。"""
        from xccv.deploy_platform import ServiceWindow

        win = self._window()
        texts = [a.text() for m in [win._tray.contextMenu()]
                 for a in m.actions()]
        self.assertTrue(any("推理服务同步停止" in t for t in texts), texts)


if __name__ == "__main__":
    unittest.main()

