"""V0.6 T2-1 回归：主线程卡顿看门狗 + T2-6 auto_cleanup_thread。

- 人为阻塞主线程 → freeze_watchdog.log 出现"卡顿开始+栈顶"；
- 事件循环恢复 → 补记"卡顿结束：持续 Xs"；
- stop() 卸载后不再记录；
- 产品式调用（丢弃返回值）心跳不得被 GC；
- 启动期（事件循环未运转）阻塞不计卡顿（布防=连续 3 次准时心跳后）；
- auto_cleanup_thread：finished 后引用置 None + deleteLater。
"""

import os
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from xccv.gui_utils import auto_cleanup_thread, install_freeze_watchdog
from xccv.splash import LoadingSplash  # noqa: F401  确保 xccv 包可导入

_APP = None


def _app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _pump(ms: int):
    """泵事件循环 ms 毫秒（喂心跳）。"""
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _pump_until(cond, timeout_ms: int = 15000, step_ms: int = 20):
    """泵事件循环直到条件成立或超时（负载鲁棒：固定 pump 时长在重载套件下不稳）。"""
    from PyQt5.QtCore import QEventLoop, QTimer

    import time as _t

    deadline = _t.monotonic() + timeout_ms / 1000.0
    while _t.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(step_ms, loop.quit)
        loop.exec_()
        try:
            if cond():
                return True
        except Exception:
            pass
    return False


class FreezeWatchdogTest(unittest.TestCase):
    def test_freeze_detected_and_recovered(self):
        _app()
        log_path = Path(os.environ.get("TEMP", "/tmp")) / "xccv_wd_test" / "freeze_watchdog.log"
        # 用独立日志目录：crash_log_path() 依赖 app_root，这里直接测行为
        import tempfile

        import xccv.gui_utils as gu

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        real_crash = gu.crash_log_path
        gu.crash_log_path = lambda: Path(tmp.name) / "crash.log"
        try:
            stop = install_freeze_watchdog(threshold_ms=120, heartbeat_ms=30)
            entry = gu._WATCHDOG_KEEPALIVE[-1]
            self.assertTrue(_pump_until(lambda: entry[2]["armed"], 15000), "看门狗未布防")
            marker = 0  # 主线程将在此处被阻塞——栈顶应含本测试文件名
            time.sleep(0.5)  # 人为卡顿：QTimer 心跳停摆
            log_file = Path(tmp.name) / "freeze_watchdog.log"

            def _read_log():
                return log_file.read_text("utf-8") if log_file.exists() else ""

            # 恢复事件循环：心跳恢复 → 补记结束（负载鲁棒：泵到出现为止）
            _pump_until(lambda: "卡顿结束" in _read_log(), 15000)
            stop()
            text = _read_log()
            self.assertIn("卡顿开始", text)
            self.assertIn("卡顿结束", text)
            self.assertIn("test_v06_freeze_watchdog.py", text)  # 栈顶留痕
            self.assertIn("主线程栈顶", text)
            del marker
        finally:
            gu.crash_log_path = real_crash

    def test_stop_disables_detection(self):
        import tempfile

        import xccv.gui_utils as gu

        _app()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        real_crash = gu.crash_log_path
        gu.crash_log_path = lambda: Path(tmp.name) / "crash.log"
        try:
            stop = install_freeze_watchdog(threshold_ms=100, heartbeat_ms=30)
            _pump(100)
            stop()
            time.sleep(0.4)  # 卸载后再卡顿不应记录
            _pump(150)
            log_file = Path(tmp.name) / "freeze_watchdog.log"
            text = log_file.read_text("utf-8") if log_file.exists() else ""
            self.assertNotIn("卡顿开始", text)
        finally:
            gu.crash_log_path = real_crash


class WatchdogLifetimeTest(unittest.TestCase):
    """产品入口丢弃返回值时，心跳 QTimer 必须存活、看门狗持续有效。"""

    def setUp(self):
        import tempfile

        import xccv.gui_utils as gu

        _app()
        self.gu = gu
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        real_crash = gu.crash_log_path
        gu.crash_log_path = lambda: Path(self.tmp.name) / "crash.log"
        self.addCleanup(setattr, gu, "crash_log_path", real_crash)
        self.log_file = Path(self.tmp.name) / "freeze_watchdog.log"

    def _read(self):
        return self.log_file.read_text("utf-8") if self.log_file.exists() else ""

    def _dispose(self, entry):
        timer, _thread, state = entry
        state["stop"] = True
        timer.stop()
        if entry in self.gu._WATCHDOG_KEEPALIVE:
            self.gu._WATCHDOG_KEEPALIVE.remove(entry)

    def test_discarded_return_value_keeps_heartbeat(self):
        import gc

        install_freeze_watchdog(threshold_ms=120, heartbeat_ms=30)  # 同产品：丢弃返回值
        entry = self.gu._WATCHDOG_KEEPALIVE[-1]
        self.addCleanup(self._dispose, entry)
        gc.collect()
        self.assertTrue(_pump_until(lambda: entry[2]["armed"], 15000),
                        "心跳 QTimer 被回收：看门狗无法布防（P153 回归）")
        time.sleep(0.45)  # 布防后的真实卡顿
        self.assertTrue(_pump_until(lambda: "卡顿结束" in self._read(), 15000),
                        "卡顿结束未记录：心跳在卡顿后未恢复（P153 回归）")
        self.assertIn("卡顿开始", self._read())

    def test_startup_block_before_event_loop_not_logged(self):
        install_freeze_watchdog(threshold_ms=120, heartbeat_ms=30)
        entry = self.gu._WATCHDOG_KEEPALIVE[-1]
        self.addCleanup(self._dispose, entry)
        time.sleep(0.45)  # 模拟 exec_ 前的窗口构建（无事件循环）
        self.assertTrue(_pump_until(lambda: entry[2]["armed"], 15000))
        _pump(150)
        self.assertNotIn("卡顿开始", self._read(),
                         "启动期（未布防）阻塞不应计为卡顿")


class AutoCleanupThreadTest(unittest.TestCase):
    def test_thread_cleanup_sets_none_and_deletes(self):
        from PyQt5.QtCore import QObject, QThread

        _app()

        class _Dummy(QThread):
            pass

        class _Holder(QObject):
            pass

        holder = _Holder()
        t = _Dummy()
        holder.worker = t
        auto_cleanup_thread(holder, "worker", t)
        t.start()
        t.quit()
        t.wait(2000)
        self.assertTrue(_pump_until(lambda: holder.worker is None, 15000),
                        "finished 队列投递未在超时内到达主线程")
        self.assertIsNone(holder.worker)

    def test_cleanup_identity_guard(self):
        """引用已被换成新线程时不误清新线程。"""
        from PyQt5.QtCore import QObject, QThread

        _app()

        class _Dummy(QThread):
            pass

        class _Holder(QObject):
            pass

        holder = _Holder()
        old = _Dummy()
        new = _Dummy()
        holder.worker = old
        auto_cleanup_thread(holder, "worker", old)
        holder.worker = new  # finished 触发前引用已换
        old.start()
        old.quit()
        old.wait(2000)
        _pump(200)
        self.assertIs(holder.worker, new)  # 新引用保持


if __name__ == "__main__":
    unittest.main()
