"""部署服务测试：3 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_v06_freeze_watchdog
- test_v06_batch4_fixes
- test_offline
"""


# == 来源模块：test_v06_freeze_watchdog ==
"""主线程卡顿看门狗 + auto_cleanup_thread。

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
            self.assertIn(os.path.basename(__file__), text)  # 栈顶留痕
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
                        "心跳 QTimer 被回收：看门狗无法布防")
        time.sleep(0.45)  # 布防后的真实卡顿
        self.assertTrue(_pump_until(lambda: "卡顿结束" in self._read(), 15000),
                        "卡顿结束未记录：心跳在卡顿后未恢复")
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

# == 来源模块：test_v06_batch4_fixes ==
"""回归锁：打包阶段反馈的 bug 修复。

覆盖：单实例业务参数转发（训练项目直达）、推理服务编译态启动命令、
模型加载窗口真实服务探测、流程图空运行守卫顺序、AI 等待横幅判定。
"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5 import QtWidgets
from PyQt5.QtNetwork import QLocalServer, QLocalSocket

_APP__v06_batch4_fixes = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class SingleInstanceForwardTest(unittest.TestCase):
    """单实例激活消息携带业务参数（训练平台已开时直达项目）。"""

    def test_payload_forwarded_to_registered_window(self):
        from xccv.single_instance import already_running, register

        received = []

        class Win(QtWidgets.QWidget):
            def on_platform_message(self, args):
                received.append(list(args))

        win = Win()
        win.show()
        register("b4test1", win)
        self.addCleanup(QLocalServer.removeServer, "xccv-b4test1")
        self.assertTrue(already_running("b4test1", ["冒烟项目"]))
        deadline = __import__("time").monotonic() + 3
        while not received and __import__("time").monotonic() < deadline:
            _APP__v06_batch4_fixes.processEvents()
        self.assertEqual(received, [["冒烟项目"]])

    def test_plain_activate_still_works(self):
        from xccv.single_instance import already_running, register

        received = []

        class Win(QtWidgets.QWidget):
            def on_platform_message(self, args):
                received.append(list(args))

        win = Win()
        win.show()
        register("b4test2", win)
        self.addCleanup(QLocalServer.removeServer, "xccv-b4test2")
        self.assertTrue(already_running("b4test2"))   # 无 payload：仅激活
        self.assertFalse(received)   # 无参数时不应触发业务钩子


class ServiceStartCommandTest(unittest.TestCase):
    """编译安装态推理服务必须走 importlib（-m 在 .pyd 下秒退）。"""

    def test_importlib_start_and_log_redirect(self):
        """服务启动一律 importlib 形式（.pyd 下 -m 秒退）+ 日志落盘。"""
        import xccv.deploy_platform as dp

        src = Path(dp.__file__).read_text(encoding="utf-8")
        self.assertIn("importlib.import_module('xccv.service.main').main()", src)
        self.assertNotIn('"-m", "xccv.service.main"', src)   # 移除 -m 形式
        self.assertIn("stdout=logf", src)                    # 日志落盘（不再 DEVNULL）
        self.assertIn("stderr=subprocess.STDOUT", src)
        self.assertIn("service.log", src)
        self.assertIn("mkdir(parents=True, exist_ok=True)", src)

    def test_log_redirect_targets_service_log(self):
        import xccv.deploy_platform as dp

        src = Path(dp.__file__).read_text(encoding="utf-8")
        self.assertIn('service.log', src)
        self.assertIn('mkdir(parents=True, exist_ok=True)', src)


class ModelLoadProbeTest(unittest.TestCase):
    """模型加载窗口独立模式不得写死"运行中"。"""

    def test_service_state_probes_in_standalone(self):
        import xccv.deploy_platform as dp

        src = Path(dp.__file__).read_text(encoding="utf-8")
        self.assertIn("_probe_service", src)
        self.assertNotIn("return True, \"\"   # 独立模式不预判", src)

    def test_probe_returns_false_when_port_closed(self):
        from xccv.deploy_platform import ModelServiceWindow

        w = ModelServiceWindow.__new__(ModelServiceWindow)
        w._page = None
        from unittest import mock

        with mock.patch.object(ModelServiceWindow, "base_url",
                               return_value="http://127.0.0.1:59999"):
            self.assertFalse(w.service_state()[0])


class FlowRunGuardTest(unittest.TestCase):
    """流程图空运行——选图守卫必须先于模型加载。"""

    def test_guard_precedes_model_load(self):
        import xccv.flow_platform as fp

        src = Path(fp.__file__).read_text(encoding="utf-8")
        run = src[src.index("def _run_flow(self):"):src.index("def _pack_flow")]
        img_guard = run.index("img_path = self._selected_image()")
        model_load = run.index("self._resolve_model(flow)")
        self.assertLess(img_guard, model_load)
        self.assertIn("_pick_image_dialog", run)   # 兜底文件选择器


class AiBannerTest(unittest.TestCase):
    """AI 等待横幅——加载中显示、就绪收起。"""

    def _win(self):
        from labelme.app import MainWindow

        # 直接构造过重：用最小替身验证判定逻辑
        class _Stub:
            pass

        w = _Stub()
        w._ai_banner = None
        return w

    def test_wait_words_shown_and_ready_hides(self):
        from xccv.annotation import ai_assist  # noqa: F401  触发依赖链检查
        import labelme.app as la

        src = Path(la.__file__).read_text(encoding="utf-8")
        self.assertIn("_on_ai_banner", src)
        self.assertIn("加载中", src)
        self.assertIn("aiWaitBanner", src)
        # 就绪路径收起横幅
        self.assertIn("self._on_ai_banner(None, None)", src)


class PlatformMessageHookTest(unittest.TestCase):
    """training_window 必须实现单实例业务钩子。"""

    def test_training_window_has_hook(self):
        from xccv.training.training_window import TrainingWindow

        self.assertTrue(hasattr(TrainingWindow, "on_platform_message"))

# == 来源模块：test_offline ==
"""离线权重：随包基准权重优先、完全离线训练的配置断言。

基准权重不随仓库分发（体积与许可各异），本地可选放置；对应断言在权重缺失时跳过。
"""

import os
import unittest
from pathlib import Path

from xccv.engine.dataset.base import TaskType
from xccv.engine.models.yolo import YoloModel
from xccv.engine.offline import WEIGHTS_DIR, setup_offline_weights

_HAS_YOLO11N = (WEIGHTS_DIR / "ultralytics" / "yolo11n.pt").is_file()
_HAS_YOLO11S_SEG = (WEIGHTS_DIR / "ultralytics" / "yolo11s-seg.pt").is_file()


class OfflineWeightsTest(unittest.TestCase):
    def test_setup_redirects_torch_home(self):
        """随包 torch 权重目录存在时才重定向 TORCH_HOME（无权重仓库不指向空目录）。"""
        setup_offline_weights()
        expected = str((Path(__file__).resolve().parents[1]
                        / "xccv" / "engine" / "weights" / "torch"))
        if Path(expected).is_dir():
            self.assertEqual(os.environ.get("TORCH_HOME"), expected)
        else:
            self.assertNotEqual(os.environ.get("TORCH_HOME"), expected)

    def test_weights_dir_points_inside_package(self):
        self.assertTrue(str(WEIGHTS_DIR).endswith(str(Path("engine") / "weights")))

    def test_yolo_pretrained_path_points_at_bundled_dir(self):
        """随包权重在位 → 指向随包目录；缺失 → 回退裸名（运行期可见下载）。"""
        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=True)
        if _HAS_YOLO11N:
            self.assertEqual(Path(m.init_weight),
                             WEIGHTS_DIR / "ultralytics" / "yolo11n.pt")
        else:
            self.assertEqual(Path(m.init_weight), Path("yolo11n.pt"))

    @unittest.skipUnless(_HAS_YOLO11N, "未放置基准权重 yolo11n.pt（可选）")
    def test_yolo_uses_bundled_pretrained(self):
        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=True)
        self.assertTrue(Path(m.init_weight).is_file())

    @unittest.skipUnless(_HAS_YOLO11S_SEG, "未放置基准权重 yolo11s-seg.pt（可选）")
    def test_yolo_seg_uses_bundled_pretrained(self):
        m = YoloModel("yolo11s", TaskType.SEGMENTATION, 2, use_pretrained=True)
        self.assertTrue(Path(m.init_weight).is_file())
        self.assertIn("yolo11s-seg.pt", str(m.init_weight))

    def test_yolo_no_pretrained_still_yaml(self):
        m = YoloModel("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
        self.assertEqual(m.init_weight, "yolo11n.yaml")

if __name__ == '__main__':
    unittest.main()
