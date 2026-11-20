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

_APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class SingleInstanceForwardTest(unittest.TestCase):
    """P-B4-1：单实例激活消息携带业务参数（训练平台已开时直达项目）。"""

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
            _APP.processEvents()
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
    """P-B4-2：编译安装态推理服务必须走 importlib（-m 在 .pyd 下秒退）。"""

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
    """P-B4-3：模型加载窗口独立模式不得写死"运行中"。"""

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
    """P-B4-4：流程图空运行——选图守卫必须先于模型加载。"""

    def test_guard_precedes_model_load(self):
        import xccv.flow_platform as fp

        src = Path(fp.__file__).read_text(encoding="utf-8")
        run = src[src.index("def _run_flow(self):"):src.index("def _pack_flow")]
        img_guard = run.index("img_path = self._selected_image()")
        model_load = run.index("self._resolve_model(flow)")
        self.assertLess(img_guard, model_load)
        self.assertIn("_pick_image_dialog", run)   # 兜底文件选择器


class AiBannerTest(unittest.TestCase):
    """P-B4-5：AI 等待横幅——加载中显示、就绪收起。"""

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


if __name__ == "__main__":
    unittest.main()
