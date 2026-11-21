"""全局"加载中"启动界面（LoadingSplash）。

- 显示即见（show+processEvents 后可见）；
- finish() 关闭并 setParent(None)+deleteLater（防残影，）;
- >20s 提示"仍在加载"文案；
- run_platform 的 heavy 判定为静态声明187原 torch 探测恒 False）。
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMainWindow

from xccv.splash import LoadingSplash

_APP = None


def _app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class LoadingSplashTest(unittest.TestCase):
    def test_show_and_finish(self):
        _app()
        splash = LoadingSplash.show_splash("测试平台", "正在加载…")
        try:
            self.assertTrue(splash.isVisible())
        finally:
            main = QMainWindow()
            main.show()
            splash.finish(main)
        self.assertFalse(splash.isVisible())
        self.assertIsNone(splash.parent())

    def test_finish_without_window(self):
        _app()
        splash = LoadingSplash.show_splash("标注平台")
        splash.finish()
        self.assertFalse(splash.isVisible())

    def test_patience_message_after_timeout(self):
        from PyQt5.QtCore import QTimer

        _app()
        splash = LoadingSplash("模型工具", "正在加载…")
        splash._patience.start(50)  # 缩短耐心阈值触发点
        splash.show()
        try:
            loop = QTimer()
            loop.singleShot(200, _app().quit)
            _app().exec_()
            self.assertIn("仍在加载", splash._message)
        finally:
            splash.finish()

    def test_icon_fallback_safe(self):
        from xccv.platform_entry import _splash_icon

        pm = _splash_icon("training")
        missing = _splash_icon("__no_such_key__")  # 回退 app.png（存在）
        self.assertTrue(pm is None or not pm.isNull())
        self.assertTrue(missing is None or not missing.isNull())


class HeavyStaticDeclarationTest(unittest.TestCase):
    """heavy 改静态声明——源码锁（torch 探测不得回归）。"""

    def test_no_torch_probe_in_run_platform(self):
        import inspect

        from xccv import platform_entry

        src = inspect.getsource(platform_entry.run_platform)
        self.assertNotIn('"torch" not in sys.modules', src)
        self.assertNotIn("'torch' not in sys.modules", src)
        self.assertIn("bool(heavy)", src)

    def test_run_platform_shows_splash_for_all(self):
        import inspect

        from xccv import platform_entry

        src = inspect.getsource(platform_entry.run_platform)
        self.assertIn("LoadingSplash.show_splash", src)


class EarlySplashTest(unittest.TestCase):
    """收尾核验：入口顶部预显示（模块级重型导入之前），run_platform 接管。

    实测（runs/probe_startup_v06.py）：训练平台首窗 765ms→~285ms——原先
    pyqtgraph/matplotlib/cv2 等模块级导入全部发生在 run_platform 显示之前。
    """

    _ENTRIES = {
        "xccv/app.py": ("main", "主界面"),
        "xccv/annotation/annotator_window.py": ("annotation", "标注平台"),
        "xccv/training/training_window.py": ("training", "训练平台", True),
        "xccv/testing/testing_window.py": ("testing", "测试平台", True),
        "xccv/modeltools/modeltools_window.py": ("tools", "模型工具", True),
        "xccv/deploy_platform.py": None,  # argv 分发，见 test_deploy_mapping
        "xccv/flow_platform.py": ("flowmain", "流程图平台"),
    }

    def tearDown(self):
        import xccv.splash as sp

        if sp._EARLY is not None:
            sp._EARLY.finish()

    def test_early_adopted_by_show_splash(self):
        _app()
        early = LoadingSplash.show_early("训练平台", "正在加载…")
        adopted = LoadingSplash.show_splash("训练平台", "正在加载深度学习组件…")
        try:
            self.assertIs(adopted, early, "show_splash 必须接管预显示的同一张卡")
            self.assertIn("深度学习", adopted._message)
            import xccv.splash as sp

            self.assertIsNone(sp._EARLY)
        finally:
            adopted.finish()

    def test_finished_early_not_adopted(self):
        _app()
        early = LoadingSplash.show_early("测试平台")
        early.finish()
        fresh = LoadingSplash.show_splash("测试平台")
        try:
            self.assertIsNot(fresh, early)
            self.assertTrue(fresh.isVisible())
        finally:
            fresh.finish()

    def test_adoption_without_change_skips_repaint(self):
        """接管时内容未变不重绘（重绘会再付 ~270ms 中文字体加载，主窗晚出）。"""
        from unittest import mock

        _app()
        early = LoadingSplash.show_early("模型工具", "正在加载…")
        try:
            with mock.patch("xccv.splash.QApplication.processEvents") as pe:
                same = LoadingSplash.show_splash("模型工具", "正在加载…")
            self.assertIs(same, early)
            pe.assert_not_called()
        finally:
            early.finish()

    def test_title_not_double_prefixed(self):
        _app()
        self.assertEqual(LoadingSplash("XCcv 推理服务").title_text(), "XCcv 推理服务")
        self.assertEqual(LoadingSplash("训练平台").title_text(), "XCcv 训练平台")

    def test_entry_modules_call_early_splash_first(self):
        """结构锁：七个入口模块 docstring 之后第一句即 __main__ 守卫的 early_splash。"""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel, expect in self._ENTRIES.items():
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
            first = tree.body[1]
            self.assertIsInstance(first, ast.If, f"{rel}：首句应为 __main__ 守卫")
            self.assertIn("__main__", ast.unparse(first.test), rel)
            body = ast.unparse(first)
            self.assertIn("early_splash(", body, rel)
            if expect is not None:
                call = f"early_splash('{expect[0]}', '{expect[1]}'"
                call += ", heavy=True)" if len(expect) > 2 else ")"
                self.assertIn(call, body, rel)  # heavy 须与 run_platform 一致（文案同→接管不重绘）

    def test_deploy_mapping_matches_modes(self):
        """部署族预显示的 key/标题须与 _MODES（run_platform 实参）一致。"""
        import ast
        from pathlib import Path

        from xccv import deploy_platform

        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "xccv/deploy_platform.py").read_text(encoding="utf-8"))
        dict_node = next(n for n in ast.walk(tree.body[1]) if isinstance(n, ast.Dict))
        early_map = ast.literal_eval(dict_node)
        modes = {m: (key, title) for m, (key, title, _cls) in deploy_platform._MODES.items()}
        self.assertEqual(early_map, modes)

    def test_early_splash_skipped_on_selftest(self):
        from unittest import mock

        import xccv.splash as sp
        from xccv import platform_entry

        _app()
        with mock.patch("sys.argv", ["x", "--selftest"]):
            platform_entry.early_splash("training", "训练平台")
        self.assertIsNone(sp._EARLY)

    def test_early_splash_exits_when_already_running(self):
        from unittest import mock

        import xccv.splash as sp
        from xccv import platform_entry

        _app()
        with mock.patch("sys.argv", ["x"]), \
                mock.patch("xccv.single_instance.already_running", return_value=True), \
                mock.patch("xccv.taskbar.set_process_app_user_model_id"):
            with self.assertRaises(SystemExit):
                platform_entry.early_splash("training", "训练平台")
        self.assertIsNone(sp._EARLY, "已有实例时不应闪加载卡")

    def test_early_splash_shows_card(self):
        from unittest import mock

        import xccv.splash as sp
        from xccv import platform_entry

        _app()
        with mock.patch("sys.argv", ["x"]), \
                mock.patch("xccv.single_instance.already_running", return_value=False), \
                mock.patch("xccv.taskbar.set_process_app_user_model_id"):
            platform_entry.early_splash("testing", "测试平台")
        self.assertIsNotNone(sp._EARLY)
        self.assertTrue(sp._EARLY.isVisible())

    def test_early_splash_keeps_app_alive_fresh_process(self):
        """early_splash 内创建的 QApplication 必须保活。

        局部变量持有时函数返回即被 GC 销毁（连带加载卡）→ run_platform 再建
        新 app+新卡、重付 ~300ms 字体加载（实测轻量平台主窗晚出 ~400ms）。
        进程内测试已有 app 测不出，故用全新子进程。
        """
        import subprocess
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        code = (
            "import sys, gc; sys.path.insert(0, r'%s')\n"
            "from unittest import mock\n"
            "import xccv.platform_entry as pe\n"
            "from PyQt5.QtWidgets import QApplication\n"
            "assert QApplication.instance() is None\n"
            "with mock.patch('xccv.single_instance.already_running', return_value=False), "
            "mock.patch('xccv.taskbar.set_process_app_user_model_id'):\n"
            "    pe.early_splash('testing', '测试平台', heavy=True)\n"
            "gc.collect()\n"
            "import xccv.splash as sp\n"
            "print('APP', QApplication.instance() is not None, 'ALIVE', sp._alive(sp._EARLY))\n"
        ) % str(root)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, env=env, cwd=str(root), timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        self.assertIn("APP True ALIVE True", proc.stdout)


if __name__ == "__main__":
    unittest.main()


