"""标注平台测试：6 个历史测试模块的合并模块。

由以下模块按域合并而成，用例逐段原样迁移、断言未改：
- test_app_shell
- test_v06_splash
- test_v06_theme_live
- test_v06_t5_sam_reload
- test_v041_fixes
- test_v042_fixes
"""


# == 来源模块：test_app_shell ==
"""集成测试：主界面整合行为 + 标注产物→训练→测试 跨模块全链路。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from tests.synthetic import make_labelme_dataset
from xccv.app import _PLATFORMS, HomeWindow
from xccv.engine.dataset.base import TaskType
from xccv.engine.io_utils import imread_unicode
from xccv.engine.models import TrainConfig, create_model, load_model
from xccv.testing.render import draw_predictions

_APP = QApplication.instance() or QApplication(sys.argv)


class HomeWindowTest(unittest.TestCase):
    """主界面改为子进程拉起平台：验证启动调用与失败提示，不真正弹窗。"""

    def setUp(self):
        self.win = HomeWindow()
        self.addCleanup(self.win.close)

    def test_has_four_platform_cards(self):
        self.assertEqual(len(_PLATFORMS), 4)
        self.assertEqual([p[0] for p in _PLATFORMS],
                         ["annotation", "training", "testing", "tools"])

    def test_open_platform_launches_independent_process(self):
        with patch("xccv.launcher.launch_platform") as mock_launch:
            self.win._open_platform("annotation")
            self.win._open_platform("training")
            self.win._open_platform("testing")
            self.win._open_platform("tools")
            self.assertEqual(
                [c.args[0] for c in mock_launch.call_args_list],
                ["annotation", "training", "testing", "tools"])

    def test_launch_failure_shows_message(self):
        """启动失败要弹窗提示，而不是点击无反应。"""
        with patch("xccv.launcher.launch_platform", side_effect=OSError("boom")), \
             patch("xccv.app.QMessageBox.critical") as mock_box:
            self.win._open_platform("training")
            self.assertTrue(mock_box.called)


class PipelineTest(unittest.TestCase):
    """跨模块全链路：标注产物（labelme json 数据集）→ 训练 → 测试推理可视化。"""

    def test_annotate_train_infer(self):
        with tempfile.TemporaryDirectory() as tmp:
            det, _ = make_labelme_dataset(Path(tmp) / "det", n_images=6)
            model = create_model("yolo11n", TaskType.DETECTION, 2, use_pretrained=False)
            metrics = model.train(TrainConfig(
                task=TaskType.DETECTION, dataset_dir=det,
                output_dir=Path(tmp) / "run", model_name="yolo11n",
                epochs=1, batch_size=2, val_ratio=0.25,
                image_size=64, workers=0, use_pretrained=False))
            self.assertIn("map50", metrics["history"][0])  # 键名标准化后可进入白名单绘制

            from xccv.testing.testing_window import latest_weights
            ckpt = latest_weights(Path(tmp))
            self.assertIsNotNone(ckpt)

            infer_model = load_model(ckpt)
            self.assertEqual(infer_model.task.value, "detection")
            img = imread_unicode(next(det.glob("*.png")))
            self.assertIsNotNone(img)
            pred = infer_model.predict(img, conf=0.1, iou=0.45)
            rendered = draw_predictions(img, pred, infer_model.classes, "detection")
            self.assertEqual(rendered.shape, img.shape)

# == 来源模块：test_v06_splash ==
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

_APP__v06_splash = None


def _app():
    global _APP__v06_splash
    if _APP__v06_splash is None:
        _APP__v06_splash = QApplication.instance() or QApplication([])
    return _APP__v06_splash


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

# == 来源模块：test_v06_theme_live ==
"""浅色/暗色主题**实时切换**回归：切换即时生效，无需重启软件。

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

_APP__v06_theme_live = QApplication.instance() or QApplication([])
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
    """暗色主题：日志/说明框不得白底浅字——QPlainTextEdit 专用规则。"""

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
        _APP__v06_theme_live.processEvents()
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

# == 来源模块：test_v06_t5_sam_reload ==
"""SAM 后台加载失败必须解锁 _ai_loading（失败不得永久吞点击）。

复测轮发现：initializeAiModelAsync 失败只发 aiModelFailed
状态栏消息，canvas._ai_loading 卡 True——后续点击全部被「AI 模型加载中」
吞掉（mousePressEvent 首锚分支守卫），S 键重试也被入口处
`if self._ai_loading: return` 拦截，AI 标注永久失效直到重启。修复：失败
路径把 _ai_loading 复位为 False，点击与重试即刻恢复可用。
"""

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from labelme.widgets import canvas as canvas_mod
from labelme.widgets.canvas import Canvas

_APP__v06_t5_sam_reload = None


def _raising_factory(msg):
    class _Boom:
        name = "boom-sam"

        def __init__(self, *a, **k):
            raise RuntimeError(msg)

    return _Boom


class FailRecoveryTest(unittest.TestCase):
    """加载失败 → _ai_loading 复位 → 点击入栈恢复、S 键重试放行。"""

    def _patch_factory(self, cls):
        orig = canvas_mod.osam.apis.get_model_type_by_name
        canvas_mod.osam.apis.get_model_type_by_name = lambda name: cls
        self.addCleanup(lambda: setattr(canvas_mod.osam.apis,
                                        "get_model_type_by_name", orig))

    def _make_canvas(self):
        global _APP__v06_t5_sam_reload
        _APP__v06_t5_sam_reload = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cv = Canvas()
        cv.resize(50, 50)   # 部件小于图 → offsetToCenter=0，点击坐标不漂移
        cv.pixmap = QtGui.QPixmap(60, 60)
        cv.pixmap.fill(QtGui.QColor(255, 255, 255))
        cv.createMode = "ai_polygon"
        cv.mode = Canvas.CREATE
        return _APP__v06_t5_sam_reload, cv

    def _wait_unlocked(self, app, cv, timeout=5.0):
        deadline = time.monotonic() + timeout
        while cv._ai_loading and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()

    def test_failed_load_unlocks_clicks(self):
        app, cv = self._make_canvas()
        self._patch_factory(_raising_factory("GPU 暂忙（模拟）"))
        failed = []
        cv.aiModelFailed.connect(failed.append)
        cv.initializeAiModelAsync("boom-sam")
        self._wait_unlocked(app, cv)
        self.assertTrue(failed)                       # 失败信号已发
        self.assertFalse(cv._ai_loading)              # 修复点：解锁
        ev = QtGui.QMouseEvent(
            QtGui.QMouseEvent.MouseButtonPress, QtCore.QPointF(10.0, 10.0),
            QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.NoModifier)
        cv.mousePressEvent(ev)
        self.assertEqual(len(cv._ai_prompts), 1)      # 点击不再被吞

    def test_retry_allowed_after_failure(self):
        app, cv = self._make_canvas()
        self._patch_factory(_raising_factory("第一次失败（模拟）"))
        cv.initializeAiModelAsync("boom-sam")
        self._wait_unlocked(app, cv)
        self.assertFalse(cv._ai_loading)
        # 第二次（换名）加载不得再被入口 _ai_loading 拦截
        self._patch_factory(_raising_factory("第二次失败（模拟）"))
        statuses = []
        cv.aiStatus.connect(statuses.append)
        cv.initializeAiModelAsync("boom-sam-2")
        self._wait_unlocked(app, cv)
        self.assertTrue(any("后台加载中（" in s for s in statuses),
                        f"重试被拦截，状态={statuses}")
        self.assertFalse(cv._ai_loading)

# == 来源模块：test_v041_fixes ==
"""v0.4.1 修复回归测试。

覆盖：工作台 Qt5 导航兼容、算子中文标题、xccv2 前缀嗅探、混图友好报错、
运行日志落盘、无图像输出节点快照叠加、纯传统流程打包/运行、ai_detection
旧链兼容、批量期间禁并发推理。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2

from xccv.engine.io_utils import imwrite_unicode
import numpy as np

from xccv.modeltools.flow_session import (
    FlowSession, _normalize_types, node_library_payload,
)
from xccv.modeltools import flow_session as _fs

node_library_payload()  # 注册全部算子

# 运行日志重定向到临时目录（避免测试污染仓库 platform_data）
import tempfile as _tempfile

_fs._RUN_LOG_DIR = Path(_tempfile.gettempdir()) / "xccv_test_v041_logs"


def _app__v041_fixes():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


_WINDOW_HOLDER = []   # 测试窗口持引用：防测试进程尾部析构原生崩溃


class WorkbenchNavTest(unittest.TestCase):
    """A1/G1 修复回归：主界面「传统平台」页挂载必须是 Qt5 兼容写法。

    根因：栈替换的 Qt6 专属 API 在 Qt 5.15 没有，实测首点「传统平台」
    即 AttributeError 崩溃。修复后为 removeWidget + setParent(None) + insertWidget。
    注：不构造真实 HomeWindow 做集成导航——重模块导入顺序会改变 sip 对
    顶层窗口的托管（测试进程内/退出期原生崩溃），改用逻辑级 + 源码级双重锁定。
    """

    def test_no_qt6_replace_widget_in_app(self):
        """源码级锁定：app.py 不得出现 Qt6-only 的 replaceWidget。"""
        src = (Path(__file__).resolve().parents[1] / "xccv" / "app.py").read_text(
            encoding="utf-8")
        self.assertNotIn("replaceWidget", src)

    def test_workbench_constructs_with_registry(self):
        """工作台本体可构建（等价于导航后首次挂载的实质内容）。

        注：本用例若与 testing_window 导入同进程会触发退出期原生崩溃
        （sip 清理顺序），工作台构建的完整覆盖在 test_batch20_workbench，
        这里仅以分类过滤下拉的选项完整性做纯逻辑校验。
        """
        from xccv.engine.flow.registry import all_categories, list_nodes

        cats = set(all_categories())
        self.assertLessEqual({"图像源", "预处理", "定位", "逻辑", "检出过滤", "AI"},
                             cats)
        self.assertGreaterEqual(len(list_nodes()), 30)


class ChineseTitleTest(unittest.TestCase):
    """节点库/画布暴露中文名称，而非注册函数名。"""

    def test_titles_are_chinese(self):
        from xccv.engine.flow.registry import list_nodes

        titles = {n["name"]: n["title"] for n in list_nodes()}
        self.assertEqual(titles["gaussian_blur"], "高斯模糊")
        self.assertEqual(titles["threshold"], "二值化")
        self.assertEqual(titles["blob_analysis"], "斑点分析")
        self.assertEqual(titles["find_circle"], "圆定位")
        self.assertEqual(titles["template_match"], "模板匹配")
        self.assertEqual(titles["model_detection"], "目标检测模型")
        self.assertEqual(titles["model_instance_seg"], "实例分割模型")
        self.assertEqual(titles["model_classification"], "分类模型")
        self.assertEqual(titles["filter_confidence"], "置信度过滤")
        self.assertEqual(titles["filter_count_check"], "数量判定")
        self.assertEqual(titles["image_loader"], "读入图片")
        # 生产算子（34 个）title 全部不含下划线（排除漏翻译的函数名）；
        # 注册表是全局的，套件中测试注册的节点（ex_/rg_ 前缀等）不在检查范围
        prod = [n for n in list_nodes() if not n["name"].startswith(("rg_", "ex_",
                                                                     "lr_"))]
        for n in prod:
            self.assertNotIn("_", n["title"],
                             f"{n['name']} 的标题仍是函数名: {n['title']}")


class SniffXccv2Test(unittest.TestCase):
    """修复：画布保存的新格式（xccv2/ 前缀）被正确识别并拒绝误跑。"""

    def test_sniff_xccv2_prefix(self):
        from xccv.engine.flow.legacy_adapter import sniff_format

        self.assertEqual(sniff_format({"nodes": [
            {"id": 1, "type": "xccv2/gaussian_blur", "properties": {}}]}), "xcv_graph")

    def test_load_flow_rejects_canvas_saved_xcv(self):
        from xccv.engine import detection_flow as df

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "检出流程.json"
            p.write_text(json.dumps({"last_node_id": 1, "last_link_id": 0,
                                     "nodes": [{"id": 1, "type": "xccv2/threshold",
                                                "properties": {}}],
                                     "links": []}, ensure_ascii=False),
                         encoding="utf-8")
            with self.assertLogs("xccv.engine.flow.legacy", level="WARNING"):
                self.assertIsNone(df.load_flow(p))


class MixedGraphTest(unittest.TestCase):
    """混图（旧版节点 + 新算子）给中文人话报错，而非"类型未注册"。"""

    def test_mixed_graph_friendly_error(self):
        from xccv.modeltools.flow_session import FlowErrorMixed

        graph = {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {"path": ""}},
            {"id": 2, "type": "xccv/post_confidence", "properties": {"min": 0.3}}],
            "links": []}
        with self.assertRaises(FlowErrorMixed) as cm:
            FlowSession().run(graph)
        self.assertIn("xccv/post_confidence", str(cm.exception))
        self.assertIn("混用", str(cm.exception))


class RunLogTest(unittest.TestCase):
    """Q4：每次运行落一份日志到 platform_data/logs/流程运行日志。"""

    def test_run_writes_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.png"
            imwrite_unicode(Path(p), np.zeros((8, 8, 3), np.uint8))
            s = FlowSession()
            s.run({"nodes": [{"id": 1, "type": "image_loader",
                              "properties": {"path": str(p)}}], "links": []},
                  label="单测")
            # glob 顺序取决于文件系统（Linux ext4 为哈希序），须排序后取最新；
            # 运行日志名带当前时间戳，必然排在 20260101 占位文件之后
            logs = sorted(_fs._RUN_LOG_DIR.glob("flow_*.txt"))
            self.assertTrue(logs)
            text = logs[-1].read_text(encoding="utf-8")
            self.assertIn("来源：单测", text)
            self.assertIn("image_loader(#1)", text)

    def test_log_dir_kept_under_cap(self):
        _fs._RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(_fs._RUN_LOG_KEEP + 6):
            (_fs._RUN_LOG_DIR / f"flow_20260101_{i:06d}_000.txt").write_text("x",
                                                                             encoding="utf-8")
        FlowSession.write_run_log({"ok_count": 1, "total": 1, "errors": {},
                                   "logs": "", "warnings": []}, label="清理")
        remaining = list(_fs._RUN_LOG_DIR.glob("flow_*.txt"))
        self.assertLessEqual(len(remaining), _fs._RUN_LOG_KEEP)


class SnapshotOverlayTest(unittest.TestCase):
    """C2/C3 修复：无图像输出节点（blob/find_circle 等）有"输入+结果叠加"快照。"""

    def _run_blob(self, tmp: Path):
        import xccv.engine.flow.ops.locate  # noqa: F401
        from xccv.engine.flow.executor import Executor
        from xccv.engine.flow.node_base import RunContext

        img = np.zeros((60, 80), np.uint8)
        cv2.circle(img, (40, 30), 12, 255, -1)
        p = tmp / "blob.png"
        imwrite_unicode(Path(p), img)
        g = {"nodes": [
            {"id": 1, "type": "image_loader", "properties": {"path": str(p)}},
            {"id": 2, "type": "blob_analysis",
             "properties": {"threshold_mode": "otsu", "thresh": 128, "invert": False,
                            "min_area": 10, "max_area": 0, "circularity": 0.0}}],
            "links": [[1, 1, 0, 2, 0]]}
        ctx = RunContext(snapshot_store=_fs.SnapshotStore())
        Executor(g, ctx).run()
        return ctx.snapshot_store.get(2)

    def test_blob_snapshot_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = self._run_blob(Path(tmp))
            self.assertIsNotNone(snap)
            # 斑点分析带图像直通输出 → 快照存输出图（键 image）
            self.assertIn("image", snap["images"])
            vis = snap["images"]["image"]
            self.assertEqual(vis.ndim, 3)   # 快照图统一转 BGR 三通道

    def test_overlay_draws_boxes_for_prediction(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.models.base import Prediction

        img = np.zeros((40, 40, 3), np.uint8)
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[5, 5, 30, 30]])
        vis = overlay_result(img, {"detection": pred})
        self.assertTrue((vis != img).any())   # 框画上去了


class PackRelaxationTest(unittest.TestCase):
    """纯传统流程可打包（不含模型）；旧链无模型仍拦截。"""

    def _xcv_flow(self):
        return {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {"path": ""}},
            {"id": 2, "type": "xccv2/gaussian_blur", "properties": {}}],
            "links": [[1, 1, 0, 2, 0]]}

    def test_pure_traditional_passes_validation(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        self.assertEqual(validate_flow_for_pack(self._xcv_flow()), [])

    def test_legacy_graph_without_model_still_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [{"id": 1, "type": "xccv/post_confidence",
                           "properties": {"min": 0.3}}], "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("没有模型节点" in p for p in problems))

    def test_xcv_model_node_without_weight_blocked(self):
        """新模型节点未选权重 → 打包校验拦截（C1：校验作用到新节点）。"""
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [{"id": 1, "type": "xccv2/model_detection",
                           "properties": {}}], "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("尚未选择权重文件" in p for p in problems))

    def test_xcv_bad_count_check_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [
            {"id": 1, "type": "xccv2/filter_count_check",
             "properties": {"min": 5, "max": 2}}],
            "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("下限 5 大于上限 2" in p for p in problems))

    def test_xcv_bad_region_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [{"id": 1, "type": "xccv2/filter_region_keep",
                           "properties": {"regions": "1,2,3"}}],
                "links": []}
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("不是 4 个数字" in p for p in problems))

    def test_xcv_disconnected_blocked(self):
        from xccv.modeltools.flow_shared import validate_flow_for_pack

        flow = {"nodes": [
            {"id": 1, "type": "xccv2/image_loader", "properties": {"path": ""}},
            {"id": 2, "type": "xccv2/gaussian_blur", "properties": {}}],
            "links": []}   # 无连线：blur 的必需输入缺失 → 结构问题
        problems = validate_flow_for_pack(flow)
        self.assertTrue(any("结构问题" in p or "缺少" in p for p in problems))

    def test_xvap_roundtrip_without_models(self):
        from xccv.engine import flow_package as fp

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pure.xvap"
            fp.save_xvap(path, self._xcv_flow(), {})
            flow, models = fp.load_xvap(path)
            self.assertEqual(len(flow["nodes"]), 2)
            self.assertEqual(models, {})


class FlowPageXcvRunTest(unittest.TestCase):
    """流程图平台页的新算子流程运行路由。"""

    def test_flow_page_runs_xcv_flow(self):
        """Q7：流程图平台页「运行流程」对纯传统流程走新引擎并回写状态栏。

        用 MagicMock 作 self 调未绑定方法（覆盖路径注入/运行/状态消息逻辑）；
        不构造真实 FlowPage——重 Qt 窗口在测试进程的析构顺序会触发原生崩溃。
        """
        _app__v041_fixes()
        from xccv.flow_platform import FlowPage

        with tempfile.TemporaryDirectory() as tmp:
            img = np.zeros((60, 80, 3), np.uint8)
            img[20:40, 30:60] = 255
            img_path = Path(tmp) / "目标.png"
            from xccv.engine.io_utils import imwrite_unicode

            imwrite_unicode(img_path, img)   # cv2.imwrite 中文文件名静默失败
            flow = {"last_node_id": 2, "last_link_id": 1, "nodes": [
                {"id": 1, "type": "xccv2/image_loader", "pos": [0, 0],
                 "properties": {"path": "", "is_grayscale": False}},
                {"id": 2, "type": "xccv2/gaussian_blur", "pos": [1, 0],
                 "properties": {"ksize": 3, "sigma": 0.0}}],
                "links": [[1, 1, 0, 2, 0]]}
            flow_path = Path(tmp) / "flow.json"
            flow_path.write_text(json.dumps(flow, ensure_ascii=False),
                                 encoding="utf-8")

            page = mock.MagicMock()
            FlowPage._run_flow_xcv(page, flow, img_path)
            msg = page.window.return_value.statusBar.return_value \
                .showMessage.call_args[0][0]
            self.assertIn("2/2 节点成功", msg)
            self.assertIn("流程运行日志", msg)
            page._show_mat.assert_called_once()   # 可视化已回显


class BatchConcurrencyGuardTest(unittest.TestCase):
    """防护：批量运行期间禁用 UI 线程单图推理（同模型跨线程并发）。

    用 MagicMock 作 self 调未绑定方法——不实例化 PyQt 对象（未初始化的
    PyQt 子类桩被 GC 时会触发 sip 原生崩溃，且崩在后续用例里极难定位）。
    """

    def test_infer_blocked_during_batch(self):
        from xccv.testing.testing_window import TestingWindow

        fake = mock.MagicMock()
        fake._xcv_flow = None                    # 非传统流程模式
        fake._current_image_path.return_value = None   # 非批量分支取不到图即返回
        fake.model = None
        TestingWindow._infer_current(fake)          # 无模型 → 早退
        fake.result_stats.setText.assert_called_once_with("未加载模型")
        fake._current_image_path.assert_not_called()

        fake.model = object()
        fake._batch_thread = object()               # 批量运行中 → 守卫拦截
        TestingWindow._infer_current(fake)
        fake._current_image_path.assert_not_called()

        fake._batch_thread = None
        TestingWindow._infer_current(fake)          # 非批量 → 正常走到取图
        fake._current_image_path.assert_called_once()

# == 来源模块：test_v042_fixes ==
"""v0.4.2 修复测试：标注平台亮度对比度对话框、叠加蒙版、日志保留数、
模型节点拆分后的打包校验（其余见 test_v041_fixes / test_flow_ai_bridge）。"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import PIL.Image



def _app__v042_fixes():
    global _APP
    if _APP is None:
        from PyQt5.QtWidgets import QApplication

        _APP = QApplication.instance() or QApplication([])
    return _APP


class BrightnessContrastDialogTest(unittest.TestCase):
    """Q6 修复：RGBA 图不再内存错位崩溃；滑条驱动回调正常。"""

    def _dialog(self, mode):
        from labelme.widgets.brightness_contrast_dialog import (
            BrightnessContrastDialog,
        )

        _app__v042_fixes()
        img = PIL.Image.new(mode, (24, 16), (120, 120, 120, 255)[:len(mode)])
        got = {}
        dlg = BrightnessContrastDialog(img, lambda qimage: got.update(qimage=qimage))
        _WINDOW_HOLDER.append(dlg)
        self.addCleanup(dlg.close)
        return dlg, got

    def test_rgba_image_callback(self):
        """RGBA 源图（PNG 常见）：不再错位崩溃，回调收到正确尺寸的 QImage。"""
        dlg, got = self._dialog("RGBA")
        dlg.slider_brightness.setValue(80)     # 1.6 倍亮度
        dlg.onNewValue(None)
        self.assertIn("qimage", got)
        self.assertEqual(got["qimage"].width(), 24)
        self.assertEqual(got["qimage"].height(), 16)

    def test_rgb_image_and_neutral(self):
        dlg, got = self._dialog("RGB")
        dlg.onNewValue(None)                    # 1.0/1.0 中性值也走通
        self.assertIn("qimage", got)

    def test_buffer_held_for_qimage(self):
        """QImage 不拷贝字节：缓冲必须被对话框持有（否则释放后绘制即崩）。"""
        from labelme.widgets.brightness_contrast_dialog import (
            BrightnessContrastDialog,
        )

        dlg, _ = self._dialog("RGB")
        dlg.onNewValue(None)
        self.assertTrue(len(dlg._buf) >= 24 * 16 * 3)


class OverlayMaskTest(unittest.TestCase):
    """ToC4/ToE2 修复：几何/检出叠加为半透明蒙版（大面积变化，非细线）。"""

    def test_circle_mask_fills_area(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.flow.ir import Geometry

        img = np.zeros((80, 80, 3), np.uint8)
        circle = Geometry(kind="circle", data=np.array([40.0, 40.0, 15.0]))
        vis = overlay_result(img, {"circle": circle})
        changed = int((vis.sum(axis=2) > 0).sum())
        fill_area = int(3.1416 * 15 * 15)
        self.assertGreater(changed, fill_area * 0.5)   # 填充蒙版生效（远超细线）
        # 圆心附近像素明显被蒙版染色
        self.assertGreater(int(vis[40, 40, 1]), 30)    # G 通道抬升

    def test_detection_boxes_mask(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.models.base import Prediction

        img = np.zeros((60, 60, 3), np.uint8)
        pred = Prediction(labels=[0], scores=[0.9], boxes=[[10, 10, 40, 40]])
        vis = overlay_result(img, {"detection": pred})
        self.assertGreater(int(vis[25, 25, 2]), 30)    # 框内橙色蒙版

    def test_line_thick_with_endpoints(self):
        from xccv.engine.flow.snapshot import overlay_result
        from xccv.engine.flow.ir import Geometry

        img = np.zeros((50, 50, 3), np.uint8)
        line = Geometry(kind="line", data=np.array([[5.0, 25.0], [45.0, 25.0]]))
        vis = overlay_result(img, {"line": line})
        self.assertGreater(int(vis[25, 25, 1]), 30)    # 线身上色
        self.assertGreater(int(vis[25, 5, 1]), 30)     # 端点圆点


class LogRetentionTest(unittest.TestCase):
    def test_keep_is_50(self):
        from xccv.modeltools import flow_session as fs

        self.assertEqual(fs._RUN_LOG_KEEP, 50)

if __name__ == '__main__':
    unittest.main()
